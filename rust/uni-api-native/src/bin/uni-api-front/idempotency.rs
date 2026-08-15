use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode, Uri};
use bytes::Bytes;
use futures_util::stream;
use sha2::{Digest, Sha256};
use tokio::sync::{watch, Mutex};

const STATUS_HEADER: &str = "x-uni-api-idempotency-status";

#[derive(Clone)]
pub struct Coordinator {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    next_owner: AtomicU64,
    ttl: Duration,
    wait_timeout: Duration,
    max_entries: usize,
    max_stored_bytes: usize,
    max_response_bytes: usize,
    max_inflight_response_bytes: usize,
    inflight_response_bytes: AtomicUsize,
}

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    stored_bytes: usize,
}

struct Entry {
    request_hash: String,
    owner_token: Option<u64>,
    notify: watch::Sender<()>,
    response: Option<CachedResponse>,
    nonreplayable_reason: Option<String>,
    completed_at: Option<Instant>,
    expires_at: Option<Instant>,
}

#[derive(Clone)]
struct CachedResponse {
    status: StatusCode,
    headers: HeaderMap,
    chunks: Arc<Vec<Bytes>>,
    bytes: usize,
}

pub enum Claim {
    Owner(Owner),
    Response(Response<Body>),
}

pub struct Owner {
    coordinator: Coordinator,
    record_key: String,
    token: u64,
    armed: bool,
}

pub struct RequestIdentity {
    record_key: String,
    request_hash: String,
}

pub struct RequestHasher {
    record_key: String,
    request: Sha256,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                next_owner: AtomicU64::new(1),
                ttl: env_duration("IDEMPOTENCY_TTL_SECONDS", 15.0 * 60.0),
                wait_timeout: env_duration("IDEMPOTENCY_WAIT_TIMEOUT_SECONDS", 30.0 * 60.0),
                max_entries: env_usize("IDEMPOTENCY_MAX_ENTRIES", 4096),
                max_stored_bytes: env_usize("IDEMPOTENCY_MAX_STORED_BYTES", 128 * 1024 * 1024),
                max_response_bytes: env_usize("IDEMPOTENCY_MAX_RESPONSE_BYTES", 16 * 1024 * 1024),
                max_inflight_response_bytes: env_usize(
                    "RUST_IDEMPOTENCY_MAX_INFLIGHT_RESPONSE_BYTES",
                    128 * 1024 * 1024,
                ),
                inflight_response_bytes: AtomicUsize::new(0),
            }),
        }
    }

    pub async fn claim(&self, identity: RequestIdentity) -> Claim {
        let RequestIdentity {
            record_key,
            request_hash,
        } = identity;

        loop {
            let wait = {
                let now = Instant::now();
                let mut state = self.inner.state.lock().await;
                prune_expired(&mut state, now);
                if let Some(entry) = state.entries.get(&record_key) {
                    if entry.request_hash != request_hash {
                        return Claim::Response(error_response(
                            StatusCode::CONFLICT,
                            "Idempotency-Key was already used for a different request",
                            "conflict",
                            false,
                        ));
                    }
                    if let Some(response) = entry.response.clone() {
                        return Claim::Response(replay_response(response));
                    }
                    if entry.nonreplayable_reason.is_some() {
                        return Claim::Response(error_response(
                            StatusCode::CONFLICT,
                            "the original request executed, but its response is no longer replayable",
                            "executed-nonreplayable",
                            false,
                        ));
                    }
                    Some(entry.notify.subscribe())
                } else {
                    while state.entries.len() >= self.inner.max_entries {
                        if !evict_oldest_completed(&mut state, None) {
                            return Claim::Response(error_response(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "idempotency coordinator capacity exhausted",
                                "capacity-exhausted",
                                true,
                            ));
                        }
                    }
                    let token = self.inner.next_owner.fetch_add(1, Ordering::Relaxed);
                    let (notify, _) = watch::channel(());
                    state.entries.insert(
                        record_key.clone(),
                        Entry {
                            request_hash: request_hash.clone(),
                            owner_token: Some(token),
                            notify,
                            response: None,
                            nonreplayable_reason: None,
                            completed_at: None,
                            expires_at: None,
                        },
                    );
                    return Claim::Owner(Owner {
                        coordinator: self.clone(),
                        record_key,
                        token,
                        armed: true,
                    });
                }
            };

            if let Some(mut wait) = wait {
                if tokio::time::timeout(self.inner.wait_timeout, wait.changed())
                    .await
                    .is_err()
                {
                    return Claim::Response(error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "timed out waiting for the original idempotent request",
                        "wait-timeout",
                        true,
                    ));
                }
            }
        }
    }

    pub async fn observability_snapshot(&self) -> serde_json::Value {
        let state = self.inner.state.lock().await;
        let completed = state
            .entries
            .values()
            .filter(|entry| entry.response.is_some())
            .count();
        let nonreplayable = state
            .entries
            .values()
            .filter(|entry| entry.nonreplayable_reason.is_some())
            .count();
        serde_json::json!({
            "entries":state.entries.len(),
            "inflight":state.entries.values().filter(|entry| entry.owner_token.is_some()).count(),
            "completed":completed,
            "nonreplayable":nonreplayable,
            "stored_bytes":state.stored_bytes,
            "inflight_response_bytes":self.inner.inflight_response_bytes.load(Ordering::Acquire),
            "max_entries":self.inner.max_entries,
            "max_stored_bytes":self.inner.max_stored_bytes,
            "max_response_bytes":self.inner.max_response_bytes,
        })
    }
}

impl RequestHasher {
    pub fn new(
        method: &Method,
        uri: &Uri,
        headers: &HeaderMap,
        idempotency_key: &str,
    ) -> Result<Self, Response<Body>> {
        if !valid_key(idempotency_key) {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "Idempotency-Key must contain 1-128 safe ASCII characters",
                "invalid-key",
                false,
            ));
        }
        let credential = ["authorization", "x-api-key"]
            .into_iter()
            .map(|name| joined_header_values(headers, name))
            .collect::<Vec<_>>()
            .join("\n");
        let credential_hash = Sha256::digest(credential.as_bytes());
        let method_bytes = method.as_str().as_bytes();
        let path_bytes = uri.path().as_bytes();
        let query = uri.query().unwrap_or_default().as_bytes();

        let mut record = Sha256::new();
        for (index, value) in [
            method_bytes,
            path_bytes,
            query,
            format!("{credential_hash:x}").as_bytes(),
            idempotency_key.as_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            if index > 0 {
                record.update([0]);
            }
            record.update(value);
        }

        let mut request = Sha256::new();
        request.update(method_bytes);
        for value in [
            path_bytes,
            query,
            joined_header_values(headers, "content-type").as_bytes(),
            joined_header_values(headers, "content-encoding").as_bytes(),
        ] {
            request.update([0]);
            request.update(value);
        }
        request.update([0]);
        Ok(Self {
            record_key: format!("{:x}", record.finalize()),
            request,
        })
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.request.update(bytes);
    }

    pub fn finish(self) -> RequestIdentity {
        RequestIdentity {
            record_key: self.record_key,
            request_hash: format!("{:x}", self.request.finalize()),
        }
    }
}

impl Owner {
    pub fn max_response_bytes(&self) -> usize {
        self.coordinator.inner.max_response_bytes
    }

    pub fn try_reserve_inflight_response(&self, bytes: usize) -> bool {
        try_reserve_atomic(
            &self.coordinator.inner.inflight_response_bytes,
            self.coordinator.inner.max_inflight_response_bytes,
            bytes,
        )
    }

    pub fn release_inflight_response(&self, bytes: usize) {
        atomic_saturating_sub(&self.coordinator.inner.inflight_response_bytes, bytes);
    }

    pub async fn complete(
        mut self,
        status: StatusCode,
        mut headers: HeaderMap,
        chunks: Vec<Bytes>,
        bytes: usize,
    ) {
        headers.remove(STATUS_HEADER);
        let cacheable_status =
            status.as_u16() <= 499 && !matches!(status.as_u16(), 408 | 425 | 429 | 499);
        if !cacheable_status {
            self.release_inner().await;
            return;
        }
        if bytes > self.coordinator.inner.max_response_bytes {
            self.nonreplayable_inner("response_too_large").await;
            return;
        }

        let now = Instant::now();
        let mut state = self.coordinator.inner.state.lock().await;
        prune_expired(&mut state, now);
        let valid_owner = state
            .entries
            .get(&self.record_key)
            .is_some_and(|entry| entry.owner_token == Some(self.token));
        if !valid_owner {
            self.armed = false;
            return;
        }
        while state.stored_bytes.saturating_add(bytes) > self.coordinator.inner.max_stored_bytes {
            if !evict_oldest_completed(&mut state, Some(&self.record_key)) {
                break;
            }
        }
        if state.stored_bytes.saturating_add(bytes) > self.coordinator.inner.max_stored_bytes {
            drop(state);
            self.nonreplayable_inner("spool_capacity_exhausted").await;
            return;
        }

        let notify = {
            let entry = state
                .entries
                .get_mut(&self.record_key)
                .expect("validated idempotency owner disappeared");
            entry.owner_token = None;
            entry.response = Some(CachedResponse {
                status,
                headers,
                chunks: Arc::new(chunks),
                bytes,
            });
            entry.completed_at = Some(now);
            entry.expires_at = Some(now + self.coordinator.inner.ttl);
            entry.notify.clone()
        };
        state.stored_bytes = state.stored_bytes.saturating_add(bytes);
        self.armed = false;
        drop(state);
        let _ = notify.send(());
    }

    pub async fn nonreplayable(mut self, reason: &str) {
        self.nonreplayable_inner(reason).await;
    }

    pub async fn release(mut self) {
        self.release_inner().await;
    }

    async fn nonreplayable_inner(&mut self, reason: &str) {
        let now = Instant::now();
        let mut state = self.coordinator.inner.state.lock().await;
        let notify = state
            .entries
            .get_mut(&self.record_key)
            .filter(|entry| entry.owner_token == Some(self.token))
            .map(|entry| {
                entry.owner_token = None;
                entry.nonreplayable_reason = Some(reason.to_owned());
                entry.completed_at = Some(now);
                entry.expires_at = Some(now + self.coordinator.inner.ttl);
                entry.notify.clone()
            });
        self.armed = false;
        drop(state);
        if let Some(notify) = notify {
            let _ = notify.send(());
        }
    }

    async fn release_inner(&mut self) {
        let mut state = self.coordinator.inner.state.lock().await;
        let notify = state
            .entries
            .get(&self.record_key)
            .filter(|entry| entry.owner_token == Some(self.token))
            .map(|entry| entry.notify.clone());
        if notify.is_some() {
            state.entries.remove(&self.record_key);
        }
        self.armed = false;
        drop(state);
        if let Some(notify) = notify {
            let _ = notify.send(());
        }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let coordinator = self.coordinator.clone();
        let record_key = self.record_key.clone();
        let token = self.token;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut state = coordinator.inner.state.lock().await;
                let notify = state
                    .entries
                    .get(&record_key)
                    .filter(|entry| entry.owner_token == Some(token))
                    .map(|entry| entry.notify.clone());
                if notify.is_some() {
                    state.entries.remove(&record_key);
                }
                drop(state);
                if let Some(notify) = notify {
                    let _ = notify.send(());
                }
            });
        }
    }
}

pub fn executed_header(headers: &mut HeaderMap) {
    headers.insert(STATUS_HEADER, HeaderValue::from_static("executed"));
}

pub fn response_from_bytes(
    status: StatusCode,
    mut headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    executed_header(&mut headers);
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

fn replay_response(response: CachedResponse) -> Response<Body> {
    let chunks = response.chunks.as_ref().clone();
    let body = Body::from_stream(stream::iter(
        chunks.into_iter().map(Ok::<Bytes, Infallible>),
    ));
    let mut output = Response::new(body);
    *output.status_mut() = response.status;
    *output.headers_mut() = response.headers;
    output
        .headers_mut()
        .insert(STATUS_HEADER, HeaderValue::from_static("replayed"));
    output
}

fn error_response(
    status: StatusCode,
    detail: &str,
    idempotency_status: &'static str,
    retry_after: bool,
) -> Response<Body> {
    let body = serde_json::json!({
        "error": {
            "message": detail,
            "type": "idempotency_error",
            "code": idempotency_status.replace('-', "_"),
        }
    })
    .to_string();
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    response
        .headers_mut()
        .insert(STATUS_HEADER, HeaderValue::from_static(idempotency_status));
    if retry_after {
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("1"));
    }
    response
}

fn valid_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

#[cfg(test)]
fn request_identity(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    idempotency_key: &str,
    body: &[u8],
) -> (String, String) {
    let mut hasher = RequestHasher::new(method, uri, headers, idempotency_key).unwrap();
    hasher.update(body);
    let identity = hasher.finish();
    (identity.record_key, identity.request_hash)
}

fn joined_header_values(headers: &HeaderMap, name: &str) -> String {
    headers
        .get_all(name)
        .iter()
        .map(|value| {
            value
                .as_bytes()
                .iter()
                .map(|byte| char::from(*byte))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn prune_expired(state: &mut State, now: Instant) {
    let expired = state
        .entries
        .iter()
        .filter(|(_, entry)| entry.expires_at.is_some_and(|deadline| deadline <= now))
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    for key in expired {
        if let Some(entry) = state.entries.remove(&key) {
            state.stored_bytes = state
                .stored_bytes
                .saturating_sub(entry.response.map_or(0, |response| response.bytes));
        }
    }
}

fn evict_oldest_completed(state: &mut State, exclude: Option<&str>) -> bool {
    let oldest = state
        .entries
        .iter()
        .filter(|(key, entry)| {
            exclude != Some(key.as_str())
                && entry.owner_token.is_none()
                && entry.completed_at.is_some()
        })
        .min_by_key(|(_, entry)| entry.completed_at)
        .map(|(key, _)| key.clone());
    let Some(key) = oldest else {
        return false;
    };
    if let Some(entry) = state.entries.remove(&key) {
        state.stored_bytes = state
            .stored_bytes
            .saturating_sub(entry.response.map_or(0, |response| response.bytes));
    }
    true
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_duration(name: &str, default_seconds: f64) -> Duration {
    let seconds = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default_seconds);
    Duration::from_secs_f64(seconds)
}

fn try_reserve_atomic(counter: &AtomicUsize, limit: usize, bytes: usize) -> bool {
    if bytes == 0 {
        return true;
    }
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(bytes).filter(|next| *next <= limit)
        })
        .is_ok()
}

fn atomic_saturating_sub(counter: &AtomicUsize, bytes: usize) {
    if bytes == 0 {
        return;
    }
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_sub(bytes))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_parts() -> (Method, Uri, HeaderMap) {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        (Method::POST, "/v1/responses".parse().unwrap(), headers)
    }

    fn identity(
        method: &Method,
        uri: &Uri,
        headers: &HeaderMap,
        key: &str,
        body: &[u8],
    ) -> RequestIdentity {
        let mut hasher = RequestHasher::new(method, uri, headers, key).unwrap();
        hasher.update(body);
        hasher.finish()
    }

    #[tokio::test]
    async fn replays_completed_response_and_rejects_conflicts() {
        let coordinator = Coordinator::new();
        let (method, uri, headers) = request_parts();
        let Claim::Owner(owner) = coordinator
            .claim(identity(&method, &uri, &headers, "request-1", b"one"))
            .await
        else {
            panic!("first claim was not owner");
        };
        owner
            .complete(
                StatusCode::OK,
                HeaderMap::new(),
                vec![Bytes::from_static(b"event\n\n")],
                7,
            )
            .await;
        let Claim::Response(replay) = coordinator
            .claim(identity(&method, &uri, &headers, "request-1", b"one"))
            .await
        else {
            panic!("completed claim was not replayed");
        };
        assert_eq!(replay.status(), StatusCode::OK);
        assert_eq!(replay.headers()[STATUS_HEADER], "replayed");

        let Claim::Response(conflict) = coordinator
            .claim(identity(&method, &uri, &headers, "request-1", b"two"))
            .await
        else {
            panic!("conflicting claim became owner");
        };
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(conflict.headers()[STATUS_HEADER], "conflict");
    }

    #[tokio::test]
    async fn waiters_resume_after_owner_release() {
        let coordinator = Coordinator::new();
        let (method, uri, headers) = request_parts();
        let Claim::Owner(owner) = coordinator
            .claim(identity(&method, &uri, &headers, "request-2", b"body"))
            .await
        else {
            panic!("first claim was not owner");
        };
        let waiting = {
            let coordinator = coordinator.clone();
            let method = method.clone();
            let uri = uri.clone();
            let headers = headers.clone();
            tokio::spawn(async move {
                coordinator
                    .claim(identity(&method, &uri, &headers, "request-2", b"body"))
                    .await
            })
        };
        tokio::task::yield_now().await;
        owner.release().await;
        assert!(matches!(waiting.await.unwrap(), Claim::Owner(_)));
    }

    #[test]
    fn request_identity_matches_python_contract() {
        let (method, _, headers) = request_parts();
        let uri = "/v1/responses?a=1%202".parse().unwrap();
        let (record_key, request_hash) =
            request_identity(&method, &uri, &headers, "request-1", br#"{"stream":true}"#);
        assert_eq!(
            record_key,
            "c31e7b1fed4be19823949def8ea72342080e37834f07ff37aca9fed9ecf92396"
        );
        assert_eq!(
            request_hash,
            "ad850636a8a309bc388ee7ec10afc635d47d2d9577dd5572f1be3d92bd2c4ef6"
        );
    }
}
