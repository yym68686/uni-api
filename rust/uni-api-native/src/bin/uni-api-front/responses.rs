use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use memchr::{memchr2, memmem};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::idempotency;
use crate::proxy::{filtered_response_headers, json_error, AppState};
use crate::responses_item_ids::{event_item_id_needs_normalization, ResponsesItemIdNormalizer};
use crate::responses_native::NativeRoute;

const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";
const MAX_ERROR_BODY_BYTES: usize = 1024 * 1024;
const DOWNSTREAM_SEGMENT_BYTES: usize = 64 * 1024;
const DOWNSTREAM_CHANNEL_SEGMENTS: usize = 16;
pub(crate) const UNLIMITED_SSE_EVENT_BYTES: usize = 0;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Plan {
    pub(crate) attempt_id: String,
    pub(crate) url: String,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) body: String,
    pub(crate) proxy: Option<String>,
    pub(crate) engine: String,
    #[serde(default)]
    pub(crate) precommit_semantic_guard: Option<bool>,
    #[serde(default)]
    pub(crate) http1_only: bool,
    #[serde(default = "default_commit_policy")]
    pub(crate) commit_policy: String,
    #[serde(default)]
    pub(crate) normalize_custom_tool_call_ids: bool,
    #[serde(default)]
    pub(crate) connect_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub(crate) write_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub(crate) pool_timeout_seconds: Option<f64>,
    pub(crate) first_byte_timeout_seconds: Option<f64>,
    pub(crate) idle_timeout_seconds: Option<f64>,
    pub(crate) total_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub(crate) provider_name: Option<String>,
    #[serde(default)]
    pub(crate) provider_key: Option<String>,
    #[serde(default)]
    pub(crate) original_model: Option<String>,
    #[serde(default = "default_max_event_bytes")]
    pub(crate) max_event_bytes: usize,
    #[serde(default = "default_max_precommit_items")]
    pub(crate) max_precommit_items: usize,
    #[serde(default = "default_max_precommit_bytes")]
    pub(crate) max_precommit_bytes: usize,
}

fn default_commit_policy() -> String {
    "real_output".to_owned()
}

fn default_max_event_bytes() -> usize {
    UNLIMITED_SSE_EVENT_BYTES
}

fn default_max_precommit_items() -> usize {
    128
}

fn default_max_precommit_bytes() -> usize {
    8 * 1024 * 1024 + 128 * 266
}

fn precommit_semantic_guard(plan: &Plan) -> bool {
    plan.precommit_semantic_guard
        .unwrap_or_else(|| plan.engine.eq_ignore_ascii_case("codex"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamMode {
    OpaqueRaw,
    GuardedThenRaw,
    SelectiveRewrite,
}

impl StreamMode {
    fn for_plan(plan: &Plan) -> Self {
        if plan.normalize_custom_tool_call_ids {
            Self::SelectiveRewrite
        } else if precommit_semantic_guard(plan) {
            Self::GuardedThenRaw
        } else {
            Self::OpaqueRaw
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::OpaqueRaw => "opaque_raw",
            Self::GuardedThenRaw => "guarded_then_raw",
            Self::SelectiveRewrite => "selective_rewrite",
        }
    }

    fn relays_raw_after_commit(self) -> bool {
        matches!(self, Self::OpaqueRaw | Self::GuardedThenRaw)
    }
}

struct StreamStats {
    stream_mode: &'static str,
    upstream_bytes: u64,
    upstream_chunks: u64,
    downstream_bytes: u64,
    downstream_chunks: u64,
    event_count: u64,
    delta_events: u64,
    normalized_events: u64,
    usage: Option<Value>,
    wire_hash: Option<Sha256>,
}

impl StreamStats {
    fn new(attempt_id: &str) -> Self {
        let sample_bps = wire_hash_sample_bps();
        let mut sampler = DefaultHasher::new();
        attempt_id.hash(&mut sampler);
        let sampled = sample_bps >= 10_000 || sampler.finish() % 10_000 < sample_bps;
        Self {
            stream_mode: "unknown",
            upstream_bytes: 0,
            upstream_chunks: 0,
            downstream_bytes: 0,
            downstream_chunks: 0,
            event_count: 0,
            delta_events: 0,
            normalized_events: 0,
            usage: None,
            wire_hash: sampled.then(Sha256::new),
        }
    }

    fn report(&self) -> Value {
        let hash = self
            .wire_hash
            .as_ref()
            .map(|hasher| format!("{:x}", hasher.clone().finalize()));
        json!({
            "upstream_bytes": self.upstream_bytes,
            "upstream_chunks": self.upstream_chunks,
            "downstream_bytes": self.downstream_bytes,
            "downstream_chunks": self.downstream_chunks,
            "event_count": self.event_count,
            "delta_events": self.delta_events,
            "normalized_events": self.normalized_events,
            "usage": self.usage,
            "wire_sha256": hash,
            "wire_hash_sampled": self.wire_hash.is_some(),
            "stream_mode": self.stream_mode,
        })
    }

    fn observe_upstream(&mut self, chunk: &[u8]) {
        self.upstream_bytes = self.upstream_bytes.saturating_add(chunk.len() as u64);
        self.upstream_chunks = self.upstream_chunks.saturating_add(1);
    }

    fn observe_wire(&mut self, wire: &[u8]) {
        self.downstream_bytes = self.downstream_bytes.saturating_add(wire.len() as u64);
        self.downstream_chunks = self.downstream_chunks.saturating_add(1);
        if let Some(hasher) = self.wire_hash.as_mut() {
            hasher.update(wire);
        }
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self::new("")
    }
}

fn wire_hash_sample_bps() -> u64 {
    static SAMPLE_BPS: OnceLock<u64> = OnceLock::new();
    *SAMPLE_BPS.get_or_init(|| {
        std::env::var("RUST_RESPONSES_WIRE_HASH_SAMPLE_BPS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(100)
            .min(10_000)
    })
}

#[derive(Debug)]
enum Terminal {
    Completed,
    Incomplete,
    SemanticFailure { event_type: String, payload: Value },
}

struct ActiveAttempt {
    plan: Plan,
    status: StatusCode,
    headers: HeaderMap,
    stream: ByteStream,
    decoder: SseDecoder,
    processor: ResponsesProcessor,
    mode: StreamMode,
    early_output: Vec<Bytes>,
    buffered: Vec<Bytes>,
    stats: StreamStats,
    terminal: Option<Terminal>,
    total_deadline: Option<tokio::time::Instant>,
    commit_reason: &'static str,
    precommit_keepalive_sent: bool,
    business_committed: bool,
    raw_pending_forwarded: bool,
}

enum PreflightResult {
    Retry(Value),
    Started(ActiveAttempt),
}

enum HedgeSignal {
    Trigger,
    Started { plan: Plan, active: ActiveAttempt },
    Retry { plan: Plan, outcome: Value },
}

enum Coordinator {
    Python { session_id: String },
    Native { route: NativeRoute },
}

impl Coordinator {
    fn is_native(&self) -> bool {
        matches!(self, Self::Native { .. })
    }

    async fn commit(&mut self, state: &AppState, observation: &Value) -> Result<(), String> {
        match self {
            Self::Python { session_id } => control_commit(state, session_id, observation)
                .await
                .map(|_| ()),
            Self::Native { .. } => Ok(()),
        }
    }

    async fn complete(&mut self, state: &AppState, outcome: &Value) -> Option<Value> {
        match self {
            Self::Python { session_id } => control_complete(state, session_id, outcome).await.ok(),
            Self::Native { route } => {
                route.complete_native(outcome).await;
                None
            }
        }
    }

    async fn retry(
        &mut self,
        state: &AppState,
        mut outcome: Value,
    ) -> Result<RetryResolution, String> {
        match self {
            Self::Python { session_id } => {
                retry_after_public_start_python(state, session_id, outcome).await
            }
            Self::Native { route } => loop {
                if !route.record_failure(&outcome).await {
                    return Ok(RetryResolution::Final(route.final_message()));
                }
                let plan = match route.next_plan().await {
                    Ok(Some(plan)) => plan,
                    Ok(None) => return Ok(RetryResolution::Final(route.final_message())),
                    Err(error) => {
                        route.emit_internal_failure(502, "native_retry_plan_error", &error);
                        return Err(error);
                    }
                };
                match preflight_attempt(state, plan, true).await {
                    Ok(PreflightResult::Started(active)) => {
                        return Ok(RetryResolution::Active(active));
                    }
                    Ok(PreflightResult::Retry(next)) => outcome = next,
                    Err(error) => {
                        outcome = json!({
                            "kind": "protocol_error",
                            "status_code": 502,
                            "detail": error,
                            "committed": false,
                        });
                    }
                }
            },
        }
    }
}

fn spawn_hedge_attempt(
    state: AppState,
    plan: Plan,
    sender: mpsc::UnboundedSender<HedgeSignal>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let result =
            preflight_attempt_with_trigger(&state, plan.clone(), false, Some(&sender)).await;
        let signal = match result {
            Ok(PreflightResult::Started(active)) => HedgeSignal::Started { plan, active },
            Ok(PreflightResult::Retry(outcome)) => HedgeSignal::Retry { plan, outcome },
            Err(error) => HedgeSignal::Retry {
                plan,
                outcome: json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                }),
            },
        };
        let _ = sender.send(signal);
    })
}

async fn preflight_native_hedged(
    state: &AppState,
    route: &mut NativeRoute,
) -> Result<Option<ActiveAttempt>, String> {
    let max_inflight = route.hedge_slots().max(1);
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let mut running = HashMap::<String, tokio::task::JoinHandle<()>>::new();

    let Some(plan) = route.next_plan().await? else {
        return Ok(None);
    };
    let attempt_id = plan.attempt_id.clone();
    let handle = spawn_hedge_attempt(state.clone(), plan, sender.clone());
    running.insert(attempt_id, handle);

    while let Some(signal) = receiver.recv().await {
        match signal {
            HedgeSignal::Trigger => {
                route.record_hedge_trigger();
                if running.len() < max_inflight {
                    if let Some(next) = route.next_plan().await? {
                        let attempt_id = next.attempt_id.clone();
                        let handle = spawn_hedge_attempt(state.clone(), next, sender.clone());
                        running.insert(attempt_id, handle);
                    }
                }
            }
            HedgeSignal::Started { plan, active } => {
                route.record_hedge_cancellations(running.len().saturating_sub(1));
                for (_, handle) in running.drain() {
                    handle.abort();
                }
                route.set_current_plan(&plan);
                return Ok(Some(active));
            }
            HedgeSignal::Retry { plan, outcome } => {
                running.remove(&plan.attempt_id);
                let retryable = route.record_failure_for(&plan, &outcome).await;
                if retryable && running.len() < max_inflight {
                    if let Some(next) = route.next_plan().await? {
                        let attempt_id = next.attempt_id.clone();
                        let handle = spawn_hedge_attempt(state.clone(), next, sender.clone());
                        running.insert(attempt_id, handle);
                    }
                }
                if running.is_empty() {
                    return Ok(None);
                }
            }
        }
    }
    Ok(None)
}

pub async fn serve_native(
    state: AppState,
    mut route: NativeRoute,
    mut idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    if !route.stream() {
        return serve_native_nonstream(state, route, idempotency_owner).await;
    }
    if route.hedging_enabled() && route.stream() {
        match preflight_native_hedged(&state, &mut route).await {
            Ok(Some(active)) => {
                let mut control_headers = HeaderMap::new();
                if let Ok(value) = HeaderValue::from_str(route.request_id()) {
                    control_headers.insert("x-request-id", value);
                }
                control_headers
                    .insert("access-control-allow-origin", HeaderValue::from_static("*"));
                return start_public_stream(
                    state,
                    Coordinator::Native { route },
                    active,
                    control_headers,
                    None,
                    idempotency_owner.take(),
                );
            }
            Ok(None) => {
                let status =
                    StatusCode::from_u16(route.last_status()).unwrap_or(StatusCode::BAD_GATEWAY);
                route.emit_final_response(status.as_u16(), "native_hedge_exhausted");
                release_owner(&mut idempotency_owner).await;
                return json_error(status, route.last_detail());
            }
            Err(error) => {
                route.emit_internal_failure(502, "native_hedge_error", &error);
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_GATEWAY, &error);
            }
        }
    }
    loop {
        let plan = match route.next_plan().await {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                let status =
                    StatusCode::from_u16(route.last_status()).unwrap_or(StatusCode::BAD_GATEWAY);
                route.emit_final_response(status.as_u16(), "native_route_exhausted");
                release_owner(&mut idempotency_owner).await;
                return json_error(status, route.last_detail());
            }
            Err(error) if error == "native-codex-oauth-fallback" => {
                route.emit_internal_failure(502, "native_codex_oauth_fallback", &error);
                release_owner(&mut idempotency_owner).await;
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "Native route encountered unsupported Codex OAuth",
                );
            }
            Err(error) => {
                route.emit_internal_failure(400, "native_plan_error", &error);
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_REQUEST, &error);
            }
        };
        match preflight_attempt(&state, plan, false).await {
            Ok(PreflightResult::Started(active)) => {
                let mut control_headers = HeaderMap::new();
                if let Ok(value) = HeaderValue::from_str(route.request_id()) {
                    control_headers.insert("x-request-id", value);
                }
                control_headers
                    .insert("access-control-allow-origin", HeaderValue::from_static("*"));
                return start_public_stream(
                    state,
                    Coordinator::Native { route },
                    active,
                    control_headers,
                    None,
                    idempotency_owner.take(),
                );
            }
            Ok(PreflightResult::Retry(outcome)) => {
                if !route.record_failure(&outcome).await {
                    let status = StatusCode::from_u16(route.last_status())
                        .unwrap_or(StatusCode::BAD_GATEWAY);
                    route.emit_final_response(status.as_u16(), "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(status, route.last_detail());
                }
            }
            Err(error) => {
                let outcome = json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
                if !route.record_failure(&outcome).await {
                    route.emit_final_response(502, "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(StatusCode::BAD_GATEWAY, route.last_detail());
                }
            }
        }
    }
}

async fn serve_native_nonstream(
    state: AppState,
    mut route: NativeRoute,
    mut idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    loop {
        let plan = match route.next_plan().await {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                let status =
                    StatusCode::from_u16(route.last_status()).unwrap_or(StatusCode::BAD_GATEWAY);
                route.emit_final_response(status.as_u16(), "native_route_exhausted");
                release_owner(&mut idempotency_owner).await;
                return json_error(status, route.last_detail());
            }
            Err(error) => {
                route.emit_internal_failure(400, "native_plan_error", &error);
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_REQUEST, &error);
            }
        };
        match send_native_nonstream_attempt(&state, &plan).await {
            Ok((status, mut headers, mut body)) if status.is_success() => {
                let usage = serde_json::from_slice::<Value>(&body)
                    .ok()
                    .and_then(|payload| {
                        payload
                            .get("usage")
                            .filter(|value| value.is_object())
                            .cloned()
                    });
                if plan.normalize_custom_tool_call_ids {
                    if let Ok(mut payload) = serde_json::from_slice::<Value>(&body) {
                        let mut normalizer = ResponsesItemIdNormalizer::default();
                        if normalizer.normalize(&mut payload).unwrap_or(false) {
                            body = serde_json::to_vec(&payload).unwrap_or(body);
                        }
                    }
                }
                headers.insert(
                    "x-uni-api-data-plane",
                    HeaderValue::from_static("rust-native-v2"),
                );
                if let Ok(value) = HeaderValue::from_str(route.request_id()) {
                    headers.insert("x-request-id", value);
                }
                headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
                if let Some(owner) = idempotency_owner.take() {
                    let bytes = Bytes::from(body.clone());
                    if body.len() <= owner.max_response_bytes()
                        && owner.try_reserve_inflight_response(body.len())
                    {
                        owner.release_inflight_response(body.len());
                        owner
                            .complete(status, headers.clone(), vec![bytes], body.len())
                            .await;
                        idempotency::executed_header(&mut headers);
                    } else {
                        owner.nonreplayable("response_too_large").await;
                        idempotency::executed_header(&mut headers);
                    }
                }
                let mut outcome = json!({
                    "kind": "completed",
                    "status_code": status.as_u16(),
                    "upstream_status_code": status.as_u16(),
                    "downstream_bytes": body.len(),
                });
                if let Some(usage) = usage {
                    outcome["usage"] = usage;
                }
                route.complete_native(&outcome).await;
                let mut response = Response::new(Body::from(body));
                *response.status_mut() = status;
                *response.headers_mut() = headers;
                return response;
            }
            Ok((status, _headers, body)) => {
                let detail = String::from_utf8_lossy(&body)
                    .chars()
                    .take(4096)
                    .collect::<String>();
                let outcome = json!({
                    "kind": "http_error",
                    "status_code": status.as_u16(),
                    "upstream_status_code": status.as_u16(),
                    "body": detail,
                    "committed": false,
                });
                if !route.record_failure(&outcome).await {
                    let final_status = StatusCode::from_u16(route.last_status())
                        .unwrap_or(StatusCode::BAD_GATEWAY);
                    route.emit_final_response(final_status.as_u16(), "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(final_status, route.last_detail());
                }
            }
            Err(error) => {
                let outcome = json!({
                    "kind": "transport_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
                if !route.record_failure(&outcome).await {
                    route.emit_final_response(502, "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(StatusCode::BAD_GATEWAY, route.last_detail());
                }
            }
        }
    }
}

async fn send_native_nonstream_attempt(
    state: &AppState,
    plan: &Plan,
) -> Result<(StatusCode, HeaderMap, Vec<u8>), String> {
    let client = state
        .upstream_client(
            plan.proxy.as_deref(),
            plan.http1_only,
            positive_duration(plan.connect_timeout_seconds),
        )
        .await?;
    let mut headers = HeaderMap::new();
    for (name, value) in &plan.headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| "native plan contains an invalid header name".to_owned())?;
        let value = HeaderValue::from_str(value)
            .map_err(|_| "native plan contains an invalid header value".to_owned())?;
        headers.append(name, value);
    }
    headers
        .entry("accept-encoding")
        .or_insert(HeaderValue::from_static("identity"));
    let timeout = earliest_timeout(&[
        plan.write_timeout_seconds,
        plan.pool_timeout_seconds,
        plan.first_byte_timeout_seconds,
        plan.total_timeout_seconds,
    ]);
    let request = client
        .post(&plan.url)
        .headers(headers)
        .body(plan.body.clone());
    let response = if let Some(timeout) = timeout {
        tokio::time::timeout(timeout, request.send())
            .await
            .map_err(|_| "upstream non-streaming response timed out".to_owned())?
            .map_err(|error| format!("upstream non-streaming request failed: {error}"))?
    } else {
        request
            .send()
            .await
            .map_err(|error| format!("upstream non-streaming request failed: {error}"))?
    };
    let status = response.status();
    let headers = filtered_response_headers(response.headers());
    let body = response
        .bytes()
        .await
        .map_err(|error| format!("read upstream non-streaming body: {error}"))?
        .to_vec();
    Ok((status, headers, body))
}

pub async fn serve_session(
    state: AppState,
    session_id: String,
    control_response: reqwest::Response,
    mut idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let public_control_headers = public_control_headers(control_response.headers());
    let mut control_response = Some(control_response);

    let mut message = match control_get_plan(&state, &session_id).await {
        Ok(message) => message,
        Err(error) => {
            release_owner(&mut idempotency_owner).await;
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Rust Responses control plan failed: {error}"),
            );
        }
    };

    loop {
        if message.get("kind").and_then(Value::as_str) == Some("final") {
            return response_from_final(
                &message,
                &public_control_headers,
                idempotency_owner.take(),
            )
            .await;
        }
        let plan = match serde_json::from_value::<Plan>(message.clone()) {
            Ok(plan) if message.get("kind").and_then(Value::as_str) == Some("plan") => plan,
            Ok(_) | Err(_) => {
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_GATEWAY, "Invalid Rust Responses plan");
            }
        };
        match preflight_attempt(&state, plan, false).await {
            Ok(PreflightResult::Retry(mut outcome)) => {
                outcome["attempt_id"] = Value::String(
                    message
                        .get("attempt_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                );
                message = match control_advance(&state, &session_id, &outcome).await {
                    Ok(next) => next,
                    Err(error) => {
                        release_owner(&mut idempotency_owner).await;
                        return json_error(
                            StatusCode::BAD_GATEWAY,
                            &format!("Rust Responses retry coordination failed: {error}"),
                        );
                    }
                };
            }
            Ok(PreflightResult::Started(active)) => {
                let observation = json!({
                    "attempt_id": active.plan.attempt_id,
                    "upstream_status_code": active.status.as_u16(),
                    "commit_reason": active.commit_reason,
                    "business_committed": active.business_committed,
                    "precommit_events": active.stats.event_count,
                    "precommit_bytes": active.buffered.iter().map(|item| item.len() as u64).sum::<u64>(),
                });
                if let Err(error) = control_commit(&state, &session_id, &observation).await {
                    release_owner(&mut idempotency_owner).await;
                    return json_error(
                        StatusCode::BAD_GATEWAY,
                        &format!("Rust Responses commit coordination failed: {error}"),
                    );
                }
                let control_drain = control_response.take().map(|response| {
                    tokio::spawn(async move {
                        let mut stream = response.bytes_stream();
                        while let Some(result) = stream.next().await {
                            if result.is_err() {
                                break;
                            }
                        }
                    })
                });
                return start_public_stream(
                    state,
                    Coordinator::Python { session_id },
                    active,
                    public_control_headers,
                    control_drain,
                    idempotency_owner.take(),
                );
            }
            Err(error) => {
                let outcome = json!({
                    "attempt_id": message.get("attempt_id").and_then(Value::as_str).unwrap_or_default(),
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
                message = match control_advance(&state, &session_id, &outcome).await {
                    Ok(next) => next,
                    Err(control_error) => {
                        release_owner(&mut idempotency_owner).await;
                        return json_error(
                            StatusCode::BAD_GATEWAY,
                            &format!("Rust Responses preflight failed: {control_error}"),
                        );
                    }
                };
            }
        }
    }
}

async fn preflight_attempt(
    state: &AppState,
    plan: Plan,
    keepalive_already_sent: bool,
) -> Result<PreflightResult, String> {
    preflight_attempt_with_trigger(state, plan, keepalive_already_sent, None).await
}

async fn preflight_attempt_with_trigger(
    state: &AppState,
    plan: Plan,
    keepalive_already_sent: bool,
    trigger: Option<&mpsc::UnboundedSender<HedgeSignal>>,
) -> Result<PreflightResult, String> {
    let client = state
        .upstream_client(
            plan.proxy.as_deref(),
            plan.http1_only,
            positive_duration(plan.connect_timeout_seconds),
        )
        .await?;
    let mut headers = HeaderMap::new();
    for (name, value) in &plan.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            return Err("upstream plan contains an invalid header name".into());
        };
        let Ok(value) = HeaderValue::from_str(value) else {
            return Err("upstream plan contains an invalid header value".into());
        };
        headers.append(name, value);
    }
    headers
        .entry("accept-encoding")
        .or_insert(HeaderValue::from_static("identity"));

    let started_at = tokio::time::Instant::now();
    let first_deadline = deadline(started_at, plan.first_byte_timeout_seconds);
    let total_deadline = deadline(started_at, plan.total_timeout_seconds);
    let send_stage_timeout =
        earliest_timeout(&[plan.write_timeout_seconds, plan.pool_timeout_seconds]);
    let send_stage_deadline = send_stage_timeout.map(|timeout| started_at + timeout);
    let send_deadline = earlier_deadline(
        earlier_deadline(first_deadline, total_deadline),
        send_stage_deadline,
    );
    let request = client
        .post(&plan.url)
        .headers(headers)
        .body(plan.body.clone());
    let mut request_future = Box::pin(request.send());
    let mut hedge_triggered = false;
    let response = if let Some(trigger) = trigger {
        let first_can_trigger = first_deadline.is_some_and(|first| {
            send_stage_deadline.is_none_or(|stage| first < stage)
                && total_deadline.is_none_or(|total| first < total)
        });
        if first_can_trigger {
            tokio::select! {
                result = &mut request_future => result
                    .map_err(|error| format!("upstream response headers failed: {error}"))?,
                _ = tokio::time::sleep_until(first_deadline.expect("checked first deadline")) => {
                    hedge_triggered = true;
                    let _ = trigger.send(HedgeSignal::Trigger);
                    await_deadline(
                        &mut request_future,
                        earlier_deadline(total_deadline, send_stage_deadline),
                    )
                    .await
                    .map_err(|error| format!("upstream response headers failed: {error}"))?
                    .map_err(|error| format!("upstream response headers failed: {error}"))?
                }
            }
        } else {
            await_deadline(&mut request_future, send_deadline)
                .await
                .map_err(|error| format!("upstream response headers failed: {error}"))?
                .map_err(|error| format!("upstream response headers failed: {error}"))?
        }
    } else {
        await_deadline(&mut request_future, send_deadline)
            .await
            .map_err(|error| format!("upstream response headers failed: {error}"))?
            .map_err(|error| format!("upstream response headers failed: {error}"))?
    };
    let status = response.status();
    let unsupported_encoding = response
        .headers()
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"));
    let response_headers = filtered_response_headers(response.headers());
    if !status.is_success() {
        let body = read_limited_body(response, total_deadline).await;
        return Ok(PreflightResult::Retry(json!({
            "kind": "http_error",
            "status_code": status.as_u16(),
            "upstream_status_code": status.as_u16(),
            "body": body,
            "committed": false,
        })));
    }
    if unsupported_encoding {
        return Ok(PreflightResult::Retry(json!({
            "kind": "protocol_error",
            "status_code": 502,
            "upstream_status_code": status.as_u16(),
            "detail": "Responses upstream ignored Accept-Encoding: identity",
            "committed": false,
        })));
    }

    let mode = StreamMode::for_plan(&plan);
    let mut stats = StreamStats::new(&plan.attempt_id);
    stats.stream_mode = mode.as_str();
    let stream = Box::pin(response.bytes_stream());
    let mut active = ActiveAttempt {
        decoder: SseDecoder::new(plan.max_event_bytes),
        processor: ResponsesProcessor::new(
            plan.commit_policy.clone(),
            plan.normalize_custom_tool_call_ids,
        ),
        mode,
        plan: plan.clone(),
        status,
        headers: response_headers,
        stream,
        early_output: Vec::new(),
        buffered: Vec::new(),
        stats,
        terminal: None,
        total_deadline,
        commit_reason: "real_output",
        precommit_keepalive_sent: keepalive_already_sent,
        business_committed: false,
        raw_pending_forwarded: false,
    };

    if active.mode == StreamMode::OpaqueRaw {
        active.commit_reason = "upstream_http_200";
        active.business_committed = true;
        return Ok(PreflightResult::Started(active));
    }
    loop {
        let next_deadline = earlier_deadline(
            (!hedge_triggered).then_some(first_deadline).flatten(),
            active.total_deadline,
        );
        let chunk = match await_deadline(active.stream.next(), next_deadline).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(error))) => {
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 502,
                    "upstream_status_code": status.as_u16(),
                    "detail": format!("upstream stream read failed: {error}"),
                    "committed": false,
                })))
            }
            Ok(None) => {
                let frames = active.decoder.finish()?;
                if !frames.is_empty() {
                    if let Some(result) = process_preflight_frames(&mut active, frames)? {
                        return Ok(result);
                    }
                }
                return Ok(PreflightResult::Retry(json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "upstream_status_code": status.as_u16(),
                    "detail": "Responses upstream closed before substantive output",
                    "committed": false,
                })));
            }
            Err(error) => {
                if trigger.is_some()
                    && first_deadline.is_some()
                    && !hedge_triggered
                    && error == "upstream deadline exceeded"
                {
                    hedge_triggered = true;
                    if let Some(trigger) = trigger {
                        let _ = trigger.send(HedgeSignal::Trigger);
                    }
                    continue;
                }
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 504,
                    "upstream_status_code": status.as_u16(),
                    "detail": error,
                    "committed": false,
                })));
            }
        };
        active.stats.observe_upstream(&chunk);
        let frames = active.decoder.feed(&chunk)?;
        if let Some(result) = process_preflight_frames(&mut active, frames)? {
            return Ok(result);
        }
    }
}

fn process_preflight_frames(
    active: &mut ActiveAttempt,
    frames: Vec<SseFrame>,
) -> Result<Option<PreflightResult>, String> {
    let mut started = false;
    let semantic_guard = precommit_semantic_guard(&active.plan);
    for frame in frames {
        let processed = active.processor.process(frame, &mut active.stats)?;
        if let Some(Terminal::SemanticFailure {
            event_type,
            payload,
        }) = processed.terminal.as_ref()
        {
            if !started {
                return Ok(Some(PreflightResult::Retry(semantic_failure_outcome(
                    active, event_type, payload, false,
                ))));
            }
            if active.business_committed {
                if let Some(wire) = processed.wire {
                    active.buffered.push(wire);
                }
            }
            active.terminal = processed.terminal;
            return Ok(Some(PreflightResult::Started(take_active(active)?)));
        }
        let early_keepalive =
            !active.precommit_keepalive_sent && processed.canonical_keepalive && semantic_guard;
        let suppress_repeated_keepalive =
            active.precommit_keepalive_sent && processed.event_type.as_deref() == Some("keepalive");
        let transparent_response_created =
            !semantic_guard && processed.event_type.as_deref() == Some("response.created");
        let commits_response =
            processed.commits || transparent_response_created || processed.terminal.is_some();
        if processed.event_type.as_deref() == Some("response.created")
            && semantic_guard
            && !active.precommit_keepalive_sent
        {
            active.buffered.push(Bytes::from_static(
                b"event: keepalive\ndata: {\"type\":\"keepalive\",\"sequence_number\":0}\n\n",
            ));
            active.precommit_keepalive_sent = true;
        }
        if let Some(wire) = processed.wire.filter(|_| !suppress_repeated_keepalive) {
            let buffered_bytes: usize = active.buffered.iter().map(Bytes::len).sum();
            if !started
                && !commits_response
                && (active.buffered.len() >= active.plan.max_precommit_items
                    || buffered_bytes.saturating_add(wire.len()) > active.plan.max_precommit_bytes)
            {
                return Ok(Some(PreflightResult::Retry(json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "upstream_status_code": active.status.as_u16(),
                    "detail": "Responses upstream precommit buffer limit exceeded",
                    "committed": false,
                }))));
            }
            if early_keepalive {
                active.early_output.push(wire);
            } else {
                active.buffered.push(wire);
            }
        }
        if processed.event_type.as_deref() == Some("keepalive") {
            active.precommit_keepalive_sent = true;
        }
        if let Some(terminal) = processed.terminal {
            active.terminal = Some(terminal);
            active.commit_reason = "semantic_terminal";
            active.business_committed = true;
            return Ok(Some(PreflightResult::Started(take_active(active)?)));
        }
        if early_keepalive {
            if !started {
                active.commit_reason = "upstream_keepalive";
            }
            started = true;
        }
        if processed.commits || transparent_response_created {
            if !started {
                active.commit_reason = if transparent_response_created {
                    "response_created"
                } else {
                    "real_output"
                };
            }
            started = true;
            active.business_committed = true;
        }
    }
    if started {
        Ok(Some(PreflightResult::Started(take_active(active)?)))
    } else {
        Ok(None)
    }
}

fn take_active(active: &mut ActiveAttempt) -> Result<ActiveAttempt, String> {
    stage_raw_pending(active);
    let placeholder = ActiveAttempt {
        plan: active.plan.clone(),
        status: active.status,
        headers: HeaderMap::new(),
        stream: Box::pin(futures_util::stream::empty()),
        decoder: SseDecoder::new(active.plan.max_event_bytes),
        processor: ResponsesProcessor::new("real_output".into(), false),
        mode: active.mode,
        early_output: Vec::new(),
        buffered: Vec::new(),
        stats: StreamStats::new(&active.plan.attempt_id),
        terminal: None,
        total_deadline: active.total_deadline,
        commit_reason: active.commit_reason,
        precommit_keepalive_sent: active.precommit_keepalive_sent,
        business_committed: active.business_committed,
        raw_pending_forwarded: false,
    };
    Ok(std::mem::replace(active, placeholder))
}

fn stage_raw_pending(active: &mut ActiveAttempt) {
    if active.business_committed
        && active.mode.relays_raw_after_commit()
        && !active.raw_pending_forwarded
    {
        if let Some(pending) = active.decoder.pending_copy() {
            active.buffered.push(pending);
        }
        active.raw_pending_forwarded = true;
    }
}

fn start_public_stream(
    state: AppState,
    coordinator: Coordinator,
    active: ActiveAttempt,
    control_headers: HeaderMap,
    control_drain: Option<tokio::task::JoinHandle<()>>,
    idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let status = active.status;
    let mut headers = active.headers.clone();
    for (name, value) in &control_headers {
        headers.insert(name.clone(), value.clone());
    }
    headers
        .entry("content-type")
        .or_insert(HeaderValue::from_static("text/event-stream"));
    headers.insert(
        "x-uni-api-data-plane",
        HeaderValue::from_static(match &coordinator {
            Coordinator::Native { .. } => "rust-native-v2",
            Coordinator::Python { .. } => "rust-v1",
        }),
    );

    let capture_headers = headers.clone();
    let capture = idempotency_owner.map(|owner| IdempotencyCapture {
        max_bytes: owner.max_response_bytes(),
        owner,
        status,
        headers: capture_headers,
        chunks: Vec::new(),
        bytes: 0,
        overflowed: false,
    });
    if capture.is_some() {
        idempotency::executed_header(&mut headers);
    }

    let (sender, receiver) = mpsc::channel(DOWNSTREAM_CHANNEL_SEGMENTS);
    let cancellation = CancellationToken::new();
    let finished = Arc::new(AtomicBool::new(false));
    let body_stream = GuardedBodyStream {
        inner: ReceiverStream::new(receiver),
        cancellation: cancellation.clone(),
        finished: finished.clone(),
    };
    let runtime = ActiveRuntime {
        output: OutputSink::new(sender, cancellation.clone(), capture),
        finished,
        control_drain,
    };
    tokio::spawn(run_active_attempt(state, coordinator, active, runtime));

    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

struct IdempotencyCapture {
    owner: idempotency::Owner,
    status: StatusCode,
    headers: HeaderMap,
    chunks: Vec<Bytes>,
    bytes: usize,
    max_bytes: usize,
    overflowed: bool,
}

struct OutputSink {
    sender: mpsc::Sender<Result<Bytes, io::Error>>,
    cancellation: CancellationToken,
    downstream_open: bool,
    capture: Option<IdempotencyCapture>,
}

struct ActiveRuntime {
    output: OutputSink,
    finished: Arc<AtomicBool>,
    control_drain: Option<tokio::task::JoinHandle<()>>,
}

impl OutputSink {
    fn new(
        sender: mpsc::Sender<Result<Bytes, io::Error>>,
        cancellation: CancellationToken,
        capture: Option<IdempotencyCapture>,
    ) -> Self {
        Self {
            sender,
            cancellation,
            downstream_open: true,
            capture,
        }
    }

    async fn send_wire(&mut self, wire: Bytes) -> Result<(), ()> {
        for offset in (0..wire.len()).step_by(DOWNSTREAM_SEGMENT_BYTES) {
            let end = (offset + DOWNSTREAM_SEGMENT_BYTES).min(wire.len());
            let segment = wire.slice(offset..end);
            if let Some(capture) = self.capture.as_mut() {
                if !capture.overflowed {
                    let next_bytes = capture.bytes.saturating_add(segment.len());
                    if next_bytes > capture.max_bytes
                        || !capture.owner.try_reserve_inflight_response(segment.len())
                    {
                        capture.owner.release_inflight_response(capture.bytes);
                        capture.overflowed = true;
                        capture.chunks.clear();
                        capture.bytes = 0;
                    } else {
                        capture.bytes = next_bytes;
                        capture.chunks.push(segment.clone());
                    }
                }
            }
            if !self.downstream_open {
                continue;
            }
            let result = tokio::select! {
                _ = self.cancellation.cancelled() => None,
                result = tokio::time::timeout(
                    downstream_write_timeout(),
                    self.sender.send(Ok(segment)),
                ) => Some(result),
            };
            if !matches!(result, Some(Ok(Ok(())))) {
                if self.capture.is_some() {
                    self.downstream_open = false;
                    continue;
                }
                return Err(());
            }
        }
        Ok(())
    }

    async fn finish(mut self, cacheable: bool) {
        let Some(capture) = self.capture.take() else {
            return;
        };
        capture.owner.release_inflight_response(capture.bytes);
        if !cacheable {
            capture.owner.release().await;
        } else if capture.overflowed {
            capture.owner.nonreplayable("response_too_large").await;
        } else {
            capture
                .owner
                .complete(
                    capture.status,
                    capture.headers,
                    capture.chunks,
                    capture.bytes,
                )
                .await;
        }
    }
}

async fn run_active_attempt(
    state: AppState,
    mut coordinator: Coordinator,
    mut active: ActiveAttempt,
    runtime: ActiveRuntime,
) {
    let ActiveRuntime {
        mut output,
        finished,
        mut control_drain,
    } = runtime;
    let cancellation = output.cancellation.clone();
    let cacheable = 'request: loop {
        if flush_initial_output(&mut active, &mut output)
            .await
            .is_err()
        {
            complete_disconnect(
                &state,
                &mut coordinator,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
            )
            .await;
            break 'request false;
        }

        if let Some(terminal) = active.terminal.take() {
            if !active.business_committed {
                let outcome = semantic_retry_outcome(&active, terminal);
                match coordinator.retry(&state, outcome).await {
                    Ok(RetryResolution::Active(next)) => {
                        active = next;
                        continue 'request;
                    }
                    Ok(RetryResolution::Final(message)) => {
                        break 'request send_final_inband(&mut output, &message).await.is_ok();
                    }
                    Err(_) => break 'request false,
                }
            }
            finish_terminal(
                &state,
                &mut coordinator,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
                terminal,
            )
            .await;
            break 'request true;
        }

        loop {
            if active.business_committed {
                let mode = active.mode;
                if mode.relays_raw_after_commit() {
                    break 'request run_raw_committed(
                        &state,
                        &mut coordinator,
                        &mut active,
                        &mut output,
                        &cancellation,
                    )
                    .await;
                }
                if mode == StreamMode::SelectiveRewrite {
                    break 'request run_selective_committed(
                        &state,
                        &mut coordinator,
                        &mut active,
                        &mut output,
                        &cancellation,
                    )
                    .await;
                }
            }
            let idle_deadline = deadline(
                tokio::time::Instant::now(),
                active.plan.idle_timeout_seconds,
            );
            let next_deadline = earlier_deadline(idle_deadline, active.total_deadline);
            let next = tokio::select! {
                _ = cancellation.cancelled(), if output.capture.is_none() => {
                    complete_disconnect(
                        &state,
                        &mut coordinator,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    break 'request false;
                }
                result = await_deadline(active.stream.next(), next_deadline) => result,
            };
            let frames = match next {
                Ok(Some(Ok(chunk))) => {
                    active.stats.observe_upstream(&chunk);
                    match active.decoder.feed(&chunk) {
                        Ok(frames) => frames,
                        Err(error) => {
                            if !active.business_committed {
                                match coordinator
                                    .retry(
                                        &state,
                                        failure_retry_outcome(&active, "protocol_error", &error),
                                    )
                                    .await
                                {
                                    Ok(RetryResolution::Active(next)) => {
                                        active = next;
                                        continue 'request;
                                    }
                                    Ok(RetryResolution::Final(message)) => {
                                        break 'request send_final_inband(&mut output, &message)
                                            .await
                                            .is_ok();
                                    }
                                    Err(_) => break 'request false,
                                }
                            } else {
                                complete_failure(
                                    &state,
                                    &mut coordinator,
                                    active.plan.attempt_id.clone(),
                                    active.status,
                                    &mut active.stats,
                                    "protocol_error",
                                    &error,
                                )
                                .await;
                            }
                            break 'request false;
                        }
                    }
                }
                Ok(None) => match active.decoder.finish() {
                    Ok(frames) if !frames.is_empty() => frames,
                    Ok(_) => {
                        let detail = "Responses upstream ended without a terminal response event";
                        if !active.business_committed {
                            match coordinator
                                .retry(
                                    &state,
                                    failure_retry_outcome(&active, "protocol_error", detail),
                                )
                                .await
                            {
                                Ok(RetryResolution::Active(next)) => {
                                    active = next;
                                    continue 'request;
                                }
                                Ok(RetryResolution::Final(message)) => {
                                    break 'request send_final_inband(&mut output, &message)
                                        .await
                                        .is_ok();
                                }
                                Err(_) => break 'request false,
                            }
                        } else {
                            complete_failure(
                                &state,
                                &mut coordinator,
                                active.plan.attempt_id.clone(),
                                active.status,
                                &mut active.stats,
                                "protocol_error",
                                detail,
                            )
                            .await;
                        }
                        break 'request false;
                    }
                    Err(error) => {
                        if !active.business_committed {
                            match coordinator
                                .retry(
                                    &state,
                                    failure_retry_outcome(&active, "protocol_error", &error),
                                )
                                .await
                            {
                                Ok(RetryResolution::Active(next)) => {
                                    active = next;
                                    continue 'request;
                                }
                                Ok(RetryResolution::Final(message)) => {
                                    break 'request send_final_inband(&mut output, &message)
                                        .await
                                        .is_ok();
                                }
                                Err(_) => break 'request false,
                            }
                        } else {
                            complete_failure(
                                &state,
                                &mut coordinator,
                                active.plan.attempt_id.clone(),
                                active.status,
                                &mut active.stats,
                                "protocol_error",
                                &error,
                            )
                            .await;
                        }
                        break 'request false;
                    }
                },
                Ok(Some(Err(error))) => {
                    let detail = format!("upstream stream read failed: {error}");
                    if !active.business_committed {
                        match coordinator
                            .retry(
                                &state,
                                failure_retry_outcome(&active, "transport_error", &detail),
                            )
                            .await
                        {
                            Ok(RetryResolution::Active(next)) => {
                                active = next;
                                continue 'request;
                            }
                            Ok(RetryResolution::Final(message)) => {
                                break 'request send_final_inband(&mut output, &message)
                                    .await
                                    .is_ok();
                            }
                            Err(_) => break 'request false,
                        }
                    } else {
                        complete_failure(
                            &state,
                            &mut coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "transport_error",
                            &detail,
                        )
                        .await;
                    }
                    break 'request false;
                }
                Err(error) => {
                    if !active.business_committed {
                        match coordinator
                            .retry(
                                &state,
                                failure_retry_outcome(&active, "transport_error", &error),
                            )
                            .await
                        {
                            Ok(RetryResolution::Active(next)) => {
                                active = next;
                                continue 'request;
                            }
                            Ok(RetryResolution::Final(message)) => {
                                break 'request send_final_inband(&mut output, &message)
                                    .await
                                    .is_ok();
                            }
                            Err(_) => break 'request false,
                        }
                    } else {
                        complete_failure(
                            &state,
                            &mut coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "transport_error",
                            &error,
                        )
                        .await;
                    }
                    break 'request false;
                }
            };

            match process_active_frames(&state, &mut coordinator, &mut active, &mut output, frames)
                .await
            {
                ActiveFrameResult::Continue => {}
                ActiveFrameResult::Done(cacheable) => break 'request cacheable,
                ActiveFrameResult::Retry(outcome) => {
                    match coordinator.retry(&state, outcome).await {
                        Ok(RetryResolution::Active(next)) => {
                            active = next;
                            continue 'request;
                        }
                        Ok(RetryResolution::Final(message)) => {
                            break 'request send_final_inband(&mut output, &message).await.is_ok();
                        }
                        Err(_) => break 'request false,
                    }
                }
            }
        }
    };
    finished.store(true, Ordering::Release);
    if let Some(task) = control_drain.take() {
        task.abort();
    }
    output.finish(cacheable).await;
}

async fn run_raw_committed(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    cancellation: &CancellationToken,
) -> bool {
    if !active.raw_pending_forwarded {
        if let Some(pending) = active.decoder.pending_copy() {
            active.stats.observe_wire(&pending);
            if output.send_wire(pending).await.is_err() {
                complete_disconnect(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                )
                .await;
                return false;
            }
        }
        active.raw_pending_forwarded = true;
    }

    loop {
        let idle_deadline = deadline(
            tokio::time::Instant::now(),
            active.plan.idle_timeout_seconds,
        );
        let next_deadline = earlier_deadline(idle_deadline, active.total_deadline);
        let next = tokio::select! {
            _ = cancellation.cancelled(), if output.capture.is_none() => {
                complete_disconnect(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                )
                .await;
                return false;
            }
            result = await_deadline(active.stream.next(), next_deadline) => result,
        };
        match next {
            Ok(Some(Ok(chunk))) => {
                active.stats.observe_upstream(&chunk);
                let inspection = match active.decoder.feed(&chunk) {
                    Ok(frames) => inspect_raw_frames(frames, &mut active.stats),
                    Err(error) => Err(error),
                };
                active.stats.observe_wire(&chunk);
                if output.send_wire(chunk).await.is_err() {
                    complete_disconnect(
                        state,
                        coordinator,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    return false;
                }
                match inspection {
                    Ok(Some(terminal)) => {
                        finish_observed_terminal(state, coordinator, active, terminal).await;
                        return true;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        complete_failure(
                            state,
                            coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "protocol_error",
                            &error,
                        )
                        .await;
                        return false;
                    }
                }
            }
            Ok(None) => {
                let inspection = match active.decoder.finish() {
                    Ok(frames) => inspect_raw_frames(frames, &mut active.stats),
                    Err(error) => Err(error),
                };
                match inspection {
                    Ok(Some(terminal)) => {
                        finish_observed_terminal(state, coordinator, active, terminal).await;
                        return true;
                    }
                    Ok(None) => {
                        complete_failure(
                            state,
                            coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "protocol_error",
                            "Responses upstream ended without a terminal response event",
                        )
                        .await;
                    }
                    Err(error) => {
                        complete_failure(
                            state,
                            coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "protocol_error",
                            &error,
                        )
                        .await;
                    }
                }
                return false;
            }
            Ok(Some(Err(error))) => {
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "transport_error",
                    &format!("upstream stream read failed: {error}"),
                )
                .await;
                return false;
            }
            Err(error) => {
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "transport_error",
                    &error,
                )
                .await;
                return false;
            }
        }
    }
}

fn inspect_raw_frames(
    frames: Vec<SseFrame>,
    stats: &mut StreamStats,
) -> Result<Option<Terminal>, String> {
    for frame in frames {
        if let Some(terminal) = inspect_terminal_frame(&frame, stats)? {
            return Ok(Some(terminal));
        }
    }
    Ok(None)
}

async fn finish_observed_terminal(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    terminal: Terminal,
) {
    if let Terminal::SemanticFailure {
        event_type,
        payload,
    } = &terminal
    {
        let outcome = semantic_failure_outcome(active, event_type, payload, true);
        let _ = coordinator.complete(state, &outcome).await;
        return;
    }
    finish_terminal(
        state,
        coordinator,
        active.plan.attempt_id.clone(),
        active.status,
        &mut active.stats,
        terminal,
    )
    .await;
}

async fn run_selective_committed(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    cancellation: &CancellationToken,
) -> bool {
    loop {
        let idle_deadline = deadline(
            tokio::time::Instant::now(),
            active.plan.idle_timeout_seconds,
        );
        let next_deadline = earlier_deadline(idle_deadline, active.total_deadline);
        let next = tokio::select! {
            _ = cancellation.cancelled(), if output.capture.is_none() => {
                complete_disconnect(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                )
                .await;
                return false;
            }
            result = await_deadline(active.stream.next(), next_deadline) => result,
        };
        match next {
            Ok(Some(Ok(chunk))) => {
                active.stats.observe_upstream(&chunk);
                let frames = match active.decoder.feed(&chunk) {
                    Ok(frames) => frames,
                    Err(error) => {
                        complete_failure(
                            state,
                            coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "protocol_error",
                            &error,
                        )
                        .await;
                        return false;
                    }
                };
                if let Some(cacheable) =
                    relay_selective_frames(state, coordinator, active, output, frames).await
                {
                    return cacheable;
                }
            }
            Ok(None) => {
                let frames = match active.decoder.finish() {
                    Ok(frames) => frames,
                    Err(error) => {
                        complete_failure(
                            state,
                            coordinator,
                            active.plan.attempt_id.clone(),
                            active.status,
                            &mut active.stats,
                            "protocol_error",
                            &error,
                        )
                        .await;
                        return false;
                    }
                };
                if let Some(cacheable) =
                    relay_selective_frames(state, coordinator, active, output, frames).await
                {
                    return cacheable;
                }
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "protocol_error",
                    "Responses upstream ended without a terminal response event",
                )
                .await;
                return false;
            }
            Ok(Some(Err(error))) => {
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "transport_error",
                    &format!("upstream stream read failed: {error}"),
                )
                .await;
                return false;
            }
            Err(error) => {
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "transport_error",
                    &error,
                )
                .await;
                return false;
            }
        }
    }
}

async fn relay_selective_frames(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    frames: Vec<SseFrame>,
) -> Option<bool> {
    let mut batch = BytesMut::new();
    for frame in frames {
        let special = selective_rewrite_candidate(&frame)
            || frame_is_comment_only(frame.raw())
            || terminal_candidate(&frame);
        if !special {
            observe_light_frame(&mut active.stats, &frame);
            batch.extend_from_slice(&frame.canonical_wire());
            continue;
        }
        if !batch.is_empty() {
            let wire = batch.split().freeze();
            active.stats.observe_wire(&wire);
            if output.send_wire(wire).await.is_err() {
                complete_disconnect(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                )
                .await;
                return Some(false);
            }
        }
        match process_active_frames(state, coordinator, active, output, vec![frame]).await {
            ActiveFrameResult::Continue => {}
            ActiveFrameResult::Done(cacheable) => return Some(cacheable),
            ActiveFrameResult::Retry(_) => return Some(false),
        }
    }
    if !batch.is_empty() {
        let wire = batch.freeze();
        active.stats.observe_wire(&wire);
        if output.send_wire(wire).await.is_err() {
            complete_disconnect(
                state,
                coordinator,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
            )
            .await;
            return Some(false);
        }
    }
    None
}

async fn flush_initial_output(
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
) -> Result<(), ()> {
    let mut wires = std::mem::take(&mut active.early_output);
    if active.business_committed {
        wires.extend(std::mem::take(&mut active.buffered));
    }
    for wire in wires {
        active.stats.observe_wire(&wire);
        output.send_wire(wire).await?;
    }
    Ok(())
}

fn semantic_failure_outcome(
    active: &ActiveAttempt,
    event_type: &str,
    payload: &Value,
    committed: bool,
) -> Value {
    let (status, detail) = responses_semantic_error(payload, event_type);
    let mut outcome = active.stats.report();
    outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
    outcome["kind"] = Value::String(
        if committed {
            "semantic_failure"
        } else {
            "semantic_error"
        }
        .into(),
    );
    outcome["status_code"] = Value::from(status);
    outcome["upstream_status_code"] = Value::from(active.status.as_u16());
    outcome["event_type"] = Value::String(event_type.to_owned());
    outcome["payload"] = payload.clone();
    outcome["detail"] = Value::String(detail);
    outcome["committed"] = Value::Bool(committed);
    outcome
}

fn responses_semantic_error(payload: &Value, event_type: &str) -> (u16, String) {
    let error = if event_type.eq_ignore_ascii_case("response.failed") {
        payload.pointer("/response/error")
    } else {
        payload.get("error")
    }
    .or_else(|| payload.get("error"));
    let detail = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| error.and_then(Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("Responses upstream returned a failure terminal")
        .chars()
        .take(4096)
        .collect::<String>();

    for candidate in [
        error.and_then(|value| value.get("status_code")),
        error.and_then(|value| value.get("status")),
        payload.get("status_code"),
        payload.get("status"),
        payload.pointer("/response/status_code"),
    ]
    .into_iter()
    .flatten()
    {
        let parsed = candidate
            .as_u64()
            .or_else(|| candidate.as_str()?.parse::<u64>().ok());
        if let Some(status @ 400..=599) = parsed {
            return (status as u16, detail);
        }
    }

    let code = error
        .and_then(|value| value.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let status = match code.as_str() {
        "account_deactivated"
        | "account_disabled"
        | "account_suspended"
        | "deactivated_workspace"
        | "permission_denied"
        | "user_deactivated"
        | "user_suspended" => Some(403),
        "authentication_error" | "incorrect_api_key_provided" | "invalid_api_key" => Some(401),
        "billing_hard_limit_reached" | "insufficient_quota" | "rate_limit_exceeded" => Some(429),
        "context_length_exceeded"
        | "invalid_request_error"
        | "invalid_type"
        | "model_not_priced"
        | "model_price_not_configured"
        | "model_pricing_not_configured"
        | "model_price_unconfigured"
        | "model_pricing_missing"
        | "unsupported_parameter" => Some(400),
        "model_not_found" | "not_found_error" => Some(404),
        _ => None,
    };
    if let Some(status) = status {
        return (status, detail);
    }

    let error_type = error
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let status = match error_type.as_str() {
        "authentication_error" => Some(401),
        "invalid_request_error" => Some(400),
        "not_found_error" => Some(404),
        "permission_error" => Some(403),
        "rate_limit_error" | "tokens" => Some(429),
        _ => None,
    };
    if let Some(status) = status {
        return (status, detail);
    }

    let message = detail.to_ascii_lowercase();
    let status = if message.contains("rate limit") || message.contains("too many requests") {
        429
    } else if [
        "context window",
        "context length",
        "maximum context",
        "too many tokens",
    ]
    .iter()
    .any(|marker| message.contains(marker))
    {
        400
    } else if message.contains("request entity too large") || message.contains("payload too large")
    {
        413
    } else if message.contains("invalid") || message.contains("unsupported") {
        400
    } else if message.contains("not found") {
        404
    } else if message.contains("permission") || message.contains("forbidden") {
        403
    } else if message.contains("auth")
        || message.contains("api key")
        || message.contains("unauthorized")
    {
        401
    } else {
        500
    };
    (status, detail)
}

fn semantic_retry_outcome(active: &ActiveAttempt, terminal: Terminal) -> Value {
    match terminal {
        Terminal::SemanticFailure {
            event_type,
            payload,
        } => semantic_failure_outcome(active, &event_type, &payload, false),
        _ => failure_retry_outcome(
            active,
            "protocol_error",
            "Responses precommit attempt ended unexpectedly",
        ),
    }
}

fn failure_retry_outcome(active: &ActiveAttempt, kind: &str, detail: &str) -> Value {
    let mut outcome = active.stats.report();
    outcome["attempt_id"] = Value::String(active.plan.attempt_id.clone());
    outcome["kind"] = Value::String(kind.into());
    outcome["status_code"] = Value::from(if kind == "transport_error" { 504 } else { 502 });
    outcome["upstream_status_code"] = Value::from(active.status.as_u16());
    outcome["detail"] = Value::String(detail.chars().take(4096).collect());
    outcome["committed"] = Value::Bool(false);
    outcome
}

fn commit_observation(active: &ActiveAttempt, reason: &str) -> Value {
    json!({
        "attempt_id": active.plan.attempt_id,
        "upstream_status_code": active.status.as_u16(),
        "commit_reason": reason,
        "business_committed": active.business_committed,
        "precommit_events": active.stats.event_count,
        "precommit_bytes": active.buffered.iter().map(|item| item.len() as u64).sum::<u64>(),
    })
}

enum RetryResolution {
    Active(ActiveAttempt),
    Final(Value),
}

async fn retry_after_public_start_python(
    state: &AppState,
    session_id: &str,
    mut outcome: Value,
) -> Result<RetryResolution, String> {
    loop {
        let message = control_advance(state, session_id, &outcome).await?;
        if message.get("kind").and_then(Value::as_str) == Some("final") {
            return Ok(RetryResolution::Final(message));
        }
        let plan: Plan = serde_json::from_value(message.clone())
            .map_err(|error| format!("retry plan is invalid: {error}"))?;
        match preflight_attempt(state, plan.clone(), true).await {
            Ok(PreflightResult::Started(active)) => {
                let observation = commit_observation(&active, active.commit_reason);
                control_commit(state, session_id, &observation).await?;
                return Ok(RetryResolution::Active(active));
            }
            Ok(PreflightResult::Retry(mut next_outcome)) => {
                next_outcome["attempt_id"] = Value::String(plan.attempt_id);
                outcome = next_outcome;
            }
            Err(error) => {
                outcome = json!({
                    "attempt_id": plan.attempt_id,
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                });
            }
        }
    }
}

async fn send_final_inband(output: &mut OutputSink, message: &Value) -> Result<(), ()> {
    if let Some(encoded) = message
        .get("stream_failure_terminal_b64")
        .and_then(Value::as_str)
    {
        if let Ok(wire) = BASE64.decode(encoded) {
            return output.send_wire(Bytes::from(wire)).await;
        }
    }
    let status = message
        .get("status_code")
        .and_then(Value::as_u64)
        .unwrap_or(502);
    let detail = message
        .get("body_b64")
        .and_then(Value::as_str)
        .and_then(|value| BASE64.decode(value).ok())
        .map(|value| String::from_utf8_lossy(&value).chars().take(2048).collect())
        .unwrap_or_else(|| "All Responses providers failed".to_owned());
    let error = json!({
        "type": "error",
        "error": {"message": detail, "status_code": status},
    });
    let wire = encode_event("error", &error).map_err(|_| ())?;
    output.send_wire(wire).await?;
    output
        .send_wire(Bytes::from_static(b"data: [DONE]\n\n"))
        .await
}

enum ActiveFrameResult {
    Continue,
    Done(bool),
    Retry(Value),
}

async fn process_active_frames(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    frames: Vec<SseFrame>,
) -> ActiveFrameResult {
    for frame in frames {
        let processed = match active.processor.process(frame, &mut active.stats) {
            Ok(processed) => processed,
            Err(error) => {
                if !active.business_committed {
                    return ActiveFrameResult::Retry(failure_retry_outcome(
                        active,
                        "protocol_error",
                        &error,
                    ));
                }
                complete_failure(
                    state,
                    coordinator,
                    active.plan.attempt_id.clone(),
                    active.status,
                    &mut active.stats,
                    "protocol_error",
                    &error,
                )
                .await;
                return ActiveFrameResult::Done(false);
            }
        };
        if let Some(Terminal::SemanticFailure {
            event_type,
            payload,
        }) = processed.terminal.as_ref()
        {
            if !active.business_committed {
                let outcome = semantic_failure_outcome(active, event_type, payload, false);
                return ActiveFrameResult::Retry(outcome);
            }
            let outcome = semantic_failure_outcome(active, event_type, payload, true);
            let mut terminal_sent = false;
            if let Some(reply) = coordinator.complete(state, &outcome).await {
                if let Some(encoded) = reply.get("terminal_b64").and_then(Value::as_str) {
                    if let Ok(terminal) = BASE64.decode(encoded) {
                        let wire = Bytes::from(terminal);
                        active.stats.observe_wire(&wire);
                        terminal_sent = output.send_wire(wire).await.is_ok();
                    }
                }
            }
            if coordinator.is_native() && !terminal_sent {
                if let Ok(wire) = encode_event(event_type, payload) {
                    active.stats.observe_wire(&wire);
                    terminal_sent = output.send_wire(wire).await.is_ok();
                }
            }
            return ActiveFrameResult::Done(terminal_sent);
        }

        let suppress_repeated_keepalive = !active.business_committed
            && active.precommit_keepalive_sent
            && processed.event_type.as_deref() == Some("keepalive");
        if let Some(wire) = processed.wire.filter(|_| !suppress_repeated_keepalive) {
            if active.business_committed {
                active.stats.observe_wire(&wire);
                if output.send_wire(wire).await.is_err() {
                    complete_disconnect(
                        state,
                        coordinator,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    return ActiveFrameResult::Done(false);
                }
            } else {
                active.buffered.push(wire);
            }
        }

        if processed.commits && !active.business_committed {
            active.business_committed = true;
            let observation = commit_observation(active, "real_output");
            if coordinator.commit(state, &observation).await.is_err() {
                return ActiveFrameResult::Done(false);
            }
            for wire in std::mem::take(&mut active.buffered) {
                active.stats.observe_wire(&wire);
                if output.send_wire(wire).await.is_err() {
                    complete_disconnect(
                        state,
                        coordinator,
                        active.plan.attempt_id.clone(),
                        active.status,
                        &mut active.stats,
                    )
                    .await;
                    return ActiveFrameResult::Done(false);
                }
            }
        }
        if let Some(terminal) = processed.terminal {
            finish_terminal(
                state,
                coordinator,
                active.plan.attempt_id.clone(),
                active.status,
                &mut active.stats,
                terminal,
            )
            .await;
            return ActiveFrameResult::Done(true);
        }
    }
    ActiveFrameResult::Continue
}

async fn finish_terminal(
    state: &AppState,
    coordinator: &mut Coordinator,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
    terminal: Terminal,
) {
    let (kind, semantic_failure) = match terminal {
        Terminal::Completed => ("completed", None),
        Terminal::Incomplete => ("incomplete", None),
        Terminal::SemanticFailure {
            event_type,
            payload,
        } => {
            let (status, detail) = responses_semantic_error(&payload, &event_type);
            ("semantic_failure", Some((status, detail, event_type)))
        }
    };
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String(kind.into());
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["committed"] = Value::Bool(true);
    if let Some((status, detail, event_type)) = semantic_failure {
        outcome["status_code"] = Value::from(status);
        outcome["detail"] = Value::String(detail);
        outcome["event_type"] = Value::String(event_type);
    }
    let _ = coordinator.complete(state, &outcome).await;
}

async fn complete_disconnect(
    state: &AppState,
    coordinator: &mut Coordinator,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
) {
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String("downstream_disconnected".into());
    outcome["status_code"] = Value::from(499);
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["committed"] = Value::Bool(true);
    let _ = coordinator.complete(state, &outcome).await;
}

async fn complete_failure(
    state: &AppState,
    coordinator: &mut Coordinator,
    attempt_id: String,
    upstream_status: StatusCode,
    stats: &mut StreamStats,
    kind: &str,
    detail: &str,
) {
    let mut outcome = stats.report();
    outcome["attempt_id"] = Value::String(attempt_id);
    outcome["kind"] = Value::String(kind.into());
    outcome["status_code"] = Value::from(502);
    outcome["upstream_status_code"] = Value::from(upstream_status.as_u16());
    outcome["detail"] = Value::String(detail.chars().take(4096).collect());
    outcome["committed"] = Value::Bool(true);
    let _ = coordinator.complete(state, &outcome).await;
}

fn downstream_write_timeout() -> Duration {
    static TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        let seconds = std::env::var("RUST_DOWNSTREAM_WRITE_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(30.0);
        Duration::from_secs_f64(seconds)
    })
}

struct GuardedBodyStream {
    inner: ReceiverStream<Result<Bytes, io::Error>>,
    cancellation: CancellationToken,
    finished: Arc<AtomicBool>,
}

impl Stream for GuardedBodyStream {
    type Item = Result<Bytes, io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for GuardedBodyStream {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Acquire) {
            self.cancellation.cancel();
        }
    }
}

#[derive(Debug)]
struct ProcessedEvent {
    wire: Option<Bytes>,
    event_type: Option<String>,
    commits: bool,
    canonical_keepalive: bool,
    terminal: Option<Terminal>,
}

struct ResponsesProcessor {
    commit_policy: String,
    normalizer: Option<ResponsesItemIdNormalizer>,
}

impl ResponsesProcessor {
    fn new(commit_policy: String, normalize_ids: bool) -> Self {
        let commit_policy = match commit_policy.trim().to_ascii_lowercase().as_str() {
            "completed_usage" => "completed_usage".to_owned(),
            _ => "real_output".to_owned(),
        };
        Self {
            commit_policy,
            normalizer: normalize_ids.then(ResponsesItemIdNormalizer::default),
        }
    }

    fn process(
        &mut self,
        frame: SseFrame,
        stats: &mut StreamStats,
    ) -> Result<ProcessedEvent, String> {
        stats.event_count = stats.event_count.saturating_add(1);
        let parsed = parse_sse_frame(&frame)?;
        if parsed.comment_only {
            if parsed.raw.starts_with(": oaix-terminal-flush-v1 ") {
                return Ok(ProcessedEvent {
                    wire: None,
                    event_type: None,
                    commits: false,
                    canonical_keepalive: false,
                    terminal: None,
                });
            }
            return Ok(ProcessedEvent {
                wire: Some(frame.canonical_wire()),
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        }
        let Some(data) = parsed.data else {
            return Ok(ProcessedEvent {
                wire: None,
                event_type: None,
                commits: false,
                canonical_keepalive: false,
                terminal: None,
            });
        };
        if data.trim() == "[DONE]" {
            return Err(
                "Responses upstream emitted [DONE] without a terminal response event".into(),
            );
        }
        let mut payload: Value = serde_json::from_str(&data)
            .map_err(|error| format!("Responses upstream event JSON is invalid: {error}"))?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "Responses upstream event payload is missing a string type".to_owned())?
            .to_owned();
        if event_type.len() > 256 || event_type.contains(['\r', '\n']) {
            return Err("Responses upstream event type is invalid".into());
        }
        if let Some(declared) = parsed.declared_event.as_deref() {
            if declared != event_type {
                return Err(format!(
                    "Responses event field {declared:?} does not match payload type {event_type:?}"
                ));
            }
        }
        validate_terminal(&event_type, &payload)?;

        let normalized = if let Some(normalizer) = self.normalizer.as_mut() {
            normalizer.normalize(&mut payload)?
        } else {
            false
        };
        let wire = if normalized {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            encode_event(&event_type, &payload)?
        } else if parsed.declared_event.is_none() {
            stats.normalized_events = stats.normalized_events.saturating_add(1);
            Bytes::from(format!("event: {event_type}\n{}\n\n", parsed.raw))
        } else {
            frame.canonical_wire()
        };
        if event_type.ends_with(".delta") {
            stats.delta_events = stats.delta_events.saturating_add(1);
        }
        if let Some(usage) = extract_usage(&payload) {
            stats.usage = Some(usage.clone());
        }

        let terminal = if semantic_failure(&event_type, &payload) {
            Some(Terminal::SemanticFailure {
                event_type: event_type.clone(),
                payload: payload.clone(),
            })
        } else if event_type == "response.completed" {
            Some(Terminal::Completed)
        } else if event_type == "response.incomplete" {
            Some(Terminal::Incomplete)
        } else {
            None
        };
        let canonical_keepalive = event_type == "keepalive" && is_canonical_keepalive(&payload);
        let commits = terminal.is_some()
            || (!matches!(
                event_type.as_str(),
                "response.created" | "response.in_progress" | "response.queued" | "keepalive"
            ) && self.commit_policy != "completed_usage"
                && has_real_output(&event_type, &payload));
        Ok(ProcessedEvent {
            wire: Some(wire),
            event_type: Some(event_type),
            commits,
            canonical_keepalive,
            terminal,
        })
    }
}

fn declared_event_bytes(raw: &[u8]) -> Option<&[u8]> {
    for line in raw.split(|byte| matches!(*byte, b'\r' | b'\n')) {
        let Some(mut value) = line.strip_prefix(b"event:") else {
            continue;
        };
        if let Some(stripped) = value.strip_prefix(b" ") {
            value = stripped;
        }
        return Some(value);
    }
    None
}

fn frame_has_data(raw: &[u8]) -> bool {
    raw.split(|byte| matches!(*byte, b'\r' | b'\n'))
        .any(|line| line == b"data" || line.starts_with(b"data:"))
}

fn frame_is_comment_only(raw: &[u8]) -> bool {
    let mut saw_comment = false;
    for line in raw.split(|byte| matches!(*byte, b'\r' | b'\n')) {
        if line.is_empty() {
            continue;
        }
        if line.starts_with(b":") {
            saw_comment = true;
        } else {
            return false;
        }
    }
    saw_comment
}

fn is_terminal_event_type(event_type: &[u8]) -> bool {
    matches!(
        event_type,
        b"error" | b"response.completed" | b"response.failed" | b"response.incomplete"
    )
}

fn terminal_candidate(frame: &SseFrame) -> bool {
    if declared_event_bytes(frame.raw()).is_some_and(is_terminal_event_type) {
        return true;
    }
    [
        b"response.completed".as_slice(),
        b"response.failed".as_slice(),
        b"response.incomplete".as_slice(),
        b"\"type\":\"error\"".as_slice(),
    ]
    .iter()
    .any(|needle| memmem::find(frame.raw(), needle).is_some())
}

fn selective_rewrite_candidate(frame: &SseFrame) -> bool {
    let raw = frame.raw();
    let declared_event = declared_event_bytes(raw);
    terminal_candidate(frame)
        || declared_event.is_none()
        || !frame_has_data(raw)
        || declared_event.is_some_and(|event_type| {
            matches!(
                event_type,
                b"response.output_item.added" | b"response.output_item.done"
            ) || json_item_id_fields(raw)
                .any(|item_id| event_item_id_needs_normalization(event_type, item_id))
        })
}

fn json_item_id_fields(raw: &[u8]) -> impl Iterator<Item = &[u8]> {
    const ITEM_ID_KEY: &[u8] = b"\"item_id\"";
    memmem::find_iter(raw, ITEM_ID_KEY).filter_map(|key_start| {
        let after_field = raw.get(key_start + ITEM_ID_KEY.len()..)?;
        let colon = after_field.iter().position(|byte| *byte == b':')?;
        let after_colon = after_field.get(colon + 1..)?;
        let quote = after_colon
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())?;
        let value = after_colon.get(quote..)?.strip_prefix(b"\"")?;
        let end = value.iter().position(|byte| *byte == b'\"')?;
        value.get(..end)
    })
}

fn observe_light_frame(stats: &mut StreamStats, frame: &SseFrame) {
    stats.event_count = stats.event_count.saturating_add(1);
    if declared_event_bytes(frame.raw()).is_some_and(|event_type| event_type.ends_with(b".delta")) {
        stats.delta_events = stats.delta_events.saturating_add(1);
    }
}

fn inspect_terminal_frame(
    frame: &SseFrame,
    stats: &mut StreamStats,
) -> Result<Option<Terminal>, String> {
    observe_light_frame(stats, frame);
    if !terminal_candidate(frame) {
        return Ok(None);
    }
    let parsed = parse_sse_frame(frame)?;
    let Some(data) = parsed.data else {
        return Ok(None);
    };
    if data.trim() == "[DONE]" {
        return Err("Responses upstream emitted [DONE] without a terminal response event".into());
    }
    let payload: Value = serde_json::from_str(&data)
        .map_err(|error| format!("Responses upstream terminal JSON is invalid: {error}"))?;
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "Responses upstream terminal payload is missing a string type".to_owned())?;
    if !is_terminal_event_type(event_type.as_bytes()) {
        return Ok(None);
    }
    if let Some(declared) = parsed.declared_event.as_deref() {
        if declared != event_type {
            return Err(format!(
                "Responses event field {declared:?} does not match payload type {event_type:?}"
            ));
        }
    }
    validate_terminal(event_type, &payload)?;
    if let Some(usage) = extract_usage(&payload) {
        stats.usage = Some(usage.clone());
    }
    if semantic_failure(event_type, &payload) {
        return Ok(Some(Terminal::SemanticFailure {
            event_type: event_type.to_owned(),
            payload,
        }));
    }
    Ok(match event_type {
        "response.completed" => Some(Terminal::Completed),
        "response.incomplete" => Some(Terminal::Incomplete),
        _ => None,
    })
}

fn validate_terminal(event_type: &str, payload: &Value) -> Result<(), String> {
    if !matches!(
        event_type,
        "error" | "response.completed" | "response.failed" | "response.incomplete"
    ) {
        return Ok(());
    }
    let object = payload
        .as_object()
        .ok_or_else(|| format!("Responses upstream {event_type} payload must be a JSON object"))?;
    if event_type.starts_with("response.") && !object.get("response").is_some_and(Value::is_object)
    {
        return Err(format!(
            "Responses upstream {event_type} payload is missing response"
        ));
    }
    if event_type == "error" && object.get("error").is_none_or(Value::is_null) {
        return Err("Responses upstream error payload is missing error".into());
    }
    Ok(())
}

fn semantic_failure(event_type: &str, payload: &Value) -> bool {
    if matches!(event_type, "error" | "response.failed") {
        return true;
    }
    let status = payload.get("status").and_then(Value::as_str);
    let response_status = payload
        .get("response")
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str);
    status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || response_status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || payload.get("error").is_some_and(Value::is_object)
}

fn is_canonical_keepalive(payload: &Value) -> bool {
    let Some(object) = payload.as_object() else {
        return false;
    };
    object.len() == 2
        && object.get("type").and_then(Value::as_str) == Some("keepalive")
        && object.get("sequence_number").and_then(Value::as_i64) == Some(0)
}

fn has_real_output(event_type: &str, payload: &Value) -> bool {
    if event_type.starts_with("response.") && event_type.ends_with(".delta") {
        return payload
            .get("delta")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty());
    }
    if matches!(
        event_type,
        "response.content_part.added" | "response.content_part.done"
    ) {
        return payload.get("part").is_some_and(part_has_text);
    }
    if event_type == "response.output_item.done" {
        return payload.get("item").is_some_and(item_has_output);
    }
    if event_type.starts_with("response.") && event_type.ends_with(".done") {
        return ["text", "refusal", "arguments"].iter().any(|field| {
            payload
                .get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        });
    }
    false
}

fn part_has_text(part: &Value) -> bool {
    ["text", "refusal"].iter().any(|field| {
        part.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    })
}

fn item_has_output(item: &Value) -> bool {
    if item
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|parts| parts.iter().any(part_has_text))
    {
        return true;
    }
    if !matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "tool_call")
    ) {
        return false;
    }
    ["name", "arguments", "call_id"].iter().any(|field| {
        item.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    })
}

fn extract_usage(payload: &Value) -> Option<&Value> {
    payload
        .get("response")
        .and_then(|value| value.get("usage"))
        .filter(|value| value.is_object())
        .or_else(|| payload.get("usage").filter(|value| value.is_object()))
}

fn encode_event(event_type: &str, payload: &Value) -> Result<Bytes, String> {
    let data = serde_json::to_string(payload)
        .map_err(|error| format!("Responses event encoding failed: {error}"))?;
    Ok(Bytes::from(format!(
        "event: {event_type}\ndata: {data}\n\n"
    )))
}

#[derive(Debug)]
struct SseFrame {
    wire: Bytes,
    raw_len: usize,
    terminated: bool,
}

impl SseFrame {
    fn raw(&self) -> &[u8] {
        &self.wire[..self.raw_len]
    }

    fn canonical_wire(&self) -> Bytes {
        if self.terminated {
            self.wire.clone()
        } else {
            let mut wire = BytesMut::with_capacity(self.wire.len().saturating_add(2));
            wire.extend_from_slice(&self.wire);
            wire.extend_from_slice(b"\n\n");
            wire.freeze()
        }
    }
}

struct ParsedSseFrame {
    raw: String,
    declared_event: Option<String>,
    data: Option<String>,
    comment_only: bool,
}

struct SseDecoder {
    buffer: BytesMut,
    scan_from: usize,
    max_event_bytes: usize,
}

impl SseDecoder {
    fn new(max_event_bytes: usize) -> Self {
        Self {
            buffer: BytesMut::new(),
            scan_from: 0,
            max_event_bytes,
        }
    }

    fn exceeds_event_limit(&self, observed_bytes: usize) -> bool {
        self.max_event_bytes != UNLIMITED_SSE_EVENT_BYTES && observed_bytes > self.max_event_bytes
    }

    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some((end, delimiter_len)) =
            find_event_delimiter_from(&self.buffer, self.scan_from)
        {
            if self.exceeds_event_limit(end) {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let wire = self.buffer.split_to(end + delimiter_len).freeze();
            self.scan_from = 0;
            if wire[..end].iter().all(|byte| byte.is_ascii_whitespace()) {
                continue;
            }
            frames.push(SseFrame {
                wire,
                raw_len: end,
                terminated: true,
            });
        }
        self.scan_from = self.buffer.len().saturating_sub(3);
        if self.max_event_bytes != UNLIMITED_SSE_EVENT_BYTES
            && self.buffer.len() > self.max_event_bytes.saturating_add(64 * 1024)
        {
            return Err("Responses upstream SSE pending frame exceeds the configured limit".into());
        }
        Ok(frames)
    }

    fn finish(&mut self) -> Result<Vec<SseFrame>, String> {
        let mut frames = self.feed(&[])?;
        if !self.buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            if self.exceeds_event_limit(self.buffer.len()) {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let raw_len = self.buffer.len();
            frames.push(SseFrame {
                wire: self.buffer.split().freeze(),
                raw_len,
                terminated: false,
            });
        }
        self.buffer.clear();
        self.scan_from = 0;
        Ok(frames)
    }

    fn pending_copy(&self) -> Option<Bytes> {
        (!self.buffer.is_empty()).then(|| Bytes::copy_from_slice(&self.buffer))
    }
}

fn find_event_delimiter_from(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut index = start.min(bytes.len());
    while index < bytes.len() {
        let relative = memchr2(b'\r', b'\n', &bytes[index..])?;
        index += relative;
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") || bytes[index..].starts_with(b"\r\r") {
            return Some((index, 2));
        }
        index += 1;
    }
    None
}

fn parse_sse_frame(frame: &SseFrame) -> Result<ParsedSseFrame, String> {
    let text = std::str::from_utf8(frame.raw())
        .map_err(|_| "Responses upstream SSE event is not valid UTF-8".to_owned())?;
    let normalized = if text.as_bytes().contains(&b'\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    let mut declared_event = None;
    let mut data_lines = Vec::new();
    let mut saw_field = false;
    let mut saw_comment = false;
    for line in normalized.split('\n') {
        if let Some(_comment) = line.strip_prefix(':') {
            saw_comment = true;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        saw_field = true;
        let (field, mut value) = line.split_once(':').unwrap_or((line, ""));
        if let Some(stripped) = value.strip_prefix(' ') {
            value = stripped;
        }
        match field {
            "event" => declared_event = Some(value.to_owned()),
            "data" => data_lines.push(value.to_owned()),
            _ => {}
        }
    }
    Ok(ParsedSseFrame {
        raw: normalized,
        declared_event,
        data: (!data_lines.is_empty()).then(|| data_lines.join("\n")),
        comment_only: saw_comment && !saw_field,
    })
}

async fn control_get_plan(state: &AppState, session_id: &str) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::GET,
        &format!("/_internal/rust-responses/{session_id}/plan"),
        None,
    )
    .await
}

async fn control_advance(
    state: &AppState,
    session_id: &str,
    outcome: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/advance"),
        Some(outcome),
    )
    .await
}

async fn control_commit(
    state: &AppState,
    session_id: &str,
    observation: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/commit"),
        Some(observation),
    )
    .await
}

async fn control_complete(
    state: &AppState,
    session_id: &str,
    outcome: &Value,
) -> Result<Value, String> {
    control_request(
        state,
        reqwest::Method::POST,
        &format!("/_internal/rust-responses/{session_id}/complete"),
        Some(outcome),
    )
    .await
}

async fn control_request(
    state: &AppState,
    method: reqwest::Method,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let mut request = state
        .backend_client
        .request(method, state.internal_url(path))
        .header(CONTROL_HEADER, state.control_token.as_ref());
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("control request failed: {error}"))?;
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("control response read failed: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "control response status {}: {}",
            status.as_u16(),
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(1024)
                .collect::<String>()
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("control response JSON is invalid: {error}"))
}

async fn response_from_final(
    message: &Value,
    control_headers: &HeaderMap,
    owner: Option<idempotency::Owner>,
) -> Response<Body> {
    let status = message
        .get("status_code")
        .and_then(Value::as_u64)
        .and_then(|value| StatusCode::from_u16(value as u16).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let body = Bytes::from(
        message
            .get("body_b64")
            .and_then(Value::as_str)
            .and_then(|value| BASE64.decode(value).ok())
            .unwrap_or_default(),
    );
    let mut headers = HeaderMap::new();
    if let Some(values) = message.get("headers").and_then(Value::as_object) {
        for (name, value) in values {
            let (Ok(name), Some(value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                value
                    .as_str()
                    .and_then(|value| HeaderValue::from_str(value).ok()),
            ) else {
                continue;
            };
            if name.as_str().eq_ignore_ascii_case("content-length") {
                continue;
            }
            headers.append(name, value);
        }
    }
    for (name, value) in control_headers {
        headers.insert(name.clone(), value.clone());
    }
    if let Some(owner) = owner {
        if body.len() > owner.max_response_bytes() {
            owner.nonreplayable("response_too_large").await;
        } else {
            owner
                .complete(status, headers.clone(), vec![body.clone()], body.len())
                .await;
        }
        return idempotency::response_from_bytes(status, headers, body);
    }
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

async fn release_owner(owner: &mut Option<idempotency::Owner>) {
    if let Some(owner) = owner.take() {
        owner.release().await;
    }
}

fn public_control_headers(headers: &HeaderMap) -> HeaderMap {
    let mut selected = HeaderMap::new();
    for (name, value) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if lower == "x-request-id" || lower.starts_with("access-control-") {
            selected.append(name.clone(), value.clone());
        }
    }
    selected
}

async fn read_limited_body(
    response: reqwest::Response,
    deadline: Option<tokio::time::Instant>,
) -> String {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while body.len() < MAX_ERROR_BODY_BYTES {
        match await_deadline(stream.next(), deadline).await {
            Ok(Some(Ok(chunk))) => {
                let remaining = MAX_ERROR_BODY_BYTES - body.len();
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            _ => break,
        }
    }
    String::from_utf8_lossy(&body).into_owned()
}

fn deadline(started: tokio::time::Instant, seconds: Option<f64>) -> Option<tokio::time::Instant> {
    seconds
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| started + Duration::from_secs_f64(value))
}

fn positive_duration(seconds: Option<f64>) -> Option<Duration> {
    seconds
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(Duration::from_secs_f64)
}

fn earliest_timeout(values: &[Option<f64>]) -> Option<Duration> {
    values
        .iter()
        .copied()
        .flatten()
        .filter(|value| value.is_finite() && *value > 0.0)
        .min_by(f64::total_cmp)
        .map(Duration::from_secs_f64)
}

fn earlier_deadline(
    first: Option<tokio::time::Instant>,
    second: Option<tokio::time::Instant>,
) -> Option<tokio::time::Instant> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

async fn await_deadline<F, T>(
    future: F,
    deadline: Option<tokio::time::Instant>,
) -> Result<T, String>
where
    F: std::future::Future<Output = T>,
{
    if let Some(deadline) = deadline {
        tokio::time::timeout_at(deadline, future)
            .await
            .map_err(|_| "upstream deadline exceeded".to_owned())
    } else {
        Ok(future.await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_active(engine: &str) -> ActiveAttempt {
        let plan = Plan {
            attempt_id: "attempt-1".into(),
            url: "http://provider.example/v1/responses".into(),
            headers: HashMap::new(),
            body: "{}".into(),
            proxy: None,
            engine: engine.into(),
            precommit_semantic_guard: None,
            http1_only: false,
            commit_policy: "real_output".into(),
            normalize_custom_tool_call_ids: false,
            connect_timeout_seconds: None,
            write_timeout_seconds: None,
            pool_timeout_seconds: None,
            first_byte_timeout_seconds: None,
            idle_timeout_seconds: None,
            total_timeout_seconds: None,
            provider_name: None,
            provider_key: None,
            original_model: None,
            max_event_bytes: 4096,
            max_precommit_items: 128,
            max_precommit_bytes: 64 * 1024,
        };
        let mode = StreamMode::for_plan(&plan);
        ActiveAttempt {
            decoder: SseDecoder::new(plan.max_event_bytes),
            processor: ResponsesProcessor::new("real_output".into(), false),
            mode,
            plan,
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            stream: Box::pin(futures_util::stream::empty()),
            early_output: Vec::new(),
            buffered: Vec::new(),
            stats: StreamStats::default(),
            terminal: None,
            total_deadline: None,
            commit_reason: "real_output",
            precommit_keepalive_sent: false,
            business_committed: false,
            raw_pending_forwarded: false,
        }
    }

    #[test]
    fn decoder_preserves_fragmented_canonical_frames() {
        let mut decoder = SseDecoder::new(1024);
        assert!(decoder.feed(b"event: response.output_").unwrap().is_empty());
        let frames = decoder
            .feed(
                b"text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n",
            )
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(
            declared_event_bytes(frames[0].raw()),
            Some(b"response.output_text.delta".as_slice())
        );
    }

    #[test]
    fn decoder_zero_limit_accepts_frames_beyond_the_legacy_bound() {
        let mut decoder = SseDecoder::new(UNLIMITED_SSE_EVENT_BYTES);
        let payload = vec![b'x'; 8 * 1024 * 1024 + 1];
        assert!(decoder.feed(b"data: ").unwrap().is_empty());
        assert!(decoder.feed(&payload).unwrap().is_empty());
        let frames = decoder.feed(b"\n\n").unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].raw_len, payload.len() + b"data: ".len());
    }

    #[test]
    fn decoder_explicit_limit_remains_enforced() {
        let mut decoder = SseDecoder::new(8);
        let error = decoder.feed(b"data: too-large\n\n").unwrap_err();
        assert_eq!(
            error,
            "Responses upstream SSE event exceeds the configured limit"
        );
    }

    #[test]
    fn stream_modes_match_engine_and_normalization_contract() {
        let gpt = test_active("gpt");
        assert_eq!(gpt.mode, StreamMode::OpaqueRaw);

        let codex = test_active("codex");
        assert_eq!(codex.mode, StreamMode::GuardedThenRaw);

        let mut normalized_plan = test_active("gpt").plan;
        normalized_plan.normalize_custom_tool_call_ids = true;
        assert_eq!(
            StreamMode::for_plan(&normalized_plan),
            StreamMode::SelectiveRewrite
        );
    }

    #[test]
    fn raw_mode_ignores_nonterminal_json_and_parses_terminal_once() {
        let mut decoder = SseDecoder::new(4096);
        let frames = decoder
            .feed(
                b"event: response.output_text.delta\ndata: this-is-not-json\n\n\
                  event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )
            .unwrap();
        let mut stats = StreamStats::new("raw-terminal-test");
        let terminal = inspect_raw_frames(frames, &mut stats).unwrap();
        assert!(matches!(terminal, Some(Terminal::Completed)));
        assert_eq!(stats.event_count, 2);
        assert_eq!(stats.delta_events, 1);
        assert_eq!(stats.usage.as_ref().unwrap()["total_tokens"], 3);
    }

    #[test]
    fn decoder_preserves_crlf_wire_without_rebuilding() {
        let wire = Bytes::from_static(
            b"event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\r\n\r\n",
        );
        let mut decoder = SseDecoder::new(4096);
        let frame = decoder.feed(&wire).unwrap().pop().unwrap();
        assert_eq!(frame.canonical_wire(), wire);
    }

    #[test]
    fn selective_mode_only_materializes_rewrite_candidates() {
        let mut decoder = SseDecoder::new(4096);
        let mut ordinary = decoder
            .feed(b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_abc\",\"delta\":\"x\"}\n\n")
            .unwrap();
        assert!(!selective_rewrite_candidate(&ordinary.remove(0)));

        let mut custom = decoder
            .feed(b"event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"item_abc\",\"delta\":\"{}\"}\n\n")
            .unwrap();
        assert!(selective_rewrite_candidate(&custom.remove(0)));

        let mut shadowed = decoder
            .feed(b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"metadata\":{\"item_id\":\"msg_nested\"},\"item_id\":\"item_actual\",\"delta\":\"x\"}\n\n")
            .unwrap();
        assert!(selective_rewrite_candidate(&shadowed.remove(0)));
    }

    #[test]
    fn processor_canonicalizes_data_only_events() {
        let mut decoder = SseDecoder::new(1024);
        let frame = decoder
            .feed(b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let mut processor = ResponsesProcessor::new("real_output".into(), false);
        let mut stats = StreamStats::default();
        let event = processor.process(frame, &mut stats).unwrap();
        assert!(event.commits);
        assert!(event
            .wire
            .unwrap()
            .starts_with(b"event: response.output_text.delta\n"));
    }

    #[test]
    fn custom_tool_call_ids_are_statefully_rewritten() {
        let mut processor = ResponsesProcessor::new("real_output".into(), true);
        let mut stats = StreamStats::default();
        let mut decoder = SseDecoder::new(4096);
        let added = decoder
            .feed(b"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"custom_tool_call\",\"id\":\"item_abc\"}}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let first = processor.process(added, &mut stats).unwrap().wire.unwrap();
        assert!(String::from_utf8_lossy(&first).contains("ctc_abc"));
        let delta = decoder
            .feed(b"event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"item_abc\",\"delta\":\"{}\"}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let second = processor.process(delta, &mut stats).unwrap().wire.unwrap();
        assert!(String::from_utf8_lossy(&second).contains("ctc_abc"));
    }

    #[test]
    fn response_failed_is_a_semantic_terminal() {
        let mut decoder = SseDecoder::new(4096);
        let frame = decoder
            .feed(b"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"nope\"}}}\n\n")
            .unwrap()
            .pop()
            .unwrap();
        let mut processor = ResponsesProcessor::new("real_output".into(), false);
        let mut stats = StreamStats::default();
        assert!(matches!(
            processor.process(frame, &mut stats).unwrap().terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }

    #[test]
    fn gpt_preflight_commits_response_created_without_synthetic_keepalive() {
        let mut active = test_active("gpt");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected response.created to commit the transparent stream");
        };
        assert!(active.business_committed);
        assert_eq!(active.commit_reason, "response_created");
        assert!(active.early_output.is_empty());
        assert_eq!(active.buffered.len(), 1);
        assert!(active.buffered[0].starts_with(b"event: response.created\n"));
        assert!(!active.buffered[0].starts_with(b"event: keepalive\n"));
    }

    #[test]
    fn gpt_preflight_keeps_every_frame_after_commit_in_the_same_chunk() {
        let mut active = test_active("gpt");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"b\"}\n\n\
                  event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected a committed stream");
        };
        assert!(active.business_committed);
        assert!(matches!(active.terminal, Some(Terminal::Completed)));
        assert_eq!(
            active.buffered.len(),
            4,
            "transparent response.created + three following frames"
        );
        assert!(active.buffered[0].starts_with(b"event: response.created\n"));
    }

    #[test]
    fn gpt_failure_after_response_created_remains_in_band() {
        let mut active = test_active("gpt");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"forward me\"}}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected the transparent stream to remain committed");
        };
        assert!(active.business_committed);
        assert_eq!(active.buffered.len(), 2);
        assert!(active.buffered[0].starts_with(b"event: response.created\n"));
        assert!(active.buffered[1].starts_with(b"event: response.failed\n"));
        assert!(matches!(
            active.terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }

    #[test]
    fn codex_response_created_still_injects_the_precommit_keepalive() {
        let mut active = test_active("codex");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected substantive Codex output to commit");
        };
        assert!(active.business_committed);
        assert_eq!(active.buffered.len(), 3);
        assert!(active.buffered[0].starts_with(b"event: keepalive\n"));
        assert!(active.buffered[1].starts_with(b"event: response.created\n"));
    }

    #[test]
    fn committing_frame_may_exceed_structural_precommit_byte_limit() {
        let mut active = test_active("codex");
        active.plan.max_precommit_bytes = 64;
        let frames = active
            .decoder
            .feed(
                b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"this frame commits the response\"}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected the substantive frame to commit");
        };
        assert!(active.business_committed);
        assert_eq!(active.buffered.len(), 1);
        assert!(active.buffered[0].len() > active.plan.max_precommit_bytes);
    }

    #[test]
    fn codex_keepalive_starts_http_but_keeps_semantic_failure_retryable() {
        let mut active = test_active("codex");
        let frames = active
            .decoder
            .feed(
                b"event: keepalive\ndata: {\"type\":\"keepalive\",\"sequence_number\":0}\n\n\
                  event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"retry me\"}}}\n\n",
            )
            .unwrap();
        let PreflightResult::Started(active) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("expected keepalive-started stream");
        };
        assert!(!active.business_committed);
        assert_eq!(active.early_output.len(), 1);
        assert_eq!(active.buffered.len(), 1, "response.created remains private");
        assert!(matches!(
            active.terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }

    #[test]
    fn semantic_failures_preserve_responses_http_status_and_detail() {
        assert_eq!(
            responses_semantic_error(
                &json!({
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "Rate limit reached"
                        }
                    }
                }),
                "response.failed",
            ),
            (429, "Rate limit reached".to_owned())
        );
        assert_eq!(
            responses_semantic_error(
                &json!({
                    "type": "error",
                    "error": {
                        "status_code": "413",
                        "message": "payload too large"
                    }
                }),
                "error",
            ),
            (413, "payload too large".to_owned())
        );
    }
}
