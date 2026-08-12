use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, RwLock};
use url::Url;

use crate::request_spool::{SpoolObservation, StoredBody};
use crate::resources::MemoryReservation;
use crate::responses::{Plan, UNLIMITED_SSE_EVENT_BYTES};
use crate::responses_item_ids::normalize_response_root;

const SNAPSHOT_SCHEMA_VERSION: u64 = 1;
const DEFAULT_MAX_PRECOMMIT_ITEMS: usize = 128;
const DEFAULT_MAX_PRECOMMIT_BYTES: usize = 8 * 1024 * 1024 + 128 * 266;
const CODEX_USER_AGENT: &str = "codex_cli_rs/0.144.0 (Debian 13.0.0; x86_64) WindowsTerminal";

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

type RouteKey = (String, String);
type RouteFailureHistory = HashMap<RouteKey, VecDeque<tokio::time::Instant>>;
type RateWindows = HashMap<(String, u64), VecDeque<tokio::time::Instant>>;

#[derive(Clone, Copy, Eq, PartialEq)]
struct SnapshotStamp {
    modified: Option<SystemTime>,
    len: u64,
}

#[derive(Clone)]
pub struct NativeConfigStore {
    path: Arc<PathBuf>,
    current: Arc<RwLock<Option<Arc<Snapshot>>>>,
    snapshot_stamp: Arc<Mutex<Option<SnapshotStamp>>>,
    provider_cursors: Arc<Mutex<HashMap<String, Arc<AtomicUsize>>>>,
    key_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    channel_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    route_failures: Arc<Mutex<RouteFailureHistory>>,
    client_windows: Arc<Mutex<RateWindows>>,
}

#[derive(Debug, Deserialize)]
struct RawSnapshot {
    schema_version: u64,
    revision: String,
    #[serde(default)]
    database_disabled: bool,
    #[serde(default)]
    preferences: Map<String, Value>,
    #[serde(default)]
    api_keys: Vec<RawApiKey>,
    #[serde(default)]
    providers: Vec<RawProvider>,
}

#[derive(Debug, Deserialize)]
struct RawApiKey {
    token: String,
    #[serde(default)]
    model_rules: Vec<Value>,
    #[serde(default)]
    role: String,
    #[serde(default)]
    weights: Map<String, Value>,
    #[serde(default)]
    preferences: Map<String, Value>,
    #[serde(default)]
    native_paid_state_safe: bool,
}

#[derive(Debug, Deserialize)]
struct RawProvider {
    name: String,
    base_url: String,
    engine: Option<String>,
    api: Value,
    #[serde(default)]
    models: HashMap<String, String>,
    #[serde(default)]
    preferences: Map<String, Value>,
    #[serde(default)]
    exclude_endpoints: Value,
}

#[derive(Clone)]
struct Snapshot {
    revision: Arc<str>,
    database_disabled: bool,
    preferences: Arc<Map<String, Value>>,
    api_keys: Arc<HashMap<String, Arc<ApiKey>>>,
    providers: Arc<Vec<Arc<Provider>>>,
    providers_by_name: Arc<HashMap<String, Arc<Provider>>>,
}

#[derive(Clone)]
struct ApiKey {
    token: Arc<str>,
    model_rules: Arc<Vec<String>>,
    role: Arc<str>,
    preferences: Arc<Map<String, Value>>,
    native_paid_state_safe: bool,
    native_supported: bool,
}

#[derive(Clone)]
struct Provider {
    name: Arc<str>,
    base_url: Arc<str>,
    engine: Arc<str>,
    api_keys: Arc<Vec<String>>,
    models: Arc<HashMap<String, String>>,
    preferences: Arc<Map<String, Value>>,
    excluded_endpoints: Arc<Vec<String>>,
    cursor: Arc<AtomicUsize>,
}

struct FailedRoute<'a> {
    provider: &'a Provider,
    key: &'a str,
    original_model: &'a str,
    has_alternative: bool,
    status: u16,
    detail: &'a str,
    force_quota_cooldown: bool,
}

#[derive(Clone, Debug)]
struct NativeAttemptObservation {
    request_id: String,
    attempt_id: String,
    attempt_index: usize,
    provider: String,
    request_model: String,
    actual_model: String,
    upstream_host: String,
    stream: bool,
    snapshot_revision: String,
    started_at: tokio::time::Instant,
}

enum ProviderKeySelection {
    Selected(String),
    NoProviderKey,
    ChannelCooling,
    AllKeysCooling,
}

struct RoutingAttemptEvent<'a> {
    attempt_number: usize,
    provider: &'a Provider,
    original_model: &'a str,
    outcome: &'a str,
    attempt_id: Option<&'a str>,
    skip_reason: Option<&'a str>,
    status: Option<u16>,
}

struct NativeRejectionObservation<'a> {
    request_id: &'a str,
    request_model: Option<&'a str>,
    stream: Option<bool>,
    request_body_bytes: u64,
    snapshot_revision: &'a str,
    reason: &'a str,
}

pub enum NativePreparation {
    Ready(NativeRoute),
    Fallback,
    Response(Response<Body>),
}

pub struct NativeRoute {
    store: NativeConfigStore,
    snapshot: Arc<Snapshot>,
    api_key: Arc<ApiKey>,
    providers: Vec<Arc<Provider>>,
    base_payload: Value,
    request_headers: HeaderMap,
    request_model: String,
    stream: bool,
    request_id: String,
    request_body_bytes: u64,
    cursor: usize,
    max_attempts: usize,
    last_provider: Option<Arc<Provider>>,
    last_provider_key: Option<String>,
    last_original_model: Option<String>,
    last_attempt: Option<NativeAttemptObservation>,
    last_status: u16,
    last_detail: String,
    has_attempt_failure: bool,
    last_failure_origin: String,
    routing_attempts: usize,
    routing_skips: usize,
    upstream_attempts: usize,
    upstream_duration_ms: u64,
    routing_ledger: Vec<Value>,
    upstream_ledger: Vec<Value>,
    started_at: tokio::time::Instant,
    final_emitted: bool,
    _memory_reservation: MemoryReservation,
}

impl NativeConfigStore {
    pub fn new() -> Self {
        let path = std::env::var("RUST_RESPONSES_CONFIG_SNAPSHOT_PATH")
            .unwrap_or_else(|_| "/tmp/uni-api-rust-responses-config-v1.json".into());
        Self {
            path: Arc::new(PathBuf::from(path)),
            current: Arc::new(RwLock::new(None)),
            snapshot_stamp: Arc::new(Mutex::new(None)),
            provider_cursors: Arc::new(Mutex::new(HashMap::new())),
            key_cooldowns: Arc::new(Mutex::new(HashMap::new())),
            channel_cooldowns: Arc::new(Mutex::new(HashMap::new())),
            route_failures: Arc::new(Mutex::new(HashMap::new())),
            client_windows: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start_watcher(&self) {
        let store = self.clone();
        tokio::spawn(async move {
            loop {
                if let Err(error) = store.refresh().await {
                    eprintln!(
                        "{}",
                        json!({
                            "event_type": "rust_responses_config_snapshot_error",
                            "error": error,
                        })
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    pub async fn refresh(&self) -> Result<bool, String> {
        let metadata = match tokio::fs::metadata(self.path.as_ref()).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(format!("stat native Responses snapshot: {error}")),
        };
        let stamp = SnapshotStamp {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        };
        if self
            .snapshot_stamp
            .lock()
            .await
            .as_ref()
            .is_some_and(|current| *current == stamp)
        {
            return Ok(false);
        }
        let bytes = match tokio::fs::read(self.path.as_ref()).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(format!("read native Responses snapshot: {error}")),
        };
        let raw: RawSnapshot = serde_json::from_slice(&bytes)
            .map_err(|error| format!("decode native Responses snapshot: {error}"))?;
        if raw.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported native Responses snapshot schema {}",
                raw.schema_version
            ));
        }
        if raw.revision.len() != 64 {
            return Err("native Responses snapshot revision is invalid".into());
        }
        *self.snapshot_stamp.lock().await = Some(stamp);
        if self
            .current
            .read()
            .await
            .as_ref()
            .is_some_and(|snapshot| snapshot.revision.as_ref() == raw.revision)
        {
            return Ok(false);
        }

        let mut cursors = self.provider_cursors.lock().await;
        let mut providers = Vec::with_capacity(raw.providers.len());
        let mut providers_by_name = HashMap::with_capacity(raw.providers.len());
        for item in raw.providers {
            let name = item.name.trim().to_owned();
            if name.is_empty() || item.models.is_empty() {
                continue;
            }
            let cursor = cursors
                .entry(name.clone())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
                .clone();
            let provider = Arc::new(Provider {
                name: name.clone().into(),
                base_url: item.base_url.trim().to_owned().into(),
                engine: item.engine.unwrap_or_else(|| "gpt".into()).into(),
                api_keys: Arc::new(provider_api_keys(&item.api)),
                models: Arc::new(item.models),
                preferences: Arc::new(item.preferences),
                excluded_endpoints: Arc::new(endpoint_values(&item.exclude_endpoints)),
                cursor,
            });
            providers_by_name.insert(name, provider.clone());
            providers.push(provider);
        }
        drop(cursors);

        let api_keys = raw
            .api_keys
            .into_iter()
            .filter_map(|item| {
                let token = item.token.trim().to_owned();
                if token.is_empty() {
                    return None;
                }
                let native_supported =
                    item.weights.is_empty() && item.model_rules.iter().all(Value::is_string);
                let rules = item
                    .model_rules
                    .into_iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect::<Vec<_>>();
                Some((
                    token.clone(),
                    Arc::new(ApiKey {
                        token: token.into(),
                        model_rules: Arc::new(rules),
                        role: item.role.into(),
                        preferences: Arc::new(item.preferences),
                        native_paid_state_safe: item.native_paid_state_safe,
                        native_supported,
                    }),
                ))
            })
            .collect::<HashMap<_, _>>();
        let snapshot = Arc::new(Snapshot {
            revision: raw.revision.into(),
            database_disabled: raw.database_disabled,
            preferences: Arc::new(raw.preferences),
            api_keys: Arc::new(api_keys),
            providers: Arc::new(providers),
            providers_by_name: Arc::new(providers_by_name),
        });
        *self.current.write().await = Some(snapshot.clone());
        eprintln!(
            "{}",
            json!({
                "event_type": "rust_responses_config_snapshot_loaded",
                "revision": snapshot.revision.as_ref(),
                "api_key_count": snapshot.api_keys.len(),
                "provider_count": snapshot.providers.len(),
            })
        );
        Ok(true)
    }

    async fn snapshot(&self) -> Option<Arc<Snapshot>> {
        self.current.read().await.clone()
    }

    async fn select_provider_key(
        &self,
        provider: &Provider,
        original_model: &str,
    ) -> ProviderKeySelection {
        if provider.api_keys.is_empty() {
            return ProviderKeySelection::NoProviderKey;
        }
        let now = tokio::time::Instant::now();
        if self
            .channel_cooldowns
            .lock()
            .await
            .get(&(provider.name.to_string(), original_model.to_owned()))
            .is_some_and(|until| *until > now)
        {
            return ProviderKeySelection::ChannelCooling;
        }
        let cooldowns = self.key_cooldowns.lock().await;
        for _ in 0..provider.api_keys.len() {
            let index = provider.cursor.fetch_add(1, Ordering::Relaxed) % provider.api_keys.len();
            let key = provider.api_keys[index].clone();
            if cooldowns
                .get(&(provider.name.to_string(), key.clone()))
                .is_none_or(|until| *until <= now)
            {
                return ProviderKeySelection::Selected(key);
            }
        }
        ProviderKeySelection::AllKeysCooling
    }

    async fn cool_failed_route(&self, failure: FailedRoute<'_>) {
        let FailedRoute {
            provider,
            key,
            original_model,
            has_alternative,
            status,
            detail,
            force_quota_cooldown,
        } = failure;
        if matches!(status, 403 | 404) {
            let now = tokio::time::Instant::now();
            let route_key = (provider.name.to_string(), original_model.to_owned());
            let mut failures = self.route_failures.lock().await;
            let history = failures.entry(route_key.clone()).or_default();
            while history.front().is_some_and(|observed| {
                now.duration_since(*observed) >= provider_model_circuit_window()
            }) {
                history.pop_front();
            }
            history.push_back(now);
            if history.len() >= provider_model_circuit_threshold() {
                self.channel_cooldowns
                    .lock()
                    .await
                    .insert(route_key, now + provider_model_circuit_open_period());
            }
        }
        let global_seconds = self
            .current
            .read()
            .await
            .as_ref()
            .and_then(|snapshot| preference_f64(&snapshot.preferences, "cooldown_period"));
        let channel_seconds = preference_f64(&provider.preferences, "cooldown_period")
            .or(global_seconds)
            .unwrap_or(0.0);
        if has_alternative && channel_seconds > 0.0 {
            self.channel_cooldowns.lock().await.insert(
                (provider.name.to_string(), original_model.to_owned()),
                tokio::time::Instant::now() + Duration::from_secs_f64(channel_seconds),
            );
        }
        if provider.api_keys.len() <= 1 {
            return;
        }
        let lower_detail = detail.to_ascii_lowercase();
        let quota_failure = force_quota_cooldown
            || matches!(status, 401..=403) && provider.engine.eq_ignore_ascii_case("codex")
            || lower_detail.contains("insufficient_quota")
            || lower_detail.contains("billing_hard_limit_reached");
        let key_seconds = if quota_failure {
            preference_f64(&provider.preferences, "api_key_quota_cooldown_period")
                .filter(|value| *value > 0.0)
                .unwrap_or(6.0 * 60.0 * 60.0)
        } else if status == 429
            && [
                "rate_limit_exceeded",
                "rate limit reached",
                "too many requests",
                "tokens per min",
                "requests per min",
                "tokens per day",
                "requests per day",
                "please try again in",
            ]
            .iter()
            .any(|marker| lower_detail.contains(marker))
        {
            preference_f64(&provider.preferences, "api_key_rate_limit_cooldown_period")
                .filter(|value| *value > 0.0)
                .unwrap_or(30.0 * 60.0)
                .max(retry_after_seconds(detail).unwrap_or(0.0))
        } else {
            preference_f64(&provider.preferences, "api_key_cooldown_period").unwrap_or(0.0)
        };
        if key_seconds > 0.0 {
            self.key_cooldowns.lock().await.insert(
                (provider.name.to_string(), key.to_owned()),
                tokio::time::Instant::now() + Duration::from_secs_f64(key_seconds),
            );
        }
    }

    async fn reset_route_failure(&self, provider: &Provider, original_model: &str) {
        self.route_failures
            .lock()
            .await
            .remove(&(provider.name.to_string(), original_model.to_owned()));
    }

    async fn admit_rate(&self, bucket: &str, rules: &[(usize, u64)]) -> bool {
        let now = tokio::time::Instant::now();
        let mut buckets = self.client_windows.lock().await;
        for (limit, seconds) in rules {
            let queue = buckets.entry((bucket.to_owned(), *seconds)).or_default();
            let window = Duration::from_secs(*seconds);
            while queue
                .front()
                .is_some_and(|started| now.duration_since(*started) >= window)
            {
                queue.pop_front();
            }
            if queue.len() >= *limit {
                return false;
            }
        }
        for (_, seconds) in rules {
            buckets
                .entry((bucket.to_owned(), *seconds))
                .or_default()
                .push_back(now);
        }
        true
    }
}

impl NativeRoute {
    pub fn stream(&self) -> bool {
        self.stream
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn last_status(&self) -> u16 {
        self.last_status
    }

    pub fn last_detail(&self) -> &str {
        &self.last_detail
    }

    pub fn has_attempts_remaining(&self) -> bool {
        self.cursor < self.max_attempts
    }

    pub async fn next_plan(&mut self) -> Result<Option<Plan>, String> {
        while self.cursor < self.max_attempts {
            let attempt_number = self.cursor;
            let provider = self.providers[attempt_number % self.providers.len()].clone();
            self.cursor += 1;
            self.routing_attempts = self.routing_attempts.saturating_add(1);
            let original_model = provider
                .models
                .get(&self.request_model)
                .ok_or_else(|| "native provider model mapping disappeared".to_owned())?
                .clone();
            let provider_key = match self
                .store
                .select_provider_key(&provider, &original_model)
                .await
            {
                ProviderKeySelection::Selected(key) => key,
                selection => {
                    self.routing_skips = self.routing_skips.saturating_add(1);
                    let reason = match selection {
                        ProviderKeySelection::NoProviderKey => "provider_has_no_api_keys",
                        ProviderKeySelection::ChannelCooling => "provider_channel_cooldown",
                        ProviderKeySelection::AllKeysCooling => "provider_keys_cooldown",
                        ProviderKeySelection::Selected(_) => unreachable!(),
                    };
                    self.emit_routing_attempt(RoutingAttemptEvent {
                        attempt_number,
                        provider: &provider,
                        original_model: &original_model,
                        outcome: "skipped",
                        attempt_id: None,
                        skip_reason: Some(reason),
                        status: None,
                    });
                    if !self.has_attempt_failure {
                        self.last_status = 429;
                        self.last_detail =
                            "All API keys are rate limited and stop auto retry!".into();
                    }
                    continue;
                }
            };
            let engine = provider.engine.to_ascii_lowercase();
            if engine != "gpt" && engine != "codex" {
                self.routing_skips = self.routing_skips.saturating_add(1);
                self.emit_routing_attempt(RoutingAttemptEvent {
                    attempt_number,
                    provider: &provider,
                    original_model: &original_model,
                    outcome: "skipped",
                    attempt_id: None,
                    skip_reason: Some("unsupported_native_engine"),
                    status: None,
                });
                continue;
            }
            if engine == "codex" && provider_key.contains(',') {
                return Err("native-codex-oauth-fallback".into());
            }
            let mut payload = self.base_payload.clone();
            compile_payload(
                &mut payload,
                &provider,
                &self.request_model,
                &original_model,
                &engine,
            )?;
            let body = serde_json::to_string(&payload)
                .map_err(|error| format!("encode native upstream payload: {error}"))?;
            let attempt_id = native_attempt_id(&self.request_id, attempt_number);
            let headers = build_headers(
                &self.request_headers,
                &provider,
                &provider_key,
                &engine,
                self.stream,
                &self.request_id,
                &attempt_id,
            )?;
            let timeout = resolve_timeouts(
                &self.snapshot,
                &provider,
                &self.request_model,
                &original_model,
                &engine,
                self.stream,
                self.api_key.role.as_ref(),
            );
            self.upstream_attempts = self.upstream_attempts.saturating_add(1);
            let observation = NativeAttemptObservation {
                request_id: self.request_id.clone(),
                attempt_id: attempt_id.clone(),
                attempt_index: attempt_number.saturating_add(1),
                provider: provider.name.to_string(),
                request_model: self.request_model.clone(),
                actual_model: original_model.clone(),
                upstream_host: upstream_host(&provider.base_url),
                stream: self.stream,
                snapshot_revision: self.snapshot.revision.to_string(),
                started_at: tokio::time::Instant::now(),
            };
            self.last_provider = Some(provider.clone());
            self.last_provider_key = Some(provider_key);
            self.last_original_model = Some(original_model.clone());
            self.last_attempt = Some(observation.clone());
            return Ok(Some(Plan {
                attempt_id,
                url: normalize_upstream_url(&provider.base_url, &engine),
                headers,
                body,
                proxy: preference_string(&provider.preferences, "proxy")
                    .or_else(|| preference_string(&self.snapshot.preferences, "proxy")),
                engine: engine.clone(),
                precommit_semantic_guard: Some(engine == "codex"),
                http1_only: engine == "codex",
                commit_policy: preference_string(
                    &provider.preferences,
                    "responses_stream_commit_policy",
                )
                .unwrap_or_else(|| "real_output".into()),
                normalize_custom_tool_call_ids: normalization_enabled(
                    &provider,
                    &self.request_model,
                    &original_model,
                ),
                first_byte_timeout_seconds: timeout.first_byte,
                idle_timeout_seconds: timeout.idle,
                total_timeout_seconds: timeout.total,
                max_event_bytes: UNLIMITED_SSE_EVENT_BYTES,
                max_precommit_items: DEFAULT_MAX_PRECOMMIT_ITEMS,
                max_precommit_bytes: DEFAULT_MAX_PRECOMMIT_BYTES,
            }));
        }
        Ok(None)
    }

    pub async fn record_failure(&mut self, outcome: &Value) -> bool {
        let original_status = outcome
            .get("status_code")
            .and_then(Value::as_u64)
            .unwrap_or(502)
            .min(u16::MAX as u64) as u16;
        let detail = outcome
            .get("detail")
            .or_else(|| outcome.get("body"))
            .and_then(Value::as_str)
            .unwrap_or("Responses upstream attempt failed");
        let status = remap_provider_status(original_status, detail);
        let codex_model_unsupported = status == 400
            && self
                .last_provider
                .as_ref()
                .is_some_and(|provider| provider.engine.eq_ignore_ascii_case("codex"))
            && detail
                .to_ascii_lowercase()
                .contains("model is not supported when using codex with a chatgpt account");
        let missing_persisted_item = status == 404 && is_missing_persisted_item_error(detail);
        let request_scoped = matches!(status, 400 | 413) || missing_persisted_item;
        self.last_status = status;
        self.last_detail = detail.chars().take(4096).collect();
        self.has_attempt_failure = true;
        self.last_failure_origin = failure_origin(outcome).to_owned();
        let upstream_status = outcome_status_from(outcome, "upstream_status_code", original_status);
        self.emit_upstream_attempt(outcome, upstream_status, false);
        if let (Some(provider), Some(original_model)) =
            (self.last_provider.clone(), self.last_original_model.clone())
        {
            let attempt_id = self
                .last_attempt
                .as_ref()
                .map(|attempt| attempt.attempt_id.clone());
            self.emit_routing_attempt(RoutingAttemptEvent {
                attempt_number: self.cursor.saturating_sub(1),
                provider: &provider,
                original_model: &original_model,
                outcome: "failed",
                attempt_id: attempt_id.as_deref(),
                skip_reason: None,
                status: Some(status),
            });
        }
        if !request_scoped || codex_model_unsupported {
            if let (Some(provider), Some(key), Some(original_model)) = (
                self.last_provider.as_ref(),
                self.last_provider_key.as_deref(),
                self.last_original_model.as_deref(),
            ) {
                self.store
                    .cool_failed_route(FailedRoute {
                        provider,
                        key,
                        original_model,
                        has_alternative: self.providers.len() > 1,
                        status,
                        detail,
                        force_quota_cooldown: codex_model_unsupported,
                    })
                    .await;
            }
        }
        self.auto_retry()
            && (!request_scoped || codex_model_unsupported)
            && self.has_attempts_remaining()
    }

    pub async fn record_success(&self) {
        if let (Some(provider), Some(original_model)) = (
            self.last_provider.as_ref(),
            self.last_original_model.as_deref(),
        ) {
            self.store
                .reset_route_failure(provider, original_model)
                .await;
        }
    }

    pub async fn complete_native(&mut self, outcome: &Value) {
        let kind = outcome
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("completed");
        let success = matches!(kind, "completed" | "incomplete");
        let status = outcome_status(outcome, if success { 200 } else { 502 });
        let upstream_status = outcome_status_from(outcome, "upstream_status_code", status);
        if success {
            self.record_success().await;
        } else {
            self.has_attempt_failure = true;
            self.last_failure_origin = failure_origin(outcome).to_owned();
        }
        self.emit_upstream_attempt(outcome, upstream_status, success);
        if let (Some(provider), Some(original_model)) =
            (self.last_provider.clone(), self.last_original_model.clone())
        {
            let attempt_id = self
                .last_attempt
                .as_ref()
                .map(|attempt| attempt.attempt_id.clone());
            self.emit_routing_attempt(RoutingAttemptEvent {
                attempt_number: self.cursor.saturating_sub(1),
                provider: &provider,
                original_model: &original_model,
                outcome: if success {
                    "succeeded"
                } else {
                    "completed_with_error"
                },
                attempt_id: attempt_id.as_deref(),
                skip_reason: None,
                status: Some(status),
            });
        }
        self.emit_final_event(status, kind, outcome);
    }

    pub fn emit_final_response(&mut self, status: u16, kind: &str) {
        self.emit_final_event(status, kind, &Value::Null);
    }

    pub fn emit_internal_failure(&mut self, status: u16, kind: &str, detail: &str) {
        self.last_status = status;
        self.last_detail = detail.chars().take(4096).collect();
        self.has_attempt_failure = true;
        self.last_failure_origin = "ember_native".into();
        self.emit_final_event(status, kind, &json!({"detail": detail}));
    }

    pub fn final_message(&mut self) -> Value {
        let detail = if self.last_detail.is_empty() {
            format!("All {} providers failed", self.request_model)
        } else {
            self.last_detail.clone()
        };
        let status = if self.last_status == 0 {
            502
        } else {
            self.last_status
        };
        self.emit_final_response(status, "failed_before_commit");
        json!({
            "kind": "final",
            "status_code": status,
            "body_b64": BASE64.encode(detail.as_bytes()),
        })
    }

    fn emit_routing_attempt(&mut self, event: RoutingAttemptEvent<'_>) {
        let status = event.status.unwrap_or_default();
        if self.routing_ledger.len() < 64 {
            self.routing_ledger.push(json!({
                "attempt_id": event.attempt_id,
                "attempt_index": event.attempt_number.saturating_add(1),
                "provider": event.provider.name.to_string(),
                "actual_model": event.original_model,
                "outcome": event.outcome,
                "status_code": event.status,
                "skip_reason": event.skip_reason,
            }));
        }
        eprintln!(
            "{}",
            json!({
                "kind": "log",
                "fugue_table": "app_events",
                "event": "routing_attempt",
                "event_type": "routing_attempt",
                "severity": event_severity(status, event.outcome),
                "source": "uni-api-ember",
                "message": "uni-api-ember native routing attempt",
                "request_id": self.request_id,
                "trace_id": self.request_id,
                "path": "/v1/responses",
                "path_template": "/v1/responses",
                "route": "POST /v1/responses",
                "method": "POST",
                "model": self.request_model,
                "provider": event.provider.name.to_string(),
                "channel": event.provider.name.to_string(),
                "actual_model": event.original_model,
                "attempt_id": event.attempt_id,
                "attempt_index": event.attempt_number.saturating_add(1),
                "attempt_outcome": event.outcome,
                "attempt_status_code": if status == 0 { None } else { Some(status) },
                "skip_reason": event.skip_reason,
                "streaming": self.stream,
                "snapshot_revision": self.snapshot.revision.to_string(),
                "rust_responses_data_plane": true,
            })
        );
    }

    fn emit_upstream_attempt(&mut self, outcome: &Value, status: u16, success: bool) {
        let Some(attempt) = self.last_attempt.clone() else {
            return;
        };
        let detail = outcome
            .get("detail")
            .or_else(|| outcome.get("body"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let error_sha256 = (!detail.is_empty()).then(|| sha256_hex(detail));
        let duration_ms = attempt
            .started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64;
        self.upstream_duration_ms = self.upstream_duration_ms.saturating_add(duration_ms);
        let attempt_outcome = outcome
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or(if success { "completed" } else { "failed" });
        if self.upstream_ledger.len() < 64 {
            self.upstream_ledger.push(json!({
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "provider": attempt.provider,
                "actual_model": attempt.actual_model,
                "upstream_host": attempt.upstream_host,
                "status_code": status,
                "success": success,
                "outcome": attempt_outcome,
                "error_sha256": error_sha256,
                "duration_ms": duration_ms,
            }));
        }
        eprintln!(
            "{}",
            json!({
                "kind": "log",
                "fugue_table": "app_events",
                "event": "upstream_attempt",
                "event_type": "upstream_attempt",
                "severity": event_severity(status, if success { "succeeded" } else { "failed" }),
                "source": "uni-api-ember",
                "message": "uni-api-ember native upstream attempt",
                "request_id": attempt.request_id,
                "trace_id": attempt.request_id,
                "path": "/v1/responses",
                "path_template": "/v1/responses",
                "route": "POST /v1/responses",
                "method": "POST",
                "model": attempt.request_model,
                "provider": attempt.provider,
                "channel": attempt.provider,
                "actual_model": attempt.actual_model,
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "attempt_status_code": status,
                "attempt_status_class": status_class(status),
                "semantic_status_code": outcome.get("status_code").and_then(Value::as_u64),
                "attempt_success": success,
                "attempt_outcome": attempt_outcome,
                "status_origin": failure_origin(outcome),
                "error_sha256": error_sha256,
                "duration_ms": duration_ms,
                "upstream_host": attempt.upstream_host,
                "streaming": attempt.stream,
                "snapshot_revision": attempt.snapshot_revision,
                "rust_responses_data_plane": true,
            })
        );
    }

    fn emit_final_event(&mut self, status: u16, kind: &str, outcome: &Value) {
        if self.final_emitted {
            return;
        }
        self.final_emitted = true;
        let elapsed_ms = self
            .started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64;
        let success = matches!(kind, "completed" | "incomplete");
        let detail = if success {
            ""
        } else {
            outcome
                .get("detail")
                .or_else(|| outcome.get("body"))
                .and_then(Value::as_str)
                .unwrap_or(&self.last_detail)
        };
        let final_provider = self
            .last_attempt
            .as_ref()
            .map(|attempt| attempt.provider.as_str());
        let final_actual_model = self
            .last_attempt
            .as_ref()
            .map(|attempt| attempt.actual_model.as_str());
        let status_origin = if success {
            "upstream_success"
        } else if self.last_failure_origin.is_empty() {
            "native_route_selection"
        } else {
            self.last_failure_origin.as_str()
        };
        let summary = json!({
            "request_kind": "responses",
            "terminal_kind": kind,
            "model": self.request_model,
            "provider": final_provider,
            "channel": final_provider,
            "actual_model": final_actual_model,
            "stream": self.stream,
            "status_code": status,
            "status_class": status_class(status),
            "status_origin": status_origin,
            "error_type": (!success || status >= 400).then_some(kind),
            "routing_attempt_count": self.routing_attempts,
            "routing_skip_count": self.routing_skips,
            "upstream_attempt_count": self.upstream_attempts,
            "upstream_duration_ms": self.upstream_duration_ms,
            "routing_attempts": self.routing_ledger,
            "upstream_attempts": self.upstream_ledger,
            "routing_attempts_omitted_count": self.routing_attempts.saturating_sub(self.routing_ledger.len()),
            "upstream_attempts_omitted_count": self.upstream_attempts.saturating_sub(self.upstream_ledger.len()),
            "last_failure_origin": self.last_failure_origin,
            "snapshot_revision": self.snapshot.revision.to_string(),
            "rust_responses_data_plane": true,
        });
        eprintln!(
            "{}",
            json!({
                "kind": "log",
                "fugue_table": "request_facts",
                "event": "request_summary",
                "event_type": "request_summary",
                "severity": event_severity(status, kind),
                "source": "uni-api-ember",
                "message": "uni-api-ember native Responses request finished",
                "request_id": self.request_id,
                "trace_id": self.request_id,
                "path": "/v1/responses",
                "path_template": "/v1/responses",
                "route": "POST /v1/responses",
                "route_id": "POST /v1/responses",
                "method": "POST",
                "model": self.request_model,
                "provider": final_provider,
                "channel": final_provider,
                "actual_model": final_actual_model,
                "status_code": status,
                "status_class": status_class(status),
                "duration_ms": elapsed_ms,
                "upstream_ms": self.upstream_duration_ms,
                "bytes_in": self.request_body_bytes,
                "bytes_out": outcome.get("downstream_bytes").and_then(Value::as_u64).unwrap_or(0),
                "streaming": self.stream,
                "error_type": (!success || status >= 400).then_some(kind),
                "status_origin": status_origin,
                "error_sha256": terminal_error_sha256(success, detail),
                "summary_json": summary.to_string(),
                "rust_responses_data_plane": true,
            })
        );
    }

    fn auto_retry(&self) -> bool {
        self.api_key
            .preferences
            .get("AUTO_RETRY")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }
}

fn emit_native_rejection(status: u16, observation: NativeRejectionObservation<'_>) {
    let status_origin = native_rejection_origin(observation.reason);
    let summary = json!({
        "request_kind": "responses",
        "terminal_kind": "native_rejection",
        "rejection_reason": observation.reason,
        "model": observation.request_model,
        "stream": observation.stream,
        "status_code": status,
        "status_class": status_class(status),
        "status_origin": status_origin,
        "error_type": observation.reason,
        "routing_attempt_count": 0,
        "routing_skip_count": 0,
        "upstream_attempt_count": 0,
        "snapshot_revision": observation.snapshot_revision,
        "rust_responses_data_plane": true,
    });
    eprintln!(
        "{}",
        json!({
            "kind": "log",
            "fugue_table": "request_facts",
            "event": "request_summary",
            "event_type": "request_summary",
            "severity": event_severity(status, "native_rejection"),
            "source": "uni-api-ember",
            "message": "uni-api-ember native Responses request rejected",
            "request_id": observation.request_id,
            "trace_id": observation.request_id,
            "path": "/v1/responses",
            "path_template": "/v1/responses",
            "route": "POST /v1/responses",
            "route_id": "POST /v1/responses",
            "method": "POST",
            "model": observation.request_model,
            "status_code": status,
            "status_class": status_class(status),
            "duration_ms": 0,
            "upstream_ms": 0,
            "bytes_in": observation.request_body_bytes,
            "bytes_out": 0,
            "streaming": observation.stream,
            "error_type": observation.reason,
            "status_origin": status_origin,
            "summary_json": summary.to_string(),
            "rust_responses_data_plane": true,
        })
    );
}

pub async fn prepare_native_request(
    store: &NativeConfigStore,
    parts: &Parts,
    storage: &StoredBody,
    observation: &SpoolObservation,
    memory_reservation: MemoryReservation,
) -> NativePreparation {
    let Some(snapshot) = store.snapshot().await else {
        return NativePreparation::Fallback;
    };
    if !is_identity_json_request(&parts.headers) {
        return NativePreparation::Fallback;
    }
    let request_id = request_id(&parts.headers);
    let Some(token) = extract_api_key(&parts.headers) else {
        emit_native_rejection(
            403,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: None,
                stream: None,
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "invalid_api_key",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "Invalid or missing API Key"}),
        ));
    };
    let Some(api_key) = snapshot.api_keys.get(&token).cloned() else {
        emit_native_rejection(
            403,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: None,
                stream: None,
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "invalid_api_key",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "Invalid or missing API Key"}),
        ));
    };
    if !snapshot.database_disabled
        || !api_key.native_supported
        || !api_key.native_paid_state_safe
        || api_key
            .preferences
            .get("ENABLE_MODERATION")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || api_key
            .preferences
            .get("SCHEDULING_ALGORITHM")
            .and_then(Value::as_str)
            .is_some_and(|algorithm| algorithm != "fixed_priority")
    {
        return NativePreparation::Fallback;
    }
    let mut payload = match storage.parse_json().await {
        Ok(Value::Object(payload)) => Value::Object(payload),
        Ok(_) => {
            emit_native_rejection(
                422,
                NativeRejectionObservation {
                    request_id: &request_id,
                    request_model: None,
                    stream: None,
                    request_body_bytes: observation.body_bytes,
                    snapshot_revision: snapshot.revision.as_ref(),
                    reason: "request_body_not_object",
                },
            );
            return NativePreparation::Response(json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail": "Request body must be a JSON object"}),
            ));
        }
        Err(error) => {
            emit_native_rejection(
                422,
                NativeRejectionObservation {
                    request_id: &request_id,
                    request_model: None,
                    stream: None,
                    request_body_bytes: observation.body_bytes,
                    snapshot_revision: snapshot.revision.as_ref(),
                    reason: "request_body_invalid_json",
                },
            );
            return NativePreparation::Response(json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail": error}),
            ));
        }
    };
    let object = payload.as_object().expect("checked JSON object");
    let Some(request_model) = object
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    else {
        emit_native_rejection(
            422,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: None,
                stream: None,
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "request_model_missing",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": "Request body requires a model"}),
        ));
    };
    if !object.contains_key("input") {
        emit_native_rejection(
            422,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: Some(&request_model),
                stream: None,
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "request_input_missing",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": "Request body requires input"}),
        ));
    }
    let stream_value = object.get("stream").cloned();
    let stream = match stream_value.as_ref() {
        None | Some(Value::Null) => false,
        Some(value) => match pydantic_bool(value) {
            Some(value) => value,
            None => {
                emit_native_rejection(
                    422,
                    NativeRejectionObservation {
                        request_id: &request_id,
                        request_model: Some(&request_model),
                        stream: None,
                        request_body_bytes: observation.body_bytes,
                        snapshot_revision: snapshot.revision.as_ref(),
                        reason: "request_stream_invalid",
                    },
                );
                return NativePreparation::Response(json_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({"detail": "stream must be a boolean"}),
                ));
            }
        },
    };
    if stream_value.is_some_and(|value| !matches!(value, Value::Null | Value::Bool(_))) {
        payload
            .as_object_mut()
            .expect("checked JSON object")
            .insert("stream".into(), Value::Bool(stream));
    }
    let providers = match matching_providers(
        &snapshot,
        &api_key,
        &request_model,
        observation.body_bytes,
    ) {
        Ok(providers) if !providers.is_empty() => providers,
        Ok(_) => {
            emit_native_rejection(
                404,
                NativeRejectionObservation {
                    request_id: &request_id,
                    request_model: Some(&request_model),
                    stream: Some(stream),
                    request_body_bytes: observation.body_bytes,
                    snapshot_revision: snapshot.revision.as_ref(),
                    reason: "no_matching_provider",
                },
            );
            return NativePreparation::Response(json_response(
                StatusCode::NOT_FOUND,
                json!({"message": format!("No available providers at the moment: {request_model}")}),
            ));
        }
        Err(()) => return NativePreparation::Fallback,
    };
    if providers.iter().any(|provider| {
        !matches!(provider.engine.as_ref(), "gpt" | "codex")
            || provider.api_keys.is_empty()
            || (provider.engine.as_ref() == "codex"
                && provider.api_keys.iter().any(|key| key.contains(',')))
            || (provider.engine.as_ref() == "gpt" && !provider.base_url.contains("v1/responses"))
            || provider_stream_override(provider).is_some_and(|value| value != stream)
    }) {
        return NativePreparation::Fallback;
    }
    let Some(global_rate_rules) = parse_rate_limits(snapshot.preferences.get("rate_limit"), None)
    else {
        return NativePreparation::Fallback;
    };
    let Some(client_rate_rules) =
        parse_rate_limits(api_key.preferences.get("rate_limit"), Some(&request_model))
    else {
        return NativePreparation::Fallback;
    };
    // Admit only after the request is proven native-safe. A compatibility
    // fallback must not consume both the Rust and Python rate-limit buckets.
    if !store.admit_rate("__global__", &global_rate_rules).await {
        emit_native_rejection(
            429,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: Some(&request_model),
                stream: Some(stream),
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "native_global_rate_limit",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "Too many requests"}),
        ));
    }
    if !store
        .admit_rate(&format!("client:{}", api_key.token), &client_rate_rules)
        .await
    {
        emit_native_rejection(
            429,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: Some(&request_model),
                stream: Some(stream),
                request_body_bytes: observation.body_bytes,
                snapshot_revision: snapshot.revision.as_ref(),
                reason: "native_client_rate_limit",
            },
        );
        return NativePreparation::Response(json_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "Too many requests"}),
        ));
    }
    let retry_count = compute_retry_count(&providers);
    NativePreparation::Ready(NativeRoute {
        store: store.clone(),
        snapshot,
        api_key,
        providers,
        base_payload: payload,
        request_headers: parts.headers.clone(),
        request_model,
        stream,
        request_id,
        request_body_bytes: observation.body_bytes,
        cursor: 0,
        max_attempts: retry_count,
        last_provider: None,
        last_provider_key: None,
        last_original_model: None,
        last_attempt: None,
        last_status: 502,
        last_detail: String::new(),
        has_attempt_failure: false,
        last_failure_origin: String::new(),
        routing_attempts: 0,
        routing_skips: 0,
        upstream_attempts: 0,
        upstream_duration_ms: 0,
        routing_ledger: Vec::new(),
        upstream_ledger: Vec::new(),
        started_at: tokio::time::Instant::now(),
        final_emitted: false,
        _memory_reservation: memory_reservation,
    })
}

fn matching_providers(
    snapshot: &Snapshot,
    api_key: &ApiKey,
    request_model: &str,
    request_body_bytes: u64,
) -> Result<Vec<Arc<Provider>>, ()> {
    let mut matches = Vec::new();
    for rule in api_key.model_rules.iter() {
        if rule == "all" {
            matches.extend(
                snapshot
                    .providers
                    .iter()
                    .filter(|provider| provider.models.contains_key(request_model))
                    .cloned(),
            );
            continue;
        }
        if rule.starts_with('<') && rule.ends_with('>') {
            if &rule[1..rule.len() - 1] == request_model {
                matches.extend(
                    snapshot
                        .providers
                        .iter()
                        .filter(|provider| provider.models.contains_key(request_model))
                        .cloned(),
                );
            }
            continue;
        }
        if let Some((provider_name, model_rule)) = rule.split_once('/') {
            let Some(provider) = snapshot.providers_by_name.get(provider_name) else {
                // Nested API-key providers remain on the Python compatibility path.
                if snapshot.api_keys.contains_key(provider_name) {
                    return Err(());
                }
                continue;
            };
            if (model_rule == "*" || model_rule == request_model)
                && provider.models.contains_key(request_model)
            {
                matches.push(provider.clone());
            }
            continue;
        }
        if rule == request_model {
            matches.extend(
                snapshot
                    .providers
                    .iter()
                    .filter(|provider| provider.models.contains_key(request_model))
                    .cloned(),
            );
        }
    }
    matches.retain(|provider| {
        !provider
            .excluded_endpoints
            .iter()
            .any(|endpoint| endpoint.trim_end_matches('/') == "/v1/responses")
            && provider_accepts_body(provider, request_body_bytes)
    });
    Ok(matches)
}

fn compute_retry_count(providers: &[Arc<Provider>]) -> usize {
    if providers.is_empty() {
        return 0;
    }
    let retry = if providers.len() == 1 && providers[0].api_keys.len() > 1 {
        providers[0].api_keys.len()
    } else {
        providers
            .iter()
            .map(|provider| provider.api_keys.len())
            .sum::<usize>()
            .saturating_mul(2)
            .min(10)
    };
    providers.len().saturating_add(retry)
}

fn provider_accepts_body(provider: &Provider, bytes: u64) -> bool {
    let Some(raw) = provider.preferences.get("max_request_body_bytes") else {
        return true;
    };
    parse_byte_limit(raw).is_none_or(|limit| bytes <= limit)
}

fn parse_byte_limit(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return (value > 0).then_some(value);
    }
    let raw = value.as_str()?.trim().to_ascii_lowercase().replace('_', "");
    let split = raw
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(raw.len());
    let number = raw[..split].trim().parse::<f64>().ok()?;
    let unit = raw[split..].trim();
    let multiplier = match unit {
        "" | "b" | "byte" | "bytes" => 1.0,
        "k" | "kb" => 1_000.0,
        "ki" | "kib" => 1_024.0,
        "m" | "mb" => 1_000_000.0,
        "mi" | "mib" => 1_048_576.0,
        "g" | "gb" => 1_000_000_000.0,
        "gi" | "gib" => 1_073_741_824.0,
        _ => return None,
    };
    let bytes = (number * multiplier) as u64;
    (bytes > 0).then_some(bytes)
}

fn compile_payload(
    payload: &mut Value,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    engine: &str,
) -> Result<(), String> {
    let root = payload
        .as_object_mut()
        .ok_or_else(|| "native Responses payload is not an object".to_owned())?;
    root.insert("model".into(), Value::String(original_model.to_owned()));
    if engine == "codex" {
        for key in [
            "previous_response_id",
            "prompt_cache_retention",
            "safety_identifier",
        ] {
            root.remove(key);
        }
        root.entry("instructions")
            .or_insert(Value::String(String::new()));
    }
    apply_overrides(root, provider, request_model);
    if engine == "codex" {
        strip_codex_fields(root);
    }
    if normalization_enabled(provider, request_model, original_model) {
        normalize_response_root(root)?;
    }
    Ok(())
}

fn apply_overrides(root: &mut Map<String, Value>, provider: &Provider, request_model: &str) {
    let Some(overrides) = provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
    else {
        return;
    };
    apply_override_section(root, overrides, provider, true);
    if let Some(model) = overrides.get(request_model).and_then(Value::as_object) {
        apply_override_section(root, model, provider, false);
    }
}

fn apply_override_section(
    root: &mut Map<String, Value>,
    section: &Map<String, Value>,
    provider: &Provider,
    skip_model_keys: bool,
) {
    for (key, value) in section {
        if key == "__remove__"
            || matches!(key.as_str(), "service_tier" | "translation_options")
            || (skip_model_keys && provider.models.contains_key(key))
        {
            continue;
        }
        merge_value(root.entry(key.clone()).or_insert(Value::Null), value);
    }
    if let Some(removals) = section.get("__remove__") {
        apply_removals(root, removals);
    }
}

fn merge_value(target: &mut Value, replacement: &Value) {
    match (target, replacement) {
        (Value::Object(target), Value::Object(replacement)) => {
            for (key, value) in replacement {
                if key == "__remove__" {
                    continue;
                }
                merge_value(target.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (target, replacement) => *target = replacement.clone(),
    }
}

fn apply_removals(root: &mut Map<String, Value>, removals: &Value) {
    let items = match removals {
        Value::Array(items) => items.clone(),
        value => vec![value.clone()],
    };
    for item in items {
        if let Some(path) = item.as_str() {
            if !matches!(path, "service_tier" | "translation_options") {
                delete_path(root, path);
            }
        } else if let Some(rule) = item.as_object() {
            if !matches!(
                rule.get("path").and_then(Value::as_str),
                Some("service_tier" | "translation_options")
            ) {
                apply_structured_removal(root, rule);
            }
        }
    }
}

fn apply_structured_removal(root: &mut Map<String, Value>, rule: &Map<String, Value>) {
    let Some(path) = rule.get("path").and_then(Value::as_str) else {
        return;
    };
    if !rule.contains_key("where") && !rule.contains_key("where_any") {
        delete_path(root, path);
        return;
    }
    let Some(target) = get_path_mut(root, path) else {
        return;
    };
    let should_remove = |value: &Value| {
        rule.get("where")
            .is_some_and(|condition| matches_condition(value, condition))
            || rule
                .get("where_any")
                .is_some_and(|condition| matches_any_condition(value, condition))
    };
    if let Some(items) = target.as_array_mut() {
        items.retain(|item| !should_remove(item));
        if items.is_empty() && rule.get("drop_empty").and_then(Value::as_bool) == Some(true) {
            delete_path(root, path);
        }
    } else if should_remove(target) {
        delete_path(root, path);
    }
}

fn matches_any_condition(value: &Value, condition: &Value) -> bool {
    condition
        .as_array()
        .map(|conditions| conditions.iter().any(|item| matches_condition(value, item)))
        .unwrap_or_else(|| matches_condition(value, condition))
}

fn matches_condition(value: &Value, condition: &Value) -> bool {
    if condition.is_string() {
        return value == condition;
    }
    let Some(condition) = condition.as_object() else {
        return false;
    };
    condition.iter().all(|(path, expected)| {
        get_value_path(value, path).is_some_and(|actual| actual == expected)
    })
}

fn delete_path(root: &mut Map<String, Value>, path: &str) {
    let parts = path
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        let Some(next) = current.get_mut(*part).and_then(Value::as_object_mut) else {
            return;
        };
        current = next;
    }
    current.remove(parts[parts.len() - 1]);
}

fn get_path_mut<'a>(root: &'a mut Map<String, Value>, path: &str) -> Option<&'a mut Value> {
    let mut parts = path.split('.').filter(|part| !part.trim().is_empty());
    let first = parts.next()?;
    let mut current = root.get_mut(first)?;
    for part in parts {
        current = current.as_object_mut()?.get_mut(part)?;
    }
    Some(current)
}

fn get_value_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for part in path.split('.').filter(|part| !part.trim().is_empty()) {
        current = current.as_object()?.get(part)?;
    }
    Some(current)
}

fn strip_codex_fields(root: &mut Map<String, Value>) {
    for key in [
        "max_output_tokens",
        "response_format",
        "top_p",
        "truncation",
    ] {
        root.remove(key);
    }
    root.remove("cache_control");
    root.remove("reasoning_content");
    for value in root.values_mut() {
        strip_key_recursive(value, "cache_control");
        strip_key_recursive(value, "reasoning_content");
    }
    if let Some(input) = root.get_mut("input").and_then(Value::as_array_mut) {
        for item in input.iter_mut().filter_map(Value::as_object_mut) {
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                item.remove("id");
            }
            if item.get("type").and_then(Value::as_str) == Some("message") {
                item.remove("reasoning");
                item.remove("reasoning_content");
            }
        }
    }
}

fn provider_stream_override(provider: &Provider) -> Option<bool> {
    provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get("stream"))
        .and_then(Value::as_bool)
}

fn strip_key_recursive(value: &mut Value, key: &str) {
    match value {
        Value::Object(object) => {
            object.remove(key);
            for child in object.values_mut() {
                strip_key_recursive(child, key);
            }
        }
        Value::Array(items) => {
            for child in items {
                strip_key_recursive(child, key);
            }
        }
        _ => {}
    }
}

fn normalization_enabled(provider: &Provider, request_model: &str, original_model: &str) -> bool {
    match provider
        .preferences
        .get("normalize_responses_custom_tool_call_ids")
    {
        Some(Value::Bool(value)) => *value,
        Some(Value::Array(models)) => models.iter().any(|model| {
            model.as_str().is_some_and(|model| {
                model == "*" || model == request_model || model == original_model
            })
        }),
        Some(_) => false,
        None => provider.engine.as_ref() == "codex",
    }
}

fn build_headers(
    incoming: &HeaderMap,
    provider: &Provider,
    provider_key: &str,
    engine: &str,
    stream: bool,
    request_id: &str,
    attempt_id: &str,
) -> Result<HashMap<String, String>, String> {
    let mut headers = HashMap::from([
        ("Content-Type".into(), "application/json".into()),
        ("Authorization".into(), format!("Bearer {provider_key}")),
        ("x-request-id".into(), request_id.to_owned()),
        ("x-uni-api-ember-request-id".into(), request_id.to_owned()),
        ("x-caller-request-id".into(), request_id.to_owned()),
        ("x-caller-app".into(), "uni-api-ember".into()),
    ]);
    if engine == "codex" {
        headers.insert(
            "Openai-Beta".into(),
            header_or(incoming, "openai-beta", "responses=experimental"),
        );
        headers.insert(
            "Originator".into(),
            header_or(incoming, "originator", "codex_cli_rs"),
        );
        headers.insert(
            "Session_id".into(),
            header_or(incoming, "session_id", request_id),
        );
        headers.insert("User-Agent".into(), CODEX_USER_AGENT.into());
        headers.insert(
            "Accept".into(),
            if stream {
                "text/event-stream"
            } else {
                "application/json"
            }
            .into(),
        );
    }
    if let Some(extra) = provider
        .preferences
        .get("headers")
        .and_then(Value::as_object)
    {
        for (name, value) in extra {
            if let Some(value) = value.as_str() {
                headers.insert(name.clone(), value.to_owned());
            }
        }
    }
    let passthrough = provider
        .preferences
        .get("passthrough_request_headers")
        .map(endpoint_values)
        .unwrap_or_default();
    for name in passthrough {
        headers.retain(|existing, _| !existing.eq_ignore_ascii_case(&name));
        if let Some(value) = incoming.get(&name).and_then(|value| value.to_str().ok()) {
            headers.insert(name, value.to_owned());
        }
    }
    if provider
        .preferences
        .get("oaix_routing_attempt_id")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        headers.insert("X-OAIX-Routing-Attempt-ID".into(), attempt_id.to_owned());
    }
    for (name, value) in &headers {
        HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| format!("provider {} produced invalid header {name}", provider.name))?;
        HeaderValue::from_str(value)
            .map_err(|_| format!("provider {} produced invalid header value", provider.name))?;
    }
    Ok(headers)
}

fn header_or(headers: &HeaderMap, name: &str, default: &str) -> String {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

struct Timeouts {
    first_byte: Option<f64>,
    idle: Option<f64>,
    total: Option<f64>,
}

fn resolve_timeouts(
    snapshot: &Snapshot,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    engine: &str,
    stream: bool,
    role: &str,
) -> Timeouts {
    let base = model_timeout(
        provider,
        &snapshot.preferences,
        request_model,
        original_model,
    );
    let context = HashMap::from([
        ("provider", provider.name.as_ref()),
        ("endpoint", "/v1/responses"),
        ("method", "POST"),
        ("engine", engine),
        ("model", request_model),
        ("request_model", request_model),
        ("upstream_model", original_model),
        ("role", role),
    ]);
    let mut values = Map::new();
    merge_timeout_policy(
        &mut values,
        snapshot.preferences.get("timeout_policy"),
        &context,
        stream,
    );
    merge_timeout_policy(
        &mut values,
        provider.preferences.get("timeout_policy"),
        &context,
        stream,
    );
    Timeouts {
        first_byte: values
            .get("first_byte")
            .and_then(Value::as_f64)
            .or(Some(base)),
        idle: values.get("idle").and_then(Value::as_f64),
        total: values.get("total").and_then(Value::as_f64),
    }
}

fn model_timeout(
    provider: &Provider,
    global: &Map<String, Value>,
    request_model: &str,
    original_model: &str,
) -> f64 {
    for preferences in [&provider.preferences, global] {
        if let Some(timeout) = preferences.get("model_timeout") {
            if let Some(value) = timeout.as_f64() {
                return value;
            }
            if let Some(values) = timeout.as_object() {
                for key in [request_model, original_model, "default"] {
                    if let Some(value) = values.get(key).and_then(Value::as_f64) {
                        return value;
                    }
                }
            }
        }
    }
    100.0
}

fn merge_timeout_policy(
    target: &mut Map<String, Value>,
    policy: Option<&Value>,
    context: &HashMap<&str, &str>,
    stream: bool,
) {
    let Some(policy) = policy.and_then(Value::as_object) else {
        return;
    };
    if let Some(default) = policy.get("default").and_then(Value::as_object) {
        target.extend(default.clone());
    }
    let mut best: Option<(&Map<String, Value>, usize)> = None;
    for rule in policy
        .get("rules")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let Some(condition) = rule.get("match").and_then(Value::as_object) else {
            continue;
        };
        if timeout_rule_matches(condition, context, stream)
            && best.is_none_or(|(_, score)| condition.len() > score)
        {
            if let Some(timeout) = rule.get("timeout").and_then(Value::as_object) {
                best = Some((timeout, condition.len()));
            }
        }
    }
    if let Some((timeout, _)) = best {
        target.extend(timeout.clone());
    }
}

fn timeout_rule_matches(
    condition: &Map<String, Value>,
    context: &HashMap<&str, &str>,
    stream: bool,
) -> bool {
    condition.iter().all(|(key, expected)| {
        if key == "stream" {
            return expected.as_bool() == Some(stream);
        }
        let actual = context.get(key.as_str()).copied().unwrap_or_default();
        timeout_value_matches(expected, actual)
    })
}

fn timeout_value_matches(expected: &Value, actual: &str) -> bool {
    if let Some(values) = expected.as_array() {
        return values
            .iter()
            .any(|value| timeout_value_matches(value, actual));
    }
    let Some(expected) = expected.as_str() else {
        return false;
    };
    expected == "*"
        || expected.eq_ignore_ascii_case(actual)
        || expected.strip_suffix('*').is_some_and(|prefix| {
            actual
                .to_ascii_lowercase()
                .starts_with(&prefix.to_ascii_lowercase())
        })
}

fn normalize_upstream_url(base_url: &str, engine: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if engine != "codex" || base.ends_with("/v1/responses") || base.ends_with("/responses") {
        base.to_owned()
    } else {
        format!("{base}/responses")
    }
}

fn upstream_host(base_url: &str) -> String {
    Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default()
}

fn outcome_status(outcome: &Value, fallback: u16) -> u16 {
    outcome_status_from(
        outcome,
        "status_code",
        outcome_status_from(outcome, "upstream_status_code", fallback),
    )
}

fn outcome_status_from(outcome: &Value, key: &str, fallback: u16) -> u16 {
    outcome
        .get(key)
        .and_then(Value::as_u64)
        .filter(|status| *status > 0)
        .unwrap_or(u64::from(fallback))
        .min(u64::from(u16::MAX)) as u16
}

fn failure_origin(outcome: &Value) -> &'static str {
    match outcome.get("kind").and_then(Value::as_str) {
        Some("http_error") => "upstream_http",
        Some("transport_error") => "ember_transport",
        Some("protocol_error") => "ember_protocol",
        Some("semantic_failure" | "semantic_error") => "upstream_semantic",
        Some("downstream_disconnected") => "downstream_client",
        Some("completed" | "incomplete") => "upstream_success",
        _ => "ember_native",
    }
}

fn native_rejection_origin(reason: &str) -> &'static str {
    match reason {
        "native_global_rate_limit" => "native_global_rate_limit",
        "native_client_rate_limit" => "native_client_rate_limit",
        "no_matching_provider" => "native_route_selection",
        "invalid_api_key" => "native_authentication",
        _ => "native_request_validation",
    }
}

fn status_class(status: u16) -> &'static str {
    match status {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "unknown",
    }
}

fn event_severity(status: u16, outcome: &str) -> &'static str {
    if status >= 500 {
        "error"
    } else if status >= 400 || matches!(outcome, "skipped" | "failed" | "completed_with_error") {
        "warning"
    } else {
        "info"
    }
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn terminal_error_sha256(success: bool, detail: &str) -> Option<String> {
    (!success && !detail.is_empty()).then(|| sha256_hex(detail))
}

fn provider_api_keys(value: &Value) -> Vec<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => vec![value.trim().to_owned()],
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn endpoint_values(value: &Value) -> Vec<String> {
    match value {
        Value::String(value) => vec![value.clone()],
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn preference_string(preferences: &Map<String, Value>, key: &str) -> Option<String> {
    preferences
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn preference_f64(preferences: &Map<String, Value>, key: &str) -> Option<f64> {
    preferences.get(key).and_then(Value::as_f64)
}

fn remap_provider_status(status: u16, detail: &str) -> u16 {
    if [
        "string_above_max_length",
        "must be less than max_seq_len",
        "please reduce the length of the messages or completion",
        "request contains text fields that are too large.",
        "please reduce the length of either one, or use the",
        "exceeds the maximum number of tokens allowed",
    ]
    .iter()
    .any(|marker| detail.to_ascii_lowercase().contains(marker))
    {
        return 413;
    }
    if detail.contains("'reason': 'API_KEY_INVALID'")
        || detail.contains("API key not valid")
        || detail.contains("API key expired")
    {
        return 401;
    }
    if detail.contains("User location is not supported for the API use.") {
        return 403;
    }
    if detail.contains("<center><h1>400 Bad Request</h1></center>")
        || detail.contains("Provider API error: bad response status code 400")
        || status == 400 && is_model_pricing_unconfigured(detail)
    {
        return 502;
    }
    if detail.contains(
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
    ) {
        return 403;
    }
    if detail.contains("<head><title>413 Request Entity Too Large</title></head>") {
        return 429;
    }
    status
}

fn is_model_pricing_unconfigured(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    [
        "model_not_priced",
        "model_price_not_configured",
        "model_pricing_not_configured",
        "model_price_unconfigured",
        "model_pricing_missing",
        "has not been priced by the administrator",
        "has not been priced by administrator",
        "price has not been configured by the administrator",
        "pricing has not been configured by the administrator",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || lower
            .split_whitespace()
            .collect::<String>()
            .contains("价格尚未由管理员配置")
}

fn is_missing_persisted_item_error(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    lower.contains("invalid_request_error")
        && lower.contains("item with id")
        && lower.contains("not found")
        && lower.contains("items are not persisted when")
        && lower.contains("store")
}

fn provider_model_circuit_threshold() -> usize {
    std::env::var("PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(3)
}

fn provider_model_circuit_window() -> Duration {
    environment_duration("PROVIDER_MODEL_CIRCUIT_WINDOW_SECONDS", 120.0)
}

fn provider_model_circuit_open_period() -> Duration {
    environment_duration("PROVIDER_MODEL_CIRCUIT_OPEN_SECONDS", 300.0)
}

fn environment_duration(name: &str, default_seconds: f64) -> Duration {
    Duration::from_secs_f64(
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(default_seconds),
    )
}

fn retry_after_seconds(detail: &str) -> Option<f64> {
    let lower = detail.to_ascii_lowercase();
    let tail = lower.split_once("try again in")?.1.trim_start();
    let number_end = tail
        .find(|character: char| !(character.is_ascii_digit() || character == '.'))
        .unwrap_or(tail.len());
    if number_end == 0 {
        return None;
    }
    let number = tail[..number_end].parse::<f64>().ok()?;
    let unit = tail[number_end..].trim_start();
    Some(
        if unit.starts_with("ms") || unit.starts_with("millisecond") {
            (number / 1000.0).ceil()
        } else if unit.starts_with('m') {
            (number * 60.0).ceil()
        } else {
            number.ceil()
        },
    )
}

fn parse_rate_limits(value: Option<&Value>, model: Option<&str>) -> Option<Vec<(usize, u64)>> {
    let raw = match value {
        None | Some(Value::Null) => "999999/min",
        Some(Value::String(value)) => value.trim(),
        Some(Value::Object(values)) => {
            let selected = if let Some(exact) = model.and_then(|model| values.get(model)) {
                Some(exact)
            } else {
                let matches = model
                    .into_iter()
                    .flat_map(|model| {
                        values.iter().filter(move |(configured, _)| {
                            configured.as_str() != "default" && model.contains(configured.as_str())
                        })
                    })
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>();
                if matches.len() > 1 {
                    return None;
                }
                matches.first().copied().or_else(|| values.get("default"))
            };
            match selected {
                Some(Value::String(value)) => value.trim(),
                Some(_) => return None,
                None => "999999/min",
            }
        }
        _ => return None,
    };
    let mut rules = Vec::new();
    for configured in raw.split(',') {
        let (count, period) = configured.trim().split_once('/')?;
        let count = count.trim().parse::<usize>().ok()?;
        let seconds = match period.trim().to_ascii_lowercase().as_str() {
            "s" | "sec" | "second" => 1,
            "m" | "min" | "minute" => 60,
            "h" | "hr" | "hour" => 3_600,
            "d" | "day" => 86_400,
            "mo" | "month" => 2_592_000,
            "y" | "year" => 31_536_000,
            "tpr" => continue,
            _ => return None,
        };
        rules.push((count, seconds));
    }
    Some(rules)
}

fn pydantic_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Number(value) if value.as_i64() == Some(1) => Some(true),
        Value::Number(value) if value.as_i64() == Some(0) => Some(false),
        Value::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "on" | "yes" | "y" => Some(true),
            "0" | "false" | "f" | "off" | "no" | "n" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(token) = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
    {
        if !token.is_empty() {
            return Some(token.to_owned());
        }
    }
    let authorization = headers.get("authorization")?.to_str().ok()?;
    authorization
        .split_once(' ')
        .map(|(_, token)| token.trim())
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

fn is_identity_json_request(headers: &HeaderMap) -> bool {
    let content_encoding = headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("identity");
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json");
    content_encoding.eq_ignore_ascii_case("identity")
        && content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
}

fn request_id(headers: &HeaderMap) -> String {
    for name in ["x-request-id", "x-caller-request-id"] {
        if let Some(value) = headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return value.chars().take(128).collect();
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    format!("{now:016x}{sequence:016x}")
}

fn native_attempt_id(request_id: &str, attempt: usize) -> String {
    format!("{request_id}-r{}", attempt + 1)
}

fn json_response(status: StatusCode, payload: Value) -> Response<Body> {
    let mut response = Response::new(Body::from(payload.to_string()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    response.headers_mut().insert(
        "x-uni-api-data-plane",
        HeaderValue::from_static("rust-native-v2"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::ResourceGovernor;

    fn provider() -> Provider {
        Provider {
            name: Arc::from("fugue-codex"),
            base_url: Arc::from("https://example.com/v1/responses"),
            engine: Arc::from("codex"),
            api_keys: Arc::new(vec!["provider-key".into()]),
            models: Arc::new(HashMap::from([(
                "gpt-public".into(),
                "gpt-upstream".into(),
            )])),
            preferences: Arc::new(Map::from_iter([(
                "post_body_parameter_overrides".into(),
                json!({"store": false, "__remove__": ["temperature"]}),
            )])),
            excluded_endpoints: Arc::new(Vec::new()),
            cursor: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn native_route_for_test(provider: Arc<Provider>, max_attempts: usize) -> NativeRoute {
        let store = NativeConfigStore::new();
        let snapshot = Arc::new(Snapshot {
            revision: Arc::from("0".repeat(64)),
            database_disabled: true,
            preferences: Arc::new(Map::new()),
            api_keys: Arc::new(HashMap::new()),
            providers: Arc::new(vec![provider.clone()]),
            providers_by_name: Arc::new(HashMap::from([(
                provider.name.to_string(),
                provider.clone(),
            )])),
        });
        let api_key = Arc::new(ApiKey {
            token: Arc::from("client-key"),
            model_rules: Arc::new(Vec::new()),
            role: Arc::from("user"),
            preferences: Arc::new(Map::from_iter([("AUTO_RETRY".into(), json!(true))])),
            native_paid_state_safe: true,
            native_supported: true,
        });
        let (_, memory_reservation) = ResourceGovernor::unconstrained_for_test()
            .reserve_memory_capacity(0)
            .await
            .unwrap();
        NativeRoute {
            store,
            snapshot,
            api_key,
            providers: vec![provider.clone(), provider],
            base_payload: json!({"model":"gpt-public","input":"hello","stream":true}),
            request_headers: HeaderMap::new(),
            request_model: "gpt-public".into(),
            stream: true,
            request_id: "request-test".into(),
            request_body_bytes: 64,
            cursor: 0,
            max_attempts,
            last_provider: None,
            last_provider_key: None,
            last_original_model: None,
            last_attempt: None,
            last_status: 502,
            last_detail: String::new(),
            has_attempt_failure: false,
            last_failure_origin: String::new(),
            routing_attempts: 0,
            routing_skips: 0,
            upstream_attempts: 0,
            upstream_duration_ms: 0,
            routing_ledger: Vec::new(),
            upstream_ledger: Vec::new(),
            started_at: tokio::time::Instant::now(),
            final_emitted: false,
            _memory_reservation: memory_reservation,
        }
    }

    #[test]
    fn payload_compiler_matches_codex_contract_without_double_json_envelope() {
        let provider = provider();
        let mut payload = json!({
            "model": "gpt-public",
            "input": [{"type":"reasoning","id":"rs_1","cache_control":{}}],
            "stream": true,
            "temperature": 1,
            "max_output_tokens": 42,
            "previous_response_id": "resp_1"
        });
        compile_payload(
            &mut payload,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
        )
        .unwrap();
        assert_eq!(payload["model"], "gpt-upstream");
        assert_eq!(payload["store"], false);
        assert_eq!(payload["instructions"], "");
        assert!(payload.get("temperature").is_none());
        assert!(payload.get("max_output_tokens").is_none());
        assert!(payload.get("previous_response_id").is_none());
        assert!(payload["input"][0].get("id").is_none());
        assert!(payload["input"][0].get("cache_control").is_none());
    }

    #[test]
    fn structured_removal_matches_python_semantics() {
        let mut root = json!({
            "tools": [
                {"type":"function","name":"ok"},
                {"type":"image_generation"}
            ]
        })
        .as_object()
        .unwrap()
        .clone();
        apply_removals(
            &mut root,
            &json!([{
                "path":"tools",
                "where":{"type":"image_generation"},
                "drop_empty":true
            }]),
        );
        assert_eq!(root["tools"], json!([{"type":"function","name":"ok"}]));
    }

    #[test]
    fn byte_limits_and_rate_limits_have_no_hard_request_ceiling() {
        assert_eq!(parse_byte_limit(&json!("20MiB")), Some(20 * 1024 * 1024));
        assert_eq!(parse_byte_limit(&Value::Null), None);
        assert_eq!(
            parse_rate_limits(Some(&json!("300/min, 1000/hour")), None),
            Some(vec![(300, 60), (1000, 3_600)])
        );
        assert_eq!(
            parse_rate_limits(
                Some(&json!({"gpt-5.6": "5/sec", "default": "300/min"})),
                Some("gpt-5.6-sol"),
            ),
            Some(vec![(5, 1)])
        );
    }

    #[test]
    fn provider_error_policy_matches_request_scoped_and_gateway_failures() {
        assert_eq!(
            remap_provider_status(
                400,
                r#"{"error":{"code":"model_not_priced","message":"missing"}}"#,
            ),
            502
        );
        assert!(is_missing_persisted_item_error(
            r#"{"error":{"type":"invalid_request_error","message":"Item with id 'rs_1' not found. Items are not persisted when store is false."}}"#,
        ));
        assert_eq!(
            retry_after_seconds("Rate limit reached. Please try again in 2500ms."),
            Some(3.0)
        );
        assert_eq!(terminal_error_sha256(true, "earlier attempt failed"), None);
        assert!(terminal_error_sha256(false, "terminal failure").is_some());
    }

    #[tokio::test]
    async fn local_cooldown_skip_does_not_overwrite_last_upstream_failure() {
        let mut provider = provider();
        provider.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
        let mut route = native_route_for_test(Arc::new(provider), 2).await;

        assert!(route.next_plan().await.unwrap().is_some());
        assert!(
            route
                .record_failure(&json!({
                    "kind": "http_error",
                    "status_code": 503,
                    "body": "no available token",
                }))
                .await
        );
        assert!(route.next_plan().await.unwrap().is_none());

        assert_eq!(route.last_status(), 503);
        assert_eq!(route.last_detail(), "no available token");
        assert_eq!(route.routing_attempts, 2);
        assert_eq!(route.routing_skips, 1);
        assert_eq!(route.upstream_attempts, 1);
        let final_message = route.final_message();
        assert_eq!(final_message["status_code"], 503);
        assert!(route.final_emitted);
    }
}
