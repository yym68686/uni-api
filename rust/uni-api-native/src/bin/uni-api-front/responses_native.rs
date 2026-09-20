use std::collections::BTreeSet;
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

use crate::codex_oauth::CodexOAuthManager;
use crate::hedging::{parse_hedging, HedgingConfig};
use crate::persistence::{ChannelStat, Persistence, RequestStat};
use crate::request_spool::{SpoolObservation, StoredBody};
use crate::resources::MemoryReservation;
use crate::responses::{Plan, UNLIMITED_SSE_EVENT_BYTES};
use crate::responses_item_ids::normalize_response_root;

const SNAPSHOT_SCHEMA_VERSION: u64 = 1;
const DEFAULT_MAX_PRECOMMIT_ITEMS: usize = 128;
const DEFAULT_MAX_PRECOMMIT_BYTES: usize = 8 * 1024 * 1024 + 128 * 266;
pub(crate) const CODEX_USER_AGENT: &str =
    "codex_cli_rs/0.153.2 (Debian 13.0.0; x86_64) WindowsTerminal";

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static SCHEDULING_NONCE: AtomicU64 = AtomicU64::new(1);

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
    pub(crate) channel_controls: Arc<RwLock<crate::channel_controls::Controls>>,
    path: Arc<PathBuf>,
    pub(crate) current: Arc<RwLock<Option<Arc<Snapshot>>>>,
    snapshot_stamp: Arc<Mutex<Option<SnapshotStamp>>>,
    provider_cursors: Arc<Mutex<HashMap<String, Arc<AtomicUsize>>>>,
    key_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    channel_cooldowns: Arc<Mutex<HashMap<(String, String), tokio::time::Instant>>>,
    route_failures: Arc<Mutex<RouteFailureHistory>>,
    client_windows: Arc<Mutex<RateWindows>>,
    provider_windows: Arc<Mutex<RateWindows>>,
    routing_cursors: Arc<Mutex<HashMap<(String, String), usize>>>,
}

#[derive(Debug, Deserialize)]
struct RawSnapshot {
    schema_version: u64,
    revision: String,
    #[serde(default)]
    preferences: Map<String, Value>,
    #[serde(default)]
    api_keys: Vec<RawApiKey>,
    #[serde(default)]
    providers: Vec<RawProvider>,
    #[serde(default)]
    api_config: Value,
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
}

#[derive(Debug, Deserialize)]
pub(crate) struct RawProvider {
    name: String,
    base_url: String,
    engine: Option<String>,
    api: Value,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    private_key: Option<String>,
    #[serde(default)]
    client_email: Option<String>,
    #[serde(default)]
    aws_access_key: Option<String>,
    #[serde(default)]
    aws_secret_key: Option<String>,
    #[serde(default)]
    aws_session_token: Option<String>,
    #[serde(default)]
    cf_account_id: Option<String>,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    models: HashMap<String, String>,
    #[serde(default)]
    preferences: Map<String, Value>,
    #[serde(default)]
    exclude_endpoints: Value,
    #[serde(default)]
    only_request_types: Value,
    #[serde(default)]
    exclude_request_types: Value,
    #[serde(default)]
    exclude_request_rules: Value,
}

#[derive(Clone)]
pub(crate) struct Snapshot {
    pub(crate) revision: Arc<str>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) api_keys: Arc<HashMap<String, Arc<ApiKey>>>,
    pub(crate) api_key_order: Arc<Vec<String>>,
    pub(crate) providers: Arc<Vec<Arc<Provider>>>,
    pub(crate) providers_by_name: Arc<HashMap<String, Arc<Provider>>>,
    pub(crate) api_config: Arc<Value>,
}

#[derive(Clone)]
pub(crate) struct ApiKey {
    pub(crate) token: Arc<str>,
    pub(crate) model_rules: Arc<Vec<String>>,
    pub(crate) role: Arc<str>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) weights: Arc<Map<String, Value>>,
    pub(crate) native_supported: bool,
}

#[derive(Clone)]
pub(crate) struct Provider {
    pub(crate) name: Arc<str>,
    pub(crate) base_url: Arc<str>,
    pub(crate) engine: Arc<str>,
    pub(crate) api_keys: Arc<Vec<String>>,
    pub(crate) project_id: Option<Arc<str>>,
    pub(crate) private_key: Option<Arc<str>>,
    pub(crate) client_email: Option<Arc<str>>,
    pub(crate) aws_access_key: Option<Arc<str>>,
    pub(crate) aws_secret_key: Option<Arc<str>>,
    pub(crate) aws_session_token: Option<Arc<str>>,
    pub(crate) cf_account_id: Option<Arc<str>>,
    pub(crate) region: Arc<str>,
    pub(crate) models: Arc<HashMap<String, String>>,
    pub(crate) preferences: Arc<Map<String, Value>>,
    pub(crate) excluded_endpoints: Arc<Vec<String>>,
    pub(crate) only_request_types: Arc<Vec<String>>,
    pub(crate) excluded_request_types: Arc<Vec<String>>,
    pub(crate) excluded_request_rules: Arc<Vec<Value>>,
    pub(crate) cursor: Arc<AtomicUsize>,
}

pub(crate) fn runtime_provider(item: RawProvider, cursor: Arc<AtomicUsize>) -> Arc<Provider> {
    let name = item.name.trim().to_owned();
    Arc::new(Provider {
        name: name.clone().into(),
        base_url: item.base_url.trim().to_owned().into(),
        engine: item.engine.unwrap_or_else(|| "gpt".into()).into(),
        api_keys: Arc::new(provider_api_keys(&item.api)),
        project_id: item
            .project_id
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        private_key: item
            .private_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        client_email: item
            .client_email
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_access_key: item
            .aws_access_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_secret_key: item
            .aws_secret_key
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        aws_session_token: item
            .aws_session_token
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        cf_account_id: item
            .cf_account_id
            .filter(|value| !value.trim().is_empty())
            .map(Into::into),
        region: item
            .region
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "global".into())
            .into(),
        models: Arc::new(item.models),
        preferences: Arc::new(item.preferences),
        excluded_endpoints: Arc::new(endpoint_values(&item.exclude_endpoints)),
        only_request_types: Arc::new(request_type_values(&item.only_request_types)),
        excluded_request_types: Arc::new(request_type_values(&item.exclude_request_types)),
        excluded_request_rules: Arc::new(request_rule_values(&item.exclude_request_rules)),
        cursor,
    })
}

pub(crate) struct FailedRoute<'a> {
    pub(crate) provider: &'a Provider,
    pub(crate) key: &'a str,
    pub(crate) original_model: &'a str,
    pub(crate) has_alternative: bool,
    pub(crate) status: u16,
    pub(crate) detail: &'a str,
    pub(crate) provider_model_unavailable: bool,
    pub(crate) force_quota_cooldown: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProviderFailurePolicy {
    pub(crate) status: u16,
    pub(crate) retryable: bool,
    pub(crate) request_scoped: bool,
    pub(crate) provider_model_unavailable: bool,
    pub(crate) force_quota_cooldown: bool,
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

pub(crate) enum ProviderKeySelection {
    Selected(String),
    NoProviderKey,
    ChannelCooling,
    AllKeysCooling,
}

pub(crate) struct ResolvedRoute {
    pub(crate) providers: Vec<Arc<Provider>>,
    pub(crate) hedging: HedgingConfig,
}

pub(crate) struct RouteResolutionError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

#[derive(Clone)]
pub(crate) struct AuthContext {
    pub(crate) api_key: Arc<ApiKey>,
    pub(crate) api_key_count: usize,
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
    role: &'a str,
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
    codex_oauth: CodexOAuthManager,
    persistence: Persistence,
    snapshot: Arc<Snapshot>,
    api_key: Arc<ApiKey>,
    providers: Vec<Arc<Provider>>,
    base_payload: Value,
    request_headers: HeaderMap,
    request_model: String,
    endpoint: String,
    request_type: Option<String>,
    wants_compact: bool,
    stream: bool,
    request_id: String,
    request_body_bytes: u64,
    cursor: usize,
    max_attempts: usize,
    hedging: HedgingConfig,
    attempt_contexts: HashMap<String, NativeAttemptObservation>,
    hedge_trigger_count: usize,
    hedge_cancelled_attempt_count: usize,
    last_provider: Option<Arc<Provider>>,
    last_provider_key: Option<String>,
    last_original_model: Option<String>,
    last_attempt: Option<NativeAttemptObservation>,
    last_status: u16,
    last_detail: String,
    last_provider_model_unavailable: bool,
    has_attempt_failure: bool,
    last_failure_origin: String,
    routing_attempts: usize,
    routing_skips: usize,
    upstream_attempts: usize,
    upstream_duration_ms: u64,
    routing_ledger: Vec<Value>,
    upstream_ledger: Vec<Value>,
    arrival: Option<crate::request_timing::RequestArrival>,
    started_at: tokio::time::Instant,
    final_emitted: bool,
    _memory_reservation: MemoryReservation,
}

impl NativeConfigStore {
    pub fn new() -> Self {
        let path = std::env::var("RUST_RESPONSES_CONFIG_SNAPSHOT_PATH")
            .unwrap_or_else(|_| "/tmp/uni-api-rust-responses-config-v1.json".into());
        Self {
            channel_controls: Arc::new(RwLock::new(crate::channel_controls::Controls::default())),
            path: Arc::new(PathBuf::from(path)),
            current: Arc::new(RwLock::new(None)),
            snapshot_stamp: Arc::new(Mutex::new(None)),
            provider_cursors: Arc::new(Mutex::new(HashMap::new())),
            key_cooldowns: Arc::new(Mutex::new(HashMap::new())),
            channel_cooldowns: Arc::new(Mutex::new(HashMap::new())),
            route_failures: Arc::new(Mutex::new(HashMap::new())),
            client_windows: Arc::new(Mutex::new(HashMap::new())),
            provider_windows: Arc::new(Mutex::new(HashMap::new())),
            routing_cursors: Arc::new(Mutex::new(HashMap::new())),
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
            let provider = runtime_provider(item, cursor);
            providers_by_name.insert(name, provider.clone());
            providers.push(provider);
        }
        drop(cursors);

        let api_key_order = raw
            .api_keys
            .iter()
            .map(|item| item.token.trim().to_owned())
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let api_keys = raw
            .api_keys
            .into_iter()
            .filter_map(|item| {
                let token = item.token.trim().to_owned();
                if token.is_empty() {
                    return None;
                }
                let native_supported = item.model_rules.iter().all(Value::is_string);
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
                        weights: Arc::new(item.weights),
                        native_supported,
                    }),
                ))
            })
            .collect::<HashMap<_, _>>();
        let snapshot = Arc::new(Snapshot {
            revision: raw.revision.into(),
            preferences: Arc::new(raw.preferences),
            api_keys: Arc::new(api_keys),
            api_key_order: Arc::new(api_key_order),
            providers: Arc::new(providers),
            providers_by_name: Arc::new(providers_by_name),
            api_config: Arc::new(raw.api_config),
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

    pub(crate) async fn base_snapshot(&self) -> Option<Arc<Snapshot>> {
        self.current.read().await.clone()
    }

    pub(crate) async fn snapshot(&self) -> Option<Arc<Snapshot>> {
        let base = self.current.read().await.clone()?;
        Some(self.channel_controls.read().await.overlay(base))
    }

    pub async fn is_ready(&self) -> bool {
        self.current.read().await.is_some()
    }

    pub async fn models_for_headers(&self, headers: &HeaderMap) -> Result<Vec<String>, u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let api_key = snapshot.api_keys.get(&token).ok_or(403u16)?;
        let mut models = BTreeSet::new();
        for rule in api_key.model_rules.iter() {
            if rule == "all" {
                for provider in snapshot
                    .providers
                    .iter()
                    .filter(|p| crate::channel_controls::temporary_allowed(p, api_key))
                {
                    models.extend(provider.models.keys().cloned());
                }
                continue;
            }
            if rule.starts_with('<') && rule.ends_with('>') {
                let model = rule[1..rule.len() - 1].to_owned();
                if snapshot
                    .providers
                    .iter()
                    .filter(|p| crate::channel_controls::temporary_allowed(p, api_key))
                    .any(|provider| provider.models.contains_key(&model))
                {
                    models.insert(model);
                }
                continue;
            }
            if let Some((provider_name, model_rule)) = rule.split_once('/') {
                if let Some(provider) = snapshot
                    .providers_by_name
                    .get(provider_name)
                    .filter(|p| crate::channel_controls::temporary_allowed(p, api_key))
                {
                    if model_rule == "*" {
                        models.extend(provider.models.keys().cloned());
                    } else if provider.models.contains_key(model_rule) {
                        models.insert(model_rule.to_owned());
                    }
                }
                continue;
            }
            if snapshot
                .providers
                .iter()
                .filter(|p| crate::channel_controls::temporary_allowed(p, api_key))
                .any(|provider| provider.models.contains_key(rule))
            {
                models.insert(rule.clone());
            }
        }
        Ok(models.into_iter().collect())
    }

    pub(crate) async fn authorize(
        &self,
        headers: &HeaderMap,
    ) -> Result<AuthContext, RouteResolutionError> {
        let token = extract_api_key(headers).ok_or_else(|| RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Invalid or missing API Key".into(),
        })?;
        let snapshot = self.snapshot().await.ok_or_else(|| RouteResolutionError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Runtime configuration is not ready".into(),
        })?;
        let api_key =
            snapshot
                .api_keys
                .get(&token)
                .cloned()
                .ok_or_else(|| RouteResolutionError {
                    status: StatusCode::FORBIDDEN,
                    message: "Invalid or missing API Key".into(),
                })?;
        Ok(AuthContext {
            api_key,
            api_key_count: snapshot.api_keys.len(),
        })
    }

    pub(crate) async fn moderation_enabled(
        &self,
        headers: &HeaderMap,
    ) -> Result<bool, RouteResolutionError> {
        Ok(self
            .authorize(headers)
            .await?
            .api_key
            .preferences
            .get("ENABLE_MODERATION")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub async fn runtime_counts(&self) -> (usize, usize, usize) {
        let Some(snapshot) = self.snapshot().await else {
            return (0, 0, 0);
        };
        let model_count = snapshot
            .providers
            .iter()
            .flat_map(|provider| provider.models.keys())
            .collect::<BTreeSet<_>>()
            .len();
        (
            snapshot.api_keys.len(),
            snapshot.providers.len(),
            model_count,
        )
    }

    pub(crate) async fn channel_catalog(
        &self,
        headers: &HeaderMap,
        endpoint: &str,
        stream: bool,
        selected_key_id: Option<&str>,
    ) -> Result<(Vec<Value>, String, String), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        let selected_id = selected_key_id.unwrap_or_default();
        let entries = crate::channel_catalog::entries(&snapshot, caller, Some(selected_id))?;
        let cooldowns = self.channel_cooldowns.lock().await.clone();
        let now = tokio::time::Instant::now();
        let controls = self.channel_controls.read().await.routing_rules();
        let mut rows: Vec<Value> = entries.into_iter().filter_map(|(provider, model)| {
            if provider.excluded_endpoints.iter().any(|v| v.trim_end_matches('/').eq_ignore_ascii_case(endpoint)) {
                return None;
            }
            let upstream = provider.models.get(&model)?;
            let route_cooling = cooldowns.get(&(provider.name.to_string(), upstream.clone())).is_some_and(|until| *until > now);
            let (eligible, reason) = if controls.disabled(selected_id,&model,&provider.name) {
                (false,"temporarily_disabled")
            } else if provider.api_keys.is_empty() && provider.client_email.is_none() {
                (false, "no_provider_key")
            } else if route_cooling { (false, "channel_cooldown") } else { (true, "eligible") };
            Some(json!({"provider":provider.name.as_ref(),"model":model,"upstream_model":upstream,"engine":provider.engine.as_ref(),"endpoint":endpoint,"stream":stream,"eligible":eligible,"reason":reason}))
        }).collect();
        rows.sort_by_key(|row| {
            controls
                .order(selected_id, row["model"].as_str().unwrap_or_default())
                .and_then(|order| {
                    order
                        .iter()
                        .position(|p| Some(p.as_str()) == row["provider"].as_str())
                })
                .unwrap_or(usize::MAX)
        });
        Ok((rows, snapshot.revision.to_string(), selected_id.to_owned()))
    }

    pub(crate) async fn authorize_catalog(&self, headers: &HeaderMap) -> Result<(), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::channel_catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        Ok(())
    }

    pub(crate) async fn balance_provider(
        &self,
        headers: &HeaderMap,
        name: &str,
    ) -> Result<(Arc<Provider>, Option<String>), u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::channel_catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        let provider = snapshot
            .providers_by_name
            .get(name)
            .cloned()
            .ok_or(404u16)?;
        let proxy = preference_string(&provider.preferences, "proxy")
            .or_else(|| preference_string(&snapshot.preferences, "proxy"));
        Ok((provider, proxy))
    }

    pub(crate) async fn api_key_catalog(&self, headers: &HeaderMap) -> Result<Value, u16> {
        let token = extract_api_key(headers).ok_or(403u16)?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        let caller = snapshot.api_keys.get(&token).ok_or(403u16)?;
        if !crate::channel_catalog::can_inspect_all(&snapshot, caller) {
            return Err(403);
        }
        Ok(
            json!({"data":crate::channel_catalog::keys(&snapshot, caller),"snapshot_revision":snapshot.revision.as_ref(),"can_inspect_all":true}),
        )
    }

    pub(crate) async fn prices_for_model(&self, model: &str) -> (f64, f64) {
        self.snapshot()
            .await
            .map(|snapshot| model_prices(&snapshot.preferences, model))
            .unwrap_or((0.3, 1.0))
    }

    pub(crate) async fn keepalive_interval(
        &self,
        provider: &Provider,
        request_model: &str,
        original_model: &str,
    ) -> Option<Duration> {
        let snapshot = self.snapshot().await?;
        let interval = model_preference(
            provider,
            &snapshot.preferences,
            request_model,
            original_model,
            "keepalive_interval",
        )
        .unwrap_or(99999.0);
        let timeout = model_timeout(
            provider,
            &snapshot.preferences,
            request_model,
            original_model,
        );
        (interval.is_finite() && interval > 0.0 && interval <= timeout)
            .then(|| Duration::from_secs_f64(interval))
    }

    pub(crate) async fn auto_retry_enabled(&self, headers: &HeaderMap) -> bool {
        self.auto_retry_budget(headers).await > 0
    }

    pub(crate) async fn auto_retry_budget(&self, headers: &HeaderMap) -> usize {
        let Some(snapshot) = self.snapshot().await else {
            return 1;
        };
        let Some(token) = extract_api_key(headers) else {
            return 1;
        };
        let Some(value) = snapshot
            .api_keys
            .get(&token)
            .and_then(|key| key.preferences.get("AUTO_RETRY"))
        else {
            return 1;
        };
        match value {
            Value::Bool(enabled) => usize::from(*enabled),
            Value::Number(number) => number.as_u64().unwrap_or(0).min(100) as usize,
            Value::String(text) => text
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| usize::from(pydantic_bool(value).unwrap_or(true)))
                .min(100),
            _ => 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn generic_timeouts(
        &self,
        headers: &HeaderMap,
        provider: &Provider,
        request_model: &str,
        original_model: &str,
        engine: &str,
        stream: bool,
        endpoint: &str,
        method: &str,
    ) -> Timeouts {
        let Some(snapshot) = self.snapshot().await else {
            return Timeouts::default();
        };
        let role = extract_api_key(headers)
            .and_then(|token| snapshot.api_keys.get(&token).cloned())
            .map(|key| key.role.to_string())
            .unwrap_or_default();
        resolve_timeouts(
            &snapshot,
            provider,
            request_model,
            original_model,
            engine,
            stream,
            None,
            &role,
            endpoint,
            method,
        )
    }

    pub async fn api_config(&self) -> Option<Value> {
        self.snapshot()
            .await
            .map(|snapshot| (*snapshot.api_config).clone())
    }

    pub async fn paid_api_key_states(&self, persistence: &Persistence) -> Value {
        let Some(snapshot) = self.snapshot().await else {
            return json!({});
        };
        let mut states = Map::new();
        for item in snapshot
            .api_config
            .get("api_keys")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(token) = item.get("api").and_then(Value::as_str) else {
                continue;
            };
            let Some(credits) = item.pointer("/preferences/credits").and_then(Value::as_f64) else {
                continue;
            };
            let created_at = item
                .pointer("/preferences/created_at")
                .and_then(Value::as_str)
                .and_then(parse_config_datetime)
                .unwrap_or_else(|| unix_seconds_i64().saturating_sub(30 * 86_400));
            let total_cost = persistence
                .total_cost(token, created_at)
                .await
                .unwrap_or(0.0);
            let all_tokens_info = persistence
                .token_usage(Some(token), None, Some(created_at), None)
                .await
                .ok()
                .and_then(|value| value.get("usage").cloned())
                .unwrap_or_else(|| json!([]));
            states.insert(
                token.to_owned(),
                json!({
                    "credits": credits,
                    "created_at": created_at,
                    "all_tokens_info": all_tokens_info,
                    "total_cost": total_cost,
                    "enabled": credits == -1.0 || total_cost <= credits,
                }),
            );
        }
        Value::Object(states)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn resolve_route(
        &self,
        persistence: &Persistence,
        headers: &HeaderMap,
        request_model: &str,
        endpoint: &str,
        request_body_bytes: u64,
        request_type: Option<&str>,
        admit_rate: bool,
    ) -> Result<ResolvedRoute, RouteResolutionError> {
        let snapshot = self.snapshot().await.ok_or_else(|| RouteResolutionError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Runtime configuration is not ready".into(),
        })?;
        let token = extract_api_key(headers).ok_or_else(|| RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Invalid or missing API Key".into(),
        })?;
        let api_key =
            snapshot
                .api_keys
                .get(&token)
                .cloned()
                .ok_or_else(|| RouteResolutionError {
                    status: StatusCode::FORBIDDEN,
                    message: "Invalid or missing API Key".into(),
                })?;
        if !api_key.native_supported {
            return Err(RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "API key contains an unsupported model rule".into(),
            });
        }
        self.ensure_paid_balance(persistence, &api_key).await?;
        let global_rules = parse_rate_limits(snapshot.preferences.get("rate_limit"), None)
            .ok_or_else(|| RouteResolutionError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "Invalid global rate-limit configuration".into(),
            })?;
        let client_rules =
            parse_rate_limits(api_key.preferences.get("rate_limit"), Some(request_model))
                .ok_or_else(|| RouteResolutionError {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    message: "Invalid client rate-limit configuration".into(),
                })?;
        let estimated_tokens = (request_body_bytes / 4).max(1) as usize;
        if tpr_exceeded(&client_rules, estimated_tokens) {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Tokens per request limit exceeded".into(),
            });
        }
        if admit_rate {
            for child in nested_keys_for_model(&snapshot, &api_key, request_model) {
                if let Some(rules) =
                    parse_rate_limits(child.preferences.get("rate_limit"), Some(request_model))
                {
                    if tpr_exceeded(&rules, estimated_tokens)
                        || !self
                            .admit_rate(&format!("client:{}", child.token), &rules)
                            .await
                    {
                        return Err(RouteResolutionError {
                            status: StatusCode::TOO_MANY_REQUESTS,
                            message: "Nested API-key rate limit exceeded".into(),
                        });
                    }
                }
            }
        }
        if admit_rate
            && (!self.admit_rate("__global__", &global_rules).await
                || !self
                    .admit_rate(&format!("client:{}", api_key.token), &client_rules)
                    .await)
        {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Too many requests".into(),
            });
        }
        let route_key = diagnostic_key(&snapshot, &api_key, headers, endpoint)?;
        let providers = matching_providers(
            &snapshot,
            &route_key,
            request_model,
            request_body_bytes,
            request_type,
            None,
            endpoint.trim_end_matches('/'),
        )
        .map_err(|_| RouteResolutionError {
            status: StatusCode::NOT_IMPLEMENTED,
            message: "Nested runtime provider is not available".into(),
        })?;
        if providers.is_empty() {
            return Err(RouteResolutionError {
                status: StatusCode::NOT_FOUND,
                message: format!("No available providers at the moment: {request_model}"),
            });
        }
        let providers = self
            .schedule_providers(&api_key, request_model, providers)
            .await;
        if providers.is_empty() {
            return Err(RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "All matching channels are temporarily disabled".into(),
            });
        }
        let hedging = if headers.contains_key(TARGET_PROVIDER_HEADER) {
            HedgingConfig::default()
        } else {
            parse_hedging(&snapshot.preferences)
        };
        Ok(ResolvedRoute { providers, hedging })
    }

    pub(crate) async fn ensure_paid_balance(
        &self,
        persistence: &Persistence,
        api_key: &ApiKey,
    ) -> Result<(), RouteResolutionError> {
        if persistence.disabled() {
            return Ok(());
        }
        let Some(credits) = api_key.preferences.get("credits").and_then(Value::as_f64) else {
            return Ok(());
        };
        if credits == -1.0 {
            return Ok(());
        }
        let created_at = api_key
            .preferences
            .get("created_at")
            .and_then(Value::as_str)
            .and_then(parse_config_datetime)
            .unwrap_or_else(|| unix_seconds_i64().saturating_sub(30 * 86_400));
        let total_cost = persistence
            .total_cost(api_key.token.as_ref(), created_at)
            .await
            .map_err(|error| RouteResolutionError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: format!("Unable to verify API-key balance: {error}"),
            })?;
        if total_cost > credits {
            return Err(RouteResolutionError {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "Balance is insufficient, please check your account.".into(),
            });
        }
        Ok(())
    }

    pub(crate) async fn schedule_providers(
        &self,
        api_key: &ApiKey,
        request_model: &str,
        providers: Vec<Arc<Provider>>,
    ) -> Vec<Arc<Provider>> {
        let controls = self.channel_controls.read().await.routing_rules();
        let key = crate::channel_catalog::key_id(&api_key.token);
        if controls.order(&key, request_model).is_some() {
            return controls.apply(&key, request_model, providers);
        }
        let scheduled = self
            .schedule_configured_providers(api_key, request_model, providers)
            .await;
        if controls.is_empty() {
            scheduled
        } else {
            controls.apply(&key, request_model, scheduled)
        }
    }

    async fn schedule_configured_providers(
        &self,
        api_key: &ApiKey,
        request_model: &str,
        providers: Vec<Arc<Provider>>,
    ) -> Vec<Arc<Provider>> {
        if providers.len() <= 1 {
            return providers;
        }
        let algorithm = api_key
            .preferences
            .get("SCHEDULING_ALGORITHM")
            .and_then(Value::as_str)
            .or_else(|| {
                api_key
                    .preferences
                    .get("api_key_schedule_algorithm")
                    .and_then(Value::as_str)
            })
            .unwrap_or("fixed_priority")
            .trim()
            .to_ascii_lowercase();
        let mut scheduled = weighted_provider_sequence(
            &providers,
            request_model,
            api_key.weights.as_ref(),
            &algorithm,
        );
        if scheduled.is_empty() {
            scheduled = providers;
        }
        if algorithm == "random" {
            shuffle_providers(
                &mut scheduled,
                scheduling_seed(&api_key.token, request_model),
            );
            return scheduled;
        }
        if algorithm == "fixed_priority" || scheduled.len() <= 1 {
            return scheduled;
        }
        let mut cursors = self.routing_cursors.lock().await;
        let cursor = cursors
            .entry((api_key.token.to_string(), request_model.to_owned()))
            .or_insert(0);
        let start = *cursor % scheduled.len();
        *cursor = (start + 1) % scheduled.len();
        scheduled.rotate_left(start);
        scheduled
    }

    pub(crate) async fn select_provider_key(
        &self,
        provider: &Provider,
        original_model: &str,
    ) -> ProviderKeySelection {
        if provider.api_keys.is_empty()
            && provider.client_email.is_some()
            && provider.private_key.is_some()
        {
            return ProviderKeySelection::Selected(String::new());
        }
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
        let algorithm = provider
            .preferences
            .get("api_key_schedule_algorithm")
            .or_else(|| provider.preferences.get("API_KEY_SCHEDULE_ALGORITHM"))
            .and_then(Value::as_str)
            .unwrap_or("round_robin")
            .trim()
            .to_ascii_lowercase();
        let cooldowns = self.key_cooldowns.lock().await;
        let mut candidates = (0..provider.api_keys.len()).collect::<Vec<_>>();
        if algorithm == "fixed_priority" || algorithm == "priority" {
            // preserve configured order
        } else if algorithm == "random" || algorithm == "lottery" {
            shuffle_indices(
                &mut candidates,
                scheduling_seed(provider.name.as_ref(), original_model),
            );
        } else {
            let start = provider.cursor.fetch_add(1, Ordering::Relaxed) % provider.api_keys.len();
            candidates.rotate_left(start);
        }
        drop(cooldowns);
        for index in candidates {
            let key = provider.api_keys[index].clone();
            let cooling = self
                .key_cooldowns
                .lock()
                .await
                .get(&(provider.name.to_string(), key.clone()))
                .is_some_and(|until| *until > now);
            if cooling {
                continue;
            }
            if let Some(rules) = parse_rate_limits(
                provider.preferences.get("api_key_rate_limit"),
                Some(original_model),
            ) {
                if !self.admit_provider_rate(provider, &key, &rules).await {
                    continue;
                }
            }
            return ProviderKeySelection::Selected(key);
        }
        ProviderKeySelection::AllKeysCooling
    }

    pub(crate) async fn cool_failed_route(&self, failure: FailedRoute<'_>) {
        let FailedRoute {
            provider,
            key,
            original_model,
            has_alternative,
            status,
            detail,
            provider_model_unavailable,
            force_quota_cooldown,
        } = failure;
        if provider_model_unavailable || matches!(status, 403 | 404) {
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
        if provider_model_unavailable || provider.api_keys.len() <= 1 {
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

    async fn admit_provider_rate(
        &self,
        provider: &Provider,
        key: &str,
        rules: &[(usize, u64)],
    ) -> bool {
        let now = tokio::time::Instant::now();
        let mut buckets = self.provider_windows.lock().await;
        for (limit, seconds) in rules {
            if *seconds == 0 {
                continue;
            }
            let bucket = format!("provider:{}:{}", provider.name, key);
            let queue = buckets.entry((bucket, *seconds)).or_default();
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
            if *seconds > 0 {
                buckets
                    .entry((format!("provider:{}:{}", provider.name, key), *seconds))
                    .or_default()
                    .push_back(now);
            }
        }
        true
    }

    pub(crate) async fn reset_route_failure(&self, provider: &Provider, original_model: &str) {
        self.route_failures
            .lock()
            .await
            .remove(&(provider.name.to_string(), original_model.to_owned()));
    }

    pub(crate) async fn admit_rate(&self, bucket: &str, rules: &[(usize, u64)]) -> bool {
        let now = tokio::time::Instant::now();
        let mut buckets = self.client_windows.lock().await;
        for (limit, seconds) in rules {
            if *seconds == 0 {
                continue;
            }
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
            if *seconds == 0 {
                continue;
            }
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

    pub(crate) fn hedging_enabled(&self) -> bool {
        self.hedging.active()
    }

    pub(crate) fn hedge_slots(&self) -> usize {
        self.hedging.max_inflight_attempts
    }

    pub(crate) fn record_hedge_trigger(&mut self) {
        self.hedge_trigger_count = self.hedge_trigger_count.saturating_add(1);
    }

    pub(crate) fn record_hedge_cancellations(&mut self, count: usize) {
        self.hedge_cancelled_attempt_count =
            self.hedge_cancelled_attempt_count.saturating_add(count);
    }

    pub(crate) fn set_current_plan(&mut self, plan: &Plan) {
        let Some(provider_name) = plan.provider_name.as_deref() else {
            return;
        };
        let Some(provider) = self
            .providers
            .iter()
            .find(|provider| provider.name.as_ref() == provider_name)
            .cloned()
        else {
            return;
        };
        self.last_provider = Some(provider);
        self.last_provider_key = plan.provider_key.clone();
        self.last_original_model = plan.original_model.clone();
        if let Some(observation) = self.attempt_contexts.get(&plan.attempt_id).cloned() {
            self.last_attempt = Some(observation);
        }
    }

    pub(crate) async fn record_failure_for(&mut self, plan: &Plan, outcome: &Value) -> bool {
        self.set_current_plan(plan);
        self.record_failure(outcome).await
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn last_status(&self) -> u16 {
        self.last_status
    }

    pub fn response_detail(&self) -> String {
        if self.last_provider_model_unavailable {
            format!(
                "All configured providers failed for model {}",
                self.request_model
            )
        } else {
            self.last_detail.clone()
        }
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
            let provider_key_raw = match self
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
            let mut provider_key = provider_key_raw.clone();
            let mut codex_account_id = None;
            if engine == "codex" && provider_key_raw.contains(',') {
                let auth = self
                    .codex_oauth
                    .resolve(
                        &provider_key_raw,
                        preference_string(&provider.preferences, "proxy").as_deref(),
                    )
                    .await?;
                provider_key = auth.bearer;
                codex_account_id = auth.account_id;
            }
            let mut payload = self.base_payload.clone();
            compile_payload(
                &mut payload,
                &provider,
                &self.request_model,
                &original_model,
                &engine,
                self.wants_compact,
            )?;
            let body = serde_json::to_string(&payload)
                .map_err(|error| format!("encode native upstream payload: {error}"))?;
            let attempt_id = native_attempt_id(&self.request_id, attempt_number);
            let mut headers = build_headers(
                &self.request_headers,
                &provider,
                &provider_key,
                &engine,
                self.stream,
                &self.request_id,
                &attempt_id,
            )?;
            if let Some(account_id) = codex_account_id {
                HeaderValue::from_str(&account_id)
                    .map_err(|_| "Codex account ID is not a valid header".to_owned())?;
                headers.insert("chatgpt-account-id".into(), account_id);
            }
            let timeout = resolve_timeouts(
                &self.snapshot,
                &provider,
                &self.request_model,
                &original_model,
                &engine,
                self.stream,
                self.request_type.as_deref(),
                self.api_key.role.as_ref(),
                &self.endpoint,
                "POST",
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
            self.last_provider_key = Some(provider_key_raw.clone());
            self.last_original_model = Some(original_model.clone());
            self.last_attempt = Some(observation.clone());
            self.attempt_contexts
                .insert(attempt_id.clone(), observation);
            crate::channel_metrics::global().start(
                provider.name.as_ref(),
                self.request_model.as_str(),
                original_model.as_str(),
                self.endpoint.as_str(),
                self.stream,
            );
            return Ok(Some(Plan {
                dispatch: self.arrival.map(|arrival| {
                    arrival.attempt(
                        crate::channel_metrics::MetricKey::new(
                            provider.name.as_ref(),
                            &self.request_model,
                            original_model.as_str(),
                            &self.endpoint,
                            self.stream,
                        ),
                        self.request_id.clone(),
                        attempt_id.clone(),
                        &self.api_key.token,
                    )
                }),
                attempt_id,
                url: normalize_upstream_url(&provider.base_url, &engine, self.wants_compact),
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
                connect_timeout_seconds: timeout.connect,
                write_timeout_seconds: timeout.write,
                pool_timeout_seconds: timeout.pool,
                first_byte_timeout_seconds: timeout.first_byte,
                idle_timeout_seconds: timeout.idle,
                total_timeout_seconds: timeout.total,
                provider_name: Some(provider.name.to_string()),
                provider_key: Some(provider_key_raw),
                original_model: Some(original_model),
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
        let policy = classify_provider_failure(
            original_status,
            detail,
            self.last_provider.as_deref(),
            &self.endpoint,
            self.auto_retry(),
        );
        let status = policy.status;
        self.last_status = status;
        self.last_detail = detail.chars().take(4096).collect();
        self.last_provider_model_unavailable = policy.provider_model_unavailable;
        self.has_attempt_failure = true;
        self.last_failure_origin = failure_origin(outcome).to_owned();
        let upstream_status = outcome_status_from(outcome, "upstream_status_code", original_status);
        self.emit_upstream_attempt(
            outcome,
            upstream_status,
            false,
            policy.provider_model_unavailable,
        );
        self.record_current_channel(false, outcome);
        if let (Some(provider), Some(original_model)) =
            (self.last_provider.clone(), self.last_original_model.clone())
        {
            let attempt = self.last_attempt.clone();
            self.emit_routing_attempt(RoutingAttemptEvent {
                attempt_number: attempt
                    .as_ref()
                    .map(|attempt| attempt.attempt_index.saturating_sub(1))
                    .unwrap_or_else(|| self.cursor.saturating_sub(1)),
                provider: &provider,
                original_model: &original_model,
                outcome: "failed",
                attempt_id: attempt.as_ref().map(|attempt| attempt.attempt_id.as_str()),
                skip_reason: None,
                status: Some(status),
            });
        }
        if !policy.request_scoped || policy.force_quota_cooldown {
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
                        provider_model_unavailable: policy.provider_model_unavailable,
                        force_quota_cooldown: policy.force_quota_cooldown,
                    })
                    .await;
            }
        }
        policy.retryable && self.has_attempts_remaining()
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
        if matches!(kind, "semantic_failure" | "semantic_error") {
            // Apply the normal failure accounting and cooldown policy, but
            // never dispatch a retry after output has been committed.
            let _ = self.record_failure(outcome).await;
            self.emit_final_event(self.last_status, kind, outcome);
            return;
        }
        let upstream_status = outcome_status_from(outcome, "upstream_status_code", status);
        if success {
            self.record_success().await;
        } else {
            self.has_attempt_failure = true;
            self.last_failure_origin = failure_origin(outcome).to_owned();
        }
        self.emit_upstream_attempt(outcome, upstream_status, success, false);
        self.record_current_channel(success, outcome);
        if let (Some(provider), Some(original_model)) =
            (self.last_provider.clone(), self.last_original_model.clone())
        {
            let attempt = self.last_attempt.clone();
            self.emit_routing_attempt(RoutingAttemptEvent {
                attempt_number: attempt
                    .as_ref()
                    .map(|attempt| attempt.attempt_index.saturating_sub(1))
                    .unwrap_or_else(|| self.cursor.saturating_sub(1)),
                provider: &provider,
                original_model: &original_model,
                outcome: if success {
                    "succeeded"
                } else {
                    "completed_with_error"
                },
                attempt_id: attempt.as_ref().map(|attempt| attempt.attempt_id.as_str()),
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
            self.response_detail()
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
        let metrics = crate::channel_metrics::global();
        let upstream_model = event
            .provider
            .models
            .get(event.original_model)
            .map(String::as_str)
            .unwrap_or(event.original_model);
        if event.outcome == "started" {
            metrics.start(
                event.provider.name.as_ref(),
                self.request_model.as_str(),
                upstream_model,
                self.endpoint.as_str(),
                self.stream,
            );
        } else if event.outcome == "skipped" {
            metrics.finish(
                event.provider.name.as_ref(),
                self.request_model.as_str(),
                upstream_model,
                self.endpoint.as_str(),
                self.stream,
                "skipped",
                None,
                None,
            );
        }
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
                "role": self.api_key.role.as_ref(),
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

    fn emit_upstream_attempt(
        &mut self,
        outcome: &Value,
        status: u16,
        success: bool,
        provider_model_unavailable: bool,
    ) {
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
        crate::channel_metrics::global().finish(
            &attempt.provider,
            &attempt.request_model,
            &attempt.actual_model,
            &self.endpoint,
            attempt.stream,
            if success { "success" } else { "failed" },
            Some(duration_ms as f64),
            outcome.get("first_output_ms").and_then(Value::as_f64),
        );
        crate::channel_metrics::global().response_timings(
            &attempt.provider,
            &attempt.request_model,
            &attempt.actual_model,
            &self.endpoint,
            attempt.stream,
            outcome.get("response_created_ms").and_then(Value::as_f64),
            outcome.get("first_text_ms").and_then(Value::as_f64),
        );
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
                "provider_model_unavailable": provider_model_unavailable,
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
                "role": self.api_key.role.as_ref(),
                "actual_model": attempt.actual_model,
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "attempt_status_code": status,
                "attempt_status_class": status_class(status),
                "semantic_status_code": outcome.get("status_code").and_then(Value::as_u64),
                "attempt_success": success,
                "attempt_outcome": attempt_outcome,
                "provider_model_unavailable": provider_model_unavailable,
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
            "role": self.api_key.role.as_ref(),
            "actual_model": final_actual_model,
            "stream": self.stream,
            "status_code": status,
            "status_class": status_class(status),
            "status_origin": status_origin,
            "error_type": (!success || status >= 400).then_some(kind),
            "routing_attempt_count": self.routing_attempts,
            "routing_skip_count": self.routing_skips,
            "upstream_attempt_count": self.upstream_attempts,
            "hedging": {
                "enabled": self.hedging.enabled,
                "max_inflight_attempts": self.hedging.max_inflight_attempts,
                "winner_policy": "first_valid_success",
                "trigger_count": self.hedge_trigger_count,
                "cancelled_attempt_count": self.hedge_cancelled_attempt_count,
            },
            "upstream_duration_ms": self.upstream_duration_ms,
            "routing_attempts": self.routing_ledger,
            "upstream_attempts": self.upstream_ledger,
            "routing_attempts_omitted_count": self.routing_attempts.saturating_sub(self.routing_ledger.len()),
            "upstream_attempts_omitted_count": self.upstream_attempts.saturating_sub(self.upstream_ledger.len()),
            "last_failure_origin": self.last_failure_origin,
            "snapshot_revision": self.snapshot.revision.to_string(),
            "rust_responses_data_plane": true,
        });
        let (prompt_tokens, completion_tokens, total_tokens) = usage_tokens(outcome);
        let (prompt_price, completion_price) =
            model_prices(&self.snapshot.preferences, &self.request_model);
        self.persistence.record_request(RequestStat {
            fact_usage: crate::fact_usage::FactUsage::from_usage(outcome.get("usage")),
            stream: self.stream,
            upstream_model: final_actual_model.unwrap_or_default().to_owned(),
            status,
            first_output_ms: outcome.get("first_output_ms").and_then(Value::as_f64),
            response_created_ms: outcome.get("response_created_ms").and_then(Value::as_f64),
            first_text_ms: outcome.get("first_text_ms").and_then(Value::as_f64),
            request_id: self.request_id.clone(),
            trace_id: trace_id(&self.request_headers, &self.request_id),
            endpoint: self.endpoint.clone(),
            client_ip: client_ip(&self.request_headers),
            process_time: elapsed_ms as f64 / 1000.0,
            first_response_time: 0.0,
            provider: final_provider.unwrap_or_default().to_owned(),
            model: self.request_model.clone(),
            api_key: self.api_key.token.to_string(),
            is_flagged: !success || status >= 400,
            text: detail.chars().take(4096).collect(),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_price,
            completion_price,
            timing_spans: summary.to_string(),
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
                "role": self.api_key.role.as_ref(),
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
            .map(|value| match value {
                Value::Bool(enabled) => *enabled,
                Value::Number(number) => number.as_u64().unwrap_or(0) > 0,
                Value::String(text) => text.trim().parse::<usize>().map(|n| n > 0).unwrap_or(true),
                _ => true,
            })
            .unwrap_or(true)
    }

    fn record_current_channel(&self, success: bool, outcome: &Value) {
        let (Some(provider), Some(provider_key)) =
            (self.last_provider.as_ref(), self.last_provider_key.as_ref())
        else {
            return;
        };
        self.persistence.record_channel(ChannelStat {
            duration_ms: self
                .last_attempt
                .as_ref()
                .map(|a| a.started_at.elapsed().as_secs_f64() * 1000.0),
            first_output_ms: outcome.get("first_output_ms").and_then(Value::as_f64),
            response_created_ms: outcome.get("response_created_ms").and_then(Value::as_f64),
            first_text_ms: outcome.get("first_text_ms").and_then(Value::as_f64),
            request_id: self.request_id.clone(),
            attempt_id: self
                .last_attempt
                .as_ref()
                .map(|attempt| attempt.attempt_id.clone())
                .unwrap_or_default(),
            provider: provider.name.to_string(),
            model: self.request_model.clone(),
            upstream_model: self.last_original_model.clone().unwrap_or_default(),
            api_key: self.api_key.token.to_string(),
            provider_api_key: provider_key.clone(),
            success,
            endpoint: self.endpoint.clone(),
            stream: self.stream,
        });
    }
}

fn usage_tokens(outcome: &Value) -> (i64, i64, i64) {
    let usage = outcome.get("usage").filter(|value| value.is_object());
    let read = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| {
                usage
                    .and_then(|value| value.get(*name))
                    .and_then(Value::as_i64)
            })
            .unwrap_or_default()
    };
    let prompt = read(&["input_tokens", "prompt_tokens"]);
    let completion = read(&["output_tokens", "completion_tokens"]);
    let total = read(&["total_tokens"]);
    (
        prompt,
        completion,
        total.max(prompt.saturating_add(completion)),
    )
}

fn model_prices(preferences: &Map<String, Value>, model: &str) -> (f64, f64) {
    let prices = preferences.get("model_price").and_then(Value::as_object);
    let encoded = prices
        .and_then(|prices| {
            prices
                .iter()
                .find(|(prefix, _)| {
                    prefix.as_str() != "default" && model.starts_with(prefix.as_str())
                })
                .map(|(_, value)| value)
                .or_else(|| prices.get("default"))
        })
        .and_then(|value| value.as_str())
        .unwrap_or("0.3,1");
    let mut parts = encoded.split(',').map(str::trim);
    let prompt = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.3);
    let completion = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);
    (prompt, completion)
}

fn trace_id(headers: &HeaderMap, fallback: &str) -> String {
    headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split('-').nth(1))
        .filter(|value| value.len() == 32)
        .unwrap_or(fallback)
        .to_owned()
}

fn client_ip(headers: &HeaderMap) -> String {
    headers
        .get("x-forwarded-for")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .or_else(|| {
            headers
                .get("x-real-ip")
                .and_then(|value| value.to_str().ok())
        })
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn emit_native_rejection(status: u16, observation: NativeRejectionObservation<'_>) {
    let status_origin = native_rejection_origin(observation.reason);
    let summary = json!({
        "request_kind": "responses",
        "terminal_kind": "native_rejection",
        "rejection_reason": observation.reason,
        "model": observation.request_model,
        "role": observation.role,
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
            "role": observation.role,
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

#[allow(clippy::too_many_arguments)]
pub async fn prepare_native_request(
    store: &NativeConfigStore,
    codex_oauth: CodexOAuthManager,
    persistence: Persistence,
    parts: &Parts,
    storage: &StoredBody,
    observation: &SpoolObservation,
    memory_reservation: MemoryReservation,
    endpoint: &str,
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
                role: "",
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
                role: "",
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
    if !api_key.native_supported {
        return NativePreparation::Fallback;
    }
    let role = api_key.role.as_ref();
    if let Err(error) = store.ensure_paid_balance(&persistence, &api_key).await {
        return NativePreparation::Response(json_response(
            error.status,
            json!({"error": error.message}),
        ));
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
                    role,
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
                    role,
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
                role,
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
                role,
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
    let request_type = detect_request_type(object);
    let reasoning_effort = request_reasoning_effort(object);
    let normalized_endpoint = endpoint.trim_end_matches('/');
    let wants_compact = normalized_endpoint == "/v1/responses/compact";
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
                        role,
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
    let route_key = match diagnostic_key(&snapshot, &api_key, &parts.headers, normalized_endpoint) {
        Ok(key) => key,
        Err(error) => {
            return NativePreparation::Response(json_response(
                error.status,
                json!({"error":error.message}),
            ))
        }
    };
    let providers = match matching_providers(
        &snapshot,
        &route_key,
        &request_model,
        observation.body_bytes,
        request_type,
        reasoning_effort.as_deref(),
        normalized_endpoint,
    ) {
        Ok(providers) if !providers.is_empty() => providers,
        Ok(_) => {
            emit_native_rejection(
                404,
                NativeRejectionObservation {
                    request_id: &request_id,
                    request_model: Some(&request_model),
                    stream: Some(stream),
                    role,
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
    let providers = store
        .schedule_providers(&api_key, &request_model, providers)
        .await;
    if providers.is_empty() {
        return NativePreparation::Response(json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"All matching channels are temporarily disabled"}),
        ));
    }
    if providers.iter().any(|provider| {
        !matches!(provider.engine.as_ref(), "gpt" | "codex")
            || provider.api_keys.is_empty()
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
    if tpr_exceeded(
        &client_rate_rules,
        (observation.body_bytes / 4).max(1) as usize,
    ) {
        return NativePreparation::Response(json_response(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error":"Tokens per request limit exceeded"}),
        ));
    }
    // Admit only after the request is proven native-safe. A compatibility
    // fallback must not consume both the Rust and Python rate-limit buckets.
    if !store.admit_rate("__global__", &global_rate_rules).await {
        emit_native_rejection(
            429,
            NativeRejectionObservation {
                request_id: &request_id,
                request_model: Some(&request_model),
                stream: Some(stream),
                role,
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
                role,
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
    let targeted = parts.headers.contains_key(TARGET_PROVIDER_HEADER);
    let retry_count = if targeted {
        1
    } else {
        compute_retry_count(&providers)
            .max(api_key_retry_budget(&api_key, providers.len()))
            .min(100)
    };
    NativePreparation::Ready(NativeRoute {
        store: store.clone(),
        codex_oauth,
        persistence,
        snapshot: snapshot.clone(),
        api_key,
        providers,
        base_payload: payload,
        request_headers: parts.headers.clone(),
        request_model,
        endpoint: normalized_endpoint.to_owned(),
        request_type: request_type.map(str::to_owned),
        wants_compact,
        stream,
        request_id,
        request_body_bytes: observation.body_bytes,
        cursor: 0,
        max_attempts: retry_count,
        hedging: if targeted {
            HedgingConfig::default()
        } else {
            parse_hedging(&snapshot.preferences)
        },
        attempt_contexts: HashMap::new(),
        hedge_trigger_count: 0,
        hedge_cancelled_attempt_count: 0,
        last_provider: None,
        last_provider_key: None,
        last_original_model: None,
        last_attempt: None,
        last_status: 502,
        last_detail: String::new(),
        last_provider_model_unavailable: false,
        has_attempt_failure: false,
        last_failure_origin: String::new(),
        routing_attempts: 0,
        routing_skips: 0,
        upstream_attempts: 0,
        upstream_duration_ms: 0,
        routing_ledger: Vec::new(),
        upstream_ledger: Vec::new(),
        arrival: parts
            .extensions
            .get::<crate::request_timing::RequestArrival>()
            .copied(),
        started_at: tokio::time::Instant::now(),
        final_emitted: false,
        _memory_reservation: memory_reservation,
    })
}

fn nested_keys_for_model(snapshot: &Snapshot, key: &ApiKey, model: &str) -> Vec<Arc<ApiKey>> {
    fn walk(
        snapshot: &Snapshot,
        key: &ApiKey,
        model: &str,
        out: &mut Vec<Arc<ApiKey>>,
        seen: &mut std::collections::BTreeSet<String>,
    ) {
        if !seen.insert(key.token.to_string()) {
            return;
        }
        let rules = key
            .preferences
            .get("__route_graph")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        for rule in rules {
            let Some((alias, requested)) = rule.split_once('/') else {
                continue;
            };
            if (requested == "*" || requested == model) && snapshot.api_keys.get(alias).is_some() {
                let child = snapshot.api_keys.get(alias).unwrap();
                out.push(child.clone());
                walk(snapshot, child, model, out, seen);
            }
        }
        seen.remove(key.token.as_ref());
    }
    let mut out = Vec::new();
    walk(
        snapshot,
        key,
        model,
        &mut out,
        &mut std::collections::BTreeSet::new(),
    );
    out
}

// Explicit administrator-only diagnostic routing. The temporary key is never
// written to the snapshot; normal traffic retains its configured routing graph.
pub(crate) const TARGET_PROVIDER_HEADER: &str = "x-uni-api-provider";

fn diagnostic_key(
    snapshot: &Snapshot,
    key: &ApiKey,
    headers: &HeaderMap,
    endpoint: &str,
) -> Result<ApiKey, RouteResolutionError> {
    let Some(value) = headers.get(TARGET_PROVIDER_HEADER) else {
        return Ok(key.clone());
    };
    if !crate::channel_catalog::can_inspect_all(snapshot, key) {
        return Err(RouteResolutionError {
            status: StatusCode::FORBIDDEN,
            message: "Targeted requests require a platform administrator key".into(),
        });
    }
    if endpoint.trim_end_matches('/') != "/v1/responses" {
        return Err(RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Targeted requests require /v1/responses".into(),
        });
    }
    let name = value
        .to_str()
        .ok()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Invalid target provider".into(),
        })?;
    if name.contains('/') || snapshot.api_keys.contains_key(name) {
        return Err(RouteResolutionError {
            status: StatusCode::BAD_REQUEST,
            message: "Ambiguous target provider name".into(),
        });
    }
    if !snapshot.providers_by_name.contains_key(name) {
        return Err(RouteResolutionError {
            status: StatusCode::NOT_FOUND,
            message: "Target provider not found".into(),
        });
    }
    let mut diagnostic = key.clone();
    diagnostic.model_rules = Arc::new(vec![format!("{name}/*")]);
    let mut preferences = (*key.preferences).clone();
    preferences.remove("__route_graph");
    preferences.insert("__diagnostic_provider".into(), json!(name));
    diagnostic.preferences = Arc::new(preferences);
    Ok(diagnostic)
}

fn matching_providers(
    snapshot: &Snapshot,
    api_key: &ApiKey,
    request_model: &str,
    request_body_bytes: u64,
    request_type: Option<&str>,
    reasoning_effort: Option<&str>,
    endpoint: &str,
) -> Result<Vec<Arc<Provider>>, ()> {
    // Walk the key graph recursively. A child key contributes only the model
    // rules it explicitly owns; its token is never turned into a synthetic
    // upstream channel, so parent and child policy boundaries stay visible to
    // scheduling and accounting.
    fn collect(
        snapshot: &Snapshot,
        key: &ApiKey,
        model: &str,
        out: &mut Vec<Arc<Provider>>,
        visiting: &mut std::collections::BTreeSet<String>,
    ) {
        if !visiting.insert(key.token.to_string()) {
            return;
        }
        let rules = key
            .preferences
            .get("__route_graph")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_else(|| key.model_rules.iter().map(String::as_str).collect());
        for rule in rules {
            let Some((alias, requested)) = rule.split_once('/') else {
                if rule == "all"
                    || rule == model
                    || (rule.starts_with('<')
                        && rule.ends_with('>')
                        && &rule[1..rule.len() - 1] == model)
                {
                    out.extend(
                        snapshot
                            .providers
                            .iter()
                            .filter(|p| p.models.contains_key(model))
                            .cloned(),
                    );
                }
                continue;
            };
            if let Some(child) = snapshot.api_keys.get(alias) {
                if requested == "*" || requested == model {
                    collect(snapshot, child, model, out, visiting);
                }
                continue;
            }
            if let Some(provider) = snapshot.providers_by_name.get(alias) {
                if (requested == "*" || requested == model) && provider.models.contains_key(model) {
                    out.push(provider.clone());
                }
            }
        }
        visiting.remove(key.token.as_ref());
    }
    let mut matches = Vec::new();
    collect(
        snapshot,
        api_key,
        request_model,
        &mut matches,
        &mut std::collections::BTreeSet::new(),
    );
    // First occurrence defines priority, including nested key and wildcard rules.
    // Sorting to deduplicate silently changes fixed_priority into name order.
    let mut seen = BTreeSet::new();
    matches.retain(|provider| seen.insert(provider.name.clone()));
    matches.retain(|provider| {
        crate::channel_controls::temporary_allowed(provider, api_key)
            && !provider.excluded_endpoints.iter().any(|excluded| {
                excluded
                    .trim_end_matches('/')
                    .eq_ignore_ascii_case(endpoint)
            })
            && provider_accepts_body(provider, request_body_bytes)
            && provider_accepts_request_type(provider, request_type)
            && provider_accepts_request_rules(
                provider,
                endpoint,
                request_model,
                reasoning_effort,
                request_type,
            )
    });
    Ok(matches)
}

fn weighted_provider_sequence(
    providers: &[Arc<Provider>],
    request_model: &str,
    weights: &Map<String, Value>,
    algorithm: &str,
) -> Vec<Arc<Provider>> {
    let weighted = providers
        .iter()
        .filter_map(|provider| {
            let exact = format!("{}/{request_model}", provider.name);
            let wildcard = format!("{}/*", provider.name);
            let weight = weights
                .get(&exact)
                .or_else(|| weights.get(&wildcard))
                .and_then(positive_weight)?;
            Some((provider.clone(), weight))
        })
        .collect::<Vec<_>>();
    if weighted.len() <= 1 {
        return providers.to_vec();
    }
    match algorithm {
        "weighted_round_robin" | "smart_round_robin" => smooth_weighted_sequence(&weighted),
        "lottery" => lottery_sequence(&weighted, scheduling_seed("lottery", request_model)),
        _ => weighted.into_iter().map(|(provider, _)| provider).collect(),
    }
}

fn positive_weight(value: &Value) -> Option<usize> {
    let weight = value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        .or_else(|| value.as_str().and_then(|value| value.parse::<u64>().ok()))?;
    usize::try_from(weight).ok().filter(|value| *value > 0)
}

fn smooth_weighted_sequence(weighted: &[(Arc<Provider>, usize)]) -> Vec<Arc<Provider>> {
    let total = weighted
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<usize>()
        .min(16_384);
    let mut current = vec![0i64; weighted.len()];
    let total_weight = weighted
        .iter()
        .map(|(_, weight)| i64::try_from(*weight).unwrap_or(i64::MAX / 4))
        .sum::<i64>()
        .max(1);
    let mut sequence = Vec::with_capacity(total);
    for _ in 0..total {
        let mut selected = 0usize;
        let mut selected_current = i64::MIN;
        let mut selected_configured = 1i64;
        for (index, (_, weight)) in weighted.iter().enumerate() {
            let configured = i64::try_from(*weight).unwrap_or(i64::MAX / 4);
            current[index] = current[index].saturating_add(configured);
            if selected_current == i64::MIN
                || current[index].saturating_mul(selected_configured)
                    > selected_current.saturating_mul(configured)
            {
                selected = index;
                selected_current = current[index];
                selected_configured = configured;
            }
        }
        sequence.push(weighted[selected].0.clone());
        current[selected] = current[selected].saturating_sub(total_weight);
    }
    sequence
}

fn lottery_sequence(weighted: &[(Arc<Provider>, usize)], mut seed: u64) -> Vec<Arc<Provider>> {
    let total = weighted
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<usize>()
        .min(16_384);
    let total_weight = weighted
        .iter()
        .map(|(_, weight)| *weight as u64)
        .sum::<u64>()
        .max(1);
    let mut sequence = Vec::with_capacity(total);
    for _ in 0..total {
        seed = xorshift(seed);
        let ticket = seed % total_weight;
        let mut cumulative = 0u64;
        let selected = weighted
            .iter()
            .find(|(_, weight)| {
                cumulative = cumulative.saturating_add(*weight as u64);
                ticket < cumulative
            })
            .map(|(provider, _)| provider)
            .unwrap_or(&weighted[0].0);
        sequence.push(selected.clone());
    }
    sequence
}

fn shuffle_indices(indices: &mut [usize], mut seed: u64) {
    for index in (1..indices.len()).rev() {
        seed = xorshift(seed);
        indices.swap(index, seed as usize % (index + 1));
    }
}

fn shuffle_providers(providers: &mut [Arc<Provider>], mut seed: u64) {
    for index in (1..providers.len()).rev() {
        seed = xorshift(seed);
        providers.swap(index, seed as usize % (index + 1));
    }
}

fn scheduling_seed(scope: &str, request_model: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(request_model.as_bytes());
    hasher.update(
        SCHEDULING_NONCE
            .fetch_add(1, Ordering::Relaxed)
            .to_le_bytes(),
    );
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"))
}

fn xorshift(mut value: u64) -> u64 {
    if value == 0 {
        value = 0x9e37_79b9_7f4a_7c15;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn unix_seconds_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

fn parse_config_datetime(value: &str) -> Option<i64> {
    if let Ok(value) = value.parse::<f64>() {
        return value.is_finite().then_some(value as i64);
    }
    let value = value.trim();
    let (date, raw_time) = value.split_once('T').or_else(|| value.split_once(' '))?;
    let (time, offset_seconds) = if let Some(time) = raw_time.strip_suffix('Z') {
        (time, 0i64)
    } else if let Some(index) = raw_time
        .char_indices()
        .skip(1)
        .filter(|(_, value)| matches!(value, '+' | '-'))
        .map(|(index, _)| index)
        .fold(None, |_, index| Some(index))
    {
        let (time, offset) = raw_time.split_at(index);
        let sign = if offset.starts_with('-') { -1 } else { 1 };
        let mut parts = offset[1..].split(':');
        let hours = parts.next()?.parse::<i64>().ok()?;
        let minutes = parts.next().unwrap_or("0").parse::<i64>().ok()?;
        (time, sign * (hours * 3600 + minutes * 60))
    } else {
        (raw_time, 0i64)
    };
    let mut date = date.split('-').map(|value| value.parse::<i64>().ok());
    let (year, month, day) = (date.next()??, date.next()??, date.next()??);
    let mut time = time.split(':');
    let hour = time.next()?.parse::<i64>().ok()?;
    let minute = time.next()?.parse::<i64>().ok()?;
    let second = time.next()?.split('.').next()?.parse::<i64>().ok()?;
    Some(
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second
            - offset_seconds,
    )
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn api_key_retry_budget(api_key: &ApiKey, provider_count: usize) -> usize {
    let configured = api_key
        .preferences
        .get("AUTO_RETRY")
        .map(|value| match value {
            Value::Bool(enabled) => usize::from(*enabled),
            Value::Number(number) => number.as_u64().unwrap_or(0) as usize,
            Value::String(text) => text.trim().parse::<usize>().unwrap_or(1),
            _ => 1,
        })
        .unwrap_or(1);
    provider_count.saturating_add(configured)
}

pub(crate) fn compute_retry_count(providers: &[Arc<Provider>]) -> usize {
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

fn detect_request_type(payload: &Map<String, Value>) -> Option<&'static str> {
    let is_compaction = payload
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
        });
    is_compaction.then_some("compaction")
}

pub(crate) fn provider_accepts_request_type(
    provider: &Provider,
    request_type: Option<&str>,
) -> bool {
    if !provider.only_request_types.is_empty()
        && !request_type.is_some_and(|value| {
            provider
                .only_request_types
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(value))
        })
    {
        return false;
    }
    !request_type.is_some_and(|value| {
        provider
            .excluded_request_types
            .iter()
            .any(|excluded| excluded.eq_ignore_ascii_case(value))
    })
}

fn request_rule_values(value: &Value) -> Vec<Value> {
    if let Some(values) = value.as_array() {
        return values
            .iter()
            .filter(|item| item.is_object())
            .cloned()
            .collect();
    }
    value
        .is_object()
        .then(|| value.clone())
        .into_iter()
        .collect()
}

fn request_reasoning_effort(payload: &Map<String, Value>) -> Option<String> {
    payload
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| {
            payload
                .get("reasoning")
                .and_then(Value::as_object)
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(crate) fn provider_accepts_request_rules(
    provider: &Provider,
    endpoint: &str,
    request_model: &str,
    reasoning_effort: Option<&str>,
    request_type: Option<&str>,
) -> bool {
    let upstream_model = provider
        .models
        .get(request_model)
        .map(String::as_str)
        .unwrap_or(request_model);
    !provider.excluded_request_rules.iter().any(|rule| {
        exclude_request_rule_matches(
            rule,
            endpoint,
            request_model,
            upstream_model,
            reasoning_effort,
            request_type,
        )
    })
}

fn exclude_request_rule_matches(
    rule: &Value,
    endpoint: &str,
    request_model: &str,
    upstream_model: &str,
    reasoning_effort: Option<&str>,
    request_type: Option<&str>,
) -> bool {
    let Some(condition) = rule.get("match").and_then(Value::as_object) else {
        return false;
    };
    if condition.is_empty() {
        return false;
    }
    condition.iter().all(|(key, expected)| match key.as_str() {
        "endpoint" => request_rule_value_matches(expected, Some(endpoint), true),
        "request_model" => request_rule_value_matches(expected, Some(request_model), false),
        "upstream_model" => request_rule_value_matches(expected, Some(upstream_model), false),
        "reasoning_effort" => request_rule_value_matches(expected, reasoning_effort, false),
        "request_type" => request_rule_value_matches(expected, request_type, false),
        _ => false,
    })
}

fn request_rule_value_matches(expected: &Value, actual: Option<&str>, endpoint: bool) -> bool {
    if let Some(values) = expected.as_array() {
        return values
            .iter()
            .any(|value| request_rule_value_matches(value, actual, endpoint));
    }
    let (Some(expected), Some(actual)) = (expected.as_str(), actual) else {
        return false;
    };
    let normalize = |value: &str| {
        let mut value = value.trim().trim_end_matches('/').to_ascii_lowercase();
        if endpoint && !value.is_empty() && !value.starts_with('/') {
            value.insert(0, '/');
        }
        value
    };
    let expected = normalize(expected);
    let actual = normalize(actual);
    if expected.is_empty() || actual.is_empty() {
        return false;
    }
    expected == "*"
        || expected == actual
        || expected
            .strip_suffix('*')
            .is_some_and(|prefix| actual.starts_with(prefix))
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
    wants_compact: bool,
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
        if wants_compact {
            root.remove("store");
        }
    }
    if normalization_enabled(provider, request_model, original_model) {
        normalize_response_root(root)?;
    }
    Ok(())
}

pub(crate) fn apply_overrides(
    root: &mut Map<String, Value>,
    provider: &Provider,
    request_model: &str,
) {
    if let Some(overrides) = provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
    {
        apply_override_section(root, overrides, provider, true);
        if let Some(model) = overrides.get(request_model).and_then(Value::as_object) {
            apply_override_section(root, model, provider, false);
        }
    }
    // Codex wire requirements also apply when no overrides are configured,
    // and cannot be undone by provider-wide or model-specific overrides.
    // Endpoint-specific sanitizers still run afterwards (e.g. compact drops store).
    if provider.engine.trim().eq_ignore_ascii_case("codex") {
        root.insert("store".into(), Value::Bool(false));
        root.remove("response_format");
        root.remove("temperature");
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

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct Timeouts {
    pub(crate) connect: Option<f64>,
    pub(crate) write: Option<f64>,
    pub(crate) pool: Option<f64>,
    pub(crate) first_byte: Option<f64>,
    pub(crate) idle: Option<f64>,
    pub(crate) total: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_timeouts(
    snapshot: &Snapshot,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    engine: &str,
    stream: bool,
    request_type: Option<&str>,
    role: &str,
    endpoint: &str,
    method: &str,
) -> Timeouts {
    let base = model_timeout(
        provider,
        &snapshot.preferences,
        request_model,
        original_model,
    );
    let context = HashMap::from([
        ("provider", provider.name.as_ref()),
        ("endpoint", endpoint),
        ("method", method),
        ("engine", engine),
        ("model", request_model),
        ("request_model", request_model),
        ("upstream_model", original_model),
        ("request_type", request_type.unwrap_or_default()),
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
        connect: values.get("connect").and_then(Value::as_f64),
        write: values.get("write").and_then(Value::as_f64),
        pool: values.get("pool").and_then(Value::as_f64),
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
    model_preference(
        provider,
        global,
        request_model,
        original_model,
        "model_timeout",
    )
    .unwrap_or(100.0)
}

fn model_preference(
    provider: &Provider,
    global: &Map<String, Value>,
    request_model: &str,
    original_model: &str,
    preference: &str,
) -> Option<f64> {
    for preferences in [&provider.preferences, global] {
        let Some(timeout) = preferences.get(preference) else {
            continue;
        };
        if let Some(value) = timeout.as_f64() {
            return Some(value);
        }
        let Some(values) = timeout.as_object() else {
            continue;
        };
        if let Some(value) = model_timeout_value(values, request_model) {
            return Some(value);
        }
        if let Some(value) = model_timeout_value(values, original_model) {
            return Some(value);
        }
        if let Some(value) = values
            .iter()
            .find(|(key, value)| key.eq_ignore_ascii_case("default") && value.as_f64().is_some())
            .and_then(|(_, value)| value.as_f64())
        {
            return Some(value);
        }
    }
    None
}

fn model_timeout_value(values: &Map<String, Value>, model: &str) -> Option<f64> {
    let normalized_model = model.to_ascii_lowercase();

    values
        .iter()
        .find(|(key, value)| {
            !key.eq_ignore_ascii_case("default")
                && key.eq_ignore_ascii_case(model)
                && value.as_f64().is_some()
        })
        .and_then(|(_, value)| value.as_f64())
        .or_else(|| {
            values
                .iter()
                .find(|(key, value)| {
                    !key.is_empty()
                        && !key.eq_ignore_ascii_case("default")
                        && normalized_model.contains(&key.to_ascii_lowercase())
                        && value.as_f64().is_some()
                })
                .and_then(|(_, value)| value.as_f64())
        })
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

fn normalize_upstream_url(base_url: &str, engine: &str, wants_compact: bool) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if wants_compact {
        if base.ends_with("/v1/responses/compact") || base.ends_with("/responses/compact") {
            return base.to_owned();
        }
        let response_url = normalize_upstream_url(base, engine, false);
        return format!("{response_url}/compact");
    }
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

fn request_type_values(value: &Value) -> Vec<String> {
    endpoint_values(value)
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
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
    if is_provider_model_unavailable(status, detail) {
        return 503;
    }
    if detail.contains("<center><h1>400 Bad Request</h1></center>")
        || detail.contains("Provider API error: bad response status code 400")
        || status == 400
            && (is_model_pricing_unconfigured(detail)
                || is_provider_minimum_input_restriction(detail))
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

pub(crate) fn classify_provider_failure(
    original_status: u16,
    detail: &str,
    provider: Option<&Provider>,
    endpoint: &str,
    auto_retry: bool,
) -> ProviderFailurePolicy {
    let provider_model_unavailable = is_provider_model_unavailable(original_status, detail);
    let status = remap_provider_status(original_status, detail);
    let codex_model_unsupported = status == 400
        && matches!(endpoint, "/v1/responses" | "/v1/responses/compact")
        && provider.is_some_and(|provider| provider.engine.eq_ignore_ascii_case("codex"))
        && detail
            .to_ascii_lowercase()
            .contains("model is not supported when using codex with a chatgpt account");
    let missing_persisted_item = status == 404 && is_missing_persisted_item_error(detail);
    let request_scoped = matches!(status, 400 | 413) || missing_persisted_item;
    let azure_request = matches!(status, 400 | 413)
        && provider.is_some_and(|provider| is_azure_provider(&provider.base_url));
    ProviderFailurePolicy {
        status,
        retryable: auto_retry && (!request_scoped || codex_model_unsupported || azure_request),
        request_scoped,
        provider_model_unavailable,
        force_quota_cooldown: codex_model_unsupported,
    }
}

fn is_provider_model_unavailable(status: u16, detail: &str) -> bool {
    if !matches!(status, 400 | 404) {
        return false;
    }

    const CODES: &[&str] = &[
        "model_not_found",
        "model_not_supported",
        "model_unsupported",
        "unknown_provider",
        "unsupported_model",
    ];
    const MARKERS: &[&str] = &[
        "unknown provider for model",
        "no provider found for model",
        "no provider available for model",
        "model is not supported by this provider",
    ];

    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let parsed = serde_json::from_str::<Value>(&candidate).ok();
        let (code, message) = if let Some(payload) = parsed.as_ref() {
            let error = payload
                .get("error")
                .filter(|value| value.is_object())
                .or_else(|| payload.get("detail").filter(|value| value.is_object()));
            (
                error
                    .and_then(|value| value.get("code"))
                    .and_then(Value::as_str),
                error
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str),
            )
        } else {
            (None, Some(candidate.as_str()))
        };

        if code.is_some_and(|value| CODES.contains(&value.trim().to_ascii_lowercase().as_str())) {
            return true;
        }
        if message.is_some_and(|value| {
            let lower = value.to_ascii_lowercase();
            // Some providers report model availability as invalid_request_error
            // without a model-specific code. Match the whole message so echoed
            // input in an ordinary validation error does not trigger failover.
            lower.trim() == "this model is not available."
                || MARKERS.iter().any(|marker| lower.contains(marker))
        }) {
            return true;
        }

        let Some(nested) = message.filter(|value| value.trim_start().starts_with('{')) else {
            break;
        };
        candidate = nested.to_owned();
    }
    false
}

fn is_azure_provider(base_url: &str) -> bool {
    let Ok(url) = Url::parse(base_url) else {
        return false;
    };
    url.host_str() == Some("models.inference.ai.azure.com")
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
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

fn is_provider_minimum_input_restriction(detail: &str) -> bool {
    // A channel key's minimum-input policy is not a malformed client request.
    // Read only error messages (including JSON-escaped/wrapped messages), never
    // echoed request fields. Neither the provider nor the numeric limit matters.
    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let parsed = serde_json::from_str::<Value>(&candidate).ok();
        let message = match parsed.as_ref() {
            Some(payload) => payload
                .pointer("/error/message")
                .or_else(|| payload.pointer("/detail/message"))
                .or_else(|| payload.get("message"))
                .or_else(|| payload.get("error"))
                .or_else(|| payload.get("detail"))
                .unwrap_or(payload)
                .as_str(),
            None => Some(candidate.as_str()),
        };
        let Some(message) = message else {
            return false;
        };
        if message.trim_start().starts_with('{') {
            candidate = message.to_owned();
            continue;
        }
        let compact = message
            .split_whitespace()
            .collect::<String>()
            .to_ascii_lowercase();
        return [
            ("该令牌不接受输入少于", "token的请求"),
            ("thiskeydoesnotacceptrequestswithfewerthan", "inputtokens"),
        ]
        .iter()
        .any(|(prefix, suffix)| {
            compact.split_once(prefix).is_some_and(|(_, tail)| {
                let after_number = tail.trim_start_matches(|c: char| c.is_ascii_digit());
                after_number.len() < tail.len() && after_number.starts_with(suffix)
            })
        });
    }
    false
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

fn tpr_exceeded(rules: &[(usize, u64)], estimated_tokens: usize) -> bool {
    rules
        .iter()
        .any(|(limit, seconds)| *seconds == 0 && estimated_tokens > *limit)
}

pub(crate) fn parse_rate_limits(
    value: Option<&Value>,
    model: Option<&str>,
) -> Option<Vec<(usize, u64)>> {
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
            "tpr" => 0,
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

pub(crate) fn extract_api_key(headers: &HeaderMap) -> Option<String> {
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

pub(crate) fn request_id(headers: &HeaderMap) -> String {
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
            project_id: None,
            private_key: None,
            client_email: None,
            aws_access_key: None,
            aws_secret_key: None,
            aws_session_token: None,
            cf_account_id: None,
            region: Arc::from("global"),
            models: Arc::new(HashMap::from([(
                "gpt-public".into(),
                "gpt-upstream".into(),
            )])),
            preferences: Arc::new(Map::from_iter([(
                "post_body_parameter_overrides".into(),
                json!({"store": false, "__remove__": ["temperature"]}),
            )])),
            excluded_endpoints: Arc::new(Vec::new()),
            only_request_types: Arc::new(Vec::new()),
            excluded_request_types: Arc::new(Vec::new()),
            excluded_request_rules: Arc::new(Vec::new()),
            cursor: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn named_provider(name: &str) -> Arc<Provider> {
        let mut value = provider();
        value.name = name.to_owned().into();
        Arc::new(value)
    }

    async fn native_route_for_test(provider: Arc<Provider>, max_attempts: usize) -> NativeRoute {
        let store = NativeConfigStore::new();
        let snapshot = Arc::new(Snapshot {
            revision: Arc::from("0".repeat(64)),
            preferences: Arc::new(Map::new()),
            api_keys: Arc::new(HashMap::new()),
            api_key_order: Arc::new(Vec::new()),
            providers: Arc::new(vec![provider.clone()]),
            providers_by_name: Arc::new(HashMap::from([(
                provider.name.to_string(),
                provider.clone(),
            )])),
            api_config: Arc::new(json!({})),
        });
        let api_key = Arc::new(ApiKey {
            token: Arc::from("client-key"),
            model_rules: Arc::new(Vec::new()),
            role: Arc::from("user"),
            preferences: Arc::new(Map::from_iter([("AUTO_RETRY".into(), json!(true))])),
            weights: Arc::new(Map::new()),
            native_supported: true,
        });
        let (_, memory_reservation) = ResourceGovernor::unconstrained_for_test()
            .reserve_memory_capacity(0)
            .await
            .unwrap();
        NativeRoute {
            store,
            codex_oauth: CodexOAuthManager::new(),
            persistence: Persistence::initialize(true).await.unwrap(),
            snapshot,
            api_key,
            providers: vec![provider.clone(), provider],
            base_payload: json!({"model":"gpt-public","input":"hello","stream":true}),
            request_headers: HeaderMap::new(),
            request_model: "gpt-public".into(),
            endpoint: "/v1/responses".into(),
            request_type: None,
            wants_compact: false,
            stream: true,
            request_id: "request-test".into(),
            request_body_bytes: 64,
            cursor: 0,
            max_attempts,
            hedging: HedgingConfig::default(),
            attempt_contexts: HashMap::new(),
            hedge_trigger_count: 0,
            hedge_cancelled_attempt_count: 0,
            last_provider: None,
            last_provider_key: None,
            last_original_model: None,
            last_attempt: None,
            last_status: 502,
            last_detail: String::new(),
            last_provider_model_unavailable: false,
            has_attempt_failure: false,
            last_failure_origin: String::new(),
            routing_attempts: 0,
            routing_skips: 0,
            upstream_attempts: 0,
            upstream_duration_ms: 0,
            routing_ledger: Vec::new(),
            upstream_ledger: Vec::new(),
            arrival: Some(crate::request_timing::RequestArrival::now()),
            started_at: tokio::time::Instant::now(),
            final_emitted: false,
            _memory_reservation: memory_reservation,
        }
    }

    async fn catalog_fixture() -> NativeConfigStore {
        let store = NativeConfigStore::new();
        let mut providers = Vec::new();
        for name in ["z-first", "a-second", "m-third", "excluded"] {
            let mut p = provider();
            p.name = name.into();
            p.models = Arc::new(HashMap::from([
                ("shared".into(), "actual-shared".into()),
                ("extra".into(), "actual-extra".into()),
                ("vendor/model".into(), "actual-slash".into()),
            ]));
            if name == "excluded" {
                p.excluded_endpoints = Arc::new(vec!["/v1/responses".into()]);
            }
            providers.push(Arc::new(p));
        }
        let mut keys = HashMap::new();
        for (token, role, rules) in [
            ("dashboard-first", "user", vec!["z-first/shared"]),
            (
                "restricted",
                "user",
                vec!["m-third/shared", "a-second/*", "m-third/shared"],
            ),
            ("parent", "user", vec!["restricted/shared", "z-first/extra"]),
            (
                "mixed",
                "user",
                vec!["m-third/extra", "shared", "<vendor/model>", "z-first/*"],
            ),
            ("admin-key", "admin", vec!["all"]),
        ] {
            keys.insert(
                token.to_owned(),
                Arc::new(ApiKey {
                    token: token.into(),
                    model_rules: Arc::new(vec!["all".into()]),
                    role: role.into(),
                    preferences: Arc::new(Map::from_iter([("__route_graph".into(), json!(rules))])),
                    weights: Arc::new(Map::new()),
                    native_supported: true,
                }),
            );
        }
        *store.current.write().await = Some(Arc::new(Snapshot {
            revision: "0".repeat(64).into(),
            preferences: Arc::new(Map::new()),
            api_keys: Arc::new(keys),
            api_key_order: Arc::new(
                [
                    "dashboard-first",
                    "restricted",
                    "parent",
                    "mixed",
                    "admin-key",
                ]
                .map(str::to_owned)
                .to_vec(),
            ),
            providers_by_name: Arc::new(
                providers
                    .iter()
                    .map(|p| (p.name.to_string(), p.clone()))
                    .collect(),
            ),
            providers: Arc::new(providers),
            api_config: Arc::new(json!({})),
        }));
        store
    }
    #[tokio::test]
    async fn temporary_channel_import_scopes_routing_and_resets_without_config_writes() {
        let store = catalog_fixture().await;
        let admin = catalog_headers("dashboard-first");
        let key = crate::channel_catalog::key_id("restricted");
        let original = store.current.read().await.clone().unwrap();
        let initial = store.controls_view(&admin).await.unwrap();
        let input = |revision: &Value, position| crate::channel_controls::ImportMutation {
            revision: revision.as_str().unwrap().into(),
            action: String::new(),
            api_key_id: key.clone(),
            provider: "sub2api-fixture".into(),
            base_url: "https://example.com/v1/responses".into(),
            api_key: "secret-import-key".into(),
            models: vec!["shared".into(), "new-model".into()],
            position,
        };
        assert!(store
            .import_temporary_channel(
                &catalog_headers("restricted"),
                input(&initial["revision"], 1)
            )
            .await
            .is_err());
        assert!(store
            .import_temporary_channel(&admin, input(&initial["revision"], 999))
            .await
            .is_err());
        let changed = store
            .import_temporary_channel(&admin, input(&initial["revision"], 1))
            .await
            .unwrap();
        assert!(!changed.to_string().contains("secret-import-key"));
        assert!(store
            .import_temporary_channel(&admin, input(&initial["revision"], 1))
            .await
            .is_err());
        let snapshot = store.snapshot().await.unwrap();
        assert!(Arc::ptr_eq(&snapshot, &store.snapshot().await.unwrap()));
        for token in ["restricted", "parent", "admin-key", "mixed"] {
            let api_key = &snapshot.api_keys[token];
            let providers =
                matching_providers(&snapshot, api_key, "shared", 0, None, None, "/v1/responses")
                    .unwrap();
            assert_eq!(
                providers
                    .iter()
                    .any(|p| p.name.as_ref() == "sub2api-fixture"),
                token == "restricted"
            );
            let ordered = store.schedule_providers(api_key, "shared", providers).await;
            if token == "restricted" {
                assert_eq!(ordered[0].name.as_ref(), "sub2api-fixture");
            }
            let rows = crate::channel_catalog::entries(
                &snapshot,
                &snapshot.api_keys["dashboard-first"],
                Some(&crate::channel_catalog::key_id(token)),
            )
            .unwrap();
            assert_eq!(
                rows.iter()
                    .any(|(p, _)| p.name.as_ref() == "sub2api-fixture"),
                token == "restricted"
            );
        }
        assert!(!store
            .models_for_headers(&catalog_headers("admin-key"))
            .await
            .unwrap()
            .contains(&"new-model".to_string()));
        assert!(store
            .models_for_headers(&catalog_headers("restricted"))
            .await
            .unwrap()
            .contains(&"new-model".to_string()));
        assert!(!original.providers_by_name.contains_key("sub2api-fixture"));
        assert!(Arc::ptr_eq(
            &original,
            &store.current.read().await.clone().unwrap()
        ));
        // Re-adding replaces this scoped provider, never creates a duplicate.
        let changed = store
            .import_temporary_channel(&admin, input(&changed["revision"], 1))
            .await
            .unwrap();
        assert_eq!(changed["temporary_channels"].as_array().unwrap().len(), 1);
        let reset = crate::channel_controls::Mutation {
            revision: changed["revision"].as_str().unwrap().into(),
            action: "reset".into(),
            api_key_id: key,
            model: "shared".into(),
            order: vec![],
            disabled: vec![],
        };
        let changed = store.mutate_controls(&admin, reset).await.unwrap();
        assert!(
            !store.snapshot().await.unwrap().providers_by_name["sub2api-fixture"]
                .models
                .contains_key("shared")
        );
        let reset = crate::channel_controls::Mutation {
            revision: changed["revision"].as_str().unwrap().into(),
            action: "reset_all".into(),
            api_key_id: "".into(),
            model: "".into(),
            order: vec![],
            disabled: vec![],
        };
        store.mutate_controls(&admin, reset).await.unwrap();
        assert!(!store
            .snapshot()
            .await
            .unwrap()
            .providers_by_name
            .contains_key("sub2api-fixture"));
    }
    #[tokio::test]
    async fn temporary_channel_management_preserves_other_routes_and_rejects_stale_edits() {
        let store = catalog_fixture().await;
        let admin = catalog_headers("dashboard-first");
        let key = crate::channel_catalog::key_id("restricted");
        let original = store.current.read().await.clone().unwrap();
        let mut view = store.controls_view(&admin).await.unwrap();
        let input = |view: &Value, provider: &str, action: &str, models: Vec<&str>| {
            crate::channel_controls::ImportMutation {
                revision: view["revision"].as_str().unwrap().into(),
                action: action.into(),
                api_key_id: key.clone(),
                provider: provider.into(),
                base_url: "https://example.com/v1/responses".into(),
                api_key: "fixture-secret".into(),
                models: models.into_iter().map(String::from).collect(),
                position: 1,
            }
        };
        for p in ["sub2api-one", "sub2api-two"] {
            view = store
                .import_temporary_channel(&admin, input(&view, p, "", vec!["shared", "new-model"]))
                .await
                .unwrap();
        }
        view = store
            .mutate_controls(
                &admin,
                crate::channel_controls::Mutation {
                    revision: view["revision"].as_str().unwrap().into(),
                    action: "set".into(),
                    api_key_id: key.clone(),
                    model: "shared".into(),
                    order: vec!["sub2api-two".into(), "sub2api-one".into()],
                    disabled: vec!["sub2api-two".into()],
                },
            )
            .await
            .unwrap();
        let before = view.clone();
        let mut wrong = input(&view, "sub2api-one", "delete", vec![]);
        wrong.api_key_id = crate::channel_catalog::key_id("admin-key");
        assert!(store.import_temporary_channel(&admin, wrong).await.is_err());
        assert_eq!(view, store.controls_view(&admin).await.unwrap());
        view = store
            .import_temporary_channel(
                &admin,
                input(&view, "sub2api-one", "replace", vec!["new-model"]),
            )
            .await
            .unwrap();
        let snapshot = store.snapshot().await.unwrap();
        assert!(!snapshot.providers_by_name["sub2api-one"]
            .models
            .contains_key("shared"));
        assert!(snapshot.providers_by_name["sub2api-two"]
            .models
            .contains_key("shared"));
        let rule = view["rules"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["model"] == "shared")
            .unwrap();
        assert_eq!(rule["order"], json!(["sub2api-two"]));
        assert_eq!(rule["disabled"], json!(["sub2api-two"]));
        let rejected = store
            .import_temporary_channel(&admin, input(&before, "sub2api-one", "delete", vec![]))
            .await
            .unwrap_err();
        assert_eq!(rejected.0, StatusCode::CONFLICT);
        view = store
            .import_temporary_channel(&admin, input(&view, "sub2api-one", "delete", vec![]))
            .await
            .unwrap();
        let snapshot = store.snapshot().await.unwrap();
        assert!(!snapshot.providers_by_name.contains_key("sub2api-one"));
        assert!(snapshot.providers_by_name.contains_key("sub2api-two"));
        assert!(!view["rules"].to_string().contains("sub2api-one"));
        assert!(!view.to_string().contains("fixture-secret"));
        assert!(Arc::ptr_eq(
            &original,
            &store.current.read().await.clone().unwrap()
        ));
        assert!(store
            .import_temporary_channel(
                &catalog_headers("restricted"),
                input(&view, "sub2api-two", "delete", vec![])
            )
            .await
            .is_err());
    }
    #[tokio::test]
    async fn retained_controls_replace_atomically_and_validate_before_serving() {
        use crate::channel_controls::{RestoreMutation, RetainedChannel, RetainedSnapshot, Rule};
        let store = catalog_fixture().await;
        let admin = catalog_headers("dashboard-first");
        let key = crate::channel_catalog::key_id("restricted");
        let snapshot = RetainedSnapshot {
            channel_settings: std::collections::BTreeMap::new(),
            version: 1,
            rules: vec![Rule {
                api_key_id: key.clone(),
                model: "shared".into(),
                order: vec!["sub2api-retained".into()],
                disabled: vec!["sub2api-retained".into()],
            }],
            temporary_channels: vec![RetainedChannel {
                provider: "sub2api-retained".into(),
                api_key_id: key,
                base_url: "https://example.com/v1/responses".into(),
                api_key: "restore-secret".into(),
                definition: None,
                models: vec!["shared".into()],
            }],
        };
        let initial = store.controls_view(&admin).await.unwrap();
        let restored = store
            .restore_controls(
                &admin,
                RestoreMutation {
                    revision: initial["revision"].as_str().unwrap().into(),
                    snapshot: snapshot.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(restored["temporary_channels"].as_array().unwrap().len(), 1);
        assert_eq!(
            restored["rules"][0]["disabled"],
            json!(["sub2api-retained"])
        );
        assert!(!restored.to_string().contains("restore-secret"));
        let before = restored.clone();
        let mut invalid = snapshot.clone();
        invalid.rules[0].order.push("missing".into());
        assert!(store
            .restore_controls(
                &admin,
                RestoreMutation {
                    revision: restored["revision"].as_str().unwrap().into(),
                    snapshot: invalid
                }
            )
            .await
            .is_err());
        assert_eq!(before, store.controls_view(&admin).await.unwrap());
        assert!(store
            .restore_controls(
                &admin,
                RestoreMutation {
                    revision: initial["revision"].as_str().unwrap().into(),
                    snapshot: snapshot.clone()
                }
            )
            .await
            .is_err());
        assert!(store
            .restore_controls(
                &catalog_headers("restricted"),
                RestoreMutation {
                    revision: restored["revision"].as_str().unwrap().into(),
                    snapshot: snapshot.clone()
                }
            )
            .await
            .is_err());
        let fresh = catalog_fixture().await;
        let view = fresh.controls_view(&admin).await.unwrap();
        let after = fresh
            .restore_controls(
                &admin,
                RestoreMutation {
                    revision: view["revision"].as_str().unwrap().into(),
                    snapshot,
                },
            )
            .await
            .unwrap();
        assert_eq!(after["rules"], restored["rules"]);
        assert_eq!(after["temporary_channels"], restored["temporary_channels"]);
        assert_ne!(after["instance_id"], restored["instance_id"]);
    }
    fn catalog_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        headers
    }
    fn catalog_pairs(rows: &[Value]) -> Vec<(&str, &str)> {
        rows.iter()
            .map(|r| {
                (
                    r["provider"].as_str().unwrap(),
                    r["model"].as_str().unwrap(),
                )
            })
            .collect()
    }
    #[tokio::test]
    async fn temporary_controls_are_scoped_reversible_and_revision_guarded() {
        use crate::channel_controls::Mutation;
        let store = catalog_fixture().await;
        let snapshot = store.snapshot().await.unwrap();
        let headers = catalog_headers("dashboard-first");
        assert_eq!(
            store
                .controls_view(&catalog_headers("restricted"))
                .await
                .unwrap_err(),
            403
        );
        let initial = store.controls_view(&headers).await.unwrap();
        let input = |revision: &Value,
                     action: &str,
                     key: &str,
                     model: &str,
                     order: Vec<&str>,
                     disabled: Vec<&str>| {
            serde_json::from_value::<Mutation>(json!({"revision":revision,"action":action,"api_key_id":key,"model":model,"order":order,"disabled":disabled})).unwrap()
        };
        assert!(store
            .mutate_controls(
                &catalog_headers("restricted"),
                input(&initial["revision"], "set", "", "", vec![], vec!["z-first"])
            )
            .await
            .is_err());
        assert!(store
            .mutate_controls(
                &headers,
                input(&initial["revision"], "set", "", "", vec!["missing"], vec![])
            )
            .await
            .is_err());
        let changed = store
            .mutate_controls(
                &headers,
                input(
                    &initial["revision"],
                    "set",
                    "",
                    "",
                    vec!["m-third", "a-second", "z-first"],
                    vec!["z-first"],
                ),
            )
            .await
            .unwrap();
        assert!(store
            .mutate_controls(
                &headers,
                input(&initial["revision"], "reset", "", "", vec![], vec![])
            )
            .await
            .is_err());
        let names = |providers: Vec<Arc<Provider>>| {
            providers
                .into_iter()
                .map(|p| p.name.to_string())
                .collect::<Vec<_>>()
        };
        let available = vec![
            snapshot.providers_by_name["z-first"].clone(),
            snapshot.providers_by_name["a-second"].clone(),
            snapshot.providers_by_name["m-third"].clone(),
        ];
        assert_eq!(
            names(
                store
                    .schedule_providers(
                        &snapshot.api_keys["admin-key"],
                        "shared",
                        available.clone()
                    )
                    .await
            ),
            vec!["m-third", "a-second"]
        );
        // No allow-list expansion: configured matches remain the sole candidates.
        assert!(store
            .schedule_providers(
                &snapshot.api_keys["dashboard-first"],
                "shared",
                vec![snapshot.providers_by_name["z-first"].clone()]
            )
            .await
            .is_empty());
        let id = crate::channel_catalog::key_id("restricted");
        let scoped = store
            .mutate_controls(
                &headers,
                input(
                    &changed["revision"],
                    "set",
                    &id,
                    "shared",
                    vec!["a-second", "m-third"],
                    vec!["m-third"],
                ),
            )
            .await
            .unwrap();
        assert_eq!(
            names(
                store
                    .schedule_providers(
                        &snapshot.api_keys["restricted"],
                        "shared",
                        available.clone()
                    )
                    .await
            ),
            vec!["a-second"]
        );
        assert_eq!(
            names(
                store
                    .schedule_providers(
                        &snapshot.api_keys["restricted"],
                        "extra",
                        available.clone()
                    )
                    .await
            ),
            vec!["m-third", "a-second"]
        );
        assert_eq!(
            names(
                store
                    .schedule_providers(
                        &snapshot.api_keys["admin-key"],
                        "shared",
                        available.clone()
                    )
                    .await
            ),
            vec!["m-third", "a-second"]
        );
        let (rows, _, _) = store
            .channel_catalog(&headers, "/v1/responses", false, Some(&id))
            .await
            .unwrap();
        assert_eq!(
            rows.iter()
                .find(|r| r["provider"] == "m-third" && r["model"] == "shared")
                .unwrap()["reason"],
            "temporarily_disabled"
        );
        let reset = store
            .mutate_controls(
                &headers,
                input(&scoped["revision"], "reset", &id, "shared", vec![], vec![]),
            )
            .await
            .unwrap();
        assert_eq!(reset["rules"].as_array().unwrap().len(), 1);
        let cleared = store
            .mutate_controls(
                &headers,
                input(&reset["revision"], "reset_all", "", "", vec![], vec![]),
            )
            .await
            .unwrap();
        assert!(cleared["rules"].as_array().unwrap().is_empty());
        assert_eq!(
            names(
                store
                    .schedule_providers(&snapshot.api_keys["admin-key"], "shared", available)
                    .await
            ),
            vec!["z-first", "a-second", "m-third"]
        );
        let restarted = catalog_fixture().await;
        let fresh = restarted.controls_view(&headers).await.unwrap();
        assert_ne!(initial["instance_id"], fresh["instance_id"]);
        assert!(fresh["rules"].as_array().unwrap().is_empty());
        assert!(restarted
            .mutate_controls(
                &headers,
                input(&cleared["revision"], "set", "", "", vec![], vec!["z-first"])
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn diagnostic_routing_is_admin_only_and_never_falls_back() {
        let store = catalog_fixture().await;
        let snapshot = store.snapshot().await.unwrap();
        for token in ["dashboard-first", "admin-key"] {
            let key = &snapshot.api_keys[token];
            let mut headers = catalog_headers(token);
            headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("m-third"));
            let targeted = diagnostic_key(&snapshot, key, &headers, "/v1/responses")
                .ok()
                .unwrap();
            let providers = matching_providers(
                &snapshot,
                &targeted,
                "shared",
                100,
                None,
                None,
                "/v1/responses",
            )
            .unwrap();
            assert_eq!(
                providers
                    .iter()
                    .map(|p| p.name.as_ref())
                    .collect::<Vec<_>>(),
                vec!["m-third"]
            );
            assert!(matching_providers(
                &snapshot,
                &targeted,
                "missing",
                100,
                None,
                None,
                "/v1/responses"
            )
            .unwrap()
            .is_empty());
            assert!(diagnostic_key(&snapshot, key, &headers, "/v1/chat/completions").is_err());
            headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("missing"));
            assert!(diagnostic_key(&snapshot, key, &headers, "/v1/responses").is_err());
            headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("excluded"));
            let excluded = diagnostic_key(&snapshot, key, &headers, "/v1/responses")
                .ok()
                .unwrap();
            assert!(matching_providers(
                &snapshot,
                &excluded,
                "shared",
                100,
                None,
                None,
                "/v1/responses"
            )
            .unwrap()
            .is_empty());
        }
        let mut headers = catalog_headers("restricted");
        headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("m-third"));
        assert!(diagnostic_key(
            &snapshot,
            &snapshot.api_keys["restricted"],
            &headers,
            "/v1/responses"
        )
        .is_err());
        assert_eq!(
            snapshot.api_keys["dashboard-first"].preferences["__route_graph"],
            json!(["z-first/shared"])
        );
    }

    #[tokio::test]
    async fn fixed_priority_preserves_key_graph_order_when_deduplicating_matches() {
        let store = catalog_fixture().await;
        let snapshot = store.snapshot().await.unwrap();
        for (token, model, endpoint, expected) in [
            (
                "restricted",
                "shared",
                "/v1/messages",
                vec!["m-third", "a-second"],
            ),
            (
                "parent",
                "shared",
                "/v1/messages",
                vec!["m-third", "a-second"],
            ),
            ("parent", "extra", "/v1/messages", vec!["z-first"]),
            (
                "mixed",
                "shared",
                "/v1/responses",
                vec!["z-first", "a-second", "m-third"],
            ),
            (
                "admin-key",
                "shared",
                "/v1/messages",
                vec!["z-first", "a-second", "m-third", "excluded"],
            ),
        ] {
            let key = snapshot.api_keys.get(token).unwrap();
            let matched =
                matching_providers(&snapshot, key, model, 0, None, None, endpoint).unwrap();
            let scheduled = store.schedule_providers(key, model, matched).await;
            assert_eq!(
                scheduled
                    .iter()
                    .map(|p| p.name.as_ref())
                    .collect::<Vec<_>>(),
                expected,
                "key={token} model={model}"
            );
        }
    }
    #[tokio::test]
    async fn catalog_default_keeps_provider_order_and_does_not_change_routing_state() {
        let store = catalog_fixture().await;
        let headers = catalog_headers("dashboard-first");
        let (rows, _, selection) = store
            .channel_catalog(&headers, "/v1/responses", true, None)
            .await
            .unwrap();
        assert!(selection.is_empty());
        assert_eq!(
            catalog_pairs(&rows),
            vec![
                ("z-first", "extra"),
                ("z-first", "shared"),
                ("z-first", "vendor/model"),
                ("a-second", "extra"),
                ("a-second", "shared"),
                ("a-second", "vendor/model"),
                ("m-third", "extra"),
                ("m-third", "shared"),
                ("m-third", "vendor/model"),
            ]
        );
        assert!(store.client_windows.lock().await.is_empty());
        assert!(store.provider_windows.lock().await.is_empty());
        assert!(store.routing_cursors.lock().await.is_empty());
        for p in store.snapshot().await.unwrap().providers.iter() {
            assert_eq!(p.cursor.load(Ordering::Relaxed), 0);
        }
    }
    #[tokio::test]
    async fn catalog_selected_key_preserves_rules_filters_and_nested_model_restriction() {
        let store = catalog_fixture().await;
        let headers = catalog_headers("dashboard-first");
        let id = crate::channel_catalog::key_id("restricted");
        let (rows, _, selected) = store
            .channel_catalog(&headers, "/v1/responses", true, Some(&id))
            .await
            .unwrap();
        assert_eq!(id, selected);
        assert_eq!(
            catalog_pairs(&rows),
            vec![
                ("m-third", "shared"),
                ("a-second", "extra"),
                ("a-second", "shared"),
                ("a-second", "vendor/model")
            ]
        );
        let id = crate::channel_catalog::key_id("parent");
        let (rows, _, _) = store
            .channel_catalog(&headers, "/v1/responses", true, Some(&id))
            .await
            .unwrap();
        assert_eq!(
            catalog_pairs(&rows),
            vec![
                ("m-third", "shared"),
                ("a-second", "shared"),
                ("z-first", "extra")
            ]
        );
        let id = crate::channel_catalog::key_id("mixed");
        let (rows, _, _) = store
            .channel_catalog(&headers, "/v1/responses", true, Some(&id))
            .await
            .unwrap();
        assert_eq!(
            catalog_pairs(&rows),
            vec![
                ("m-third", "extra"),
                ("z-first", "shared"),
                ("a-second", "shared"),
                ("m-third", "shared"),
                ("z-first", "vendor/model"),
                ("a-second", "vendor/model"),
                ("m-third", "vendor/model"),
                ("z-first", "extra"),
            ]
        );
    }
    #[tokio::test]
    async fn catalog_key_selection_is_redacted_and_cannot_escalate_access() {
        let store = catalog_fixture().await;
        for token in ["restricted", "parent", "mixed", "invalid"] {
            let headers = catalog_headers(token);
            assert_eq!(store.api_key_catalog(&headers).await.unwrap_err(), 403);
            assert_eq!(store.authorize_catalog(&headers).await.unwrap_err(), 403);
            assert!(matches!(
                store.balance_provider(&headers, "z-first").await,
                Err(403)
            ));
            for selected in [
                None,
                Some(crate::channel_catalog::key_id(token)),
                Some(crate::channel_catalog::key_id("dashboard-first")),
            ] {
                assert_eq!(
                    store
                        .channel_catalog(&headers, "/v1/responses", true, selected.as_deref())
                        .await
                        .unwrap_err(),
                    403
                );
            }
            if token != "invalid" {
                assert!(store.models_for_headers(&headers).await.is_ok());
            }
        }
        for token in ["dashboard-first", "admin-key"] {
            let headers = catalog_headers(token);
            assert_eq!(
                store
                    .balance_provider(&headers, "z-first")
                    .await
                    .unwrap()
                    .0
                    .name
                    .as_ref(),
                "z-first"
            );
            assert!(matches!(
                store.balance_provider(&headers, "unknown").await,
                Err(404)
            ));
            let listing = store.api_key_catalog(&headers).await.unwrap();
            assert_eq!(listing["data"].as_array().unwrap().len(), 5);
            assert_eq!(listing["can_inspect_all"], true);
            assert!(!listing.to_string().contains("restricted"));
            assert!(store.authorize_catalog(&headers).await.is_ok());
            assert_eq!(
                store
                    .channel_catalog(&headers, "/v1/responses", true, Some("stale-id"))
                    .await
                    .unwrap_err(),
                404
            );
        }
    }

    #[tokio::test]
    async fn catalog_metrics_keep_selected_order_and_channel_wide_statistics() {
        let store = catalog_fixture().await;
        let headers = catalog_headers("dashboard-first");
        let id = crate::channel_catalog::key_id("parent");
        let (rows, revision, _) = store
            .channel_catalog(&headers, "/v1/responses", true, Some(&id))
            .await
            .unwrap();
        let metrics = crate::channel_metrics::ChannelMetrics::new();
        metrics.start("m-third", "shared", "actual-shared", "/v1/responses", true);
        metrics.finish(
            "m-third",
            "shared",
            "actual-shared",
            "/v1/responses",
            true,
            "success",
            Some(1000.),
            Some(200.),
        );
        for timeseries in [false, true] {
            let response = metrics.query(rows.clone(), &revision, 15, timeseries);
            assert_eq!(
                catalog_pairs(response["data"].as_array().unwrap()),
                catalog_pairs(&rows)
            );
            assert_eq!(response["data"][0]["stats"]["success"], 1);
            assert_eq!(
                response["data"][0]["stats"]["first_output"]["last_ms"],
                200.
            );
        }
    }

    #[test]
    fn payload_compiler_matches_codex_contract_without_double_json_envelope() {
        let mut provider = provider();
        provider.preferences = Arc::new(Map::new());
        let mut payload = json!({
            "model": "gpt-public",
            "input": [{"type":"reasoning","id":"rs_1","cache_control":{}}],
            "stream": true,
            "temperature": 1,
            "response_format": {"type": "json_object"},
            "store": true,
            "max_output_tokens": 42,
            "previous_response_id": "resp_1"
        });
        compile_payload(
            &mut payload,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
            false,
        )
        .unwrap();
        assert_eq!(payload["model"], "gpt-upstream");
        assert_eq!(payload["store"], false);
        assert_eq!(payload["instructions"], "");
        assert!(payload.get("temperature").is_none());
        assert!(payload.get("response_format").is_none());
        assert!(payload.get("max_output_tokens").is_none());
        assert!(payload.get("previous_response_id").is_none());
        assert!(payload["input"][0].get("id").is_none());
        assert!(payload["input"][0].get("cache_control").is_none());
    }

    #[test]
    fn codex_defaults_follow_overrides_without_mutating_provider_or_other_engines() {
        for engine in ["codex", " CODEX ", "gpt"] {
            let mut provider = provider();
            provider.engine = Arc::from(engine);
            let preferences = Map::from_iter([(
                "post_body_parameter_overrides".into(),
                json!({
                    "store": true,
                    "temperature": 0.5,
                    "response_format": {"type": "json_object"},
                    "__remove__": ["metadata.remove"],
                    "gpt-public": {
                        "__remove__": ["store"],
                        "temperature": 0.7,
                        "response_format": {"type": "text"},
                        "metadata": {"model_specific": true}
                    }
                }),
            )]);
            provider.preferences = Arc::new(preferences.clone());
            let mut payload = json!({"metadata": {"remove": true, "keep": true}});
            apply_overrides(payload.as_object_mut().unwrap(), &provider, "gpt-public");
            assert_eq!(
                payload["metadata"],
                json!({"keep": true, "model_specific": true})
            );
            if engine == "gpt" {
                assert!(payload.get("store").is_none());
                assert_eq!(payload["temperature"], 0.7);
                assert_eq!(payload["response_format"], json!({"type": "text"}));
            } else {
                assert_eq!(payload["store"], false);
                assert!(payload.get("temperature").is_none());
                assert!(payload.get("response_format").is_none());
            }
            assert_eq!(*provider.preferences, preferences);
        }
    }

    #[test]
    fn codex_compact_omits_store_after_applying_defaults() {
        let mut provider = provider();
        provider.preferences = Arc::new(Map::new());
        let mut payload = json!({
            "model": "gpt-public", "input": "hello", "store": true,
            "temperature": 1, "response_format": {"type": "json_object"}
        });
        compile_payload(
            &mut payload,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
            true,
        )
        .unwrap();
        for field in ["store", "temperature", "response_format"] {
            assert!(payload.get(field).is_none(), "unexpected field {field}");
        }
    }

    #[test]
    fn hedging_preferences_parse_with_safe_defaults() {
        let config = parse_hedging(&Map::from_iter([(
            "hedging".into(),
            json!({
                "enabled": true,
                "max_inflight_attempts": 2,
                "winner_policy": "first_valid_success"
            }),
        )]));
        assert!(config.enabled);
        assert_eq!(config.max_inflight_attempts, 2);
        assert_eq!(
            config.winner_policy,
            crate::hedging::WinnerPolicy::FirstValidSuccess
        );

        let invalid = parse_hedging(&Map::from_iter([(
            "hedging".into(),
            json!({"enabled": true, "winner_policy": "unknown"}),
        )]));
        assert_eq!(invalid, HedgingConfig::default());
    }

    #[test]
    fn timeout_policy_matches_compaction_request_type_without_affecting_regular_requests() {
        let provider = provider();
        let snapshot = Snapshot {
            revision: Arc::from("0".repeat(64)),
            preferences: Arc::new(Map::from_iter([
                ("model_timeout".into(), json!({"gpt-public": 20})),
                (
                    "timeout_policy".into(),
                    json!({
                        "rules": [{
                            "match": {
                                "endpoint": "/v1/responses",
                                "stream": true,
                                "request_type": "compaction",
                                "engine": "codex",
                                "model": ["gpt-5.6*", "gpt-public", "gpt-5.4*"]
                            },
                            "timeout": {"first_byte": 300, "total": 3000}
                        }]
                    }),
                ),
            ])),
            api_keys: Arc::new(HashMap::new()),
            api_key_order: Arc::new(Vec::new()),
            providers: Arc::new(Vec::new()),
            providers_by_name: Arc::new(HashMap::new()),
            api_config: Arc::new(json!({})),
        };

        let compaction = resolve_timeouts(
            &snapshot,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
            true,
            Some("compaction"),
            "user",
            "/v1/responses",
            "POST",
        );
        let regular = resolve_timeouts(
            &snapshot,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
            true,
            None,
            "user",
            "/v1/responses",
            "POST",
        );

        assert_eq!(compaction.first_byte, Some(300.0));
        assert_eq!(compaction.total, Some(3000.0));
        assert_eq!(regular.first_byte, Some(20.0));
        assert_eq!(regular.total, None);
    }

    #[test]
    fn keepalive_matches_request_then_upstream_then_provider_default_then_global() {
        let mut provider = provider();
        provider.preferences = Arc::new(Map::from_iter([(
            "keepalive_interval".into(),
            json!({
                "PUBLIC-MODEL": 1, "public": 2, "upstream": 3, "default": 4
            }),
        )]));
        let global = Map::from_iter([(
            "keepalive_interval".into(),
            json!({"global": 5, "default": 6}),
        )]);
        let resolve = |p: &Provider, model, upstream| {
            model_preference(p, &global, model, upstream, "keepalive_interval")
        };
        assert_eq!(resolve(&provider, "public-model", "upstream"), Some(1.0));
        assert_eq!(resolve(&provider, "public-other", "upstream"), Some(2.0));
        assert_eq!(resolve(&provider, "alias", "upstream-v2"), Some(3.0));
        assert_eq!(resolve(&provider, "global", "unknown"), Some(4.0));
        provider.preferences = Arc::new(Map::from_iter([(
            "keepalive_interval".into(),
            json!({"local": 7}),
        )]));
        assert_eq!(resolve(&provider, "global-v2", "unknown"), Some(5.0));
        assert_eq!(resolve(&provider, "unknown", "unknown"), Some(6.0));
        provider.preferences = Arc::new(Map::from_iter([("keepalive_interval".into(), json!(0))]));
        assert_eq!(resolve(&provider, "global", "unknown"), Some(0.0));
    }

    #[test]
    fn model_timeout_matches_legacy_fuzzy_and_fallback_order() {
        let mut provider = provider();
        provider.preferences = Arc::new(Map::from_iter([(
            "model_timeout".into(),
            json!({
                "GPT-5.6-SOL": 15,
                "gpt-5.6": 20,
                "upstream-special": 40,
                "default": 90
            }),
        )]));
        let global = Map::from_iter([(
            "model_timeout".into(),
            json!({"gpt-5.6": 20, "global-only": 30, "default": 2000}),
        )]);

        assert_eq!(
            model_timeout(&provider, &global, "gpt-5.6-sol", "gpt-5.6-sol"),
            15.0
        );
        assert_eq!(
            model_timeout(&provider, &global, "public-alias", "upstream-special-v2"),
            40.0
        );
        assert_eq!(
            model_timeout(&provider, &global, "unknown", "unknown-upstream"),
            90.0
        );

        provider.preferences = Arc::new(Map::from_iter([(
            "model_timeout".into(),
            json!({"provider-only": 50}),
        )]));
        assert_eq!(
            model_timeout(&provider, &global, "gpt-5.6-terra", "gpt-5.6-terra"),
            20.0
        );
        assert_eq!(
            model_timeout(&provider, &global, "global-only-v2", "other"),
            30.0
        );

        let resolved = resolve_timeouts(
            &Snapshot {
                revision: Arc::from("0".repeat(64)),
                preferences: Arc::new(global),
                api_keys: Arc::new(HashMap::new()),
                api_key_order: Arc::new(Vec::new()),
                providers: Arc::new(Vec::new()),
                providers_by_name: Arc::new(HashMap::new()),
                api_config: Arc::new(json!({})),
            },
            &provider,
            "gpt-5.6-sol",
            "gpt-5.6-sol",
            "codex",
            true,
            None,
            "user",
            "/v1/responses",
            "POST",
        );
        assert_eq!(resolved.first_byte, Some(20.0));
        assert_eq!(resolved.idle, None);
        assert_eq!(resolved.total, None);
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
    fn shared_python_rust_contract_fixture_covers_routes_and_provider_fields() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../test/runtime_contracts.fixture"
        ))
        .unwrap();
        assert!(fixture["routes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "/v1/video/tasks"));
        assert!(fixture["provider_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "tools"));
        assert_eq!(fixture["nested_key"]["rule"], "child-key/*");
    }

    #[test]
    fn tpr_rate_limits_are_preserved_as_request_token_rules() {
        let rules = parse_rate_limits(Some(&json!("2/tpr, 10/min")), Some("m")).unwrap();
        assert!(rules.contains(&(2, 0)));
        assert!(rules.contains(&(10, 60)));
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

    #[tokio::test]
    async fn weighted_round_robin_matches_legacy_sequence_and_rotates_requests() {
        let providers = vec![named_provider("a"), named_provider("b")];
        let weights = Map::from_iter([
            ("a/gpt-public".into(), json!(3)),
            ("b/gpt-public".into(), json!(1)),
        ]);
        let sequence =
            weighted_provider_sequence(&providers, "gpt-public", &weights, "weighted_round_robin");
        assert_eq!(
            sequence
                .iter()
                .map(|provider| provider.name.as_ref())
                .collect::<Vec<_>>(),
            vec!["a", "b", "a", "a"]
        );

        let api_key = ApiKey {
            token: "client".into(),
            model_rules: Arc::new(vec!["all".into()]),
            role: "client".into(),
            preferences: Arc::new(Map::from_iter([(
                "SCHEDULING_ALGORITHM".into(),
                json!("weighted_round_robin"),
            )])),
            weights: Arc::new(weights),
            native_supported: true,
        };
        let store = NativeConfigStore::new();
        let first = store
            .schedule_providers(&api_key, "gpt-public", providers.clone())
            .await;
        let second = store
            .schedule_providers(&api_key, "gpt-public", providers)
            .await;
        assert_eq!(first[0].name.as_ref(), "a");
        assert_eq!(second[0].name.as_ref(), "b");
    }

    #[test]
    fn request_type_detection_and_provider_filters_match_python_semantics() {
        let regular = json!({"input": [{"type":"message"}]});
        let compact = json!({
            "input": [
                {"type":"message"},
                {"type":"compaction_trigger"}
            ]
        });
        assert_eq!(detect_request_type(regular.as_object().unwrap()), None);
        assert_eq!(
            detect_request_type(compact.as_object().unwrap()),
            Some("compaction")
        );

        let mut provider = provider();
        provider.only_request_types = Arc::new(vec!["compaction".into()]);
        assert!(!provider_accepts_request_type(&provider, None));
        assert!(provider_accepts_request_type(&provider, Some("compaction")));

        provider.excluded_request_types = Arc::new(vec!["compaction".into()]);
        assert!(!provider_accepts_request_type(
            &provider,
            Some("compaction")
        ));
    }

    #[test]
    fn provider_request_rules_match_public_upstream_model_and_reasoning_effort() {
        let mut provider = provider();
        provider.excluded_request_rules = Arc::new(vec![json!({
            "match": {
                "endpoint": "/v1/responses",
                "request_model": "gpt-public",
                "upstream_model": "gpt-upstream",
                "reasoning_effort": ["MAX"]
            },
            "reason": "unsupported_reasoning_effort"
        })]);

        assert!(!provider_accepts_request_rules(
            &provider,
            "/v1/responses/",
            "gpt-public",
            Some("max"),
            None,
        ));
        assert!(provider_accepts_request_rules(
            &provider,
            "/v1/responses",
            "gpt-public",
            Some("high"),
            None,
        ));
        assert!(provider_accepts_request_rules(
            &provider,
            "/v1/responses",
            "gpt-public",
            None,
            None,
        ));
        assert!(provider_accepts_request_rules(
            &provider,
            "/v1/chat/completions",
            "gpt-public",
            Some("max"),
            None,
        ));
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

    #[test]
    fn unavailable_model_message_is_a_retryable_channel_failure() {
        let body = r#"{"error":{"message":"This model is not available.","type":"invalid_request_error"}}"#;
        for detail in [
            body.to_owned(),
            json!({"error": {"message": body}}).to_string(),
            json!({"detail": {"message": "  THIS MODEL IS NOT AVAILABLE.  "}}).to_string(),
            "This model is not available.".to_owned(),
        ] {
            for endpoint in [
                "/v1/responses",
                "/v1/responses/compact",
                "/v1/chat/completions",
            ] {
                let policy = classify_provider_failure(400, &detail, None, endpoint, true);
                assert_eq!(policy.status, 503, "{detail}");
                assert!(policy.retryable);
                assert!(!policy.request_scoped);
                assert!(policy.provider_model_unavailable);
                assert!(!policy.force_quota_cooldown);
                let disabled = classify_provider_failure(400, &detail, None, endpoint, false);
                assert_eq!(disabled.status, 503);
                assert!(!disabled.retryable);
            }
        }
        for detail in [
            r#"{"error":{"type":"invalid_request_error","message":"Missing required parameter: input"}}"#,
            r#"{"error":{"message":"Invalid input"},"input":"This model is not available."}"#,
            r#"{"error":{"message":"Invalid input"},"debug":{"message":"This model is not available."}}"#,
            r#"{"error":{"message":"Invalid input: expected 'This model is not available.'"}}"#,
        ] {
            let policy = classify_provider_failure(400, detail, None, "/v1/responses", true);
            assert_eq!(policy.status, 400, "{detail}");
            assert!(policy.request_scoped);
            assert!(!policy.retryable);
            assert!(!policy.provider_model_unavailable);
        }
        assert_eq!(remap_provider_status(413, body), 413);
    }

    #[tokio::test]
    async fn unavailable_model_retries_next_channel_and_preserves_upstream_status() {
        let mut first = provider();
        first.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
        let mut route = native_route_for_test(Arc::new(first), 3).await;
        route.providers.insert(1, named_provider("fallback"));
        let first_plan = route.next_plan().await.unwrap().unwrap();
        assert!(route
            .record_failure_for(&first_plan, &json!({
                "kind": "http_error",
                "status_code": 400,
                "body": r#"{"error":{"message":"This model is not available.","type":"invalid_request_error"}}"#,
            }))
            .await);
        assert_eq!(route.last_status(), 503);
        assert_eq!(route.upstream_ledger[0]["status_code"], 400);
        assert_eq!(route.upstream_ledger[0]["provider_model_unavailable"], true);
        assert_eq!(route.routing_ledger[0]["status_code"], 503);
        let fallback = route.next_plan().await.unwrap().unwrap();
        assert_eq!(fallback.provider_name.as_deref(), Some("fallback"));
        // The failed provider/model is cooling, so a later turn skips it.
        assert!(route.next_plan().await.unwrap().is_none());
        assert_eq!(route.routing_skips, 1);
        assert_eq!(route.last_status(), 503);
    }

    #[test]
    fn minimum_input_key_restrictions_are_retryable_gateway_errors() {
        let chinese = "该令牌不接受输入少于 2000 token 的请求(按请求体大小判定)。";
        let english = "This key does not accept requests with fewer than 2000 input tokens (judged by request body size).";
        for message in [
            format!("{chinese}{english}"),
            chinese.into(),
            english.into(),
            english.replace("2000", "8192").to_uppercase(),
        ] {
            let body =
                json!({"error": {"type": "invalid_request_error", "message": message}}).to_string();
            for detail in [
                message,
                body.clone(),
                json!({"error": {"message": body}}).to_string(),
            ] {
                for endpoint in [
                    "/v1/responses",
                    "/v1/responses/compact",
                    "/v1/chat/completions",
                ] {
                    let policy = classify_provider_failure(400, &detail, None, endpoint, true);
                    assert_eq!(policy.status, 502, "{detail}");
                    assert!(policy.retryable);
                    assert!(!policy.request_scoped);
                    assert!(!policy.provider_model_unavailable);
                    assert!(!policy.force_quota_cooldown);
                    assert!(
                        !classify_provider_failure(400, &detail, None, endpoint, false).retryable
                    );
                }
            }
        }
        let escaped = r#"{"error":{"message":"\u8be5\u4ee4\u724c\u4e0d\u63a5\u53d7\u8f93\u5165\u5c11\u4e8e 4096 token \u7684\u8bf7\u6c42"}}"#;
        assert_eq!(remap_provider_status(400, escaped), 502);
        assert_eq!(remap_provider_status(413, escaped), 413);
    }

    #[test]
    fn minimum_input_detection_preserves_client_validation_errors() {
        for detail in [
            r#"{"error":{"type":"invalid_request_error","message":"Missing required parameter: input"}}"#,
            "Input must contain at least 1 token.",
            "This model requires at least 2000 input tokens.",
            "This key does not accept requests with fewer than two input tokens.",
            "This key does not accept requests with fewer than 2000 output tokens.",
            "该令牌不接受输入少于 token 的请求",
            r#"{"error":{"message":"Invalid input"},"input":"该令牌不接受输入少于 2000 token 的请求"}"#,
            r#"{"error":{"message":"Invalid input"},"debug":{"message":"This key does not accept requests with fewer than 2000 input tokens"}}"#,
        ] {
            let policy = classify_provider_failure(400, detail, None, "/v1/responses", true);
            assert_eq!(policy.status, 400, "{detail}");
            assert!(policy.request_scoped);
            assert!(!policy.retryable);
        }
    }

    #[test]
    fn provider_failure_policy_matches_python_retry_matrix() {
        let base_provider = provider();
        for status in [401, 402, 403, 404, 409, 422, 429, 500, 502, 503, 504] {
            let policy = classify_provider_failure(
                status,
                "upstream failure",
                Some(&base_provider),
                "/v1/chat/completions",
                true,
            );
            assert!(policy.retryable, "status {status} should fail over");
            assert!(
                !policy.request_scoped,
                "status {status} should not be request scoped"
            );
        }

        assert!(
            !classify_provider_failure(
                400,
                "invalid request",
                Some(&base_provider),
                "/v1/chat/completions",
                true
            )
            .retryable
        );
        assert!(
            !classify_provider_failure(
                413,
                "payload too large",
                Some(&base_provider),
                "/v1/chat/completions",
                true
            )
            .retryable
        );
        assert!(
            !classify_provider_failure(
                403,
                "upstream failure",
                Some(&base_provider),
                "/v1/chat/completions",
                false
            )
            .retryable
        );
        assert!(!classify_provider_failure(
            404,
            "{\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Item with id 'rs_1' not found. Items are not persisted when store is false.\"}}",
            Some(&base_provider),
            "/v1/responses",
            true,
        ).retryable);

        let pricing = classify_provider_failure(
            400,
            r#"{"error":{"code":"model_not_priced","message":"missing"}}"#,
            Some(&base_provider),
            "/v1/chat/completions",
            true,
        );
        assert_eq!(pricing.status, 502);
        assert!(pricing.retryable);

        let model_unavailable = classify_provider_failure(
            400,
            r#"{"error":{"code":"model_not_found","message":"unknown provider for model gpt-5.6-sol"}}"#,
            Some(&base_provider),
            "/v1/responses",
            true,
        );
        assert_eq!(model_unavailable.status, 503);
        assert!(model_unavailable.retryable);
        assert!(!model_unavailable.request_scoped);
        assert!(model_unavailable.provider_model_unavailable);

        let wrapped_model_unavailable = classify_provider_failure(
            400,
            r#"{"error":{"message":"{\"error\":{\"code\":\"model_not_found\",\"message\":\"unknown provider for model gpt-5.6-sol\"}}"}}"#,
            Some(&base_provider),
            "/v1/responses",
            true,
        );
        assert_eq!(wrapped_model_unavailable.status, 503);
        assert!(wrapped_model_unavailable.retryable);

        let ordinary_bad_request = classify_provider_failure(
            400,
            r#"{"error":{"code":"invalid_type","message":"messages: field required"}}"#,
            Some(&base_provider),
            "/v1/responses",
            true,
        );
        assert_eq!(ordinary_bad_request.status, 400);
        assert!(!ordinary_bad_request.retryable);
        assert!(!ordinary_bad_request.provider_model_unavailable);

        let codex = classify_provider_failure(
            400,
            "model is not supported when using codex with a ChatGPT account",
            Some(&base_provider),
            "/v1/responses",
            true,
        );
        assert!(codex.retryable);
        assert!(codex.force_quota_cooldown);
        assert!(
            !classify_provider_failure(
                400,
                "model is not supported when using codex with a ChatGPT account",
                Some(&base_provider),
                "/v1/chat/completions",
                true,
            )
            .retryable
        );

        let mut azure = provider();
        azure.engine = Arc::from("gpt");
        azure.base_url = Arc::from("https://models.inference.ai.azure.com");
        assert!(
            classify_provider_failure(
                400,
                "provider validation",
                Some(&azure),
                "/v1/chat/completions",
                true
            )
            .retryable
        );
        assert!(
            classify_provider_failure(
                413,
                "provider validation",
                Some(&azure),
                "/v1/chat/completions",
                true
            )
            .retryable
        );
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
        assert_eq!(route.response_detail(), "no available token");
        assert_eq!(route.routing_attempts, 2);
        assert_eq!(route.routing_skips, 1);
        assert_eq!(route.upstream_attempts, 1);
        let final_message = route.final_message();
        assert_eq!(final_message["status_code"], 503);
        assert!(route.final_emitted);
    }
}
