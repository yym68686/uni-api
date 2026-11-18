//! Console-owned configuration. A fresh gateway can wait without a PVC, and
//! an already serving gateway never replaces a verified bundle with a draft.
use crate::api::platform::json_response;
use crate::config::{compiler::compile_config_document, source};
use crate::control::channels::{RestoreMutation, RetainedSnapshot};
use crate::runtime::{context::AppState, state::GatewayRuntime};
use crate::transport::http::json_error;
use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{HeaderMap, Response, StatusCode},
};
use futures_util::StreamExt;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};

const POLL: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub(crate) struct ManagedConfig {
    url: reqwest::Url,
    credential: Arc<str>,
    client: reqwest::Client,
    active: Arc<AtomicU64>,
    serial: Arc<Mutex<()>>,
    allow_local_seed: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema: u8,
    base: Value,
    controls: RetainedSnapshot,
}

#[derive(Deserialize)]
struct Version {
    revision: u64,
    bundle: Bundle,
}

#[derive(Deserialize)]
struct Versions {
    revision: u64,
    #[serde(default)]
    uninitialized: bool,
    published: Option<Version>,
    pending: Option<Version>,
}

#[derive(Deserialize)]
struct Committed {
    revision: u64,
    credits: Option<f64>,
    #[serde(default)]
    duplicate: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyncError {
    Unavailable,
    Unauthorized,
    Changed,
    Invalid,
}

impl ManagedConfig {
    pub(crate) fn from_env() -> Result<Option<Self>, String> {
        let url = match std::env::var("UNI_API_MANAGED_CONFIG_URL") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(_) => return Err("invalid managed configuration URL".into()),
        };
        let url = reqwest::Url::parse(&url).map_err(|_| "invalid managed configuration URL")?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path().ends_with('/')
        {
            return Err("invalid managed configuration URL".into());
        }
        let token = std::env::var("UNI_API_MANAGED_CONFIG_ADMIN_KEY")
            .map_err(|_| "managed configuration admin key missing")?;
        if token.len() < 8 || token.len() > 4096 || !token.bytes().all(|c| c.is_ascii_graphic()) {
            return Err("invalid managed configuration admin key".into());
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(8))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|_| "managed configuration client unavailable")?;
        Ok(Some(Self {
            url,
            credential: token.into(),
            client,
            active: Arc::new(AtomicU64::new(0)),
            serial: Arc::new(Mutex::new(())),
            allow_local_seed: crate::runtime::context::env_bool(
                "UNI_API_MANAGED_ALLOW_LOCAL_SEED",
                true,
            ),
        }))
    }

    pub(crate) fn applied_revision(&self) -> Option<u64> {
        let revision = self.active.load(Ordering::Acquire);
        (revision > 0).then_some(revision)
    }

    pub(crate) async fn read_guard(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.serial.lock().await
    }

    async fn exchange<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, SyncError> {
        let response = request
            .bearer_auth(self.credential.as_ref())
            .send()
            .await
            .map_err(|_| SyncError::Unavailable)?;
        match response.status().as_u16() {
            200 => {}
            401 | 403 => return Err(SyncError::Unauthorized),
            409 => return Err(SyncError::Changed),
            400 | 413 | 422 => return Err(SyncError::Invalid),
            _ => return Err(SyncError::Unavailable),
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| SyncError::Unavailable)?;
            if bytes.len().saturating_add(chunk.len()) > config_limit() {
                return Err(SyncError::Invalid);
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| SyncError::Invalid)
    }

    async fn post(&self, body: Value) -> Result<Committed, SyncError> {
        let bytes = serde_json::to_vec(&body).map_err(|_| SyncError::Invalid)?;
        if bytes.len() > config_limit() {
            return Err(SyncError::Invalid);
        }
        self.exchange(
            self.client
                .post(self.url.clone())
                .header("content-type", "application/json")
                .body(bytes),
        )
        .await
    }

    async fn install(
        &self,
        state: &AppState,
        next: GatewayRuntime,
        revision: u64,
    ) -> Result<(), SyncError> {
        if revision == 0 {
            return Err(SyncError::Invalid);
        }
        let base = next.base_snapshot().await.ok_or(SyncError::Invalid)?;
        let controls = next.channel_controls.read().await.clone();
        state.runtime.install_bundle(base, controls).await;
        self.active.store(revision, Ordering::Release);
        eprintln!(
            "{}",
            json!({"event_type":"managed_configuration_applied","revision":revision})
        );
        Ok(())
    }

    async fn synchronize(&self, state: &AppState) -> Result<(), SyncError> {
        let _writer = self.serial.lock().await;
        let active = self.applied_revision().unwrap_or(0);
        let versions: Versions = self
            .exchange(
                self.client
                    .get(self.url.clone())
                    .query(&[("applied_revision", active)]),
            )
            .await?;
        if versions.revision < active {
            return Err(SyncError::Changed);
        }
        if let Some(pending) = versions.pending {
            if pending.revision <= versions.revision {
                return Err(SyncError::Invalid);
            }
            let candidate = build_candidate(state, pending.bundle, false).await;
            let committed = self
                .post(json!({
                    "action": if candidate.is_ok() {"promote"} else {"reject"},
                    "expected_revision":versions.revision,"draft_revision":pending.revision
                }))
                .await?;
            if let Ok(candidate) = candidate {
                if committed.revision != pending.revision {
                    return Err(SyncError::Invalid);
                }
                return self.install(state, candidate, committed.revision).await;
            }
        }
        if versions.revision > active {
            let version = versions.published.ok_or(SyncError::Invalid)?;
            if version.revision != versions.revision {
                return Err(SyncError::Invalid);
            }
            let candidate = build_candidate(state, version.bundle, false).await?;
            self.install(state, candidate, version.revision).await?;
        } else if versions.revision == 0
            && versions.uninitialized
            && self.allow_local_seed
            && !state.runtime.is_ready().await
        {
            // This branch is reached only after a successful authenticated
            // reply from an explicitly uninitialized source.
            if let Ok(bytes) = state.config_publisher.compile_source().await {
                let raw =
                    GatewayRuntime::decode_snapshot(&bytes).map_err(|_| SyncError::Invalid)?;
                let base = state.runtime.build_snapshot(raw).await;
                let legacy = detached(&state.runtime).await;
                *legacy.current.write().await = Some(base.clone());
                tokio::time::timeout(Duration::from_secs(22), legacy.restore_controls_on_start())
                    .await
                    .map_err(|_| SyncError::Unavailable)?
                    .map_err(|_| SyncError::Invalid)?;
                let controls = legacy.channel_controls.read().await.clone();
                state.runtime.install_bundle(base, controls).await;
                eprintln!(
                    "{}",
                    json!({"event_type":"managed_configuration_using_local_seed"})
                );
            }
        }
        Ok(())
    }

    pub(crate) async fn wait_for_first_configuration(&self, state: &AppState) {
        loop {
            let result = self.synchronize(state).await;
            if state.runtime.is_ready().await {
                return;
            }
            eprintln!(
                "{}",
                json!({"event_type":"managed_configuration_waiting","reason":match result {
                    Ok(()) => "not_configured", Err(SyncError::Unauthorized) => "unauthorized",
                    Err(SyncError::Changed) => "revision_changed", Err(SyncError::Invalid) => "invalid_configuration",
                    Err(SyncError::Unavailable) => "control_plane_unavailable"
                }})
            );
            tokio::time::sleep(POLL).await;
        }
    }

    pub(crate) fn start_watcher(&self, state: AppState) {
        let managed = self.clone();
        tokio::spawn(async move {
            let mut last_error = None;
            loop {
                tokio::time::sleep(POLL).await;
                let error = managed.synchronize(&state).await.err();
                if error != last_error {
                    eprintln!(
                        "{}",
                        json!({"event_type":"managed_configuration_sync_status","error":error.map(|e|format!("{e:?}"))})
                    );
                    last_error = error;
                }
            }
        });
    }

    // Use the existing gateway authorization and mutation semantics on a
    // detached copy, then commit to the console before exposing any change.
    pub(crate) async fn mutate(&self, state: &AppState, request: Request) -> Response<Body> {
        let _writer = self.serial.lock().await;
        let Some(expected) = self.applied_revision() else {
            return json_error(
                StatusCode::CONFLICT,
                "Publish the initial managed configuration before modifying it",
            );
        };
        let path = request.uri().path().trim_end_matches('/').to_owned();
        let mut staged = state.clone();
        staged.runtime = detached(&state.runtime).await;
        let before = match snapshot_bundle(&staged.runtime).await {
            Some(bundle) => bundle,
            None => {
                return json_error(StatusCode::SERVICE_UNAVAILABLE, "Configuration unavailable")
            }
        };
        let (bundle, response, op) =
            if matches!(path.as_str(), "/v1/api_config/update" | "/v1/add_credits") {
                if let Err(response) =
                    crate::api::platform::require_admin(state, request.headers()).await
                {
                    return response;
                }
                let mut bundle = before.clone();
                if path == "/v1/add_credits" {
                    let Some(id) = request
                        .headers()
                        .get("idempotency-key")
                        .and_then(|v| v.to_str().ok())
                        .filter(|value| {
                            (1..=128).contains(&value.len())
                                && value.bytes().all(|b| b.is_ascii_graphic())
                        })
                    else {
                        return json_error(
                            StatusCode::BAD_REQUEST,
                            "Managed credits require an Idempotency-Key",
                        );
                    };
                    let paid_key = crate::upstream::generic::query_value(request.uri(), "paid_key")
                        .unwrap_or_default();
                    let amount: f64 =
                        crate::upstream::generic::query_value(request.uri(), "amount")
                            .and_then(|value| value.parse().ok())
                            .unwrap_or_default();
                    let digest = format!(
                        "{:x}",
                        Sha256::digest(format!(
                            "{}:{amount}",
                            crate::routing::catalog::key_id(&paid_key)
                        ))
                    );
                    let credits = match source::credit_document(&mut bundle.base, &paid_key, amount)
                    {
                        Ok(credits) => credits,
                        Err(_) => {
                            return json_error(StatusCode::BAD_REQUEST, "Invalid credit update")
                        }
                    };
                    (
                        bundle,
                        json_response(
                            StatusCode::OK,
                            json!({"paid_key":paid_key,"credits":credits}),
                        ),
                        Some((id.to_owned(), digest, credits, paid_key)),
                    )
                } else {
                    let Ok(bytes) =
                        to_bytes(request.into_body(), crate::control::config_body_limit()).await
                    else {
                        return json_error(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "Configuration update body is too large",
                        );
                    };
                    let Ok(patch) = serde_json::from_slice::<Value>(&bytes) else {
                        return json_error(StatusCode::BAD_REQUEST, "Invalid configuration patch");
                    };
                    if source::patch_document(&mut bundle.base, &patch).is_err() {
                        return json_error(StatusCode::BAD_REQUEST, "Invalid configuration patch");
                    }
                    (
                        bundle,
                        json_response(StatusCode::OK, json!({"message":"API config updated"})),
                        None,
                    )
                }
            } else {
                let response = crate::api::platform::handle_local_mutation(&staged, request).await;
                if response.status() != StatusCode::OK {
                    return response;
                }
                let Some(bundle) = snapshot_bundle(&staged.runtime).await else {
                    return json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Configuration unavailable",
                    );
                };
                (bundle, response, None)
            };
        let changed_base = before.base != bundle.base;
        // Round-trip validation also proves that the next fresh pod can load
        // the exact version we are about to publish.
        let candidate = match build_candidate(&staged, bundle.clone(), changed_base).await {
            Ok(validated) if changed_base => validated,
            Ok(_) => staged.runtime,
            Err(_) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "Managed configuration validation failed",
                )
            }
        };
        let committed = self
            .post(
                json!({"action":"commit","expected_revision":expected,"bundle":bundle,
            "operation_id":op.as_ref().map(|x| &x.0),"operation_hash":op.as_ref().map(|x| &x.1),
            "credits":op.as_ref().map(|x| x.2)}),
            )
            .await;
        match committed {
            Ok(result) => {
                // An idempotent replay returns the same credits without
                // reapplying an older or differently staged bundle.
                if !result.duplicate
                    && self
                        .install(state, candidate, result.revision)
                        .await
                        .is_err()
                {
                    return json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Configuration committed; refresh before retrying",
                    );
                }
                if let Some((_, _, _, paid_key)) = op {
                    return json_response(
                        StatusCode::OK,
                        json!({"paid_key":paid_key,"credits":result.credits}),
                    );
                }
                response
            }
            Err(SyncError::Changed) => json_error(
                StatusCode::CONFLICT,
                "Managed configuration changed; refresh before retrying",
            ),
            Err(_) => json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Managed configuration commit not confirmed; refresh before retrying",
            ),
        }
    }
}

fn config_limit() -> usize {
    crate::control::config_body_limit().min((32 << 20) + 4096)
}

async fn snapshot_bundle(runtime: &GatewayRuntime) -> Option<Bundle> {
    let controls = runtime.channel_controls.read().await;
    let base = runtime.base_snapshot().await?;
    Some(Bundle {
        schema: 1,
        base: base.api_config.as_ref().clone(),
        controls: controls.retained(),
    })
}

async fn detached(runtime: &GatewayRuntime) -> GatewayRuntime {
    let mut detached = runtime.clone();
    let controls = runtime.channel_controls.read().await.clone();
    detached.current = Arc::new(RwLock::new(runtime.base_snapshot().await));
    detached.channel_controls = Arc::new(RwLock::new(controls));
    detached
}

async fn build_candidate(
    state: &AppState,
    bundle: Bundle,
    preserve_operations: bool,
) -> Result<GatewayRuntime, SyncError> {
    if bundle.schema != 1 {
        return Err(SyncError::Invalid);
    }
    let bytes = compile_config_document(bundle.base, state.config_publisher.database_disabled)
        .map_err(|_| SyncError::Invalid)?;
    let raw = GatewayRuntime::decode_snapshot(&bytes).map_err(|_| SyncError::Invalid)?;
    let staged = detached(&state.runtime).await;
    let base = staged.build_snapshot(raw).await;
    if state
        .managed_config
        .as_ref()
        .is_some_and(|managed| !is_admin_key(&base, managed.credential.as_ref()))
        || base.api_keys.is_empty()
        || (base.providers.is_empty() && bundle.controls.temporary_channels.is_empty())
    {
        return Err(SyncError::Invalid);
    }
    let revision = staged.channel_controls.read().await.revision(&base);
    *staged.current.write().await = Some(base);
    let operations = if preserve_operations {
        staged
            .channel_controls
            .read()
            .await
            .settings_operations
            .clone()
    } else {
        Default::default()
    };
    staged
        .restore_controls_authorized(RestoreMutation {
            revision,
            snapshot: bundle.controls,
        })
        .await
        .map_err(|_| SyncError::Invalid)?;
    staged.channel_controls.write().await.settings_operations = operations;
    Ok(staged)
}

pub(crate) async fn export_response(state: &AppState, headers: &HeaderMap) -> Response<Body> {
    let controls = state.runtime.channel_controls.read().await;
    let Some(base) = state.runtime.base_snapshot().await else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "Configuration unavailable");
    };
    if crate::routing::access::extract_api_key(headers)
        .is_none_or(|token| !is_admin_key(&base, &token))
    {
        return json_error(StatusCode::FORBIDDEN, "Administrator key required");
    }
    let bundle = Bundle {
        schema: 1,
        base: base.api_config.as_ref().clone(),
        controls: controls.retained(),
    };
    let mut response = json_response(
        StatusCode::OK,
        json!({"bundle":bundle, "revision":controls.revision(&base)}),
    );
    response.headers_mut().insert(
        "cache-control",
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

fn is_admin_key(base: &crate::config::snapshot::Snapshot, token: &str) -> bool {
    base.api_keys.get(token).is_some_and(|admin| {
        let role = admin.role.to_ascii_lowercase();
        role == "admin" || role.starts_with("admin-")
    })
}
