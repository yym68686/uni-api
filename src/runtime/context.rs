use crate::config::source::RuntimeConfigPublisher;
use crate::providers::codex::oauth::CodexOAuthManager;
use crate::runtime::idempotency;
use crate::runtime::resources::ResourceGovernor;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::Persistence;
use crate::transport::spool::SpoolManager;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub struct AppState {
    pub backend_origin: Arc<str>,
    pub control_token: Arc<str>,
    pub backend_client: reqwest::Client,
    upstream_clients: crate::upstream::client::ClientPool,
    pub(crate) resource_governor: ResourceGovernor,
    pub(crate) request_spool: SpoolManager,
    pub(crate) idempotency: idempotency::Coordinator,
    pub(crate) responses_data_plane_enabled: bool,
    pub python_compat_enabled: bool,
    pub persistence: Persistence,
    pub config_publisher: RuntimeConfigPublisher,
    pub runtime: GatewayRuntime,
    pub codex_oauth: CodexOAuthManager,
    pub(crate) channel_metrics: crate::observability::metrics::ChannelMetrics,
}

impl AppState {
    pub fn new(
        backend_origin: String,
        control_token: String,
        python_compat_enabled: bool,
        persistence: Persistence,
        config_publisher: RuntimeConfigPublisher,
    ) -> Result<Self, String> {
        let backend_client = reqwest::Client::builder()
            .http1_only()
            .pool_max_idle_per_host(256)
            .build()
            .map_err(|error| format!("build Python backend client: {error}"))?;
        let resource_governor = ResourceGovernor::new();
        let request_spool = SpoolManager::new(resource_governor.clone())?;
        Ok(Self {
            backend_origin: backend_origin.into(),
            control_token: control_token.into(),
            backend_client,
            upstream_clients: crate::upstream::client::ClientPool::default(),
            resource_governor,
            request_spool,
            idempotency: idempotency::Coordinator::new(),
            responses_data_plane_enabled: env_bool("UNI_API_RUST_RESPONSES_DATA_PLANE", true),
            python_compat_enabled,
            persistence,
            config_publisher,
            runtime: GatewayRuntime::new(),
            codex_oauth: CodexOAuthManager::new(),
            channel_metrics: crate::observability::metrics::global(),
        })
    }

    pub async fn upstream_client(
        &self,
        proxy: Option<&str>,
        http1_only: bool,
        connect_timeout: Option<Duration>,
    ) -> Result<reqwest::Client, String> {
        self.upstream_clients
            .get(proxy, http1_only, connect_timeout)
            .await
    }

    pub fn internal_url(&self, path: &str) -> String {
        format!("{}{}", self.backend_origin, path)
    }

    pub async fn runtime_observability(&self) -> serde_json::Value {
        serde_json::json!({
            "resource_governor":self.resource_governor.observability_snapshot(),
            "idempotency":self.idempotency.observability_snapshot().await,
            "upstream_http_clients":{
                "pooled_clients":self.upstream_clients.len().await,
            },
        })
    }
}

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}
