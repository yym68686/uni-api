use crate::config::snapshot::ApiKey;
use crate::config::snapshot::Provider;
use crate::upstream::hedging::HedgingConfig;
use axum::http::StatusCode;
use std::sync::Arc;

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
