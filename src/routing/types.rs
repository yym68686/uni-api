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

#[derive(Clone)]
pub(crate) struct ResolvedRoute {
    pub(crate) providers: Vec<Arc<Provider>>,
    // An optional one-pass tier-specific prefix followed by the unchanged ordinary schedule.
    pub(crate) preferred_count: usize,
    pub(crate) hedging: HedgingConfig,
    pub(crate) unknown_400_retries: u8,
}

impl ResolvedRoute {
    pub(crate) fn scan_limit(&self, original: usize) -> usize {
        if self.preferred_count != 0
            || self
                .providers
                .first()
                .is_some_and(|provider| super::service_tier::is_standard(provider))
        {
            original.max(self.providers.len())
        } else {
            original
        }
    }

    pub(crate) fn candidate_index(&self, cursor: usize, provider_count: usize) -> usize {
        if cursor < self.preferred_count {
            cursor
        } else {
            self.preferred_count
                + (cursor - self.preferred_count) % (provider_count - self.preferred_count)
        }
    }

    pub(crate) fn has_alternative(&self, provider: &Provider) -> bool {
        self.providers
            .iter()
            .any(|other| other.name != provider.name)
    }
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
