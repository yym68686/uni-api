//! Resolve configuration once per attempt from the request's pinned snapshot.
use crate::config::snapshot::{Provider, Snapshot};
use crate::routing::context::RequestFacts;
use crate::routing::timeouts::{
    model_preference, model_timeout, positive_duration, resolve_timeouts, Timeouts,
};
use std::time::Duration;

pub(crate) const MAX_PRECOMMIT_ITEMS: usize = 128;
pub(crate) const MAX_PRECOMMIT_BYTES: usize = 8 * 1024 * 1024 + 128 * 266;

#[derive(Clone, Debug)]
pub(crate) struct ResponseStreamPolicy {
    pub commit_policy: String,
    pub normalize_ids: bool,
    pub max_precommit_items: usize,
    pub max_precommit_bytes: usize,
}

impl ResponseStreamPolicy {
    pub fn resolve(provider: &Provider, model: &str, original_model: &str) -> Self {
        Self {
            commit_policy: crate::routing::access::preference_string(
                &provider.preferences,
                "responses_stream_commit_policy",
            )
            .unwrap_or_else(|| "real_output".into()),
            normalize_ids: crate::providers::responses::normalization_enabled(
                provider,
                model,
                original_model,
            ),
            max_precommit_items: MAX_PRECOMMIT_ITEMS,
            max_precommit_bytes: MAX_PRECOMMIT_BYTES,
        }
    }
}

pub(crate) struct ResolvedAttemptPolicy {
    pub proxy: Option<String>,
    pub http1_only: bool,
    pub timeouts: Timeouts,
    pub keepalive_interval: Option<Duration>,
    pub responses: ResponseStreamPolicy,
}

impl ResolvedAttemptPolicy {
    pub fn proxy(snapshot: &Snapshot, provider: &Provider) -> Option<String> {
        crate::routing::access::preference_string(&provider.preferences, "proxy")
            .or_else(|| crate::routing::access::preference_string(&snapshot.preferences, "proxy"))
    }

    pub fn resolve(
        snapshot: &Snapshot,
        provider: &Provider,
        facts: &RequestFacts,
        original_model: &str,
        role: &str,
        method: &str,
        stream: bool,
    ) -> Self {
        let interval = model_preference(
            provider,
            &snapshot.preferences,
            &facts.model,
            original_model,
            "keepalive_interval",
        )
        .unwrap_or(99999.0);
        let timeout = model_timeout(
            provider,
            &snapshot.preferences,
            &facts.model,
            original_model,
        );
        Self {
            proxy: Self::proxy(snapshot, provider),
            http1_only: provider.engine.eq_ignore_ascii_case("codex"),
            timeouts: resolve_timeouts(
                snapshot,
                provider,
                &facts.model,
                original_model,
                &provider.engine,
                stream,
                facts.request_type,
                role,
                &facts.endpoint,
                method,
            ),
            keepalive_interval: (interval <= timeout)
                .then(|| positive_duration(Some(interval)))
                .flatten(),
            responses: ResponseStreamPolicy::resolve(provider, &facts.model, original_model),
        }
    }
}
