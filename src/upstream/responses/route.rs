use crate::config::snapshot::endpoint_values;
use crate::config::snapshot::ApiKey;
use crate::config::snapshot::Provider;
use crate::config::snapshot::Snapshot;
use crate::protocols::responses::item_ids::normalize_response_root;
use crate::protocols::sse::UNLIMITED_SSE_EVENT_BYTES;
use crate::providers::codex::oauth::CodexOAuthManager;
use crate::providers::codex::CODEX_USER_AGENT;
use crate::providers::overrides::apply_overrides;
use crate::routing::access::model_prices;
use crate::routing::access::preference_string;
use crate::routing::failure::classify_provider_failure;
use crate::routing::planner::TARGET_PROVIDER_HEADER;
use crate::routing::timeouts::resolve_timeouts;
use crate::routing::types::FailedRoute;
use crate::runtime::resources::MemoryReservation;
use crate::runtime::scheduling::ProviderKeySelection;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::ChannelStat;
use crate::storage::database::Persistence;
use crate::storage::database::RequestStat;
use crate::upstream::hedging::HedgingConfig;
use crate::upstream::responses::Plan;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use url::Url;

pub(crate) const DEFAULT_MAX_PRECOMMIT_ITEMS: usize = 128;

pub(crate) const DEFAULT_MAX_PRECOMMIT_BYTES: usize = 8 * 1024 * 1024 + 128 * 266;

#[derive(Clone, Debug)]
pub(crate) struct NativeAttemptObservation {
    pub(crate) request_id: String,
    pub(crate) attempt_id: String,
    pub(crate) attempt_index: usize,
    pub(crate) provider: String,
    pub(crate) request_model: String,
    pub(crate) actual_model: String,
    pub(crate) wire_model: Option<String>,
    pub(crate) upstream_host: String,
    pub(crate) stream: bool,
    pub(crate) snapshot_revision: String,
    pub(crate) started_at: tokio::time::Instant,
}

pub(crate) struct RoutingAttemptEvent<'a> {
    pub(crate) attempt_number: usize,
    pub(crate) provider: &'a Provider,
    pub(crate) original_model: &'a str,
    pub(crate) outcome: &'a str,
    pub(crate) attempt_id: Option<&'a str>,
    pub(crate) skip_reason: Option<&'a str>,
    pub(crate) status: Option<u16>,
}

pub enum ResponsesPreparation {
    Ready(ResponsesRoute),
    Fallback,
    Response(Response<Body>),
}

pub struct ResponsesRoute {
    pub(crate) store: GatewayRuntime,
    pub(crate) codex_oauth: CodexOAuthManager,
    pub(crate) persistence: Persistence,
    pub(crate) snapshot: Arc<Snapshot>,
    pub(crate) api_key: Arc<ApiKey>,
    pub(crate) providers: Vec<Arc<Provider>>,
    pub(crate) base_payload: Value,
    pub(crate) request_headers: HeaderMap,
    pub(crate) request_model: String,
    pub(crate) endpoint: String,
    pub(crate) request_type: Option<String>,
    pub(crate) wants_compact: bool,
    pub(crate) stream: bool,
    pub(crate) request_id: String,
    pub(crate) request_body_bytes: u64,
    pub(crate) cursor: usize,
    pub(crate) max_attempts: usize,
    pub(crate) hedging: HedgingConfig,
    pub(crate) attempt_contexts: HashMap<String, NativeAttemptObservation>,
    pub(crate) history_repair_attempted: bool,
    pub(crate) pending_history_repair: Option<Plan>,
    pub(crate) empty_name_repair_attempt_id: Option<String>,
    pub(crate) missing_item_repair_attempt_id: Option<String>,
    pub(crate) hedge_trigger_count: usize,
    pub(crate) hedge_cancelled_attempt_count: usize,
    pub(crate) last_provider: Option<Arc<Provider>>,
    pub(crate) last_provider_key: Option<String>,
    pub(crate) last_original_model: Option<String>,
    pub(crate) last_attempt: Option<NativeAttemptObservation>,
    pub(crate) last_status: u16,
    pub(crate) last_detail: String,
    pub(crate) last_provider_model_unavailable: bool,
    pub(crate) has_attempt_failure: bool,
    pub(crate) last_failure_origin: String,
    pub(crate) routing_attempts: usize,
    pub(crate) routing_skips: usize,
    pub(crate) upstream_attempts: usize,
    pub(crate) upstream_duration_ms: u64,
    pub(crate) routing_ledger: Vec<Value>,
    pub(crate) upstream_ledger: Vec<Value>,
    pub(crate) arrival: Option<crate::observability::timing::RequestArrival>,
    pub(crate) started_at: tokio::time::Instant,
    pub(crate) final_emitted: bool,
    pub(crate) _memory_reservation: MemoryReservation,
}

impl ResponsesRoute {
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

    pub(crate) async fn record_plan_failure(&mut self, mut plan: Plan, outcome: &Value) -> bool {
        self.set_current_plan(&plan);
        let retryable = self.record_failure(outcome).await;
        if !self.auto_retry()
            || self.request_headers.contains_key(TARGET_PROVIDER_HEADER)
            || self.endpoint != "/v1/responses"
        {
            return retryable;
        }
        // The repaired request was already tried on this exact channel/key.
        // A second HTTP 400 must not suppress the user's configured fallback
        // chain. Preserve the original failure status and normal retry budget.
        if self.empty_name_repair_attempt_id.as_deref() == Some(plan.attempt_id.as_str())
            && crate::protocols::responses::empty_name::is_uncommitted_400(outcome)
        {
            return self.has_attempts_remaining();
        }
        if self.missing_item_repair_attempt_id.as_deref() == Some(plan.attempt_id.as_str())
            && (crate::protocols::responses::missing_item::is_uncommitted_404(outcome)
                || crate::protocols::responses::empty_name::is_uncommitted_400(outcome))
        {
            return self.has_attempts_remaining();
        }
        let empty_name = crate::protocols::responses::empty_name::repair(&plan.body, outcome);
        let missing_item = crate::protocols::responses::missing_item::repair(&plan.body, outcome);
        if self.history_repair_attempted {
            // Later channels still get their own compilation of the original
            // input. Recognize the same confirmed bad history, without another
            // repair resend or an unbounded loop.
            return retryable
                || (self.empty_name_repair_attempt_id.is_some()
                    && empty_name.is_some()
                    && self.has_attempts_remaining())
                || (self.missing_item_repair_attempt_id.is_some()
                    && missing_item.is_some()
                    && self.has_attempts_remaining());
        }
        let is_empty_name = empty_name.is_some();
        let is_missing_item = missing_item.is_some();
        let Some((body, changed)) = empty_name
            .or(missing_item)
            .or_else(|| crate::protocols::responses::heartbeat::repair(&plan.body, outcome))
        else {
            return retryable;
        };
        let Some(mut observation) = self.last_attempt.clone() else {
            return retryable;
        };
        self.history_repair_attempted = true;
        let original_attempt_id = plan.attempt_id.clone();
        plan.body = body;
        plan.attempt_id = native_attempt_id(&self.request_id, self.routing_attempts);
        if is_empty_name {
            self.empty_name_repair_attempt_id = Some(plan.attempt_id.clone());
        }
        if is_missing_item {
            self.missing_item_repair_attempt_id = Some(plan.attempt_id.clone());
        }
        for (name, value) in &mut plan.headers {
            if name.eq_ignore_ascii_case("x-oaix-routing-attempt-id") {
                *value = plan.attempt_id.clone();
            }
        }
        self.routing_attempts = self.routing_attempts.saturating_add(1);
        self.upstream_attempts = self.upstream_attempts.saturating_add(1);
        observation.attempt_id = plan.attempt_id.clone();
        observation.attempt_index = self.routing_attempts;
        observation.started_at = tokio::time::Instant::now();
        self.attempt_contexts
            .insert(plan.attempt_id.clone(), observation.clone());
        plan.dispatch = self.arrival.map(|arrival| {
            arrival.attempt(
                crate::observability::metrics::MetricKey::new(
                    &observation.provider,
                    &self.request_model,
                    &observation.actual_model,
                    &self.endpoint,
                    self.stream,
                ),
                self.request_id.clone(),
                plan.attempt_id.clone(),
                &self.api_key.token,
            )
        });
        crate::observability::metrics::global().start(
            &observation.provider,
            &self.request_model,
            &observation.actual_model,
            &self.endpoint,
            self.stream,
        );
        let event = if is_empty_name {
            "responses_empty_name_repair"
        } else if is_missing_item {
            "responses_missing_item_repair"
        } else {
            "responses_heartbeat_repair"
        };
        let mut log = json!({
            "kind":"log", "event":event, "event_type":event, "severity":"info",
            "source":"uni-api-ember", "fugue_table":"app_events",
            "request_id":self.request_id, "original_attempt_id":original_attempt_id,
            "attempt_id":plan.attempt_id, "provider":observation.provider,
            "model":self.request_model,
        });
        log[if is_empty_name {
            "tool_pairs_converted"
        } else if is_missing_item {
            "reasoning_items_converted"
        } else {
            "heartbeat_items_converted"
        }] = json!(changed);
        eprintln!("{log}");
        self.pending_history_repair = Some(plan);
        true
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
        if let Some(plan) = self.pending_history_repair.take() {
            self.set_current_plan(&plan);
            return Ok(Some(plan));
        }
        while self.cursor < self.max_attempts {
            let attempt_number = self.routing_attempts;
            let provider = self.providers[self.cursor % self.providers.len()].clone();
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
                wire_model: payload
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
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
            crate::observability::metrics::global().start(
                provider.name.as_ref(),
                self.request_model.as_str(),
                original_model.as_str(),
                self.endpoint.as_str(),
                self.stream,
            );
            return Ok(Some(Plan {
                dispatch: self.arrival.map(|arrival| {
                    arrival.attempt(
                        crate::observability::metrics::MetricKey::new(
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
            self.last_attempt
                .as_ref()
                .and_then(|attempt| attempt.wire_model.as_deref()),
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

    pub(crate) fn emit_routing_attempt(&mut self, event: RoutingAttemptEvent<'_>) {
        let status = event.status.unwrap_or_default();
        let metrics = crate::observability::metrics::global();
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

    pub(crate) fn emit_upstream_attempt(
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
        crate::observability::metrics::global().finish(
            &attempt.provider,
            &attempt.request_model,
            &attempt.actual_model,
            &self.endpoint,
            attempt.stream,
            if success { "success" } else { "failed" },
            Some(duration_ms as f64),
            outcome.get("first_output_ms").and_then(Value::as_f64),
        );
        crate::observability::metrics::global().response_timings(
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

    pub(crate) fn emit_final_event(&mut self, status: u16, kind: &str, outcome: &Value) {
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
            fact_usage: crate::observability::usage::FactUsage::from_usage(outcome.get("usage")),
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

    pub(crate) fn auto_retry(&self) -> bool {
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

    pub(crate) fn record_current_channel(&self, success: bool, outcome: &Value) {
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

pub(crate) fn usage_tokens(outcome: &Value) -> (i64, i64, i64) {
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

pub(crate) fn trace_id(headers: &HeaderMap, fallback: &str) -> String {
    headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split('-').nth(1))
        .filter(|value| value.len() == 32)
        .unwrap_or(fallback)
        .to_owned()
}

pub(crate) fn client_ip(headers: &HeaderMap) -> String {
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

pub(crate) fn compile_payload(
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

pub(crate) fn strip_codex_fields(root: &mut Map<String, Value>) {
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

pub(crate) fn provider_stream_override(provider: &Provider) -> Option<bool> {
    provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get("stream"))
        .and_then(Value::as_bool)
}

pub(crate) fn strip_key_recursive(value: &mut Value, key: &str) {
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

pub(crate) fn normalization_enabled(
    provider: &Provider,
    request_model: &str,
    original_model: &str,
) -> bool {
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

pub(crate) fn build_headers(
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
    if let Some(value) = incoming
        .get("x-oaix-settlement-nonce")
        .and_then(|v| v.to_str().ok())
    {
        headers.insert("X-OAIX-Settlement-Nonce".into(), value.to_owned());
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

pub(crate) fn header_or(headers: &HeaderMap, name: &str, default: &str) -> String {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

pub(crate) fn normalize_upstream_url(base_url: &str, engine: &str, wants_compact: bool) -> String {
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

pub(crate) fn upstream_host(base_url: &str) -> String {
    Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default()
}

pub(crate) fn outcome_status(outcome: &Value, fallback: u16) -> u16 {
    outcome_status_from(
        outcome,
        "status_code",
        outcome_status_from(outcome, "upstream_status_code", fallback),
    )
}

pub(crate) fn outcome_status_from(outcome: &Value, key: &str, fallback: u16) -> u16 {
    outcome
        .get(key)
        .and_then(Value::as_u64)
        .filter(|status| *status > 0)
        .unwrap_or(u64::from(fallback))
        .min(u64::from(u16::MAX)) as u16
}

pub(crate) fn failure_origin(outcome: &Value) -> &'static str {
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

pub(crate) fn native_rejection_origin(reason: &str) -> &'static str {
    match reason {
        "native_global_rate_limit" => "native_global_rate_limit",
        "native_client_rate_limit" => "native_client_rate_limit",
        "no_matching_provider" => "native_route_selection",
        "invalid_api_key" => "native_authentication",
        _ => "native_request_validation",
    }
}

pub(crate) fn status_class(status: u16) -> &'static str {
    match status {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "unknown",
    }
}

pub(crate) fn event_severity(status: u16, outcome: &str) -> &'static str {
    if status >= 500 {
        "error"
    } else if status >= 400 || matches!(outcome, "skipped" | "failed" | "completed_with_error") {
        "warning"
    } else {
        "info"
    }
}

pub(crate) fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(crate) fn terminal_error_sha256(success: bool, detail: &str) -> Option<String> {
    (!success && !detail.is_empty()).then(|| sha256_hex(detail))
}

pub(crate) fn native_attempt_id(request_id: &str, attempt: usize) -> String {
    format!("{request_id}-r{}", attempt + 1)
}

#[cfg(test)]
mod tests;
