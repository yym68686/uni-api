use crate::config::snapshot::Provider;
use crate::protocols::conversion::chat_to_responses_response;
use crate::protocols::conversion::cloudflare_to_chat;
use crate::protocols::conversion::cohere_to_chat;
use crate::protocols::conversion::normalize_search_response;
use crate::protocols::conversion::responses_to_chat;
use crate::protocols::conversion::synthetic_chat_stream;
use crate::protocols::conversion::synthetic_responses_stream;
use crate::protocols::provider_stream;
use crate::protocols::provider_stream::OutputProtocol as StreamOutputProtocol;
use crate::protocols::provider_stream::Protocol as StreamProtocol;
use crate::providers::anthropic::claude_to_chat;
use crate::providers::gemini::gemini_to_chat;
use crate::providers::request::build_attempt;
use crate::providers::types::AttemptBody;
use crate::providers::types::DownstreamProtocol;
use crate::providers::types::PreparedAttempt;
use crate::providers::types::PreparedInput;
use crate::providers::types::ResponseAdapter;
use crate::providers::vertex::vertex_access_token;
use crate::providers::video::normalize_callxyq_video_response;
use crate::providers::video::normalize_lingjing_video_response;
use crate::providers::video::remember_video_task;
use crate::providers::video::VideoTaskRoute;
use crate::routing::failure::classify_provider_failure;
use crate::routing::timeouts::{earliest_timeout, positive_duration, remaining_timeout};
use crate::routing::types::FailedRoute;
use crate::runtime::context::AppState;
use crate::runtime::resources::MemoryReservation;
use crate::runtime::scheduling::ProviderKeySelection;
use crate::storage::database::ChannelStat;
use crate::storage::database::RequestStat;
use crate::transport::body::read_limited_upstream_body_with_idle;
use crate::transport::body::upstream_chunks;
use crate::transport::body::upstream_response_max_bytes;
use crate::transport::body::UPSTREAM_ERROR_MAX_BYTES;
use crate::transport::http::filtered_response_headers;
use crate::transport::http::json_error;
use crate::transport::multipart::multipart_rewrite_body;
use crate::transport::multipart::prepare_dashscope_transcription;
use crate::upstream::hedging::await_with_hedge;
use crate::upstream::hedging::deadline as hedge_deadline;
use crate::upstream::hedging::HedgeEvent;
use crate::upstream::hedging::HedgeScheduler;
use crate::upstream::hedging::HedgeTrigger;
use crate::upstream::hedging::HedgingConfig;
use axum::body::Body;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode, Uri};
use bytes::Bytes;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use url::Url;

pub(crate) struct AttemptLoop {
    pub(crate) state: AppState,
    pub(crate) started: Instant,
    pub(crate) arrival: Option<crate::observability::timing::RequestArrival>,
    pub(crate) method: Method,
    pub(crate) uri: Uri,
    pub(crate) path: String,
    pub(crate) headers: HeaderMap,
    pub(crate) request_id: String,
    pub(crate) trace_id: String,
    pub(crate) client_ip: String,
    pub(crate) api_key: String,
    pub(crate) api_key_role: String,
    pub(crate) request_model: String,
    pub(crate) video_task_route: Option<VideoTaskRoute>,
    pub(crate) providers: Vec<Arc<Provider>>,
    pub(crate) max_attempts: usize,
    pub(crate) auto_retry: bool,
    pub(crate) input: PreparedInput,
    pub(crate) image_reservations: Vec<MemoryReservation>,
    pub(crate) prompt_price: f64,
    pub(crate) completion_price: f64,
    pub(crate) keepalive_updates: Option<crate::protocols::chat::stream::KeepaliveUpdates>,
}

pub(crate) fn chat_nonstream_hedging_enabled(
    path: &str,
    payload: Option<&Value>,
    hedging: HedgingConfig,
) -> bool {
    path == "/v1/chat/completions"
        && payload
            .and_then(|payload| payload.get("stream"))
            .and_then(Value::as_bool)
            .is_none_or(|stream| !stream)
        && hedging.active()
}

#[derive(Clone)]
pub(crate) struct GenericHedgeContext {
    pub(crate) attempt_index: usize,
    pub(crate) attempt_started: Instant,
    pub(crate) provider: Arc<Provider>,
    pub(crate) original_model: String,
    pub(crate) wire_model: Option<String>,
    pub(crate) provider_key: String,
    pub(crate) upstream_url: String,
    pub(crate) downstream_stream: bool,
}

pub(crate) struct GenericHedgePlan {
    pub(crate) context: GenericHedgeContext,
    pub(crate) prepared: PreparedAttempt,
}

pub(crate) struct GenericHedgeSuccess {
    pub(crate) context: GenericHedgeContext,
    pub(crate) success: AttemptSuccess,
}

pub(crate) struct GenericHedgeFailure {
    pub(crate) context: GenericHedgeContext,
    pub(crate) failure: AttemptFailure,
}

pub(crate) async fn next_generic_hedge_plan(
    execution: &AttemptLoop,
    cursor: &mut usize,
    last_status: &mut StatusCode,
    last_detail: &mut String,
) -> Option<GenericHedgePlan> {
    while *cursor < execution.max_attempts {
        let attempt_index = *cursor;
        *cursor += 1;
        let provider = execution.providers[attempt_index % execution.providers.len()].clone();
        let Some(original_model) = provider.models.get(&execution.request_model).cloned() else {
            continue;
        };
        let key_selection = if let Some(route) = execution
            .video_task_route
            .as_ref()
            .filter(|route| route.provider_name == provider.name.as_ref())
        {
            ProviderKeySelection::Selected(route.provider_key.clone())
        } else {
            execution
                .state
                .runtime
                .select_provider_key(&provider, &original_model)
                .await
        };
        let provider_key_raw = match key_selection {
            ProviderKeySelection::Selected(key) => key,
            ProviderKeySelection::NoProviderKey => {
                *last_status = StatusCode::BAD_GATEWAY;
                *last_detail = format!("Provider {} has no API key", provider.name);
                emit_routing_skip(
                    execution,
                    attempt_index,
                    &provider,
                    &original_model,
                    "provider_has_no_api_keys",
                );
                continue;
            }
            ProviderKeySelection::ChannelCooling => {
                *last_status = StatusCode::TOO_MANY_REQUESTS;
                *last_detail = "All matching provider routes are cooling down".into();
                emit_routing_skip(
                    execution,
                    attempt_index,
                    &provider,
                    &original_model,
                    "provider_channel_cooldown",
                );
                continue;
            }
            ProviderKeySelection::AllKeysCooling => {
                *last_status = StatusCode::TOO_MANY_REQUESTS;
                *last_detail = "All matching provider routes are cooling down".into();
                emit_routing_skip(
                    execution,
                    attempt_index,
                    &provider,
                    &original_model,
                    "provider_keys_cooldown",
                );
                continue;
            }
        };
        let mut provider_key = provider_key_raw.clone();
        let mut codex_account_id = None;
        if provider.engine.eq_ignore_ascii_case("codex") && provider_key_raw.contains(',') {
            match execution
                .state
                .codex_oauth
                .resolve(
                    &provider_key_raw,
                    provider.preferences.get("proxy").and_then(Value::as_str),
                )
                .await
            {
                Ok(auth) => {
                    provider_key = auth.bearer;
                    codex_account_id = auth.account_id;
                }
                Err(error) => {
                    *last_status = StatusCode::UNAUTHORIZED;
                    *last_detail = error;
                    emit_routing_skip(
                        execution,
                        attempt_index,
                        &provider,
                        &original_model,
                        "codex_oauth_resolution_failed",
                    );
                    continue;
                }
            }
        }
        let mut prepared = match build_attempt(
            &provider,
            &provider_key,
            &execution.request_model,
            &original_model,
            &execution.method,
            &execution.uri,
            &execution.path,
            &execution.headers,
            &execution.input,
            &execution.request_id,
        ) {
            Ok(attempt) => attempt,
            Err(error) => {
                *last_status = StatusCode::BAD_REQUEST;
                *last_detail = error;
                emit_routing_skip(
                    execution,
                    attempt_index,
                    &provider,
                    &original_model,
                    "attempt_build_failed",
                );
                continue;
            }
        };
        if let Some(account_id) = codex_account_id {
            if let Ok(value) = HeaderValue::from_str(&account_id) {
                prepared.headers.insert("chatgpt-account-id", value);
            }
        }
        let attempt_id = format!("{}-r{}", execution.request_id, attempt_index + 1);
        if let Ok(value) = HeaderValue::from_str(&attempt_id) {
            prepared
                .headers
                .insert("x-uni-api-attempt-id", value.clone());
            if provider
                .preferences
                .get("oaix_routing_attempt_id")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                prepared.headers.insert("x-oaix-routing-attempt-id", value);
            }
        }
        let attempt_started = Instant::now();
        prepared.dispatch = execution.arrival.map(|arrival| {
            arrival.attempt(
                crate::observability::metrics::MetricKey::new(
                    provider.name.as_ref(),
                    &execution.request_model,
                    &original_model,
                    &execution.path,
                    prepared.downstream_stream,
                ),
                execution.request_id.clone(),
                attempt_id,
                &execution.api_key,
            )
        });
        emit_attempt(
            &execution.request_id,
            &execution.trace_id,
            &execution.api_key_role,
            attempt_index,
            &provider,
            &execution.request_model,
            &original_model,
            &execution.path,
            prepared.downstream_stream,
            &execution.method,
            &prepared.url,
            "started",
            None,
        );
        let upstream_url = prepared.url.clone();
        return Some(GenericHedgePlan {
            context: GenericHedgeContext {
                attempt_index,
                attempt_started,
                provider,
                original_model,
                wire_model: prepared.wire_model.clone(),
                provider_key: provider_key_raw,
                upstream_url,
                downstream_stream: prepared.downstream_stream,
            },
            prepared,
        });
    }
    None
}

pub(crate) fn emit_routing_skip(
    execution: &AttemptLoop,
    attempt_index: usize,
    provider: &Provider,
    original_model: &str,
    skip_reason: &str,
) {
    crate::observability::telemetry::emit(json!({
        "kind": "log",
        "fugue_table": "app_events",
        "event": "routing_attempt",
        "event_type": "routing_attempt",
        "severity": "info",
        "source": "uni-api-ember",
        "message": "uni-api-ember generic routing attempt",
        "request_id": execution.request_id,
        "trace_id": execution.trace_id,
        "path": execution.path,
        "path_template": execution.path,
        "route": format!("{} {}", execution.method, execution.path),
        "method": execution.method.as_str(),
        "model": execution.request_model,
        "provider": provider.name.as_ref(),
        "channel": provider.name.as_ref(),
        "role": execution.api_key_role,
        "actual_model": original_model,
        "attempt_id": format!("{}-r{}", execution.request_id, attempt_index + 1),
        "attempt_index": attempt_index + 1,
        "attempt_outcome": "skipped",
        "skip_reason": skip_reason,
        "streaming": false,
        "rust_generic_data_plane": true,
    }));
}

pub(crate) fn spawn_generic_hedge_attempt(
    scheduler: &mut HedgeScheduler<usize, GenericHedgeSuccess, GenericHedgeFailure>,
    execution: &AttemptLoop,
    plan: GenericHedgePlan,
) {
    let state = execution.state.clone();
    let incoming_headers = execution.headers.clone();
    let endpoint = execution.path.clone();
    let context = plan.context;
    let key = context.attempt_index;
    scheduler.spawn(key, move |trigger| async move {
        match send_attempt(
            &state,
            &context.provider,
            plan.prepared,
            &incoming_headers,
            &endpoint,
            None,
            Some(&trigger),
        )
        .await
        {
            Ok(success) => Ok(GenericHedgeSuccess { context, success }),
            Err(failure) => Err(GenericHedgeFailure { context, failure }),
        }
    });
}

pub(crate) async fn run_hedged_attempt_loop(
    execution: AttemptLoop,
    hedging: HedgingConfig,
) -> Response<Body> {
    let mut scheduler = HedgeScheduler::new(hedging.max_inflight_attempts);
    let mut pending = HashMap::<usize, GenericHedgeContext>::new();
    let mut cursor = 0usize;
    let mut trigger_count = 0usize;
    let mut cancelled_count = 0usize;
    let mut last_status = StatusCode::BAD_GATEWAY;
    let mut last_detail = String::from("No upstream attempt succeeded");
    let mut last_upstream_response = None;

    if let Some(plan) =
        next_generic_hedge_plan(&execution, &mut cursor, &mut last_status, &mut last_detail).await
    {
        pending.insert(plan.context.attempt_index, plan.context.clone());
        spawn_generic_hedge_attempt(&mut scheduler, &execution, plan);
    }

    while !scheduler.is_empty() {
        match scheduler.next_event().await {
            HedgeEvent::Triggered { key } => {
                debug_assert!(pending.contains_key(&key));
                trigger_count = trigger_count.saturating_add(1);
                if scheduler.has_capacity() {
                    if let Some(plan) = next_generic_hedge_plan(
                        &execution,
                        &mut cursor,
                        &mut last_status,
                        &mut last_detail,
                    )
                    .await
                    {
                        pending.insert(plan.context.attempt_index, plan.context.clone());
                        spawn_generic_hedge_attempt(&mut scheduler, &execution, plan);
                    }
                }
            }
            HedgeEvent::Succeeded { key, output } => {
                pending.remove(&key);
                let cancelled = scheduler.cancel_remaining();
                cancelled_count = cancelled_count.saturating_add(cancelled.len());
                for cancelled_key in cancelled {
                    if let Some(context) = pending.remove(&cancelled_key) {
                        emit_attempt(
                            &execution.request_id,
                            &execution.trace_id,
                            &execution.api_key_role,
                            context.attempt_index,
                            &context.provider,
                            &execution.request_model,
                            &context.original_model,
                            &execution.path,
                            context.downstream_stream,
                            &execution.method,
                            &context.upstream_url,
                            "cancelled",
                            None,
                        );
                    }
                }
                let GenericHedgeSuccess { context, success } = output;
                let AttemptSuccess {
                    response,
                    status,
                    usage,
                    fact_usage,
                    stream_outcome,
                    upstream_url,
                } = success;
                debug_assert!(stream_outcome.is_none());
                execution.state.persistence.record_channel(ChannelStat {
                    transport_timing: None,
                    duration_ms: Some(context.attempt_started.elapsed().as_secs_f64() * 1000.0),
                    first_output_ms: None,
                    response_created_ms: None,
                    first_text_ms: None,
                    request_id: execution.request_id.clone(),
                    attempt_id: format!("{}-r{}", execution.request_id, context.attempt_index + 1),
                    provider: context.provider.name.to_string(),
                    model: execution.request_model.clone(),
                    upstream_model: context.original_model.clone(),
                    api_key: execution.api_key.clone(),
                    provider_api_key: context.provider_key.clone(),
                    success: true,
                    endpoint: execution.path.clone(),
                    stream: context.downstream_stream,
                });
                execution
                    .state
                    .runtime
                    .reset_route_failure(&context.provider, &context.original_model)
                    .await;
                execution.state.persistence.record_request(RequestStat {
                    fact_usage,
                    stream: context.downstream_stream,
                    status: status.as_u16(),
                    request_id: execution.request_id.clone(),
                    trace_id: execution.trace_id.clone(),
                    endpoint: execution.path.clone(),
                    client_ip: execution.client_ip.clone(),
                    process_time: execution.started.elapsed().as_secs_f64(),
                    first_response_time: context.attempt_started.elapsed().as_secs_f64(),
                    provider: context.provider.name.to_string(),
                    model: execution.request_model.clone(),
                    upstream_model: context.original_model.clone(),
                    api_key: execution.api_key.clone(),
                    prompt_tokens: usage.0,
                    completion_tokens: usage.1,
                    total_tokens: usage.2,
                    prompt_price: execution.prompt_price,
                    completion_price: execution.completion_price,
                    timing_spans: json!({
                        "runtime": "rust",
                        "terminal": "hedge_winner",
                        "attempt_count": cursor,
                        "winner_attempt_index": context.attempt_index + 1,
                        "hedge_trigger_count": trigger_count,
                        "hedge_cancelled_attempt_count": cancelled_count,
                        "upstream_ms": context.attempt_started.elapsed().as_millis(),
                    })
                    .to_string(),
                    ..RequestStat::default()
                });
                emit_attempt(
                    &execution.request_id,
                    &execution.trace_id,
                    &execution.api_key_role,
                    context.attempt_index,
                    &context.provider,
                    &execution.request_model,
                    &context.original_model,
                    &execution.path,
                    context.downstream_stream,
                    &execution.method,
                    &upstream_url,
                    "completed",
                    Some(status.as_u16()),
                );
                return response;
            }
            HedgeEvent::Failed { key, failure } => {
                pending.remove(&key);
                let GenericHedgeFailure {
                    context,
                    mut failure,
                } = failure;
                let policy = classify_provider_failure(
                    failure.status.as_u16(),
                    &failure.detail,
                    Some(&context.provider),
                    &execution.path,
                    execution.auto_retry,
                    context.wire_model.as_deref(),
                );
                failure.status =
                    StatusCode::from_u16(policy.status).unwrap_or(StatusCode::BAD_GATEWAY);
                if let Some(response) = failure.response.as_mut() {
                    *response.status_mut() = failure.status;
                }
                last_status = failure.status;
                if policy.provider_model_unavailable {
                    last_detail = format!(
                        "All configured providers failed for model {}",
                        execution.request_model
                    );
                    last_upstream_response = None;
                } else {
                    last_detail = failure.detail.clone();
                    if let Some(response) = failure.response.take() {
                        last_upstream_response = Some(response);
                    }
                }
                execution.state.persistence.record_channel(ChannelStat {
                    transport_timing: None,
                    duration_ms: Some(context.attempt_started.elapsed().as_secs_f64() * 1000.0),
                    first_output_ms: None,
                    response_created_ms: None,
                    first_text_ms: None,
                    request_id: execution.request_id.clone(),
                    attempt_id: format!("{}-r{}", execution.request_id, context.attempt_index + 1),
                    provider: context.provider.name.to_string(),
                    model: execution.request_model.clone(),
                    upstream_model: context.original_model.clone(),
                    api_key: execution.api_key.clone(),
                    provider_api_key: context.provider_key.clone(),
                    success: false,
                    endpoint: execution.path.clone(),
                    stream: context.downstream_stream,
                });
                if !policy.request_scoped || policy.force_quota_cooldown {
                    execution
                        .state
                        .runtime
                        .cool_failed_route(FailedRoute {
                            provider: &context.provider,
                            key: &context.provider_key,
                            original_model: &context.original_model,
                            has_alternative: execution.providers.len() > 1,
                            status: failure.status.as_u16(),
                            detail: &failure.detail,
                            provider_model_unavailable: policy.provider_model_unavailable,
                            force_quota_cooldown: policy.force_quota_cooldown,
                        })
                        .await;
                }
                if context.provider.engine.eq_ignore_ascii_case("codex")
                    && context.provider_key.contains(',')
                    && matches!(failure.status.as_u16(), 401..=403)
                {
                    execution
                        .state
                        .codex_oauth
                        .clear(&context.provider_key)
                        .await;
                }
                emit_attempt(
                    &execution.request_id,
                    &execution.trace_id,
                    &execution.api_key_role,
                    context.attempt_index,
                    &context.provider,
                    &execution.request_model,
                    &context.original_model,
                    &execution.path,
                    context.downstream_stream,
                    &execution.method,
                    &failure.upstream_url,
                    "failed",
                    Some(failure.status.as_u16()),
                );
                if policy.retryable && scheduler.has_capacity() {
                    if let Some(plan) = next_generic_hedge_plan(
                        &execution,
                        &mut cursor,
                        &mut last_status,
                        &mut last_detail,
                    )
                    .await
                    {
                        pending.insert(plan.context.attempt_index, plan.context.clone());
                        spawn_generic_hedge_attempt(&mut scheduler, &execution, plan);
                    }
                }
            }
        }
    }

    execution.state.persistence.record_request(RequestStat {
        is_flagged: true,
        status: last_status.as_u16(),
        request_id: execution.request_id,
        trace_id: execution.trace_id,
        endpoint: execution.path,
        client_ip: execution.client_ip,
        process_time: execution.started.elapsed().as_secs_f64(),
        model: execution.request_model,
        api_key: execution.api_key,
        timing_spans: json!({
            "runtime":"rust",
            "terminal":"hedge_exhausted",
            "attempt_count": cursor,
            "hedge_trigger_count": trigger_count,
            "hedge_cancelled_attempt_count": cancelled_count,
        })
        .to_string(),
        ..RequestStat::default()
    });
    last_upstream_response.unwrap_or_else(|| json_error(last_status, &last_detail))
}

pub(crate) async fn run_attempt_loop(execution: AttemptLoop) -> Response<Body> {
    let AttemptLoop {
        state,
        started,
        arrival,
        method,
        uri,
        path,
        headers,
        request_id,
        trace_id,
        client_ip,
        api_key,
        api_key_role,
        request_model,
        video_task_route,
        providers,
        max_attempts,
        auto_retry,
        input,
        image_reservations: _image_reservations,
        prompt_price,
        completion_price,
        keepalive_updates,
    } = execution;
    let mut last_status = StatusCode::BAD_GATEWAY;
    let mut last_detail = String::from("No upstream attempt succeeded");
    let mut last_upstream_response = None;
    let mut upstream_failed = false;

    for attempt_index in 0..max_attempts {
        let provider = providers[attempt_index % providers.len()].clone();
        let Some(original_model) = provider.models.get(&request_model).cloned() else {
            continue;
        };
        let key_selection = if let Some(route) = video_task_route
            .as_ref()
            .filter(|route| route.provider_name == provider.name.as_ref())
        {
            ProviderKeySelection::Selected(route.provider_key.clone())
        } else {
            state
                .runtime
                .select_provider_key(&provider, &original_model)
                .await
        };
        let provider_key_raw = match key_selection {
            ProviderKeySelection::Selected(key) => key,
            ProviderKeySelection::NoProviderKey => {
                last_status = StatusCode::BAD_GATEWAY;
                last_detail = format!("Provider {} has no API key", provider.name);
                continue;
            }
            ProviderKeySelection::ChannelCooling | ProviderKeySelection::AllKeysCooling => {
                if !upstream_failed {
                    last_status = StatusCode::TOO_MANY_REQUESTS;
                    last_detail = "All matching provider routes are cooling down".into();
                }
                continue;
            }
        };
        let mut provider_key = provider_key_raw.clone();
        let mut codex_account_id = None;
        if provider.engine.eq_ignore_ascii_case("codex") && provider_key_raw.contains(',') {
            match state
                .codex_oauth
                .resolve(
                    &provider_key_raw,
                    provider.preferences.get("proxy").and_then(Value::as_str),
                )
                .await
            {
                Ok(auth) => {
                    provider_key = auth.bearer;
                    codex_account_id = auth.account_id;
                }
                Err(error) => {
                    last_status = StatusCode::UNAUTHORIZED;
                    last_detail = error;
                    continue;
                }
            }
        }
        let mut prepared = match build_attempt(
            &provider,
            &provider_key,
            &request_model,
            &original_model,
            &method,
            &uri,
            &path,
            &headers,
            &input,
            &request_id,
        ) {
            Ok(attempt) => attempt,
            Err(error) => {
                last_status = StatusCode::BAD_REQUEST;
                last_detail = error;
                continue;
            }
        };
        if let Some(account_id) = codex_account_id {
            if let Ok(value) = HeaderValue::from_str(&account_id) {
                prepared.headers.insert("chatgpt-account-id", value);
            }
        }
        let attempt_started = Instant::now();
        prepared.dispatch = arrival.map(|arrival| {
            arrival.attempt(
                crate::observability::metrics::MetricKey::new(
                    provider.name.as_ref(),
                    &request_model,
                    &original_model,
                    &path,
                    prepared.downstream_stream,
                ),
                request_id.clone(),
                format!("{request_id}-r{}", attempt_index + 1),
                &api_key,
            )
        });
        let downstream_stream = prepared.downstream_stream;
        let wire_model = prepared.wire_model.clone();
        emit_attempt(
            &request_id,
            &trace_id,
            &api_key_role,
            attempt_index,
            &provider,
            &request_model,
            &original_model,
            &path,
            downstream_stream,
            &method,
            &prepared.url,
            "started",
            None,
        );
        match send_attempt(
            &state,
            &provider,
            prepared,
            &headers,
            &path,
            keepalive_updates.as_ref(),
            None,
        )
        .await
        {
            Ok(success) => {
                let AttemptSuccess {
                    response,
                    status,
                    usage,
                    fact_usage,
                    stream_outcome,
                    upstream_url,
                } = success;
                let request_stat = RequestStat {
                    fact_usage,
                    stream: downstream_stream,
                    upstream_model: original_model.clone(),
                    status: status.as_u16(),
                    request_id: request_id.clone(),
                    trace_id: trace_id.clone(),
                    endpoint: path.clone(),
                    client_ip: client_ip.clone(),
                    process_time: started.elapsed().as_secs_f64(),
                    first_response_time: attempt_started.elapsed().as_secs_f64(),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    api_key: api_key.clone(),
                    prompt_tokens: usage.0,
                    completion_tokens: usage.1,
                    total_tokens: usage.2,
                    prompt_price,
                    completion_price,
                    timing_spans: json!({
                        "runtime": "rust",
                        "attempt_count": attempt_index + 1,
                        "upstream_ms": attempt_started.elapsed().as_millis(),
                    })
                    .to_string(),
                    ..RequestStat::default()
                };
                if let Some(stream_outcome) = stream_outcome {
                    let outcome_state = state.clone();
                    let outcome_provider = provider.clone();
                    let outcome_original_model = original_model.clone();
                    let outcome_provider_key = provider_key_raw.clone();
                    let outcome_request_id = request_id.clone();
                    let outcome_trace_id = trace_id.clone();
                    let outcome_role = api_key_role.clone();
                    let outcome_model = request_model.clone();
                    let outcome_api_key = api_key.clone();
                    let outcome_path = path.clone();
                    let outcome_method = method.clone();
                    let has_alternative = providers.len() > 1;
                    tokio::spawn(async move {
                        let outcome = stream_outcome.await.unwrap_or_else(|_| {
                            provider_stream::StreamOutcome {
                                usage: (0, 0, 0),
                                fact_usage: Default::default(),
                                success: false,
                                observational_only: false,
                                status_code: 502,
                                detail: "provider stream outcome was canceled".into(),
                                first_output_ms: None,
                                response_created_ms: None,
                                first_text_ms: None,
                            }
                        });
                        let mut request_stat = request_stat;
                        request_stat.process_time = started.elapsed().as_secs_f64();
                        request_stat.prompt_tokens = outcome.usage.0;
                        request_stat.completion_tokens = outcome.usage.1;
                        request_stat.total_tokens = outcome.usage.2;
                        request_stat.fact_usage = outcome.fact_usage.clone();
                        request_stat.first_output_ms = outcome.first_output_ms;
                        request_stat.response_created_ms = outcome.response_created_ms;
                        request_stat.first_text_ms = outcome.first_text_ms;
                        request_stat.status = outcome.status_code;
                        request_stat.is_flagged = !outcome.success;
                        request_stat.timing_spans = json!({
                            "runtime": "rust",
                            "attempt_count": attempt_index + 1,
                            "upstream_ms": attempt_started.elapsed().as_millis(),
                            "terminal": if outcome.success { "stream_completed" } else { "stream_failed" },
                            "status_code": outcome.status_code,
                            "first_output_ms": outcome.first_output_ms,
                        })
                        .to_string();
                        outcome_state.persistence.record_channel(ChannelStat {
                            transport_timing: None,
                            duration_ms: Some(attempt_started.elapsed().as_secs_f64() * 1000.0),
                            first_output_ms: outcome.first_output_ms,
                            response_created_ms: outcome.response_created_ms,
                            first_text_ms: outcome.first_text_ms,
                            request_id: outcome_request_id.clone(),
                            attempt_id: format!("{}-r{}", outcome_request_id, attempt_index + 1),
                            provider: outcome_provider.name.to_string(),
                            model: outcome_model.clone(),
                            upstream_model: outcome_original_model.clone(),
                            api_key: outcome_api_key,
                            provider_api_key: outcome_provider_key.clone(),
                            success: outcome.success,
                            endpoint: outcome_path.clone(),
                            stream: downstream_stream,
                        });
                        let recorded_status = if outcome.success {
                            if !outcome.observational_only {
                                outcome_state
                                    .runtime
                                    .reset_route_failure(&outcome_provider, &outcome_original_model)
                                    .await;
                            }
                            status.as_u16()
                        } else if outcome.observational_only || outcome.status_code == 499 {
                            outcome.status_code
                        } else {
                            let policy = classify_provider_failure(
                                outcome.status_code,
                                &outcome.detail,
                                Some(&outcome_provider),
                                &outcome_path,
                                auto_retry,
                                wire_model.as_deref(),
                            );
                            if !policy.request_scoped || policy.force_quota_cooldown {
                                outcome_state
                                    .runtime
                                    .cool_failed_route(FailedRoute {
                                        provider: &outcome_provider,
                                        key: &outcome_provider_key,
                                        original_model: &outcome_original_model,
                                        has_alternative,
                                        status: policy.status,
                                        detail: &outcome.detail,
                                        provider_model_unavailable: policy
                                            .provider_model_unavailable,
                                        force_quota_cooldown: policy.force_quota_cooldown,
                                    })
                                    .await;
                            }
                            if outcome_provider.engine.eq_ignore_ascii_case("codex")
                                && outcome_provider_key.contains(',')
                                && matches!(policy.status, 401..=403)
                            {
                                outcome_state.codex_oauth.clear(&outcome_provider_key).await;
                            }
                            policy.status
                        };
                        outcome_state.persistence.record_request(request_stat);
                        crate::observability::metrics::global().response_timings(
                            &outcome_provider.name,
                            &outcome_model,
                            &outcome_original_model,
                            &outcome_path,
                            downstream_stream,
                            outcome.response_created_ms,
                            outcome.first_text_ms,
                        );
                        emit_attempt_with_first_output(
                            &outcome_request_id,
                            &outcome_trace_id,
                            &outcome_role,
                            attempt_index,
                            &outcome_provider,
                            &outcome_model,
                            &outcome_original_model,
                            &outcome_path,
                            downstream_stream,
                            &outcome_method,
                            &upstream_url,
                            if outcome.success {
                                "completed"
                            } else {
                                "failed"
                            },
                            Some(recorded_status),
                            outcome.first_output_ms,
                        );
                    });
                    return response;
                }
                state.persistence.record_channel(ChannelStat {
                    transport_timing: None,
                    duration_ms: Some(attempt_started.elapsed().as_secs_f64() * 1000.0),
                    first_output_ms: None,
                    response_created_ms: None,
                    first_text_ms: None,
                    request_id: request_id.clone(),
                    attempt_id: format!("{}-r{}", request_id, attempt_index + 1),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    upstream_model: original_model.clone(),
                    api_key: api_key.clone(),
                    provider_api_key: provider_key_raw.clone(),
                    success: true,
                    endpoint: path.clone(),
                    stream: downstream_stream,
                });
                state
                    .runtime
                    .reset_route_failure(&provider, &original_model)
                    .await;
                state.persistence.record_request(request_stat);
                emit_attempt(
                    &request_id,
                    &trace_id,
                    &api_key_role,
                    attempt_index,
                    &provider,
                    &request_model,
                    &original_model,
                    &path,
                    downstream_stream,
                    &method,
                    &upstream_url,
                    "completed",
                    Some(status.as_u16()),
                );
                return response;
            }
            Err(mut failure) => {
                upstream_failed = true;
                let policy = classify_provider_failure(
                    failure.status.as_u16(),
                    &failure.detail,
                    Some(&provider),
                    &path,
                    auto_retry,
                    wire_model.as_deref(),
                );
                failure.status =
                    StatusCode::from_u16(policy.status).unwrap_or(StatusCode::BAD_GATEWAY);
                if let Some(response) = failure.response.as_mut() {
                    *response.status_mut() = failure.status;
                }
                last_status = failure.status;
                if policy.provider_model_unavailable {
                    last_detail =
                        format!("All configured providers failed for model {request_model}");
                    last_upstream_response = None;
                } else {
                    last_detail = failure.detail.clone();
                    if let Some(response) = failure.response.take() {
                        last_upstream_response = Some(response);
                    }
                }
                state.persistence.record_channel(ChannelStat {
                    transport_timing: None,
                    duration_ms: Some(attempt_started.elapsed().as_secs_f64() * 1000.0),
                    first_output_ms: None,
                    response_created_ms: None,
                    first_text_ms: None,
                    request_id: request_id.clone(),
                    attempt_id: format!("{}-r{}", request_id, attempt_index + 1),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    upstream_model: original_model.clone(),
                    api_key: api_key.clone(),
                    provider_api_key: provider_key_raw.clone(),
                    success: false,
                    endpoint: path.clone(),
                    stream: downstream_stream,
                });
                if !policy.request_scoped || policy.force_quota_cooldown {
                    state
                        .runtime
                        .cool_failed_route(FailedRoute {
                            provider: &provider,
                            key: &provider_key_raw,
                            original_model: &original_model,
                            has_alternative: providers.len() > 1,
                            status: failure.status.as_u16(),
                            detail: &failure.detail,
                            provider_model_unavailable: policy.provider_model_unavailable,
                            force_quota_cooldown: policy.force_quota_cooldown,
                        })
                        .await;
                }
                if provider.engine.eq_ignore_ascii_case("codex")
                    && provider_key_raw.contains(',')
                    && matches!(failure.status.as_u16(), 401..=403)
                {
                    state.codex_oauth.clear(&provider_key_raw).await;
                }
                emit_attempt(
                    &request_id,
                    &trace_id,
                    &api_key_role,
                    attempt_index,
                    &provider,
                    &request_model,
                    &original_model,
                    &path,
                    downstream_stream,
                    &method,
                    &failure.upstream_url,
                    "failed",
                    Some(failure.status.as_u16()),
                );
                if !policy.retryable {
                    break;
                }
            }
        }
    }

    state.persistence.record_request(RequestStat {
        is_flagged: true,
        status: last_status.as_u16(),
        stream: input
            .payload
            .as_ref()
            .and_then(|v| v.get("stream"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        request_id,
        trace_id,
        endpoint: path,
        client_ip,
        process_time: started.elapsed().as_secs_f64(),
        model: request_model,
        api_key,
        timing_spans: json!({"runtime":"rust","terminal":"route_exhausted"}).to_string(),
        ..RequestStat::default()
    });
    last_upstream_response.unwrap_or_else(|| json_error(last_status, &last_detail))
}

pub(crate) struct AttemptSuccess {
    pub(crate) response: Response<Body>,
    pub(crate) status: StatusCode,
    pub(crate) usage: (i64, i64, i64),
    pub(crate) fact_usage: crate::observability::usage::FactUsage,
    pub(crate) stream_outcome:
        Option<tokio::sync::oneshot::Receiver<provider_stream::StreamOutcome>>,
    pub(crate) upstream_url: String,
}

// Only attribution headers survive a protocol conversion. Forwarding content
// type/length or other representation headers would describe the wrong body.
pub(crate) fn copy_oaix_headers(source: &HeaderMap, target: &mut HeaderMap) {
    for (name, value) in source {
        if name.as_str().starts_with("x-oaix-") {
            target.append(name.clone(), value.clone());
        }
    }
}

pub(crate) struct AttemptFailure {
    pub(crate) status: StatusCode,
    pub(crate) detail: String,
    pub(crate) upstream_url: String,
    pub(crate) response: Option<Response<Body>>,
}

pub(crate) async fn send_attempt(
    state: &AppState,
    provider: &Provider,
    mut prepared: PreparedAttempt,
    incoming_headers: &HeaderMap,
    endpoint: &str,
    keepalive_updates: Option<&crate::protocols::chat::stream::KeepaliveUpdates>,
    hedge_trigger: Option<&HedgeTrigger<usize>>,
) -> Result<AttemptSuccess, AttemptFailure> {
    let proxy = provider.preferences.get("proxy").and_then(Value::as_str);
    let http1_only = provider.engine.eq_ignore_ascii_case("codex");
    let timeouts = state
        .runtime
        .generic_timeouts(
            incoming_headers,
            provider,
            &prepared.request_model,
            &prepared.original_model,
            provider.engine.as_ref(),
            prepared.upstream_stream,
            endpoint,
            prepared.method.as_str(),
        )
        .await;
    if let Some(updates) = keepalive_updates {
        let interval = state
            .runtime
            .keepalive_interval(provider, &prepared.request_model, &prepared.original_model)
            .await;
        updates.send_replace(interval);
    }
    let connect_timeout = positive_duration(timeouts.connect);
    let client = state
        .upstream_client(proxy, http1_only, connect_timeout)
        .await
        .map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: error,
            upstream_url: prepared.url.clone(),
            response: None,
        })?;
    // The resolved policy is also returned by the control-plane preview. Never
    // infer a second, hidden total timeout from model_timeout or the endpoint.
    let request_timeout = positive_duration(timeouts.total);
    if matches!(
        provider.engine.to_ascii_lowercase().as_str(),
        "vertex" | "vertex-gemini" | "vertex-claude"
    ) && provider.client_email.is_some()
        && provider.private_key.is_some()
    {
        let token = vertex_access_token(&state.backend_client, provider)
            .await
            .map_err(|error| AttemptFailure {
                status: StatusCode::BAD_GATEWAY,
                detail: error,
                upstream_url: prepared.url.clone(),
                response: None,
            })?;
        prepared.headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| AttemptFailure {
                status: StatusCode::BAD_GATEWAY,
                detail: "Vertex OAuth token is not a valid header value".into(),
                upstream_url: prepared.url.clone(),
                response: None,
            })?,
        );
    }
    let mut request = client
        .request(prepared.method.clone(), &prepared.url)
        .headers(prepared.headers.clone());
    if let Some(timeout) = request_timeout {
        request = request.timeout(timeout);
    }
    request = match prepared.body {
        AttemptBody::Json(body) => request.body(body),
        AttemptBody::Replay(storage, observation) => {
            let body = storage
                .into_body(&observation)
                .await
                .map_err(|error| AttemptFailure {
                    status: error.status,
                    detail: error.message,
                    upstream_url: prepared.url.clone(),
                    response: None,
                })?;
            request.body(reqwest::Body::wrap_stream(body.into_data_stream()))
        }
        AttemptBody::MultipartRewrite {
            storage,
            observation,
            source_content_type,
            boundary,
            model,
        } => request.body(
            multipart_rewrite_body(storage, observation, &source_content_type, boundary, model)
                .await
                .map_err(|detail| AttemptFailure {
                    status: StatusCode::BAD_REQUEST,
                    detail,
                    upstream_url: prepared.url.clone(),
                    response: None,
                })?,
        ),
        AttemptBody::DashscopeTranscription {
            storage,
            observation,
            source_content_type,
            model,
            provider_key,
        } => {
            let (headers, body) = prepare_dashscope_transcription(
                &client,
                prepared.headers.clone(),
                storage,
                observation,
                &source_content_type,
                &model,
                &provider_key,
            )
            .await
            .map_err(|detail| AttemptFailure {
                status: StatusCode::BAD_GATEWAY,
                detail,
                upstream_url: prepared.url.clone(),
                response: None,
            })?;
            let mut request = client
                .request(prepared.method.clone(), &prepared.url)
                .headers(headers);
            if let Some(timeout) = request_timeout {
                request = request.timeout(timeout);
            }
            request.body(body)
        }
        AttemptBody::Empty => request,
    };
    let billing_secret =
        crate::observability::billing::request_key(&prepared.headers, &prepared.url);
    if let Some(dispatch) = &prepared.dispatch {
        dispatch.billing.target(&prepared.url, &billing_secret);
        dispatch.record(&state.channel_metrics);
    }
    let send_started = Instant::now();
    let started = tokio::time::Instant::from_std(send_started);
    let response = await_with_hedge(
        request.send(),
        hedge_deadline(started, timeouts.first_byte),
        earliest_timeout(&[timeouts.write, timeouts.pool, timeouts.total])
            .map(|timeout| started + timeout),
        hedge_trigger,
    )
    .await
    .map_err(|_| AttemptFailure {
        status: StatusCode::GATEWAY_TIMEOUT,
        detail: "Upstream response headers timed out".into(),
        upstream_url: prepared.url.clone(),
        response: None,
    })?
    .output
    .map_err(|error| AttemptFailure {
        status: StatusCode::BAD_GATEWAY,
        detail: format!("Upstream transport error: {error}"),
        upstream_url: prepared.url.clone(),
        response: None,
    })?;
    if let Some(dispatch) = &prepared.dispatch {
        dispatch.billing.headers(
            response.headers(),
            response.status().as_u16(),
            &billing_secret,
        );
    }
    let status = response.status();
    let attribution_headers = response.headers().clone();
    if !status.is_success() {
        let headers = filtered_response_headers(response.headers());
        let body = read_limited_upstream_body_with_idle(
            response,
            UPSTREAM_ERROR_MAX_BYTES,
            positive_duration(timeouts.idle),
        )
        .await
        .unwrap_or_else(|error| Bytes::from(format!("read upstream error response: {error}")));
        if let Some(dispatch) = &prepared.dispatch {
            dispatch.billing.error_body(status.as_u16(), &body);
        }
        let detail = String::from_utf8_lossy(&body).into_owned();
        let mut output = Response::new(Body::from(body));
        *output.status_mut() = status;
        *output.headers_mut() = headers;
        output
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Err(AttemptFailure {
            status,
            detail: truncate_detail(&detail),
            upstream_url: prepared.url,
            response: Some(output),
        });
    }
    if prepared.adapter == ResponseAdapter::Passthrough
        && prepared.downstream_protocol == DownstreamProtocol::Native
        && prepared.upstream_stream
    {
        if response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("text/event-stream"))
        {
            let guarded_chat = endpoint == "/v1/chat/completions";
            // Chat routing is finalized by the stream outcome, not HTTP headers.
            if !guarded_chat {
                state
                    .runtime
                    .reset_route_failure(provider, &prepared.original_model)
                    .await;
            }
            let mut translation = crate::observability::passthrough::observe(
                response,
                send_started,
                guarded_chat,
                positive_duration(timeouts.idle),
            );
            if guarded_chat {
                translation = crate::protocols::chat::stream::preflight(
                    translation,
                    positive_duration(timeouts.first_byte)
                        .map(|duration| tokio::time::Instant::from_std(send_started + duration)),
                )
                .await
                .map_err(|failure| AttemptFailure {
                    status: StatusCode::from_u16(failure.status_code)
                        .unwrap_or(StatusCode::BAD_GATEWAY),
                    detail: failure.detail,
                    upstream_url: prepared.url.clone(),
                    response: None,
                })?;
            }
            translation
                .response
                .headers_mut()
                .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
            return Ok(AttemptSuccess {
                response: translation.response,
                status,
                usage: (0, 0, 0),
                fact_usage: Default::default(),
                stream_outcome: Some(translation.outcome),
                upstream_url: prepared.url,
            });
        }
        let headers = filtered_response_headers(response.headers());
        let mut output = Response::new(Body::from_stream(upstream_chunks(
            response,
            positive_duration(timeouts.idle),
        )));
        *output.status_mut() = status;
        *output.headers_mut() = headers;
        output
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Ok(AttemptSuccess {
            response: output,
            status,
            usage: (0, 0, 0),
            fact_usage: Default::default(),
            stream_outcome: None,
            upstream_url: prepared.url,
        });
    }
    if prepared.upstream_stream {
        let protocol = match prepared.adapter {
            ResponseAdapter::ResponsesToChat => StreamProtocol::Responses,
            ResponseAdapter::GeminiToChat => StreamProtocol::Gemini,
            ResponseAdapter::ClaudeToChat
                if provider.engine.eq_ignore_ascii_case("vertex-claude") =>
            {
                StreamProtocol::VertexClaude
            }
            ResponseAdapter::ClaudeToChat => StreamProtocol::Claude,
            ResponseAdapter::CohereToChat => StreamProtocol::Cohere,
            ResponseAdapter::CloudflareToChat => StreamProtocol::Cloudflare,
            ResponseAdapter::AwsToChat => StreamProtocol::AwsBedrock,
            ResponseAdapter::LingjingVideo | ResponseAdapter::CallxyqVideo => unreachable!(),
            ResponseAdapter::Passthrough => StreamProtocol::Chat,
            ResponseAdapter::Search => unreachable!(),
        };
        let output_protocol = if prepared.downstream_protocol == DownstreamProtocol::ResponsesCompat
        {
            StreamOutputProtocol::Responses
        } else {
            StreamOutputProtocol::Chat
        };
        let mut translation = if prepared.adapter == ResponseAdapter::ResponsesToChat {
            provider_stream::translate_responses_to_chat(
                response,
                output_protocol,
                prepared.request_model.clone(),
                prepared.chat_stream_include_usage,
                remaining_timeout(timeouts.first_byte, send_started.elapsed()),
                timeouts.idle,
                remaining_timeout(timeouts.total, send_started.elapsed()),
                false,
            )
            .await
            .map_err(|failure| AttemptFailure {
                status: StatusCode::from_u16(failure.status_code)
                    .unwrap_or(StatusCode::BAD_GATEWAY),
                detail: failure.detail,
                upstream_url: prepared.url.clone(),
                response: None,
            })?
        } else {
            provider_stream::translate(
                response,
                protocol,
                output_protocol,
                prepared.request_model.clone(),
                prepared.chat_stream_include_usage,
                timeouts.idle,
                remaining_timeout(timeouts.total, send_started.elapsed()),
            )
        };
        if endpoint == "/v1/chat/completions"
            && prepared.adapter != ResponseAdapter::ResponsesToChat
        {
            translation = crate::protocols::chat::stream::preflight(
                translation,
                positive_duration(timeouts.first_byte)
                    .map(|duration| tokio::time::Instant::from_std(send_started + duration)),
            )
            .await
            .map_err(|failure| AttemptFailure {
                status: StatusCode::from_u16(failure.status_code)
                    .unwrap_or(StatusCode::BAD_GATEWAY),
                detail: failure.detail,
                upstream_url: prepared.url.clone(),
                response: None,
            })?;
        }
        copy_oaix_headers(&attribution_headers, translation.response.headers_mut());
        translation
            .response
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Ok(AttemptSuccess {
            response: translation.response,
            status,
            usage: (0, 0, 0),
            fact_usage: Default::default(),
            stream_outcome: Some(translation.outcome),
            upstream_url: prepared.url,
        });
    }
    if prepared.adapter == ResponseAdapter::Passthrough
        && prepared.downstream_protocol == DownstreamProtocol::Native
    {
        let headers = filtered_response_headers(response.headers());
        let body = read_limited_upstream_body_with_idle(
            response,
            upstream_response_max_bytes(),
            positive_duration(timeouts.idle),
        )
        .await
        .map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: format!("Read upstream response failed: {error}"),
            upstream_url: prepared.url.clone(),
            response: None,
        })?;
        let parsed = serde_json::from_slice::<Value>(&body).ok();
        let fact_usage = crate::observability::usage::FactUsage::from_usage(
            parsed.as_ref().and_then(|v| v.get("usage")),
        );
        let usage = parsed.as_ref().map(usage).unwrap_or((0, 0, 0));
        let mut output = Response::new(Body::from(body));
        *output.status_mut() = status;
        *output.headers_mut() = headers;
        output
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Ok(AttemptSuccess {
            response: output,
            status,
            usage,
            fact_usage,
            stream_outcome: None,
            upstream_url: prepared.url,
        });
    }
    let upstream = if prepared.adapter == ResponseAdapter::Search {
        let bytes = read_limited_upstream_body_with_idle(
            response,
            upstream_response_max_bytes(),
            positive_duration(timeouts.idle),
        )
        .await
        .map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: format!("Read upstream response failed: {error}"),
            upstream_url: prepared.url.clone(),
            response: None,
        })?;
        serde_json::from_slice::<Value>(&bytes)
            .unwrap_or_else(|_| json!({"text":String::from_utf8_lossy(&bytes)}))
    } else {
        let bytes = read_limited_upstream_body_with_idle(
            response,
            upstream_response_max_bytes(),
            positive_duration(timeouts.idle),
        )
        .await
        .map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: format!("Read upstream response failed: {error}"),
            upstream_url: prepared.url.clone(),
            response: None,
        })?;
        serde_json::from_slice::<Value>(&bytes).map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: format!("Decode upstream response failed: {error}"),
            upstream_url: prepared.url.clone(),
            response: None,
        })?
    };
    let fact_usage = crate::observability::usage::FactUsage::from_usage(
        upstream
            .get("usage")
            .or_else(|| upstream.get("usageMetadata")),
    );
    let normalized = match prepared.adapter {
        ResponseAdapter::Search => normalize_search_response(&prepared.url, &upstream),
        ResponseAdapter::ResponsesToChat => responses_to_chat(&upstream, &prepared.original_model),
        ResponseAdapter::GeminiToChat => gemini_to_chat(&upstream, &prepared.original_model),
        ResponseAdapter::ClaudeToChat => claude_to_chat(&upstream, &prepared.original_model),
        ResponseAdapter::CohereToChat => cohere_to_chat(&upstream, &prepared.original_model),
        ResponseAdapter::CloudflareToChat => {
            cloudflare_to_chat(&upstream, &prepared.original_model)
        }
        ResponseAdapter::AwsToChat => claude_to_chat(&upstream, &prepared.original_model),
        ResponseAdapter::LingjingVideo => normalize_lingjing_video_response(
            &prepared.method,
            &prepared.request_model,
            &prepared.url,
            &upstream,
        ),
        ResponseAdapter::CallxyqVideo => normalize_callxyq_video_response(
            &prepared.method,
            &prepared.request_model,
            &upstream,
            prepared.estimated_video_tokens,
        ),
        ResponseAdapter::Passthrough => upstream,
    };
    let normalized = if prepared.downstream_protocol == DownstreamProtocol::ResponsesCompat {
        chat_to_responses_response(&normalized, &prepared.request_model)
    } else {
        normalized
    };
    if matches!(
        prepared.adapter,
        ResponseAdapter::LingjingVideo | ResponseAdapter::CallxyqVideo
    ) && prepared.method == Method::POST
    {
        if let Some(task_id) = normalized.get("id").and_then(Value::as_str) {
            remember_video_task(
                task_id,
                provider.name.as_ref(),
                &prepared.request_model,
                &prepared.provider_key,
                prepared.estimated_video_tokens,
            );
        }
    }
    let usage = usage(&normalized);
    let mut output = if prepared.downstream_stream {
        if prepared.downstream_protocol == DownstreamProtocol::ResponsesCompat {
            synthetic_responses_stream(normalized)
        } else {
            synthetic_chat_stream(normalized)
        }
    } else {
        json_response(StatusCode::OK, normalized)
    };
    copy_oaix_headers(&attribution_headers, output.headers_mut());
    output
        .headers_mut()
        .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
    Ok(AttemptSuccess {
        response: output,
        status: StatusCode::OK,
        usage,
        fact_usage,
        stream_outcome: None,
        upstream_url: prepared.url,
    })
}

pub(crate) fn usage(value: &Value) -> (i64, i64, i64) {
    let usage = value.get("usage").unwrap_or(&Value::Null);
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(prompt.saturating_add(completion));
    (prompt, completion, total)
}

pub(crate) fn query_value(uri: &Uri, name: &str) -> Option<String> {
    url::form_urlencoded::parse(uri.query()?.as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
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

pub(crate) fn truncate_detail(value: &str) -> String {
    value.chars().take(4096).collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_attempt(
    request_id: &str,
    trace_id: &str,
    role: &str,
    attempt_index: usize,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    endpoint: &str,
    downstream_stream: bool,
    method: &Method,
    url: &str,
    outcome: &str,
    status: Option<u16>,
) {
    emit_attempt_with_first_output(
        request_id,
        trace_id,
        role,
        attempt_index,
        provider,
        request_model,
        original_model,
        endpoint,
        downstream_stream,
        method,
        url,
        outcome,
        status,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_attempt_with_first_output(
    request_id: &str,
    trace_id: &str,
    role: &str,
    attempt_index: usize,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    endpoint: &str,
    downstream_stream: bool,
    method: &Method,
    url: &str,
    outcome: &str,
    status: Option<u16>,
    first_output_ms: Option<f64>,
) {
    let metrics = crate::observability::metrics::global();
    let upstream_model = original_model;
    if outcome == "started" {
        metrics.start(
            provider.name.as_ref(),
            request_model,
            upstream_model,
            endpoint,
            downstream_stream,
        );
    } else {
        metrics.finish(
            provider.name.as_ref(),
            request_model,
            upstream_model,
            endpoint,
            downstream_stream,
            outcome,
            None,
            first_output_ms,
        );
    }
    let upstream_host = Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default();
    let event = if outcome == "started" {
        "routing_attempt"
    } else {
        "upstream_attempt"
    };
    let attempt_success = match outcome {
        "completed" => Some(true),
        "failed" | "cancelled" => Some(false),
        _ => None,
    };
    crate::observability::telemetry::emit(json!({
            "kind": "log",
            "fugue_table": "app_events",
            "event": event,
            "event_type": event,
            "severity": if status.is_some_and(|status| status >= 400) { "warn" } else { "info" },
            "source": "uni-api-ember",
            "message": format!("uni-api-ember generic {event}"),
            "request_id": request_id,
            "trace_id": trace_id,
            "path": endpoint,
            "streaming": downstream_stream,
            "path_template": endpoint,
            "route": format!("{} {endpoint}", method.as_str()),
            "method": method.as_str(),
            "role": role,
            "attempt_id": format!("{request_id}-r{}", attempt_index + 1),
            "attempt_index": attempt_index + 1,
            "provider": provider.name.as_ref(),
            "channel": provider.name.as_ref(),
            "model": request_model,
            "request_model": request_model,
            "actual_model": original_model,
            "upstream_host": upstream_host,
            "attempt_outcome": outcome,
            "attempt_status_code": status,
            "attempt_success": attempt_success,
            "outcome": outcome,
            "status_code": status,
            "rust_generic_data_plane": true,
        }
    ));
}

pub(crate) fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    let mut response = Response::new(Body::from(value.to_string()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    response
}

#[cfg(test)]
mod tests;
