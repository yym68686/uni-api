use crate::observability::stream::StreamStats;
use crate::protocols::responses::events::encode_event;
use crate::protocols::responses::events::item_has_output;
use crate::protocols::responses::events::responses_semantic_error;
use crate::protocols::responses::events::Terminal;
use crate::protocols::responses::processor::frame_is_comment_only;
use crate::protocols::responses::processor::inspect_terminal_frame;
use crate::protocols::responses::processor::selective_rewrite_candidate;
use crate::protocols::responses::processor::terminal_candidate;
use crate::protocols::responses::processor::ResponsesProcessor;
use crate::protocols::sse::SseDecoder;
use crate::protocols::sse::SseFrame;
use crate::protocols::sse::UNLIMITED_SSE_EVENT_BYTES;

pub(crate) mod prepare;
pub(crate) mod route;

use crate::protocols::responses::item_ids::ResponsesItemIdNormalizer;
use crate::routing::timeouts::{earliest_timeout, positive_duration};
use crate::runtime::context::AppState;
use crate::runtime::idempotency;
use crate::transport::http::filtered_response_headers;
use crate::transport::http::json_error;
use crate::upstream::hedging::HedgeEvent;
use crate::upstream::hedging::HedgeScheduler;
use crate::upstream::hedging::HedgeTrigger;
use crate::upstream::responses::route::ResponsesRoute;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";

const MAX_ERROR_BODY_BYTES: usize = 1024 * 1024;

const DOWNSTREAM_SEGMENT_BYTES: usize = 64 * 1024;

const DOWNSTREAM_CHANNEL_SEGMENTS: usize = 16;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Plan {
    #[serde(skip)]
    pub(crate) dispatch: Option<crate::observability::timing::AttemptDispatch>,
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
}

enum PreflightResult {
    Retry(Value),
    HttpError(Box<Plan>, Value),
    Started(ActiveAttempt),
}

enum Coordinator {
    Python { session_id: String },
    Native { route: ResponsesRoute },
}

impl Coordinator {
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

    async fn retry(&mut self, state: &AppState, outcome: Value) -> Result<RetryResolution, String> {
        match self {
            Self::Python { session_id } => {
                retry_after_public_start_python(state, session_id, outcome).await
            }
            Self::Native { route } => {
                let mut retryable = route.record_failure(&outcome).await;
                loop {
                    if !retryable {
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
                        Ok(PreflightResult::Retry(next)) => {
                            retryable = route.record_failure(&next).await;
                        }
                        Ok(PreflightResult::HttpError(plan, next)) => {
                            retryable = route.record_plan_failure(*plan, &next).await;
                        }
                        Err(error) => {
                            let outcome = json!({
                                "kind": "protocol_error",
                                "status_code": 502,
                                "detail": error,
                                "committed": false,
                            });
                            retryable = route.record_failure(&outcome).await;
                        }
                    }
                }
            }
        }
    }
}

fn spawn_hedge_attempt(
    scheduler: &mut HedgeScheduler<String, (Plan, ActiveAttempt), (Plan, Value)>,
    state: AppState,
    plan: Plan,
) {
    let attempt_id = plan.attempt_id.clone();
    scheduler.spawn(attempt_id, move |trigger| async move {
        match preflight_attempt_with_trigger(&state, plan.clone(), false, Some(&trigger)).await {
            Ok(PreflightResult::Started(active)) => Ok((plan, active)),
            Ok(PreflightResult::Retry(outcome)) => Err((plan, outcome)),
            Ok(PreflightResult::HttpError(plan, outcome)) => Err((*plan, outcome)),
            Err(error) => Err((
                plan,
                json!({
                    "kind": "protocol_error",
                    "status_code": 502,
                    "detail": error,
                    "committed": false,
                }),
            )),
        }
    });
}

async fn preflight_native_hedged(
    state: &AppState,
    route: &mut ResponsesRoute,
) -> Result<Option<ActiveAttempt>, String> {
    let max_inflight = route.hedge_slots().max(1);
    let mut scheduler = HedgeScheduler::new(max_inflight);

    let Some(plan) = route.next_plan().await? else {
        return Ok(None);
    };
    spawn_hedge_attempt(&mut scheduler, state.clone(), plan);

    loop {
        match scheduler.next_event().await {
            HedgeEvent::Triggered { key: _ } => {
                route.record_hedge_trigger();
                if scheduler.has_capacity() {
                    if let Some(next) = route.next_plan().await? {
                        spawn_hedge_attempt(&mut scheduler, state.clone(), next);
                    }
                }
            }
            HedgeEvent::Succeeded {
                key: _,
                output: (plan, active),
            } => {
                route.record_hedge_cancellations(scheduler.cancel_remaining().len());
                route.set_current_plan(&plan);
                return Ok(Some(active));
            }
            HedgeEvent::Failed {
                key: _,
                failure: (plan, outcome),
            } => {
                let retryable = route.record_plan_failure(plan, &outcome).await;
                if retryable && scheduler.has_capacity() {
                    if let Some(next) = route.next_plan().await? {
                        spawn_hedge_attempt(&mut scheduler, state.clone(), next);
                    }
                }
                if scheduler.is_empty() {
                    return Ok(None);
                }
            }
        }
    }
}

pub async fn serve_native(
    state: AppState,
    mut route: ResponsesRoute,
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
                return json_error(status, &route.response_detail());
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
                return json_error(status, &route.response_detail());
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
                    return json_error(status, &route.response_detail());
                }
            }
            Ok(PreflightResult::HttpError(plan, outcome)) => {
                if !route.record_plan_failure(*plan, &outcome).await {
                    let status = StatusCode::from_u16(route.last_status())
                        .unwrap_or(StatusCode::BAD_GATEWAY);
                    route.emit_final_response(status.as_u16(), "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(status, &route.response_detail());
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
                    return json_error(StatusCode::BAD_GATEWAY, &route.response_detail());
                }
            }
        }
    }
}

async fn serve_native_nonstream(
    state: AppState,
    mut route: ResponsesRoute,
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
                return json_error(status, &route.response_detail());
            }
            Err(error) => {
                route.emit_internal_failure(400, "native_plan_error", &error);
                release_owner(&mut idempotency_owner).await;
                return json_error(StatusCode::BAD_REQUEST, &error);
            }
        };
        match send_native_nonstream_attempt(&state, &plan).await {
            Ok((status, mut headers, mut body, elapsed_ms)) if status.is_success() => {
                let first_output_ms = serde_json::from_slice::<Value>(&body)
                    .ok()
                    .filter(|p| {
                        p.get("output")
                            .and_then(Value::as_array)
                            .is_some_and(|items| items.iter().any(item_has_output))
                    })
                    .map(|_| elapsed_ms);
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
                    "first_output_ms":first_output_ms,
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
            Ok((status, _headers, body, _elapsed_ms)) => {
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
                if !route.record_plan_failure(plan, &outcome).await {
                    let final_status = StatusCode::from_u16(route.last_status())
                        .unwrap_or(StatusCode::BAD_GATEWAY);
                    route.emit_final_response(final_status.as_u16(), "failed_before_commit");
                    release_owner(&mut idempotency_owner).await;
                    return json_error(final_status, &route.response_detail());
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
                    return json_error(StatusCode::BAD_GATEWAY, &route.response_detail());
                }
            }
        }
    }
}

async fn send_native_nonstream_attempt(
    state: &AppState,
    plan: &Plan,
) -> Result<(StatusCode, HeaderMap, Vec<u8>, f64), String> {
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
    let billing_secret = crate::observability::billing::request_key(&headers, &plan.url);
    let mut request = client
        .post(&plan.url)
        .headers(headers)
        .body(plan.body.clone());
    // RequestBuilder::timeout covers both headers and the complete body. A
    // timeout around send() alone stops protecting the request after headers.
    if let Some(total) = positive_duration(plan.total_timeout_seconds) {
        request = request.timeout(total);
    }
    if let Some(dispatch) = &plan.dispatch {
        dispatch.billing.target(&plan.url, &billing_secret);
        dispatch.record(&state.channel_metrics);
    }
    let observation_started = tokio::time::Instant::now();
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
    if let Some(dispatch) = &plan.dispatch {
        dispatch.billing.headers(
            response.headers(),
            response.status().as_u16(),
            &billing_secret,
        );
    }
    let status = response.status();
    let headers = filtered_response_headers(response.headers());
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    loop {
        let next = if let Some(idle) = positive_duration(plan.idle_timeout_seconds) {
            tokio::time::timeout(idle, stream.next())
                .await
                .map_err(|_| "upstream non-streaming body idle timed out".to_owned())?
        } else {
            stream.next().await
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| format!("read upstream non-streaming body: {error}"))?;
        body.extend_from_slice(&chunk);
    }
    if let Some(dispatch) = &plan.dispatch {
        dispatch.billing.error_body(status.as_u16(), &body);
    }
    Ok((
        status,
        headers,
        body,
        observation_started.elapsed().as_secs_f64() * 1000.0,
    ))
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
            Ok(
                PreflightResult::Retry(mut outcome) | PreflightResult::HttpError(_, mut outcome),
            ) => {
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
    trigger: Option<&HedgeTrigger<String>>,
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
    let billing_secret = crate::observability::billing::request_key(&headers, &plan.url);
    let request = client
        .post(&plan.url)
        .headers(headers)
        .body(plan.body.clone());
    if let Some(dispatch) = &plan.dispatch {
        dispatch.billing.target(&plan.url, &billing_secret);
        dispatch.record(&state.channel_metrics);
    }
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
                    trigger.fire();
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
    let headers_received_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let mut stats = StreamStats::new(&plan.attempt_id);
    stats.started_at = started_at;
    stats.headers_received_ms = Some(headers_received_ms);
    stats.observe_response_connection(&response);
    if let Some(dispatch) = &plan.dispatch {
        dispatch.billing.headers(
            response.headers(),
            response.status().as_u16(),
            &billing_secret,
        );
    }
    let status = response.status();
    let unsupported_encoding = response
        .headers()
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"));
    let response_headers = filtered_response_headers(response.headers());
    if !status.is_success() {
        let error_read_started = tokio::time::Instant::now();
        let body = read_limited_body(
            response,
            total_deadline,
            positive_duration(plan.idle_timeout_seconds),
        )
        .await;
        stats.error_body_read_ms = Some(error_read_started.elapsed().as_secs_f64() * 1000.0);
        if let Some(dispatch) = &plan.dispatch {
            dispatch
                .billing
                .error_body(status.as_u16(), body.as_bytes());
        }
        return Ok(PreflightResult::HttpError(
            Box::new(plan),
            json!({
                "kind": "http_error",
                "status_code": status.as_u16(),
                "upstream_status_code": status.as_u16(),
                "body": body,
                "transport_timing": stats.transport_timing(),
                "committed": false,
            }),
        ));
    }
    if unsupported_encoding {
        return Ok(PreflightResult::Retry(json!({
            "kind": "protocol_error",
            "status_code": 502,
            "upstream_status_code": status.as_u16(),
            "detail": "Responses upstream ignored Accept-Encoding: identity",
            "transport_timing": stats.transport_timing(),
            "committed": false,
        })));
    }

    let mode = StreamMode::for_plan(&plan);
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
    };

    if active.mode == StreamMode::OpaqueRaw {
        active.commit_reason = "upstream_http_200";
        active.business_committed = true;
        return Ok(PreflightResult::Started(active));
    }
    let mut idle_deadline = deadline(
        tokio::time::Instant::now(),
        active.plan.idle_timeout_seconds,
    );
    loop {
        let hard_deadline = earlier_deadline(idle_deadline, active.total_deadline);
        let next_deadline = earlier_deadline(
            (!hedge_triggered).then_some(first_deadline).flatten(),
            hard_deadline,
        );
        let read_started = tokio::time::Instant::now();
        let read_result = await_deadline(active.stream.next(), next_deadline).await;
        active.stats.preflight_read_wait_ms += read_started.elapsed().as_secs_f64() * 1000.0;
        active.stats.preflight_read_calls = active.stats.preflight_read_calls.saturating_add(1);
        let chunk = match read_result {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(error))) => {
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 502,
                    "upstream_status_code": status.as_u16(),
                    "detail": format!("upstream stream read failed: {error}"),
                    "transport_timing": active.stats.transport_timing(),
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
                    "transport_timing": active.stats.transport_timing(),
                    "committed": false,
                })));
            }
            Err(error) => {
                if trigger.is_some()
                    && first_deadline
                        .is_some_and(|first| hard_deadline.is_none_or(|hard| first < hard))
                    && !hedge_triggered
                    && error == "upstream deadline exceeded"
                {
                    hedge_triggered = true;
                    if let Some(trigger) = trigger {
                        trigger.fire();
                    }
                    continue;
                }
                return Ok(PreflightResult::Retry(json!({
                    "kind": "transport_error",
                    "status_code": 504,
                    "upstream_status_code": status.as_u16(),
                    "detail": error,
                    "transport_timing": active.stats.transport_timing(),
                    "committed": false,
                })));
            }
        };
        idle_deadline = deadline(
            tokio::time::Instant::now(),
            active.plan.idle_timeout_seconds,
        );
        active.stats.observe_upstream(&chunk);
        let processing_started = tokio::time::Instant::now();
        let frames = active.decoder.feed(&chunk)?;
        active.stats.preflight_decode_ms += processing_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(result) = process_preflight_frames(&mut active, frames)? {
            return Ok(result);
        }
    }
}

fn process_preflight_frames(
    active: &mut ActiveAttempt,
    frames: Vec<SseFrame>,
) -> Result<Option<PreflightResult>, String> {
    let started = tokio::time::Instant::now();
    let mut result = process_preflight_frames_inner(active, frames);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match &mut result {
        Ok(Some(PreflightResult::Started(next))) => next.stats.preflight_process_ms += elapsed_ms,
        Ok(Some(PreflightResult::Retry(outcome))) => {
            active.stats.preflight_process_ms += elapsed_ms;
            outcome["transport_timing"] = json!(active.stats.transport_timing());
        }
        _ => active.stats.preflight_process_ms += elapsed_ms,
    }
    result
}

fn process_preflight_frames_inner(
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
    };
    Ok(std::mem::replace(active, placeholder))
}

fn start_public_stream(
    state: AppState,
    coordinator: Coordinator,
    mut active: ActiveAttempt,
    control_headers: HeaderMap,
    control_drain: Option<tokio::task::JoinHandle<()>>,
    idempotency_owner: Option<idempotency::Owner>,
) -> Response<Body> {
    active.stats.public_stream_ready_ms =
        Some(active.stats.started_at.elapsed().as_secs_f64() * 1000.0);
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

    async fn finish(mut self, cacheable: bool, transport_failed: bool) {
        // A transport timeout must not look like a clean HTTP EOF. Preserve the
        // existing semantic-failure protocol and do not invent frames or replay.
        if transport_failed && self.downstream_open && !self.cancellation.is_cancelled() {
            let _ = tokio::time::timeout(
                downstream_write_timeout(),
                self.sender.send(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "upstream stream ended without a successful terminal event",
                ))),
            )
            .await;
        }
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
            if let Terminal::SemanticFailure {
                event_type,
                payload,
            } = terminal
            {
                let outcome = semantic_failure_outcome(&active, &event_type, &payload, true);
                let _ = coordinator.complete(&state, &outcome).await;
                break 'request false;
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
    output
        .finish(cacheable, active.stats.transport_failed)
        .await;
}

async fn run_raw_committed(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    output: &mut OutputSink,
    cancellation: &CancellationToken,
) -> bool {
    active.stats.begin_raw_observation();
    loop {
        let idle_deadline = deadline(
            tokio::time::Instant::now(),
            active.plan.idle_timeout_seconds,
        );
        let next_deadline = earlier_deadline(idle_deadline, active.total_deadline);
        let read_started = tokio::time::Instant::now();
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
        active
            .stats
            .raw_read_wait(read_started.elapsed().as_secs_f64() * 1000.0);
        match next {
            Ok(Some(Ok(chunk))) => {
                active.stats.observe_upstream(&chunk);
                let process_started = tokio::time::Instant::now();
                let inspection = match active.decoder.feed(&chunk) {
                    Ok(frames) => raw_terminal_prefix(frames, &mut active.stats),
                    Err(error) => Err(error),
                };
                active.stats.raw_processed(
                    process_started.elapsed().as_secs_f64() * 1000.0,
                    active.decoder.buffer.len(),
                );
                match inspection {
                    Ok((prefix, terminal)) => {
                        if !prefix.is_empty() {
                            active.stats.observe_wire(&prefix);
                        }
                        if !prefix.is_empty()
                            && send_raw_wire(output, prefix, &mut active.stats)
                                .await
                                .is_err()
                        {
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
                        if let Some(terminal) = terminal {
                            return finish_observed_terminal(state, coordinator, active, terminal)
                                .await;
                        }
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
                        return false;
                    }
                }
            }
            Ok(None) => {
                let process_started = tokio::time::Instant::now();
                let inspection = match active.decoder.finish() {
                    Ok(frames) => raw_terminal_prefix(frames, &mut active.stats),
                    Err(error) => Err(error),
                };
                active.stats.raw_processed(
                    process_started.elapsed().as_secs_f64() * 1000.0,
                    active.decoder.buffer.len(),
                );
                match inspection {
                    Ok((prefix, terminal)) => {
                        if !prefix.is_empty() {
                            active.stats.observe_wire(&prefix);
                        }
                        if !prefix.is_empty()
                            && send_raw_wire(output, prefix, &mut active.stats)
                                .await
                                .is_err()
                        {
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
                        if let Some(terminal) = terminal {
                            return finish_observed_terminal(state, coordinator, active, terminal)
                                .await;
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

async fn send_raw_wire(
    output: &mut OutputSink,
    wire: Bytes,
    stats: &mut StreamStats,
) -> Result<(), ()> {
    let started = tokio::time::Instant::now();
    let result = output.send_wire(wire).await;
    stats.raw_output_wait(started.elapsed().as_secs_f64() * 1000.0);
    result
}

/// Return only the safe wire prefix before the first terminal event.
///
/// Upstream providers sometimes append heartbeats or unrelated bytes after
/// `response.completed` in the same HTTP chunk. Raw forwarding must stop at
/// the semantic terminal instead of sending that suffix first. A semantic
/// failure is omitted from the returned wire prefix so the downstream stream
/// closes without exposing the provider's error event.
fn raw_terminal_prefix(
    frames: Vec<SseFrame>,
    stats: &mut StreamStats,
) -> Result<(Bytes, Option<Terminal>), String> {
    let mut prefix = BytesMut::new();
    for frame in frames {
        if let Some(raw) = stats.raw_stream.as_mut() {
            raw.totals.frames = raw.totals.frames.saturating_add(1);
            if frame_is_comment_only(frame.raw()) {
                raw.totals.comment_frames = raw.totals.comment_frames.saturating_add(1);
            }
        }
        let wire = frame.wire.clone();
        let terminal = inspect_terminal_frame(&frame, stats)?;
        if let Some(terminal) = terminal {
            // A semantic failure after committed output is an internal
            // terminal only. Do not leak the provider's error event to the
            // downstream SSE client; the transport closes after the already
            // emitted content.
            if !matches!(terminal, Terminal::SemanticFailure { .. }) {
                prefix.extend_from_slice(&wire);
            }
            return Ok((prefix.freeze(), Some(terminal)));
        }
        prefix.extend_from_slice(&wire);
    }
    Ok((prefix.freeze(), None))
}

async fn finish_observed_terminal(
    state: &AppState,
    coordinator: &mut Coordinator,
    active: &mut ActiveAttempt,
    terminal: Terminal,
) -> bool {
    if let Terminal::SemanticFailure {
        event_type,
        payload,
    } = &terminal
    {
        let outcome = semantic_failure_outcome(active, event_type, payload, true);
        let _ = coordinator.complete(state, &outcome).await;
        return false;
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
    true
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
            let _ = inspect_terminal_frame(&frame, &mut active.stats);
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
            Ok(
                PreflightResult::Retry(mut next_outcome)
                | PreflightResult::HttpError(_, mut next_outcome),
            ) => {
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
            let _ = coordinator.complete(state, &outcome).await;
            return ActiveFrameResult::Done(false);
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
    stats.transport_failed = kind == "transport_error";
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
    idle: Option<Duration>,
) -> String {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while body.len() < MAX_ERROR_BODY_BYTES {
        let idle_deadline = idle.and_then(|idle| tokio::time::Instant::now().checked_add(idle));
        match await_deadline(stream.next(), earlier_deadline(idle_deadline, deadline)).await {
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
    crate::upstream::hedging::deadline(started, seconds)
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
    use crate::protocols::responses::processor::declared_event_bytes;

    fn test_active(engine: &str) -> ActiveAttempt {
        let plan = Plan {
            dispatch: None,
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
        }
    }

    #[tokio::test]
    async fn raw_stage_observation_distinguishes_output_queue_wait() {
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(Ok(Bytes::from_static(b"occupied")))
            .await
            .unwrap();
        let mut output = OutputSink::new(sender, CancellationToken::new(), None);
        let release = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            assert_eq!(receiver.recv().await.unwrap().unwrap(), "occupied");
            receiver.recv().await.unwrap().unwrap()
        });
        let mut stats = StreamStats::new("queue-wait");
        stats.begin_raw_observation();
        send_raw_wire(&mut output, Bytes::from_static(b"unchanged"), &mut stats)
            .await
            .unwrap();
        assert_eq!(release.await.unwrap(), "unchanged");
        let raw = stats.raw_stream.unwrap();
        assert!(raw.totals.output_send_ms >= 40.0);
        assert_eq!(raw.totals.output_calls, 1);
        assert_eq!(raw.totals.upstream_read_wait_ms, 0.0);
        assert!(raw.at_response_created.is_none());
    }

    #[test]
    fn raw_stream_observes_semantic_latency_without_rewriting_or_counting_keepalive() {
        let mut stats = StreamStats::new("latency");
        stats.started_at = tokio::time::Instant::now() - Duration::from_millis(400);
        let mut decoder = SseDecoder::new(4096);
        let keepalive =
            b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n";
        let (wire, _) = raw_terminal_prefix(decoder.feed(keepalive).unwrap(), &mut stats).unwrap();
        assert_eq!(wire.as_ref(), keepalive);
        assert!(stats.first_output_ms.is_none());
        let created = stats.response_created_ms.unwrap();
        assert!(created >= 400.0);
        assert!(stats.first_text_ms.is_none());
        stats.started_at = tokio::time::Instant::now() - Duration::from_millis(800);
        let delta = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"test\"}\n\n";
        let (wire, _) = raw_terminal_prefix(decoder.feed(delta).unwrap(), &mut stats).unwrap();
        assert_eq!(wire.as_ref(), delta);
        let first = stats.first_output_ms.unwrap();
        assert!(first >= 800.0);
        let first_text = stats.first_text_ms.unwrap();
        assert!(first_text >= 800.0 && first_text <= first);
        assert_eq!(stats.response_created_ms, Some(created));
        raw_terminal_prefix(decoder.feed(delta).unwrap(), &mut stats).unwrap();
        assert_eq!(stats.first_output_ms, Some(first));
        assert_eq!(stats.first_text_ms, Some(first_text));
        let mut failed = StreamStats::new("failure");
        let body=b"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n";
        raw_terminal_prefix(decoder.feed(body).unwrap(), &mut failed).unwrap();
        assert!(failed.first_output_ms.is_none());
        assert!(failed.response_created_ms.is_none());
        assert!(failed.first_text_ms.is_none());
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
        let (_prefix, terminal) = raw_terminal_prefix(frames, &mut stats).unwrap();
        let terminal = terminal.expect("completed terminal expected");
        assert!(matches!(terminal, Terminal::Completed));
        assert_eq!(stats.event_count, 2);
        assert_eq!(stats.delta_events, 1);
        assert_eq!(stats.usage.as_ref().unwrap()["total_tokens"], 3);
    }

    #[test]
    fn raw_mode_stops_at_completed_with_same_chunk_suffix() {
        let mut decoder = SseDecoder::new(4096);
        let frames = decoder
            .feed(
                b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n\
                  event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n\
                  : provider heartbeat after completed\n\n",
            )
            .unwrap();
        let mut stats = StreamStats::new("raw-terminal-prefix-test");
        let (prefix, terminal) = raw_terminal_prefix(frames, &mut stats).unwrap();
        let terminal = terminal.expect("completed must terminate raw forwarding");
        let wire = String::from_utf8(prefix.to_vec()).unwrap();
        assert!(matches!(terminal, Terminal::Completed));
        assert!(wire.contains("event: response.completed"));
        assert!(!wire.contains("provider heartbeat after completed"));
    }

    #[test]
    fn raw_mode_filters_semantic_failure_after_committed_output() {
        let mut decoder = SseDecoder::new(4096);
        let frames = decoder
            .feed(
                b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"test\"}\n\n\\
                  event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\",\"code\":\"server_error\",\"message\":\"temporarily unavailable\"}}\n\n",
            )
            .unwrap();
        let mut stats = StreamStats::new("raw-semantic-failure-test");
        let (prefix, terminal) = raw_terminal_prefix(frames, &mut stats).unwrap();
        let terminal = terminal.expect("semantic terminal expected");
        let wire = String::from_utf8(prefix.to_vec()).unwrap();
        assert!(matches!(terminal, Terminal::SemanticFailure { .. }));
        assert!(wire.contains("\"delta\":\"test\""));
        assert!(!wire.contains("event: error"));
    }

    #[test]
    fn raw_mode_filters_fragmented_semantic_failure_without_leaking_partial_error() {
        let mut decoder = SseDecoder::new(4096);
        let first = decoder
            .feed(
                b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"test\"}\n\n\
                  event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\"",
            )
            .unwrap();
        let mut stats = StreamStats::new("raw-fragmented-semantic-failure-test");
        let (first_wire, first_terminal) = raw_terminal_prefix(first, &mut stats).unwrap();
        assert!(first_terminal.is_none());
        assert!(String::from_utf8(first_wire.to_vec())
            .unwrap()
            .contains("delta"));

        let second = decoder
            .feed(b",\"message\":\"temporarily unavailable\"}}\n\n")
            .unwrap();
        let (second_wire, second_terminal) = raw_terminal_prefix(second, &mut stats).unwrap();
        assert!(second_wire.is_empty());
        assert!(matches!(
            second_terminal,
            Some(Terminal::SemanticFailure { .. })
        ));
    }

    #[test]
    fn raw_mode_does_not_repeat_fragmented_terminal_prefix() {
        let mut decoder = SseDecoder::new(4096);
        let first = b"event: response.completed\ndata: {\"type\":\"response.com";
        assert!(decoder.feed(first).unwrap().is_empty());
        let second = b"pleted\",\"response\":{\"status\":\"completed\"}}\n\n";
        let frames = decoder.feed(second).unwrap();
        let mut stats = StreamStats::new("raw-terminal-fragment-test");
        let (new_bytes, terminal) = raw_terminal_prefix(frames, &mut stats).unwrap();
        let terminal = terminal.expect("completed terminal expected");
        assert!(matches!(terminal, Terminal::Completed));
        assert!(new_bytes.ends_with(second));
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
    fn codex_whitespace_priming_does_not_commit_before_rate_limit_failure() {
        let mut active = test_active("codex");
        let frames = active
            .decoder
            .feed(
                b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"status\":\"in_progress\"}}\n\n\
                  event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"content\":[]}}\n\n\
                  event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n\
                  event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\" \"}\n\n\
                  event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"type\":\"rate_limit_error\",\"code\":null,\"message\":\"Upstream rate limit exceeded, please retry later\"}}}\n\n",
            )
            .unwrap();

        let PreflightResult::Retry(outcome) = process_preflight_frames(&mut active, frames)
            .unwrap()
            .unwrap()
        else {
            panic!("whitespace-only priming must remain retryable");
        };
        assert_eq!(outcome["status_code"], 429);
        assert_eq!(outcome["committed"], false);
        assert!(!active.business_committed);
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
        assert_eq!(active.buffered.len(), 1);
        assert!(active.buffered[0].starts_with(b"event: response.created\n"));
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
        assert_eq!(
            responses_semantic_error(
                &json!({
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "upstream_unavailable",
                            "message": "Service temporarily unavailable. Please try again later."
                        }
                    }
                }),
                "response.failed",
            ),
            (
                503,
                "Service temporarily unavailable. Please try again later.".to_owned()
            )
        );
    }
}
