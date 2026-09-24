use crate::observability::context::request_id;
use crate::providers::codex::oauth::CodexOAuthManager;
use crate::routing::access::extract_api_key;
use crate::routing::filters::detect_request_type;
use crate::routing::filters::request_reasoning_effort;
use crate::routing::planner::api_key_retry_budget;
use crate::routing::planner::compute_retry_count;
use crate::routing::planner::diagnostic_key;
use crate::routing::planner::matching_providers;
use crate::routing::planner::TARGET_PROVIDER_HEADER;
use crate::runtime::resources::MemoryReservation;
use crate::runtime::scheduling::parse_rate_limits;
use crate::runtime::scheduling::tpr_exceeded;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::Persistence;
use crate::transport::spool::SpoolObservation;
use crate::transport::spool::StoredBody;
use crate::upstream::hedging::parse_hedging;
use crate::upstream::hedging::HedgingConfig;
use crate::upstream::responses::route::event_severity;
use crate::upstream::responses::route::native_rejection_origin;
use crate::upstream::responses::route::provider_stream_override;
use crate::upstream::responses::route::status_class;
use crate::upstream::responses::route::ResponsesPreparation;
use crate::upstream::responses::route::ResponsesRoute;
use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;

pub(crate) struct NativeRejectionObservation<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) request_model: Option<&'a str>,
    pub(crate) stream: Option<bool>,
    pub(crate) role: &'a str,
    pub(crate) request_body_bytes: u64,
    pub(crate) snapshot_revision: &'a str,
    pub(crate) reason: &'a str,
}

pub(crate) fn emit_native_rejection(status: u16, observation: NativeRejectionObservation<'_>) {
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
pub async fn prepare_request(
    store: &GatewayRuntime,
    codex_oauth: CodexOAuthManager,
    persistence: Persistence,
    parts: &Parts,
    storage: &StoredBody,
    observation: &SpoolObservation,
    memory_reservation: MemoryReservation,
    endpoint: &str,
) -> ResponsesPreparation {
    let Some(snapshot) = store.snapshot().await else {
        return ResponsesPreparation::Fallback;
    };
    if !is_identity_json_request(&parts.headers) {
        return ResponsesPreparation::Fallback;
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
        return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Response(json_response(
            StatusCode::FORBIDDEN,
            json!({"error": "Invalid or missing API Key"}),
        ));
    };
    if !api_key.native_supported {
        return ResponsesPreparation::Fallback;
    }
    let role = api_key.role.as_ref();
    if let Err(error) = store.ensure_paid_balance(&persistence, &api_key).await {
        return ResponsesPreparation::Response(json_response(
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
            return ResponsesPreparation::Response(json_response(
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
            return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Response(json_response(
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
                return ResponsesPreparation::Response(json_response(
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
            return ResponsesPreparation::Response(json_response(
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
            return ResponsesPreparation::Response(json_response(
                StatusCode::NOT_FOUND,
                json!({"message": format!("No available providers at the moment: {request_model}")}),
            ));
        }
        Err(()) => return ResponsesPreparation::Fallback,
    };
    let providers = store
        .schedule_providers(&api_key, &request_model, providers)
        .await;
    if providers.is_empty() {
        return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Fallback;
    }
    let Some(global_rate_rules) = parse_rate_limits(snapshot.preferences.get("rate_limit"), None)
    else {
        return ResponsesPreparation::Fallback;
    };
    let Some(client_rate_rules) =
        parse_rate_limits(api_key.preferences.get("rate_limit"), Some(&request_model))
    else {
        return ResponsesPreparation::Fallback;
    };
    if tpr_exceeded(
        &client_rate_rules,
        (observation.body_bytes / 4).max(1) as usize,
    ) {
        return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Response(json_response(
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
        return ResponsesPreparation::Response(json_response(
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
    ResponsesPreparation::Ready(ResponsesRoute {
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
        history_repair_attempted: false,
        pending_history_repair: None,
        empty_name_repair_attempt_id: None,
        missing_item_repair_attempt_id: None,
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
            .get::<crate::observability::timing::RequestArrival>()
            .copied(),
        started_at: tokio::time::Instant::now(),
        final_emitted: false,
        _memory_reservation: memory_reservation,
    })
}

pub(crate) fn pydantic_bool(value: &Value) -> Option<bool> {
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

pub(crate) fn is_identity_json_request(headers: &HeaderMap) -> bool {
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

pub(crate) fn json_response(status: StatusCode, payload: Value) -> Response<Body> {
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
