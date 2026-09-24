use crate::runtime::context::AppState;
use crate::runtime::idempotency::RequestHasher;
use crate::runtime::idempotency::{self, Claim};
use crate::runtime::resources::CapacityFailure;
use crate::transport::body::read_spooled_body;
use crate::transport::body::RequestBodySpoolError;
use crate::transport::http::filtered_request_headers;
use crate::transport::http::filtered_response_headers;
use crate::transport::http::json_error;
use crate::transport::http::relay_response;
use crate::transport::http::CONTROL_HEADER;
use crate::transport::spool::RequestSpool;
use crate::transport::spool::SpoolFailure;
use crate::transport::spool::SpoolObservation;
use crate::upstream::responses;
use crate::upstream::responses::prepare::prepare_request;
use crate::upstream::responses::route::ResponsesPreparation;
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Response, StatusCode};
use axum::response::IntoResponse;
use std::time::Duration;

const INTERNAL_PREFIX: &str = "/_internal/rust-responses";

pub async fn handler(State(state): State<AppState>, mut request: Request) -> Response<Body> {
    request
        .extensions_mut()
        .insert(crate::observability::timing::RequestArrival::now());
    let cors = crate::api::cors::CorsRequest::from_headers(request.headers());
    if let Some(response) = cors.preflight_response(request.method(), request.headers()) {
        return response;
    }
    let mut response = handler_inner(state, request).await;
    cors.apply(&mut response);
    response
}

async fn handler_inner(state: AppState, request: Request) -> Response<Body> {
    let request = match crate::transport::decompression::decode(request).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    let raw_path = request.uri().path();
    let path = raw_path.trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path }.to_owned();
    if path.starts_with(INTERNAL_PREFIX) {
        return json_error(StatusCode::NOT_FOUND, "Not found");
    }
    let native_method = request.method().clone();
    let native_path = path.clone();
    let native_uri = request.uri().clone();
    let native_headers = request.headers().clone();
    if let Some(response) = crate::api::platform::handle(
        &state,
        &native_method,
        &native_uri,
        &native_path,
        &native_headers,
    )
    .await
    {
        return response;
    }
    let use_rust_responses = state.responses_data_plane_enabled
        && request.method() == Method::POST
        && matches!(path.as_str(), "/v1/responses" | "/v1/responses/compact");
    let generic_idempotency_request = request.method() == Method::POST
        && matches!(path.as_str(), "/v1/chat/completions" | "/v1/messages")
        && request.headers().contains_key("idempotency-key");
    let use_spooled_dispatch = use_rust_responses || generic_idempotency_request;
    let resource_wait = if path == "/healthz" {
        Duration::ZERO
    } else {
        match state.resource_governor.wait_for_global_headroom().await {
            Ok(wait) => wait.waited,
            Err(failure) => {
                if use_spooled_dispatch {
                    let observation = SpoolObservation {
                        resource_wait_ms: duration_ms(failure.waited),
                        failure_resource: Some(failure.resource.as_str()),
                        ..SpoolObservation::default()
                    };
                    log_spool_observation(
                        request.headers(),
                        failure.resource.exhausted_status(),
                        &observation,
                    );
                }
                return resource_capacity_response(failure);
            }
        }
    };
    if use_spooled_dispatch {
        let idempotency_values = request
            .headers()
            .get_all("idempotency-key")
            .iter()
            .collect::<Vec<_>>();
        if idempotency_values.len() > 1 {
            return idempotency_error(
                StatusCode::BAD_REQUEST,
                "multiple Idempotency-Key headers are not allowed",
                "invalid-key",
                false,
            );
        }
        let idempotency_key = match idempotency_values.first() {
            Some(value) => {
                let Ok(key) = value.to_str().map(str::to_owned) else {
                    return idempotency_error(
                        StatusCode::BAD_REQUEST,
                        "Idempotency-Key must contain 1-128 safe ASCII characters",
                        "invalid-key",
                        false,
                    );
                };
                Some(key)
            }
            None => None,
        };
        let idempotent_request = idempotency_key.is_some();
        let (parts, body) = request.into_parts();
        let request_hasher = match idempotency_key.as_deref() {
            Some(key) => match RequestHasher::new(&parts.method, &parts.uri, &parts.headers, key) {
                Ok(hasher) => Some(hasher),
                Err(response) => return response,
            },
            None => None,
        };
        let content_length = parts
            .headers
            .get("content-length")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        let spool = match read_spooled_body(
            body,
            &state.request_spool,
            request_hasher,
            content_length,
            resource_wait,
        )
        .await
        {
            Ok(result) => result,
            Err(RequestBodySpoolError::Timeout) => {
                let observation = SpoolObservation {
                    resource_wait_ms: duration_ms(resource_wait),
                    failure_resource: Some("request_body_idle_timeout"),
                    ..SpoolObservation::default()
                };
                log_spool_observation(&parts.headers, StatusCode::REQUEST_TIMEOUT, &observation);
                return request_spool_input_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "request body upload timed out",
                    "request-timeout",
                    "request_body_idle_timeout",
                    idempotent_request,
                );
            }
            Err(RequestBodySpoolError::Read) => {
                let observation = SpoolObservation {
                    resource_wait_ms: duration_ms(resource_wait),
                    failure_resource: Some("request_body_read"),
                    ..SpoolObservation::default()
                };
                log_spool_observation(&parts.headers, StatusCode::BAD_REQUEST, &observation);
                return request_spool_input_error(
                    StatusCode::BAD_REQUEST,
                    "request body upload failed",
                    "request-failed",
                    "request_body_read",
                    idempotent_request,
                );
            }
            Err(RequestBodySpoolError::Spool(failure)) => {
                log_spool_observation(&parts.headers, failure.status, &failure.observation);
                return spool_failure_response(failure, idempotent_request);
            }
        };
        let RequestSpool {
            identity,
            observation,
            storage,
        } = spool;
        let owner = match identity {
            Some(identity) => {
                let claim = state.idempotency.claim(identity).await;
                let Claim::Owner(owner) = claim else {
                    let Claim::Response(response) = claim else {
                        unreachable!()
                    };
                    return response;
                };
                Some(owner)
            }
            None => None,
        };
        if !use_rust_responses {
            let body = match storage.into_body(&observation).await {
                Ok(body) => body,
                Err(failure) => {
                    if let Some(owner) = owner {
                        owner.release().await;
                    }
                    log_spool_observation(&parts.headers, failure.status, &failure.observation);
                    return spool_failure_response(failure, idempotent_request);
                }
            };
            let request = Request::from_parts(parts, body);
            let response = crate::api::gateway::handle(state, request, resource_wait).await;
            if let Some(owner) = owner {
                return relay_idempotent_axum_response(response, owner).await;
            }
            return response;
        }
        if state
            .runtime
            .moderation_enabled(&parts.headers)
            .await
            .unwrap_or(false)
        {
            if let Ok(payload) = storage.parse_json().await {
                if let Some(text) = crate::api::gateway::moderation_text(&payload) {
                    if let Err(response) =
                        crate::api::gateway::run_moderation_preflight(&state, &parts.headers, &text)
                            .await
                    {
                        if let Some(owner) = owner {
                            owner.release().await;
                        }
                        return response;
                    }
                }
            }
        }
        let native_json_memory_bytes = observation.body_bytes.saturating_mul(
            std::env::var("RUST_RESPONSES_JSON_MEMORY_MULTIPLIER")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(3),
        );
        let native_memory_reservation = match state
            .resource_governor
            .reserve_memory_capacity(native_json_memory_bytes)
            .await
        {
            Ok((_, reservation)) => reservation,
            Err(failure) => {
                if let Some(owner) = owner {
                    owner.release().await;
                }
                return resource_capacity_response(failure);
            }
        };
        match prepare_request(
            &state.runtime,
            state.codex_oauth.clone(),
            state.persistence.clone(),
            &parts,
            &storage,
            &observation,
            native_memory_reservation,
            &path,
        )
        .await
        {
            ResponsesPreparation::Ready(route) => {
                return responses::serve_native(state, route, owner).await;
            }
            ResponsesPreparation::Response(response) => {
                if let Some(owner) = owner {
                    owner.release().await;
                }
                return response;
            }
            ResponsesPreparation::Fallback => {}
        }
        if !matches!(path.as_str(), "/v1/responses" | "/v1/responses/compact")
            && !state.python_compat_enabled
        {
            if let Some(owner) = owner {
                owner.release().await;
            }
            return json_error(
                StatusCode::NOT_IMPLEMENTED,
                "Request is not supported by the Rust runtime",
            );
        }
        let body = match storage.into_body(&observation).await {
            Ok(body) => body,
            Err(failure) => {
                if let Some(owner) = owner {
                    owner.release().await;
                }
                log_spool_observation(&parts.headers, failure.status, &failure.observation);
                return spool_failure_response(failure, idempotent_request);
            }
        };
        let request = Request::from_parts(parts, body);
        if matches!(path.as_str(), "/v1/responses" | "/v1/responses/compact") {
            let response = crate::api::gateway::handle(state, request, resource_wait).await;
            if let Some(owner) = owner {
                return relay_idempotent_axum_response(response, owner).await;
            }
            return response;
        }
        let proxied = proxy_to_backend(
            &state,
            request,
            true,
            idempotent_request,
            Some(&observation),
        )
        .await;
        return match (proxied, owner) {
            (Ok((response, Some(session_id))), owner) => {
                responses::serve_session(state, session_id, response, owner).await
            }
            (Ok((response, None)), Some(owner)) => relay_idempotent_response(response, owner).await,
            (Ok((response, None)), None) => relay_response(response),
            (Err(error), Some(owner)) => {
                owner.release().await;
                json_error(
                    StatusCode::BAD_GATEWAY,
                    &format!("Python control plane unavailable: {error}"),
                )
            }
            (Err(error), None) => json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Python control plane unavailable: {error}"),
            ),
        };
    }
    if crate::api::gateway::supports(request.method(), &path) {
        return crate::api::gateway::handle(state, request, resource_wait).await;
    }
    if crate::api::platform::supports_mutation(request.method(), &path) {
        return crate::api::platform::handle_mutation(&state, request).await;
    }
    if !state.python_compat_enabled {
        if crate::api::gateway::known_path(&path) {
            return json_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed");
        }
        return json_error(StatusCode::NOT_FOUND, "Not found");
    }
    match proxy_to_backend(&state, request, use_rust_responses, false, None).await {
        Ok((response, session_id)) => {
            if let Some(session_id) = session_id {
                responses::serve_session(state, session_id, response, None).await
            } else {
                relay_response(response)
            }
        }
        Err(error) => json_error(
            StatusCode::BAD_GATEWAY,
            &format!("Python control plane unavailable: {error}"),
        ),
    }
}

async fn proxy_to_backend(
    state: &AppState,
    request: Request,
    rust_responses: bool,
    strip_idempotency_key: bool,
    spool_observation: Option<&SpoolObservation>,
) -> Result<(reqwest::Response, Option<String>), reqwest::Error> {
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", state.backend_origin, path_and_query);
    let mut headers = filtered_request_headers(&parts.headers);
    if strip_idempotency_key {
        headers.remove("idempotency-key");
    }
    if rust_responses {
        if let Ok(value) = HeaderValue::from_str(&state.control_token) {
            headers.insert(HeaderName::from_static(CONTROL_HEADER), value);
        }
    }
    if let Some(observation) = spool_observation {
        insert_spool_observation_headers(&mut headers, observation);
    }
    let body_stream = body.into_data_stream();
    let response = state
        .backend_client
        .request(parts.method, url)
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body_stream))
        .send()
        .await?;
    let session_id = response
        .headers()
        .get("x-uni-api-rust-responses-session")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    Ok((response, session_id))
}

async fn relay_idempotent_response(
    response: reqwest::Response,
    owner: idempotency::Owner,
) -> Response<Body> {
    let status = response.status();
    let headers = filtered_response_headers(response.headers());
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(error) => {
            owner.release().await;
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Python control response read failed: {error}"),
            );
        }
    };
    if body.len() > owner.max_response_bytes() {
        owner.nonreplayable("response_too_large").await;
    } else {
        owner
            .complete(status, headers.clone(), vec![body.clone()], body.len())
            .await;
    }
    idempotency::response_from_bytes(status, headers, body)
}

async fn relay_idempotent_axum_response(
    response: Response<Body>,
    owner: idempotency::Owner,
) -> Response<Body> {
    if response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        owner.nonreplayable("streaming_response").await;
        return response;
    }
    let (parts, body) = response.into_parts();
    let maximum = std::env::var("RUST_GENERIC_UPSTREAM_RESPONSE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64 * 1024 * 1024);
    let body = match to_bytes(body, maximum.saturating_add(1)).await {
        Ok(body) => body,
        Err(error) => {
            owner.release().await;
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Rust compatibility response read failed: {error}"),
            );
        }
    };
    if body.len() > owner.max_response_bytes() {
        owner.nonreplayable("response_too_large").await;
    } else {
        owner
            .complete(
                parts.status,
                parts.headers.clone(),
                vec![body.clone()],
                body.len(),
            )
            .await;
    }
    Response::from_parts(parts, Body::from(body))
}

fn idempotency_error(
    status: StatusCode,
    message: &str,
    idempotency_status: &'static str,
    retry_after: bool,
) -> Response<Body> {
    let mut response = (
        status,
        [("content-type", "application/json")],
        serde_json::json!({
            "error": {
                "message": message,
                "type": "idempotency_error",
                "code": idempotency_status.replace('-', "_"),
            }
        })
        .to_string(),
    )
        .into_response();
    response.headers_mut().insert(
        "x-uni-api-idempotency-status",
        HeaderValue::from_static(idempotency_status),
    );
    response
        .headers_mut()
        .insert("connection", HeaderValue::from_static("close"));
    if retry_after {
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("1"));
    }
    response
}

fn insert_spool_observation_headers(headers: &mut HeaderMap, observation: &SpoolObservation) {
    for (name, value) in [
        (
            "x-uni-api-rust-request-spool-body-bytes",
            observation.body_bytes,
        ),
        (
            "x-uni-api-rust-request-spool-memory-peak-bytes",
            observation.memory_peak_bytes,
        ),
        (
            "x-uni-api-rust-request-spool-local-disk-bytes",
            observation.local_disk_bytes,
        ),
        (
            "x-uni-api-rust-request-spool-local-free-bytes-start",
            observation.local_free_bytes_at_start,
        ),
        (
            "x-uni-api-rust-request-spool-local-writable-bytes-start",
            observation.local_writable_bytes_at_start,
        ),
        (
            "x-uni-api-rust-request-spool-local-free-inodes-start",
            observation.local_free_inodes_at_start,
        ),
        (
            "x-uni-api-rust-request-spool-local-writable-inodes-start",
            observation.local_writable_inodes_at_start,
        ),
        (
            "x-uni-api-rust-request-spool-resource-wait-ms",
            observation.resource_wait_ms,
        ),
    ] {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(&value.to_string()),
        ) {
            headers.insert(name, value);
        }
    }
    if let Ok(value) = HeaderValue::from_str(observation.final_tier) {
        headers.insert(
            HeaderName::from_static("x-uni-api-rust-request-spool-final-tier"),
            value,
        );
    }
}

fn resource_capacity_response(failure: CapacityFailure) -> Response<Body> {
    let mut response = json_error(
        failure.resource.exhausted_status(),
        &format!(
            "Local resource wait timed out: {}",
            failure.resource.as_str()
        ),
    );
    response
        .headers_mut()
        .insert("retry-after", HeaderValue::from_static("1"));
    response.headers_mut().insert(
        "x-uni-api-status-origin",
        HeaderValue::from_static("rust_local_admission"),
    );
    if let Ok(value) = HeaderValue::from_str(failure.resource.as_str()) {
        response
            .headers_mut()
            .insert("x-uni-api-admission-reason", value);
    }
    response
}

fn request_spool_input_error(
    status: StatusCode,
    message: &str,
    idempotency_status: &'static str,
    resource: &'static str,
    idempotent_request: bool,
) -> Response<Body> {
    let mut response = if idempotent_request {
        idempotency_error(status, message, idempotency_status, false)
    } else {
        json_error(status, message)
    };
    response.headers_mut().insert(
        "x-uni-api-status-origin",
        HeaderValue::from_static("rust_request_spool"),
    );
    response.headers_mut().insert(
        "x-uni-api-admission-reason",
        HeaderValue::from_static(resource),
    );
    response
}

fn spool_failure_response(failure: SpoolFailure, idempotent_request: bool) -> Response<Body> {
    let mut response = if idempotent_request {
        idempotency_error(
            failure.status,
            &failure.message,
            if failure.status == StatusCode::PAYLOAD_TOO_LARGE {
                "request-too-large"
            } else {
                "capacity-exhausted"
            },
            failure.retry_after,
        )
    } else {
        let mut response = json_error(failure.status, &failure.message);
        if failure.retry_after {
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
        }
        response
    };
    response.headers_mut().insert(
        "x-uni-api-status-origin",
        HeaderValue::from_static("rust_request_spool"),
    );
    if let Some(resource) = failure.resource {
        if let Ok(value) = HeaderValue::from_str(resource.as_str()) {
            response
                .headers_mut()
                .insert("x-uni-api-admission-reason", value);
        }
    }
    response
}

fn log_spool_observation(headers: &HeaderMap, status: StatusCode, observation: &SpoolObservation) {
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-correlation-id"))
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let trace_id = headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split('-').nth(1))
        .unwrap_or(request_id);
    eprintln!(
        "{}",
        serde_json::json!({
            "event_type": "rust_request_spool",
            "request_id": request_id,
            "trace_id": trace_id,
            "status_code": status.as_u16(),
            "body_bytes": observation.body_bytes,
            "memory_peak_bytes": observation.memory_peak_bytes,
            "local_disk_bytes": observation.local_disk_bytes,
            "local_free_bytes_at_start": observation.local_free_bytes_at_start,
            "local_writable_bytes_at_start": observation.local_writable_bytes_at_start,
            "local_free_inodes_at_start": observation.local_free_inodes_at_start,
            "local_writable_inodes_at_start": observation.local_writable_inodes_at_start,
            "resource_wait_ms": observation.resource_wait_ms,
            "final_tier": observation.final_tier,
            "failure_resource": observation.failure_resource,
        })
    );
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
