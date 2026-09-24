use crate::observability::context::request_id;
use crate::providers::request::build_attempt;
use crate::providers::request::ALPHA_SEARCH_ENDPOINT;
use crate::providers::types::PreparedInput;
use crate::providers::video::video_task_route_for_path;
use crate::routing::access::extract_api_key;
use crate::routing::planner::compute_retry_count;
use crate::runtime::context::AppState;
use crate::runtime::scheduling::ProviderKeySelection;
use crate::transport::http::json_error;
use crate::transport::spool::SpoolObservation;
use crate::upstream::generic::chat_nonstream_hedging_enabled;
use crate::upstream::generic::client_ip;
use crate::upstream::generic::json_response;
use crate::upstream::generic::query_value;
use crate::upstream::generic::run_attempt_loop;
use crate::upstream::generic::run_hedged_attempt_loop;
use crate::upstream::generic::send_attempt;
use crate::upstream::generic::trace_id;
use crate::upstream::generic::AttemptLoop;
use crate::upstream::input::prepare_image_inputs;
use crate::upstream::input::prepare_input;
use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode, Uri};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub(crate) const PUBLIC_JSON_ROUTES: &[&str] = &[
    "/v1/systemone",
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/images/generations",
    "/v1/embeddings",
    "/v1/audio/speech",
    "/v1/moderations",
    "/v1/video/tasks",
    "/v1/asset-groups",
    "/v1/assets",
    ALPHA_SEARCH_ENDPOINT,
];

pub fn supports(method: &Method, path: &str) -> bool {
    if *method == Method::POST
        && (PUBLIC_JSON_ROUTES.contains(&path)
            || matches!(
                path,
                "/v1/images/edits"
                    | "/v1/audio/transcriptions"
                    | "/v1/responses"
                    | "/v1/responses/compact"
            ))
    {
        return true;
    }
    if *method == Method::GET
        && (matches!(path, "/search" | "/v1/search")
            || path.starts_with("/v1/video/tasks/")
            || path.starts_with("/v1/asset-groups/")
            || path.starts_with("/v1/assets/"))
    {
        return true;
    }
    false
}

pub fn known_path(path: &str) -> bool {
    PUBLIC_JSON_ROUTES.contains(&path)
        || matches!(
            path,
            "/search"
                | "/v1/search"
                | "/v1/images/edits"
                | "/v1/audio/transcriptions"
                | "/v1/responses"
                | "/v1/responses/compact"
        )
        || path.starts_with("/v1/video/tasks/")
        || path.starts_with("/v1/asset-groups/")
        || path.starts_with("/v1/assets/")
}

pub async fn handle(state: AppState, request: Request, resource_wait: Duration) -> Response<Body> {
    let started = Instant::now();
    let arrival = request
        .extensions()
        .get::<crate::observability::timing::RequestArrival>()
        .copied();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().trim_end_matches('/').to_owned();
    let path = if path.is_empty() { "/".into() } else { path };
    let headers = request.headers().clone();
    let request_id = request_id(&headers);
    let trace_id = trace_id(&headers, &request_id);
    let client_ip = client_ip(&headers);
    let api_key = extract_api_key(&headers).unwrap_or_default();
    let api_key_role = state
        .runtime
        .authorize(&headers)
        .await
        .ok()
        .map(|auth| auth.api_key.role.to_string())
        .unwrap_or_default();

    let input = match prepare_input(&state, request, &method, &uri, &path, resource_wait).await {
        Ok(input) => input,
        Err(response) => return response,
    };
    if method != Method::GET {
        if let Some(payload) = input.payload.as_ref() {
            if let Some(field) = missing_required_field(&path, payload) {
                return validation_error(field);
            }
        }
    }
    let query_model = query_value(&uri, "model");
    let video_task_route = video_task_route_for_path(&path);
    let mut request_model = input
        .payload
        .as_ref()
        .and_then(|payload| payload.get("model"))
        .and_then(Value::as_str)
        .or(query_model.as_deref())
        .unwrap_or(input.default_model.as_str())
        .trim()
        .to_owned();
    if let Some(route) = video_task_route.as_ref() {
        request_model.clone_from(&route.request_model);
    }
    if request_model.is_empty() {
        request_model = default_model_for_path(&state, &headers, &path).await;
    }
    if request_model.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "Request model is required");
    }
    if path != "/v1/moderations"
        && state
            .runtime
            .moderation_enabled(&headers)
            .await
            .unwrap_or(false)
    {
        if let Some(text) = input.payload.as_ref().and_then(moderation_text) {
            if let Err(response) = run_moderation_preflight(&state, &headers, &text).await {
                return response;
            }
        }
    }
    let body_bytes = input.observation.body_bytes;
    let request_type = (path == "/v1/responses/compact").then_some("compaction");
    let mut resolved = match state
        .runtime
        .resolve_route(
            &state.persistence,
            &headers,
            &request_model,
            &path,
            body_bytes,
            request_type,
            true,
        )
        .await
    {
        Ok(route) => route,
        Err(error) => return json_error(error.status, &error.message),
    };
    if let Some(route) = video_task_route.as_ref() {
        resolved
            .providers
            .retain(|provider| provider.name.as_ref() == route.provider_name);
        if resolved.providers.is_empty() {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "The provider used to create this video task is unavailable",
            );
        }
    }
    let retry_budget = state.runtime.auto_retry_budget(&headers).await;
    let max_attempts = if headers.contains_key(crate::routing::planner::TARGET_PROVIDER_HEADER)
        || retry_budget == 0
    {
        1
    } else {
        compute_retry_count(&resolved.providers)
            .max(resolved.providers.len().saturating_add(retry_budget))
            .min(100)
    };
    let hedging = resolved.hedging;
    let auto_retry = state.runtime.auto_retry_enabled(&headers).await;
    let (input, image_reservations) = match prepare_image_inputs(&state, input).await {
        Ok(prepared) => prepared,
        Err((status, detail)) => return json_error(status, &detail),
    };
    let (prompt_price, completion_price) = state.runtime.prices_for_model(&request_model).await;
    let use_chat_stream = path == "/v1/chat/completions"
        && input
            .payload
            .as_ref()
            .and_then(|payload| payload.get("stream"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut execution = AttemptLoop {
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
        providers: resolved.providers,
        max_attempts,
        auto_retry,
        input,
        image_reservations,
        prompt_price,
        completion_price,
        keepalive_updates: None,
    };
    if use_chat_stream {
        let (tx, rx) = tokio::sync::watch::channel(None);
        execution.keepalive_updates = Some(tx);
        return crate::protocols::chat::stream::with_keepalive(run_attempt_loop(execution), rx)
            .await;
    }
    if chat_nonstream_hedging_enabled(&execution.path, execution.input.payload.as_ref(), hedging) {
        return run_hedged_attempt_loop(execution, hedging).await;
    }
    run_attempt_loop(execution).await
}

pub(crate) fn missing_required_field(path: &str, payload: &Value) -> Option<&'static str> {
    let root = payload.as_object()?;
    let missing = |field: &'static str| {
        root.get(field).is_none_or(|value| {
            value.is_null()
                || value.as_str().is_some_and(|value| value.trim().is_empty())
                || value.as_array().is_some_and(Vec::is_empty)
        })
    };
    match path {
        "/v1/systemone" if !root.get("model").is_some_and(Value::is_string) => Some("model"),
        "/v1/systemone" if !root.contains_key("state") => Some("state"),
        "/v1/systemone" if !root.get("questions").is_some_and(Value::is_object) => {
            Some("questions")
        }
        "/v1/chat/completions" | "/v1/messages" if missing("messages") => Some("messages"),
        "/v1/images/generations" if missing("prompt") => Some("prompt"),
        "/v1/embeddings" if missing("input") => Some("input"),
        "/v1/audio/speech" if missing("input") => Some("input"),
        "/v1/audio/speech" if missing("voice") => Some("voice"),
        "/v1/moderations" if missing("input") => Some("input"),
        "/v1/responses" if missing("input") => Some("input"),
        "/v1/video/tasks" if missing("prompt") && missing("content") && missing("taskParams") => {
            Some("prompt")
        }
        _ => None,
    }
}

pub(crate) fn validation_error(field: &str) -> Response<Body> {
    let mut response = json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({
            "detail":[{
                "type":"missing",
                "loc":["body",field],
                "msg":"Field required",
                "input":Value::Null,
            }]
        }),
    );
    response
        .headers_mut()
        .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
    response
}

pub(crate) fn moderation_text(payload: &Value) -> Option<String> {
    let root = payload.as_object()?;
    if let Some(messages) = root.get("messages").and_then(Value::as_array) {
        for message in messages.iter().rev() {
            if let Some(text) = moderation_content_text(message.get("content")) {
                return Some(text);
            }
        }
    }
    if let Some(input) = root.get("input") {
        if let Some(text) = input
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            return Some(text.to_owned());
        }
        if let Some(items) = input.as_array() {
            if items.iter().all(Value::is_string) {
                let text = items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.trim().is_empty() {
                    return Some(text);
                }
            }
            for item in items.iter().rev() {
                if item
                    .get("role")
                    .and_then(Value::as_str)
                    .is_some_and(|role| role.eq_ignore_ascii_case("user"))
                {
                    if let Some(text) = moderation_content_text(item.get("content")) {
                        return Some(text);
                    }
                }
            }
        }
    }
    root.get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

pub(crate) fn moderation_content_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(text) = value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return Some(text.to_owned());
    }
    value
        .as_array()?
        .iter()
        .rev()
        .find_map(|part| {
            matches!(
                part.get("type").and_then(Value::as_str),
                Some("text" | "input_text")
            )
            .then(|| part.get("text").and_then(Value::as_str))
            .flatten()
        })
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

pub(crate) async fn run_moderation_preflight(
    state: &AppState,
    headers: &HeaderMap,
    text: &str,
) -> Result<(), Response<Body>> {
    let model = "omni-moderation-latest";
    let payload = json!({"model": model, "input": text, "stream": false});
    let resolved = state
        .runtime
        .resolve_route(
            &state.persistence,
            headers,
            model,
            "/v1/moderations",
            serde_json::to_vec(&payload)
                .map(|body| body.len() as u64)
                .unwrap_or(0),
            Some("moderation"),
            false,
        )
        .await
        .map_err(|error| json_error(error.status, &error.message))?;
    let input = PreparedInput {
        payload: Some(payload),
        replay: None,
        observation: SpoolObservation::default(),
        default_model: model.into(),
        content_type: "application/json".into(),
    };
    let uri = Uri::from_static("/v1/moderations");
    let mut last_error = json_error(
        StatusCode::BAD_GATEWAY,
        "Moderation preflight did not reach an upstream provider",
    );
    for provider in resolved.providers {
        let Some(original_model) = provider.models.get(model).cloned() else {
            continue;
        };
        let provider_key = match state
            .runtime
            .select_provider_key(&provider, &original_model)
            .await
        {
            ProviderKeySelection::Selected(key) => key,
            _ => continue,
        };
        let prepared = build_attempt(
            &provider,
            &provider_key,
            model,
            &original_model,
            &Method::POST,
            &uri,
            "/v1/moderations",
            headers,
            &input,
            &request_id(headers),
        )
        .map_err(|error| json_error(StatusCode::BAD_REQUEST, &error))?;
        match send_attempt(
            state,
            &provider,
            prepared,
            headers,
            "/v1/moderations",
            None,
            None,
        )
        .await
        {
            Ok(success) => {
                let bytes = to_bytes(success.response.into_body(), 4 * 1024 * 1024)
                    .await
                    .map_err(|error| {
                        json_error(
                            StatusCode::BAD_GATEWAY,
                            &format!("Read moderation response failed: {error}"),
                        )
                    })?;
                let response = serde_json::from_slice::<Value>(&bytes).map_err(|error| {
                    json_error(
                        StatusCode::BAD_GATEWAY,
                        &format!("Decode moderation response failed: {error}"),
                    )
                })?;
                let flagged = response
                    .pointer("/results/0/flagged")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if flagged {
                    return Err(json_error(
                        StatusCode::BAD_REQUEST,
                        "Content did not pass the moral check, please modify and try again.",
                    ));
                }
                return Ok(());
            }
            Err(failure) => {
                last_error = json_error(failure.status, &failure.detail);
            }
        }
    }
    Err(last_error)
}

pub(crate) async fn default_model_for_path(
    state: &AppState,
    headers: &HeaderMap,
    path: &str,
) -> String {
    let Ok(models) = state.runtime.models_for_headers(headers).await else {
        return String::new();
    };
    let preferred = if path.contains("video")
        || path.contains("asset-groups")
        || path.starts_with("/v1/assets")
    {
        ["seedance", "sora", "video", "veo"].as_slice()
    } else {
        [].as_slice()
    };
    for token in preferred {
        if let Some(model) = models
            .iter()
            .find(|model| model.to_ascii_lowercase().contains(token))
        {
            return model.clone();
        }
    }
    models.into_iter().next().unwrap_or_default()
}
