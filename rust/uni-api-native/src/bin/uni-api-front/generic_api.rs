use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Response, StatusCode, Uri};
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use bytes::Bytes;
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use ring::rand::SystemRandom;
use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use url::Url;

use crate::persistence::{ChannelStat, RequestStat};
use crate::provider_stream::{
    self, OutputProtocol as StreamOutputProtocol, Protocol as StreamProtocol,
};
use crate::proxy::{
    filtered_response_headers, json_error, read_spooled_body, AppState, RequestBodySpoolError,
};
use crate::request_spool::{SpoolObservation, StoredBody};
use crate::resources::MemoryReservation;
use crate::responses_native::{
    apply_overrides, classify_provider_failure, compute_retry_count, extract_api_key, request_id,
    FailedRoute, Provider, ProviderKeySelection, CODEX_USER_AGENT,
};

const ALPHA_SEARCH_ENDPOINT: &str = "/v1/alpha/search";
const IMAGE_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_IMAGE_MAX_BYTES: usize = 12 * 1024 * 1024;
const DEFAULT_UPSTREAM_RESPONSE_MAX_BYTES: usize = 64 * 1024 * 1024;
const UPSTREAM_ERROR_MAX_BYTES: usize = 1024 * 1024;

const PUBLIC_JSON_ROUTES: &[&str] = &[
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseAdapter {
    Passthrough,
    Search,
    ResponsesToChat,
    GeminiToChat,
    ClaudeToChat,
    CohereToChat,
    CloudflareToChat,
    AwsToChat,
    LingjingVideo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DownstreamProtocol {
    Native,
    ResponsesCompat,
}

struct PreparedAttempt {
    method: Method,
    url: String,
    headers: HeaderMap,
    body: AttemptBody,
    adapter: ResponseAdapter,
    downstream_stream: bool,
    upstream_stream: bool,
    request_model: String,
    original_model: String,
    downstream_protocol: DownstreamProtocol,
    chat_stream_include_usage: bool,
    provider_key: String,
    estimated_video_tokens: Option<i64>,
}

#[derive(Clone)]
struct CachedVertexToken {
    value: String,
    expires_at: Instant,
}

static VERTEX_TOKEN_CACHE: OnceLock<tokio::sync::Mutex<HashMap<String, CachedVertexToken>>> =
    OnceLock::new();

struct ThoughtSignatureCache {
    values: HashMap<String, String>,
    order: VecDeque<String>,
    bytes: usize,
}

static GEMINI_THOUGHT_SIGNATURES: OnceLock<Mutex<ThoughtSignatureCache>> = OnceLock::new();

#[derive(Clone)]
struct VideoTaskRoute {
    provider_name: String,
    request_model: String,
    provider_key: String,
    video_tokens: Option<i64>,
    created_at: Instant,
}

static VIDEO_TASK_ROUTES: OnceLock<Mutex<HashMap<String, VideoTaskRoute>>> = OnceLock::new();

enum AttemptBody {
    Json(Vec<u8>),
    Replay(StoredBody, SpoolObservation),
    MultipartRewrite {
        storage: StoredBody,
        observation: SpoolObservation,
        source_content_type: String,
        boundary: String,
        model: String,
    },
    DashscopeTranscription {
        storage: StoredBody,
        observation: SpoolObservation,
        source_content_type: String,
        model: String,
        provider_key: String,
    },
    Empty,
}

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
        .native_responses_config
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
            .native_responses_config
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
        .native_responses_config
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
    let max_attempts = compute_retry_count(&resolved.providers)
        .max(resolved.providers.len())
        .min(10);
    let auto_retry = state
        .native_responses_config
        .auto_retry_enabled(&headers)
        .await;
    let (input, _image_reservations) = match prepare_image_inputs(&state, input).await {
        Ok(prepared) => prepared,
        Err((status, detail)) => return json_error(status, &detail),
    };
    let (prompt_price, completion_price) = state
        .native_responses_config
        .prices_for_model(&request_model)
        .await;
    let mut last_status = StatusCode::BAD_GATEWAY;
    let mut last_detail = String::from("No upstream attempt succeeded");
    let mut last_upstream_response = None;

    for attempt_index in 0..max_attempts {
        let provider = resolved.providers[attempt_index % resolved.providers.len()].clone();
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
                .native_responses_config
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
                last_status = StatusCode::TOO_MANY_REQUESTS;
                last_detail = "All matching provider routes are cooling down".into();
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
        emit_attempt(
            &request_id,
            &trace_id,
            &api_key_role,
            attempt_index,
            &provider,
            &request_model,
            &original_model,
            &prepared.url,
            "started",
            None,
        );
        match send_attempt(&state, &provider, prepared, &headers, &path).await {
            Ok(success) => {
                state.persistence.record_channel(ChannelStat {
                    request_id: request_id.clone(),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    api_key: api_key.clone(),
                    provider_api_key: provider_key_raw.clone(),
                    success: true,
                });
                state
                    .native_responses_config
                    .reset_route_failure(&provider, &original_model)
                    .await;
                let mut request_stat = RequestStat {
                    request_id: request_id.clone(),
                    trace_id: trace_id.clone(),
                    endpoint: path.clone(),
                    client_ip: client_ip.clone(),
                    process_time: started.elapsed().as_secs_f64(),
                    first_response_time: attempt_started.elapsed().as_secs_f64(),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    api_key: api_key.clone(),
                    prompt_tokens: success.usage.0,
                    completion_tokens: success.usage.1,
                    total_tokens: success.usage.2,
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
                if let Some(usage_completion) = success.usage_completion {
                    let persistence = state.persistence.clone();
                    tokio::spawn(async move {
                        if let Ok(usage) = usage_completion.await {
                            request_stat.prompt_tokens = usage.0;
                            request_stat.completion_tokens = usage.1;
                            request_stat.total_tokens = usage.2;
                        }
                        persistence.record_request(request_stat);
                    });
                } else {
                    state.persistence.record_request(request_stat);
                }
                emit_attempt(
                    &request_id,
                    &trace_id,
                    &api_key_role,
                    attempt_index,
                    &provider,
                    &request_model,
                    &original_model,
                    &success.upstream_url,
                    "completed",
                    Some(success.status.as_u16()),
                );
                return success.response;
            }
            Err(mut failure) => {
                let policy = classify_provider_failure(
                    failure.status.as_u16(),
                    &failure.detail,
                    Some(&provider),
                    &path,
                    auto_retry,
                );
                failure.status =
                    StatusCode::from_u16(policy.status).unwrap_or(StatusCode::BAD_GATEWAY);
                if let Some(response) = failure.response.as_mut() {
                    *response.status_mut() = failure.status;
                }
                last_status = failure.status;
                last_detail = failure.detail.clone();
                if let Some(response) = failure.response.take() {
                    last_upstream_response = Some(response);
                }
                state.persistence.record_channel(ChannelStat {
                    request_id: request_id.clone(),
                    provider: provider.name.to_string(),
                    model: request_model.clone(),
                    api_key: api_key.clone(),
                    provider_api_key: provider_key_raw.clone(),
                    success: false,
                });
                if !policy.request_scoped || policy.force_quota_cooldown {
                    state
                        .native_responses_config
                        .cool_failed_route(FailedRoute {
                            provider: &provider,
                            key: &provider_key_raw,
                            original_model: &original_model,
                            has_alternative: resolved.providers.len() > 1,
                            status: failure.status.as_u16(),
                            detail: &failure.detail,
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

fn missing_required_field(path: &str, payload: &Value) -> Option<&'static str> {
    let root = payload.as_object()?;
    let missing = |field: &'static str| {
        root.get(field).is_none_or(|value| {
            value.is_null()
                || value.as_str().is_some_and(|value| value.trim().is_empty())
                || value.as_array().is_some_and(Vec::is_empty)
        })
    };
    match path {
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

fn validation_error(field: &str) -> Response<Body> {
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

fn moderation_content_text(value: Option<&Value>) -> Option<String> {
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

fn content_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(text) = value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return Some(text.to_owned());
    }
    let parts = value.as_array()?;
    let mut output = String::new();
    for part in parts {
        let text = part
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty());
        if let Some(text) = text {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(text);
        }
    }
    (!output.is_empty()).then_some(output)
}

pub(crate) async fn run_moderation_preflight(
    state: &AppState,
    headers: &HeaderMap,
    text: &str,
) -> Result<(), Response<Body>> {
    let model = "omni-moderation-latest";
    let payload = json!({"model": model, "input": text, "stream": false});
    let resolved = state
        .native_responses_config
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
            .native_responses_config
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
        match send_attempt(state, &provider, prepared, headers, "/v1/moderations").await {
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

struct PreparedInput {
    payload: Option<Value>,
    replay: Option<(StoredBody, SpoolObservation)>,
    observation: SpoolObservation,
    default_model: String,
    content_type: String,
}

async fn prepare_image_inputs(
    state: &AppState,
    mut input: PreparedInput,
) -> Result<(PreparedInput, Vec<MemoryReservation>), (StatusCode, String)> {
    let Some(payload) = input.payload.as_mut() else {
        return Ok((input, Vec::new()));
    };
    let mut reservations = Vec::new();
    if let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("image_url") {
                    continue;
                }
                let Some(url) = part
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                else {
                    continue;
                };
                let normalized = normalize_image_url(state, &url, &mut reservations).await?;
                if let Some(root) = part.get_mut("image_url").and_then(Value::as_object_mut) {
                    root.insert("url".into(), Value::String(normalized));
                }
            }
        }
    }
    if let Some(items) = payload.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            let Some(parts) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("input_image") {
                    continue;
                }
                let Some(url) = part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                else {
                    continue;
                };
                let normalized = normalize_image_url(state, &url, &mut reservations).await?;
                part.as_object_mut()
                    .expect("responses input image object")
                    .insert("image_url".into(), Value::String(normalized));
            }
        }
    }
    Ok((input, reservations))
}

async fn normalize_image_url(
    state: &AppState,
    value: &str,
    reservations: &mut Vec<MemoryReservation>,
) -> Result<String, (StatusCode, String)> {
    if value.starts_with("data:") {
        validate_image_data_url(state, value, reservations).await?;
        return Ok(value.to_owned());
    }
    let parsed =
        Url::parse(value).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid image URL".into()))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err((StatusCode::BAD_REQUEST, "Invalid image URL".into()));
    }
    let client = state
        .upstream_client(None, false, None)
        .await
        .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;
    let response = tokio::time::timeout(
        IMAGE_FETCH_TIMEOUT,
        client
            .get(value)
            .header("accept-encoding", "identity")
            .timeout(IMAGE_FETCH_TIMEOUT)
            .send(),
    )
    .await
    .map_err(|_| (StatusCode::REQUEST_TIMEOUT, "Image fetch timed out".into()))?
    .map_err(|_| (StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()))?;
    if !response.status().is_success() {
        return Err((StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()));
    }
    let maximum = image_max_bytes();
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Image input exceeds the configured size limit".into(),
        ));
    }
    if let Some(length) = response.content_length() {
        let (_, reservation) = state
            .resource_governor
            .reserve_memory_capacity(length.saturating_mul(3))
            .await
            .map_err(|_| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Insufficient memory capacity for image input".into(),
                )
            })?;
        reservations.push(reservation);
    }
    let known_length = response.content_length().is_some();
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|_| (StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()))?;
        if bytes.len().saturating_add(chunk.len()) > maximum {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "Image input exceeds the configured size limit".into(),
            ));
        }
        if !known_length {
            let (_, reservation) = state
                .resource_governor
                .reserve_memory_capacity((chunk.len() as u64).saturating_mul(3))
                .await
                .map_err(|_| {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Insufficient memory capacity for image input".into(),
                    )
                })?;
            reservations.push(reservation);
        }
        bytes.extend_from_slice(&chunk);
    }
    let media_type = detect_image_media_type(&bytes).ok_or_else(|| {
        (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Unsupported image media type".into(),
        )
    })?;
    Ok(format!(
        "data:{media_type};base64,{}",
        BASE64.encode(&bytes)
    ))
}

async fn validate_image_data_url(
    state: &AppState,
    value: &str,
    reservations: &mut Vec<MemoryReservation>,
) -> Result<(), (StatusCode, String)> {
    let (header, encoded) = value
        .split_once(',')
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Invalid image data URL".into()))?;
    if header.len() > 128
        || !header.to_ascii_lowercase().starts_with("data:")
        || !header.to_ascii_lowercase().ends_with(";base64")
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "Image input must be a base64 data URL".into(),
        ));
    }
    let declared = header[5..header.len().saturating_sub(7)].to_ascii_lowercase();
    let declared = if declared == "image/jpg" {
        "image/jpeg"
    } else {
        declared.as_str()
    };
    if !matches!(declared, "image/jpeg" | "image/png" | "image/webp") {
        return Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Unsupported image media type".into(),
        ));
    }
    let predicted = encoded.len().saturating_add(3) / 4 * 3;
    if predicted > image_max_bytes() {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Image input exceeds the configured size limit".into(),
        ));
    }
    let (_, reservation) = state
        .resource_governor
        .reserve_memory_capacity(predicted as u64)
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Insufficient memory capacity for image input".into(),
            )
        })?;
    reservations.push(reservation);
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = BASE64
        .decode(padded.as_bytes())
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid image base64".into()))?;
    let detected = detect_image_media_type(&decoded).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Image bytes do not match the declared media type".into(),
        )
    })?;
    if detected != declared {
        return Err((
            StatusCode::BAD_REQUEST,
            "Image bytes do not match the declared media type".into(),
        ));
    }
    Ok(())
}

fn image_max_bytes() -> usize {
    std::env::var("RUST_IMAGE_INPUT_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_IMAGE_MAX_BYTES)
}

fn detect_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

async fn prepare_input(
    state: &AppState,
    request: Request,
    method: &Method,
    uri: &Uri,
    path: &str,
    resource_wait: Duration,
) -> Result<PreparedInput, Response<Body>> {
    if *method == Method::GET {
        let payload = if matches!(path, "/search" | "/v1/search") {
            let query = query_value(uri, "q").unwrap_or_else(|| "Jina+AI".into());
            Some(json!({
                "model": "search",
                "messages": [{"role":"user","content":query}],
                "stream": false,
            }))
        } else {
            let model = query_value(uri, "model").unwrap_or_else(|| "video".into());
            Some(json!({"model": model}))
        };
        return Ok(PreparedInput {
            payload,
            replay: None,
            observation: SpoolObservation::default(),
            default_model: if matches!(path, "/search" | "/v1/search") {
                "search".into()
            } else {
                String::new()
            },
            content_type: String::new(),
        });
    }

    let content_length = request
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();
    let normalized_content_type = content_type.to_ascii_lowercase();
    let (_, body) = request.into_parts();
    let spool = read_spooled_body(
        body,
        &state.request_spool,
        None,
        content_length,
        resource_wait,
    )
    .await
    .map_err(|error| match error {
        RequestBodySpoolError::Timeout => {
            json_error(StatusCode::REQUEST_TIMEOUT, "Request body upload timed out")
        }
        RequestBodySpoolError::Read => {
            json_error(StatusCode::BAD_REQUEST, "Request body upload failed")
        }
        RequestBodySpoolError::Spool(failure) => json_error(failure.status, &failure.message),
    })?;
    if normalized_content_type.starts_with("application/json") {
        let payload = spool
            .storage
            .parse_json()
            .await
            .map_err(|error| json_error(StatusCode::BAD_REQUEST, &error))?;
        if !payload.is_object() {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                "JSON request body must be an object",
            ));
        }
        return Ok(PreparedInput {
            payload: Some(payload),
            replay: None,
            observation: spool.observation,
            default_model: String::new(),
            content_type,
        });
    }
    let model = match query_value(uri, "model") {
        Some(model) => model,
        None if normalized_content_type.starts_with("multipart/form-data") => spool
            .storage
            .multipart_text_field(&content_type, "model", 4096)
            .await
            .map_err(|error| json_error(StatusCode::BAD_REQUEST, &error))?
            .unwrap_or_else(|| {
                if path == "/v1/images/edits" {
                    "gpt-image-2".into()
                } else {
                    String::new()
                }
            }),
        None => String::new(),
    };
    Ok(PreparedInput {
        payload: Some(json!({"model": model})),
        replay: Some((spool.storage, spool.observation.clone())),
        observation: spool.observation,
        default_model: String::new(),
        content_type,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_attempt(
    provider: &Provider,
    provider_key: &str,
    request_model: &str,
    original_model: &str,
    method: &Method,
    uri: &Uri,
    path: &str,
    incoming_headers: &HeaderMap,
    input: &PreparedInput,
    request_id: &str,
) -> Result<PreparedAttempt, String> {
    let is_alpha_search = path == ALPHA_SEARCH_ENDPOINT;
    let engine = provider.engine.trim().to_ascii_lowercase();
    let native_responses_wire = matches!(path, "/v1/responses" | "/v1/responses/compact")
        && (engine == "codex"
            || (engine == "gpt"
                && provider
                    .base_url
                    .to_ascii_lowercase()
                    .contains("/responses")));
    let downstream_protocol = if path == "/v1/responses" && !native_responses_wire {
        DownstreamProtocol::ResponsesCompat
    } else {
        DownstreamProtocol::Native
    };
    let downstream_stream = !is_alpha_search
        && input
            .payload
            .as_ref()
            .and_then(|payload| payload.get("stream"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let provider_stream = downstream_stream;
    let chat_stream_include_usage = path == "/v1/chat/completions"
        && input
            .payload
            .as_ref()
            .and_then(|payload| payload.pointer("/stream_options/include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let is_search = matches!(path, "/search" | "/v1/search");
    let jina_search = is_search
        && (provider.name.eq_ignore_ascii_case("jina")
            || provider
                .base_url
                .to_ascii_lowercase()
                .contains("api.jina.ai"));
    let proxy = provider
        .preferences
        .get("proxy")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let _ = proxy;

    if let Some((storage, observation)) = &input.replay {
        let url = endpoint_url(provider.base_url.as_ref(), path, method, uri)?;
        let dashscope_transcription = path == "/v1/audio/transcriptions"
            && provider
                .base_url
                .to_ascii_lowercase()
                .contains("dashscope.aliyuncs.com");
        let multipart_boundary = (!dashscope_transcription
            && input.content_type.starts_with("multipart/form-data"))
        .then(|| multipart_output_boundary(request_id));
        let outgoing_content_type = multipart_boundary
            .as_ref()
            .map(|boundary| format!("multipart/form-data; boundary={boundary}"))
            .or_else(|| dashscope_transcription.then(|| "application/json".into()));
        let mut headers = provider_headers(
            provider,
            provider_key,
            incoming_headers,
            request_id,
            &engine,
            false,
            outgoing_content_type
                .as_deref()
                .or(Some(input.content_type.as_str())),
        )?;
        headers.remove("content-length");
        let body = if dashscope_transcription {
            AttemptBody::DashscopeTranscription {
                storage: storage.clone_for_replay(),
                observation: observation.clone(),
                source_content_type: input.content_type.clone(),
                model: original_model.to_owned(),
                provider_key: provider_key.to_owned(),
            }
        } else if let Some(boundary) = multipart_boundary {
            AttemptBody::MultipartRewrite {
                storage: storage.clone_for_replay(),
                observation: observation.clone(),
                source_content_type: input.content_type.clone(),
                boundary,
                model: original_model.to_owned(),
            }
        } else {
            AttemptBody::Replay(storage.clone_for_replay(), observation.clone())
        };
        return Ok(PreparedAttempt {
            method: method.clone(),
            url,
            headers,
            body,
            adapter: ResponseAdapter::Passthrough,
            downstream_stream: false,
            upstream_stream: false,
            request_model: request_model.to_owned(),
            original_model: original_model.to_owned(),
            downstream_protocol,
            chat_stream_include_usage,
            provider_key: provider_key.to_owned(),
            estimated_video_tokens: None,
        });
    }

    let mut payload = input.payload.clone().unwrap_or_else(|| json!({}));
    if downstream_protocol == DownstreamProtocol::ResponsesCompat {
        payload = responses_to_chat_request(&payload, original_model)?;
    }
    let wire_path = if downstream_protocol == DownstreamProtocol::ResponsesCompat {
        "/v1/chat/completions"
    } else {
        path
    };
    let estimated_video_tokens = (path == "/v1/video/tasks")
        .then(|| estimate_video_tokens(&payload))
        .flatten();
    let (url, adapter, upstream_stream) = match engine.as_str() {
        "codex" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_responses(&payload, original_model)?;
            (
                responses_url(provider.base_url.as_ref()),
                ResponseAdapter::ResponsesToChat,
                provider_stream,
            )
        }
        "gpt" | "openrouter" | "azure" | "azure-databricks" | "cloudflare"
            if wire_path == "/v1/chat/completions"
                && provider
                    .base_url
                    .to_ascii_lowercase()
                    .contains("/responses") =>
        {
            payload = chat_to_responses(&payload, original_model)?;
            (
                responses_url(provider.base_url.as_ref()),
                ResponseAdapter::ResponsesToChat,
                provider_stream,
            )
        }
        "gemini" | "vertex" | "vertex-gemini" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_gemini(&payload, original_model)?;
            (
                if matches!(engine.as_str(), "vertex" | "vertex-gemini") {
                    vertex_gemini_url(provider, original_model, provider_key, provider_stream)?
                } else {
                    gemini_url(
                        provider.base_url.as_ref(),
                        original_model,
                        provider_key,
                        provider_stream,
                    )?
                },
                ResponseAdapter::GeminiToChat,
                provider_stream,
            )
        }
        "vertex-claude" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_claude(&payload, original_model)?;
            (
                vertex_claude_url(provider, original_model)?,
                ResponseAdapter::ClaudeToChat,
                provider_stream,
            )
        }
        "claude" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_claude(&payload, original_model)?;
            (
                messages_url(provider.base_url.as_ref()),
                ResponseAdapter::ClaudeToChat,
                provider_stream,
            )
        }
        "aws" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_claude(&payload, original_model)?;
            if let Some(root) = payload.as_object_mut() {
                root.remove("model");
                root.remove("stream");
                root.insert(
                    "anthropic_version".into(),
                    Value::String("bedrock-2023-05-31".into()),
                );
            }
            (
                aws_bedrock_url(provider, original_model, provider_stream)?,
                ResponseAdapter::AwsToChat,
                provider_stream,
            )
        }
        "cohere" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_cohere(&payload, original_model)?;
            (
                provider.base_url.to_string(),
                ResponseAdapter::CohereToChat,
                provider_stream,
            )
        }
        "doubao-translation" if wire_path == "/v1/chat/completions" => {
            payload =
                chat_to_doubao_translation(&payload, original_model, request_model, provider)?;
            (
                provider.base_url.to_string(),
                ResponseAdapter::ResponsesToChat,
                provider_stream,
            )
        }
        "azure" if wire_path == "/v1/chat/completions" => {
            set_model(&mut payload, original_model)?;
            normalize_azure_token_limit(&mut payload, original_model);
            (
                azure_chat_url(provider.base_url.as_ref(), original_model)?,
                ResponseAdapter::Passthrough,
                provider_stream,
            )
        }
        "azure-databricks" if wire_path == "/v1/chat/completions" => {
            set_model(&mut payload, original_model)?;
            (
                databricks_chat_url(provider.base_url.as_ref(), original_model)?,
                ResponseAdapter::Passthrough,
                provider_stream,
            )
        }
        "cloudflare" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_cloudflare(&payload)?;
            (
                cloudflare_url(provider, original_model)?,
                ResponseAdapter::CloudflareToChat,
                provider_stream,
            )
        }
        "claude" if wire_path == "/v1/messages" => {
            set_model(&mut payload, original_model)?;
            (
                messages_url(provider.base_url.as_ref()),
                ResponseAdapter::Passthrough,
                provider_stream,
            )
        }
        "lingjing" if path == "/v1/video/tasks" => {
            payload = content_generation_to_lingjing(&payload, original_model)?;
            (
                lingjing_url(provider.base_url.as_ref(), "/draw/task/submit", None)?,
                ResponseAdapter::LingjingVideo,
                false,
            )
        }
        "lingjing" if path.starts_with("/v1/video/tasks/") => {
            let task_id = path.trim_start_matches("/v1/video/tasks/");
            let query = url::form_urlencoded::Serializer::new(String::new())
                .append_pair("taskId", task_id)
                .finish();
            (
                lingjing_url(provider.base_url.as_ref(), "/draw/task/query", Some(&query))?,
                ResponseAdapter::LingjingVideo,
                false,
            )
        }
        "lingjing"
            if path == "/v1/asset-groups"
                || path.starts_with("/v1/asset-groups/")
                || path == "/v1/assets"
                || path.starts_with("/v1/assets/") =>
        {
            (
                endpoint_url(provider.base_url.as_ref(), wire_path, method, uri)?,
                ResponseAdapter::Passthrough,
                false,
            )
        }
        _ if path == "/v1/audio/speech"
            && provider
                .base_url
                .to_ascii_lowercase()
                .contains("api.minimaxi.com") =>
        {
            payload = openai_tts_to_minimax(&payload, original_model)?;
            (
                endpoint_url(provider.base_url.as_ref(), wire_path, method, uri)?,
                ResponseAdapter::Passthrough,
                false,
            )
        }
        _ if path == "/v1/embeddings"
            && provider
                .base_url
                .to_ascii_lowercase()
                .starts_with("https://api.jina.ai") =>
        {
            normalize_jina_embedding(&mut payload, original_model)?;
            (
                endpoint_url(provider.base_url.as_ref(), wire_path, method, uri)?,
                ResponseAdapter::Passthrough,
                false,
            )
        }
        _ if matches!(path, "/search" | "/v1/search") => {
            let query = search_query(&payload)?;
            payload = if jina_search {
                json!({"q":query})
            } else {
                let defaults = provider
                    .preferences
                    .get("search_defaults")
                    .and_then(Value::as_object);
                json!({
                    "query":query,
                    "topic":defaults.and_then(|value| value.get("topic")).cloned().unwrap_or_else(|| json!("general")),
                    "search_depth":defaults.and_then(|value| value.get("search_depth")).cloned().unwrap_or_else(|| json!("basic")),
                    "chunks_per_source":defaults.and_then(|value| value.get("chunks_per_source")).cloned().unwrap_or_else(|| json!(3)),
                    "max_results":defaults.and_then(|value| value.get("max_results")).cloned().unwrap_or_else(|| json!(7)),
                })
            };
            (
                if jina_search {
                    let mut url = Url::parse("https://s.jina.ai/")
                        .map_err(|error| format!("invalid Jina search URL: {error}"))?;
                    url.query_pairs_mut().append_pair(
                        "q",
                        payload.get("q").and_then(Value::as_str).unwrap_or_default(),
                    );
                    url.to_string()
                } else {
                    endpoint_url(provider.base_url.as_ref(), wire_path, method, uri)?
                },
                ResponseAdapter::Search,
                false,
            )
        }
        _ => {
            set_model(&mut payload, original_model)?;
            (
                endpoint_url(provider.base_url.as_ref(), wire_path, method, uri)?,
                ResponseAdapter::Passthrough,
                provider_stream,
            )
        }
    };
    if !matches!(
        adapter,
        ResponseAdapter::GeminiToChat | ResponseAdapter::AwsToChat | ResponseAdapter::LingjingVideo
    ) && engine != "vertex-claude"
        && !is_alpha_search
    {
        if let Some(root) = payload.as_object_mut() {
            root.insert("stream".into(), Value::Bool(upstream_stream));
        }
    }
    if let Some(root) = payload.as_object_mut() {
        if is_alpha_search {
            sanitize_alpha_search_payload(root);
        } else {
            apply_overrides(root, provider, request_model);
            if path == "/v1/responses/compact" {
                root.remove("store");
            }
            if engine == "doubao-translation" {
                root.remove("translation_options");
            }
        }
    }
    let mut headers = provider_headers(
        provider,
        provider_key,
        incoming_headers,
        request_id,
        &engine,
        upstream_stream,
        None,
    )?;
    if is_alpha_search && engine == "codex" {
        apply_alpha_search_headers(&mut headers, incoming_headers, &payload)?;
    }
    if jina_search {
        headers.insert("accept", HeaderValue::from_static("application/json"));
        headers.insert("x-respond-with", HeaderValue::from_static("no-content"));
        headers.remove("content-type");
    }
    let outgoing_method = if jina_search {
        Method::GET
    } else if is_search {
        Method::POST
    } else {
        method.clone()
    };
    let body = if outgoing_method == Method::GET {
        AttemptBody::Empty
    } else {
        let body = serde_json::to_vec(&payload)
            .map_err(|error| format!("encode upstream request body: {error}"))?;
        if engine == "aws" {
            sign_aws_request(provider, &url, &body, &mut headers)?;
        }
        AttemptBody::Json(body)
    };
    Ok(PreparedAttempt {
        method: outgoing_method,
        url,
        headers,
        body,
        adapter,
        downstream_stream,
        upstream_stream,
        request_model: request_model.to_owned(),
        original_model: original_model.to_owned(),
        downstream_protocol,
        chat_stream_include_usage,
        provider_key: provider_key.to_owned(),
        estimated_video_tokens,
    })
}

fn search_query(payload: &Value) -> Result<String, String> {
    let query = payload
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .filter_map(Value::as_object)
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .map(extract_translation_text)
        .or_else(|| payload.get("q").and_then(value_text))
        .unwrap_or_default();
    let query = query.trim();
    if query.is_empty() {
        Err("Missing search query".into())
    } else {
        Ok(query.to_owned())
    }
}

fn multipart_output_boundary(request_id: &str) -> String {
    let digest = Sha256::digest(request_id.as_bytes());
    format!("uni-api-{}", hex_bytes(&digest[..12]))
}

async fn multipart_rewrite_body(
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    boundary: String,
    model: String,
) -> Result<reqwest::Body, String> {
    Ok(reqwest::Body::wrap_stream(
        multipart_rewrite_stream(storage, observation, source_content_type, boundary, model)
            .await?,
    ))
}

async fn prepare_dashscope_transcription(
    client: &reqwest::Client,
    mut headers: HeaderMap,
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    model: &str,
    provider_key: &str,
) -> Result<(HeaderMap, Vec<u8>), String> {
    let audio = storage
        .multipart_file(source_content_type, "file")
        .await?
        .ok_or_else(|| "audio transcription requires multipart file field".to_owned())?;
    let certificate_response = client
        .get("https://dashscope.aliyuncs.com/api/v1/uploads")
        .bearer_auth(provider_key)
        .header("accept-encoding", "identity")
        .query(&[("action", "getPolicy"), ("model", model)])
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("request DashScope upload certificate: {error}"))?;
    if !certificate_response.status().is_success() {
        return Err(format!(
            "DashScope upload certificate returned HTTP {}",
            certificate_response.status()
        ));
    }
    let certificate_body = read_limited_upstream_body(certificate_response, 256 * 1024).await?;
    let certificate: Value = serde_json::from_slice(&certificate_body)
        .map_err(|error| format!("decode DashScope upload certificate: {error}"))?;
    let data = certificate
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| "DashScope upload certificate is missing data".to_owned())?;
    let field = |name: &str, maximum: usize| -> Result<String, String> {
        let value = data
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("DashScope upload certificate is missing {name}"))?;
        if value.len() > maximum {
            return Err(format!(
                "DashScope upload certificate field {name} is too large"
            ));
        }
        Ok(value.to_owned())
    };
    let upload_host = field("upload_host", 2048)?;
    let upload_dir = field("upload_dir", 1024)?;
    let parsed_upload_host = Url::parse(&upload_host)
        .map_err(|error| format!("invalid DashScope upload host: {error}"))?;
    if !matches!(parsed_upload_host.scheme(), "http" | "https")
        || parsed_upload_host.host_str().is_none()
    {
        return Err("DashScope upload host must be HTTP(S)".into());
    }
    if audio.filename.len() > 512 {
        return Err("DashScope upload filename exceeds 512 bytes".into());
    }
    let object_key = format!("{upload_dir}/{}", audio.filename);
    let audio_body = audio
        .storage
        .into_body(&observation)
        .await
        .map_err(|error| error.message)?;
    let mut part = reqwest::multipart::Part::stream_with_length(
        reqwest::Body::wrap_stream(audio_body.into_data_stream()),
        audio.bytes,
    )
    .file_name(audio.filename);
    if let Some(content_type) = audio.content_type {
        part = part
            .mime_str(&content_type)
            .map_err(|error| format!("invalid audio MIME type: {error}"))?;
    }
    let form = reqwest::multipart::Form::new()
        .text("key", object_key.clone())
        .text("policy", field("policy", 64 * 1024)?)
        .text("OSSAccessKeyId", field("oss_access_key_id", 4096)?)
        .text("signature", field("signature", 64 * 1024)?)
        .text("success_action_status", "200")
        .text("x-oss-object-acl", field("x_oss_object_acl", 256)?)
        .text(
            "x-oss-forbid-overwrite",
            field("x_oss_forbid_overwrite", 256)?,
        )
        .part("file", part);
    let upload = client
        .post(upload_host)
        .timeout(Duration::from_secs(3600))
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("upload DashScope transcription input: {error}"))?;
    if !upload.status().is_success() {
        return Err(format!(
            "DashScope OSS upload returned HTTP {}",
            upload.status()
        ));
    }
    let mut payload = json!({
        "model":model,
        "input":{"messages":[{"role":"user","content":[{"audio":format!("oss://{object_key}")}]}]},
    });
    for field_name in [
        "prompt",
        "response_format",
        "temperature",
        "language",
        "timestamp_granularities[]",
    ] {
        if let Some(value) = storage
            .multipart_text_field(source_content_type, field_name, 64 * 1024)
            .await?
            .filter(|value| !value.is_empty())
        {
            payload[field_name] = Value::String(value);
        }
    }
    headers.remove("content-length");
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert(
        "x-dashscope-ossresourceresolve",
        HeaderValue::from_static("enable"),
    );
    serde_json::to_vec(&payload)
        .map(|body| (headers, body))
        .map_err(|error| format!("encode DashScope transcription request: {error}"))
}

async fn multipart_rewrite_stream(
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    boundary: String,
    model: String,
) -> Result<tokio_stream::wrappers::ReceiverStream<Result<Bytes, io::Error>>, String> {
    let source_boundary = multer::parse_boundary(source_content_type)
        .map_err(|error| format!("invalid multipart boundary: {error}"))?;
    let body = storage
        .into_body(&observation)
        .await
        .map_err(|error| error.message)?;
    let stream = body.into_data_stream();
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, io::Error>>(8);
    tokio::spawn(async move {
        let mut multipart = multer::Multipart::new(stream, source_boundary);
        let mut rewrote_model = false;
        loop {
            let mut field = match multipart.next_field().await {
                Ok(Some(field)) => field,
                Ok(None) => break,
                Err(error) => {
                    let _ = sender
                        .send(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("parse multipart request: {error}"),
                        )))
                        .await;
                    return;
                }
            };
            let rewrite_model = field.name() == Some("model");
            rewrote_model |= rewrite_model;
            let mut prefix = Vec::new();
            prefix.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            for (name, value) in field.headers() {
                prefix.extend_from_slice(name.as_str().as_bytes());
                prefix.extend_from_slice(b": ");
                prefix.extend_from_slice(value.as_bytes());
                prefix.extend_from_slice(b"\r\n");
            }
            prefix.extend_from_slice(b"\r\n");
            if sender.send(Ok(Bytes::from(prefix))).await.is_err() {
                return;
            }
            if rewrite_model && sender.send(Ok(Bytes::from(model.clone()))).await.is_err() {
                return;
            }
            loop {
                match field.chunk().await {
                    Ok(Some(chunk)) if !rewrite_model => {
                        if sender.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(error) => {
                        let _ = sender
                            .send(Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("read multipart field: {error}"),
                            )))
                            .await;
                        return;
                    }
                }
            }
            if sender.send(Ok(Bytes::from_static(b"\r\n"))).await.is_err() {
                return;
            }
        }
        if !rewrote_model {
            let field = format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            );
            if sender.send(Ok(Bytes::from(field))).await.is_err() {
                return;
            }
        }
        let _ = sender
            .send(Ok(Bytes::from(format!("--{boundary}--\r\n"))))
            .await;
    });
    Ok(tokio_stream::wrappers::ReceiverStream::new(receiver))
}

fn sanitize_alpha_search_payload(root: &mut Map<String, Value>) {
    for field in [
        "store",
        "stream",
        "prompt_cache_key",
        "prompt_cache_retention",
    ] {
        root.remove(field);
    }
}

fn apply_alpha_search_headers(
    headers: &mut HeaderMap,
    incoming: &HeaderMap,
    payload: &Value,
) -> Result<(), String> {
    if headers.get("openai-beta").is_none() {
        let value = incoming
            .get("openai-beta")
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("responses=experimental"));
        headers.insert("openai-beta", value);
    }
    if headers.get("originator").is_none() {
        let value = incoming
            .get("originator")
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("codex_cli_rs"));
        headers.insert("originator", value);
    }
    if let Some(session_id) = payload
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        headers.insert(
            "session_id",
            HeaderValue::from_str(session_id)
                .map_err(|_| "alpha/search id is not a valid header value".to_owned())?,
        );
    }
    headers.insert("user-agent", HeaderValue::from_static(CODEX_USER_AGENT));
    headers.insert("accept", HeaderValue::from_static("application/json"));
    Ok(())
}

struct AttemptSuccess {
    response: Response<Body>,
    status: StatusCode,
    usage: (i64, i64, i64),
    usage_completion: Option<tokio::sync::oneshot::Receiver<(i64, i64, i64)>>,
    upstream_url: String,
}

struct AttemptFailure {
    status: StatusCode,
    detail: String,
    upstream_url: String,
    response: Option<Response<Body>>,
}

async fn send_attempt(
    state: &AppState,
    provider: &Provider,
    mut prepared: PreparedAttempt,
    incoming_headers: &HeaderMap,
    endpoint: &str,
) -> Result<AttemptSuccess, AttemptFailure> {
    let proxy = provider.preferences.get("proxy").and_then(Value::as_str);
    let http1_only = provider.engine.eq_ignore_ascii_case("codex");
    let timeouts = state
        .native_responses_config
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
    let base_timeout = provider_timeout(provider, &prepared.original_model);
    let total_timeout = positive_duration(timeouts.total).unwrap_or(base_timeout);
    let send_timeout = [
        timeouts.first_byte,
        timeouts.write,
        timeouts.pool,
        timeouts.total,
    ]
    .into_iter()
    .flatten()
    .filter(|value| value.is_finite() && *value > 0.0)
    .min_by(f64::total_cmp)
    .map(Duration::from_secs_f64)
    .unwrap_or(base_timeout);
    if matches!(
        provider.engine.to_ascii_lowercase().as_str(),
        "vertex" | "vertex-gemini" | "vertex-claude"
    ) && provider.client_email.is_some()
        && provider.private_key.is_some()
    {
        let token = vertex_access_token(state, provider)
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
        .headers(prepared.headers.clone())
        .timeout(total_timeout);
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
            client
                .request(prepared.method.clone(), &prepared.url)
                .headers(headers)
                .timeout(total_timeout)
                .body(body)
        }
        AttemptBody::Empty => request,
    };
    let response = tokio::time::timeout(send_timeout, request.send())
        .await
        .map_err(|_| AttemptFailure {
            status: StatusCode::GATEWAY_TIMEOUT,
            detail: "Upstream response headers timed out".into(),
            upstream_url: prepared.url.clone(),
            response: None,
        })?
        .map_err(|error| AttemptFailure {
            status: StatusCode::BAD_GATEWAY,
            detail: format!("Upstream transport error: {error}"),
            upstream_url: prepared.url.clone(),
            response: None,
        })?;
    let status = response.status();
    if !status.is_success() {
        let headers = filtered_response_headers(response.headers());
        let body = read_limited_upstream_body(response, UPSTREAM_ERROR_MAX_BYTES)
            .await
            .unwrap_or_else(|error| Bytes::from(format!("read upstream error response: {error}")));
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
        let headers = filtered_response_headers(response.headers());
        let mut output = Response::new(Body::from_stream(response.bytes_stream()));
        *output.status_mut() = status;
        *output.headers_mut() = headers;
        output
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Ok(AttemptSuccess {
            response: output,
            status,
            usage: (0, 0, 0),
            usage_completion: None,
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
            ResponseAdapter::LingjingVideo => unreachable!(),
            ResponseAdapter::Passthrough => StreamProtocol::Chat,
            ResponseAdapter::Search => unreachable!(),
        };
        let output_protocol = if prepared.downstream_protocol == DownstreamProtocol::ResponsesCompat
        {
            StreamOutputProtocol::Responses
        } else {
            StreamOutputProtocol::Chat
        };
        let mut translation = provider_stream::translate(
            response,
            protocol,
            output_protocol,
            prepared.request_model.clone(),
            prepared.chat_stream_include_usage,
            timeouts.idle,
            timeouts.total,
        );
        translation
            .response
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
        return Ok(AttemptSuccess {
            response: translation.response,
            status,
            usage: (0, 0, 0),
            usage_completion: Some(translation.usage),
            upstream_url: prepared.url,
        });
    }
    if prepared.adapter == ResponseAdapter::Passthrough
        && prepared.downstream_protocol == DownstreamProtocol::Native
    {
        let headers = filtered_response_headers(response.headers());
        let body = read_limited_upstream_body(response, upstream_response_max_bytes())
            .await
            .map_err(|error| AttemptFailure {
                status: StatusCode::BAD_GATEWAY,
                detail: format!("Read upstream response failed: {error}"),
                upstream_url: prepared.url.clone(),
                response: None,
            })?;
        let usage = serde_json::from_slice::<Value>(&body)
            .ok()
            .map(|value| usage(&value))
            .unwrap_or((0, 0, 0));
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
            usage_completion: None,
            upstream_url: prepared.url,
        });
    }
    let upstream = if prepared.adapter == ResponseAdapter::Search {
        let bytes = read_limited_upstream_body(response, upstream_response_max_bytes())
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
        let bytes = read_limited_upstream_body(response, upstream_response_max_bytes())
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
        ResponseAdapter::Passthrough => upstream,
    };
    let normalized = if prepared.downstream_protocol == DownstreamProtocol::ResponsesCompat {
        chat_to_responses_response(&normalized, &prepared.request_model)
    } else {
        normalized
    };
    if prepared.adapter == ResponseAdapter::LingjingVideo && prepared.method == Method::POST {
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
    output
        .headers_mut()
        .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
    Ok(AttemptSuccess {
        response: output,
        status: StatusCode::OK,
        usage,
        usage_completion: None,
        upstream_url: prepared.url,
    })
}

async fn read_limited_upstream_body(
    response: reqwest::Response,
    maximum: usize,
) -> Result<Bytes, String> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err(format!(
            "upstream response exceeds the configured {maximum} byte limit"
        ));
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        if body.len().saturating_add(chunk.len()) > maximum {
            return Err(format!(
                "upstream response exceeds the configured {maximum} byte limit"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

fn upstream_response_max_bytes() -> usize {
    std::env::var("RUST_GENERIC_UPSTREAM_RESPONSE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_UPSTREAM_RESPONSE_MAX_BYTES)
}

fn provider_headers(
    provider: &Provider,
    provider_key: &str,
    incoming: &HeaderMap,
    request_id: &str,
    engine: &str,
    stream: bool,
    content_type: Option<&str>,
) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    if let Some(content_type) = content_type.filter(|value| !value.is_empty()) {
        headers.insert(
            "content-type",
            HeaderValue::from_str(content_type)
                .map_err(|_| "request Content-Type is not a valid header".to_owned())?,
        );
    }
    if engine == "lingjing" {
        let access_key = provider
            .preferences
            .get("access_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "Lingjing provider requires preferences.access_key".to_owned())?;
        let secret_key = provider
            .preferences
            .get("secret_key")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "Lingjing provider requires preferences.secret_key".to_owned())?;
        headers.insert(
            "x-access-key",
            HeaderValue::from_str(access_key)
                .map_err(|_| "Lingjing access key is not a valid header value".to_owned())?,
        );
        headers.insert(
            "x-secret-key",
            HeaderValue::from_str(secret_key)
                .map_err(|_| "Lingjing secret key is not a valid header value".to_owned())?,
        );
    } else if engine == "claude" {
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(provider_key)
                .map_err(|_| "provider API key is not a valid header".to_owned())?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert(
            "anthropic-beta",
            HeaderValue::from_static("tools-2024-05-16"),
        );
    } else if engine == "azure" {
        headers.insert(
            "api-key",
            HeaderValue::from_str(provider_key)
                .map_err(|_| "provider API key is not a valid header".to_owned())?,
        );
    } else if engine == "azure-databricks" {
        let encoded = BASE64.encode(format!("token:{provider_key}"));
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Basic {encoded}"))
                .map_err(|_| "provider API key is not a valid header".to_owned())?,
        );
    } else if !matches!(
        engine,
        "gemini" | "vertex" | "vertex-gemini" | "vertex-claude" | "aws"
    ) {
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {provider_key}"))
                .map_err(|_| "provider API key is not a valid header".to_owned())?,
        );
    }
    if let Ok(value) = HeaderValue::from_str(request_id) {
        headers.insert("x-request-id", value.clone());
        headers.insert("x-caller-request-id", value);
    }
    if stream {
        headers.insert("accept", HeaderValue::from_static("text/event-stream"));
    }
    if engine == "openrouter" && provider.base_url.contains("openrouter.ai") {
        headers.insert(
            "http-referer",
            HeaderValue::from_static("https://github.com/yym68686/uni-api"),
        );
        headers.insert("x-title", HeaderValue::from_static("Uni API"));
    }
    if let Some(extra) = provider
        .preferences
        .get("headers")
        .and_then(Value::as_object)
    {
        for (name, value) in extra {
            let Some(value) = value.as_str() else {
                continue;
            };
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| format!("provider {} has an invalid header", provider.name))?;
            let value = HeaderValue::from_str(value)
                .map_err(|_| format!("provider {} has an invalid header value", provider.name))?;
            headers.insert(name, value);
        }
    }
    let passthrough = provider
        .preferences
        .get("passthrough_request_headers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    for name in passthrough {
        if let (Ok(header_name), Some(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            incoming.get(name).cloned(),
        ) {
            headers.insert(header_name, value);
        }
    }
    Ok(headers)
}

fn set_model(payload: &mut Value, model: &str) -> Result<(), String> {
    payload
        .as_object_mut()
        .ok_or_else(|| "request body must be a JSON object".to_owned())?
        .insert("model".into(), Value::String(model.to_owned()));
    Ok(())
}

fn responses_to_chat_request(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "responses request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    if let Some(instructions) = root
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        messages.push(json!({"role":"system","content":instructions}));
    }
    match root.get("input") {
        Some(Value::String(text)) => messages.push(json!({"role":"user","content":text})),
        Some(Value::Array(items)) => {
            for item in items {
                let item_type = item.get("type").and_then(Value::as_str);
                match item_type {
                    Some("function_call") => messages.push(json!({
                        "role":"assistant",
                        "content":Value::Null,
                        "tool_calls":[{
                            "id":item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Value::Null),
                            "type":"function",
                            "function":{
                                "name":item.get("name").cloned().unwrap_or(Value::Null),
                                "arguments":item.get("arguments").cloned().unwrap_or_else(|| Value::String("{}".into())),
                            }
                        }]
                    })),
                    Some("function_call_output") => messages.push(json!({
                        "role":"tool",
                        "tool_call_id":item.get("call_id").cloned().unwrap_or(Value::Null),
                        "content":item.get("output").cloned().unwrap_or_else(|| Value::String(String::new())),
                    })),
                    Some("message") | None if item.get("role").is_some() => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        messages.push(json!({
                            "role":role,
                            "content":responses_input_to_chat_content(item.get("content")),
                        }));
                    }
                    _ => {}
                }
            }
        }
        Some(value) if !value.is_null() => {
            messages.push(json!({"role":"user","content":value.clone()}));
        }
        _ => {}
    }
    if messages.is_empty() {
        return Err("responses request requires input".into());
    }

    let mut output = Map::new();
    output.insert("model".into(), Value::String(original_model.to_owned()));
    output.insert("messages".into(), Value::Array(messages));
    output.insert(
        "stream".into(),
        Value::Bool(root.get("stream").and_then(Value::as_bool).unwrap_or(false)),
    );
    if let Some(value) = root.get("max_output_tokens") {
        output.insert("max_tokens".into(), value.clone());
    }
    for name in [
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "service_tier",
        "reasoning",
        "reasoning_effort",
        "modalities",
        "audio",
        "metadata",
        "user",
    ] {
        if let Some(value) = root.get(name) {
            output.insert(name.into(), value.clone());
        }
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        output.insert(
            "tools".into(),
            Value::Array(
                tools
                    .iter()
                    .filter_map(|tool| {
                        if tool.get("type").and_then(Value::as_str) != Some("function") {
                            return None;
                        }
                        let function = tool.get("function").unwrap_or(tool);
                        let mut definition = Map::new();
                        definition.insert(
                            "name".into(),
                            function.get("name").cloned().unwrap_or(Value::Null),
                        );
                        if let Some(value) = function.get("description") {
                            definition.insert("description".into(), value.clone());
                        }
                        definition.insert(
                            "parameters".into(),
                            function
                                .get("parameters")
                                .cloned()
                                .unwrap_or_else(|| json!({"type":"object","properties":{}})),
                        );
                        if let Some(value) = function.get("strict") {
                            definition.insert("strict".into(), value.clone());
                        }
                        Some(json!({"type":"function","function":definition}))
                    })
                    .collect(),
            ),
        );
    }
    if let Some(choice) = root.get("tool_choice") {
        let choice = if choice.get("type").and_then(Value::as_str) == Some("function") {
            json!({
                "type":"function",
                "function":{"name":choice.get("name").cloned().unwrap_or(Value::Null)}
            })
        } else {
            choice.clone()
        };
        output.insert("tool_choice".into(), choice);
    }
    Ok(Value::Object(output))
}

fn responses_input_to_chat_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return Value::String(String::new());
    };
    if let Some(text) = content.as_str() {
        return Value::String(text.to_owned());
    }
    Value::Array(
        content
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("input_text" | "output_text" | "text") => Some(json!({
                    "type":"text",
                    "text":part.get("text").cloned().unwrap_or_else(|| Value::String(String::new())),
                })),
                Some("input_image") => Some(json!({
                    "type":"image_url",
                    "image_url":{"url":part.get("image_url").cloned().unwrap_or(Value::Null)},
                })),
                Some("input_audio") => Some(json!({
                    "type":"input_audio",
                    "input_audio":part.get("input_audio").cloned().unwrap_or_else(|| {
                        json!({
                            "data":part.get("data").cloned().unwrap_or(Value::Null),
                            "format":part.get("format").cloned().unwrap_or(Value::Null),
                        })
                    }),
                })),
                _ => None,
            })
            .collect(),
    )
}

fn chat_to_responses(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut items = Vec::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(message) = message.as_object() else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if role == "tool" {
            items.push(json!({
                "type":"function_call_output",
                "call_id": message.get("tool_call_id").cloned().unwrap_or(Value::Null),
                "output": message.get("content").cloned().unwrap_or(Value::String(String::new())),
            }));
            continue;
        }
        if let Some(content) = message.get("content") {
            items.push(json!({"role":role,"content":responses_content(content, role)}));
        }
        for tool in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            items.push(json!({
                "type":"function_call",
                "call_id":tool.get("id").cloned().unwrap_or(Value::Null),
                "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments":tool.pointer("/function/arguments").cloned().unwrap_or(Value::String("{}".into())),
            }));
        }
    }
    let mut output = Map::new();
    output.insert("model".into(), Value::String(original_model.to_owned()));
    output.insert("input".into(), Value::Array(items));
    output.insert("stream".into(), Value::Bool(false));
    output.insert("store".into(), Value::Bool(false));
    if let Some(max_tokens) = root.get("max_tokens") {
        output.insert("max_output_tokens".into(), max_tokens.clone());
    }
    for name in [
        "temperature",
        "top_p",
        "reasoning",
        "parallel_tool_calls",
        "service_tier",
    ] {
        if let Some(value) = root.get(name) {
            output.insert(name.into(), value.clone());
        }
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        output.insert(
            "tools".into(),
            Value::Array(
                tools
                    .iter()
                    .filter_map(|tool| {
                        let function = tool.get("function")?;
                        Some(json!({
                            "type":"function",
                            "name":function.get("name").cloned().unwrap_or(Value::Null),
                            "description":function.get("description").cloned().unwrap_or(Value::Null),
                            "parameters":function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                            "strict":function.get("strict").and_then(Value::as_bool).unwrap_or(false),
                        }))
                    })
                    .collect(),
            ),
        );
    }
    if let Some(choice) = root.get("tool_choice") {
        output.insert("tool_choice".into(), chat_tool_choice_to_responses(choice));
    }
    Ok(Value::Object(output))
}

fn chat_tool_choice_to_responses(choice: &Value) -> Value {
    let Some(root) = choice.as_object() else {
        return choice.clone();
    };
    match root.get("type").and_then(Value::as_str) {
        Some("function") => json!({
            "type":"function",
            "name":choice
                .pointer("/function/name")
                .or_else(|| choice.get("name"))
                .cloned()
                .unwrap_or(Value::Null),
        }),
        Some("allowed_tools") => {
            let allowed = root
                .get("allowed_tools")
                .and_then(Value::as_object)
                .unwrap_or(root);
            let tools = allowed
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|tool| {
                    if tool.get("type").and_then(Value::as_str) == Some("function") {
                        json!({
                            "type":"function",
                            "name":tool
                                .pointer("/function/name")
                                .or_else(|| tool.get("name"))
                                .cloned()
                                .unwrap_or(Value::Null),
                        })
                    } else {
                        tool.clone()
                    }
                })
                .collect::<Vec<_>>();
            json!({
                "type":"allowed_tools",
                "mode":allowed.get("mode").cloned().unwrap_or_else(|| Value::String("auto".into())),
                "tools":tools,
            })
        }
        _ => choice.clone(),
    }
}

fn responses_content(content: &Value, role: &str) -> Value {
    if let Some(text) = content.as_str() {
        return Value::Array(vec![json!({
            "type": if role == "assistant" { "output_text" } else { "input_text" },
            "text": text,
        })]);
    }
    let mut parts = Vec::new();
    for item in content.as_array().into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(json!({
                "type": if role == "assistant" { "output_text" } else { "input_text" },
                "text": item.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            Some("image_url") => parts.push(json!({
                "type":"input_image",
                "image_url":item.pointer("/image_url/url").cloned().unwrap_or(Value::Null),
            })),
            _ => parts.push(item.clone()),
        }
    }
    Value::Array(parts)
}

fn chat_to_gemini(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    let mut tool_names = HashMap::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for tool in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(id), Some(name)) = (
                tool.get("id").and_then(Value::as_str),
                tool.pointer("/function/name").and_then(Value::as_str),
            ) {
                tool_names.insert(id.to_owned(), name.to_owned());
            }
        }
    }
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let mut parts = gemini_parts(message.get("content").unwrap_or(&Value::Null))?;
        if role == "system" {
            system_parts.extend(parts);
        } else if role == "tool" {
            let call_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let name = tool_names
                .get(call_id)
                .cloned()
                .unwrap_or_else(|| "tool".into());
            let response = message
                .get("content")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            contents.push(json!({
                "role":"user",
                "parts":[{"functionResponse":{"name":name,"response":{"result":response}}}],
            }));
        } else {
            for tool in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let arguments = tool
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .and_then(|value| serde_json::from_str::<Value>(value).ok())
                    .unwrap_or_else(|| json!({}));
                let mut part = json!({
                    "functionCall":{
                        "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                        "args":arguments,
                    }
                });
                if let Some(signature) = tool
                    .get("id")
                    .and_then(Value::as_str)
                    .and_then(decode_gemini_thought_signature)
                {
                    part.as_object_mut()
                        .expect("Gemini function call part")
                        .insert("thoughtSignature".into(), Value::String(signature));
                }
                parts.push(part);
            }
            if !parts.is_empty() {
                contents.push(json!({
                    "role": if role == "assistant" { "model" } else { "user" },
                    "parts": parts,
                }));
            }
        }
    }
    if contents.is_empty() {
        contents.push(json!({"role":"user","parts":[{"text":"No messages"}]}));
    }
    let mut generation = Map::new();
    if let Some(value) = root.get("temperature") {
        generation.insert("temperature".into(), value.clone());
    }
    if let Some(value) = root.get("top_p") {
        generation.insert("topP".into(), value.clone());
    }
    generation.insert(
        "maxOutputTokens".into(),
        root.get("max_tokens").cloned().unwrap_or(json!(8192)),
    );
    let mut output = json!({
        "contents": contents,
        "generationConfig": generation,
        "safetySettings": [
            {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_NONE"}
        ],
        "_uni_api_model": original_model,
    });
    output
        .as_object_mut()
        .expect("Gemini payload object")
        .remove("_uni_api_model");
    if !system_parts.is_empty() {
        output
            .as_object_mut()
            .expect("Gemini payload object")
            .insert("systemInstruction".into(), json!({"parts":system_parts}));
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        let declarations = tools
            .iter()
            .filter_map(|tool| {
                let mut function = tool.get("function")?.clone();
                if let Some(function) = function.as_object_mut() {
                    function.remove("strict");
                    if let Some(parameters) = function.get_mut("parameters") {
                        sanitize_gemini_schema(parameters);
                    }
                }
                Some(function)
            })
            .collect::<Vec<_>>();
        if !declarations.is_empty() {
            output
                .as_object_mut()
                .expect("Gemini payload object")
                .insert(
                    "tools".into(),
                    json!([{"functionDeclarations":declarations}]),
                );
        }
    }
    apply_gemini_request_controls(&mut output, root, original_model);
    Ok(output)
}

fn sanitize_gemini_schema(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("additionalProperties");
            if let Some(default) = object.remove("default") {
                let description = object
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                object.insert(
                    "description".into(),
                    Value::String(format!("{description}\nDefault: {default}")),
                );
            }
            for value in object.values_mut() {
                sanitize_gemini_schema(value);
            }
        }
        Value::Array(items) => {
            for value in items {
                sanitize_gemini_schema(value);
            }
        }
        _ => {}
    }
}

fn gemini_parts(content: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"text":text})]);
    }
    let mut parts = Vec::new();
    for item in content.as_array().into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(
                json!({"text":item.get("text").cloned().unwrap_or(Value::String(String::new()))}),
            ),
            Some("image_url") => {
                if let Some(part) = item
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .and_then(data_url_part)
                {
                    parts.push(part);
                }
            }
            Some("input_audio") => {
                if let Some(part) = input_audio_part(item)? {
                    parts.push(part);
                }
            }
            _ => {}
        }
    }
    Ok(parts)
}

fn data_url_part(value: &str) -> Option<Value> {
    let data = value.strip_prefix("data:")?;
    let (metadata, body) = data.split_once(',')?;
    let mime = metadata
        .split(';')
        .next()
        .unwrap_or("application/octet-stream");
    let mut part = json!({"inlineData":{"mimeType":mime,"data":body}});
    if let Some(signature) = gemini_image_thought_signature(body) {
        part.as_object_mut()
            .expect("Gemini image part")
            .insert("thoughtSignature".into(), Value::String(signature));
    }
    Some(part)
}

fn input_audio_part(item: &Value) -> Result<Option<Value>, String> {
    let Some(input) = item.get("input_audio") else {
        return Ok(None);
    };
    let Some(data) = input.get("data").and_then(Value::as_str) else {
        return Err("input_audio.data must be a non-empty string".into());
    };
    let format = input
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wav")
        .to_ascii_lowercase();
    if matches!(
        Url::parse(data)
            .ok()
            .map(|url| url.scheme().to_owned())
            .as_deref(),
        Some("http" | "https" | "gs")
    ) {
        return Ok(Some(
            json!({"fileData":{"mimeType":audio_mime_type(&format),"fileUri":data}}),
        ));
    }
    let (mime, encoded) = if let Some(rest) = data.strip_prefix("data:") {
        let (metadata, encoded) = rest
            .split_once(',')
            .ok_or_else(|| "input_audio data URL is invalid".to_owned())?;
        (
            metadata.split(';').next().unwrap_or("audio/wav").to_owned(),
            encoded,
        )
    } else {
        (audio_mime_type(&format).to_owned(), data)
    };
    if encoded.is_empty() || encoded.len() > 8 * 1024 * 1024 {
        return Err("input_audio base64 exceeds the supported size limit".into());
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    BASE64
        .decode(padded.as_bytes())
        .map_err(|_| "input_audio data must be valid base64".to_owned())?;
    Ok(Some(json!({"inlineData":{"mimeType":mime,"data":encoded}})))
}

fn audio_mime_type(format: &str) -> &'static str {
    match format {
        "mp3" | "mpeg" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        "opus" => "audio/opus",
        "webm" => "audio/webm",
        _ => "audio/wav",
    }
}

fn apply_gemini_request_controls(output: &mut Value, root: &Map<String, Value>, model: &str) {
    let output = output.as_object_mut().expect("Gemini payload object");
    if let Some(choice) = root.get("tool_choice") {
        let mut config = Map::new();
        match choice.as_str() {
            Some("none") => {
                config.insert("mode".into(), Value::String("NONE".into()));
            }
            Some("required" | "any") => {
                config.insert("mode".into(), Value::String("ANY".into()));
            }
            Some("auto") => {
                config.insert("mode".into(), Value::String("AUTO".into()));
            }
            _ if choice.get("type").and_then(Value::as_str) == Some("function") => {
                config.insert("mode".into(), Value::String("ANY".into()));
                if let Some(name) = choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .and_then(Value::as_str)
                {
                    config.insert(
                        "allowedFunctionNames".into(),
                        Value::Array(vec![Value::String(name.to_owned())]),
                    );
                }
            }
            _ => {}
        }
        if !config.is_empty() {
            output.insert("toolConfig".into(), json!({"functionCallingConfig":config}));
        }
    }
    if let Some(tier) = root
        .get("service_tier")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
    {
        let tier = match tier.as_str() {
            "default" | "standard" => "STANDARD",
            "priority" => "PRIORITY",
            "flex" => "FLEX",
            value => value,
        };
        output.insert(
            "serviceTier".into(),
            Value::String(tier.to_ascii_uppercase()),
        );
    }
    let effort = root
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| {
            root.get("reasoning")
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str)
        })
        .map(|value| value.to_ascii_lowercase().replace('-', "_"));
    if let Some(effort) = effort {
        let generation = output
            .entry("generationConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("Gemini generation config");
        if model.to_ascii_lowercase().contains("gemini-3") {
            let level = match effort.as_str() {
                "minimal" | "low" => "low",
                "medium" => "medium",
                "high" | "extra_high" | "xhigh" => "high",
                _ => "minimal",
            };
            generation.insert("thinkingConfig".into(), json!({"thinkingLevel":level}));
        } else if model.to_ascii_lowercase().contains("gemini-2.5") {
            let maximum = if model.to_ascii_lowercase().contains("pro") {
                32768
            } else {
                24576
            };
            let budget = match effort.as_str() {
                "none" => 0,
                "minimal" | "low" => maximum / 4,
                "medium" => maximum / 2,
                "high" => maximum * 3 / 4,
                "extra_high" | "xhigh" => maximum,
                _ => 0,
            };
            generation.insert(
                "thinkingConfig".into(),
                json!({"includeThoughts":budget > 0,"thinkingBudget":budget}),
            );
        }
    }
    let wants_audio = root
        .get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case("audio"))
            })
        })
        || root.get("audio").is_some();
    if wants_audio {
        let voice = root
            .get("audio")
            .and_then(|value| value.get("voice"))
            .and_then(Value::as_str)
            .unwrap_or("Kore");
        let generation = output
            .entry("generationConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("Gemini generation config");
        generation.insert("responseModalities".into(), json!(["AUDIO"]));
        generation.insert(
            "speechConfig".into(),
            json!({"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}),
        );
    }
}

fn decode_gemini_thought_signature(call_id: &str) -> Option<String> {
    let encoded = call_id.strip_prefix("call_")?.split('.').next()?;
    if encoded.is_empty() || encoded.len() > 90_000 {
        return None;
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(padded.as_bytes()))
        .ok()?;
    (decoded.len() <= 64 * 1024)
        .then(|| String::from_utf8(decoded).ok())
        .flatten()
}

fn gemini_image_key(encoded: &str) -> Option<String> {
    if encoded.is_empty() || encoded.len() > 16 * 1024 * 1024 {
        return None;
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = BASE64.decode(padded.as_bytes()).ok()?;
    Some(format!("{:x}", Sha256::digest(decoded)))
}

fn gemini_image_thought_signature(encoded: &str) -> Option<String> {
    let key = gemini_image_key(encoded)?;
    let cache = GEMINI_THOUGHT_SIGNATURES.get_or_init(|| {
        Mutex::new(ThoughtSignatureCache {
            values: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        })
    });
    cache.lock().ok()?.values.get(&key).cloned()
}

pub(crate) fn cache_gemini_image_thought_signature(encoded: &str, signature: &str) {
    if signature.is_empty() || signature.len() > 64 * 1024 {
        return;
    }
    let Some(key) = gemini_image_key(encoded) else {
        return;
    };
    let cache = GEMINI_THOUGHT_SIGNATURES.get_or_init(|| {
        Mutex::new(ThoughtSignatureCache {
            values: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        })
    });
    let Ok(mut cache) = cache.lock() else {
        return;
    };
    if let Some(previous) = cache.values.remove(&key) {
        cache.bytes = cache.bytes.saturating_sub(previous.len());
        cache.order.retain(|item| item != &key);
    }
    cache.bytes = cache.bytes.saturating_add(signature.len());
    cache.order.push_back(key.clone());
    cache.values.insert(key, signature.to_owned());
    while cache.values.len() > 100 || cache.bytes > 4 * 1024 * 1024 {
        let Some(oldest) = cache.order.pop_front() else {
            break;
        };
        if let Some(value) = cache.values.remove(&oldest) {
            cache.bytes = cache.bytes.saturating_sub(value.len());
        }
    }
}

pub(crate) fn gemini_call_id(signature: Option<&str>, fallback_index: usize) -> String {
    signature
        .filter(|signature| !signature.is_empty() && signature.len() <= 64 * 1024)
        .map(|signature| {
            format!(
                "call_{}.{}",
                URL_SAFE_NO_PAD.encode(signature.as_bytes()),
                fallback_index
            )
        })
        .unwrap_or_else(|| format!("call_{fallback_index}"))
}

fn chat_to_claude(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    let mut system = Vec::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message.get("content").cloned().unwrap_or(Value::Null);
        if role == "system" {
            if let Some(text) = content_text(Some(&content)) {
                system.push(text);
            }
            continue;
        }
        if role == "tool" {
            messages.push(json!({
                "role":"user",
                "content":[{"type":"tool_result","tool_use_id":message.get("tool_call_id").cloned().unwrap_or(Value::Null),"content":content}],
            }));
            continue;
        }
        let mut blocks = claude_content(&content);
        for tool in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let arguments = tool
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|value| serde_json::from_str::<Value>(value).ok())
                .unwrap_or_else(|| json!({}));
            blocks.push(json!({
                "type":"tool_use",
                "id":tool.get("id").cloned().unwrap_or(Value::Null),
                "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "input":arguments,
            }));
        }
        messages.push(json!({
            "role": if role == "assistant" { "assistant" } else { "user" },
            "content": blocks,
        }));
    }
    let mut output = json!({
        "model": original_model,
        "messages": messages,
        "max_tokens": root.get("max_tokens").cloned().unwrap_or(json!(4096)),
        "stream": false,
    });
    if !system.is_empty() {
        output
            .as_object_mut()
            .expect("Claude payload object")
            .insert("system".into(), Value::String(system.join("\n")));
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        let tools = tools
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                Some(json!({
                    "name":function.get("name").cloned().unwrap_or(Value::Null),
                    "description":function.get("description").cloned().unwrap_or(Value::Null),
                    "input_schema":function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                }))
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            output
                .as_object_mut()
                .expect("Claude payload object")
                .insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(output) = output.as_object_mut() {
        for name in ["temperature", "top_p", "top_k", "stop_sequences"] {
            if let Some(value) = root.get(name) {
                output.insert(name.into(), value.clone());
            }
        }
        if let Some(choice) = root.get("tool_choice") {
            let choice = match choice.as_str() {
                Some("auto") => Some(json!({"type":"auto"})),
                Some("required" | "any") => Some(json!({"type":"any"})),
                Some("none") => None,
                _ if choice.get("type").and_then(Value::as_str) == Some("function") => choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .cloned()
                    .map(|name| json!({"type":"tool","name":name})),
                _ => None,
            };
            if let Some(choice) = choice {
                output.insert("tool_choice".into(), choice);
            } else if root.get("tool_choice").and_then(Value::as_str) == Some("none") {
                output.remove("tools");
            }
        }
        let explicit_thinking = root.get("thinking").filter(|value| value.is_object());
        let effort = root
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .or_else(|| {
                root.get("reasoning")
                    .and_then(|value| value.get("effort"))
                    .and_then(Value::as_str)
            });
        if let Some(thinking) = explicit_thinking {
            output.insert("thinking".into(), thinking.clone());
            output.insert("temperature".into(), json!(1));
            output.remove("top_p");
            output.remove("top_k");
            if let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_i64) {
                let max_tokens = output
                    .get("max_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(4096)
                    .max(budget + 1024);
                output.insert("max_tokens".into(), json!(max_tokens));
            }
        } else if let Some(effort) = effort {
            let budget = match effort.to_ascii_lowercase().as_str() {
                "minimal" | "low" => 1024,
                "medium" => 4096,
                "high" => 8192,
                "extra_high" | "xhigh" => 16384,
                _ => 0,
            };
            if budget > 0 {
                output.insert(
                    "thinking".into(),
                    json!({"type":"enabled","budget_tokens":budget}),
                );
                output.insert("temperature".into(), json!(1));
                output.remove("top_p");
                output.remove("top_k");
                let max_tokens = output
                    .get("max_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(4096)
                    .max(budget + 1024);
                output.insert("max_tokens".into(), json!(max_tokens));
            }
        }
        if let Some(tier) = root.get("service_tier").and_then(Value::as_str) {
            let tier = match tier.to_ascii_lowercase().as_str() {
                "flex" | "standard" | "standard_only" => "standard_only",
                _ => "auto",
            };
            output.insert("service_tier".into(), Value::String(tier.into()));
        }
    }
    Ok(output)
}

fn chat_to_cohere(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = match message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
        {
            "assistant" => "CHATBOT",
            "system" => "SYSTEM",
            _ => "USER",
        };
        if let Some(text) = content_text(message.get("content")) {
            messages.push(json!({"role":role,"message":text}));
        }
    }
    let last = messages
        .pop()
        .ok_or_else(|| "Cohere chat request requires at least one message".to_owned())?;
    let mut output = json!({
        "model":original_model,
        "message":last.get("message").cloned().unwrap_or(Value::String(String::new())),
    });
    if !messages.is_empty() {
        output
            .as_object_mut()
            .expect("Cohere payload object")
            .insert("chat_history".into(), Value::Array(messages));
    }
    Ok(output)
}

fn chat_to_doubao_translation(
    input: &Value,
    original_model: &str,
    request_model: &str,
    provider: &Provider,
) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let user_text = root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .filter_map(Value::as_object)
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .map(extract_translation_text)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "No user message".to_owned())?;
    let translation_overrides = provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get(request_model))
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get("translation_options"))
        .and_then(Value::as_object);
    let mut options = Map::from_iter([("target_language".into(), Value::String("zh".into()))]);
    if let Some(overrides) = translation_overrides {
        for key in ["source_language", "target_language"] {
            if let Some(value) = overrides
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                options.insert(key.into(), Value::String(value.to_owned()));
            }
        }
    }
    let mut output = json!({
        "model":original_model,
        "input":[{
            "role":"user",
            "content":[{
                "type":"input_text",
                "text":user_text,
                "translation_options":options,
            }],
        }],
    });
    if root.get("stream").and_then(Value::as_bool) == Some(true) {
        output["stream"] = Value::Bool(true);
    }
    Ok(output)
}

fn extract_translation_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(extract_translation_text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(item) => ["text", "content", "input"]
            .into_iter()
            .find_map(|key| item.get(key).map(extract_translation_text))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn openai_tts_to_minimax(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "text-to-speech request body must be an object".to_owned())?;
    let text = root
        .get("input")
        .cloned()
        .ok_or_else(|| "text-to-speech input is required".to_owned())?;
    let voice = root
        .get("voice")
        .cloned()
        .ok_or_else(|| "text-to-speech voice is required".to_owned())?;
    let mut output = Map::from_iter([
        ("model".into(), Value::String(original_model.to_owned())),
        ("text".into(), text),
        ("voice_setting".into(), json!({"voice_id":voice})),
    ]);
    for key in ["response_format", "speed", "stream"] {
        if let Some(value) = root.get(key) {
            output.insert(key.into(), value.clone());
        }
    }
    Ok(Value::Object(output))
}

fn normalize_jina_embedding(input: &mut Value, original_model: &str) -> Result<(), String> {
    let root = input
        .as_object_mut()
        .ok_or_else(|| "embedding request body must be an object".to_owned())?;
    root.insert("model".into(), Value::String(original_model.to_owned()));
    if let Some(format) = root.remove("encoding_format") {
        root.insert("embedding_type".into(), format);
    }
    Ok(())
}

fn chat_to_cloudflare(input: &Value) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let prompt = root
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| content_text(message.get("content")))
        .ok_or_else(|| "Cloudflare chat request requires a text message".to_owned())?;
    let mut output = json!({"prompt":prompt});
    if let Some(root) = output.as_object_mut() {
        for name in ["temperature", "top_p", "max_tokens", "seed"] {
            if let Some(value) = input.get(name) {
                root.insert(name.into(), value.clone());
            }
        }
    }
    Ok(output)
}

fn claude_content(content: &Value) -> Vec<Value> {
    if let Some(text) = content.as_str() {
        return vec![json!({"type":"text","text":text})];
    }
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| match item.get("type").and_then(Value::as_str) {
            Some("text") => Some(json!({"type":"text","text":item.get("text").cloned().unwrap_or(Value::String(String::new()))})),
            Some("image_url") => item
                .pointer("/image_url/url")
                .and_then(Value::as_str)
                .and_then(|value| {
                    let data = value.strip_prefix("data:")?;
                    let (metadata, body) = data.split_once(',')?;
                    Some(json!({
                        "type":"image",
                        "source":{"type":"base64","media_type":metadata.split(';').next().unwrap_or("image/png"),"data":body},
                    }))
                }),
            _ => None,
        })
        .collect()
}

fn responses_to_chat(value: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for item in value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for content in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(delta) = content
                        .get("text")
                        .or_else(|| content.get("output_text"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(delta);
                    }
                }
            }
            Some("function_call") => tool_calls.push(json!({
                "id":item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Value::Null),
                "type":"function",
                "function":{"name":item.get("name").cloned().unwrap_or(Value::Null),"arguments":item.get("arguments").cloned().unwrap_or(Value::String("{}".into()))},
            })),
            _ => {}
        }
    }
    let has_tool_calls = !tool_calls.is_empty();
    let content = if text.is_empty() && has_tool_calls {
        Value::Null
    } else {
        Value::String(text)
    };
    let mut message = json!({"role":"assistant","content":content});
    if has_tool_calls {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("tool_calls".into(), Value::Array(tool_calls));
    }
    json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}", unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":if has_tool_calls { "tool_calls" } else { "stop" }}],
        "usage":provider_stream::responses_usage_to_chat(value.get("usage")),
    })
}

fn gemini_to_chat(value: &Value, model: &str) -> Value {
    let candidate = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|items| items.first());
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for part in candidate
        .and_then(|candidate| candidate.pointer("/content/parts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(data), Some(signature)) = (
            part.pointer("/inlineData/data")
                .or_else(|| part.pointer("/inline_data/data"))
                .and_then(Value::as_str),
            part.get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(Value::as_str),
        ) {
            cache_gemini_image_thought_signature(data, signature);
        }
        if let Some(value) = part.get("text").and_then(Value::as_str) {
            if part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                reasoning.push_str(value);
            } else {
                text.push_str(value);
            }
        }
        if let Some(call) = part.get("functionCall") {
            let call_id = gemini_call_id(
                part.get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(Value::as_str),
                tool_calls.len() + 1,
            );
            tool_calls.push(json!({
                "id":call_id,
                "type":"function",
                "function":{
                    "name":call.get("name").cloned().unwrap_or(Value::Null),
                    "arguments":serde_json::to_string(call.get("args").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into()),
                }
            }));
        }
    }
    let usage = value
        .get("usageMetadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut message = json!({"role":"assistant","content":text});
    if !reasoning.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("tool_calls".into(), Value::Array(tool_calls));
    }
    json!({
        "id":format!("chatcmpl-{}",unix_seconds()),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":"stop"}],
        "usage":{
            "prompt_tokens":usage.get("promptTokenCount").cloned().unwrap_or(json!(0)),
            "completion_tokens":usage.get("candidatesTokenCount").cloned().unwrap_or(json!(0)),
            "total_tokens":usage.get("totalTokenCount").cloned().unwrap_or(json!(0)),
        },
    })
}

fn claude_to_chat(value: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for item in value
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(item.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => reasoning.push_str(
                item.get("thinking")
                    .or_else(|| item.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ),
            Some("tool_use") => tool_calls.push(json!({
                "id":item.get("id").cloned().unwrap_or(Value::Null),
                "type":"function",
                "function":{
                    "name":item.get("name").cloned().unwrap_or(Value::Null),
                    "arguments":serde_json::to_string(item.get("input").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into()),
                }
            })),
            _ => {}
        }
    }
    let mut message = json!({"role":"assistant","content":text});
    if !reasoning.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("tool_calls".into(), Value::Array(tool_calls));
    }
    let usage = value.get("usage").cloned().unwrap_or_else(|| json!({}));
    json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":"stop"}],
        "usage":{
            "prompt_tokens":usage.get("input_tokens").cloned().unwrap_or(json!(0)),
            "completion_tokens":usage.get("output_tokens").cloned().unwrap_or(json!(0)),
            "total_tokens":usage.get("input_tokens").and_then(Value::as_i64).unwrap_or(0) + usage.get("output_tokens").and_then(Value::as_i64).unwrap_or(0),
        },
    })
}

fn cohere_to_chat(value: &Value, model: &str) -> Value {
    let prompt_tokens = value
        .pointer("/meta/billed_units/input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion_tokens = value
        .pointer("/meta/billed_units/output_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    json!({
        "id":value.get("generation_id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":value.get("text").cloned().unwrap_or(Value::String(String::new()))},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"total_tokens":prompt_tokens + completion_tokens},
    })
}

fn cloudflare_to_chat(value: &Value, model: &str) -> Value {
    let text = value
        .pointer("/result/response")
        .or_else(|| value.get("response"))
        .cloned()
        .unwrap_or(Value::String(String::new()));
    json!({
        "id":format!("chatcmpl-{}",unix_seconds()),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
    })
}

fn chat_to_responses_response(value: &Value, model: &str) -> Value {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({"role":"assistant","content":""}));
    let mut output = Vec::new();
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        output.push(json!({
            "id":format!("rs_{}", unix_seconds()),
            "type":"reasoning",
            "summary":[{"type":"summary_text","text":reasoning}],
        }));
    }
    if !text.is_empty() || message.get("tool_calls").is_none() {
        output.push(json!({
            "id":format!("msg_{}", unix_seconds()),
            "type":"message",
            "status":"completed",
            "role":"assistant",
            "content":[{"type":"output_text","text":text,"annotations":[]}],
        }));
    }
    for tool in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        output.push(json!({
            "id":tool.get("id").cloned().unwrap_or_else(|| Value::String(format!("fc_{}", unix_seconds()))),
            "type":"function_call",
            "status":"completed",
            "call_id":tool.get("id").cloned().unwrap_or_else(|| Value::String(format!("call_{}", unix_seconds()))),
            "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
            "arguments":tool.pointer("/function/arguments").cloned().unwrap_or_else(|| Value::String("{}".into())),
        }));
    }
    let chat_usage = value.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = chat_usage
        .get("prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output_tokens = chat_usage
        .get("completion_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    json!({
        "id":format!("resp_{}", unix_seconds()),
        "object":"response",
        "created_at":unix_seconds(),
        "status":"completed",
        "model":model,
        "output":output,
        "output_text":text,
        "usage":{
            "input_tokens":input_tokens,
            "output_tokens":output_tokens,
            "total_tokens":chat_usage.get("total_tokens").cloned().unwrap_or(json!(input_tokens + output_tokens)),
            "input_tokens_details":chat_usage.get("prompt_tokens_details").cloned().unwrap_or_else(|| json!({})),
            "output_tokens_details":chat_usage.get("completion_tokens_details").cloned().unwrap_or_else(|| json!({})),
        },
        "error":Value::Null,
        "incomplete_details":Value::Null,
    })
}

fn normalize_search_response(url: &str, value: &Value) -> Value {
    let host = Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .unwrap_or_default();
    if value.is_object() && (host.ends_with("tavily.com") || value.get("results").is_some()) {
        let data = value
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let item = item.as_object()?;
                let mut normalized = item.clone();
                let content = item
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let description = if content.chars().count() > 240 {
                    format!("{}...", content.chars().take(237).collect::<String>())
                } else {
                    content.to_owned()
                };
                normalized.insert(
                    "title".into(),
                    item.get("title")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                );
                normalized.insert(
                    "url".into(),
                    item.get("url")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                );
                normalized.insert("description".into(), Value::String(description));
                normalized.insert("content".into(), Value::String(content.to_owned()));
                for name in ["usage", "score", "raw_content"] {
                    normalized.entry(name).or_insert(Value::Null);
                }
                Some(Value::Object(normalized))
            })
            .collect::<Vec<_>>();
        let mut meta = Map::new();
        meta.insert("provider".into(), Value::String("tavily".into()));
        if let Some(root) = value.as_object() {
            for (name, value) in root {
                if name != "results" {
                    meta.insert(name.clone(), value.clone());
                }
            }
        }
        return json!({"code":200,"status":20000,"data":data,"meta":meta});
    }
    if let Some(root) = value.as_object().filter(|root| root.contains_key("data")) {
        let mut output = root.clone();
        output.entry("code").or_insert(json!(200));
        output.entry("status").or_insert(json!(20000));
        let mut meta = output
            .remove("meta")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        meta.entry("provider")
            .or_insert_with(|| Value::String("jina".into()));
        output.insert("meta".into(), Value::Object(meta));
        let data = output
            .remove("data")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|item| {
                let mut item = item.as_object()?.clone();
                for name in ["title", "url", "description", "content"] {
                    item.entry(name)
                        .or_insert_with(|| Value::String(String::new()));
                }
                for name in ["usage", "score", "raw_content"] {
                    item.entry(name).or_insert(Value::Null);
                }
                Some(Value::Object(item))
            })
            .collect();
        output.insert("data".into(), Value::Array(data));
        return Value::Object(output);
    }
    json!({
        "code":200,
        "status":20000,
        "data":[],
        "meta":{"provider":"unknown","raw":value},
    })
}

fn synthetic_chat_stream(value: Value) -> Response<Body> {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({"role":"assistant","content":""}));
    let chunk = json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion.chunk",
        "created":value.get("created").cloned().unwrap_or_else(|| json!(unix_seconds())),
        "model":value.get("model").cloned().unwrap_or(Value::Null),
        "choices":[{"index":0,"delta":message,"finish_reason":"stop"}],
        "usage":value.get("usage").cloned().unwrap_or(Value::Null),
    });
    let wire = format!("data: {}\n\ndata: [DONE]\n\n", chunk);
    let mut response = Response::new(Body::from(wire));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

fn synthetic_responses_stream(value: Value) -> Response<Body> {
    let mut created_response = value.clone();
    if let Some(root) = created_response.as_object_mut() {
        root.insert("status".into(), Value::String("in_progress".into()));
        root.insert("output".into(), Value::Array(Vec::new()));
        root.insert("output_text".into(), Value::String(String::new()));
    }
    let created = json!({"type":"response.created","response":created_response});
    let completed = json!({"type":"response.completed","response":value});
    let wire = format!(
        "event: response.created\ndata: {created}\n\nevent: response.completed\ndata: {completed}\n\n"
    );
    let mut response = Response::new(Body::from(wire));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

fn azure_chat_url(base: &str, deployment: &str) -> Result<String, String> {
    let mut url = Url::parse(base).map_err(|error| format!("invalid Azure base URL: {error}"))?;
    let path = url.path().trim_end_matches('/');
    if !path.contains("/models/chat/completions")
        && !path.contains("/openai/deployments/")
        && !path.ends_with("/chat/completions")
    {
        url.set_path(&format!(
            "/openai/deployments/{deployment}/chat/completions"
        ));
    }
    let has_version = url.query_pairs().any(|(name, _)| name == "api-version");
    if !has_version {
        url.query_pairs_mut()
            .append_pair("api-version", "2025-01-01-preview");
    }
    Ok(url.to_string())
}

fn databricks_chat_url(base: &str, deployment: &str) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid Databricks base URL: {error}"))?;
    url.set_path(&format!("/serving-endpoints/{deployment}/invocations"));
    url.set_query(None);
    Ok(url.to_string())
}

fn cloudflare_url(provider: &Provider, model: &str) -> Result<String, String> {
    let account = provider.cf_account_id.as_deref().ok_or_else(|| {
        format!(
            "Cloudflare provider {} is missing cf_account_id",
            provider.name
        )
    })?;
    let mut url = Url::parse(provider.base_url.as_ref())
        .map_err(|error| format!("invalid Cloudflare base URL: {error}"))?;
    url.set_path(&format!("/client/v4/accounts/{account}/ai/run/{model}"));
    url.set_query(None);
    Ok(url.to_string())
}

fn vertex_claude_url(provider: &Provider, model: &str) -> Result<String, String> {
    let project = provider
        .project_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("Vertex provider {} is missing project_id", provider.name))?;
    let region = provider.region.trim();
    let origin = if provider.base_url.contains("google-vertex-ai") {
        provider.base_url.trim_end_matches('/').to_owned()
    } else if region == "global" {
        "https://aiplatform.googleapis.com".into()
    } else {
        format!("https://{region}-aiplatform.googleapis.com")
    };
    Ok(format!(
        "{origin}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict"
    ))
}

fn aws_bedrock_url(provider: &Provider, model: &str, stream: bool) -> Result<String, String> {
    let mut url = Url::parse(provider.base_url.as_ref())
        .map_err(|error| format!("invalid AWS Bedrock base URL: {error}"))?;
    url.set_path(&format!(
        "/model/{model}/{}",
        if stream {
            "invoke-with-response-stream"
        } else {
            "invoke"
        }
    ));
    url.set_query(None);
    Ok(url.to_string())
}

fn normalize_azure_token_limit(payload: &mut Value, model: &str) {
    if !model.to_ascii_lowercase().contains("gpt-5") {
        return;
    }
    let Some(root) = payload.as_object_mut() else {
        return;
    };
    if let Some(value) = root.remove("max_tokens") {
        root.insert("max_completion_tokens".into(), value);
    }
}

fn sign_aws_request(
    provider: &Provider,
    url: &str,
    body: &[u8],
    headers: &mut HeaderMap,
) -> Result<(), String> {
    sign_aws_request_at(provider, url, body, headers, SystemTime::now())
}

fn sign_aws_request_at(
    provider: &Provider,
    url: &str,
    body: &[u8],
    headers: &mut HeaderMap,
    now: SystemTime,
) -> Result<(), String> {
    type HmacSha256 = Hmac<Sha256>;

    let access_key = provider
        .aws_access_key
        .as_deref()
        .ok_or_else(|| format!("AWS provider {} is missing aws_access_key", provider.name))?;
    let secret_key = provider
        .aws_secret_key
        .as_deref()
        .ok_or_else(|| format!("AWS provider {} is missing aws_secret_key", provider.name))?;
    let parsed = Url::parse(url).map_err(|error| format!("invalid AWS URL: {error}"))?;
    let host = match parsed.port() {
        Some(port) => format!("{}:{port}", parsed.host_str().unwrap_or_default()),
        None => parsed.host_str().unwrap_or_default().to_owned(),
    };
    let region = aws_region(provider, &host)?;
    let (amz_date, date_stamp) = aws_timestamp(now);
    let payload_hash = format!("{:x}", Sha256::digest(body));
    let accept = if parsed.path().ends_with("invoke-with-response-stream") {
        "application/vnd.amazon.bedrock.payload+json"
    } else {
        "application/json"
    };
    let mut canonical_headers = format!(
        "accept:{accept}\ncontent-type:application/json\nhost:{host}\nx-amz-bedrock-accept:{accept}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    );
    let mut signed_headers =
        "accept;content-type;host;x-amz-bedrock-accept;x-amz-content-sha256;x-amz-date".to_owned();
    if let Some(token) = provider.aws_session_token.as_deref() {
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }
    let canonical_query = parsed.query().unwrap_or_default();
    let canonical_request = format!(
        "POST\n{}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        aws_uri_encode(parsed.path())
    );
    let scope = format!("{date_stamp}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{:x}",
        Sha256::digest(canonical_request.as_bytes())
    );
    let k_date = hmac_bytes::<HmacSha256>(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    )?;
    let k_region = hmac_bytes::<HmacSha256>(&k_date, region.as_bytes())?;
    let k_service = hmac_bytes::<HmacSha256>(&k_region, b"bedrock")?;
    let k_signing = hmac_bytes::<HmacSha256>(&k_service, b"aws4_request")?;
    let signature = hex_bytes(&hmac_bytes::<HmacSha256>(
        &k_signing,
        string_to_sign.as_bytes(),
    )?);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    for (name, value) in [
        ("accept", accept),
        ("content-type", "application/json"),
        ("host", host.as_str()),
        ("x-amz-bedrock-accept", accept),
        ("x-amz-content-sha256", payload_hash.as_str()),
        ("x-amz-date", amz_date.as_str()),
        ("authorization", authorization.as_str()),
    ] {
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).expect("static AWS header name"),
            HeaderValue::from_str(value)
                .map_err(|_| format!("AWS provider {} has an invalid header", provider.name))?,
        );
    }
    if let Some(token) = provider.aws_session_token.as_deref() {
        headers.insert(
            "x-amz-security-token",
            HeaderValue::from_str(token)
                .map_err(|_| "AWS session token is not a valid header value".to_owned())?,
        );
    }
    Ok(())
}

fn hmac_bytes<M>(key: &[u8], message: &[u8]) -> Result<Vec<u8>, String>
where
    M: Mac + hmac::digest::KeyInit,
{
    let mut mac = <M as Mac>::new_from_slice(key).map_err(|_| "invalid HMAC key".to_owned())?;
    mac.update(message);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn aws_region(provider: &Provider, host: &str) -> Result<String, String> {
    if provider.region.as_ref() != "global" && !provider.region.trim().is_empty() {
        return Ok(provider.region.to_string());
    }
    let parts = host.split('.').collect::<Vec<_>>();
    if parts.len() >= 3 && parts[0].starts_with("bedrock-runtime") {
        return Ok(parts[1].to_owned());
    }
    Err(format!(
        "AWS provider {} has no usable Bedrock region",
        provider.name
    ))
}

fn aws_timestamp(now: SystemTime) -> (String, String) {
    let total = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let days = total.div_euclid(86_400);
    let seconds = total.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds / 3600;
    let minute = seconds % 3600 / 60;
    let second = seconds % 60;
    (
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"),
        format!("{year:04}{month:02}{day:02}"),
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn aws_uri_encode(path: &str) -> String {
    let mut output = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(char::from(b"0123456789ABCDEF"[(byte >> 4) as usize]));
            output.push(char::from(b"0123456789ABCDEF"[(byte & 0x0f) as usize]));
        }
    }
    output
}

fn hex_bytes(value: &[u8]) -> String {
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        output.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    output
}

fn endpoint_url(base: &str, endpoint: &str, method: &Method, uri: &Uri) -> Result<String, String> {
    if endpoint.starts_with("/v1/video/tasks/") {
        let task = endpoint.trim_start_matches("/v1/video/tasks/");
        return Ok(format!(
            "{}/{}",
            video_tasks_url(base),
            url::form_urlencoded::byte_serialize(task.as_bytes()).collect::<String>()
        ));
    }
    if endpoint == "/v1/video/tasks" {
        return Ok(video_tasks_url(base));
    }
    if endpoint.starts_with("/v1/asset-groups/") {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(
            base,
            &format!(
                "/material/asset-groups/{}",
                endpoint.trim_start_matches("/v1/asset-groups/")
            ),
            query.as_deref(),
        );
    }
    if endpoint == "/v1/asset-groups" {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(base, "/material/asset-groups", query.as_deref());
    }
    if endpoint.starts_with("/v1/assets/") {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(
            base,
            &format!(
                "/material/assets/{}",
                endpoint.trim_start_matches("/v1/assets/")
            ),
            query.as_deref(),
        );
    }
    if endpoint == "/v1/assets" {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(base, "/material/assets/create", query.as_deref());
    }
    if matches!(endpoint, "/search" | "/v1/search") && *method == Method::GET {
        return Ok(base.to_owned());
    }
    replace_known_endpoint(base, endpoint)
}

fn filtered_lingjing_query(query: Option<&str>) -> Option<String> {
    let mut output = url::form_urlencoded::Serializer::new(String::new());
    let mut retained = false;
    for (key, value) in url::form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        if matches!(key.as_ref(), "model" | "request_model") {
            continue;
        }
        retained = true;
        output.append_pair(&key, &value);
    }
    retained.then(|| output.finish())
}

fn replace_known_endpoint(base: &str, endpoint: &str) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid provider base URL: {error}"))?;
    let known = [
        "/chat/completions",
        "/images/generations",
        "/images/edits",
        "/audio/transcriptions",
        "/audio/speech",
        "/moderations",
        "/embeddings",
        "/responses/compact",
        "/responses",
        "/messages",
    ];
    let mut path = url.path().trim_end_matches('/').to_owned();
    for suffix in known {
        if path.ends_with(suffix) {
            path.truncate(path.len() - suffix.len());
            break;
        }
    }
    let endpoint = endpoint.trim_start_matches("/v1/");
    url.set_path(&format!("{}/{}", path.trim_end_matches('/'), endpoint));
    Ok(url.to_string())
}

fn responses_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/responses") {
        base.to_owned()
    } else {
        format!("{base}/responses")
    }
}

fn messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/messages") {
        base.to_owned()
    } else {
        format!("{base}/messages")
    }
}

fn gemini_url(base: &str, model: &str, key: &str, stream: bool) -> Result<String, String> {
    let mut url = Url::parse(base).map_err(|error| format!("invalid Gemini base URL: {error}"))?;
    let path = url
        .path()
        .split("/models/")
        .next()
        .unwrap_or(url.path())
        .trim_end_matches('/');
    url.set_path(&format!(
        "{path}/models/{model}:{}",
        if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        }
    ));
    url.query_pairs_mut().clear().append_pair("key", key);
    Ok(url.to_string())
}

fn vertex_gemini_url(
    provider: &Provider,
    model: &str,
    key: &str,
    stream: bool,
) -> Result<String, String> {
    let operation = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };
    if key.as_bytes().get(2) == Some(&b'.') {
        let mut url = Url::parse(provider.base_url.trim_end_matches('/'))
            .map_err(|error| format!("invalid Vertex base URL: {error}"))?;
        let base_path = url.path().trim_end_matches('/');
        url.set_path(&format!(
            "{base_path}/v1/publishers/google/models/{model}:{operation}"
        ));
        url.query_pairs_mut().clear().append_pair("key", key);
        return Ok(url.to_string());
    }
    let project = provider
        .project_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("Vertex provider {} is missing project_id", provider.name))?;
    let region = provider.region.trim();
    let origin = if provider.base_url.contains("google-vertex-ai") {
        provider.base_url.trim_end_matches('/').to_owned()
    } else if region == "global" {
        "https://aiplatform.googleapis.com".into()
    } else {
        format!("https://{region}-aiplatform.googleapis.com")
    };
    Ok(format!(
        "{origin}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{operation}"
    ))
}

async fn vertex_access_token(state: &AppState, provider: &Provider) -> Result<String, String> {
    let email = provider
        .client_email
        .as_deref()
        .ok_or_else(|| format!("Vertex provider {} is missing client_email", provider.name))?;
    let private_key = provider
        .private_key
        .as_deref()
        .ok_or_else(|| format!("Vertex provider {} is missing private_key", provider.name))?;
    let cache = VERTEX_TOKEN_CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    if let Some(token) = cache
        .lock()
        .await
        .get(email)
        .filter(|token| token.expires_at > Instant::now() + Duration::from_secs(60))
        .cloned()
    {
        return Ok(token.value);
    }
    let assertion = service_account_jwt(email, private_key)?;
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer")
        .append_pair("assertion", &assertion)
        .finish();
    let response = state
        .backend_client
        .post("https://oauth2.googleapis.com/token")
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept-encoding", "identity")
        .body(body)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("Vertex OAuth token request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Vertex OAuth token endpoint returned HTTP {}",
            response.status().as_u16()
        ));
    }
    let payload = response
        .json::<Value>()
        .await
        .map_err(|error| format!("decode Vertex OAuth token response: {error}"))?;
    let value = payload
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 16 * 1024)
        .ok_or_else(|| "Vertex OAuth token response is invalid".to_owned())?
        .to_owned();
    let expires_in = payload
        .get("expires_in")
        .and_then(Value::as_u64)
        .unwrap_or(3600)
        .clamp(120, 86_400);
    cache.lock().await.insert(
        email.to_owned(),
        CachedVertexToken {
            value: value.clone(),
            expires_at: Instant::now() + Duration::from_secs(expires_in),
        },
    );
    Ok(value)
}

fn service_account_jwt(email: &str, private_key: &str) -> Result<String, String> {
    let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
    let now = unix_seconds();
    let claims = serde_json::to_vec(&json!({
        "iss": email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now.saturating_add(3600),
        "iat": now,
    }))
    .map_err(|error| format!("encode Vertex OAuth claims: {error}"))?;
    let claims = URL_SAFE_NO_PAD.encode(claims);
    let signing_input = format!("{header}.{claims}");
    let encoded_key = private_key
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect::<String>();
    let der = BASE64
        .decode(encoded_key)
        .map_err(|error| format!("decode Vertex private key: {error}"))?;
    let key = RsaKeyPair::from_pkcs8(&der)
        .or_else(|_| RsaKeyPair::from_der(&der))
        .map_err(|_| "parse Vertex RSA private key".to_owned())?;
    let mut signature = vec![0; key.public().modulus_len()];
    key.sign(
        &RSA_PKCS1_SHA256,
        &SystemRandom::new(),
        signing_input.as_bytes(),
        &mut signature,
    )
    .map_err(|_| "sign Vertex OAuth assertion".to_owned())?;
    Ok(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature)
    ))
}

fn video_tasks_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/contents/generations/tasks") {
        base.to_owned()
    } else if Url::parse(base)
        .ok()
        .is_some_and(|url| url.path().is_empty() || url.path() == "/")
    {
        format!("{base}/api/v3/contents/generations/tasks")
    } else {
        format!("{base}/contents/generations/tasks")
    }
}

fn content_generation_to_lingjing(input: &Value, model_code: &str) -> Result<Value, String> {
    let request = input
        .as_object()
        .ok_or_else(|| "video task request body must be an object".to_owned())?;
    if request.contains_key("taskParams") || request.contains_key("modelCode") {
        let mut payload = request.clone();
        payload.insert("modelCode".into(), Value::String(model_code.to_owned()));
        for key in [
            "model",
            "request_model",
            "provider",
            "provider_options",
            "route",
        ] {
            payload.remove(key);
        }
        return Ok(Value::Object(payload));
    }

    let mut prompt_parts = Vec::new();
    let mut content_resources = Vec::new();
    for part in request
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        match part.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => {
                if let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                {
                    prompt_parts.push(text.to_owned());
                }
            }
            kind @ ("image_url" | "video_url" | "audio_url") => {
                let resource_type = kind.trim_end_matches("_url");
                if let Some(url) = content_part_url(part, kind) {
                    let mut resource = Map::new();
                    resource.insert("type".into(), Value::String(resource_type.to_owned()));
                    resource.insert(
                        "usage".into(),
                        Value::String(lingjing_resource_usage(
                            part.get("role"),
                            resource_type,
                            content_resources.len(),
                        )),
                    );
                    resource.insert("source".into(), lingjing_source(&url));
                    if let Some(reference_key) = part.get("reference_key") {
                        resource.insert("reference_key".into(), reference_key.clone());
                    }
                    content_resources.push(Value::Object(resource));
                }
            }
            _ => {}
        }
    }

    let prompt = request
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| prompt_parts.join("\n"));
    let mut task_input = Map::from_iter([("prompt".into(), Value::String(prompt))]);
    let quality = request.get("quality").cloned().or_else(|| {
        request.get("resolution").and_then(|value| {
            let raw = value.as_str()?.trim().to_ascii_lowercase();
            Some(Value::String(
                raw.strip_suffix('p').unwrap_or(raw.as_str()).to_owned(),
            ))
        })
    });
    if let Some(quality) = quality.filter(|value| !value.is_null()) {
        task_input.insert(
            "quality".into(),
            Value::String(value_text(&quality).unwrap_or_default()),
        );
    }
    for key in ["duration", "ratio", "generate_num", "prompt_optimizer"] {
        if let Some(value) = request.get(key).filter(|value| !value.is_null()) {
            task_input.insert(key.into(), value.clone());
        }
    }
    let unified_resources = request
        .get("resources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, resource)| normalize_lingjing_resource(resource, index))
        .collect::<Vec<_>>();
    if !unified_resources.is_empty() {
        task_input.insert("resources".into(), Value::Array(unified_resources));
    } else if !content_resources.is_empty() {
        task_input.insert("resources".into(), Value::Array(content_resources));
    }
    if let Some(options) = request
        .get("provider_options")
        .and_then(Value::as_object)
        .and_then(|options| {
            options
                .get("lingjing")
                .and_then(Value::as_object)
                .or(Some(options))
        })
    {
        for (key, value) in options {
            if !value.is_object() && !value.is_null() {
                task_input.insert(key.clone(), value.clone());
            }
        }
    }
    for key in ["generate_audio", "need_audio", "audio"] {
        if let Some(value) = request.get(key) {
            task_input.insert(
                "need_audio".into(),
                Value::Bool(value.as_bool().unwrap_or(false)),
            );
        }
    }
    Ok(json!({"modelCode":model_code,"taskParams":{"input":task_input}}))
}

fn content_part_url(part: &Map<String, Value>, key: &str) -> Option<String> {
    let value = part.get(key)?;
    let raw = value
        .as_str()
        .or_else(|| value.get("url").and_then(Value::as_str))?
        .trim();
    (!raw.is_empty()).then(|| raw.to_owned())
}

fn lingjing_source(value: &str) -> Value {
    if let Some(asset_id) = value.strip_prefix("asset://") {
        return json!({"kind":"asset_id","value":asset_id});
    }
    if value.starts_with("Asset-") {
        return json!({"kind":"asset_id","value":value});
    }
    json!({"kind":"url","value":value})
}

fn lingjing_resource_usage(role: Option<&Value>, resource_type: &str, index: usize) -> String {
    let role = role
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if matches!(
        role.as_str(),
        "first_frame" | "last_frame" | "reference" | "keyframe" | "source"
    ) {
        return role;
    }
    if matches!(
        role.as_str(),
        "reference_image" | "reference_video" | "reference_audio"
    ) {
        return "reference".into();
    }
    if resource_type == "image" && index == 0 {
        "first_frame".into()
    } else {
        "reference".into()
    }
}

fn normalize_lingjing_resource(value: &Value, index: usize) -> Option<Value> {
    let resource = value.as_object()?;
    let resource_type = resource
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("image")
        .trim()
        .to_ascii_lowercase();
    if !matches!(resource_type.as_str(), "image" | "video" | "audio") {
        return None;
    }
    let source = resource
        .get("source")
        .filter(|value| value.is_object())
        .cloned()
        .or_else(|| {
            ["url", "asset_id", "assetId", "value"]
                .into_iter()
                .find_map(|key| resource.get(key).and_then(value_text))
                .map(|value| lingjing_source(&value))
        })?;
    let mut normalized = Map::from_iter([
        ("type".into(), Value::String(resource_type.clone())),
        (
            "usage".into(),
            Value::String(lingjing_resource_usage(
                resource.get("usage").or_else(|| resource.get("role")),
                &resource_type,
                index,
            )),
        ),
        ("source".into(), source),
    ]);
    if let Some(reference_key) = resource
        .get("reference_key")
        .or_else(|| resource.get("referenceKey"))
    {
        normalized.insert("reference_key".into(), reference_key.clone());
    }
    Some(Value::Object(normalized))
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn normalize_lingjing_video_response(
    method: &Method,
    request_model: &str,
    url: &str,
    upstream: &Value,
) -> Value {
    let Some(root) = upstream.as_object() else {
        return upstream.clone();
    };
    let data = root.get("data").and_then(Value::as_object);
    if *method == Method::POST {
        let task_id = data.and_then(|data| {
            data.get("taskId")
                .or_else(|| data.get("task_id"))
                .and_then(value_text)
        });
        return task_id.map_or_else(
            || upstream.clone(),
            |task_id| {
                json!({
                    "id":task_id,
                    "model":request_model,
                    "provider":"lingjing",
                    "status":"queued",
                    "created_at":unix_seconds(),
                })
            },
        );
    }
    if *method != Method::GET {
        return upstream.clone();
    }
    let data = data.cloned().unwrap_or_default();
    let query_task_id = Url::parse(url).ok().and_then(|url| {
        url.query_pairs()
            .find(|(key, _)| key == "taskId")
            .map(|(_, value)| value.into_owned())
    });
    let task_id = data
        .get("task_id")
        .or_else(|| data.get("taskId"))
        .and_then(value_text)
        .or(query_task_id)
        .unwrap_or_default();
    let upstream_status = data
        .get("status")
        .and_then(value_text)
        .unwrap_or_default()
        .to_ascii_uppercase();
    let status = match upstream_status.as_str() {
        "SUCCESS" => "succeeded",
        "CANCELED" => "cancelled",
        "FAIL" | "FAILED" | "UNKNOWN" => "failed",
        "WAITING" | "QUEUED" | "SUBMITTED" | "RUNNING" | "" => "running",
        other => other,
    };
    let video_url = data
        .get("result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|item| item.get("url").and_then(Value::as_str));
    let mut normalized = json!({
        "id":task_id,
        "model":request_model,
        "provider":"lingjing",
        "status":status.to_ascii_lowercase(),
        "video":{},
    });
    if let Some(url) = video_url {
        normalized["video"]["url"] = Value::String(url.to_owned());
    }
    if let Some(error) = data.get("external_error").filter(|value| !value.is_null()) {
        normalized["error"] = json!({"message":error});
    }
    if normalized["status"] == "succeeded" {
        let video_tokens = video_task_route(&task_id)
            .and_then(|route| route.video_tokens)
            .unwrap_or(108_900);
        normalized["usage"] = json!({
            "video_tokens":video_tokens,
            "completion_tokens":video_tokens,
            "total_tokens":video_tokens,
        });
    }
    normalized
}

fn estimate_video_tokens(payload: &Value) -> Option<i64> {
    let root = payload.as_object()?;
    let positive = |key: &str| {
        root.get(key).and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_f64().map(|value| value as i64))
                .or_else(|| value.as_str()?.trim_end_matches(['p', 'P']).parse().ok())
                .filter(|value| *value > 0)
        })
    };
    let duration = positive("duration").unwrap_or(5);
    let fps = positive("fps")
        .or_else(|| positive("framespersecond"))
        .unwrap_or(24);
    let resolution = positive("quality")
        .or_else(|| positive("resolution"))
        .unwrap_or(720);
    let scale = (resolution as f64 / 720.0).powi(2);
    Some((duration as f64 * fps as f64 * 907.5 * scale).round() as i64)
}

fn video_task_routes() -> &'static Mutex<HashMap<String, VideoTaskRoute>> {
    VIDEO_TASK_ROUTES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn video_task_route_for_path(path: &str) -> Option<VideoTaskRoute> {
    let task_id = path.strip_prefix("/v1/video/tasks/")?;
    video_task_route(task_id)
}

fn video_task_route(task_id: &str) -> Option<VideoTaskRoute> {
    let mut routes = video_task_routes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let now = Instant::now();
    routes.retain(|_, route| now.duration_since(route.created_at) < Duration::from_secs(86_400));
    routes.get(task_id).cloned()
}

fn remember_video_task(
    task_id: &str,
    provider_name: &str,
    request_model: &str,
    provider_key: &str,
    video_tokens: Option<i64>,
) {
    let mut routes = video_task_routes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if routes.len() >= 4096 {
        if let Some(oldest) = routes
            .iter()
            .min_by_key(|(_, route)| route.created_at)
            .map(|(task_id, _)| task_id.clone())
        {
            routes.remove(&oldest);
        }
    }
    routes.insert(
        task_id.to_owned(),
        VideoTaskRoute {
            provider_name: provider_name.to_owned(),
            request_model: request_model.to_owned(),
            provider_key: provider_key.to_owned(),
            video_tokens,
            created_at: Instant::now(),
        },
    );
}

fn lingjing_url(base: &str, openapi_path: &str, query: Option<&str>) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid Lingjing base URL: {error}"))?;
    let base_path = url.path().trim_end_matches('/');
    let suffix = openapi_path.trim_start_matches('/');
    let path = if base_path.ends_with("/api/entrance/openapi") {
        format!("{base_path}/{suffix}")
    } else if base_path.ends_with("/api/entrance") {
        format!("{base_path}/openapi/{suffix}")
    } else {
        format!("{base_path}/api/entrance/openapi/{suffix}")
    };
    url.set_path(&path);
    url.set_query(query);
    Ok(url.to_string())
}

fn provider_timeout(provider: &Provider, model: &str) -> Duration {
    let raw = provider.preferences.get("model_timeout");
    let seconds = match raw {
        Some(Value::Number(value)) => value.as_f64(),
        Some(Value::Object(values)) => values
            .get(model)
            .or_else(|| values.get("default"))
            .and_then(Value::as_f64),
        _ => None,
    }
    .filter(|value| value.is_finite() && *value > 0.0)
    .unwrap_or(200.0);
    Duration::from_secs_f64(seconds)
}

fn positive_duration(value: Option<f64>) -> Option<Duration> {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(Duration::from_secs_f64)
}

fn usage(value: &Value) -> (i64, i64, i64) {
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

fn query_value(uri: &Uri, name: &str) -> Option<String> {
    url::form_urlencoded::parse(uri.query()?.as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

async fn default_model_for_path(state: &AppState, headers: &HeaderMap, path: &str) -> String {
    let Ok(models) = state
        .native_responses_config
        .models_for_headers(headers)
        .await
    else {
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

fn trace_id(headers: &HeaderMap, fallback: &str) -> String {
    headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split('-').nth(1))
        .filter(|value| value.len() == 32)
        .unwrap_or(fallback)
        .to_owned()
}

fn client_ip(headers: &HeaderMap) -> String {
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

fn truncate_detail(value: &str) -> String {
    value.chars().take(4096).collect()
}

#[allow(clippy::too_many_arguments)]
fn emit_attempt(
    request_id: &str,
    trace_id: &str,
    role: &str,
    attempt_index: usize,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    url: &str,
    outcome: &str,
    status: Option<u16>,
) {
    let upstream_host = Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default();
    eprintln!(
        "{}",
        json!({
            "event_type":"rust_native_api_attempt",
            "request_id":request_id,
            "trace_id":trace_id,
            "role":role,
            "attempt_index":attempt_index + 1,
            "provider":provider.name.as_ref(),
            "request_model":request_model,
            "actual_model":original_model,
            "upstream_host":upstream_host,
            "outcome":outcome,
            "status_code":status,
        })
    );
}

fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    let mut response = Response::new(Body::from(value.to_string()));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    response
}

fn unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_provider(engine: &str, base_url: &str) -> Provider {
        Provider {
            name: "provider-a".into(),
            base_url: base_url.to_owned().into(),
            engine: engine.to_owned().into(),
            api_keys: std::sync::Arc::new(vec!["upstream-key".into()]),
            project_id: None,
            private_key: None,
            client_email: None,
            aws_access_key: None,
            aws_secret_key: None,
            aws_session_token: None,
            cf_account_id: None,
            region: "global".into(),
            models: std::sync::Arc::new(HashMap::new()),
            preferences: std::sync::Arc::new(Map::new()),
            excluded_endpoints: std::sync::Arc::new(Vec::new()),
            only_request_types: std::sync::Arc::new(Vec::new()),
            excluded_request_types: std::sync::Arc::new(Vec::new()),
            excluded_request_rules: std::sync::Arc::new(Vec::new()),
            cursor: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    #[test]
    fn every_python_model_route_has_a_native_dispatch() {
        for route in [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/images/generations",
            "/v1/images/edits",
            "/v1/embeddings",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
            "/v1/moderations",
            "/v1/video/tasks",
            "/v1/asset-groups",
            "/v1/assets",
            "/v1/alpha/search",
            "/v1/responses",
        ] {
            assert!(supports(&Method::POST, route), "missing {route}");
        }
        assert!(supports(&Method::GET, "/v1/search"));
        assert!(supports(&Method::GET, "/v1/video/tasks/task-1"));
    }

    #[test]
    fn codex_chat_is_compiled_to_responses_wire() {
        let input = json!({
            "model":"gpt-public",
            "messages":[
                {"role":"system","content":"be concise"},
                {"role":"user","content":"hello"}
            ],
            "max_tokens":42,
            "stream":true
        });
        let output = chat_to_responses(&input, "gpt-upstream").unwrap();
        assert_eq!(output["model"], "gpt-upstream");
        assert_eq!(output["max_output_tokens"], 42);
        assert_eq!(output["stream"], false);
        assert_eq!(output["input"][1]["content"][0]["type"], "input_text");
    }

    #[test]
    fn chat_to_responses_preserves_tool_constraints() {
        let input = json!({
            "messages":[{"role":"user","content":"What time is it?"}],
            "tools":[
                {
                    "type":"function",
                    "function":{
                        "name":"now",
                        "parameters":{"type":"object","properties":{}}
                    }
                },
                {
                    "type":"function",
                    "function":{
                        "name":"weather",
                        "parameters":{"type":"object","properties":{}},
                        "strict":true
                    }
                }
            ],
            "tool_choice":{"type":"function","function":{"name":"now"}}
        });

        let output = chat_to_responses(&input, "gpt-upstream").unwrap();

        assert_eq!(output["tools"][0]["strict"], false);
        assert_eq!(output["tools"][1]["strict"], true);
        assert_eq!(
            output["tool_choice"],
            json!({"type":"function","name":"now"})
        );
    }

    #[test]
    fn chat_to_responses_flattens_allowed_tools_choice() {
        let input = json!({
            "messages":[{"role":"user","content":"Use a tool"}],
            "tool_choice":{
                "type":"allowed_tools",
                "allowed_tools":{
                    "mode":"required",
                    "tools":[
                        {"type":"function","function":{"name":"now"}},
                        {"type":"function","name":"weather"}
                    ]
                }
            }
        });

        let output = chat_to_responses(&input, "gpt-upstream").unwrap();

        assert_eq!(
            output["tool_choice"],
            json!({
                "type":"allowed_tools",
                "mode":"required",
                "tools":[
                    {"type":"function","name":"now"},
                    {"type":"function","name":"weather"}
                ]
            })
        );
    }

    #[test]
    fn responses_to_chat_maps_tool_finish_and_usage() {
        let response = responses_to_chat(
            &json!({
                "id":"resp-a",
                "output":[{
                    "type":"function_call",
                    "id":"fc-a",
                    "call_id":"call-a",
                    "name":"now",
                    "arguments":"{\"timezone\":\"Europe/Berlin\"}"
                }],
                "usage":{
                    "input_tokens":3,
                    "output_tokens":5,
                    "total_tokens":8,
                    "input_tokens_details":{"cached_tokens":2,"cache_write_tokens":1},
                    "output_tokens_details":{"reasoning_tokens":4}
                }
            }),
            "public-model",
        );

        assert_eq!(response["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["id"],
            "call-a"
        );
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(response["usage"]["prompt_tokens"], 3);
        assert_eq!(response["usage"]["completion_tokens"], 5);
        assert_eq!(response["usage"]["total_tokens"], 8);
        assert_eq!(
            response["usage"]["prompt_tokens_details"]["cached_tokens"],
            2
        );
        assert_eq!(
            response["usage"]["prompt_tokens_details"]["cache_write_tokens"],
            1
        );
        assert_eq!(
            response["usage"]["completion_tokens_details"]["reasoning_tokens"],
            4
        );
    }

    #[test]
    fn gemini_and_claude_payloads_preserve_system_and_tools() {
        let input = json!({
            "messages":[
                {"role":"system","content":"system"},
                {"role":"user","content":"hello"}
            ],
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
        });
        let gemini = chat_to_gemini(&input, "gemini-upstream").unwrap();
        assert_eq!(gemini["systemInstruction"]["parts"][0]["text"], "system");
        assert_eq!(
            gemini["tools"][0]["functionDeclarations"][0]["name"],
            "lookup"
        );
        let claude = chat_to_claude(&input, "claude-upstream").unwrap();
        assert_eq!(claude["system"], "system");
        assert_eq!(claude["tools"][0]["name"], "lookup");
    }

    #[test]
    fn responses_compat_compiles_to_gemini_and_restores_responses_shape() {
        let provider = test_provider("gemini", "https://generativelanguage.googleapis.com/v1beta");
        let input = PreparedInput {
            payload: Some(json!({
                "model":"public-model",
                "instructions":"be concise",
                "input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],
                "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
                "tool_choice":{"type":"function","name":"lookup"},
                "reasoning":{"effort":"high"},
                "service_tier":"priority",
                "stream":true
            })),
            replay: None,
            observation: SpoolObservation::default(),
            default_model: String::new(),
            content_type: "application/json".into(),
        };
        let uri: Uri = "/v1/responses".parse().unwrap();
        let prepared = build_attempt(
            &provider,
            "key",
            "public-model",
            "gemini-2.5-pro",
            &Method::POST,
            &uri,
            "/v1/responses",
            &HeaderMap::new(),
            &input,
            "request-a",
        )
        .unwrap();
        assert_eq!(
            prepared.downstream_protocol,
            DownstreamProtocol::ResponsesCompat
        );
        assert!(prepared.downstream_stream);
        assert!(prepared.upstream_stream);
        assert_eq!(prepared.adapter, ResponseAdapter::GeminiToChat);
        let AttemptBody::Json(body) = prepared.body else {
            panic!("responses compatibility request must use JSON");
        };
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be concise");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(body["serviceTier"], "PRIORITY");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            24576
        );

        let response = chat_to_responses_response(
            &json!({
                "choices":[{"message":{"role":"assistant","content":"done","tool_calls":[{"id":"call-a","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}],
                "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
            }),
            "public-model",
        );
        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "public-model");
        assert_eq!(response["output"][0]["content"][0]["text"], "done");
        assert_eq!(response["output"][1]["type"], "function_call");
        assert_eq!(response["usage"]["total_tokens"], 5);
    }

    #[test]
    fn search_responses_are_normalized_without_losing_provider_fields() {
        let normalized = normalize_search_response(
            "https://api.tavily.com/search",
            &json!({
                "query":"rust",
                "results":[{"title":"Rust","url":"https://rust-lang.org","content":"language","score":0.9}],
                "request_id":"search-a"
            }),
        );
        assert_eq!(normalized["code"], 200);
        assert_eq!(normalized["data"][0]["description"], "language");
        assert_eq!(normalized["data"][0]["score"], 0.9);
        assert_eq!(normalized["meta"]["provider"], "tavily");
        assert_eq!(normalized["meta"]["request_id"], "search-a");
    }

    #[test]
    fn gemini_thought_signatures_round_trip_for_images_and_tools() {
        let encoded = BASE64.encode(b"\x89PNG\r\n\x1a\nimage");
        cache_gemini_image_thought_signature(&encoded, "image-signature");
        let part = data_url_part(&format!("data:image/png;base64,{encoded}")).unwrap();
        assert_eq!(part["thoughtSignature"], "image-signature");

        let chat = gemini_to_chat(
            &json!({
                "candidates":[{"content":{"parts":[{
                    "functionCall":{"name":"lookup","args":{"q":"rust"}},
                    "thoughtSignature":"tool-signature"
                }]}}]
            }),
            "gemini-public",
        );
        let call_id = chat["choices"][0]["message"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap();
        assert_eq!(
            decode_gemini_thought_signature(call_id).as_deref(),
            Some("tool-signature")
        );
    }

    #[test]
    fn gemini_audio_and_claude_controls_are_translated() {
        let input = json!({
            "messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}],
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            "tool_choice":"required",
            "reasoning_effort":"medium",
            "service_tier":"flex",
            "modalities":["audio"],
            "audio":{"voice":"Aoede"}
        });
        let gemini = chat_to_gemini(&input, "gemini-3-flash").unwrap();
        assert_eq!(
            gemini["contents"][0]["parts"][0]["inlineData"]["mimeType"],
            "audio/wav"
        );
        assert_eq!(
            gemini["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
                ["voiceName"],
            "Aoede"
        );
        let claude = chat_to_claude(&input, "claude-sonnet").unwrap();
        assert_eq!(claude["tool_choice"]["type"], "any");
        assert_eq!(claude["thinking"]["budget_tokens"], 4096);
        assert_eq!(claude["service_tier"], "standard_only");
    }

    #[tokio::test]
    async fn multipart_media_is_parsed_and_rebuilt_with_the_upstream_model() {
        let source_boundary = "CaseSensitiveBoundary";
        let source = format!(
            "--{source_boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit this\r\n--{source_boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.bin\"\r\nContent-Type: application/octet-stream\r\n\r\nbinary\0payload\r\n--{source_boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\npublic-model\r\n--{source_boundary}--\r\n"
        );
        let manager = crate::request_spool::SpoolManager::new(
            crate::resources::ResourceGovernor::unconstrained_for_test(),
        )
        .unwrap();
        let mut writer = manager
            .begin(None, Some(source.len() as u64), Duration::ZERO)
            .await
            .unwrap();
        writer.append(Bytes::from(source)).await.unwrap();
        let spool = writer.finish().await.unwrap();
        let content_type = format!("multipart/form-data; boundary={source_boundary}");
        assert_eq!(
            spool
                .storage
                .multipart_text_field(&content_type, "model", 4096)
                .await
                .unwrap()
                .as_deref(),
            Some("public-model")
        );
        let output_boundary = "rewritten-boundary".to_owned();
        let stream = multipart_rewrite_stream(
            spool.storage,
            spool.observation,
            &content_type,
            output_boundary.clone(),
            "upstream-model".into(),
        )
        .await
        .unwrap();
        let mut multipart = multer::Multipart::new(stream, output_boundary);
        let mut fields = HashMap::new();
        while let Some(field) = multipart.next_field().await.unwrap() {
            let name = field.name().unwrap().to_owned();
            fields.insert(name, field.bytes().await.unwrap());
        }
        assert_eq!(fields["prompt"], "edit this");
        assert_eq!(fields["model"], "upstream-model");
        assert_eq!(fields["image"], b"binary\0payload"[..]);
    }

    #[test]
    fn upstream_responses_are_normalized_to_openai_chat() {
        let gemini = gemini_to_chat(
            &json!({
                "candidates":[{"content":{"parts":[{"text":"hello"}]}}],
                "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}
            }),
            "gemini-public",
        );
        assert_eq!(gemini["choices"][0]["message"]["content"], "hello");
        assert_eq!(gemini["usage"]["total_tokens"], 5);
        let claude = claude_to_chat(
            &json!({"content":[{"type":"text","text":"world"}],"usage":{"input_tokens":1,"output_tokens":2}}),
            "claude-public",
        );
        assert_eq!(claude["choices"][0]["message"]["content"], "world");
        assert_eq!(claude["usage"]["total_tokens"], 3);
    }

    #[test]
    fn special_provider_urls_and_payloads_match_legacy_contracts() {
        assert_eq!(
            azure_chat_url("https://azure.example.com", "deployment-a").unwrap(),
            "https://azure.example.com/openai/deployments/deployment-a/chat/completions?api-version=2025-01-01-preview"
        );
        assert_eq!(
            databricks_chat_url("https://dbc.example.com", "serving-a").unwrap(),
            "https://dbc.example.com/serving-endpoints/serving-a/invocations"
        );
        let cohere = chat_to_cohere(
            &json!({"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hello"}]}),
            "command-r",
        )
        .unwrap();
        assert_eq!(cohere["message"], "hello");
        assert_eq!(cohere["chat_history"][0]["role"], "SYSTEM");

        let mut cloudflare = test_provider("cloudflare", "https://api.cloudflare.com");
        cloudflare.cf_account_id = Some("account-a".into());
        assert_eq!(
            cloudflare_url(&cloudflare, "@cf/meta/llama").unwrap(),
            "https://api.cloudflare.com/client/v4/accounts/account-a/ai/run/@cf/meta/llama"
        );
    }

    #[test]
    fn lingjing_video_contract_and_task_affinity_match_legacy_runtime() {
        let converted = content_generation_to_lingjing(
            &json!({
                "model":"seedance-2-0",
                "prompt":"sunlight",
                "resources":[{"type":"image","url":"asset://Asset-test","role":"first_frame"}],
                "duration":5,
                "resolution":"720p",
                "ratio":"16:9",
                "generate_audio":false,
            }),
            "sd_2_0",
        )
        .unwrap();
        assert_eq!(converted["modelCode"], "sd_2_0");
        assert_eq!(converted["taskParams"]["input"]["quality"], "720");
        assert_eq!(
            converted["taskParams"]["input"]["resources"][0]["source"],
            json!({"kind":"asset_id","value":"Asset-test"})
        );

        let mut provider = test_provider("lingjing", "https://api-llm.lingjingai.cn");
        provider.preferences = std::sync::Arc::new(Map::from_iter([
            ("access_key".into(), json!("ak-test")),
            ("secret_key".into(), json!("sk-test")),
        ]));
        let headers = provider_headers(
            &provider,
            "routing-key",
            &HeaderMap::new(),
            "request-a",
            "lingjing",
            false,
            None,
        )
        .unwrap();
        assert_eq!(headers["x-access-key"], "ak-test");
        assert_eq!(headers["x-secret-key"], "sk-test");
        assert!(headers.get("authorization").is_none());

        remember_video_task(
            "task-rust-lingjing",
            "lingjing",
            "seedance-2-0",
            "routing-key",
            Some(108_900),
        );
        let route = video_task_route_for_path("/v1/video/tasks/task-rust-lingjing").unwrap();
        assert_eq!(route.provider_name, "lingjing");
        assert_eq!(route.provider_key, "routing-key");
        let normalized = normalize_lingjing_video_response(
            &Method::GET,
            "seedance-2-0",
            "https://api-llm.lingjingai.cn/draw/task/query?taskId=task-rust-lingjing",
            &json!({"data":{"task_id":"task-rust-lingjing","status":"SUCCESS","result":[{"url":"https://example.com/out.mp4"}]}}),
        );
        assert_eq!(normalized["status"], "succeeded");
        assert_eq!(normalized["video"]["url"], "https://example.com/out.mp4");
        assert_eq!(normalized["usage"]["video_tokens"], 108_900);
    }

    #[test]
    fn long_tail_payload_adapters_match_python_contracts() {
        let mut doubao = test_provider("doubao-translation", "https://example.com/responses");
        doubao.preferences = std::sync::Arc::new(Map::from_iter([(
            "post_body_parameter_overrides".into(),
            json!({"translate-public":{"translation_options":{"source_language":"en","target_language":"ja"}}}),
        )]));
        let translated = chat_to_doubao_translation(
            &json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"stream":true}),
            "doubao-upstream",
            "translate-public",
            &doubao,
        )
        .unwrap();
        assert_eq!(translated["input"][0]["content"][0]["text"], "hello");
        assert_eq!(
            translated["input"][0]["content"][0]["translation_options"]["target_language"],
            "ja"
        );
        assert_eq!(translated["stream"], true);

        let minimax = openai_tts_to_minimax(
            &json!({"input":"speak","voice":"alloy","speed":1.25}),
            "speech-02-hd",
        )
        .unwrap();
        assert_eq!(minimax["text"], "speak");
        assert_eq!(minimax["voice_setting"]["voice_id"], "alloy");

        let mut jina = json!({"input":"text","encoding_format":"float"});
        normalize_jina_embedding(&mut jina, "jina-embeddings-v3").unwrap();
        assert_eq!(jina["embedding_type"], "float");
        assert!(jina.get("encoding_format").is_none());
    }

    #[test]
    fn compact_and_validation_routes_are_native() {
        assert!(supports(&Method::POST, "/v1/responses/compact"));
        assert!(known_path("/v1/responses/compact"));
        assert_eq!(
            missing_required_field("/v1/chat/completions", &json!({"model":"gpt"})),
            Some("messages")
        );
        assert_eq!(
            missing_required_field(
                "/v1/chat/completions",
                &json!({"model":"gpt","messages":[{"role":"user","content":"hi"}]})
            ),
            None
        );
    }

    #[test]
    fn provider_headers_use_protocol_specific_authentication() {
        let request_headers = HeaderMap::new();
        let azure = test_provider("azure", "https://azure.example.com");
        let headers = provider_headers(
            &azure,
            "azure-key",
            &request_headers,
            "request-a",
            "azure",
            false,
            None,
        )
        .unwrap();
        assert_eq!(headers["api-key"], "azure-key");
        assert!(headers.get("authorization").is_none());

        let openrouter = test_provider(
            "openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
        );
        let headers = provider_headers(
            &openrouter,
            "key",
            &request_headers,
            "request-a",
            "openrouter",
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            headers["http-referer"],
            "https://github.com/yym68686/uni-api"
        );
        assert_eq!(headers["x-title"], "Uni API");
    }

    #[test]
    fn alpha_search_strips_responses_fields_and_skips_provider_overrides() {
        let mut provider = test_provider("codex", "https://example.com/v1/responses");
        let mut preferences = Map::new();
        preferences.insert(
            "post_body_parameter_overrides".into(),
            json!({"store": false, "metadata": {"source": "generic-override"}}),
        );
        provider.preferences = std::sync::Arc::new(preferences);
        let input = PreparedInput {
            payload: Some(json!({
                "id": "search-session-a",
                "model": "gpt-public",
                "input": "query",
                "commands": [],
                "settings": {},
                "max_output_tokens": 256,
                "store": true,
                "stream": true,
                "prompt_cache_key": "cache-a",
                "prompt_cache_retention": "24h",
                "future_search_field": {"enabled": true}
            })),
            replay: None,
            observation: SpoolObservation::default(),
            default_model: String::new(),
            content_type: "application/json".into(),
        };
        let uri: Uri = ALPHA_SEARCH_ENDPOINT.parse().unwrap();
        let prepared = build_attempt(
            &provider,
            "key",
            "gpt-public",
            "gpt-upstream",
            &Method::POST,
            &uri,
            ALPHA_SEARCH_ENDPOINT,
            &HeaderMap::new(),
            &input,
            "request-a",
        )
        .unwrap();
        let AttemptBody::Json(body) = prepared.body else {
            panic!("alpha/search must use a JSON body");
        };
        let body: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(prepared.url, "https://example.com/v1/alpha/search");
        assert!(!prepared.downstream_stream);
        assert!(!prepared.upstream_stream);
        assert_eq!(body["model"], "gpt-upstream");
        assert_eq!(body["future_search_field"]["enabled"], true);
        for field in [
            "store",
            "stream",
            "prompt_cache_key",
            "prompt_cache_retention",
            "metadata",
        ] {
            assert!(body.get(field).is_none(), "unexpected field {field}");
        }
        assert_eq!(prepared.headers["openai-beta"], "responses=experimental");
        assert_eq!(prepared.headers["originator"], "codex_cli_rs");
        assert_eq!(prepared.headers["session_id"], "search-session-a");
        assert_eq!(prepared.headers["user-agent"], CODEX_USER_AGENT);
        assert_eq!(prepared.headers["accept"], "application/json");
    }

    #[test]
    fn non_alpha_routes_still_apply_provider_overrides() {
        let mut provider = test_provider("codex", "https://example.com/v1/responses");
        let mut preferences = Map::new();
        preferences.insert(
            "post_body_parameter_overrides".into(),
            json!({"store": false}),
        );
        provider.preferences = std::sync::Arc::new(preferences);
        let input = PreparedInput {
            payload: Some(json!({"model": "gpt-public", "input": "hello"})),
            replay: None,
            observation: SpoolObservation::default(),
            default_model: String::new(),
            content_type: "application/json".into(),
        };
        let uri: Uri = "/v1/moderations".parse().unwrap();
        let prepared = build_attempt(
            &provider,
            "key",
            "gpt-public",
            "gpt-upstream",
            &Method::POST,
            &uri,
            "/v1/moderations",
            &HeaderMap::new(),
            &input,
            "request-a",
        )
        .unwrap();
        let AttemptBody::Json(body) = prepared.body else {
            panic!("moderations must use a JSON body");
        };
        let body: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn aws_bedrock_request_is_signed_with_sigv4() {
        let mut provider = test_provider("aws", "https://bedrock-runtime.us-east-1.amazonaws.com");
        provider.aws_access_key = Some("AKIA_TEST".into());
        provider.aws_secret_key = Some("secret".into());
        let url = aws_bedrock_url(&provider, "anthropic.claude-3-haiku:0", true).unwrap();
        let mut headers = HeaderMap::new();
        sign_aws_request_at(
            &provider,
            &url,
            br#"{"messages":[]}"#,
            &mut headers,
            UNIX_EPOCH,
        )
        .unwrap();
        assert_eq!(headers["x-amz-date"], "19700101T000000Z");
        assert!(headers["authorization"].to_str().unwrap().starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIA_TEST/19700101/us-east-1/bedrock/aws4_request"
        ));
        assert_eq!(
            headers["accept"],
            "application/vnd.amazon.bedrock.payload+json"
        );
    }

    #[test]
    fn vertex_claude_uses_project_and_region() {
        let mut provider = test_provider("vertex-claude", "https://aiplatform.googleapis.com");
        provider.project_id = Some("project-a".into());
        provider.region = "europe-west1".into();
        assert_eq!(
            vertex_claude_url(&provider, "claude-sonnet-4-5@20250929").unwrap(),
            "https://europe-west1-aiplatform.googleapis.com/v1/projects/project-a/locations/europe-west1/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict"
        );
    }

    #[test]
    fn moderation_extracts_the_legacy_last_text_shapes() {
        assert_eq!(
            moderation_text(&json!({
                "messages":[
                    {"role":"user","content":"earlier"},
                    {"role":"assistant","content":[{"type":"text","text":"latest"}]}
                ]
            }))
            .as_deref(),
            Some("latest")
        );
        assert_eq!(
            moderation_text(&json!({"model":"text-embedding-3-small","input":["one","two"]}))
                .as_deref(),
            Some("one\ntwo")
        );
        assert_eq!(
            moderation_text(&json!({
                "input":[{"role":"user","content":[
                    {"type":"input_text","text":"first"},
                    {"type":"input_text","text":"last"}
                ]}]
            }))
            .as_deref(),
            Some("last")
        );
    }
}
