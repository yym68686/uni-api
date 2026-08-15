use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use serde_json::{json, Value};

use crate::proxy::{json_error, AppState};

const MODEL_CREATED: u64 = 1_720_524_448_858;

pub async fn handle(
    state: &AppState,
    method: &Method,
    uri: &Uri,
    request_path: &str,
    headers: &HeaderMap,
) -> Option<Response<Body>> {
    let path = request_path.trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };
    let mut response = match (method, path) {
        (&Method::GET, "/healthz") => Some(json_response(
            StatusCode::OK,
            json!({"status":"ok","version": app_version()}),
        )),
        (&Method::GET, "/v1/observability/runtime") => {
            let (api_keys, providers, models) =
                state.native_responses_config.runtime_counts().await;
            let details = state.runtime_observability().await;
            Some(json_response(
                StatusCode::OK,
                json!({
                    "runtime": "rust",
                    "python_compat_enabled": state.python_compat_enabled,
                    "configuration_ready": state.native_responses_config.is_ready().await,
                    "database_disabled": state.persistence.disabled(),
                    "persistence": if state.persistence.disabled() { "disabled" } else { "rust" },
                    "api_key_count": api_keys,
                    "provider_count": providers,
                    "model_count": models,
                    "resource_governor":details.get("resource_governor").cloned().unwrap_or(Value::Null),
                    "idempotency":details.get("idempotency").cloned().unwrap_or(Value::Null),
                    "upstream_http_clients":details.get("upstream_http_clients").cloned().unwrap_or(Value::Null),
                }),
            ))
        }
        (&Method::GET, "/v1/models") => Some(models_response(state, uri, headers).await),
        (&Method::GET, "/v1/generate-api-key") => {
            if let Err(error) = state.native_responses_config.authorize(headers).await {
                Some(json_error(error.status, &error.message))
            } else {
                Some(json_response(
                    StatusCode::OK,
                    json!({"api_key": generate_api_key()}),
                ))
            }
        }
        (&Method::GET, "/v1/stats") => Some(stats_response(state, uri, headers).await),
        (&Method::GET, "/v1/token_usage") => Some(token_usage_response(state, uri, headers).await),
        (&Method::GET, "/v1/channel_key_rankings") => {
            Some(channel_rankings_response(state, uri, headers).await)
        }
        (&Method::GET, "/v1/api_keys_states") => {
            Some(api_key_states_response(state, headers).await)
        }
        (&Method::GET, "/v1/api_config") => Some(api_config_response(state, headers).await),
        (&Method::GET, "/openapi.json") => Some(json_response(StatusCode::OK, openapi_document())),
        (&Method::GET, "/favicon.ico") => Some(binary_response(
            "image/x-icon",
            include_bytes!("../../../../../static/favicon.ico"),
        )),
        (&Method::GET, "/apple-touch-icon.png") => Some(binary_response(
            "image/png",
            include_bytes!("../../../../../static/apple-touch-icon.png"),
        )),
        (&Method::GET, "/apple-touch-icon-precomposed.png") => Some(binary_response(
            "image/png",
            include_bytes!("../../../../../static/apple-touch-icon-precomposed.png"),
        )),
        (&Method::GET, "/docs/markdown") => Some(text_response(
            StatusCode::OK,
            "text/markdown; charset=utf-8",
            include_str!("../../../../../README.md"),
        )),
        (&Method::GET, "/docs") | (&Method::GET, "/redoc") => Some(text_response(
            StatusCode::OK,
            "text/html; charset=utf-8",
            r#"<!doctype html><meta charset="utf-8"><title>uni-api</title><script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"><div id="swagger"></div><script>SwaggerUIBundle({url:'/openapi.json',dom_id:'#swagger'})</script>"#,
        )),
        (&Method::GET, "/") => Some(redirect_response("https://uni-api-web.pages.dev")),
        _ => None,
    };
    if let Some(response) = response.as_mut() {
        insert_request_id(response.headers_mut(), headers);
        response
            .headers_mut()
            .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
    }
    response
}

pub fn supports_mutation(method: &Method, path: &str) -> bool {
    *method == Method::POST && matches!(path, "/v1/api_config/update" | "/v1/add_credits")
}

pub async fn handle_mutation(state: &AppState, request: Request) -> Response<Body> {
    let path = request.uri().path().trim_end_matches('/').to_owned();
    let headers = request.headers().clone();
    if let Err(response) = require_admin(state, &headers).await {
        return response;
    }
    match path.as_str() {
        "/v1/api_config/update" => {
            let limit = std::env::var("RUST_ADMIN_CONFIG_MAX_BYTES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(16 * 1024 * 1024);
            let body = match to_bytes(request.into_body(), limit).await {
                Ok(body) => body,
                Err(error) => {
                    return json_error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!("Configuration update body is too large: {error}"),
                    )
                }
            };
            let patch = match serde_json::from_slice::<Value>(&body) {
                Ok(Value::Object(patch)) => Value::Object(patch),
                Ok(_) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "Configuration patch must be an object",
                    )
                }
                Err(error) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        &format!("Invalid configuration patch: {error}"),
                    )
                }
            };
            match state.config_publisher.apply_patch(&patch).await {
                Ok(()) => {
                    let _ = state.native_responses_config.refresh().await;
                    json_response(StatusCode::OK, json!({"message":"API config updated"}))
                }
                Err(error) => json_error(StatusCode::CONFLICT, &error),
            }
        }
        "/v1/add_credits" => {
            let paid_key = query_value(request.uri(), "paid_key").unwrap_or_default();
            let amount = query_value(request.uri(), "amount")
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0);
            match state.config_publisher.add_credits(&paid_key, amount).await {
                Ok(credits) => {
                    let _ = state.native_responses_config.refresh().await;
                    json_response(
                        StatusCode::OK,
                        json!({"paid_key":paid_key,"credits":credits}),
                    )
                }
                Err(error) if error == "Paid API key not found" => {
                    json_error(StatusCode::NOT_FOUND, &error)
                }
                Err(error) => json_error(StatusCode::BAD_REQUEST, &error),
            }
        }
        _ => json_error(StatusCode::NOT_FOUND, "Not found"),
    }
}

async fn models_response(state: &AppState, uri: &Uri, headers: &HeaderMap) -> Response<Body> {
    let models = match state
        .native_responses_config
        .models_for_headers(headers)
        .await
    {
        Ok(models) => models,
        Err(403) => return json_error(StatusCode::FORBIDDEN, "Invalid or missing API Key"),
        Err(_) => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Runtime configuration is not ready",
            )
        }
    };
    if query_value(uri, "client_version").is_some() {
        let catalog: Value = serde_json::from_str(include_str!(
            "../../../../../uni_api/api/codex_models_pro_0_144_0.json"
        ))
        .unwrap_or_else(|_| json!({"models":[]}));
        let allowed = models.iter().map(String::as_str).collect::<HashSet<_>>();
        let catalog_models = catalog
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let by_slug = catalog_models
            .iter()
            .filter_map(|model| {
                Some((
                    model.get("slug")?.as_str()?.to_ascii_lowercase(),
                    model.clone(),
                ))
            })
            .collect::<HashMap<_, _>>();
        let mut filtered = catalog_models
            .into_iter()
            .filter(|model| {
                model
                    .get("slug")
                    .and_then(Value::as_str)
                    .is_some_and(|slug| allowed.contains(slug))
            })
            .collect::<Vec<_>>();
        let mut included = filtered
            .iter()
            .filter_map(|model| model.get("slug").and_then(Value::as_str))
            .map(str::to_owned)
            .collect::<HashSet<_>>();
        for model in &models {
            if included.contains(model) || !is_codex_catalog_model(model) {
                continue;
            }
            filtered.push(codex_compatible_model(
                model,
                100 + filtered.len(),
                &by_slug,
            ));
            included.insert(model.clone());
        }
        let mut response = json_response(StatusCode::OK, json!({"models":filtered}));
        response.headers_mut().insert(
            "x-uni-api-models-source",
            HeaderValue::from_static("codex-pro-snapshot"),
        );
        return response;
    }
    json_response(
        StatusCode::OK,
        json!({
            "object": "list",
            "data": models.into_iter().map(|model| json!({
                "id": model,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "uni-api",
            })).collect::<Vec<_>>(),
        }),
    )
}

fn is_codex_catalog_model(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    !lower.is_empty()
        && ![
            "audio",
            "dall-e",
            "embedding",
            "image",
            "moderation",
            "rerank",
            "seedance",
            "sora",
            "speech",
            "tts",
            "video",
            "whisper",
        ]
        .into_iter()
        .any(|token| lower.contains(token))
}

fn codex_compatible_model(model: &str, priority: usize, catalog: &HashMap<String, Value>) -> Value {
    let lower = model.to_ascii_lowercase();
    if let Some(mut template) = catalog
        .iter()
        .filter(|(family, _)| lower == **family || lower.starts_with(&format!("{family}-")))
        .max_by_key(|(family, _)| family.len())
        .map(|(_, value)| value.clone())
    {
        template["slug"] = Value::String(model.to_owned());
        template["display_name"] = Value::String(model.to_owned());
        template["description"] = Value::String("Available through uni-api.".into());
        template["priority"] = json!(priority);
        return template;
    }
    let supports_reasoning = lower.contains("codex")
        || ["gpt-5", "o1", "o3", "o4"]
            .into_iter()
            .any(|prefix| lower.starts_with(prefix));
    let supports_images = !lower.contains("deepseek");
    let mut fallback = json!({
        "slug":model,
        "display_name":model,
        "description":"Available through uni-api.",
        "supported_reasoning_levels":if supports_reasoning { json!([
            {"effort":"low","description":"Fast responses with lighter reasoning"},
            {"effort":"medium","description":"Balances speed and reasoning depth for everyday tasks"},
            {"effort":"high","description":"Greater reasoning depth for complex problems"},
            {"effort":"xhigh","description":"Extra high reasoning depth for complex problems"}
        ]) } else { json!([]) },
        "shell_type":"shell_command",
        "visibility":"list",
        "supported_in_api":true,
        "priority":priority,
        "additional_speed_tiers":["fast"],
        "service_tiers":[{"id":"priority","name":"Fast","description":"1.5x speed, increased usage"}],
        "availability_nux":Value::Null,
        "upgrade":Value::Null,
        "base_instructions":"You are Codex, a coding agent. Read the codebase before making focused changes, preserve unrelated work, validate the result, and communicate concise progress.",
        "supports_reasoning_summaries":supports_reasoning,
        "default_reasoning_summary":"auto",
        "support_verbosity":false,
        "default_verbosity":Value::Null,
        "apply_patch_tool_type":"freeform",
        "web_search_tool_type":"text",
        "truncation_policy":{"mode":"tokens","limit":10000},
        "supports_parallel_tool_calls":true,
        "supports_image_detail_original":supports_images,
        "context_window":272000,
        "max_context_window":272000,
        "auto_compact_token_limit":Value::Null,
        "effective_context_window_percent":95,
        "experimental_supported_tools":[],
        "input_modalities":if supports_images { json!(["text","image"]) } else { json!(["text"]) },
        "supports_search_tool":false,
        "use_responses_lite":false,
    });
    if supports_reasoning {
        fallback["default_reasoning_level"] = Value::String("medium".into());
    }
    fallback
}

async fn stats_response(state: &AppState, uri: &Uri, headers: &HeaderMap) -> Response<Body> {
    if let Err(response) = require_admin(state, headers).await {
        return response;
    }
    let hours = query_value(uri, "hours")
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(24)
        .clamp(1, 720);
    match state.persistence.stats_summary(hours).await {
        Ok(value) => json_response(StatusCode::OK, value),
        Err(error) => json_error(StatusCode::SERVICE_UNAVAILABLE, &error),
    }
}

async fn token_usage_response(state: &AppState, uri: &Uri, headers: &HeaderMap) -> Response<Body> {
    let auth = match state.native_responses_config.authorize(headers).await {
        Ok(auth) => auth,
        Err(error) => return json_error(error.status, &error.message),
    };
    let is_admin =
        auth.api_key.role.to_ascii_lowercase().contains("admin") || auth.api_key_count == 1;
    let requested_key = query_value(uri, "api_key_param");
    let filter_key = if is_admin {
        requested_key.as_deref()
    } else {
        Some(auth.api_key.token.as_ref())
    };
    let model = query_value(uri, "model");
    let (start, end) = time_range(uri, 30);
    match state
        .persistence
        .token_usage(filter_key, model.as_deref(), start, end)
        .await
    {
        Ok(value) => json_response(StatusCode::OK, value),
        Err(error) => json_error(StatusCode::SERVICE_UNAVAILABLE, &error),
    }
}

async fn channel_rankings_response(
    state: &AppState,
    uri: &Uri,
    headers: &HeaderMap,
) -> Response<Body> {
    if let Err(response) = require_admin(state, headers).await {
        return response;
    }
    let provider = query_value(uri, "provider_name").unwrap_or_default();
    if provider.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "provider_name is required");
    }
    let (start, end) = time_range(uri, 1);
    match state
        .persistence
        .channel_key_rankings(&provider, start, end)
        .await
    {
        Ok(value) => json_response(StatusCode::OK, value),
        Err(error) => json_error(StatusCode::SERVICE_UNAVAILABLE, &error),
    }
}

async fn api_key_states_response(state: &AppState, headers: &HeaderMap) -> Response<Body> {
    if let Err(response) = require_admin(state, headers).await {
        return response;
    }
    json_response(
        StatusCode::OK,
        json!({"api_keys_states":state.native_responses_config.paid_api_key_states(&state.persistence).await}),
    )
}

async fn api_config_response(state: &AppState, headers: &HeaderMap) -> Response<Body> {
    if let Err(response) = require_admin(state, headers).await {
        return response;
    }
    match state.native_responses_config.api_config().await {
        Some(config) => json_response(StatusCode::OK, json!({"api_config":config})),
        None => json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Runtime configuration is not ready",
        ),
    }
}

async fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(), Response<Body>> {
    let auth = state
        .native_responses_config
        .authorize(headers)
        .await
        .map_err(|error| json_error(error.status, &error.message))?;
    if auth.api_key_count > 1 && !auth.api_key.role.to_ascii_lowercase().contains("admin") {
        return Err(json_error(StatusCode::FORBIDDEN, "Permission denied"));
    }
    Ok(())
}

fn time_range(uri: &Uri, default_days: i64) -> (Option<i64>, Option<i64>) {
    if let Some(days) = query_value(uri, "last_n_days").and_then(|value| value.parse::<i64>().ok())
    {
        let end = unix_seconds();
        return (
            Some(end.saturating_sub(days.max(1).saturating_mul(86_400))),
            Some(end),
        );
    }
    let start = query_value(uri, "start_datetime").and_then(|value| parse_datetime(&value));
    let end = query_value(uri, "end_datetime").and_then(|value| parse_datetime(&value));
    if start.is_some() || end.is_some() {
        (start, end)
    } else {
        let end = unix_seconds();
        (
            Some(end.saturating_sub(default_days.saturating_mul(86_400))),
            Some(end),
        )
    }
}

fn parse_datetime(value: &str) -> Option<i64> {
    if let Ok(value) = value.parse::<f64>() {
        return value.is_finite().then_some(value as i64);
    }
    let value = value.trim_end_matches('Z');
    let (date, raw_time) = value.split_once('T').or_else(|| value.split_once(' '))?;
    let mut time = raw_time.split('+').next().unwrap_or(raw_time);
    if let Some(index) = time.rfind('-').filter(|index| *index > 2) {
        time = &time[..index];
    }
    let mut date = date
        .split('-')
        .filter_map(|value| value.parse::<i64>().ok());
    let (year, month, day) = (date.next()?, date.next()?, date.next()?);
    let mut time = time.split(':');
    let hour = time.next()?.parse::<i64>().ok()?;
    let minute = time.next()?.parse::<i64>().ok()?;
    let second = time.next()?.split('.').next()?.parse::<i64>().ok()?;
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn query_value(uri: &Uri, name: &str) -> Option<String> {
    url::form_urlencoded::parse(uri.query()?.as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

fn openapi_document() -> Value {
    let mut paths = serde_json::Map::new();
    for (method, path) in [
        ("post", "/v1/chat/completions"),
        ("post", "/v1/responses"),
        ("post", "/v1/responses/compact"),
        ("post", "/v1/messages"),
        ("get", "/v1/models"),
        ("post", "/v1/images/generations"),
        ("post", "/v1/images/edits"),
        ("post", "/v1/embeddings"),
        ("post", "/v1/audio/speech"),
        ("post", "/v1/audio/transcriptions"),
        ("post", "/v1/moderations"),
        ("post", "/v1/video/tasks"),
        ("get", "/v1/video/tasks/{task_id}"),
        ("post", "/v1/asset-groups"),
        ("get", "/v1/asset-groups/{group_id}"),
        ("post", "/v1/assets"),
        ("get", "/v1/assets/{asset_id}"),
        ("get", "/v1/stats"),
        ("get", "/v1/token_usage"),
    ] {
        paths.entry(path).or_insert_with(|| json!({}))[method] =
            json!({"responses":{"200":{"description":"OK"}}});
    }
    json!({"openapi":"3.1.0","info":{"title":"uni-api","version":app_version()},"paths":paths})
}

fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    (
        status,
        [("content-type", "application/json")],
        value.to_string(),
    )
        .into_response()
}

fn text_response(
    status: StatusCode,
    content_type: &'static str,
    value: &'static str,
) -> Response<Body> {
    (status, [("content-type", content_type)], value).into_response()
}

fn binary_response(content_type: &'static str, value: &'static [u8]) -> Response<Body> {
    (StatusCode::OK, [("content-type", content_type)], value).into_response()
}

fn redirect_response(location: &str) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::FOUND;
    if let Ok(value) = HeaderValue::from_str(location) {
        response.headers_mut().insert("location", value);
    }
    response
}

fn insert_request_id(output: &mut HeaderMap, input: &HeaderMap) {
    let request_id = input
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split('-').nth(1))
        .filter(|value| value.len() == 32)
        .or_else(|| {
            input
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
        });
    if let Some(request_id) = request_id.and_then(|value| HeaderValue::from_str(value).ok()) {
        output.insert("x-request-id", request_id);
    }
}

fn generate_api_key() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    let digest = format!("{:x}", hasher.finalize());
    format!("sk-{}", &digest[..48])
}

fn unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

fn app_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            let project = include_str!("../../../../../pyproject.toml");
            project
                .lines()
                .find_map(|line| line.trim().strip_prefix("version = \"")?.strip_suffix('"'))
                .unwrap_or("unknown")
                .to_owned()
        })
        .as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_version_is_embedded() {
        assert!(app_version().starts_with("1.7."));
    }

    #[test]
    fn generated_keys_match_public_shape() {
        let key = generate_api_key();
        assert!(key.starts_with("sk-"));
        assert_eq!(key.len(), 51);
    }

    #[test]
    fn parses_unix_and_iso_time_ranges() {
        assert_eq!(parse_datetime("1700000000"), Some(1_700_000_000));
        assert_eq!(parse_datetime("1970-01-01T00:00:00Z"), Some(0));
    }

    #[test]
    fn codex_catalog_generates_safe_fallback_models() {
        let fallback = codex_compatible_model("gpt-5-custom", 101, &HashMap::new());
        assert_eq!(fallback["slug"], "gpt-5-custom");
        assert_eq!(fallback["default_reasoning_level"], "medium");
        assert_eq!(fallback["input_modalities"], json!(["text", "image"]));
        assert!(is_codex_catalog_model("deepseek-chat"));
        assert!(!is_codex_catalog_model("text-embedding-3-large"));
    }
}
