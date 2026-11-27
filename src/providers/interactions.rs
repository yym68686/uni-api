use super::request::provider_headers;
use super::types::{AttemptBody, DownstreamProtocol, PreparedAttempt, ResponseAdapter};
use crate::config::snapshot::Provider;
use crate::protocols::interactions;
use axum::http::{HeaderMap, HeaderValue, Method};
use serde_json::{json, Value};
use url::Url;

const GOOGLE_HOST: &str = "generativelanguage.googleapis.com";
pub(crate) const VERSION: &str = "gemini_interactions_version";

pub(crate) fn supports_model(provider: &Provider, model: &str) -> bool {
    let conversational = crate::providers::codex::models::is_conversational_model;
    conversational(model)
        && provider
            .models
            .get(model)
            .is_some_and(|upstream| conversational(upstream))
}

pub(crate) fn native(provider: &Provider, model: &str) -> bool {
    crate::config::model_protocol::selected(provider, model).is_none()
        && upstream_url(provider).is_some()
}

pub(crate) fn supports_provider(provider: &Provider) -> bool {
    upstream_url(provider).is_some()
        || matches!(
            provider.engine.trim().to_ascii_lowercase().as_str(),
            "gpt"
                | "codex"
                | "openrouter"
                | "azure"
                | "azure-databricks"
                | "gemini"
                | "vertex"
                | "vertex-gemini"
        )
}

pub(crate) fn finalize_translation(
    provider: &Provider,
    attempt: &mut PreparedAttempt,
) -> Result<(), String> {
    let AttemptBody::Json(bytes) = &mut attempt.body else {
        return Err("Invalid translated Interactions body".into());
    };
    let mut value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let body = value
        .as_object_mut()
        .ok_or("Invalid translated Interactions object")?;
    // Existing channel overrides are applied by the wire builder. They must
    // not re-enable state or execution features excluded by this contract.
    for field in [
        "background",
        "previous_response_id",
        "previous_interaction_id",
        "conversation",
        "continuation_token",
        "tools",
        "tool_choice",
        "tool_config",
        "toolConfig",
        "functions",
        "function_call",
        "cachedContent",
        "modalities",
        "audio",
        "response_format",
    ] {
        if body.get(field).is_some_and(|value| {
            !value.is_null()
                && value != &Value::Bool(false)
                && !value.as_array().is_some_and(Vec::is_empty)
        }) {
            return Err(format!(
                "Incompatible channel setting for Interactions: {field}"
            ));
        }
        body.remove(field);
    }
    if let Some(config) = body.get("generationConfig") {
        if [
            "responseModalities",
            "responseSchema",
            "responseJsonSchema",
            "candidateCount",
        ]
        .iter()
        .any(|field| config.get(field).is_some())
            || config
                .get("responseMimeType")
                .is_some_and(|v| v != "text/plain")
        {
            return Err("Incompatible Gemini output settings for Interactions".into());
        }
    }
    let responses = attempt.adapter == ResponseAdapter::ResponsesToChat;
    if !responses && attempt.upstream_stream && !attempt.downstream_stream {
        return Err("This Interactions channel cannot return non-streaming output".into());
    }
    if body.get("n").is_some_and(|n| n != 1) {
        return Err("Interactions requires a single output candidate".into());
    }
    if responses {
        body.insert("store".into(), Value::Bool(false));
        if body.get("text").is_some_and(|text| {
            text.pointer("/format/type")
                .is_some_and(|kind| kind != "text")
        }) {
            return Err("Incompatible Responses output format for Interactions".into());
        }
    } else {
        body.remove("store");
        if attempt.adapter == ResponseAdapter::Passthrough && attempt.upstream_stream {
            body.insert("stream_options".into(), json!({"include_usage":true}));
        }
    }
    if attempt.adapter == ResponseAdapter::GeminiToChat
        && provider.engine.eq_ignore_ascii_case("gemini")
        && Url::parse(&attempt.url).is_ok_and(|url| url.host_str() == Some(GOOGLE_HOST))
    {
        let mut url = Url::parse(&attempt.url).map_err(|_| "Invalid Gemini upstream URL")?;
        let safe_query: Vec<_> = url
            .query_pairs()
            .filter(|(k, _)| k != "key")
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        url.set_query(None);
        if !safe_query.is_empty() {
            url.query_pairs_mut().extend_pairs(safe_query);
        }
        attempt.url = url.into();
        attempt.headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(&attempt.provider_key)
                .map_err(|_| "Invalid Gemini provider key")?,
        );
    }
    *bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    attempt.downstream_protocol = DownstreamProtocol::InteractionsCompat;
    Ok(())
}

pub(crate) fn upstream_url(provider: &Provider) -> Option<Url> {
    if !provider.engine.eq_ignore_ascii_case("gemini") {
        return None;
    }
    let mut url = Url::parse(&provider.base_url).ok()?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.has_host()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let path = url.path().trim_end_matches('/');
    let prefix = [
        "/v1beta/interactions",
        "/v1/interactions",
        "/v1beta/models",
        "/v1/models",
        "/v1beta",
        "/v1",
    ]
    .iter()
    .find_map(|suffix| path.strip_suffix(suffix))
    .unwrap_or(path);
    let version = match provider.preferences.get(VERSION) {
        Some(value) => match value.as_str()? {
            "v1" => "v1",
            "v1beta" => "v1beta",
            _ => return None,
        },
        // Only Google's documented Developer API origin gets a default.
        None if url.scheme() == "https"
            && url.host_str() == Some(GOOGLE_HOST)
            && url.port().is_none()
            && prefix.is_empty() =>
        {
            "v1"
        }
        None => return None,
    };
    url.set_path(&format!("{prefix}/{version}/interactions"));
    Some(url)
}

pub(crate) fn prepare(
    provider: &Provider,
    key: &str,
    model: &str,
    upstream_model: &str,
    payload: Option<&Value>,
    request_id: &str,
) -> Result<PreparedAttempt, String> {
    let url = upstream_url(provider).ok_or("Provider does not support Interactions")?;
    if !supports_model(provider, model) {
        return Err("Interactions requires a text generation model".into());
    }
    let mut payload = payload.cloned().ok_or("Interactions requires JSON")?;
    interactions::validate(&payload)?;
    let stream = payload["stream"].as_bool().unwrap_or(false);
    super::request::set_model(&mut payload, upstream_model)?;
    payload["store"] = Value::Bool(false);
    // Provider header settings may supply proxy-specific headers, but no caller
    // credentials, cookies, settlement metadata or arbitrary header forwarding.
    let mut headers = provider_headers(
        provider,
        key,
        &HeaderMap::new(),
        request_id,
        "gemini",
        stream,
        None,
    )?;
    for name in [
        "authorization",
        "x-api-key",
        "content-length",
        "content-encoding",
        "idempotency-key",
    ] {
        headers.remove(name);
    }
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert(
        "accept",
        HeaderValue::from_static(if stream {
            "text/event-stream"
        } else {
            "application/json"
        }),
    );
    headers.insert(
        "x-goog-api-key",
        HeaderValue::from_str(key).map_err(|_| "Invalid Gemini provider key")?,
    );
    Ok(PreparedAttempt {
        dispatch: None,
        method: Method::POST,
        url: url.into(),
        headers,
        body: AttemptBody::Json(
            serde_json::to_vec(&payload).map_err(|_| "Invalid Interactions JSON")?,
        ),
        adapter: ResponseAdapter::Passthrough,
        downstream_stream: stream,
        upstream_stream: stream,
        request_model: model.to_owned(),
        original_model: upstream_model.to_owned(),
        request_type: None,
        wire_model: Some(upstream_model.to_owned()),
        downstream_protocol: DownstreamProtocol::InteractionsNative,
        chat_stream_include_usage: false,
        prompt_compaction: false,
        provider_key: key.to_owned(),
        estimated_video_tokens: None,
    })
}
