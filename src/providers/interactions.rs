use super::request::provider_headers;
use super::types::{AttemptBody, DownstreamProtocol, PreparedAttempt, ResponseAdapter};
use crate::config::snapshot::Provider;
use crate::protocols::interactions;
use crate::protocols::interactions::{Compat, Omitted};
use axum::http::{HeaderMap, HeaderValue, Method};
use serde_json::{json, Value};
use url::Url;

const GOOGLE_HOST: &str = "generativelanguage.googleapis.com";
pub(crate) const VERSION: &str = "gemini_interactions_version";
pub(crate) const THINKING_SUMMARIES: &str = "interactions_thinking_summaries";

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
                | "claude"
                | "vertex-claude"
                | "aws"
        )
}

pub(crate) fn translate_thinking_level(
    provider: &Provider,
    model: &str,
    source: &Value,
    chat: &mut Value,
    compat: &mut Compat,
) {
    let setting = &source["generation_config"]["thinking_level"];
    if setting.is_null() {
        return;
    }
    let level = setting.as_str().and_then(interactions::thinking_level);
    if let Some(level) = level.filter(|level| {
        if let Some((preset, _)) = super::gemini_effort::route(provider, model) {
            return preset.level(level).is_some();
        }
        let model = model.rsplit('/').next().unwrap_or(model);
        matches!(
            provider.engine.as_ref(),
            "gemini" | "vertex" | "vertex-gemini"
        ) && model.starts_with("gemini-3")
            && (matches!(*level, "low" | "high")
                || (model.contains("-flash") && matches!(*level, "minimal" | "medium")))
    }) {
        chat["reasoning_effort"] = json!(level);
    } else {
        compat.omit(Omitted::ThinkingLevel);
    }
}

fn object_at<'a>(
    object: &'a mut serde_json::Map<String, Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, Value> {
    let value = object.entry(key).or_insert_with(|| json!({}));
    if !value.is_object() {
        *value = json!({});
    }
    value.as_object_mut().expect("object normalized above")
}

fn supports_summaries(provider: &Provider, attempt: &PreparedAttempt) -> bool {
    if !matches!(
        attempt.adapter,
        ResponseAdapter::GeminiToChat | ResponseAdapter::ResponsesToChat
    ) {
        return false;
    }
    if let Some(enabled) = provider
        .preferences
        .get(THINKING_SUMMARIES)
        .and_then(Value::as_bool)
    {
        return enabled;
    }
    let model = attempt
        .wire_model
        .as_deref()
        .unwrap_or(&attempt.original_model);
    let leaf = model.rsplit('/').next().unwrap_or(model);
    if attempt.adapter == ResponseAdapter::GeminiToChat {
        return [
            "gemini-2.5-",
            "gemini-3-",
            "gemini-3.1-",
            "gemini-3.5-",
            "gemini-3.8-",
            "gemini-4-",
        ]
        .into_iter()
        .any(|prefix| leaf.starts_with(prefix));
    }
    // An OpenAI-shaped HTTP endpoint alone is not evidence that it implements
    // summary requests. Allow custom gateways to opt in with a channel setting.
    Url::parse(&attempt.url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| matches!(host.as_str(), "api.openai.com" | "chatgpt.com"))
        && ["gpt-5", "o3", "o4-mini"].into_iter().any(|prefix| {
            leaf == prefix
                || leaf
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with(['-', '.']))
        })
}

pub(crate) fn finalize_translation(
    provider: &Provider,
    source: &Value,
    mut compat: Compat,
    attempt: &mut PreparedAttempt,
) -> Result<(), String> {
    let summary_setting = &source["generation_config"]["thinking_summaries"];
    compat.summaries = summary_setting == "auto" && supports_summaries(provider, attempt);
    if !summary_setting.is_null() && summary_setting != "none" && !compat.summaries {
        compat.omit(Omitted::ThinkingSummaries);
    }
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
        "parallel_tool_calls",
        "plugins",
        "functions",
        "function_call",
        "cachedContent",
        "cached_content",
        "modalities",
        "audio",
        "response_format",
        "service_tier",
        "serviceTier",
        "include",
    ] {
        if body.get(field).is_some_and(|value| {
            !value.is_null()
                && value != &Value::Bool(false)
                && !value.as_array().is_some_and(Vec::is_empty)
        }) {
            compat.omit(Omitted::ChannelDefaults);
        }
        body.remove(field);
    }
    let responses = attempt.adapter == ResponseAdapter::ResponsesToChat;
    if body.get("n").is_some_and(|n| n != 1) {
        body.remove("n");
        compat.omit(Omitted::ChannelDefaults);
    }
    if responses {
        body.insert("store".into(), Value::Bool(false));
        if let Some(text) = body.get_mut("text").and_then(Value::as_object_mut) {
            if text
                .get("format")
                .is_some_and(|format| format["type"] != "text")
            {
                text.remove("format");
                compat.omit(Omitted::ChannelDefaults);
            }
        }
        if let Some(reasoning) = body.get_mut("reasoning").and_then(Value::as_object_mut) {
            reasoning.remove("summary");
            reasoning.remove("generate_summary");
            if reasoning.is_empty() {
                body.remove("reasoning");
            }
        }
        if compat.summaries {
            object_at(body, "reasoning").insert("summary".into(), json!("auto"));
        }
    } else {
        if attempt.adapter == ResponseAdapter::Passthrough
            && Url::parse(&attempt.url)
                .ok()
                .is_some_and(|url| url.host_str() == Some("api.openai.com"))
        {
            body.insert("store".into(), Value::Bool(false));
        } else if body.remove("store").is_some_and(|v| v == true) {
            compat.omit(Omitted::ChannelDefaults);
        }
        if attempt.adapter == ResponseAdapter::Passthrough && attempt.upstream_stream {
            body.insert("stream_options".into(), json!({"include_usage":true}));
        }
    }
    if attempt.adapter == ResponseAdapter::GeminiToChat {
        for key in ["generationConfig", "generation_config"] {
            if let Some(config) = body.get_mut(key).and_then(Value::as_object_mut) {
                for field in [
                    "responseModalities",
                    "response_modalities",
                    "responseSchema",
                    "response_schema",
                    "responseJsonSchema",
                    "response_json_schema",
                    "candidateCount",
                    "candidate_count",
                    "responseMimeType",
                    "response_mime_type",
                ] {
                    if config.remove(field).is_some() {
                        compat.omit(Omitted::ChannelDefaults);
                    }
                }
                for key in ["thinkingConfig", "thinking_config"] {
                    if let Some(thinking) = config.get_mut(key).and_then(Value::as_object_mut) {
                        thinking.remove("includeThoughts");
                        thinking.remove("include_thoughts");
                        if thinking.is_empty() {
                            config.remove(key);
                        }
                    }
                }
            }
        }
        if compat.summaries {
            let config = object_at(body, "generationConfig");
            object_at(config, "thinkingConfig").insert("includeThoughts".into(), Value::Bool(true));
        }
    }
    if matches!(
        attempt.adapter,
        ResponseAdapter::ClaudeToChat | ResponseAdapter::AwsToChat
    ) {
        if let Some(output) = body.get_mut("output_config").and_then(Value::as_object_mut) {
            if output.remove("format").is_some() {
                compat.omit(Omitted::ChannelDefaults);
            }
        }
        if body
            .get("temperature")
            .and_then(Value::as_f64)
            .is_some_and(|v| v > 1.0)
        {
            body.remove("temperature");
        }
    }
    if let Some(limit) =
        interactions::max_output_limit(&source["generation_config"]["max_output_tokens"])
    {
        let clamp = |root: &mut serde_json::Map<String, Value>, key: &str| {
            let selected = root.get(key).and_then(interactions::max_output_limit);
            root.insert(key.into(), json!(selected.unwrap_or(limit).min(limit)));
        };
        if attempt.adapter == ResponseAdapter::GeminiToChat {
            clamp(object_at(body, "generationConfig"), "maxOutputTokens");
        } else if responses {
            clamp(body, "max_output_tokens");
        } else if attempt.adapter == ResponseAdapter::Passthrough
            && body.contains_key("max_completion_tokens")
        {
            body.remove("max_tokens");
            clamp(body, "max_completion_tokens");
        } else {
            clamp(body, "max_tokens");
        }
    }
    for (field, gemini) in [("temperature", "temperature"), ("top_p", "topP")] {
        let requested = &source["generation_config"][field];
        if !requested.is_null()
            && if attempt.adapter == ResponseAdapter::GeminiToChat {
                body.get("generationConfig")
                    .is_none_or(|config| &config[gemini] != requested)
            } else {
                body.get(field) != Some(requested)
            }
        {
            compat.omit(Omitted::GenerationConfig);
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
    attempt.downstream_protocol = DownstreamProtocol::InteractionsCompat(compat);
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
    let stream = payload["stream"].as_bool().unwrap_or(false);
    super::request::set_model(&mut payload, upstream_model)?;
    payload["store"] = Value::Bool(false);
    let mut compat = Compat::default();
    if let Some(root) = payload.as_object_mut() {
        if root
            .remove("service_tier")
            .is_some_and(|value| !value.is_null())
        {
            compat.omit(Omitted::ServiceTier);
        }
        if root
            .remove("tools")
            .is_some_and(|v| !v.is_null() && !v.as_array().is_some_and(Vec::is_empty))
        {
            compat.omit(Omitted::Tools);
        }
        root.remove("tool_choice");
        if let Some(config) = root
            .get_mut("generation_config")
            .and_then(Value::as_object_mut)
        {
            config.remove("tool_choice");
        } else if root
            .remove("generation_config")
            .is_some_and(|v| !v.is_null())
        {
            compat.omit(Omitted::GenerationConfig);
        }
        if root.get("stream").is_some_and(|value| !value.is_boolean()) {
            root.remove("stream");
            compat.omit(Omitted::Extensions);
        }
        if root
            .get("background")
            .is_some_and(|value| !value.is_boolean())
        {
            root.remove("background");
            compat.omit(Omitted::Extensions);
        }
    }
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
        downstream_protocol: DownstreamProtocol::InteractionsNative(compat),
        chat_stream_include_usage: false,
        prompt_compaction: false,
        provider_key: key.to_owned(),
        estimated_video_tokens: None,
    })
}
