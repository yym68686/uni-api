use crate::config::snapshot::Provider;
use crate::protocols::conversion::chat_to_cloudflare;
use crate::protocols::conversion::chat_to_cohere;
use crate::protocols::conversion::chat_to_doubao_translation;
use crate::protocols::conversion::chat_to_responses;
use crate::protocols::conversion::extract_translation_text;
use crate::protocols::conversion::normalize_jina_embedding;
use crate::protocols::conversion::openai_tts_to_minimax;
use crate::protocols::conversion::responses_to_chat_request;
use crate::providers::anthropic::chat_to_claude;
use crate::providers::cloud::aws_bedrock_url;
use crate::providers::cloud::azure_chat_url;
use crate::providers::cloud::cloudflare_url;
use crate::providers::cloud::databricks_chat_url;
use crate::providers::cloud::normalize_azure_token_limit;
use crate::providers::cloud::sign_aws_request;
use crate::providers::cloud::vertex_claude_url;
use crate::providers::codex::CODEX_USER_AGENT;
use crate::providers::endpoints::endpoint_url;
use crate::providers::endpoints::messages_url;
use crate::providers::endpoints::responses_url;
use crate::providers::endpoints::typesafe_endpoint_url;
use crate::providers::gemini::chat_to_gemini;
use crate::providers::gemini::gemini_url;
use crate::providers::overrides::apply_overrides;
use crate::providers::types::AttemptBody;
use crate::providers::types::DownstreamProtocol;
use crate::providers::types::PreparedAttempt;
use crate::providers::types::PreparedInput;
use crate::providers::types::ResponseAdapter;
use crate::providers::vertex::vertex_gemini_url;
use crate::providers::video::callxyq_video_payload;
use crate::providers::video::callxyq_video_url;
use crate::providers::video::content_generation_to_lingjing;
use crate::providers::video::estimate_video_tokens;
use crate::providers::video::lingjing_url;
use crate::providers::video::value_text;
use crate::transport::multipart::multipart_output_boundary;
use crate::transport::spool::SpoolObservation;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Uri};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{json, Map, Value};
use url::Url;

pub(crate) const ALPHA_SEARCH_ENDPOINT: &str = "/v1/alpha/search";

// Pure request preparation for the administrator preview. No send, OAuth
// refresh, key scheduling or model invocation occurs here.
pub(crate) fn preview_channel_request(
    provider: &Provider,
    model: &str,
    upstream: &str,
    endpoint: &str,
    payload: Value,
) -> Result<Value, String> {
    let uri: Uri = endpoint
        .parse()
        .map_err(|_| "invalid preview endpoint".to_owned())?;
    let input = PreparedInput {
        payload: Some(payload),
        replay: None,
        observation: SpoolObservation::default(),
        default_model: String::new(),
        content_type: "application/json".into(),
    };
    let attempt = build_attempt(
        provider,
        "preview-key",
        model,
        upstream,
        &Method::POST,
        &uri,
        endpoint,
        &HeaderMap::new(),
        &input,
        "preview",
    )?;
    let body = match attempt.body {
        AttemptBody::Json(bytes) => serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
        _ => Value::Null,
    };
    let mut url = Url::parse(&attempt.url).map_err(|_| "invalid prepared endpoint".to_owned())?;
    url.set_query(None);
    Ok(
        json!({"url":url.to_string(),"method":attempt.method.as_str(),"header_names":attempt.headers.keys().map(|h|h.as_str()).collect::<Vec<_>>(),"body":body,"stream":attempt.upstream_stream}),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_attempt(
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
    let is_systemone = path == "/v1/systemone";
    let engine = provider.engine.trim().to_ascii_lowercase();
    if !crate::routing::filters::provider_accepts_endpoint(provider, path) {
        return Err("Provider does not support this endpoint".into());
    }
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
        && !is_systemone
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
            dispatch: None,
            method: method.clone(),
            url,
            headers,
            body,
            adapter: ResponseAdapter::Passthrough,
            downstream_stream: false,
            upstream_stream: false,
            request_model: request_model.to_owned(),
            original_model: original_model.to_owned(),
            wire_model: None,
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
        "typesafe" => {
            set_model(&mut payload, original_model)?;
            (
                typesafe_endpoint_url(provider.base_url.as_ref())?,
                ResponseAdapter::Passthrough,
                false,
            )
        }
        "codex" if wire_path == "/v1/chat/completions" => {
            payload = chat_to_responses(&payload, original_model)?;
            (
                responses_url(provider.base_url.as_ref()),
                ResponseAdapter::ResponsesToChat,
                provider_stream,
            )
        }
        "gpt" | "openrouter" | "requesty" | "azure" | "azure-databricks" | "cloudflare"
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
        _ if path == "/v1/video/tasks"
            && (provider.name.eq_ignore_ascii_case("callxyq")
                || provider
                    .base_url
                    .to_ascii_lowercase()
                    .contains("callxyq.xyz")) =>
        {
            payload = callxyq_video_payload(&payload, original_model)?;
            (
                callxyq_video_url(provider, None, true)?,
                ResponseAdapter::CallxyqVideo,
                false,
            )
        }
        _ if path.starts_with("/v1/video/tasks/")
            && (provider.name.eq_ignore_ascii_case("callxyq")
                || provider
                    .base_url
                    .to_ascii_lowercase()
                    .contains("callxyq.xyz")) =>
        {
            let task_id = path.trim_start_matches("/v1/video/tasks/");
            (
                callxyq_video_url(provider, Some(task_id), false)?,
                ResponseAdapter::CallxyqVideo,
                false,
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
        ResponseAdapter::GeminiToChat
            | ResponseAdapter::AwsToChat
            | ResponseAdapter::LingjingVideo
            | ResponseAdapter::CallxyqVideo
    ) && engine != "vertex-claude"
        && !is_alpha_search
        && !is_systemone
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
    // OpenAI-compatible overrides may explicitly disable upstream streaming.
    // Decode the wire format we actually requested before adapting the result.
    let upstream_stream = if matches!(
        adapter,
        ResponseAdapter::ResponsesToChat | ResponseAdapter::Passthrough
    ) {
        payload
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(upstream_stream)
    } else {
        upstream_stream
    };
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
        dispatch: None,
        method: outgoing_method,
        url,
        headers,
        body,
        adapter,
        downstream_stream,
        upstream_stream,
        request_model: request_model.to_owned(),
        original_model: original_model.to_owned(),
        wire_model: payload
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        downstream_protocol,
        chat_stream_include_usage,
        provider_key: provider_key.to_owned(),
        estimated_video_tokens,
    })
}

pub(crate) fn search_query(payload: &Value) -> Result<String, String> {
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

pub(crate) fn sanitize_alpha_search_payload(root: &mut Map<String, Value>) {
    for field in [
        "store",
        "stream",
        "prompt_cache_key",
        "prompt_cache_retention",
    ] {
        root.remove(field);
    }
}

pub(crate) fn apply_alpha_search_headers(
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

pub(crate) fn provider_headers(
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
    if engine == "requesty" && provider.base_url.contains("requesty.ai") {
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
    if let Some(value) = incoming.get("x-oaix-settlement-nonce") {
        headers.insert("x-oaix-settlement-nonce", value.clone());
    }
    Ok(headers)
}

pub(crate) fn set_model(payload: &mut Value, model: &str) -> Result<(), String> {
    payload
        .as_object_mut()
        .ok_or_else(|| "request body must be a JSON object".to_owned())?
        .insert("model".into(), Value::String(model.to_owned()));
    Ok(())
}
