use super::request::provider_headers;
use super::types::{AttemptBody, DownstreamProtocol, PreparedAttempt, ResponseAdapter};
use crate::config::snapshot::Provider;
use crate::protocols::interactions;
use axum::http::{HeaderMap, HeaderValue, Method};
use serde_json::Value;
use url::Url;

const GOOGLE_HOST: &str = "generativelanguage.googleapis.com";
pub(crate) const VERSION: &str = "gemini_interactions_version";

pub(crate) fn supports_model(provider: &Provider, model: &str) -> bool {
    crate::providers::codex::models::is_conversational_model(model)
        && provider
            .models
            .get(model)
            .is_some_and(|upstream| upstream == model)
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

pub(crate) struct RequestEntity<'a> {
    pub(crate) metadata: &'a Value,
    pub(crate) body: AttemptBody,
    pub(crate) content_type: &'a str,
}

pub(crate) fn prepare(
    provider: &Provider,
    key: &str,
    model: &str,
    upstream_model: &str,
    entity: RequestEntity<'_>,
    request_id: &str,
) -> Result<PreparedAttempt, String> {
    let url = upstream_url(provider).ok_or("Provider does not support Interactions")?;
    if upstream_model != model || !supports_model(provider, model) {
        return Err("Interactions passthrough requires the configured upstream model name".into());
    }
    interactions::validate(entity.metadata)?;
    let stream = entity.metadata["stream"].as_bool().unwrap_or(false);
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
    headers.insert(
        "content-type",
        HeaderValue::from_str(entity.content_type)
            .map_err(|_| "Invalid Interactions Content-Type")?,
    );
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
        body: entity.body,
        adapter: ResponseAdapter::Passthrough,
        downstream_stream: stream,
        upstream_stream: stream,
        request_model: model.to_owned(),
        original_model: upstream_model.to_owned(),
        request_type: None,
        wire_model: Some(upstream_model.to_owned()),
        downstream_protocol: DownstreamProtocol::Native,
        chat_stream_include_usage: false,
        prompt_compaction: false,
        provider_key: key.to_owned(),
        estimated_video_tokens: None,
    })
}
