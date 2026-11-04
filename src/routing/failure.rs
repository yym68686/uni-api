use crate::config::snapshot::Provider;
use crate::routing::types::ProviderFailurePolicy;
use serde_json::Value;
use url::Url;

pub(crate) fn remap_provider_status(status: u16, detail: &str) -> u16 {
    if status == 400 {
        return 400;
    }
    if [
        "string_above_max_length",
        "must be less than max_seq_len",
        "please reduce the length of the messages or completion",
        "request contains text fields that are too large.",
        "please reduce the length of either one, or use the",
        "exceeds the maximum number of tokens allowed",
    ]
    .iter()
    .any(|marker| detail.to_ascii_lowercase().contains(marker))
    {
        return 413;
    }
    if detail.contains("'reason': 'API_KEY_INVALID'")
        || detail.contains("API key not valid")
        || detail.contains("API key expired")
    {
        return 401;
    }
    if detail.contains("User location is not supported for the API use.") {
        return 403;
    }
    if is_provider_model_unavailable(status, detail) {
        return 503;
    }
    if detail.contains("<center><h1>400 Bad Request</h1></center>")
        || detail.contains("Provider API error: bad response status code 400")
    {
        return 502;
    }
    if detail.contains(
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
    ) {
        return 403;
    }
    if detail.contains("<head><title>413 Request Entity Too Large</title></head>") {
        return 429;
    }
    status
}

pub(crate) fn classify_provider_failure(
    original_status: u16,
    detail: &str,
    provider: Option<&Provider>,
    _endpoint: &str,
    auto_retry: bool,
    _upstream_model: Option<&str>,
) -> ProviderFailurePolicy {
    if original_status == 400 {
        return ProviderFailurePolicy {
            status: 400,
            retryable: false,
            request_scoped: true,
            provider_model_unavailable: false,
            force_quota_cooldown: false,
        };
    }
    let provider_model_unavailable = is_provider_model_unavailable(original_status, detail);
    let status = remap_provider_status(original_status, detail);
    let missing_persisted_item = status == 404 && is_missing_persisted_item_error(detail);
    let request_scoped = status == 413 || missing_persisted_item;
    let azure_request = status == 413 && provider.is_some_and(|p| is_azure_provider(&p.base_url));
    ProviderFailurePolicy {
        status,
        retryable: auto_retry && (!request_scoped || azure_request),
        request_scoped,
        provider_model_unavailable,
        force_quota_cooldown: false,
    }
}

pub(crate) fn is_provider_model_unavailable(status: u16, detail: &str) -> bool {
    if status != 404 {
        return false;
    }

    const CODES: &[&str] = &[
        "model_not_found",
        "model_not_supported",
        "model_unsupported",
        "unknown_provider",
        "unsupported_model",
    ];
    const MARKERS: &[&str] = &[
        "unknown provider for model",
        "no provider found for model",
        "no provider available for model",
        "model is not supported by this provider",
    ];

    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let parsed = serde_json::from_str::<Value>(&candidate).ok();
        let (code, message) = if let Some(payload) = parsed.as_ref() {
            let error = payload
                .get("error")
                .filter(|value| value.is_object())
                .or_else(|| payload.get("detail").filter(|value| value.is_object()));
            (
                error
                    .and_then(|value| value.get("code"))
                    .and_then(Value::as_str),
                error
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str),
            )
        } else {
            (None, Some(candidate.as_str()))
        };

        if code.is_some_and(|value| CODES.contains(&value.trim().to_ascii_lowercase().as_str())) {
            return true;
        }
        if message.is_some_and(|value| {
            let lower = value.to_ascii_lowercase();
            // Some providers report model availability as invalid_request_error
            // without a model-specific code. Match the whole message so echoed
            // input in an ordinary validation error does not trigger failover.
            lower.trim() == "this model is not available."
                || MARKERS.iter().any(|marker| lower.contains(marker))
        }) {
            return true;
        }

        let Some(nested) = message.filter(|value| value.trim_start().starts_with('{')) else {
            break;
        };
        candidate = nested.to_owned();
    }
    false
}

pub(crate) fn is_generic_upstream_error(error: &Value) -> bool {
    let matches = |field: &str, expected: &str| {
        error
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim().eq_ignore_ascii_case(expected))
    };
    // This relay's generic failure uses an empty param, not a field name.
    // Require both upstream_error markers; a Chinese validation message alone
    // must not turn a client error into provider failover.
    if matches("message", "上游错误") {
        return matches("type", "upstream_error")
            && matches("code", "upstream_error")
            && error
                .get("param")
                .is_none_or(|value| value.is_null() || value.as_str() == Some(""));
    }
    // Gateways sometimes replace all three fields with "upstream_error".
    // Only recognize the complete generic envelope; specific validation codes,
    // parameters and text echoed from input must keep their client-error policy.
    matches("type", "upstream_error")
        && (matches("message", "Upstream request failed") || matches("message", "upstream_error"))
        && (error.get("code").is_none_or(Value::is_null) || matches("code", "upstream_error"))
        && error.get("param").is_none_or(Value::is_null)
}

pub(crate) fn is_azure_provider(base_url: &str) -> bool {
    let Ok(url) = Url::parse(base_url) else {
        return false;
    };
    url.host_str() == Some("models.inference.ai.azure.com")
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
}

pub(crate) fn is_missing_persisted_item_error(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    lower.contains("invalid_request_error")
        && lower.contains("item with id")
        && lower.contains("not found")
        && lower.contains("items are not persisted when")
        && lower.contains("store")
}
