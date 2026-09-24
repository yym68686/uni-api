use crate::config::snapshot::Provider;
use crate::routing::types::ProviderFailurePolicy;
use serde_json::Value;
use url::Url;

pub(crate) fn remap_provider_status(status: u16, detail: &str) -> u16 {
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
    if is_provider_request_processing_failure(status, detail) {
        return 502;
    }
    if detail.contains("<center><h1>400 Bad Request</h1></center>")
        || detail.contains("Provider API error: bad response status code 400")
        || status == 400
            && (is_model_pricing_unconfigured(detail)
                || is_provider_minimum_input_restriction(detail))
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
    endpoint: &str,
    auto_retry: bool,
    upstream_model: Option<&str>,
) -> ProviderFailurePolicy {
    let provider_model_unavailable = is_provider_model_unavailable(original_status, detail);
    let status = if original_status == 400
        && upstream_model.is_some_and(|model| is_provider_error_model_mismatch(detail, model))
    {
        502
    } else {
        remap_provider_status(original_status, detail)
    };
    let codex_model_unsupported = status == 400
        && matches!(endpoint, "/v1/responses" | "/v1/responses/compact")
        && provider.is_some_and(|provider| provider.engine.eq_ignore_ascii_case("codex"))
        && detail
            .to_ascii_lowercase()
            .contains("model is not supported when using codex with a chatgpt account");
    let missing_persisted_item = status == 404 && is_missing_persisted_item_error(detail);
    let request_scoped = matches!(status, 400 | 413) || missing_persisted_item;
    let azure_request = matches!(status, 400 | 413)
        && provider.is_some_and(|provider| is_azure_provider(&provider.base_url));
    ProviderFailurePolicy {
        status,
        retryable: auto_retry && (!request_scoped || codex_model_unsupported || azure_request),
        request_scoped,
        provider_model_unavailable,
        force_quota_cooldown: codex_model_unsupported,
    }
}

pub(crate) fn is_provider_error_model_mismatch(detail: &str, upstream_model: &str) -> bool {
    let upstream_model = upstream_model.trim();
    if upstream_model.is_empty() {
        return false;
    }
    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let Ok(payload) = serde_json::from_str::<Value>(&candidate) else {
            return false;
        };
        let Some(error) = payload
            .get("error")
            .or_else(|| payload.get("detail"))
            .filter(|error| error.is_object())
        else {
            return false;
        };
        let Some(message) = error.get("message").and_then(Value::as_str) else {
            return false;
        };
        if message.trim_start().starts_with('{') {
            candidate = message.to_owned();
            continue;
        }
        if error.get("code").and_then(Value::as_str) != Some("unsupported_value")
            || error.get("type").and_then(Value::as_str) != Some("invalid_request_error")
            || error
                .get("param")
                .and_then(Value::as_str)
                .is_none_or(|param| param.trim().is_empty())
        {
            return false;
        }
        // Only this explicit model-specific validation message identifies a
        // mismatched backend. Never search echoed input or arbitrary prose.
        let Some(rest) = message.strip_prefix("Unsupported value: '") else {
            return false;
        };
        let Some((value, rest)) = rest.split_once("' is not supported with the '") else {
            return false;
        };
        let Some((model, supported)) = rest.split_once("' model. Supported values are: ") else {
            return false;
        };
        return !value.is_empty()
            && !model.is_empty()
            && model
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_.:/".contains(&c))
            && supported.starts_with('\'')
            && supported.ends_with("'.")
            && !model.eq_ignore_ascii_case(upstream_model);
    }
    false
}

pub(crate) fn is_provider_model_unavailable(status: u16, detail: &str) -> bool {
    if !matches!(status, 400 | 404) {
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

pub(crate) fn is_provider_request_processing_failure(status: u16, detail: &str) -> bool {
    if status != 400 {
        return false;
    }

    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let parsed = serde_json::from_str::<Value>(&candidate).ok();
        if let Some(error) = parsed.as_ref().and_then(|payload| {
            payload
                .get("error")
                .filter(|value| value.is_object())
                .or_else(|| payload.get("detail").filter(|value| value.is_object()))
        }) {
            // Some gateways hide the original failure behind this generic 400.
            // Require the explicit upstream type and whole message; a specific
            // validation code or echoed input must keep its client-error policy.
            let matches = |field: &str, expected: &str| {
                error
                    .get(field)
                    .and_then(Value::as_str)
                    .is_some_and(|value| value.trim().eq_ignore_ascii_case(expected))
            };
            let generic_code =
                error.get("code").is_none_or(Value::is_null) || matches("code", "upstream_error");
            if matches("type", "upstream_error")
                && matches("message", "Upstream request failed")
                && generic_code
            {
                return true;
            }
        }
        let message = match parsed.as_ref() {
            Some(payload) => payload
                .pointer("/error/message")
                .or_else(|| payload.pointer("/detail/message"))
                .or_else(|| payload.get("message"))
                .or_else(|| payload.get("error"))
                .or_else(|| payload.get("detail"))
                .unwrap_or(payload)
                .as_str(),
            None => Some(candidate.as_str()),
        };
        let Some(message) = message else {
            return false;
        };
        if message.trim_start().starts_with('{') {
            candidate = message.to_owned();
            continue;
        }
        return message
            .trim()
            .eq_ignore_ascii_case("The upstream service could not process this request.");
    }
    false
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

pub(crate) fn is_model_pricing_unconfigured(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    [
        "model_not_priced",
        "model_price_not_configured",
        "model_pricing_not_configured",
        "model_price_unconfigured",
        "model_pricing_missing",
        "has not been priced by the administrator",
        "has not been priced by administrator",
        "price has not been configured by the administrator",
        "pricing has not been configured by the administrator",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
        || lower
            .split_whitespace()
            .collect::<String>()
            .contains("价格尚未由管理员配置")
}

pub(crate) fn is_provider_minimum_input_restriction(detail: &str) -> bool {
    // A channel key's minimum-input policy is not a malformed client request.
    // Read only error messages (including JSON-escaped/wrapped messages), never
    // echoed request fields. Neither the provider nor the numeric limit matters.
    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let parsed = serde_json::from_str::<Value>(&candidate).ok();
        let message = match parsed.as_ref() {
            Some(payload) => payload
                .pointer("/error/message")
                .or_else(|| payload.pointer("/detail/message"))
                .or_else(|| payload.get("message"))
                .or_else(|| payload.get("error"))
                .or_else(|| payload.get("detail"))
                .unwrap_or(payload)
                .as_str(),
            None => Some(candidate.as_str()),
        };
        let Some(message) = message else {
            return false;
        };
        if message.trim_start().starts_with('{') {
            candidate = message.to_owned();
            continue;
        }
        let compact = message
            .split_whitespace()
            .collect::<String>()
            .to_ascii_lowercase();
        return [
            ("该令牌不接受输入少于", "token的请求"),
            ("thiskeydoesnotacceptrequestswithfewerthan", "inputtokens"),
        ]
        .iter()
        .any(|(prefix, suffix)| {
            compact.split_once(prefix).is_some_and(|(_, tail)| {
                let after_number = tail.trim_start_matches(|c: char| c.is_ascii_digit());
                after_number.len() < tail.len() && after_number.starts_with(suffix)
            })
        });
    }
    false
}

pub(crate) fn is_missing_persisted_item_error(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    lower.contains("invalid_request_error")
        && lower.contains("item with id")
        && lower.contains("not found")
        && lower.contains("items are not persisted when")
        && lower.contains("store")
}
