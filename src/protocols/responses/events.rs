use bytes::Bytes;
use serde_json::Value;

#[derive(Debug)]
pub(crate) enum Terminal {
    Completed,
    Incomplete,
    SemanticFailure { event_type: String, payload: Value },
}

pub(crate) fn responses_semantic_error(payload: &Value, event_type: &str) -> (u16, String) {
    let error = if event_type.eq_ignore_ascii_case("response.failed") {
        payload.pointer("/response/error")
    } else {
        payload.get("error")
    }
    .or_else(|| payload.get("error"));
    let detail = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| error.and_then(Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("Responses upstream returned a failure terminal")
        .chars()
        .take(4096)
        .collect::<String>();

    for candidate in [
        error.and_then(|value| value.get("status_code")),
        error.and_then(|value| value.get("status")),
        payload.get("status_code"),
        payload.get("status"),
        payload.pointer("/response/status_code"),
    ]
    .into_iter()
    .flatten()
    {
        let parsed = candidate
            .as_u64()
            .or_else(|| candidate.as_str()?.parse::<u64>().ok());
        if let Some(status @ 400..=599) = parsed {
            return (status as u16, detail);
        }
    }

    let code = error
        .and_then(|value| value.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let status = match code.as_str() {
        "account_deactivated"
        | "account_disabled"
        | "account_suspended"
        | "deactivated_workspace"
        | "permission_denied"
        | "user_deactivated"
        | "user_suspended" => Some(403),
        "authentication_error" | "incorrect_api_key_provided" | "invalid_api_key" => Some(401),
        "billing_hard_limit_reached" | "insufficient_quota" | "rate_limit_exceeded" => Some(429),
        "upstream_unavailable" => Some(503),
        "context_length_exceeded"
        | "invalid_request_error"
        | "invalid_type"
        | "model_not_priced"
        | "model_price_not_configured"
        | "model_pricing_not_configured"
        | "model_price_unconfigured"
        | "model_pricing_missing"
        | "unsupported_parameter" => Some(400),
        "model_not_found" | "not_found_error" => Some(404),
        _ => None,
    };
    if let Some(status) = status {
        return (status, detail);
    }

    let error_type = error
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let status = match error_type.as_str() {
        "authentication_error" => Some(401),
        "invalid_request_error" => Some(400),
        "not_found_error" => Some(404),
        "permission_error" => Some(403),
        "rate_limit_error" | "tokens" => Some(429),
        "upstream_unavailable" => Some(503),
        _ => None,
    };
    if let Some(status) = status {
        return (status, detail);
    }

    let message = detail.to_ascii_lowercase();
    let status = if message.contains("rate limit") || message.contains("too many requests") {
        429
    } else if [
        "context window",
        "context length",
        "maximum context",
        "too many tokens",
    ]
    .iter()
    .any(|marker| message.contains(marker))
    {
        400
    } else if message.contains("request entity too large") || message.contains("payload too large")
    {
        413
    } else if message.contains("invalid") || message.contains("unsupported") {
        400
    } else if message.contains("not found") {
        404
    } else if message.contains("permission") || message.contains("forbidden") {
        403
    } else if message.contains("auth")
        || message.contains("api key")
        || message.contains("unauthorized")
    {
        401
    } else {
        500
    };
    (status, detail)
}

pub(crate) fn validate_terminal(event_type: &str, payload: &Value) -> Result<(), String> {
    if !matches!(
        event_type,
        "error" | "response.completed" | "response.failed" | "response.incomplete"
    ) {
        return Ok(());
    }
    let object = payload
        .as_object()
        .ok_or_else(|| format!("Responses upstream {event_type} payload must be a JSON object"))?;
    if event_type.starts_with("response.") && !object.get("response").is_some_and(Value::is_object)
    {
        return Err(format!(
            "Responses upstream {event_type} payload is missing response"
        ));
    }
    if event_type == "error" && object.get("error").is_none_or(Value::is_null) {
        return Err("Responses upstream error payload is missing error".into());
    }
    Ok(())
}

pub(crate) fn semantic_failure(event_type: &str, payload: &Value) -> bool {
    if matches!(event_type, "error" | "response.failed") {
        return true;
    }
    let status = payload.get("status").and_then(Value::as_str);
    let response_status = payload
        .get("response")
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str);
    status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || response_status.is_some_and(|value| value.eq_ignore_ascii_case("failed"))
        || payload.get("error").is_some_and(Value::is_object)
}

pub(crate) fn is_canonical_keepalive(payload: &Value) -> bool {
    let Some(object) = payload.as_object() else {
        return false;
    };
    object.len() == 2
        && object.get("type").and_then(Value::as_str) == Some("keepalive")
        && object.get("sequence_number").and_then(Value::as_i64) == Some(0)
}

pub(crate) fn has_real_output(event_type: &str, payload: &Value) -> bool {
    if event_type.starts_with("response.") && event_type.ends_with(".delta") {
        return payload
            .get("delta")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
    }
    if matches!(
        event_type,
        "response.content_part.added" | "response.content_part.done"
    ) {
        return payload.get("part").is_some_and(part_has_text);
    }
    if event_type == "response.output_item.done" {
        return payload.get("item").is_some_and(item_has_output);
    }
    if event_type.starts_with("response.") && event_type.ends_with(".done") {
        return ["text", "refusal", "arguments"].iter().any(|field| {
            payload
                .get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        });
    }
    false
}

pub(crate) fn part_has_text(part: &Value) -> bool {
    ["text", "refusal"].iter().any(|field| {
        part.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    })
}

pub(crate) fn item_has_output(item: &Value) -> bool {
    if item
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|parts| parts.iter().any(part_has_text))
    {
        return true;
    }
    if !matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call" | "tool_call")
    ) {
        return false;
    }
    ["name", "arguments", "call_id"].iter().any(|field| {
        item.get(*field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    })
}

pub(crate) fn extract_usage(payload: &Value) -> Option<&Value> {
    payload
        .get("response")
        .and_then(|value| value.get("usage"))
        .filter(|value| value.is_object())
        .or_else(|| payload.get("usage").filter(|value| value.is_object()))
}

pub(crate) fn encode_event(event_type: &str, payload: &Value) -> Result<Bytes, String> {
    let data = serde_json::to_string(payload)
        .map_err(|error| format!("Responses event encoding failed: {error}"))?;
    Ok(Bytes::from(format!(
        "event: {event_type}\ndata: {data}\n\n"
    )))
}
