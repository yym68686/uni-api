use crate::config::snapshot::Provider;
use serde_json::{Map, Value};

pub(crate) fn provider_accepts_endpoint(provider: &Provider, endpoint: &str) -> bool {
    let endpoint = endpoint.trim_end_matches('/');
    if endpoint == "all" {
        return true;
    }
    let typesafe = provider.engine.trim().eq_ignore_ascii_case("typesafe");
    typesafe == (endpoint == "/v1/systemone")
}

pub(crate) fn provider_accepts_body(provider: &Provider, bytes: u64) -> bool {
    let Some(raw) = provider.preferences.get("max_request_body_bytes") else {
        return true;
    };
    parse_byte_limit(raw).is_none_or(|limit| bytes <= limit)
}

pub(crate) fn detect_request_type(payload: &Map<String, Value>) -> Option<&'static str> {
    let is_compaction = payload
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
        });
    is_compaction.then_some("compaction")
}

pub(crate) fn provider_accepts_request_type(
    provider: &Provider,
    request_type: Option<&str>,
) -> bool {
    if !provider.only_request_types.is_empty()
        && !request_type.is_some_and(|value| {
            provider
                .only_request_types
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(value))
        })
    {
        return false;
    }
    !request_type.is_some_and(|value| {
        provider
            .excluded_request_types
            .iter()
            .any(|excluded| excluded.eq_ignore_ascii_case(value))
    })
}

pub(crate) fn request_reasoning_effort(payload: &Map<String, Value>) -> Option<String> {
    payload
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| {
            payload
                .get("reasoning")
                .and_then(Value::as_object)
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(crate) fn provider_accepts_request_rules(
    provider: &Provider,
    endpoint: &str,
    request_model: &str,
    reasoning_effort: Option<&str>,
    request_type: Option<&str>,
) -> bool {
    let upstream_model = provider
        .models
        .get(request_model)
        .map(String::as_str)
        .unwrap_or(request_model);
    !provider.excluded_request_rules.iter().any(|rule| {
        exclude_request_rule_matches(
            rule,
            endpoint,
            request_model,
            upstream_model,
            reasoning_effort,
            request_type,
        )
    })
}

pub(crate) fn exclude_request_rule_matches(
    rule: &Value,
    endpoint: &str,
    request_model: &str,
    upstream_model: &str,
    reasoning_effort: Option<&str>,
    request_type: Option<&str>,
) -> bool {
    let Some(condition) = rule.get("match").and_then(Value::as_object) else {
        return false;
    };
    if condition.is_empty() {
        return false;
    }
    condition.iter().all(|(key, expected)| match key.as_str() {
        "endpoint" => request_rule_value_matches(expected, Some(endpoint), true),
        "request_model" => request_rule_value_matches(expected, Some(request_model), false),
        "upstream_model" => request_rule_value_matches(expected, Some(upstream_model), false),
        "reasoning_effort" => request_rule_value_matches(expected, reasoning_effort, false),
        "request_type" => request_rule_value_matches(expected, request_type, false),
        _ => false,
    })
}

pub(crate) fn request_rule_value_matches(
    expected: &Value,
    actual: Option<&str>,
    endpoint: bool,
) -> bool {
    if let Some(values) = expected.as_array() {
        return values
            .iter()
            .any(|value| request_rule_value_matches(value, actual, endpoint));
    }
    let (Some(expected), Some(actual)) = (expected.as_str(), actual) else {
        return false;
    };
    let normalize = |value: &str| {
        let mut value = value.trim().trim_end_matches('/').to_ascii_lowercase();
        if endpoint && !value.is_empty() && !value.starts_with('/') {
            value.insert(0, '/');
        }
        value
    };
    let expected = normalize(expected);
    let actual = normalize(actual);
    if expected.is_empty() || actual.is_empty() {
        return false;
    }
    expected == "*"
        || expected == actual
        || expected
            .strip_suffix('*')
            .is_some_and(|prefix| actual.starts_with(prefix))
}

pub(crate) fn parse_byte_limit(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return (value > 0).then_some(value);
    }
    let raw = value.as_str()?.trim().to_ascii_lowercase().replace('_', "");
    let split = raw
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(raw.len());
    let number = raw[..split].trim().parse::<f64>().ok()?;
    let unit = raw[split..].trim();
    let multiplier = match unit {
        "" | "b" | "byte" | "bytes" => 1.0,
        "k" | "kb" => 1_000.0,
        "ki" | "kib" => 1_024.0,
        "m" | "mb" => 1_000_000.0,
        "mi" | "mib" => 1_048_576.0,
        "g" | "gb" => 1_000_000_000.0,
        "gi" | "gib" => 1_073_741_824.0,
        _ => return None,
    };
    let bytes = (number * multiplier) as u64;
    (bytes > 0).then_some(bytes)
}
