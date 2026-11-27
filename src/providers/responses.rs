//! Responses wire construction shared by native, compatibility execution and preview.
use crate::config::snapshot::{endpoint_values, Provider};
use crate::protocols::responses::item_ids::normalize_response_root;
use crate::providers::codex::CODEX_USER_AGENT;
use crate::providers::overrides::apply_overrides;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub(crate) const COMPACTION_MODE: &str = "compaction_mode";

pub(crate) fn validate_compaction_mode(value: &Value) -> Result<(), String> {
    let valid = |v: &Value| matches!(v.as_str(), Some("auto" | "native" | "prompt"));
    if valid(value)
        || value.as_object().is_some_and(|scope| {
            scope
                .keys()
                .all(|key| matches!(key.as_str(), "default" | "models"))
                && scope.get("default").is_none_or(valid)
                && scope.get("models").is_none_or(|v| {
                    v.as_object().is_some_and(|models| {
                        models
                            .iter()
                            .all(|(model, mode)| !model.trim().is_empty() && valid(mode))
                    })
                })
        })
    {
        Ok(())
    } else {
        Err("Invalid channel setting: /preferences/compaction_mode".into())
    }
}

pub(crate) fn prompt_compaction(provider: &Provider, model: &str, upstream: &str) -> bool {
    if !super::stream_mode::responses_wire(&provider.engine, &provider.base_url) {
        return true;
    }
    let preference = provider.preferences.get(COMPACTION_MODE).and_then(|value| {
        value
            .get("models")
            .and_then(|m| m.get(model).or_else(|| m.get(upstream)))
            .or_else(|| value.get("default"))
            .unwrap_or(value)
            .as_str()
    });
    match preference {
        Some("native") => false,
        Some("prompt") => true,
        // A Responses URL describes the wire, not the model's ability to create
        // OpenAI compaction. Resolve the actual channel model, not the public alias.
        _ => {
            let leaf = upstream
                .rsplit('/')
                .next()
                .unwrap_or(upstream)
                .to_ascii_lowercase();
            let leaf = leaf.strip_prefix("openai.").unwrap_or(&leaf);
            !["gpt-", "codex-", "chatgpt-"]
                .iter()
                .any(|prefix| leaf.starts_with(prefix))
                && !["o1", "o3", "o4"].iter().any(|model| {
                    leaf == *model
                        || leaf
                            .strip_prefix(model)
                            .is_some_and(|suffix| suffix.starts_with('-'))
                })
        }
    }
}

pub(crate) fn compile_payload(
    payload: &mut Value,
    provider: &Provider,
    request_model: &str,
    original_model: &str,
    engine: &str,
    wants_compact: bool,
) -> Result<(), String> {
    let gemini_effort = if !wants_compact {
        super::gemini_effort::route(provider, original_model)
            .map(|(preset, _)| super::gemini_effort::selected(preset, original_model, payload))
            .transpose()?
            .flatten()
    } else {
        None
    };
    let root = payload
        .as_object_mut()
        .ok_or_else(|| "native Responses payload is not an object".to_owned())?;
    root.insert("model".into(), Value::String(original_model.to_owned()));
    if engine == "codex" {
        for key in [
            "previous_response_id",
            "prompt_cache_retention",
            "safety_identifier",
        ] {
            root.remove(key);
        }
        root.entry("instructions")
            .or_insert(Value::String(String::new()));
    }
    apply_overrides(root, provider, request_model);
    if let Some(effort) = gemini_effort {
        super::gemini_effort::Wire::Responses.apply(root, effort);
    }
    if !wants_compact {
        crate::routing::service_tier::strip_for_standard(root, provider);
        if let Some(tier) = provider
            .preferences
            .get(crate::routing::service_tier::ROUTED)
            .filter(|tier| tier.is_string())
        {
            root.insert("service_tier".into(), tier.clone());
            root.insert("model".into(), Value::String(original_model.to_owned()));
        }
    }
    if engine == "codex" {
        strip_codex_fields(root);
        if wants_compact {
            root.remove("store");
        }
    }
    if normalization_enabled(provider, request_model, original_model) {
        normalize_response_root(root)?;
    }
    Ok(())
}

pub(crate) fn strip_codex_fields(root: &mut Map<String, Value>) {
    for key in [
        "max_output_tokens",
        "response_format",
        "top_p",
        "truncation",
    ] {
        root.remove(key);
    }
    root.remove("cache_control");
    root.remove("reasoning_content");
    for value in root.values_mut() {
        strip_key_recursive(value, "cache_control");
        strip_key_recursive(value, "reasoning_content");
    }
    if let Some(input) = root.get_mut("input").and_then(Value::as_array_mut) {
        for item in input.iter_mut().filter_map(Value::as_object_mut) {
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                item.remove("id");
            }
            if item.get("type").and_then(Value::as_str) == Some("message") {
                item.remove("reasoning");
                item.remove("reasoning_content");
            }
        }
    }
}

pub(crate) fn provider_stream_override(provider: &Provider) -> Option<bool> {
    provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get("stream"))
        .and_then(Value::as_bool)
}

pub(crate) fn strip_key_recursive(value: &mut Value, key: &str) {
    match value {
        Value::Object(object) => {
            object.remove(key);
            for child in object.values_mut() {
                strip_key_recursive(child, key);
            }
        }
        Value::Array(items) => {
            for child in items {
                strip_key_recursive(child, key);
            }
        }
        _ => {}
    }
}

pub(crate) fn normalization_enabled(
    provider: &Provider,
    request_model: &str,
    original_model: &str,
) -> bool {
    match provider
        .preferences
        .get("normalize_responses_custom_tool_call_ids")
    {
        Some(Value::Bool(value)) => *value,
        Some(Value::Array(models)) => models.iter().any(|model| {
            model.as_str().is_some_and(|model| {
                model == "*" || model == request_model || model == original_model
            })
        }),
        Some(_) => false,
        None => provider.engine.as_ref() == "codex",
    }
}

pub(crate) fn build_headers(
    incoming: &HeaderMap,
    provider: &Provider,
    provider_key: &str,
    engine: &str,
    stream: bool,
    request_id: &str,
    attempt_id: &str,
) -> Result<HashMap<String, String>, String> {
    let mut headers = HashMap::from([
        ("Accept-Encoding".into(), "identity".into()),
        ("Content-Type".into(), "application/json".into()),
        ("Authorization".into(), format!("Bearer {provider_key}")),
        ("x-request-id".into(), request_id.to_owned()),
        ("x-uni-api-ember-request-id".into(), request_id.to_owned()),
        ("x-caller-request-id".into(), request_id.to_owned()),
        ("x-caller-app".into(), "uni-api-ember".into()),
    ]);
    if engine == "codex" {
        headers.insert(
            "Openai-Beta".into(),
            header_or(incoming, "openai-beta", "responses=experimental"),
        );
        headers.insert(
            "Originator".into(),
            header_or(incoming, "originator", "codex_cli_rs"),
        );
        headers.insert(
            "Session_id".into(),
            header_or(incoming, "session_id", request_id),
        );
        headers.insert("User-Agent".into(), CODEX_USER_AGENT.into());
        headers.insert(
            "Accept".into(),
            if stream {
                "text/event-stream"
            } else {
                "application/json"
            }
            .into(),
        );
    }
    if let Some(extra) = provider
        .preferences
        .get("headers")
        .and_then(Value::as_object)
    {
        for (name, value) in extra {
            if let Some(value) = value.as_str() {
                headers.insert(name.clone(), value.to_owned());
            }
        }
    }
    let passthrough = provider
        .preferences
        .get("passthrough_request_headers")
        .map(endpoint_values)
        .unwrap_or_default();
    for name in passthrough {
        headers.retain(|existing, _| !existing.eq_ignore_ascii_case(&name));
        if let Some(value) = incoming.get(&name).and_then(|value| value.to_str().ok()) {
            headers.insert(name, value.to_owned());
        }
    }
    if let Some(value) = incoming
        .get("x-oaix-settlement-nonce")
        .and_then(|v| v.to_str().ok())
    {
        headers.insert("X-OAIX-Settlement-Nonce".into(), value.to_owned());
    }
    if provider
        .preferences
        .get("oaix_routing_attempt_id")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        headers.insert("X-OAIX-Routing-Attempt-ID".into(), attempt_id.to_owned());
    }
    for (name, value) in &headers {
        HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| format!("provider {} produced invalid header {name}", provider.name))?;
        HeaderValue::from_str(value)
            .map_err(|_| format!("provider {} produced invalid header value", provider.name))?;
    }
    Ok(headers)
}

pub(crate) fn header_or(headers: &HeaderMap, name: &str, default: &str) -> String {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

pub(crate) fn normalize_upstream_url(base_url: &str, engine: &str, wants_compact: bool) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if wants_compact {
        if base.ends_with("/v1/responses/compact") || base.ends_with("/responses/compact") {
            return base.to_owned();
        }
        let response_url = normalize_upstream_url(base, engine, false);
        return format!("{response_url}/compact");
    }
    if engine != "codex" || base.ends_with("/v1/responses") || base.ends_with("/responses") {
        base.to_owned()
    } else {
        format!("{base}/responses")
    }
}
