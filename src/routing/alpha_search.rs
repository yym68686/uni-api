//! Search relay rejection contradicted by a text-only wire input.
//! Never rewrite input or reinterpret genuine encrypted/tool-item validation.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "alpha_search_text_incompatible";
const MESSAGE: &str = "Search input supports text messages, not encrypted history or tool items";

pub(crate) fn rejected_input(status: u16, detail: &str, request: &[u8]) -> bool {
    if status != 400 {
        return false;
    }
    let Ok(response) = serde_json::from_str::<Value>(detail) else {
        return false;
    };
    let Some(error) = response.get("error").and_then(Value::as_object) else {
        return false;
    };
    if error.get("type").and_then(Value::as_str) != Some("invalid_request_error")
        || error.get("code").and_then(Value::as_str) != Some("bad_request")
        || error.get("message").and_then(Value::as_str) != Some(MESSAGE)
        || !error.get("param").is_none_or(Value::is_null)
    {
        return false;
    }
    let Ok(payload) = serde_json::from_slice::<Value>(request) else {
        return false;
    };
    match payload.get("input") {
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Array(items)) => !items.is_empty() && items.iter().all(text_message),
        _ => false,
    }
}

fn text_message(item: &Value) -> bool {
    let Some(message) = item.as_object() else {
        return false;
    };
    if !message.keys().all(|key| {
        matches!(
            key.as_str(),
            "type"
                | "role"
                | "content"
                | "id"
                | "status"
                | "internal_chat_message_metadata_passthrough"
        )
    }) || message
        .get("type")
        .is_some_and(|v| v.as_str() != Some("message"))
        || !matches!(
            message.get("role").and_then(Value::as_str),
            Some("user" | "assistant" | "system" | "developer")
        )
    {
        return false;
    }
    match message.get("content") {
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Array(parts)) => {
            !parts.is_empty()
                && parts.iter().all(|part| {
                    part.as_object().is_some_and(|p| {
                        p.len() == 2
                            && matches!(
                                p.get("type").and_then(Value::as_str),
                                Some("input_text" | "output_text")
                            )
                            && p.get("text").and_then(Value::as_str).is_some()
                    })
                })
                && parts.iter().any(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.trim().is_empty())
                })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn error() -> Value {
        json!({"error":{"message":MESSAGE,"type":"invalid_request_error","code":"bad_request"}})
    }
    fn wire() -> Value {
        json!({"model":"fixture","input":[{"type":"message","role":"user",
            "content":[{"type":"input_text","text":"fixture"}],
            "internal_chat_message_metadata_passthrough":{"content_item_kinds":["user.image"]}}]})
    }
    fn matches(error: &Value, wire: &Value) -> bool {
        rejected_input(400, &error.to_string(), wire.to_string().as_bytes())
    }

    #[test]
    fn exact_error_requires_text_only_wire() {
        assert!(matches(&error(), &wire()));
        for input in [
            json!("query"),
            json!([{"role":"user","content":"query"}]),
            json!([{"type":"message","role":"assistant","content":[{"type":"output_text","text":"query"}]}]),
        ] {
            let mut w = wire();
            w["input"] = input;
            assert!(matches(&error(), &w));
        }
        let mut e = error();
        e["error"]["param"] = Value::Null;
        assert!(matches(&e, &wire()));
        for status in [200, 401, 403, 404, 413, 422, 429, 500, 502] {
            assert!(!rejected_input(
                status,
                &error().to_string(),
                wire().to_string().as_bytes()
            ));
        }
    }

    #[test]
    fn unrelated_envelopes_and_messages_stay_client_errors() {
        for (key, value) in [
            ("type", json!("upstream_error")),
            ("type", Value::Null),
            ("code", json!("invalid_type")),
            ("code", json!(400)),
            ("code", Value::Null),
            ("param", json!("input")),
            ("param", json!("")),
            ("message", json!(format!("{MESSAGE}."))),
            ("message", json!(format!("Invalid input: {MESSAGE}"))),
            ("message", json!(MESSAGE.to_ascii_lowercase())),
        ] {
            let mut e = error();
            e["error"][key] = value;
            assert!(!matches(&e, &wire()), "{e}");
        }
        for e in [
            json!({"detail":error()["error"]}),
            json!({"error":{"message":error().to_string()}}),
            json!({"error":{"message":"invalid input"},"debug":error()}),
        ] {
            assert!(!matches(&e, &wire()));
        }
        assert!(!rejected_input(400, MESSAGE, wire().to_string().as_bytes()));
        assert!(!rejected_input(400, &error().to_string(), b"not json"));
    }

    #[test]
    fn genuine_history_tools_images_and_invalid_inputs_stay_client_errors() {
        for input in [
            Value::Null,
            json!(""),
            json!(" "),
            json!([]),
            json!({}),
            json!([{"type":"reasoning","encrypted_content":"opaque"}]),
            json!([{"type":"function_call","name":"lookup","arguments":"{}"}]),
            json!([{"type":"function_call_output","call_id":"x","output":"fixture"}]),
            json!([{"role":"tool","content":"fixture"}]),
            json!([{"role":"user","content":[]}]),
            json!([{"role":"user","content":[{"type":"input_image","image_url":"fixture"}]}]),
            json!([{"role":"user","content":[{"type":"input_text","text":1}]}]),
            json!([{"role":"user","content":"fixture","encrypted_content":"opaque"}]),
            json!([{"role":"assistant","content":"fixture","tool_calls":[]}]),
        ] {
            let mut w = wire();
            w["input"] = input;
            assert!(!matches(&error(), &w), "{w}");
        }
        let mut w = wire();
        w["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"reasoning","encrypted_content":"opaque"}));
        assert!(!matches(&error(), &w));
    }
}
