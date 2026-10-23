//! A signed history block may be accepted by one Messages provider and rejected
//! by another. Recognize only the observed signature error and its actual wire
//! block; never remove or rewrite signed history to make a request pass.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "thinking_signature_incompatible";

pub(crate) fn rejected_block(status: u16, detail: &str, request: &[u8]) -> bool {
    if status != 400 {
        return false;
    }
    let Some((message_index, block_index)) = error_location(detail) else {
        return false;
    };
    let Ok(payload) = serde_json::from_slice::<Value>(request) else {
        return false;
    };
    let Some(message) = payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.get(message_index))
    else {
        return false;
    };
    let block = message
        .get("content")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.get(block_index));
    message.get("role").and_then(Value::as_str) == Some("assistant")
        && block.is_some_and(|block| {
            block.get("type").and_then(Value::as_str) == Some("thinking")
                && block.get("thinking").is_some_and(Value::is_string)
                && block
                    .get("signature")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.trim().is_empty())
        })
}

fn error_location(detail: &str) -> Option<(usize, usize)> {
    let mut candidate = detail.to_owned();
    for _ in 0..3 {
        let payload: Value = serde_json::from_str(&candidate).ok()?;
        let error = payload.get("error").or_else(|| payload.get("detail"))?;
        let message = error.get("message")?.as_str()?;
        if !error
            .get("code")
            .is_none_or(|code| code.is_null() || code.as_str() == Some("invalid_request_error"))
            || !error.get("param").is_none_or(Value::is_null)
        {
            return None;
        }
        if message.trim_start().starts_with('{') {
            if !error.get("type").is_none_or(|kind| {
                kind.is_null()
                    || matches!(
                        kind.as_str(),
                        Some("invalid_request_error" | "upstream_error")
                    )
            }) {
                return None;
            }
            candidate = message.to_owned();
            continue;
        }
        if error.get("type").and_then(Value::as_str) != Some("invalid_request_error") {
            return None;
        }
        // Deliberately exclude modified-thinking and prefix-binding errors,
        // additional prose, quoted examples, and arbitrary fields in the body.
        let location = message
            .strip_prefix("messages.")?
            .strip_suffix(": Invalid `signature` in `thinking` block")?;
        let (message, block) = location.split_once(".content.")?;
        let index = |s: &str| {
            (!s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()))
                .then(|| s.parse::<usize>().ok())
                .flatten()
        };
        return Some((index(message)?, index(block)?));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn error(message: &str) -> Value {
        json!({"error":{"type":"invalid_request_error","message":message}})
    }
    const MESSAGE: &str = "messages.1.content.0: Invalid `signature` in `thinking` block";
    fn request() -> Value {
        json!({"messages":[{"role":"user","content":"question"},
            {"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"opaque"},
            {"type":"tool_use","id":"tool_1","name":"read","input":{}}]}]})
    }

    #[test]
    fn requires_exact_error_and_signed_wire_block() {
        let raw = request().to_string();
        let body = error(MESSAGE);
        for detail in [
            body.to_string(),
            json!({"error":{"message":body.to_string()}}).to_string(),
        ] {
            assert!(rejected_block(400, &detail, raw.as_bytes()));
            assert!(!rejected_block(422, &detail, raw.as_bytes()));
        }
        for message in [
            MESSAGE.replace("1.content.0", "999.content.0"),
            MESSAGE.replace("1.content.0", "1.content.1"),
            MESSAGE.replace("1.content.0", "-1.content.0"),
            MESSAGE.replace("1.content.0", "99999999999999999999999999.content.0"),
            format!("{MESSAGE}. The block is bound to a different conversation."),
            format!("Invalid input: '{MESSAGE}'"),
            MESSAGE.replace("signature", "text"),
        ] {
            assert!(!rejected_block(
                400,
                &error(&message).to_string(),
                raw.as_bytes()
            ));
        }
        for (field, value) in [
            ("code", json!("invalid_type")),
            ("param", json!("messages")),
            ("type", json!("authentication_error")),
        ] {
            let mut other = body.clone();
            other["error"][field] = value;
            assert!(!rejected_block(400, &other.to_string(), raw.as_bytes()));
        }
        assert!(!rejected_block(
            400,
            &json!({"error":{"code":"invalid_type","message":body.to_string()}}).to_string(),
            raw.as_bytes()
        ));
        assert!(!rejected_block(
            400,
            &json!({"error":{"type":"authentication_error","message":body.to_string()}})
                .to_string(),
            raw.as_bytes()
        ));
        assert!(!rejected_block(400, MESSAGE, raw.as_bytes()));
        assert!(!rejected_block(
            400,
            &json!({"input":body}).to_string(),
            raw.as_bytes()
        ));
    }

    #[test]
    fn preserves_validation_for_missing_or_malformed_blocks() {
        for value in [Value::Null, json!(""), json!(" \n"), json!(23), json!({})] {
            let mut body = request();
            body["messages"][1]["content"][0]["signature"] = value;
            assert!(!rejected_block(
                400,
                &error(MESSAGE).to_string(),
                body.to_string().as_bytes()
            ));
        }
        for (path, value) in [
            ("/messages/1/role", json!("user")),
            ("/messages/1/content/0/type", json!("redacted_thinking")),
            ("/messages/1/content/0/thinking", Value::Null),
            ("/messages/1/content", json!("text")),
        ] {
            let mut body = request();
            *body.pointer_mut(path).unwrap() = value;
            assert!(!rejected_block(
                400,
                &error(MESSAGE).to_string(),
                body.to_string().as_bytes()
            ));
        }
    }
}
