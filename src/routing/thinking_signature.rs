//! A signed history block may be accepted by one Messages provider and rejected
//! by another. Recognize only the observed signature error and its actual wire
//! block (or the exact reindexed relay envelope); never rewrite signed history.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "thinking_signature_incompatible";

pub(crate) fn rejected_block(status: u16, detail: &str, request: &[u8]) -> bool {
    if status != 400 {
        return false;
    }
    let Some(rejection) = error_location(detail) else {
        return false;
    };
    let Ok(payload) = serde_json::from_slice::<Value>(request) else {
        return false;
    };
    let Some(messages) = payload.get("messages").and_then(Value::as_array) else {
        return false;
    };
    match rejection {
        Rejection::Block(message_index, block_index) => {
            messages.get(message_index).is_some_and(|message| {
                message.get("role").and_then(Value::as_str) == Some("assistant")
                    && message
                        .get("content")
                        .and_then(Value::as_array)
                        .and_then(|blocks| blocks.get(block_index))
                        .is_some_and(signed_thinking)
            })
        }
        Rejection::ReindexedRelay => {
            // This relay reports positions after its own transformation. The
            // original location cannot identify a block. Require signed history
            // and leave missing/malformed thinking signatures as client errors.
            let mut found = false;
            for message in messages {
                for block in message
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if block.get("type").and_then(Value::as_str) == Some("thinking") {
                        if message.get("role").and_then(Value::as_str) != Some("assistant")
                            || !signed_thinking(block)
                        {
                            return false;
                        }
                        found = true;
                    }
                }
            }
            found
        }
    }
}

fn signed_thinking(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("thinking")
        && block.get("thinking").is_some_and(Value::is_string)
        && block
            .get("signature")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
}

enum Rejection {
    Block(usize, usize),
    ReindexedRelay,
}

// Only the two observed attribution suffixes are accepted, with bounded opaque
// IDs. Extra prose (including prefix-binding guidance) must not trigger failover.
fn strip_relay_receipts(message: &str) -> Option<&str> {
    let (message, receipts) = message.split_once(" (luban request id: ")?;
    let (luban, request) = receipts.split_once(") (request id: ")?;
    let request = request.strip_suffix(')')?;
    let valid_id = |s: &str| {
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
    };
    if !luban.strip_prefix("req_").is_some_and(valid_id) || luban.len() > 128 || !valid_id(request)
    {
        return None;
    }
    Some(message)
}

fn error_location(detail: &str) -> Option<Rejection> {
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
        let relay = error.get("type").and_then(Value::as_str) == Some("<nil>");
        let message = if relay {
            if !error.get("code").is_none_or(Value::is_null) {
                return None;
            }
            strip_relay_receipts(message)?
        } else if error.get("type").and_then(Value::as_str) == Some("invalid_request_error") {
            message
        } else {
            return None;
        };
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
        let (message_index, block_index) = (index(message)?, index(block)?);
        return Some(if relay {
            Rejection::ReindexedRelay
        } else {
            Rejection::Block(message_index, block_index)
        });
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
    fn relay_error() -> Value {
        json!({"error":{"type":"<nil>","message":
            "messages.3.content.79: Invalid `signature` in `thinking` block (luban request id: req_fixture_1) (request id: fixture-request-2)"}})
    }

    #[test]
    fn reindexed_relay_preserves_history_and_retries_exact_envelope() {
        let raw = request().to_string();
        let error = relay_error();
        for detail in [
            error.to_string(),
            json!({"error":{"message":error.to_string()}}).to_string(),
            json!({"detail":error["error"]}).to_string(),
        ] {
            assert!(rejected_block(400, &detail, raw.as_bytes()));
            for status in [200, 401, 403, 422, 500] {
                assert!(!rejected_block(status, &detail, raw.as_bytes()));
            }
        }
    }

    #[test]
    fn relay_receipts_are_not_a_general_signature_substring_match() {
        let raw = request().to_string();
        let original = relay_error();
        let message = original["error"]["message"].as_str().unwrap();
        for changed in [
            message.replace("<nil>", "unknown") + " extra text",
            message.replace("req_fixture_1", ""),
            message.replace("req_fixture_1", "req_"),
            message.replace("req_fixture_1", "req_contains space"),
            message.replace("fixture-request-2", "https://example.com"),
            message.replace("fixture-request-2", &"a".repeat(129)),
            message.replace(" (luban request id: req_fixture_1)", ""),
            message.replace(" (request id: fixture-request-2)", ""),
            message.replace("3.content.79", "-3.content.79"),
            message.replace("3.content.79", "3.content.99999999999999999999999"),
            message.replace("signature", "text"),
            format!("Quoted input: {message}"),
            format!("{message}. The block is bound to a different conversation."),
        ] {
            let mut other = original.clone();
            other["error"]["message"] = json!(changed);
            assert!(!rejected_block(400, &other.to_string(), raw.as_bytes()));
        }
        for (field, value) in [
            ("type", json!("invalid_request_error")),
            ("type", json!("nil")),
            ("type", Value::Null),
            ("code", json!("invalid_type")),
            ("code", json!("invalid_request_error")),
            ("param", json!("messages")),
        ] {
            let mut other = original.clone();
            other["error"][field] = value;
            assert!(!rejected_block(400, &other.to_string(), raw.as_bytes()));
        }
        assert!(!rejected_block(
            400,
            &json!({"debug": original}).to_string(),
            raw.as_bytes()
        ));
        assert!(!rejected_block(
            400,
            &json!({"error":{"code":"invalid_type","message":original.to_string()}}).to_string(),
            raw.as_bytes()
        ));
    }

    #[test]
    fn relay_requires_signed_history_without_malformed_thinking_blocks() {
        let detail = relay_error().to_string();
        for body in [
            json!({}),
            json!({"messages":[]}),
            json!({"messages":[{"role":"user","content":"hello"}]}),
        ] {
            assert!(!rejected_block(400, &detail, body.to_string().as_bytes()));
        }
        for invalid in [Value::Null, json!(""), json!(" "), json!(42)] {
            let mut body = request();
            let mut second = body["messages"][1]["content"][0].clone();
            second["signature"] = invalid;
            body["messages"][1]["content"]
                .as_array_mut()
                .unwrap()
                .push(second);
            assert!(!rejected_block(400, &detail, body.to_string().as_bytes()));
        }
        let mut body = request();
        body["messages"][1]["role"] = json!("user");
        assert!(!rejected_block(400, &detail, body.to_string().as_bytes()));
        let mut body = request();
        body["messages"][1]["content"][0]["thinking"] = Value::Null;
        assert!(!rejected_block(400, &detail, body.to_string().as_bytes()));
    }
}
