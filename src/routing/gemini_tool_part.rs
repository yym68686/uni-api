//! Narrow relay compatibility failure, checked against the actual Gemini wire.
//! A genuinely absent/malformed data union must remain a client validation 400.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "gemini_tool_part_incompatible";

pub(crate) fn rejected_block(status: u16, detail: &str, request: &[u8]) -> bool {
    if status != 400 {
        return false;
    }
    let Some(locations) = error_locations(detail) else {
        return false;
    };
    let Ok(payload) = serde_json::from_slice::<Value>(request) else {
        return false;
    };
    locations.iter().all(|&(content, part)| {
        let Some(turn) = payload
            .get("contents")
            .and_then(Value::as_array)
            .and_then(|v| v.get(content))
        else {
            return false;
        };
        let Some(parts) = turn.get("parts").and_then(Value::as_array) else {
            return false;
        };
        turn.get("role").and_then(Value::as_str) == Some("model")
            && parts.iter().any(valid_function_call)
            && parts.get(part).is_some_and(|part| {
                valid_function_call(part)
                    || part.as_object().is_some_and(|p| {
                        p.len() == 1 && p.get("text").and_then(Value::as_str) == Some("")
                    })
            })
    })
}

fn valid_function_call(part: &Value) -> bool {
    let Some(p) = part.as_object() else {
        return false;
    };
    if !p
        .keys()
        .all(|k| matches!(k.as_str(), "functionCall" | "thoughtSignature"))
        || p.get("thoughtSignature")
            .is_some_and(|s| s.as_str().is_none_or(|s| s.is_empty()))
    {
        return false;
    }
    let Some(call) = p.get("functionCall").and_then(Value::as_object) else {
        return false;
    };
    call.get("name")
        .and_then(Value::as_str)
        .is_some_and(|n| !n.trim().is_empty())
        && call.get("args").is_some_and(Value::is_object)
}

fn error_locations(detail: &str) -> Option<Vec<(usize, usize)>> {
    let payload: Value = serde_json::from_str(detail).ok()?;
    let error = payload.get("error")?;
    // Hash-verified relay envelope from the incident. Do not accept other
    // validation codes, quoted examples, nested errors, or generic oneof prose.
    if error.get("type")?.as_str()? != "upstream_error"
        || error.get("code")?.as_u64()? != 400
        || !error
            .get("param")
            .is_none_or(|p| p.is_null() || p.as_str() == Some(""))
    {
        return None;
    }
    let message = error.get("message")?.as_str()?;
    if message.len() > 16_384 {
        return None;
    }
    let mut locations = Vec::new();
    for line in message.strip_suffix('\n').unwrap_or(message).split('\n') {
        let path = line
            .strip_prefix("* ***.contents[")?
            .strip_suffix("].data: required oneof field 'data' must have one initialized field")?;
        let (content, part) = path.split_once("].parts[")?;
        let index = |s: &str| {
            (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                .then(|| s.parse::<usize>().ok())
                .flatten()
        };
        locations.push((index(content)?, index(part)?));
        if locations.len() > 64 {
            return None;
        }
    }
    (!locations.is_empty()).then_some(locations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn error() -> Value {
        json!({"error":{"type":"upstream_error","code":400,"param":"","message":
            "* ***.contents[1].parts[0].data: required oneof field 'data' must have one initialized field\n"}})
    }
    fn wire() -> Value {
        json!({"contents":[{"role":"user","parts":[{"text":"fixture"}]},
            {"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"text":""}},"thoughtSignature":"opaque"}]}]})
    }
    fn matches(error: &Value, wire: &Value) -> bool {
        rejected_block(400, &error.to_string(), wire.to_string().as_bytes())
    }

    #[test]
    fn accepts_only_indexed_tool_data_and_exact_relay_envelope() {
        assert!(matches(&error(), &wire()));
        let mut native = wire();
        native["contents"][1]["parts"]
            .as_array_mut()
            .unwrap()
            .insert(0, json!({"text":""}));
        assert!(matches(&error(), &native));
        assert!(!rejected_block(
            422,
            &error().to_string(),
            wire().to_string().as_bytes()
        ));
        for (key, value) in [
            ("type", json!("invalid_request_error")),
            ("code", json!("400")),
            ("code", json!("invalid_type")),
            ("code", Value::Null),
            ("param", json!("contents")),
        ] {
            let mut e = error();
            e["error"][key] = value;
            assert!(!matches(&e, &wire()));
        }
        for param in [Value::Null, json!("")] {
            let mut e = error();
            e["error"]["param"] = param;
            assert!(matches(&e, &wire()));
        }
        for message in ["", "required oneof field 'data' must have one initialized field",
            "* ***.contents[1].parts[0].data: required oneof field 'data' must have one initialized field\nadditional validation",
            "* ***.contents[999999999999999999999999].parts[0].data: required oneof field 'data' must have one initialized field"] {
            let mut e = error(); e["error"]["message"] = json!(message);
            assert!(!matches(&e, &wire()));
        }
        let mut two = error();
        two["error"]["message"] = json!(format!(
            "{}{}",
            error()["error"]["message"].as_str().unwrap(),
            error()["error"]["message"]
                .as_str()
                .unwrap()
                .replace("contents[1]", "contents[9]")
        ));
        assert!(
            !matches(&two, &wire()),
            "all listed positions must be compatible"
        );
    }

    #[test]
    fn preserves_real_missing_data_and_other_validation() {
        for part in [
            json!({}),
            json!({"thoughtSignature":"opaque"}),
            json!({"text":null}),
            json!({"text":"valid"}),
            json!({"functionCall":{"name":"lookup"}}),
            json!({"functionCall":{"name":"","args":{}}}),
            json!({"functionCall":{"name":"lookup","args":[]}}),
            json!({"functionCall":{"name":"lookup","args":{}},"text":""}),
        ] {
            let mut w = wire();
            w["contents"][1]["parts"][0] = part;
            assert!(!matches(&error(), &w));
        }
        let mut w = wire();
        w["contents"][1]["role"] = json!("user");
        assert!(!matches(&error(), &w));
        w = wire();
        w["contents"][1]["parts"][0] = json!({"text":""});
        assert!(
            !matches(&error(), &w),
            "standalone empty text is not a tool placeholder"
        );
    }
}
