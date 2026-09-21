//! Reactive repair for scheduler context incorrectly serialized as tool output.
use serde_json::{json, Value};

const ERROR_MESSAGE: &str =
    "The tool output does not match a previous tool call. Resend the tool-call context.";

pub(crate) fn repair(body: &str, outcome: &Value) -> Option<(String, usize)> {
    if outcome.get("kind").and_then(Value::as_str) != Some("http_error")
        || outcome.get("status_code").and_then(Value::as_u64) != Some(400)
        || outcome.get("upstream_status_code").and_then(Value::as_u64) != Some(400)
        || outcome.get("committed").and_then(Value::as_bool) != Some(false)
    {
        return None;
    }
    let detail = outcome.get("body").and_then(Value::as_str)?;
    if !matches_error(detail) {
        return None;
    }
    let mut payload: Value = serde_json::from_str(body).ok()?;
    let items = payload.get_mut("input")?.as_array_mut()?;
    let mut changed = 0;
    for item in items {
        let Some(output) = heartbeat_output(item) else {
            continue;
        };
        *item = json!({
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": format!("Historical scheduled heartbeat context:\n{output}")
            }]
        });
        changed += 1;
    }
    (changed > 0)
        .then(|| {
            serde_json::to_string(&payload)
                .ok()
                .map(|body| (body, changed))
        })
        .flatten()
}

fn matches_error(detail: &str) -> bool {
    let Ok(mut payload) = serde_json::from_str::<Value>(detail) else {
        return false;
    };
    for _ in 0..3 {
        let Some(error) = payload.get("error").and_then(Value::as_object) else {
            return false;
        };
        if error.get("code").and_then(Value::as_str) == Some("function_call_output_not_found")
            && error.get("type").and_then(Value::as_str) == Some("invalid_request_error")
            && error.get("param").and_then(Value::as_str) == Some("input")
            && error.get("message").and_then(Value::as_str) == Some(ERROR_MESSAGE)
        {
            return true;
        }
        // Some gateways wrap the original JSON in error.message.
        if error.contains_key("code") || error.contains_key("param") {
            return false;
        }
        let Some(inner) = error.get("message").and_then(Value::as_str) else {
            return false;
        };
        let Ok(inner) = serde_json::from_str(inner) else {
            return false;
        };
        payload = inner;
    }
    false
}

fn heartbeat_output(item: &Value) -> Option<&str> {
    if item.get("type").and_then(Value::as_str) != Some("function_call_output")
        || item.get("name").and_then(Value::as_str) != Some("automation_update")
        || item.get("namespace").and_then(Value::as_str) != Some("codex_app")
        || !matches!(item.get("call_id"), None | Some(Value::Null))
            && item.get("call_id").and_then(Value::as_str) != Some("")
    {
        return None;
    }
    let output = item.get("output")?.as_str()?;
    (output.trim_start().starts_with("<heartbeat>") && output.trim_end().ends_with("</heartbeat>"))
        .then_some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error() -> Value {
        json!({"error": {"code":"function_call_output_not_found",
            "type":"invalid_request_error", "param":"input", "message":ERROR_MESSAGE}})
    }

    fn outcome() -> Value {
        json!({"kind":"http_error", "status_code":400, "upstream_status_code":400,
            "committed":false, "body":error().to_string()})
    }

    fn heartbeat() -> Value {
        json!({"type":"function_call_output", "name":"automation_update",
            "namespace":"codex_app", "output":" <heartbeat>keep all text</heartbeat>\n"})
    }

    #[test]
    fn repairs_only_injected_heartbeats_and_preserves_other_context() {
        let mut real = heartbeat();
        real["call_id"] = json!("call-real");
        let mut malformed_id = heartbeat();
        malformed_id["call_id"] = json!(false);
        let untouched = vec![
            json!({"type":"function_call", "name":"automation_update", "call_id":"call-real", "arguments":"{}"}),
            real,
            malformed_id,
            json!({"type":"custom_tool_call_output", "call_id":"custom", "output":"result"}),
            json!({"type":"reasoning", "encrypted_content":"preserve"}),
            json!({"role":"user", "content":"preserve user text"}),
        ];
        let mut items = untouched.clone();
        items.extend(vec![heartbeat(); 16]);
        let body =
            json!({"model":"model", "input":items, "tools":[{"type":"custom","name":"exec"}]});
        let (repaired, count) = repair(&body.to_string(), &outcome()).unwrap();
        let repaired: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(count, 16);
        assert_eq!(
            repaired["input"].as_array().unwrap()[..untouched.len()],
            untouched
        );
        assert_eq!(repaired["tools"], body["tools"]);
        assert_eq!(
            repaired["input"][6]["content"][0]["text"],
            "Historical scheduled heartbeat context:\n <heartbeat>keep all text</heartbeat>\n"
        );
        assert!(repair(&repaired.to_string(), &outcome()).is_none());
    }

    #[test]
    fn accepts_exact_error_and_bounded_json_wrappers() {
        let mut value = error();
        for _ in 0..3 {
            assert!(matches_error(&value.to_string()));
            value = json!({"error":{"message":value.to_string()}});
        }
        assert!(!matches_error(&value.to_string()));
        assert!(!matches_error(ERROR_MESSAGE));
        for field in ["code", "type", "param", "message"] {
            let mut value = error();
            value["error"][field] = json!("different");
            assert!(!matches_error(&value.to_string()));
        }
    }

    #[test]
    fn refuses_other_statuses_semantic_errors_or_committed_output() {
        let body = json!({"input":[heartbeat()]}).to_string();
        for (field, value) in [
            ("kind", json!("semantic_failure")),
            ("status_code", json!(500)),
            ("upstream_status_code", json!(200)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut other = outcome();
            other[field] = value;
            assert!(repair(&body, &other).is_none());
        }
    }

    #[test]
    fn refuses_unrelated_or_malformed_tool_outputs() {
        for (field, value) in [
            ("name", json!("other")),
            ("namespace", json!("other")),
            ("type", json!("custom_tool_call_output")),
            ("call_id", json!("call-present")),
            ("call_id", json!(0)),
            ("output", json!("<heartbeat>unfinished")),
            (
                "output",
                json!([{"type":"text","text":"<heartbeat>x</heartbeat>"}]),
            ),
        ] {
            let mut item = heartbeat();
            item[field] = value;
            assert!(repair(&json!({"input":[item]}).to_string(), &outcome()).is_none());
        }
    }
}
