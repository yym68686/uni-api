//! Reactive repair of a rejected, demonstrably unexecuted historical tool pair.
use std::collections::HashMap;

use serde_json::{json, Value};

const MESSAGE_SUFFIX: &str =
    "': empty string. Expected a string with minimum length 1, but got an empty string instead.";

pub(crate) fn is_uncommitted_400(outcome: &Value) -> bool {
    outcome.get("kind").and_then(Value::as_str) == Some("http_error")
        && outcome.get("status_code").and_then(Value::as_u64) == Some(400)
        && outcome.get("upstream_status_code").and_then(Value::as_u64) == Some(400)
        && outcome.get("committed").and_then(Value::as_bool) == Some(false)
}

pub(crate) fn repair(body: &str, outcome: &Value) -> Option<(String, usize)> {
    if !is_uncommitted_400(outcome) || !matches_error(outcome.get("body")?.as_str()?, 0) {
        return None;
    }
    let mut payload: Value = serde_json::from_str(body).ok()?;
    let items = payload.get_mut("input")?.as_array_mut()?;
    let mut by_id: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        if let Some(id) = item.get("call_id").and_then(Value::as_str) {
            by_id.entry(id).or_default().push(index);
        }
    }
    let mut replacements = Vec::new();
    for (index, call) in items.iter().enumerate() {
        if call.get("type").and_then(Value::as_str) != Some("function_call")
            || call.get("name").and_then(Value::as_str) != Some("")
            || call.get("arguments").and_then(Value::as_str).is_none()
        {
            continue;
        }
        let Some(id) = call
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        else {
            continue;
        };
        let Some(indices) = by_id.get(id) else {
            continue;
        };
        let [first, second] = indices.as_slice() else {
            continue;
        };
        if *first != index {
            continue;
        }
        let output = &items[*second];
        if output.get("type").and_then(Value::as_str) != Some("function_call_output")
            || output.get("output").and_then(Value::as_str) != Some("unsupported call: ")
        {
            continue;
        }
        replacements.push((
            index,
            json!({
                "type":"message", "role":"assistant", "content":[{
                    "type":"output_text",
                    "text":format!("Historical malformed tool call (not executed): {call}")
                }]
            }),
        ));
        replacements.push((
            *second,
            json!({
                "type":"message", "role":"user", "content":[{
                    "type":"input_text",
                    "text":format!("Historical tool execution error: {output}")
                }]
            }),
        ));
    }
    let pairs = replacements.len() / 2;
    if pairs == 0 {
        return None;
    }
    for (index, replacement) in replacements {
        items[index] = replacement;
    }
    serde_json::to_string(&payload)
        .ok()
        .map(|body| (body, pairs))
}

fn name_param(param: &str) -> bool {
    param
        .strip_prefix("input[")
        .and_then(|s| s.strip_suffix("].name"))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}

fn message_param(message: &str) -> Option<&str> {
    let param = message
        .strip_prefix("Invalid '")?
        .strip_suffix(MESSAGE_SUFFIX)?;
    name_param(param).then_some(param)
}

fn matches_error(detail: &str, depth: usize) -> bool {
    if depth >= 3 {
        return false;
    }
    let payload = serde_json::from_str::<Value>(detail).ok().or_else(|| {
        // OAIX may return one SSE error inside an HTTP 400 / error.message.
        // Never search arbitrary text, additional events, or request echoes.
        let mut lines = detail.trim().lines();
        if lines.next()?.trim_end_matches('\r') != "event: error" {
            return None;
        }
        let data = lines
            .next()?
            .trim_end_matches('\r')
            .strip_prefix("data: ")?;
        if lines.any(|line| !line.trim().is_empty()) {
            return None;
        }
        serde_json::from_str(data).ok()
    });
    let Some(error) = payload
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(Value::as_object)
    else {
        return false;
    };
    let Some(message) = error.get("message").and_then(Value::as_str) else {
        return false;
    };
    let param = message_param(message);
    let kind = error.get("type").and_then(Value::as_str);
    let code = error.get("code").and_then(Value::as_str);
    if kind == Some("invalid_request_error") && code == Some("empty_string") {
        return param.is_some() && param == error.get("param").and_then(Value::as_str);
    }
    // These two shapes were verified against real upstream responses.
    if param.is_some() && !error.contains_key("param") {
        if kind == Some("upstream_error") && !error.contains_key("code") {
            return true;
        }
        if kind == Some("gateway_error")
            && code == Some("oaix_gateway_error")
            && error.get("status").and_then(Value::as_u64) == Some(400)
        {
            return true;
        }
    }
    !error.contains_key("type")
        && !error.contains_key("code")
        && !error.contains_key("param")
        && matches_error(message, depth + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error() -> Value {
        json!({"error":{"code":"empty_string","type":"invalid_request_error",
            "param":"input[82].name",
            "message":format!("Invalid 'input[82].name{MESSAGE_SUFFIX}")}})
    }
    fn outcome() -> Value {
        json!({"kind":"http_error","status_code":400,"upstream_status_code":400,
            "committed":false,"body":error().to_string()})
    }
    fn pair(id: &str) -> Vec<Value> {
        vec![
            json!({"type":"function_call","name":"","call_id":id,"arguments":"{\"ok\":true}"}),
            json!({"type":"function_call_output","call_id":id,"output":"unsupported call: "}),
        ]
    }

    #[test]
    fn converts_all_confirmed_pairs_in_place_and_preserves_context() {
        let mut items = pair("bad1");
        items.extend([
            json!({"role":"user","content":"preserve"}),
            json!({"type":"reasoning","encrypted_content":"preserve"}),
            json!({"type":"function_call","name":"real","call_id":"real","arguments":"{}"}),
            json!({"type":"function_call_output","call_id":"real","output":"OK"}),
        ]);
        items.extend(pair("bad2"));
        let original =
            json!({"input":items,"model":"model","tools":[{"type":"custom","name":"exec"}]});
        let (body, count) = repair(&original.to_string(), &outcome()).unwrap();
        let fixed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(count, 2);
        assert_eq!(fixed["input"].as_array().unwrap()[2..6], items[2..6]);
        assert_eq!(fixed["tools"], original["tools"]);
        assert_eq!(fixed["model"], original["model"]);
        for index in [0, 1, 6, 7] {
            let text = fixed["input"][index]["content"][0]["text"]
                .as_str()
                .unwrap();
            let (_, raw) = text.split_once(": ").unwrap();
            assert_eq!(serde_json::from_str::<Value>(raw).unwrap(), items[index]);
        }
        assert!(repair(&body, &outcome()).is_none());
    }

    #[test]
    fn accepts_only_exact_official_and_observed_gateway_shapes() {
        let mut value = error();
        for _ in 0..3 {
            assert!(matches_error(&value.to_string(), 0));
            value = json!({"error":{"message":value.to_string()}});
        }
        assert!(!matches_error(&value.to_string(), 0));
        let message = error()["error"]["message"].clone();
        let tokens = json!({"error":{"type":"upstream_error","message":message}});
        assert!(matches_error(&tokens.to_string(), 0));
        let oaix = json!({"error":{"type":"gateway_error","code":"oaix_gateway_error","status":400,"message":message}});
        let sse = format!("event: error\ndata: {oaix}\n\n");
        assert!(matches_error(&sse, 0));
        assert!(matches_error(
            &json!({"error":{"message":sse}}).to_string(),
            0
        ));
        assert!(!matches_error(
            &format!("{sse}event: response.completed\ndata: {{}}\n\n"),
            0
        ));
        assert!(!matches_error(message.as_str().unwrap(), 0));
        for field in ["code", "type", "param", "message"] {
            let mut wrong = error();
            wrong["error"][field] = json!("other");
            assert!(!matches_error(&wrong.to_string(), 0));
        }
        for param in [
            "tools[0].name",
            "input[-1].name",
            "input[].name",
            "input[1].arguments",
        ] {
            let wrong = json!({"error":{"code":"empty_string","type":"invalid_request_error",
                "param":param,"message":format!("Invalid '{param}{MESSAGE_SUFFIX}")}});
            assert!(!matches_error(&wrong.to_string(), 0));
        }
        for mut wrong in [tokens, oaix] {
            wrong["error"]["code"] = json!("invalid_type");
            assert!(!matches_error(&wrong.to_string(), 0));
        }
    }

    #[test]
    fn refuses_other_statuses_and_committed_output() {
        for (field, value) in [
            ("kind", json!("semantic_failure")),
            ("status_code", json!(500)),
            ("upstream_status_code", json!(200)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut other = outcome();
            other[field] = value;
            assert!(repair(&json!({"input":pair("bad")}).to_string(), &other).is_none());
        }
    }

    #[test]
    fn refuses_ambiguous_or_executed_calls() {
        let cases = [
            (0, "name", json!("valid")),
            (0, "name", json!(" ")),
            (0, "name", Value::Null),
            (0, "call_id", json!("")),
            (0, "arguments", json!({})),
            (0, "type", json!("custom_tool_call")),
            (1, "output", json!("OK")),
            (1, "output", json!("unsupported call: real")),
            (1, "type", json!("custom_tool_call_output")),
        ];
        for (index, field, value) in cases {
            let mut items = pair("bad");
            items[index][field] = value;
            assert!(repair(&json!({"input":items}).to_string(), &outcome()).is_none());
        }
        let pair = pair("bad");
        for items in [
            vec![pair[0].clone()],
            vec![pair[1].clone(), pair[0].clone()],
            vec![pair[0].clone(), pair[1].clone(), pair[1].clone()],
        ] {
            assert!(repair(&json!({"input":items}).to_string(), &outcome()).is_none());
        }
    }
}
