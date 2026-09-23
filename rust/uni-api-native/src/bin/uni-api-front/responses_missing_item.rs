//! Recover the visible summary of one unpersisted, unencrypted reasoning item.
use serde_json::{json, Value};

const PREFIX: &str = "Item with id '";
const SUFFIX: &str = "' not found. Items are not persisted when `store` is set to false. Try again with `store` set to true, or remove this item from your input.";

pub(crate) fn is_uncommitted_404(outcome: &Value) -> bool {
    outcome.get("kind").and_then(Value::as_str) == Some("http_error")
        && outcome.get("status_code").and_then(Value::as_u64) == Some(404)
        && outcome.get("upstream_status_code").and_then(Value::as_u64) == Some(404)
        && outcome.get("committed").and_then(Value::as_bool) == Some(false)
}

pub(crate) fn repair(body: &str, outcome: &Value) -> Option<(String, usize)> {
    if !is_uncommitted_404(outcome) {
        return None;
    }
    let target = missing_id(outcome.get("body")?.as_str()?)?;
    let mut payload: Value = serde_json::from_str(body).ok()?;
    if payload.get("store").and_then(Value::as_bool) != Some(false) {
        return None;
    }
    let items = payload.get_mut("input")?.as_array_mut()?;
    let mut matches = items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.get("id").and_then(Value::as_str) == Some(target.as_str()));
    let (index, item) = matches.next()?;
    if matches.next().is_some() || item.get("type").and_then(Value::as_str) != Some("reasoning") {
        return None;
    }
    match item.get("encrypted_content") {
        None | Some(Value::Null) => {}
        Some(Value::String(value)) if value.is_empty() => {}
        _ => return None,
    }
    match item.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::Array(value)) if value.is_empty() => {}
        _ => return None,
    }
    let summary = item.get("summary")?.as_array()?;
    let texts: Option<Vec<&str>> = summary
        .iter()
        .map(|part| {
            (part.get("type").and_then(Value::as_str) == Some("summary_text"))
                .then(|| part.get("text").and_then(Value::as_str))
                .flatten()
        })
        .collect();
    let texts = texts?;
    if !texts.iter().any(|text| !text.trim().is_empty()) {
        return None;
    }
    // A summary-only reasoning object can be silently ignored after removing
    // its ID. Keep every visible summary in an ordinary assistant message.
    items[index] = json!({
        "type":"message", "role":"assistant", "content":[{
            "type":"output_text",
            "text":format!("Historical reasoning summary:\n{}", texts.join("\n\n"))
        }]
    });
    serde_json::to_string(&payload).ok().map(|body| (body, 1))
}

fn missing_id(detail: &str) -> Option<String> {
    let mut payload: Value = serde_json::from_str(detail).ok()?;
    for _ in 0..3 {
        let error = payload.get("error")?.as_object()?;
        let message = error.get("message")?.as_str()?;
        if error.get("type").and_then(Value::as_str) == Some("invalid_request_error")
            && error.get("param").and_then(Value::as_str) == Some("input")
            && error.get("code") == Some(&Value::Null)
        {
            let id = message.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?;
            let suffix = id.strip_prefix("rs_")?;
            return (!suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphanumeric()))
                .then(|| id.to_owned());
        }
        if ["type", "param", "code"]
            .iter()
            .any(|key| error.contains_key(*key))
        {
            return None;
        }
        payload = serde_json::from_str(message).ok()?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "rs_c65ee145dd863116153ca83b";
    fn error() -> Value {
        json!({"error":{"code":null,"type":"invalid_request_error","param":"input",
            "message":format!("{PREFIX}{ID}{SUFFIX}")}})
    }
    fn outcome() -> Value {
        json!({"kind":"http_error","status_code":404,"upstream_status_code":404,
            "committed":false,"body":error().to_string()})
    }
    fn body() -> Value {
        json!({"model":"model","store":false,"input":[
            {"type":"reasoning","id":ID,"summary":[
                {"type":"summary_text","text":"one\n"},{"type":"summary_text","text":"two"}],
                "encrypted_content":null,"content":null},
            {"type":"reasoning","id":"rs_preserve","encrypted_content":"opaque","summary":[]},
            {"type":"function_call","name":"real","call_id":"call-real","arguments":"{}"},
            {"type":"function_call_output","call_id":"call-real","output":"result"},
            {"role":"user","content":"continue"}],
            "tools":[{"type":"function","name":"real"}],"reasoning":{"effort":"high"}})
    }

    #[test]
    fn replaces_only_the_error_target_preserving_all_summary_text() {
        let original = body();
        let (wire, count) = repair(&original.to_string(), &outcome()).unwrap();
        let fixed: Value = serde_json::from_str(&wire).unwrap();
        let mut expected = original;
        expected["input"][0] = json!({"type":"message","role":"assistant","content":[{
            "type":"output_text","text":"Historical reasoning summary:\none\n\n\ntwo"}]});
        assert_eq!(fixed, expected);
        assert_eq!(count, 1);
        assert!(repair(&wire, &outcome()).is_none());
    }

    #[test]
    fn recognizes_exact_error_and_only_bounded_untyped_wrappers() {
        let mut wrapped = error();
        for _ in 0..3 {
            assert_eq!(missing_id(&wrapped.to_string()).as_deref(), Some(ID));
            wrapped = json!({"error":{"message":wrapped.to_string()}});
        }
        assert!(missing_id(&wrapped.to_string()).is_none());
        assert!(missing_id(&format!("{PREFIX}{ID}{SUFFIX}")).is_none());
        for field in ["code", "type", "param", "message"] {
            let mut wrong = error();
            wrong["error"][field] = json!("different");
            assert!(missing_id(&wrong.to_string()).is_none());
            let mut absent = error();
            absent["error"].as_object_mut().unwrap().remove(field);
            assert!(missing_id(&absent.to_string()).is_none());
        }
        for id in ["rs_", "msg_other", "rs_bad/path", "rs_bad'quote", "rs_é"] {
            let mut wrong = error();
            wrong["error"]["message"] = json!(format!("{PREFIX}{id}{SUFFIX}"));
            assert!(missing_id(&wrong.to_string()).is_none());
        }
        assert!(missing_id(
            &json!({"error":{"type":"upstream_error","message":error().to_string()}}).to_string()
        )
        .is_none());
        assert!(missing_id(
            &json!({"error":{"message":"invalid input"},"debug":error()}).to_string()
        )
        .is_none());
    }

    #[test]
    fn requires_stateless_precommit_http_404() {
        for (field, value) in [
            ("kind", json!("semantic_error")),
            ("status_code", json!(400)),
            ("upstream_status_code", json!(200)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut other = outcome();
            other[field] = value;
            assert!(repair(&body().to_string(), &other).is_none());
        }
        for value in [json!(true), json!("false"), Value::Null] {
            let mut other = body();
            other["store"] = value;
            assert!(repair(&other.to_string(), &outcome()).is_none());
        }
        let mut other = body();
        other.as_object_mut().unwrap().remove("store");
        assert!(repair(&other.to_string(), &outcome()).is_none());
    }

    #[test]
    fn protects_encrypted_content_references_and_ambiguous_ids() {
        for (field, value) in [
            ("type", json!("item_reference")),
            ("id", json!("rs_other")),
            ("encrypted_content", json!("opaque")),
            ("encrypted_content", json!({})),
            ("content", json!([{"type":"reasoning_text","text":"raw"}])),
            ("content", json!("raw")),
            ("summary", json!([])),
            ("summary", Value::Null),
            ("summary", json!([{"type":"summary_text","text":" "}])),
            ("summary", json!([{"type":"unknown","text":"preserve"}])),
            ("summary", json!([{"type":"summary_text","text":false}])),
        ] {
            let mut other = body();
            other["input"][0][field] = value;
            assert!(repair(&other.to_string(), &outcome()).is_none(), "{field}");
        }
        let mut duplicate = body();
        duplicate["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"item_reference","id":ID}));
        assert!(repair(&duplicate.to_string(), &outcome()).is_none());
    }

    #[test]
    fn accepts_absent_or_empty_encrypted_fields_but_not_blank_ciphertext() {
        let mut other = body();
        let item = other["input"][0].as_object_mut().unwrap();
        item.remove("encrypted_content");
        item.remove("content");
        assert!(repair(&other.to_string(), &outcome()).is_some());
        other["input"][0]["encrypted_content"] = json!("");
        other["input"][0]["content"] = json!([]);
        assert!(repair(&other.to_string(), &outcome()).is_some());
        other["input"][0]["encrypted_content"] = json!(" ");
        assert!(repair(&other.to_string(), &outcome()).is_none());
    }
}
