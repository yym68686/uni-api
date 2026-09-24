//! Recover visible summaries from a rejected, non-native Cursor envelope.

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

const ERROR_PREFIX: &str = "The encrypted content for item ";
const ERROR_SUFFIX: &str =
    " could not be verified. Reason: Encrypted content could not be decrypted or parsed.";
const ENVELOPE_PREFIX: &str = "cursor-sand-v1:";
const MAX_ENVELOPE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CursorEnvelope {
    signature: String,
    text: String,
}

pub(crate) fn repair(body: &str, outcome: &Value) -> Option<(String, usize)> {
    if !super::empty_name::is_uncommitted_400(outcome) {
        return None;
    }
    let target = rejected_id(outcome.get("body")?.as_str()?)?;
    let mut payload: Value = serde_json::from_str(body).ok()?;
    if payload.get("store").and_then(Value::as_bool) != Some(false) {
        return None;
    }
    let items = payload.get_mut("input")?.as_array_mut()?;
    let mut id_counts = HashMap::new();
    for item in items.iter() {
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            *id_counts.entry(id).or_insert(0usize) += 1;
        }
    }
    if id_counts.get(target.as_str()) != Some(&1) {
        return None;
    }
    let target_item = items
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(target.as_str()))?;
    compatible_summary(target_item)?;

    // A real history can contain several adjacent envelopes. Repair every
    // proven equivalent envelope in this one resend, not one resend per ID.
    let replacements: Vec<_> = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let id = item.get("id")?.as_str()?;
            if !valid_id(id) || id_counts.get(id) != Some(&1) {
                return None;
            }
            let summary = compatible_summary(item)?;
            Some((
                index,
                json!({
                    "type":"message", "role":"assistant", "content":[{
                        "type":"output_text",
                        "text":format!("Historical reasoning summary:\n{summary}")
                    }]
                }),
            ))
        })
        .collect();
    let count = replacements.len();
    for (index, replacement) in replacements {
        items[index] = replacement;
    }
    serde_json::to_string(&payload)
        .ok()
        .map(|body| (body, count))
}

fn valid_id(id: &str) -> bool {
    id.strip_prefix("rs_").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

fn rejected_id(detail: &str) -> Option<String> {
    let mut payload: Value = serde_json::from_str(detail).ok()?;
    for _ in 0..3 {
        let error = payload.get("error")?.as_object()?;
        let message = error.get("message")?.as_str()?;
        if error.get("code").and_then(Value::as_str) == Some("invalid_encrypted_content")
            && error.get("type").and_then(Value::as_str) == Some("invalid_request_error")
            && error.get("param") == Some(&Value::Null)
        {
            let id = message
                .strip_prefix(ERROR_PREFIX)?
                .strip_suffix(ERROR_SUFFIX)?;
            return valid_id(id).then(|| id.to_owned());
        }
        if ["code", "type", "param"]
            .iter()
            .any(|key| error.contains_key(*key))
        {
            return None;
        }
        payload = serde_json::from_str(message).ok()?;
    }
    None
}

fn compatible_summary(item: &Value) -> Option<String> {
    if item.get("type").and_then(Value::as_str) != Some("reasoning") {
        return None;
    }
    match item.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::Array(parts)) if parts.is_empty() => {}
        _ => return None,
    }
    let encrypted = item.get("encrypted_content")?.as_str()?;
    if encrypted.len() > MAX_ENVELOPE_BYTES {
        return None;
    }
    let encoded = encrypted.strip_prefix(ENVELOPE_PREFIX)?;
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
    let text = texts.join("\n\n");
    // This is decoding a known JSON envelope, not decrypting native reasoning.
    // Never replace an opaque native/unknown blob with a guessed summary.
    let raw = STANDARD.decode(encoded).ok()?;
    let envelope: CursorEnvelope = serde_json::from_slice(&raw).ok()?;
    if envelope.signature.is_empty() || envelope.text != text {
        return None;
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "rs_c1bc50c20fa1492e929b32afd1ccef85";
    fn error() -> Value {
        json!({"error":{"code":"invalid_encrypted_content","type":"invalid_request_error",
            "param":null,"message":format!("{ERROR_PREFIX}{ID}{ERROR_SUFFIX}")}})
    }
    fn outcome() -> Value {
        json!({"kind":"http_error","status_code":400,"upstream_status_code":400,
            "committed":false,"body":error().to_string()})
    }
    fn encrypted(text: &str) -> String {
        format!(
            "{ENVELOPE_PREFIX}{}",
            STANDARD.encode(json!({"signature":"fixture-signature","text":text}).to_string())
        )
    }
    fn item(id: &str) -> Value {
        json!({"type":"reasoning","id":id,"content":null,
            "summary":[{"type":"summary_text","text":"first\n"},{"type":"summary_text","text":"second 中文"}],
            "encrypted_content":encrypted("first\n\n\nsecond 中文")})
    }
    fn body() -> Value {
        json!({"model":"model","store":false,"input":[
            item(ID), item("rs_second"), item("rs_third"),
            {"type":"reasoning","id":"rs_native","encrypted_content":"gAAAAopaque","summary":[]},
            {"type":"function_call","name":"real","call_id":"call-real","arguments":"{}"},
            {"type":"function_call_output","call_id":"call-real","output":"result"},
            {"role":"user","content":"continue"}],
            "tools":[{"type":"function","name":"real"}],"reasoning":{"effort":"high"}})
    }

    #[test]
    fn converts_all_proven_envelopes_and_preserves_other_input() {
        let original = body();
        let (wire, count) = repair(&original.to_string(), &outcome()).unwrap();
        let fixed: Value = serde_json::from_str(&wire).unwrap();
        let mut expected = original;
        for index in 0..3 {
            expected["input"][index] = json!({"type":"message","role":"assistant","content":[{
                "type":"output_text","text":"Historical reasoning summary:\nfirst\n\n\nsecond 中文"}]});
        }
        assert_eq!(count, 3);
        assert_eq!(fixed, expected);
        assert!(repair(&wire, &outcome()).is_none());
    }

    #[test]
    fn accepts_only_exact_error_and_two_untyped_wrappers() {
        let mut wrapped = error();
        for _ in 0..3 {
            assert_eq!(rejected_id(&wrapped.to_string()).as_deref(), Some(ID));
            wrapped = json!({"error":{"message":wrapped.to_string()}});
        }
        assert!(rejected_id(&wrapped.to_string()).is_none());
        for field in ["code", "type", "param", "message"] {
            let mut other = error();
            other["error"][field] = json!("wrong");
            assert!(rejected_id(&other.to_string()).is_none());
            let mut other = error();
            other["error"].as_object_mut().unwrap().remove(field);
            assert!(rejected_id(&other.to_string()).is_none());
        }
        for id in ["rs_", "msg_x", "rs_bad/path", "rs_bad'quote", "rs_é"] {
            let other = json!({"error":{"code":"invalid_encrypted_content","type":"invalid_request_error",
                "param":null,"message":format!("{ERROR_PREFIX}{id}{ERROR_SUFFIX}")}});
            assert!(rejected_id(&other.to_string()).is_none());
        }
        for other in [
            json!({"error":{"type":"gateway_error","message":error().to_string()}}),
            json!({"error":{"message":"bad"},"debug":error()}),
        ] {
            assert!(rejected_id(&other.to_string()).is_none());
        }
        assert!(rejected_id(&format!("{ERROR_PREFIX}{ID}{ERROR_SUFFIX}")).is_none());
    }

    #[test]
    fn requires_stateless_precommit_http_400() {
        for (field, value) in [
            ("kind", json!("semantic_failure")),
            ("status_code", json!(404)),
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
    fn target_must_be_unique_and_provably_recoverable() {
        for (field, value) in [
            ("type", json!("item_reference")),
            ("id", json!("rs_other")),
            ("encrypted_content", json!("gAAAAopaque")),
            ("encrypted_content", Value::Null),
            ("content", json!([{"type":"reasoning_text","text":"raw"}])),
            ("content", json!("raw")),
            ("summary", json!([])),
            ("summary", Value::Null),
            ("summary", json!([{"type":"unknown","text":"first"}])),
            ("summary", json!([{"type":"summary_text","text":false}])),
        ] {
            let mut other = body();
            other["input"][0][field] = value;
            assert!(repair(&other.to_string(), &outcome()).is_none(), "{field}");
        }
        let mut other = body();
        other["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"item_reference","id":ID}));
        assert!(repair(&other.to_string(), &outcome()).is_none());
    }

    #[test]
    fn malformed_or_opaque_envelopes_never_activate_repair() {
        let text = "first\n\n\nsecond 中文";
        let mut cases = vec![
            "gAAAAopaque".to_owned(),
            "unknown:opaque".to_owned(),
            format!("{ENVELOPE_PREFIX}not-base64"),
            encrypted("different"),
            format!("{ENVELOPE_PREFIX}{}", "A".repeat(MAX_ENVELOPE_BYTES)),
        ];
        for envelope in [
            json!({"signature":"","text":text}),
            json!({"signature":42,"text":text}),
            json!({"text":text}),
            json!({"signature":"sig","text":text,"unknown":"preserve"}),
            json!({"signature":"sig","text":null}),
        ] {
            cases.push(format!(
                "{ENVELOPE_PREFIX}{}",
                STANDARD.encode(envelope.to_string())
            ));
        }
        // Duplicate JSON keys are ambiguous even if their final values match.
        cases.push(format!(
            "{ENVELOPE_PREFIX}{}",
            STANDARD.encode(r#"{"signature":"a","signature":"b","text":"first"}"#)
        ));
        for encrypted in cases {
            let mut other = body();
            other["input"][0]["encrypted_content"] = json!(encrypted);
            assert!(repair(&other.to_string(), &outcome()).is_none());
        }
    }

    #[test]
    fn non_target_unknown_and_duplicate_items_are_preserved() {
        let mut original = body();
        original["input"][1]["encrypted_content"] = json!("unknown:keep");
        original["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"item_reference","id":"rs_third"}));
        let (wire, count) = repair(&original.to_string(), &outcome()).unwrap();
        let fixed: Value = serde_json::from_str(&wire).unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            fixed["input"].as_array().unwrap()[1..],
            original["input"].as_array().unwrap()[1..]
        );
    }

    #[test]
    fn optional_empty_content_preserves_every_summary_character() {
        for content in [None, Some(json!([]))] {
            let mut other = body();
            let item = other["input"][0].as_object_mut().unwrap();
            if let Some(content) = content {
                item.insert("content".into(), content);
            } else {
                item.remove("content");
            }
            assert_eq!(repair(&other.to_string(), &outcome()).unwrap().1, 3);
        }
        let mut other = body();
        other["input"][0]["summary"] = json!([{"type":"summary_text","text":" "}]);
        other["input"][0]["encrypted_content"] = json!(encrypted(" "));
        assert!(repair(&other.to_string(), &outcome()).is_none());
    }
}
