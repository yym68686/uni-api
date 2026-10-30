//! A precise upstream compaction-state rejection may be channel-specific.
//! Keep the opaque state intact; never decode, delete or repair it here.

use serde_json::Value;

const MESSAGE: &str = "压缩状态无效或不属于当前调用方，请使用原调用凭据，或携带完整历史重试。";
const ENCRYPTED_PREFIX: &str = "The encrypted content for item ";
const ENCRYPTED_SUFFIX: &str =
    " could not be verified. Reason: Encrypted content could not be decrypted or parsed.";

pub(crate) fn is_channel_rejection(body: &str, outcome: &Value) -> bool {
    if !super::empty_name::is_uncommitted_400(outcome) {
        return false;
    }
    let Some(error) = outcome
        .get("body")
        .and_then(Value::as_str)
        .and_then(|detail| serde_json::from_str::<Value>(detail).ok())
        .and_then(|payload| payload.get("error").cloned())
    else {
        return false;
    };
    let canonical = error.get("type").and_then(Value::as_str) == Some("invalid_request_error")
        && error.get("code").and_then(Value::as_str) == Some("invalid_compaction_state")
        && error.get("message").and_then(Value::as_str) == Some(MESSAGE)
        && error.get("param").is_none_or(Value::is_null);
    let rejected_id = encrypted_compaction_id(&error);
    if !canonical && rejected_id.is_none() {
        return false;
    }
    let Ok(payload) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let Some(items) = payload.get("input").and_then(Value::as_array) else {
        return false;
    };
    let mut has_compaction = false;
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("compaction") {
            continue;
        }
        if item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_none_or(|state| state.trim().is_empty())
        {
            return false;
        }
        has_compaction = true;
    }
    has_compaction
        && rejected_id.is_none_or(|id| {
            let mut matching = items
                .iter()
                .filter(|item| item.get("id").and_then(Value::as_str) == Some(id));
            matching
                .next()
                .is_some_and(|item| item.get("type").and_then(Value::as_str) == Some("compaction"))
                && matching.next().is_none()
        })
}

fn encrypted_compaction_id(error: &Value) -> Option<&str> {
    if error.get("type").and_then(Value::as_str) != Some("upstream_error")
        || !error.get("code").is_none_or(Value::is_null)
        || !error.get("param").is_none_or(Value::is_null)
    {
        return None;
    }
    let id = error
        .get("message")?
        .as_str()?
        .strip_prefix(ENCRYPTED_PREFIX)?
        .strip_suffix(ENCRYPTED_SUFFIX)?;
    let uuid = id.strip_prefix("cmp_")?;
    (uuid.len() == 36
        && uuid.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        }))
    .then_some(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outcome() -> Value {
        json!({"kind":"http_error","status_code":400,"upstream_status_code":400,
            "committed":false,"body":json!({"error":{
                "type":"invalid_request_error","code":"invalid_compaction_state",
                "message":MESSAGE
            }}).to_string()})
    }

    fn body() -> Value {
        json!({"input":[{"type":"compaction","encrypted_content":"opaque-keep"},
            {"type":"reasoning","encrypted_content":"also-keep"},
            {"role":"user","content":"continue"}]})
    }

    const ID: &str = "cmp_01234567-89ab-cdef-0123-456789abcdef";

    fn identified_outcome() -> Value {
        let mut failure = outcome();
        failure["body"] = json!({"error":{"type":"upstream_error",
            "message":format!("{ENCRYPTED_PREFIX}{ID}{ENCRYPTED_SUFFIX}")}})
        .to_string()
        .into();
        failure
    }

    fn identified_body() -> Value {
        let mut payload = body();
        payload["input"][0]["id"] = ID.into();
        payload
    }

    #[test]
    fn identified_rejection_requires_exact_envelope_and_uncommitted_400() {
        let wire = identified_body().to_string();
        let failure = identified_outcome();
        assert!(is_channel_rejection(&wire, &failure));
        let error: Value = serde_json::from_str(failure["body"].as_str().unwrap()).unwrap();
        let mut nulls = error.clone();
        nulls["error"]["code"] = Value::Null;
        nulls["error"]["param"] = Value::Null;
        let mut other = failure.clone();
        other["body"] = nulls.to_string().into();
        assert!(is_channel_rejection(&wire, &other));
        for (field, value) in [
            ("type", json!("invalid_request_error")),
            ("type", Value::Null),
            ("code", json!("invalid_encrypted_content")),
            ("code", json!("")),
            ("code", json!(400)),
            ("param", json!("input")),
            ("param", json!("")),
            (
                "message",
                json!(format!("Quoted: {ENCRYPTED_PREFIX}{ID}{ENCRYPTED_SUFFIX}")),
            ),
            (
                "message",
                json!(format!("{ENCRYPTED_PREFIX}{ID}{ENCRYPTED_SUFFIX} extra")),
            ),
        ] {
            let mut detail = error.clone();
            detail["error"][field] = value;
            other["body"] = detail.to_string().into();
            assert!(!is_channel_rejection(&wire, &other));
        }
        for detail in [
            "{".to_owned(),
            error["error"].to_string(),
            json!({"error":{"message":error.to_string()}}).to_string(),
            json!({"detail":error}).to_string(),
        ] {
            other["body"] = detail.into();
            assert!(!is_channel_rejection(&wire, &other));
        }
        for (field, value) in [
            ("kind", json!("semantic_error")),
            ("status_code", json!(403)),
            ("upstream_status_code", json!(200)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut other = failure.clone();
            other[field] = value;
            assert!(!is_channel_rejection(&wire, &other));
        }
        for field in ["kind", "status_code", "upstream_status_code", "committed"] {
            let mut other = failure.clone();
            other.as_object_mut().unwrap().remove(field);
            assert!(!is_channel_rejection(&wire, &other));
        }
    }

    #[test]
    fn identified_rejection_requires_unique_actual_compaction_and_valid_uuid() {
        let failure = identified_outcome();
        for (field, value) in [
            ("id", json!("cmp_other")),
            ("id", Value::Null),
            ("type", json!("reasoning")),
            ("encrypted_content", Value::Null),
            ("encrypted_content", json!("")),
            ("encrypted_content", json!(" \n")),
            ("encrypted_content", json!(42)),
        ] {
            let mut payload = identified_body();
            payload["input"][0][field] = value;
            assert!(!is_channel_rejection(&payload.to_string(), &failure));
        }
        for field in ["id", "encrypted_content", "type"] {
            let mut payload = identified_body();
            payload["input"][0].as_object_mut().unwrap().remove(field);
            assert!(!is_channel_rejection(&payload.to_string(), &failure));
        }
        for duplicate in [
            json!({"type":"compaction","id":ID,"encrypted_content":"keep"}),
            json!({"type":"item_reference","id":ID}),
            json!({"type":"compaction","encrypted_content":""}),
        ] {
            let mut payload = identified_body();
            payload["input"].as_array_mut().unwrap().push(duplicate);
            assert!(!is_channel_rejection(&payload.to_string(), &failure));
        }
        for input in [Value::Null, json!("plain"), json!([])] {
            assert!(!is_channel_rejection(
                &json!({"input":input}).to_string(),
                &failure
            ));
        }
        assert!(!is_channel_rejection("{", &failure));
        for id in [
            "cmp_",
            "rs_0123456789abcdef",
            "cmp_0123456789abcdef0123456789abcdef",
            "cmp_01234567-89ab-cdef-0123-456789abcdeg",
            "cmp_01234567_89ab-cdef-0123-456789abcdef",
            "cmp_01234567-89ab-cdef-0123-456789abcde/",
            "cmp_01234567-89ab-cdef-0123-456789abcdé",
        ] {
            let mut payload = identified_body();
            payload["input"][0]["id"] = id.into();
            let mut other = failure.clone();
            other["body"] = json!({"error":{"type":"upstream_error",
                "message":format!("{ENCRYPTED_PREFIX}{id}{ENCRYPTED_SUFFIX}")}})
            .to_string()
            .into();
            assert!(!is_channel_rejection(&payload.to_string(), &other));
        }
        let mut payload = identified_body();
        payload["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"compaction",
            "id":"cmp_11111111-2222-3333-4444-555555555555","encrypted_content":"also-keep"}));
        let wire = payload.to_string();
        assert!(is_channel_rejection(&wire, &failure));
        assert_eq!(wire, payload.to_string());
    }

    #[test]
    fn exact_rejection_requires_uncommitted_upstream_400_and_compaction() {
        let wire = body().to_string();
        assert!(is_channel_rejection(&wire, &outcome()));
        for (field, value) in [
            ("kind", json!("semantic_error")),
            ("status_code", json!(403)),
            ("upstream_status_code", json!(200)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut failure = outcome();
            failure[field] = value;
            assert!(!is_channel_rejection(&wire, &failure));
        }
        for field in ["kind", "status_code", "upstream_status_code", "committed"] {
            let mut failure = outcome();
            failure.as_object_mut().unwrap().remove(field);
            assert!(!is_channel_rejection(&wire, &failure));
        }
        for input in [
            json!([]),
            json!("plain input"),
            Value::Null,
            json!([{"type":"reasoning","encrypted_content":"opaque"}]),
            json!([{"role":"user","content":"compaction","encrypted_content":"echo"}]),
        ] {
            assert!(!is_channel_rejection(
                &json!({"input":input}).to_string(),
                &outcome()
            ));
        }
        for state in [Value::Null, json!(""), json!(" \n"), json!(1), json!({})] {
            let mut payload = body();
            payload["input"][0]["encrypted_content"] = state.clone();
            assert!(!is_channel_rejection(&payload.to_string(), &outcome()));
            let mut payload = body();
            payload["input"]
                .as_array_mut()
                .unwrap()
                .push(json!({"type":"compaction","encrypted_content":state}));
            assert!(!is_channel_rejection(&payload.to_string(), &outcome()));
        }
        let mut payload = body();
        payload["input"][0]
            .as_object_mut()
            .unwrap()
            .remove("encrypted_content");
        assert!(!is_channel_rejection(&payload.to_string(), &outcome()));
        assert!(!is_channel_rejection("{", &outcome()));
    }

    #[test]
    fn exact_error_does_not_generalize_auth_or_state_validation() {
        let wire = body().to_string();
        let error: Value = serde_json::from_str(outcome()["body"].as_str().unwrap()).unwrap();
        let mut null_param = error.clone();
        null_param["error"]["param"] = Value::Null;
        let mut failure = outcome();
        failure["body"] = null_param.to_string().into();
        assert!(is_channel_rejection(&wire, &failure));
        for (field, value) in [
            ("type", json!("authentication_error")),
            ("type", Value::Null),
            ("code", json!("invalid_encrypted_content")),
            ("code", Value::Null),
            ("message", json!("Invalid compaction state")),
            ("message", json!(format!("Quoted: {MESSAGE}"))),
            ("message", json!(format!("{MESSAGE} Extra"))),
            ("param", json!("input")),
            ("param", json!("")),
            ("param", json!(0)),
        ] {
            let mut error = error.clone();
            error["error"][field] = value;
            failure["body"] = error.to_string().into();
            assert!(!is_channel_rejection(&wire, &failure));
        }
        for detail in [
            MESSAGE.to_owned(),
            "{".to_owned(),
            error["error"].to_string(),
            json!({"detail":error["error"]}).to_string(),
            json!({"error":{"message":error.to_string()}}).to_string(),
            json!({"error":{"message":"Invalid input"},"input":error}).to_string(),
        ] {
            failure["body"] = detail.into();
            assert!(!is_channel_rejection(&wire, &failure));
        }
    }
}
