//! A precise upstream compaction-state rejection may be channel-specific.
//! Keep the opaque state intact; never decode, delete or repair it here.

use serde_json::Value;

const MESSAGE: &str = "压缩状态无效或不属于当前调用方，请使用原调用凭据，或携带完整历史重试。";

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
    if error.get("type").and_then(Value::as_str) != Some("invalid_request_error")
        || error.get("code").and_then(Value::as_str) != Some("invalid_compaction_state")
        || error.get("message").and_then(Value::as_str) != Some(MESSAGE)
        || !error.get("param").is_none_or(Value::is_null)
    {
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
