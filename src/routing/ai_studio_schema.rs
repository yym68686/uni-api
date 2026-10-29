//! Only the relay's explicit examples capability rejection, never schema validation.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "ai_studio_examples_incompatible";

pub(crate) fn rejected_examples(status: u16, detail: &str) -> bool {
    if status != 400 {
        return false;
    }
    let Ok(payload) = serde_json::from_str::<Value>(detail) else {
        return false;
    };
    let Some(error) = payload.get("error").and_then(Value::as_object) else {
        return false;
    };
    if error.get("type").and_then(Value::as_str) != Some("invalid_request_error")
        || error.get("code").and_then(Value::as_str) != Some("invalid_request")
        || !error
            .get("param")
            .is_none_or(|p| p.is_null() || p.as_str() == Some(""))
    {
        return false;
    }
    let Some(message) = error.get("message").and_then(Value::as_str) else {
        return false;
    };
    if message.len() > 4096 {
        return false;
    }
    let Some(location) = message
        .strip_prefix("AI Studio 请求参数无效: 编码 function declaration ")
        .and_then(|s| {
            s.strip_suffix(": AI Studio 协议能力尚无成功现场证据: JSON schema 字段 examples")
        })
    else {
        return false;
    };
    let Some((index, path)) = location.split_once(": parameters: ***.") else {
        return false;
    };
    if index.is_empty()
        || index.len() > 1 && index.starts_with('0')
        || !index.bytes().all(|b| b.is_ascii_digit())
        || index.parse::<u32>().is_err()
    {
        return false;
    }
    // Only the observed dot-separated property path grammar. No echoed prose,
    // alternate masks, arbitrary JSON pointers, other schema fields or suffixes.
    let mut segments = path.split('.');
    let mut properties = 0;
    while let Some(segment) = segments.next() {
        if segment != "properties" {
            return false;
        }
        let Some(name) = segments.next() else {
            return false;
        };
        let mut bytes = name.bytes();
        if !bytes
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
            || !bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return false;
        }
        properties += 1;
    }
    properties > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(location: &str) -> String {
        format!("AI Studio 请求参数无效: 编码 function declaration {location}: AI Studio 协议能力尚无成功现场证据: JSON schema 字段 examples")
    }
    fn error() -> Value {
        json!({"error":{"type":"invalid_request_error","code":"invalid_request","param":"",
            "message":message("4: parameters: ***.properties.duration_min")}})
    }

    #[test]
    fn exact_capability_envelope_accepts_canonical_indexed_property_paths() {
        for location in [
            "0: parameters: ***.properties.duration_min",
            "4: parameters: ***.properties.duration_min",
            "17: parameters: ***.properties.command",
            "35: parameters: ***.properties.options.properties.duration_min",
        ] {
            let mut e = error();
            e["error"]["message"] = json!(message(location));
            for param in [Some(Value::Null), Some(json!("")), None] {
                let object = e["error"].as_object_mut().unwrap();
                if let Some(param) = param {
                    object.insert("param".into(), param);
                } else {
                    object.remove("param");
                }
                assert!(rejected_examples(400, &e.to_string()));
            }
        }
    }

    #[test]
    fn unrelated_statuses_envelopes_and_validation_remain_unchanged() {
        for status in [200, 401, 403, 404, 413, 422, 429, 500, 502, 503] {
            assert!(!rejected_examples(status, &error().to_string()));
        }
        for (field, value) in [
            ("type", json!("upstream_error")),
            ("type", Value::Null),
            ("code", json!("invalid_type")),
            ("code", json!(400)),
            ("code", Value::Null),
            ("param", json!("tools")),
            ("param", json!(false)),
            (
                "message",
                json!(format!(
                    "{}.",
                    error()["error"]["message"].as_str().unwrap()
                )),
            ),
            (
                "message",
                json!(format!(
                    "Quoted: {}",
                    error()["error"]["message"].as_str().unwrap()
                )),
            ),
            ("message", json!("examples must be an array")),
        ] {
            let mut e = error();
            e["error"][field] = value;
            assert!(!rejected_examples(400, &e.to_string()));
        }
        for e in [
            json!({"detail":error()["error"]}),
            json!({"error":{"message":error().to_string()}}),
            json!({"error":{"message":"invalid input"},"debug":error()}),
            error()["error"].clone(),
            json!({"input":error()}),
        ] {
            assert!(!rejected_examples(400, &e.to_string()));
        }
        assert!(!rejected_examples(
            400,
            error()["error"]["message"].as_str().unwrap()
        ));
    }

    #[test]
    fn malformed_or_different_capability_messages_do_not_match() {
        for location in [
            ": parameters: ***.properties.duration_min",
            "04: parameters: ***.properties.duration_min",
            "-1: parameters: ***.properties.duration_min",
            "4294967296: parameters: ***.properties.duration_min",
            "4: parameters: **.properties.duration_min",
            "4: parameters: ***.properties.",
            "4: parameters: ***.properties.0",
            "4: parameters: ***.properties.duration_min ignored",
            "4: parameters: ***.properties.duration_min.examples",
            "4: parameters: ***.properties.duration_min\n",
            "4: parameters: ***.items.duration_min",
        ] {
            let mut e = error();
            e["error"]["message"] = json!(message(location));
            assert!(!rejected_examples(400, &e.to_string()));
        }
        for changed in [
            message("4: parameters: ***.properties.duration_min")
                .replace("字段 examples", "字段 default"),
            message("4: parameters: ***.properties.duration_min")
                .replace("尚无成功现场证据", "校验失败"),
            format!(
                "{}\nmore validation",
                message("4: parameters: ***.properties.duration_min")
            ),
        ] {
            let mut e = error();
            e["error"]["message"] = json!(changed);
            assert!(!rejected_examples(400, &e.to_string()));
        }
    }
}
