//! A relay that only serves standard cannot honor a valid priority request.
//! Keep the requested tier intact and try another channel; never downgrade it.
use serde_json::Value;

pub(crate) const FAILURE_REASON: &str = "service_tier_incompatible";
pub(crate) const EVENT: &str = "service_tier_fallback";
const MESSAGE: &str = "oai-basispoints supports only the standard service tier; omit service_tier or use auto/default; Fast/priority is not supported";

pub(crate) fn rejected_request(status: u16, detail: &str, request: &[u8]) -> bool {
    if status != 400 {
        return false;
    }
    let Ok(response) = serde_json::from_str::<Value>(detail) else {
        return false;
    };
    let Some(error) = response.get("error").and_then(Value::as_object) else {
        return false;
    };
    if error.get("type").and_then(Value::as_str) != Some("upstream_error")
        || error.get("code").and_then(Value::as_str) != Some("unsupported_service_tier")
        || error.get("message").and_then(Value::as_str) != Some(MESSAGE)
        || !error.get("param").is_none_or(Value::is_null)
        || !error
            .get("details")
            .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
    {
        return false;
    }
    serde_json::from_slice::<Value>(request)
        .ok()
        .is_some_and(|wire| wire.get("service_tier").and_then(Value::as_str) == Some("priority"))
}

pub(crate) fn rejected_outcome(request: &str, outcome: &Value) -> bool {
    outcome.get("kind").and_then(Value::as_str) == Some("http_error")
        && outcome.get("status_code").and_then(Value::as_u64) == Some(400)
        && outcome.get("upstream_status_code").and_then(Value::as_u64) == Some(400)
        && outcome.get("committed").and_then(Value::as_bool) == Some(false)
        && outcome
            .get("body")
            .and_then(Value::as_str)
            .is_some_and(|body| rejected_request(400, body, request.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn error() -> Value {
        json!({"error":{"type":"upstream_error","code":"unsupported_service_tier","message":MESSAGE}})
    }

    #[test]
    fn exact_relay_envelope_and_actual_priority_wire_are_required() {
        let wire = br#"{"service_tier":"priority"}"#;
        assert!(rejected_request(400, &error().to_string(), wire));
        for (key, value) in [
            ("type", json!("invalid_request_error")),
            ("code", json!("invalid_value")),
            ("param", json!("service_tier")),
            ("param", json!("")),
            ("message", json!(format!("{MESSAGE}."))),
            ("message", json!(format!("Invalid input: {MESSAGE}"))),
            ("details", json!([{"reason":"validation"}])),
        ] {
            let mut e = error();
            e["error"][key] = value;
            assert!(!rejected_request(400, &e.to_string(), wire));
        }
        for tier in [
            json!(null),
            json!("default"),
            json!("auto"),
            json!("fast"),
            json!("PRIORITY"),
            json!(1),
        ] {
            assert!(!rejected_request(
                400,
                &error().to_string(),
                json!({"service_tier":tier}).to_string().as_bytes()
            ));
        }
        for body in [
            b"{}".as_slice(),
            b"not json",
            br#"{"input":{"service_tier":"priority"}}"#,
        ] {
            assert!(!rejected_request(400, &error().to_string(), body));
        }
        for e in [
            json!({"detail":error()["error"]}),
            json!({"error":{"message":error().to_string()}}),
            json!({"input":error()}),
        ] {
            assert!(!rejected_request(400, &e.to_string(), wire));
        }
        assert!(!rejected_request(400, MESSAGE, wire));
        for status in [200, 401, 403, 404, 422, 429, 500, 502] {
            assert!(!rejected_request(status, &error().to_string(), wire));
        }
        let mut e = error();
        e["error"]["param"] = Value::Null;
        assert!(rejected_request(400, &e.to_string(), wire));
    }

    #[test]
    fn only_uncommitted_http_400_outcomes_match() {
        let request = r#"{"service_tier":"priority"}"#;
        let outcome = json!({"kind":"http_error","status_code":400,"upstream_status_code":400,"committed":false,"body":error().to_string()});
        assert!(rejected_outcome(request, &outcome));
        for (key, value) in [
            ("kind", json!("semantic_error")),
            ("upstream_status_code", json!(200)),
            ("status_code", json!(502)),
            ("committed", json!(true)),
            ("committed", Value::Null),
        ] {
            let mut o = outcome.clone();
            o[key] = value;
            assert!(!rejected_outcome(request, &o));
        }
    }
}
