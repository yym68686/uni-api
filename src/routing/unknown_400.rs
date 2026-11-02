//! Opt-in, request-scoped rescue of unclassified upstream 400s. This does not
//! assert that an upstream failure is a 502, or that inference is idempotent.
use serde_json::{Map, Value};
use std::collections::HashSet;
use url::Url;

pub(crate) const KEY: &str = "unknown_upstream_400_retries";

pub(crate) fn validate(value: Option<&Value>) -> Result<u8, String> {
    match value {
        None => Ok(0),
        Some(value) => value
            .as_u64()
            .filter(|n| *n <= 3)
            .map(|n| n as u8)
            .ok_or_else(|| format!("{KEY} must be an integer from 0 to 3")),
    }
}

pub(crate) fn configured(preferences: &Map<String, Value>) -> u8 {
    validate(preferences.get(KEY)).unwrap_or(0)
}

/// Restrict this experimental policy to inference without known hosted-tool,
/// storage, background-job, image/video or other side-effect contracts.
pub(crate) fn safe_request(endpoint: &str, payload: &Value) -> bool {
    let text_endpoint = matches!(
        endpoint,
        "/v1/chat/completions" | "/v1/responses" | "/v1/messages"
    ) || crate::protocols::gemini::route(endpoint)
        .is_some_and(|(_, action)| matches!(action, "generateContent" | "streamGenerateContent"));
    text_endpoint
        && payload.is_object()
        && ![
            "web_search_options",
            "mcp_servers",
            "code_execution",
            "previous_response_id",
            "conversation",
        ]
        .iter()
        .any(|key| payload.get(*key).is_some_and(|v| !v.is_null()))
        && !payload
            .get("background")
            .is_some_and(|v| v != &Value::Bool(false))
        && !payload
            .get("store")
            .is_some_and(|v| v != &Value::Bool(false))
        && payload.get("tools").is_none_or(|tools| {
            tools.as_array().is_some_and(|tools| {
                tools.iter().all(|tool| {
                    tool.get("type").and_then(Value::as_str) == Some("function")
                        || (endpoint == "/v1/messages"
                            && tool.get("type").is_none()
                            && tool.get("name").is_some_and(Value::is_string))
                        || (tool.as_object().is_some_and(|o| o.len() == 1)
                            && tool
                                .get("functionDeclarations")
                                .is_some_and(Value::is_array))
                })
            })
        })
}

// Only error-envelope fields are inspected. Never classify echoed input as an
// error, and never discard a specific validation or safety refusal.
pub(crate) fn unknown_error(error: &Value) -> bool {
    let Some(error) = error.as_object() else {
        return false;
    };
    if error
        .get("param")
        .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
    {
        return false;
    }
    let Some(message) = error
        .get("message")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    else {
        return false;
    };
    if error.get("type").is_some_and(|v| {
        !v.as_str().is_some_and(|s| {
            matches!(
                s,
                "invalid_request_error"
                    | "upstream_error"
                    | "server_error"
                    | "bad_request"
                    | "error"
            )
        })
    }) {
        return false;
    }
    let code = error.get("code");
    if code.is_some_and(|v| {
        !v.is_null()
            && v.as_u64() != Some(400)
            && !v.as_str().is_some_and(|s| {
                matches!(
                    s,
                    "400" | "bad_request" | "invalid_request_error" | "upstream_error"
                )
            })
    }) {
        return false;
    }
    if error.get("status").is_some_and(|v| {
        v.as_str()
            .is_none_or(|s| !matches!(s, "UNKNOWN" | "INTERNAL"))
    }) {
        return false;
    }
    let lower = message.to_lowercase();
    !message.trim_start().starts_with(['{', '['])
        // This generic envelope names no invalid field. Keep it behind the
        // operator's opt-in budget rather than treating "invalid" as proof of
        // a specific validation failure. Envelope guards still apply.
        && (message == "Invalid request parameter" || ![
            "invalid",
            "validation",
            "required",
            "missing",
            "must ",
            "cannot",
            "can't",
            "expected",
            "unsupported",
            "not supported",
            "malformed",
            "too long",
            "too large",
            "maximum",
            "context length",
            "safety",
            "policy",
            "refus",
            "blocked",
            "prohibit",
            "content_filter",
            "moderation",
            "permission",
            "已执行",
            "参数",
            "不能为空",
            "不合法",
            "无效",
            "缺少",
            "必须",
            "超出",
            "安全",
            "违规",
            "拒绝",
            "不允许",
            "不支持",
        ]
        .iter()
        .any(|marker| lower.contains(marker)))
        && error
            .get("details")
            .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty))
}

pub(crate) fn unknown_detail(detail: &str) -> bool {
    // Truncated/flattened/non-JSON diagnostics lack enough evidence to rule out
    // a field error. They are deliberately not rescued by this first version.
    serde_json::from_str::<Value>(detail).ok().is_some_and(|v| {
        v.get("error")
            .or_else(|| v.get("detail"))
            .map_or_else(|| unknown_error(&v), unknown_error)
    })
}

pub(crate) fn unknown_outcome(outcome: &Value) -> bool {
    if outcome.get("status_code").and_then(Value::as_u64) != Some(400)
        || outcome.get("committed").and_then(Value::as_bool) != Some(false)
    {
        return false;
    }
    match outcome.get("kind").and_then(Value::as_str) {
        Some("http_error") if outcome["upstream_status_code"] == 400 => outcome
            .get("body")
            .and_then(Value::as_str)
            .is_some_and(unknown_detail),
        Some("semantic_error" | "semantic_failure") if outcome["upstream_status_code"] == 200 => {
            match outcome.get("event_type").and_then(Value::as_str) {
                Some("error") => outcome.pointer("/payload/error").is_some_and(unknown_error),
                Some("response.failed") => outcome
                    .pointer("/payload/response/error")
                    .is_some_and(unknown_error),
                _ => false,
            }
        }
        _ => false,
    }
}

#[derive(Default)]
pub(crate) struct Budget {
    limit: u8,
    used: u8,
    restricted: bool,
    pending: bool,
    visited: HashSet<String>,
    cause: Option<(u16, String)>,
}

impl Budget {
    pub(crate) fn enabled(&self) -> bool {
        self.limit > 0
    }

    pub(crate) fn can_plan(&self) -> bool {
        !self.restricted || self.pending
    }

    pub(crate) fn set_cause(&mut self, raw_status: u16, attempt_id: String) {
        self.cause = Some((raw_status, attempt_id));
    }

    pub(crate) fn may_dispatch_payload(&self, safe: bool) -> bool {
        !self.restricted || safe
    }
    pub(crate) fn new(limit: u8, enabled: bool, endpoint: &str, payload: &Value) -> Self {
        Self {
            limit: if enabled && safe_request(endpoint, payload) {
                limit
            } else {
                0
            },
            ..Self::default()
        }
    }

    /// Existing exact rules remain independent and do not consume this budget.
    pub(crate) fn failure(&mut self, existing_retryable: bool, unknown: bool) -> bool {
        if existing_retryable {
            self.restricted = false;
            self.pending = false;
            return true;
        }
        if self.limit > 0 && unknown {
            self.restricted = true;
            self.pending = self.used < self.limit;
            return self.pending;
        }
        self.pending = false;
        false
    }

    pub(crate) fn may_dispatch(&self, url: &str) -> bool {
        !self.restricted
            || (self.pending && origin(url).is_some_and(|o| !self.visited.contains(&o)))
    }

    /// Call only after building a request, immediately before scheduling send.
    /// A rescue cannot spawn speculative hedges without another authorization.
    pub(crate) fn dispatch(&mut self, url: &str, request_id: &str, attempt_id: &str) {
        if !self.enabled() {
            return;
        }
        if let Some(origin) = origin(url) {
            self.visited.insert(origin);
        }
        if self.restricted && self.pending {
            self.pending = false;
            self.used += 1;
            crate::observability::request_trace::log(serde_json::json!({
                "kind":"log", "fugue_table":"app_events", "source":"uni-api-ember",
                "event":"unknown_upstream_400_retry", "event_type":"unknown_upstream_400_retry",
                "severity":"warn", "request_id":request_id, "attempt_id":attempt_id,
                "unknown_400_retry_limit":self.limit, "unknown_400_retry_used":self.used,
                "routing_status_code":400, "semantic_status_code":400,
                "attempt_status_code":self.cause.as_ref().map(|c| c.0),
                "original_attempt_id":self.cause.as_ref().map(|c| c.1.as_str()),
            }));
        }
    }

    pub(crate) fn used(&self) -> u8 {
        self.used
    }
}

fn origin(url: &str) -> Option<String> {
    let url = Url::parse(url).ok()?;
    Some(format!(
        "{}:{}",
        url.host_str()?,
        url.port_or_known_default()?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_configuration_and_safe_requests() {
        assert_eq!(validate(None), Ok(0));
        for n in 0..=3 {
            assert_eq!(validate(Some(&json!(n))), Ok(n));
        }
        for bad in [
            json!(4),
            json!(-1),
            json!(1.0),
            json!(null),
            json!(true),
            json!("1"),
        ] {
            assert!(validate(Some(&bad)).is_err());
        }
        assert!(safe_request(
            "/v1/responses",
            &json!({"store":false,"tools":[{"type":"function"}]})
        ));
        for payload in [
            json!({"store":true}),
            json!({"background":true}),
            json!({"tools":[{"type":"computer_use"}]}),
            json!({"tools":[{"type":"web_search"}]}),
        ] {
            assert!(!safe_request("/v1/responses", &payload));
        }
        assert!(!safe_request("/v1/images/generations", &json!({})));
    }

    #[test]
    fn only_unknown_envelopes_not_validation_safety_or_echoed_fields() {
        assert!(unknown_detail(
            r#"{"error":{"code":400,"message":"Relay backend hiccup"}}"#
        ));
        for error in [
            json!({"message":"Missing messages"}),
            json!({"message":"Content policy rejected"}),
            json!({"message":"Relay hiccup","param":"input"}),
            json!({"message":"Relay hiccup","code":"invalid_value"}),
            json!({"message":"Relay hiccup","status":"INVALID_ARGUMENT"}),
            json!({"message":"Relay hiccup","details":[{}]}),
        ] {
            assert!(!unknown_error(&error));
        }
        assert!(!unknown_detail(
            r#"{"input":{"message":"Relay backend hiccup"}}"#
        ));
        assert!(!unknown_detail("Relay backend hiccup"));
        let mut outcome = json!({"kind":"semantic_error","status_code":400,"upstream_status_code":200,"committed":false,"event_type":"error","payload":{"error":{"message":"Relay backend hiccup"}}});
        assert!(unknown_outcome(&outcome));
        outcome["committed"] = json!(true);
        assert!(!unknown_outcome(&outcome));
    }

    #[test]
    fn generic_parameter_message_remains_unknown_without_field_evidence() {
        let generic = json!({"type":"invalid_request_error","code":"invalid_request_error","message":"Invalid request parameter"});
        assert!(unknown_error(&generic));
        for param in [json!(null), json!("")] {
            let mut error = generic.clone();
            error["param"] = param;
            assert!(unknown_error(&error));
        }
        for (field, value) in [
            ("param", json!("input")),
            ("code", json!("invalid_value")),
            ("type", json!("safety_error")),
            ("status", json!("INVALID_ARGUMENT")),
            ("details", json!([{"field":"input"}])),
            ("message", json!("Invalid request parameter: input")),
            (
                "message",
                json!("Invalid request parameter (content policy)"),
            ),
            ("message", json!("Invalid request parameter\nMissing input")),
        ] {
            let mut error = generic.clone();
            error[field] = value;
            assert!(!unknown_error(&error), "{field}");
        }
        assert!(!unknown_detail("Invalid request parameter"));
        assert!(!unknown_detail(
            r#"{"error":{"message":"Invalid request parameter"}"#
        ));
    }

    #[test]
    fn independent_bounded_dispatch_budget_and_no_same_site_key_rotation() {
        let mut b = Budget::new(2, true, "/v1/responses", &json!({}));
        b.dispatch("https://a.test/v1", "r", "r1");
        assert!(b.failure(false, true));
        assert!(!b.may_dispatch("http://a.test:443/other-key"));
        assert!(b.may_dispatch("https://b.test/v1"));
        assert_eq!(b.used(), 0); // skipped routes do not consume rescue credit
        b.dispatch("https://b.test/v1", "r", "r2");
        assert!(!b.may_dispatch("https://c.test/v1")); // no speculative rescue hedge
        assert!(!b.can_plan()); // a hedge trigger must not exhaust the route cursor
        assert!(b.failure(true, false)); // ordinary 500 follows its existing policy
        assert_eq!(b.used(), 1);
        b.dispatch("https://c.test/v1", "r", "r3");
        assert!(b.failure(false, true));
        assert!(!b.may_dispatch("https://a.test/another-key"));
        b.dispatch("https://d.test/v1", "r", "r4");
        assert!(!b.failure(false, true));
        assert!(!b.may_dispatch("https://e.test/v1"));
        assert_eq!(b.used(), 2);
        let mut disabled = Budget::new(3, false, "/v1/responses", &json!({}));
        assert!(!disabled.failure(false, true));
    }
}
