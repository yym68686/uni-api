//! One request-scoped policy for every upstream HTTP/semantic 400.
//! Keep the persisted key so existing console settings survive upgrades.
use serde_json::{Map, Value};
use std::collections::HashSet;

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

/// Inspect transport/parser status and explicit commit boundary, never wording,
/// envelope shape, request features, or provider identity.
pub(crate) fn upstream_outcome(outcome: &Value) -> bool {
    outcome["status_code"] == 400
        && outcome["committed"] == false
        && matches!(
            outcome["kind"].as_str(),
            Some("http_error" | "semantic_error" | "semantic_failure")
        )
        && outcome["upstream_status_code"]
            .as_u64()
            .is_some_and(|status| (200..300).contains(&status) || status == 400)
}

#[derive(Default)]
pub(crate) struct Budget {
    limit: u8,
    used: u8,
    restricted: bool,
    pending: bool,
    sent: usize,
    visited: HashSet<String>,
    cause: Option<(u16, String)>,
}
impl Budget {
    pub(crate) fn new(limit: u8, enabled: bool) -> Self {
        Self {
            limit: if enabled { limit } else { 0 },
            ..Self::default()
        }
    }
    pub(crate) fn enabled(&self) -> bool {
        self.limit > 0
    }
    pub(crate) fn can_plan(&self) -> bool {
        !self.restricted || self.pending
    }
    pub(crate) fn pending(&self) -> bool {
        self.pending
    }
    pub(crate) fn sent(&self) -> usize {
        self.sent
    }
    /// Global 400 allowance is independent of AUTO_RETRY for other failures.
    /// Cooling/skipped channels do not spend the allowance.
    pub(crate) fn scan_limit(&self, ordinary_limit: usize, providers: usize) -> usize {
        if self.enabled() {
            ordinary_limit.max(providers)
        } else {
            ordinary_limit
        }
    }

    pub(crate) fn can_attempt(&self, ordinary_limit: usize) -> bool {
        self.sent < 100 && self.can_plan() && (self.sent < ordinary_limit || self.pending)
    }
    pub(crate) fn set_cause(&mut self, raw_status: u16, attempt_id: String) {
        self.cause = Some((raw_status, attempt_id));
    }
    pub(crate) fn failure(&mut self, ordinary_retryable: bool, upstream_400: bool) -> bool {
        if upstream_400 {
            self.restricted = true;
            self.pending = self.used < self.limit;
            self.pending
        } else {
            self.restricted = false;
            self.pending = false;
            ordinary_retryable
        }
    }
    pub(crate) fn may_dispatch(&self, channel: &str) -> bool {
        !self.restricted || (self.pending && !self.visited.contains(channel))
    }
    pub(crate) fn dispatch(&mut self, channel: &str, request_id: &str, attempt_id: &str) {
        self.sent = self.sent.saturating_add(1);
        self.visited.insert(channel.to_owned());
        if self.restricted && self.pending {
            self.pending = false;
            self.used += 1;
            crate::observability::request_trace::log(serde_json::json!({
                "kind":"log", "fugue_table":"app_events", "source":"uni-api-ember",
                "event":"upstream_400_retry", "event_type":"upstream_400_retry",
                "severity":"warn", "request_id":request_id, "attempt_id":attempt_id,
                "upstream_400_retry_limit":self.limit, "upstream_400_retry_used":self.used,
                "provider":channel, "routing_status_code":400, "semantic_status_code":400,
                "attempt_status_code":self.cause.as_ref().map(|c| c.0),
                "original_attempt_id":self.cause.as_ref().map(|c| c.1.as_str()),
            }));
        }
    }
    pub(crate) fn used(&self) -> u8 {
        self.used
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn settings_keep_the_existing_storage_contract() {
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
    }
    #[test]
    fn every_upstream_400_uses_status_and_commit_only() {
        for (kind, raw) in [
            ("http_error", 400),
            ("semantic_error", 200),
            ("semantic_error", 201),
            ("semantic_error", 299),
            ("semantic_failure", 200),
        ] {
            let mut o =
                json!({"kind":kind,"status_code":400,"upstream_status_code":raw,"committed":false});
            assert!(upstream_outcome(&o));
            o["body"] = json!("not JSON, truncated, validation or safety error");
            assert!(upstream_outcome(&o));
            o["committed"] = json!(true);
            assert!(!upstream_outcome(&o));
            o.as_object_mut().unwrap().remove("committed");
            assert!(!upstream_outcome(&o));
        }
        assert!(!upstream_outcome(
            &json!({"kind":"local_validation","status_code":400,"upstream_status_code":400,"committed":false})
        ));
        assert!(!upstream_outcome(
            &json!({"kind":"semantic_error","status_code":400,"upstream_status_code":500,"committed":false})
        ));
    }
    #[test]
    fn all_400s_share_one_budget_and_different_channels() {
        let mut b = Budget::new(3, true);
        b.dispatch("a", "r", "r1");
        for n in 1..=3 {
            assert!(b.failure(true, true));
            assert_eq!(b.scan_limit(1, 5), 5);
            assert!(!b.may_dispatch("a"));
            let channel = format!("b{n}");
            assert!(b.may_dispatch(&channel));
            b.dispatch(&channel, "r", &format!("r{}", n + 1));
            assert_eq!(b.used(), n);
            assert!(!b.can_plan());
        }
        assert!(!b.failure(true, true));
        assert!(!b.can_plan());
        assert!(b.failure(true, false));
        let mut zero = Budget::new(0, true);
        assert!(!zero.failure(true, true));
        let mut targeted = Budget::new(3, false);
        assert!(!targeted.failure(true, true));
    }
}
