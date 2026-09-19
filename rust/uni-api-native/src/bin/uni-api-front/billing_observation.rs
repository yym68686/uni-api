//! Receipt correlation facts only. No synchronous storage or billing decisions
//! run on the serving path. Every actually sent attempt emits once, even when a
//! hedge is cancelled or a response body fails after its headers arrived.
use axum::http::HeaderMap;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::channel_metrics::MetricKey;

#[derive(Debug, Default)]
struct Receipt {
    base: String,
    key_hash: String,
    ids: Vec<String>,
    status: u16,
    error_sha256: String,
}

#[derive(Debug)]
pub(crate) struct BillingAttempt {
    key: MetricKey,
    request_id: String,
    attempt_id: String,
    caller_key_id: String,
    started: OnceLock<i64>,
    receipt: Mutex<Receipt>,
}
impl BillingAttempt {
    pub(crate) fn new(
        key: MetricKey,
        request_id: String,
        attempt_id: String,
        caller_key_id: String,
    ) -> Self {
        Self {
            key,
            request_id,
            attempt_id,
            caller_key_id,
            started: OnceLock::new(),
            receipt: Mutex::new(Receipt::default()),
        }
    }
    pub(crate) fn start(&self) {
        self.started.get_or_init(now_ms);
    }
    pub(crate) fn target(&self, url: &str, secret: &str) {
        if let Ok(mut r) = self.receipt.lock() {
            r.base = billing_base(url);
            if !secret.is_empty() {
                r.key_hash = hex::encode(Sha256::digest(secret.as_bytes()));
            }
        }
    }
    pub(crate) fn headers(&self, headers: &HeaderMap, status: u16, secret: &str) {
        if let Ok(mut r) = self.receipt.lock() {
            r.status = status;
            // The direct sub2api server's identifier takes precedence over any
            // upstream/echoed ID. Do not associate one attempt with both bills.
            if let Some(id) = safe_header(headers, "x-client-request-id", secret) {
                r.ids = vec![format!("client:{id}")];
            }
        }
    }
    // Observe already-buffered HTTP errors without changing reads, retries or
    // charging decisions. The digest retains evidence without response content.
    pub(crate) fn error_body(&self, status: u16, body: &[u8]) {
        if (400..600).contains(&status) && !body.is_empty() {
            if let Ok(mut r) = self.receipt.lock() {
                if r.status == status {
                    r.error_sha256 = hex::encode(Sha256::digest(body));
                }
            }
        }
    }
    fn event(&self) -> Option<Value> {
        let started = self.started.get()?;
        let r = self.receipt.lock().ok()?;
        let mut event = crate::facts_s3::dispatch_event(
            &self.key,
            &self.request_id,
            &self.attempt_id,
            &self.caller_key_id,
            0.0,
        );
        event["kind"] = json!("billing");
        event["started_ms"] = json!(started);
        event["upstream_base"] = json!(r.base);
        event["upstream_key_hash"] = json!(r.key_hash);
        event["billing_request_ids"] = json!(r.ids);
        event["status"] = json!(r.status);
        event["upstream_error_sha256"] = json!(r.error_sha256);
        event.as_object_mut()?.remove("dispatch_ms");
        Some(event)
    }
}
impl Drop for BillingAttempt {
    fn drop(&mut self) {
        if let Some(event) = self.event() {
            if let Some(writer) = crate::facts_s3::global() {
                writer.enqueue(event);
            }
        }
    }
}
pub(crate) fn request_key(headers: &HeaderMap, url: &str) -> String {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, key)| key.trim().to_owned())
        .or_else(|| {
            headers
                .get("x-api-key")
                .or_else(|| headers.get("x-goog-api-key"))
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        })
        .or_else(|| {
            url::Url::parse(url).ok().and_then(|u| {
                u.query_pairs()
                    .find(|(k, _)| k == "key")
                    .map(|(_, v)| v.into_owned())
            })
        })
        .unwrap_or_default()
}
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
fn safe_header(headers: &HeaderMap, name: &str, secret: &str) -> Option<String> {
    // Duplicate headers are ambiguous; never pick one arbitrarily.
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?.trim();
    if values.next().is_some()
        || value.is_empty()
        || value.len() > 192
        || value.starts_with("sk-")
        || (!secret.is_empty() && value.contains(secret))
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
    {
        return None;
    }
    Some(value.to_owned())
}
pub(crate) fn billing_base(raw: &str) -> String {
    let Ok(mut url) = url::Url::parse(raw) else {
        return String::new();
    };
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return String::new();
    }
    url.set_query(None);
    url.set_fragment(None);
    let path = url.path().trim_end_matches('/').to_owned();
    let base = if let Some((base, _)) = path.split_once("/v1beta/models/") {
        base.to_owned()
    } else {
        let mut base = path.clone();
        for suffix in [
            "/v1/chat/completions",
            "/v1/responses/compact",
            "/v1/responses",
            "/v1/messages",
            "/v1beta/models",
            "/v1/embeddings",
            "/v1beta",
            "/v1",
        ] {
            if let Some(v) = path.strip_suffix(suffix) {
                base = v.to_owned();
                break;
            }
        }
        base
    };
    url.set_path(&base);
    url.as_str().trim_end_matches('/').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    fn attempt() -> BillingAttempt {
        BillingAttempt::new(
            MetricKey::new("p", "m", "up", "/v1/responses", true),
            "req".into(),
            "req-r1".into(),
            "key-fingerprint".into(),
        )
    }
    #[test]
    fn captures_direct_receipt_and_no_secret_or_body() {
        let a = attempt();
        assert!(a.event().is_none());
        a.start();
        a.target(
            "https://SITE.test:443/tenant/v1/responses?key=secret",
            "secret",
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-client-request-id", "site-uuid".parse().unwrap());
        headers.insert("x-request-id", "other-upstream".parse().unwrap());
        a.headers(&headers, 200, "secret");
        let e = a.event().unwrap();
        assert_eq!(e["kind"], "billing");
        assert_eq!(e["billing_request_ids"], json!(["client:site-uuid"]));
        assert_eq!(e["upstream_base"], "https://site.test/tenant");
        assert_eq!(
            e["upstream_key_hash"],
            hex::encode(Sha256::digest(b"secret"))
        );
        assert!(!e.to_string().contains("secret"));
        assert!(e.get("dispatch_ms").is_none());
        assert_eq!(e["attempt_id"], "req-r1");
    }
    #[test]
    fn independent_attempts_and_cancelled_attempts_do_not_reuse_ids() {
        let first = Arc::new(attempt());
        let clone = first.clone();
        first.start();
        first.target("https://site.test/v1/responses", "k");
        let mut h = HeaderMap::new();
        h.insert("x-client-request-id", "first-id".parse().unwrap());
        first.headers(&h, 200, "k");
        drop(first);
        assert_eq!(
            clone.event().unwrap()["billing_request_ids"],
            json!(["client:first-id"])
        );
        let second = attempt();
        second.start();
        assert_eq!(second.event().unwrap()["billing_request_ids"], json!([]));
        assert_eq!(second.event().unwrap()["status"], 0);
    }
    #[test]
    fn error_evidence_requires_matching_http_failure_and_retains_no_body() {
        let a = attempt();
        a.start();
        let h = HeaderMap::new();
        a.headers(&h, 403, "secret");
        let body = br#"{"code":"INSUFFICIENT_BALANCE","message":"Insufficient account balance"}"#;
        a.error_body(403, body);
        let e = a.event().unwrap();
        assert_eq!(
            e["upstream_error_sha256"],
            "7650844e093da022f530f60d448c6e401ca17d5efd97d38978acf34e43cdcb71"
        );
        assert!(!e.to_string().contains("INSUFFICIENT_BALANCE"));
        let b = attempt();
        b.start();
        b.headers(&h, 200, "secret");
        b.error_body(200, body);
        b.error_body(403, body);
        assert_eq!(b.event().unwrap()["upstream_error_sha256"], "");
    }
    #[test]
    fn rejects_ambiguous_and_secret_response_identifiers() {
        let mut h = HeaderMap::new();
        h.append("x-client-request-id", "one".parse().unwrap());
        h.append("x-client-request-id", "two".parse().unwrap());
        assert!(safe_header(&h, "x-client-request-id", "secret").is_none());
        h.insert("x-client-request-id", "prefix-secret".parse().unwrap());
        assert!(safe_header(&h, "x-client-request-id", "secret").is_none());
        assert_eq!(
            billing_base("https://site.test/base/v1beta/models/m:streamGenerateContent?key=hidden"),
            "https://site.test/base"
        );
    }
}
