//! Diagnostic facts only; no synchronous I/O and no routing decisions.
use axum::body::{Body, Bytes, HttpBody};
use axum::http::{HeaderMap, Response};
use http_body::{Frame, SizeHint};
use serde_json::{json, Value};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// Reuse existing bounded structured events, copying only diagnostic metadata.
// Never export role, headers, payloads, prompts, raw response bodies or secrets.
pub(crate) fn record_log(log: &Value) {
    let stage = log
        .get("event_type")
        .or_else(|| log.get("event"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(
        stage,
        "routing_attempt"
            | "upstream_attempt"
            | "rust_request_spool"
            | "responses_empty_name_repair"
            | "responses_missing_item_repair"
            | "responses_encrypted_content_repair"
            | "responses_heartbeat_repair"
    ) {
        return;
    }
    let request = log
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if request.is_empty() {
        return;
    }
    let mut detail = json!({});
    for name in [
        "attempt_index",
        "attempt_outcome",
        "attempt_status_code",
        "semantic_status_code",
        "skip_reason",
        "status_origin",
        "error_sha256",
        "duration_ms",
        "body_bytes",
        "resource_wait_ms",
        "failure_resource",
        "original_attempt_id",
        "tool_pairs_converted",
        "reasoning_items_converted",
        "heartbeat_items_converted",
        "error",
    ] {
        if let Some(value) = log.get(name).filter(|v| !v.is_null()) {
            detail[name] = value.clone();
        }
    }
    emit(
        json!({"request_id":request,"trace_id":log.get("trace_id"),"attempt_id":log.get("attempt_id"),"provider":log.get("provider"),"model":log.get("model"),"upstream_model":log.get("actual_model"),"endpoint":log.get("path"),"stream":log.get("streaming"),"stage":stage,"trace_detail":detail}),
    );
}

pub(crate) fn log(event: Value) {
    record_log(&event);
    eprintln!("{event}");
}

fn emit(mut value: Value) {
    if let Some(writer) = crate::storage::s3::global() {
        value["schema"] = json!(1);
        value["kind"] = json!("trace");
        value["event_id"] = json!(crate::storage::s3::new_event_id("trace"));
        value["instance_id"] = json!(crate::storage::s3::instance_id());
        value["at_ms"] = json!(now_ms());
        writer.enqueue(value);
    }
}

// Error fields are kept separate from arbitrary echoed body fields. Known
// credentials are removed before token redaction; unstructured bodies stay out.
pub(crate) fn error_fields(body: &[u8], secrets: &[&str]) -> Value {
    let Ok(root) = serde_json::from_slice::<Value>(body) else {
        return json!({});
    };
    let error = root
        .pointer("/response/error")
        .or_else(|| root.get("error"))
        .unwrap_or(&root);
    let mut out = json!({});
    for name in ["code", "type", "message"] {
        if let Some(text) = error.get(name).and_then(Value::as_str) {
            let mut clean = text.to_owned();
            for secret in secrets.iter().filter(|s| !s.is_empty()) {
                clean = clean.replace(secret, "[redacted]");
            }
            // Strip URLs (including credential query strings), email addresses,
            // key/token-shaped words and long opaque identifiers.
            let mut redact_next = false;
            clean = clean
                .split_whitespace()
                .map(|word| {
                    let lower = word.to_ascii_lowercase();
                    let previous = redact_next;
                    redact_next = matches!(
                        lower.trim_matches([':', '"', '\'']),
                        "bearer" | "authorization" | "api_key" | "token" | "password"
                    );
                    if previous
                        || redact_next
                        || word.contains("sk-")
                        || lower.contains("api_key=")
                        || lower.contains("token=")
                        || lower.contains("password=")
                        || word.contains("://")
                        || word.contains('@')
                        || word.len() > 96
                    {
                        "[redacted]"
                    } else {
                        word
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            out[format!("error_{name}")] = json!(clean
                .chars()
                .filter(|c| !c.is_control())
                .take(1024)
                .collect::<String>());
        }
    }
    out
}

pub(crate) struct RequestTrace {
    request_id: String,
    trace_id: String,
    endpoint: String,
    key_id: String,
    caller_secret: String,
    started: Instant,
    status: u16,
    done: bool,
}

impl RequestTrace {
    pub(crate) fn new(request_id: String, headers: &HeaderMap, endpoint: &str) -> Self {
        let key = crate::routing::access::extract_api_key(headers).unwrap_or_default();
        let trace_id = crate::upstream::generic::trace_id(headers, &request_id);
        let trace = Self {
            request_id,
            trace_id,
            endpoint: endpoint.to_owned(),
            key_id: if key.is_empty() {
                String::new()
            } else {
                crate::routing::catalog::key_id(&key)
            },
            started: Instant::now(),
            caller_secret: key,
            status: 0,
            done: false,
        };
        trace.record("request_received", json!({}));
        trace
    }
    fn record(&self, stage: &str, detail: Value) {
        emit(
            json!({"request_id":self.request_id,"trace_id":self.trace_id,"endpoint":self.endpoint,"key_id":self.key_id,"stage":stage,"status":self.status,"duration_ms":self.started.elapsed().as_secs_f64()*1000.,"trace_detail":detail}),
        );
    }
    pub(crate) fn response(mut self, response: Response<Body>) -> Response<Body> {
        self.status = response.status().as_u16();
        self.record("response_headers", json!({}));
        let (parts, inner) = response.into_parts();
        if inner.is_end_stream() {
            self.finish("response_body_finished");
        }
        Response::from_parts(
            parts,
            Body::new(ObservedBody {
                inner,
                trace: self,
                error_body: Vec::new(),
            }),
        )
    }
    fn finish(&mut self, stage: &str) {
        if !self.done {
            self.done = true;
            self.record(stage, json!({}));
        }
    }
}
impl Drop for RequestTrace {
    fn drop(&mut self) {
        self.finish("downstream_closed");
    }
}

struct ObservedBody {
    inner: Body,
    trace: RequestTrace,
    error_body: Vec<u8>,
}
impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let result = Pin::new(&mut self.inner).poll_frame(cx);
        if self.trace.status >= 400 && !self.trace.done {
            if let Poll::Ready(Some(Ok(frame))) = &result {
                if let Some(data) = frame.data_ref() {
                    let remaining = 4096usize.saturating_sub(self.error_body.len());
                    self.error_body
                        .extend_from_slice(&data[..data.len().min(remaining)]);
                }
            }
            if matches!(&result, Poll::Ready(None))
                || (matches!(&result, Poll::Ready(Some(_))) && self.inner.is_end_stream())
            {
                let detail = error_fields(&self.error_body, &[&self.trace.caller_secret]);
                if detail.as_object().is_some_and(|v| !v.is_empty()) {
                    self.trace.record("gateway_error", detail);
                }
            }
        }
        match &result {
            Poll::Ready(None) => self.trace.finish("response_body_finished"),
            Poll::Ready(Some(Err(_))) => self.trace.finish("response_body_error"),
            Poll::Ready(Some(Ok(_))) if self.inner.is_end_stream() => {
                self.trace.finish("response_body_finished")
            }
            _ => {}
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn errors_exclude_body_fields_and_credentials() {
        let error = error_fields(br#"{"error":{"code":"INSUFFICIENT_BALANCE","message":"Insufficient account balance secret-value sk-secret https://test/?token=secret Bearer opaque-secret"},"input":"private prompt","api_key":"secret-value"}"#, &["secret-value"]);
        let text = error.to_string();
        assert_eq!(error["error_code"], "INSUFFICIENT_BALANCE");
        for forbidden in [
            "private prompt",
            "secret-value",
            "sk-secret",
            "https://",
            "opaque-secret",
        ] {
            assert!(!text.contains(forbidden));
        }
        assert!(error_fields(b"not json with secrets", &[])
            .as_object()
            .unwrap()
            .is_empty());
    }
    #[tokio::test]
    async fn body_observation_preserves_bytes_status_and_headers() {
        let trace = RequestTrace::new("fixture".into(), &HeaderMap::new(), "/v1/responses");
        let response = Response::builder()
            .status(403)
            .header("x-test", "yes")
            .body(Body::from("unchanged"))
            .unwrap();
        let observed = trace.response(response);
        assert_eq!(observed.status(), 403);
        assert_eq!(observed.headers()["x-test"], "yes");
        assert_eq!(
            axum::body::to_bytes(observed.into_body(), 100)
                .await
                .unwrap(),
            "unchanged"
        );
    }

    #[tokio::test]
    async fn body_observation_preserves_trailers() {
        struct Frames(std::collections::VecDeque<Frame<Bytes>>);
        impl HttpBody for Frames {
            type Data = Bytes;
            type Error = std::convert::Infallible;
            fn poll_frame(
                mut self: Pin<&mut Self>,
                _: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
                Poll::Ready(self.0.pop_front().map(Ok))
            }
            fn is_end_stream(&self) -> bool {
                self.0.is_empty()
            }
        }
        let mut trailers = HeaderMap::new();
        trailers.insert("x-checksum", "verified".parse().unwrap());
        let frames = Frames(std::collections::VecDeque::from([
            Frame::data(Bytes::from_static(b"bytes")),
            Frame::trailers(trailers),
        ]));
        let trace = RequestTrace::new("trailers".into(), &HeaderMap::new(), "/v1/responses");
        let mut body = trace.response(Response::new(Body::new(frames))).into_body();
        let first = futures_util::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.data_ref().unwrap(), "bytes");
        let last = futures_util::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(last.trailers_ref().unwrap()["x-checksum"], "verified");
        assert!(body.is_end_stream());
    }
}
