//! Bounded, byte-preserving observation of native SSE passthrough responses.
use crate::fact_usage::FactUsage;
use crate::provider_stream::{StreamOutcome, Translation};
use axum::{body::Body, http::Response};
use futures_util::StreamExt;
use serde_json::Value;
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

const MAX_LINE: usize = 1 << 20;

pub(crate) fn observe(
    response: reqwest::Response,
    started: Instant,
    control_routing: bool,
    idle: Option<Duration>,
) -> Translation {
    let status = response.status();
    let headers = crate::proxy::filtered_response_headers(response.headers());
    let (tx, rx) = oneshot::channel();
    let mut observer = Observer::new();
    observer.start = started;
    observer.observational_only = !control_routing;
    observer.sender = Some(tx);
    let body = futures_util::stream::unfold(
        (response.bytes_stream(), observer),
        move |(mut upstream, mut observer)| async move {
            let next = if let Some(idle) = idle {
                match tokio::time::timeout(idle, upstream.next()).await {
                    Ok(next) => next.map(|chunk| chunk.map_err(io::Error::other)),
                    Err(_) => Some(Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "upstream stream idle timeout exceeded",
                    ))),
                }
            } else {
                upstream
                    .next()
                    .await
                    .map(|chunk| chunk.map_err(io::Error::other))
            };
            match next {
                Some(result) => {
                    match &result {
                        Ok(bytes) => observer.feed(bytes),
                        Err(_) => {
                            observer.failed = true;
                            observer.ended = true;
                        }
                    }
                    Some((result, (upstream, observer)))
                }
                None => {
                    observer.ended = true;
                    None
                }
            }
        },
    );
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Translation {
        response,
        outcome: rx,
    }
}

struct Observer {
    observational_only: bool,
    line: Vec<u8>,
    overflow: bool,
    usage: FactUsage,
    first_output_ms: Option<f64>,
    response_created_ms: Option<f64>,
    first_text_ms: Option<f64>,
    start: Instant,
    terminal: bool,
    failed: bool,
    ended: bool,
    sender: Option<oneshot::Sender<StreamOutcome>>,
}

impl Observer {
    fn new() -> Self {
        Self {
            observational_only: true,
            line: Vec::new(),
            overflow: false,
            usage: Default::default(),
            first_output_ms: None,
            response_created_ms: None,
            first_text_ms: None,
            start: Instant::now(),
            terminal: false,
            failed: false,
            ended: false,
            sender: None,
        }
    }
    fn feed(&mut self, bytes: &[u8]) {
        for piece in bytes.split_inclusive(|b| *b == b'\n') {
            if !self.overflow {
                if self.line.len().saturating_add(piece.len()) > MAX_LINE {
                    self.line.clear();
                    self.overflow = true;
                } else {
                    self.line.extend_from_slice(piece);
                }
            }
            if piece.last() == Some(&b'\n') {
                if !self.overflow {
                    self.read_line();
                }
                self.line.clear();
                self.overflow = false;
            }
        }
    }
    fn read_line(&mut self) {
        let Ok(line) = std::str::from_utf8(&self.line) else {
            return;
        };
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
            return;
        };
        if data == "[DONE]" {
            self.terminal = true;
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return;
        };
        self.usage.merge(FactUsage::from_usage(
            value
                .get("usage")
                .or_else(|| value.pointer("/message/usage"))
                .or_else(|| value.pointer("/response/usage")),
        ));
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let elapsed = self.start.elapsed().as_secs_f64() * 1000.0;
        if kind == "response.created" && self.response_created_ms.is_none() {
            self.response_created_ms = Some(elapsed);
        }
        if kind == "response.output_text.delta"
            && self.first_text_ms.is_none()
            && value
                .get("delta")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        {
            self.first_text_ms = Some(elapsed);
        }

        if kind == "error"
            || kind == "response.failed"
            || value.get("error").is_some_and(|v| !v.is_null())
            || value.pointer("/response/status").and_then(Value::as_str) == Some("failed")
        {
            self.failed = true;
        }
        if matches!(
            kind,
            "message_stop" | "response.completed" | "response.incomplete"
        ) || value
            .pointer("/choices/0/finish_reason")
            .is_some_and(|v| !v.is_null())
        {
            self.terminal = true;
        }
        let semantic = [
            "/delta/text",
            "/delta/thinking",
            "/delta/partial_json",
            "/choices/0/delta/content",
            "/choices/0/delta/reasoning_content",
        ]
        .iter()
        .any(|p| {
            value
                .pointer(p)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        }) || value
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
            || (matches!(
                kind,
                "response.output_text.delta" | "response.function_call_arguments.delta"
            ) && value
                .get("delta")
                .and_then(Value::as_str)
                .is_some_and(|v| !v.is_empty()));
        if semantic && self.first_output_ms.is_none() {
            self.first_output_ms = Some(self.start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    fn outcome(&self) -> StreamOutcome {
        let success = self.ended && self.terminal && !self.failed;
        StreamOutcome {
            // Existing passthrough billing used no parsed usage. Only facts
            // adopt these observations; keep billing behavior unchanged.
            usage: (0, 0, 0),
            fact_usage: self.usage.clone(),
            first_output_ms: self.first_output_ms,
            response_created_ms: self.response_created_ms,
            first_text_ms: self.first_text_ms,
            success,
            observational_only: self.observational_only,
            status_code: if success {
                200
            } else if !self.ended {
                499
            } else {
                502
            },
            detail: if success {
                String::new()
            } else if !self.ended {
                "downstream disconnected".into()
            } else {
                "passthrough stream failed or ended without a terminal event".into()
            },
        }
    }
}
impl Drop for Observer {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(self.outcome());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn observer_preserves_wire_headers_and_waits_for_body_completion() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let wire = b": keepalive\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":60}}}\n\ndata: [DONE]\n\n";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let upstream = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut input = [0u8; 4096];
            let n = socket.read(&mut input).await.unwrap();
            assert!(n > 0);
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Test: same\r\nContent-Length: {}\r\n\r\n",wire.len()).as_bytes()).await.unwrap();
            socket.write_all(wire).await.unwrap();
        });
        let source = reqwest::Client::new()
            .get(format!("http://{addr}"))
            .send()
            .await
            .unwrap();
        let Translation {
            response,
            mut outcome,
        } = observe(
            source,
            Instant::now() - std::time::Duration::from_millis(100),
            false,
            None,
        );
        assert_eq!(response.headers()["x-test"], "same");
        assert!(matches!(
            outcome.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let bytes = axum::body::to_bytes(response.into_body(), MAX_LINE)
            .await
            .unwrap();
        assert_eq!(&bytes[..], wire);
        let observed = outcome.await.unwrap();
        assert!(observed.success);
        assert_eq!(
            (observed.fact_usage.input, observed.fact_usage.cache_read),
            (Some(100), Some(60))
        );
        upstream.await.unwrap();
    }
    #[test]
    fn fragmented_claude_usage_and_errors_are_observed_without_accumulating_payloads() {
        let mut o = Observer::new();
        let wire = b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"cache_read_input_tokens\":11}}}\r\n\r\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hello\"}}\n\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":3}}\n\ndata: {\"type\":\"message_stop\"}\n\n";
        for bytes in wire.chunks(7) {
            o.feed(bytes);
        }
        o.ended = true;
        let result = o.outcome();
        assert!(result.success && result.observational_only);
        assert_eq!(result.usage, (0, 0, 0));
        assert_eq!(
            (result.fact_usage.input, result.fact_usage.output),
            (Some(16), Some(3))
        );
        assert_eq!(result.fact_usage.cache_read, Some(11));
        assert!(result.first_output_ms.is_some());
        assert!(o.line.is_empty());
        o.feed(b"data: {\"type\":\"error\",\"error\":{\"message\":\"bad\"}}\n\n");
        assert!(!o.outcome().success);
    }
    #[test]
    fn oversized_lines_and_cancelled_streams_stay_bounded() {
        let mut o = Observer::new();
        o.feed(&vec![b'a'; MAX_LINE + 1]);
        assert!(o.line.is_empty());
        o.feed(b"\ndata: [DONE]\n\n");
        assert_eq!(o.outcome().status_code, 499);
        o.ended = true;
        assert!(o.outcome().success);
    }
}
