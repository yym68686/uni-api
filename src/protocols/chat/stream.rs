//! Chat heartbeats keep the connection alive without committing an upstream attempt.

use axum::{
    body::Body,
    http::{HeaderValue, Response},
};
use std::{future::Future, io, time::Duration};

use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};
use tokio::{
    sync::{mpsc, watch},
    time::Instant,
};

use crate::protocols::provider_stream::PrecommitFailure;
use crate::protocols::provider_stream::Translation;
use tokio_stream::wrappers::ReceiverStream;

const MAX_PRECOMMIT_BYTES: usize = 1024 * 1024;

const HEARTBEAT: &[u8] = b": keepalive\n\n";

pub type KeepaliveUpdates = watch::Sender<Option<Duration>>;

async fn tick(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn next_tick(interval: Option<Duration>) -> Option<Instant> {
    interval.map(|duration| Instant::now() + duration)
}

/// Keep HTTP error statuses until the first heartbeat is actually due. Afterwards
/// the complete retry loop still runs, and a terminal failure becomes an SSE error.
pub async fn with_keepalive<F>(
    future: F,
    mut updates: watch::Receiver<Option<Duration>>,
) -> Response<Body>
where
    F: Future<Output = Response<Body>> + Send + 'static,
{
    let mut future = Box::pin(future);
    let mut interval = *updates.borrow_and_update();
    let mut deadline = next_tick(interval);
    let mut updates_open = true;
    loop {
        tokio::select! {
            biased;
            response = &mut future => return heartbeat_body(response, *updates.borrow()),
            changed = updates.changed(), if updates_open => {
                updates_open = changed.is_ok();
                interval = *updates.borrow_and_update();
                deadline = next_tick(interval);
            }
            () = tick(deadline) => break,
        }
    }
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(async move {
        if tx.send(Ok(Bytes::from_static(HEARTBEAT))).await.is_err() {
            return;
        }
        let mut deadline = next_tick(interval);
        let response = loop {
            tokio::select! {
                biased;
                () = tx.closed() => return,
                response = &mut future => break response,
                changed = updates.changed(), if updates_open => {
                    updates_open = changed.is_ok();
                    interval = *updates.borrow_and_update();
                    deadline = next_tick(interval);
                }
                () = tick(deadline) => {
                    // A slow reader must not stop polling the upstream retry loop.
                    if matches!(tx.try_send(Ok(Bytes::from_static(HEARTBEAT))), Err(mpsc::error::TrySendError::Closed(_))) { return; }
                    deadline = next_tick(interval);
                }
            }
        };
        interval = *updates.borrow();
        if is_sse(&response) && response.status().is_success() {
            let mut body = heartbeat_body(response, interval)
                .into_body()
                .into_data_stream();
            loop {
                tokio::select! {
                    biased;
                    () = tx.closed() => return,
                    chunk = body.next() => match chunk {
                        Some(chunk) => if tx.send(chunk.map_err(io::Error::other)).await.is_err() { return; },
                        None => return,
                    }
                }
            }
        }
        let status = response.status().as_u16();
        let body = axum::body::to_bytes(response.into_body(), MAX_PRECOMMIT_BYTES).await;
        let mut payload = body
            .ok()
            .and_then(|body| serde_json::from_slice::<Value>(&body).ok())
            .unwrap_or_else(|| json!({"error":{"message":"All upstream attempts failed"}}));
        if !payload.get("error").is_some_and(Value::is_object) {
            payload = json!({"error":{"message":payload.to_string()}});
        }
        payload["error"]["status_code"] = json!(status);
        let _ = tx
            .send(Ok(Bytes::from(format!("data: {payload}\n\n"))))
            .await;
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response.headers_mut().insert(
        "cache-control",
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response
        .headers_mut()
        .insert("x-uni-api-runtime", HeaderValue::from_static("rust"));
    response
}

fn is_sse(response: &Response<Body>) -> bool {
    response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"))
}

/// Never insert a comment into a partially forwarded SSE frame.
#[derive(Default)]
struct Boundary {
    tail: Vec<u8>,
}

impl Boundary {
    fn feed(&mut self, bytes: &[u8]) {
        if bytes.len() >= 4 {
            self.tail.clear();
            self.tail.extend_from_slice(&bytes[bytes.len() - 4..]);
        } else {
            self.tail.extend_from_slice(bytes);
            if self.tail.len() > 4 {
                self.tail.drain(..self.tail.len() - 4);
            }
        }
    }
    fn complete(&self) -> bool {
        self.tail.is_empty() || self.tail.ends_with(b"\n\n") || self.tail.ends_with(b"\r\n\r\n")
    }
}

fn heartbeat_body(response: Response<Body>, interval: Option<Duration>) -> Response<Body> {
    if interval.is_none() || !is_sse(&response) {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    parts.headers.remove("content-length");
    let output = stream::unfold(
        (
            body.into_data_stream(),
            Boundary::default(),
            next_tick(interval),
        ),
        move |(mut body, mut boundary, mut deadline)| async move {
            tokio::select! {
                biased;
                chunk = body.next() => {
                    let chunk = chunk?;
                    if let Ok(bytes) = &chunk { boundary.feed(bytes); }
                    deadline = next_tick(interval);
                    Some((chunk.map_err(io::Error::other), (body, boundary, deadline)))
                }
                () = tick(deadline), if boundary.complete() => {
                    deadline = next_tick(interval);
                    Some((Ok(Bytes::from_static(HEARTBEAT)), (body, boundary, deadline)))
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(output))
}

/// Headers, comments, role-only deltas and usage do not constitute model output.
/// Retain bounded metadata for the winning attempt; discard it on retry.
pub async fn preflight(
    translation: Translation,
    deadline: Option<Instant>,
) -> Result<Translation, PrecommitFailure> {
    let Translation { response, outcome } = translation;
    let (parts, body) = response.into_parts();
    let mut body = body.into_data_stream();
    let mut buffer = Vec::new();
    let mut retained = Vec::new();
    loop {
        while let Some((end, separator)) =
            crate::protocols::provider_stream::next_event_boundary(&buffer)
        {
            let commit = real_output(&buffer[..end])?;
            retained.extend(buffer.drain(..end + separator));
            if retained.len() > MAX_PRECOMMIT_BYTES {
                return Err(failure("Chat metadata exceeded 1 MiB before real output"));
            }
            if commit {
                retained.extend_from_slice(&buffer);
                let stream =
                    stream::once(async move { Ok::<_, axum::Error>(Bytes::from(retained)) })
                        .chain(body);
                return Ok(Translation {
                    response: Response::from_parts(parts, Body::from_stream(stream)),
                    outcome,
                });
            }
        }
        if buffer.len().saturating_add(retained.len()) > MAX_PRECOMMIT_BYTES {
            return Err(failure("Chat SSE frame exceeded 1 MiB before real output"));
        }
        let next = tokio::select! {
            biased;
            () = tick(deadline) => return Err(PrecommitFailure { status_code: 504, detail: "Upstream chat first output timed out".into() }),
            chunk = body.next() => chunk,
        };
        match next {
            Some(Ok(bytes)) => buffer.extend_from_slice(&bytes),
            Some(Err(error)) => {
                return Err(failure(&format!(
                    "Upstream chat stream failed before real output: {error}"
                )))
            }
            None => {
                return Err(failure(
                    "Upstream chat stream ended before real output or a finish reason",
                ))
            }
        }
    }
}

fn failure(detail: &str) -> PrecommitFailure {
    PrecommitFailure {
        status_code: 502,
        detail: detail.into(),
    }
}

fn real_output(event: &[u8]) -> Result<bool, PrecommitFailure> {
    let text =
        std::str::from_utf8(event).map_err(|_| failure("Invalid UTF-8 in chat SSE event"))?;
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");
    if data.trim().is_empty() {
        return Ok(false);
    }
    if data.trim() == "[DONE]" {
        return Err(failure(
            "Upstream chat stream emitted [DONE] before real output or a finish reason",
        ));
    }
    let payload: Value = serde_json::from_str(&data)
        .map_err(|error| failure(&format!("Invalid chat SSE JSON: {error}")))?;
    if let Some(error) = payload.get("error").filter(|error| !error.is_null()) {
        let status = error
            .get("status_code")
            .or_else(|| error.get("code"))
            .and_then(Value::as_u64)
            .filter(|status| (400..=599).contains(status))
            .unwrap_or(502) as u16;
        return Err(PrecommitFailure {
            status_code: status,
            detail: error.to_string(),
        });
    }
    Ok(payload
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("finish_reason")
                    .is_some_and(|value| !value.is_null())
                    || choice
                        .get("delta")
                        .and_then(Value::as_object)
                        .is_some_and(|delta| {
                            delta.iter().any(|(key, value)| {
                                key != "role"
                                    && match value {
                                        Value::Null => false,
                                        Value::String(text) => !text.is_empty(),
                                        Value::Array(items) => !items.is_empty(),
                                        Value::Object(items) => !items.is_empty(),
                                        _ => true,
                                    }
                            })
                        })
            })
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn sse(body: Body) -> Response<Body> {
        let mut response = Response::new(body);
        response.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        response
    }

    #[test]
    fn only_real_chat_output_or_a_finish_reason_commits() {
        for event in [
            ": keepalive",
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}",
            "data: {\"choices\":[],\"usage\":{\"total_tokens\":1}}",
        ] {
            assert!(!real_output(event.as_bytes()).unwrap());
        }
        for event in [
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\"}]}}]}",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
        ] {
            assert!(real_output(event.as_bytes()).unwrap());
        }
        assert!(real_output(b"data: [DONE]").is_err());
        assert_eq!(
            real_output(b"data: {\"error\":{\"message\":\"busy\",\"status_code\":429}}")
                .unwrap_err()
                .status_code,
            429
        );
    }

    #[tokio::test]
    async fn fragmented_first_output_preserves_the_winning_wire() {
        let wire = b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\ndata: [DONE]\n\n";
        let body = Body::from_stream(stream::iter(
            wire.chunks(3)
                .map(|bytes| Ok::<_, io::Error>(Bytes::copy_from_slice(bytes)))
                .collect::<Vec<_>>(),
        ));
        let (_tx, outcome) = tokio::sync::oneshot::channel();
        let translation = preflight(
            Translation {
                response: sse(body),
                outcome,
            },
            Some(Instant::now() + Duration::from_secs(1)),
        )
        .await
        .unwrap();
        assert_eq!(
            to_bytes(translation.response.into_body(), 4096)
                .await
                .unwrap()
                .as_ref(),
            wire
        );
    }

    #[tokio::test]
    async fn fast_failure_keeps_http_status_but_heartbeat_failure_is_explicit_sse() {
        let (tx, rx) = watch::channel(Some(Duration::from_millis(15)));
        let response = with_keepalive(
            async {
                crate::transport::http::json_error(
                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                    "timed out",
                )
            },
            rx,
        )
        .await;
        assert_eq!(response.status(), 504);
        let response = with_keepalive(
            async {
                tokio::time::sleep(Duration::from_millis(65)).await;
                crate::transport::http::json_error(
                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                    "timed out",
                )
            },
            tx.subscribe(),
        )
        .await;
        assert_eq!(response.status(), 200);
        let body = String::from_utf8(to_bytes(response.into_body(), 4096).await.unwrap().to_vec())
            .unwrap();
        assert!(body.matches(": keepalive").count() >= 2, "{body}");
        assert!(body.contains("\"status_code\":504"), "{body}");
        assert!(!body.contains("[DONE]"));
    }

    #[tokio::test]
    async fn heartbeat_does_not_split_an_sse_frame() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
        let response = heartbeat_body(
            sse(Body::from_stream(ReceiverStream::new(rx))),
            Some(Duration::from_millis(10)),
        );
        let mut body = response.into_body().into_data_stream();
        tx.send(Ok(Bytes::from_static(b"data: {\"text\":\"")))
            .await
            .unwrap();
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            b"data: {\"text\":\""
        );
        assert!(tokio::time::timeout(Duration::from_millis(35), body.next())
            .await
            .is_err());
        tx.send(Ok(Bytes::from_static(b"OK\"}\n\n"))).await.unwrap();
        assert_eq!(body.next().await.unwrap().unwrap().as_ref(), b"OK\"}\n\n");
        assert_eq!(body.next().await.unwrap().unwrap().as_ref(), HEARTBEAT);
    }

    #[tokio::test]
    async fn disconnect_drops_the_pending_retry_loop() {
        struct OnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let (dropped, observed) = tokio::sync::oneshot::channel();
        let (_tx, rx) = watch::channel(Some(Duration::from_millis(5)));
        let response = with_keepalive(
            async move {
                let _guard = OnDrop(Some(dropped));
                std::future::pending::<Response<Body>>().await
            },
            rx,
        )
        .await;
        drop(response);
        tokio::time::timeout(Duration::from_secs(1), observed)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn a_full_heartbeat_queue_does_not_block_upstream_progress() {
        let (_tx, rx) = watch::channel(Some(Duration::from_millis(1)));
        let (completed, observed) = tokio::sync::oneshot::channel();
        let response = with_keepalive(
            async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let _ = completed.send(());
                crate::transport::http::json_error(
                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                    "timed out",
                )
            },
            rx,
        )
        .await;
        // Retain the body without consuming it until the heartbeat queue is full.
        tokio::time::timeout(Duration::from_secs(1), observed)
            .await
            .unwrap()
            .unwrap();
        drop(response);
    }
}
