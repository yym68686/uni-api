//! Only gateway-owned diagnostics may cross the upstream error boundary.
use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};
use std::{collections::VecDeque, io};

#[derive(Clone, Copy)]
pub(crate) struct PublicError(StatusCode);

impl PublicError {
    pub(crate) fn new(status: StatusCode) -> Self {
        Self(status)
    }

    fn from_value(value: &Value) -> Self {
        let number = [
            "/error/status_code",
            "/error/code",
            "/response/error/status_code",
            "/response/error/code",
            "/interaction/error/status_code",
            "/interaction/error/code",
            "/status_code",
        ]
        .iter()
        .filter_map(|path| value.pointer(path).and_then(Value::as_u64))
        .find(|status| (400..600).contains(status))
        .and_then(|status| StatusCode::from_u16(status as u16).ok());
        let symbolic = [
            "/error/status",
            "/error/code",
            "/error/type",
            "/response/error/code",
            "/response/error/type",
            "/interaction/error/status",
            "/interaction/error/code",
            "/code",
        ]
        .iter()
        .filter_map(|path| value.pointer(path).and_then(Value::as_str))
        .find_map(|code| match code {
            "RESOURCE_EXHAUSTED"
            | "rate_limit_error"
            | "rate_limit_exceeded"
            | "upstream_rate_limited" => Some(StatusCode::TOO_MANY_REQUESTS),
            "overloaded_error" | "UNAVAILABLE" => Some(StatusCode::SERVICE_UNAVAILABLE),
            "DEADLINE_EXCEEDED" => Some(StatusCode::GATEWAY_TIMEOUT),
            "INVALID_ARGUMENT" | "invalid_request_error" => Some(StatusCode::BAD_REQUEST),
            "NOT_FOUND" => Some(StatusCode::NOT_FOUND),
            "UNAUTHENTICATED" => Some(StatusCode::UNAUTHORIZED),
            "PERMISSION_DENIED" => Some(StatusCode::FORBIDDEN),
            _ => None,
        });
        Self(number.or(symbolic).unwrap_or(StatusCode::BAD_GATEWAY))
    }

    pub(crate) fn status(self) -> StatusCode {
        self.0
    }

    pub(crate) fn code(self) -> &'static str {
        match self.0.as_u16() {
            300..=399 => "upstream_redirect",
            400 | 413 | 415 | 422 => "upstream_invalid_request",
            401 | 403 => "upstream_authentication_failed",
            402 => "upstream_billing_error",
            404 => "upstream_not_found",
            408 | 504 => "upstream_timeout",
            429 => "upstream_rate_limited",
            503 | 529 => "upstream_unavailable",
            _ => "upstream_error",
        }
    }

    pub(crate) fn message(self) -> &'static str {
        match self.code() {
            "upstream_redirect" => "The upstream service returned an unsupported redirect",
            "upstream_invalid_request" => "The upstream service rejected the request",
            "upstream_authentication_failed" => "The upstream service denied access",
            "upstream_billing_error" => "The upstream service could not process billing",
            "upstream_not_found" => "The upstream resource was not found",
            "upstream_timeout" => "The upstream service timed out",
            "upstream_rate_limited" => "The upstream service rate limit was reached",
            "upstream_unavailable" => "The upstream service is temporarily unavailable",
            _ => "The upstream service could not complete the request",
        }
    }

    pub(crate) fn object(self) -> Value {
        json!({"message":self.message(),"type":"upstream_error","code":self.code(),
            "status_code":self.0.as_u16(),"retryable":matches!(self.0.as_u16(), 408 | 429 | 500..=599)})
    }

    pub(crate) fn response(self, upstream_headers: &HeaderMap) -> Response<Body> {
        let mut response = Response::new(Body::from(json!({"error":self.object()}).to_string()));
        *response.status_mut() = self.0;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if matches!(self.0.as_u16(), 429 | 503 | 529) {
            if let Some(delay) = upstream_headers
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .filter(|v| !v.is_empty() && v.len() <= 10 && v.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|v| v.parse::<u32>().ok())
            {
                response
                    .headers_mut()
                    .insert(header::RETRY_AFTER, HeaderValue::from(delay));
            }
        }
        response
    }
}

pub(crate) fn response(status: StatusCode) -> Response<Body> {
    PublicError::new(status).response(&HeaderMap::new())
}

pub(crate) fn semantic_status(value: &Value) -> StatusCode {
    PublicError::from_value(value).status()
}

pub(crate) fn transport_reason(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "upstream transport timed out"
    } else if error.is_connect() {
        "upstream connection failed"
    } else if error.is_builder() {
        "upstream request could not be built"
    } else if error.is_body() || error.is_decode() {
        "upstream body transfer failed"
    } else {
        "upstream transport failed"
    }
}

fn error_event(event: &str) -> bool {
    matches!(
        event,
        "error" | "response.error" | "response.failed" | "interaction.error" | "interaction.failed"
    )
}

fn failed(value: &Value) -> bool {
    value.get("error").is_some_and(|error| !error.is_null())
        || matches!(
            value.get("status").and_then(Value::as_str),
            Some("failed" | "FAILED")
        )
}

pub(crate) fn is_error(value: &Value) -> bool {
    value
        .get("type")
        .or_else(|| value.get("event_type"))
        .and_then(Value::as_str)
        .is_some_and(error_event)
        || failed(value)
        || value.get("response").is_some_and(failed)
        || value.get("interaction").is_some_and(failed)
}

// Reconstruct the envelope as well as its error object. Arbitrary sibling
// properties, IDs, metadata and debug fields are not safe.
fn safe_payload(
    value: &Value,
    declared: Option<&str>,
    interactions: bool,
    gemini: bool,
) -> (Option<&'static str>, Value) {
    let err = PublicError::from_value(value);
    let kind = value
        .get("type")
        .or_else(|| value.get("event_type"))
        .and_then(Value::as_str);
    if interactions {
        (
            Some("error"),
            crate::protocols::interactions::stream_error(err.status().as_u16(), err.message()),
        )
    } else if gemini {
        (
            None,
            crate::protocols::gemini::error(err.status().as_u16(), err.message()),
        )
    } else if declared == Some("response.failed")
        || kind == Some("response.failed")
        || value.get("response").is_some_and(failed)
        || value["object"] == "response"
    {
        (
            Some("response.failed"),
            json!({"type":"response.failed","response":{"status":"failed","error":err.object()}}),
        )
    } else if declared == Some("error")
        || kind == Some("error")
        || declared.is_some_and(error_event)
    {
        (Some("error"), json!({"type":"error","error":err.object()}))
    } else {
        (None, json!({"error":err.object()}))
    }
}

pub(crate) fn safe_json(value: &Value) -> Value {
    let (_, mut safe) = safe_payload(value, None, false, false);
    if let Some(response) = safe.get_mut("response") {
        return std::mem::take(response);
    }
    safe
}

pub(crate) fn clean_json_body(body: Bytes) -> Bytes {
    let Ok(value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if is_error(&value) {
        Bytes::from(safe_json(&value).to_string())
    } else {
        body
    }
}

fn clean_frame(
    frame: crate::protocols::sse::SseFrame,
    interactions: bool,
    gemini: bool,
) -> io::Result<Bytes> {
    let parsed = crate::protocols::sse::parse_sse_frame(&frame)
        .map_err(|_| io::Error::other("Invalid upstream event"))?;
    let declared = parsed.declared_event.as_deref();
    let value = match parsed.data.as_deref().map(str::trim) {
        Some(data) if data != "[DONE]" => match serde_json::from_str::<Value>(data) {
            Ok(value) => Some(value),
            Err(_) if declared.is_some_and(error_event) => None,
            // Never forward opaque data on an LLM SSE wire. Let the upstream
            // observer retain ownership of termination and failure attribution.
            Err(_) => return Ok(Bytes::new()),
        },
        _ => None,
    };
    if !declared.is_some_and(error_event) && !value.as_ref().is_some_and(is_error) {
        // SSE comments carry no model output or protocol event data.
        return if parsed.comment_only {
            Ok(Bytes::from_static(b": keepalive\n\n"))
        } else {
            Ok(frame.canonical_wire())
        };
    }
    let (event, value) = safe_payload(
        value.as_ref().unwrap_or(&Value::Null),
        declared,
        interactions,
        gemini,
    );
    let prefix = event.map_or(String::new(), |event| format!("event: {event}\n"));
    Ok(Bytes::from(format!("{prefix}data: {value}\n\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn http_error_does_not_relay_headers_or_values() {
        let mut headers = HeaderMap::new();
        for name in [
            "location",
            "www-authenticate",
            "set-cookie",
            "x-oaix-request-id",
        ] {
            headers.insert(
                name,
                HeaderValue::from_static("internal-fixture.invalid/opaque"),
            );
        }
        headers.insert("retry-after", HeaderValue::from_static("00017"));
        let output = PublicError::new(StatusCode::TOO_MANY_REQUESTS).response(&headers);
        assert_eq!(output.status(), 429);
        assert_eq!(output.headers().len(), 2);
        assert_eq!(output.headers()["retry-after"], "17");
        let body = to_bytes(output.into_body(), 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "upstream_rate_limited");
        assert_eq!(value["error"]["retryable"], true);
        headers.insert("retry-after", HeaderValue::from_static("fixture%2Fopaque"));
        assert!(!PublicError::new(StatusCode::TOO_MANY_REQUESTS)
            .response(&headers)
            .headers()
            .contains_key("retry-after"));
    }

    #[tokio::test]
    async fn fragmented_and_unterminated_error_events_are_rebuilt() {
        let good = b"event: step.delta\r\ndata: {\"delta\":\"hello\"}\r\n\r\n";
        let failed = b"\xef\xbb\xbfevent: interaction.failed\r\ndata: {\"interaction\":{\"status\":\"failed\",\"error\":{\"code\":429,\"message\":\"opaque-credential\"}},\"debug\":\"private-fixture.invalid%2F\"}\r\n\n";
        let incomplete = b"event: error\ndata: partial opaque-credential";
        let chunks = [good.as_slice(), failed.as_slice(), incomplete.as_slice()].concat();
        let pieces = chunks
            .chunks(3)
            .map(Bytes::copy_from_slice)
            .collect::<Vec<_>>();
        let response = Response::builder()
            .header("content-type", "text/event-stream; charset=utf-8")
            .body(Body::from_stream(stream::iter(
                pieces.into_iter().map(Ok::<_, io::Error>),
            )))
            .unwrap();
        let output = to_bytes(sanitize_stream(response, true, false).into_body(), 8192)
            .await
            .unwrap();
        assert!(output.starts_with(good));
        let text = std::str::from_utf8(&output).unwrap();
        assert_eq!(text.matches("event: error").count(), 2);
        assert!(text.contains("RESOURCE_EXHAUSTED"));
        for needle in ["opaque-credential", "private-fixture", "debug"] {
            assert!(!text.contains(needle));
        }
    }
}

#[derive(Clone)]
struct Sanitized;

/// Frame before exporting, so a split error is never partially forwarded.
pub(crate) fn sanitize_stream(
    response: Response<Body>,
    interactions: bool,
    gemini: bool,
) -> Response<Body> {
    if response.extensions().get::<Sanitized>().is_some()
        || !response.status().is_success()
        || !response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"))
            })
    {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    parts.extensions.insert(Sanitized);
    parts.headers.remove(header::CONTENT_LENGTH);
    let sse = crate::protocols::sse::SseDecoder::new(0);
    let output = stream::try_unfold(
        (body.into_data_stream(), sse, VecDeque::new(), false),
        move |(mut body, mut sse, mut frames, mut eof)| async move {
            loop {
                if let Some(frame) = frames.pop_front() {
                    let safe = clean_frame(frame, interactions, gemini)?;
                    if safe.is_empty() {
                        continue;
                    }
                    return Ok::<_, io::Error>(Some((safe, (body, sse, frames, eof))));
                }
                if eof {
                    return Ok(None);
                }
                let next = match body.next().await {
                    Some(Ok(bytes)) => sse.feed(&bytes),
                    Some(Err(_)) => return Err(io::Error::other("Upstream stream interrupted")),
                    None => {
                        eof = true;
                        sse.finish()
                    }
                };
                frames = next
                    .map_err(|_| io::Error::other("Invalid upstream event"))?
                    .into();
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(output))
}
