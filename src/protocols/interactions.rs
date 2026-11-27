//! The stateless Gemini Developer API Interactions contract.
use crate::protocols::provider_stream::{StreamOutcome, Translation};
use crate::protocols::sse::{parse_sse_frame, SseDecoder};
use axum::{body::Body, http::Response};
use futures_util::StreamExt;
use serde_json::Value;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

pub(crate) const PATH: &str = "/v1/interactions";

pub(crate) fn is_route(path: &str) -> bool {
    path.trim_end_matches('/') == PATH
}

pub(crate) fn is_resource(path: &str) -> bool {
    path.strip_prefix(PATH)
        .is_some_and(|tail| tail.starts_with('/') && !tail.trim_matches('/').is_empty())
        || path.trim_end_matches('/') == "/v1beta/interactions"
        || path.starts_with("/v1beta/interactions/")
}

pub(crate) fn validate(value: &Value) -> Result<(), &'static str> {
    let object = value
        .as_object()
        .ok_or("Interactions requires a JSON object")?;
    if value["model"]
        .as_str()
        .is_none_or(|s| s.is_empty() || s.trim() != s)
    {
        return Err("Interactions requires an exact upstream model name");
    }
    // Google defaults to server-side storage. Do not silently change the
    // caller's storage setting or the managed gateway's no-storage policy.
    if value["store"] != Value::Bool(false) {
        return Err("Interactions passthrough requires explicit store:false");
    }
    if object
        .get("background")
        .is_some_and(|v| v != &Value::Bool(false))
    {
        return Err("Background Interactions are not supported");
    }
    if [
        "previous_interaction_id",
        "continuation_token",
        "agent",
        "agent_config",
        "max_total_tokens",
        "service_tier",
        "response_modalities",
    ]
    .iter()
    .any(|field| object.get(*field).is_some_and(|v| !v.is_null()))
    {
        return Err("Interactions gateway only supports stateless model requests without custom service tiers or response modalities");
    }
    if object.get("stream").is_some_and(|v| !v.is_boolean()) {
        return Err("Interactions stream must be a boolean");
    }
    if let Some(format) = object.get("response_format").filter(|v| !v.is_null()) {
        let text = |v: &Value| v["type"] == "text";
        if !(text(format)
            || format
                .as_array()
                .is_some_and(|values| !values.is_empty() && values.iter().all(text)))
        {
            return Err("Interactions currently supports only text response formats");
        }
    }
    Ok(())
}

pub(crate) fn status_kind(status: &str) -> &'static str {
    match status {
        "completed" => "completed",
        "requires_action" => "requires_action",
        "requires_continuation" | "in_progress" | "incomplete" => "incomplete",
        "failed" | "cancelled" | "canceled" => "semantic_failure",
        _ => "protocol_error",
    }
}

pub(crate) fn moderation_text(payload: &Value) -> Option<String> {
    // Extract text for the existing moderation policy. Unrecognized parts
    // are ignored here, not rejected or removed from the wire body.
    fn content(value: &Value, parts: &mut Vec<String>) {
        if let Some(text) = value.as_str().filter(|text| !text.trim().is_empty()) {
            parts.push(text.to_owned());
        } else if let Some(items) = value.as_array() {
            parts.extend(items.iter().filter_map(|item| {
                (item["type"] == "text")
                    .then(|| item["text"].as_str())
                    .flatten()
                    .filter(|text| !text.trim().is_empty())
                    .map(str::to_owned)
            }));
        }
    }
    let input = &payload["input"];
    let mut parts = Vec::new();
    content(input, &mut parts);
    if let Some(steps) = input.as_array() {
        for step in steps {
            match step["type"].as_str() {
                Some("user_input" | "model_output") => content(&step["content"], &mut parts),
                Some("function_result") => content(&step["result"], &mut parts),
                _ => {}
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

#[derive(Default)]
pub(crate) struct Usage {
    pub fact: crate::observability::usage::FactUsage,
    pub total: Option<i64>,
}

impl Usage {
    pub(crate) fn read(usage: Option<&Value>) -> Self {
        let Some(u) = usage else {
            return Self::default();
        };
        let read = |key: &str| u[key].as_i64().filter(|n| *n >= 0);
        Self {
            fact: crate::observability::usage::FactUsage {
                input: read("total_input_tokens"),
                output: read("total_output_tokens")
                    .and_then(|n| n.checked_add(read("total_thought_tokens").unwrap_or(0))),
                cache_read: read("total_cached_tokens"),
                ..Default::default()
            },
            total: read("total_tokens"),
        }
    }

    pub(crate) fn counters(&self) -> (i64, i64, i64) {
        (
            self.fact.input.unwrap_or_default(),
            self.fact.output.unwrap_or_default(),
            self.total.unwrap_or_default(),
        )
    }
}

pub(crate) fn observe_stream(
    upstream: reqwest::Response,
    started: Instant,
    idle: Option<Duration>,
) -> Translation {
    let status = upstream.status();
    let mut headers = crate::transport::http::filtered_response_headers(upstream.headers());
    headers.insert(
        "x-accel-buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    let (sender, outcome) = oneshot::channel();
    let observation = Observation {
        decoder: Some(SseDecoder::new(1024 * 1024)),
        started,
        sender: Some(sender),
        eof: false,
        kind: None,
        usage: Usage::default(),
        created_ms: None,
        text_ms: None,
    };
    let chunks = Box::pin(crate::transport::body::upstream_chunks(upstream, idle));
    let body = futures_util::stream::unfold(
        (chunks, observation),
        |(mut chunks, mut observation)| async move {
            match chunks.next().await {
                Some(chunk) => {
                    match &chunk {
                        Ok(bytes) => observation.feed(bytes),
                        Err(_) => observation.kind = Some("transport_error"),
                    }
                    Some((chunk, (chunks, observation)))
                }
                None => {
                    observation.eof = true;
                    None
                }
            }
        },
    );
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Translation { response, outcome }
}

struct Observation {
    decoder: Option<SseDecoder>,
    started: Instant,
    sender: Option<oneshot::Sender<StreamOutcome>>,
    eof: bool,
    kind: Option<&'static str>,
    usage: Usage,
    created_ms: Option<f64>,
    text_ms: Option<f64>,
}

impl Observation {
    fn feed(&mut self, bytes: &[u8]) {
        let Some(decoder) = &mut self.decoder else {
            return;
        };
        let Ok(frames) = decoder.feed(bytes) else {
            // Observation is bounded. Oversized events are still relayed verbatim.
            self.decoder = None;
            return;
        };
        for frame in frames {
            let Ok(frame) = parse_sse_frame(&frame) else {
                continue;
            };
            let Some(data) = frame.data else { continue };
            let Ok(value) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            let kind = value["event_type"]
                .as_str()
                .or(frame.declared_event.as_deref());
            let elapsed = self.started.elapsed().as_secs_f64() * 1000.0;
            match kind {
                Some("interaction.created") => {
                    self.created_ms.get_or_insert(elapsed);
                }
                Some("step.delta")
                    if value["delta"]["type"] == "text"
                        && value["delta"]["text"]
                            .as_str()
                            .is_some_and(|s| !s.is_empty()) =>
                {
                    self.text_ms.get_or_insert(elapsed);
                }
                Some("interaction.completed") if self.kind.is_none() => {
                    self.kind = Some(status_kind(
                        value["interaction"]["status"].as_str().unwrap_or(""),
                    ));
                    self.usage = Usage::read(value.pointer("/interaction/usage"));
                }
                Some("error") => self.kind = Some("semantic_failure"),
                _ => {}
            }
        }
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        let kind = self.kind.unwrap_or(if self.eof {
            "protocol_error"
        } else {
            "downstream_disconnected"
        });
        let success = matches!(kind, "completed" | "requires_action");
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(StreamOutcome {
                usage: self.usage.counters(),
                fact_usage: std::mem::take(&mut self.usage.fact),
                first_output_ms: self.text_ms,
                first_text_ms: self.text_ms,
                response_created_ms: self.created_ms,
                success,
                observational_only: true,
                terminal_kind: Some(kind),
                response_completed: Some(kind == "completed"),
                status_code: match kind {
                    "completed" | "requires_action" | "incomplete" => 200,
                    "downstream_disconnected" => 499,
                    _ => 502,
                },
                failure_reason: (!success).then_some(kind),
                detail: if success {
                    String::new()
                } else {
                    format!("Interactions stream: {kind}")
                },
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_moderation_text_without_constraining_protocol_steps() {
        for (input, expected) in [
            (json!("  你好  "), "  你好  "),
            (
                json!([{"type":"text", "text":"first"}, {"type":"text", "text":"second"}]),
                "first\nsecond",
            ),
            (
                json!([
                    {"type":"user_input", "content":[{"type":"text", "text":"first"}, {"type":"text", "text":"second"}]},
                    {"type":"model_output", "content":[{"type":"text", "text":"answer"}]},
                    {"type":"user_input", "content":[{"type":"text", "text":"next"}]}
                ]),
                "first\nsecond\nanswer\nnext",
            ),
            (
                json!([
                    {"type":"user_input","content":"Use the tool"},
                    {"type":"thought","signature":"keep-opaque"},
                    {"type":"function_call","id":"call-1","arguments":{"value":"opaque"}},
                    {"type":"function_result","call_id":"call-1","result":[
                        {"type":"text","text":"42"}, {"type":"image","uri":"fixture"}
                    ]}
                ]),
                "Use the tool\n42",
            ),
        ] {
            let payload = json!({"model":"fixture", "store":false, "input":input});
            assert_eq!(validate(&payload), Ok(()));
            assert_eq!(moderation_text(&payload).as_deref(), Some(expected));
        }
        for input in [
            json!("  "),
            json!([]),
            json!([{"type":"user_input", "content":[]}]),
            json!([{"type":"user_input", "content":[{"type":"text", "text":"ok"}, {"type":"image", "uri":"fixture"}]}]),
            json!([{"type":"model_output", "content":[{"type":"text", "text":" "}]}]),
            json!([{"type":"function_result", "content":[{"type":"text", "text":"ok"}]}]),
            json!([{"type":"text", "text":"flat"}, {"type":"user_input", "content":[{"type":"text", "text":"step"}]}]),
        ] {
            let payload = json!({"model":"fixture", "store":false, "input":input});
            assert_eq!(validate(&payload), Ok(()), "{payload}");
        }
    }

    #[test]
    fn observes_only_complete_terminal_frames_and_does_not_double_count() {
        let (sender, outcome) = oneshot::channel();
        let mut o = Observation {
            decoder: Some(SseDecoder::new(1024)),
            started: Instant::now(),
            sender: Some(sender),
            eof: false,
            kind: None,
            usage: Usage::default(),
            created_ms: None,
            text_ms: None,
        };
        let frame = b"event: interaction.completed\r\ndata: {\"interaction\":{\"status\":\"requires_action\",\"usage\":{\"total_input_tokens\":100,\"total_output_tokens\":12,\"total_thought_tokens\":8,\"total_cached_tokens\":60,\"total_tokens\":120}}}\r\n\r\n";
        for chunk in frame[..frame.len() - 1].chunks(7) {
            o.feed(chunk);
        }
        assert!(o.kind.is_none());
        o.feed(&frame[frame.len() - 1..]);
        o.feed(frame);
        assert_eq!(o.kind, Some("requires_action"));
        assert_eq!(o.usage.counters(), (100, 20, 120));
        assert_eq!(o.usage.fact.cache_read, Some(60));
        assert!(Usage::read(Some(&json!({"total_output_tokens":-1})))
            .fact
            .output
            .is_none());
        drop(o);
        let mut outcome = outcome;
        let outcome = outcome.try_recv().unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.response_completed, Some(false));
        assert_eq!(outcome.status_code, 200);
        assert_eq!(outcome.usage, (100, 20, 120));
    }
}
