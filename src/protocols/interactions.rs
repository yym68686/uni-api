//! The stateless Gemini Developer API Interactions contract.
use crate::protocols::provider_stream::{StreamOutcome, Translation};
use crate::protocols::sse::{parse_sse_frame, SseDecoder};
use axum::{body::Body, http::Response};
use futures_util::StreamExt;
use serde_json::{json, Value};
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

// Use the same traversal for admission and moderation; never flatten the
// actual wire payload. An array is either content parts or explicit steps.
fn visit_input_text<'a>(input: &'a Value, mut visit: impl FnMut(&'a str)) -> bool {
    fn nonempty_text(value: &Value) -> Option<&str> {
        value.as_str().filter(|text| !text.trim().is_empty())
    }
    if let Some(text) = nonempty_text(input) {
        visit(text);
        return true;
    }
    let mut visit_part = |part: &'a Value| match nonempty_text(&part["text"]) {
        Some(text) if part["type"] == "text" => {
            visit(text);
            true
        }
        _ => false,
    };
    let Some(parts) = input.as_array().filter(|parts| !parts.is_empty()) else {
        return false;
    };
    if parts[0]["type"] == "text" {
        return parts.iter().all(visit_part);
    }
    parts.iter().all(|step| {
        matches!(step["type"].as_str(), Some("user_input" | "model_output"))
            && step["content"]
                .as_array()
                .is_some_and(|content| !content.is_empty() && content.iter().all(&mut visit_part))
    })
}

pub(crate) fn validate(value: &Value) -> Result<(), &'static str> {
    let object = value
        .as_object()
        .ok_or("Interactions requires a JSON object")?;
    if value["model"].as_str().is_none_or(|s| s.trim().is_empty()) {
        return Err("Interactions requires a model");
    }
    if !visit_input_text(&value["input"], |_| {}) {
        return Err("Interactions requires nonempty text input (a string, text parts, or text-only user_input/model_output steps)");
    }
    for field in ["background", "store"] {
        if object.get(field).is_some_and(|v| v != &Value::Bool(false)) {
            return Err("Interactions requires store and background to be false when specified");
        }
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
        return Err("Interactions currently supports only stateless text model requests without custom service tiers");
    }
    if ["tools", "tool_choice"]
        .iter()
        .any(|field| object.get(*field).is_some_and(|v| !v.is_null()))
    {
        return Err("Interactions currently does not support tools or tool_choice");
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
    if let Some(config) = object.get("generation_config").filter(|v| !v.is_null()) {
        if !config.is_object() || !config["tool_choice"].is_null() {
            return Err("Interactions currently does not support tool configuration");
        }
    }
    Ok(())
}

pub(crate) fn compatibility(value: &Value) -> Result<(), String> {
    let Some(root) = value.as_object() else {
        return Err("Interactions requires a JSON object".into());
    };
    for (key, _) in root.iter().filter(|(_, value)| !value.is_null()) {
        match key.as_str() {
            "model" | "input" | "store" | "background" | "stream" => {}
            "system_instruction" if root[key].is_string() => {}
            "response_format" if text_format(&root[key]) => {}
            "generation_config" => {
                if let Some(config) = root[key].as_object() {
                    for (field, setting) in config.iter().filter(|(_, value)| !value.is_null()) {
                        match field.as_str() {
                            "temperature" | "top_p" if setting.is_number() => {}
                            "max_output_tokens" if setting.as_u64().is_some_and(|n| n > 0) => {}
                            _ => {
                                return Err(native_required(&format!("generation_config.{field}")))
                            }
                        }
                    }
                } else {
                    return Err(native_required(key));
                }
            }
            _ => return Err(native_required(key)),
        }
    }
    if let Some(items) = value["input"].as_array() {
        for item in items {
            let (allowed, parts): (&[&str], _) = if item["type"] == "text" {
                (&["type", "text"], None)
            } else {
                (&["type", "content"], item["content"].as_array())
            };
            let allowed_fields = |item: &Value, allowed: &[&str]| {
                item.as_object()
                    .is_some_and(|object| object.keys().all(|key| allowed.contains(&key.as_str())))
            };
            if !allowed_fields(item, allowed)
                || parts.is_some_and(|parts| {
                    !parts.iter().all(|p| allowed_fields(p, &["type", "text"]))
                })
            {
                return Err(native_required("input"));
            }
        }
    }
    Ok(())
}

fn text_format(value: &Value) -> bool {
    let text = |v: &Value| v["type"] == "text" && v.as_object().is_some_and(|o| o.len() == 1);
    text(value)
        || value
            .as_array()
            .is_some_and(|v| v.len() == 1 && text(&v[0]))
}

fn native_required(field: &str) -> String {
    format!("Interactions field {field} requires a native Interactions channel for this model")
}

pub(crate) fn from_wire_usage(usage: Option<&Value>) -> serde_json::Map<String, Value> {
    let mut result = serde_json::Map::new();
    let Some(usage) = usage.filter(|u| u.is_object()) else {
        return result;
    };
    let usage = usage
        .get("uniApiUsage")
        .filter(|v| v.is_object())
        .unwrap_or(usage);
    let facts = crate::observability::usage::FactUsage::from_usage(Some(usage));
    let read = |keys: &[&str]| {
        keys.iter().find_map(|key| {
            usage
                .pointer(key)
                .and_then(Value::as_i64)
                .filter(|n| *n >= 0)
        })
    };
    let thought = read(&[
        "/thoughtsTokenCount",
        "/completion_tokens_details/reasoning_tokens",
        "/output_tokens_details/reasoning_tokens",
    ]);
    let text = if usage.get("candidatesTokenCount").is_some() {
        read(&["/candidatesTokenCount"])
    } else {
        facts
            .output
            .and_then(|n| n.checked_sub(thought.unwrap_or(0)).filter(|n| *n >= 0))
    };
    for (name, count) in [
        ("total_input_tokens", facts.input),
        ("total_output_tokens", text),
        ("total_thought_tokens", thought),
        ("total_cached_tokens", facts.cache_read),
        ("total_tokens", read(&["/total_tokens", "/totalTokenCount"])),
    ] {
        if let Some(count) = count {
            result.insert(name.into(), json!(count));
        }
    }
    result
}

pub(crate) fn stream_error(code: u16, message: &str) -> Value {
    let error = crate::protocols::gemini::error(code, message);
    json!({"event_type":"error","error":error["error"]})
}

fn text_step(text: &str) -> Value {
    json!({"type":"model_output","content":[{"type":"text","text":text}]})
}

pub(crate) fn output_status(reason: &str) -> Result<&'static str, &'static str> {
    match reason {
        "stop" => Ok("completed"),
        "length" => Ok("incomplete"),
        "content_filter" => Ok("failed"),
        _ => Err("Upstream returned an unsupported Interactions finish reason"),
    }
}

pub(crate) fn chat_non_text(message: &Value) -> bool {
    ["tool_calls", "function_call", "audio", "refusal", "images"]
        .iter()
        .any(|field| {
            message.get(field).is_some_and(|value| {
                !value.is_null()
                    && !value.as_array().is_some_and(Vec::is_empty)
                    && value.as_str() != Some("")
            })
        })
        || message
            .get("content")
            .is_some_and(|v| !v.is_null() && !v.is_string())
}

pub(crate) fn gemini_non_text(value: &Value) -> bool {
    value
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .is_some_and(|parts| {
            parts.iter().any(|part| {
                !part["text"].is_string()
                    && !part.as_object().is_some_and(|fields| {
                        fields.keys().all(|key| {
                            matches!(
                                key.as_str(),
                                "thought" | "thoughtSignature" | "thought_signature"
                            )
                        })
                    })
            })
        })
}

pub(crate) fn responses_reason(value: &Value) -> Result<&'static str, &'static str> {
    if value.get("error").is_some_and(|error| !error.is_null()) {
        return Err("Upstream Responses returned an error");
    }
    if value
        .get("output")
        .and_then(Value::as_array)
        .is_none_or(|items| {
            items.iter().any(|item| match item["type"].as_str() {
                Some("reasoning") => false,
                Some("message") => item["content"].as_array().is_none_or(|parts| {
                    parts
                        .iter()
                        .any(|part| !matches!(part["type"].as_str(), Some("output_text" | "text")))
                }),
                _ => true,
            })
        })
    {
        return Err("Upstream Responses returned unsupported output");
    }
    match value["status"].as_str() {
        Some("completed") => Ok("stop"),
        Some("incomplete") => match value
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens" | "length") => Ok("length"),
            Some("content_filter") => Ok("content_filter"),
            _ => Err("Upstream Responses returned an unknown incomplete reason"),
        },
        _ => Err("Upstream Responses did not return a completed or incomplete response"),
    }
}

pub(crate) struct TextOutput {
    id: String,
    model: String,
    created: String,
    pub(crate) emitted_text: bool,
    pub(crate) terminal: Option<&'static str>,
    pub(crate) input_done: bool,
    pub(crate) usage: serde_json::Map<String, Value>,
}

impl TextOutput {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            id: format!(
                "int_{}",
                crate::observability::context::request_id(&Default::default())
            ),
            model: model.into(),
            created: timestamp(),
            emitted_text: false,
            terminal: None,
            input_done: false,
            usage: Default::default(),
        }
    }

    fn interaction(&self, status: &str, steps: Option<Value>) -> Value {
        let updated = if status == "in_progress" {
            self.created.clone()
        } else {
            timestamp()
        };
        let mut result = json!({
            "id":self.id,"model":self.model,"status":status,"store":false,
            "created":self.created,"updated":updated,
        });
        if !self.usage.is_empty() && status != "in_progress" {
            result["usage"] = Value::Object(self.usage.clone());
        }
        if let Some(steps) = steps {
            result["steps"] = steps;
        }
        result
    }

    pub(crate) fn created(&self) -> Value {
        json!({"event_type":"interaction.created","interaction":self.interaction("in_progress", None)})
    }

    pub(crate) fn delta(&mut self, text: &str) -> Vec<Value> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if !self.emitted_text {
            self.emitted_text = true;
            events.push(json!({"event_type":"step.start","index":0,"step":{"type":"model_output","content":[]}}));
        }
        events
            .push(json!({"event_type":"step.delta","index":0,"delta":{"type":"text","text":text}}));
        events
    }

    pub(crate) fn finish(&self, status: &str) -> Vec<Value> {
        let mut events = Vec::new();
        if self.emitted_text {
            events.push(json!({"event_type":"step.stop","index":0}));
        }
        events.push(json!({"event_type":"interaction.completed","interaction":self.interaction(status, None)}));
        events
    }

    pub(crate) fn json(&self, status: &str, text: &str) -> Value {
        let steps = if text.is_empty() {
            json!([])
        } else {
            json!([text_step(text)])
        };
        self.interaction(status, Some(steps))
    }
}

fn timestamp() -> String {
    let now = crate::runtime::clock::unix_seconds();
    let (year, month, day) = crate::providers::cloud::civil_from_days(now / 86_400);
    let (hour, minute, second) = (now / 3_600 % 24, now / 60 % 60, now % 60);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

pub(crate) fn sse_frame(output: &mut Vec<u8>, event: &Value) -> Result<(), serde_json::Error> {
    if let Some(kind) = event["event_type"].as_str() {
        output.extend_from_slice(b"event: ");
        output.extend_from_slice(kind.as_bytes());
        output.push(b'\n');
    }
    output.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut *output, event)?;
    output.extend_from_slice(b"\n\n");
    Ok(())
}

pub(crate) fn to_chat(value: &Value) -> Result<Value, String> {
    validate(value)?;
    compatibility(value)?;
    let mut messages = Vec::new();
    if let Some(instruction) = value["system_instruction"].as_str() {
        messages.push(json!({"role":"system","content":instruction}));
    }
    let input = &value["input"];
    if let Some(text) = input.as_str() {
        messages.push(json!({"role":"user","content":text}));
    } else if input[0]["type"] == "text" {
        messages.push(json!({"role":"user","content":input}));
    } else if let Some(steps) = input.as_array() {
        messages.extend(steps.iter().map(|step| {
            let role = if step["type"] == "model_output" {
                "assistant"
            } else {
                "user"
            };
            json!({"role":role,"content":step["content"]})
        }));
    }
    let stream = value["stream"].as_bool().unwrap_or(false);
    let mut request = json!({"model":value["model"],"messages":messages,"stream":stream});
    for (source, target) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
    ] {
        if let Some(setting) = value["generation_config"]
            .get(source)
            .filter(|v| !v.is_null())
        {
            request[target] = setting.clone();
        }
    }
    if stream {
        request["stream_options"] = json!({"include_usage":true});
    }
    Ok(request)
}

pub(crate) fn status_kind(status: &str) -> &'static str {
    match status {
        "completed" => "completed",
        "requires_action" | "requires_continuation" | "in_progress" | "incomplete" => "incomplete",
        "failed" | "cancelled" | "canceled" => "semantic_failure",
        _ => "protocol_error",
    }
}

pub(crate) fn moderation_text(payload: &Value) -> Option<String> {
    let mut parts = Vec::new();
    visit_input_text(&payload["input"], |text| parts.push(text)).then(|| parts.join("\n"))
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
        let success = kind == "completed";
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
                response_completed: Some(success),
                status_code: match kind {
                    "completed" | "incomplete" => 200,
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
    fn admits_and_moderates_each_text_input_shape_without_losing_steps() {
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
        ] {
            let payload = json!({"model":"fixture", "input":input});
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
            let payload = json!({"model":"fixture", "input":input});
            assert!(validate(&payload).is_err(), "{payload}");
            assert_eq!(moderation_text(&payload), None, "{payload}");
        }
    }

    #[test]
    fn observes_only_complete_terminal_frames_and_does_not_double_count() {
        let mut o = Observation {
            decoder: Some(SseDecoder::new(1024)),
            started: Instant::now(),
            sender: None,
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
        assert_eq!(o.kind, Some("incomplete"));
        assert_eq!(o.usage.counters(), (100, 20, 120));
        assert_eq!(o.usage.fact.cache_read, Some(60));
        assert!(Usage::read(Some(&json!({"total_output_tokens":-1})))
            .fact
            .output
            .is_none());
    }
}
