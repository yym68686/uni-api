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
    if value["store"] == true || value["background"] == true {
        return Err("Interactions capability unavailable: stored or background execution");
    }
    if [
        "previous_interaction_id",
        "continuation_token",
        "agent",
        "agent_config",
        "max_total_tokens",
    ]
    .iter()
    .any(|field| object.get(*field).is_some_and(|v| !v.is_null()))
    {
        return Err(
            "Interactions capability unavailable: continuation, agents, or total token budgets",
        );
    }
    let requires_tool = |v: &Value| {
        matches!(v.as_str(), Some("required" | "any"))
            || v.get("type").is_some_and(|v| v == "function")
            || matches!(
                v.get("mode").and_then(Value::as_str),
                Some("any" | "required")
            )
    };
    if requires_tool(&value["tool_choice"])
        || requires_tool(&value["generation_config"]["tool_choice"])
    {
        return Err("Interactions capability unavailable: required tool calls");
    }
    if object.get("response_modalities").is_some_and(|v| {
        !v.is_null()
            && !v.as_array().is_some_and(|items| {
                items.iter().all(|item| {
                    item.as_str()
                        .is_some_and(|s| s.eq_ignore_ascii_case("text"))
                })
            })
    }) {
        return Err("Interactions capability unavailable: non-text output modalities");
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

pub(crate) fn structured_text_requested(value: &Value) -> bool {
    let format = &value["response_format"];
    let is_structured = |v: &Value| v["mime_type"] == "application/json" || !v["schema"].is_null();
    is_structured(format)
        || format
            .as_array()
            .is_some_and(|items| items.iter().any(is_structured))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Compat {
    pub summaries: bool,
    warnings: u16,
}

#[derive(Clone, Copy)]
pub(crate) enum Omitted {
    Extensions,
    InputMetadata,
    GenerationConfig,
    ThinkingSummaries,
    ThinkingLevel,
    ServiceTier,
    ChannelDefaults,
    Tools,
}

impl Compat {
    pub(crate) fn omit(&mut self, field: Omitted) {
        self.warnings |= 1 << field as u16;
    }

    pub(crate) fn warning(self) -> Option<String> {
        (self.warnings != 0).then(|| {
            [
                "extensions",
                "input_metadata",
                "generation_config",
                "thinking_summaries",
                "thinking_level",
                "service_tier",
                "channel_defaults",
                "tools",
            ]
            .into_iter()
            .enumerate()
            .filter_map(|(i, label)| (self.warnings & (1 << i) != 0).then_some(label))
            .collect::<Vec<_>>()
            .join(",")
        })
    }
}

pub(crate) fn thinking_level(level: &str) -> Option<&'static str> {
    match level {
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" | "extra_high" => Some("xhigh"),
        _ => None,
    }
}

pub(crate) fn max_output_limit(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .filter(|v| *v > 0 && *v <= u64::from(u32::MAX))
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

pub(crate) fn responses_summaries(value: &Value) -> impl Iterator<Item = &str> {
    value["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "reasoning")
        .flat_map(|item| item["summary"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "summary_text")
        .filter_map(|part| part["text"].as_str().filter(|text| !text.is_empty()))
}

pub(crate) fn gemini_summaries(value: &Value) -> impl Iterator<Item = &str> {
    value
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part["thought"] == true)
        .filter_map(|part| part["text"].as_str().filter(|text| !text.is_empty()))
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
    pub(crate) emitted_summary: bool,
    active_step: Option<(usize, bool)>,
    next_step: usize,
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
            emitted_summary: false,
            active_step: None,
            next_step: 0,
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
        self.step_delta(text, false)
    }

    pub(crate) fn summary(&mut self, text: &str) -> Vec<Value> {
        self.step_delta(text, true)
    }

    fn step_delta(&mut self, text: &str, summary: bool) -> Vec<Value> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        self.emitted_text |= !summary;
        self.emitted_summary |= summary;
        let index = match self.active_step {
            Some((index, kind)) if kind == summary => index,
            previous => {
                if let Some((index, _)) = previous {
                    events.push(json!({"event_type":"step.stop","index":index}));
                }
                let index = self.next_step;
                self.next_step += 1;
                self.active_step = Some((index, summary));
                let step = if summary {
                    json!({"type":"thought","summary":[]})
                } else {
                    json!({"type":"model_output","content":[]})
                };
                events.push(json!({"event_type":"step.start","index":index,"step":step}));
                index
            }
        };
        let delta = if summary {
            json!({"type":"thought_summary","content":{"type":"text","text":text}})
        } else {
            json!({"type":"text","text":text})
        };
        events.push(json!({"event_type":"step.delta","index":index,"delta":delta}));
        events
    }

    pub(crate) fn finish(&self, status: &str) -> Vec<Value> {
        let mut events = Vec::new();
        if let Some((index, _)) = self.active_step {
            events.push(json!({"event_type":"step.stop","index":index}));
        }
        events.push(json!({"event_type":"interaction.completed","interaction":self.interaction(status, None)}));
        events
    }

    pub(crate) fn json(&self, status: &str, text: &str, summaries: &[String]) -> Value {
        let mut steps = Vec::new();
        if !summaries.is_empty() {
            steps.push(json!({"type":"thought","summary":summaries.iter()
                .map(|text| json!({"type":"text","text":text})).collect::<Vec<_>>()}));
        }
        if !text.is_empty() {
            steps.push(text_step(text));
        }
        self.interaction(status, Some(json!(steps)))
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

// The input here is the gateway's own bounded translated SSE, not arbitrary
// provider events. Use the same terminal and usage validation as the SSE path.
pub(crate) async fn collect_translated(
    body: Body,
    limit: usize,
) -> Result<Option<Value>, &'static str> {
    const INVALID: &str = "Invalid translated Interactions stream";
    let mut stream = body.into_data_stream();
    let mut decoder = SseDecoder::new(limit);
    let mut received = 0usize;
    let mut steps: Vec<(bool, String)> = Vec::new();
    let mut final_payload = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| INVALID)?;
        received = received.saturating_add(chunk.len());
        if received > limit {
            return Err("Translated Interactions stream exceeded the response size limit");
        }
        for frame in decoder.feed(&chunk).map_err(|_| INVALID)? {
            let Some(data) = parse_sse_frame(&frame).map_err(|_| INVALID)?.data else {
                continue;
            };
            let mut event: Value = serde_json::from_str(&data).map_err(|_| INVALID)?;
            let index = || {
                event["index"]
                    .as_u64()
                    .and_then(|i| usize::try_from(i).ok())
            };
            match event["event_type"].as_str() {
                Some("step.start") if index() == Some(steps.len()) => {
                    steps.push((event["step"]["type"] == "thought", String::new()));
                }
                Some("step.delta") => {
                    let (is_summary, text) =
                        index().and_then(|i| steps.get_mut(i)).ok_or(INVALID)?;
                    let path = if *is_summary {
                        "/delta/content/text"
                    } else {
                        "/delta/text"
                    };
                    text.push_str(event.pointer(path).and_then(Value::as_str).ok_or(INVALID)?);
                }
                Some("interaction.completed") if final_payload.is_none() => {
                    final_payload = Some(event["interaction"].take());
                }
                Some("error") => return Ok(None),
                _ => {}
            }
        }
    }
    if let Some(payload) = &mut final_payload {
        payload["steps"] = json!(steps
            .into_iter()
            .map(|(summary, text)| {
                if summary {
                    json!({"type":"thought","summary":[{"type":"text","text":text}]})
                } else {
                    text_step(&text)
                }
            })
            .collect::<Vec<_>>());
    }
    Ok(final_payload)
}

pub(crate) fn to_chat(value: &Value, compat: &mut Compat) -> Value {
    if !value["tools"].is_null() && !value["tools"].as_array().is_some_and(Vec::is_empty) {
        compat.omit(Omitted::Tools);
    }
    if value.as_object().is_some_and(|root| {
        root.iter().any(|(key, v)| {
            !v.is_null()
                && !matches!(
                    key.as_str(),
                    "model"
                        | "input"
                        | "stream"
                        | "store"
                        | "background"
                        | "system_instruction"
                        | "generation_config"
                        | "service_tier"
                        | "response_format"
                        | "response_modalities"
                        | "tools"
                        | "tool_choice"
                )
        })
    }) || ["stream", "store", "background"]
        .iter()
        .any(|key| !value[*key].is_null() && !value[*key].is_boolean())
        || (!value["system_instruction"].is_null() && !value["system_instruction"].is_string())
    {
        compat.omit(Omitted::Extensions);
    }
    if !value["service_tier"].is_null() {
        compat.omit(Omitted::ServiceTier);
    }
    let mut parts = |content: &Value| {
        content
            .as_array()
            .into_iter()
            .flatten()
            .map(|part| {
                if part
                    .as_object()
                    .is_some_and(|o| o.keys().any(|key| !matches!(key.as_str(), "type" | "text")))
                {
                    compat.omit(Omitted::InputMetadata);
                }
                json!({"type":"text","text":part["text"]})
            })
            .collect::<Vec<_>>()
    };
    let mut messages = Vec::new();
    if let Some(instruction) = value["system_instruction"].as_str() {
        messages.push(json!({"role":"system","content":instruction}));
    }
    let input = &value["input"];
    if let Some(text) = input.as_str() {
        messages.push(json!({"role":"user","content":text}));
    } else if input[0]["type"] == "text" {
        messages.push(json!({"role":"user","content":parts(input)}));
    } else if let Some(steps) = input.as_array() {
        let mut metadata = false;
        for step in steps {
            let role = if step["type"] == "model_output" {
                "assistant"
            } else {
                "user"
            };
            metadata |= step.as_object().is_some_and(|o| {
                o.keys()
                    .any(|key| !matches!(key.as_str(), "type" | "content"))
            });
            messages.push(json!({"role":role,"content":parts(&step["content"])}));
        }
        if metadata {
            compat.omit(Omitted::InputMetadata);
        }
    }
    let stream = value["stream"].as_bool().unwrap_or(false);
    let mut request = json!({"model":value["model"],"messages":messages,"stream":stream});
    let config = &value["generation_config"];
    if let Some(config) = config.as_object() {
        for (name, v) in config.iter().filter(|(_, v)| !v.is_null()) {
            match name.as_str() {
                "temperature" if v.as_f64().is_some_and(|v| (0.0..=2.0).contains(&v)) => {
                    request[name] = v.clone()
                }
                "top_p" if v.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v)) => {
                    request[name] = v.clone()
                }
                "max_output_tokens" if max_output_limit(v).is_some() => {
                    request["max_tokens"] = v.clone()
                }
                "thinking_level" | "thinking_summaries" => {}
                "tool_choice" if matches!(v.as_str(), Some("auto" | "none")) => {}
                _ => compat.omit(Omitted::GenerationConfig),
            }
        }
    } else if !config.is_null() {
        compat.omit(Omitted::GenerationConfig);
    }
    if stream {
        request["stream_options"] = json!({"include_usage":true});
    }
    request
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
    if let Some(system) = payload["system_instruction"].as_str() {
        parts.push(system);
    }
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
