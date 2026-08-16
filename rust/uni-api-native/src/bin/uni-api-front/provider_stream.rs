use std::collections::HashMap;
use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{HeaderValue, Response};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use crc32fast::hash as crc32;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

const MAX_STREAM_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Protocol {
    Chat,
    Responses,
    Gemini,
    Claude,
    VertexClaude,
    Cohere,
    AwsBedrock,
    Cloudflare,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputProtocol {
    Chat,
    Responses,
}

pub struct Translation {
    pub response: Response<Body>,
    pub usage: oneshot::Receiver<(i64, i64, i64)>,
}

#[derive(Clone, Copy)]
struct StreamTimeouts {
    idle: Option<Duration>,
    total: Option<Duration>,
}

#[derive(Clone, Copy)]
struct TranslationOptions {
    include_usage: bool,
    timeouts: StreamTimeouts,
}

pub fn translate(
    response: reqwest::Response,
    protocol: Protocol,
    output_protocol: OutputProtocol,
    model: String,
    include_usage: bool,
    idle_timeout_seconds: Option<f64>,
    total_timeout_seconds: Option<f64>,
) -> Translation {
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(16);
    let (usage_tx, usage_rx) = oneshot::channel();
    tokio::spawn(async move {
        let options = TranslationOptions {
            include_usage,
            timeouts: StreamTimeouts {
                idle: positive_duration(idle_timeout_seconds),
                total: positive_duration(total_timeout_seconds),
            },
        };
        let result = run_translation(
            response,
            protocol,
            output_protocol,
            &model,
            &content_type,
            &tx,
            options,
        )
        .await;
        let usage = match result {
            Ok(usage) => usage,
            Err(error) => {
                let payload = match output_protocol {
                    OutputProtocol::Chat => json!({"error":{"message":error}}),
                    OutputProtocol::Responses => json!({
                        "type":"response.failed",
                        "response":{
                            "status":"failed",
                            "error":{"message":error,"type":"upstream_stream_error"}
                        }
                    }),
                };
                let _ = send_wire(&tx, &payload, output_protocol).await;
                (0, 0, 0)
            }
        };
        if output_protocol == OutputProtocol::Chat {
            let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
        }
        let _ = usage_tx.send(usage);
    });
    let mut output = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    *output.status_mut() = status;
    output.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    output.headers_mut().insert(
        "cache-control",
        HeaderValue::from_static("no-cache, no-transform"),
    );
    Translation {
        response: output,
        usage: usage_rx,
    }
}

async fn run_translation(
    response: reqwest::Response,
    protocol: Protocol,
    output_protocol: OutputProtocol,
    model: &str,
    content_type: &str,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    options: TranslationOptions,
) -> Result<(i64, i64, i64), String> {
    let mut state = StreamState::new_with_options(model, output_protocol, options.include_usage);
    let timeouts = options.timeouts;
    for event in state.start_chunks() {
        send_wire(tx, &event, output_protocol).await?;
    }
    let mut stream = response.bytes_stream();
    let total_deadline = timeouts
        .total
        .map(|timeout| tokio::time::Instant::now() + timeout);
    if protocol == Protocol::AwsBedrock {
        let mut buffer = Vec::new();
        while let Some(chunk) =
            next_upstream_chunk(&mut stream, timeouts.idle, total_deadline).await?
        {
            buffer.extend_from_slice(&chunk.map_err(|error| error.to_string())?);
            drain_aws_frames(&mut buffer, &mut state, tx).await?;
            if buffer.len() > MAX_STREAM_FRAME_BYTES {
                return Err("AWS event-stream frame exceeded 1 MiB".into());
            }
        }
        if !buffer.is_empty() {
            return Err("AWS event-stream ended with an incomplete frame".into());
        }
        state.finish_if_needed(tx).await?;
        return Ok(state.usage());
    }

    let sse = content_type.contains("text/event-stream")
        || matches!(
            protocol,
            Protocol::Responses | Protocol::Claude | Protocol::Cloudflare | Protocol::Chat
        );
    if sse {
        let mut buffer = Vec::new();
        while let Some(chunk) =
            next_upstream_chunk(&mut stream, timeouts.idle, total_deadline).await?
        {
            buffer.extend_from_slice(&chunk.map_err(|error| error.to_string())?);
            drain_sse_events(&mut buffer, protocol, &mut state, tx).await?;
            if buffer.len() > MAX_STREAM_FRAME_BYTES {
                return Err("upstream SSE event exceeded 1 MiB".into());
            }
        }
        if !buffer.iter().all(u8::is_ascii_whitespace) {
            drain_final_sse_event(&mut buffer, protocol, &mut state, tx).await?;
        }
    } else if protocol == Protocol::Cohere {
        let mut buffer = Vec::new();
        while let Some(chunk) =
            next_upstream_chunk(&mut stream, timeouts.idle, total_deadline).await?
        {
            buffer.extend_from_slice(&chunk.map_err(|error| error.to_string())?);
            while let Some(index) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=index).collect::<Vec<_>>();
                process_json_bytes(&line, protocol, &mut state, tx).await?;
            }
            if buffer.len() > MAX_STREAM_FRAME_BYTES {
                return Err("Cohere stream frame exceeded 1 MiB".into());
            }
        }
        process_json_bytes(&buffer, protocol, &mut state, tx).await?;
    } else {
        let mut framer = JsonObjectFramer::default();
        while let Some(chunk) =
            next_upstream_chunk(&mut stream, timeouts.idle, total_deadline).await?
        {
            for frame in framer.feed(&chunk.map_err(|error| error.to_string())?)? {
                process_json_bytes(&frame, protocol, &mut state, tx).await?;
            }
        }
        framer.finish()?;
    }
    state.finish_if_needed(tx).await?;
    Ok(state.usage())
}

fn positive_duration(value: Option<f64>) -> Option<Duration> {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(Duration::from_secs_f64)
}

async fn next_upstream_chunk<S>(
    stream: &mut S,
    idle_timeout: Option<Duration>,
    total_deadline: Option<tokio::time::Instant>,
) -> Result<Option<Result<Bytes, reqwest::Error>>, String>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let total_remaining = total_deadline
        .map(|deadline| deadline.saturating_duration_since(tokio::time::Instant::now()));
    let timeout = match (idle_timeout, total_remaining) {
        (Some(idle), Some(total)) => Some(idle.min(total)),
        (Some(idle), None) => Some(idle),
        (None, Some(total)) => Some(total),
        (None, None) => None,
    };
    let Some(timeout) = timeout else {
        return Ok(stream.next().await);
    };
    if timeout.is_zero() {
        return Err("upstream stream total timeout exceeded".into());
    }
    tokio::time::timeout(timeout, stream.next())
        .await
        .map_err(|_| "upstream stream idle or total timeout exceeded".to_owned())
}

async fn drain_sse_events(
    buffer: &mut Vec<u8>,
    protocol: Protocol,
    state: &mut StreamState,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    while let Some((end, separator)) = next_event_boundary(buffer) {
        let event = buffer.drain(..end).collect::<Vec<_>>();
        buffer.drain(..separator);
        process_sse_event(&event, protocol, state, tx).await?;
    }
    Ok(())
}

async fn drain_final_sse_event(
    buffer: &mut Vec<u8>,
    protocol: Protocol,
    state: &mut StreamState,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let event = std::mem::take(buffer);
    process_sse_event(&event, protocol, state, tx).await
}

fn next_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len().saturating_sub(1) {
        if buffer[index..].starts_with(b"\n\n") {
            return Some((index, 2));
        }
        if buffer[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
    }
    None
}

async fn process_sse_event(
    event: &[u8],
    protocol: Protocol,
    state: &mut StreamState,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let text = std::str::from_utf8(event).map_err(|_| "upstream SSE was not UTF-8")?;
    let mut data = String::new();
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.trim_start());
        }
    }
    if data.trim().is_empty() || data.trim() == "[DONE]" {
        return Ok(());
    }
    process_json_bytes(data.as_bytes(), protocol, state, tx).await
}

async fn process_json_bytes(
    bytes: &[u8],
    protocol: Protocol,
    state: &mut StreamState,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let bytes = trim_ascii(bytes);
    if bytes.is_empty() {
        return Ok(());
    }
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("decode upstream stream event: {error}"))?;
    let chunks = state.convert(protocol, &value);
    for chunk in chunks {
        send_wire(tx, &chunk, state.output_protocol).await?;
    }
    Ok(())
}

async fn drain_aws_frames(
    buffer: &mut Vec<u8>,
    state: &mut StreamState,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    loop {
        if buffer.len() < 12 {
            return Ok(());
        }
        let total = u32::from_be_bytes(buffer[0..4].try_into().expect("four bytes")) as usize;
        let headers = u32::from_be_bytes(buffer[4..8].try_into().expect("four bytes")) as usize;
        if !(16..=MAX_STREAM_FRAME_BYTES).contains(&total) || headers > total - 16 {
            return Err("invalid AWS event-stream frame length".into());
        }
        if buffer.len() < total {
            return Ok(());
        }
        let frame = buffer.drain(..total).collect::<Vec<_>>();
        let expected_prelude = u32::from_be_bytes(frame[8..12].try_into().expect("four bytes"));
        if crc32(&frame[..8]) != expected_prelude {
            return Err("invalid AWS event-stream prelude CRC".into());
        }
        let expected_message =
            u32::from_be_bytes(frame[total - 4..total].try_into().expect("four bytes"));
        if crc32(&frame[..total - 4]) != expected_message {
            return Err("invalid AWS event-stream message CRC".into());
        }
        let envelope: Value = serde_json::from_slice(&frame[12 + headers..total - 4])
            .map_err(|error| format!("decode AWS event envelope: {error}"))?;
        let Some(encoded) = envelope.get("bytes").and_then(Value::as_str) else {
            continue;
        };
        let decoded = BASE64
            .decode(encoded)
            .map_err(|_| "AWS event-stream bytes field is not valid base64")?;
        let payload: Value = serde_json::from_slice(&decoded)
            .map_err(|error| format!("decode AWS Bedrock event: {error}"))?;
        let event_type = payload.get("type").and_then(Value::as_str);
        let chunks = match event_type {
            Some("message_delta") => {
                state.completion_tokens = number(payload.pointer("/usage/output_tokens"));
                Vec::new()
            }
            Some("message_stop") => Vec::new(),
            _ => state.convert(Protocol::Claude, &payload),
        };
        for chunk in chunks {
            send_wire(tx, &chunk, state.output_protocol).await?;
        }
        if let Some(metrics) = payload.get("amazon-bedrock-invocationMetrics") {
            state.prompt_tokens = number(metrics.get("inputTokenCount"));
            state.completion_tokens = number(metrics.get("outputTokenCount"));
            for chunk in state.finish_chunks_for_output("stop") {
                send_wire(tx, &chunk, state.output_protocol).await?;
            }
        }
    }
}

async fn send_wire(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    value: &Value,
    output_protocol: OutputProtocol,
) -> Result<(), String> {
    let mut wire = Vec::new();
    if output_protocol == OutputProtocol::Responses {
        if let Some(event_type) = value.get("type").and_then(Value::as_str) {
            wire.extend_from_slice(b"event: ");
            wire.extend_from_slice(event_type.as_bytes());
            wire.push(b'\n');
        }
    }
    wire.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut wire, value).map_err(|error| error.to_string())?;
    wire.extend_from_slice(b"\n\n");
    tx.send(Ok(Bytes::from(wire)))
        .await
        .map_err(|_| "downstream disconnected".to_owned())
}

struct StreamState {
    id: String,
    model: String,
    created: u64,
    next_tool_index: usize,
    tools: HashMap<String, usize>,
    response_tool_output_indexes: HashMap<u64, usize>,
    prompt_tokens: i64,
    completion_tokens: i64,
    chat_usage: Option<Value>,
    terminal: bool,
    output_protocol: OutputProtocol,
    include_usage: bool,
    responses: ResponsesOutputState,
}

impl StreamState {
    fn new_with_options(model: &str, output_protocol: OutputProtocol, include_usage: bool) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: format!("chatcmpl-{created}"),
            model: model.to_owned(),
            created,
            next_tool_index: 0,
            tools: HashMap::new(),
            response_tool_output_indexes: HashMap::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            chat_usage: None,
            terminal: false,
            output_protocol,
            include_usage,
            responses: ResponsesOutputState::new(model, created),
        }
    }

    fn start_chunks(&mut self) -> Vec<Value> {
        if self.output_protocol == OutputProtocol::Responses {
            self.responses.start_chunks()
        } else {
            Vec::new()
        }
    }

    fn convert(&mut self, protocol: Protocol, value: &Value) -> Vec<Value> {
        let chunks = match protocol {
            Protocol::Chat => self.chat(value),
            Protocol::Responses => self.responses(value),
            Protocol::Gemini => self.gemini(value),
            Protocol::Claude => self.claude(value),
            Protocol::VertexClaude => self.claude(value),
            Protocol::Cohere => self.cohere(value),
            Protocol::Cloudflare => self.cloudflare(value),
            Protocol::AwsBedrock => Vec::new(),
        };
        self.encode_chunks(chunks)
    }

    fn chat(&mut self, value: &Value) -> Vec<Value> {
        if let Some(id) = value.get("id").and_then(Value::as_str) {
            self.id = id.to_owned();
        }
        if let Some(usage) = value.get("usage") {
            self.prompt_tokens = number(usage.get("prompt_tokens"));
            self.completion_tokens = number(usage.get("completion_tokens"));
        }
        if value
            .pointer("/choices/0/finish_reason")
            .is_some_and(|value| !value.is_null())
        {
            self.terminal = true;
        }
        vec![value.clone()]
    }

    fn encode_chunks(&mut self, chunks: Vec<Value>) -> Vec<Value> {
        if self.output_protocol == OutputProtocol::Chat {
            chunks
        } else {
            chunks
                .iter()
                .flat_map(|chunk| self.responses.encode_chat_chunk(chunk))
                .collect()
        }
    }

    fn responses(&mut self, value: &Value) -> Vec<Value> {
        let event = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event {
            "response.output_text.delta" => value
                .get("delta")
                .and_then(Value::as_str)
                .map(|text| vec![self.chunk(json!({"content":text}), None, None)])
                .unwrap_or_default(),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => value
                .get("delta")
                .and_then(Value::as_str)
                .map(|text| vec![self.chunk(json!({"reasoning_content":text}), None, None)])
                .unwrap_or_default(),
            "response.output_item.added"
                if value.pointer("/item/type").and_then(Value::as_str) == Some("function_call") =>
            {
                let call_id = value.pointer("/item/call_id").and_then(Value::as_str);
                let item_id = value.pointer("/item/id").and_then(Value::as_str);
                let output_index = value.get("output_index").and_then(Value::as_u64);
                let index = self.responses_tool_index(call_id, item_id, output_index);
                let emitted_call_id = call_id.or(item_id).unwrap_or("call");
                vec![self.chunk(
                    json!({"tool_calls":[{"index":index,"id":emitted_call_id,"type":"function","function":{"name":value.pointer("/item/name").cloned().unwrap_or(Value::Null),"arguments":""}}]}),
                    None,
                    None,
                )]
            }
            "response.function_call_arguments.delta" => {
                let call_id = value.get("call_id").and_then(Value::as_str);
                let item_id = value.get("item_id").and_then(Value::as_str);
                let output_index = value.get("output_index").and_then(Value::as_u64);
                let index = self.responses_tool_index(call_id, item_id, output_index);
                vec![self.chunk(
                    json!({"tool_calls":[{"index":index,"function":{"arguments":value.get("delta").cloned().unwrap_or(Value::String(String::new()))}}]}),
                    None,
                    None,
                )]
            }
            "response.completed" => {
                let usage = responses_usage_to_chat(value.pointer("/response/usage"));
                self.prompt_tokens = number(usage.get("prompt_tokens"));
                self.completion_tokens = number(usage.get("completion_tokens"));
                self.chat_usage = Some(usage);
                self.finish_chunks(if self.tools.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                })
            }
            "response.failed" | "error" => vec![
                json!({"error":{"message":value.pointer("/response/error/message").or_else(|| value.pointer("/error/message")).and_then(Value::as_str).unwrap_or("upstream Responses request failed")}}),
            ],
            _ => Vec::new(),
        }
    }

    fn gemini(&mut self, value: &Value) -> Vec<Value> {
        let mut output = Vec::new();
        for part in value
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                let field = if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    "reasoning_content"
                } else {
                    "content"
                };
                output.push(self.chunk(json!({field:text}), None, None));
            }
            if let Some(call) = part.get("functionCall") {
                let call_id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        crate::generic_api::gemini_call_id(
                            part.get("thoughtSignature")
                                .or_else(|| part.get("thought_signature"))
                                .and_then(Value::as_str),
                            self.next_tool_index + 1,
                        )
                    });
                let index = self.tool_index(&call_id);
                output.push(self.chunk(
                    json!({"tool_calls":[{"index":index,"id":call_id,"type":"function","function":{"name":call.get("name").cloned().unwrap_or(Value::Null),"arguments":serde_json::to_string(call.get("args").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into())}}]}),
                    None,
                    None,
                ));
            }
            if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) {
                if let Some(signature) = part
                    .get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(Value::as_str)
                {
                    crate::generic_api::cache_gemini_image_thought_signature(data, signature);
                }
                let mime = part
                    .pointer("/inlineData/mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                output.push(self.chunk(
                    json!({"content":format!("\n![image](data:{mime};base64,{data})")}),
                    None,
                    None,
                ));
            }
        }
        if let Some(usage) = value.get("usageMetadata") {
            self.prompt_tokens = number(usage.get("promptTokenCount"));
            self.completion_tokens =
                number(usage.get("candidatesTokenCount")) + number(usage.get("thoughtsTokenCount"));
        }
        if value
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
            .is_some()
            || value.pointer("/promptFeedback/blockReason").is_some()
        {
            output.extend(self.finish_chunks(if self.tools.is_empty() {
                "stop"
            } else {
                "tool_calls"
            }));
        }
        output
    }

    fn claude(&mut self, value: &Value) -> Vec<Value> {
        let event = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event {
            "message_start" => {
                if let Some(id) = value.pointer("/message/id").and_then(Value::as_str) {
                    self.id = id.to_owned();
                }
                self.prompt_tokens = number(value.pointer("/message/usage/input_tokens"));
                vec![self.chunk(json!({"role":"assistant"}), None, None)]
            }
            "content_block_start" => {
                match value.pointer("/content_block/type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let call_id = value
                            .pointer("/content_block/id")
                            .and_then(Value::as_str)
                            .unwrap_or("call");
                        let index = self.tool_index(call_id);
                        vec![self.chunk(
                        json!({"tool_calls":[{"index":index,"id":call_id,"type":"function","function":{"name":value.pointer("/content_block/name").cloned().unwrap_or(Value::Null),"arguments":""}}]}),
                        None,
                        None,
                    )]
                    }
                    Some("thinking") => value
                        .pointer("/content_block/thinking")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                        .map(|text| vec![self.chunk(json!({"reasoning_content":text}), None, None)])
                        .unwrap_or_default(),
                    _ => value
                        .pointer("/content_block/text")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                        .map(|text| vec![self.chunk(json!({"content":text}), None, None)])
                        .unwrap_or_default(),
                }
            }
            "content_block_delta" => match value.pointer("/delta/type").and_then(Value::as_str) {
                Some("text_delta") => value
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .map(|text| vec![self.chunk(json!({"content":text}), None, None)])
                    .unwrap_or_default(),
                Some("thinking_delta") => value
                    .pointer("/delta/thinking")
                    .and_then(Value::as_str)
                    .map(|text| vec![self.chunk(json!({"reasoning_content":text}), None, None)])
                    .unwrap_or_default(),
                Some("input_json_delta") => {
                    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
                    vec![self.chunk(
                        json!({"tool_calls":[{"index":index,"function":{"arguments":value.pointer("/delta/partial_json").cloned().unwrap_or(Value::String(String::new()))}}]}),
                        None,
                        None,
                    )]
                }
                _ => Vec::new(),
            },
            "message_delta" => {
                self.completion_tokens = number(value.pointer("/usage/output_tokens"));
                value
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(|reason| {
                        self.finish_chunks(if reason == "tool_use" {
                            "tool_calls"
                        } else {
                            "stop"
                        })
                    })
                    .unwrap_or_default()
            }
            "message_stop" => self.finish_chunks(if self.tools.is_empty() {
                "stop"
            } else {
                "tool_calls"
            }),
            "error" => vec![
                json!({"error":{"message":value.pointer("/error/message").and_then(Value::as_str).unwrap_or("Anthropic stream error")}}),
            ],
            _ => self.vertex_claude_fallback(value),
        }
    }

    fn vertex_claude_fallback(&mut self, value: &Value) -> Vec<Value> {
        let mut output = Vec::new();
        collect_named_text(value, "text", &mut |text| {
            output.push(self.chunk(json!({"content":text}), None, None));
        });
        if find_key(value, "finishReason").is_some() || find_key(value, "stop_reason").is_some() {
            output.extend(self.finish_chunks(if self.tools.is_empty() {
                "stop"
            } else {
                "tool_calls"
            }));
        }
        output
    }

    fn cohere(&mut self, value: &Value) -> Vec<Value> {
        if value.get("event_type").and_then(Value::as_str) == Some("text-generation") {
            return value
                .get("text")
                .and_then(Value::as_str)
                .map(|text| vec![self.chunk(json!({"content":text}), None, None)])
                .unwrap_or_default();
        }
        if value.get("is_finished").and_then(Value::as_bool) == Some(true) {
            self.prompt_tokens = number(value.pointer("/meta/billed_units/input_tokens"));
            self.completion_tokens = number(value.pointer("/meta/billed_units/output_tokens"));
            return self.finish_chunks("stop");
        }
        Vec::new()
    }

    fn cloudflare(&mut self, value: &Value) -> Vec<Value> {
        let text = value
            .get("response")
            .or_else(|| value.pointer("/result/response"))
            .and_then(Value::as_str);
        let mut output = text
            .map(|text| vec![self.chunk(json!({"content":text}), None, None)])
            .unwrap_or_default();
        if value.get("done").and_then(Value::as_bool) == Some(true)
            || matches!(
                value.get("event").and_then(Value::as_str),
                Some("done" | "completed")
            )
        {
            output.extend(self.finish_chunks("stop"));
        }
        output
    }

    fn tool_index(&mut self, id: &str) -> usize {
        if let Some(index) = self.tools.get(id) {
            return *index;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        self.tools.insert(id.to_owned(), index);
        index
    }

    fn responses_tool_index(
        &mut self,
        call_id: Option<&str>,
        item_id: Option<&str>,
        output_index: Option<u64>,
    ) -> usize {
        let existing = call_id
            .and_then(|id| self.tools.get(id).copied())
            .or_else(|| item_id.and_then(|id| self.tools.get(id).copied()))
            .or_else(|| {
                output_index
                    .and_then(|index| self.response_tool_output_indexes.get(&index).copied())
            });
        let index = existing.unwrap_or_else(|| {
            let index = self.next_tool_index;
            self.next_tool_index += 1;
            index
        });
        for id in [call_id, item_id].into_iter().flatten() {
            if !id.is_empty() {
                self.tools.insert(id.to_owned(), index);
            }
        }
        if let Some(output_index) = output_index {
            self.response_tool_output_indexes
                .insert(output_index, index);
        }
        index
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>, usage: Option<Value>) -> Value {
        let mut chunk = json!({
            "id":self.id,
            "object":"chat.completion.chunk",
            "created":self.created,
            "model":self.model,
            "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}],
        });
        if let Some(usage) = usage {
            chunk
                .as_object_mut()
                .expect("chunk object")
                .insert("usage".into(), usage);
        } else if self.output_protocol == OutputProtocol::Chat && self.include_usage {
            chunk
                .as_object_mut()
                .expect("chunk object")
                .insert("usage".into(), Value::Null);
        }
        chunk
    }

    fn finish_chunks(&mut self, reason: &str) -> Vec<Value> {
        if self.terminal {
            return Vec::new();
        }
        self.terminal = true;
        let usage = self.chat_usage.clone().unwrap_or_else(|| {
            json!({
                "prompt_tokens":self.prompt_tokens,
                "completion_tokens":self.completion_tokens,
                "total_tokens":self.prompt_tokens + self.completion_tokens,
            })
        });
        if self.output_protocol == OutputProtocol::Responses {
            return vec![self.chunk(json!({}), Some(reason), Some(usage))];
        }
        let mut chunks = vec![self.chunk(json!({}), Some(reason), None)];
        if self.include_usage {
            chunks.push(json!({
                "id":self.id,
                "object":"chat.completion.chunk",
                "created":self.created,
                "model":self.model,
                "choices":[],
                "usage":usage,
            }));
        }
        chunks
    }

    fn finish_chunks_for_output(&mut self, reason: &str) -> Vec<Value> {
        let chunks = self.finish_chunks(reason);
        self.encode_chunks(chunks)
    }

    fn usage(&self) -> (i64, i64, i64) {
        (
            self.prompt_tokens,
            self.completion_tokens,
            self.prompt_tokens + self.completion_tokens,
        )
    }

    async fn finish_if_needed(
        &mut self,
        tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        if self.output_protocol == OutputProtocol::Responses {
            for chunk in self.responses.finish() {
                send_wire(tx, &chunk, self.output_protocol).await?;
            }
            return Ok(());
        }
        for chunk in self.finish_chunks_for_output(if self.tools.is_empty() {
            "stop"
        } else {
            "tool_calls"
        }) {
            send_wire(tx, &chunk, self.output_protocol).await?;
        }
        Ok(())
    }
}

struct ResponsesOutputState {
    id: String,
    model: String,
    created: u64,
    sequence: u64,
    next_output_index: usize,
    message: Option<ResponsesTextItem>,
    reasoning: Option<ResponsesReasoningItem>,
    tools: HashMap<usize, ResponsesToolItem>,
    prompt_tokens: i64,
    completion_tokens: i64,
    completed: bool,
}

#[derive(Clone)]
struct ResponsesTextItem {
    output_index: usize,
    id: String,
    text: String,
}

#[derive(Clone)]
struct ResponsesReasoningItem {
    output_index: usize,
    id: String,
    text: String,
}

#[derive(Clone)]
struct ResponsesToolItem {
    output_index: usize,
    id: String,
    call_id: String,
    name: String,
    arguments: String,
}

impl ResponsesOutputState {
    fn new(model: &str, created: u64) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            id: format!("resp_{unique:x}"),
            model: model.to_owned(),
            created,
            sequence: 0,
            next_output_index: 0,
            message: None,
            reasoning: None,
            tools: HashMap::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            completed: false,
        }
    }

    fn start_chunks(&mut self) -> Vec<Value> {
        let response = self.response("in_progress", Vec::new(), false);
        vec![
            self.event("response.created", json!({"response":response.clone()})),
            self.event("response.in_progress", json!({"response":response})),
        ]
    }

    fn encode_chat_chunk(&mut self, chunk: &Value) -> Vec<Value> {
        if self.completed {
            return Vec::new();
        }
        if let Some(error) = chunk.get("error") {
            self.completed = true;
            let response = json!({
                "id":self.id,
                "object":"response",
                "created_at":self.created,
                "status":"failed",
                "model":self.model,
                "output":self.completed_output(),
                "error":error,
                "incomplete_details":Value::Null,
            });
            return vec![self.event("response.failed", json!({"response":response}))];
        }
        if let Some(usage) = chunk.get("usage").filter(|value| value.is_object()) {
            self.prompt_tokens = number(usage.get("prompt_tokens"));
            self.completion_tokens = number(usage.get("completion_tokens"));
        }

        let mut output = Vec::new();
        let delta = chunk
            .pointer("/choices/0/delta")
            .filter(|value| value.is_object());
        if let Some(reasoning) = delta
            .and_then(|value| value.get("reasoning_content"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            output.extend(self.ensure_reasoning_started());
            let item = self.reasoning.as_mut().expect("reasoning item");
            item.text.push_str(reasoning);
            let output_index = item.output_index;
            let item_id = item.id.clone();
            output.push(self.event(
                "response.reasoning_summary_text.delta",
                json!({
                    "output_index":output_index,
                    "item_id":item_id,
                    "summary_index":0,
                    "delta":reasoning,
                }),
            ));
        }
        if let Some(text) = delta
            .and_then(|value| value.get("content"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            output.extend(self.ensure_message_started());
            let item = self.message.as_mut().expect("message item");
            item.text.push_str(text);
            let output_index = item.output_index;
            let item_id = item.id.clone();
            output.push(self.event(
                "response.output_text.delta",
                json!({
                    "output_index":output_index,
                    "item_id":item_id,
                    "content_index":0,
                    "delta":text,
                    "logprobs":[],
                }),
            ));
        }
        for tool in delta
            .and_then(|value| value.get("tool_calls"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = tool.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let call_id = tool.get("id").and_then(Value::as_str);
            let name = tool.pointer("/function/name").and_then(Value::as_str);
            let arguments = tool
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or_default();
            output.extend(self.ensure_tool_started(index, call_id, name));
            let item = self.tools.get_mut(&index).expect("tool item");
            if let Some(call_id) = call_id.filter(|value| !value.is_empty()) {
                item.call_id = call_id.to_owned();
            }
            if let Some(name) = name.filter(|value| !value.is_empty()) {
                item.name = name.to_owned();
            }
            item.arguments.push_str(arguments);
            let output_index = item.output_index;
            let item_id = item.id.clone();
            if !arguments.is_empty() {
                output.push(self.event(
                    "response.function_call_arguments.delta",
                    json!({
                        "output_index":output_index,
                        "item_id":item_id,
                        "delta":arguments,
                    }),
                ));
            }
        }

        output
    }

    fn ensure_message_started(&mut self) -> Vec<Value> {
        if self.message.is_some() {
            return Vec::new();
        }
        let output_index = self.take_output_index();
        let id = format!("msg_{}_{output_index}", self.id.trim_start_matches("resp_"));
        self.message = Some(ResponsesTextItem {
            output_index,
            id: id.clone(),
            text: String::new(),
        });
        let item = json!({
            "id":id,
            "type":"message",
            "status":"in_progress",
            "role":"assistant",
            "content":[],
        });
        vec![
            self.event(
                "response.output_item.added",
                json!({"output_index":output_index,"item":item}),
            ),
            self.event(
                "response.content_part.added",
                json!({
                    "output_index":output_index,
                    "item_id":id,
                    "content_index":0,
                    "part":{"type":"output_text","text":"","annotations":[],"logprobs":[]},
                }),
            ),
        ]
    }

    fn ensure_reasoning_started(&mut self) -> Vec<Value> {
        if self.reasoning.is_some() {
            return Vec::new();
        }
        let output_index = self.take_output_index();
        let id = format!("rs_{}_{output_index}", self.id.trim_start_matches("resp_"));
        self.reasoning = Some(ResponsesReasoningItem {
            output_index,
            id: id.clone(),
            text: String::new(),
        });
        vec![
            self.event(
                "response.output_item.added",
                json!({
                    "output_index":output_index,
                    "item":{"id":id,"type":"reasoning","status":"in_progress","summary":[]},
                }),
            ),
            self.event(
                "response.reasoning_summary_part.added",
                json!({
                    "output_index":output_index,
                    "item_id":id,
                    "summary_index":0,
                    "part":{"type":"summary_text","text":""},
                }),
            ),
        ]
    }

    fn ensure_tool_started(
        &mut self,
        index: usize,
        call_id: Option<&str>,
        name: Option<&str>,
    ) -> Vec<Value> {
        if self.tools.contains_key(&index) {
            return Vec::new();
        }
        let output_index = self.take_output_index();
        let suffix = self.id.trim_start_matches("resp_");
        let id = format!("fc_{suffix}_{index}");
        let call_id = call_id
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("call_{suffix}_{index}"));
        let name = name.unwrap_or_default().to_owned();
        self.tools.insert(
            index,
            ResponsesToolItem {
                output_index,
                id: id.clone(),
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: String::new(),
            },
        );
        vec![self.event(
            "response.output_item.added",
            json!({
                "output_index":output_index,
                "item":{
                    "id":id,
                    "type":"function_call",
                    "status":"in_progress",
                    "call_id":call_id,
                    "name":name,
                    "arguments":"",
                },
            }),
        )]
    }

    fn finish(&mut self) -> Vec<Value> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;
        let mut output = Vec::new();
        if let Some(reasoning) = self.reasoning.clone() {
            let output_index = reasoning.output_index;
            let item_id = reasoning.id.clone();
            let text = reasoning.text.clone();
            output.push(self.event(
                "response.reasoning_summary_text.done",
                json!({"output_index":output_index,"item_id":item_id,"summary_index":0,"text":text}),
            ));
            output.push(self.event(
                "response.reasoning_summary_part.done",
                json!({
                    "output_index":output_index,
                    "item_id":item_id,
                    "summary_index":0,
                    "part":{"type":"summary_text","text":text},
                }),
            ));
            output.push(self.event(
                "response.output_item.done",
                json!({"output_index":output_index,"item":self.reasoning_value(&reasoning)}),
            ));
        }
        if let Some(message) = self.message.clone() {
            let output_index = message.output_index;
            let item_id = message.id.clone();
            let text = message.text.clone();
            output.push(self.event(
                "response.output_text.done",
                json!({
                    "output_index":output_index,
                    "item_id":item_id,
                    "content_index":0,
                    "text":text,
                    "logprobs":[],
                }),
            ));
            output.push(self.event(
                "response.content_part.done",
                json!({
                    "output_index":output_index,
                    "item_id":item_id,
                    "content_index":0,
                    "part":{"type":"output_text","text":text,"annotations":[],"logprobs":[]},
                }),
            ));
            output.push(self.event(
                "response.output_item.done",
                json!({"output_index":output_index,"item":self.message_value(&message)}),
            ));
        }
        let mut tools = self.tools.values().cloned().collect::<Vec<_>>();
        tools.sort_by_key(|item| item.output_index);
        for tool in tools {
            let output_index = tool.output_index;
            let item_id = tool.id.clone();
            let arguments = tool.arguments.clone();
            output.push(self.event(
                "response.function_call_arguments.done",
                json!({"output_index":output_index,"item_id":item_id,"arguments":arguments}),
            ));
            output.push(self.event(
                "response.output_item.done",
                json!({"output_index":output_index,"item":self.tool_value(&tool)}),
            ));
        }
        let response = self.response("completed", self.completed_output(), true);
        output.push(self.event("response.completed", json!({"response":response})));
        output
    }

    fn take_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn event(&mut self, event_type: &str, mut payload: Value) -> Value {
        let sequence = self.sequence;
        self.sequence += 1;
        let object = payload.as_object_mut().expect("event payload object");
        object.insert("type".into(), Value::String(event_type.to_owned()));
        object.insert("sequence_number".into(), json!(sequence));
        payload
    }

    fn response(&self, status: &str, output: Vec<Value>, include_usage: bool) -> Value {
        json!({
            "id":self.id,
            "object":"response",
            "created_at":self.created,
            "status":status,
            "model":self.model,
            "output":output,
            "output_text":self.message.as_ref().map(|item| item.text.as_str()).unwrap_or_default(),
            "usage":include_usage.then(|| json!({
                "input_tokens":self.prompt_tokens,
                "output_tokens":self.completion_tokens,
                "total_tokens":self.prompt_tokens + self.completion_tokens,
                "input_tokens_details":{},
                "output_tokens_details":{},
            })),
            "error":Value::Null,
            "incomplete_details":Value::Null,
        })
    }

    fn completed_output(&self) -> Vec<Value> {
        let mut output = Vec::new();
        if let Some(reasoning) = &self.reasoning {
            output.push((reasoning.output_index, self.reasoning_value(reasoning)));
        }
        if let Some(message) = &self.message {
            output.push((message.output_index, self.message_value(message)));
        }
        for tool in self.tools.values() {
            output.push((tool.output_index, self.tool_value(tool)));
        }
        output.sort_by_key(|(index, _)| *index);
        output.into_iter().map(|(_, value)| value).collect()
    }

    fn message_value(&self, item: &ResponsesTextItem) -> Value {
        json!({
            "id":item.id,
            "type":"message",
            "status":"completed",
            "role":"assistant",
            "content":[{
                "type":"output_text",
                "text":item.text,
                "annotations":[],
                "logprobs":[],
            }],
        })
    }

    fn reasoning_value(&self, item: &ResponsesReasoningItem) -> Value {
        json!({
            "id":item.id,
            "type":"reasoning",
            "status":"completed",
            "summary":[{"type":"summary_text","text":item.text}],
        })
    }

    fn tool_value(&self, item: &ResponsesToolItem) -> Value {
        json!({
            "id":item.id,
            "type":"function_call",
            "status":"completed",
            "call_id":item.call_id,
            "name":item.name,
            "arguments":item.arguments,
        })
    }
}

fn number(value: Option<&Value>) -> i64 {
    value
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().map(|value| value as i64))
        })
        .unwrap_or(0)
}

pub(crate) fn responses_usage_to_chat(usage: Option<&Value>) -> Value {
    let prompt_tokens = number(usage.and_then(|value| {
        value
            .get("prompt_tokens")
            .or_else(|| value.get("input_tokens"))
    }));
    let completion_tokens = number(usage.and_then(|value| {
        value
            .get("completion_tokens")
            .or_else(|| value.get("output_tokens"))
    }));
    let total_tokens = usage
        .and_then(|value| value.get("total_tokens"))
        .map(|value| number(Some(value)))
        .unwrap_or(prompt_tokens + completion_tokens);
    let prompt_details = usage.and_then(|value| {
        value
            .get("prompt_tokens_details")
            .or_else(|| value.get("input_tokens_details"))
    });
    let completion_details = usage.and_then(|value| {
        value
            .get("completion_tokens_details")
            .or_else(|| value.get("output_tokens_details"))
    });
    json!({
        "prompt_tokens":prompt_tokens,
        "completion_tokens":completion_tokens,
        "total_tokens":total_tokens,
        "prompt_tokens_details":{
            "cached_tokens":number(prompt_details.and_then(|value| value.get("cached_tokens"))),
            "cache_write_tokens":number(prompt_details.and_then(|value| value.get("cache_write_tokens"))),
            "audio_tokens":number(prompt_details.and_then(|value| value.get("audio_tokens"))),
        },
        "completion_tokens_details":{
            "reasoning_tokens":number(completion_details.and_then(|value| value.get("reasoning_tokens"))),
            "audio_tokens":number(completion_details.and_then(|value| value.get("audio_tokens"))),
            "accepted_prediction_tokens":number(completion_details.and_then(|value| value.get("accepted_prediction_tokens"))),
            "rejected_prediction_tokens":number(completion_details.and_then(|value| value.get("rejected_prediction_tokens"))),
        },
    })
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

fn collect_named_text(value: &Value, key: &str, callback: &mut impl FnMut(&str)) {
    match value {
        Value::Object(object) => {
            if let Some(text) = object.get(key).and_then(Value::as_str) {
                callback(text);
            }
            for nested in object.values() {
                collect_named_text(nested, key, callback);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_named_text(item, key, callback);
            }
        }
        _ => {}
    }
}

fn find_key<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match value {
        Value::Object(object) => object
            .get(key)
            .or_else(|| object.values().find_map(|value| find_key(value, key))),
        Value::Array(items) => items.iter().find_map(|value| find_key(value, key)),
        _ => None,
    }
}

#[derive(Default)]
struct JsonObjectFramer {
    current: Vec<u8>,
    depth: usize,
    in_string: bool,
    escaped: bool,
}

impl JsonObjectFramer {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let mut frames = Vec::new();
        for byte in bytes.iter().copied() {
            if self.depth == 0 {
                if byte != b'{' {
                    continue;
                }
                self.current.clear();
                self.current.push(byte);
                self.depth = 1;
                continue;
            }
            self.current.push(byte);
            if self.current.len() > MAX_STREAM_FRAME_BYTES {
                return Err("upstream JSON stream frame exceeded 1 MiB".into());
            }
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => self.in_string = true,
                b'{' => self.depth += 1,
                b'}' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        frames.push(std::mem::take(&mut self.current));
                    }
                }
                _ => {}
            }
        }
        Ok(frames)
    }

    fn finish(self) -> Result<(), String> {
        if self.depth == 0 {
            Ok(())
        } else {
            Err("upstream JSON stream ended with an incomplete object".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_stream_maps_text_tools_and_usage() {
        let mut state = StreamState::new_with_options("gemini-public", OutputProtocol::Chat, true);
        let chunks = state.gemini(&json!({
            "candidates":[{"content":{"parts":[
                {"text":"hello"},
                {"functionCall":{"name":"lookup","args":{"q":"x"}},"thoughtSignature":"tool-signature"}
            ]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}
        }));
        assert_eq!(
            chunks[0].pointer("/choices/0/delta/content"),
            Some(&json!("hello"))
        );
        assert_eq!(
            chunks[1].pointer("/choices/0/delta/tool_calls/0/function/name"),
            Some(&json!("lookup"))
        );
        assert!(chunks[1]
            .pointer("/choices/0/delta/tool_calls/0/id")
            .and_then(Value::as_str)
            .is_some_and(|value| value.starts_with("call_dG9vbC1zaWduYXR1cmU.")));
        assert_eq!(
            chunks.last().unwrap().pointer("/usage/total_tokens"),
            Some(&json!(7))
        );
    }

    #[test]
    fn claude_stream_maps_incremental_tool_arguments() {
        let mut state = StreamState::new_with_options("claude-public", OutputProtocol::Chat, false);
        let start = state.claude(&json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup"}}));
        let delta = state.claude(&json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}));
        assert_eq!(
            start[0].pointer("/choices/0/delta/tool_calls/0/id"),
            Some(&json!("toolu_1"))
        );
        assert_eq!(
            delta[0].pointer("/choices/0/delta/tool_calls/0/function/arguments"),
            Some(&json!("{\"q\":"))
        );
    }

    #[test]
    fn responses_stream_reuses_chat_tool_indexes_across_response_identifiers() {
        let mut state = StreamState::new_with_options("public-model", OutputProtocol::Chat, false);
        let first_start = state.responses(&json!({
            "type":"response.output_item.added",
            "output_index":1,
            "item":{
                "type":"function_call",
                "id":"fc-a",
                "call_id":"call-a",
                "name":"now",
                "arguments":"",
            },
        }));
        let first_delta = state.responses(&json!({
            "type":"response.function_call_arguments.delta",
            "output_index":1,
            "item_id":"fc-a",
            "delta":"{\"timezone\":\"Europe/Berlin\"}",
        }));
        let second_start = state.responses(&json!({
            "type":"response.output_item.added",
            "output_index":2,
            "item":{
                "type":"function_call",
                "id":"fc-b",
                "call_id":"call-b",
                "name":"weather",
                "arguments":"",
            },
        }));
        let second_delta = state.responses(&json!({
            "type":"response.function_call_arguments.delta",
            "output_index":2,
            "item_id":"fc-b",
            "delta":"{\"city\":\"Berlin\"}",
        }));

        assert_eq!(
            first_start[0].pointer("/choices/0/delta/tool_calls/0/index"),
            Some(&json!(0))
        );
        assert_eq!(
            first_delta[0].pointer("/choices/0/delta/tool_calls/0/index"),
            Some(&json!(0))
        );
        assert_eq!(
            second_start[0].pointer("/choices/0/delta/tool_calls/0/index"),
            Some(&json!(1))
        );
        assert_eq!(
            second_delta[0].pointer("/choices/0/delta/tool_calls/0/index"),
            Some(&json!(1))
        );
        assert_eq!(state.next_tool_index, 2);
    }

    #[test]
    fn responses_stream_emits_usage_only_in_requested_final_chunk() {
        let mut state = StreamState::new_with_options("public-model", OutputProtocol::Chat, true);
        let chunks = state.responses(&json!({
            "type":"response.completed",
            "response":{
                "usage":{
                    "input_tokens":3,
                    "output_tokens":5,
                    "total_tokens":8,
                    "input_tokens_details":{"cached_tokens":2},
                    "output_tokens_details":{"reasoning_tokens":4},
                },
            },
        }));

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0]["usage"], Value::Null);
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[1]["choices"], json!([]));
        assert_eq!(chunks[1]["usage"]["prompt_tokens"], 3);
        assert_eq!(chunks[1]["usage"]["completion_tokens"], 5);
        assert_eq!(
            chunks[1]["usage"]["prompt_tokens_details"]["cached_tokens"],
            2
        );
        assert_eq!(
            chunks[1]["usage"]["completion_tokens_details"]["reasoning_tokens"],
            4
        );
    }

    #[test]
    fn responses_stream_omits_usage_when_not_requested() {
        let mut state = StreamState::new_with_options("public-model", OutputProtocol::Chat, false);
        let chunks = state.responses(&json!({
            "type":"response.completed",
            "response":{"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}},
        }));

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].get("usage").is_none());
    }

    #[test]
    fn chat_stream_maps_to_incremental_responses_events() {
        let mut state =
            StreamState::new_with_options("public-model", OutputProtocol::Responses, false);
        let started = state.start_chunks();
        let first = state.convert(
            Protocol::Chat,
            &json!({
                "id":"chatcmpl-a",
                "choices":[{"delta":{"role":"assistant","content":"hello "},"finish_reason":null}]
            }),
        );
        let second = state.convert(
            Protocol::Chat,
            &json!({
                "id":"chatcmpl-a",
                "choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]
            }),
        );
        state.convert(
            Protocol::Chat,
            &json!({
                "id":"chatcmpl-a",
                "choices":[],
                "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
            }),
        );
        let completed = state.responses.finish();

        assert_eq!(started[0]["type"], "response.created");
        assert_eq!(first[0]["type"], "response.output_item.added");
        assert_eq!(first[1]["type"], "response.content_part.added");
        assert_eq!(first[2]["type"], "response.output_text.delta");
        assert_eq!(second[0]["type"], "response.output_text.delta");
        assert_eq!(completed.last().unwrap()["type"], "response.completed");
        assert_eq!(
            completed.last().unwrap()["response"]["output_text"],
            "hello world"
        );
        assert_eq!(
            completed.last().unwrap()["response"]["usage"]["total_tokens"],
            5
        );
    }

    #[test]
    fn chat_tool_stream_maps_to_responses_function_call_events() {
        let mut state =
            StreamState::new_with_options("public-model", OutputProtocol::Responses, false);
        state.start_chunks();
        let start = state.convert(
            Protocol::Chat,
            &json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-a","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null}]
            }),
        );
        let delta = state.convert(
            Protocol::Chat,
            &json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":"tool_calls"}]
            }),
        );
        let completed = state.responses.finish();

        assert_eq!(start[0]["type"], "response.output_item.added");
        assert_eq!(start[1]["type"], "response.function_call_arguments.delta");
        assert_eq!(delta[0]["type"], "response.function_call_arguments.delta");
        let response = &completed.last().unwrap()["response"];
        assert_eq!(response["output"][0]["call_id"], "call-a");
        assert_eq!(response["output"][0]["name"], "lookup");
        assert_eq!(response["output"][0]["arguments"], "{\"q\":\"x\"}");
    }

    #[test]
    fn json_object_framer_handles_pretty_printed_arrays() {
        let mut framer = JsonObjectFramer::default();
        let mut frames = framer.feed(b"[\n {\"a\":1},\n {\"b\":{\"c\":2}}").unwrap();
        frames.extend(framer.feed(b"}\n]").unwrap());
        framer.finish().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(
            serde_json::from_slice::<Value>(&frames[1]).unwrap(),
            json!({"b":{"c":2}})
        );
    }
}
