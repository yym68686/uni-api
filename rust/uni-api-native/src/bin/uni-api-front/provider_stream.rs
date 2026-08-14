use std::collections::HashMap;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{HeaderValue, Response};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use crc32fast::hash as crc32;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

const MAX_STREAM_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Protocol {
    Responses,
    Gemini,
    Claude,
    VertexClaude,
    Cohere,
    AwsBedrock,
    Cloudflare,
}

pub struct Translation {
    pub response: Response<Body>,
    pub usage: oneshot::Receiver<(i64, i64, i64)>,
}

pub fn translate(response: reqwest::Response, protocol: Protocol, model: String) -> Translation {
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
        let result = run_translation(response, protocol, &model, &content_type, &tx).await;
        let usage = match result {
            Ok(usage) => usage,
            Err(error) => {
                let payload = json!({"error":{"message":error}});
                let _ = send_wire(&tx, &payload).await;
                (0, 0, 0)
            }
        };
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
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
    model: &str,
    content_type: &str,
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
) -> Result<(i64, i64, i64), String> {
    let mut state = StreamState::new(model);
    let mut stream = response.bytes_stream();
    if protocol == Protocol::AwsBedrock {
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
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
            Protocol::Responses | Protocol::Claude | Protocol::Cloudflare
        );
    if sse {
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
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
        while let Some(chunk) = stream.next().await {
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
        while let Some(chunk) = stream.next().await {
            for frame in framer.feed(&chunk.map_err(|error| error.to_string())?)? {
                process_json_bytes(&frame, protocol, &mut state, tx).await?;
            }
        }
        framer.finish()?;
    }
    state.finish_if_needed(tx).await?;
    Ok(state.usage())
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
        send_wire(tx, &chunk).await?;
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
            send_wire(tx, &chunk).await?;
        }
        if let Some(metrics) = payload.get("amazon-bedrock-invocationMetrics") {
            state.prompt_tokens = number(metrics.get("inputTokenCount"));
            state.completion_tokens = number(metrics.get("outputTokenCount"));
            for chunk in state.finish_chunks("stop") {
                send_wire(tx, &chunk).await?;
            }
        }
    }
}

async fn send_wire(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    value: &Value,
) -> Result<(), String> {
    let mut wire = b"data: ".to_vec();
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
    prompt_tokens: i64,
    completion_tokens: i64,
    terminal: bool,
}

impl StreamState {
    fn new(model: &str) -> Self {
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
            prompt_tokens: 0,
            completion_tokens: 0,
            terminal: false,
        }
    }

    fn convert(&mut self, protocol: Protocol, value: &Value) -> Vec<Value> {
        match protocol {
            Protocol::Responses => self.responses(value),
            Protocol::Gemini => self.gemini(value),
            Protocol::Claude => self.claude(value),
            Protocol::VertexClaude => self.claude(value),
            Protocol::Cohere => self.cohere(value),
            Protocol::Cloudflare => self.cloudflare(value),
            Protocol::AwsBedrock => Vec::new(),
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
                let call_id = value
                    .pointer("/item/call_id")
                    .or_else(|| value.pointer("/item/id"))
                    .and_then(Value::as_str)
                    .unwrap_or("call");
                let index = self.tool_index(call_id);
                vec![self.chunk(
                    json!({"tool_calls":[{"index":index,"id":call_id,"type":"function","function":{"name":value.pointer("/item/name").cloned().unwrap_or(Value::Null),"arguments":""}}]}),
                    None,
                    None,
                )]
            }
            "response.function_call_arguments.delta" => {
                let call_id = value
                    .get("call_id")
                    .or_else(|| value.get("item_id"))
                    .and_then(Value::as_str)
                    .unwrap_or("call");
                let index = self.tool_index(call_id);
                vec![self.chunk(
                    json!({"tool_calls":[{"index":index,"function":{"arguments":value.get("delta").cloned().unwrap_or(Value::String(String::new()))}}]}),
                    None,
                    None,
                )]
            }
            "response.completed" => {
                self.prompt_tokens = number(value.pointer("/response/usage/input_tokens"));
                self.completion_tokens = number(value.pointer("/response/usage/output_tokens"));
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
                    .unwrap_or_else(|| format!("call_{}", self.next_tool_index + 1));
                let index = self.tool_index(&call_id);
                output.push(self.chunk(
                    json!({"tool_calls":[{"index":index,"id":call_id,"type":"function","function":{"name":call.get("name").cloned().unwrap_or(Value::Null),"arguments":serde_json::to_string(call.get("args").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into())}}]}),
                    None,
                    None,
                ));
            }
            if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) {
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
        }
        chunk
    }

    fn finish_chunks(&mut self, reason: &str) -> Vec<Value> {
        if self.terminal {
            return Vec::new();
        }
        self.terminal = true;
        let usage = json!({
            "prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "total_tokens":self.prompt_tokens + self.completion_tokens,
        });
        vec![self.chunk(json!({}), Some(reason), Some(usage))]
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
        for chunk in self.finish_chunks(if self.tools.is_empty() {
            "stop"
        } else {
            "tool_calls"
        }) {
            send_wire(tx, &chunk).await?;
        }
        Ok(())
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
        let mut state = StreamState::new("gemini-public");
        let chunks = state.gemini(&json!({
            "candidates":[{"content":{"parts":[
                {"text":"hello"},
                {"functionCall":{"name":"lookup","args":{"q":"x"}}}
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
        assert_eq!(
            chunks.last().unwrap().pointer("/usage/total_tokens"),
            Some(&json!(7))
        );
    }

    #[test]
    fn claude_stream_maps_incremental_tool_arguments() {
        let mut state = StreamState::new("claude-public");
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
