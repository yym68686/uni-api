//! Gemini's public wire contract; provider selection remains in the gateway.
use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{HeaderValue, Response},
};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};

pub(crate) fn route(path: &str) -> Option<(&str, &str)> {
    let tail = path
        .strip_prefix("/v1beta/models/")
        .or_else(|| path.strip_prefix("/v1/models/"))?;
    let (model, action) = tail.rsplit_once(':')?;
    if model.is_empty()
        || model.contains(['?', '#', '%'])
        || model
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
    {
        return None;
    }
    matches!(
        action,
        "generateContent" | "streamGenerateContent" | "countTokens"
    )
    .then_some((model, action))
}

pub(crate) fn normalize_auth(request: &mut Request) {
    let query: Vec<(String, String)> =
        url::form_urlencoded::parse(request.uri().query().unwrap_or("").as_bytes())
            .into_owned()
            .collect();
    let key = request
        .headers()
        .get("x-goog-api-key")
        .cloned()
        .or_else(|| {
            query
                .iter()
                .find(|(k, _)| k == "key")
                .and_then(|(_, v)| HeaderValue::from_str(v).ok())
        });
    if !request.headers().contains_key("authorization")
        && !request.headers().contains_key("x-api-key")
    {
        if let Some(key) = key {
            request.headers_mut().insert("x-api-key", key);
        }
    }
    request.headers_mut().remove("x-goog-api-key");
    let mut safe = url::form_urlencoded::Serializer::new(String::new());
    for (k, v) in query.iter().filter(|(k, _)| k != "key") {
        safe.append_pair(k, v);
    }
    let query = safe.finish();
    let uri = format!(
        "{}{}{}",
        request.uri().path(),
        if query.is_empty() { "" } else { "?" },
        query
    );
    if let Ok(uri) = uri.parse() {
        *request.uri_mut() = uri;
    }
}

pub(crate) fn error(status: u16, message: &str) -> Value {
    let code = match status {
        400 | 422 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        429 => "RESOURCE_EXHAUSTED",
        501 => "UNIMPLEMENTED",
        503 => "UNAVAILABLE",
        504 => "DEADLINE_EXCEEDED",
        _ => "INTERNAL",
    };
    json!({"error":{"code":status,"message":message,"status":code}})
}

pub(crate) async fn wire_response(response: Response<Body>, array: bool) -> Response<Body> {
    let (mut parts, body) = response.into_parts();
    if !parts.status.is_success() {
        let bytes = to_bytes(body, 1024 * 1024).await.unwrap_or_default();
        let value = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
        let message = value
            .pointer("/error/message")
            .or_else(|| value.get("detail"))
            .and_then(Value::as_str)
            .unwrap_or("Gemini request failed");
        parts.headers.remove("content-length");
        parts.headers.remove("content-encoding");
        parts
            .headers
            .insert("content-type", HeaderValue::from_static("application/json"));
        let payload = error(parts.status.as_u16(), message).to_string();
        return Response::from_parts(parts, Body::from(payload));
    }
    if !array
        || !parts
            .headers
            .get("content-type")
            .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"))
    {
        return Response::from_parts(parts, body);
    }
    parts.headers.remove("content-length");
    parts
        .headers
        .insert("content-type", HeaderValue::from_static("application/json"));
    // One bounded SSE frame at a time; JSON-array clients still receive a stream.
    let output = stream::try_unfold(
        (body.into_data_stream(), Vec::new(), 0usize, false),
        |(mut input, mut buffer, mut count, mut done)| async move {
            if done {
                return Ok::<_, std::io::Error>(None);
            }
            if count == 0 {
                count = 1;
                return Ok(Some((
                    Bytes::from_static(b"["),
                    (input, buffer, count, done),
                )));
            }
            loop {
                if let Some((end, separator)) = super::provider_stream::next_event_boundary(&buffer)
                {
                    let frame: Vec<u8> = buffer.drain(..end + separator).collect();
                    let text = std::str::from_utf8(&frame).map_err(std::io::Error::other)?;
                    let data = text
                        .lines()
                        .filter_map(|l| l.strip_prefix("data:"))
                        .map(str::trim)
                        .collect::<Vec<_>>()
                        .join("\n");
                    if data.is_empty() {
                        continue;
                    }
                    let value: Value =
                        serde_json::from_str(&data).map_err(std::io::Error::other)?;
                    let encoded = value.to_string();
                    let bytes =
                        Bytes::from(format!("{}{encoded}", if count == 1 { "" } else { "," }));
                    count = 2;
                    return Ok(Some((bytes, (input, buffer, count, done))));
                }
                match input.next().await {
                    Some(Ok(chunk)) => {
                        buffer.extend_from_slice(&chunk);
                        if buffer.len() > 2 * 1024 * 1024 {
                            return Err(std::io::Error::other("Gemini frame too large"));
                        }
                    }
                    Some(Err(e)) => return Err(std::io::Error::other(e)),
                    None => {
                        if !buffer.iter().all(u8::is_ascii_whitespace) {
                            return Err(std::io::Error::other("incomplete Gemini SSE frame"));
                        }
                        done = true;
                        return Ok(Some((
                            Bytes::from_static(b"]"),
                            (input, buffer, count, done),
                        )));
                    }
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(output))
}

pub(crate) fn request_to_chat(value: &Value, model: &str, stream: bool) -> Result<Value, String> {
    let root = value
        .as_object()
        .ok_or("Gemini request must be an object")?;
    for key in root.keys() {
        if !matches!(
            key.as_str(),
            "contents"
                | "systemInstruction"
                | "system_instruction"
                | "generationConfig"
                | "generation_config"
                | "tools"
                | "toolConfig"
                | "tool_config"
        ) {
            return Err(format!(
                "Gemini field {key} requires a native Gemini provider"
            ));
        }
    }
    let contents = root
        .get("contents")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or("Gemini contents must be a nonempty array")?;
    let mut messages = Vec::new();
    if let Some(system) = value
        .get("systemInstruction")
        .or_else(|| value.get("system_instruction"))
    {
        let mut text = Vec::new();
        for part in system
            .get("parts")
            .and_then(Value::as_array)
            .ok_or("systemInstruction.parts must be an array")?
        {
            text.push(
                part.get("text")
                    .and_then(Value::as_str)
                    .ok_or("Only text system instructions can be converted")?,
            );
        }
        messages.push(json!({"role":"system","content":text.join("\n")}));
    }
    let mut pending = VecDeque::<(String, String)>::new();
    let mut ordinal = 0;
    for content in contents {
        let role = match content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
        {
            "model" => "assistant",
            "user" => "user",
            _ => return Err("Invalid Gemini content role".into()),
        };
        let parts = content
            .get("parts")
            .and_then(Value::as_array)
            .filter(|p| !p.is_empty())
            .ok_or("Gemini content parts must be nonempty")?;
        let mut blocks = Vec::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part.get("thought").and_then(Value::as_bool) != Some(true) {
                    blocks.push(json!({"type":"text","text":text}));
                }
            } else if let Some(call) = part
                .get("functionCall")
                .or_else(|| part.get("function_call"))
            {
                if role != "assistant" {
                    return Err("functionCall requires model role".into());
                }
                ordinal += 1;
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("gemini_call_{ordinal}"));
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or("Function name is required")?;
                let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                if !args.is_object() {
                    return Err("Function args must be an object".into());
                }
                pending.push_back((id.clone(), name.into()));
                calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}));
            } else if let Some(result) = part
                .get("functionResponse")
                .or_else(|| part.get("function_response"))
            {
                if role != "user" || !result.get("response").is_some_and(Value::is_object) {
                    return Err("functionResponse requires user role and an object response".into());
                }
                let name = result
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("Function response name is required")?;
                let id = result.get("id").and_then(Value::as_str);
                let pos = pending
                    .iter()
                    .position(|(i, n)| n == name && id.is_none_or(|id| id == i))
                    .ok_or("Function response has no matching call")?;
                let (id, _) = pending.remove(pos).expect("matched call");
                results.push(json!({"role":"tool","tool_call_id":id,"content":result.get("response").unwrap_or(&Value::Null).to_string()}));
            } else if let Some(data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
                let mime = data
                    .get("mimeType")
                    .or_else(|| data.get("mime_type"))
                    .and_then(Value::as_str)
                    .ok_or("inlineData.mimeType is required")?;
                if !mime.starts_with("image/") {
                    return Err(format!("{mime} requires a native Gemini provider"));
                }
                let data = data
                    .get("data")
                    .and_then(Value::as_str)
                    .ok_or("inlineData.data is required")?;
                blocks.push(json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{data}")}}));
            } else {
                return Err("This Gemini content part requires a native Gemini provider".into());
            }
        }
        // Tool results must precede the user's next text turn.
        messages.extend(results);
        if !blocks.is_empty() || !calls.is_empty() {
            let mut message = json!({"role":role,"content":blocks});
            if !calls.is_empty() {
                message["tool_calls"] = json!(calls);
            }
            messages.push(message);
        }
    }
    let mut out = json!({"model":model,"messages":messages,"stream":stream});
    if let Some(config) = value
        .get("generationConfig")
        .or_else(|| value.get("generation_config"))
    {
        let config = config
            .as_object()
            .ok_or("generationConfig must be an object")?;
        for (key, v) in config {
            let target = match key.as_str() {
                "temperature" => "temperature",
                "topP" | "top_p" => "top_p",
                "topK" | "top_k" => "top_k",
                "maxOutputTokens" | "max_output_tokens" => "max_tokens",
                "stopSequences" | "stop_sequences" => "stop",
                "candidateCount" | "candidate_count" => "n",
                "seed" => "seed",
                "thinkingConfig" | "thinking_config" => {
                    if v.get("thinkingBudget")
                        .or_else(|| v.get("thinking_budget"))
                        .is_some()
                    {
                        return Err(
                            "Exact thinking budgets require a native Gemini provider".into()
                        );
                    }
                    if let Some(effort) = v.get("thinkingLevel").or_else(|| v.get("thinking_level"))
                    {
                        out["reasoning_effort"] = effort.clone();
                    } else {
                        return Err(
                            "Exact thinking budgets require a native Gemini provider".into()
                        );
                    }
                    continue;
                }
                "responseMimeType"
                | "response_mime_type"
                | "responseSchema"
                | "response_schema"
                | "responseJsonSchema"
                | "response_json_schema" => continue,
                _ => {
                    return Err(format!(
                        "generationConfig.{key} requires a native Gemini provider"
                    ))
                }
            };
            out[target] = v.clone();
        }
        let mime = config
            .get("responseMimeType")
            .or_else(|| config.get("response_mime_type"))
            .and_then(Value::as_str);
        if mime == Some("application/json") {
            let schema = config
                .get("responseJsonSchema")
                .or_else(|| config.get("response_json_schema"))
                .or_else(|| config.get("responseSchema"))
                .or_else(|| config.get("response_schema"));
            out["response_format"] = if let Some(schema) = schema {
                let mut schema = schema.clone();
                normalize_schema(&mut schema);
                json!({"type":"json_schema","json_schema":{"name":"response","schema":schema}})
            } else {
                json!({"type":"json_object"})
            };
        } else if mime.is_some_and(|m| m != "text/plain") {
            return Err("Non-text response modalities require a native Gemini provider".into());
        }
    }
    if let Some(tools) = value.get("tools") {
        let mut functions = Vec::new();
        for tool in tools.as_array().ok_or("tools must be an array")? {
            if tool.as_object().is_none_or(|tool| {
                tool.keys()
                    .any(|key| key != "functionDeclarations" && key != "function_declarations")
            }) {
                return Err("Hosted Gemini tools require a native Gemini provider".into());
            }
            let declarations = tool
                .get("functionDeclarations")
                .or_else(|| tool.get("function_declarations"))
                .and_then(Value::as_array)
                .ok_or("Hosted Gemini tools require a native Gemini provider")?;
            for function in declarations {
                let mut function = function.clone();
                if function.get("behavior").is_some() {
                    return Err("Asynchronous functions require a native Gemini provider".into());
                }
                if let Some(schema) = function
                    .get("parametersJsonSchema")
                    .or_else(|| function.get("parameters_json_schema"))
                    .cloned()
                {
                    function["parameters"] = schema;
                    function
                        .as_object_mut()
                        .expect("function")
                        .remove("parametersJsonSchema");
                    function
                        .as_object_mut()
                        .expect("function")
                        .remove("parameters_json_schema");
                }
                if let Some(schema) = function.get_mut("parameters") {
                    normalize_schema(schema);
                }
                functions.push(json!({"type":"function","function":function}));
            }
        }
        out["tools"] = json!(functions);
    }
    if let Some(config) = value.get("toolConfig").or_else(|| value.get("tool_config")) {
        let c = config
            .get("functionCallingConfig")
            .or_else(|| config.get("function_calling_config"))
            .ok_or("Unsupported toolConfig")?;
        let mode = c.get("mode").and_then(Value::as_str).unwrap_or("AUTO");
        out["tool_choice"] = match mode {
            "AUTO" => json!("auto"),
            "NONE" => json!("none"),
            "ANY" => json!("required"),
            _ => return Err("Unsupported function calling mode".into()),
        };
        if let Some(names) = c
            .get("allowedFunctionNames")
            .or_else(|| c.get("allowed_function_names"))
        {
            let names = names
                .as_array()
                .ok_or("allowedFunctionNames must be an array")?;
            if mode != "ANY" || names.len() != 1 {
                return Err("Conversion requires exactly one allowed function in ANY mode".into());
            }
            out["tool_choice"] = json!({"type":"function","function":{"name":names[0]}});
        }
    }
    Ok(out)
}

fn normalize_schema(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(kind)) = map.get_mut("type") {
                *kind = kind.to_ascii_lowercase();
            }
            for (k, v) in map.iter_mut() {
                if k != "enum" && k != "const" {
                    normalize_schema(v);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_schema(item);
            }
        }
        _ => {}
    }
}

pub(crate) fn usage_from_chat(u: &Value) -> Value {
    let n = |v: Option<&Value>| v.and_then(Value::as_i64).unwrap_or(0).max(0);
    let thoughts = n(u.pointer("/completion_tokens_details/reasoning_tokens"));
    let output = n(u.get("completion_tokens"));
    let input = n(u.get("prompt_tokens"));
    // Anthropic reports cache creation/read outside input_tokens, whereas Chat
    // providers include cached tokens in prompt_tokens already.
    let input = if u.get("input_tokens").is_some() {
        input
            .saturating_add(
                n(u.get("cache_creation_input_tokens"))
                    .max(n(u.pointer("/prompt_tokens_details/cache_write_tokens"))),
            )
            .saturating_add(n(u.get("cache_read_input_tokens")))
    } else {
        input
    };
    json!({"promptTokenCount":input,"candidatesTokenCount":output.saturating_sub(thoughts),"thoughtsTokenCount":thoughts,"totalTokenCount":u.get("total_tokens").cloned().unwrap_or(json!(input.saturating_add(output))),"cachedContentTokenCount":n(u.pointer("/prompt_tokens_details/cached_tokens")),"uniApiUsage":u})
}

pub(crate) fn chat_finish_reason(reason: &str) -> &'static str {
    match reason {
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            "content_filter"
        }
        _ => "stop",
    }
}

pub(crate) fn finish_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

fn message_parts(message: &Value) -> Vec<Value> {
    let mut parts = Vec::new();
    if let Some(text) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        parts.push(json!({"text":text,"thought":true}));
    }
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        parts.push(json!({"text":text}));
    }
    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        for block in blocks {
            if let Some(text) = block.get("text") {
                parts.push(json!({"text":text}));
            }
        }
    }
    for tool in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let args = tool
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let args =
            serde_json::from_str::<Value>(args).unwrap_or_else(|_| json!({"arguments":args}));
        parts.push(
            json!({"functionCall":{"id":tool["id"],"name":tool["function"]["name"],"args":args}}),
        );
    }
    parts
}

pub(crate) fn response_from_chat(value: &Value, model: &str) -> Value {
    let candidates: Vec<_> = value.get("choices").and_then(Value::as_array).into_iter().flatten().enumerate().map(|(index, c)| json!({"index":c.get("index").cloned().unwrap_or(json!(index)),"content":{"role":"model","parts":message_parts(&c["message"])},"finishReason":finish_reason(c["finish_reason"].as_str().unwrap_or("stop"))})).collect();
    let mut out = json!({"candidates":candidates,"modelVersion":model,"responseId":value["id"]});
    if let Some(u) = value.get("usage").filter(|u| u.is_object()) {
        out["usageMetadata"] = usage_from_chat(u);
    }
    out
}

#[derive(Default)]
pub(crate) struct OutputState {
    tools: BTreeMap<(i64, i64), Value>,
    bytes: usize,
}

impl OutputState {
    pub(crate) fn encode(&mut self, chunk: &Value, model: &str) -> Result<Vec<Value>, String> {
        if let Some(e) = chunk.get("error") {
            return Ok(vec![error(
                e.get("status_code").and_then(Value::as_u64).unwrap_or(502) as u16,
                e.get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Upstream failed"),
            )]);
        }
        let mut candidates = Vec::new();
        for c in chunk
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = c["index"].as_i64().unwrap_or(0);
            let delta = &c["delta"];
            let mut parts = message_parts(
                &json!({"content":delta["content"],"reasoning_content":delta["reasoning_content"]}),
            );
            for t in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let key = (index, t["index"].as_i64().unwrap_or(0));
                let target = self.tools.entry(key).or_insert_with(
                    || json!({"id":"","type":"function","function":{"name":"","arguments":""}}),
                );
                if let Some(id) = t.get("id") {
                    target["id"] = id.clone();
                }
                for field in ["name", "arguments"] {
                    if let Some(s) = t["function"][field].as_str() {
                        self.bytes = self.bytes.saturating_add(s.len());
                        if self.bytes > 1024 * 1024 {
                            return Err("Gemini tool arguments exceeded 1 MiB".into());
                        }
                        let old = target["function"][field].as_str().unwrap_or("");
                        target["function"][field] = json!(format!("{old}{s}"));
                    }
                }
            }
            let finish = c.get("finish_reason").and_then(Value::as_str);
            if finish.is_some() {
                let keys: Vec<_> = self
                    .tools
                    .keys()
                    .filter(|(choice, _)| *choice == index)
                    .copied()
                    .collect();
                for key in keys {
                    let tool = self.tools.remove(&key).expect("tool");
                    let raw = tool["function"]["arguments"].as_str().unwrap_or("");
                    let args: Value = serde_json::from_str(if raw.is_empty() { "{}" } else { raw })
                        .map_err(|_| "Upstream returned incomplete function arguments")?;
                    if !args.is_object() {
                        return Err("Function arguments must be a JSON object".into());
                    }
                    parts.extend(message_parts(&json!({"tool_calls":[tool]})));
                }
            }
            if !parts.is_empty() || finish.is_some() {
                let mut candidate = json!({"index":index,"content":{"role":"model","parts":parts}});
                if let Some(f) = finish {
                    candidate["finishReason"] = json!(finish_reason(f));
                }
                candidates.push(candidate);
            }
        }
        let mut out = json!({"candidates":candidates,"modelVersion":model});
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            out["usageMetadata"] = usage_from_chat(u);
        }
        if out["candidates"].as_array().is_some_and(Vec::is_empty)
            && out.get("usageMetadata").is_none()
        {
            Ok(vec![])
        } else {
            Ok(vec![out])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_parallel_tool_history_and_json_schema() {
        let request = request_to_chat(&json!({
            "systemInstruction":{"parts":[{"text":"Be brief"}]},
            "contents":[
                {"role":"model","parts":[{"functionCall":{"id":"a","name":"lookup","args":{"q":"a"}}},{"functionCall":{"id":"b","name":"lookup","args":{"q":"b"}}}]},
                {"role":"user","parts":[{"functionResponse":{"id":"b","name":"lookup","response":{"result":2}}},{"functionResponse":{"id":"a","name":"lookup","response":{"result":1}}},{"text":"summarize"}]}
            ],
            "generationConfig":{"responseMimeType":"application/json","responseSchema":{"type":"OBJECT","properties":{"name":{"type":"STRING"}}}},
            "tools":[{"functionDeclarations":[{"name":"lookup","parametersJsonSchema":{"type":"object"}}]}]
        }), "target", true).unwrap();
        assert_eq!(request["messages"][2]["tool_call_id"], "b");
        assert_eq!(request["messages"][3]["tool_call_id"], "a");
        assert_eq!(request["messages"][4]["role"], "user");
        assert_eq!(
            request["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        assert_eq!(
            request["tools"][0]["function"]["parameters"]["type"],
            "object"
        );
    }

    #[test]
    fn rejects_native_only_controls_without_losing_them() {
        let base = json!({"contents":[{"parts":[{"text":"hi"}]}]});
        for (field, value) in [
            ("cachedContent", json!("cachedContents/1")),
            (
                "tools",
                json!([{"functionDeclarations":[],"googleSearch":{}}]),
            ),
            (
                "generationConfig",
                json!({"thinkingConfig":{"thinkingLevel":"high","thinkingBudget":10}}),
            ),
        ] {
            let mut request = base.clone();
            request[field] = value;
            assert!(request_to_chat(&request, "model", false).is_err());
        }
    }

    #[test]
    fn preserves_anthropic_cache_accounting_and_reasoning() {
        let original = crate::protocols::provider_stream::claude_usage_to_chat(&json!({
            "input_tokens":40,"cache_read_input_tokens":60,"cache_creation_input_tokens":10,
            "output_tokens":20,"output_tokens_details":{"thinking_tokens":8}
        }));
        let usage = usage_from_chat(&original);
        assert_eq!(usage["promptTokenCount"], 110);
        assert_eq!(usage["candidatesTokenCount"], 12);
        assert_eq!(usage["thoughtsTokenCount"], 8);
        assert_eq!(usage["totalTokenCount"], 130);
        assert_eq!(usage["uniApiUsage"], original);
    }

    #[test]
    fn fragmented_tools_reject_incomplete_arguments() {
        let mut state = OutputState::default();
        assert!(state.encode(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}),"model").unwrap().is_empty());
        assert!(state
            .encode(
                &json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
                "model"
            )
            .is_err());
    }
}
