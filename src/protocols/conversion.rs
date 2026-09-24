use crate::config::snapshot::Provider;
use crate::protocols::content::content_text;
use crate::protocols::provider_stream;
use crate::runtime::clock::unix_seconds;
use axum::body::Body;
use axum::http::{HeaderValue, Response};
use serde_json::{json, Map, Value};
use url::Url;

pub(crate) fn responses_to_chat_request(
    input: &Value,
    original_model: &str,
) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "responses request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    if let Some(instructions) = root
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        messages.push(json!({"role":"system","content":instructions}));
    }
    match root.get("input") {
        Some(Value::String(text)) => messages.push(json!({"role":"user","content":text})),
        Some(Value::Array(items)) => {
            for item in items {
                let item_type = item.get("type").and_then(Value::as_str);
                match item_type {
                    Some("function_call") => messages.push(json!({
                        "role":"assistant",
                        "content":Value::Null,
                        "tool_calls":[{
                            "id":item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Value::Null),
                            "type":"function",
                            "function":{
                                "name":item.get("name").cloned().unwrap_or(Value::Null),
                                "arguments":item.get("arguments").cloned().unwrap_or_else(|| Value::String("{}".into())),
                            }
                        }]
                    })),
                    Some("function_call_output") => messages.push(json!({
                        "role":"tool",
                        "tool_call_id":item.get("call_id").cloned().unwrap_or(Value::Null),
                        "content":item.get("output").cloned().unwrap_or_else(|| Value::String(String::new())),
                    })),
                    Some("message") | None if item.get("role").is_some() => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        messages.push(json!({
                            "role":role,
                            "content":responses_input_to_chat_content(item.get("content")),
                        }));
                    }
                    _ => {}
                }
            }
        }
        Some(value) if !value.is_null() => {
            messages.push(json!({"role":"user","content":value.clone()}));
        }
        _ => {}
    }
    if messages.is_empty() {
        return Err("responses request requires input".into());
    }

    let mut output = Map::new();
    output.insert("model".into(), Value::String(original_model.to_owned()));
    output.insert("messages".into(), Value::Array(messages));
    output.insert(
        "stream".into(),
        Value::Bool(root.get("stream").and_then(Value::as_bool).unwrap_or(false)),
    );
    if let Some(value) = root.get("max_output_tokens") {
        output.insert("max_tokens".into(), value.clone());
    }
    for name in [
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "service_tier",
        "reasoning",
        "reasoning_effort",
        "modalities",
        "audio",
        "metadata",
        "user",
    ] {
        if let Some(value) = root.get(name) {
            output.insert(name.into(), value.clone());
        }
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        output.insert(
            "tools".into(),
            Value::Array(
                tools
                    .iter()
                    .filter_map(|tool| {
                        if tool.get("type").and_then(Value::as_str) != Some("function") {
                            return None;
                        }
                        let function = tool.get("function").unwrap_or(tool);
                        let mut definition = Map::new();
                        definition.insert(
                            "name".into(),
                            function.get("name").cloned().unwrap_or(Value::Null),
                        );
                        if let Some(value) = function.get("description") {
                            definition.insert("description".into(), value.clone());
                        }
                        definition.insert(
                            "parameters".into(),
                            function
                                .get("parameters")
                                .cloned()
                                .unwrap_or_else(|| json!({"type":"object","properties":{}})),
                        );
                        if let Some(value) = function.get("strict") {
                            definition.insert("strict".into(), value.clone());
                        }
                        Some(json!({"type":"function","function":definition}))
                    })
                    .collect(),
            ),
        );
    }
    if let Some(choice) = root.get("tool_choice") {
        let choice = if choice.get("type").and_then(Value::as_str) == Some("function") {
            json!({
                "type":"function",
                "function":{"name":choice.get("name").cloned().unwrap_or(Value::Null)}
            })
        } else {
            choice.clone()
        };
        output.insert("tool_choice".into(), choice);
    }
    Ok(Value::Object(output))
}

pub(crate) fn responses_input_to_chat_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return Value::String(String::new());
    };
    if let Some(text) = content.as_str() {
        return Value::String(text.to_owned());
    }
    Value::Array(
        content
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("input_text" | "output_text" | "text") => Some(json!({
                    "type":"text",
                    "text":part.get("text").cloned().unwrap_or_else(|| Value::String(String::new())),
                })),
                Some("input_image") => Some(json!({
                    "type":"image_url",
                    "image_url":{"url":part.get("image_url").cloned().unwrap_or(Value::Null)},
                })),
                Some("input_audio") => Some(json!({
                    "type":"input_audio",
                    "input_audio":part.get("input_audio").cloned().unwrap_or_else(|| {
                        json!({
                            "data":part.get("data").cloned().unwrap_or(Value::Null),
                            "format":part.get("format").cloned().unwrap_or(Value::Null),
                        })
                    }),
                })),
                _ => None,
            })
            .collect(),
    )
}

pub(crate) fn chat_to_responses(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut items = Vec::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(message) = message.as_object() else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if role == "tool" {
            items.push(json!({
                "type":"function_call_output",
                "call_id": message.get("tool_call_id").cloned().unwrap_or(Value::Null),
                "output": message.get("content").cloned().unwrap_or(Value::String(String::new())),
            }));
            continue;
        }
        if let Some(content) = message.get("content") {
            items.push(json!({"role":role,"content":responses_content(content, role)}));
        }
        for tool in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            items.push(json!({
                "type":"function_call",
                "call_id":tool.get("id").cloned().unwrap_or(Value::Null),
                "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments":tool.pointer("/function/arguments").cloned().unwrap_or(Value::String("{}".into())),
            }));
        }
    }
    let mut output = Map::new();
    output.insert("model".into(), Value::String(original_model.to_owned()));
    output.insert("input".into(), Value::Array(items));
    output.insert("stream".into(), Value::Bool(false));
    output.insert("store".into(), Value::Bool(false));
    if let Some(max_tokens) = root.get("max_tokens") {
        output.insert("max_output_tokens".into(), max_tokens.clone());
    }
    for name in [
        "temperature",
        "top_p",
        "reasoning",
        "parallel_tool_calls",
        "service_tier",
    ] {
        if let Some(value) = root.get(name) {
            output.insert(name.into(), value.clone());
        }
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        output.insert(
            "tools".into(),
            Value::Array(
                tools
                    .iter()
                    .filter_map(|tool| {
                        let function = tool.get("function")?;
                        Some(json!({
                            "type":"function",
                            "name":function.get("name").cloned().unwrap_or(Value::Null),
                            "description":function.get("description").cloned().unwrap_or(Value::Null),
                            "parameters":function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                            "strict":function.get("strict").and_then(Value::as_bool).unwrap_or(false),
                        }))
                    })
                    .collect(),
            ),
        );
    }
    if let Some(choice) = root.get("tool_choice") {
        output.insert("tool_choice".into(), chat_tool_choice_to_responses(choice));
    }
    Ok(Value::Object(output))
}

pub(crate) fn chat_tool_choice_to_responses(choice: &Value) -> Value {
    let Some(root) = choice.as_object() else {
        return choice.clone();
    };
    match root.get("type").and_then(Value::as_str) {
        Some("function") => json!({
            "type":"function",
            "name":choice
                .pointer("/function/name")
                .or_else(|| choice.get("name"))
                .cloned()
                .unwrap_or(Value::Null),
        }),
        Some("allowed_tools") => {
            let allowed = root
                .get("allowed_tools")
                .and_then(Value::as_object)
                .unwrap_or(root);
            let tools = allowed
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|tool| {
                    if tool.get("type").and_then(Value::as_str) == Some("function") {
                        json!({
                            "type":"function",
                            "name":tool
                                .pointer("/function/name")
                                .or_else(|| tool.get("name"))
                                .cloned()
                                .unwrap_or(Value::Null),
                        })
                    } else {
                        tool.clone()
                    }
                })
                .collect::<Vec<_>>();
            json!({
                "type":"allowed_tools",
                "mode":allowed.get("mode").cloned().unwrap_or_else(|| Value::String("auto".into())),
                "tools":tools,
            })
        }
        _ => choice.clone(),
    }
}

pub(crate) fn responses_content(content: &Value, role: &str) -> Value {
    if let Some(text) = content.as_str() {
        return Value::Array(vec![json!({
            "type": if role == "assistant" { "output_text" } else { "input_text" },
            "text": text,
        })]);
    }
    let mut parts = Vec::new();
    for item in content.as_array().into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(json!({
                "type": if role == "assistant" { "output_text" } else { "input_text" },
                "text": item.get("text").cloned().unwrap_or(Value::String(String::new())),
            })),
            Some("image_url") => parts.push(json!({
                "type":"input_image",
                "image_url":item.pointer("/image_url/url").cloned().unwrap_or(Value::Null),
            })),
            _ => parts.push(item.clone()),
        }
    }
    Value::Array(parts)
}

pub(crate) fn chat_to_cohere(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = match message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
        {
            "assistant" => "CHATBOT",
            "system" => "SYSTEM",
            _ => "USER",
        };
        if let Some(text) = content_text(message.get("content")) {
            messages.push(json!({"role":role,"message":text}));
        }
    }
    let last = messages
        .pop()
        .ok_or_else(|| "Cohere chat request requires at least one message".to_owned())?;
    let mut output = json!({
        "model":original_model,
        "message":last.get("message").cloned().unwrap_or(Value::String(String::new())),
    });
    if !messages.is_empty() {
        output
            .as_object_mut()
            .expect("Cohere payload object")
            .insert("chat_history".into(), Value::Array(messages));
    }
    Ok(output)
}

pub(crate) fn chat_to_doubao_translation(
    input: &Value,
    original_model: &str,
    request_model: &str,
    provider: &Provider,
) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let user_text = root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .filter_map(Value::as_object)
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .map(extract_translation_text)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "No user message".to_owned())?;
    let translation_overrides = provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get(request_model))
        .and_then(Value::as_object)
        .and_then(|overrides| overrides.get("translation_options"))
        .and_then(Value::as_object);
    let mut options = Map::from_iter([("target_language".into(), Value::String("zh".into()))]);
    if let Some(overrides) = translation_overrides {
        for key in ["source_language", "target_language"] {
            if let Some(value) = overrides
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                options.insert(key.into(), Value::String(value.to_owned()));
            }
        }
    }
    let mut output = json!({
        "model":original_model,
        "input":[{
            "role":"user",
            "content":[{
                "type":"input_text",
                "text":user_text,
                "translation_options":options,
            }],
        }],
    });
    if root.get("stream").and_then(Value::as_bool) == Some(true) {
        output["stream"] = Value::Bool(true);
    }
    Ok(output)
}

pub(crate) fn extract_translation_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(extract_translation_text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(item) => ["text", "content", "input"]
            .into_iter()
            .find_map(|key| item.get(key).map(extract_translation_text))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(crate) fn openai_tts_to_minimax(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "text-to-speech request body must be an object".to_owned())?;
    let text = root
        .get("input")
        .cloned()
        .ok_or_else(|| "text-to-speech input is required".to_owned())?;
    let voice = root
        .get("voice")
        .cloned()
        .ok_or_else(|| "text-to-speech voice is required".to_owned())?;
    let mut output = Map::from_iter([
        ("model".into(), Value::String(original_model.to_owned())),
        ("text".into(), text),
        ("voice_setting".into(), json!({"voice_id":voice})),
    ]);
    for key in ["response_format", "speed", "stream"] {
        if let Some(value) = root.get(key) {
            output.insert(key.into(), value.clone());
        }
    }
    Ok(Value::Object(output))
}

pub(crate) fn normalize_jina_embedding(
    input: &mut Value,
    original_model: &str,
) -> Result<(), String> {
    let root = input
        .as_object_mut()
        .ok_or_else(|| "embedding request body must be an object".to_owned())?;
    root.insert("model".into(), Value::String(original_model.to_owned()));
    if let Some(format) = root.remove("encoding_format") {
        root.insert("embedding_type".into(), format);
    }
    Ok(())
}

pub(crate) fn chat_to_cloudflare(input: &Value) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let prompt = root
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| content_text(message.get("content")))
        .ok_or_else(|| "Cloudflare chat request requires a text message".to_owned())?;
    let mut output = json!({"prompt":prompt});
    if let Some(root) = output.as_object_mut() {
        for name in ["temperature", "top_p", "max_tokens", "seed"] {
            if let Some(value) = input.get(name) {
                root.insert(name.into(), value.clone());
            }
        }
    }
    Ok(output)
}

pub(crate) fn responses_to_chat(value: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for item in value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for content in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(delta) = content
                        .get("text")
                        .or_else(|| content.get("output_text"))
                        .and_then(Value::as_str)
                    {
                        text.push_str(delta);
                    }
                }
            }
            Some("function_call") => tool_calls.push(json!({
                "id":item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(Value::Null),
                "type":"function",
                "function":{"name":item.get("name").cloned().unwrap_or(Value::Null),"arguments":item.get("arguments").cloned().unwrap_or(Value::String("{}".into()))},
            })),
            _ => {}
        }
    }
    let has_tool_calls = !tool_calls.is_empty();
    let content = if text.is_empty() && has_tool_calls {
        Value::Null
    } else {
        Value::String(text)
    };
    let mut message = json!({"role":"assistant","content":content});
    if has_tool_calls {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("tool_calls".into(), Value::Array(tool_calls));
    }
    json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}", unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":if has_tool_calls { "tool_calls" } else { "stop" }}],
        "usage":provider_stream::responses_usage_to_chat(value.get("usage")),
    })
}

pub(crate) fn cohere_to_chat(value: &Value, model: &str) -> Value {
    let prompt_tokens = value
        .pointer("/meta/billed_units/input_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion_tokens = value
        .pointer("/meta/billed_units/output_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    json!({
        "id":value.get("generation_id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":value.get("text").cloned().unwrap_or(Value::String(String::new()))},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"total_tokens":prompt_tokens + completion_tokens},
    })
}

pub(crate) fn cloudflare_to_chat(value: &Value, model: &str) -> Value {
    let text = value
        .pointer("/result/response")
        .or_else(|| value.get("response"))
        .cloned()
        .unwrap_or(Value::String(String::new()));
    json!({
        "id":format!("chatcmpl-{}",unix_seconds()),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
    })
}

pub(crate) fn chat_to_responses_response(value: &Value, model: &str) -> Value {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({"role":"assistant","content":""}));
    let mut output = Vec::new();
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        output.push(json!({
            "id":format!("rs_{}", unix_seconds()),
            "type":"reasoning",
            "summary":[{"type":"summary_text","text":reasoning}],
        }));
    }
    if !text.is_empty() || message.get("tool_calls").is_none() {
        output.push(json!({
            "id":format!("msg_{}", unix_seconds()),
            "type":"message",
            "status":"completed",
            "role":"assistant",
            "content":[{"type":"output_text","text":text,"annotations":[]}],
        }));
    }
    for tool in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        output.push(json!({
            "id":tool.get("id").cloned().unwrap_or_else(|| Value::String(format!("fc_{}", unix_seconds()))),
            "type":"function_call",
            "status":"completed",
            "call_id":tool.get("id").cloned().unwrap_or_else(|| Value::String(format!("call_{}", unix_seconds()))),
            "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
            "arguments":tool.pointer("/function/arguments").cloned().unwrap_or_else(|| Value::String("{}".into())),
        }));
    }
    let chat_usage = value.get("usage").cloned().unwrap_or_else(|| json!({}));
    let input_tokens = chat_usage
        .get("prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output_tokens = chat_usage
        .get("completion_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let mut response = json!({
        "id":format!("resp_{}", unix_seconds()),
        "object":"response",
        "created_at":unix_seconds(),
        "status":"completed",
        "model":model,
        "output":output,
        "output_text":text,
        "usage":{
            "input_tokens":input_tokens,
            "output_tokens":output_tokens,
            "total_tokens":chat_usage.get("total_tokens").cloned().unwrap_or(json!(input_tokens + output_tokens)),
            "input_tokens_details":chat_usage.get("prompt_tokens_details").cloned().unwrap_or_else(|| json!({})),
            "output_tokens_details":chat_usage.get("completion_tokens_details").cloned().unwrap_or_else(|| json!({})),
        },
        "error":Value::Null,
        "incomplete_details":Value::Null,
    });
    if let Some(receipt) = chat_usage.get("oaix_settlement_receipt") {
        response["usage"]["oaix_settlement_receipt"] = receipt.clone();
    }
    response
}

pub(crate) fn normalize_search_response(url: &str, value: &Value) -> Value {
    let host = Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .unwrap_or_default();
    if value.is_object() && (host.ends_with("tavily.com") || value.get("results").is_some()) {
        let data = value
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let item = item.as_object()?;
                let mut normalized = item.clone();
                let content = item
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let description = if content.chars().count() > 240 {
                    format!("{}...", content.chars().take(237).collect::<String>())
                } else {
                    content.to_owned()
                };
                normalized.insert(
                    "title".into(),
                    item.get("title")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                );
                normalized.insert(
                    "url".into(),
                    item.get("url")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                );
                normalized.insert("description".into(), Value::String(description));
                normalized.insert("content".into(), Value::String(content.to_owned()));
                for name in ["usage", "score", "raw_content"] {
                    normalized.entry(name).or_insert(Value::Null);
                }
                Some(Value::Object(normalized))
            })
            .collect::<Vec<_>>();
        let mut meta = Map::new();
        meta.insert("provider".into(), Value::String("tavily".into()));
        if let Some(root) = value.as_object() {
            for (name, value) in root {
                if name != "results" {
                    meta.insert(name.clone(), value.clone());
                }
            }
        }
        return json!({"code":200,"status":20000,"data":data,"meta":meta});
    }
    if let Some(root) = value.as_object().filter(|root| root.contains_key("data")) {
        let mut output = root.clone();
        output.entry("code").or_insert(json!(200));
        output.entry("status").or_insert(json!(20000));
        let mut meta = output
            .remove("meta")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        meta.entry("provider")
            .or_insert_with(|| Value::String("jina".into()));
        output.insert("meta".into(), Value::Object(meta));
        let data = output
            .remove("data")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|item| {
                let mut item = item.as_object()?.clone();
                for name in ["title", "url", "description", "content"] {
                    item.entry(name)
                        .or_insert_with(|| Value::String(String::new()));
                }
                for name in ["usage", "score", "raw_content"] {
                    item.entry(name).or_insert(Value::Null);
                }
                Some(Value::Object(item))
            })
            .collect();
        output.insert("data".into(), Value::Array(data));
        return Value::Object(output);
    }
    json!({
        "code":200,
        "status":20000,
        "data":[],
        "meta":{"provider":"unknown","raw":value},
    })
}

pub(crate) fn synthetic_chat_stream(value: Value) -> Response<Body> {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({"role":"assistant","content":""}));
    let chunk = json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion.chunk",
        "created":value.get("created").cloned().unwrap_or_else(|| json!(unix_seconds())),
        "model":value.get("model").cloned().unwrap_or(Value::Null),
        "choices":[{"index":0,"delta":message,"finish_reason":"stop"}],
        "usage":value.get("usage").cloned().unwrap_or(Value::Null),
    });
    let wire = format!("data: {}\n\ndata: [DONE]\n\n", chunk);
    let mut response = Response::new(Body::from(wire));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

pub(crate) fn synthetic_responses_stream(value: Value) -> Response<Body> {
    let mut created_response = value.clone();
    if let Some(root) = created_response.as_object_mut() {
        root.insert("status".into(), Value::String("in_progress".into()));
        root.insert("output".into(), Value::Array(Vec::new()));
        root.insert("output_text".into(), Value::String(String::new()));
    }
    let created = json!({"type":"response.created","response":created_response});
    let completed = json!({"type":"response.completed","response":value});
    let wire = format!(
        "event: response.created\ndata: {created}\n\nevent: response.completed\ndata: {completed}\n\n"
    );
    let mut response = Response::new(Body::from(wire));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}
