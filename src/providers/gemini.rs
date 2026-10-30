use crate::runtime::clock::unix_seconds;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};
use url::Url;

pub(crate) struct ThoughtSignatureCache {
    pub(crate) values: HashMap<String, String>,
    pub(crate) order: VecDeque<String>,
    pub(crate) bytes: usize,
}

pub(crate) static GEMINI_THOUGHT_SIGNATURES: OnceLock<Mutex<ThoughtSignatureCache>> =
    OnceLock::new();

pub(crate) fn chat_to_gemini(
    input: &Value,
    original_model: &str,
    allow_continuation: bool,
) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    let mut batch = super::gemini_tools::Batch::default();
    let messages = root
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let append_continuation = allow_continuation && should_append_gemini_continuation(messages);
    for (message_index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let field = format!("messages[{message_index}]");
        if role == "tool" {
            if let Some(content) = batch.result(message, &format!("{field}.tool_call_id"))? {
                contents.push(content);
            }
            continue;
        }
        if batch.pending() {
            return Err(super::gemini_tools::error(
                &field,
                "missing tool results before the next message",
            ));
        }
        let mut parts = gemini_message_parts(message)
            .map_err(|message| format!("gemini_part: {field}.content: {message}"))?;
        if role == "system" {
            system_parts.extend(parts);
        } else {
            if let Some(tools) = message.get("tool_calls").filter(|v| !v.is_null()) {
                let tools = tools.as_array().ok_or_else(|| {
                    super::gemini_tools::error(
                        &format!("{field}.tool_calls"),
                        "tool_calls must be an array",
                    )
                })?;
                if !tools.is_empty() && role != "assistant" {
                    return Err(super::gemini_tools::error(
                        &format!("{field}.tool_calls"),
                        "function calls require an assistant message",
                    ));
                }
                parts.extend(batch.start(tools, &format!("{field}.tool_calls"))?);
            }
            if !parts.is_empty() {
                contents.push(json!({
                    "role": if role == "assistant" { "model" } else { "user" },
                    "parts": parts,
                }));
            }
        }
    }
    if batch.pending() {
        return Err(super::gemini_tools::error(
            "messages",
            "missing tool results at the end of history",
        ));
    }
    if contents.is_empty() {
        contents.push(json!({"role":"user","parts":[{"text":"No messages"}]}));
    }
    // Some Gemini providers reject requests ending with a model turn. Keep the
    // converted assistant history intact, then explicitly ask Gemini to
    // continue only when the final assistant turn is ordinary text and every
    // preceding tool call has a matching result. Native Gemini requests do not
    // pass through this converter and remain unchanged.
    if append_continuation {
        contents.push(json!({"role":"user","parts":[{"text":"Continue."}]}));
    }
    let mut generation = Map::new();
    if let Some(value) = root.get("temperature") {
        generation.insert("temperature".into(), value.clone());
    }
    if let Some(value) = root.get("top_p") {
        generation.insert("topP".into(), value.clone());
    }
    generation.insert(
        "maxOutputTokens".into(),
        root.get("max_tokens").cloned().unwrap_or(json!(8192)),
    );
    let mut output = json!({
        "contents": contents,
        "generationConfig": generation,
        "safetySettings": [
            {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_NONE"},
            {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_NONE"}
        ],
        "_uni_api_model": original_model,
    });
    output
        .as_object_mut()
        .expect("Gemini payload object")
        .remove("_uni_api_model");
    if !system_parts.is_empty() {
        output
            .as_object_mut()
            .expect("Gemini payload object")
            .insert("systemInstruction".into(), json!({"parts":system_parts}));
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        let declarations = tools
            .iter()
            .filter_map(|tool| {
                let mut function = tool.get("function")?.clone();
                if let Some(function) = function.as_object_mut() {
                    function.remove("strict");
                    if let Some(parameters) = function.get_mut("parameters") {
                        sanitize_gemini_schema(parameters);
                    }
                }
                Some(function)
            })
            .collect::<Vec<_>>();
        if !declarations.is_empty() {
            output
                .as_object_mut()
                .expect("Gemini payload object")
                .insert(
                    "tools".into(),
                    json!([{"functionDeclarations":declarations}]),
                );
        }
    }
    apply_gemini_request_controls(&mut output, root, original_model);
    Ok(output)
}

// Responses conversion can omit unsupported items/parts. Never let such loss
// make an unsupported ending or tool history eligible for continuation.
pub(crate) fn responses_allow_gemini_continuation(input: &Value) -> bool {
    let Some(items) = input.get("input").and_then(Value::as_array) else {
        return false;
    };
    if !items
        .iter()
        .all(|item| match item.get("type").and_then(Value::as_str) {
            Some("message") | None => item.get("role").and_then(Value::as_str).is_some(),
            Some("function_call") => {
                ["call_id", "name"].iter().all(|key| {
                    item.get(*key)
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.trim().is_empty())
                }) && item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .is_some_and(|v| v.is_object())
            }
            Some("function_call_output") => {
                item.get("call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
                    && item.get("output").is_some_and(|v| !v.is_null())
            }
            _ => false,
        })
    {
        return false;
    }
    let Some(last) = items.last() else {
        return false;
    };
    if last.get("role").and_then(Value::as_str) != Some("assistant")
        || !last.as_object().is_some_and(|o| {
            o.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "type"
                        | "role"
                        | "content"
                        | "id"
                        | "status"
                        | "internal_chat_message_metadata_passthrough"
                )
            })
        })
    {
        return false;
    }
    plain_text_content(last.get("content"), &["input_text", "output_text", "text"])
}

fn should_append_gemini_continuation(messages: &[Value]) -> bool {
    if !messages.last().is_some_and(is_plain_assistant_text_turn) {
        return false;
    }

    let mut pending = HashSet::new();
    let mut seen = HashSet::new();
    let mut receiving_results = false;
    for message in messages {
        if message.get("function_call").is_some_and(|v| !v.is_null()) {
            return false;
        }
        let calls = match message.get("tool_calls") {
            None | Some(Value::Null) => &[][..],
            Some(Value::Array(calls)) => calls.as_slice(),
            _ => return false,
        };
        match message.get("role").and_then(Value::as_str) {
            Some("assistant") if !calls.is_empty() => {
                // Responses serializes parallel calls as adjacent assistant
                // messages. Permit the batch until its first tool result.
                if !pending.is_empty() && receiving_results {
                    return false;
                }
                receiving_results = false;
                for call in calls {
                    let Some(id) = call.get("id").and_then(Value::as_str) else {
                        return false;
                    };
                    let valid_function = call.get("type").and_then(Value::as_str)
                        == Some("function")
                        && call
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .is_some_and(|name| !name.trim().is_empty())
                        && call
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .and_then(|args| serde_json::from_str::<Value>(args).ok())
                            .is_some_and(|args| args.is_object());
                    if id.trim().is_empty() || !valid_function || !seen.insert(id) {
                        return false;
                    }
                    pending.insert(id);
                }
            }
            Some("tool") if calls.is_empty() => {
                let Some(id) = message.get("tool_call_id").and_then(Value::as_str) else {
                    return false;
                };
                if !pending.remove(id) || message.get("content").is_none_or(Value::is_null) {
                    return false;
                }
                receiving_results = true;
            }
            Some("user" | "system" | "developer" | "assistant")
                if calls.is_empty() && pending.is_empty() => {}
            _ => return false,
        }
    }
    pending.is_empty()
}

fn is_plain_assistant_text_turn(message: &Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    if message.get("role").and_then(Value::as_str) != Some("assistant")
        || !object
            .keys()
            .all(|key| matches!(key.as_str(), "role" | "content" | "name" | "tool_calls"))
        || match message.get("tool_calls") {
            None | Some(Value::Null) => false,
            Some(Value::Array(calls)) => !calls.is_empty(),
            _ => true,
        }
    {
        return false;
    }
    plain_text_content(message.get("content"), &["text"])
}

fn plain_text_content(content: Option<&Value>, types: &[&str]) -> bool {
    match content {
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Array(parts)) => {
            !parts.is_empty()
                && parts.iter().all(|part| {
                    part.as_object().is_some_and(|o| o.len() == 2)
                        && part
                            .get("type")
                            .and_then(Value::as_str)
                            .is_some_and(|t| types.contains(&t))
                        && part.get("text").and_then(Value::as_str).is_some()
                })
                && parts.iter().any(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.trim().is_empty())
                })
        }
        _ => false,
    }
}

pub(crate) fn sanitize_gemini_schema(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("additionalProperties");
            if let Some(default) = object.remove("default") {
                let description = object
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                object.insert(
                    "description".into(),
                    Value::String(format!("{description}\nDefault: {default}")),
                );
            }
            for value in object.values_mut() {
                sanitize_gemini_schema(value);
            }
        }
        Value::Array(items) => {
            for value in items {
                sanitize_gemini_schema(value);
            }
        }
        _ => {}
    }
}

// Empty Chat text alongside a real assistant tool call is a placeholder, not
// a Gemini data Part. Keep standalone/whitespace text, malformed inputs, native
// Gemini payloads and explicit wire overrides under their existing validation.
fn gemini_message_parts(message: &Value) -> Result<Vec<Value>, String> {
    let content = message.get("content").unwrap_or(&Value::Null);
    if content.as_str() != Some("")
        && !content
            .as_array()
            .is_some_and(|items| items.iter().any(empty_chat_text_part))
    {
        return gemini_parts(content);
    }
    let tool_turn = message.get("role").and_then(Value::as_str) == Some("assistant")
        && message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| {
                !calls.is_empty()
                    && calls.iter().all(|call| {
                        call.get("type").and_then(Value::as_str) == Some("function")
                            && call
                                .get("id")
                                .and_then(Value::as_str)
                                .is_some_and(|id| !id.trim().is_empty())
                            && call
                                .pointer("/function/name")
                                .and_then(Value::as_str)
                                .is_some_and(|name| !name.trim().is_empty())
                            && call
                                .pointer("/function/arguments")
                                .and_then(Value::as_str)
                                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                                .is_some_and(|args| args.is_object())
                    })
            });
    if tool_turn {
        if content.as_str() == Some("") {
            return Ok(Vec::new());
        }
        if let Some(items) = content.as_array() {
            let filtered = Value::Array(
                items
                    .iter()
                    .filter(|item| !empty_chat_text_part(item))
                    .cloned()
                    .collect(),
            );
            return gemini_parts(&filtered);
        }
    }
    gemini_parts(content)
}

fn empty_chat_text_part(item: &Value) -> bool {
    item.as_object().is_some_and(|object| object.len() == 2)
        && item.get("type").and_then(Value::as_str) == Some("text")
        && item.get("text").and_then(Value::as_str) == Some("")
}

pub(crate) fn gemini_parts(content: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"text":text})]);
    }
    let mut parts = Vec::new();
    for (index, item) in content.as_array().into_iter().flatten().enumerate() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(
                json!({"text":item.get("text").cloned().unwrap_or(Value::String(String::new()))}),
            ),
            Some("image_url") => {
                let url = item
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        format!("content[{index}].image_url.url must be a non-empty data URL")
                    })?;
                if let Some(part) = validated_data_url_part(url)
                    .map_err(|message| format!("content[{index}].image_url.url: {message}"))?
                {
                    parts.push(part);
                }
            }
            Some("input_audio") => {
                if let Some(part) = input_audio_part(item)? {
                    parts.push(part);
                }
            }
            _ => {}
        }
    }
    Ok(parts)
}

#[cfg(test)]
pub(crate) fn data_url_part(value: &str) -> Option<Value> {
    validated_data_url_part(value).ok().flatten()
}

fn validated_data_url_part(value: &str) -> Result<Option<Value>, String> {
    let Some(data) = value.strip_prefix("data:") else {
        return Err("only image data URLs are supported by this adapter".into());
    };
    let (metadata, body) = data.split_once(',').ok_or("invalid image data URL")?;
    let mime = metadata
        .split(';')
        .next()
        .unwrap_or("application/octet-stream");
    if !mime.starts_with("image/")
        || mime.len() <= 6
        || !metadata.split(';').any(|s| s == "base64")
        || body.is_empty()
        || body.len() > 16 * 1024 * 1024
        || BASE64.decode(body).is_err()
    {
        return Err(
            "image data URL requires an image MIME type and non-empty valid base64 data".into(),
        );
    }
    let mut part = json!({"inlineData":{"mimeType":mime,"data":body}});
    if let Some(signature) = gemini_image_thought_signature(body) {
        part.as_object_mut()
            .expect("Gemini image part")
            .insert("thoughtSignature".into(), Value::String(signature));
    }
    Ok(Some(part))
}

pub(crate) fn input_audio_part(item: &Value) -> Result<Option<Value>, String> {
    let Some(input) = item.get("input_audio") else {
        return Ok(None);
    };
    let Some(data) = input.get("data").and_then(Value::as_str) else {
        return Err("input_audio.data must be a non-empty string".into());
    };
    let format = input
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wav")
        .to_ascii_lowercase();
    if matches!(
        Url::parse(data)
            .ok()
            .map(|url| url.scheme().to_owned())
            .as_deref(),
        Some("http" | "https" | "gs")
    ) {
        return Ok(Some(
            json!({"fileData":{"mimeType":audio_mime_type(&format),"fileUri":data}}),
        ));
    }
    let (mime, encoded) = if let Some(rest) = data.strip_prefix("data:") {
        let (metadata, encoded) = rest
            .split_once(',')
            .ok_or_else(|| "input_audio data URL is invalid".to_owned())?;
        (
            metadata.split(';').next().unwrap_or("audio/wav").to_owned(),
            encoded,
        )
    } else {
        (audio_mime_type(&format).to_owned(), data)
    };
    if encoded.is_empty() || encoded.len() > 8 * 1024 * 1024 {
        return Err("input_audio base64 exceeds the supported size limit".into());
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    BASE64
        .decode(padded.as_bytes())
        .map_err(|_| "input_audio data must be valid base64".to_owned())?;
    Ok(Some(json!({"inlineData":{"mimeType":mime,"data":encoded}})))
}

pub(crate) fn audio_mime_type(format: &str) -> &'static str {
    match format {
        "mp3" | "mpeg" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        "opus" => "audio/opus",
        "webm" => "audio/webm",
        _ => "audio/wav",
    }
}

pub(crate) fn apply_gemini_request_controls(
    output: &mut Value,
    root: &Map<String, Value>,
    model: &str,
) {
    let output = output.as_object_mut().expect("Gemini payload object");
    if let Some(choice) = root.get("tool_choice") {
        let mut config = Map::new();
        match choice.as_str() {
            Some("none") => {
                config.insert("mode".into(), Value::String("NONE".into()));
            }
            Some("required" | "any") => {
                config.insert("mode".into(), Value::String("ANY".into()));
            }
            Some("auto") => {
                config.insert("mode".into(), Value::String("AUTO".into()));
            }
            _ if choice.get("type").and_then(Value::as_str) == Some("function") => {
                config.insert("mode".into(), Value::String("ANY".into()));
                if let Some(name) = choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .and_then(Value::as_str)
                {
                    config.insert(
                        "allowedFunctionNames".into(),
                        Value::Array(vec![Value::String(name.to_owned())]),
                    );
                }
            }
            _ => {}
        }
        if !config.is_empty() {
            output.insert("toolConfig".into(), json!({"functionCallingConfig":config}));
        }
    }
    if let Some(tier) = root
        .get("service_tier")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
    {
        let tier = match tier.as_str() {
            "default" | "standard" => "STANDARD",
            "priority" => "PRIORITY",
            "flex" => "FLEX",
            value => value,
        };
        output.insert(
            "serviceTier".into(),
            Value::String(tier.to_ascii_uppercase()),
        );
    }
    if let Some(format) = root.get("response_format") {
        let generation = output
            .entry("generationConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("Gemini generation config");
        match format.get("type").and_then(Value::as_str) {
            Some("json_object") => {
                generation.insert("responseMimeType".into(), json!("application/json"));
            }
            Some("json_schema") => {
                if let Some(schema) = format.pointer("/json_schema/schema") {
                    generation.insert("responseMimeType".into(), json!("application/json"));
                    generation.insert("responseJsonSchema".into(), schema.clone());
                }
            }
            Some("text") | None => {}
            _ => {}
        }
    }
    let effort = root
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| {
            root.get("reasoning")
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str)
        })
        .map(|value| value.to_ascii_lowercase().replace('-', "_"));
    if let Some(effort) = effort {
        let generation = output
            .entry("generationConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("Gemini generation config");
        if model.to_ascii_lowercase().contains("gemini-3") {
            let level = match effort.as_str() {
                "minimal" | "low" => "low",
                "medium" => "medium",
                "high" | "extra_high" | "xhigh" => "high",
                _ => "minimal",
            };
            generation.insert("thinkingConfig".into(), json!({"thinkingLevel":level}));
        } else if model.to_ascii_lowercase().contains("gemini-2.5") {
            let maximum = if model.to_ascii_lowercase().contains("pro") {
                32768
            } else {
                24576
            };
            let budget = match effort.as_str() {
                "none" => 0,
                "minimal" | "low" => maximum / 4,
                "medium" => maximum / 2,
                "high" => maximum * 3 / 4,
                "extra_high" | "xhigh" => maximum,
                _ => 0,
            };
            generation.insert(
                "thinkingConfig".into(),
                json!({"includeThoughts":budget > 0,"thinkingBudget":budget}),
            );
        }
    }
    let wants_audio = root
        .get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.as_str()
                    .is_some_and(|value| value.eq_ignore_ascii_case("audio"))
            })
        })
        || root.get("audio").is_some();
    if wants_audio {
        let voice = root
            .get("audio")
            .and_then(|value| value.get("voice"))
            .and_then(Value::as_str)
            .unwrap_or("Kore");
        let generation = output
            .entry("generationConfig")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("Gemini generation config");
        generation.insert("responseModalities".into(), json!(["AUDIO"]));
        generation.insert(
            "speechConfig".into(),
            json!({"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}}),
        );
    }
}

pub(crate) fn decode_gemini_thought_signature(call_id: &str) -> Option<String> {
    let encoded = call_id.strip_prefix("call_")?.split('.').next()?;
    if encoded.is_empty() || encoded.len() > 90_000 {
        return None;
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(padded.as_bytes()))
        .ok()?;
    (decoded.len() <= 64 * 1024)
        .then(|| String::from_utf8(decoded).ok())
        .flatten()
}

pub(crate) fn gemini_image_key(encoded: &str) -> Option<String> {
    if encoded.is_empty() || encoded.len() > 16 * 1024 * 1024 {
        return None;
    }
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = BASE64.decode(padded.as_bytes()).ok()?;
    Some(format!("{:x}", Sha256::digest(decoded)))
}

pub(crate) fn gemini_image_thought_signature(encoded: &str) -> Option<String> {
    let key = gemini_image_key(encoded)?;
    let cache = GEMINI_THOUGHT_SIGNATURES.get_or_init(|| {
        Mutex::new(ThoughtSignatureCache {
            values: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        })
    });
    cache.lock().ok()?.values.get(&key).cloned()
}

pub(crate) fn cache_gemini_image_thought_signature(encoded: &str, signature: &str) {
    if signature.is_empty() || signature.len() > 64 * 1024 {
        return;
    }
    let Some(key) = gemini_image_key(encoded) else {
        return;
    };
    let cache = GEMINI_THOUGHT_SIGNATURES.get_or_init(|| {
        Mutex::new(ThoughtSignatureCache {
            values: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        })
    });
    let Ok(mut cache) = cache.lock() else {
        return;
    };
    if let Some(previous) = cache.values.remove(&key) {
        cache.bytes = cache.bytes.saturating_sub(previous.len());
        cache.order.retain(|item| item != &key);
    }
    cache.bytes = cache.bytes.saturating_add(signature.len());
    cache.order.push_back(key.clone());
    cache.values.insert(key, signature.to_owned());
    while cache.values.len() > 100 || cache.bytes > 4 * 1024 * 1024 {
        let Some(oldest) = cache.order.pop_front() else {
            break;
        };
        if let Some(value) = cache.values.remove(&oldest) {
            cache.bytes = cache.bytes.saturating_sub(value.len());
        }
    }
}

pub(crate) fn gemini_call_id(signature: Option<&str>, fallback_index: usize) -> String {
    signature
        .filter(|signature| !signature.is_empty() && signature.len() <= 64 * 1024)
        .map(|signature| {
            format!(
                "call_{}.{}",
                URL_SAFE_NO_PAD.encode(signature.as_bytes()),
                fallback_index
            )
        })
        .unwrap_or_else(|| format!("call_{fallback_index}"))
}

pub(crate) fn gemini_to_chat(value: &Value, model: &str) -> Value {
    let candidate = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|items| items.first());
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for part in candidate
        .and_then(|candidate| candidate.pointer("/content/parts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(data), Some(signature)) = (
            part.pointer("/inlineData/data")
                .or_else(|| part.pointer("/inline_data/data"))
                .and_then(Value::as_str),
            part.get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(Value::as_str),
        ) {
            cache_gemini_image_thought_signature(data, signature);
        }
        if let Some(value) = part.get("text").and_then(Value::as_str) {
            if part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                reasoning.push_str(value);
            } else {
                text.push_str(value);
            }
        }
        if let Some(call) = part.get("functionCall") {
            let call_id = super::gemini_tools::call_id(
                call.get("id").and_then(Value::as_str),
                part.get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(Value::as_str),
                tool_calls.len() + 1,
            );
            tool_calls.push(json!({
                "id":call_id,
                "type":"function",
                "function":{
                    "name":call.get("name").cloned().unwrap_or(Value::Null),
                    "arguments":serde_json::to_string(call.get("args").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into()),
                }
            }));
        }
    }
    let usage = value
        .get("usageMetadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut message = json!({"role":"assistant","content":text});
    if !reasoning.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message
            .as_object_mut()
            .expect("chat message")
            .insert("tool_calls".into(), Value::Array(tool_calls));
    }
    json!({
        "id":format!("chatcmpl-{}",unix_seconds()),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":crate::protocols::gemini::chat_finish_reason(value.pointer("/candidates/0/finishReason").and_then(Value::as_str).unwrap_or("STOP"))}],
        "usage":crate::protocols::provider_stream::gemini_usage_to_chat(&usage),
    })
}

pub(crate) fn gemini_url(
    base: &str,
    model: &str,
    key: &str,
    stream: bool,
) -> Result<String, String> {
    let mut url = Url::parse(base).map_err(|error| format!("invalid Gemini base URL: {error}"))?;
    let path = url
        .path()
        .split("/models/")
        .next()
        .unwrap_or(url.path())
        .trim_end_matches('/');
    url.set_path(&format!(
        "{path}/models/{model}:{}",
        if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        }
    ));
    url.query_pairs_mut().clear().append_pair("key", key);
    if stream {
        url.query_pairs_mut().append_pair("alt", "sse");
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod gemini_model_turn_tests {
    use super::{
        chat_to_gemini, responses_allow_gemini_continuation, should_append_gemini_continuation,
    };
    use serde_json::{json, Value};

    fn assistant_text(text: &str) -> Value {
        json!({"role":"assistant","content":text})
    }

    #[test]
    fn omits_only_empty_tool_text_and_preserves_signed_function_and_parameters() {
        let id = super::gemini_call_id(Some("opaque-signature"), 0);
        for content in [json!(""), json!([{"type":"text","text":""}])] {
            let input = json!({"messages":[
                {"role":"user","content":"start"},
                {"role":"assistant","content":content,"tool_calls":[{"id":id,"type":"function","function":{"name":"lookup","arguments":"{\"text\":\"\"}"}}]},
                {"role":"tool","tool_call_id":id,"content":""}],
                "temperature":0.2,"top_p":0.8,"max_tokens":77});
            let original = input.clone();
            let wire = chat_to_gemini(&input, "gemini", true).unwrap();
            assert_eq!(
                wire["contents"][1]["parts"],
                json!([{"functionCall":{"name":"lookup","args":{"text":""}},"thoughtSignature":"opaque-signature"}])
            );
            assert_eq!(
                wire["contents"][2]["parts"][0]["functionResponse"]["response"]["result"],
                ""
            );
            assert_eq!(
                wire["generationConfig"],
                json!({"temperature":0.2,"topP":0.8,"maxOutputTokens":77})
            );
            assert_eq!(input, original);
        }
    }

    #[test]
    fn does_not_remove_standalone_whitespace_or_malformed_text() {
        let call =
            json!({"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}});
        for content in [
            json!(" \n"),
            json!("visible"),
            json!([{"type":"text","text":null}]),
            json!([{"type":"text"}]),
            json!([{"type":"text","text":"","extra":"metadata"}]),
        ] {
            let message = json!({"role":"assistant","content":content,"tool_calls":[call]});
            assert_eq!(
                super::gemini_message_parts(&message).unwrap(),
                super::gemini_parts(&content).unwrap()
            );
        }
        for message in [
            json!({"role":"assistant","content":""}),
            json!({"role":"user","content":"","tool_calls":[call]}),
            json!({"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"bad json"}}]}),
        ] {
            assert_eq!(
                super::gemini_message_parts(&message).unwrap(),
                json!([{"text":""}]).as_array().unwrap().clone()
            );
        }
        let message = json!({"role":"assistant","tool_calls":[call],"content":[
            {"type":"text","text":""},{"type":"text","text":"visible"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]});
        assert_eq!(
            super::gemini_message_parts(&message).unwrap(),
            vec![
                json!({"text":"visible"}),
                json!({"inlineData":{"mimeType":"image/png","data":"AAAA"}})
            ]
        );
    }

    #[test]
    fn appends_user_turn_after_resolved_tool_history_and_preserves_every_turn() {
        let messages = vec![
            json!({"role":"user","content":"run the tool"}),
            json!({"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"call-1","content":"tool result"}),
            assistant_text("The result is ready."),
        ];
        assert!(should_append_gemini_continuation(&messages));
        let input = json!({"model":"gemini-test","messages":messages,"max_tokens":77,"tools":[{"function":{"name":"lookup","parameters":{"type":"object"}}}]});
        let output = chat_to_gemini(&input, "gemini-test", true).unwrap();
        let contents = output["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 5);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["functionCall"]["name"], "lookup");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["name"],
            "lookup"
        );
        assert_eq!(contents[3]["role"], "model");
        assert_eq!(contents[3]["parts"][0]["text"], "The result is ready.");
        assert_eq!(contents[4]["role"], "user");
        assert_eq!(contents[4]["parts"][0]["text"], "Continue.");
        assert_eq!(output["generationConfig"]["maxOutputTokens"], 77);
        assert_eq!(
            output["tools"][0]["functionDeclarations"][0]["name"],
            "lookup"
        );
    }

    #[test]
    fn supports_parallel_responses_calls_and_keeps_signatures() {
        let signature = "test-signature";
        let id = super::gemini_call_id(Some(signature), 0);
        let input = json!({"input":[
            {"type":"message","role":"user","content":"start"},
            {"type":"function_call","call_id":id,"name":"one","arguments":"{}"},
            {"type":"function_call","call_id":"b","name":"two","arguments":"{}"},
            {"type":"function_call_output","call_id":"b","output":"result b"},
            {"type":"function_call_output","call_id":id,"output":"result a"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"ready"}]}
        ]});
        assert!(responses_allow_gemini_continuation(&input));
        let chat =
            crate::protocols::conversion::responses_to_gemini_chat_request(&input, "gemini-test")
                .unwrap();
        let wire = chat_to_gemini(&chat, "gemini-test", true).unwrap();
        let turns = wire["contents"].as_array().unwrap();
        assert_eq!(turns.len(), 5);
        assert_eq!(turns[1]["parts"][0]["thoughtSignature"], signature);
        assert_eq!(turns[1]["parts"].as_array().unwrap().len(), 2);
        assert_eq!(turns[2]["parts"].as_array().unwrap().len(), 2);
        assert_eq!(turns[3]["parts"][0]["text"], "ready");
        assert_eq!(
            turns[4],
            json!({"role":"user","parts":[{"text":"Continue."}]})
        );
    }

    #[test]
    fn skips_unsupported_or_lossy_responses_endings_and_invalid_tools() {
        let plain = json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"ready"}]});
        for unsupported in [
            json!({"type":"custom_tool_call","call_id":"a"}),
            json!({"type":"reasoning","encrypted_content":"signature"}),
        ] {
            assert!(!responses_allow_gemini_continuation(
                &json!({"input":[unsupported,plain.clone()]})
            ));
        }
        let mut mixed = plain.clone();
        mixed["content"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"refusal","refusal":"no"}));
        assert!(!responses_allow_gemini_continuation(
            &json!({"input":[mixed]})
        ));
        for args in [Value::Null, json!("not json"), json!("[]")] {
            assert!(!responses_allow_gemini_continuation(&json!({"input":[
                {"type":"function_call","call_id":"a","name":"lookup","arguments":args},
                {"type":"function_call_output","call_id":"a","output":"ok"},plain.clone()]})));
        }
        let mut signed = plain.clone();
        signed["content"][0]["signature"] = json!("signature");
        assert!(!responses_allow_gemini_continuation(
            &json!({"input":[signed]})
        ));
    }

    #[test]
    fn leaves_nonplain_chat_endings_and_bad_tool_fields_unchanged() {
        for extra in [
            json!({"tool_calls":{}}),
            json!({"function_call":{"name":"legacy"}}),
            json!({"reasoning_content":"thought"}),
        ] {
            let mut message = assistant_text("ready");
            message
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert!(!should_append_gemini_continuation(&[message]));
        }
        let input = json!({"messages":[assistant_text("ready")]});
        let unchanged = chat_to_gemini(&input, "gemini-test", false).unwrap();
        assert_eq!(unchanged["contents"].as_array().unwrap().len(), 1);
        assert_eq!(unchanged["contents"][0]["role"], "model");
    }

    #[test]
    fn does_not_append_after_user_turn_or_without_assistant_text() {
        assert!(!should_append_gemini_continuation(&[
            json!({"role":"assistant","content":"answer"}),
            json!({"role":"user","content":"next"}),
        ]));
        assert!(!should_append_gemini_continuation(&[assistant_text(
            "  \n"
        )]));
        assert!(!should_append_gemini_continuation(&[json!({
            "role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]
        })]));
        assert!(!should_append_gemini_continuation(&[json!({
            "role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]
        })]));
    }

    #[test]
    fn does_not_append_when_tool_history_is_unresolved_or_malformed() {
        let call = json!({"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]});
        let result = json!({"role":"tool","tool_call_id":"call-1","content":"result"});
        assert!(!should_append_gemini_continuation(&[
            call.clone(),
            assistant_text("done")
        ]));
        assert!(!should_append_gemini_continuation(&[
            result,
            assistant_text("done")
        ]));
        assert!(!should_append_gemini_continuation(&[
            call.clone(),
            json!({"role":"tool","tool_call_id":"call-1","content":"first"}),
            json!({"role":"tool","tool_call_id":"call-1","content":"duplicate"}),
            assistant_text("done"),
        ]));
        assert!(!should_append_gemini_continuation(&[
            call,
            json!({"role":"user","content":"interleaved before result"}),
            json!({"role":"tool","tool_call_id":"call-1","content":"late result"}),
            assistant_text("done"),
        ]));
    }
}
