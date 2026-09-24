use crate::runtime::clock::unix_seconds;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use url::Url;

pub(crate) struct ThoughtSignatureCache {
    pub(crate) values: HashMap<String, String>,
    pub(crate) order: VecDeque<String>,
    pub(crate) bytes: usize,
}

pub(crate) static GEMINI_THOUGHT_SIGNATURES: OnceLock<Mutex<ThoughtSignatureCache>> =
    OnceLock::new();

pub(crate) fn chat_to_gemini(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    let mut tool_names = HashMap::new();
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for tool in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(id), Some(name)) = (
                tool.get("id").and_then(Value::as_str),
                tool.pointer("/function/name").and_then(Value::as_str),
            ) {
                tool_names.insert(id.to_owned(), name.to_owned());
            }
        }
    }
    for message in root
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let mut parts = gemini_parts(message.get("content").unwrap_or(&Value::Null))?;
        if role == "system" {
            system_parts.extend(parts);
        } else if role == "tool" {
            let call_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let name = tool_names
                .get(call_id)
                .cloned()
                .unwrap_or_else(|| "tool".into());
            let response = message
                .get("content")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            contents.push(json!({
                "role":"user",
                "parts":[{"functionResponse":{"name":name,"response":{"result":response}}}],
            }));
        } else {
            for tool in message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let arguments = tool
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .and_then(|value| serde_json::from_str::<Value>(value).ok())
                    .unwrap_or_else(|| json!({}));
                let mut part = json!({
                    "functionCall":{
                        "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                        "args":arguments,
                    }
                });
                if let Some(signature) = tool
                    .get("id")
                    .and_then(Value::as_str)
                    .and_then(decode_gemini_thought_signature)
                {
                    part.as_object_mut()
                        .expect("Gemini function call part")
                        .insert("thoughtSignature".into(), Value::String(signature));
                }
                parts.push(part);
            }
            if !parts.is_empty() {
                contents.push(json!({
                    "role": if role == "assistant" { "model" } else { "user" },
                    "parts": parts,
                }));
            }
        }
    }
    if contents.is_empty() {
        contents.push(json!({"role":"user","parts":[{"text":"No messages"}]}));
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

pub(crate) fn gemini_parts(content: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"text":text})]);
    }
    let mut parts = Vec::new();
    for item in content.as_array().into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(
                json!({"text":item.get("text").cloned().unwrap_or(Value::String(String::new()))}),
            ),
            Some("image_url") => {
                if let Some(part) = item
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .and_then(data_url_part)
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

pub(crate) fn data_url_part(value: &str) -> Option<Value> {
    let data = value.strip_prefix("data:")?;
    let (metadata, body) = data.split_once(',')?;
    let mime = metadata
        .split(';')
        .next()
        .unwrap_or("application/octet-stream");
    let mut part = json!({"inlineData":{"mimeType":mime,"data":body}});
    if let Some(signature) = gemini_image_thought_signature(body) {
        part.as_object_mut()
            .expect("Gemini image part")
            .insert("thoughtSignature".into(), Value::String(signature));
    }
    Some(part)
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
            let call_id = gemini_call_id(
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
        "choices":[{"index":0,"message":message,"finish_reason":"stop"}],
        "usage":{
            "prompt_tokens":usage.get("promptTokenCount").cloned().unwrap_or(json!(0)),
            "completion_tokens":usage.get("candidatesTokenCount").cloned().unwrap_or(json!(0)),
            "total_tokens":usage.get("totalTokenCount").cloned().unwrap_or(json!(0)),
        },
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
    Ok(url.to_string())
}
