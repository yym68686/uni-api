use crate::protocols::content::content_text;
use crate::runtime::clock::unix_seconds;
use serde_json::{json, Value};

pub(crate) fn chat_to_claude(input: &Value, original_model: &str) -> Result<Value, String> {
    let root = input
        .as_object()
        .ok_or_else(|| "chat request body must be an object".to_owned())?;
    let mut messages = Vec::new();
    let mut system = Vec::new();
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
        let content = message.get("content").cloned().unwrap_or(Value::Null);
        if role == "system" {
            if let Some(text) = content_text(Some(&content)) {
                system.push(text);
            }
            continue;
        }
        if role == "tool" {
            messages.push(json!({
                "role":"user",
                "content":[{"type":"tool_result","tool_use_id":message.get("tool_call_id").cloned().unwrap_or(Value::Null),"content":content}],
            }));
            continue;
        }
        let mut blocks = claude_content(&content);
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
            blocks.push(json!({
                "type":"tool_use",
                "id":tool.get("id").cloned().unwrap_or(Value::Null),
                "name":tool.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "input":arguments,
            }));
        }
        messages.push(json!({
            "role": if role == "assistant" { "assistant" } else { "user" },
            "content": blocks,
        }));
    }
    let mut output = json!({
        "model": original_model,
        "messages": messages,
        "max_tokens": root.get("max_tokens").cloned().unwrap_or(json!(4096)),
        "stream": false,
    });
    if !system.is_empty() {
        output
            .as_object_mut()
            .expect("Claude payload object")
            .insert("system".into(), Value::String(system.join("\n")));
    }
    if let Some(tools) = root.get("tools").and_then(Value::as_array) {
        let tools = tools
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                Some(json!({
                    "name":function.get("name").cloned().unwrap_or(Value::Null),
                    "description":function.get("description").cloned().unwrap_or(Value::Null),
                    "input_schema":function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                }))
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            output
                .as_object_mut()
                .expect("Claude payload object")
                .insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(output) = output.as_object_mut() {
        for name in ["temperature", "top_p", "top_k", "stop_sequences"] {
            if let Some(value) = root.get(name) {
                output.insert(name.into(), value.clone());
            }
        }
        if let Some(choice) = root.get("tool_choice") {
            let choice = match choice.as_str() {
                Some("auto") => Some(json!({"type":"auto"})),
                Some("required" | "any") => Some(json!({"type":"any"})),
                Some("none") => None,
                _ if choice.get("type").and_then(Value::as_str) == Some("function") => choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .cloned()
                    .map(|name| json!({"type":"tool","name":name})),
                _ => None,
            };
            if let Some(choice) = choice {
                output.insert("tool_choice".into(), choice);
            } else if root.get("tool_choice").and_then(Value::as_str) == Some("none") {
                output.remove("tools");
            }
        }
        let explicit_thinking = root.get("thinking").filter(|value| value.is_object());
        let effort = root
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .or_else(|| {
                root.get("reasoning")
                    .and_then(|value| value.get("effort"))
                    .and_then(Value::as_str)
            });
        if let Some(thinking) = explicit_thinking {
            output.insert("thinking".into(), thinking.clone());
            output.insert("temperature".into(), json!(1));
            output.remove("top_p");
            output.remove("top_k");
            if let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_i64) {
                let max_tokens = output
                    .get("max_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(4096)
                    .max(budget + 1024);
                output.insert("max_tokens".into(), json!(max_tokens));
            }
        } else if let Some(effort) = effort {
            let budget = match effort.to_ascii_lowercase().as_str() {
                "minimal" | "low" => 1024,
                "medium" => 4096,
                "high" => 8192,
                "extra_high" | "xhigh" => 16384,
                _ => 0,
            };
            if budget > 0 {
                output.insert(
                    "thinking".into(),
                    json!({"type":"enabled","budget_tokens":budget}),
                );
                output.insert("temperature".into(), json!(1));
                output.remove("top_p");
                output.remove("top_k");
                let max_tokens = output
                    .get("max_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(4096)
                    .max(budget + 1024);
                output.insert("max_tokens".into(), json!(max_tokens));
            }
        }
        if let Some(tier) = root.get("service_tier").and_then(Value::as_str) {
            let tier = match tier.to_ascii_lowercase().as_str() {
                "flex" | "standard" | "standard_only" => "standard_only",
                _ => "auto",
            };
            output.insert("service_tier".into(), Value::String(tier.into()));
        }
    }
    Ok(output)
}

pub(crate) fn claude_content(content: &Value) -> Vec<Value> {
    if let Some(text) = content.as_str() {
        return vec![json!({"type":"text","text":text})];
    }
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| match item.get("type").and_then(Value::as_str) {
            Some("text") => Some(json!({"type":"text","text":item.get("text").cloned().unwrap_or(Value::String(String::new()))})),
            Some("image_url") => item
                .pointer("/image_url/url")
                .and_then(Value::as_str)
                .and_then(|value| {
                    let data = value.strip_prefix("data:")?;
                    let (metadata, body) = data.split_once(',')?;
                    Some(json!({
                        "type":"image",
                        "source":{"type":"base64","media_type":metadata.split(';').next().unwrap_or("image/png"),"data":body},
                    }))
                }),
            _ => None,
        })
        .collect()
}

pub(crate) fn claude_to_chat(value: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for item in value
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(item.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => reasoning.push_str(
                item.get("thinking")
                    .or_else(|| item.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ),
            Some("tool_use") => tool_calls.push(json!({
                "id":item.get("id").cloned().unwrap_or(Value::Null),
                "type":"function",
                "function":{
                    "name":item.get("name").cloned().unwrap_or(Value::Null),
                    "arguments":serde_json::to_string(item.get("input").unwrap_or(&json!({}))).unwrap_or_else(|_| "{}".into()),
                }
            })),
            _ => {}
        }
    }
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
    let usage = value.get("usage").cloned().unwrap_or_else(|| json!({}));
    json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":"stop"}],
        "usage":{
            "prompt_tokens":usage.get("input_tokens").cloned().unwrap_or(json!(0)),
            "completion_tokens":usage.get("output_tokens").cloned().unwrap_or(json!(0)),
            "total_tokens":usage.get("input_tokens").and_then(Value::as_i64).unwrap_or(0) + usage.get("output_tokens").and_then(Value::as_i64).unwrap_or(0),
        },
    })
}
