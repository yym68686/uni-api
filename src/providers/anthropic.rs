use crate::protocols::content::content_text;
use crate::runtime::clock::unix_seconds;
use serde_json::{json, Map, Value};

// Bedrock rejects whitespace-only text blocks, including assistant placeholders
// next to tool calls. Remove only plain placeholders from valid tool turns;
// never alter signed thinking, tool arguments, or an otherwise empty message.
pub(crate) fn strip_blank_tool_turn_text(payload: &mut Value) {
    let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let has_tool = blocks.iter().any(|block| {
            block.get("type").and_then(Value::as_str) == Some("tool_use")
                && ["id", "name"].iter().all(|field| {
                    block
                        .get(field)
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty())
                })
                && block.get("input").is_some_and(Value::is_object)
        });
        if has_tool {
            blocks.retain(|block| {
                !(block.as_object().is_some_and(|fields| fields.len() == 2)
                    && block.get("type").and_then(Value::as_str) == Some("text")
                    && block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.trim().is_empty()))
            });
        }
    }
}

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
        if let Some(stop) = root.get("stop") {
            output.insert(
                "stop_sequences".into(),
                if stop.is_string() {
                    json!([stop])
                } else {
                    stop.clone()
                },
            );
        }
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
        apply_thinking(root, output, original_model);
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

// Capability is deliberately exact: aliases are resolved by the caller. Do not
// infer new model capabilities from prefixes or silently change older models.
fn uses_adaptive_thinking(model: &str) -> bool {
    model == "claude-sonnet-5"
}

fn apply_thinking(root: &Map<String, Value>, output: &mut Map<String, Value>, model: &str) {
    if let Some(config) = root.get("output_config") {
        // Preserve format extensions and even invalid values for upstream validation.
        output.insert("output_config".into(), config.clone());
    }
    let effort = root
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| {
            root.get("reasoning")
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str)
        });
    if let Some(thinking) = root.get("thinking") {
        // Explicit disabled/adaptive/manual/invalid values must not be replaced
        // by an inferred budget merely because reasoning_effort is present.
        output.insert("thinking".into(), thinking.clone());
    } else if uses_adaptive_thinking(model) {
        let selected = output
            .get("output_config")
            .and_then(|v| v.get("effort"))
            .and_then(Value::as_str)
            .or(effort);
        if let Some(effort) = selected {
            output.insert(
                "thinking".into(),
                json!({"type":
                if effort.eq_ignore_ascii_case("none") { "disabled" } else { "adaptive" }}),
            );
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
        }
    }
    let mode = output
        .get("thinking")
        .and_then(|v| v.get("type"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if mode.as_deref() == Some("adaptive") {
        if let Some(effort) = effort.filter(|v| !v.eq_ignore_ascii_case("none")) {
            let config = output.entry("output_config").or_insert_with(|| json!({}));
            if let Some(config) = config.as_object_mut() {
                // Explicit effort (including null/invalid) wins; no guessed xhigh/max mapping.
                config.entry("effort").or_insert_with(|| json!(effort));
            }
        }
    } else if mode.as_deref() == Some("enabled") {
        if let Some(budget) = output
            .get("thinking")
            .and_then(|v| v.get("budget_tokens"))
            .and_then(Value::as_i64)
        {
            let max_tokens = output
                .get("max_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(4096)
                .max(budget.saturating_add(1024));
            output.insert("max_tokens".into(), json!(max_tokens));
        }
    }
    if matches!(mode.as_deref(), Some("enabled" | "adaptive")) {
        output.insert("temperature".into(), json!(1));
        output.remove("top_p");
        output.remove("top_k");
    }
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
    let usage = crate::protocols::provider_stream::claude_usage_to_chat(
        value.get("usage").unwrap_or(&Value::Null),
    );
    json!({
        "id":value.get("id").cloned().unwrap_or_else(|| Value::String(format!("chatcmpl-{}",unix_seconds()))),
        "object":"chat.completion",
        "created":unix_seconds(),
        "model":model,
        "choices":[{"index":0,"message":message,"finish_reason":match value.get("stop_reason").and_then(Value::as_str) { Some("max_tokens") => "length", Some("tool_use") => "tool_calls", Some("refusal") => "content_filter", _ => "stop" }}],
        "usage":usage,
    })
}

#[cfg(test)]
mod blank_tool_text_tests {
    use super::strip_blank_tool_turn_text;
    use serde_json::json;

    #[test]
    fn removes_only_plain_blank_assistant_tool_turn_text() {
        let thinking =
            json!({"type":"thinking","thinking":"signed content","signature":"signature"});
        let tool = json!({"type":"tool_use","id":"tool_1","name":"run","input":{"text":" "}});
        let meaningful = json!({"type":"text","text":" preserve spaces "});
        let cache = json!({"type":"text","text":" ","cache_control":{"type":"ephemeral"}});
        let mut payload = json!({"system":[{"type":"text","text":" "}],"messages":[
            {"role":"assistant","content":[thinking,{"type":"text","text":" \n\t"},
                {"type":"text","text":""},tool,meaningful,cache]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":" "}]}
        ]});
        let before = payload.clone();
        strip_blank_tool_turn_text(&mut payload);
        assert_eq!(
            payload["messages"][0]["content"],
            json!([thinking, tool, meaningful, cache])
        );
        assert_eq!(payload["system"], before["system"]);
        assert_eq!(payload["messages"][1], before["messages"][1]);
        let once = payload.clone();
        strip_blank_tool_turn_text(&mut payload);
        assert_eq!(payload, once);
    }

    #[test]
    fn preserves_invalid_or_non_tool_messages() {
        for message in [
            json!({"role":"assistant","content":[{"type":"text","text":" "}]}),
            json!({"role":"assistant","content":" "}),
            json!({"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":"s"},{"type":"text","text":" "}]}),
            json!({"role":"assistant","content":[{"type":"tool_use","id":"","name":"run","input":{}},{"type":"text","text":" "}]}),
            json!({"role":"user","content":[{"type":"tool_use","id":"tool_1","name":"run","input":{}},{"type":"text","text":" "}]}),
        ] {
            let mut payload = json!({"messages":[message]});
            let before = payload.clone();
            strip_blank_tool_turn_text(&mut payload);
            assert_eq!(payload, before);
        }
    }
}

#[cfg(test)]
mod adaptive_thinking_tests {
    use super::chat_to_claude;
    use serde_json::{json, Value};

    fn convert(fields: Value, model: &str) -> Value {
        let mut body = json!({"messages":[{"role":"user","content":"fixture"}],"max_tokens":256});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        chat_to_claude(&body, model).unwrap()
    }

    #[test]
    fn supported_model_preserves_effort_and_output_limit() {
        for effort in ["low", "medium", "high"] {
            for field in [
                json!({"reasoning_effort":effort}),
                json!({"reasoning":{"effort":effort}}),
            ] {
                let body = convert(field, "claude-sonnet-5");
                assert_eq!(body["thinking"], json!({"type":"adaptive"}));
                assert_eq!(body["output_config"]["effort"], effort);
                assert_eq!(body["max_tokens"], 256);
            }
        }
    }

    #[test]
    fn older_and_unknown_models_keep_manual_budget() {
        for model in [
            "claude-sonnet-4-6",
            "claude-sonnet-5-5",
            "claude-sonnet-5-custom",
            "unknown",
        ] {
            let body = convert(json!({"reasoning_effort":"high"}), model);
            assert_eq!(
                body["thinking"],
                json!({"type":"enabled","budget_tokens":8192})
            );
            assert_eq!(body["max_tokens"], 9216);
            assert!(body.get("output_config").is_none());
        }
    }

    #[test]
    fn explicit_output_config_is_merged_without_overwrite() {
        let config =
            json!({"effort":"low","format":{"type":"json_schema","schema":{"type":"object"}}});
        let body = convert(
            json!({"reasoning_effort":"high","output_config":config}),
            "claude-sonnet-5",
        );
        assert_eq!(body["output_config"], config);
        let body = convert(
            json!({"reasoning_effort":"high","output_config":{"format":{"type":"text"}}}),
            "claude-sonnet-5",
        );
        assert_eq!(
            body["output_config"],
            json!({"format":{"type":"text"},"effort":"high"})
        );
    }

    #[test]
    fn explicit_disable_preserves_sampling_and_prevents_inference() {
        let body = convert(
            json!({"thinking":{"type":"disabled"},"reasoning_effort":"high","temperature":0.4,"top_p":0.8,"top_k":8}),
            "claude-sonnet-5",
        );
        assert_eq!(body["thinking"], json!({"type":"disabled"}));
        assert_eq!(body["temperature"], 0.4);
        assert_eq!(body["top_p"], 0.8);
        assert_eq!(body["top_k"], 8);
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("output_config").is_none());
        let body = convert(json!({"reasoning_effort":"none"}), "claude-sonnet-5");
        assert_eq!(body["thinking"], json!({"type":"disabled"}));
    }

    #[test]
    fn explicit_manual_and_adaptive_intent_is_preserved() {
        let body = convert(
            json!({"thinking":{"type":"enabled","budget_tokens":1024},"reasoning_effort":"high"}),
            "claude-sonnet-5",
        );
        assert_eq!(
            body["thinking"],
            json!({"type":"enabled","budget_tokens":1024})
        );
        assert_eq!(body["max_tokens"], 2048);
        assert!(body.get("output_config").is_none());
        let body = convert(
            json!({"thinking":{"type":"adaptive","display":"omitted"},"reasoning_effort":"high"}),
            "custom-model",
        );
        assert_eq!(
            body["thinking"],
            json!({"type":"adaptive","display":"omitted"})
        );
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["max_tokens"], 256);
    }

    #[test]
    fn invalid_explicit_fields_and_unmapped_efforts_reach_validation() {
        for thinking in [Value::Null, json!("invalid"), json!({"type":"invalid"})] {
            let body = convert(
                json!({"thinking":thinking,"reasoning_effort":"high"}),
                "claude-sonnet-5",
            );
            assert_eq!(body["thinking"], thinking);
            assert!(body.get("output_config").is_none());
        }
        for config in [Value::Null, json!("invalid"), json!({"effort":null})] {
            let body = convert(
                json!({"output_config":config,"reasoning_effort":"high"}),
                "claude-sonnet-5",
            );
            assert_eq!(body["output_config"], config);
        }
        for effort in ["xhigh", "max", "bogus"] {
            let body = convert(json!({"reasoning_effort":effort}), "claude-sonnet-5");
            assert_eq!(body["output_config"]["effort"], effort);
        }
    }

    #[test]
    fn no_thinking_is_invented_without_intent_and_forced_tools_stay_forced() {
        let body = convert(json!({}), "claude-sonnet-5");
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());
        let body = convert(
            json!({"reasoning_effort":"high","tool_choice":{"type":"function","function":{"name":"lookup"}}}),
            "claude-sonnet-5",
        );
        assert_eq!(body["tool_choice"], json!({"type":"tool","name":"lookup"}));
        assert_eq!(body["thinking"]["type"], "adaptive");
    }
}
