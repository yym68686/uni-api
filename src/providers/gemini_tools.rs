//! Request-local Gemini tool identity and batch association. No signature cache,
//! signature synthesis, cross-request state, or name-only result matching.
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::{json, Value};
use std::collections::HashMap;

const ID_PREFIX: &str = "call_g2_";
pub(crate) const MAX_SIGNATURE_BYTES: usize = 1024 * 1024;
const MAX_NATIVE_ID_BYTES: usize = 64 * 1024;
// JSON can escape each input byte as six bytes. Bound decoding before allocating,
// while accepting every envelope the encoder can produce from valid inputs.
const MAX_ENVELOPE_BYTES: usize = 6 * (MAX_SIGNATURE_BYTES + MAX_NATIVE_ID_BYTES) + 16;
const MAX_ENCODED_ENVELOPE_BYTES: usize = ((MAX_ENVELOPE_BYTES + 2) / 3) * 4;
pub(crate) const MAX_LEGACY_SIGNATURE_ENCODED_BYTES: usize = ((MAX_SIGNATURE_BYTES + 2) / 3) * 4;

fn validate_identity(native: Option<&str>, signature: Option<&str>) -> Result<(), String> {
    if native.is_some_and(|id| id.is_empty() || id.len() > MAX_NATIVE_ID_BYTES) {
        return Err("Gemini native tool id exceeds its bounded capacity".into());
    }
    if signature.is_some_and(|sig| sig.is_empty() || sig.len() > MAX_SIGNATURE_BYTES) {
        return Err("Gemini thought signature exceeds its bounded capacity".into());
    }
    Ok(())
}

pub(crate) fn call_id(
    native: Option<&str>,
    signature: Option<&str>,
    ordinal: usize,
) -> Result<String, String> {
    let native = native.filter(|s| !s.is_empty());
    let signature = signature.filter(|s| !s.is_empty());
    validate_identity(native, signature)?;
    match (native, signature) {
        (Some(native), signature) => Ok(format!(
            "{ID_PREFIX}{}.{ordinal}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&(native, signature)).unwrap())
        )),
        (None, signature) => super::gemini::gemini_call_id(signature, ordinal),
    }
}

fn identity(id: &str) -> Result<(Option<String>, Option<String>), String> {
    if let Some(encoded) = id.strip_prefix(ID_PREFIX) {
        let (encoded, ordinal) = encoded
            .rsplit_once('.')
            .ok_or("invalid Gemini tool identity")?;
        if encoded.len() > MAX_ENCODED_ENVELOPE_BYTES || ordinal.parse::<usize>().is_err() {
            return Err("invalid Gemini tool identity".into());
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| "invalid Gemini tool identity")?;
        if bytes.len() > MAX_ENVELOPE_BYTES {
            return Err("invalid Gemini tool identity".into());
        }
        let (native, signature): (String, Option<String>) =
            serde_json::from_slice(&bytes).map_err(|_| "invalid Gemini tool identity")?;
        validate_identity(Some(&native), signature.as_deref())?;
        return Ok((Some(native), signature));
    }
    // Legacy IDs always contain the numeric ordinal. A normal call_1 is not a signature.
    if id
        .rsplit_once('.')
        .is_some_and(|(_, n)| n.parse::<usize>().is_ok())
    {
        if let Some(signature) = super::gemini::decode_gemini_thought_signature(id) {
            return Ok((None, Some(signature)));
        }
    }
    validate_identity(Some(id), None)?;
    Ok((Some(id.to_owned()), None))
}

pub(crate) fn error(field: &str, message: &str) -> String {
    // Safe, machine-readable diagnostics: paths and fixed messages only, never
    // arguments, results, signatures or caller-supplied IDs.
    format!("gemini_tool_history: {field}: {message}")
}

pub(crate) fn call_part(tool: &Value, field: &str) -> Result<Value, String> {
    let id = tool
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| error(field, "function call requires a non-empty id"))?;
    let name = tool
        .pointer("/function/name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| error(field, "function call requires a non-empty name"))?;
    let args = tool
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .filter(Value::is_object)
        .ok_or_else(|| error(field, "function arguments must be a JSON object string"))?;
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err(error(field, "only function tool calls can be converted"));
    }
    let (native, mut signature) = identity(id).map_err(|message| error(field, &message))?;
    for value in [
        tool.get("thoughtSignature"),
        tool.get("thought_signature"),
        tool.pointer("/extra_content/google/thought_signature"),
    ]
    .into_iter()
    .flatten()
    {
        let actual = value
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= MAX_SIGNATURE_BYTES)
            .ok_or_else(|| {
                error(
                    field,
                    "thought signature must be a non-empty bounded string",
                )
            })?;
        if signature.as_deref().is_some_and(|old| old != actual) {
            return Err(error(field, "conflicting thought signature metadata"));
        }
        signature = Some(actual.to_owned());
    }
    let mut part = json!({"functionCall":{"name":name,"args":args}});
    if let Some(native) = native {
        part["functionCall"]["id"] = json!(native);
    }
    if let Some(signature) = signature {
        part["thoughtSignature"] = json!(signature);
    }
    Ok(part)
}

#[derive(Default)]
pub(crate) struct Batch {
    calls: Vec<(Value, Option<Value>)>,
    indexes: HashMap<String, usize>,
    remaining: usize,
}

impl Batch {
    pub(crate) fn pending(&self) -> bool {
        !self.calls.is_empty()
    }

    pub(crate) fn start(&mut self, tools: &[Value], field: &str) -> Result<Vec<Value>, String> {
        let mut parts = Vec::new();
        for (i, tool) in tools.iter().enumerate() {
            let part = call_part(tool, &format!("{field}[{i}]"))?;
            let id = tool["id"].as_str().unwrap();
            if self
                .indexes
                .insert(id.to_owned(), self.calls.len())
                .is_some()
            {
                return Err(error(field, "duplicate function call id within one batch"));
            }
            self.calls.push((part["functionCall"].clone(), None));
            self.remaining += 1;
            parts.push(part);
        }
        Ok(parts)
    }

    pub(crate) fn result(&mut self, message: &Value, field: &str) -> Result<Option<Value>, String> {
        let index = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .and_then(|id| self.indexes.get(id))
            .copied()
            .ok_or_else(|| {
                error(
                    field,
                    "orphan tool result: no matching call in the pending batch",
                )
            })?;
        let (call, slot) = &mut self.calls[index];
        if slot.is_some() {
            return Err(error(field, "duplicate tool result"));
        }
        let mut response = json!({"functionResponse":{"name":call["name"],"response":{"result":message.get("content").cloned().unwrap_or(json!(""))}}});
        if let Some(id) = call.get("id") {
            response["functionResponse"]["id"] = id.clone();
        }
        *slot = Some(response);
        self.remaining -= 1;
        if self.remaining != 0 {
            return Ok(None);
        }
        let parts = std::mem::take(&mut self.calls)
            .into_iter()
            .map(|(_, slot)| slot.unwrap())
            .collect::<Vec<_>>();
        self.indexes.clear();
        Ok(Some(json!({"role":"user","parts":parts})))
    }
}

// Gemini normally sends a complete functionCall Part. Buffer tool Parts until
// the terminal frame so late metadata for the SAME native ID cannot change an
// already emitted OpenAI call ID. Text continues to stream immediately.
#[derive(Default)]
pub(crate) struct StreamTools {
    parts: Vec<Value>,
    ids: HashMap<String, usize>,
    bytes: usize,
}

impl StreamTools {
    pub(crate) fn push(&mut self, part: &Value) -> Result<(), String> {
        let size = serde_json::to_vec(part).unwrap_or_default().len();
        self.bytes = self.bytes.saturating_add(size);
        if self.bytes > 1024 * 1024 || self.parts.len() >= 1024 {
            return Err("Gemini tool stream exceeded its bounded buffer".into());
        }
        if let Some(id) = part
            .pointer("/functionCall/id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            if let Some(index) = self.ids.get(id).copied() {
                merge(&mut self.parts[index], part)?;
                return Ok(());
            }
            self.ids.insert(id.to_owned(), self.parts.len());
        }
        self.parts.push(part.clone());
        Ok(())
    }

    pub(crate) fn take(&mut self) -> Vec<Value> {
        self.ids.clear();
        self.bytes = 0;
        std::mem::take(&mut self.parts)
    }
}

fn merge(target: &mut Value, source: &Value) -> Result<(), String> {
    if let (Some(target), Some(source)) = (target.as_object_mut(), source.as_object()) {
        for (key, value) in source {
            if let Some(old) = target.get_mut(key) {
                merge(old, value)?;
            } else {
                target.insert(key.clone(), value.clone());
            }
        }
        Ok(())
    } else if target == source {
        Ok(())
    } else {
        Err("Gemini tool stream contains conflicting metadata or arguments".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(id: &str) -> Value {
        json!({"id":id,"type":"function","function":{"name":"read","arguments":"{\"path\":\"a.md\"}"}})
    }

    #[test]
    fn preserves_native_id_and_signature_without_copying_between_calls() {
        for (native, sig) in [
            (Some("native-1"), Some("opaque")),
            (None, Some("opaque")),
            (Some("native-1"), None),
            (None, None),
        ] {
            let id = call_id(native, sig, 1).unwrap();
            let part = call_part(&tool(&id), "tool_calls[0]").unwrap();
            assert_eq!(part.get("thoughtSignature").and_then(Value::as_str), sig);
            assert_eq!(
                part.pointer("/functionCall/id").and_then(Value::as_str),
                native.or(if sig.is_none() { Some("call_1") } else { None })
            );
        }
    }

    #[test]
    fn accepts_explicit_signature_extensions_and_rejects_conflicts() {
        for (field, value) in [
            ("thoughtSignature", json!("opaque")),
            ("thought_signature", json!("opaque")),
            (
                "extra_content",
                json!({"google":{"thought_signature":"opaque"}}),
            ),
        ] {
            let mut call = tool("native-1");
            call[field] = value;
            assert_eq!(
                call_part(&call, "tool_calls[0]").unwrap()["thoughtSignature"],
                "opaque"
            );
        }
        let mut call = tool(&call_id(Some("native"), Some("opaque"), 1).unwrap());
        call["thoughtSignature"] = json!("different");
        assert!(call_part(&call, "tool_calls[0]")
            .unwrap_err()
            .contains("conflicting"));
        for bad in [json!(""), json!(null), json!(10)] {
            call["thoughtSignature"] = bad;
            assert!(call_part(&call, "tool_calls[0]").is_err());
        }
    }

    #[test]
    fn malformed_identity_and_invalid_calls_fail_without_echoing_secrets() {
        for id in [
            "call_g2_bad.1",
            "call_g2_W10.1",
            "call_g2_bad",
            "call_g2_WyJhIiwiYiJd.not-an-ordinal",
        ] {
            let message = call_part(&tool(id), "messages[1].tool_calls[0]").unwrap_err();
            assert!(!message.contains(id));
        }
        for args in [
            json!(null),
            json!("not-json-secret"),
            json!("[]"),
            json!("10"),
        ] {
            let mut call = tool("id");
            call["function"]["arguments"] = args;
            let message = call_part(&call, "tool_calls[0]").unwrap_err();
            assert!(!message.contains("not-json-secret"));
        }
    }

    #[test]
    fn long_signatures_round_trip_in_every_identity_form() {
        for size in [65536, 65537, 76928, MAX_SIGNATURE_BYTES] {
            let signature = "s".repeat(size);
            for native in [None, Some("native-1")] {
                let id = call_id(native, Some(&signature), 1).unwrap();
                let part = call_part(&tool(&id), "tool_calls[0]").unwrap();
                assert_eq!(part["thoughtSignature"], signature);
            }
            for name in ["thoughtSignature", "thought_signature"] {
                let mut call = tool("native-1");
                call[name] = json!(signature);
                assert_eq!(
                    call_part(&call, "tool_calls[0]").unwrap()["thoughtSignature"],
                    signature
                );
            }
        }
        // Exercise maximum JSON escaping, not just ASCII signatures.
        let signature = "\u{0}".repeat(MAX_SIGNATURE_BYTES);
        let native = "\u{0}".repeat(MAX_NATIVE_ID_BYTES);
        let id = call_id(Some(&native), Some(&signature), 1).unwrap();
        let part = call_part(&tool(&id), "tool_calls[0]").unwrap();
        assert_eq!(part["thoughtSignature"], signature);
        assert_eq!(part["functionCall"]["id"], native);
    }

    #[test]
    fn oversized_signatures_are_rejected_without_silent_loss() {
        let signature = "s".repeat(MAX_SIGNATURE_BYTES + 1);
        for native in [None, Some("native-1")] {
            assert!(call_id(native, Some(&signature), 1).is_err());
        }
        let encoded =
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&("native-1", &signature)).unwrap());
        let call = tool(&format!("{ID_PREFIX}{encoded}.1"));
        assert!(call_part(&call, "tool_calls[0]").is_err());
        let mut call = tool("native-1");
        call["thoughtSignature"] = json!(signature);
        assert!(call_part(&call, "tool_calls[0]").is_err());
        assert!(call_id(Some(&"n".repeat(MAX_NATIVE_ID_BYTES + 1)), None, 1).is_err());
    }

    #[test]
    fn same_name_parallel_results_are_reordered_by_id_and_batches_do_not_leak() {
        let mut batch = Batch::default();
        for _ in 0..2 {
            let calls = [tool("a"), tool("b"), tool("c")];
            assert_eq!(batch.start(&calls, "tool_calls").unwrap().len(), 3);
            for id in ["c", "b"] {
                assert!(batch
                    .result(&json!({"tool_call_id":id,"content":id}), "tool_call_id")
                    .unwrap()
                    .is_none());
            }
            let wire = batch
                .result(&json!({"tool_call_id":"a","content":"a"}), "tool_call_id")
                .unwrap()
                .unwrap();
            assert_eq!(
                wire["parts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["functionResponse"]["response"]["result"]
                        .as_str()
                        .unwrap())
                    .collect::<Vec<_>>(),
                vec!["a", "b", "c"]
            );
            assert!(!batch.pending());
        }
        assert!(batch
            .result(&json!({"tool_call_id":"a"}), "tool_call_id")
            .is_err());
    }

    #[test]
    fn duplicate_calls_and_duplicate_or_orphan_results_are_rejected() {
        let mut batch = Batch::default();
        assert!(batch.start(&[tool("a"), tool("a")], "tool_calls").is_err());
        let mut batch = Batch::default();
        batch.start(&[tool("a"), tool("b")], "tool_calls").unwrap();
        assert!(batch
            .result(&json!({"tool_call_id":"orphan-secret"}), "tool_call_id")
            .is_err());
        assert!(batch
            .result(&json!({"tool_call_id":"a"}), "tool_call_id")
            .unwrap()
            .is_none());
        assert!(batch
            .result(&json!({"tool_call_id":"a"}), "tool_call_id")
            .unwrap_err()
            .contains("duplicate"));
        assert!(batch.pending());
    }

    #[test]
    fn stream_merges_only_identical_native_ids_and_refuses_conflicts() {
        let mut tools = StreamTools::default();
        tools
            .push(&json!({"functionCall":{"id":"a","name":"read","args":{}}}))
            .unwrap();
        tools
            .push(&json!({"functionCall":{"id":"b","name":"read","args":{"n":2}}}))
            .unwrap();
        tools
            .push(&json!({"functionCall":{"id":"a","args":{"n":1}},"thoughtSignature":"opaque"}))
            .unwrap();
        let parts = tools.take();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionCall"]["args"], json!({"n":1}));
        assert_eq!(parts[0]["thoughtSignature"], "opaque");
        assert!(parts[1].get("thoughtSignature").is_none());
        tools.push(&parts[0]).unwrap();
        assert!(tools
            .push(&json!({"functionCall":{"id":"a","args":{"n":2}}}))
            .is_err());
        let mut tools = StreamTools::default();
        assert!(tools
            .push(&json!({"functionCall":{"args":{"large":"x".repeat(1024*1024)}}}))
            .is_err());
    }
}
