use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const MAX_ITEM_ID_LENGTH: usize = 64;
const HASHED_ITEM_ID_HEX_LENGTH: usize = 40;

#[derive(Default)]
pub(crate) struct ResponsesItemIdNormalizer {
    id_map: HashMap<String, String>,
    seen_ids: HashSet<String>,
}

impl ResponsesItemIdNormalizer {
    pub(crate) fn normalize(&mut self, payload: &mut Value) -> Result<bool, String> {
        let Some(root) = payload.as_object_mut() else {
            return Ok(false);
        };
        let existing = collect_item_ids(root);
        let mut pending = HashMap::new();
        let mut target_owners = HashMap::new();

        for item in item_locations(root) {
            let Some(original) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(prefix) = item
                .get("type")
                .and_then(Value::as_str)
                .and_then(item_prefix)
            else {
                continue;
            };
            let Some(normalized) = normalized_id(original, prefix) else {
                continue;
            };
            self.register_pending(
                &mut pending,
                &mut target_owners,
                &existing,
                original,
                normalized,
            )?;
        }

        let item_reference = root
            .get("item_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if let Some(original) = item_reference.as_deref() {
            if let Some(normalized) = pending
                .get(original)
                .or_else(|| self.id_map.get(original))
                .cloned()
                .or_else(|| {
                    root.get("type")
                        .and_then(Value::as_str)
                        .and_then(event_prefix)
                        .and_then(|prefix| normalized_id(original, prefix))
                })
            {
                self.register_pending(
                    &mut pending,
                    &mut target_owners,
                    &existing,
                    original,
                    normalized,
                )?;
            }
        }

        let mut changed = false;
        visit_items_mut(root, |item| {
            let current = item.get("id").and_then(Value::as_str).map(str::to_owned);
            if let Some(normalized) = current
                .as_deref()
                .and_then(|current| pending.get(current).or_else(|| self.id_map.get(current)))
            {
                item.insert("id".into(), Value::String(normalized.clone()));
                changed = true;
            }
        });
        if let Some(current) = item_reference {
            if let Some(normalized) = pending.get(&current).or_else(|| self.id_map.get(&current)) {
                root.insert("item_id".into(), Value::String(normalized.clone()));
                changed = true;
            }
        }

        self.id_map.extend(pending);
        self.seen_ids.extend(collect_item_ids(root));
        Ok(changed)
    }

    fn register_pending(
        &self,
        pending: &mut HashMap<String, String>,
        target_owners: &mut HashMap<String, String>,
        existing: &HashSet<String>,
        original: &str,
        normalized: String,
    ) -> Result<(), String> {
        if let Some(prior) = pending.get(original).or_else(|| self.id_map.get(original)) {
            if prior != &normalized {
                return Err(
                    "Responses item ID normalization produced an inconsistent mapping".into(),
                );
            }
            return Ok(());
        }
        if normalized != original && existing.contains(&normalized) {
            return Err(
                "Responses item ID normalization would collide with an existing item ID".into(),
            );
        }
        if let Some(owner) = target_owners.get(&normalized) {
            if owner != original {
                return Err(
                    "Responses item ID normalization would collide with another normalized item ID"
                        .into(),
                );
            }
        }
        if self.seen_ids.contains(&normalized)
            && self.id_map.get(original).map(String::as_str) != Some(normalized.as_str())
        {
            return Err(
                "Responses item ID normalization would collide with a prior stream item ID".into(),
            );
        }
        target_owners.insert(normalized.clone(), original.to_owned());
        pending.insert(original.to_owned(), normalized);
        Ok(())
    }
}

pub(crate) fn normalize_response_root(root: &mut Map<String, Value>) -> Result<bool, String> {
    let mut payload = Value::Object(std::mem::take(root));
    let result = ResponsesItemIdNormalizer::default().normalize(&mut payload);
    *root = payload
        .as_object_mut()
        .map(std::mem::take)
        .ok_or_else(|| "Responses item ID normalization lost object payload".to_owned())?;
    result
}

pub(crate) fn event_item_id_needs_normalization(event_type: &[u8], item_id: &[u8]) -> bool {
    let Some(prefix) = event_prefix_bytes(event_type) else {
        return false;
    };
    let Some(suffix) = item_id.strip_prefix(prefix) else {
        return true;
    };
    item_id.len() > MAX_ITEM_ID_LENGTH
        || suffix.is_empty()
        || !suffix.iter().all(u8::is_ascii_alphanumeric)
}

fn normalized_id(original: &str, prefix: &str) -> Option<String> {
    let canonical_prefix = format!("{prefix}_");
    if let Some(suffix) = original.strip_prefix(&canonical_prefix) {
        if !suffix.is_empty()
            && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric())
            && original.len() <= MAX_ITEM_ID_LENGTH
        {
            return None;
        }
    }
    if let Some((_current_prefix, suffix)) = original.split_once('_') {
        let candidate = format!("{prefix}_{suffix}");
        if !suffix.is_empty()
            && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric())
            && candidate.len() <= MAX_ITEM_ID_LENGTH
        {
            return (candidate != original).then_some(candidate);
        }
    }
    let digest = Sha256::digest(original.as_bytes());
    let hex = format!("{digest:x}");
    Some(format!("{prefix}_{}", &hex[..HASHED_ITEM_ID_HEX_LENGTH]))
}

fn item_prefix(item_type: &str) -> Option<&'static str> {
    match item_type {
        "message" => Some("msg"),
        "reasoning" => Some("rs"),
        "function_call" => Some("fc"),
        "function_call_output" => Some("fco"),
        "tool_search_call" => Some("tsc"),
        "custom_tool_call" => Some("ctc"),
        "custom_tool_call_output" => Some("ctco"),
        _ => None,
    }
}

fn event_prefix(event_type: &str) -> Option<&'static str> {
    event_prefix_bytes(event_type.as_bytes()).and_then(|prefix| {
        std::str::from_utf8(prefix)
            .ok()
            .map(|prefix| prefix.trim_end_matches('_'))
    })
}

fn event_prefix_bytes(event_type: &[u8]) -> Option<&'static [u8]> {
    match event_type {
        b"response.output_text.delta" | b"response.output_text.done" => Some(b"msg_"),
        b"response.reasoning_summary_text.delta"
        | b"response.reasoning_summary_text.done"
        | b"response.reasoning_text.delta"
        | b"response.reasoning_text.done" => Some(b"rs_"),
        b"response.function_call_arguments.delta" | b"response.function_call_arguments.done" => {
            Some(b"fc_")
        }
        b"response.custom_tool_call_input.delta" | b"response.custom_tool_call_input.done" => {
            Some(b"ctc_")
        }
        _ => None,
    }
}

fn item_locations(root: &Map<String, Value>) -> Vec<&Map<String, Value>> {
    let mut locations = Vec::new();
    if let Some(item) = root.get("item").and_then(Value::as_object) {
        locations.push(item);
    }
    for collection in ["input", "output"] {
        if let Some(items) = root.get(collection).and_then(Value::as_array) {
            locations.extend(items.iter().filter_map(Value::as_object));
        }
    }
    if let Some(items) = root
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("output"))
        .and_then(Value::as_array)
    {
        locations.extend(items.iter().filter_map(Value::as_object));
    }
    locations
}

fn visit_items_mut(root: &mut Map<String, Value>, mut visit: impl FnMut(&mut Map<String, Value>)) {
    if let Some(item) = root.get_mut("item").and_then(Value::as_object_mut) {
        visit(item);
    }
    for collection in ["input", "output"] {
        if let Some(items) = root.get_mut(collection).and_then(Value::as_array_mut) {
            for item in items.iter_mut().filter_map(Value::as_object_mut) {
                visit(item);
            }
        }
    }
    if let Some(items) = root
        .get_mut("response")
        .and_then(Value::as_object_mut)
        .and_then(|response| response.get_mut("output"))
        .and_then(Value::as_array_mut)
    {
        for item in items.iter_mut().filter_map(Value::as_object_mut) {
            visit(item);
        }
    }
}

fn collect_item_ids(root: &Map<String, Value>) -> HashSet<String> {
    item_locations(root)
        .into_iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_type_prefixes_wrong_prefixes_and_long_ids() {
        let mut payload = json!({"input":[
            {"type":"message","id":"item_message123"},
            {"type":"function_call","id":"item_function123"},
            {"type":"tool_search_call","id":"fc_3e812ff47c529a7d687ecf89239a0f57433566d7"},
            {"type":"custom_tool_call","id":"fc_custom123"},
            {"type":"custom_tool_call","id":"ctc_not-canonical"},
            {"type":"message","id":format!("resp_{}_msg", "a".repeat(76))},
            {"type":"future_item","id":"item_future123"}
        ]});
        let mut normalizer = ResponsesItemIdNormalizer::default();

        assert!(normalizer.normalize(&mut payload).unwrap());
        assert_eq!(payload["input"][0]["id"], "msg_message123");
        assert_eq!(payload["input"][1]["id"], "fc_function123");
        assert_eq!(
            payload["input"][2]["id"],
            "tsc_3e812ff47c529a7d687ecf89239a0f57433566d7"
        );
        assert_eq!(payload["input"][3]["id"], "ctc_custom123");
        assert!(payload["input"][4]["id"]
            .as_str()
            .unwrap()
            .starts_with("ctc_"));
        assert!(!payload["input"][4]["id"].as_str().unwrap().contains('-'));
        assert!(payload["input"][5]["id"]
            .as_str()
            .unwrap()
            .starts_with("msg_"));
        assert!(payload["input"][5]["id"].as_str().unwrap().len() <= 64);
        assert_eq!(payload["input"][6]["id"], "item_future123");
    }

    #[test]
    fn rewrites_event_references_with_or_without_prior_item_event() {
        let mut normalizer = ResponsesItemIdNormalizer::default();
        let mut item = json!({
            "type":"response.output_item.added",
            "item":{"type":"message","id":"item_message123"}
        });
        let mut delta = json!({
            "type":"response.output_text.delta",
            "item_id":"item_message123",
            "delta":"x"
        });

        normalizer.normalize(&mut item).unwrap();
        normalizer.normalize(&mut delta).unwrap();
        assert_eq!(item["item"]["id"], "msg_message123");
        assert_eq!(delta["item_id"], "msg_message123");

        let mut direct = json!({
            "type":"response.function_call_arguments.delta",
            "item_id":"item_function123",
            "delta":"{}"
        });
        ResponsesItemIdNormalizer::default()
            .normalize(&mut direct)
            .unwrap();
        assert_eq!(direct["item_id"], "fc_function123");
    }
}
