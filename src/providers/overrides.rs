use crate::config::snapshot::Provider;
use serde_json::{Map, Value};

pub(crate) fn apply_overrides(
    root: &mut Map<String, Value>,
    provider: &Provider,
    request_model: &str,
) {
    if let Some(overrides) = provider
        .preferences
        .get("post_body_parameter_overrides")
        .and_then(Value::as_object)
    {
        apply_override_section(root, overrides, provider, true);
        if let Some(model) = overrides.get(request_model).and_then(Value::as_object) {
            apply_override_section(root, model, provider, false);
        }
    }
    // Codex wire requirements also apply when no overrides are configured,
    // and cannot be undone by provider-wide or model-specific overrides.
    // Endpoint-specific sanitizers still run afterwards (e.g. compact drops store).
    if provider.engine.trim().eq_ignore_ascii_case("codex") {
        root.insert("store".into(), Value::Bool(false));
        root.remove("response_format");
        root.remove("temperature");
    }
}

pub(crate) fn apply_override_section(
    root: &mut Map<String, Value>,
    section: &Map<String, Value>,
    provider: &Provider,
    skip_model_keys: bool,
) {
    for (key, value) in section {
        if key == "__remove__"
            || matches!(key.as_str(), "service_tier" | "translation_options")
            || (skip_model_keys && provider.models.contains_key(key))
        {
            continue;
        }
        merge_value(root.entry(key.clone()).or_insert(Value::Null), value);
    }
    if let Some(removals) = section.get("__remove__") {
        apply_removals(root, removals);
    }
}

pub(crate) fn merge_value(target: &mut Value, replacement: &Value) {
    match (target, replacement) {
        (Value::Object(target), Value::Object(replacement)) => {
            for (key, value) in replacement {
                if key == "__remove__" {
                    continue;
                }
                merge_value(target.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (target, replacement) => *target = replacement.clone(),
    }
}

pub(crate) fn apply_removals(root: &mut Map<String, Value>, removals: &Value) {
    let items = match removals {
        Value::Array(items) => items.clone(),
        value => vec![value.clone()],
    };
    for item in items {
        if let Some(path) = item.as_str() {
            if !matches!(path, "service_tier" | "translation_options") {
                delete_path(root, path);
            }
        } else if let Some(rule) = item.as_object() {
            if !matches!(
                rule.get("path").and_then(Value::as_str),
                Some("service_tier" | "translation_options")
            ) {
                apply_structured_removal(root, rule);
            }
        }
    }
}

pub(crate) fn apply_structured_removal(root: &mut Map<String, Value>, rule: &Map<String, Value>) {
    let Some(path) = rule.get("path").and_then(Value::as_str) else {
        return;
    };
    if !rule.contains_key("where") && !rule.contains_key("where_any") {
        delete_path(root, path);
        return;
    }
    let Some(target) = get_path_mut(root, path) else {
        return;
    };
    let should_remove = |value: &Value| {
        rule.get("where")
            .is_some_and(|condition| matches_condition(value, condition))
            || rule
                .get("where_any")
                .is_some_and(|condition| matches_any_condition(value, condition))
    };
    if let Some(items) = target.as_array_mut() {
        items.retain(|item| !should_remove(item));
        if items.is_empty() && rule.get("drop_empty").and_then(Value::as_bool) == Some(true) {
            delete_path(root, path);
        }
    } else if should_remove(target) {
        delete_path(root, path);
    }
}

pub(crate) fn matches_any_condition(value: &Value, condition: &Value) -> bool {
    condition
        .as_array()
        .map(|conditions| conditions.iter().any(|item| matches_condition(value, item)))
        .unwrap_or_else(|| matches_condition(value, condition))
}

pub(crate) fn matches_condition(value: &Value, condition: &Value) -> bool {
    if condition.is_string() {
        return value == condition;
    }
    let Some(condition) = condition.as_object() else {
        return false;
    };
    condition.iter().all(|(path, expected)| {
        get_value_path(value, path).is_some_and(|actual| actual == expected)
    })
}

pub(crate) fn delete_path(root: &mut Map<String, Value>, path: &str) {
    let parts = path
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        let Some(next) = current.get_mut(*part).and_then(Value::as_object_mut) else {
            return;
        };
        current = next;
    }
    current.remove(parts[parts.len() - 1]);
}

pub(crate) fn get_path_mut<'a>(
    root: &'a mut Map<String, Value>,
    path: &str,
) -> Option<&'a mut Value> {
    let mut parts = path.split('.').filter(|part| !part.trim().is_empty());
    let first = parts.next()?;
    let mut current = root.get_mut(first)?;
    for part in parts {
        current = current.as_object_mut()?.get_mut(part)?;
    }
    Some(current)
}

pub(crate) fn get_value_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for part in path.split('.').filter(|part| !part.trim().is_empty()) {
        current = current.as_object()?.get(part)?;
    }
    Some(current)
}
