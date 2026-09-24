use crate::config::snapshot::SNAPSHOT_SCHEMA_VERSION;
use crate::runtime::clock::unix_millis;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use url::Url;

pub fn compile_snapshot_bytes(raw: &[u8], database_disabled: bool) -> Result<Vec<u8>, String> {
    let yaml: serde_yaml::Value = serde_yaml::from_slice(raw)
        .map_err(|error| format!("decode uni-api YAML configuration: {error}"))?;
    let mut config = serde_json::to_value(yaml)
        .map_err(|error| format!("convert uni-api configuration: {error}"))?;
    expand_environment(&mut config);
    let root = config
        .as_object()
        .ok_or_else(|| "uni-api configuration must be a mapping".to_owned())?;

    let mut providers = root
        .get("providers")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(compile_provider)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    // Video providers use a dedicated schema in Python. Compile them into the
    // same immutable provider graph so every Rust endpoint can route them.
    if let Some(video) = root.get("video_providers").and_then(Value::as_array) {
        for item in video {
            if let Some(mut provider) = compile_video_provider(item) {
                let name = provider
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !providers
                    .iter()
                    .any(|p| p.get("name").and_then(Value::as_str) == Some(name))
                {
                    providers.push(provider.take());
                }
            }
        }
    }
    let mut api_keys = root
        .get("api_keys")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| compile_api_key(item, database_disabled))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    expand_api_key_aliases(&mut api_keys)?;
    if api_keys.is_empty() {
        return Err("uni-api configuration contains no usable API keys".into());
    }
    if providers.is_empty() {
        return Err("uni-api configuration contains no usable providers".into());
    }

    let mut snapshot = json!({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_unix_ms": unix_millis(),
        "database_disabled": database_disabled,
        "preferences": root.get("preferences").cloned().unwrap_or_else(|| json!({})),
        "api_keys": api_keys,
        "providers": providers,
        "api_config": config,
    });
    let mut revision_payload = snapshot.clone();
    revision_payload
        .as_object_mut()
        .expect("snapshot object")
        .remove("generated_unix_ms");
    let canonical = serde_json::to_vec(&revision_payload)
        .map_err(|error| format!("encode runtime configuration revision: {error}"))?;
    let revision = format!("{:x}", Sha256::digest(canonical));
    snapshot
        .as_object_mut()
        .expect("snapshot object")
        .insert("revision".into(), Value::String(revision));
    serde_json::to_vec(&snapshot)
        .map_err(|error| format!("encode compiled runtime configuration: {error}"))
}

pub(crate) fn compile_video_provider(value: &Value) -> Option<Value> {
    let item = value.as_object()?;
    let name = item
        .get("name")
        .or_else(|| item.get("provider"))
        .and_then(Value::as_str)?
        .trim();
    let base_url = item
        .get("base_url")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if name.is_empty() || base_url.is_empty() {
        return None;
    }
    let models = item
        .get("models")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(request, cfg)| {
                    let upstream = cfg
                        .as_str()
                        .map(str::to_owned)
                        .or_else(|| {
                            cfg.get("upstream_model")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .unwrap_or_else(|| request.clone());
                    (request.clone(), upstream)
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    if models.is_empty() {
        return None;
    }
    let mut preferences = item
        .get("preferences")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(routes) = item.get("routes") {
        preferences.insert("video_routes".into(), routes.clone());
    }
    preferences.insert(
        "video_adapter".into(),
        item.get("adapter")
            .cloned()
            .unwrap_or_else(|| json!("http_json")),
    );
    let api = item
        .get("auth")
        .and_then(|a| a.get("api_key"))
        .cloned()
        .or_else(|| item.get("api").cloned())
        .unwrap_or(Value::Null);
    Some(
        json!({"name":name,"base_url":base_url,"engine":item.get("adapter").and_then(Value::as_str).unwrap_or("video"),"api":api,"models":models,"model_order":models.keys().collect::<Vec<_>>(),"preferences":preferences,"exclude_endpoints":[],"only_request_types":null,"exclude_request_types":null,"exclude_request_rules":[]}),
    )
}

pub(crate) fn compile_provider(value: &Value) -> Option<Value> {
    let item = value.as_object()?;
    let name = scalar_string(item.get("provider")?).trim().to_owned();
    if name.is_empty() {
        return None;
    }
    // Keep the configured URL byte-for-byte.  The Python runtime also keeps
    // proxy/gateway URLs here and derives the provider protocol separately.
    // Replacing a project-backed URL with the public Google endpoint would
    // silently bypass an operator-configured gateway.
    let base_url = scalar_string(item.get("base_url").unwrap_or(&Value::Null));
    let (models, model_order) = compile_models(item.get("model"));
    if models.is_empty() {
        return None;
    }
    let mut preferences = item
        .get("preferences")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !preferences.contains_key("max_request_body_bytes") {
        if let Some(limit) = item.get("max_request_body_bytes") {
            preferences.insert("max_request_body_bytes".into(), limit.clone());
        }
    }
    // Preserve provider fields consumed by Python adapters. Keeping these in
    // the immutable preferences map makes the Rust snapshot forward compatible
    // without dropping unknown provider options during compilation.
    for field in [
        "tools",
        "image",
        "video_route",
        "video_model",
        "video_provider",
        "api_key_rate_limit",
        "api_key_schedule_algorithm",
        "AUTO_RETRY",
        "project_id",
        "private_key",
        "client_email",
    ] {
        if let Some(value) = item.get(field) {
            preferences.insert(field.to_owned(), value.clone());
        }
    }
    let engine = item
        .get("engine")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| infer_engine(&base_url));
    let aws_access_key = scalar_string(item.get("aws_access_key").unwrap_or(&Value::Null));
    let aws_secret_key = scalar_string(item.get("aws_secret_key").unwrap_or(&Value::Null));
    let aws_session_token = scalar_string(item.get("aws_session_token").unwrap_or(&Value::Null));
    let client_email = scalar_string(item.get("client_email").unwrap_or(&Value::Null));
    let private_key = scalar_string(item.get("private_key").unwrap_or(&Value::Null));
    let provider_api = match item.get("api") {
        Some(value) if !value.is_null() => value.clone(),
        _ if engine.eq_ignore_ascii_case("aws") && !aws_access_key.trim().is_empty() => {
            Value::String(aws_access_key.clone())
        }
        _ if matches!(
            engine.to_ascii_lowercase().as_str(),
            "vertex" | "vertex-gemini" | "vertex-claude"
        ) && !client_email.trim().is_empty()
            && !private_key.trim().is_empty() =>
        {
            Value::String("__vertex_oauth__".into())
        }
        _ => Value::Null,
    };
    let region = scalar_string(item.get("region").unwrap_or(&Value::Null));
    let region = if region.is_empty() {
        "global".to_owned()
    } else {
        region
    };
    Some(json!({
        "name": name,
        "base_url": base_url,
        "engine": engine,
        "api": provider_api,
        "project_id": scalar_string(item.get("project_id").unwrap_or(&Value::Null)),
        "private_key": private_key,
        "client_email": client_email,
        "aws_access_key": aws_access_key,
        "aws_secret_key": aws_secret_key,
        "aws_session_token": aws_session_token,
        "cf_account_id": scalar_string(item.get("cf_account_id").unwrap_or(&Value::Null)),
        "region": region,
        "models": models,
        "model_order": model_order,
        "preferences": preferences,
        "exclude_endpoints": merge_endpoint_values(
            item.get("exclude_endpoints"),
            preferences.get("exclude_endpoints"),
        ),
        "only_request_types": item.get("only_request_types").cloned().unwrap_or(Value::Null),
        "exclude_request_types": item.get("exclude_request_types").cloned().unwrap_or(Value::Null),
        "exclude_request_rules": merge_rule_values(
            item.get("exclude_request_rules"),
            preferences.get("exclude_request_rules"),
        ),
    }))
}

pub(crate) fn infer_engine(base_url: &str) -> String {
    let lower = base_url.trim().to_ascii_lowercase();
    if Url::parse(base_url)
        .ok()
        .is_some_and(|url| url.host_str() == Some("api.typesafe.ai"))
        || lower.trim_end_matches('/').ends_with("/v1/systemone")
    {
        return "typesafe".into();
    }
    if lower.contains("/v1/messages") || lower.contains("/claude/") {
        return "claude".into();
    }
    if lower.contains("/v1beta") || lower.contains("generativelanguage.googleapis.com") {
        return "gemini".into();
    }
    if lower.contains("aiplatform.googleapis.com") {
        return "vertex".into();
    }
    if lower.contains("amazonaws.com") {
        return "aws".into();
    }
    if lower.contains("api.cohere.com") {
        return "cohere".into();
    }
    if lower.contains("volces.com/api/v3") || lower.contains("doubao") {
        return "doubao-translation".into();
    }
    if lower.contains("azure.com") {
        return "azure".into();
    }
    if lower.contains("databricks") {
        return "azure-databricks".into();
    }
    if lower.contains("cloudflare") || lower.contains("workers.dev") {
        return "cloudflare".into();
    }
    "gpt".into()
}

pub(crate) fn expand_api_key_aliases(api_keys: &mut [Value]) -> Result<(), String> {
    let mut by_token = BTreeMap::new();
    for (index, item) in api_keys.iter().enumerate() {
        if let Some(token) = item
            .get("token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            by_token.insert(token.to_owned(), index);
        }
    }
    let original = api_keys.to_vec();
    for index in 0..api_keys.len() {
        let rules = original[index]
            .get("model_rules")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut expanded = Vec::new();
        let mut visiting = Vec::new();
        for rule in rules {
            expand_rule(&rule, &original, &by_token, &mut visiting, &mut expanded)?;
        }
        if expanded.is_empty() {
            expanded.push(Value::String("all".into()));
        }
        api_keys[index]
            .as_object_mut()
            .expect("compiled API key is an object")
            .insert(
                "model_rules".into(),
                Value::Array(deduplicate_values(expanded)),
            );
    }
    Ok(())
}

pub(crate) fn expand_rule(
    rule: &Value,
    api_keys: &[Value],
    by_token: &BTreeMap<String, usize>,
    visiting: &mut Vec<String>,
    output: &mut Vec<Value>,
) -> Result<(), String> {
    let Some(raw) = rule
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let Some((alias, requested_model)) = raw.split_once('/') else {
        output.push(Value::String(raw.to_owned()));
        return Ok(());
    };
    let Some(alias_index) = by_token.get(alias).copied() else {
        output.push(Value::String(raw.to_owned()));
        return Ok(());
    };
    if visiting.iter().any(|item| item == alias) {
        return Err(format!("cyclic API key model alias involving {alias}"));
    }
    visiting.push(alias.to_owned());
    let alias_rules = api_keys[alias_index]
        .get("model_rules")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for alias_rule in alias_rules {
        let Some(alias_rule_text) = alias_rule.as_str() else {
            continue;
        };
        if requested_model == "*" {
            expand_rule(
                &Value::String(alias_rule_text.to_owned()),
                api_keys,
                by_token,
                visiting,
                output,
            )?;
            continue;
        }
        let matches = alias_rule_text == requested_model
            || alias_rule_text.ends_with("/*")
            || alias_rule_text.ends_with(&format!("/{requested_model}"));
        if matches {
            let concrete = if alias_rule_text.ends_with("/*") {
                alias_rule_text.trim_end_matches('*').to_owned() + requested_model
            } else {
                alias_rule_text.to_owned()
            };
            expand_rule(
                &Value::String(concrete),
                api_keys,
                by_token,
                visiting,
                output,
            )?;
        }
    }
    visiting.pop();
    Ok(())
}

pub(crate) fn deduplicate_values(values: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::BTreeSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(value.to_string()))
        .collect()
}

pub(crate) fn compile_api_key(value: &Value, _database_disabled: bool) -> Option<Value> {
    let item = value.as_object()?;
    let token = scalar_string(item.get("api")?).trim().to_owned();
    if token.is_empty() {
        return None;
    }
    let configured_rules = item
        .get("model")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![Value::String("all".into())]);
    let mut model_rules = Vec::with_capacity(configured_rules.len());
    let mut weights = item
        .get("weights")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for rule in configured_rules {
        if let Some(value) = rule.as_str() {
            model_rules.push(Value::String(value.to_owned()));
            continue;
        }
        let Some(object) = rule.as_object() else {
            continue;
        };
        if let Some((name, weight)) = object.iter().next() {
            model_rules.push(Value::String(name.clone()));
            weights.insert(name.clone(), weight.clone());
        }
    }
    if model_rules.is_empty() {
        model_rules.push(Value::String("all".into()));
    }
    let mut preferences = item
        .get("preferences")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    // Keep the original rules as a route graph. The flattened model_rules
    // remain useful for fast matching, while this metadata lets the native
    // router traverse parent/child key boundaries without inventing channels.
    preferences.insert("__route_graph".into(), Value::Array(model_rules.clone()));
    Some(json!({
        "token": token,
        "model_rules": model_rules,
        "role": item.get("role").and_then(Value::as_str).unwrap_or_else(|| token.get(..8).unwrap_or(&token)),
        "weights": weights,
        "preferences": preferences,
        "native_paid_state_safe": true,
    }))
}

pub(crate) fn compile_models(value: Option<&Value>) -> (BTreeMap<String, String>, Vec<String>) {
    let mut models = BTreeMap::new();
    let mut order = Vec::new();
    for item in value.and_then(Value::as_array).into_iter().flatten() {
        if let Some(model) = item.as_str() {
            if !models.contains_key(model) {
                order.push(model.to_owned());
            }
            models.insert(model.to_owned(), model.to_owned());
            continue;
        }
        let Some(mapping) = item.as_object() else {
            continue;
        };
        for (upstream, exposed) in mapping {
            let exposed = scalar_string(exposed);
            if exposed.is_empty() {
                continue;
            }
            if !models.contains_key(&exposed) {
                order.push(exposed.clone());
            }
            models.insert(exposed, upstream.clone());
        }
    }
    (models, order)
}

pub(crate) fn merge_endpoint_values(first: Option<&Value>, second: Option<&Value>) -> Vec<Value> {
    let mut values = Vec::new();
    for source in [first, second].into_iter().flatten() {
        if let Some(items) = source.as_array() {
            values.extend(items.iter().cloned());
        } else if !source.is_null() {
            values.push(source.clone());
        }
    }
    values
}

pub(crate) fn merge_rule_values(first: Option<&Value>, second: Option<&Value>) -> Vec<Value> {
    let mut values = Vec::new();
    for source in [first, second].into_iter().flatten() {
        if let Some(items) = source.as_array() {
            values.extend(items.iter().filter(|item| item.is_object()).cloned());
        } else if source.is_object() {
            values.push(source.clone());
        }
    }
    values
}

pub(crate) fn scalar_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        _ => String::new(),
    }
}

pub(crate) fn expand_environment(value: &mut Value) {
    match value {
        Value::Object(object) => object.values_mut().for_each(expand_environment),
        Value::Array(items) => items.iter_mut().for_each(expand_environment),
        Value::String(text) => *text = expand_string(text),
        _ => {}
    }
}

pub(crate) fn expand_string(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = String::with_capacity(value.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        let Some(relative) = value[cursor..].find("${") else {
            output.push_str(&value[cursor..]);
            break;
        };
        let start = cursor + relative;
        output.push_str(&value[cursor..start]);
        let Some(end_relative) = value[start + 2..].find('}') else {
            output.push_str(&value[start..]);
            break;
        };
        let end = start + 2 + end_relative;
        let expression = &value[start + 2..end];
        let (name, default) = expression
            .split_once(":-")
            .map(|(name, default)| (name, Some(default)))
            .unwrap_or((expression, None));
        let replacement = std::env::var(name)
            .ok()
            .or_else(|| default.map(str::to_owned));
        if let Some(replacement) = replacement {
            output.push_str(&replacement);
        }
        cursor = end + 1;
    }
    output
}
