//! Per-provider intent over the base snapshot. No YAML write, model request or
//! paid probe is performed by editing/validation. Secrets are opaque references.
use crate::responses_native::{NativeConfigStore, Provider, Snapshot};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

type Failure = (StatusCode, String);
fn bad(s: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, s.into())
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProviderSettings {
    #[serde(default)]
    pub set: BTreeMap<String, Value>,
    #[serde(default)]
    pub remove: Vec<String>,
    #[serde(skip)]
    pub compiled: Option<Arc<Provider>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Mutation {
    pub revision: String,
    pub operation_id: String,
    #[serde(default)]
    pub changes: Vec<Change>,
    #[serde(default)]
    pub sample: Value,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Change {
    pub provider: String,
    #[serde(default)]
    pub set: BTreeMap<String, Value>,
    #[serde(default)]
    pub remove: Vec<String>,
    #[serde(default)]
    pub reset: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub copy_to_key: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub create_to_key: String,
    #[serde(default)]
    pub delete_copy: bool,
}

pub(crate) fn identity_changed(s: &ProviderSettings) -> bool {
    s.set
        .keys()
        .chain(s.remove.iter())
        .any(|p| p == "/api" || p.starts_with("/api/") || p == "/base_url")
}
pub(crate) fn digest<T: Serialize>(v: &T) -> String {
    fn normalize(v: Value) -> Value {
        match v {
            Value::Object(o) => {
                Value::Object(o.into_iter().map(|(k, v)| (k, normalize(v))).collect())
            }
            Value::Array(a) => Value::Array(a.into_iter().map(normalize).collect()),
            Value::Number(ref n) if n.is_f64() => {
                let f = n.as_f64().unwrap();
                if f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0 {
                    json!(f as i64)
                } else {
                    v
                }
            }
            _ => v,
        }
    }
    hex::encode(Sha256::digest(
        serde_json::to_vec(&normalize(serde_json::to_value(v).unwrap_or(Value::Null)))
            .unwrap_or_default(),
    ))
}

fn pointer_parts(path: &str) -> Vec<String> {
    path.trim_start_matches('/')
        .split('/')
        .map(|p| p.replace("~1", "/").replace("~0", "~"))
        .collect()
}
fn assign(root: &mut Value, parts: &[String], value: Option<Value>) {
    if parts.is_empty() {
        return;
    }
    if !root.is_object() {
        *root = json!({})
    }
    let obj = root.as_object_mut().unwrap();
    if parts.len() == 1 {
        if let Some(v) = value {
            obj.insert(parts[0].clone(), v);
        } else {
            obj.remove(&parts[0]);
        }
        return;
    }
    if value.is_none() && !obj.contains_key(&parts[0]) {
        return;
    }
    assign(
        obj.entry(parts[0].clone()).or_insert(json!({})),
        &parts[1..],
        value,
    );
}
pub(crate) fn merge(base: &Value, set: &BTreeMap<String, Value>, remove: &[String]) -> Value {
    let mut out = base.clone();
    for path in remove {
        assign(&mut out, &pointer_parts(path), None)
    }
    for (path, v) in set {
        assign(&mut out, &pointer_parts(path), Some(v.clone()))
    }
    out
}
fn collect_diff(before: &Value, after: &Value, path: &str, out: &mut ProviderSettings) {
    if before == after {
        return;
    }
    if let (Some(a), Some(b)) = (before.as_object(), after.as_object()) {
        for (k, v) in a {
            let child = format!("{path}/{}", k.replace('~', "~0").replace('/', "~1"));
            if let Some(next) = b.get(k) {
                collect_diff(v, next, &child, out)
            } else {
                out.remove.push(child)
            }
        }
        for (k, v) in b {
            if !a.contains_key(k) {
                collect_diff(
                    &json!({}),
                    v,
                    &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                    out,
                );
            }
        }
    } else {
        out.set.insert(path.into(), after.clone());
    }
}
pub(crate) fn document(base: &Snapshot, p: &Provider) -> Value {
    if let Some(value) = base
        .api_config
        .get("providers")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row["provider"].as_str() == Some(p.name.as_ref()))
        })
    {
        return value.clone();
    }
    let models: Vec<Value> = p
        .models
        .iter()
        .map(|(public, upstream)| {
            if public == upstream {
                json!(public)
            } else {
                json!({upstream:public})
            }
        })
        .collect();
    json!({"provider":p.name.as_ref(),"base_url":p.base_url.as_ref(),"engine":p.engine.as_ref(),"api":p.api_keys.as_ref(),"model":models,"preferences":{},"exclude_endpoints":p.excluded_endpoints.as_ref(),"only_request_types":p.only_request_types.as_ref(),"exclude_request_types":p.excluded_request_types.as_ref(),"exclude_request_rules":p.excluded_request_rules.as_ref()})
}
fn is_secret(path: &str) -> bool {
    path == "/api"
        || [
            "password",
            "secret",
            "private_key",
            "access_token",
            "authorization",
        ]
        .iter()
        .any(|k| {
            path.to_ascii_lowercase()
                .split('/')
                .any(|part| part.contains(k))
        })
        || path.starts_with("/api/")
        || [
            "/private_key",
            "/aws_access_key",
            "/aws_secret_key",
            "/aws_session_token",
            "/client_email",
            "/preferences/proxy",
        ]
        .contains(&path)
        || path.starts_with("/preferences/headers/")
        || path == "/preferences/headers"
        || (path.starts_with("/preferences/post_body_parameter_overrides/")
            && ["password", "secret", "token", "api_key", "authorization"]
                .iter()
                .any(|key| path.to_ascii_lowercase().contains(key)))
}
fn secret_ref(path: &str, v: &Value) -> Value {
    json!({"$secret":format!("{}:{}",path,digest(v))})
}
fn redact_at(v: &Value, path: &str) -> Value {
    if is_secret(path) && !v.is_null() {
        if path == "/api" {
            return match v {
                Value::Array(a) => Value::Array(
                    a.iter()
                        .enumerate()
                        .map(|(i, v)| secret_ref(&format!("{path}/{i}"), v))
                        .collect(),
                ),
                _ => secret_ref(path, v),
            };
        }
        return secret_ref(path, v);
    }
    match v {
        Value::Object(o) => Value::Object(
            o.iter()
                .filter(|(k, _)| !k.starts_with("__temporary"))
                .map(|(k, v)| {
                    (
                        k.clone(),
                        redact_at(
                            v,
                            &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                        ),
                    )
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(
            a.iter()
                .enumerate()
                .map(|(i, v)| redact_at(v, &format!("{path}/{i}")))
                .collect(),
        ),
        _ => v.clone(),
    }
}
fn secrets(v: &Value, path: &str, out: &mut BTreeMap<String, Value>) {
    if is_secret(path) {
        if let Some(k) = secret_ref(path, v)["$secret"].as_str() {
            out.insert(k.into(), v.clone());
        }
    }
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                secrets(
                    v,
                    &format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")),
                    out,
                )
            }
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                secrets(v, &format!("{path}/{i}"), out)
            }
        }
        _ => {}
    }
}
fn resolve_refs(v: &Value, available: &BTreeMap<String, Value>) -> Result<Value, Failure> {
    if let Some(reference) = v.get("$secret").and_then(Value::as_str) {
        return available
            .get(reference)
            .cloned()
            .ok_or_else(|| bad("Secret reference expired; refresh this channel"));
    }
    Ok(match v {
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| Ok((k.clone(), resolve_refs(v, available)?)))
                .collect::<Result<_, Failure>>()?,
        ),
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|v| resolve_refs(v, available))
                .collect::<Result<_, _>>()?,
        ),
        _ => v.clone(),
    })
}
fn supported_paths() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("base_url", "基本与模型", "string"),
        ("engine", "基本与模型", "engine"),
        ("model", "基本与模型", "models"),
        ("api", "密钥", "keys"),
        ("api_key_schedule_algorithm", "密钥", "algorithm"),
        ("api_key_rate_limit", "密钥", "json"),
        (
            "preferences/api_key_schedule_algorithm",
            "密钥",
            "algorithm",
        ),
        ("preferences/api_key_rate_limit", "密钥", "json"),
        ("preferences/model_timeout", "超时与冷却", "json"),
        ("preferences/timeout_policy", "超时与冷却", "json"),
        ("preferences/keepalive_interval", "超时与冷却", "json"),
        ("preferences/cooldown_period", "超时与冷却", "number"),
        (
            "preferences/api_key_cooldown_period",
            "超时与冷却",
            "number",
        ),
        (
            "preferences/api_key_rate_limit_cooldown_period",
            "超时与冷却",
            "number",
        ),
        (
            "preferences/api_key_quota_cooldown_period",
            "超时与冷却",
            "number",
        ),
        ("AUTO_RETRY", "超时与冷却", "boolean"),
        ("exclude_endpoints", "请求规则", "json"),
        ("only_request_types", "请求规则", "json"),
        ("exclude_request_types", "请求规则", "json"),
        ("exclude_request_rules", "请求规则", "json"),
        ("preferences/max_request_body_bytes", "请求规则", "json"),
        ("preferences/headers", "请求改写", "json"),
        (
            "preferences/post_body_parameter_overrides",
            "请求改写",
            "json",
        ),
        (
            "preferences/normalize_responses_custom_tool_call_ids",
            "请求改写",
            "json",
        ),
        ("tools", "网络与适配", "boolean"),
        ("image", "网络与适配", "boolean"),
        ("preferences/proxy", "网络与适配", "secret"),
        ("preferences/balance_query", "网络与适配", "boolean"),
        ("project_id", "网络与适配", "string"),
        ("region", "网络与适配", "string"),
        ("client_email", "网络与适配", "secret"),
        ("private_key", "网络与适配", "secret"),
        ("aws_access_key", "网络与适配", "secret"),
        ("aws_secret_key", "网络与适配", "secret"),
        ("aws_session_token", "网络与适配", "secret"),
        ("cf_account_id", "网络与适配", "string"),
    ]
}
pub(crate) fn schema() -> Value {
    json!({"version":1,"supported":true,"create_typesafe":true,"storage":"console_overlay","fields":supported_paths().iter().map(|(p,g,t)|json!({"path":format!("/{p}"),"group":g,"type":t,"hot_update":true})).collect::<Vec<_>>(),"engines":["typesafe","gpt","codex","claude","gemini","vertex","vertex-gemini","vertex-claude","aws","azure","azure-databricks","openrouter","cloudflare","cohere","jina","tavily","exa","doubao-translation"],"key_algorithms":["round_robin","fixed_priority","random","lottery"],"algorithm_note":"smart_round_robin 在本运行时按轮询执行，不能作为成功率策略","separate_scopes":{"api_key":["SCHEDULING_ALGORITHM","weights","AUTO_RETRY"],"global":["hedging"]}})
}
fn validate_paths(c: &Change) -> Result<(), Failure> {
    let roots: BTreeSet<String> = supported_paths()
        .iter()
        .map(|(p, _, _)| format!("/{p}"))
        .collect();
    if c.provider.is_empty() || c.provider.len() > 256 || c.set.len() + c.remove.len() > 128 {
        return Err(bad("Channel settings patch is too large"));
    }
    for p in c.set.keys().chain(c.remove.iter()) {
        if !roots
            .iter()
            .any(|r| p == r || (p.starts_with(&(r.clone() + "/")) && r != "/api" && r != "/model"))
            || p.contains("__temporary")
            || p.contains("__route_graph")
        {
            return Err(bad(format!("Unsupported setting path: {p}")));
        }
    }
    Ok(())
}
fn field_error(path: &str) -> String {
    format!("Invalid channel setting: {path}")
}
pub(crate) fn compile(raw: &Value, previous: &Arc<Provider>) -> Result<Arc<Provider>, String> {
    if raw["provider"].as_str() != Some(previous.name.as_ref()) {
        return Err(field_error("/provider (identity cannot change)"));
    }
    let url = url::Url::parse(raw["base_url"].as_str().ok_or(field_error("/base_url"))?)
        .map_err(|_| field_error("/base_url"))?;
    if !["http", "https"].contains(&url.scheme())
        || url.host_str().is_none()
        || url.password().is_some()
        || !url.username().is_empty()
    {
        return Err(field_error("/base_url"));
    }
    if let Some(e) = raw.get("engine") {
        if !schema()["engines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == e)
            && e.as_str() != Some(previous.engine.as_ref())
        {
            return Err(field_error("/engine"));
        }
    }
    if raw
        .get("model")
        .is_some_and(|v| !v.is_null() && !v.is_array() && !v.is_object())
    {
        return Err(field_error("/model"));
    }
    if let Some(api) = raw.get("api") {
        let valid = |v: &Value| {
            v.as_str()
                .is_some_and(|s| !s.is_empty() && s.len() <= 16384 && !s.contains(['\r', '\n']))
        };
        if !(api.is_null()
            || valid(api)
            || api
                .as_array()
                .is_some_and(|a| !a.is_empty() && a.len() <= 1024 && a.iter().all(valid)))
        {
            return Err(field_error("/api"));
        }
    }
    if let Some(p) = raw.get("preferences") {
        let p = p.as_object().ok_or(field_error("/preferences"))?;
        for (k, v) in p {
            if k.starts_with("__") {
                return Err(field_error("/preferences/internal"));
            }
            if k.ends_with("cooldown_period")
                && !v.as_f64().is_some_and(|n| n >= 0.0 && n.is_finite())
            {
                return Err(field_error(&format!("/preferences/{k}")));
            }
        }
        if let Some(a) = p.get("api_key_schedule_algorithm") {
            if ![
                "round_robin",
                "fixed_priority",
                "random",
                "lottery",
                "smart_round_robin",
            ]
            .contains(&a.as_str().unwrap_or(""))
            {
                return Err(field_error("/preferences/api_key_schedule_algorithm"));
            }
        }
        if let Some(h) = p.get("headers") {
            let h = h.as_object().ok_or(field_error("/preferences/headers"))?;
            for (k, v) in h {
                if reqwest::header::HeaderName::from_bytes(k.as_bytes()).is_err()
                    || v.as_str()
                        .is_none_or(|v| reqwest::header::HeaderValue::from_str(v).is_err())
                {
                    return Err(field_error("/preferences/headers"));
                }
            }
        }
        if let Some(p) = p.get("proxy") {
            let u = url::Url::parse(p.as_str().ok_or(field_error("/preferences/proxy"))?)
                .map_err(|_| field_error("/preferences/proxy"))?;
            if !["http", "https", "socks5", "socks5h"].contains(&u.scheme()) {
                return Err(field_error("/preferences/proxy"));
            }
        }
        if let Some(p) = p.get("post_body_parameter_overrides") {
            if !p.is_object() {
                return Err(field_error("/preferences/post_body_parameter_overrides"));
            }
        }
    }
    let mut materialized = raw.clone();
    if raw.get("model").is_none_or(Value::is_null) {
        materialized["model"] = json!(previous
            .models
            .iter()
            .map(|(public, upstream)| json!({upstream:public}))
            .collect::<Vec<_>>());
    }
    if let Some(models) = materialized.get("model").and_then(Value::as_array) {
        let mut exposed = BTreeSet::new();
        if models.is_empty() || models.len() > 1024 {
            return Err(field_error("/model"));
        }
        for model in models {
            match model {
                Value::String(name) if !name.trim().is_empty() => {
                    if !exposed.insert(name.clone()) {
                        return Err(field_error("/model duplicate alias"));
                    }
                }
                Value::Object(map) if !map.is_empty() => {
                    for (upstream, public) in map {
                        let public = public
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                            .ok_or(field_error("/model"))?;
                        if upstream.trim().is_empty() || !exposed.insert(public.into()) {
                            return Err(field_error("/model duplicate alias"));
                        }
                    }
                }
                _ => return Err(field_error("/model")),
            }
        }
    } else {
        return Err(field_error("/model"));
    }
    for key in [
        "exclude_endpoints",
        "only_request_types",
        "exclude_request_types",
    ] {
        if let Some(v) = raw.get(key) {
            if !(v.is_string() || v.as_array().is_some_and(|a| a.iter().all(Value::is_string))) {
                return Err(field_error(&format!("/{key}")));
            }
        }
    }
    for key in ["tools", "image", "AUTO_RETRY"] {
        if let Some(v) = raw.get(key) {
            if !v.is_boolean() {
                return Err(field_error(&format!("/{key}")));
            }
        }
    }
    if let Some(p) = raw.get("preferences").and_then(Value::as_object) {
        for key in ["model_timeout", "keepalive_interval"] {
            if let Some(v) = p.get(key) {
                let numeric = |v: &Value| v.as_f64().is_some_and(|n| n >= 0.0 && n.is_finite());
                if !(numeric(v) || v.as_object().is_some_and(|m| m.values().all(numeric))) {
                    return Err(field_error(&format!("/preferences/{key}")));
                }
            }
        }
        if let Some(policy) = p.get("timeout_policy") {
            let timeouts = |v: &Value| {
                v.as_object().is_some_and(|m| {
                    m.iter().all(|(k, v)| {
                        ["connect", "write", "pool", "first_byte", "idle", "total"]
                            .contains(&k.as_str())
                            && v.as_f64().is_some_and(|n| n > 0.0 && n.is_finite())
                    })
                })
            };
            if !policy.is_object()
                || policy.get("default").is_some_and(|v| !timeouts(v))
                || policy.get("rules").is_some_and(|v| {
                    !v.as_array().is_some_and(|rules| {
                        rules.iter().all(|r| {
                            r.get("match").is_some_and(Value::is_object)
                                && r.get("timeout").is_some_and(timeouts)
                        })
                    })
                })
            {
                return Err(field_error("/preferences/timeout_policy"));
            }
        }
        if let Some(rate) = p.get("api_key_rate_limit") {
            if !(rate.is_string()
                || rate
                    .as_object()
                    .is_some_and(|m| m.values().all(Value::is_string)))
            {
                return Err(field_error("/preferences/api_key_rate_limit"));
            }
        }
    }
    let compiled = crate::config::compile_provider(&materialized).ok_or(field_error("/model"))?;
    let parsed = serde_json::from_value(compiled).map_err(|_| field_error("/"))?;
    let mut next =
        (*crate::responses_native::runtime_provider(parsed, previous.cursor.clone())).clone();
    if next.models.is_empty() {
        return Err(field_error("/model"));
    }
    for k in ["tools", "image", "AUTO_RETRY", "balance_query"] {
        if let Some(v) = next.preferences.get(k) {
            if !v.is_boolean() {
                return Err(field_error(k));
            }
        }
    }
    if let Some(v) = next.preferences.get("api_key_schedule_algorithm") {
        if ![
            "round_robin",
            "fixed_priority",
            "random",
            "lottery",
            "smart_round_robin",
        ]
        .contains(&v.as_str().unwrap_or(""))
        {
            return Err(field_error("/api_key_schedule_algorithm"));
        }
    }
    if let Some(v) = next.preferences.get("api_key_rate_limit") {
        let values: Vec<&Value> = v
            .as_object()
            .map(|o| o.values().collect())
            .unwrap_or_else(|| vec![v]);
        if values
            .iter()
            .any(|v| crate::responses_native::parse_rate_limits(Some(v), None).is_none())
        {
            return Err(field_error("/api_key_rate_limit"));
        }
    }
    if let Some(rules) = raw.get("exclude_request_rules") {
        let valid = |r: &Value| {
            r.get("match").and_then(Value::as_object).is_some_and(|m| {
                !m.is_empty()
                    && m.keys().all(|k| {
                        [
                            "endpoint",
                            "request_model",
                            "upstream_model",
                            "reasoning_effort",
                            "request_type",
                        ]
                        .contains(&k.as_str())
                    })
            })
        };
        if !(valid(rules) || rules.as_array().is_some_and(|a| a.iter().all(valid))) {
            return Err(field_error("/exclude_request_rules"));
        }
    }
    if let Some(owner) = previous.preferences.get("__temporary_key_id") {
        let mut prefs = (*next.preferences).clone();
        prefs.insert("__temporary_key_id".into(), owner.clone());
        next.preferences = Arc::new(prefs);
    }
    Ok(Arc::new(next))
}
fn affected(snapshot: &Snapshot, provider: &str) -> Vec<Value> {
    let Some(inspector) = snapshot
        .api_keys
        .values()
        .find(|key| crate::channel_catalog::can_inspect_all(snapshot, key))
    else {
        return Vec::new();
    };
    snapshot
        .api_key_order
        .iter()
        .filter_map(|token| {
            snapshot.api_keys.get(token)?;
            let id = crate::channel_catalog::key_id(token);
            let entries = crate::channel_catalog::entries(snapshot, inspector, Some(&id)).ok()?;
            let models: BTreeSet<_> = entries
                .into_iter()
                .filter(|(p, _)| p.name.as_ref() == provider)
                .map(|(_, m)| m)
                .collect();
            if models.is_empty() {
                None
            } else {
                Some(json!({"key_id":id,"models":models}))
            }
        })
        .collect()
}
impl NativeConfigStore {
    // Separate, explicitly requested projection. Resolve by opaque reference so
    // reordering/deleting a draft row cannot reveal a different row's key.
    pub(crate) async fn settings_secrets(
        &self,
        provider: &str,
        revision: &str,
    ) -> Result<Value, Failure> {
        let state = self.channel_controls.read().await;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        if revision.is_empty() || state.revision(&base) != revision {
            return Err((
                StatusCode::CONFLICT,
                "Configuration changed; refresh before revealing keys".into(),
            ));
        }
        let original = state
            .temporary
            .get(provider)
            .or_else(|| base.providers_by_name.get(provider))
            .ok_or((StatusCode::NOT_FOUND, "Channel not found".into()))?;
        let raw = state
            .temporary_documents
            .get(provider)
            .cloned()
            .unwrap_or_else(|| document(&base, original));
        let setting = state.settings.get(provider).cloned().unwrap_or_default();
        let merged = merge(&raw, &setting.set, &setting.remove);
        let mut references = BTreeMap::new();
        // Only API keys from this provider's base and active intent, never other
        // credentials or another provider. Base refs also support reset drafts.
        for document in [&raw, &merged] {
            if let Some(api) = document.get("api") {
                secrets(api, "/api", &mut references);
            }
        }
        references.retain(|reference, _| {
            reference.starts_with("/api:") || reference.starts_with("/api/")
        });
        references.retain(|_, value| value.is_string());
        Ok(json!({"provider":provider,"revision":revision,"keys":references}))
    }

    pub(crate) async fn settings_view(&self, provider: &str) -> Result<Value, Failure> {
        let state = self.channel_controls.read().await;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let effective = state.overlay(base.clone());
        let p = effective
            .providers_by_name
            .get(provider)
            .ok_or((StatusCode::NOT_FOUND, "Channel not found".into()))?;
        let original = state
            .temporary
            .get(provider)
            .or_else(|| base.providers_by_name.get(provider))
            .ok_or(bad("Channel base missing"))?;
        let raw = state
            .temporary_documents
            .get(provider)
            .cloned()
            .unwrap_or_else(|| document(&base, original));
        let settings = state.settings.get(provider).cloned().unwrap_or_default();
        let merged = merge(&raw, &settings.set, &settings.remove);
        let mut schema = schema();
        schema["fields"].as_array_mut().unwrap().retain(|field| {
            let path = field["path"].as_str().unwrap_or("");
            for key in ["api_key_schedule_algorithm", "api_key_rate_limit"] {
                if path == format!("/{key}") {
                    return merged.get(key).is_some();
                }
                if path == format!("/preferences/{key}") {
                    return merged.get(key).is_none();
                }
            }
            true
        });
        let conflict = compile(&merged, original).err();
        Ok(
            json!({"provider":provider,"revision":state.revision(&base),"config_revision":base.revision.as_ref(),"kind":if state.temporary.contains_key(provider){"imported"}else{"configured"},"base":redact_at(&raw,""),"effective":redact_at(&merged,""),"override_paths":settings.set.keys().chain(settings.remove.iter()).collect::<Vec<_>>(),"affected_keys":affected(&effective,provider),"api_key_id":p.preferences.get("__temporary_key_id"),"schema":schema,"conflict":conflict,"available_keys":base.api_key_order.iter().enumerate().map(|(i,token)|json!({"key_id":crate::channel_catalog::key_id(token),"position":i+1})).collect::<Vec<_>>(),"global_preferences":redact_at(&json!({"preferences":base.preferences.as_ref()}),"")["preferences"]}),
        )
    }
    pub(crate) async fn settings_providers(&self) -> Result<Value, Failure> {
        let state = self.channel_controls.read().await;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let effective = state.overlay(base);
        Ok(
            json!({"providers":effective.providers.iter().map(|p|json!({"provider":p.name.as_ref(),"base_url":p.base_url.as_ref(),"api":p.api_keys.as_ref(),"temporary":state.temporary.contains_key(p.name.as_ref()),"identity_changed":p.name.starts_with("sub2api-copy-")||state.settings.get(p.name.as_ref()).is_some_and(identity_changed)})).collect::<Vec<_>>()}),
        )
    }
    pub(crate) async fn settings_discover(&self, input: Mutation) -> Result<Value, Failure> {
        if input.changes.len() != 1 {
            return Err(bad("Discovery requires one channel"));
        }
        self.settings_change(
            Mutation {
                revision: input.revision.clone(),
                operation_id: input.operation_id.clone(),
                changes: input.changes.clone(),
                sample: Value::Null,
            },
            false,
        )
        .await?;
        let state = self.channel_controls.read().await;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        if state.revision(&base) != input.revision {
            return Err((StatusCode::CONFLICT, "Configuration changed".into()));
        }
        let c = &input.changes[0];
        let p = state
            .temporary
            .get(&c.provider)
            .or_else(|| base.providers_by_name.get(&c.provider))
            .ok_or(bad("Channel missing"))?;
        let raw = state
            .temporary_documents
            .get(&c.provider)
            .cloned()
            .unwrap_or_else(|| document(&base, p));
        let old = state.settings.get(&c.provider).cloned().unwrap_or_default();
        let current = merge(&raw, &old.set, &old.remove);
        let mut refs = BTreeMap::new();
        secrets(&raw, "", &mut refs);
        secrets(&current, "", &mut refs);
        let resolved = c
            .set
            .iter()
            .map(|(p, v)| Ok((p.clone(), resolve_refs(v, &refs)?)))
            .collect::<Result<BTreeMap<_, _>, Failure>>()?;
        let raw = merge(if c.reset { &raw } else { &current }, &resolved, &c.remove);
        drop(state);
        let mut builder = reqwest::Client::builder();
        if let Some(proxy) = raw.pointer("/preferences/proxy").and_then(Value::as_str) {
            builder = builder.proxy(reqwest::Proxy::all(proxy).map_err(|_| bad("Invalid proxy"))?);
        }
        let client = builder
            .build()
            .map_err(|_| bad("Discovery client unavailable"))?;
        let models = crate::config::discover_provider_models(&client, &raw)
            .await
            .map_err(|_| bad("Model discovery failed; check address, engine and credentials"))?;
        Ok(json!({"models":models}))
    }
    pub(crate) async fn settings_export(&self) -> Result<Value, Failure> {
        let state = self.channel_controls.read().await;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        Ok(
            json!({"revision":state.revision(&base),"channel_settings":state.settings,"digest":digest(&state.settings),"temporary_definitions":state.settings_definitions(&base)}),
        )
    }
    pub(crate) async fn settings_operation(&self, id: &str) -> Result<Value, Failure> {
        self.channel_controls
            .read()
            .await
            .settings_operations
            .get(id)
            .map(|(_, v)| v.as_ref().clone())
            .ok_or((
                StatusCode::NOT_FOUND,
                "Operation not found in this instance; reconcile persisted intent".into(),
            ))
    }
    pub(crate) async fn settings_change(
        &self,
        input: Mutation,
        apply: bool,
    ) -> Result<Value, Failure> {
        if input.changes.is_empty()
            || input.changes.len() > 100
            || input.operation_id.is_empty()
            || input.operation_id.len() > 128
        {
            return Err(bad("Invalid settings operation"));
        }
        let hash = digest(&input.changes);
        let mut state = self.channel_controls.write().await;
        if apply {
            if let Some((old, result)) = state.settings_operations.get(&input.operation_id) {
                return if old == &hash {
                    Ok(result.as_ref().clone())
                } else {
                    Err((
                        StatusCode::CONFLICT,
                        "Operation ID already used for another change".into(),
                    ))
                };
            }
        }
        let base_guard = self.current.read().await;
        let base = base_guard.clone().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        if state.revision(&base) != input.revision {
            return Err((
                StatusCode::CONFLICT,
                "Configuration or controls changed; refresh before applying".into(),
            ));
        }
        let mut candidate = state.clone();
        candidate.sequence += 1;
        let mut previews = Vec::new();
        let mut seen = BTreeSet::new();
        for original_change in &input.changes {
            if !original_change.create_to_key.is_empty() {
                let name = &original_change.provider;
                if !original_change.copy_to_key.is_empty()
                    || original_change.delete_copy
                    || original_change.reset
                    || !original_change.remove.is_empty()
                    || !name.starts_with("typesafe-")
                    || name.len() > 100
                    || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
                    || base.providers_by_name.contains_key(name)
                    || candidate.temporary.contains_key(name)
                {
                    return Err(bad(
                        "Invalid channel setting: new TypeSafe channel identity",
                    ));
                }
                if candidate.temporary.len() >= 128 {
                    return Err(bad("Too many temporary channels"));
                }
                if !base.api_key_order.iter().any(|token| {
                    crate::channel_catalog::key_id(token) == original_change.create_to_key
                }) {
                    return Err(bad("Destination API key not found"));
                }
                validate_paths(original_change)?;
                let raw = merge(&json!({"provider":name}), &original_change.set, &[]);
                if raw["engine"] != "typesafe" {
                    return Err(bad(
                        "Invalid channel setting: new channel must use typesafe",
                    ));
                }
                let compiled = crate::config::compile_provider(&raw)
                    .ok_or(bad("Invalid channel setting: /model"))?;
                let parsed = serde_json::from_value(compiled)
                    .map_err(|_| bad("Invalid channel setting: definition"))?;
                let mut prototype =
                    (*crate::responses_native::runtime_provider(parsed, Arc::default())).clone();
                let mut prefs = (*prototype.preferences).clone();
                prefs.insert(
                    "__temporary_key_id".into(),
                    json!(original_change.create_to_key),
                );
                prototype.preferences = Arc::new(prefs);
                let built = compile(&raw, &Arc::new(prototype)).map_err(bad)?;
                if built.api_keys.is_empty() {
                    return Err(bad("Invalid channel setting: /api"));
                }
                candidate.temporary_documents.insert(name.clone(), raw);
                candidate.temporary.insert(name.clone(), built);
            }
            let mut copied_change;
            let c = if !original_change.copy_to_key.is_empty() {
                if !base
                    .api_key_order
                    .iter()
                    .any(|t| crate::channel_catalog::key_id(t) == original_change.copy_to_key)
                {
                    return Err(bad("Destination API key not found"));
                }
                if state.temporary.len() >= 128 {
                    return Err(bad("Too many temporary channels"));
                }
                let source = state
                    .overlay(base.clone())
                    .providers_by_name
                    .get(&original_change.provider)
                    .cloned()
                    .ok_or(bad("Copy source not found"))?;
                let name = format!(
                    "sub2api-copy-{}",
                    &digest(&format!(
                        "{}:{}:{}",
                        input.operation_id, original_change.provider, original_change.copy_to_key
                    ))[..20]
                );
                let source_raw = state
                    .temporary_documents
                    .get(&original_change.provider)
                    .cloned()
                    .unwrap_or_else(|| document(&base, &source));
                let source_settings = state
                    .settings
                    .get(&original_change.provider)
                    .cloned()
                    .unwrap_or_default();
                let mut raw = merge(&source_raw, &source_settings.set, &source_settings.remove);
                raw["provider"] = json!(name);
                let mut prototype = (*source).clone();
                prototype.name = name.clone().into();
                let mut prefs = (*prototype.preferences).clone();
                prefs.insert(
                    "__temporary_key_id".into(),
                    json!(original_change.copy_to_key),
                );
                prototype.preferences = Arc::new(prefs);
                let built = compile(&raw, &Arc::new(prototype)).map_err(bad)?;
                candidate.temporary_documents.insert(name.clone(), raw);
                candidate.temporary.insert(name.clone(), built);
                copied_change = original_change.clone();
                copied_change.provider = name;
                &copied_change
            } else {
                original_change
            };
            validate_paths(c)?;
            if !seen.insert(c.provider.clone()) {
                return Err(bad("Duplicate channel in batch"));
            }
            let old = candidate
                .temporary
                .get(&c.provider)
                .or_else(|| base.providers_by_name.get(&c.provider))
                .ok_or((StatusCode::NOT_FOUND, "Channel not found".into()))?;
            let raw = candidate
                .temporary_documents
                .get(&c.provider)
                .cloned()
                .unwrap_or_else(|| document(&base, old));
            let existing = state.settings.get(&c.provider).cloned().unwrap_or_default();
            let current = merge(&raw, &existing.set, &existing.remove);
            let mut refs = BTreeMap::new();
            secrets(&raw, "", &mut refs);
            secrets(&current, "", &mut refs);
            if c.delete_copy {
                if !candidate.remove_settings_copy(&c.provider) {
                    return Err(bad("Only console copies can be deleted here"));
                }
                previews.push(json!({"provider":c.provider,"before":redact_at(&current,""),"after":null,"deleted":true,"sample":null}));
                continue;
            }
            let start = if c.reset {
                raw.clone()
            } else {
                current.clone()
            };
            let resolved = c
                .set
                .iter()
                .map(|(p, v)| Ok((p.clone(), resolve_refs(v, &refs)?)))
                .collect::<Result<BTreeMap<_, _>, Failure>>()?;
            let merged = merge(&start, &resolved, &c.remove);
            let mut settings = ProviderSettings::default();
            collect_diff(&raw, &merged, "", &mut settings);
            let compiled = compile(&merged, old).map_err(bad)?;
            settings.compiled = Some(compiled.clone());
            if settings.set.is_empty() && settings.remove.is_empty() {
                candidate.settings.remove(&c.provider);
            } else {
                candidate.settings.insert(c.provider.clone(), settings);
            }
            let request_model = input
                .sample
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    compiled
                        .models
                        .keys()
                        .next()
                        .map(String::as_str)
                        .unwrap_or("")
                });
            let upstream = compiled
                .models
                .get(request_model)
                .map(String::as_str)
                .unwrap_or(request_model);
            let endpoint = input
                .sample
                .get("endpoint")
                .and_then(Value::as_str)
                .unwrap_or("/v1/responses");
            let stream = input
                .sample
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let timeout = crate::responses_native::resolve_timeouts(
                &base,
                &compiled,
                request_model,
                upstream,
                &compiled.engine,
                stream,
                input.sample.get("request_type").and_then(Value::as_str),
                "",
                endpoint,
                "POST",
            );
            let mut body=input.sample.get("body").cloned().unwrap_or(json!({"model":request_model,"input":"Preview","messages":[{"role":"user","content":"Preview"}]}));
            if let Some(o) = body.as_object_mut() {
                o.insert("model".into(), json!(request_model));
                if endpoint != "/v1/systemone" {
                    o.insert("stream".into(), json!(stream));
                }
            }
            let prepared = crate::generic_api::preview_channel_request(
                &compiled,
                request_model,
                upstream,
                endpoint,
                body.clone(),
            );
            let wire = match prepared {
                Ok(mut v) => {
                    v["body"] = redact_at(
                        &json!({"preferences":{"post_body_parameter_overrides":v["body"].clone()}}),
                        "",
                    )["preferences"]["post_body_parameter_overrides"]
                        .clone();
                    v
                }
                Err(_) => json!({"error":"该示例不能按当前协议编译，请检查模型、引擎和请求体"}),
            };
            if let Some(o) = body.as_object_mut() {
                o.insert("model".into(), json!(upstream));
                crate::responses_native::apply_overrides(o, &compiled, request_model);
            }
            // Payload may contain configured secrets. Only the field names leave
            // the gateway preview; unredacted headers/body never enter logs.
            previews.push(json!({"provider":c.provider,"created":!original_change.copy_to_key.is_empty() || !original_change.create_to_key.is_empty(),"before":redact_at(&current,""),"after":redact_at(&merged,""),"sample":{"model":request_model,"upstream_model":upstream,"endpoint":endpoint,"timeouts":timeout,"excluded_endpoint":compiled.excluded_endpoints.iter().any(|e|e==endpoint),"excluded_request_type":!crate::responses_native::provider_accepts_request_type(&compiled,input.sample.get("request_type").and_then(Value::as_str)),"engine":compiled.engine.as_ref(),"base_url":compiled.base_url.as_ref(),"wire_request":wire,"excluded_rule":!crate::responses_native::provider_accepts_request_rules(&compiled,endpoint,request_model,input.sample.get("reasoning_effort").and_then(Value::as_str),input.sample.get("request_type").and_then(Value::as_str)),"body":redact_at(&json!({"preferences":{"post_body_parameter_overrides":body}}),"")["preferences"]["post_body_parameter_overrides"]}}));
        }
        let effective = candidate.overlay(base.clone());
        for p in &mut previews {
            let name = p["provider"].as_str().unwrap().to_owned();
            p["previous_keys"] = json!(affected(&state.overlay(base.clone()), &name));
            p["affected_keys"] = json!(affected(&effective, &name));
        }
        let mut result = json!({"operation_id":input.operation_id,"status":if apply{"applied"}else{"validated"},"revision":if apply{candidate.revision(&base)}else{state.revision(&base)},"previews":previews,"settings_digest":digest(&candidate.settings)});
        result["intent"] = json!({"settings":candidate.settings,"temporary_definitions":candidate.settings_definitions(&base)});
        if apply {
            candidate
                .settings_operations
                .insert(input.operation_id, (hash, Arc::new(result.clone())));
            if candidate.settings_operations.len() > 256 {
                if let Some(k) = candidate.settings_operations.keys().next().cloned() {
                    candidate.settings_operations.remove(&k);
                }
            }
            *state = candidate;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_native::{ApiKey, RawProvider};
    use axum::http::HeaderMap;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;
    #[tokio::test]
    async fn typesafe_creation_is_scoped_atomic_restorable_and_reversible() {
        let store = fixture().await;
        let revision = store.settings_view("one").await.unwrap()["revision"]
            .as_str()
            .unwrap()
            .to_owned();
        let create = |revision: &str, operation: &str, key: &str| {
            serde_json::from_value::<Mutation>(json!({
            "revision":revision,"operation_id":operation,
            "changes":[{"provider":"typesafe-jev","create_to_key":key,"set":{
                "/engine":"typesafe","/base_url":"https://api.typesafe.ai/v1/systemone",
                "/api":"jev-secret","/model":["jev-latest","jev-1.13.0"]
            }}],
            "sample":{"endpoint":"/v1/systemone","stream":false,"body":{"state":["a",{"b":true}],"questions":{"q":{"type":"noul","instructions":"Is b true?"}}}}
        })).unwrap()
        };
        let owner = crate::channel_catalog::key_id("caller-a");
        assert!(store
            .settings_change(create(&revision, "invalid", "missing"), true)
            .await
            .is_err());
        assert_eq!(
            store.settings_view("one").await.unwrap()["revision"],
            revision
        );
        let preview = store
            .settings_change(create(&revision, "create", &owner), false)
            .await
            .unwrap();
        assert_eq!(preview["previews"][0]["created"], true);
        assert!(preview["previews"][0]["sample"]["wire_request"]["body"]
            .get("stream")
            .is_none());
        assert!(store.settings_view("typesafe-jev").await.is_err());
        let applied = store
            .settings_change(create(&revision, "create", &owner), true)
            .await
            .unwrap();
        assert_eq!(
            applied,
            store
                .settings_change(create(&revision, "create", &owner), true)
                .await
                .unwrap()
        );
        assert_eq!(
            applied["previews"][0]["affected_keys"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let snapshot = store.snapshot().await.unwrap();
        let provider = &snapshot.providers_by_name["typesafe-jev"];
        assert_eq!(provider.engine.as_ref(), "typesafe");
        assert!(crate::channel_controls::temporary_allowed(
            provider,
            &snapshot.api_keys["caller-a"]
        ));
        assert!(!crate::channel_controls::temporary_allowed(
            provider,
            &snapshot.api_keys["caller-b"]
        ));
        let exported = store.settings_export().await.unwrap();
        let restored = fixture().await;
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer admin-token".parse().unwrap());
        let restore_revision = restored.controls_view(&headers).await.unwrap()["revision"]
            .as_str()
            .unwrap()
            .to_owned();
        restored.restore_controls(&headers, serde_json::from_value(json!({
            "revision":restore_revision,"snapshot":{"version":2,"rules":[],"temporary_channels":[{
                "provider":"typesafe-jev","api_key_id":owner,"base_url":"https://api.typesafe.ai/v1/systemone",
                "api_key":"jev-secret","models":["jev-latest","jev-1.13.0"],
                "definition":exported["temporary_definitions"]["typesafe-jev"]
            }]}
        })).unwrap()).await.unwrap();
        assert_eq!(
            restored.snapshot().await.unwrap().providers_by_name["typesafe-jev"]
                .engine
                .as_ref(),
            "typesafe"
        );
        let next_revision = applied["revision"].as_str().unwrap();
        assert!(store
            .settings_change(create(next_revision, "duplicate", &owner), true)
            .await
            .is_err());
        store.settings_change(serde_json::from_value(json!({"revision":next_revision,"operation_id":"rollback","changes":[{"provider":"typesafe-jev","delete_copy":true}]})).unwrap(),true).await.unwrap();
        assert!(store.settings_view("typesafe-jev").await.is_err());
        assert_eq!(store.snapshot().await.unwrap().providers.len(), 1);
    }
    async fn fixture() -> NativeConfigStore {
        let raw = json!({"provider":"one","base_url":"https://example.com/v1/responses","engine":"gpt","api":["secret-a","secret-b"],"model":["public",{"upstream":"alias"},{"vendor/model":"vendor/public"}],"preferences":{"cooldown_period":30,"headers":{"x-private":"secret-header"}},"unknown_extension":{"keep":true}});
        let item: RawProvider =
            serde_json::from_value(crate::config::compile_provider(&raw).unwrap()).unwrap();
        let p = crate::responses_native::runtime_provider(item, Arc::new(AtomicUsize::new(0)));
        let keys = HashMap::from_iter(["admin-token", "caller-a", "caller-b"].into_iter().map(
            |token| {
                (
                    token.into(),
                    Arc::new(ApiKey {
                        token: token.into(),
                        model_rules: Arc::new(vec!["all".into()]),
                        role: if token == "admin-token" { "admin" } else { "" }.into(),
                        preferences: Arc::new(serde_json::Map::new()),
                        weights: Arc::new(serde_json::Map::new()),
                        native_supported: true,
                    }),
                )
            },
        ));
        let base = Snapshot {
            revision: "base-one".into(),
            preferences: Arc::new(serde_json::Map::new()),
            api_keys: Arc::new(keys),
            api_key_order: Arc::new(vec![
                "admin-token".into(),
                "caller-a".into(),
                "caller-b".into(),
            ]),
            providers: Arc::new(vec![p.clone()]),
            providers_by_name: Arc::new(HashMap::from([("one".into(), p)])),
            api_config: Arc::new(json!({"providers":[raw]})),
        };
        let store = NativeConfigStore::new();
        *store.current.write().await = Some(Arc::new(base));
        store
    }
    fn change(revision: &str, id: &str, set: Value) -> Mutation {
        serde_json::from_value(
            json!({"revision":revision,"operation_id":id,"changes":[{"provider":"one","set":set}]}),
        )
        .unwrap()
    }
    #[tokio::test]
    async fn edits_are_atomic_keep_secrets_and_existing_requests_and_check_revision() {
        let store = fixture().await;
        let view = store.settings_view("one").await.unwrap();
        let original = store.snapshot().await.unwrap();
        let rev = view["revision"].as_str().unwrap();
        assert_eq!(
            view["affected_keys"].as_array().unwrap().len(),
            3,
            "ordinary calling keys must appear in the impact preview"
        );
        assert!(!view.to_string().contains("secret-a"));
        assert!(!view.to_string().contains("secret-header"));
        let input = change(
            rev,
            "first",
            json!({"/preferences/cooldown_period":0,"/api":[view["effective"]["api"][1].clone(),view["effective"]["api"][0].clone()]}),
        );
        let preview = store.settings_change(input, false).await.unwrap();
        assert_eq!(preview["status"], "validated");
        assert_eq!(store.settings_view("one").await.unwrap()["revision"], rev);
        let applied = store
            .settings_change(
                change(rev, "first", json!({"/preferences/cooldown_period":0})),
                true,
            )
            .await
            .unwrap();
        assert_eq!(original.providers[0].preferences["cooldown_period"], 30);
        let next = store.snapshot().await.unwrap();
        assert_eq!(next.providers[0].preferences["cooldown_period"], 0);
        assert_eq!(next.providers[0].api_keys[0], "secret-a");
        assert!(Arc::ptr_eq(
            &original.providers[0].cursor,
            &next.providers[0].cursor
        ));
        let again = store
            .settings_change(
                change(rev, "first", json!({"/preferences/cooldown_period":0})),
                true,
            )
            .await
            .unwrap();
        assert_eq!(applied, again);
        assert_eq!(
            store
                .settings_change(change(rev, "stale", json!({"/tools":false})), true)
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
        let view = store.settings_view("one").await.unwrap();
        assert_eq!(view["effective"]["unknown_extension"]["keep"], true);
    }
    #[tokio::test]
    async fn invalid_batch_cannot_partially_apply_and_nested_removal_is_preserved() {
        let store = fixture().await;
        let view = store.settings_view("one").await.unwrap();
        let rev = view["revision"].as_str().unwrap();
        let mut input = change(rev, "bad", json!({"/tools":false}));
        input.changes.push(Change {
            provider: "missing".into(),
            set: BTreeMap::new(),
            remove: vec![],
            reset: false,
            copy_to_key: String::new(),
            delete_copy: false,
            create_to_key: String::new(),
        });
        assert!(store.settings_change(input, true).await.is_err());
        assert_eq!(store.settings_view("one").await.unwrap()["revision"], rev);
        let a=store.settings_change(change(rev,"object",json!({"/preferences/post_body_parameter_overrides":{"temperature":1,"max_output_tokens":50}})),true).await.unwrap();
        let mut input = change(a["revision"].as_str().unwrap(), "remove", json!({}));
        input.changes[0].remove =
            vec!["/preferences/post_body_parameter_overrides/temperature".into()];
        store.settings_change(input, true).await.unwrap();
        let next = store.snapshot().await.unwrap();
        let body = &next.providers[0].preferences["post_body_parameter_overrides"];
        assert!(body.get("temperature").is_none());
        assert_eq!(body["max_output_tokens"], 50);
    }
    #[tokio::test]
    async fn copy_is_only_available_to_destination_key_and_restores_full_definition() {
        let store = fixture().await;
        let view = store.settings_view("one").await.unwrap();
        let mut input = change(
            view["revision"].as_str().unwrap(),
            "copy",
            json!({"/preferences/model_timeout":{"default":125}}),
        );
        input.changes[0].copy_to_key = crate::channel_catalog::key_id("caller-a");
        let result = store.settings_change(input, true).await.unwrap();
        assert_eq!(
            result["previews"][0]["affected_keys"]
                .as_array()
                .unwrap()
                .len(),
            1,
            "a copy must only affect its destination caller"
        );
        let name = result["previews"][0]["provider"].as_str().unwrap();
        let snapshot = store.snapshot().await.unwrap();
        let provider = &snapshot.providers_by_name[name];
        assert_eq!(provider.api_keys.len(), 2);
        assert_eq!(provider.preferences["model_timeout"]["default"], 125);
        assert!(crate::channel_controls::temporary_allowed(
            provider,
            &snapshot.api_keys["caller-a"]
        ));
        assert!(!crate::channel_controls::temporary_allowed(
            provider,
            &snapshot.api_keys["caller-b"]
        ));
        let exported = store.settings_export().await.unwrap();
        let settings: BTreeMap<String, ProviderSettings> =
            serde_json::from_value(exported["channel_settings"].clone()).unwrap();
        let raw = exported["temporary_definitions"][name].clone();
        let restored = fixture().await;
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer admin-token".parse().unwrap());
        let view = restored.controls_view(&headers).await.unwrap();
        restored
            .restore_controls(
                &headers,
                crate::channel_controls::RestoreMutation {
                    revision: view["revision"].as_str().unwrap().into(),
                    snapshot: crate::channel_controls::RetainedSnapshot {
                        version: 2,
                        rules: vec![],
                        channel_settings: settings,
                        temporary_channels: vec![crate::channel_controls::RetainedChannel {
                            provider: name.into(),
                            api_key_id: crate::channel_catalog::key_id("caller-a"),
                            base_url: raw["base_url"].as_str().unwrap().into(),
                            api_key: "secret-a".into(),
                            models: vec!["public".into(), "alias".into(), "vendor/public".into()],
                            definition: Some(raw),
                        }],
                    },
                },
            )
            .await
            .unwrap();
        let snap = restored.snapshot().await.unwrap();
        assert_eq!(snap.providers_by_name[name].api_keys.len(), 2);
        assert_eq!(snap.providers_by_name[name].models["alias"], "upstream");
        assert_eq!(
            snap.providers_by_name[name].preferences["model_timeout"]["default"],
            125
        );
    }
    #[test]
    fn checks_invalid_values_and_preserves_false_zero() {
        let raw = json!({"provider":"one","base_url":"https://example.com/v1/responses","api":"secret-a","model":["one"]});
        let p = crate::responses_native::runtime_provider(
            serde_json::from_value(crate::config::compile_provider(&raw).unwrap()).unwrap(),
            Arc::new(AtomicUsize::new(0)),
        );
        for patch in [
            json!({"/model":[]}),
            json!({"/preferences/model_timeout":{"default":"bad"}}),
            json!({"/preferences/headers":{"x":"a\nb"}}),
            json!({"/preferences/api_key_schedule_algorithm":"imaginary"}),
        ] {
            let set = serde_json::from_value(patch).unwrap();
            assert!(compile(&merge(&raw, &set, &[]), &p).is_err())
        }
        assert!(compile(
            &merge(
                &raw,
                &BTreeMap::from([
                    ("/tools".into(), json!(false)),
                    ("/preferences/cooldown_period".into(), json!(0))
                ]),
                &[]
            ),
            &p
        )
        .is_ok());
    }
    #[tokio::test]
    async fn protocols_preview_inherit_models_and_delete_copy_atomically() {
        let store = fixture().await;
        for (i, (engine, url, endpoint, payload, field)) in [
            (
                "gpt",
                "https://example.com/v1/responses",
                "/v1/responses",
                json!({"input":"Hi"}),
                "input",
            ),
            (
                "claude",
                "https://example.com/v1/messages",
                "/v1/chat/completions",
                json!({"messages":[{"role":"user","content":"Hi"}]}),
                "messages",
            ),
            (
                "gemini",
                "https://example.com/v1beta",
                "/v1/chat/completions",
                json!({"messages":[{"role":"user","content":"Hi"}]}),
                "contents",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let view = store.settings_view("one").await.unwrap();
            let mut input = change(
                view["revision"].as_str().unwrap(),
                &format!("proto-{i}"),
                json!({"/engine":engine,"/base_url":url,"/preferences/model_timeout":{"default":123},"/preferences/post_body_parameter_overrides":{"temperature":0.2}}),
            );
            input.sample =
                json!({"model":"alias","endpoint":endpoint,"stream":true,"body":payload});
            let result = store.settings_change(input, true).await.unwrap();
            let wire = &result["previews"][0]["sample"]["wire_request"];
            assert!(wire.get("error").is_none(), "{wire}");
            assert!(wire["body"].get(field).is_some());
            assert_eq!(wire["body"]["temperature"], 0.2);
            assert_eq!(
                result["previews"][0]["sample"]["timeouts"]["first_byte"],
                123.0
            );
            assert!(!result["previews"].to_string().contains("secret-a"));
        }
        let view = store.settings_view("one").await.unwrap();
        let mut input = change(view["revision"].as_str().unwrap(), "copy-delete", json!({}));
        input.changes[0].copy_to_key = crate::channel_catalog::key_id("caller-a");
        let result = store.settings_change(input, true).await.unwrap();
        let name = result["previews"][0]["provider"].as_str().unwrap();
        let input:Mutation=serde_json::from_value(json!({"revision":result["revision"],"operation_id":"delete-copy","changes":[{"provider":name,"delete_copy":true}]})).unwrap();
        store.settings_change(input, true).await.unwrap();
        let snapshot = store.snapshot().await.unwrap();
        assert!(!snapshot.providers_by_name.contains_key(name));
        assert_eq!(snapshot.providers.len(), 1);
    }
    #[test]
    fn canonical_digest_uses_sorted_object_keys_for_retention_interop() {
        assert_eq!(digest(&json!({"a":1.0})), digest(&json!({"a":1})));
        let value = ProviderSettings {
            set: BTreeMap::from([("/api".into(), json!("<secret>"))]),
            ..ProviderSettings::default()
        };
        assert_eq!(
            digest(&value),
            digest(&json!({"remove":[],"set":{"/api":"<secret>"}}))
        );
    }
    #[tokio::test]
    async fn reveal_api_refs_is_read_only_scoped_and_revision_checked() {
        let store = fixture().await;
        let before = store.settings_view("one").await.unwrap();
        let revision = before["revision"].as_str().unwrap();
        let result = store.settings_secrets("one", revision).await.unwrap();
        assert_eq!(result["keys"].as_object().unwrap().len(), 2);
        for (i, expected) in ["secret-a", "secret-b"].iter().enumerate() {
            assert_eq!(
                result["keys"][before["effective"]["api"][i]["$secret"].as_str().unwrap()],
                *expected
            );
        }
        assert!(!result.to_string().contains("secret-header"));
        assert_eq!(store.settings_view("one").await.unwrap(), before);
        assert_eq!(
            store
                .settings_secrets("missing", revision)
                .await
                .unwrap_err()
                .0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            store.settings_secrets("one", "").await.unwrap_err().0,
            StatusCode::CONFLICT
        );
        let applied = store
            .settings_change(
                change(revision, "new-key", json!({"/api":"replacement-key"})),
                true,
            )
            .await
            .unwrap();
        assert_eq!(
            store.settings_secrets("one", revision).await.unwrap_err().0,
            StatusCode::CONFLICT
        );
        let current = store.settings_view("one").await.unwrap();
        let result = store
            .settings_secrets("one", applied["revision"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            result["keys"][current["effective"]["api"]["$secret"].as_str().unwrap()],
            "replacement-key"
        );
        assert_eq!(
            result["keys"][before["base"]["api"][0]["$secret"].as_str().unwrap()],
            "secret-a"
        );
        assert!(!current.to_string().contains("replacement-key"));
        let operation = store.settings_operation("new-key").await.unwrap();
        assert!(!operation["previews"]
            .to_string()
            .contains("replacement-key"));
    }
}
