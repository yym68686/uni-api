//! Process-local routing intent. Never written to configuration or persistent
//! storage; a new NativeConfigStore starts with no overrides and a new revision.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::channel_catalog::entries;
use crate::responses_native::{NativeConfigStore, Provider, Snapshot};

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Rule {
    #[serde(default)]
    pub api_key_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub order: Vec<String>,
    #[serde(default)]
    pub disabled: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Mutation {
    pub revision: String,
    pub action: String,
    #[serde(default)]
    pub api_key_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub order: Vec<String>,
    #[serde(default)]
    pub disabled: Vec<String>,
}
type OverlayCache = Arc<Mutex<Option<(String, u64, Arc<Snapshot>)>>>;

#[derive(Clone)]
pub(crate) struct Controls {
    instance: String,
    sequence: u64,
    rules: BTreeMap<(String, String), Rule>,
    temporary: BTreeMap<String, Arc<Provider>>,
    overlay_cache: OverlayCache,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            instance: format!(
                "{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            sequence: 0,
            rules: BTreeMap::new(),
            temporary: BTreeMap::new(),
            overlay_cache: Arc::new(Mutex::new(None)),
        }
    }
}
impl Controls {
    fn revision(&self, snapshot: &Snapshot) -> String {
        format!("{}:{}:{}", self.instance, self.sequence, snapshot.revision)
    }
    fn view(&self, snapshot: &Snapshot) -> Value {
        json!({"revision":self.revision(snapshot),"instance_id":self.instance,"config_revision":snapshot.revision.as_ref(),"storage":"process_memory","temporary_channel_import":true,"temporary_channel_management":true,"reset_on_restart":true,"expires_at":null,"rules":self.rules.values().collect::<Vec<_>>(),"temporary_channels":self.temporary.values().map(|p|json!({"provider":p.name.as_ref(),"api_key_id":p.preferences.get("__temporary_key_id"),"models":p.models.keys().collect::<BTreeSet<_>>()})).collect::<Vec<_>>()})
    }
    pub fn overlay(&self, base: Arc<Snapshot>) -> Arc<Snapshot> {
        if self.temporary.is_empty() {
            return base;
        }
        let mut cache = self.overlay_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((revision, sequence, snapshot)) = cache.as_ref() {
            if revision == base.revision.as_ref() && *sequence == self.sequence {
                return snapshot.clone();
            }
        }
        let mut snapshot = (*base).clone();
        let mut providers = (*base.providers).clone();
        let mut by_name = (*base.providers_by_name).clone();
        let mut keys = (*base.api_keys).clone();
        for provider in self.temporary.values() {
            if by_name.contains_key(provider.name.as_ref()) {
                continue;
            }
            let owner = provider
                .preferences
                .get("__temporary_key_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let Some(token) = base
                .api_key_order
                .iter()
                .find(|token| crate::channel_catalog::key_id(token) == owner)
            else {
                continue;
            };
            let Some(key) = keys.get(token) else { continue };
            let mut key = (**key).clone();
            let rule = format!("{}/*", provider.name);
            let mut rules = (*key.model_rules).clone();
            rules.push(rule.clone());
            key.model_rules = Arc::new(rules);
            let mut preferences = (*key.preferences).clone();
            if let Some(graph) = preferences
                .get_mut("__route_graph")
                .and_then(Value::as_array_mut)
            {
                graph.push(Value::String(rule));
            }
            key.preferences = Arc::new(preferences);
            keys.insert(token.clone(), Arc::new(key));
            providers.push(provider.clone());
            by_name.insert(provider.name.to_string(), provider.clone());
        }
        snapshot.providers = Arc::new(providers);
        snapshot.providers_by_name = Arc::new(by_name);
        snapshot.api_keys = Arc::new(keys);
        let snapshot = Arc::new(snapshot);
        *cache = Some((base.revision.to_string(), self.sequence, snapshot.clone()));
        snapshot
    }
    fn reset_temporary(&mut self, key: &str, model: &str) {
        self.temporary.retain(|_, provider| {
            let owner = provider
                .preferences
                .get("__temporary_key_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !key.is_empty() && owner != key {
                return true;
            }
            if model.is_empty() {
                return false;
            }
            let mut p = (**provider).clone();
            let mut models = (*p.models).clone();
            models.remove(model);
            p.models = Arc::new(models);
            *provider = Arc::new(p);
            !provider.models.is_empty()
        });
    }
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
    fn applicable<'a>(&'a self, key: &'a str, model: &'a str) -> impl Iterator<Item = &'a Rule> {
        self.rules.values().filter(move |r| {
            (r.api_key_id.is_empty() || r.api_key_id == key)
                && (r.model.is_empty() || r.model == model)
        })
    }
    pub fn disabled(&self, key: &str, model: &str, provider: &str) -> bool {
        self.applicable(key, model)
            .any(|r| r.disabled.iter().any(|p| p == provider))
    }
    // More specific scope wins for ordering: key+model > key > model > global.
    pub fn order(&self, key: &str, model: &str) -> Option<&[String]> {
        [(key, model), (key, ""), ("", model), ("", "")]
            .into_iter()
            .filter_map(|(k, m)| self.rules.get(&(k.to_owned(), m.to_owned())))
            .find(|r| !r.order.is_empty())
            .map(|r| r.order.as_slice())
    }
    pub fn apply(
        &self,
        key: &str,
        model: &str,
        mut providers: Vec<Arc<Provider>>,
    ) -> Vec<Arc<Provider>> {
        providers.retain(|p| !self.disabled(key, model, p.name.as_ref()));
        if let Some(order) = self.order(key, model) {
            let ranks: HashMap<_, _> = order
                .iter()
                .enumerate()
                .map(|(i, p)| (p.as_str(), i))
                .collect();
            let mut seen = BTreeSet::new();
            providers.retain(|p| seen.insert(p.name.to_string()));
            providers.sort_by_key(|p| ranks.get(p.name.as_ref()).copied().unwrap_or(usize::MAX));
        }
        providers
    }
}
impl NativeConfigStore {
    pub(crate) async fn controls_view(&self, headers: &HeaderMap) -> Result<Value, u16> {
        self.authorize_catalog(headers).await?;
        let snapshot = self.snapshot().await.ok_or(503u16)?;
        Ok(self.channel_controls.read().await.view(&snapshot))
    }
    pub(crate) async fn mutate_controls(
        &self,
        headers: &HeaderMap,
        input: Mutation,
    ) -> Result<Value, (StatusCode, String)> {
        self.authorize_catalog(headers).await.map_err(|status| {
            (
                StatusCode::from_u16(status).unwrap_or(StatusCode::FORBIDDEN),
                "Platform administrator key required".into(),
            )
        })?;
        let snapshot = self.snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let bad = |text: &str| (StatusCode::BAD_REQUEST, text.to_owned());
        if !["set", "reset", "reset_all"].contains(&input.action.as_str()) {
            return Err(bad("Invalid action"));
        }
        if input.order.len() > 1024
            || input.disabled.len() > 1024
            || input.model.len() > 512
            || input.api_key_id.len() > 128
        {
            return Err(bad("Control scope too large"));
        }
        if input.action == "set" {
            let caller = snapshot
                .api_keys
                .values()
                .find(|k| crate::channel_catalog::can_inspect_all(&snapshot, k))
                .ok_or(bad("Administrator unavailable"))?;
            let allowed = entries(&snapshot, caller, Some(&input.api_key_id))
                .map_err(|_| bad("Selected API key does not exist"))?;
            let names: BTreeSet<_> = allowed
                .into_iter()
                .filter(|(_, m)| input.model.is_empty() || m == &input.model)
                .map(|(p, _)| p.name.to_string())
                .collect();
            if names.is_empty() {
                return Err(bad("No channels in selected scope"));
            }
            for list in [&input.order, &input.disabled] {
                if list.iter().collect::<BTreeSet<_>>().len() != list.len()
                    || list.iter().any(|p| !names.contains(p))
                {
                    return Err(bad("Unknown, duplicate or unauthorized channel in scope"));
                }
            }
        }
        let mut state = self.channel_controls.write().await;
        if input.revision != state.revision(&snapshot) {
            return Err((
                StatusCode::CONFLICT,
                "Controls or configuration changed, or instance restarted; refresh before applying"
                    .into(),
            ));
        }
        let scope = (input.api_key_id.clone(), input.model.clone());
        let audit = json!({"order":input.order,"disabled":input.disabled});
        match input.action.as_str() {
            "reset_all" => {
                state.rules.clear();
                state.temporary.clear();
            }
            "reset" => {
                state.rules.remove(&scope);
                state.reset_temporary(&input.api_key_id, &input.model);
            }
            _ => {
                if input.order.is_empty() && input.disabled.is_empty() {
                    state.rules.remove(&scope);
                } else {
                    if state.rules.len() >= 128 && !state.rules.contains_key(&scope) {
                        return Err(bad("Too many temporary scopes"));
                    }
                    state.rules.insert(
                        scope,
                        Rule {
                            api_key_id: input.api_key_id.clone(),
                            model: input.model.clone(),
                            order: input.order,
                            disabled: input.disabled,
                        },
                    );
                }
            }
        }
        state.sequence += 1;
        eprintln!(
            "{}",
            json!({"event_type":"channel_controls_changed","action":input.action,"api_key_id":input.api_key_id,"model":input.model,"revision":state.revision(&snapshot),"rule_count":state.rules.len(),"change":audit})
        );
        Ok(state.view(&snapshot))
    }
}

// A temporary provider is authorized for exactly one destination API key.
// Wildcard and nested-key expansion must not widen that scope.
pub(crate) fn temporary_allowed(
    provider: &Provider,
    key: &crate::responses_native::ApiKey,
) -> bool {
    provider
        .preferences
        .get("__temporary_key_id")
        .and_then(Value::as_str)
        .is_none_or(|owner| {
            owner == crate::channel_catalog::key_id(&key.token)
                || key
                    .preferences
                    .get("__diagnostic_provider")
                    .and_then(Value::as_str)
                    == Some(provider.name.as_ref())
        })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportMutation {
    pub revision: String,
    pub api_key_id: String,
    pub provider: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub position: usize,
}
impl NativeConfigStore {
    pub(crate) async fn import_temporary_channel(
        &self,
        headers: &HeaderMap,
        input: ImportMutation,
    ) -> Result<Value, (StatusCode, String)> {
        self.authorize_catalog(headers).await.map_err(|s| {
            (
                StatusCode::from_u16(s).unwrap_or(StatusCode::FORBIDDEN),
                "Platform administrator key required".into(),
            )
        })?;
        let snapshot = self.snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let bad = |m: &str| (StatusCode::BAD_REQUEST, m.to_string());
        if input.action == "delete" || input.action == "replace" {
            return self.manage_temporary_channel(snapshot, input).await;
        }
        if !input.action.is_empty() {
            return Err(bad("Invalid action"));
        }

        if !input.provider.starts_with("sub2api-")
            || input.provider.len() > 100
            || !input
                .provider
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-')
            || input.api_key.is_empty()
            || input.api_key.len() > 8192
            || input.api_key.contains(['\r', '\n'])
            || input.models.is_empty()
            || input.models.len() > 32
            || input.position == 0
            || input.position > 1025
        {
            return Err(bad("Invalid temporary channel"));
        }
        if input
            .models
            .iter()
            .any(|m| m.is_empty() || m.len() > 256 || m.contains('/') || m.contains(['\r', '\n']))
            || input.models.iter().collect::<BTreeSet<_>>().len() != input.models.len()
        {
            return Err(bad("Invalid models"));
        }
        let url = url::Url::parse(&input.base_url).map_err(|_| bad("Invalid upstream address"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !url.path().ends_with("/v1/responses")
        {
            return Err(bad("Invalid Responses address"));
        }
        let selected = snapshot
            .api_keys
            .values()
            .find(|k| crate::channel_catalog::key_id(&k.token) == input.api_key_id)
            .ok_or(bad("Selected API key does not exist"))?;
        let caller = snapshot
            .api_keys
            .values()
            .find(|k| crate::channel_catalog::can_inspect_all(&snapshot, k))
            .ok_or(bad("Administrator unavailable"))?;
        let entries = entries(&snapshot, caller, Some(&input.api_key_id))
            .map_err(|_| bad("Selected key unavailable"))?;
        let mut state = self.channel_controls.write().await;
        if input.revision != state.revision(&snapshot) {
            return Err((
                StatusCode::CONFLICT,
                "Controls changed; refresh before adding".into(),
            ));
        }
        if let Some(existing) = snapshot.providers_by_name.get(&input.provider) {
            if !state.temporary.contains_key(&input.provider)
                || existing
                    .preferences
                    .get("__temporary_key_id")
                    .and_then(Value::as_str)
                    != Some(input.api_key_id.as_str())
            {
                return Err(bad("Provider name already in use"));
            }
        }
        if state.temporary.len() >= 128 && !state.temporary.contains_key(&input.provider) {
            return Err(bad("Too many temporary channels"));
        }
        let new_scopes = input
            .models
            .iter()
            .filter(|m| {
                !state
                    .rules
                    .contains_key(&(input.api_key_id.clone(), (*m).clone()))
            })
            .count();
        if state.rules.len() + new_scopes > 128 {
            return Err(bad("Too many temporary scopes"));
        }
        // Keep other already imported models when adding another tested subset.
        let mut models = state
            .temporary
            .get(&input.provider)
            .map(|p| (*p.models).clone())
            .unwrap_or_default();
        for model in &input.models {
            models.insert(model.clone(), model.clone());
        }
        let provider = Arc::new(Provider {
            name: input.provider.clone().into(),
            base_url: input.base_url.into(),
            engine: "gpt".into(),
            api_keys: Arc::new(vec![input.api_key]),
            project_id: None,
            private_key: None,
            client_email: None,
            aws_access_key: None,
            aws_secret_key: None,
            aws_session_token: None,
            cf_account_id: None,
            region: "global".into(),
            models: Arc::new(models),
            preferences: Arc::new(serde_json::Map::from_iter([(
                "__temporary_key_id".into(),
                json!(input.api_key_id),
            )])),
            excluded_endpoints: Arc::new(Vec::new()),
            only_request_types: Arc::new(Vec::new()),
            excluded_request_types: Arc::new(Vec::new()),
            excluded_request_rules: Arc::new(Vec::new()),
            cursor: Arc::new(AtomicUsize::new(0)),
        });
        let mut orders = Vec::new();
        for model in &input.models {
            let mut names = entries
                .iter()
                .filter(|(p, m)| m == model && p.name.as_ref() != input.provider)
                .map(|(p, _)| p.name.to_string())
                .collect::<Vec<_>>();
            if let Some(order) = state.order(&input.api_key_id, model) {
                let ranks: HashMap<_, _> = order.iter().enumerate().map(|(i, n)| (n, i)).collect();
                names.sort_by_key(|n| ranks.get(n).copied().unwrap_or(usize::MAX));
            }
            if input.position > names.len() + 1 {
                return Err(bad("Position exceeds this model's channel count"));
            }
            names.insert(input.position - 1, input.provider.clone());
            orders.push((model.clone(), names));
        }
        let _ = selected;
        state.temporary.insert(input.provider.clone(), provider);
        for (model, order) in orders {
            let rule = state
                .rules
                .entry((input.api_key_id.clone(), model.clone()))
                .or_insert_with(|| Rule {
                    api_key_id: input.api_key_id.clone(),
                    model,
                    ..Rule::default()
                });
            rule.order = order;
            rule.disabled.retain(|p| p != &input.provider);
        }
        state.sequence += 1;
        eprintln!(
            "{}",
            json!({"event_type":"temporary_channel_added","provider":input.provider,"api_key_id":input.api_key_id,"models":input.models,"position":input.position})
        );
        Ok(state.view(&snapshot))
    }
}

impl NativeConfigStore {
    async fn manage_temporary_channel(
        &self,
        snapshot: Arc<Snapshot>,
        input: ImportMutation,
    ) -> Result<Value, (StatusCode, String)> {
        let bad = |m: &str| (StatusCode::BAD_REQUEST, m.to_string());
        let mut state = self.channel_controls.write().await;
        if input.revision != state.revision(&snapshot) {
            return Err((
                StatusCode::CONFLICT,
                "Controls changed; refresh before editing".into(),
            ));
        }
        let existing = state.temporary.get(&input.provider).cloned().ok_or((
            StatusCode::NOT_FOUND,
            "Temporary channel no longer exists".into(),
        ))?;
        if existing
            .preferences
            .get("__temporary_key_id")
            .and_then(Value::as_str)
            != Some(input.api_key_id.as_str())
        {
            return Err(bad("Temporary channel belongs to another key"));
        }
        let mut orders = Vec::new();
        if input.action == "replace" {
            if input.models.is_empty()
                || input.models.len() > 32
                || input.position == 0
                || input.position > 1025
                || input.models.iter().any(|m| {
                    m.is_empty() || m.len() > 256 || m.contains('/') || m.contains(['\r', '\n'])
                })
                || input.models.iter().collect::<BTreeSet<_>>().len() != input.models.len()
            {
                return Err(bad("Invalid models or position"));
            }
            let caller = snapshot
                .api_keys
                .values()
                .find(|k| crate::channel_catalog::can_inspect_all(&snapshot, k))
                .ok_or(bad("Administrator unavailable"))?;
            let available = entries(&snapshot, caller, Some(&input.api_key_id))
                .map_err(|_| bad("Selected key unavailable"))?;
            for model in &input.models {
                let mut names = available
                    .iter()
                    .filter(|(p, m)| m == model && p.name.as_ref() != input.provider)
                    .map(|(p, _)| p.name.to_string())
                    .collect::<Vec<_>>();
                if let Some(order) = state.order(&input.api_key_id, model) {
                    let ranks: HashMap<_, _> =
                        order.iter().enumerate().map(|(i, n)| (n, i)).collect();
                    names.sort_by_key(|n| ranks.get(n).copied().unwrap_or(usize::MAX));
                }
                if input.position > names.len() + 1 {
                    return Err(bad("Position exceeds model channel count"));
                }
                names.insert(input.position - 1, input.provider.clone());
                orders.push((model.clone(), names));
            }
            let new_scopes = input
                .models
                .iter()
                .filter(|m| {
                    !state
                        .rules
                        .contains_key(&(input.api_key_id.clone(), (*m).clone()))
                })
                .count();
            if state.rules.len() + new_scopes > 128 {
                return Err(bad("Too many temporary scopes"));
            }
        }
        // Remove only this provider from affected orders/disable lists. Preserve
        // other providers and their settings, even in the same key/model scope.
        let removed: BTreeSet<_> = existing
            .models
            .keys()
            .filter(|m| input.action == "delete" || !input.models.contains(m))
            .cloned()
            .collect();
        state.rules.retain(|_, rule| {
            if input.action == "delete" || (!rule.model.is_empty() && removed.contains(&rule.model))
            {
                rule.order.retain(|p| p != &input.provider);
                rule.disabled.retain(|p| p != &input.provider);
            }
            !rule.order.is_empty() || !rule.disabled.is_empty()
        });
        if input.action == "delete" {
            state.temporary.remove(&input.provider);
        } else {
            let mut provider = (*existing).clone();
            provider.models = Arc::new(
                input
                    .models
                    .iter()
                    .map(|m| (m.clone(), m.clone()))
                    .collect(),
            );
            state
                .temporary
                .insert(input.provider.clone(), Arc::new(provider));
            for (model, order) in orders {
                let rule = state
                    .rules
                    .entry((input.api_key_id.clone(), model.clone()))
                    .or_insert_with(|| Rule {
                        api_key_id: input.api_key_id.clone(),
                        model,
                        ..Rule::default()
                    });
                rule.order = order;
            }
        }
        state.sequence += 1;
        eprintln!(
            "{}",
            json!({"event_type":"temporary_channel_managed","action":input.action,"provider":input.provider,"api_key_id":input.api_key_id,"models":input.models})
        );
        Ok(state.view(&snapshot))
    }
}
