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
struct BootstrapReceipt {
    snapshot_id: String,
    applied_revision: String,
}

pub(crate) struct Controls {
    instance: String,
    pub(crate) sequence: u64,
    rules: BTreeMap<(String, String), Rule>,
    pub(crate) temporary: BTreeMap<String, Arc<Provider>>,
    overlay_cache: OverlayCache,
    pub(crate) settings: BTreeMap<String, crate::channel_settings::ProviderSettings>,
    pub(crate) temporary_documents: BTreeMap<String, Value>,
    pub(crate) settings_operations: BTreeMap<String, (String, Arc<Value>)>,
    bootstrap_restore: Option<BootstrapReceipt>,
}
// A validation candidate must never poison the serving snapshot cache.
impl Clone for Controls {
    fn clone(&self) -> Self {
        Self {
            instance: self.instance.clone(),
            sequence: self.sequence,
            rules: self.rules.clone(),
            temporary: self.temporary.clone(),
            settings: self.settings.clone(),
            temporary_documents: self.temporary_documents.clone(),
            settings_operations: self.settings_operations.clone(),
            bootstrap_restore: self.bootstrap_restore.clone(),
            overlay_cache: Arc::new(Mutex::new(None)),
        }
    }
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
            settings: BTreeMap::new(),
            temporary_documents: BTreeMap::new(),
            settings_operations: BTreeMap::new(),
            bootstrap_restore: None,
            rules: BTreeMap::new(),
            temporary: BTreeMap::new(),
            overlay_cache: Arc::new(Mutex::new(None)),
        }
    }
}
impl Controls {
    // Scheduling only needs routing rules, never full provider intent or audit
    // payloads. Keep request-path work independent of settings history size.
    pub(crate) fn routing_rules(&self) -> Self {
        Self {
            instance: self.instance.clone(),
            sequence: self.sequence,
            rules: self.rules.clone(),
            temporary: BTreeMap::new(),
            settings: BTreeMap::new(),
            temporary_documents: BTreeMap::new(),
            settings_operations: BTreeMap::new(),
            overlay_cache: self.overlay_cache.clone(),
            bootstrap_restore: None,
        }
    }
    pub(crate) fn revision(&self, snapshot: &Snapshot) -> String {
        format!("{}:{}:{}", self.instance, self.sequence, snapshot.revision)
    }
    pub(crate) fn settings_definitions(&self, base: &Snapshot) -> BTreeMap<String, Value> {
        let mut definitions = self.temporary_documents.clone();
        for name in self.settings.keys() {
            if let Some(p) = self.temporary.get(name) {
                definitions
                    .entry(name.clone())
                    .or_insert_with(|| crate::channel_settings::document(base, p));
            }
        }
        definitions
    }
    fn effective_temporary(&self, p: &Arc<Provider>, base: &Snapshot) -> Arc<Provider> {
        if let Some(setting) = self.settings.get(p.name.as_ref()) {
            let raw = self
                .temporary_documents
                .get(p.name.as_ref())
                .cloned()
                .unwrap_or_else(|| crate::channel_settings::document(base, p));
            crate::channel_settings::compile(
                &crate::channel_settings::merge(&raw, &setting.set, &setting.remove),
                p,
            )
            .unwrap_or_else(|_| setting.compiled.clone().unwrap_or_else(|| p.clone()))
        } else {
            p.clone()
        }
    }
    pub(crate) fn view(&self, snapshot: &Snapshot) -> Value {
        let bootstrap = self.bootstrap_restore.as_ref().map(|b| {
            json!({"snapshot_id":b.snapshot_id,"applied_revision":b.applied_revision,"unchanged":b.applied_revision == self.revision(snapshot)})
        });
        json!({"revision":self.revision(snapshot),"instance_id":self.instance,"config_revision":snapshot.revision.as_ref(),"bootstrap_restore":bootstrap,"storage":"process_memory","channel_definitions":!self.temporary_documents.is_empty(),"channel_definitions_digest":crate::channel_settings::digest(&self.settings_definitions(snapshot)),"channel_settings":true,"channel_settings_digest":crate::channel_settings::digest(&self.settings),"temporary_channel_import":true,"temporary_channel_management":true,"temporary_channel_restore":true,"reset_on_restart":true,"expires_at":null,"rules":self.rules.values().collect::<Vec<_>>(),"temporary_channels":self.temporary.values().map(|p|json!({"provider":p.name.as_ref(),"identity_changed":self.settings.get(p.name.as_ref()).is_some_and(crate::channel_settings::identity_changed),"api_key_id":p.preferences.get("__temporary_key_id"),"models":self.effective_temporary(p,snapshot).models.keys().cloned().collect::<BTreeSet<_>>()})).collect::<Vec<_>>()})
    }
    pub fn overlay(&self, base: Arc<Snapshot>) -> Arc<Snapshot> {
        if self.temporary.is_empty() && self.settings.is_empty() {
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
        for provider in &mut providers {
            if let Some(settings) = self.settings.get(provider.name.as_ref()) {
                let raw = self
                    .temporary_documents
                    .get(provider.name.as_ref())
                    .cloned()
                    .unwrap_or_else(|| crate::channel_settings::document(&base, provider));
                let effective =
                    crate::channel_settings::merge(&raw, &settings.set, &settings.remove);
                let next =
                    crate::channel_settings::compile(&effective, provider).unwrap_or_else(|_| {
                        settings
                            .compiled
                            .clone()
                            .unwrap_or_else(|| provider.clone())
                    });
                by_name.insert(provider.name.to_string(), next.clone());
                *provider = next;
            }
        }
        snapshot.providers = Arc::new(providers);
        snapshot.providers_by_name = Arc::new(by_name);
        snapshot.api_keys = Arc::new(keys);
        let snapshot = Arc::new(snapshot);
        *cache = Some((base.revision.to_string(), self.sequence, snapshot.clone()));
        snapshot
    }
    pub(crate) fn remove_settings_copy(&mut self, name: &str) -> bool {
        if !(name.starts_with("sub2api-copy-") || name.starts_with("typesafe-"))
            || !self.temporary_documents.contains_key(name)
        {
            return false;
        }
        self.temporary.remove(name);
        self.temporary_documents.remove(name);
        self.settings.remove(name);
        self.rules.retain(|_, r| {
            r.order.retain(|p| p != name);
            r.disabled.retain(|p| p != name);
            !r.order.is_empty() || !r.disabled.is_empty()
        });
        true
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
        self.settings.retain(|name, _| {
            !(name.starts_with("sub2api-") || name.starts_with("typesafe-"))
                || self.temporary.contains_key(name)
        });
        self.temporary_documents
            .retain(|name, _| self.temporary.contains_key(name));
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
            state.settings.remove(&input.provider);
            state.temporary_documents.remove(&input.provider);
        } else {
            // Model management changes only model intent, preserving credentials,
            // aliases for retained models, and unrelated advanced overrides.
            if let Some(settings) = state.settings.get_mut(&input.provider) {
                settings.set.remove("/model");
                settings.remove.retain(|p| p != "/model");
            }
            if let Some(raw) = state.temporary_documents.get_mut(&input.provider) {
                raw["model"] = json!(input
                    .models
                    .iter()
                    .map(|m| json!({existing.models.get(m).unwrap_or(m):m}))
                    .collect::<Vec<_>>());
            }
            let mut provider = (*existing).clone();
            provider.models = Arc::new(
                input
                    .models
                    .iter()
                    .map(|m| {
                        (
                            m.clone(),
                            existing.models.get(m).cloned().unwrap_or_else(|| m.clone()),
                        )
                    })
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

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RetainedChannel {
    pub provider: String,
    pub api_key_id: String,
    pub base_url: String,
    pub api_key: String,
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<Value>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RetainedSnapshot {
    pub version: u8,
    #[serde(default)]
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub temporary_channels: Vec<RetainedChannel>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channel_settings: BTreeMap<String, crate::channel_settings::ProviderSettings>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RestoreMutation {
    pub revision: String,
    pub snapshot: RetainedSnapshot,
}
impl NativeConfigStore {
    pub(crate) async fn restore_controls(
        &self,
        headers: &HeaderMap,
        input: RestoreMutation,
    ) -> Result<Value, (StatusCode, String)> {
        self.authorize_catalog(headers).await.map_err(|s| {
            (
                StatusCode::from_u16(s).unwrap_or(StatusCode::FORBIDDEN),
                "Platform administrator key required".into(),
            )
        })?;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let bad = |s: &str| (StatusCode::BAD_REQUEST, s.to_string());
        if ![1, 2].contains(&input.snapshot.version)
            || input.snapshot.rules.len() > 128
            || input.snapshot.temporary_channels.len() > 128
        {
            return Err(bad("Invalid retained configuration"));
        }
        let mut state = self.channel_controls.write().await;
        if state.revision(&base) != input.revision {
            return Err((
                StatusCode::CONFLICT,
                "Controls changed before restoration".into(),
            ));
        }
        // Construct and validate the whole overlay before touching serving state.
        let mut candidate = Controls {
            instance: state.instance.clone(),
            sequence: state.sequence + 1,
            bootstrap_restore: state.bootstrap_restore.clone(),
            ..Controls::default()
        };
        for p in input.snapshot.temporary_channels {
            if !(p.provider.starts_with("sub2api-")
                || (p.provider.starts_with("typesafe-") && p.definition.is_some()))
                || p.provider.len() > 100
                || !p
                    .provider
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-')
                || base.providers_by_name.contains_key(&p.provider)
                || candidate.temporary.contains_key(&p.provider)
                || p.api_key.is_empty()
                || p.api_key.len() > 8192
                || p.api_key.contains(['\r', '\n'])
                || p.models.is_empty()
                || p.models.len() > if p.definition.is_some() { 1024 } else { 32 }
                || p.models.iter().any(|m| {
                    m.is_empty()
                        || m.len() > 256
                        || (p.definition.is_none() && m.contains('/'))
                        || m.contains(['\r', '\n'])
                })
                || p.models.iter().collect::<BTreeSet<_>>().len() != p.models.len()
            {
                return Err(bad("Invalid retained temporary channel"));
            }
            if !base
                .api_keys
                .values()
                .any(|k| crate::channel_catalog::key_id(&k.token) == p.api_key_id)
            {
                return Err(bad("Retained API key no longer exists"));
            }
            let url = url::Url::parse(&p.base_url).map_err(|_| bad("Invalid retained address"))?;
            if !matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || (p.definition.is_none() && url.query().is_some())
                || url.fragment().is_some()
                || (p.definition.is_none() && !url.path().ends_with("/v1/responses"))
            {
                return Err(bad("Invalid retained address"));
            }
            let provider = Arc::new(Provider {
                name: p.provider.clone().into(),
                base_url: p.base_url.into(),
                engine: "gpt".into(),
                api_keys: Arc::new(vec![p.api_key]),
                project_id: None,
                private_key: None,
                client_email: None,
                aws_access_key: None,
                aws_secret_key: None,
                aws_session_token: None,
                cf_account_id: None,
                region: "global".into(),
                models: Arc::new(p.models.into_iter().map(|m| (m.clone(), m)).collect()),
                preferences: Arc::new(serde_json::Map::from_iter([(
                    "__temporary_key_id".into(),
                    json!(p.api_key_id),
                )])),
                excluded_endpoints: Arc::new(Vec::new()),
                only_request_types: Arc::new(Vec::new()),
                excluded_request_types: Arc::new(Vec::new()),
                excluded_request_rules: Arc::new(Vec::new()),
                cursor: Arc::new(AtomicUsize::new(0)),
            });
            let provider = if let Some(raw) = p.definition {
                let built =
                    crate::channel_settings::compile(&raw, &provider).map_err(|e| bad(&e))?;
                candidate
                    .temporary_documents
                    .insert(p.provider.clone(), raw);
                built
            } else {
                provider
            };
            candidate.temporary.insert(p.provider, provider);
        }
        if input.snapshot.channel_settings.len() > 1024 {
            return Err(bad("Too many channel settings"));
        }
        for (name, mut setting) in input.snapshot.channel_settings {
            let provider = candidate
                .temporary
                .get(&name)
                .or_else(|| base.providers_by_name.get(&name))
                .ok_or(bad("Configured channel no longer exists"))?;
            let raw = candidate
                .temporary_documents
                .get(&name)
                .cloned()
                .unwrap_or_else(|| crate::channel_settings::document(&base, provider));
            setting.compiled = Some(
                crate::channel_settings::compile(
                    &crate::channel_settings::merge(&raw, &setting.set, &setting.remove),
                    provider,
                )
                .map_err(|e| bad(&e))?,
            );
            candidate.settings.insert(name, setting);
        }
        let overlay = candidate.overlay(base.clone());
        let caller = overlay
            .api_keys
            .values()
            .find(|k| crate::channel_catalog::can_inspect_all(&overlay, k))
            .ok_or(bad("Administrator unavailable"))?;
        for rule in input.snapshot.rules {
            if rule.order.len() > 1024
                || rule.disabled.len() > 1024
                || rule.model.len() > 512
                || rule.api_key_id.len() > 128
            {
                return Err(bad("Retained rule too large"));
            }
            let allowed = entries(&overlay, caller, Some(&rule.api_key_id))
                .map_err(|_| bad("Retained key unavailable"))?;
            let names: BTreeSet<_> = allowed
                .into_iter()
                .filter(|(_, m)| rule.model.is_empty() || m == &rule.model)
                .map(|(p, _)| p.name.to_string())
                .collect();
            for values in [&rule.order, &rule.disabled] {
                if values.iter().collect::<BTreeSet<_>>().len() != values.len()
                    || values.iter().any(|p| !names.contains(p))
                {
                    return Err(bad("Retained rule references an unavailable channel"));
                }
            }
            let scope = (rule.api_key_id.clone(), rule.model.clone());
            if candidate.rules.contains_key(&scope) {
                return Err(bad("Duplicate retained rule"));
            }
            if !rule.order.is_empty() || !rule.disabled.is_empty() {
                candidate.rules.insert(scope, rule);
            }
        }
        if self
            .base_snapshot()
            .await
            .is_none_or(|latest| latest.revision != base.revision)
        {
            return Err((
                StatusCode::CONFLICT,
                "Configuration changed during restoration".into(),
            ));
        }
        *state = candidate;
        eprintln!(
            "{}",
            json!({"event_type":"channel_controls_restored","channels":state.temporary.len(),"rules":state.rules.len()})
        );
        Ok(state.view(&base))
    }
    // Optional control-plane intent is fetched before opening the model-serving
    // socket. Standalone uni-api remains database-free and unchanged by default.
    pub(crate) async fn restore_controls_on_start(&self) -> Result<(), String> {
        let Ok(url) = std::env::var("UNI_API_CONTROL_RESTORE_URL") else {
            return Ok(());
        };
        if url.trim().is_empty() {
            return Ok(());
        }
        let token = std::env::var("UNI_API_CONTROL_RESTORE_TOKEN")
            .map_err(|_| "Restore credential missing")?;
        let parsed = url::Url::parse(&url).map_err(|_| "Invalid restore URL")?;
        if !matches!(parsed.scheme(), "https" | "http")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err("Invalid restore URL".into());
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .map_err(|_| "Restore client unavailable")?;
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            format!("Bearer {token}")
                .parse()
                .map_err(|_| "Invalid restore credential")?,
        );
        loop {
            let attempt = async {
                let response = client
                    .get(&url)
                    .header("X-Uni-Channel-Settings-Version", "1")
                    .bearer_auth(&token)
                    .send()
                    .await
                    .map_err(|_| "Restore service unavailable")?;
                if !response.status().is_success() {
                    return Err("Restore service rejected request");
                }
                use futures_util::StreamExt;
                let mut stream = response.bytes_stream();
                let mut bytes = Vec::new();
                while let Some(part) = stream.next().await {
                    let part = part.map_err(|_| "Restore response interrupted")?;
                    if bytes.len() + part.len() > 2 * 1024 * 1024 {
                        return Err("Restore snapshot too large");
                    };
                    bytes.extend_from_slice(&part);
                }
                #[derive(Deserialize)]
                struct Bootstrap {
                    enabled: bool,
                    snapshot: Option<RetainedSnapshot>,
                    snapshot_id: Option<String>,
                }
                let payload: Bootstrap =
                    serde_json::from_slice(&bytes).map_err(|_| "Invalid restore response")?;
                if !payload.enabled {
                    return Ok(());
                }
                let snapshot = payload.snapshot.ok_or("Restore snapshot missing")?;
                // Only the authenticated startup response can supply a receipt;
                // runtime restoration must never reset the unchanged fence.
                let snapshot_id = payload.snapshot_id.filter(|id| {
                    id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit())
                });
                let view = self
                    .controls_view(&headers)
                    .await
                    .map_err(|_| "Restore authentication failed")?;
                let restored = self.restore_controls(
                    &headers,
                    RestoreMutation {
                        revision: view["revision"].as_str().unwrap_or_default().into(),
                        snapshot,
                    },
                )
                .await
                .map_err(|_| "Retained configuration validation failed")?;
                if let Some(snapshot_id) = snapshot_id {
                    let applied_revision = restored["revision"].as_str().unwrap_or_default().to_string();
                    self.channel_controls.write().await.bootstrap_restore = Some(BootstrapReceipt {
                        snapshot_id: snapshot_id.clone(), applied_revision: applied_revision.clone(),
                    });
                    eprintln!("{}", json!({"event_type":"channel_controls_bootstrapped","snapshot_id":snapshot_id,"applied_revision":applied_revision}));
                }
                Ok(())
            }
            .await;
            match attempt {
                Ok(()) => return Ok(()),
                Err(reason) => {
                    eprintln!("channel_controls_restore_waiting reason={reason}");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
}
