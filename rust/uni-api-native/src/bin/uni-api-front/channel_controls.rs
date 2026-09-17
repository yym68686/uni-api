//! Process-local routing intent. Never written to configuration or persistent
//! storage; a new NativeConfigStore starts with no overrides and a new revision.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
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
#[derive(Clone)]
pub(crate) struct Controls {
    instance: String,
    sequence: u64,
    rules: BTreeMap<(String, String), Rule>,
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
        }
    }
}
impl Controls {
    fn revision(&self, snapshot: &Snapshot) -> String {
        format!("{}:{}:{}", self.instance, self.sequence, snapshot.revision)
    }
    fn view(&self, snapshot: &Snapshot) -> Value {
        json!({"revision":self.revision(snapshot),"instance_id":self.instance,"config_revision":snapshot.revision.as_ref(),"storage":"process_memory","reset_on_restart":true,"expires_at":null,"rules":self.rules.values().collect::<Vec<_>>()})
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
            "reset_all" => state.rules.clear(),
            "reset" => {
                state.rules.remove(&scope);
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
