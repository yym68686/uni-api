//! Read-only projections of configured channel order. This module never invokes
//! route scheduling, key selection, rate admission, or upstream requests.
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::responses_native::{ApiKey, Provider, Snapshot};

type Entry = (Arc<Provider>, String);

pub(crate) fn key_id(token: &str) -> String {
    format!("key-{:x}", Sha256::digest(token.as_bytes()))
}

// The first configured key is the dashboard connection credential. This grants
// catalog inspection, diagnostics and process-local routing controls. It does
// not grant persistent configuration mutation or unrelated administrator APIs.
pub(crate) fn can_inspect_all(snapshot: &Snapshot, caller: &ApiKey) -> bool {
    snapshot
        .api_key_order
        .first()
        .is_some_and(|token| token == caller.token.as_ref())
        || caller.role.eq_ignore_ascii_case("admin")
        || caller.role.to_ascii_lowercase().starts_with("admin-")
}

pub(crate) fn keys(snapshot: &Snapshot, caller: &ApiKey) -> Vec<Value> {
    snapshot.api_key_order.iter().enumerate()
        .filter_map(|(index, token)| {
            snapshot.api_keys.get(token)?;
            // Never return a complete credential, including unusually short keys.
            let prefix = if token.chars().count() > 14 {
                format!("{}…{}", token.chars().take(7).collect::<String>(), token.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<String>())
            } else { "••••".to_owned() };
            Some(json!({"key_id":key_id(token),"prefix":prefix,"position":index+1,"is_current":token==caller.token.as_ref()}))
        }).collect()
}

pub(crate) fn entries(
    snapshot: &Snapshot,
    caller: &ApiKey,
    selected_id: Option<&str>,
) -> Result<Vec<Entry>, u16> {
    if !can_inspect_all(snapshot, caller) {
        return Err(403);
    }
    if let Some(id) = selected_id.filter(|id| !id.is_empty()) {
        let selected = snapshot
            .api_key_order
            .iter()
            .filter_map(|token| snapshot.api_keys.get(token))
            .find(|key| key_id(&key.token) == id)
            .ok_or(404u16)?;
        return Ok(ordered_entries(snapshot, selected));
    }
    Ok(snapshot
        .providers
        .iter()
        .flat_map(|p| provider_entries(p, None))
        .collect())
}

fn provider_entries(provider: &Arc<Provider>, model: Option<&str>) -> Vec<Entry> {
    let models: BTreeSet<_> = provider
        .models
        .keys()
        .filter(|name| model.is_none_or(|wanted| name.as_str() == wanted))
        .cloned()
        .collect();
    models
        .into_iter()
        .map(|name| (provider.clone(), name))
        .collect()
}

#[derive(Default)]
struct Expansion {
    entries: Vec<Entry>,
    seen_entries: BTreeSet<(String, String)>,
    expanded_keys: BTreeSet<(String, Option<String>)>,
}
impl Expansion {
    fn append(&mut self, entries: Vec<Entry>) {
        for (provider, model) in entries {
            if self
                .seen_entries
                .insert((provider.name.to_string(), model.clone()))
            {
                self.entries.push((provider, model));
            }
        }
    }
    fn walk(&mut self, snapshot: &Snapshot, key: &ApiKey, only_model: Option<&str>) {
        // Memoizing each key/model pair preserves first occurrence order, avoids
        // exponential expansion of shared subgraphs, and terminates cycles.
        if !self
            .expanded_keys
            .insert((key.token.to_string(), only_model.map(str::to_owned)))
        {
            return;
        }
        let rules: Vec<&str> = key
            .preferences
            .get("__route_graph")
            .and_then(Value::as_array)
            .map(|rules| rules.iter().filter_map(Value::as_str).collect())
            .unwrap_or_else(|| key.model_rules.iter().map(String::as_str).collect());
        for rule in rules {
            let rule = rule.trim();
            if rule == "all" {
                for provider in snapshot.providers.iter() {
                    self.append(provider_entries(provider, only_model));
                }
            } else if let Some(model) = rule.strip_prefix('<').and_then(|v| v.strip_suffix('>')) {
                self.global_model(snapshot, model, only_model);
            } else if let Some((alias, requested)) = rule.split_once('/') {
                if requested != "*" && only_model.is_some_and(|wanted| wanted != requested) {
                    continue;
                }
                let model = if requested == "*" {
                    only_model
                } else {
                    Some(requested)
                };
                if let Some(child) = snapshot.api_keys.get(alias) {
                    self.walk(snapshot, child, model);
                } else if let Some(provider) = snapshot.providers_by_name.get(alias) {
                    self.append(provider_entries(provider, model));
                }
            } else {
                self.global_model(snapshot, rule, only_model);
            }
        }
    }
    fn global_model(&mut self, snapshot: &Snapshot, model: &str, only_model: Option<&str>) {
        if only_model.is_some_and(|wanted| wanted != model) {
            return;
        }
        for provider in snapshot.providers.iter() {
            self.append(provider_entries(provider, Some(model)));
        }
    }
}
fn ordered_entries(snapshot: &Snapshot, key: &ApiKey) -> Vec<Entry> {
    let mut expansion = Expansion::default();
    expansion.walk(snapshot, key, None);
    expansion.entries
}
