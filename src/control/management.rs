//! One current configuration read for the console. This never invokes route
//! scheduling, an upstream, a persistent writer, or a background refresh.

use crate::config::snapshot::{ApiKey, Snapshot};
use crate::control::channels::Controls;
use crate::runtime::state::GatewayRuntime;
use axum::http::{HeaderMap, StatusCode};
use serde_json::{json, Map, Value};

fn catalog(
    snapshot: &Snapshot,
    controls: &Controls,
    caller: &ApiKey,
    key: Option<&str>,
) -> Vec<Value> {
    // Same all-endpoint/all-stream directory and stable routing order as
    // /v1/model-channels. Transient cooldown/eligibility is not part of a
    // configuration snapshot and is not used by channel management.
    let mut rows: Vec<Value> = crate::routing::catalog::entries(snapshot, caller, key)
        .expect("snapshot keys were enumerated from this same snapshot")
        .into_iter()
        .filter(|(p, _)| !p.excluded_endpoints.iter().any(|v| v.trim_end_matches('/').eq_ignore_ascii_case("all")))
        .filter_map(|(p, model)| {
            let upstream = p.models.get(&model)?;
            Some(json!({"provider":p.name.as_ref(),"model":model,"upstream_model":upstream,"engine":p.engine.as_ref(),"engine_mode":p.engine_mode.as_ref()}))
        })
        .collect();
    rows.sort_by_key(|row| {
        controls
            .order(
                key.unwrap_or_default(),
                row["model"].as_str().unwrap_or_default(),
            )
            .and_then(|order| {
                order
                    .iter()
                    .position(|p| Some(p.as_str()) == row["provider"].as_str())
            })
            .unwrap_or(usize::MAX)
    });
    rows
}

impl GatewayRuntime {
    pub(crate) async fn management_snapshot(
        &self,
        headers: &HeaderMap,
        revision_only: bool,
    ) -> Result<Value, (StatusCode, String)> {
        let token = crate::routing::access::extract_api_key(headers)
            .ok_or((StatusCode::FORBIDDEN, "Invalid API key".into()))?;
        let base = self.base_snapshot().await.ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration unavailable".into(),
        ))?;
        let controls = self.channel_controls.read().await;
        if self
            .base_snapshot()
            .await
            .is_none_or(|current| current.revision != base.revision)
        {
            return Err((
                StatusCode::CONFLICT,
                "Configuration changed during read".into(),
            ));
        }
        let effective = controls.overlay(base.clone());
        let caller = effective
            .api_keys
            .get(&token)
            .ok_or((StatusCode::FORBIDDEN, "Invalid API key".into()))?;
        // Match the existing settings/export authorization at this very
        // snapshot too; a concurrently revoked administrator cannot read it.
        if !crate::routing::catalog::can_inspect_all(&effective, caller)
            || effective.api_keys.len() > 1 && !caller.role.to_ascii_lowercase().contains("admin")
        {
            return Err((StatusCode::FORBIDDEN, "Permission denied".into()));
        }
        let revision = controls.revision(&base);
        if revision_only {
            return Ok(
                json!({"format":"management-revision-v1","revision":revision,"config_revision":base.revision.as_ref()}),
            );
        }
        // Retain one version, then release the serving lock before expanding
        // every caller's directory or serializing exported intent.
        let captured = controls.management_read_copy();
        drop(controls);
        let controls = captured;
        let keys = crate::routing::catalog::keys(&effective, caller);
        let mut routes = Map::new();
        for key in &keys {
            let id = key["key_id"].as_str().expect("catalog key identity");
            routes.insert(
                id.to_owned(),
                json!(catalog(&effective, &controls, caller, Some(id))),
            );
        }
        // Provider credentials and exported intent have the exact same admin
        // scope as the existing settings endpoints. The console keeps these
        // fields private and never forwards them to a browser.
        let providers: Vec<Value> = effective.providers.iter().map(|p| json!({
            "provider":p.name.as_ref(),"base_url":p.base_url.as_ref(),"api":p.api_keys.as_ref(),
            "temporary":controls.temporary.contains_key(p.name.as_ref()),
            "identity_changed":p.name.starts_with("sub2api-copy-") || controls.settings.get(p.name.as_ref()).is_some_and(crate::control::settings::identity_changed)
        })).collect();
        Ok(
            json!({"format":"management-snapshot-v1","revision":revision,"config_revision":base.revision.as_ref(),
            "controls":controls.view(&base),"keys":keys,"catalog":catalog(&effective,&controls,caller,None),
            "routes":routes,"providers":providers,"export":controls.export(&base)}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::snapshot::{runtime_provider, RawProvider};
    use std::collections::HashMap;
    use std::sync::{atomic::AtomicUsize, Arc};

    fn headers(token: &str) -> HeaderMap {
        let mut out = HeaderMap::new();
        out.insert("authorization", format!("Bearer {token}").parse().unwrap());
        out
    }

    async fn fixture() -> GatewayRuntime {
        let providers = ["first", "second"].map(|name| {
            let raw: RawProvider = serde_json::from_value(json!({"name":name,
                "base_url":"https://unused.invalid/v1/responses","engine":"gpt","engine_mode":"auto",
                "api":["private-provider-key"],"models":{"a":"upstream-a","b":"b"}})).unwrap();
            runtime_provider(raw, Arc::new(AtomicUsize::new(0)))
        });
        let tokens = ["catalog-only", "admin-token", "ordinary"];
        let keys = tokens
            .into_iter()
            .map(|token| {
                (
                    token.into(),
                    Arc::new(ApiKey {
                        token: token.into(),
                        model_rules: Arc::new(vec!["second/*".into(), "first/*".into()]),
                        role: if token == "admin-token" { "admin" } else { "" }.into(),
                        preferences: Arc::new(Map::new()),
                        weights: Arc::new(Map::new()),
                        native_supported: true,
                    }),
                )
            })
            .collect();
        let snapshot = Snapshot {
            revision: "base-one".into(),
            preferences: Arc::new(Map::new()),
            api_keys: Arc::new(keys),
            api_key_order: Arc::new(tokens.into_iter().map(str::to_owned).collect()),
            providers: Arc::new(providers.to_vec()),
            providers_by_name: Arc::new(
                providers
                    .into_iter()
                    .map(|p| (p.name.to_string(), p))
                    .collect::<HashMap<_, _>>(),
            ),
            api_config: Arc::new(json!({"must":"remain unchanged"})),
        };
        let store = GatewayRuntime::new();
        *store.current.write().await = Some(Arc::new(snapshot));
        store
    }

    fn management_fields(rows: Vec<Value>) -> Vec<Value> {
        rows.into_iter().map(|v| json!({"provider":v["provider"],"model":v["model"],
            "upstream_model":v["upstream_model"],"engine":v["engine"],"engine_mode":v["engine_mode"]})).collect()
    }

    #[tokio::test]
    async fn current_snapshot_matches_directories_and_exports_without_a_write() {
        let store = fixture().await;
        let admin = headers("admin-token");
        let kid = crate::routing::catalog::key_id("ordinary");
        let before = store.controls_view(&admin).await.unwrap();
        store.mutate_controls(&admin, serde_json::from_value(json!({"revision":before["revision"],
            "action":"set","api_key_id":kid,"model":"a","order":["first","second"],"disabled":["second"]})).unwrap()).await.unwrap();
        let current = store.controls_view(&admin).await.unwrap();
        let snapshot = store.management_snapshot(&admin, false).await.unwrap();
        assert_eq!(snapshot["format"], "management-snapshot-v1");
        assert_eq!(snapshot["revision"], current["revision"]);
        assert_eq!(snapshot["controls"], current);
        assert_eq!(snapshot["export"], store.settings_export().await.unwrap());
        assert_eq!(
            snapshot["providers"],
            store.settings_providers().await.unwrap()["providers"]
        );
        assert_eq!(
            snapshot["keys"],
            store.api_key_catalog(&admin).await.unwrap()["data"]
        );
        let (catalog, _, _) = store
            .channel_catalog(&admin, "all", false, None)
            .await
            .unwrap();
        assert_eq!(snapshot["catalog"], json!(management_fields(catalog)));
        for key in snapshot["keys"].as_array().unwrap() {
            let id = key["key_id"].as_str().unwrap();
            let (rows, _, _) = store
                .channel_catalog(&admin, "all", false, Some(id))
                .await
                .unwrap();
            assert_eq!(snapshot["routes"][id], json!(management_fields(rows)));
        }
        let revision = store.management_snapshot(&admin, true).await.unwrap();
        assert_eq!(
            revision,
            json!({"format":"management-revision-v1","revision":snapshot["revision"],"config_revision":"base-one"})
        );
        assert_eq!(current, store.controls_view(&admin).await.unwrap());
        assert_eq!(
            store.base_snapshot().await.unwrap().api_config.as_ref(),
            &json!({"must":"remain unchanged"})
        );
    }

    #[tokio::test]
    async fn authorization_and_latest_effective_identity_are_checked_at_read_time() {
        let store = fixture().await;
        for token in ["", "unknown", "ordinary", "catalog-only"] {
            assert_eq!(
                store
                    .management_snapshot(&headers(token), false)
                    .await
                    .unwrap_err()
                    .0,
                StatusCode::FORBIDDEN
            );
            assert_eq!(
                store
                    .management_snapshot(&headers(token), true)
                    .await
                    .unwrap_err()
                    .0,
                StatusCode::FORBIDDEN
            );
        }
        let admin = headers("admin-token");
        let old = store.management_snapshot(&admin, false).await.unwrap();
        let mutation = serde_json::from_value(
            json!({"revision":old["revision"],"operation_id":"current-identity",
            "changes":[{"provider":"first","set":{"/api":["rotated-key"]}}]}),
        )
        .unwrap();
        store.settings_change(mutation, true).await.unwrap();
        let fresh = store.management_snapshot(&admin, false).await.unwrap();
        assert_ne!(old["revision"], fresh["revision"]);
        assert_eq!(fresh["providers"][0]["api"], json!(["rotated-key"]));
        assert_eq!(fresh["providers"][0]["identity_changed"], true);
        assert_eq!(fresh["revision"], fresh["controls"]["revision"]);
        assert_eq!(fresh["revision"], fresh["export"]["revision"]);
        let mut revoked = (*store.base_snapshot().await.unwrap()).clone();
        let mut keys = (*revoked.api_keys).clone();
        keys.remove("admin-token");
        revoked.api_keys = Arc::new(keys);
        revoked.revision = "base-two".into();
        *store.current.write().await = Some(Arc::new(revoked));
        assert_eq!(
            store
                .management_snapshot(&admin, false)
                .await
                .unwrap_err()
                .0,
            StatusCode::FORBIDDEN
        );
        *store.current.write().await = None;
        assert_eq!(
            store
                .management_snapshot(&admin, false)
                .await
                .unwrap_err()
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
