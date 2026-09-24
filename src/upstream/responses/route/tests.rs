use crate::config::snapshot::ApiKey;
use crate::config::snapshot::Provider;
use crate::config::snapshot::Snapshot;
use crate::providers::codex::oauth::CodexOAuthManager;
use crate::providers::overrides::apply_overrides;
use crate::providers::overrides::apply_removals;
use crate::routing::failure::classify_provider_failure;
use crate::routing::failure::is_missing_persisted_item_error;
use crate::routing::failure::is_provider_request_processing_failure;
use crate::routing::failure::remap_provider_status;
use crate::routing::filters::detect_request_type;
use crate::routing::filters::parse_byte_limit;
use crate::routing::filters::provider_accepts_request_rules;
use crate::routing::filters::provider_accepts_request_type;
use crate::routing::planner::diagnostic_key;
use crate::routing::planner::matching_providers;
use crate::routing::planner::weighted_provider_sequence;
use crate::routing::planner::PROBE_UNCONFIGURED_MODEL_HEADER;
use crate::routing::planner::TARGET_PROVIDER_HEADER;
use crate::routing::timeouts::model_preference;
use crate::routing::timeouts::model_timeout;
use crate::routing::timeouts::resolve_timeouts;
use crate::runtime::resources::ResourceGovernor;
use crate::runtime::scheduling::parse_rate_limits;
use crate::runtime::scheduling::retry_after_seconds;
use crate::runtime::state::GatewayRuntime;
use crate::storage::database::Persistence;
use crate::upstream::hedging::parse_hedging;
use crate::upstream::hedging::HedgingConfig;
use crate::upstream::responses::route::compile_payload;
use crate::upstream::responses::route::sha256_hex;
use crate::upstream::responses::route::terminal_error_sha256;
use crate::upstream::responses::route::ResponsesRoute;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn provider() -> Provider {
    Provider {
        name: Arc::from("fugue-codex"),
        base_url: Arc::from("https://example.com/v1/responses"),
        engine: Arc::from("codex"),
        api_keys: Arc::new(vec!["provider-key".into()]),
        project_id: None,
        private_key: None,
        client_email: None,
        aws_access_key: None,
        aws_secret_key: None,
        aws_session_token: None,
        cf_account_id: None,
        region: Arc::from("global"),
        models: Arc::new(HashMap::from([(
            "gpt-public".into(),
            "gpt-upstream".into(),
        )])),
        preferences: Arc::new(Map::from_iter([(
            "post_body_parameter_overrides".into(),
            json!({"store": false, "__remove__": ["temperature"]}),
        )])),
        excluded_endpoints: Arc::new(Vec::new()),
        only_request_types: Arc::new(Vec::new()),
        excluded_request_types: Arc::new(Vec::new()),
        excluded_request_rules: Arc::new(Vec::new()),
        cursor: Arc::new(AtomicUsize::new(0)),
    }
}

fn named_provider(name: &str) -> Arc<Provider> {
    let mut value = provider();
    value.name = name.to_owned().into();
    Arc::new(value)
}

async fn native_route_for_test(provider: Arc<Provider>, max_attempts: usize) -> ResponsesRoute {
    let store = GatewayRuntime::new();
    let snapshot = Arc::new(Snapshot {
        revision: Arc::from("0".repeat(64)),
        preferences: Arc::new(Map::new()),
        api_keys: Arc::new(HashMap::new()),
        api_key_order: Arc::new(Vec::new()),
        providers: Arc::new(vec![provider.clone()]),
        providers_by_name: Arc::new(HashMap::from([(
            provider.name.to_string(),
            provider.clone(),
        )])),
        api_config: Arc::new(json!({})),
    });
    let api_key = Arc::new(ApiKey {
        token: Arc::from("client-key"),
        model_rules: Arc::new(Vec::new()),
        role: Arc::from("user"),
        preferences: Arc::new(Map::from_iter([("AUTO_RETRY".into(), json!(true))])),
        weights: Arc::new(Map::new()),
        native_supported: true,
    });
    let (_, memory_reservation) = ResourceGovernor::unconstrained_for_test()
        .reserve_memory_capacity(0)
        .await
        .unwrap();
    ResponsesRoute {
        store,
        codex_oauth: CodexOAuthManager::new(),
        persistence: Persistence::initialize(true).await.unwrap(),
        snapshot,
        api_key,
        providers: vec![provider.clone(), provider],
        base_payload: json!({"model":"gpt-public","input":"hello","stream":true}),
        request_headers: HeaderMap::new(),
        request_model: "gpt-public".into(),
        endpoint: "/v1/responses".into(),
        request_type: None,
        wants_compact: false,
        stream: true,
        request_id: "request-test".into(),
        request_body_bytes: 64,
        cursor: 0,
        max_attempts,
        hedging: HedgingConfig::default(),
        attempt_contexts: HashMap::new(),
        history_repair_attempted: false,
        pending_history_repair: None,
        empty_name_repair_attempt_id: None,
        missing_item_repair_attempt_id: None,
        hedge_trigger_count: 0,
        hedge_cancelled_attempt_count: 0,
        last_provider: None,
        last_provider_key: None,
        last_original_model: None,
        last_attempt: None,
        last_status: 502,
        last_detail: String::new(),
        last_provider_model_unavailable: false,
        has_attempt_failure: false,
        last_failure_origin: String::new(),
        routing_attempts: 0,
        routing_skips: 0,
        upstream_attempts: 0,
        upstream_duration_ms: 0,
        routing_ledger: Vec::new(),
        upstream_ledger: Vec::new(),
        arrival: Some(crate::observability::timing::RequestArrival::now()),
        started_at: tokio::time::Instant::now(),
        final_emitted: false,
        _memory_reservation: memory_reservation,
    }
}

async fn catalog_fixture() -> GatewayRuntime {
    let store = GatewayRuntime::new();
    let mut providers = Vec::new();
    for name in ["z-first", "a-second", "m-third", "excluded"] {
        let mut p = provider();
        p.name = name.into();
        p.models = Arc::new(HashMap::from([
            ("shared".into(), "actual-shared".into()),
            ("extra".into(), "actual-extra".into()),
            ("vendor/model".into(), "actual-slash".into()),
        ]));
        if name == "excluded" {
            p.excluded_endpoints = Arc::new(vec!["/v1/responses".into()]);
        }
        providers.push(Arc::new(p));
    }
    let mut keys = HashMap::new();
    for (token, role, rules) in [
        ("dashboard-first", "user", vec!["z-first/shared"]),
        (
            "restricted",
            "user",
            vec!["m-third/shared", "a-second/*", "m-third/shared"],
        ),
        ("parent", "user", vec!["restricted/shared", "z-first/extra"]),
        (
            "mixed",
            "user",
            vec!["m-third/extra", "shared", "<vendor/model>", "z-first/*"],
        ),
        ("admin-key", "admin", vec!["all"]),
    ] {
        keys.insert(
            token.to_owned(),
            Arc::new(ApiKey {
                token: token.into(),
                model_rules: Arc::new(vec!["all".into()]),
                role: role.into(),
                preferences: Arc::new(Map::from_iter([("__route_graph".into(), json!(rules))])),
                weights: Arc::new(Map::new()),
                native_supported: true,
            }),
        );
    }
    *store.current.write().await = Some(Arc::new(Snapshot {
        revision: "0".repeat(64).into(),
        preferences: Arc::new(Map::new()),
        api_keys: Arc::new(keys),
        api_key_order: Arc::new(
            [
                "dashboard-first",
                "restricted",
                "parent",
                "mixed",
                "admin-key",
            ]
            .map(str::to_owned)
            .to_vec(),
        ),
        providers_by_name: Arc::new(
            providers
                .iter()
                .map(|p| (p.name.to_string(), p.clone()))
                .collect(),
        ),
        providers: Arc::new(providers),
        api_config: Arc::new(json!({})),
    }));
    store
}

#[tokio::test]
async fn temporary_channel_import_scopes_routing_and_resets_without_config_writes() {
    let store = catalog_fixture().await;
    let admin = catalog_headers("dashboard-first");
    let key = crate::routing::catalog::key_id("restricted");
    let original = store.current.read().await.clone().unwrap();
    let initial = store.controls_view(&admin).await.unwrap();
    let input = |revision: &Value, position| crate::control::channels::ImportMutation {
        revision: revision.as_str().unwrap().into(),
        action: String::new(),
        api_key_id: key.clone(),
        provider: "sub2api-fixture".into(),
        base_url: "https://example.com/v1/responses".into(),
        api_key: "secret-import-key".into(),
        models: vec!["shared".into(), "new-model".into()],
        position,
    };
    assert!(store
        .import_temporary_channel(
            &catalog_headers("restricted"),
            input(&initial["revision"], 1)
        )
        .await
        .is_err());
    assert!(store
        .import_temporary_channel(&admin, input(&initial["revision"], 999))
        .await
        .is_err());
    let changed = store
        .import_temporary_channel(&admin, input(&initial["revision"], 1))
        .await
        .unwrap();
    assert!(!changed.to_string().contains("secret-import-key"));
    assert!(store
        .import_temporary_channel(&admin, input(&initial["revision"], 1))
        .await
        .is_err());
    let snapshot = store.snapshot().await.unwrap();
    assert!(Arc::ptr_eq(&snapshot, &store.snapshot().await.unwrap()));
    for token in ["restricted", "parent", "admin-key", "mixed"] {
        let api_key = &snapshot.api_keys[token];
        let providers =
            matching_providers(&snapshot, api_key, "shared", 0, None, None, "/v1/responses")
                .unwrap();
        assert_eq!(
            providers
                .iter()
                .any(|p| p.name.as_ref() == "sub2api-fixture"),
            token == "restricted"
        );
        let ordered = store.schedule_providers(api_key, "shared", providers).await;
        if token == "restricted" {
            assert_eq!(ordered[0].name.as_ref(), "sub2api-fixture");
        }
        let rows = crate::routing::catalog::entries(
            &snapshot,
            &snapshot.api_keys["dashboard-first"],
            Some(&crate::routing::catalog::key_id(token)),
        )
        .unwrap();
        assert_eq!(
            rows.iter()
                .any(|(p, _)| p.name.as_ref() == "sub2api-fixture"),
            token == "restricted"
        );
    }
    assert!(!store
        .models_for_headers(&catalog_headers("admin-key"))
        .await
        .unwrap()
        .contains(&"new-model".to_string()));
    assert!(store
        .models_for_headers(&catalog_headers("restricted"))
        .await
        .unwrap()
        .contains(&"new-model".to_string()));
    assert!(!original.providers_by_name.contains_key("sub2api-fixture"));
    assert!(Arc::ptr_eq(
        &original,
        &store.current.read().await.clone().unwrap()
    ));
    // Re-adding replaces this scoped provider, never creates a duplicate.
    let changed = store
        .import_temporary_channel(&admin, input(&changed["revision"], 1))
        .await
        .unwrap();
    assert_eq!(changed["temporary_channels"].as_array().unwrap().len(), 1);
    let reset = crate::control::channels::Mutation {
        revision: changed["revision"].as_str().unwrap().into(),
        action: "reset".into(),
        api_key_id: key,
        model: "shared".into(),
        order: vec![],
        disabled: vec![],
    };
    let changed = store.mutate_controls(&admin, reset).await.unwrap();
    assert!(
        !store.snapshot().await.unwrap().providers_by_name["sub2api-fixture"]
            .models
            .contains_key("shared")
    );
    let reset = crate::control::channels::Mutation {
        revision: changed["revision"].as_str().unwrap().into(),
        action: "reset_all".into(),
        api_key_id: "".into(),
        model: "".into(),
        order: vec![],
        disabled: vec![],
    };
    store.mutate_controls(&admin, reset).await.unwrap();
    assert!(!store
        .snapshot()
        .await
        .unwrap()
        .providers_by_name
        .contains_key("sub2api-fixture"));
}

#[tokio::test]
async fn temporary_channel_management_preserves_other_routes_and_rejects_stale_edits() {
    let store = catalog_fixture().await;
    let admin = catalog_headers("dashboard-first");
    let key = crate::routing::catalog::key_id("restricted");
    let original = store.current.read().await.clone().unwrap();
    let mut view = store.controls_view(&admin).await.unwrap();
    let input = |view: &Value, provider: &str, action: &str, models: Vec<&str>| {
        crate::control::channels::ImportMutation {
            revision: view["revision"].as_str().unwrap().into(),
            action: action.into(),
            api_key_id: key.clone(),
            provider: provider.into(),
            base_url: "https://example.com/v1/responses".into(),
            api_key: "fixture-secret".into(),
            models: models.into_iter().map(String::from).collect(),
            position: 1,
        }
    };
    for p in ["sub2api-one", "sub2api-two"] {
        view = store
            .import_temporary_channel(&admin, input(&view, p, "", vec!["shared", "new-model"]))
            .await
            .unwrap();
    }
    view = store
        .mutate_controls(
            &admin,
            crate::control::channels::Mutation {
                revision: view["revision"].as_str().unwrap().into(),
                action: "set".into(),
                api_key_id: key.clone(),
                model: "shared".into(),
                order: vec!["sub2api-two".into(), "sub2api-one".into()],
                disabled: vec!["sub2api-two".into()],
            },
        )
        .await
        .unwrap();
    let before = view.clone();
    let mut wrong = input(&view, "sub2api-one", "delete", vec![]);
    wrong.api_key_id = crate::routing::catalog::key_id("admin-key");
    assert!(store.import_temporary_channel(&admin, wrong).await.is_err());
    assert_eq!(view, store.controls_view(&admin).await.unwrap());
    view = store
        .import_temporary_channel(
            &admin,
            input(&view, "sub2api-one", "replace", vec!["new-model"]),
        )
        .await
        .unwrap();
    let snapshot = store.snapshot().await.unwrap();
    assert!(!snapshot.providers_by_name["sub2api-one"]
        .models
        .contains_key("shared"));
    assert!(snapshot.providers_by_name["sub2api-two"]
        .models
        .contains_key("shared"));
    let rule = view["rules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model"] == "shared")
        .unwrap();
    assert_eq!(rule["order"], json!(["sub2api-two"]));
    assert_eq!(rule["disabled"], json!(["sub2api-two"]));
    let rejected = store
        .import_temporary_channel(&admin, input(&before, "sub2api-one", "delete", vec![]))
        .await
        .unwrap_err();
    assert_eq!(rejected.0, StatusCode::CONFLICT);
    view = store
        .import_temporary_channel(&admin, input(&view, "sub2api-one", "delete", vec![]))
        .await
        .unwrap();
    let snapshot = store.snapshot().await.unwrap();
    assert!(!snapshot.providers_by_name.contains_key("sub2api-one"));
    assert!(snapshot.providers_by_name.contains_key("sub2api-two"));
    assert!(!view["rules"].to_string().contains("sub2api-one"));
    assert!(!view.to_string().contains("fixture-secret"));
    assert!(Arc::ptr_eq(
        &original,
        &store.current.read().await.clone().unwrap()
    ));
    assert!(store
        .import_temporary_channel(
            &catalog_headers("restricted"),
            input(&view, "sub2api-two", "delete", vec![])
        )
        .await
        .is_err());
}

#[tokio::test]
async fn retained_controls_replace_atomically_and_validate_before_serving() {
    use crate::control::channels::RestoreMutation;
    use crate::control::channels::RetainedChannel;
    use crate::control::channels::RetainedSnapshot;
    use crate::control::channels::Rule;
    let store = catalog_fixture().await;
    let admin = catalog_headers("dashboard-first");
    let key = crate::routing::catalog::key_id("restricted");
    let snapshot = RetainedSnapshot {
        channel_settings: std::collections::BTreeMap::new(),
        version: 1,
        rules: vec![Rule {
            api_key_id: key.clone(),
            model: "shared".into(),
            order: vec!["sub2api-retained".into()],
            disabled: vec!["sub2api-retained".into()],
        }],
        temporary_channels: vec![RetainedChannel {
            provider: "sub2api-retained".into(),
            api_key_id: key,
            base_url: "https://example.com/v1/responses".into(),
            api_key: "restore-secret".into(),
            definition: None,
            models: vec!["shared".into()],
        }],
    };
    let initial = store.controls_view(&admin).await.unwrap();
    let restored = store
        .restore_controls(
            &admin,
            RestoreMutation {
                revision: initial["revision"].as_str().unwrap().into(),
                snapshot: snapshot.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(restored["temporary_channels"].as_array().unwrap().len(), 1);
    assert_eq!(
        restored["rules"][0]["disabled"],
        json!(["sub2api-retained"])
    );
    assert!(!restored.to_string().contains("restore-secret"));
    let before = restored.clone();
    let mut invalid = snapshot.clone();
    invalid.rules[0].order.push("missing".into());
    assert!(store
        .restore_controls(
            &admin,
            RestoreMutation {
                revision: restored["revision"].as_str().unwrap().into(),
                snapshot: invalid
            }
        )
        .await
        .is_err());
    assert_eq!(before, store.controls_view(&admin).await.unwrap());
    assert!(store
        .restore_controls(
            &admin,
            RestoreMutation {
                revision: initial["revision"].as_str().unwrap().into(),
                snapshot: snapshot.clone()
            }
        )
        .await
        .is_err());
    assert!(store
        .restore_controls(
            &catalog_headers("restricted"),
            RestoreMutation {
                revision: restored["revision"].as_str().unwrap().into(),
                snapshot: snapshot.clone()
            }
        )
        .await
        .is_err());
    let fresh = catalog_fixture().await;
    let view = fresh.controls_view(&admin).await.unwrap();
    let after = fresh
        .restore_controls(
            &admin,
            RestoreMutation {
                revision: view["revision"].as_str().unwrap().into(),
                snapshot,
            },
        )
        .await
        .unwrap();
    assert_eq!(after["rules"], restored["rules"]);
    assert_eq!(after["temporary_channels"], restored["temporary_channels"]);
    assert_ne!(after["instance_id"], restored["instance_id"]);
}

fn catalog_headers(token: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
    );
    headers
}

fn catalog_pairs(rows: &[Value]) -> Vec<(&str, &str)> {
    rows.iter()
        .map(|r| {
            (
                r["provider"].as_str().unwrap(),
                r["model"].as_str().unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn temporary_controls_are_scoped_reversible_and_revision_guarded() {
    use crate::control::channels::Mutation;
    let store = catalog_fixture().await;
    let snapshot = store.snapshot().await.unwrap();
    let headers = catalog_headers("dashboard-first");
    assert_eq!(
        store
            .controls_view(&catalog_headers("restricted"))
            .await
            .unwrap_err(),
        403
    );
    let initial = store.controls_view(&headers).await.unwrap();
    let input = |revision: &Value,
                 action: &str,
                 key: &str,
                 model: &str,
                 order: Vec<&str>,
                 disabled: Vec<&str>| {
        serde_json::from_value::<Mutation>(json!({"revision":revision,"action":action,"api_key_id":key,"model":model,"order":order,"disabled":disabled})).unwrap()
    };
    assert!(store
        .mutate_controls(
            &catalog_headers("restricted"),
            input(&initial["revision"], "set", "", "", vec![], vec!["z-first"])
        )
        .await
        .is_err());
    assert!(store
        .mutate_controls(
            &headers,
            input(&initial["revision"], "set", "", "", vec!["missing"], vec![])
        )
        .await
        .is_err());
    let changed = store
        .mutate_controls(
            &headers,
            input(
                &initial["revision"],
                "set",
                "",
                "",
                vec!["m-third", "a-second", "z-first"],
                vec!["z-first"],
            ),
        )
        .await
        .unwrap();
    assert!(store
        .mutate_controls(
            &headers,
            input(&initial["revision"], "reset", "", "", vec![], vec![])
        )
        .await
        .is_err());
    let names = |providers: Vec<Arc<Provider>>| {
        providers
            .into_iter()
            .map(|p| p.name.to_string())
            .collect::<Vec<_>>()
    };
    let available = vec![
        snapshot.providers_by_name["z-first"].clone(),
        snapshot.providers_by_name["a-second"].clone(),
        snapshot.providers_by_name["m-third"].clone(),
    ];
    assert_eq!(
        names(
            store
                .schedule_providers(&snapshot.api_keys["admin-key"], "shared", available.clone())
                .await
        ),
        vec!["m-third", "a-second"]
    );
    // No allow-list expansion: configured matches remain the sole candidates.
    assert!(store
        .schedule_providers(
            &snapshot.api_keys["dashboard-first"],
            "shared",
            vec![snapshot.providers_by_name["z-first"].clone()]
        )
        .await
        .is_empty());
    let id = crate::routing::catalog::key_id("restricted");
    let scoped = store
        .mutate_controls(
            &headers,
            input(
                &changed["revision"],
                "set",
                &id,
                "shared",
                vec!["a-second", "m-third"],
                vec!["m-third"],
            ),
        )
        .await
        .unwrap();
    assert_eq!(
        names(
            store
                .schedule_providers(
                    &snapshot.api_keys["restricted"],
                    "shared",
                    available.clone()
                )
                .await
        ),
        vec!["a-second"]
    );
    assert_eq!(
        names(
            store
                .schedule_providers(&snapshot.api_keys["restricted"], "extra", available.clone())
                .await
        ),
        vec!["m-third", "a-second"]
    );
    assert_eq!(
        names(
            store
                .schedule_providers(&snapshot.api_keys["admin-key"], "shared", available.clone())
                .await
        ),
        vec!["m-third", "a-second"]
    );
    let (rows, _, _) = store
        .channel_catalog(&headers, "/v1/responses", false, Some(&id))
        .await
        .unwrap();
    assert_eq!(
        rows.iter()
            .find(|r| r["provider"] == "m-third" && r["model"] == "shared")
            .unwrap()["reason"],
        "temporarily_disabled"
    );
    let reset = store
        .mutate_controls(
            &headers,
            input(&scoped["revision"], "reset", &id, "shared", vec![], vec![]),
        )
        .await
        .unwrap();
    assert_eq!(reset["rules"].as_array().unwrap().len(), 1);
    let cleared = store
        .mutate_controls(
            &headers,
            input(&reset["revision"], "reset_all", "", "", vec![], vec![]),
        )
        .await
        .unwrap();
    assert!(cleared["rules"].as_array().unwrap().is_empty());
    assert_eq!(
        names(
            store
                .schedule_providers(&snapshot.api_keys["admin-key"], "shared", available)
                .await
        ),
        vec!["z-first", "a-second", "m-third"]
    );
    let restarted = catalog_fixture().await;
    let fresh = restarted.controls_view(&headers).await.unwrap();
    assert_ne!(initial["instance_id"], fresh["instance_id"]);
    assert!(fresh["rules"].as_array().unwrap().is_empty());
    assert!(restarted
        .mutate_controls(
            &headers,
            input(&cleared["revision"], "set", "", "", vec![], vec!["z-first"])
        )
        .await
        .is_err());
}

#[tokio::test]
async fn unconfigured_model_probe_is_opt_in_and_does_not_mutate_snapshot() {
    let store = catalog_fixture().await;
    let snapshot = store.snapshot().await.unwrap();
    let mut headers = catalog_headers("admin-key");
    headers.insert(
        PROBE_UNCONFIGURED_MODEL_HEADER,
        HeaderValue::from_static("true"),
    );
    assert!(diagnostic_key(
        &snapshot,
        &snapshot.api_keys["admin-key"],
        &headers,
        "/v1/responses"
    )
    .is_err());
    headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("m-third"));
    assert!(diagnostic_key(
        &snapshot,
        &snapshot.api_keys["restricted"],
        &headers,
        "/v1/responses"
    )
    .is_err());
    let key = diagnostic_key(
        &snapshot,
        &snapshot.api_keys["admin-key"],
        &headers,
        "/v1/responses",
    )
    .ok()
    .unwrap();
    let probes = matching_providers(
        &snapshot,
        &key,
        "new-model",
        100,
        None,
        None,
        "/v1/responses",
    )
    .unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].name.as_ref(), "m-third");
    assert_eq!(probes[0].models["new-model"], "new-model");
    assert!(!snapshot.providers_by_name["m-third"]
        .models
        .contains_key("new-model"));
    assert!(!snapshot.api_keys["admin-key"]
        .preferences
        .contains_key("__diagnostic_unconfigured_model"));
    headers.remove(PROBE_UNCONFIGURED_MODEL_HEADER);
    let regular = diagnostic_key(
        &snapshot,
        &snapshot.api_keys["admin-key"],
        &headers,
        "/v1/responses",
    )
    .ok()
    .unwrap();
    assert!(matching_providers(
        &snapshot,
        &regular,
        "new-model",
        100,
        None,
        None,
        "/v1/responses"
    )
    .unwrap()
    .is_empty());
    headers.insert(
        PROBE_UNCONFIGURED_MODEL_HEADER,
        HeaderValue::from_static("true"),
    );
    headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("excluded"));
    let excluded = diagnostic_key(
        &snapshot,
        &snapshot.api_keys["admin-key"],
        &headers,
        "/v1/responses",
    )
    .ok()
    .unwrap();
    assert!(matching_providers(
        &snapshot,
        &excluded,
        "new-model",
        100,
        None,
        None,
        "/v1/responses"
    )
    .unwrap()
    .is_empty());
}

#[tokio::test]
async fn diagnostic_routing_is_admin_only_and_never_falls_back() {
    let store = catalog_fixture().await;
    let snapshot = store.snapshot().await.unwrap();
    for token in ["dashboard-first", "admin-key"] {
        let key = &snapshot.api_keys[token];
        let mut headers = catalog_headers(token);
        headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("m-third"));
        let targeted = diagnostic_key(&snapshot, key, &headers, "/v1/responses")
            .ok()
            .unwrap();
        let providers = matching_providers(
            &snapshot,
            &targeted,
            "shared",
            100,
            None,
            None,
            "/v1/responses",
        )
        .unwrap();
        assert_eq!(
            providers
                .iter()
                .map(|p| p.name.as_ref())
                .collect::<Vec<_>>(),
            vec!["m-third"]
        );
        assert!(matching_providers(
            &snapshot,
            &targeted,
            "missing",
            100,
            None,
            None,
            "/v1/responses"
        )
        .unwrap()
        .is_empty());
        assert!(diagnostic_key(&snapshot, key, &headers, "/v1/chat/completions").is_err());
        headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("missing"));
        assert!(diagnostic_key(&snapshot, key, &headers, "/v1/responses").is_err());
        headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("excluded"));
        let excluded = diagnostic_key(&snapshot, key, &headers, "/v1/responses")
            .ok()
            .unwrap();
        assert!(matching_providers(
            &snapshot,
            &excluded,
            "shared",
            100,
            None,
            None,
            "/v1/responses"
        )
        .unwrap()
        .is_empty());
    }
    let mut headers = catalog_headers("restricted");
    headers.insert(TARGET_PROVIDER_HEADER, HeaderValue::from_static("m-third"));
    assert!(diagnostic_key(
        &snapshot,
        &snapshot.api_keys["restricted"],
        &headers,
        "/v1/responses"
    )
    .is_err());
    assert_eq!(
        snapshot.api_keys["dashboard-first"].preferences["__route_graph"],
        json!(["z-first/shared"])
    );
}

#[tokio::test]
async fn fixed_priority_preserves_key_graph_order_when_deduplicating_matches() {
    let store = catalog_fixture().await;
    let snapshot = store.snapshot().await.unwrap();
    for (token, model, endpoint, expected) in [
        (
            "restricted",
            "shared",
            "/v1/messages",
            vec!["m-third", "a-second"],
        ),
        (
            "parent",
            "shared",
            "/v1/messages",
            vec!["m-third", "a-second"],
        ),
        ("parent", "extra", "/v1/messages", vec!["z-first"]),
        (
            "mixed",
            "shared",
            "/v1/responses",
            vec!["z-first", "a-second", "m-third"],
        ),
        (
            "admin-key",
            "shared",
            "/v1/messages",
            vec!["z-first", "a-second", "m-third", "excluded"],
        ),
    ] {
        let key = snapshot.api_keys.get(token).unwrap();
        let matched = matching_providers(&snapshot, key, model, 0, None, None, endpoint).unwrap();
        let scheduled = store.schedule_providers(key, model, matched).await;
        assert_eq!(
            scheduled
                .iter()
                .map(|p| p.name.as_ref())
                .collect::<Vec<_>>(),
            expected,
            "key={token} model={model}"
        );
    }
}

#[tokio::test]
async fn catalog_default_keeps_provider_order_and_does_not_change_routing_state() {
    let store = catalog_fixture().await;
    let headers = catalog_headers("dashboard-first");
    let (rows, _, selection) = store
        .channel_catalog(&headers, "/v1/responses", true, None)
        .await
        .unwrap();
    assert!(selection.is_empty());
    assert_eq!(
        catalog_pairs(&rows),
        vec![
            ("z-first", "extra"),
            ("z-first", "shared"),
            ("z-first", "vendor/model"),
            ("a-second", "extra"),
            ("a-second", "shared"),
            ("a-second", "vendor/model"),
            ("m-third", "extra"),
            ("m-third", "shared"),
            ("m-third", "vendor/model"),
        ]
    );
    assert!(store.scheduling.client_windows.lock().await.is_empty());
    assert!(store.scheduling.provider_windows.lock().await.is_empty());
    assert!(store.scheduling.routing_cursors.lock().await.is_empty());
    for p in store.snapshot().await.unwrap().providers.iter() {
        assert_eq!(p.cursor.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn catalog_selected_key_preserves_rules_filters_and_nested_model_restriction() {
    let store = catalog_fixture().await;
    let headers = catalog_headers("dashboard-first");
    let id = crate::routing::catalog::key_id("restricted");
    let (rows, _, selected) = store
        .channel_catalog(&headers, "/v1/responses", true, Some(&id))
        .await
        .unwrap();
    assert_eq!(id, selected);
    assert_eq!(
        catalog_pairs(&rows),
        vec![
            ("m-third", "shared"),
            ("a-second", "extra"),
            ("a-second", "shared"),
            ("a-second", "vendor/model")
        ]
    );
    let id = crate::routing::catalog::key_id("parent");
    let (rows, _, _) = store
        .channel_catalog(&headers, "/v1/responses", true, Some(&id))
        .await
        .unwrap();
    assert_eq!(
        catalog_pairs(&rows),
        vec![
            ("m-third", "shared"),
            ("a-second", "shared"),
            ("z-first", "extra")
        ]
    );
    let id = crate::routing::catalog::key_id("mixed");
    let (rows, _, _) = store
        .channel_catalog(&headers, "/v1/responses", true, Some(&id))
        .await
        .unwrap();
    assert_eq!(
        catalog_pairs(&rows),
        vec![
            ("m-third", "extra"),
            ("z-first", "shared"),
            ("a-second", "shared"),
            ("m-third", "shared"),
            ("z-first", "vendor/model"),
            ("a-second", "vendor/model"),
            ("m-third", "vendor/model"),
            ("z-first", "extra"),
        ]
    );
}

#[tokio::test]
async fn catalog_key_selection_is_redacted_and_cannot_escalate_access() {
    let store = catalog_fixture().await;
    for token in ["restricted", "parent", "mixed", "invalid"] {
        let headers = catalog_headers(token);
        assert_eq!(store.api_key_catalog(&headers).await.unwrap_err(), 403);
        assert_eq!(store.authorize_catalog(&headers).await.unwrap_err(), 403);
        assert!(matches!(
            store.balance_provider(&headers, "z-first").await,
            Err(403)
        ));
        for selected in [
            None,
            Some(crate::routing::catalog::key_id(token)),
            Some(crate::routing::catalog::key_id("dashboard-first")),
        ] {
            assert_eq!(
                store
                    .channel_catalog(&headers, "/v1/responses", true, selected.as_deref())
                    .await
                    .unwrap_err(),
                403
            );
        }
        if token != "invalid" {
            assert!(store.models_for_headers(&headers).await.is_ok());
        }
    }
    for token in ["dashboard-first", "admin-key"] {
        let headers = catalog_headers(token);
        assert_eq!(
            store
                .balance_provider(&headers, "z-first")
                .await
                .unwrap()
                .0
                .name
                .as_ref(),
            "z-first"
        );
        assert!(matches!(
            store.balance_provider(&headers, "unknown").await,
            Err(404)
        ));
        let listing = store.api_key_catalog(&headers).await.unwrap();
        assert_eq!(listing["data"].as_array().unwrap().len(), 5);
        assert_eq!(listing["can_inspect_all"], true);
        assert!(!listing.to_string().contains("restricted"));
        assert!(store.authorize_catalog(&headers).await.is_ok());
        assert_eq!(
            store
                .channel_catalog(&headers, "/v1/responses", true, Some("stale-id"))
                .await
                .unwrap_err(),
            404
        );
    }
}

#[tokio::test]
async fn catalog_metrics_keep_selected_order_and_channel_wide_statistics() {
    let store = catalog_fixture().await;
    let headers = catalog_headers("dashboard-first");
    let id = crate::routing::catalog::key_id("parent");
    let (rows, revision, _) = store
        .channel_catalog(&headers, "/v1/responses", true, Some(&id))
        .await
        .unwrap();
    let metrics = crate::observability::metrics::ChannelMetrics::new();
    metrics.start("m-third", "shared", "actual-shared", "/v1/responses", true);
    metrics.finish(
        "m-third",
        "shared",
        "actual-shared",
        "/v1/responses",
        true,
        "success",
        Some(1000.),
        Some(200.),
    );
    for timeseries in [false, true] {
        let response = metrics.query(rows.clone(), &revision, 15, timeseries);
        assert_eq!(
            catalog_pairs(response["data"].as_array().unwrap()),
            catalog_pairs(&rows)
        );
        assert_eq!(response["data"][0]["stats"]["success"], 1);
        assert_eq!(
            response["data"][0]["stats"]["first_output"]["last_ms"],
            200.
        );
    }
}

#[test]
fn payload_compiler_matches_codex_contract_without_double_json_envelope() {
    let mut provider = provider();
    provider.preferences = Arc::new(Map::new());
    let mut payload = json!({
        "model": "gpt-public",
        "input": [{"type":"reasoning","id":"rs_1","cache_control":{}}],
        "stream": true,
        "temperature": 1,
        "response_format": {"type": "json_object"},
        "store": true,
        "max_output_tokens": 42,
        "previous_response_id": "resp_1"
    });
    compile_payload(
        &mut payload,
        &provider,
        "gpt-public",
        "gpt-upstream",
        "codex",
        false,
    )
    .unwrap();
    assert_eq!(payload["model"], "gpt-upstream");
    assert_eq!(payload["store"], false);
    assert_eq!(payload["instructions"], "");
    assert!(payload.get("temperature").is_none());
    assert!(payload.get("response_format").is_none());
    assert!(payload.get("max_output_tokens").is_none());
    assert!(payload.get("previous_response_id").is_none());
    assert!(payload["input"][0].get("id").is_none());
    assert!(payload["input"][0].get("cache_control").is_none());
}

#[test]
fn codex_defaults_follow_overrides_without_mutating_provider_or_other_engines() {
    for engine in ["codex", " CODEX ", "gpt"] {
        let mut provider = provider();
        provider.engine = Arc::from(engine);
        let preferences = Map::from_iter([(
            "post_body_parameter_overrides".into(),
            json!({
                "store": true,
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
                "__remove__": ["metadata.remove"],
                "gpt-public": {
                    "__remove__": ["store"],
                    "temperature": 0.7,
                    "response_format": {"type": "text"},
                    "metadata": {"model_specific": true}
                }
            }),
        )]);
        provider.preferences = Arc::new(preferences.clone());
        let mut payload = json!({"metadata": {"remove": true, "keep": true}});
        apply_overrides(payload.as_object_mut().unwrap(), &provider, "gpt-public");
        assert_eq!(
            payload["metadata"],
            json!({"keep": true, "model_specific": true})
        );
        if engine == "gpt" {
            assert!(payload.get("store").is_none());
            assert_eq!(payload["temperature"], 0.7);
            assert_eq!(payload["response_format"], json!({"type": "text"}));
        } else {
            assert_eq!(payload["store"], false);
            assert!(payload.get("temperature").is_none());
            assert!(payload.get("response_format").is_none());
        }
        assert_eq!(*provider.preferences, preferences);
    }
}

#[test]
fn codex_compact_omits_store_after_applying_defaults() {
    let mut provider = provider();
    provider.preferences = Arc::new(Map::new());
    let mut payload = json!({
        "model": "gpt-public", "input": "hello", "store": true,
        "temperature": 1, "response_format": {"type": "json_object"}
    });
    compile_payload(
        &mut payload,
        &provider,
        "gpt-public",
        "gpt-upstream",
        "codex",
        true,
    )
    .unwrap();
    for field in ["store", "temperature", "response_format"] {
        assert!(payload.get(field).is_none(), "unexpected field {field}");
    }
}

#[test]
fn hedging_preferences_parse_with_safe_defaults() {
    let config = parse_hedging(&Map::from_iter([(
        "hedging".into(),
        json!({
            "enabled": true,
            "max_inflight_attempts": 2,
            "winner_policy": "first_valid_success"
        }),
    )]));
    assert!(config.enabled);
    assert_eq!(config.max_inflight_attempts, 2);
    assert_eq!(
        config.winner_policy,
        crate::upstream::hedging::WinnerPolicy::FirstValidSuccess
    );

    let invalid = parse_hedging(&Map::from_iter([(
        "hedging".into(),
        json!({"enabled": true, "winner_policy": "unknown"}),
    )]));
    assert_eq!(invalid, HedgingConfig::default());
}

#[test]
fn nonstream_total_policy_replaces_only_the_implicit_first_byte_fallback() {
    let mut provider = provider();
    let mut snapshot = Snapshot {
        revision: Arc::from("0".repeat(64)),
        preferences: Arc::new(Map::from_iter([("model_timeout".into(), json!(20))])),
        api_keys: Arc::new(HashMap::new()),
        api_key_order: Arc::new(Vec::new()),
        providers: Arc::new(Vec::new()),
        providers_by_name: Arc::new(HashMap::new()),
        api_config: Arc::new(json!({})),
    };
    for (stream, policy, expected_first, expected_total) in [
        (false, json!({}), 20.0, None),
        (false, json!({"total":100}), 100.0, Some(100.0)),
        (false, json!({"total":3000}), 3000.0, Some(3000.0)),
        (
            false,
            json!({"first_byte":10,"total":100}),
            10.0,
            Some(100.0),
        ),
        (false, json!({"first_byte":0,"total":100}), 0.0, Some(100.0)),
        (false, json!({"total":0}), 0.0, Some(0.0)),
        (true, json!({"total":100}), 20.0, Some(100.0)),
    ] {
        provider.preferences = Arc::new(Map::from_iter([(
            "timeout_policy".into(),
            json!({"default":policy}),
        )]));
        let actual = resolve_timeouts(
            &snapshot,
            &provider,
            "gpt-public",
            "gpt-upstream",
            "codex",
            stream,
            None,
            "user",
            "/v1/responses",
            "POST",
        );
        assert_eq!(
            actual.first_byte,
            Some(expected_first),
            "stream={stream} policy={policy}"
        );
        assert_eq!(actual.total, expected_total);
    }
    // A configured global first_byte is explicit even if a provider adds
    // only total. Keep that limit rather than mistaking it for the fallback.
    Arc::make_mut(&mut snapshot.preferences).insert(
        "timeout_policy".into(),
        json!({"default":{"first_byte":30}}),
    );
    let actual = resolve_timeouts(
        &snapshot,
        &provider,
        "gpt-public",
        "gpt-upstream",
        "codex",
        false,
        None,
        "user",
        "/v1/responses",
        "POST",
    );
    assert_eq!(actual.first_byte, Some(30.0));
    assert_eq!(actual.total, Some(100.0));
}

#[test]
fn timeout_policy_matches_compaction_request_type_without_affecting_regular_requests() {
    let provider = provider();
    let snapshot = Snapshot {
        revision: Arc::from("0".repeat(64)),
        preferences: Arc::new(Map::from_iter([
            ("model_timeout".into(), json!({"gpt-public": 20})),
            (
                "timeout_policy".into(),
                json!({
                    "rules": [{
                        "match": {
                            "endpoint": "/v1/responses",
                            "stream": true,
                            "request_type": "compaction",
                            "engine": "codex",
                            "model": ["gpt-5.6*", "gpt-public", "gpt-5.4*"]
                        },
                        "timeout": {"first_byte": 300, "total": 3000}
                    }]
                }),
            ),
        ])),
        api_keys: Arc::new(HashMap::new()),
        api_key_order: Arc::new(Vec::new()),
        providers: Arc::new(Vec::new()),
        providers_by_name: Arc::new(HashMap::new()),
        api_config: Arc::new(json!({})),
    };

    let compaction = resolve_timeouts(
        &snapshot,
        &provider,
        "gpt-public",
        "gpt-upstream",
        "codex",
        true,
        Some("compaction"),
        "user",
        "/v1/responses",
        "POST",
    );
    let regular = resolve_timeouts(
        &snapshot,
        &provider,
        "gpt-public",
        "gpt-upstream",
        "codex",
        true,
        None,
        "user",
        "/v1/responses",
        "POST",
    );

    assert_eq!(compaction.first_byte, Some(300.0));
    assert_eq!(compaction.total, Some(3000.0));
    assert_eq!(regular.first_byte, Some(20.0));
    assert_eq!(regular.total, None);
}

#[test]
fn keepalive_matches_request_then_upstream_then_provider_default_then_global() {
    let mut provider = provider();
    provider.preferences = Arc::new(Map::from_iter([(
        "keepalive_interval".into(),
        json!({
            "PUBLIC-MODEL": 1, "public": 2, "upstream": 3, "default": 4
        }),
    )]));
    let global = Map::from_iter([(
        "keepalive_interval".into(),
        json!({"global": 5, "default": 6}),
    )]);
    let resolve = |p: &Provider, model, upstream| {
        model_preference(p, &global, model, upstream, "keepalive_interval")
    };
    assert_eq!(resolve(&provider, "public-model", "upstream"), Some(1.0));
    assert_eq!(resolve(&provider, "public-other", "upstream"), Some(2.0));
    assert_eq!(resolve(&provider, "alias", "upstream-v2"), Some(3.0));
    assert_eq!(resolve(&provider, "global", "unknown"), Some(4.0));
    provider.preferences = Arc::new(Map::from_iter([(
        "keepalive_interval".into(),
        json!({"local": 7}),
    )]));
    assert_eq!(resolve(&provider, "global-v2", "unknown"), Some(5.0));
    assert_eq!(resolve(&provider, "unknown", "unknown"), Some(6.0));
    provider.preferences = Arc::new(Map::from_iter([("keepalive_interval".into(), json!(0))]));
    assert_eq!(resolve(&provider, "global", "unknown"), Some(0.0));
}

#[test]
fn model_timeout_matches_legacy_fuzzy_and_fallback_order() {
    let mut provider = provider();
    provider.preferences = Arc::new(Map::from_iter([(
        "model_timeout".into(),
        json!({
            "GPT-5.6-SOL": 15,
            "gpt-5.6": 20,
            "upstream-special": 40,
            "default": 90
        }),
    )]));
    let global = Map::from_iter([(
        "model_timeout".into(),
        json!({"gpt-5.6": 20, "global-only": 30, "default": 2000}),
    )]);

    assert_eq!(
        model_timeout(&provider, &global, "gpt-5.6-sol", "gpt-5.6-sol"),
        15.0
    );
    assert_eq!(
        model_timeout(&provider, &global, "public-alias", "upstream-special-v2"),
        40.0
    );
    assert_eq!(
        model_timeout(&provider, &global, "unknown", "unknown-upstream"),
        90.0
    );

    provider.preferences = Arc::new(Map::from_iter([(
        "model_timeout".into(),
        json!({"provider-only": 50}),
    )]));
    assert_eq!(
        model_timeout(&provider, &global, "gpt-5.6-terra", "gpt-5.6-terra"),
        20.0
    );
    assert_eq!(
        model_timeout(&provider, &global, "global-only-v2", "other"),
        30.0
    );

    let resolved = resolve_timeouts(
        &Snapshot {
            revision: Arc::from("0".repeat(64)),
            preferences: Arc::new(global),
            api_keys: Arc::new(HashMap::new()),
            api_key_order: Arc::new(Vec::new()),
            providers: Arc::new(Vec::new()),
            providers_by_name: Arc::new(HashMap::new()),
            api_config: Arc::new(json!({})),
        },
        &provider,
        "gpt-5.6-sol",
        "gpt-5.6-sol",
        "codex",
        true,
        None,
        "user",
        "/v1/responses",
        "POST",
    );
    assert_eq!(resolved.first_byte, Some(20.0));
    assert_eq!(resolved.idle, None);
    assert_eq!(resolved.total, None);
}

#[test]
fn structured_removal_matches_python_semantics() {
    let mut root = json!({
        "tools": [
            {"type":"function","name":"ok"},
            {"type":"image_generation"}
        ]
    })
    .as_object()
    .unwrap()
    .clone();
    apply_removals(
        &mut root,
        &json!([{
            "path":"tools",
            "where":{"type":"image_generation"},
            "drop_empty":true
        }]),
    );
    assert_eq!(root["tools"], json!([{"type":"function","name":"ok"}]));
}

#[test]
fn shared_python_rust_contract_fixture_covers_routes_and_provider_fields() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/runtime_contracts.fixture"
    )))
    .unwrap();
    assert!(fixture["routes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "/v1/video/tasks"));
    assert!(fixture["provider_fields"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "tools"));
    assert_eq!(fixture["nested_key"]["rule"], "child-key/*");
}

#[test]
fn tpr_rate_limits_are_preserved_as_request_token_rules() {
    let rules = parse_rate_limits(Some(&json!("2/tpr, 10/min")), Some("m")).unwrap();
    assert!(rules.contains(&(2, 0)));
    assert!(rules.contains(&(10, 60)));
}

#[test]
fn byte_limits_and_rate_limits_have_no_hard_request_ceiling() {
    assert_eq!(parse_byte_limit(&json!("20MiB")), Some(20 * 1024 * 1024));
    assert_eq!(parse_byte_limit(&Value::Null), None);
    assert_eq!(
        parse_rate_limits(Some(&json!("300/min, 1000/hour")), None),
        Some(vec![(300, 60), (1000, 3_600)])
    );
    assert_eq!(
        parse_rate_limits(
            Some(&json!({"gpt-5.6": "5/sec", "default": "300/min"})),
            Some("gpt-5.6-sol"),
        ),
        Some(vec![(5, 1)])
    );
}

#[tokio::test]
async fn weighted_round_robin_matches_legacy_sequence_and_rotates_requests() {
    let providers = vec![named_provider("a"), named_provider("b")];
    let weights = Map::from_iter([
        ("a/gpt-public".into(), json!(3)),
        ("b/gpt-public".into(), json!(1)),
    ]);
    let sequence =
        weighted_provider_sequence(&providers, "gpt-public", &weights, "weighted_round_robin");
    assert_eq!(
        sequence
            .iter()
            .map(|provider| provider.name.as_ref())
            .collect::<Vec<_>>(),
        vec!["a", "b", "a", "a"]
    );

    let api_key = ApiKey {
        token: "client".into(),
        model_rules: Arc::new(vec!["all".into()]),
        role: "client".into(),
        preferences: Arc::new(Map::from_iter([(
            "SCHEDULING_ALGORITHM".into(),
            json!("weighted_round_robin"),
        )])),
        weights: Arc::new(weights),
        native_supported: true,
    };
    let store = GatewayRuntime::new();
    let first = store
        .schedule_providers(&api_key, "gpt-public", providers.clone())
        .await;
    let second = store
        .schedule_providers(&api_key, "gpt-public", providers)
        .await;
    assert_eq!(first[0].name.as_ref(), "a");
    assert_eq!(second[0].name.as_ref(), "b");
}

#[test]
fn request_type_detection_and_provider_filters_match_python_semantics() {
    let regular = json!({"input": [{"type":"message"}]});
    let compact = json!({
        "input": [
            {"type":"message"},
            {"type":"compaction_trigger"}
        ]
    });
    assert_eq!(detect_request_type(regular.as_object().unwrap()), None);
    assert_eq!(
        detect_request_type(compact.as_object().unwrap()),
        Some("compaction")
    );

    let mut provider = provider();
    provider.only_request_types = Arc::new(vec!["compaction".into()]);
    assert!(!provider_accepts_request_type(&provider, None));
    assert!(provider_accepts_request_type(&provider, Some("compaction")));

    provider.excluded_request_types = Arc::new(vec!["compaction".into()]);
    assert!(!provider_accepts_request_type(
        &provider,
        Some("compaction")
    ));
}

#[test]
fn provider_request_rules_match_public_upstream_model_and_reasoning_effort() {
    let mut provider = provider();
    provider.excluded_request_rules = Arc::new(vec![json!({
        "match": {
            "endpoint": "/v1/responses",
            "request_model": "gpt-public",
            "upstream_model": "gpt-upstream",
            "reasoning_effort": ["MAX"]
        },
        "reason": "unsupported_reasoning_effort"
    })]);

    assert!(!provider_accepts_request_rules(
        &provider,
        "/v1/responses/",
        "gpt-public",
        Some("max"),
        None,
    ));
    assert!(provider_accepts_request_rules(
        &provider,
        "/v1/responses",
        "gpt-public",
        Some("high"),
        None,
    ));
    assert!(provider_accepts_request_rules(
        &provider,
        "/v1/responses",
        "gpt-public",
        None,
        None,
    ));
    assert!(provider_accepts_request_rules(
        &provider,
        "/v1/chat/completions",
        "gpt-public",
        Some("max"),
        None,
    ));
}

#[test]
fn provider_error_policy_matches_request_scoped_and_gateway_failures() {
    assert_eq!(
        remap_provider_status(
            400,
            r#"{"error":{"code":"model_not_priced","message":"missing"}}"#,
        ),
        502
    );
    assert!(is_missing_persisted_item_error(
        r#"{"error":{"type":"invalid_request_error","message":"Item with id 'rs_1' not found. Items are not persisted when store is false."}}"#,
    ));
    assert_eq!(
        retry_after_seconds("Rate limit reached. Please try again in 2500ms."),
        Some(3.0)
    );
    assert_eq!(terminal_error_sha256(true, "earlier attempt failed"), None);
    assert!(terminal_error_sha256(false, "terminal failure").is_some());
}

#[test]
fn unavailable_model_message_is_a_retryable_channel_failure() {
    let body =
        r#"{"error":{"message":"This model is not available.","type":"invalid_request_error"}}"#;
    for detail in [
        body.to_owned(),
        json!({"error": {"message": body}}).to_string(),
        json!({"detail": {"message": "  THIS MODEL IS NOT AVAILABLE.  "}}).to_string(),
        "This model is not available.".to_owned(),
    ] {
        for endpoint in [
            "/v1/responses",
            "/v1/responses/compact",
            "/v1/chat/completions",
        ] {
            let policy = classify_provider_failure(400, &detail, None, endpoint, true, None);
            assert_eq!(policy.status, 503, "{detail}");
            assert!(policy.retryable);
            assert!(!policy.request_scoped);
            assert!(policy.provider_model_unavailable);
            assert!(!policy.force_quota_cooldown);
            let disabled = classify_provider_failure(400, &detail, None, endpoint, false, None);
            assert_eq!(disabled.status, 503);
            assert!(!disabled.retryable);
        }
    }
    for detail in [
        r#"{"error":{"type":"invalid_request_error","message":"Missing required parameter: input"}}"#,
        r#"{"error":{"message":"Invalid input"},"input":"This model is not available."}"#,
        r#"{"error":{"message":"Invalid input"},"debug":{"message":"This model is not available."}}"#,
        r#"{"error":{"message":"Invalid input: expected 'This model is not available.'"}}"#,
    ] {
        let policy = classify_provider_failure(400, detail, None, "/v1/responses", true, None);
        assert_eq!(policy.status, 400, "{detail}");
        assert!(policy.request_scoped);
        assert!(!policy.retryable);
        assert!(!policy.provider_model_unavailable);
    }
    assert_eq!(remap_provider_status(413, body), 413);
}

#[test]
fn upstream_processing_failure_is_a_retryable_gateway_error() {
    let body = r#"{"error":{"message":"The upstream service could not process this request.","type":"invalid_request_error"}}"#;
    for detail in [
        body.to_owned(),
        json!({"error": {"message": body}}).to_string(),
        json!({"detail": {"message": " THE UPSTREAM SERVICE COULD NOT PROCESS THIS REQUEST. "}})
            .to_string(),
        "The upstream service could not process this request.".to_owned(),
    ] {
        for endpoint in [
            "/v1/responses",
            "/v1/responses/compact",
            "/v1/chat/completions",
        ] {
            let policy = classify_provider_failure(400, &detail, None, endpoint, true, None);
            assert_eq!(policy.status, 502, "{detail}");
            assert!(policy.retryable);
            assert!(!policy.request_scoped);
            assert!(!policy.provider_model_unavailable);
            let disabled = classify_provider_failure(400, &detail, None, endpoint, false, None);
            assert_eq!(disabled.status, 502);
            assert!(!disabled.retryable);
        }
    }
    for detail in [
        r#"{"error":{"message":"Invalid input"},"input":"The upstream service could not process this request."}"#,
        r#"{"error":{"message":"Invalid input"},"debug":{"message":"The upstream service could not process this request."}}"#,
        r#"{"error":{"message":"Invalid input: expected 'The upstream service could not process this request.'"}}"#,
    ] {
        let policy = classify_provider_failure(400, detail, None, "/v1/responses", true, None);
        assert_eq!(policy.status, 400, "{detail}");
        assert!(policy.request_scoped);
        assert!(!policy.retryable);
    }
    assert!(!is_provider_request_processing_failure(401, body));
    assert!(!is_provider_request_processing_failure(502, body));
}

#[tokio::test]
async fn generic_upstream_error_retries_and_cools_the_failed_channel() {
    let body = r#"{"error":{"code":"upstream_error","message":"Upstream request failed","type":"upstream_error"}}"#;
    let mut first = provider();
    first.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
    let mut route = native_route_for_test(Arc::new(first), 3).await;
    route.providers.insert(1, named_provider("fallback"));
    let first_plan = route.next_plan().await.unwrap().unwrap();
    assert!(
        route
            .record_plan_failure(
                first_plan,
                &json!({
                    "kind": "http_error", "status_code": 400, "body": body,
                })
            )
            .await
    );
    assert_eq!(route.last_status(), 502);
    assert_eq!(route.upstream_ledger[0]["status_code"], 400);
    assert_eq!(route.upstream_ledger[0]["error_sha256"], sha256_hex(body));
    assert_eq!(route.routing_ledger[0]["status_code"], 502);
    let fallback = route.next_plan().await.unwrap().unwrap();
    assert_eq!(fallback.provider_name.as_deref(), Some("fallback"));
    assert!(route.next_plan().await.unwrap().is_none());
    assert_eq!(route.routing_skips, 1);
    assert_eq!(route.last_status(), 502);
}

#[test]
fn generic_upstream_error_requires_the_gateway_error_envelope() {
    let body = r#"{"error":{"code":"upstream_error","message":"Upstream request failed","type":"upstream_error"}}"#;
    for detail in [
        body.to_owned(),
        json!({"error": {"message": body}}).to_string(),
        json!({"detail": {"message": body}}).to_string(),
        json!({"error": {"type": "upstream_error", "message": "Upstream request failed"}}).to_string(),
        json!({"error": {"type": " UPSTREAM_ERROR ", "code": null, "message": " UPSTREAM REQUEST FAILED "}}).to_string(),
    ] {
        for endpoint in ["/v1/responses", "/v1/responses/compact", "/v1/chat/completions"] {
            let policy = classify_provider_failure(400, &detail, None, endpoint, true, None);
            assert_eq!(policy.status, 502, "{detail}");
            assert!(policy.retryable);
            assert!(!policy.request_scoped);
            assert!(!policy.provider_model_unavailable);
            assert!(!policy.force_quota_cooldown);
            let disabled = classify_provider_failure(400, &detail, None, endpoint, false, None);
            assert_eq!(disabled.status, 502);
            assert!(!disabled.retryable);
        }
    }
    for detail in [
        "Upstream request failed".to_owned(),
        json!({"error": {"message": "Upstream request failed", "type": "invalid_request_error"}}).to_string(),
        json!({"error": {"message": "Upstream request failed", "type": "upstream_error", "code": "invalid_type"}}).to_string(),
        json!({"error": {"message": "Missing required parameter: input", "type": "upstream_error"}}).to_string(),
        json!({"error": {"message": "Invalid input: expected 'Upstream request failed'", "type": "upstream_error"}}).to_string(),
        json!({"error": {"message": "Invalid input"}, "input": body}).to_string(),
        json!({"error": {"message": "Invalid input"}, "debug": serde_json::from_str::<Value>(body).unwrap()}).to_string(),
    ] {
        let policy = classify_provider_failure(400, &detail, None, "/v1/responses", true, None);
        assert_eq!(policy.status, 400, "{detail}");
        assert!(policy.request_scoped);
        assert!(!policy.retryable);
    }
    for status in [401, 403, 404, 413, 429, 500, 503] {
        assert_eq!(remap_provider_status(status, body), status);
    }
}

#[tokio::test]
async fn upstream_processing_failure_retries_next_channel() {
    let mut first = provider();
    first.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
    let mut route = native_route_for_test(Arc::new(first), 3).await;
    route.providers.insert(1, named_provider("fallback"));
    let first_plan = route.next_plan().await.unwrap().unwrap();
    assert!(route
        .record_plan_failure(first_plan, &json!({
            "kind": "http_error",
            "status_code": 400,
            "body": r#"{"error":{"message":"The upstream service could not process this request.","type":"invalid_request_error"}}"#,
        }))
        .await);
    assert_eq!(route.last_status(), 502);
    assert_eq!(route.upstream_ledger[0]["status_code"], 400);
    assert_eq!(route.routing_ledger[0]["status_code"], 502);
    let fallback = route.next_plan().await.unwrap().unwrap();
    assert_eq!(fallback.provider_name.as_deref(), Some("fallback"));
}

#[tokio::test]
async fn unavailable_model_retries_next_channel_and_preserves_upstream_status() {
    let mut first = provider();
    first.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
    let mut route = native_route_for_test(Arc::new(first), 3).await;
    route.providers.insert(1, named_provider("fallback"));
    let first_plan = route.next_plan().await.unwrap().unwrap();
    assert!(route
        .record_plan_failure(first_plan, &json!({
            "kind": "http_error",
            "status_code": 400,
            "body": r#"{"error":{"message":"This model is not available.","type":"invalid_request_error"}}"#,
        }))
        .await);
    assert_eq!(route.last_status(), 503);
    assert_eq!(route.upstream_ledger[0]["status_code"], 400);
    assert_eq!(route.upstream_ledger[0]["provider_model_unavailable"], true);
    assert_eq!(route.routing_ledger[0]["status_code"], 503);
    let fallback = route.next_plan().await.unwrap().unwrap();
    assert_eq!(fallback.provider_name.as_deref(), Some("fallback"));
    // The failed provider/model is cooling, so a later turn skips it.
    assert!(route.next_plan().await.unwrap().is_none());
    assert_eq!(route.routing_skips, 1);
    assert_eq!(route.last_status(), 503);
}

fn model_mismatch_error(model: &str) -> Value {
    json!({"error": {
        "code": "unsupported_value",
        "message": format!("Unsupported value: 'max' is not supported with the '{model}' model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."),
        "param": "reasoning.effort",
        "type": "invalid_request_error",
    }})
}

#[test]
fn mismatched_model_validation_is_a_retryable_gateway_error() {
    let body = model_mismatch_error("gpt-5.5").to_string();
    assert_eq!(
        sha256_hex(&body),
        "a64a8cb7fea16f8b54289a847e80ad3c7ef1b2f85626af3ad7cbc75e215ca03f"
    );
    for detail in [
        body.clone(),
        json!({"error": {"message": body}}).to_string(),
        json!({"detail": {"message": json!({"error": {"message": body}}).to_string()}}).to_string(),
    ] {
        for endpoint in [
            "/v1/responses",
            "/v1/responses/compact",
            "/v1/chat/completions",
        ] {
            let policy =
                classify_provider_failure(400, &detail, None, endpoint, true, Some("gpt-6-sol"));
            assert_eq!(policy.status, 502);
            assert!(policy.retryable);
            assert!(!policy.request_scoped);
            assert!(!policy.provider_model_unavailable);
            assert!(!policy.force_quota_cooldown);
            let disabled =
                classify_provider_failure(400, &detail, None, endpoint, false, Some("gpt-6-sol"));
            assert_eq!(disabled.status, 502);
            assert!(!disabled.retryable);
        }
    }
    for status in [401, 403, 404, 413, 429, 500, 503] {
        assert_eq!(
            classify_provider_failure(
                status,
                &body,
                None,
                "/v1/responses",
                true,
                Some("gpt-6-sol")
            )
            .status,
            status
        );
    }
}

#[test]
fn model_mismatch_detection_preserves_client_errors_and_ignores_echoes() {
    let body = model_mismatch_error("gpt-5.5");
    for expected in [None, Some(""), Some("gpt-5.5"), Some(" GPT-5.5 ")] {
        let policy = classify_provider_failure(
            400,
            &body.to_string(),
            None,
            "/v1/responses",
            true,
            expected,
        );
        assert_eq!(policy.status, 400);
        assert!(!policy.retryable);
        assert!(policy.request_scoped);
    }
    let mut invalid_code = body.clone();
    invalid_code["error"]["code"] = json!("invalid_type");
    let mut invalid_type = body.clone();
    invalid_type["error"]["type"] = json!("upstream_error");
    let mut no_param = body.clone();
    no_param["error"]["param"] = Value::Null;
    let mut quoted_message = body.clone();
    quoted_message["error"]["message"] = json!(format!(
        "Invalid input: expected {}",
        body["error"]["message"]
    ));
    for detail in [
        invalid_code.to_string(),
        invalid_type.to_string(),
        no_param.to_string(),
        quoted_message.to_string(),
        body["error"]["message"].as_str().unwrap().to_owned(),
        json!({"error": {"message": "Invalid input"}, "input": body}).to_string(),
        json!({"error": {"message": "Invalid input"}, "debug": body}).to_string(),
        model_mismatch_error("").to_string(),
        model_mismatch_error("gpt-5.5' or 'gpt-6-sol").to_string(),
    ] {
        let policy =
            classify_provider_failure(400, &detail, None, "/v1/responses", true, Some("gpt-6-sol"));
        assert_eq!(policy.status, 400, "{detail}");
        assert!(!policy.retryable);
    }
}

#[tokio::test]
async fn mismatched_model_retries_and_cools_only_the_failed_route() {
    let mut first = provider();
    first.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
    let mut route = native_route_for_test(Arc::new(first), 3).await;
    route.providers.insert(1, named_provider("fallback"));
    let plan = route.next_plan().await.unwrap().unwrap();
    assert!(route.record_plan_failure(plan, &json!({
        "kind": "http_error", "status_code": 400, "body": model_mismatch_error("gpt-5.5").to_string(),
    })).await);
    assert_eq!(route.last_status(), 502);
    assert_eq!(route.upstream_ledger[0]["status_code"], 400);
    assert_eq!(route.routing_ledger[0]["status_code"], 502);
    let fallback = route.next_plan().await.unwrap().unwrap();
    assert_eq!(fallback.provider_name.as_deref(), Some("fallback"));
    assert!(route.next_plan().await.unwrap().is_none());
    assert_eq!(route.routing_skips, 1);
}

#[tokio::test]
async fn model_validation_uses_the_final_wire_model() {
    for override_model in [None, Some("gpt-5.5")] {
        let mut first = provider();
        if let Some(model) = override_model {
            first.preferences = Arc::new(Map::from_iter([(
                "post_body_parameter_overrides".into(),
                json!({"model": model}),
            )]));
        }
        let mut route = native_route_for_test(Arc::new(first), 2).await;
        let plan = route.next_plan().await.unwrap().unwrap();
        let expected = override_model.unwrap_or("gpt-upstream");
        assert_eq!(
            serde_json::from_str::<Value>(&plan.body).unwrap()["model"],
            expected
        );
        assert!(!route.record_plan_failure(plan, &json!({
            "kind": "http_error", "status_code": 400, "body": model_mismatch_error(expected).to_string(),
        })).await);
        assert_eq!(route.last_status(), 400);
        assert!(route
            .store
            .scheduling
            .route_failures
            .lock()
            .await
            .is_empty());
    }
}

#[test]
fn minimum_input_key_restrictions_are_retryable_gateway_errors() {
    let chinese = "该令牌不接受输入少于 2000 token 的请求(按请求体大小判定)。";
    let english = "This key does not accept requests with fewer than 2000 input tokens (judged by request body size).";
    for message in [
        format!("{chinese}{english}"),
        chinese.into(),
        english.into(),
        english.replace("2000", "8192").to_uppercase(),
    ] {
        let body =
            json!({"error": {"type": "invalid_request_error", "message": message}}).to_string();
        for detail in [
            message,
            body.clone(),
            json!({"error": {"message": body}}).to_string(),
        ] {
            for endpoint in [
                "/v1/responses",
                "/v1/responses/compact",
                "/v1/chat/completions",
            ] {
                let policy = classify_provider_failure(400, &detail, None, endpoint, true, None);
                assert_eq!(policy.status, 502, "{detail}");
                assert!(policy.retryable);
                assert!(!policy.request_scoped);
                assert!(!policy.provider_model_unavailable);
                assert!(!policy.force_quota_cooldown);
                assert!(
                    !classify_provider_failure(400, &detail, None, endpoint, false, None).retryable
                );
            }
        }
    }
    let escaped = r#"{"error":{"message":"\u8be5\u4ee4\u724c\u4e0d\u63a5\u53d7\u8f93\u5165\u5c11\u4e8e 4096 token \u7684\u8bf7\u6c42"}}"#;
    assert_eq!(remap_provider_status(400, escaped), 502);
    assert_eq!(remap_provider_status(413, escaped), 413);
}

#[test]
fn minimum_input_detection_preserves_client_validation_errors() {
    for detail in [
        r#"{"error":{"type":"invalid_request_error","message":"Missing required parameter: input"}}"#,
        "Input must contain at least 1 token.",
        "This model requires at least 2000 input tokens.",
        "This key does not accept requests with fewer than two input tokens.",
        "This key does not accept requests with fewer than 2000 output tokens.",
        "该令牌不接受输入少于 token 的请求",
        r#"{"error":{"message":"Invalid input"},"input":"该令牌不接受输入少于 2000 token 的请求"}"#,
        r#"{"error":{"message":"Invalid input"},"debug":{"message":"This key does not accept requests with fewer than 2000 input tokens"}}"#,
    ] {
        let policy = classify_provider_failure(400, detail, None, "/v1/responses", true, None);
        assert_eq!(policy.status, 400, "{detail}");
        assert!(policy.request_scoped);
        assert!(!policy.retryable);
    }
}

#[test]
fn provider_failure_policy_matches_python_retry_matrix() {
    let base_provider = provider();
    for status in [401, 402, 403, 404, 409, 422, 429, 500, 502, 503, 504] {
        let policy = classify_provider_failure(
            status,
            "upstream failure",
            Some(&base_provider),
            "/v1/chat/completions",
            true,
            None,
        );
        assert!(policy.retryable, "status {status} should fail over");
        assert!(
            !policy.request_scoped,
            "status {status} should not be request scoped"
        );
    }

    assert!(
        !classify_provider_failure(
            400,
            "invalid request",
            Some(&base_provider),
            "/v1/chat/completions",
            true,
            None
        )
        .retryable
    );
    assert!(
        !classify_provider_failure(
            413,
            "payload too large",
            Some(&base_provider),
            "/v1/chat/completions",
            true,
            None
        )
        .retryable
    );
    assert!(
        !classify_provider_failure(
            403,
            "upstream failure",
            Some(&base_provider),
            "/v1/chat/completions",
            false,
            None
        )
        .retryable
    );
    assert!(!classify_provider_failure(
        404,
        "{\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Item with id 'rs_1' not found. Items are not persisted when store is false.\"}}",
        Some(&base_provider),
        "/v1/responses",
        true, None,
    ).retryable);

    let pricing = classify_provider_failure(
        400,
        r#"{"error":{"code":"model_not_priced","message":"missing"}}"#,
        Some(&base_provider),
        "/v1/chat/completions",
        true,
        None,
    );
    assert_eq!(pricing.status, 502);
    assert!(pricing.retryable);

    let model_unavailable = classify_provider_failure(
        400,
        r#"{"error":{"code":"model_not_found","message":"unknown provider for model gpt-5.6-sol"}}"#,
        Some(&base_provider),
        "/v1/responses",
        true,
        None,
    );
    assert_eq!(model_unavailable.status, 503);
    assert!(model_unavailable.retryable);
    assert!(!model_unavailable.request_scoped);
    assert!(model_unavailable.provider_model_unavailable);

    let wrapped_model_unavailable = classify_provider_failure(
        400,
        r#"{"error":{"message":"{\"error\":{\"code\":\"model_not_found\",\"message\":\"unknown provider for model gpt-5.6-sol\"}}"}}"#,
        Some(&base_provider),
        "/v1/responses",
        true,
        None,
    );
    assert_eq!(wrapped_model_unavailable.status, 503);
    assert!(wrapped_model_unavailable.retryable);

    let ordinary_bad_request = classify_provider_failure(
        400,
        r#"{"error":{"code":"invalid_type","message":"messages: field required"}}"#,
        Some(&base_provider),
        "/v1/responses",
        true,
        None,
    );
    assert_eq!(ordinary_bad_request.status, 400);
    assert!(!ordinary_bad_request.retryable);
    assert!(!ordinary_bad_request.provider_model_unavailable);

    let codex = classify_provider_failure(
        400,
        "model is not supported when using codex with a ChatGPT account",
        Some(&base_provider),
        "/v1/responses",
        true,
        None,
    );
    assert!(codex.retryable);
    assert!(codex.force_quota_cooldown);
    assert!(
        !classify_provider_failure(
            400,
            "model is not supported when using codex with a ChatGPT account",
            Some(&base_provider),
            "/v1/chat/completions",
            true,
            None,
        )
        .retryable
    );

    let mut azure = provider();
    azure.engine = Arc::from("gpt");
    azure.base_url = Arc::from("https://models.inference.ai.azure.com");
    assert!(
        classify_provider_failure(
            400,
            "provider validation",
            Some(&azure),
            "/v1/chat/completions",
            true,
            None
        )
        .retryable
    );
    assert!(
        classify_provider_failure(
            413,
            "provider validation",
            Some(&azure),
            "/v1/chat/completions",
            true,
            None
        )
        .retryable
    );
}

#[tokio::test]
async fn local_cooldown_skip_does_not_overwrite_last_upstream_failure() {
    let mut provider = provider();
    provider.preferences = Arc::new(Map::from_iter([("cooldown_period".into(), json!(60.0))]));
    let mut route = native_route_for_test(Arc::new(provider), 2).await;

    assert!(route.next_plan().await.unwrap().is_some());
    assert!(
        route
            .record_failure(&json!({
                "kind": "http_error",
                "status_code": 503,
                "body": "no available token",
            }))
            .await
    );
    assert!(route.next_plan().await.unwrap().is_none());

    assert_eq!(route.last_status(), 503);
    assert_eq!(route.response_detail(), "no available token");
    assert_eq!(route.routing_attempts, 2);
    assert_eq!(route.routing_skips, 1);
    assert_eq!(route.upstream_attempts, 1);
    let final_message = route.final_message();
    assert_eq!(final_message["status_code"], 503);
    assert!(route.final_emitted);
}
