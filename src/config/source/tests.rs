use crate::config::compiler::compile_snapshot_bytes;
use crate::config::discovery::discovered_model_ids;
use crate::config::discovery::provider_models_url;
use serde_json::{json, Value};

#[test]
fn compiles_python_compatible_model_mappings_and_paid_state() {
    let raw = br#"
providers:
  - provider: codex-a
    base_url: https://example.com/v1/responses
    engine: codex
    api: secret-upstream
    model:
      - upstream-sol: gpt-5.6-sol
api_keys:
  - api: client-key
    model:
      - codex-a/*
    preferences:
      rate_limit: 10/min
"#;
    let bytes = compile_snapshot_bytes(raw, false).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value["providers"][0]["models"]["gpt-5.6-sol"],
        "upstream-sol"
    );
    assert_eq!(value["providers"][0]["model_order"][0], "gpt-5.6-sol");
    assert_eq!(value["api_keys"][0]["native_paid_state_safe"], true);
    assert_eq!(value["revision"].as_str().unwrap().len(), 64);
}

#[test]
fn preserves_provider_capabilities_and_video_provider_routes() {
    let raw = br#"
providers:
  - provider: p
    base_url: https://example.com/v1/chat/completions
    api: upstream
    model: [m]
    tools: true
    image: false
    preferences:
      api_key_schedule_algorithm: smart_round_robin
video_providers:
  - name: callxyq
    adapter: callxyq
    base_url: https://api.callxyq.xyz
    models:
      sora-2: sora-2
    routes:
      create_task: /v1/videos
api_keys:
  - api: parent
    model: [child/*]
  - api: child
    model: [callxyq/sora-2]
"#;
    let value: Value = serde_json::from_slice(&compile_snapshot_bytes(raw, true).unwrap()).unwrap();
    let p = value["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == "p")
        .unwrap();
    assert_eq!(p["preferences"]["tools"], true);
    assert_eq!(p["preferences"]["image"], false);
    assert!(value["providers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v["name"] == "callxyq"));
    assert_eq!(
        value["api_keys"][0]["preferences"]["__route_graph"][0],
        "child/*"
    );
}

#[test]
fn preserves_provider_exclude_request_rules() {
    let raw = br#"
providers:
  - provider: codex-a
    base_url: https://example.com/v1/responses
    engine: codex
    api: secret-upstream
    model:
      - codex-auto-review: gpt-5.6-luna
    exclude_request_rules:
      - match:
          endpoint: /v1/responses
          request_model: gpt-5.6-luna
          upstream_model: codex-auto-review
          reasoning_effort: [max]
        reason: unsupported_reasoning_effort
api_keys:
  - api: client-key
    model: [codex-a/*]
"#;
    let value: Value = serde_json::from_slice(&compile_snapshot_bytes(raw, true).unwrap()).unwrap();
    assert_eq!(
        value["providers"][0]["exclude_request_rules"][0]["match"]["reasoning_effort"][0],
        "max"
    );
}

#[test]
fn paid_key_is_admitted_to_native_database_balance_checks() {
    let raw = br#"
providers:
  - provider: p
    base_url: https://example.com/v1/responses
    api: upstream
    model: [m]
api_keys:
  - api: client
    model: [p/m]
    preferences:
      credits: 1.5
"#;
    let bytes = compile_snapshot_bytes(raw, false).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["api_keys"][0]["native_paid_state_safe"], true);
}

#[test]
fn expands_nested_api_key_model_rules_and_infers_engines() {
    let raw = br#"
providers:
  - provider: claude-a
    base_url: https://example.com/v1/messages
    api: upstream
    model: [claude-opus]
api_keys:
  - api: child-key
    model: [claude-a/*]
  - api: parent-key
    model: [child-key/*]
"#;
    let bytes = compile_snapshot_bytes(raw, true).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["providers"][0]["engine"], "claude");
    assert_eq!(value["api_keys"][1]["model_rules"][0], "claude-a/*");
}

#[test]
fn preserves_operator_gateway_for_project_backed_provider() {
    let raw = br#"
providers:
  - provider: vertex-through-gateway
    base_url: https://gateway.example/proxy/https://aiplatform.googleapis.com/
    project_id: project-a
    api: upstream
    model: [gemini]
api_keys:
  - api: client
    model: [vertex-through-gateway/*]
"#;
    let bytes = compile_snapshot_bytes(raw, true).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        value["providers"][0]["base_url"],
        "https://gateway.example/proxy/https://aiplatform.googleapis.com/"
    );
}

#[test]
fn model_discovery_normalizes_openai_and_gemini_shapes() {
    assert_eq!(
        provider_models_url("https://example.com/v1/chat/completions")
            .unwrap()
            .as_str(),
        "https://example.com/v1/models"
    );
    assert_eq!(
        discovered_model_ids(&json!({
            "models":[
                {"name":"models/gemini-2.5-pro"},
                {"name":"models/gemini-2.5-flash"},
                {"name":"models/gemini-2.5-pro"}
            ]
        })),
        vec!["gemini-2.5-flash", "gemini-2.5-pro"]
    );
}

#[test]
fn credential_only_special_providers_receive_native_route_keys() {
    let raw = br#"
providers:
  - provider: bedrock
    base_url: https://bedrock-runtime.us-east-1.amazonaws.com
    engine: aws
    aws_access_key: AKIA_TEST
    aws_secret_key: secret
    model: [anthropic.claude]
  - provider: vertex
    base_url: https://aiplatform.googleapis.com
    engine: vertex-claude
    client_email: svc@example.com
    private_key: key
    project_id: project-a
    model: [claude]
api_keys:
  - api: client
    model: [bedrock/*, vertex/*]
"#;
    let value: Value = serde_json::from_slice(&compile_snapshot_bytes(raw, true).unwrap()).unwrap();
    assert_eq!(value["providers"][0]["api"], "AKIA_TEST");
    assert_eq!(value["providers"][1]["api"], "__vertex_oauth__");
    assert_eq!(value["providers"][0]["aws_secret_key"], "secret");
}
