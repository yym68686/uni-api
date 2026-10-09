//! Version-independent Codex model cards derived from the caller's routes.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::OnceLock;

fn templates() -> &'static [Value] {
    static MODELS: OnceLock<Vec<Value>> = OnceLock::new();
    MODELS.get_or_init(|| {
        let catalog: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/codex/codex_models_pro_0_153_2.json"
        )))
        .expect("valid embedded Codex model templates");
        catalog["models"]
            .as_array()
            .expect("Codex model template array")
            .iter()
            .cloned()
            .map(compact_codex_card)
            .collect()
    })
}

// Check both the public alias and upstream model name. Modality-specific models
// cannot become coding models merely by being given a conversational alias.
pub(crate) fn is_conversational_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let leaf = model.rsplit('/').next().unwrap_or(&model);
    if ["embed", "rerank", "moderation"]
        .iter()
        .any(|part| leaf.contains(part))
    {
        return false;
    }
    if [
        "gpt-image",
        "dall-e",
        "dalle",
        "imagen",
        "flux",
        "stable-diffusion",
        "sora",
        "veo",
        "seedance",
        "whisper",
        "tts-",
        "speech-",
        "audio-",
    ]
    .iter()
    .any(|prefix| leaf.starts_with(prefix))
    {
        return false;
    }
    ![
        "-image",
        "-tts",
        "-speech",
        "-audio",
        "-transcribe",
        "-video",
        "-veo",
    ]
    .iter()
    .any(|part| leaf.contains(part))
}

// Keep canonical instructions byte-for-byte. Codex promotes base_instructions
// into ModelMessages when instructions_template is absent, including when the
// remaining structured tool/approval messages are present. Older clients also
// understand base_instructions, so keep that field instead of two prompt copies.
fn compact_codex_card(mut card: Value) -> Value {
    let fields = card.as_object_mut().expect("model card object");
    if let Some(messages) = fields
        .get_mut("model_messages")
        .and_then(Value::as_object_mut)
    {
        let instructions = messages.remove("instructions_template");
        // Current Codex treats instructions_template as literal text. These old
        // personality substitutions are no longer read by ModelMessages.
        messages.remove("instructions_variables");
        // ModelMessages' optional top-level fields all default to None. Do not
        // recursively strip nulls from tool schemas or other structured values.
        messages.retain(|_, value| !value.is_null());
        if let Some(Value::String(instructions)) = instructions {
            fields.insert("base_instructions".into(), Value::String(instructions));
        }
    }
    if fields
        .get("model_messages")
        .and_then(Value::as_object)
        .is_some_and(serde_json::Map::is_empty)
    {
        fields.remove("model_messages");
    }

    for field in [
        "guardian",
        "description",
        "default_reasoning_level",
        "default_service_tier",
        "available_access_programs",
        "availability_nux",
        "upgrade",
        "model_messages",
        "default_verbosity",
        "apply_patch_tool_type",
        "context_window",
        "max_context_window",
        "auto_compact_token_limit",
        "comp_hash",
        "auto_review_model_override",
        "model_specialty",
        "tool_mode",
        "multi_agent_version",
        "multi_agent_reasoning_effort",
    ] {
        if fields.get(field).is_some_and(Value::is_null) {
            fields.remove(field);
        }
    }
    for field in [
        "include_skills_usage_instructions",
        "include_plugin_usage_instructions",
        "supports_image_detail_original",
        "supports_search_tool",
        "supports_experimental_context",
        "use_responses_lite",
        "supports_reasoning_effort_updates",
        "node_repl_auto_review_required",
        "node_repl_disabled",
    ] {
        if fields.get(field) == Some(&Value::Bool(false)) {
            fields.remove(field);
        }
    }
    for field in [
        "include_apps_usage_instructions",
        "supports_reasoning_summary_parameter",
    ] {
        if fields.get(field) == Some(&Value::Bool(true)) {
            fields.remove(field);
        }
    }
    for (field, default) in [
        ("additional_speed_tiers", json!([])),
        ("service_tiers", json!([])),
        ("default_reasoning_summary", json!("auto")),
        ("web_search_tool_type", json!("text")),
        ("effective_context_window_percent", json!(95)),
        ("input_modalities", json!(["text", "image"])),
    ] {
        if fields.get(field) == Some(&default) {
            fields.remove(field);
        }
    }
    // These are not consumed by Codex's ModelInfo. Keep the legacy required
    // supports_reasoning_summaries / supports_parallel_tool_calls booleans.
    for field in [
        "available_in_plans",
        "minimal_client_version",
        "prefer_websockets",
        "requires_sandboxed_review",
    ] {
        fields.remove(field);
    }
    card
}

fn uses_codex_instructions(name: &str) -> bool {
    let leaf = name.rsplit('/').next().unwrap_or(name).to_ascii_lowercase();
    leaf.starts_with("gpt-") || leaf.starts_with("codex-")
}

fn generic_card(name: &str, priority: u64) -> Value {
    json!({
        "slug": name,
        "display_name": name,
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": true,
        "priority": priority,
        "support_verbosity": false,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "experimental_supported_tools": [],
        "base_instructions": include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"), "/assets/codex/generic_instructions.md"
        )).trim_end(),
        // Retain the previous generic catalog's context/modality settings. They
        // are compatibility metadata, not inferred provider-specific limits.
        "context_window": 272000,
        "max_context_window": 872000,
        "apply_patch_tool_type": "freeform",
        "input_modalities": ["text", "image"],
        "supports_reasoning_summary_parameter": false,
        "supports_reasoning_summaries": false,
        "supports_parallel_tool_calls": true,
    })
}

fn catalog(models: &[String]) -> Value {
    let mut remaining: BTreeSet<&str> = models.iter().map(String::as_str).collect();
    let templates = templates();
    let mut cards = Vec::with_capacity(remaining.len());
    for template in templates {
        if template["slug"]
            .as_str()
            .is_some_and(|id| remaining.remove(id))
        {
            cards.push(template.clone());
        }
    }
    let fallback = templates
        .iter()
        .find(|model| model["slug"] == "gpt-5.6-sol")
        .expect("generic Codex compatibility template");
    let first_priority = templates
        .iter()
        .filter_map(|model| model["priority"].as_u64())
        .max()
        .unwrap_or(0)
        + 1;
    for (index, name) in remaining.into_iter().enumerate() {
        let priority = first_priority + index as u64;
        let card = if uses_codex_instructions(name) {
            let mut card = fallback.clone();
            card["slug"] = json!(name);
            card["display_name"] = json!(name);
            card["description"] = json!(name);
            card["priority"] = json!(priority);
            card
        } else {
            generic_card(name, priority)
        };
        cards.push(card);
    }
    json!({"models": cards})
}

pub(crate) fn response(models: &[String], headers: &HeaderMap) -> Response<Body> {
    let body = serde_json::to_vec(&catalog(models)).expect("serializable model catalog");
    let etag = format!("\"{:x}\"", Sha256::digest(&body));
    let unchanged = headers.get_all("if-none-match").iter().any(|value| {
        value.to_str().ok().is_some_and(|value| {
            value.split(',').any(|candidate| {
                let candidate = candidate.trim();
                candidate == "*" || candidate.strip_prefix("W/").unwrap_or(candidate) == etag
            })
        })
    });
    let mut response = Response::builder()
        .status(if unchanged {
            StatusCode::NOT_MODIFIED
        } else {
            StatusCode::OK
        })
        .header("etag", etag)
        .header("cache-control", "private, no-cache")
        .header("vary", "Authorization, X-Api-Key")
        .header("x-uni-api-models-source", "key-scoped-catalog");
    if !unchanged {
        response = response
            .header("content-type", HeaderValue::from_static("application/json"))
            .header("content-length", body.len());
    }
    response
        .body(if unchanged {
            Body::empty()
        } else {
            Body::from(body)
        })
        .expect("valid model catalog response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_filter_keeps_conversation_and_rejects_generation_and_embeddings() {
        for model in [
            "gpt-6-astra",
            "gpt-6-sol",
            "claude-opus-5",
            "gemini-3.8-flash",
            "grok-4.6",
            "deepseek-v4-pro",
            "vendor/claude-sonnet",
            "codex-auto-review",
        ] {
            assert!(is_conversational_model(model), "{model}");
        }
        for model in [
            "gpt-image-2",
            "gpt-image-2.5",
            "GPT-IMAGE-2.5",
            "vendor/gpt-image-2",
            "gemini-embedding-001",
            "text-embedding-004",
            "jina-embeddings-v3",
            "bge-m3-embedding",
            "rerank-v3",
            "gemini-3-pro-image",
            "gemini-3.1-flash-image-preview",
            "gemini-2.5-flash-tts",
            "dall-e-3",
            "sora-2",
            "gemini-veo-3",
            "seedance-2-0",
            "whisper-1",
        ] {
            assert!(!is_conversational_model(model), "{model}");
        }
    }

    #[test]
    fn preserves_known_cards_and_synthesizes_new_names_without_duplicates() {
        let models = [
            "gpt-6-sol",
            "claude-opus-5",
            "gpt-6-astra",
            "gpt-6-sol",
            "codex-auto-review",
        ]
        .map(str::to_owned);
        let value = catalog(&models);
        let cards = value["models"].as_array().unwrap();
        assert_eq!(cards.len(), 4);
        let astra = cards.iter().find(|m| m["slug"] == "gpt-6-astra").unwrap();
        assert_eq!(astra["context_window"], 600000);
        assert_eq!(astra["max_context_window"], 872000);
        assert_eq!(
            astra,
            templates()
                .iter()
                .find(|m| m["slug"] == "gpt-6-astra")
                .unwrap()
        );
        let helper = cards
            .iter()
            .find(|m| m["slug"] == "codex-auto-review")
            .unwrap();
        assert_eq!(helper["visibility"], "hide");
        let fallback = templates()
            .iter()
            .find(|m| m["slug"] == "gpt-5.6-sol")
            .unwrap();
        for slug in ["gpt-6-sol"] {
            let card = cards.iter().find(|m| m["slug"] == slug).unwrap();
            assert_eq!(card["display_name"], slug);
            for (field, value) in fallback.as_object().unwrap() {
                if ![
                    "slug",
                    "display_name",
                    "description",
                    "priority",
                    "model_messages",
                ]
                .contains(&field.as_str())
                {
                    assert_eq!(&card[field], value, "{slug}: {field}");
                }
            }
            assert!(card.get("model_messages").is_none());
            assert_eq!(card["base_instructions"], fallback["base_instructions"]);
        }
        assert_eq!(catalog(&[]), json!({"models":[]}));
        assert_eq!(
            catalog(&models),
            catalog(&models.into_iter().rev().collect::<Vec<_>>())
        );
    }

    #[test]
    fn synthesized_catalog_stays_under_codex_size_limit_and_preserves_astra_context() {
        let mut models = templates()
            .iter()
            .filter_map(|model| model["slug"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        models.extend((0..26).map(|index| format!("custom-model-{index}")));
        let value = catalog(&models);
        let body = serde_json::to_vec(&value).unwrap();
        assert!(body.len() < 1024 * 1024, "catalog is {} bytes", body.len());
        assert!(!String::from_utf8_lossy(&body)
            .to_ascii_lowercase()
            .contains("uni-api"));

        let astra = value["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["slug"] == "gpt-6-astra")
            .unwrap();
        assert_eq!(astra["context_window"], 600000);
        assert_eq!(astra["max_context_window"], 872000);

        let custom = value["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["slug"] == "custom-model-0")
            .unwrap();
        assert!(custom.get("model_messages").is_none());
        assert!(custom["base_instructions"].as_str().is_some());
    }

    #[test]
    fn all_gpt_prompts_and_structured_rules_are_preserved() {
        let original: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/codex/codex_models_pro_0_153_2.json"
        )))
        .unwrap();
        for before in original["models"].as_array().unwrap() {
            let after = templates()
                .iter()
                .find(|m| m["slug"] == before["slug"])
                .unwrap();
            let prompt = before["model_messages"]["instructions_template"]
                .as_str()
                .unwrap_or_else(|| before["base_instructions"].as_str().unwrap());
            assert_eq!(after["base_instructions"], prompt, "{}", before["slug"]);
            for (field, value) in before["model_messages"].as_object().unwrap() {
                if !["instructions_template", "instructions_variables"].contains(&field.as_str())
                    && !value.is_null()
                {
                    assert_eq!(
                        &after["model_messages"][field], value,
                        "{}: {field}",
                        before["slug"]
                    );
                }
            }
            for field in [
                "context_window",
                "max_context_window",
                "visibility",
                "priority",
                "supported_reasoning_levels",
                "supports_parallel_tool_calls",
                "supports_reasoning_summaries",
            ] {
                assert_eq!(after[field], before[field], "{}: {field}", before["slug"]);
            }
        }
    }

    #[test]
    fn unknown_models_use_small_cards_but_gpt_names_keep_full_instructions() {
        for name in ["gpt-6-sol", "vendor/gpt-6.1-sol", "codex-future"] {
            let value = catalog(&[name.to_owned()]);
            let card = &value["models"][0];
            assert_eq!(
                card["base_instructions"],
                templates()
                    .iter()
                    .find(|m| m["slug"] == "gpt-5.6-sol")
                    .unwrap()["base_instructions"]
            );
        }
        for name in [
            "claude-opus-5",
            "gemini-3.8-flash",
            "vendor/claude-sonnet",
            "custom-chat",
        ] {
            let value = catalog(&[name.to_owned()]);
            let card = &value["models"][0];
            assert!(serde_json::to_vec(card).unwrap().len() < 900);
            assert!(card["base_instructions"]
                .as_str()
                .unwrap()
                .contains("AGENTS.md"));
            assert!(!card["base_instructions"]
                .as_str()
                .unwrap()
                .contains("GPT-5"));
            assert_eq!(card["supported_reasoning_levels"], json!([]));
            assert_eq!(card["experimental_supported_tools"], json!([]));
            assert_eq!(card["context_window"], 272000);
            assert_eq!(card["max_context_window"], 872000);
            assert_eq!(card["supports_reasoning_summary_parameter"], false);
            assert_eq!(card["support_verbosity"], false);
            assert!(card.get("service_tiers").is_none());
            assert!(card.get("use_responses_lite").is_none());
            assert!(card.get("tool_mode").is_none());
        }
    }

    #[test]
    fn representative_catalog_stays_below_300kb_without_dropping_models() {
        let mut models = templates()
            .iter()
            .filter(|m| m["slug"] != "gpt-reserve" && m["slug"] != "gpt-5.3-codex-spark")
            .map(|m| m["slug"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        models.extend(["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"].map(str::to_owned));
        models.extend((0..32).map(|i| format!("third-party-model-{i}")));
        let value = catalog(&models);
        assert_eq!(value["models"].as_array().unwrap().len(), 43);
        let body = serde_json::to_vec(&value).unwrap();
        assert!(body.len() < 300_000, "catalog is {} bytes", body.len());
        // More routes must not silently remove cards or alter existing ones.
        models.extend((32..132).map(|i| format!("third-party-model-{i}")));
        let expanded = catalog(&models);
        assert_eq!(expanded["models"].as_array().unwrap().len(), 143);
        assert!(serde_json::to_vec(&expanded).unwrap().len() < 400_000);
        for card in value["models"].as_array().unwrap() {
            assert_eq!(
                expanded["models"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|m| m["slug"] == card["slug"])
                    .unwrap()["base_instructions"],
                card["base_instructions"]
            );
        }
    }

    #[tokio::test]
    async fn etag_tracks_the_authorized_body_and_never_uses_upstream_etag() {
        let models = vec!["gpt-6-sol".to_owned()];
        let response = response(&models, &HeaderMap::new());
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["cache-control"], "private, no-cache");
        let etag = response.headers()["etag"].clone();
        let length = response.headers()["content-length"]
            .to_str()
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.len(), length);
        let mut headers = HeaderMap::new();
        headers.insert("if-none-match", etag);
        let cached = super::response(&models, &headers);
        assert_eq!(cached.status(), StatusCode::NOT_MODIFIED);
        assert!(axum::body::to_bytes(cached.into_body(), usize::MAX)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(super::response(&[], &headers).status(), StatusCode::OK);
    }
}
