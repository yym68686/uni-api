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
            .clone()
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

fn catalog(models: &[String]) -> Value {
    let mut remaining: BTreeSet<&str> = models.iter().map(String::as_str).collect();
    let templates = templates();
    let mut cards = Vec::with_capacity(remaining.len());
    for template in templates {
        if template["slug"]
            .as_str()
            .is_some_and(|id| remaining.remove(id))
        {
            // Preserve established GPT metadata byte-for-value, especially
            // gpt-6-astra's 600000 context_window and hidden helper cards.
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
        let mut card = fallback.clone();
        card["slug"] = json!(name);
        card["display_name"] = json!(name);
        card["description"] = json!(name);
        card["priority"] = json!(first_priority + index as u64);
        // Unknown routes inherit the legacy instruction field from the fallback
        // template. The same instructions are also present in model_messages;
        // dropping that duplicate keeps the Codex catalog below its 1 MiB
        // client-side limit while preserving the complete prompt semantics.
        if let Some(card) = card.as_object_mut() {
            card.remove("model_messages");
        }
        cards.push(card);
    }
    let mut catalog = json!({"models": cards});
    let mut size = serde_json::to_vec(&catalog).unwrap().len();
    // A growing route list can exceed the client limit again. Only remove a
    // legacy prompt's exact duplicate, never structured behavior or variables.
    for card in catalog["models"].as_array_mut().unwrap() {
        if size < 1024 * 1024 {
            break;
        }
        let duplicate = card["model_messages"].as_object().is_some_and(|messages| {
            card["base_instructions"].is_string()
                && messages.get("instructions_template") == Some(&card["base_instructions"])
                && messages
                    .iter()
                    .all(|(field, value)| field == "instructions_template" || value.is_null())
        });
        if duplicate {
            let before = serde_json::to_vec(card).unwrap().len();
            card.as_object_mut().unwrap().remove("model_messages");
            size -= before - serde_json::to_vec(card).unwrap().len();
        }
    }
    catalog
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
        for slug in ["gpt-6-sol", "claude-opus-5"] {
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
    fn growing_catalog_removes_only_exact_prompt_duplicates_when_over_limit() {
        let known = templates()
            .iter()
            .filter(|model| {
                model["slug"] != "gpt-reserve" && model["slug"] != "gpt-5.3-codex-spark"
            })
            .collect::<Vec<_>>();
        let mut models = known
            .iter()
            .map(|model| model["slug"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        models.extend((0..34).map(|index| format!("custom-model-{index}")));
        let value = catalog(&models);
        let body = serde_json::to_vec(&value).unwrap();
        let cards = value["models"].as_array().unwrap();
        assert_eq!(cards.len(), 42);
        assert!(body.len() < 1024 * 1024, "catalog is {} bytes", body.len());
        let mut removed = 0;
        for original in known {
            let card = cards
                .iter()
                .find(|m| m["slug"] == original["slug"])
                .unwrap();
            let mut expected = original.clone();
            if card.get("model_messages").is_none() {
                removed += 1;
                let messages = expected
                    .as_object_mut()
                    .unwrap()
                    .remove("model_messages")
                    .unwrap();
                assert_eq!(messages["instructions_template"], card["base_instructions"]);
                for (field, value) in messages.as_object().unwrap() {
                    assert!(field == "instructions_template" || value.is_null());
                }
            }
            assert_eq!(card, &expected);
        }
        assert!(removed > 0, "fixture must exercise size compaction");
        let astra = cards.iter().find(|m| m["slug"] == "gpt-6-astra").unwrap();
        assert_eq!(astra["context_window"], 600000);
        assert_eq!(astra["max_context_window"], 872000);
        assert!(astra.get("model_messages").is_some());
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
