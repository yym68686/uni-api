use crate::api::gateway::known_path;
use crate::api::gateway::missing_required_field;
use crate::api::gateway::moderation_text;
use crate::api::gateway::supports;
use crate::config::snapshot::Provider;
use crate::protocols::conversion::chat_to_cohere;
use crate::protocols::conversion::chat_to_doubao_translation;
use crate::protocols::conversion::chat_to_responses;
use crate::protocols::conversion::chat_to_responses_response;
use crate::protocols::conversion::normalize_jina_embedding;
use crate::protocols::conversion::normalize_search_response;
use crate::protocols::conversion::openai_tts_to_minimax;
use crate::protocols::conversion::responses_to_chat;
use crate::providers::anthropic::chat_to_claude;
use crate::providers::anthropic::claude_to_chat;
use crate::providers::cloud::aws_bedrock_url;
use crate::providers::cloud::azure_chat_url;
use crate::providers::cloud::cloudflare_url;
use crate::providers::cloud::databricks_chat_url;
use crate::providers::cloud::sign_aws_request_at;
use crate::providers::cloud::vertex_claude_url;
use crate::providers::codex::CODEX_USER_AGENT;
use crate::providers::gemini::cache_gemini_image_thought_signature;
use crate::providers::gemini::chat_to_gemini;
use crate::providers::gemini::data_url_part;
use crate::providers::gemini::decode_gemini_thought_signature;
use crate::providers::gemini::gemini_to_chat;
use crate::providers::request::build_attempt;
use crate::providers::request::provider_headers;
use crate::providers::request::ALPHA_SEARCH_ENDPOINT;
use crate::providers::types::AttemptBody;
use crate::providers::types::DownstreamProtocol;
use crate::providers::types::PreparedInput;
use crate::providers::types::ResponseAdapter;
use crate::providers::video::callxyq_video_payload;
use crate::providers::video::content_generation_to_lingjing;
use crate::providers::video::normalize_callxyq_video_response;
use crate::providers::video::normalize_lingjing_video_response;
use crate::providers::video::remember_video_task;
use crate::providers::video::video_task_route_for_path;
use crate::transport::multipart::multipart_rewrite_stream;
use crate::transport::spool::SpoolObservation;
use crate::upstream::generic::chat_nonstream_hedging_enabled;
use crate::upstream::hedging::HedgingConfig;
use crate::upstream::input::inspect_image_media_type;
use crate::upstream::input::ImageMediaTypeInspection;
use axum::http::{HeaderMap, Method, Uri};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::{Duration, UNIX_EPOCH};

const SINGLE_FRAME_GIF_BASE64: &str =
    "R0lGODlhAgACAIEAAP8AAAAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQACgAAACwAAAAAAgACAAAIBgABCAQQEAA7";

const ANIMATED_GIF_BASE64: &str =
    "R0lGODlhAgACAIEAAP8AAAAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQACgAAACwAAAAAAgACAAAIBgABCAQQEAAh+QQACgAAACwAAAAAAgACAIEAAP8AAAAAAAAAAAAIBgABCAQQEAA7";

#[test]
fn image_inspection_accepts_single_frame_gif() {
    let bytes = BASE64.decode(SINGLE_FRAME_GIF_BASE64).unwrap();
    assert_eq!(
        inspect_image_media_type(&bytes),
        ImageMediaTypeInspection::Supported("image/gif")
    );
}

#[test]
fn image_inspection_rejects_animated_gif() {
    let bytes = BASE64.decode(ANIMATED_GIF_BASE64).unwrap();
    assert_eq!(
        inspect_image_media_type(&bytes),
        ImageMediaTypeInspection::AnimatedGif
    );
}

#[test]
fn image_inspection_rejects_truncated_gif() {
    let mut bytes = BASE64.decode(SINGLE_FRAME_GIF_BASE64).unwrap();
    bytes.pop();
    assert_eq!(
        inspect_image_media_type(&bytes),
        ImageMediaTypeInspection::InvalidGif
    );
}

#[test]
fn hedging_applies_only_to_nonstream_chat_completions() {
    let enabled = HedgingConfig {
        enabled: true,
        max_inflight_attempts: 2,
        winner_policy: crate::upstream::hedging::WinnerPolicy::FirstValidSuccess,
    };
    assert!(chat_nonstream_hedging_enabled(
        "/v1/chat/completions",
        Some(&json!({"stream": false})),
        enabled,
    ));
    assert!(chat_nonstream_hedging_enabled(
        "/v1/chat/completions",
        Some(&json!({})),
        enabled,
    ));
    assert!(!chat_nonstream_hedging_enabled(
        "/v1/chat/completions",
        Some(&json!({"stream": true})),
        enabled,
    ));
    assert!(!chat_nonstream_hedging_enabled(
        "/v1/responses",
        Some(&json!({"stream": false})),
        enabled,
    ));
    assert!(!chat_nonstream_hedging_enabled(
        "/v1/chat/completions",
        Some(&json!({"stream": false})),
        HedgingConfig::default(),
    ));
}

fn test_provider(engine: &str, base_url: &str) -> Provider {
    Provider {
        name: "provider-a".into(),
        base_url: base_url.to_owned().into(),
        engine: engine.to_owned().into(),
        api_keys: std::sync::Arc::new(vec!["upstream-key".into()]),
        project_id: None,
        private_key: None,
        client_email: None,
        aws_access_key: None,
        aws_secret_key: None,
        aws_session_token: None,
        cf_account_id: None,
        region: "global".into(),
        models: std::sync::Arc::new(HashMap::new()),
        preferences: std::sync::Arc::new(Map::new()),
        excluded_endpoints: std::sync::Arc::new(Vec::new()),
        only_request_types: std::sync::Arc::new(Vec::new()),
        excluded_request_types: std::sync::Arc::new(Vec::new()),
        excluded_request_rules: std::sync::Arc::new(Vec::new()),
        cursor: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    }
}

#[test]
fn every_python_model_route_has_a_native_dispatch() {
    for route in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/embeddings",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/moderations",
        "/v1/video/tasks",
        "/v1/asset-groups",
        "/v1/assets",
        "/v1/alpha/search",
        "/v1/responses",
    ] {
        assert!(supports(&Method::POST, route), "missing {route}");
    }
    assert!(supports(&Method::GET, "/v1/search"));
    assert!(supports(&Method::GET, "/v1/video/tasks/task-1"));
}

#[test]
fn codex_chat_is_compiled_to_responses_wire() {
    let input = json!({
        "model":"gpt-public",
        "messages":[
            {"role":"system","content":"be concise"},
            {"role":"user","content":"hello"}
        ],
        "max_tokens":42,
        "stream":true
    });
    let output = chat_to_responses(&input, "gpt-upstream").unwrap();
    assert_eq!(output["model"], "gpt-upstream");
    assert_eq!(output["max_output_tokens"], 42);
    assert_eq!(output["stream"], false);
    assert_eq!(output["input"][1]["content"][0]["type"], "input_text");
}

#[test]
fn chat_to_responses_preserves_tool_constraints() {
    let input = json!({
        "messages":[{"role":"user","content":"What time is it?"}],
        "tools":[
            {
                "type":"function",
                "function":{
                    "name":"now",
                    "parameters":{"type":"object","properties":{}}
                }
            },
            {
                "type":"function",
                "function":{
                    "name":"weather",
                    "parameters":{"type":"object","properties":{}},
                    "strict":true
                }
            }
        ],
        "tool_choice":{"type":"function","function":{"name":"now"}}
    });

    let output = chat_to_responses(&input, "gpt-upstream").unwrap();

    assert_eq!(output["tools"][0]["strict"], false);
    assert_eq!(output["tools"][1]["strict"], true);
    assert_eq!(
        output["tool_choice"],
        json!({"type":"function","name":"now"})
    );
}

#[test]
fn chat_to_responses_flattens_allowed_tools_choice() {
    let input = json!({
        "messages":[{"role":"user","content":"Use a tool"}],
        "tool_choice":{
            "type":"allowed_tools",
            "allowed_tools":{
                "mode":"required",
                "tools":[
                    {"type":"function","function":{"name":"now"}},
                    {"type":"function","name":"weather"}
                ]
            }
        }
    });

    let output = chat_to_responses(&input, "gpt-upstream").unwrap();

    assert_eq!(
        output["tool_choice"],
        json!({
            "type":"allowed_tools",
            "mode":"required",
            "tools":[
                {"type":"function","name":"now"},
                {"type":"function","name":"weather"}
            ]
        })
    );
}

#[test]
fn responses_to_chat_maps_tool_finish_and_usage() {
    let response = responses_to_chat(
        &json!({
            "id":"resp-a",
            "output":[{
                "type":"function_call",
                "id":"fc-a",
                "call_id":"call-a",
                "name":"now",
                "arguments":"{\"timezone\":\"Europe/Berlin\"}"
            }],
            "usage":{
                "input_tokens":3,
                "output_tokens":5,
                "total_tokens":8,
                "input_tokens_details":{"cached_tokens":2,"cache_write_tokens":1},
                "output_tokens_details":{"reasoning_tokens":4}
            }
        }),
        "public-model",
    );

    assert_eq!(response["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["id"],
        "call-a"
    );
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(response["usage"]["completion_tokens"], 5);
    assert_eq!(response["usage"]["total_tokens"], 8);
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"],
        2
    );
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cache_write_tokens"],
        1
    );
    assert_eq!(
        response["usage"]["completion_tokens_details"]["reasoning_tokens"],
        4
    );
}

#[test]
fn gemini_and_claude_payloads_preserve_system_and_tools() {
    let input = json!({
        "messages":[
            {"role":"system","content":"system"},
            {"role":"user","content":"hello"}
        ],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
    });
    let gemini = chat_to_gemini(&input, "gemini-upstream").unwrap();
    assert_eq!(gemini["systemInstruction"]["parts"][0]["text"], "system");
    assert_eq!(
        gemini["tools"][0]["functionDeclarations"][0]["name"],
        "lookup"
    );
    let claude = chat_to_claude(&input, "claude-upstream").unwrap();
    assert_eq!(claude["system"], "system");
    assert_eq!(claude["tools"][0]["name"], "lookup");
}

#[test]
fn responses_compat_compiles_to_gemini_and_restores_responses_shape() {
    let provider = test_provider("gemini", "https://generativelanguage.googleapis.com/v1beta");
    let input = PreparedInput {
        payload: Some(json!({
            "model":"public-model",
            "instructions":"be concise",
            "input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],
            "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
            "tool_choice":{"type":"function","name":"lookup"},
            "reasoning":{"effort":"high"},
            "service_tier":"priority",
            "stream":true
        })),
        replay: None,
        observation: SpoolObservation::default(),
        default_model: String::new(),
        content_type: "application/json".into(),
    };
    let uri: Uri = "/v1/responses".parse().unwrap();
    let prepared = build_attempt(
        &provider,
        "key",
        "public-model",
        "gemini-2.5-pro",
        &Method::POST,
        &uri,
        "/v1/responses",
        &HeaderMap::new(),
        &input,
        "request-a",
    )
    .unwrap();
    assert_eq!(
        prepared.downstream_protocol,
        DownstreamProtocol::ResponsesCompat
    );
    assert!(prepared.downstream_stream);
    assert!(prepared.upstream_stream);
    assert_eq!(prepared.adapter, ResponseAdapter::GeminiToChat);
    let AttemptBody::Json(body) = prepared.body else {
        panic!("responses compatibility request must use JSON");
    };
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be concise");
    assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
    assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    assert_eq!(body["serviceTier"], "PRIORITY");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        24576
    );

    let response = chat_to_responses_response(
        &json!({
            "choices":[{"message":{"role":"assistant","content":"done","tool_calls":[{"id":"call-a","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}],
            "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
        }),
        "public-model",
    );
    assert_eq!(response["object"], "response");
    assert_eq!(response["model"], "public-model");
    assert_eq!(response["output"][0]["content"][0]["text"], "done");
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["usage"]["total_tokens"], 5);
}

#[test]
fn search_responses_are_normalized_without_losing_provider_fields() {
    let normalized = normalize_search_response(
        "https://api.tavily.com/search",
        &json!({
            "query":"rust",
            "results":[{"title":"Rust","url":"https://rust-lang.org","content":"language","score":0.9}],
            "request_id":"search-a"
        }),
    );
    assert_eq!(normalized["code"], 200);
    assert_eq!(normalized["data"][0]["description"], "language");
    assert_eq!(normalized["data"][0]["score"], 0.9);
    assert_eq!(normalized["meta"]["provider"], "tavily");
    assert_eq!(normalized["meta"]["request_id"], "search-a");
}

#[test]
fn gemini_thought_signatures_round_trip_for_images_and_tools() {
    let encoded = BASE64.encode(b"\x89PNG\r\n\x1a\nimage");
    cache_gemini_image_thought_signature(&encoded, "image-signature");
    let part = data_url_part(&format!("data:image/png;base64,{encoded}")).unwrap();
    assert_eq!(part["thoughtSignature"], "image-signature");

    let chat = gemini_to_chat(
        &json!({
            "candidates":[{"content":{"parts":[{
                "functionCall":{"name":"lookup","args":{"q":"rust"}},
                "thoughtSignature":"tool-signature"
            }]}}]
        }),
        "gemini-public",
    );
    let call_id = chat["choices"][0]["message"]["tool_calls"][0]["id"]
        .as_str()
        .unwrap();
    assert_eq!(
        decode_gemini_thought_signature(call_id).as_deref(),
        Some("tool-signature")
    );
}

#[test]
fn gemini_audio_and_claude_controls_are_translated() {
    let input = json!({
        "messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
        "tool_choice":"required",
        "reasoning_effort":"medium",
        "service_tier":"flex",
        "modalities":["audio"],
        "audio":{"voice":"Aoede"}
    });
    let gemini = chat_to_gemini(&input, "gemini-3-flash").unwrap();
    assert_eq!(
        gemini["contents"][0]["parts"][0]["inlineData"]["mimeType"],
        "audio/wav"
    );
    assert_eq!(
        gemini["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
            ["voiceName"],
        "Aoede"
    );
    let claude = chat_to_claude(&input, "claude-sonnet").unwrap();
    assert_eq!(claude["tool_choice"]["type"], "any");
    assert_eq!(claude["thinking"]["budget_tokens"], 4096);
    assert_eq!(claude["service_tier"], "standard_only");
}

#[tokio::test]
async fn multipart_media_is_parsed_and_rebuilt_with_the_upstream_model() {
    let source_boundary = "CaseSensitiveBoundary";
    let source = format!(
        "--{source_boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit this\r\n--{source_boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.bin\"\r\nContent-Type: application/octet-stream\r\n\r\nbinary\0payload\r\n--{source_boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\npublic-model\r\n--{source_boundary}--\r\n"
    );
    let manager = crate::transport::spool::SpoolManager::new(
        crate::runtime::resources::ResourceGovernor::unconstrained_for_test(),
    )
    .unwrap();
    let mut writer = manager
        .begin(None, Some(source.len() as u64), Duration::ZERO)
        .await
        .unwrap();
    writer.append(Bytes::from(source)).await.unwrap();
    let spool = writer.finish().await.unwrap();
    let content_type = format!("multipart/form-data; boundary={source_boundary}");
    assert_eq!(
        spool
            .storage
            .multipart_text_field(&content_type, "model", 4096)
            .await
            .unwrap()
            .as_deref(),
        Some("public-model")
    );
    let output_boundary = "rewritten-boundary".to_owned();
    let stream = multipart_rewrite_stream(
        spool.storage,
        spool.observation,
        &content_type,
        output_boundary.clone(),
        "upstream-model".into(),
    )
    .await
    .unwrap();
    let mut multipart = multer::Multipart::new(stream, output_boundary);
    let mut fields = HashMap::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_owned();
        fields.insert(name, field.bytes().await.unwrap());
    }
    assert_eq!(fields["prompt"], "edit this");
    assert_eq!(fields["model"], "upstream-model");
    assert_eq!(fields["image"], b"binary\0payload"[..]);
}

#[test]
fn upstream_responses_are_normalized_to_openai_chat() {
    let gemini = gemini_to_chat(
        &json!({
            "candidates":[{"content":{"parts":[{"text":"hello"}]}}],
            "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}
        }),
        "gemini-public",
    );
    assert_eq!(gemini["choices"][0]["message"]["content"], "hello");
    assert_eq!(gemini["usage"]["total_tokens"], 5);
    let claude = claude_to_chat(
        &json!({"content":[{"type":"text","text":"world"}],"usage":{"input_tokens":1,"output_tokens":2}}),
        "claude-public",
    );
    assert_eq!(claude["choices"][0]["message"]["content"], "world");
    assert_eq!(claude["usage"]["total_tokens"], 3);
}

#[test]
fn special_provider_urls_and_payloads_match_legacy_contracts() {
    assert_eq!(
        azure_chat_url("https://azure.example.com", "deployment-a").unwrap(),
        "https://azure.example.com/openai/deployments/deployment-a/chat/completions?api-version=2025-01-01-preview"
    );
    assert_eq!(
        databricks_chat_url("https://dbc.example.com", "serving-a").unwrap(),
        "https://dbc.example.com/serving-endpoints/serving-a/invocations"
    );
    let cohere = chat_to_cohere(
        &json!({"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hello"}]}),
        "command-r",
    )
    .unwrap();
    assert_eq!(cohere["message"], "hello");
    assert_eq!(cohere["chat_history"][0]["role"], "SYSTEM");

    let mut cloudflare = test_provider("cloudflare", "https://api.cloudflare.com");
    cloudflare.cf_account_id = Some("account-a".into());
    assert_eq!(
        cloudflare_url(&cloudflare, "@cf/meta/llama").unwrap(),
        "https://api.cloudflare.com/client/v4/accounts/account-a/ai/run/@cf/meta/llama"
    );
}

#[test]
fn callxyq_sora_and_veo_payloads_match_python_validation_contract() {
    let sora = callxyq_video_payload(
        &json!({"prompt":"make a film","model":"sora-2","duration":8,"aspect_ratio":"16:9"}),
        "sora-2",
    )
    .unwrap();
    assert_eq!(sora["model"], "sora-2");
    assert_eq!(sora["seconds"], 8);
    assert!(callxyq_video_payload(&json!({"prompt":"x","duration":3}), "sora-2").is_err());
    let veo = callxyq_video_payload(
        &json!({"prompt":"make a film","aspect_ratio":"9:16","resolution":"1080p"}),
        "gemini-veo-3-8s",
    )
    .unwrap();
    assert_eq!(veo["size"], "1080x1920");
    assert!(callxyq_video_payload(
        &json!({"prompt":"x","content":[{"type":"video_url","video_url":"https://x"}]}),
        "gemini-veo-3-8s"
    )
    .is_err());
}

#[test]
fn callxyq_response_normalization_matches_unified_video_shape() {
    let created = normalize_callxyq_video_response(
        &Method::POST,
        "sora-2",
        &json!({"task_id":"t1","status":"queued"}),
        Some(12),
    );
    assert_eq!(created["id"], "t1");
    assert_eq!(created["status"], "queued");
    let completed = normalize_callxyq_video_response(
        &Method::GET,
        "sora-2",
        &json!({"id":"t1","status":"completed","video_url":"https://cdn/video.mp4","size":"1280x720"}),
        Some(12),
    );
    assert_eq!(completed["status"], "succeeded");
    assert_eq!(completed["video"]["url"], "https://cdn/video.mp4");
    assert_eq!(completed["usage"]["video_tokens"], 12);
}

#[test]
fn lingjing_video_contract_and_task_affinity_match_legacy_runtime() {
    let converted = content_generation_to_lingjing(
        &json!({
            "model":"seedance-2-0",
            "prompt":"sunlight",
            "resources":[{"type":"image","url":"asset://Asset-test","role":"first_frame"}],
            "duration":5,
            "resolution":"720p",
            "ratio":"16:9",
            "generate_audio":false,
        }),
        "sd_2_0",
    )
    .unwrap();
    assert_eq!(converted["modelCode"], "sd_2_0");
    assert_eq!(converted["taskParams"]["input"]["quality"], "720");
    assert_eq!(
        converted["taskParams"]["input"]["resources"][0]["source"],
        json!({"kind":"asset_id","value":"Asset-test"})
    );

    let mut provider = test_provider("lingjing", "https://api-llm.lingjingai.cn");
    provider.preferences = std::sync::Arc::new(Map::from_iter([
        ("access_key".into(), json!("ak-test")),
        ("secret_key".into(), json!("sk-test")),
    ]));
    let headers = provider_headers(
        &provider,
        "routing-key",
        &HeaderMap::new(),
        "request-a",
        "lingjing",
        false,
        None,
    )
    .unwrap();
    assert_eq!(headers["x-access-key"], "ak-test");
    assert_eq!(headers["x-secret-key"], "sk-test");
    assert!(headers.get("authorization").is_none());

    remember_video_task(
        "task-rust-lingjing",
        "lingjing",
        "seedance-2-0",
        "routing-key",
        Some(108_900),
    );
    let route = video_task_route_for_path("/v1/video/tasks/task-rust-lingjing").unwrap();
    assert_eq!(route.provider_name, "lingjing");
    assert_eq!(route.provider_key, "routing-key");
    let normalized = normalize_lingjing_video_response(
        &Method::GET,
        "seedance-2-0",
        "https://api-llm.lingjingai.cn/draw/task/query?taskId=task-rust-lingjing",
        &json!({"data":{"task_id":"task-rust-lingjing","status":"SUCCESS","result":[{"url":"https://example.com/out.mp4"}]}}),
    );
    assert_eq!(normalized["status"], "succeeded");
    assert_eq!(normalized["video"]["url"], "https://example.com/out.mp4");
    assert_eq!(normalized["usage"]["video_tokens"], 108_900);
}

#[test]
fn long_tail_payload_adapters_match_python_contracts() {
    let mut doubao = test_provider("doubao-translation", "https://example.com/responses");
    doubao.preferences = std::sync::Arc::new(Map::from_iter([(
        "post_body_parameter_overrides".into(),
        json!({"translate-public":{"translation_options":{"source_language":"en","target_language":"ja"}}}),
    )]));
    let translated = chat_to_doubao_translation(
        &json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"stream":true}),
        "doubao-upstream",
        "translate-public",
        &doubao,
    )
    .unwrap();
    assert_eq!(translated["input"][0]["content"][0]["text"], "hello");
    assert_eq!(
        translated["input"][0]["content"][0]["translation_options"]["target_language"],
        "ja"
    );
    assert_eq!(translated["stream"], true);

    let minimax = openai_tts_to_minimax(
        &json!({"input":"speak","voice":"alloy","speed":1.25}),
        "speech-02-hd",
    )
    .unwrap();
    assert_eq!(minimax["text"], "speak");
    assert_eq!(minimax["voice_setting"]["voice_id"], "alloy");

    let mut jina = json!({"input":"text","encoding_format":"float"});
    normalize_jina_embedding(&mut jina, "jina-embeddings-v3").unwrap();
    assert_eq!(jina["embedding_type"], "float");
    assert!(jina.get("encoding_format").is_none());
}

#[test]
fn compact_and_validation_routes_are_native() {
    assert!(supports(&Method::POST, "/v1/responses/compact"));
    assert!(known_path("/v1/responses/compact"));
    assert_eq!(
        missing_required_field("/v1/chat/completions", &json!({"model":"gpt"})),
        Some("messages")
    );
    assert_eq!(
        missing_required_field(
            "/v1/chat/completions",
            &json!({"model":"gpt","messages":[{"role":"user","content":"hi"}]})
        ),
        None
    );
}

#[test]
fn provider_headers_use_protocol_specific_authentication() {
    let request_headers = HeaderMap::new();
    let azure = test_provider("azure", "https://azure.example.com");
    let headers = provider_headers(
        &azure,
        "azure-key",
        &request_headers,
        "request-a",
        "azure",
        false,
        None,
    )
    .unwrap();
    assert_eq!(headers["api-key"], "azure-key");
    assert!(headers.get("authorization").is_none());

    let openrouter = test_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
    );
    let headers = provider_headers(
        &openrouter,
        "key",
        &request_headers,
        "request-a",
        "openrouter",
        false,
        None,
    )
    .unwrap();
    assert_eq!(
        headers["http-referer"],
        "https://github.com/yym68686/uni-api"
    );
    assert_eq!(headers["x-title"], "Uni API");

    let requesty = test_provider("requesty", "https://router.requesty.ai/v1/chat/completions");
    let headers = provider_headers(
        &requesty,
        "key",
        &request_headers,
        "request-a",
        "requesty",
        false,
        None,
    )
    .unwrap();
    assert_eq!(headers["authorization"], "Bearer key");
    assert_eq!(
        headers["http-referer"],
        "https://github.com/yym68686/uni-api"
    );
    assert_eq!(headers["x-title"], "Uni API");
}

#[test]
fn alpha_search_strips_responses_fields_and_skips_provider_overrides() {
    let mut provider = test_provider("codex", "https://example.com/v1/responses");
    let mut preferences = Map::new();
    preferences.insert(
        "post_body_parameter_overrides".into(),
        json!({"store": false, "metadata": {"source": "generic-override"}}),
    );
    provider.preferences = std::sync::Arc::new(preferences);
    let input = PreparedInput {
        payload: Some(json!({
            "id": "search-session-a",
            "model": "gpt-public",
            "input": "query",
            "commands": [],
            "settings": {},
            "max_output_tokens": 256,
            "store": true,
            "stream": true,
            "prompt_cache_key": "cache-a",
            "prompt_cache_retention": "24h",
            "future_search_field": {"enabled": true}
        })),
        replay: None,
        observation: SpoolObservation::default(),
        default_model: String::new(),
        content_type: "application/json".into(),
    };
    let uri: Uri = ALPHA_SEARCH_ENDPOINT.parse().unwrap();
    let prepared = build_attempt(
        &provider,
        "key",
        "gpt-public",
        "gpt-upstream",
        &Method::POST,
        &uri,
        ALPHA_SEARCH_ENDPOINT,
        &HeaderMap::new(),
        &input,
        "request-a",
    )
    .unwrap();
    let AttemptBody::Json(body) = prepared.body else {
        panic!("alpha/search must use a JSON body");
    };
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(prepared.url, "https://example.com/v1/alpha/search");
    assert!(!prepared.downstream_stream);
    assert!(!prepared.upstream_stream);
    assert_eq!(body["model"], "gpt-upstream");
    assert_eq!(body["future_search_field"]["enabled"], true);
    for field in [
        "store",
        "stream",
        "prompt_cache_key",
        "prompt_cache_retention",
        "metadata",
    ] {
        assert!(body.get(field).is_none(), "unexpected field {field}");
    }
    assert_eq!(prepared.headers["openai-beta"], "responses=experimental");
    assert_eq!(prepared.headers["originator"], "codex_cli_rs");
    assert_eq!(prepared.headers["session_id"], "search-session-a");
    assert_eq!(prepared.headers["user-agent"], CODEX_USER_AGENT);
    assert_eq!(prepared.headers["accept"], "application/json");
}

#[test]
fn non_alpha_routes_still_apply_provider_overrides() {
    let mut provider = test_provider("codex", "https://example.com/v1/responses");
    let mut preferences = Map::new();
    preferences.insert(
        "post_body_parameter_overrides".into(),
        json!({"store": false}),
    );
    provider.preferences = std::sync::Arc::new(preferences);
    let input = PreparedInput {
        payload: Some(json!({"model": "gpt-public", "input": "hello"})),
        replay: None,
        observation: SpoolObservation::default(),
        default_model: String::new(),
        content_type: "application/json".into(),
    };
    let uri: Uri = "/v1/moderations".parse().unwrap();
    let prepared = build_attempt(
        &provider,
        "key",
        "gpt-public",
        "gpt-upstream",
        &Method::POST,
        &uri,
        "/v1/moderations",
        &HeaderMap::new(),
        &input,
        "request-a",
    )
    .unwrap();
    let AttemptBody::Json(body) = prepared.body else {
        panic!("moderations must use a JSON body");
    };
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], false);
}

#[test]
fn aws_bedrock_request_is_signed_with_sigv4() {
    let mut provider = test_provider("aws", "https://bedrock-runtime.us-east-1.amazonaws.com");
    provider.aws_access_key = Some("AKIA_TEST".into());
    provider.aws_secret_key = Some("secret".into());
    let url = aws_bedrock_url(&provider, "anthropic.claude-3-haiku:0", true).unwrap();
    let mut headers = HeaderMap::new();
    sign_aws_request_at(
        &provider,
        &url,
        br#"{"messages":[]}"#,
        &mut headers,
        UNIX_EPOCH,
    )
    .unwrap();
    assert_eq!(headers["x-amz-date"], "19700101T000000Z");
    assert!(headers["authorization"].to_str().unwrap().starts_with(
        "AWS4-HMAC-SHA256 Credential=AKIA_TEST/19700101/us-east-1/bedrock/aws4_request"
    ));
    assert_eq!(
        headers["accept"],
        "application/vnd.amazon.bedrock.payload+json"
    );
}

#[test]
fn vertex_claude_uses_project_and_region() {
    let mut provider = test_provider("vertex-claude", "https://aiplatform.googleapis.com");
    provider.project_id = Some("project-a".into());
    provider.region = "europe-west1".into();
    assert_eq!(
        vertex_claude_url(&provider, "claude-sonnet-4-5@20250929").unwrap(),
        "https://europe-west1-aiplatform.googleapis.com/v1/projects/project-a/locations/europe-west1/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict"
    );
}

#[test]
fn moderation_extracts_the_legacy_last_text_shapes() {
    assert_eq!(
        moderation_text(&json!({
            "messages":[
                {"role":"user","content":"earlier"},
                {"role":"assistant","content":[{"type":"text","text":"latest"}]}
            ]
        }))
        .as_deref(),
        Some("latest")
    );
    assert_eq!(
        moderation_text(&json!({"model":"text-embedding-3-small","input":["one","two"]}))
            .as_deref(),
        Some("one\ntwo")
    );
    assert_eq!(
        moderation_text(&json!({
            "input":[{"role":"user","content":[
                {"type":"input_text","text":"first"},
                {"type":"input_text","text":"last"}
            ]}]
        }))
        .as_deref(),
        Some("last")
    );
}
