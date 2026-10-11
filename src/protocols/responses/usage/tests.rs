use super::*;
use crate::protocols::sse::SseDecoder;
use serde_json::json;

fn conflicting_usage() -> Value {
    json!({
        "input_tokens":11476,"output_tokens":107,"total_tokens":11583,
        "prompt_tokens":0,"completion_tokens":0,
        "input_tokens_details":{"cached_tokens":11264,"cache_write_tokens":32,"audio_tokens":4},
        "prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
        "output_tokens_details":{"reasoning_tokens":67,"audio_tokens":3},
        "completion_tokens_details":{"reasoning_tokens":0},
        "oaix_settlement_receipt":"fixture-receipt","vendor_extension":{"retained":true}
    })
}

#[test]
fn native_counters_and_details_win_over_conflicting_aliases() {
    let source = conflicting_usage();
    let mut usage = source.clone();
    assert!(normalize_aliases(&mut usage));
    for (native, alias) in ALIASES {
        assert_eq!(usage[native], source[native]);
        assert_eq!(usage[alias], source[native]);
    }
    assert!(!normalize_aliases(&mut usage));
    assert_eq!(usage["vendor_extension"], source["vendor_extension"]);
    let mapped = to_chat(Some(&source));
    assert_eq!(mapped["prompt_tokens"], 11476);
    assert_eq!(mapped["completion_tokens"], 107);
    assert_eq!(mapped["total_tokens"], 11583);
    assert_eq!(mapped["prompt_tokens_details"]["cached_tokens"], 11264);
    assert_eq!(mapped["prompt_tokens_details"]["cache_write_tokens"], 32);
    assert_eq!(mapped["completion_tokens_details"]["reasoning_tokens"], 67);
    assert_eq!(mapped["oaix_settlement_receipt"], "fixture-receipt");
    assert!(mapped.get("vendor_extension").is_none());
    assert!(mapped.get("input_tokens").is_none());
    assert_eq!(
        crate::observability::usage::FactUsage::from_usage(Some(&mapped)).cache_read,
        Some(11264)
    );
}

#[test]
fn zero_is_authoritative_but_missing_null_invalid_and_overflow_are_not_zero() {
    let mut zero =
        json!({"input_tokens":0,"output_tokens":0,"prompt_tokens":9,"completion_tokens":8});
    assert!(normalize_aliases(&mut zero));
    assert_eq!(
        to_chat(Some(&zero)),
        json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0})
    );
    assert_eq!(to_chat(None), Value::Null);
    assert_eq!(to_chat(Some(&Value::Null)), Value::Null);
    assert_eq!(to_chat(Some(&json!({}))), json!({}));
    assert_eq!(
        to_chat(Some(&json!({"input_tokens":7}))),
        json!({"prompt_tokens":7})
    );
    assert_eq!(
        to_chat(Some(&json!({"total_tokens":7}))),
        json!({"total_tokens":7})
    );
    for invalid in [Value::Null, json!(-1), json!(1.5), json!("7"), json!(true)] {
        let mut usage = json!({"input_tokens":invalid,"prompt_tokens":9,"output_tokens":2});
        assert!(!normalize_aliases(&mut usage));
        let mapped = to_chat(Some(&usage));
        assert_eq!(mapped["prompt_tokens"], invalid);
        assert!(mapped.get("total_tokens").is_none());
    }
    let large = to_chat(Some(&json!({"input_tokens":i64::MAX,"output_tokens":1})));
    assert!(large.get("total_tokens").is_none());
}

#[test]
fn alias_only_usage_remains_compatible_without_fabricating_details() {
    let usage = json!({"prompt_tokens":6,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":5}});
    let mapped = to_chat(Some(&usage));
    assert_eq!(mapped["total_tokens"], 7);
    assert_eq!(mapped["prompt_tokens_details"]["cached_tokens"], 5);
    assert!(mapped.get("completion_tokens_details").is_none());
    let mut payload = json!({"response":{"usage":usage}});
    assert!(!normalize_payload(&mut payload));
}

#[test]
fn sse_repair_preserves_metadata_boundaries_and_arbitrary_chunk_splits() {
    for newline in ["\n", "\r\n", "\r"] {
        for terminated in [true, false] {
            let event = json!({"type":"response.completed","sequence_number":42,"response":{"status":"completed","usage":conflicting_usage(),"output":[{"text":"unchanged"}]}});
            // Multiline data is legal SSE. Preserve id/retry/comment fields.
            let data = serde_json::to_string(&event).unwrap().replacen(
                "\"response\":",
                &format!("{newline}data: \"response\":"),
                1,
            );
            let raw = format!("id: usage-42{newline}retry: 123{newline}: diagnostic{newline}event: response.completed{newline}data: {data}");
            let wire = format!(
                "{raw}{}",
                if terminated {
                    newline.repeat(2)
                } else {
                    String::new()
                }
            );
            for split in 1..=wire.len() {
                let mut decoder = SseDecoder::new(0);
                let mut frames = decoder.feed(&wire.as_bytes()[..split]).unwrap();
                frames.extend(decoder.feed(&wire.as_bytes()[split..]).unwrap());
                frames.extend(decoder.finish().unwrap());
                assert_eq!(frames.len(), 1);
                let (frame, changed) = normalize_frame(frames.pop().unwrap());
                assert!(changed);
                assert_eq!(frame.terminated, terminated);
                assert!(frame.wire.starts_with(
                    format!("id: usage-42{newline}retry: 123{newline}: diagnostic{newline}")
                        .as_bytes()
                ));
                let payload: Value =
                    serde_json::from_str(&parse_sse_frame(&frame).unwrap().data.unwrap()).unwrap();
                assert_eq!(payload["sequence_number"], 42);
                assert_eq!(payload["response"]["output"], event["response"]["output"]);
                assert_eq!(payload["response"]["usage"]["prompt_tokens"], 11476);
                assert!(!normalize_frame(frame).1);
            }
        }
    }
}

#[test]
fn unrelated_or_unreported_sse_is_unchanged() {
    for raw in [
        ": keepalive\r\n\r\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"usage prompt_tokens\"}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":null}}\n\n",
        "event: response.completed\ndata: {broken}\n\n",
        "data: [DONE]\n\n",
    ] {
        let frame = SseDecoder::new(0).feed(raw.as_bytes()).unwrap().pop().unwrap();
        let (frame, changed) = normalize_frame(frame);
        assert!(!changed);
        assert_eq!(frame.wire, raw.as_bytes());
    }
}
