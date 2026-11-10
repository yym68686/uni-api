//! Keep candidate-specific metadata private until substantive Responses output.
//! The winning stream retains its original bytes, headers and observation clock.
use super::events::{responses_semantic_error, Terminal};
use super::processor::ResponsesProcessor;
use crate::observability::{stream::StreamStats, usage::FactUsage};
use crate::protocols::provider_stream::{PrecommitFailure, StreamOutcome, Translation};
use crate::protocols::sse::{SseDecoder, SseFrame};
use crate::routing::attempt_policy::ResponseStreamPolicy;
use crate::transport::{body::upstream_chunks, http::filtered_response_headers};
use axum::{body::Body, http::Response};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use std::{io, time::Duration};
use tokio::{sync::oneshot, time::Instant};

fn failure(reason: &'static str, status: u16, detail: impl Into<String>) -> PrecommitFailure {
    PrecommitFailure {
        failure_reason: Some(reason),
        status_code: status,
        detail: detail.into(),
    }
}

/// Stable diagnostics without retaining malformed JSON or upstream text.
pub(crate) fn protocol_error_code(detail: &str) -> &'static str {
    if detail.contains("without a terminal response event") {
        if detail.contains("[DONE]") {
            "done_without_terminal"
        } else {
            "eof_without_terminal"
        }
    } else if detail.starts_with("Responses upstream event JSON is invalid:") {
        "invalid_event_json"
    } else if detail.starts_with("Responses event field ") {
        "event_type_mismatch"
    } else if detail == "Responses upstream SSE event is not valid UTF-8" {
        "invalid_event_utf8"
    } else if detail == "Responses upstream event payload is missing a string type" {
        "missing_event_type"
    } else {
        "other_protocol_error"
    }
}

struct Observer {
    policy: ResponseStreamPolicy,
    wire: bytes::BytesMut,
    decoder: SseDecoder,
    processor: ResponsesProcessor,
    stats: StreamStats,
    committed: bool,
    terminal: bool,
    terminal_kind: Option<&'static str>,
    ended: bool,
    failure: Option<PrecommitFailure>,
    sender: Option<oneshot::Sender<StreamOutcome>>,
}

impl Observer {
    fn new(started: Instant, policy: ResponseStreamPolicy) -> Self {
        let mut stats = StreamStats::new("generic-responses");
        stats.started_at = started;
        Self {
            decoder: SseDecoder::new(0),
            processor: ResponsesProcessor::new(policy.commit_policy.clone(), policy.normalize_ids),
            policy,
            wire: bytes::BytesMut::new(),
            stats,
            committed: false,
            terminal: false,
            terminal_kind: None,
            ended: false,
            failure: None,
            sender: None,
        }
    }

    fn frames(&mut self, frames: Vec<SseFrame>) -> Result<(), PrecommitFailure> {
        for frame in frames {
            if self.terminal || self.failure.is_some() {
                break;
            }
            let event = self
                .processor
                .process(frame, &mut self.stats)
                .map_err(|detail| failure("protocol_error", 502, detail))?;
            if self.policy.normalize_ids {
                if let Some(wire) = &event.wire {
                    self.wire.extend_from_slice(wire);
                }
            }
            if let Some(terminal) = event.terminal.as_ref() {
                self.terminal = true;
                self.terminal_kind = Some(match terminal {
                    Terminal::Completed => "completed",
                    Terminal::Incomplete => "incomplete",
                    Terminal::SemanticFailure { .. } => "semantic_failure",
                });
                // A validated Responses terminal ends the logical stream.
                // HTTP EOF and later client cancellation cannot override it.
                self.ended = true;
            }
            if let Some(Terminal::SemanticFailure {
                event_type,
                payload,
            }) = event.terminal.as_ref()
            {
                let (status, detail) = responses_semantic_error(payload, event_type);
                return Err(failure(
                    if status == 400 {
                        "upstream_400"
                    } else {
                        "upstream_response_failed"
                    },
                    status,
                    detail,
                ));
            }
            self.committed |= event.commits;
            if matches!(event.terminal, Some(Terminal::Incomplete)) {
                return Err(failure(
                    match self.stats.incomplete_reason {
                        Some("max_output_tokens") => "responses_max_output_tokens",
                        Some("content_filter") => "responses_content_filter",
                        _ => "responses_incomplete",
                    },
                    200,
                    "Responses upstream ended incomplete after output",
                ));
            }
            if !self.committed && self.stats.event_count > self.policy.max_precommit_items as u64 {
                return Err(failure(
                    "protocol_error",
                    502,
                    "Responses precommit event limit exceeded",
                ));
            }
        }
        Ok(())
    }

    fn feed(&mut self, bytes: &[u8]) -> Result<(), PrecommitFailure> {
        if self.terminal || self.failure.is_some() {
            return Ok(());
        }
        let frames = self
            .decoder
            .feed(bytes)
            .map_err(|detail| failure("protocol_error", 502, detail))?;
        self.frames(frames)
    }

    fn finish(&mut self) -> Result<(), PrecommitFailure> {
        self.ended = true;
        if self.terminal || self.failure.is_some() {
            return Ok(());
        }
        let frames = self
            .decoder
            .finish()
            .map_err(|detail| failure("protocol_error", 502, detail))?;
        self.frames(frames)?;
        if !self.terminal {
            return Err(failure(
                "protocol_error",
                502,
                "Responses upstream ended without a terminal response event",
            ));
        }
        Ok(())
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        let Some(sender) = self.sender.take() else {
            return;
        };
        let success = self.ended && self.terminal && self.failure.is_none();
        let _ = sender.send(StreamOutcome {
            usage: (0, 0, 0),
            fact_usage: FactUsage::from_usage(self.stats.usage.as_ref()),
            success,
            observational_only: false,
            status_code: if !self.ended {
                499
            } else if success {
                200
            } else {
                self.failure.as_ref().map_or(502, |f| f.status_code)
            },
            detail: if !self.ended {
                "downstream disconnected".into()
            } else {
                self.failure
                    .as_ref()
                    .map_or_else(String::new, |f| f.detail.clone())
            },
            failure_reason: if !self.ended {
                Some("downstream_disconnected")
            } else {
                self.failure.as_ref().and_then(|f| f.failure_reason)
            },
            terminal_kind: self.terminal_kind.or_else(|| {
                Some(if !self.ended {
                    "cancelled"
                } else if self.failure.as_ref().and_then(|f| f.failure_reason)
                    == Some("transport_error")
                {
                    "transport_error"
                } else {
                    "protocol_error"
                })
            }),
            response_completed: Some(self.terminal_kind == Some("completed")),
            first_output_ms: self.stats.first_output_ms,
            response_created_ms: self.stats.response_created_ms,
            first_text_ms: self.stats.first_text_ms,
        });
    }
}

pub(crate) async fn prepare(
    response: reqwest::Response,
    started: Instant,
    deadline: Option<Instant>,
    idle: Option<Duration>,
    policy: ResponseStreamPolicy,
) -> Result<Translation, PrecommitFailure> {
    if !response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        })
    {
        return Err(failure(
            "protocol_error",
            502,
            "Responses stream upstream did not return text/event-stream",
        ));
    }
    let status = response.status();
    let mut headers = filtered_response_headers(response.headers());
    // Body adapters and heartbeat insertion can change HTTP transfer framing.
    headers.remove("content-length");
    let mut upstream = Box::pin(upstream_chunks(response, idle));
    let max_precommit_bytes = policy.max_precommit_bytes;
    let mut observer = Observer::new(started, policy);
    let mut retained = Vec::<Bytes>::new();
    let mut size = 0usize;
    loop {
        let next = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, upstream.next())
                .await
                .map_err(|_| failure("transport_error", 504, "Responses first output timed out"))?,
            None => upstream.next().await,
        };
        let observed = match next {
            Some(Ok(bytes)) => {
                size = size.saturating_add(bytes.len());
                let result = observer.feed(&bytes);
                if observer.policy.normalize_ids {
                    let wire = observer.wire.split().freeze();
                    if !wire.is_empty() {
                        retained.push(wire);
                    }
                } else {
                    retained.push(bytes);
                }
                result
            }
            Some(Err(error)) => return Err(failure("transport_error", 502, error.to_string())),
            None => observer.finish(),
        };
        if let Err(error) = observed {
            if !observer.committed {
                return Err(error);
            }
            observer.failure = Some(error);
            observer.ended = true;
        }
        if observer.committed {
            break;
        }
        if size > max_precommit_bytes {
            return Err(failure(
                "protocol_error",
                502,
                "Responses precommit byte limit exceeded",
            ));
        }
    }
    let (sender, outcome) = oneshot::channel();
    observer.sender = Some(sender);
    let tail = stream::unfold(
        (upstream, observer),
        |(mut upstream, mut observer)| async move {
            if observer.terminal || observer.failure.is_some() {
                return None;
            }
            match upstream.next().await {
                Some(result) => {
                    let inspected = match &result {
                        Ok(bytes) => observer.feed(bytes),
                        Err(error) => {
                            observer.ended = true;
                            Err(failure("transport_error", 502, error.to_string()))
                        }
                    };
                    if let Err(error) = inspected {
                        observer.failure = Some(error);
                        observer.ended = true;
                    }
                    let result = if observer.policy.normalize_ids {
                        result.map(|_| observer.wire.split().freeze())
                    } else {
                        result
                    };
                    Some((result, (upstream, observer)))
                }
                None => {
                    if let Err(error) = observer.finish() {
                        observer.failure = Some(error);
                    }
                    None
                }
            }
        },
    );
    let body = stream::iter(retained.into_iter().map(Ok::<_, io::Error>)).chain(tail);
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(Translation { response, outcome })
}

/// Apply the same commit boundary to provider-to-Responses translations. Their
/// existing outcome receiver continues to own billing and upstream observations.
pub(crate) async fn preflight_translation(
    translation: Translation,
    started: Instant,
    deadline: Option<Instant>,
    mut policy: ResponseStreamPolicy,
) -> Result<Translation, PrecommitFailure> {
    let Translation { response, outcome } = translation;
    let (parts, body) = response.into_parts();
    let mut body = body.into_data_stream();
    // The translator owns IDs and bytes; this layer only controls commitment.
    policy.normalize_ids = false;
    let max_precommit_bytes = policy.max_precommit_bytes;
    let mut observer = Observer::new(started, policy);
    let mut retained = Vec::new();
    let mut size = 0usize;
    loop {
        let next = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, body.next())
                .await
                .map_err(|_| failure("transport_error", 504, "Responses first output timed out"))?,
            None => body.next().await,
        };
        let inspected = match next {
            Some(Ok(bytes)) => {
                size = size.saturating_add(bytes.len());
                let inspected = observer.feed(&bytes);
                retained.push(bytes);
                inspected
            }
            Some(Err(error)) => Err(failure("transport_error", 502, error.to_string())),
            None => observer.finish(),
        };
        if observer.committed {
            break;
        }
        inspected?;
        if size > max_precommit_bytes {
            return Err(failure(
                "protocol_error",
                502,
                "Responses precommit byte limit exceeded",
            ));
        }
    }
    let stream = stream::iter(retained.into_iter().map(Ok::<_, axum::Error>)).chain(body);
    Ok(Translation {
        response: Response::from_parts(parts, Body::from_stream(stream)),
        outcome,
    })
}
