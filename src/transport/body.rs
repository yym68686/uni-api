use crate::runtime::idempotency::RequestHasher;
use crate::transport::spool::{RequestSpool, SpoolFailure, SpoolManager};
use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use std::time::Duration;

pub(crate) enum RequestBodySpoolError {
    Timeout,
    Read,
    Spool(SpoolFailure),
}

pub(crate) async fn read_spooled_body(
    body: Body,
    manager: &SpoolManager,
    request_hasher: Option<RequestHasher>,
    content_length: Option<u64>,
    initial_wait: Duration,
) -> Result<RequestSpool, RequestBodySpoolError> {
    let mut stream = body.into_data_stream();
    let mut spool = manager
        .begin(request_hasher, content_length, initial_wait)
        .await
        .map_err(RequestBodySpoolError::Spool)?;
    let idle_timeout = Duration::from_secs_f64(
        std::env::var("REQUEST_BODY_IDLE_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(15.0),
    );
    loop {
        let next = tokio::time::timeout(idle_timeout, stream.next())
            .await
            .map_err(|_| RequestBodySpoolError::Timeout)?;
        let Some(chunk) = next else {
            return spool.finish().await.map_err(RequestBodySpoolError::Spool);
        };
        let chunk = chunk.map_err(|_| RequestBodySpoolError::Read)?;
        spool
            .append(chunk)
            .await
            .map_err(RequestBodySpoolError::Spool)?;
    }
}

pub(crate) const DEFAULT_UPSTREAM_RESPONSE_MAX_BYTES: usize = 64 * 1024 * 1024;

pub(crate) const UPSTREAM_ERROR_MAX_BYTES: usize = 1024 * 1024;

pub(crate) async fn read_limited_upstream_body(
    response: reqwest::Response,
    maximum: usize,
) -> Result<Bytes, String> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err(format!(
            "upstream response exceeds the configured {maximum} byte limit"
        ));
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        if body.len().saturating_add(chunk.len()) > maximum {
            return Err(format!(
                "upstream response exceeds the configured {maximum} byte limit"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

pub(crate) fn upstream_response_max_bytes() -> usize {
    std::env::var("RUST_GENERIC_UPSTREAM_RESPONSE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_UPSTREAM_RESPONSE_MAX_BYTES)
}
