use crate::runtime::resources::{MemoryReservation, ResourceGovernor};
use crate::transport::http::json_error;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{Response, StatusCode};
use futures_util::StreamExt;
use serde::Serialize;
use std::io::Read;

const DEFAULT_MAX_BODY_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RequestBodyLimits {
    pub max_bytes: usize,
    pub zstd_compressed_max_bytes: usize,
    pub zstd_decompressed_max_bytes: usize,
}

impl RequestBodyLimits {
    pub fn from_env() -> Self {
        let max_bytes = env_limit("REQUEST_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES);
        let legacy_limit = env_limit("ZSTD_REQUEST_MAX_BODY_BYTES", max_bytes).min(max_bytes);
        Self {
            max_bytes,
            zstd_compressed_max_bytes: env_limit(
                "ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES",
                legacy_limit,
            )
            .min(max_bytes),
            zstd_decompressed_max_bytes: env_limit(
                "ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES",
                legacy_limit,
            )
            .min(max_bytes),
        }
    }
}

pub async fn decode(
    request: Request,
    governor: &ResourceGovernor,
) -> Result<Request, Response<Body>> {
    decode_governed(
        request,
        RequestBodyLimits::from_env(),
        governor,
        crate::transport::body::request_body_idle_timeout(),
    )
    .await
}

#[cfg(test)]
async fn decode_with_limits(
    request: Request,
    limits: RequestBodyLimits,
) -> Result<Request, Response<Body>> {
    decode_governed(
        request,
        limits,
        &ResourceGovernor::unconstrained_for_test(),
        crate::transport::body::request_body_idle_timeout(),
    )
    .await
}

async fn decode_governed(
    mut request: Request,
    limits: RequestBodyLimits,
    governor: &ResourceGovernor,
    idle: std::time::Duration,
) -> Result<Request, Response<Body>> {
    let encodings = content_encodings(request.headers())?;
    if encodings.is_empty() || encodings.iter().all(|value| value == "identity") {
        if content_length(request.headers())?.is_some_and(|length| length > limits.max_bytes as u64)
        {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
            ));
        }
        // The spool checks actual bytes incrementally, including chunked uploads.
        return Ok(request);
    }
    if encodings.as_slice() != ["zstd"] {
        return Err(json_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            &format!("unsupported content encoding: {}", encodings.join(", ")),
        ));
    }

    let compressed_limit = limits.zstd_compressed_max_bytes;
    let decompressed_limit = limits.zstd_decompressed_max_bytes;
    if let Some(length) = content_length(request.headers())? {
        if length > compressed_limit as u64 {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
            ));
        }
    }

    let (mut parts, body) = request.into_parts();
    let mut compressed = Vec::new();
    let mut compressed_reservation = None;
    let mut stream = body.into_data_stream();
    loop {
        let Some(chunk) = tokio::time::timeout(idle, stream.next())
            .await
            .map_err(|_| {
                json_error(StatusCode::REQUEST_TIMEOUT, "request body upload timed out")
            })?
        else {
            break;
        };
        let chunk =
            chunk.map_err(|_| json_error(StatusCode::BAD_REQUEST, "request body upload failed"))?;
        if chunk.len() > compressed_limit.saturating_sub(compressed.len()) {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
            ));
        }
        let needed = compressed.len() + chunk.len();
        if needed > compressed.capacity() {
            let capacity = needed
                .max(compressed.capacity().saturating_mul(2))
                .max(64 * 1024);
            let capacity = capacity.min(compressed_limit);
            // Claim the new allocation before Vec can reallocate. The old
            // allocation is still resident until reserve_exact completes.
            let (_, mut reservation) = governor
                .reserve_memory_capacity(capacity as u64)
                .await
                .map_err(crate::api::handler::resource_capacity_response)?;
            compressed.reserve_exact(capacity - compressed.len());
            compressed.extend_from_slice(&chunk);
            reservation.materialize_bytes(compressed.len() as u64);
            compressed_reservation = Some(reservation);
        } else {
            compressed.extend_from_slice(&chunk);
            if let Some(reservation) = compressed_reservation.as_mut() {
                reservation.materialize_bytes(compressed.len() as u64);
            }
        }
    }

    let window_log = window_log_limit(decompressed_limit);
    let decoder_budget = decoder_memory_budget(&compressed, 1_u64 << window_log)
        .map_err(|_| json_error(StatusCode::BAD_REQUEST, "invalid zstd body"))?;
    let (_, decoder_reservation) = governor
        .reserve_memory_capacity(decoder_budget)
        .await
        .map_err(crate::api::handler::resource_capacity_response)?;
    let governor = governor.clone();
    let runtime = tokio::runtime::Handle::current();
    let cancellation = tokio_util::sync::CancellationToken::new();
    let _cancel_on_drop = cancellation.clone().drop_guard();
    let decoded = tokio::task::spawn_blocking(move || -> Result<DecodedBody, DecodeFailure> {
        let _compressed_reservation = compressed_reservation;
        let _decoder_reservation = decoder_reservation;
        let mut decoder = zstd::stream::read::Decoder::new(compressed.as_slice())?;
        decoder.window_log_max(window_log)?;
        let mut decoded = Vec::new();
        let mut reservation = None;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            if cancellation.is_cancelled() { return Err(DecodeFailure::Cancelled); }
            let read = decoder.read(&mut buffer)?;
            if read == 0 { break; }
            let needed = decoded.len().saturating_add(read);
            if needed > decompressed_limit { return Err(DecodeFailure::TooLarge); }
            if needed > decoded.capacity() {
                let capacity = needed.max(decoded.capacity().saturating_mul(2)).min(decompressed_limit);
                let (_, claim) = runtime.block_on(async {
                    tokio::select! {
                        result = governor.reserve_memory_capacity(capacity as u64) => result.map_err(DecodeFailure::Capacity),
                        () = cancellation.cancelled() => Err(DecodeFailure::Cancelled),
                    }
                })?;
                decoded.reserve_exact(capacity - decoded.len());
                reservation = Some(claim);
            }
            decoded.extend_from_slice(&buffer[..read]);
            if let Some(claim) = reservation.as_mut() { claim.materialize_bytes(decoded.len() as u64); }
        }
        if let Some(claim) = reservation.as_mut() { claim.materialize(); }
        Ok(DecodedBody { bytes: decoded, _reservation: reservation })
    })
    .await
    .map_err(|_| json_error(StatusCode::BAD_REQUEST, "invalid zstd body"))?
    .map_err(DecodeFailure::response)?;

    parts.headers.remove("content-encoding");
    parts.headers.remove("content-length");
    request = Request::from_parts(parts, Body::from(bytes::Bytes::from_owner(decoded)));
    Ok(request)
}

struct DecodedBody {
    bytes: Vec<u8>,
    _reservation: Option<MemoryReservation>,
}

enum DecodeFailure {
    Invalid(std::io::Error),
    Capacity(crate::runtime::resources::CapacityFailure),
    TooLarge,
    Cancelled,
}

impl From<std::io::Error> for DecodeFailure {
    fn from(error: std::io::Error) -> Self {
        Self::Invalid(error)
    }
}

impl DecodeFailure {
    fn response(self) -> Response<Body> {
        match self {
            Self::Capacity(failure) => crate::api::handler::resource_capacity_response(failure),
            Self::TooLarge => json_error(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
            Self::Invalid(_error) => json_error(StatusCode::BAD_REQUEST, "invalid zstd body"),
            Self::Cancelled => json_error(
                StatusCode::REQUEST_TIMEOUT,
                "request body decoding cancelled",
            ),
        }
    }
}

fn window_log_limit(limit: usize) -> u32 {
    limit
        .max(8 * 1024 * 1024)
        .checked_next_power_of_two()
        .map(|power| power.trailing_zeros())
        .unwrap_or(31)
        .clamp(10, 31)
}

fn decoder_memory_budget(mut frames: &[u8], maximum_window: u64) -> std::io::Result<u64> {
    let invalid =
        || std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid zstd frame header");
    let mut largest_window = 0;
    while !frames.is_empty() {
        let size = zstd::zstd_safe::find_frame_compressed_size(frames).map_err(|_| invalid())?;
        let frame = frames.get(..size).ok_or_else(invalid)?;
        let magic = u32::from_le_bytes(
            frame
                .get(..4)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?,
        );
        let window = if (0x184D2A50..=0x184D2A5F).contains(&magic) {
            0
        } else if magic != 0xFD2FB528 {
            return Err(invalid());
        } else if frame.get(4).ok_or_else(invalid)? & 0x20 != 0 {
            zstd::zstd_safe::get_frame_content_size(frame)
                .map_err(|_| invalid())?
                .ok_or_else(invalid)?
        } else {
            let descriptor = *frame.get(5).ok_or_else(invalid)?;
            let base = 1_u64 << (10 + (descriptor >> 3));
            base + (base / 8) * u64::from(descriptor & 7)
        };
        if window > maximum_window {
            return Err(invalid());
        }
        largest_window = largest_window.max(window);
        frames = &frames[size..];
    }
    // libzstd's history ring uses window + two <=128 KiB blocks. An additional
    // 8 MiB covers its context/tables, input buffer and our bounded output block.
    Ok(largest_window.saturating_add(8 * 1024 * 1024))
}

impl AsRef<[u8]> for DecodedBody {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

fn content_encodings(headers: &axum::http::HeaderMap) -> Result<Vec<String>, Response<Body>> {
    let mut encodings = Vec::new();
    for value in headers.get_all("content-encoding") {
        let value = value.to_str().map_err(|_| {
            json_error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported content encoding",
            )
        })?;
        encodings.extend(
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_ascii_lowercase),
        );
    }
    Ok(encodings)
}

fn content_length(headers: &axum::http::HeaderMap) -> Result<Option<u64>, Response<Body>> {
    let values = headers.get_all("content-length").iter().collect::<Vec<_>>();
    if values.len() > 1 {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid content-length",
        ));
    }
    values
        .first()
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| json_error(StatusCode::BAD_REQUEST, "invalid content-length"))
        })
        .transpose()
}

fn env_limit(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

#[cfg(test)]
fn decompress_zstd(compressed: &[u8], limit: usize) -> std::io::Result<Vec<u8>> {
    let mut decoder = zstd::stream::read::Decoder::new(compressed)?;
    let window_log = window_log_limit(limit);
    decoder.window_log_max(window_log)?;
    let mut output = Vec::with_capacity(limit.saturating_add(1));
    decoder
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request as HttpRequest;

    #[test]
    fn frame_budget_covers_the_real_decoder_and_does_not_reserve_the_body_maximum() {
        assert_eq!(window_log_limit(usize::MAX), 31);
        let payload = vec![b'x'; 2 * 1024 * 1024];
        let stream_frame = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
        let sized_frame = zstd::bulk::compress(&payload, 3).unwrap();
        for frames in [
            stream_frame.clone(),
            sized_frame.clone(),
            [stream_frame, sized_frame].concat(),
        ] {
            let budget = decoder_memory_budget(&frames, 128 * 1024 * 1024).unwrap();
            let mut context = zstd::zstd_safe::DCtx::create();
            context.init().unwrap();
            let mut decoder =
                zstd::stream::read::Decoder::with_context(frames.as_slice(), &mut context);
            std::io::copy(&mut decoder, &mut std::io::sink()).unwrap();
            drop(decoder);
            assert!(budget >= context.sizeof() as u64);
            assert!(budget < 32 * 1024 * 1024);
        }
        let small = zstd::stream::encode_all(b"small".as_slice(), 3).unwrap();
        assert!(decoder_memory_budget(&small, 128 * 1024 * 1024).unwrap() < 16 * 1024 * 1024);
    }

    #[tokio::test]
    async fn stalled_zstd_upload_obeys_the_upload_idle_deadline() {
        let body = Body::from_stream(futures_util::stream::pending::<
            Result<bytes::Bytes, std::io::Error>,
        >());
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .body(body)
            .unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            decode_governed(
                request,
                limits(1024, 1024, 1024),
                &ResourceGovernor::unconstrained_for_test(),
                std::time::Duration::from_millis(10),
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn decoded_body_owns_its_memory_claim_until_dropped() {
        let governor = ResourceGovernor::unconstrained_for_test();
        let compressed = zstd::stream::encode_all(b"{\"ok\":true}".as_slice(), 3).unwrap();
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .body(Body::from(compressed))
            .unwrap();
        let request = decode_governed(
            request,
            limits(1024, 1024, 1024),
            &governor,
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert!(
            governor.observability_snapshot()["reserved_memory_bytes"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(governor.observability_snapshot()["pending_memory_bytes"], 0);
        drop(request);
        assert_eq!(
            governor.observability_snapshot()["reserved_memory_bytes"],
            0
        );
    }

    fn limits(max_bytes: usize, compressed: usize, decompressed: usize) -> RequestBodyLimits {
        RequestBodyLimits {
            max_bytes,
            zstd_compressed_max_bytes: compressed,
            zstd_decompressed_max_bytes: decompressed,
        }
    }

    #[tokio::test]
    async fn identity_content_length_accepts_128_mib_and_rejects_one_byte_more() {
        assert_eq!(DEFAULT_MAX_BODY_BYTES, 134_217_728);
        let limits = limits(
            DEFAULT_MAX_BODY_BYTES,
            DEFAULT_MAX_BODY_BYTES,
            DEFAULT_MAX_BODY_BYTES,
        );
        for encoding in [None, Some("identity")] {
            for size in [DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_BODY_BYTES + 1] {
                let mut request = HttpRequest::builder().header("content-length", size);
                if let Some(encoding) = encoding {
                    request = request.header("content-encoding", encoding);
                }
                let result = decode_with_limits(request.body(Body::empty()).unwrap(), limits).await;
                if size == DEFAULT_MAX_BODY_BYTES {
                    assert!(result.is_ok());
                } else {
                    assert_eq!(result.unwrap_err().status(), StatusCode::PAYLOAD_TOO_LARGE);
                }
            }
        }
    }

    #[tokio::test]
    async fn zstd_enforces_actual_compressed_and_decoded_boundaries() {
        let original = vec![b'x'; 8192];
        let compressed = zstd::stream::encode_all(original.as_slice(), 3).unwrap();
        for content_length in [false, true] {
            for (wire_limit, decoded_limit, expected) in [
                (compressed.len(), original.len(), None),
                (
                    compressed.len() - 1,
                    original.len(),
                    Some(StatusCode::PAYLOAD_TOO_LARGE),
                ),
                (
                    compressed.len(),
                    original.len() - 1,
                    Some(StatusCode::PAYLOAD_TOO_LARGE),
                ),
            ] {
                let mut request = HttpRequest::builder().header("content-encoding", "zstd");
                if content_length {
                    request = request.header("content-length", compressed.len());
                }
                let chunks = compressed
                    .chunks(3)
                    .map(|chunk| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(chunk)))
                    .collect::<Vec<_>>();
                let request = request
                    .body(Body::from_stream(futures_util::stream::iter(chunks)))
                    .unwrap();
                let result =
                    decode_with_limits(request, limits(original.len(), wire_limit, decoded_limit))
                        .await;
                match expected {
                    Some(status) => assert_eq!(result.unwrap_err().status(), status),
                    None => assert_eq!(
                        to_bytes(result.unwrap().into_body(), original.len())
                            .await
                            .unwrap()
                            .as_ref(),
                        original
                    ),
                }
            }
        }
    }

    #[tokio::test]
    async fn compressed_transport_failure_is_not_reported_as_size_rejection() {
        let chunks = futures_util::stream::iter([Err::<bytes::Bytes, _>(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "fixture disconnect",
        ))]);
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .body(Body::from_stream(chunks))
            .unwrap();
        assert_eq!(
            decode_with_limits(request, limits(1024, 1024, 1024))
                .await
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn decodes_zstd_and_strips_wire_headers() {
        let compressed = zstd::stream::encode_all(b"{\"ok\":true}".as_slice(), 3).unwrap();
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .header("content-length", compressed.len())
            .body(Body::from(compressed))
            .unwrap();
        let request = decode(request, &ResourceGovernor::unconstrained_for_test())
            .await
            .unwrap();
        assert!(request.headers().get("content-encoding").is_none());
        assert!(request.headers().get("content-length").is_none());
        assert_eq!(
            to_bytes(request.into_body(), 1024).await.unwrap().as_ref(),
            b"{\"ok\":true}"
        );
    }

    #[tokio::test]
    async fn rejects_invalid_zstd() {
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .body(Body::from("not-zstd"))
            .unwrap();
        assert_eq!(
            decode(request, &ResourceGovernor::unconstrained_for_test())
                .await
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn accepts_concatenated_frames_and_rejects_truncation() {
        let first = zstd::stream::encode_all(b"abc".as_slice(), 3).unwrap();
        let second = zstd::stream::encode_all(b"defg".as_slice(), 3).unwrap();
        assert_eq!(
            decompress_zstd(&[first, second].concat(), 7).unwrap(),
            b"abcdefg"
        );
        let complete = zstd::stream::encode_all(b"truncated".as_slice(), 3).unwrap();
        assert!(decompress_zstd(&complete[..complete.len() - 1], 1024).is_err());
    }

    #[test]
    fn decoded_limit_stops_compression_bombs() {
        let compressed = zstd::stream::encode_all(vec![b'x'; 4096].as_slice(), 3).unwrap();
        assert_eq!(decompress_zstd(&compressed, 1024).unwrap().len(), 1025);
    }

    #[tokio::test]
    async fn rejects_unsupported_encoding() {
        let request = HttpRequest::builder()
            .header("content-encoding", "gzip")
            .body(Body::from("body"))
            .unwrap();
        assert_eq!(
            decode(request, &ResourceGovernor::unconstrained_for_test())
                .await
                .unwrap_err()
                .status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }
}
