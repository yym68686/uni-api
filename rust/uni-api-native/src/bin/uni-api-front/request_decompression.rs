use std::io::Read;

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{Response, StatusCode};

use crate::proxy::json_error;

const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

pub async fn decode(mut request: Request) -> Result<Request, Response<Body>> {
    let encodings = content_encodings(request.headers())?;
    if encodings.is_empty() || encodings.iter().all(|value| value == "identity") {
        return Ok(request);
    }
    if encodings.as_slice() != ["zstd"] {
        return Err(json_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            &format!("unsupported content encoding: {}", encodings.join(", ")),
        ));
    }

    let legacy_limit = env_limit("ZSTD_REQUEST_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES);
    let compressed_limit = env_limit("ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES", legacy_limit);
    let decompressed_limit = env_limit("ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES", legacy_limit);
    if let Some(length) = content_length(request.headers())? {
        if length > compressed_limit as u64 {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
            ));
        }
    }

    let (mut parts, body) = request.into_parts();
    let compressed = to_bytes(body, compressed_limit.saturating_add(1))
        .await
        .map_err(|_| json_error(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"))?;
    if compressed.len() > compressed_limit {
        return Err(json_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large",
        ));
    }

    let decoded = tokio::task::spawn_blocking(move || {
        decompress_zstd(compressed.as_ref(), decompressed_limit)
    })
    .await
    .map_err(|_| json_error(StatusCode::BAD_REQUEST, "invalid zstd body"))?
    .map_err(|_| json_error(StatusCode::BAD_REQUEST, "invalid zstd body"))?;
    if decoded.len() > decompressed_limit {
        return Err(json_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large",
        ));
    }

    parts.headers.remove("content-encoding");
    parts.headers.remove("content-length");
    request = Request::from_parts(parts, Body::from(decoded));
    Ok(request)
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

fn decompress_zstd(compressed: &[u8], limit: usize) -> std::io::Result<Vec<u8>> {
    let mut decoder = zstd::stream::read::Decoder::new(compressed)?;
    let window_log = limit
        // libzstd's streaming decoder uses internal tables in addition to the
        // advertised history window. Keep an 8 MiB floor while still
        // preventing a tiny wire body from advertising a 128+ MiB window.
        .max(8 * 1024 * 1024)
        .next_power_of_two()
        .trailing_zeros()
        .clamp(10, 31);
    decoder.window_log_max(window_log)?;
    let mut output = Vec::with_capacity(compressed.len().saturating_mul(4).min(limit));
    decoder
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request as HttpRequest;

    #[tokio::test]
    async fn decodes_zstd_and_strips_wire_headers() {
        let compressed = zstd::stream::encode_all(b"{\"ok\":true}".as_slice(), 3).unwrap();
        let request = HttpRequest::builder()
            .header("content-encoding", "zstd")
            .header("content-length", compressed.len())
            .body(Body::from(compressed))
            .unwrap();
        let request = decode(request).await.unwrap();
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
            decode(request).await.unwrap_err().status(),
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
            decode(request).await.unwrap_err().status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }
}
