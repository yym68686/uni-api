use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode};
use axum::response::IntoResponse;

pub(crate) const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";

pub(crate) const SPOOL_HEADER_PREFIX: &str = "x-uni-api-rust-request-spool-";

pub fn relay_response(response: reqwest::Response) -> Response<Body> {
    let status = response.status();
    if !status.is_success() {
        return super::upstream_error::PublicError::new(status).response(response.headers());
    }
    let headers = filtered_response_headers(response.headers());
    let stream = response.bytes_stream();
    let mut output = Response::new(Body::from_stream(stream));
    *output.status_mut() = status;
    *output.headers_mut() = headers;
    output
}

pub fn filtered_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str())
            || name.as_str().eq_ignore_ascii_case("host")
            || name.as_str().eq_ignore_ascii_case(CONTROL_HEADER)
            || name
                .as_str()
                .eq_ignore_ascii_case("x-uni-api-rust-responses-session")
            || name.as_str().starts_with(SPOOL_HEADER_PREFIX)
        {
            continue;
        }
        filtered.append(name.clone(), value.clone());
    }
    filtered
}

pub fn filtered_response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if name == axum::http::header::CONTENT_TYPE {
            // Only retain the media type, not arbitrary provider parameters.
            if let Some(media) = value.to_str().ok().and_then(|v| v.split(';').next()) {
                let media = media.trim().to_ascii_lowercase();
                if (matches!(
                    media.as_str(),
                    "application/json"
                        | "application/octet-stream"
                        | "text/event-stream"
                        | "text/plain"
                        | "application/zip"
                ) || ["audio/", "image/", "video/"]
                    .iter()
                    .any(|prefix| media.starts_with(prefix))
                    && media.len() <= 48
                    && media
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"/.+-".contains(&b)))
                {
                    if let Ok(value) = axum::http::HeaderValue::from_str(&media) {
                        filtered.insert(name.clone(), value);
                    }
                }
            }
        } else if name.as_str() == "x-client-request-id" {
            // Billing-correlation receipt on successful responses. Never
            // permit URLs or arbitrary text in it.
            if value.to_str().ok().is_some_and(|v| {
                !v.is_empty()
                    && v.len() <= 128
                    && v.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
            }) {
                filtered.insert(name.clone(), value.clone());
            }
        } else if let Some(value) = oaix_header(name.as_str(), value) {
            filtered.append(name.clone(), value);
        }
    }
    filtered
}

// The settlement contract is used by 0-0. Do not expose arbitrary x-oaix-*
// metadata; accept only the explicitly consumed, typed success fields.
pub(crate) fn oaix_header(
    name: &str,
    value: &axum::http::HeaderValue,
) -> Option<axum::http::HeaderValue> {
    let text = value.to_str().ok()?;
    let opaque = |max| {
        !text.is_empty()
            && text.len() <= max
            && text
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    };
    let valid = match name {
        "x-oaix-attribution-contract" => text == "receipt-v1",
        "x-oaix-request-id" | "x-oaix-connection-id" | "x-oaix-marketplace-contract-key" => {
            opaque(128)
        }
        "x-oaix-token-id" | "x-oaix-token-owner-user-id" | "x-oaix-marketplace-price-bps" => {
            text.parse::<u64>().is_ok() && text.bytes().all(|b| b.is_ascii_digit())
        }
        "x-oaix-marketplace-price-source" | "x-oaix-selection-mode" => opaque(40),
        "x-oaix-marketplace-price-locked" => matches!(text, "true" | "false" | "1" | "0"),
        "x-oaix-marketplace-price-locked-at" => {
            text.len() <= 40
                && text.contains('T')
                && text
                    .bytes()
                    .all(|b| b.is_ascii_digit() || b"TtZz+-:.".contains(&b))
        }
        _ => false,
    };
    valid.then(|| value.clone())
}

pub(crate) fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

// Only fixed, gateway-generated diagnostics qualify for export. Arbitrary
// preparation errors can contain URLs or caller values and must stay private.
pub(crate) fn conversion_error_location(message: &str) -> Option<(&str, &str)> {
    let (code @ ("gemini_tool_history" | "gemini_part"), rest) = message.split_once(": ")? else {
        return None;
    };
    let (field, _) = rest.split_once(": ")?;
    (!field.is_empty()
        && field.len() <= 256
        && field
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"[]._-".contains(&b)))
    .then_some((code, field))
}

pub fn json_error(status: StatusCode, message: &str) -> Response<Body> {
    let mut error = serde_json::json!({"message":message});
    if let Some((code, field)) = (status == StatusCode::BAD_REQUEST)
        .then(|| conversion_error_location(message))
        .flatten()
    {
        error["type"] = serde_json::json!("invalid_request_error");
        error["code"] = serde_json::json!(code);
        error["param"] = serde_json::json!(field);
    }
    (
        status,
        [("content-type", "application/json")],
        serde_json::json!({"error":error}).to_string(),
    )
        .into_response()
}
