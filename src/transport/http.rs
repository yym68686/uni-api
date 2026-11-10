use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode};
use axum::response::IntoResponse;

pub(crate) const CONTROL_HEADER: &str = "x-uni-api-rust-control-token";

pub(crate) const SPOOL_HEADER_PREFIX: &str = "x-uni-api-rust-request-spool-";

pub fn relay_response(response: reqwest::Response) -> Response<Body> {
    let status = response.status();
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
        if is_hop_by_hop(name.as_str())
            || name.as_str().eq_ignore_ascii_case("content-length")
            || name.as_str().eq_ignore_ascii_case("content-encoding")
            || name
                .as_str()
                .eq_ignore_ascii_case("x-uni-api-rust-responses-session")
        {
            continue;
        }
        filtered.append(name.clone(), value.clone());
    }
    filtered
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
