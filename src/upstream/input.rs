use crate::providers::types::PreparedInput;
use crate::runtime::context::AppState;
use crate::runtime::resources::MemoryReservation;
use crate::transport::body::read_spooled_body;
use crate::transport::body::RequestBodySpoolError;
use crate::transport::http::json_error;
use crate::transport::spool::SpoolObservation;
use crate::upstream::generic::query_value;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{Method, Response, StatusCode, Uri};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Duration;
use url::Url;

pub(crate) const IMAGE_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) const DEFAULT_IMAGE_MAX_BYTES: usize = 12 * 1024 * 1024;

pub(crate) async fn prepare_image_inputs(
    state: &AppState,
    mut input: PreparedInput,
) -> Result<(PreparedInput, Vec<MemoryReservation>), (StatusCode, String)> {
    let Some(payload) = input.payload.as_mut() else {
        return Ok((input, Vec::new()));
    };
    let mut reservations = Vec::new();
    if let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("image_url") {
                    continue;
                }
                let Some(url) = part
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                else {
                    continue;
                };
                let normalized = normalize_image_url(state, &url, &mut reservations).await?;
                if let Some(root) = part.get_mut("image_url").and_then(Value::as_object_mut) {
                    root.insert("url".into(), Value::String(normalized));
                }
            }
        }
    }
    if let Some(items) = payload.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            let Some(parts) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("input_image") {
                    continue;
                }
                let Some(url) = part
                    .get("image_url")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                else {
                    continue;
                };
                let normalized = normalize_image_url(state, &url, &mut reservations).await?;
                part.as_object_mut()
                    .expect("responses input image object")
                    .insert("image_url".into(), Value::String(normalized));
            }
        }
    }
    Ok((input, reservations))
}

pub(crate) async fn normalize_image_url(
    state: &AppState,
    value: &str,
    reservations: &mut Vec<MemoryReservation>,
) -> Result<String, (StatusCode, String)> {
    if value.starts_with("data:") {
        validate_image_data_url(state, value, reservations).await?;
        return Ok(value.to_owned());
    }
    let parsed =
        Url::parse(value).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid image URL".into()))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err((StatusCode::BAD_REQUEST, "Invalid image URL".into()));
    }
    let client = state
        .upstream_client(None, false, None)
        .await
        .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;
    let response = tokio::time::timeout(
        IMAGE_FETCH_TIMEOUT,
        client
            .get(value)
            .header("accept-encoding", "identity")
            .timeout(IMAGE_FETCH_TIMEOUT)
            .send(),
    )
    .await
    .map_err(|_| (StatusCode::REQUEST_TIMEOUT, "Image fetch timed out".into()))?
    .map_err(|_| (StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()))?;
    if !response.status().is_success() {
        return Err((StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()));
    }
    let maximum = image_max_bytes();
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Image input exceeds the configured size limit".into(),
        ));
    }
    if let Some(length) = response.content_length() {
        let (_, reservation) = state
            .resource_governor
            .reserve_memory_capacity(length.saturating_mul(3))
            .await
            .map_err(|_| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Insufficient memory capacity for image input".into(),
                )
            })?;
        reservations.push(reservation);
    }
    let known_length = response.content_length().is_some();
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|_| (StatusCode::BAD_REQUEST, "Unable to fetch image URL".into()))?;
        if bytes.len().saturating_add(chunk.len()) > maximum {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                "Image input exceeds the configured size limit".into(),
            ));
        }
        if !known_length {
            let (_, reservation) = state
                .resource_governor
                .reserve_memory_capacity((chunk.len() as u64).saturating_mul(3))
                .await
                .map_err(|_| {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Insufficient memory capacity for image input".into(),
                    )
                })?;
            reservations.push(reservation);
        }
        bytes.extend_from_slice(&chunk);
    }
    let media_type = match inspect_image_media_type(&bytes) {
        ImageMediaTypeInspection::Supported(media_type) => media_type,
        ImageMediaTypeInspection::AnimatedGif => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Animated GIF image input is not supported".into(),
            ));
        }
        ImageMediaTypeInspection::InvalidGif => {
            return Err((StatusCode::BAD_REQUEST, "Invalid GIF image input".into()));
        }
        ImageMediaTypeInspection::Unsupported => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Unsupported image media type".into(),
            ));
        }
    };
    Ok(format!(
        "data:{media_type};base64,{}",
        BASE64.encode(&bytes)
    ))
}

pub(crate) async fn validate_image_data_url(
    state: &AppState,
    value: &str,
    reservations: &mut Vec<MemoryReservation>,
) -> Result<(), (StatusCode, String)> {
    let (header, encoded) = value
        .split_once(',')
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Invalid image data URL".into()))?;
    if header.len() > 128
        || !header.to_ascii_lowercase().starts_with("data:")
        || !header.to_ascii_lowercase().ends_with(";base64")
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "Image input must be a base64 data URL".into(),
        ));
    }
    let declared = header[5..header.len().saturating_sub(7)].to_ascii_lowercase();
    let declared = if declared == "image/jpg" {
        "image/jpeg"
    } else {
        declared.as_str()
    };
    if !matches!(
        declared,
        "image/gif" | "image/jpeg" | "image/png" | "image/webp"
    ) {
        return Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Unsupported image media type".into(),
        ));
    }
    let predicted = encoded.len().saturating_add(3) / 4 * 3;
    if predicted > image_max_bytes() {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Image input exceeds the configured size limit".into(),
        ));
    }
    let (_, reservation) = state
        .resource_governor
        .reserve_memory_capacity(predicted as u64)
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "Insufficient memory capacity for image input".into(),
            )
        })?;
    reservations.push(reservation);
    let padded = format!("{encoded}{}", "=".repeat((4 - encoded.len() % 4) % 4));
    let decoded = BASE64
        .decode(padded.as_bytes())
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid image base64".into()))?;
    let detected = match inspect_image_media_type(&decoded) {
        ImageMediaTypeInspection::Supported(media_type) => media_type,
        ImageMediaTypeInspection::AnimatedGif => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Animated GIF image input is not supported".into(),
            ));
        }
        ImageMediaTypeInspection::InvalidGif => {
            return Err((StatusCode::BAD_REQUEST, "Invalid GIF image input".into()));
        }
        ImageMediaTypeInspection::Unsupported => {
            return Err((
                StatusCode::BAD_REQUEST,
                "Image bytes do not match the declared media type".into(),
            ));
        }
    };
    if detected != declared {
        return Err((
            StatusCode::BAD_REQUEST,
            "Image bytes do not match the declared media type".into(),
        ));
    }
    Ok(())
}

pub(crate) fn image_max_bytes() -> usize {
    std::env::var("RUST_IMAGE_INPUT_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_IMAGE_MAX_BYTES)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ImageMediaTypeInspection {
    Supported(&'static str),
    AnimatedGif,
    InvalidGif,
    Unsupported,
}

pub(crate) fn inspect_image_media_type(bytes: &[u8]) -> ImageMediaTypeInspection {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return ImageMediaTypeInspection::Supported("image/png");
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return ImageMediaTypeInspection::Supported("image/jpeg");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return ImageMediaTypeInspection::Supported("image/webp");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return match gif_frame_count(bytes) {
            Some(1) => ImageMediaTypeInspection::Supported("image/gif"),
            Some(frames) if frames > 1 => ImageMediaTypeInspection::AnimatedGif,
            _ => ImageMediaTypeInspection::InvalidGif,
        };
    }
    ImageMediaTypeInspection::Unsupported
}

pub(crate) fn gif_frame_count(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 13 || (!bytes.starts_with(b"GIF87a") && !bytes.starts_with(b"GIF89a")) {
        return None;
    }
    let mut cursor = 13usize;
    let logical_screen_packed = bytes[10];
    if logical_screen_packed & 0x80 != 0 {
        let table_bytes = 3usize.checked_mul(1usize << ((logical_screen_packed & 0x07) + 1))?;
        cursor = cursor.checked_add(table_bytes)?;
        if cursor > bytes.len() {
            return None;
        }
    }

    let mut frames = 0usize;
    loop {
        let introducer = *bytes.get(cursor)?;
        cursor += 1;
        match introducer {
            0x3B => return Some(frames),
            0x21 => {
                bytes.get(cursor)?;
                cursor += 1;
                skip_gif_subblocks(bytes, &mut cursor)?;
            }
            0x2C => {
                let descriptor_end = cursor.checked_add(9)?;
                let descriptor = bytes.get(cursor..descriptor_end)?;
                cursor = descriptor_end;
                let image_packed = descriptor[8];
                if image_packed & 0x80 != 0 {
                    let table_bytes = 3usize.checked_mul(1usize << ((image_packed & 0x07) + 1))?;
                    cursor = cursor.checked_add(table_bytes)?;
                    if cursor > bytes.len() {
                        return None;
                    }
                }
                let lzw_code_size = *bytes.get(cursor)?;
                if !(2..=12).contains(&lzw_code_size) {
                    return None;
                }
                cursor += 1;
                skip_gif_subblocks(bytes, &mut cursor)?;
                frames += 1;
                if frames > 1 {
                    return Some(frames);
                }
            }
            _ => return None,
        }
    }
}

pub(crate) fn skip_gif_subblocks(bytes: &[u8], cursor: &mut usize) -> Option<()> {
    loop {
        let block_size = *bytes.get(*cursor)? as usize;
        *cursor += 1;
        if block_size == 0 {
            return Some(());
        }
        *cursor = cursor.checked_add(block_size)?;
        if *cursor > bytes.len() {
            return None;
        }
    }
}

pub(crate) async fn prepare_input(
    state: &AppState,
    request: Request,
    method: &Method,
    uri: &Uri,
    path: &str,
    resource_wait: Duration,
) -> Result<PreparedInput, Response<Body>> {
    if *method == Method::GET {
        let payload = if matches!(path, "/search" | "/v1/search") {
            let query = query_value(uri, "q").unwrap_or_else(|| "Jina+AI".into());
            Some(json!({
                "model": "search",
                "messages": [{"role":"user","content":query}],
                "stream": false,
            }))
        } else {
            let model = query_value(uri, "model").unwrap_or_else(|| "video".into());
            Some(json!({"model": model}))
        };
        return Ok(PreparedInput {
            payload,
            replay: None,
            observation: SpoolObservation::default(),
            default_model: if matches!(path, "/search" | "/v1/search") {
                "search".into()
            } else {
                String::new()
            },
            content_type: String::new(),
        });
    }

    let content_length = request
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();
    let normalized_content_type = content_type.to_ascii_lowercase();
    let (_, body) = request.into_parts();
    let spool = read_spooled_body(
        body,
        &state.request_spool,
        None,
        content_length,
        resource_wait,
    )
    .await
    .map_err(|error| match error {
        RequestBodySpoolError::Timeout => {
            json_error(StatusCode::REQUEST_TIMEOUT, "Request body upload timed out")
        }
        RequestBodySpoolError::Read => {
            json_error(StatusCode::BAD_REQUEST, "Request body upload failed")
        }
        RequestBodySpoolError::Spool(failure) => json_error(failure.status, &failure.message),
    })?;
    if normalized_content_type.starts_with("application/json") {
        let payload = spool
            .storage
            .parse_json()
            .await
            .map_err(|error| json_error(StatusCode::BAD_REQUEST, &error))?;
        if !payload.is_object() {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                "JSON request body must be an object",
            ));
        }
        return Ok(PreparedInput {
            payload: Some(payload),
            replay: None,
            observation: spool.observation,
            default_model: String::new(),
            content_type,
        });
    }
    let model = match query_value(uri, "model") {
        Some(model) => model,
        None if normalized_content_type.starts_with("multipart/form-data") => spool
            .storage
            .multipart_text_field(&content_type, "model", 4096)
            .await
            .map_err(|error| json_error(StatusCode::BAD_REQUEST, &error))?
            .unwrap_or_else(|| {
                if path == "/v1/images/edits" {
                    "gpt-image-2".into()
                } else {
                    String::new()
                }
            }),
        None => String::new(),
    };
    Ok(PreparedInput {
        payload: Some(json!({"model": model})),
        replay: Some((spool.storage, spool.observation.clone())),
        observation: spool.observation,
        default_model: String::new(),
        content_type,
    })
}
