use crate::providers::cloud::hex_bytes;
use crate::transport::body::read_limited_upstream_body;
use crate::transport::spool::SpoolObservation;
use crate::transport::spool::StoredBody;
use axum::http::{HeaderMap, HeaderValue};
use bytes::Bytes;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io;
use std::time::Duration;
use url::Url;

pub(crate) fn multipart_output_boundary(request_id: &str) -> String {
    let digest = Sha256::digest(request_id.as_bytes());
    format!("uni-api-{}", hex_bytes(&digest[..12]))
}

pub(crate) async fn multipart_rewrite_body(
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    boundary: String,
    model: String,
) -> Result<reqwest::Body, String> {
    Ok(reqwest::Body::wrap_stream(
        multipart_rewrite_stream(storage, observation, source_content_type, boundary, model)
            .await?,
    ))
}

pub(crate) async fn prepare_dashscope_transcription(
    client: &reqwest::Client,
    mut headers: HeaderMap,
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    model: &str,
    provider_key: &str,
) -> Result<(HeaderMap, Vec<u8>), String> {
    let audio = storage
        .multipart_file(source_content_type, "file")
        .await?
        .ok_or_else(|| "audio transcription requires multipart file field".to_owned())?;
    let certificate_response = client
        .get("https://dashscope.aliyuncs.com/api/v1/uploads")
        .bearer_auth(provider_key)
        .header("accept-encoding", "identity")
        .query(&[("action", "getPolicy"), ("model", model)])
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("request DashScope upload certificate: {error}"))?;
    if !certificate_response.status().is_success() {
        return Err(format!(
            "DashScope upload certificate returned HTTP {}",
            certificate_response.status()
        ));
    }
    let certificate_body = read_limited_upstream_body(certificate_response, 256 * 1024).await?;
    let certificate: Value = serde_json::from_slice(&certificate_body)
        .map_err(|error| format!("decode DashScope upload certificate: {error}"))?;
    let data = certificate
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| "DashScope upload certificate is missing data".to_owned())?;
    let field = |name: &str, maximum: usize| -> Result<String, String> {
        let value = data
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("DashScope upload certificate is missing {name}"))?;
        if value.len() > maximum {
            return Err(format!(
                "DashScope upload certificate field {name} is too large"
            ));
        }
        Ok(value.to_owned())
    };
    let upload_host = field("upload_host", 2048)?;
    let upload_dir = field("upload_dir", 1024)?;
    let parsed_upload_host = Url::parse(&upload_host)
        .map_err(|error| format!("invalid DashScope upload host: {error}"))?;
    if !matches!(parsed_upload_host.scheme(), "http" | "https")
        || parsed_upload_host.host_str().is_none()
    {
        return Err("DashScope upload host must be HTTP(S)".into());
    }
    if audio.filename.len() > 512 {
        return Err("DashScope upload filename exceeds 512 bytes".into());
    }
    let object_key = format!("{upload_dir}/{}", audio.filename);
    let audio_body = audio
        .storage
        .into_body(&observation)
        .await
        .map_err(|error| error.message)?;
    let mut part = reqwest::multipart::Part::stream_with_length(
        reqwest::Body::wrap_stream(audio_body.into_data_stream()),
        audio.bytes,
    )
    .file_name(audio.filename);
    if let Some(content_type) = audio.content_type {
        part = part
            .mime_str(&content_type)
            .map_err(|error| format!("invalid audio MIME type: {error}"))?;
    }
    let form = reqwest::multipart::Form::new()
        .text("key", object_key.clone())
        .text("policy", field("policy", 64 * 1024)?)
        .text("OSSAccessKeyId", field("oss_access_key_id", 4096)?)
        .text("signature", field("signature", 64 * 1024)?)
        .text("success_action_status", "200")
        .text("x-oss-object-acl", field("x_oss_object_acl", 256)?)
        .text(
            "x-oss-forbid-overwrite",
            field("x_oss_forbid_overwrite", 256)?,
        )
        .part("file", part);
    let upload = client
        .post(upload_host)
        .timeout(Duration::from_secs(3600))
        .multipart(form)
        .send()
        .await
        .map_err(|error| format!("upload DashScope transcription input: {error}"))?;
    if !upload.status().is_success() {
        return Err(format!(
            "DashScope OSS upload returned HTTP {}",
            upload.status()
        ));
    }
    let mut payload = json!({
        "model":model,
        "input":{"messages":[{"role":"user","content":[{"audio":format!("oss://{object_key}")}]}]},
    });
    for field_name in [
        "prompt",
        "response_format",
        "temperature",
        "language",
        "timestamp_granularities[]",
    ] {
        if let Some(value) = storage
            .multipart_text_field(source_content_type, field_name, 64 * 1024)
            .await?
            .filter(|value| !value.is_empty())
        {
            payload[field_name] = Value::String(value);
        }
    }
    headers.remove("content-length");
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert(
        "x-dashscope-ossresourceresolve",
        HeaderValue::from_static("enable"),
    );
    serde_json::to_vec(&payload)
        .map(|body| (headers, body))
        .map_err(|error| format!("encode DashScope transcription request: {error}"))
}

pub(crate) async fn multipart_rewrite_stream(
    storage: StoredBody,
    observation: SpoolObservation,
    source_content_type: &str,
    boundary: String,
    model: String,
) -> Result<tokio_stream::wrappers::ReceiverStream<Result<Bytes, io::Error>>, String> {
    let source_boundary = multer::parse_boundary(source_content_type)
        .map_err(|error| format!("invalid multipart boundary: {error}"))?;
    let body = storage
        .into_body(&observation)
        .await
        .map_err(|error| error.message)?;
    let stream = body.into_data_stream();
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, io::Error>>(8);
    tokio::spawn(async move {
        let mut multipart = multer::Multipart::new(stream, source_boundary);
        let mut rewrote_model = false;
        loop {
            let mut field = match multipart.next_field().await {
                Ok(Some(field)) => field,
                Ok(None) => break,
                Err(error) => {
                    let _ = sender
                        .send(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("parse multipart request: {error}"),
                        )))
                        .await;
                    return;
                }
            };
            let rewrite_model = field.name() == Some("model");
            rewrote_model |= rewrite_model;
            let mut prefix = Vec::new();
            prefix.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            for (name, value) in field.headers() {
                prefix.extend_from_slice(name.as_str().as_bytes());
                prefix.extend_from_slice(b": ");
                prefix.extend_from_slice(value.as_bytes());
                prefix.extend_from_slice(b"\r\n");
            }
            prefix.extend_from_slice(b"\r\n");
            if sender.send(Ok(Bytes::from(prefix))).await.is_err() {
                return;
            }
            if rewrite_model && sender.send(Ok(Bytes::from(model.clone()))).await.is_err() {
                return;
            }
            loop {
                match field.chunk().await {
                    Ok(Some(chunk)) if !rewrite_model => {
                        if sender.send(Ok(chunk)).await.is_err() {
                            return;
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(error) => {
                        let _ = sender
                            .send(Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("read multipart field: {error}"),
                            )))
                            .await;
                        return;
                    }
                }
            }
            if sender.send(Ok(Bytes::from_static(b"\r\n"))).await.is_err() {
                return;
            }
        }
        if !rewrote_model {
            let field = format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            );
            if sender.send(Ok(Bytes::from(field))).await.is_err() {
                return;
            }
        }
        let _ = sender
            .send(Ok(Bytes::from(format!("--{boundary}--\r\n"))))
            .await;
    });
    Ok(tokio_stream::wrappers::ReceiverStream::new(receiver))
}
