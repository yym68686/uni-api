use crate::config::snapshot::Provider;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use url::Url;

pub(crate) fn azure_chat_url(base: &str, deployment: &str) -> Result<String, String> {
    let mut url = Url::parse(base).map_err(|error| format!("invalid Azure base URL: {error}"))?;
    let path = url.path().trim_end_matches('/');
    if !path.contains("/models/chat/completions")
        && !path.contains("/openai/deployments/")
        && !path.ends_with("/chat/completions")
    {
        url.set_path(&format!(
            "/openai/deployments/{deployment}/chat/completions"
        ));
    }
    let has_version = url.query_pairs().any(|(name, _)| name == "api-version");
    if !has_version {
        url.query_pairs_mut()
            .append_pair("api-version", "2025-01-01-preview");
    }
    Ok(url.to_string())
}

pub(crate) fn databricks_chat_url(base: &str, deployment: &str) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid Databricks base URL: {error}"))?;
    url.set_path(&format!("/serving-endpoints/{deployment}/invocations"));
    url.set_query(None);
    Ok(url.to_string())
}

pub(crate) fn cloudflare_url(provider: &Provider, model: &str) -> Result<String, String> {
    let account = provider.cf_account_id.as_deref().ok_or_else(|| {
        format!(
            "Cloudflare provider {} is missing cf_account_id",
            provider.name
        )
    })?;
    let mut url = Url::parse(provider.base_url.as_ref())
        .map_err(|error| format!("invalid Cloudflare base URL: {error}"))?;
    url.set_path(&format!("/client/v4/accounts/{account}/ai/run/{model}"));
    url.set_query(None);
    Ok(url.to_string())
}

pub(crate) fn vertex_claude_url(provider: &Provider, model: &str) -> Result<String, String> {
    let project = provider
        .project_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("Vertex provider {} is missing project_id", provider.name))?;
    let region = provider.region.trim();
    let origin = if provider.base_url.contains("google-vertex-ai") {
        provider.base_url.trim_end_matches('/').to_owned()
    } else if region == "global" {
        "https://aiplatform.googleapis.com".into()
    } else {
        format!("https://{region}-aiplatform.googleapis.com")
    };
    Ok(format!(
        "{origin}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict"
    ))
}

pub(crate) fn aws_bedrock_url(
    provider: &Provider,
    model: &str,
    stream: bool,
) -> Result<String, String> {
    let mut url = Url::parse(provider.base_url.as_ref())
        .map_err(|error| format!("invalid AWS Bedrock base URL: {error}"))?;
    url.set_path(&format!(
        "/model/{model}/{}",
        if stream {
            "invoke-with-response-stream"
        } else {
            "invoke"
        }
    ));
    url.set_query(None);
    Ok(url.to_string())
}

pub(crate) fn normalize_azure_token_limit(payload: &mut Value, model: &str) {
    if !model.to_ascii_lowercase().contains("gpt-5") {
        return;
    }
    let Some(root) = payload.as_object_mut() else {
        return;
    };
    if let Some(value) = root.remove("max_tokens") {
        root.insert("max_completion_tokens".into(), value);
    }
}

pub(crate) fn sign_aws_request(
    provider: &Provider,
    url: &str,
    body: &[u8],
    headers: &mut HeaderMap,
) -> Result<(), String> {
    sign_aws_request_at(provider, url, body, headers, SystemTime::now())
}

pub(crate) fn sign_aws_request_at(
    provider: &Provider,
    url: &str,
    body: &[u8],
    headers: &mut HeaderMap,
    now: SystemTime,
) -> Result<(), String> {
    type HmacSha256 = Hmac<Sha256>;

    let access_key = provider
        .aws_access_key
        .as_deref()
        .ok_or_else(|| format!("AWS provider {} is missing aws_access_key", provider.name))?;
    let secret_key = provider
        .aws_secret_key
        .as_deref()
        .ok_or_else(|| format!("AWS provider {} is missing aws_secret_key", provider.name))?;
    let parsed = Url::parse(url).map_err(|error| format!("invalid AWS URL: {error}"))?;
    let host = match parsed.port() {
        Some(port) => format!("{}:{port}", parsed.host_str().unwrap_or_default()),
        None => parsed.host_str().unwrap_or_default().to_owned(),
    };
    let region = aws_region(provider, &host)?;
    let (amz_date, date_stamp) = aws_timestamp(now);
    let payload_hash = format!("{:x}", Sha256::digest(body));
    let accept = if parsed.path().ends_with("invoke-with-response-stream") {
        "application/vnd.amazon.bedrock.payload+json"
    } else {
        "application/json"
    };
    let mut canonical_headers = format!(
        "accept:{accept}\ncontent-type:application/json\nhost:{host}\nx-amz-bedrock-accept:{accept}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    );
    let mut signed_headers =
        "accept;content-type;host;x-amz-bedrock-accept;x-amz-content-sha256;x-amz-date".to_owned();
    if let Some(token) = provider.aws_session_token.as_deref() {
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }
    let canonical_query = parsed.query().unwrap_or_default();
    let canonical_request = format!(
        "POST\n{}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        aws_uri_encode(parsed.path())
    );
    let scope = format!("{date_stamp}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{:x}",
        Sha256::digest(canonical_request.as_bytes())
    );
    let k_date = hmac_bytes::<HmacSha256>(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    )?;
    let k_region = hmac_bytes::<HmacSha256>(&k_date, region.as_bytes())?;
    let k_service = hmac_bytes::<HmacSha256>(&k_region, b"bedrock")?;
    let k_signing = hmac_bytes::<HmacSha256>(&k_service, b"aws4_request")?;
    let signature = hex_bytes(&hmac_bytes::<HmacSha256>(
        &k_signing,
        string_to_sign.as_bytes(),
    )?);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    for (name, value) in [
        ("accept", accept),
        ("content-type", "application/json"),
        ("host", host.as_str()),
        ("x-amz-bedrock-accept", accept),
        ("x-amz-content-sha256", payload_hash.as_str()),
        ("x-amz-date", amz_date.as_str()),
        ("authorization", authorization.as_str()),
    ] {
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).expect("static AWS header name"),
            HeaderValue::from_str(value)
                .map_err(|_| format!("AWS provider {} has an invalid header", provider.name))?,
        );
    }
    if let Some(token) = provider.aws_session_token.as_deref() {
        headers.insert(
            "x-amz-security-token",
            HeaderValue::from_str(token)
                .map_err(|_| "AWS session token is not a valid header value".to_owned())?,
        );
    }
    Ok(())
}

pub(crate) fn hmac_bytes<M>(key: &[u8], message: &[u8]) -> Result<Vec<u8>, String>
where
    M: Mac + hmac::digest::KeyInit,
{
    let mut mac = <M as Mac>::new_from_slice(key).map_err(|_| "invalid HMAC key".to_owned())?;
    mac.update(message);
    Ok(mac.finalize().into_bytes().to_vec())
}

pub(crate) fn aws_region(provider: &Provider, host: &str) -> Result<String, String> {
    if provider.region.as_ref() != "global" && !provider.region.trim().is_empty() {
        return Ok(provider.region.to_string());
    }
    let parts = host.split('.').collect::<Vec<_>>();
    if parts.len() >= 3 && parts[0].starts_with("bedrock-runtime") {
        return Ok(parts[1].to_owned());
    }
    Err(format!(
        "AWS provider {} has no usable Bedrock region",
        provider.name
    ))
}

pub(crate) fn aws_timestamp(now: SystemTime) -> (String, String) {
    let total = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let days = total.div_euclid(86_400);
    let seconds = total.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds / 3600;
    let minute = seconds % 3600 / 60;
    let second = seconds % 60;
    (
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"),
        format!("{year:04}{month:02}{day:02}"),
    )
}

pub(crate) fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

pub(crate) fn aws_uri_encode(path: &str) -> String {
    let mut output = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(char::from(b"0123456789ABCDEF"[(byte >> 4) as usize]));
            output.push(char::from(b"0123456789ABCDEF"[(byte & 0x0f) as usize]));
        }
    }
    output
}

pub(crate) fn hex_bytes(value: &[u8]) -> String {
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        output.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    output
}
