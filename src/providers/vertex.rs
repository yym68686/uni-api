use crate::config::snapshot::Provider;
use crate::runtime::clock::unix_seconds;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use ring::rand::SystemRandom;
use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use url::Url;

#[derive(Clone)]
pub(crate) struct CachedVertexToken {
    pub(crate) value: String,
    pub(crate) expires_at: Instant,
}

pub(crate) static VERTEX_TOKEN_CACHE: OnceLock<
    tokio::sync::Mutex<HashMap<String, CachedVertexToken>>,
> = OnceLock::new();

pub(crate) fn vertex_gemini_url(
    provider: &Provider,
    model: &str,
    key: &str,
    stream: bool,
) -> Result<String, String> {
    let operation = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };
    if key.as_bytes().get(2) == Some(&b'.') {
        let mut url = Url::parse(provider.base_url.trim_end_matches('/'))
            .map_err(|error| format!("invalid Vertex base URL: {error}"))?;
        let base_path = url.path().trim_end_matches('/');
        url.set_path(&format!(
            "{base_path}/v1/publishers/google/models/{model}:{operation}"
        ));
        url.query_pairs_mut().clear().append_pair("key", key);
        return Ok(url.to_string());
    }
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
        "{origin}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{operation}"
    ))
}

pub(crate) async fn vertex_access_token(
    client: &reqwest::Client,
    provider: &Provider,
) -> Result<String, String> {
    let email = provider
        .client_email
        .as_deref()
        .ok_or_else(|| format!("Vertex provider {} is missing client_email", provider.name))?;
    let private_key = provider
        .private_key
        .as_deref()
        .ok_or_else(|| format!("Vertex provider {} is missing private_key", provider.name))?;
    let cache = VERTEX_TOKEN_CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    if let Some(token) = cache
        .lock()
        .await
        .get(email)
        .filter(|token| token.expires_at > Instant::now() + Duration::from_secs(60))
        .cloned()
    {
        return Ok(token.value);
    }
    let assertion = service_account_jwt(email, private_key)?;
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer")
        .append_pair("assertion", &assertion)
        .finish();
    let response = client
        .post("https://oauth2.googleapis.com/token")
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept-encoding", "identity")
        .body(body)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("Vertex OAuth token request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Vertex OAuth token endpoint returned HTTP {}",
            response.status().as_u16()
        ));
    }
    let payload = response
        .json::<Value>()
        .await
        .map_err(|error| format!("decode Vertex OAuth token response: {error}"))?;
    let value = payload
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 16 * 1024)
        .ok_or_else(|| "Vertex OAuth token response is invalid".to_owned())?
        .to_owned();
    let expires_in = payload
        .get("expires_in")
        .and_then(Value::as_u64)
        .unwrap_or(3600)
        .clamp(120, 86_400);
    cache.lock().await.insert(
        email.to_owned(),
        CachedVertexToken {
            value: value.clone(),
            expires_at: Instant::now() + Duration::from_secs(expires_in),
        },
    );
    Ok(value)
}

pub(crate) fn service_account_jwt(email: &str, private_key: &str) -> Result<String, String> {
    let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
    let now = unix_seconds();
    let claims = serde_json::to_vec(&json!({
        "iss": email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now.saturating_add(3600),
        "iat": now,
    }))
    .map_err(|error| format!("encode Vertex OAuth claims: {error}"))?;
    let claims = URL_SAFE_NO_PAD.encode(claims);
    let signing_input = format!("{header}.{claims}");
    let encoded_key = private_key
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect::<String>();
    let der = BASE64
        .decode(encoded_key)
        .map_err(|error| format!("decode Vertex private key: {error}"))?;
    let key = RsaKeyPair::from_pkcs8(&der)
        .or_else(|_| RsaKeyPair::from_der(&der))
        .map_err(|_| "parse Vertex RSA private key".to_owned())?;
    let mut signature = vec![0; key.public().modulus_len()];
    key.sign(
        &RSA_PKCS1_SHA256,
        &SystemRandom::new(),
        signing_input.as_bytes(),
        &mut signature,
    )
    .map_err(|_| "sign Vertex OAuth assertion".to_owned())?;
    Ok(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature)
    ))
}
