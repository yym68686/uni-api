use crate::providers::video::lingjing_url;
use crate::providers::video::video_tasks_url;
use axum::http::{Method, Uri};
use url::Url;

pub(crate) fn endpoint_url(
    base: &str,
    endpoint: &str,
    method: &Method,
    uri: &Uri,
) -> Result<String, String> {
    if endpoint.starts_with("/v1/video/tasks/") {
        let task = endpoint.trim_start_matches("/v1/video/tasks/");
        return Ok(format!(
            "{}/{}",
            video_tasks_url(base),
            url::form_urlencoded::byte_serialize(task.as_bytes()).collect::<String>()
        ));
    }
    if endpoint == "/v1/video/tasks" {
        return Ok(video_tasks_url(base));
    }
    if endpoint.starts_with("/v1/asset-groups/") {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(
            base,
            &format!(
                "/material/asset-groups/{}",
                endpoint.trim_start_matches("/v1/asset-groups/")
            ),
            query.as_deref(),
        );
    }
    if endpoint == "/v1/asset-groups" {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(base, "/material/asset-groups", query.as_deref());
    }
    if endpoint.starts_with("/v1/assets/") {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(
            base,
            &format!(
                "/material/assets/{}",
                endpoint.trim_start_matches("/v1/assets/")
            ),
            query.as_deref(),
        );
    }
    if endpoint == "/v1/assets" {
        let query = filtered_lingjing_query(uri.query());
        return lingjing_url(base, "/material/assets/create", query.as_deref());
    }
    if matches!(endpoint, "/search" | "/v1/search") && *method == Method::GET {
        return Ok(base.to_owned());
    }
    replace_known_endpoint(base, endpoint)
}

pub(crate) fn filtered_lingjing_query(query: Option<&str>) -> Option<String> {
    let mut output = url::form_urlencoded::Serializer::new(String::new());
    let mut retained = false;
    for (key, value) in url::form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        if matches!(key.as_ref(), "model" | "request_model") {
            continue;
        }
        retained = true;
        output.append_pair(&key, &value);
    }
    retained.then(|| output.finish())
}

pub(crate) fn typesafe_endpoint_url(base: &str) -> Result<String, String> {
    let mut url = Url::parse(base).map_err(|_| "Invalid TypeSafe base URL".to_owned())?;
    let mut path = url.path().trim_end_matches('/').to_owned();
    for suffix in ["/systemone", "/models"] {
        if path.ends_with(suffix) {
            path.truncate(path.len() - suffix.len());
            break;
        }
    }
    if !path.ends_with("/v1") {
        path.push_str("/v1");
    }
    url.set_path(&format!("{path}/systemone"));
    Ok(url.to_string())
}

pub(crate) fn replace_known_endpoint(base: &str, endpoint: &str) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid provider base URL: {error}"))?;
    let known = [
        "/chat/completions",
        "/images/generations",
        "/images/edits",
        "/audio/transcriptions",
        "/audio/speech",
        "/moderations",
        "/embeddings",
        "/responses/compact",
        "/responses",
        "/messages",
        "/systemone",
    ];
    let mut path = url.path().trim_end_matches('/').to_owned();
    for suffix in known {
        if path.ends_with(suffix) {
            path.truncate(path.len() - suffix.len());
            break;
        }
    }
    let endpoint = endpoint.trim_start_matches("/v1/");
    url.set_path(&format!("{}/{}", path.trim_end_matches('/'), endpoint));
    Ok(url.to_string())
}

pub(crate) fn responses_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/responses") {
        base.to_owned()
    } else {
        format!("{base}/responses")
    }
}

pub(crate) fn messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/messages") {
        base.to_owned()
    } else {
        format!("{base}/messages")
    }
}
