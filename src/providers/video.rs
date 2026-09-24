use crate::config::snapshot::Provider;
use crate::runtime::clock::unix_seconds;
use axum::http::Method;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use url::Url;

#[derive(Clone)]
pub(crate) struct VideoTaskRoute {
    pub(crate) provider_name: String,
    pub(crate) request_model: String,
    pub(crate) provider_key: String,
    pub(crate) video_tokens: Option<i64>,
    pub(crate) created_at: Instant,
}

pub(crate) static VIDEO_TASK_ROUTES: OnceLock<Mutex<HashMap<String, VideoTaskRoute>>> =
    OnceLock::new();

pub(crate) fn callxyq_video_url(
    provider: &Provider,
    task_id: Option<&str>,
    create: bool,
) -> Result<String, String> {
    let root = provider.base_url.trim_end_matches('/');
    let route_name = if create { "create_task" } else { "get_task" };
    let configured = provider
        .preferences
        .get("video_routes")
        .and_then(|v| v.get(route_name));
    let mut path = configured
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if create {
                "/v1/videos".into()
            } else {
                "/v1/videos/{task_id}".into()
            }
        });
    if let Some(id) = task_id {
        path = path.replace("{task_id}", &id.replace('/', "%2F"));
    }
    if !path.starts_with('/') {
        path.insert(0, '/');
    }
    Ok(format!("{root}{path}"))
}

pub(crate) fn callxyq_video_payload(input: &Value, model: &str) -> Result<Value, String> {
    let object = input
        .as_object()
        .ok_or_else(|| "video task request body must be an object".to_owned())?;
    let options = object
        .get("provider_options")
        .and_then(|v| v.get("callxyq"))
        .or_else(|| object.get("provider_options"))
        .and_then(Value::as_object);
    let get = |key: &str| object.get(key).or_else(|| options.and_then(|o| o.get(key)));
    let prompt = object
        .get("prompt")
        .and_then(Value::as_str)
        .or_else(|| object.get("content").and_then(Value::as_str))
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| "callxyq video requests require prompt".to_owned())?;
    let protocol = if model.to_ascii_lowercase().starts_with("gemini-veo") {
        "veo"
    } else {
        "sora"
    };
    let mut images = Vec::new();
    let mut videos = Vec::new();
    let mut audios = Vec::new();
    if let Some(parts) = object.get("content").and_then(Value::as_array) {
        for part in parts.iter().filter_map(Value::as_object) {
            let kind = part
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("image")
                .to_ascii_lowercase();
            let value = part
                .get("image_url")
                .or_else(|| part.get("video_url"))
                .or_else(|| part.get("audio_url"))
                .and_then(|v| v.as_str().or_else(|| v.get("url").and_then(Value::as_str)))
                .map(str::trim)
                .unwrap_or("");
            if value.is_empty() {
                continue;
            }
            if value.starts_with("asset://") || value.starts_with("Asset-") {
                return Err("callxyq resources require public URL or data URL; asset_id resources are not supported".into());
            }
            match kind.as_str() {
                "image" | "image_url" => images.push(value.to_owned()),
                "video" | "video_url" => videos.push(value.to_owned()),
                "audio" | "audio_url" => audios.push(value.to_owned()),
                other => return Err(format!("Unsupported callxyq resource type: {other}")),
            }
        }
    }
    let ratio = get("aspect_ratio")
        .or_else(|| get("ratio"))
        .and_then(Value::as_str)
        .unwrap_or("16:9")
        .to_owned();
    let size = get("size").and_then(Value::as_str).map(str::to_owned);
    let resolution = get("resolution")
        .and_then(Value::as_str)
        .unwrap_or("720p")
        .to_ascii_lowercase();
    let mut payload = Map::new();
    payload.insert("model".into(), json!(model));
    payload.insert("prompt".into(), json!(prompt));
    if protocol == "sora" {
        let sora2 = model.eq_ignore_ascii_case("sora-2");
        let sora3 = matches!(
            model.to_ascii_lowercase().as_str(),
            "sora-v3-fast" | "sora-v3-pro"
        );
        if !sora2 && !sora3 {
            return Err(format!("Unsupported callxyq Sora model: {model}"));
        }
        let allowed = if sora2 {
            ["16:9", "9:16"].as_slice()
        } else {
            ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"].as_slice()
        };
        if !allowed.contains(&ratio.as_str()) {
            return Err(format!(
                "{model} does not support ratio/aspect_ratio {ratio}"
            ));
        }
        if resolution != "720p" && (sora2 || resolution != "480p") {
            return Err("callxyq Sora resolution is unsupported".into());
        }
        let seconds = get("seconds")
            .or_else(|| get("duration"))
            .and_then(|v| v.as_i64())
            .unwrap_or(if sora2 { 4 } else { 5 });
        if (sora2 && ![4, 8, 12].contains(&seconds)) || (!sora2 && !(5..=15).contains(&seconds)) {
            return Err("callxyq Sora duration is unsupported".into());
        }
        if (sora2 && (!videos.is_empty() || !audios.is_empty()))
            || images.len() > if sora2 { 1 } else { 4 }
            || videos.len() > 3
            || audios.len() > 1
            || (!audios.is_empty() && images.is_empty())
        {
            return Err("callxyq Sora resource limits exceeded".into());
        }
        if images.len() > 1 || !videos.is_empty() || !audios.is_empty() {
            for i in 1..=images.len() {
                if !prompt.contains(&format!("@Image{i}")) {
                    return Err(format!(
                        "callxyq Sora multi-resource prompts must reference @Image{i}"
                    ));
                }
            }
            for i in 1..=videos.len() {
                if !prompt.contains(&format!("@Video{i}")) {
                    return Err(format!(
                        "callxyq Sora multi-resource prompts must reference @Video{i}"
                    ));
                }
            }
            for i in 1..=audios.len() {
                if !prompt.contains(&format!("@Audio{i}")) {
                    return Err(format!(
                        "callxyq Sora audio prompts must reference @Audio{i}"
                    ));
                }
            }
        }
        payload.insert("aspect_ratio".into(), json!(ratio));
        payload.insert("resolution".into(), json!(resolution));
        payload.insert("seconds".into(), json!(seconds));
        let computed_size = size.unwrap_or_else(|| {
            if ratio == "9:16" {
                format!("720x{}", resolution.trim_end_matches('p'))
            } else {
                format!(
                    "{}x{}",
                    (16 * resolution
                        .trim_end_matches('p')
                        .parse::<i64>()
                        .unwrap_or(720))
                        / 9,
                    resolution.trim_end_matches('p')
                )
            }
        });
        payload.insert("size".into(), json!(computed_size));
        if let Some(first) = images.first() {
            payload.insert(
                if sora2 || images.len() == 1 {
                    "image_url"
                } else {
                    "reference_image_urls"
                }
                .into(),
                if sora2 || images.len() == 1 {
                    json!(first)
                } else {
                    json!(images)
                },
            );
        }
        if let Some(first) = videos.first() {
            payload.insert("reference_video".into(), json!(first));
            payload.insert("reference_videos".into(), json!(videos));
        }
        if let Some(first) = audios.first() {
            payload.insert("audio_url".into(), json!(first));
            payload.insert(
                "video_config".into(),
                get("video_config").cloned().unwrap_or_else(
                    || json!({"reference_mode":"image_reference","motion_has_audio":true}),
                ),
            );
        }
    } else {
        if !videos.is_empty() || !audios.is_empty() {
            return Err("callxyq Veo models do not support video or audio resources".into());
        }
        let max_images = if model.contains("-ref-") { 3 } else { 2 };
        if let Some(encoded) = model
            .rsplit_once('-')
            .and_then(|(_, suffix)| suffix.strip_suffix('s'))
            .and_then(|v| v.parse::<i64>().ok())
        {
            if let Some(requested) = get("duration")
                .or_else(|| get("seconds"))
                .and_then(Value::as_i64)
            {
                if requested != encoded {
                    return Err(format!("Veo duration is encoded in model name ({encoded}s); request duration {requested}s does not match"));
                }
            }
        }
        if images.len() > max_images {
            return Err("callxyq Veo image reference limit exceeded".into());
        }
        let size = size.unwrap_or_else(|| {
            if ratio == "9:16" {
                if resolution == "1080p" {
                    "1080x1920"
                } else {
                    "720x1280"
                }
            } else if ratio == "16:9" {
                if resolution == "1080p" {
                    "1920x1080"
                } else {
                    "1280x720"
                }
            } else {
                ""
            }
            .into()
        });
        if !["1280x720", "720x1280", "1920x1080", "1080x1920"].contains(&size.as_str()) {
            return Err(format!("Unsupported callxyq Veo size: {size}"));
        }
        payload.insert("size".into(), json!(size));
        if let Some(v) = get("generate_audio").or_else(|| get("audio")) {
            payload.insert("generate_audio".into(), json!(v.as_bool().unwrap_or(false)));
        }
        if let Some(first) = images.first() {
            payload.insert(
                if images.len() == 1 {
                    "image_url"
                } else {
                    "images"
                }
                .into(),
                if images.len() == 1 {
                    json!(first)
                } else {
                    json!(images)
                },
            );
        }
    }
    Ok(Value::Object(payload))
}

pub(crate) fn normalize_callxyq_video_response(
    method: &Method,
    model: &str,
    obj: &Value,
    estimated: Option<i64>,
) -> Value {
    let root = obj.as_object().cloned().unwrap_or_default();
    let id = root
        .get("id")
        .or_else(|| root.get("task_id"))
        .cloned()
        .unwrap_or(Value::Null);
    let status = match root
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "completed" => "succeeded",
        "in_progress" | "processing" => "running",
        "canceled" | "cancelled" => "cancelled",
        "failed" => "failed",
        "queued" => "queued",
        _ => "queued",
    };
    let mut out = json!({"id":id,"model":model,"provider":"callxyq","status":status});
    if method == Method::POST {
        if let Some(v) = root.get("created_at") {
            out["created_at"] = v.clone();
        }
    } else {
        let mut video = json!({});
        if let Some(v) = root.get("video_url") {
            video["url"] = v.clone();
        }
        if let Some(v) = root.get("size") {
            video["size"] = v.clone();
        }
        out["video"] = video;
        if status == "succeeded" {
            out["usage"] = json!({"video_tokens":estimated.unwrap_or(0),"completion_tokens":estimated.unwrap_or(0),"total_tokens":estimated.unwrap_or(0)});
        }
    }
    out
}

pub(crate) fn video_tasks_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/contents/generations/tasks") {
        base.to_owned()
    } else if Url::parse(base)
        .ok()
        .is_some_and(|url| url.path().is_empty() || url.path() == "/")
    {
        format!("{base}/api/v3/contents/generations/tasks")
    } else {
        format!("{base}/contents/generations/tasks")
    }
}

pub(crate) fn content_generation_to_lingjing(
    input: &Value,
    model_code: &str,
) -> Result<Value, String> {
    let request = input
        .as_object()
        .ok_or_else(|| "video task request body must be an object".to_owned())?;
    if request.contains_key("taskParams") || request.contains_key("modelCode") {
        let mut payload = request.clone();
        payload.insert("modelCode".into(), Value::String(model_code.to_owned()));
        for key in [
            "model",
            "request_model",
            "provider",
            "provider_options",
            "route",
        ] {
            payload.remove(key);
        }
        return Ok(Value::Object(payload));
    }

    let mut prompt_parts = Vec::new();
    let mut content_resources = Vec::new();
    for part in request
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        match part.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => {
                if let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                {
                    prompt_parts.push(text.to_owned());
                }
            }
            kind @ ("image_url" | "video_url" | "audio_url") => {
                let resource_type = kind.trim_end_matches("_url");
                if let Some(url) = content_part_url(part, kind) {
                    let mut resource = Map::new();
                    resource.insert("type".into(), Value::String(resource_type.to_owned()));
                    resource.insert(
                        "usage".into(),
                        Value::String(lingjing_resource_usage(
                            part.get("role"),
                            resource_type,
                            content_resources.len(),
                        )),
                    );
                    resource.insert("source".into(), lingjing_source(&url));
                    if let Some(reference_key) = part.get("reference_key") {
                        resource.insert("reference_key".into(), reference_key.clone());
                    }
                    content_resources.push(Value::Object(resource));
                }
            }
            _ => {}
        }
    }

    let prompt = request
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| prompt_parts.join("\n"));
    let mut task_input = Map::from_iter([("prompt".into(), Value::String(prompt))]);
    let quality = request.get("quality").cloned().or_else(|| {
        request.get("resolution").and_then(|value| {
            let raw = value.as_str()?.trim().to_ascii_lowercase();
            Some(Value::String(
                raw.strip_suffix('p').unwrap_or(raw.as_str()).to_owned(),
            ))
        })
    });
    if let Some(quality) = quality.filter(|value| !value.is_null()) {
        task_input.insert(
            "quality".into(),
            Value::String(value_text(&quality).unwrap_or_default()),
        );
    }
    for key in ["duration", "ratio", "generate_num", "prompt_optimizer"] {
        if let Some(value) = request.get(key).filter(|value| !value.is_null()) {
            task_input.insert(key.into(), value.clone());
        }
    }
    let unified_resources = request
        .get("resources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, resource)| normalize_lingjing_resource(resource, index))
        .collect::<Vec<_>>();
    if !unified_resources.is_empty() {
        task_input.insert("resources".into(), Value::Array(unified_resources));
    } else if !content_resources.is_empty() {
        task_input.insert("resources".into(), Value::Array(content_resources));
    }
    if let Some(options) = request
        .get("provider_options")
        .and_then(Value::as_object)
        .and_then(|options| {
            options
                .get("lingjing")
                .and_then(Value::as_object)
                .or(Some(options))
        })
    {
        for (key, value) in options {
            if !value.is_object() && !value.is_null() {
                task_input.insert(key.clone(), value.clone());
            }
        }
    }
    for key in ["generate_audio", "need_audio", "audio"] {
        if let Some(value) = request.get(key) {
            task_input.insert(
                "need_audio".into(),
                Value::Bool(value.as_bool().unwrap_or(false)),
            );
        }
    }
    Ok(json!({"modelCode":model_code,"taskParams":{"input":task_input}}))
}

pub(crate) fn content_part_url(part: &Map<String, Value>, key: &str) -> Option<String> {
    let value = part.get(key)?;
    let raw = value
        .as_str()
        .or_else(|| value.get("url").and_then(Value::as_str))?
        .trim();
    (!raw.is_empty()).then(|| raw.to_owned())
}

pub(crate) fn lingjing_source(value: &str) -> Value {
    if let Some(asset_id) = value.strip_prefix("asset://") {
        return json!({"kind":"asset_id","value":asset_id});
    }
    if value.starts_with("Asset-") {
        return json!({"kind":"asset_id","value":value});
    }
    json!({"kind":"url","value":value})
}

pub(crate) fn lingjing_resource_usage(
    role: Option<&Value>,
    resource_type: &str,
    index: usize,
) -> String {
    let role = role
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if matches!(
        role.as_str(),
        "first_frame" | "last_frame" | "reference" | "keyframe" | "source"
    ) {
        return role;
    }
    if matches!(
        role.as_str(),
        "reference_image" | "reference_video" | "reference_audio"
    ) {
        return "reference".into();
    }
    if resource_type == "image" && index == 0 {
        "first_frame".into()
    } else {
        "reference".into()
    }
}

pub(crate) fn normalize_lingjing_resource(value: &Value, index: usize) -> Option<Value> {
    let resource = value.as_object()?;
    let resource_type = resource
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("image")
        .trim()
        .to_ascii_lowercase();
    if !matches!(resource_type.as_str(), "image" | "video" | "audio") {
        return None;
    }
    let source = resource
        .get("source")
        .filter(|value| value.is_object())
        .cloned()
        .or_else(|| {
            ["url", "asset_id", "assetId", "value"]
                .into_iter()
                .find_map(|key| resource.get(key).and_then(value_text))
                .map(|value| lingjing_source(&value))
        })?;
    let mut normalized = Map::from_iter([
        ("type".into(), Value::String(resource_type.clone())),
        (
            "usage".into(),
            Value::String(lingjing_resource_usage(
                resource.get("usage").or_else(|| resource.get("role")),
                &resource_type,
                index,
            )),
        ),
        ("source".into(), source),
    ]);
    if let Some(reference_key) = resource
        .get("reference_key")
        .or_else(|| resource.get("referenceKey"))
    {
        normalized.insert("reference_key".into(), reference_key.clone());
    }
    Some(Value::Object(normalized))
}

pub(crate) fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

pub(crate) fn normalize_lingjing_video_response(
    method: &Method,
    request_model: &str,
    url: &str,
    upstream: &Value,
) -> Value {
    let Some(root) = upstream.as_object() else {
        return upstream.clone();
    };
    let data = root.get("data").and_then(Value::as_object);
    if *method == Method::POST {
        let task_id = data.and_then(|data| {
            data.get("taskId")
                .or_else(|| data.get("task_id"))
                .and_then(value_text)
        });
        return task_id.map_or_else(
            || upstream.clone(),
            |task_id| {
                json!({
                    "id":task_id,
                    "model":request_model,
                    "provider":"lingjing",
                    "status":"queued",
                    "created_at":unix_seconds(),
                })
            },
        );
    }
    if *method != Method::GET {
        return upstream.clone();
    }
    let data = data.cloned().unwrap_or_default();
    let query_task_id = Url::parse(url).ok().and_then(|url| {
        url.query_pairs()
            .find(|(key, _)| key == "taskId")
            .map(|(_, value)| value.into_owned())
    });
    let task_id = data
        .get("task_id")
        .or_else(|| data.get("taskId"))
        .and_then(value_text)
        .or(query_task_id)
        .unwrap_or_default();
    let upstream_status = data
        .get("status")
        .and_then(value_text)
        .unwrap_or_default()
        .to_ascii_uppercase();
    let status = match upstream_status.as_str() {
        "SUCCESS" => "succeeded",
        "CANCELED" => "cancelled",
        "FAIL" | "FAILED" | "UNKNOWN" => "failed",
        "WAITING" | "QUEUED" | "SUBMITTED" | "RUNNING" | "" => "running",
        other => other,
    };
    let video_url = data
        .get("result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|item| item.get("url").and_then(Value::as_str));
    let mut normalized = json!({
        "id":task_id,
        "model":request_model,
        "provider":"lingjing",
        "status":status.to_ascii_lowercase(),
        "video":{},
    });
    if let Some(url) = video_url {
        normalized["video"]["url"] = Value::String(url.to_owned());
    }
    if let Some(error) = data.get("external_error").filter(|value| !value.is_null()) {
        normalized["error"] = json!({"message":error});
    }
    if normalized["status"] == "succeeded" {
        let video_tokens = video_task_route(&task_id)
            .and_then(|route| route.video_tokens)
            .unwrap_or(108_900);
        normalized["usage"] = json!({
            "video_tokens":video_tokens,
            "completion_tokens":video_tokens,
            "total_tokens":video_tokens,
        });
    }
    normalized
}

pub(crate) fn estimate_video_tokens(payload: &Value) -> Option<i64> {
    let root = payload.as_object()?;
    let positive = |key: &str| {
        root.get(key).and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_f64().map(|value| value as i64))
                .or_else(|| value.as_str()?.trim_end_matches(['p', 'P']).parse().ok())
                .filter(|value| *value > 0)
        })
    };
    let duration = positive("duration").unwrap_or(5);
    let fps = positive("fps")
        .or_else(|| positive("framespersecond"))
        .unwrap_or(24);
    let resolution = positive("quality")
        .or_else(|| positive("resolution"))
        .unwrap_or(720);
    let scale = (resolution as f64 / 720.0).powi(2);
    Some((duration as f64 * fps as f64 * 907.5 * scale).round() as i64)
}

pub(crate) fn video_task_routes() -> &'static Mutex<HashMap<String, VideoTaskRoute>> {
    VIDEO_TASK_ROUTES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn video_task_route_for_path(path: &str) -> Option<VideoTaskRoute> {
    let task_id = path.strip_prefix("/v1/video/tasks/")?;
    video_task_route(task_id)
}

pub(crate) fn video_task_route(task_id: &str) -> Option<VideoTaskRoute> {
    let mut routes = video_task_routes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let now = Instant::now();
    routes.retain(|_, route| now.duration_since(route.created_at) < Duration::from_secs(86_400));
    routes.get(task_id).cloned()
}

pub(crate) fn remember_video_task(
    task_id: &str,
    provider_name: &str,
    request_model: &str,
    provider_key: &str,
    video_tokens: Option<i64>,
) {
    let mut routes = video_task_routes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if routes.len() >= 4096 {
        if let Some(oldest) = routes
            .iter()
            .min_by_key(|(_, route)| route.created_at)
            .map(|(task_id, _)| task_id.clone())
        {
            routes.remove(&oldest);
        }
    }
    routes.insert(
        task_id.to_owned(),
        VideoTaskRoute {
            provider_name: provider_name.to_owned(),
            request_model: request_model.to_owned(),
            provider_key: provider_key.to_owned(),
            video_tokens,
            created_at: Instant::now(),
        },
    );
}

pub(crate) fn lingjing_url(
    base: &str,
    openapi_path: &str,
    query: Option<&str>,
) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid Lingjing base URL: {error}"))?;
    let base_path = url.path().trim_end_matches('/');
    let suffix = openapi_path.trim_start_matches('/');
    let path = if base_path.ends_with("/api/entrance/openapi") {
        format!("{base_path}/{suffix}")
    } else if base_path.ends_with("/api/entrance") {
        format!("{base_path}/openapi/{suffix}")
    } else {
        format!("{base_path}/api/entrance/openapi/{suffix}")
    };
    url.set_path(&path);
    url.set_query(query);
    Ok(url.to_string())
}
