use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode};

const ALLOW_METHODS: &str = "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT";

pub struct CorsRequest {
    origin: Option<HeaderValue>,
    has_cookie: bool,
}

impl CorsRequest {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            origin: headers.get("origin").cloned(),
            has_cookie: headers.contains_key("cookie"),
        }
    }

    pub fn preflight_response(
        &self,
        method: &Method,
        headers: &HeaderMap,
    ) -> Option<Response<Body>> {
        let origin = self.origin.as_ref()?;
        if *method != Method::OPTIONS {
            return None;
        }
        let requested_method = headers.get("access-control-request-method")?;
        let allowed = matches!(
            requested_method.as_bytes(),
            b"DELETE" | b"GET" | b"HEAD" | b"OPTIONS" | b"PATCH" | b"POST" | b"PUT"
        );
        let (status, message) = if allowed {
            (StatusCode::OK, "OK")
        } else {
            (StatusCode::BAD_REQUEST, "Disallowed CORS method")
        };
        let mut response = Response::new(Body::from(message));
        *response.status_mut() = status;
        let response_headers = response.headers_mut();
        response_headers.insert(
            "content-type",
            HeaderValue::from_static("text/plain; charset=utf-8"),
        );
        response_headers.insert(
            "content-length",
            HeaderValue::from_static(if allowed { "2" } else { "22" }),
        );
        response_headers.insert("vary", HeaderValue::from_static("Origin"));
        response_headers.insert(
            "access-control-allow-methods",
            HeaderValue::from_static(ALLOW_METHODS),
        );
        response_headers.insert("access-control-max-age", HeaderValue::from_static("600"));
        response_headers.insert(
            "access-control-allow-credentials",
            HeaderValue::from_static("true"),
        );
        response_headers.insert("access-control-allow-origin", origin.clone());
        if let Some(requested_headers) = headers.get("access-control-request-headers") {
            response_headers.insert("access-control-allow-headers", requested_headers.clone());
        }
        Some(response)
    }

    pub fn apply(&self, response: &mut Response<Body>) {
        let Some(origin) = self.origin.as_ref() else {
            return;
        };
        let headers = response.headers_mut();
        headers.insert(
            "access-control-allow-credentials",
            HeaderValue::from_static("true"),
        );
        if self.has_cookie {
            headers.insert("access-control-allow-origin", origin.clone());
            append_vary_origin(headers);
        } else {
            headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
        }
    }
}

fn append_vary_origin(headers: &mut HeaderMap) {
    let value = match headers.get("vary") {
        Some(existing) => {
            let mut combined = existing.as_bytes().to_vec();
            combined.extend_from_slice(b", Origin");
            HeaderValue::from_bytes(&combined)
                .unwrap_or_else(|_| HeaderValue::from_static("Origin"))
        }
        None => HeaderValue::from_static("Origin"),
    };
    headers.insert("vary", value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[test]
    fn simple_cross_origin_response_matches_starlette_without_cookies() {
        let mut request_headers = HeaderMap::new();
        request_headers.insert("origin", HeaderValue::from_static("https://client.example"));
        let cors = CorsRequest::from_headers(&request_headers);
        let mut response = Response::new(Body::empty());

        cors.apply(&mut response);

        assert_eq!(response.headers()["access-control-allow-origin"], "*");
        assert_eq!(
            response.headers()["access-control-allow-credentials"],
            "true"
        );
        assert!(!response.headers().contains_key("vary"));
    }

    #[test]
    fn credentialed_response_mirrors_origin_and_appends_vary() {
        let mut request_headers = HeaderMap::new();
        request_headers.insert("origin", HeaderValue::from_static("https://client.example"));
        request_headers.insert("cookie", HeaderValue::from_static("session=one"));
        let cors = CorsRequest::from_headers(&request_headers);
        let mut response = Response::new(Body::empty());
        response
            .headers_mut()
            .insert("vary", HeaderValue::from_static("Accept-Encoding"));

        cors.apply(&mut response);

        assert_eq!(
            response.headers()["access-control-allow-origin"],
            "https://client.example"
        );
        assert_eq!(response.headers()["vary"], "Accept-Encoding, Origin");
    }

    #[test]
    fn request_without_origin_is_unchanged() {
        let cors = CorsRequest::from_headers(&HeaderMap::new());
        let mut response = Response::new(Body::empty());

        cors.apply(&mut response);

        assert!(!response
            .headers()
            .contains_key("access-control-allow-origin"));
        assert!(!response
            .headers()
            .contains_key("access-control-allow-credentials"));
    }

    #[tokio::test]
    async fn preflight_mirrors_origin_and_requested_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("https://client.example"));
        headers.insert(
            "access-control-request-method",
            HeaderValue::from_static("POST"),
        );
        headers.insert(
            "access-control-request-headers",
            HeaderValue::from_static("authorization, x-request-id"),
        );
        let cors = CorsRequest::from_headers(&headers);

        let response = cors
            .preflight_response(&Method::OPTIONS, &headers)
            .expect("preflight response");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["access-control-allow-origin"],
            "https://client.example"
        );
        assert_eq!(
            response.headers()["access-control-allow-methods"],
            ALLOW_METHODS
        );
        assert_eq!(
            response.headers()["access-control-allow-headers"],
            "authorization, x-request-id"
        );
        assert_eq!(response.headers()["vary"], "Origin");
        assert_eq!(
            to_bytes(response.into_body(), 64).await.unwrap().as_ref(),
            b"OK"
        );
    }

    #[tokio::test]
    async fn preflight_rejects_methods_outside_starlette_allow_all_set() {
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("https://client.example"));
        headers.insert(
            "access-control-request-method",
            HeaderValue::from_static("CONNECT"),
        );
        let cors = CorsRequest::from_headers(&headers);

        let response = cors
            .preflight_response(&Method::OPTIONS, &headers)
            .expect("preflight response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()["access-control-allow-origin"],
            "https://client.example"
        );
        assert_eq!(
            to_bytes(response.into_body(), 64).await.unwrap().as_ref(),
            b"Disallowed CORS method"
        );
    }

    #[test]
    fn ordinary_options_request_is_not_treated_as_preflight() {
        let mut headers = HeaderMap::new();
        headers.insert("origin", HeaderValue::from_static("https://client.example"));
        let cors = CorsRequest::from_headers(&headers);

        assert!(cors
            .preflight_response(&Method::OPTIONS, &headers)
            .is_none());
    }
}
