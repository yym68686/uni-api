//! Bounded response capture shared by both execution paths.
use crate::runtime::idempotency::Owner;
use axum::{
    body::Body,
    http::{HeaderMap, Response, StatusCode},
};
use bytes::Bytes;
use futures_util::StreamExt;

pub(crate) struct ReplayCapture {
    owner: Option<Owner>,
    status: StatusCode,
    headers: HeaderMap,
    chunks: Vec<Bytes>,
    bytes: usize,
    overflowed: bool,
}

impl ReplayCapture {
    pub fn new(owner: Owner, status: StatusCode, headers: HeaderMap) -> Self {
        Self {
            owner: Some(owner),
            status,
            headers,
            chunks: Vec::new(),
            bytes: 0,
            overflowed: false,
        }
    }
    pub fn push(&mut self, chunk: &Bytes) {
        let owner = self.owner.as_ref().expect("capture owns its claim");
        if self.overflowed {
            return;
        }
        let next = self.bytes.saturating_add(chunk.len());
        if next > owner.max_response_bytes() || !owner.try_reserve_inflight_response(chunk.len()) {
            owner.release_inflight_response(self.bytes);
            self.bytes = 0;
            self.chunks.clear();
            self.overflowed = true;
        } else {
            self.bytes = next;
            self.chunks.push(chunk.clone());
        }
    }
    pub async fn finish(mut self, completed: bool) {
        let owner = self.owner.take().expect("capture owns its claim");
        let bytes = self.bytes;
        owner.release_inflight_response(self.bytes);
        self.bytes = 0;
        if !completed {
            owner.nonreplayable("incomplete_response").await;
        } else if self.overflowed {
            owner.nonreplayable("response_too_large").await;
        } else {
            owner
                .complete(
                    self.status,
                    std::mem::take(&mut self.headers),
                    std::mem::take(&mut self.chunks),
                    bytes,
                )
                .await;
        }
    }
}

impl Drop for ReplayCapture {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.release_inflight_response(self.bytes);
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(owner.nonreplayable("capture_interrupted"));
            }
        }
    }
}

pub(crate) async fn capture_stream(mut response: Response<Body>, owner: Owner) -> Response<Body> {
    let Some(completion) = response
        .extensions_mut()
        .remove::<crate::observability::outcome::CompletionReceipt>()
    else {
        owner.nonreplayable("stream_completion_unknown").await;
        return response;
    };
    crate::runtime::idempotency::executed_header(response.headers_mut());
    let (parts, body) = response.into_parts();
    let capture = ReplayCapture::new(owner, parts.status, parts.headers.clone());
    let stream =
        futures_util::stream::unfold(Some((body.into_data_stream(), capture)), move |state| {
            let mut receipt = completion.0.clone();
            async move {
                let (mut body, mut capture) = state?;
                match body.next().await {
                    Some(Ok(bytes)) => {
                        capture.push(&bytes);
                        Some((Ok(bytes), Some((body, capture))))
                    }
                    Some(Err(error)) => {
                        capture.finish(false).await;
                        Some((Err(error), None))
                    }
                    None => {
                        if receipt.borrow().is_none() {
                            let _ = tokio::time::timeout(
                                std::time::Duration::from_secs(5),
                                receipt.changed(),
                            )
                            .await;
                        }
                        let completed = *receipt.borrow() == Some(true);
                        capture.finish(completed).await;
                        None
                    }
                }
            }
        });
    Response::from_parts(parts, Body::from_stream(stream))
}
