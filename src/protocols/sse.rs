use bytes::{Bytes, BytesMut};
use memchr::memchr2;

pub(crate) const UNLIMITED_SSE_EVENT_BYTES: usize = 0;

#[derive(Debug)]
pub(crate) struct SseFrame {
    pub(crate) wire: Bytes,
    pub(crate) raw_len: usize,
    pub(crate) terminated: bool,
}

impl SseFrame {
    pub(crate) fn raw(&self) -> &[u8] {
        &self.wire[..self.raw_len]
    }

    pub(crate) fn canonical_wire(&self) -> Bytes {
        if self.terminated {
            self.wire.clone()
        } else {
            let mut wire = BytesMut::with_capacity(self.wire.len().saturating_add(2));
            wire.extend_from_slice(&self.wire);
            wire.extend_from_slice(b"\n\n");
            wire.freeze()
        }
    }
}

pub(crate) struct ParsedSseFrame {
    pub(crate) raw: String,
    pub(crate) declared_event: Option<String>,
    pub(crate) data: Option<String>,
    pub(crate) comment_only: bool,
}

pub(crate) struct SseDecoder {
    pub(crate) buffer: BytesMut,
    pub(crate) scan_from: usize,
    pub(crate) max_event_bytes: usize,
}

impl SseDecoder {
    pub(crate) fn new(max_event_bytes: usize) -> Self {
        Self {
            buffer: BytesMut::new(),
            scan_from: 0,
            max_event_bytes,
        }
    }

    pub(crate) fn exceeds_event_limit(&self, observed_bytes: usize) -> bool {
        self.max_event_bytes != UNLIMITED_SSE_EVENT_BYTES && observed_bytes > self.max_event_bytes
    }

    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some((end, delimiter_len)) =
            find_event_delimiter_from(&self.buffer, self.scan_from)
        {
            if self.exceeds_event_limit(end) {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let wire = self.buffer.split_to(end + delimiter_len).freeze();
            self.scan_from = 0;
            if wire[..end].iter().all(|byte| byte.is_ascii_whitespace()) {
                continue;
            }
            frames.push(SseFrame {
                wire,
                raw_len: end,
                terminated: true,
            });
        }
        self.scan_from = self.buffer.len().saturating_sub(3);
        if self.max_event_bytes != UNLIMITED_SSE_EVENT_BYTES
            && self.buffer.len() > self.max_event_bytes.saturating_add(64 * 1024)
        {
            return Err("Responses upstream SSE pending frame exceeds the configured limit".into());
        }
        Ok(frames)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<SseFrame>, String> {
        let mut frames = self.feed(&[])?;
        if !self.buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            if self.exceeds_event_limit(self.buffer.len()) {
                return Err("Responses upstream SSE event exceeds the configured limit".into());
            }
            let raw_len = self.buffer.len();
            frames.push(SseFrame {
                wire: self.buffer.split().freeze(),
                raw_len,
                terminated: false,
            });
        }
        self.buffer.clear();
        self.scan_from = 0;
        Ok(frames)
    }
}

pub(crate) fn find_event_delimiter_from(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut index = start.min(bytes.len());
    while index < bytes.len() {
        let relative = memchr2(b'\r', b'\n', &bytes[index..])?;
        index += relative;
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") || bytes[index..].starts_with(b"\r\r") {
            return Some((index, 2));
        }
        index += 1;
    }
    None
}

pub(crate) fn parse_sse_frame(frame: &SseFrame) -> Result<ParsedSseFrame, String> {
    let text = std::str::from_utf8(frame.raw())
        .map_err(|_| "Responses upstream SSE event is not valid UTF-8".to_owned())?;
    let normalized = if text.as_bytes().contains(&b'\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    let mut declared_event = None;
    let mut data_lines = Vec::new();
    let mut saw_field = false;
    let mut saw_comment = false;
    for line in normalized.split('\n') {
        if let Some(_comment) = line.strip_prefix(':') {
            saw_comment = true;
            continue;
        }
        if line.is_empty() {
            continue;
        }
        saw_field = true;
        let (field, mut value) = line.split_once(':').unwrap_or((line, ""));
        if let Some(stripped) = value.strip_prefix(' ') {
            value = stripped;
        }
        match field {
            "event" => declared_event = Some(value.to_owned()),
            "data" => data_lines.push(value.to_owned()),
            _ => {}
        }
    }
    Ok(ParsedSseFrame {
        raw: normalized,
        declared_event,
        data: (!data_lines.is_empty()).then(|| data_lines.join("\n")),
        comment_only: saw_comment && !saw_field,
    })
}
