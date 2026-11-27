use bytes::{Bytes, BytesMut};
use memchr::memchr2;

// Match the default bounded non-stream response body. Zero from older plans
// selects the default instead of disabling protection.
pub(crate) const DEFAULT_MAX_SSE_EVENT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn max_event_bytes() -> usize {
    std::env::var("RUST_RESPONSES_MAX_EVENT_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_SSE_EVENT_BYTES)
}

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
            max_event_bytes: if max_event_bytes == 0 {
                self::max_event_bytes()
            } else {
                max_event_bytes
            },
        }
    }

    pub(crate) fn exceeds_event_limit(&self, observed_bytes: usize) -> bool {
        observed_bytes > self.max_event_bytes
    }

    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, String> {
        self.feed_inner(chunk, false)
    }

    fn feed_inner(&mut self, mut chunk: &[u8], eof: bool) -> Result<Vec<SseFrame>, String> {
        let mut frames = Vec::new();
        loop {
            // Do not copy an arbitrarily large transport chunk before checking the
            // bound. Leave room for the longest delimiter, including split CRLFs.
            let room = self
                .max_event_bytes
                .saturating_add(4)
                .saturating_sub(self.buffer.len());
            let consumed = room.min(chunk.len());
            self.buffer.extend_from_slice(&chunk[..consumed]);
            chunk = &chunk[consumed..];
            while let Some((end, delimiter_len)) =
                find_event_delimiter_from(&self.buffer, self.scan_from, eof)
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
            if self.buffer.len() > self.max_event_bytes.saturating_add(3) {
                return Err(
                    "Responses upstream SSE pending frame exceeds the configured limit".into(),
                );
            }
            if chunk.is_empty() {
                return Ok(frames);
            }
        }
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<SseFrame>, String> {
        let mut frames = self.feed_inner(&[], true)?;
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

fn find_event_delimiter_from(bytes: &[u8], start: usize, eof: bool) -> Option<(usize, usize)> {
    let mut index = start.min(bytes.len());
    while index < bytes.len() {
        let relative = memchr2(b'\r', b'\n', &bytes[index..])?;
        index += relative;
        let first = if bytes[index..].starts_with(b"\r\n") {
            2
        } else {
            1
        };
        let tail = &bytes[index + first..];
        match tail.first() {
            Some(b'\n') => return Some((index, first + 1)),
            Some(b'\r') if tail.len() > 1 => {
                return Some((index, first + if tail[1] == b'\n' { 2 } else { 1 }));
            }
            Some(b'\r') if eof => return Some((index, first + 1)),
            // The next chunk may complete the second CRLF.
            Some(b'\r') | None => return None,
            _ => {}
        }
        index += first;
    }
    None
}

pub(crate) fn parse_sse_frame(frame: &SseFrame) -> Result<ParsedSseFrame, String> {
    let text = std::str::from_utf8(frame.raw())
        .map_err(|_| "Responses upstream SSE event is not valid UTF-8".to_owned())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
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

#[cfg(test)]
mod bounded_decoder_tests {
    use super::*;

    #[test]
    fn pending_and_complete_frames_are_checked_before_large_copies() {
        for suffix in [b"".as_slice(), b"\n\n".as_slice()] {
            let mut decoder = SseDecoder::new(1024);
            let mut chunk = vec![b'x'; 1024 * 1024];
            chunk.extend_from_slice(suffix);
            assert!(decoder.feed(&chunk).is_err());
            assert!(decoder.buffer.len() <= 1028);
        }
        let mut decoder = SseDecoder::new(1024);
        for _ in 0..4 {
            assert!(decoder.feed(&[b'x'; 256]).unwrap().is_empty());
        }
        assert!(decoder.feed(&[b'x'; 4]).is_err());
    }

    #[test]
    fn large_chunks_of_small_events_and_split_delimiters_remain_valid() {
        let mut decoder = SseDecoder::new(8);
        assert_eq!(
            decoder.feed(&b"data: x\n\n".repeat(100)).unwrap().len(),
            100
        );
        assert!(decoder.feed(b"12345678\r\n\r").unwrap().is_empty());
        assert_eq!(decoder.feed(b"\n").unwrap()[0].raw(), b"12345678");
    }

    #[test]
    fn legacy_zero_selects_a_finite_default() {
        assert!(SseDecoder::new(0).max_event_bytes > 0);
    }
}
