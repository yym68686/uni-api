from __future__ import annotations

from typing import Any

from uni_api.serialization import json


class IncrementalSSEParser:
    """Incrementally split SSE text into complete raw event frames."""

    def __init__(self):
        self._buffer = ""

    @property
    def pending_text(self) -> str:
        return self._buffer

    def feed(self, chunk: str | bytes | bytearray) -> list[str]:
        if isinstance(chunk, (bytes, bytearray)):
            chunk = bytes(chunk).decode("utf-8", errors="replace")
        self._buffer += str(chunk).replace("\r\n", "\n").replace("\r", "\n")

        events = []
        while True:
            separator_index = self._buffer.find("\n\n")
            if separator_index < 0:
                break
            events.append(self._buffer[:separator_index])
            self._buffer = self._buffer[separator_index + 2 :]
        return events


def is_sse_comment_frame(raw_event: str) -> bool:
    has_line = False
    for line in raw_event.splitlines():
        if not line:
            continue
        has_line = True
        if not line.startswith(":"):
            return False
    return has_line


def parse_sse_event(raw_event: str) -> tuple[str, Any]:
    event_name = ""
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    data_str = "\n".join(data_lines).strip()
    if data_str == "[DONE]":
        return "[DONE]", "[DONE]"

    parsed_payload: Any = data_str
    if data_str:
        try:
            parsed_payload = json.loads(data_str)
        except Exception:
            parsed_payload = data_str

    if not event_name and isinstance(parsed_payload, dict):
        event_name = str(parsed_payload.get("type") or "").strip()

    return event_name, parsed_payload
