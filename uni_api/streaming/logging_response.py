from __future__ import annotations

import asyncio
from time import time
from typing import Any, Awaitable, Callable, Optional

from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from core.log_config import logger
from core.utils import safe_get
from uni_api.serialization import json
from uni_api.streaming.cleanup import call_cleanup_safely
from uni_api.streaming.sse import is_sse_comment_frame


class LoggingStreamingResponse(Response):
    def __init__(
        self,
        content,
        status_code=200,
        headers=None,
        media_type=None,
        current_info=None,
        *,
        mark_first_byte_observed: Optional[Callable[[dict[str, Any]], None]] = None,
        emit_request_observability: Optional[Callable[[dict[str, Any]], None]] = None,
        update_stats: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        trace_type: Optional[type] = None,
        debug: bool = False,
    ):
        super().__init__(content=None, status_code=status_code, headers=headers, media_type=media_type)
        self.body_iterator = content
        self._closed = False
        self.current_info = current_info or {}
        self._sse_buffer = ""
        self._mark_first_byte_observed = mark_first_byte_observed or (lambda current_info: None)
        self._emit_request_observability = emit_request_observability or (lambda current_info: None)
        self._update_stats = update_stats
        self._trace_type = trace_type
        self._debug = debug

        if "content-length" in self.headers:
            del self.headers["content-length"]
        self.headers["transfer-encoding"] = "chunked"

    def _is_trace(self, value: Any) -> bool:
        return self._trace_type is not None and isinstance(value, self._trace_type)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        _ = scope, receive
        trace = self.current_info.get("trace") if isinstance(self.current_info, dict) else None
        self.current_info["status_code"] = self.status_code
        if self._is_trace(trace):
            trace.mark("downstream_response_start")
            self.current_info["timing_spans"] = trace.snapshot()
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        try:
            async for chunk in self._logging_iterator():
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
        except Exception as e:
            logger.error("Error in streaming response: %s: %s", type(e).__name__, e)
            if self._debug:
                import traceback

                traceback.print_exc()
            try:
                error_data = json.dumps({"error": f"Streaming error: {str(e)}"})
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"data: {error_data}\n\n".encode("utf-8"),
                        "more_body": True,
                    }
                )
            except Exception as send_error:
                logger.error("Error sending error message: %s", send_error)
        finally:
            if hasattr(self.body_iterator, "aclose") and not self._closed:
                await call_cleanup_safely(
                    self.body_iterator.aclose,
                    label="Downstream streaming body iterator",
                )
                self._closed = True

            final_send_cancelled: asyncio.CancelledError | None = None
            try:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except asyncio.CancelledError as exc:
                final_send_cancelled = exc
            except Exception as exc:
                logger.warning(
                    "Error sending final streaming response body",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

            self.current_info["process_time"] = time() - self.current_info["start_time"]
            if self._is_trace(trace):
                trace.mark("stream_end")
                trace.mark("usage_recorded")
                self.current_info["timing_spans"] = trace.snapshot()
                logger.info(
                    "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                    self.current_info.get("trace_id"),
                    self.current_info.get("request_id"),
                    self.current_info.get("endpoint"),
                    self.current_info.get("timing_spans"),
                )
            self._emit_request_observability(self.current_info)
            if self._update_stats is not None:
                await self._update_stats(self.current_info)
            if final_send_cancelled is not None:
                raise final_send_cancelled

    async def _logging_iterator(self):
        async for chunk in self.body_iterator:
            self._mark_first_byte_observed(self.current_info)
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            if str(self.current_info.get("endpoint") or "").endswith("/v1/audio/speech"):
                yield chunk
                continue

            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                yield chunk
                continue

            if self._debug:
                try:
                    logger.info(text.encode("utf-8").decode("unicode_escape"))
                except Exception:
                    logger.info(text)

            self._sse_buffer += text
            while "\n" in self._sse_buffer:
                line, self._sse_buffer = self._sse_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line or is_sse_comment_frame(line) or line.startswith("event:"):
                    continue

                data = None
                if line.startswith("data:"):
                    data = line.removeprefix("data:").lstrip()
                elif line.startswith("{") or line.startswith("["):
                    data = line

                if not data or data.startswith("[DONE]") or data.startswith("OK") or "\"usage\"" not in data:
                    continue

                try:
                    resp = await asyncio.to_thread(json.loads, data)
                except Exception:
                    continue

                usage_obj = None
                if isinstance(resp, dict):
                    usage_obj = (
                        resp.get("usage")
                        or safe_get(resp, "response", "usage", default=None)
                        or safe_get(resp, "message", "usage", default=None)
                    )
                if not isinstance(usage_obj, dict):
                    continue

                prompt_tokens = usage_obj.get("prompt_tokens")
                completion_tokens = usage_obj.get("completion_tokens")
                if prompt_tokens is None and "input_tokens" in usage_obj:
                    prompt_tokens = usage_obj.get("input_tokens")
                if completion_tokens is None and "output_tokens" in usage_obj:
                    completion_tokens = usage_obj.get("output_tokens")

                try:
                    prompt_tokens = int(prompt_tokens or 0)
                except Exception:
                    prompt_tokens = 0
                try:
                    completion_tokens = int(completion_tokens or 0)
                except Exception:
                    completion_tokens = 0

                total_tokens = usage_obj.get("total_tokens")
                try:
                    total_tokens = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
                except Exception:
                    total_tokens = prompt_tokens + completion_tokens

                self.current_info["prompt_tokens"] = prompt_tokens
                self.current_info["completion_tokens"] = completion_tokens
                self.current_info["total_tokens"] = total_tokens
            yield chunk

    async def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self.body_iterator, "aclose"):
                await call_cleanup_safely(
                    self.body_iterator.aclose,
                    label="Downstream streaming body iterator",
                )
