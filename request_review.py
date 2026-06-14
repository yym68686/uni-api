import asyncio
import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import httpx
from fastapi import Request

from core.log_config import logger


DEFAULT_QUEUE_SIZE = 2048
DEFAULT_BATCH_SIZE = 20
DEFAULT_FLUSH_INTERVAL_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_PAYLOAD_BYTES = 1 << 20


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


def _review_config(config: dict[str, Any] | None) -> dict[str, Any]:
    preferences = (config or {}).get("preferences")
    if not isinstance(preferences, dict):
        return {}
    review = preferences.get("request_review")
    if isinstance(review, dict):
        return review
    review = preferences.get("review00")
    if isinstance(review, dict):
        return review
    return {}


def is_request_review_enabled(config: dict[str, Any] | None) -> bool:
    review = _review_config(config)
    base_url = _normalize_base_url(str(review.get("base_url") or ""))
    api_key = str(review.get("api_key") or "").strip()
    enabled = review.get("enabled", True)
    return enabled is not False and bool(base_url and api_key)


def build_review_event(
    *,
    request: Request,
    current_info: dict[str, Any],
    raw_body: bytes,
    parsed_body: Any = None,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    payload_bytes = raw_body or b""
    truncated = False
    if len(payload_bytes) > max_payload_bytes:
        payload_bytes = payload_bytes[:max_payload_bytes]
        truncated = True

    if isinstance(parsed_body, dict):
        model = parsed_body.get("model") or current_info.get("model") or ""
        stream = bool(parsed_body.get("stream"))
    else:
        model = current_info.get("model") or ""
        stream = bool(current_info.get("stream"))

    return {
        "eventId": str(uuid.uuid4()),
        "requestId": str(current_info.get("request_id") or ""),
        "occurredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "orgId": str(current_info.get("org_id") or ""),
        "userId": str(current_info.get("user_id") or ""),
        "apiKeyId": str(current_info.get("api_key_id") or ""),
        "method": request.method,
        "path": request.url.path,
        "queryString": request.url.query,
        "contentType": request.headers.get("content-type", ""),
        "model": str(model or ""),
        "stream": stream,
        "sourceIp": current_info.get("client_ip"),
        "payload": {
            "encoding": "base64",
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "bytes": len(payload_bytes),
            "truncated": truncated,
            "data": base64.b64encode(payload_bytes).decode("ascii"),
        },
    }


def synthetic_review_event(path: str = "/_review00/connection-test") -> dict[str, Any]:
    payload_bytes = json.dumps(
        {"type": "connection_test", "source": "uni-api-web-api"},
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "eventId": str(uuid.uuid4()),
        "requestId": str(uuid.uuid4()),
        "occurredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "POST",
        "path": path,
        "contentType": "application/json",
        "payload": {
            "encoding": "base64",
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "bytes": len(payload_bytes),
            "truncated": False,
            "data": base64.b64encode(payload_bytes).decode("ascii"),
        },
    }


class RequestReviewDispatcher:
    def __init__(
        self,
        *,
        source_service: str = "uni-api-web-api",
        environment: str = "production",
        project: str = "uni-api-web",
        queue_size: int = DEFAULT_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self.source_service = source_service
        self.environment = environment
        self.project = project
        self.queue_size = max(1, queue_size)
        self.batch_size = max(1, batch_size)
        self.flush_interval_seconds = max(0.1, flush_interval_seconds)
        self.timeout_seconds = max(0.1, timeout_seconds)
        self.max_payload_bytes = max(1024, max_payload_bytes)
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        self._client: httpx.AsyncClient | None = None
        self._worker_task: asyncio.Task | None = None
        self.dropped_events = 0

    @classmethod
    def from_env(cls) -> "RequestReviewDispatcher":
        return cls(
            source_service=os.getenv("REQUEST_REVIEW_SOURCE_SERVICE", "uni-api-web-api"),
            environment=os.getenv("REQUEST_REVIEW_ENVIRONMENT", "production"),
            project=os.getenv("REQUEST_REVIEW_PROJECT", "uni-api-web"),
            queue_size=_env_int("REQUEST_REVIEW_QUEUE_SIZE", DEFAULT_QUEUE_SIZE),
            batch_size=_env_int("REQUEST_REVIEW_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            flush_interval_seconds=_env_float("REQUEST_REVIEW_FLUSH_INTERVAL_SECONDS", DEFAULT_FLUSH_INTERVAL_SECONDS),
            timeout_seconds=_env_float("REQUEST_REVIEW_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_payload_bytes=_env_int("REQUEST_REVIEW_MAX_PAYLOAD_BYTES", DEFAULT_MAX_PAYLOAD_BYTES),
        )

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        self._worker_task = asyncio.create_task(self._worker(), name="request-review-dispatcher")

    async def stop(self) -> None:
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def enqueue(self, config: dict[str, Any] | None, event: dict[str, Any]) -> bool:
        review = _review_config(config)
        base_url = _normalize_base_url(str(review.get("base_url") or ""))
        api_key = str(review.get("api_key") or "").strip()
        enabled = review.get("enabled", True)
        if enabled is False or not base_url or not api_key:
            return False

        item = {
            "base_url": base_url,
            "api_key": api_key,
            "event": event,
        }
        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.dropped_events += 1
            logger.warning(
                "request review queue full; dropping event request_id=%s dropped=%s",
                event.get("requestId"),
                self.dropped_events,
            )
            return False

    async def _worker(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.flush_interval_seconds

            while len(batch) < self.batch_size:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in batch:
                grouped.setdefault((item["base_url"], item["api_key"]), []).append(item["event"])

            for (base_url, api_key), events in grouped.items():
                await self._send_batch(base_url, api_key, events)

            for _ in batch:
                self.queue.task_done()

    async def _send_batch(self, base_url: str, api_key: str, events: list[dict[str, Any]]) -> None:
        client = self._client
        if client is None:
            return
        await self._post_batch(client, base_url, api_key, events)

    async def send_test(self, base_url: str, api_key: str) -> tuple[int, str]:
        event = synthetic_review_event()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await self._post_batch(client, base_url, api_key, [event], raise_for_error=True)
        if response is None:
            raise RuntimeError("request_review_test_failed")
        return response.status_code, response.text[:500]

    async def _post_batch(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        events: list[dict[str, Any]],
        *,
        raise_for_error: bool = False,
    ) -> httpx.Response | None:
        payload = {
            "source": {
                "service": self.source_service,
                "environment": self.environment,
                "project": self.project,
            },
            "events": events,
        }
        url = urljoin(base_url + "/", "v1/request-reviews/batch")
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code >= 300:
                if raise_for_error:
                    response.raise_for_status()
                logger.warning(
                    "request review delivery failed status=%s events=%s response=%s",
                    response.status_code,
                    len(events),
                    response.text[:500],
                )
            return response
        except Exception as exc:
            logger.warning(
                "request review delivery error type=%s events=%s: %s",
                type(exc).__name__,
                len(events),
                exc,
            )
            if raise_for_error:
                raise
            return None


def config_for_api_response(config: dict[str, Any]) -> dict[str, Any]:
    encoded = json.loads(json.dumps(config, default=str))
    review = _review_config(encoded)
    if review and review.get("api_key"):
        review["api_key_set"] = True
    return encoded
