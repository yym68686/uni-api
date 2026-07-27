from __future__ import annotations

from typing import Any

from core.utils import safe_get
from uni_api.serialization import json
from uni_api.streaming.error_text import bounded_stream_error_text


RESPONSES_FAILURE_STATUS_BY_CODE = {
    "account_deactivated": 403,
    "account_disabled": 403,
    "account_suspended": 403,
    "authentication_error": 401,
    "billing_hard_limit_reached": 429,
    "context_length_exceeded": 400,
    "deactivated_workspace": 403,
    "incorrect_api_key_provided": 401,
    "insufficient_quota": 429,
    "invalid_api_key": 401,
    "invalid_request_error": 400,
    "invalid_type": 400,
    "model_not_found": 404,
    # Compatible gateways sometimes expose their administrator-side pricing
    # gate as a Responses semantic 400.  The upstream policy layer remaps the
    # resulting error to 502 so it can fail over to another provider.
    "model_not_priced": 400,
    "model_price_not_configured": 400,
    "model_pricing_not_configured": 400,
    "model_price_unconfigured": 400,
    "model_pricing_missing": 400,
    "not_found_error": 404,
    "permission_denied": 403,
    "rate_limit_exceeded": 429,
    "unsupported_parameter": 400,
    "user_deactivated": 403,
    "user_suspended": 403,
}

RESPONSES_FAILURE_STATUS_BY_TYPE = {
    "authentication_error": 401,
    "invalid_request_error": 400,
    "not_found_error": 404,
    "permission_error": 403,
    "rate_limit_error": 429,
    "tokens": 429,
}

_FAILURE_EVENT_TYPES = frozenset({"error", "response.failed"})
_MAX_SEQUENCE_NUMBER = (1 << 63) - 1


def _bounded_optional_text(value: Any, *, limit_bytes: int = 256) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = bounded_stream_error_text(value, limit_bytes=limit_bytes).strip()
    return text or None


def _bounded_identifier(value: Any) -> str | None:
    text = _bounded_optional_text(value)
    return text.lower() if text is not None else None


def _normalized_protocol_token(
    value: Any,
    *,
    limit_bytes: int = 128,
) -> tuple[str, bool]:
    if value is None:
        return "", True
    if not isinstance(value, str):
        return "", False
    text = _bounded_optional_text(value, limit_bytes=limit_bytes)
    return (text or "").lower(), True


def _canonical_response_failed_payload(
    payload: dict[str, Any],
    *,
    message: str,
    error_code: str | None,
    error_type: str | None,
    param: str | None,
) -> dict[str, Any] | None:
    """Build a small, SDK-compatible Responses failure terminal.

    Keep only bounded protocol fields instead of retaining the parsed upstream
    object graph behind an exception.  This is used when the provider emitted
    ``response.failed``.  A separately guarded builder below handles only
    validated provider-declared ``type:error`` SSE events; transport and local
    failures never enter either builder.
    """

    response_obj = payload.get("response")
    if not isinstance(response_obj, dict):
        return None

    if "status" in response_obj:
        response_status, valid_response_status = _normalized_protocol_token(
            response_obj.get("status"),
        )
        if (
            not valid_response_status
            or response_status not in {"", "failed"}
        ):
            return None

    normalized_error: dict[str, Any] = {"message": message}
    normalized_error_type = _bounded_identifier(error_type)
    normalized_error_code = _bounded_identifier(error_code)
    if normalized_error_type:
        normalized_error["type"] = normalized_error_type
    if normalized_error_code:
        normalized_error["code"] = normalized_error_code
    if param:
        normalized_error["param"] = param

    normalized_response: dict[str, Any] = {
        "status": "failed",
        "error": normalized_error,
    }
    for key in ("id", "object", "model"):
        value = _bounded_optional_text(response_obj.get(key))
        if value is not None:
            normalized_response[key] = value

    normalized_payload: dict[str, Any] = {
        "type": "response.failed",
        "response": normalized_response,
    }
    sequence_number = payload.get("sequence_number")
    if (
        isinstance(sequence_number, int)
        and not isinstance(sequence_number, bool)
        and 0 <= sequence_number <= _MAX_SEQUENCE_NUMBER
    ):
        normalized_payload["sequence_number"] = sequence_number
    return normalized_payload


def _canonical_provider_error_response_failed_payload(
    payload: dict[str, Any],
    *,
    message: str,
    error_code: str | None,
    error_type: str | None,
    param: str | None,
) -> dict[str, Any] | None:
    """Normalize one explicitly validated provider ``type:error`` terminal."""

    if payload.get("type") != "error":
        return None

    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return None
    raw_message = error_obj.get("message")
    if not isinstance(raw_message, str) or not _bounded_optional_text(
        raw_message,
        limit_bytes=4096,
    ):
        return None

    normalized_error: dict[str, Any] = {"message": message}
    normalized_error_type = _bounded_identifier(error_type)
    normalized_error_code = _bounded_identifier(error_code)
    if normalized_error_type:
        normalized_error["type"] = normalized_error_type
    if normalized_error_code:
        normalized_error["code"] = normalized_error_code
    if param:
        normalized_error["param"] = param

    normalized_payload: dict[str, Any] = {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": normalized_error,
        },
    }
    sequence_number = payload.get("sequence_number")
    if (
        isinstance(sequence_number, int)
        and not isinstance(sequence_number, bool)
        and 0 <= sequence_number <= _MAX_SEQUENCE_NUMBER
    ):
        normalized_payload["sequence_number"] = sequence_number
    return normalized_payload


def _explicit_status_code(*values: Any) -> int | None:
    for value in values:
        try:
            status_code = int(value)
        except (TypeError, ValueError):
            continue
        if 400 <= status_code <= 599:
            return status_code
    return None


def responses_error_status_code(error_obj: Any, *, payload: Any = None) -> int:
    if isinstance(error_obj, dict):
        explicit_status = _explicit_status_code(
            error_obj.get("status_code"),
            error_obj.get("status"),
            safe_get(payload, "status_code", default=None),
            safe_get(payload, "status", default=None),
            safe_get(payload, "response", "status_code", default=None),
        )
        if explicit_status is not None:
            return explicit_status

        error_code = (
            _bounded_optional_text(error_obj.get("code")) or ""
        ).lower()
        if error_code in RESPONSES_FAILURE_STATUS_BY_CODE:
            return RESPONSES_FAILURE_STATUS_BY_CODE[error_code]

        error_type = (
            _bounded_optional_text(error_obj.get("type")) or ""
        ).lower()
        if error_type in RESPONSES_FAILURE_STATUS_BY_TYPE:
            return RESPONSES_FAILURE_STATUS_BY_TYPE[error_type]

        message = bounded_stream_error_text(
            error_obj.get("message"),
            limit_bytes=4096,
        ).lower()
    else:
        explicit_status = _explicit_status_code(
            safe_get(payload, "status_code", default=None),
            safe_get(payload, "status", default=None),
            safe_get(payload, "response", "status_code", default=None),
        )
        if explicit_status is not None:
            return explicit_status
        message = bounded_stream_error_text(
            error_obj,
            limit_bytes=4096,
        ).lower()

    if "rate limit" in message or "too many requests" in message:
        return 429
    if (
        "context window" in message
        or "context length" in message
        or "maximum context" in message
        or "too many tokens" in message
    ):
        return 400
    if "request entity too large" in message or "payload too large" in message:
        return 413
    if "invalid" in message or "unsupported" in message:
        return 400
    if "not found" in message:
        return 404
    if "permission" in message or "forbidden" in message:
        return 403
    if "auth" in message or "api key" in message or "unauthorized" in message:
        return 401
    return 500


class ResponsesSemanticError(Exception):
    """A valid Responses failure terminal with preserved HTTP semantics."""

    upstream_semantic_error = True

    def __init__(
        self,
        *,
        status_code: int,
        event_type: str,
        message: str,
        error_code: str | None,
        error_type: str | None,
        param: str | None,
        wire_status_code: int | None = None,
        passthrough_error_body: dict[str, Any] | None = None,
        responses_sse_payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = int(status_code)
        self.event_type = event_type
        self.message = bounded_stream_error_text(message)
        self.error_code = error_code
        self.error_type = error_type
        self.param = param
        self.wire_status_code = (
            int(wire_status_code)
            if isinstance(wire_status_code, int)
            else None
        )
        self.passthrough_error_body = passthrough_error_body

        normalized_error: dict[str, Any] = {
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.error_type:
            normalized_error["type"] = self.error_type
        if self.error_code:
            normalized_error["code"] = self.error_code
        if self.param:
            normalized_error["param"] = self.param

        self.error_body = {"error": normalized_error}
        self.sse_payload = {"type": "error", **self.error_body}
        self.responses_sse_event_type = "error"
        self.responses_sse_payload = self.sse_payload
        if (
            isinstance(responses_sse_payload, dict)
            and responses_sse_payload.get("type") == "response.failed"
            and isinstance(responses_sse_payload.get("response"), dict)
        ):
            self.responses_sse_event_type = "response.failed"
            self.responses_sse_payload = responses_sse_payload
        self.detail_json = json.dumps(
            self.error_body,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        super().__init__(self.detail_json)


def responses_failure_error(
    payload: Any,
    *,
    event_type: str | None = None,
    wire_status_code: int | None = None,
    preserve_error_body: bool = False,
    validated_provider_sse: bool = False,
) -> ResponsesSemanticError | None:
    if not isinstance(payload, dict):
        return None

    raw_event_type = (
        event_type
        if event_type is not None
        else safe_get(payload, "type", default=None)
    )
    normalized_event_type, valid_event_type = _normalized_protocol_token(
        raw_event_type,
    )
    if not valid_event_type:
        normalized_event_type = ""
    if (
        normalized_event_type not in _FAILURE_EVENT_TYPES
        and "response" not in payload
        and "status" not in payload
        and "error" not in payload
    ):
        # Ordinary delta/progress events cannot satisfy any failure condition
        # below. Avoid nested status/error inspection on the dominant path.
        return None
    response_status, valid_response_status = _normalized_protocol_token(
        safe_get(payload, "response", "status", default=None),
    )
    payload_status, valid_payload_status = _normalized_protocol_token(
        safe_get(payload, "status", default=None),
    )
    if not valid_payload_status:
        payload_status = ""

    error_obj: Any = None
    if normalized_event_type == "error":
        if "error" in payload:
            error_obj = safe_get(payload, "error", default=None)
        elif any(
            key in payload
            for key in ("message", "code", "error_type", "status", "status_code")
        ):
            flattened_type = payload.get("error_type")
            if flattened_type is None and payload.get("type") not in _FAILURE_EVENT_TYPES:
                flattened_type = payload.get("type")
            error_obj = {
                "message": payload.get("message"),
                "code": payload.get("code"),
                "type": flattened_type,
                "param": payload.get("param"),
                "status": payload.get("status"),
                "status_code": payload.get("status_code"),
            }
        else:
            return None
        if error_obj is None:
            return None
    elif normalized_event_type == "response.failed":
        response_obj = payload.get("response")
        if isinstance(response_obj, dict):
            error_obj = response_obj.get("error")
        elif "error" not in payload:
            return None
        if error_obj is None and "error" in payload:
            error_obj = payload.get("error")
    elif response_status == "failed":
        error_obj = safe_get(payload, "response", "error", default=None)
    elif payload_status == "failed":
        error_obj = safe_get(payload, "error", default=None)
    elif isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error_obj = payload.get("error")

    is_failure = (
        normalized_event_type in _FAILURE_EVENT_TYPES
        or response_status == "failed"
        or payload_status == "failed"
    )
    if not is_failure:
        return None
    if (
        normalized_event_type == "response.failed"
        and (
            not valid_response_status
            or (response_status and response_status != "failed")
        )
    ):
        # A contradictory/ill-typed status is a protocol problem, not a
        # provider-declared semantic failure.
        return None

    if isinstance(error_obj, dict):
        message = _bounded_optional_text(error_obj.get("message"), limit_bytes=4096)
        error_code = _bounded_optional_text(error_obj.get("code"))
        error_type = _bounded_optional_text(error_obj.get("type"))
        param = _bounded_optional_text(error_obj.get("param"))
    else:
        message = _bounded_optional_text(error_obj, limit_bytes=4096)
        error_code = None
        error_type = None
        param = None

    if message is None:
        message = f"Responses upstream returned {normalized_event_type or 'a failure terminal'}"

    passthrough_error_body = None
    if preserve_error_body:
        bounded_error: dict[str, Any] = {"message": message}
        if error_type:
            bounded_error["type"] = error_type
        if error_code:
            bounded_error["code"] = error_code
        if param:
            bounded_error["param"] = param
        passthrough_error_body = {"error": bounded_error}

    responses_sse_payload = None
    if normalized_event_type == "response.failed":
        responses_sse_payload = _canonical_response_failed_payload(
            payload,
            message=message,
            error_code=error_code,
            error_type=error_type,
            param=param,
        )
    elif normalized_event_type == "error" and validated_provider_sse:
        responses_sse_payload = (
            _canonical_provider_error_response_failed_payload(
                payload,
                message=message,
                error_code=error_code,
                error_type=error_type,
                param=param,
            )
        )

    return ResponsesSemanticError(
        status_code=responses_error_status_code(error_obj, payload=payload),
        event_type=normalized_event_type or "error",
        message=message,
        error_code=error_code,
        error_type=error_type,
        param=param,
        wire_status_code=wire_status_code,
        passthrough_error_body=passthrough_error_body,
        responses_sse_payload=responses_sse_payload,
    )
