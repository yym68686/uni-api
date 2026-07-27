from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from fastapi import HTTPException

from uni_api.observability.exceptions import exception_diagnostics
from uni_api.upstream.responses_errors import ResponsesSemanticError
from uni_api.upstream.transport_errors import (
    HTTPX_TRANSPORT_ERRORS,
    TransportErrorClassification,
    classify_httpx_transport_error,
)


NETWORK_ERRORS = HTTPX_TRANSPORT_ERRORS


# Some OpenAI-compatible gateways incorrectly use HTTP 400 for a provider-side
# model/billing configuration failure.  Keep this allow-list deliberately
# narrow: a generic "model unavailable" message can still be a real caller
# error, while these codes/phrases identify the gateway's administrator-side
# pricing gate.
_MODEL_PRICING_UNCONFIGURED_CODES = frozenset(
    {
        "model_not_priced",
        "model_price_not_configured",
        "model_pricing_not_configured",
        "model_price_unconfigured",
        "model_pricing_missing",
    }
)
_MODEL_PRICING_UNCONFIGURED_MARKERS = (
    "has not been priced by the administrator",
    "has not been priced by administrator",
    "price has not been configured by the administrator",
    "pricing has not been configured by the administrator",
    "价格尚未由管理员配置",
)


def _timeout_seconds_text(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return f"{seconds:g}"


def _transport_error_message(
    exc: BaseException,
    classification: TransportErrorClassification,
    safe_get: Callable[..., Any],
) -> str:
    if classification.kind in {
        "local_protocol_error",
        "remote_protocol_error",
    }:
        reason = exception_diagnostics(exc)["protocol_error_reason"]
        prefix = (
            "Local"
            if classification.kind == "local_protocol_error"
            else "Remote"
        )
        return f"{prefix} protocol error: {reason}"

    timeout_key = classification.timeout_extension_key
    if timeout_key is None:
        return classification.message
    try:
        request = getattr(exc, "request", None)
    except RuntimeError:
        request = None
    timeout_extensions = getattr(request, "extensions", {}) or {}
    timeout_value = safe_get(
        timeout_extensions,
        "timeout",
        timeout_key,
        default=None,
    )
    timeout_text = _timeout_seconds_text(timeout_value)
    if timeout_text is None:
        return classification.message
    return f"{classification.message} after {timeout_text} seconds"


@dataclass(frozen=True, slots=True)
class ProviderErrorClassifier:
    safe_get: Callable[..., Any]

    def details_parts(self, details: Any) -> tuple[Optional[str], Optional[str], Optional[str], str]:
        raw = str(details or "")
        code = None
        error_type = None
        message = None

        parsed = self._json_or_python_dict(raw)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                error_type = err.get("type")
                message = err.get("message")
            detail = parsed.get("detail")
            if isinstance(detail, dict):
                code = detail.get("code") or code
                error_type = detail.get("type") or error_type
                message = detail.get("message") or message

        return (
            str(code).strip().lower() or None,
            str(error_type).strip().lower() or None,
            str(message).strip() or None,
            raw,
        )

    def normalize_exception(self, exc: Exception) -> tuple[int, str]:
        if isinstance(exc, ResponsesSemanticError):
            return exc.status_code, exc.detail_json
        # Local admission errors deliberately expose a bounded HTTP outcome.
        # Recognize the small protocol by attributes to avoid coupling this
        # policy module back to the client-pool implementation.
        local_status = getattr(exc, "status_code", None)
        local_reason = str(getattr(exc, "reason", "") or "").strip()
        if (
            bool(getattr(exc, "local_admission_rejection", False))
            and local_reason
            and isinstance(local_status, int)
            and 400 <= local_status <= 599
        ):
            return local_status, local_reason
        if (
            local_reason
            and isinstance(local_status, int)
            and 400 <= local_status <= 599
        ):
            return local_status, local_reason
        transport_failure = classify_httpx_transport_error(exc)
        if transport_failure is not None:
            return (
                transport_failure.status_code,
                _transport_error_message(
                    exc,
                    transport_failure,
                    self.safe_get,
                ),
            )
        if isinstance(exc, HTTPException):
            return exc.status_code, str(exc.detail)
        return 500, str(exc) or f"Unknown error: {exc.__class__.__name__}"

    def remap_status_code(self, status_code: int, error_message: str) -> int:
        if "string_above_max_length" in error_message:
            return 413
        if "must be less than max_seq_len" in error_message:
            return 413
        if "Please reduce the length of the messages or completion" in error_message:
            return 413
        if "Request contains text fields that are too large." in error_message:
            return 413
        if "Please reduce the length of either one, or use the" in error_message:
            return 413
        if "exceeds the maximum number of tokens allowed" in error_message:
            return 413
        if "'reason': 'API_KEY_INVALID'" in error_message or "API key not valid" in error_message or "API key expired" in error_message:
            return 401
        if "User location is not supported for the API use." in error_message:
            return 403
        if "<center><h1>400 Bad Request</h1></center>" in error_message:
            return 502
        if "Provider API error: bad response status code 400" in error_message:
            return 502
        if self.is_model_pricing_unconfigured_error(status_code, error_message):
            # The caller's request is valid; the upstream gateway rejected it
            # because its own administrator-side model pricing is incomplete.
            # Treat that provider response as a bad gateway response so the
            # routing layer can fail over to another channel.
            return 502
        if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in error_message:
            return 403
        if "<head><title>413 Request Entity Too Large</title></head>" in error_message:
            return 429
        return status_code

    def is_model_pricing_unconfigured_error(
        self,
        status_code: int,
        details: Any,
    ) -> bool:
        """Identify an upstream 400 caused by missing model pricing config.

        This is intentionally not a general-purpose 400 retry heuristic.  It
        only recognizes the stable error codes/phrases emitted by the affected
        compatible gateways, leaving malformed requests and provider parameter
        validation errors non-retryable.
        """

        if status_code != 400:
            return False

        code, _error_type, message, raw = self.details_parts(details)
        if code in _MODEL_PRICING_UNCONFIGURED_CODES:
            return True

        haystack = " ".join(
            part for part in (message, raw) if part
        ).casefold()
        if any(marker.casefold() in haystack for marker in _MODEL_PRICING_UNCONFIGURED_MARKERS):
            return True

        # Chinese gateways occasionally insert spaces or line breaks between
        # the same words.  Compact only whitespace for this narrow fallback;
        # do not broaden the match to every "model unavailable" response.
        compact = re.sub(r"\s+", "", haystack)
        return "价格尚未由管理员配置" in compact

    def is_retryable_rate_limit_error(self, status_code: int, details: Any) -> bool:
        if status_code != 429:
            return False

        code, error_type, message, raw = self.details_parts(details)
        haystack = " ".join(part for part in (code, error_type, message, raw) if part).lower()
        return any(
            token in haystack
            for token in (
                "rate_limit_exceeded",
                "rate limit reached",
                "too many requests",
                "tokens per min",
                "requests per min",
                "tokens per day",
                "requests per day",
                "please try again in",
            )
        )

    def retry_after_seconds(self, details: Any) -> int:
        _, _, message, raw = self.details_parts(details)
        haystack = " ".join(part for part in (message, raw) if part)
        match = re.search(
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?)\b",
            haystack,
            re.IGNORECASE,
        )
        if not match:
            return 0

        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("ms"):
            seconds = value / 1000.0
        elif unit.startswith("m") and not unit.startswith("ms"):
            seconds = value * 60.0
        else:
            seconds = value
        return max(1, int(math.ceil(seconds)))

    def is_quota_exhausted_error(self, status_code: int, details: str) -> bool:
        if status_code == 401:
            return False
        text = (details or "").lower()
        return any(
            token in text
            for token in (
                "insufficient_quota",
                "billing_hard_limit_reached",
                "quota exceeded",
                "exceeded your current quota",
                "usage limit",
                "out of credits",
                "payment required",
            )
        )

    def is_codex_permanent_auth_error(self, status_code: int, details: str) -> bool:
        if status_code not in (401, 403, 402):
            return False

        raw = str(details or "")
        code = None
        message = None
        parsed = self._json_or_python_dict(raw)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                message = err.get("message")
            detail = parsed.get("detail")
            if code is None and isinstance(detail, dict):
                code = detail.get("code")
                message = detail.get("message") or message

        permanent_codes = {
            "account_deactivated",
            "account_disabled",
            "account_suspended",
            "deactivated_workspace",
            "user_deactivated",
            "user_suspended",
            "organization_deactivated",
            "organization_suspended",
        }
        if code and str(code).strip() in permanent_codes:
            return True

        haystack = (message or raw).lower()
        return any(
            token in haystack
            for token in (
                "account_deactivated",
                "account_disabled",
                "account_suspended",
                "deactivated_workspace",
                "organization_deactivated",
                "user_deactivated",
                "has been deactivated",
                "has been suspended",
            )
        )

    @staticmethod
    def _json_or_python_dict(raw: str) -> Any:
        if not (raw.startswith("{") or raw.startswith("[")):
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    classifier: ProviderErrorClassifier
    get_engine: Callable[..., tuple[Any, Any]]

    def should_retry(
        self,
        auto_retry: Any,
        status_code: int,
        provider: dict,
        *,
        error_message: Any = None,
        endpoint: Optional[str] = None,
        original_model: Optional[str] = None,
    ) -> bool:
        if not auto_retry:
            return False
        if self.classifier.is_model_pricing_unconfigured_error(
            status_code,
            error_message,
        ):
            return True
        if self.is_codex_chatgpt_model_unsupported_error(status_code, error_message, provider, endpoint, original_model):
            return True
        if self.is_missing_persisted_responses_item_error(status_code, error_message):
            return False
        return status_code not in (400, 413) or urlparse(provider.get("base_url", "")).netloc == "models.inference.ai.azure.com"

    def is_codex_chatgpt_model_unsupported_error(
        self,
        status_code: int,
        details: Any,
        provider: dict,
        endpoint: Optional[str],
        original_model: Optional[str],
    ) -> bool:
        if status_code != 400:
            return False
        if endpoint not in ("/v1/responses", "/v1/responses/compact"):
            return False

        try:
            engine, _ = self.get_engine(provider, endpoint, original_model or "")
        except Exception:
            engine = None
        if engine != "codex":
            return False

        _, _, message, raw = self.classifier.details_parts(details)
        haystack = " ".join(part for part in (message, raw) if part).lower()
        return "model is not supported when using codex with a chatgpt account" in haystack

    def is_missing_persisted_responses_item_error(self, status_code: int, details: Any) -> bool:
        if status_code != 404:
            return False

        _, error_type, message, raw = self.classifier.details_parts(details)
        haystack = " ".join(part for part in (error_type, message, raw) if part).lower()
        return (
            "invalid_request_error" in haystack
            and "item with id" in haystack
            and "not found" in haystack
            and "items are not persisted when" in haystack
            and "store" in haystack
        )


@dataclass(frozen=True, slots=True)
class CooldownPolicy:
    classifier: ProviderErrorClassifier
    get_engine: Callable[..., tuple[Any, Any]]

    def rate_limit_cooling_time(self, provider: dict, status_code: int, details: Any) -> int:
        if not self.classifier.is_retryable_rate_limit_error(status_code, details):
            return 0

        configured = self.classifier.safe_get(
            provider,
            "preferences",
            "api_key_rate_limit_cooldown_period",
            default=30 * 60,
        )
        try:
            configured_seconds = int(configured)
        except Exception:
            configured_seconds = 30 * 60

        retry_after_seconds = self.classifier.retry_after_seconds(details)
        if configured_seconds > 0:
            return max(configured_seconds, retry_after_seconds)
        if retry_after_seconds > 0:
            return retry_after_seconds
        return 30 * 60

    def should_use_quota_cooldown(
        self,
        provider: dict,
        status_code: int,
        error_message: str,
        *,
        endpoint: Optional[str],
        original_model: str,
        retry_policy: RetryPolicy,
    ) -> bool:
        if retry_policy.is_codex_chatgpt_model_unsupported_error(
            status_code,
            error_message,
            provider,
            endpoint,
            original_model,
        ):
            return True
        if self.classifier.is_quota_exhausted_error(status_code, error_message):
            return True

        try:
            engine, _ = self.get_engine(provider, endpoint, original_model)
        except Exception:
            engine = None
        if engine != "codex" or status_code not in (401, 403, 402):
            return False
        if "Codex token refresh" in error_message or "refresh_token_reused" in error_message:
            return True
        return self.classifier.is_codex_permanent_auth_error(status_code, error_message)
