from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException


NETWORK_ERRORS = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.LocalProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectError,
)


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
        if isinstance(exc, httpx.ReadTimeout):
            timeout_extensions = getattr(getattr(exc, "request", None), "extensions", {}) or {}
            timeout_value = self.safe_get(timeout_extensions, "timeout", "read", default=-1)
            return 504, f"Request timed out after {timeout_value} seconds"
        if isinstance(exc, httpx.ConnectError):
            return 503, "Unable to connect to service"
        if isinstance(exc, httpx.ReadError):
            return 502, "Network read error"
        if isinstance(exc, httpx.RemoteProtocolError):
            return 502, "Remote protocol error"
        if isinstance(exc, httpx.LocalProtocolError):
            return 502, "Local protocol error"
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
        if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in error_message:
            return 403
        if "<head><title>413 Request Entity Too Large</title></head>" in error_message:
            return 429
        return status_code

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
