import asyncio
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.utils import get_engine, provider_api_circular_list, safe_get
from routing import RoutingPlan, select_provider_api_key_raw

UPSTREAM_NETWORK_ERRORS = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.LocalProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectError,
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _extract_error_details_parts(details: Any) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    raw = str(details or "")
    code = None
    error_type = None
    message = None

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

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

    if code is None and (raw.startswith("{") or raw.startswith("[")):
        try:
            import ast

            parsed_py = ast.literal_eval(raw)
        except Exception:
            parsed_py = None
        if isinstance(parsed_py, dict):
            err = parsed_py.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                error_type = err.get("type")
                message = err.get("message")
            detail = parsed_py.get("detail")
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


def _is_retryable_rate_limit_error(status_code: int, details: Any) -> bool:
    if status_code != 429:
        return False

    code, error_type, message, raw = _extract_error_details_parts(details)
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


def _extract_retry_after_seconds(details: Any) -> int:
    _, _, message, raw = _extract_error_details_parts(details)
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


def _get_rate_limit_cooling_time(provider: dict, status_code: int, details: Any) -> int:
    if not _is_retryable_rate_limit_error(status_code, details):
        return 0

    configured = safe_get(
        provider,
        "preferences",
        "api_key_rate_limit_cooldown_period",
        default=30 * 60,
    )
    try:
        configured_seconds = int(configured)
    except Exception:
        configured_seconds = 30 * 60

    retry_after_seconds = _extract_retry_after_seconds(details)
    if configured_seconds > 0:
        return max(configured_seconds, retry_after_seconds)
    if retry_after_seconds > 0:
        return retry_after_seconds
    return 30 * 60


def _is_quota_exhausted_error(status_code: int, details: str) -> bool:
    if status_code == 401:
        return False
    text = (details or "").lower()
    return any(
        k in text
        for k in (
            "insufficient_quota",
            "billing_hard_limit_reached",
            "quota exceeded",
            "exceeded your current quota",
            "usage limit",
            "out of credits",
            "payment required",
        )
    )


def _is_codex_chatgpt_model_unsupported_error(
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
        engine, _ = get_engine(provider, endpoint, original_model or "")
    except Exception:
        engine = None
    if engine != "codex":
        return False

    _, _, message, raw = _extract_error_details_parts(details)
    haystack = " ".join(part for part in (message, raw) if part).lower()
    return "model is not supported when using codex with a chatgpt account" in haystack


def _is_codex_permanent_auth_error(status_code: int, details: str) -> bool:
    if status_code not in (401, 403, 402):
        return False

    raw = str(details or "")
    code = None
    message = None

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            message = err.get("message")
        detail = parsed.get("detail")
        if code is None and isinstance(detail, dict):
            code = detail.get("code")
            message = detail.get("message") or message

    if code is None and (raw.startswith("{") or raw.startswith("[")):
        try:
            import ast

            parsed_py = ast.literal_eval(raw)
        except Exception:
            parsed_py = None
        if isinstance(parsed_py, dict):
            err = parsed_py.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                message = err.get("message")
            detail = parsed_py.get("detail")
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
        k in haystack
        for k in (
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


def normalize_provider_exception(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, httpx.ReadTimeout):
        timeout_extensions = getattr(getattr(exc, "request", None), "extensions", {}) or {}
        timeout_value = safe_get(timeout_extensions, "timeout", "read", default=-1)
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


def remap_status_code_from_error(status_code: int, error_message: str) -> int:
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


def should_retry_provider(
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
    if _is_codex_chatgpt_model_unsupported_error(status_code, error_message, provider, endpoint, original_model):
        return True
    return status_code not in (400, 413) or urlparse(provider.get("base_url", "")).netloc == "models.inference.ai.azure.com"


async def maybe_cool_provider_api_key(
    provider: dict,
    provider_name: str,
    provider_api_key_raw: Optional[str],
    status_code: int,
    error_message: str,
    *,
    original_model: str,
    endpoint: Optional[str] = None,
    exclude_error_substrings: Optional[list[str]] = None,
) -> bool:
    if not provider_api_key_raw or provider_name.startswith("sk-"):
        return False

    api_key_count = provider_api_circular_list[provider_name].get_items_count()
    if api_key_count <= 1:
        return False

    quota_cooling_time = safe_get(provider, "preferences", "api_key_quota_cooldown_period", default=0)
    cooling_time = safe_get(provider, "preferences", "api_key_cooldown_period", default=0)
    rate_limit_cooling_time = _get_rate_limit_cooling_time(provider, status_code, error_message)
    is_codex_chatgpt_model_unsupported_failure = _is_codex_chatgpt_model_unsupported_error(
        status_code,
        error_message,
        provider,
        endpoint,
        original_model,
    )

    is_codex_refresh_failure = False
    is_codex_permanent_auth_failure = False
    try:
        failed_engine_for_cooldown, _ = get_engine(provider, endpoint, original_model)
        if failed_engine_for_cooldown == "codex" and status_code in (401, 403, 402):
            if "Codex token refresh" in error_message or "refresh_token_reused" in error_message:
                is_codex_refresh_failure = True
            elif _is_codex_permanent_auth_error(status_code, error_message):
                is_codex_permanent_auth_failure = True
    except Exception:
        pass

    if (
        is_codex_refresh_failure
        or is_codex_permanent_auth_failure
        or is_codex_chatgpt_model_unsupported_failure
        or _is_quota_exhausted_error(status_code, error_message)
    ):
        effective_quota_cooldown = int(quota_cooling_time) if int(quota_cooling_time) > 0 else 6 * 60 * 60
        await provider_api_circular_list[provider_name].set_cooling(
            provider_api_key_raw,
            cooling_time=effective_quota_cooldown,
        )
        return True

    if rate_limit_cooling_time > 0:
        await provider_api_circular_list[provider_name].set_cooling(
            provider_api_key_raw,
            cooling_time=rate_limit_cooling_time,
        )
        return True

    if int(cooling_time) <= 0:
        return False

    if exclude_error_substrings and any(error in error_message for error in exclude_error_substrings):
        return False

    await provider_api_circular_list[provider_name].set_cooling(
        provider_api_key_raw,
        cooling_time=int(cooling_time),
    )
    return True


def rollback_failed_rate_limit_record(
    provider_name: str,
    provider_api_key_raw: Optional[str],
    original_model: str,
    error_message: str,
    rollback_errors: list[str],
) -> None:
    if not provider_api_key_raw or not any(error in error_message for error in rollback_errors):
        return
    circular_list = provider_api_circular_list[provider_name]
    if hasattr(circular_list, "rollback_rate_limit_record"):
        circular_list.rollback_rate_limit_record(provider_api_key_raw, original_model)
        return

    requests = circular_list.requests[provider_api_key_raw][original_model]
    if requests:
        requests.pop()


async def maybe_exclude_failed_channel(
    plan: RoutingPlan,
    provider_name: str,
    error_message: str,
    *,
    exclude_error_substrings: Optional[list[str]] = None,
    debug: bool = False,
) -> None:
    channel_manager = getattr(plan.app.state, "channel_manager", None)
    exclude_error_substrings = exclude_error_substrings or []
    if not channel_manager or channel_manager.cooldown_period <= 0 or plan.num_matching_providers <= 1:
        return
    if any(error in error_message for error in exclude_error_substrings):
        return

    await channel_manager.exclude_model(provider_name, plan.request_model_name)
    await plan.refresh_matching_providers(debug=debug)


async def maybe_clear_provider_auth_cache(
    attempt: "UpstreamAttemptContext",
    endpoint: Optional[str],
    status_code: int,
    clear_provider_auth_cache: Optional[Callable[[str], Any]],
) -> None:
    if not clear_provider_auth_cache or not attempt.provider_api_key_raw or status_code not in (401, 403):
        return
    try:
        failed_engine, _ = get_engine(attempt.provider, endpoint, attempt.original_model)
    except Exception:
        failed_engine = None
    if failed_engine == "codex":
        await _maybe_await(clear_provider_auth_cache(attempt.provider_api_key_raw))


def build_upstream_error_response(status_code: int, error_message: Any, fallback_prefix: Optional[str] = None) -> JSONResponse:
    parsed_error = None
    if isinstance(error_message, (dict, list)):
        parsed_error = error_message
    elif isinstance(error_message, str):
        stripped = error_message.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed_error = json.loads(stripped)
            except Exception:
                parsed_error = None

    if parsed_error is not None:
        return JSONResponse(status_code=status_code, content=parsed_error)

    message_text = str(error_message)
    if fallback_prefix:
        message_text = f"{fallback_prefix}: {message_text}"
    return JSONResponse(status_code=status_code, content={"error": message_text})


@dataclass
class UpstreamAttemptContext:
    plan: RoutingPlan
    provider: dict
    provider_name: str
    original_model: str
    provider_api_key_raw: Optional[str] = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpstreamAttemptResult:
    response: Any = None
    should_retry: bool = False
    finalize: bool = False


class UpstreamRunner:
    def __init__(
        self,
        plan: RoutingPlan,
        *,
        endpoint: Optional[str] = None,
        debug: bool = False,
        provider_api_key_selector=None,
        clear_provider_auth_cache: Optional[Callable[[str], Any]] = None,
    ):
        self.plan = plan
        self.endpoint = endpoint
        self.debug = debug
        self.provider_api_key_selector = provider_api_key_selector or select_provider_api_key_raw
        self.clear_provider_auth_cache = clear_provider_auth_cache

    def _runtime_api_list(self) -> list[str]:
        api_list = getattr(self.plan.app.state, "api_list", None)
        if api_list:
            return api_list
        config = getattr(self.plan.app.state, "config", {}) or {}
        return [item.get("api") for item in config.get("api_keys", []) if item.get("api")]

    async def next_attempt(self) -> Optional[UpstreamAttemptContext]:
        attempt = await self.plan.next_provider()
        if attempt is None:
            return None
        return UpstreamAttemptContext(
            plan=self.plan,
            provider=attempt.provider,
            provider_name=attempt.provider_name,
            original_model=attempt.original_model,
        )

    async def select_provider_api_key(self, attempt: UpstreamAttemptContext) -> Optional[str]:
        attempt.provider_api_key_raw = await self.provider_api_key_selector(
            attempt.provider,
            attempt.original_model,
            self._runtime_api_list(),
        )
        return attempt.provider_api_key_raw

    async def run(
        self,
        execute_attempt,
        *,
        prepare_attempt=None,
        before_next_attempt=None,
        after_failure=None,
        build_error_response=None,
        build_final_response=None,
        exclude_error_substrings: Optional[list[str]] = None,
        rollback_rate_limit_errors: Optional[list[str]] = None,
        allow_channel_exclusion: bool = False,
        should_cool_down=None,
        on_retry=None,
        on_cooldown=None,
    ) -> Any:
        while True:
            if before_next_attempt is not None:
                maybe_response = await _maybe_await(before_next_attempt())
                if maybe_response is not None:
                    return maybe_response

            attempt = await self.next_attempt()
            if attempt is None:
                break

            result = await self._run_attempt(
                attempt,
                execute_attempt,
                prepare_attempt=prepare_attempt,
                after_failure=after_failure,
                build_error_response=build_error_response,
                exclude_error_substrings=exclude_error_substrings,
                rollback_rate_limit_errors=rollback_rate_limit_errors,
                allow_channel_exclusion=allow_channel_exclusion,
                should_cool_down=should_cool_down,
                on_retry=on_retry,
                on_cooldown=on_cooldown,
            )
            if result.should_retry:
                continue
            if result.finalize:
                break
            if result.response is not None:
                return result.response

        if build_final_response is not None:
            return await _maybe_await(build_final_response(self.plan))

        return JSONResponse(
            status_code=self.plan.status_code,
            content={"error": f"All {self.plan.request_model_name} error: {self.plan.error_message}"},
        )

    async def _run_attempt(
        self,
        attempt: UpstreamAttemptContext,
        execute_attempt,
        *,
        prepare_attempt=None,
        after_failure=None,
        build_error_response=None,
        exclude_error_substrings: Optional[list[str]] = None,
        rollback_rate_limit_errors: Optional[list[str]] = None,
        allow_channel_exclusion: bool = False,
        should_cool_down=None,
        on_retry=None,
        on_cooldown=None,
    ) -> UpstreamAttemptResult:
        try:
            if prepare_attempt is not None:
                try:
                    await _maybe_await(prepare_attempt(attempt))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return await self._handle_failure(
                        attempt,
                        exc,
                        after_failure=after_failure,
                        exclude_error_substrings=exclude_error_substrings,
                        rollback_rate_limit_errors=rollback_rate_limit_errors,
                        should_cool_down=should_cool_down,
                        on_retry=on_retry,
                        on_cooldown=on_cooldown,
                        prepare_failure=True,
                    )
            response = await execute_attempt(attempt)
            if isinstance(response, UpstreamAttemptResult):
                return response
            return UpstreamAttemptResult(response=response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await self._handle_failure(
                attempt,
                exc,
                after_failure=after_failure,
                build_error_response=build_error_response,
                exclude_error_substrings=exclude_error_substrings,
                rollback_rate_limit_errors=rollback_rate_limit_errors,
                allow_channel_exclusion=allow_channel_exclusion,
                should_cool_down=should_cool_down,
                on_retry=on_retry,
                on_cooldown=on_cooldown,
                prepare_failure=False,
            )

    async def _handle_failure(
        self,
        attempt: UpstreamAttemptContext,
        exc: Exception,
        *,
        after_failure=None,
        build_error_response=None,
        exclude_error_substrings: Optional[list[str]] = None,
        rollback_rate_limit_errors: Optional[list[str]] = None,
        allow_channel_exclusion: bool = False,
        should_cool_down=None,
        on_retry=None,
        on_cooldown=None,
        prepare_failure: bool,
    ) -> UpstreamAttemptResult:
        status_code, error_message = normalize_provider_exception(exc)
        status_code = remap_status_code_from_error(status_code, error_message)
        self.plan.status_code = status_code
        self.plan.error_message = error_message

        if allow_channel_exclusion and not prepare_failure:
            await maybe_exclude_failed_channel(
                self.plan,
                attempt.provider_name,
                error_message,
                exclude_error_substrings=exclude_error_substrings,
                debug=self.debug,
            )

        should_cool_key = True
        force_cool_key = _is_codex_chatgpt_model_unsupported_error(
            status_code,
            error_message,
            attempt.provider,
            self.endpoint,
            attempt.original_model,
        )
        if should_cool_down is not None:
            should_cool_key = force_cool_key or bool(
                await _maybe_await(
                    should_cool_down(exc, status_code, error_message, attempt)
                )
            )
        if should_cool_key:
            cooled = await maybe_cool_provider_api_key(
                attempt.provider,
                attempt.provider_name,
                attempt.provider_api_key_raw,
                status_code,
                error_message,
                original_model=attempt.original_model,
                endpoint=self.endpoint,
                exclude_error_substrings=exclude_error_substrings,
            )
            if cooled and on_cooldown is not None:
                await _maybe_await(on_cooldown(attempt, status_code, error_message))

        if rollback_rate_limit_errors and not prepare_failure:
            rollback_failed_rate_limit_record(
                attempt.provider_name,
                attempt.provider_api_key_raw,
                attempt.original_model,
                error_message,
                rollback_rate_limit_errors,
            )

        await maybe_clear_provider_auth_cache(
            attempt,
            self.endpoint,
            status_code,
            self.clear_provider_auth_cache,
        )

        if after_failure is not None:
            await _maybe_await(after_failure(attempt, exc, status_code, error_message))

        if prepare_failure:
            if self.plan.auto_retry:
                if on_retry is not None:
                    await _maybe_await(on_retry(attempt, status_code, error_message))
                return UpstreamAttemptResult(should_retry=True)
            return UpstreamAttemptResult(finalize=True)

        if should_retry_provider(
            self.plan.auto_retry,
            status_code,
            attempt.provider,
            error_message=error_message,
            endpoint=self.endpoint,
            original_model=attempt.original_model,
        ):
            if on_retry is not None:
                await _maybe_await(on_retry(attempt, status_code, error_message))
            return UpstreamAttemptResult(should_retry=True)

        if build_error_response is not None:
            response = await _maybe_await(build_error_response(status_code, error_message))
        else:
            response = JSONResponse(
                status_code=status_code,
                content={"error": f"Error: Current provider response failed: {error_message}"},
            )
        return UpstreamAttemptResult(response=response)
