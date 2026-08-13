import asyncio
import hashlib
import inspect
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.log_config import logger
from core.utils import get_engine, provider_api_circular_list, safe_get
from uni_api.admission import get_request_admission_lease
from uni_api.observability.upstream_transport import (
    UpstreamTransportDiagnostics,
    bind_upstream_transport_diagnostics,
    reset_upstream_transport_diagnostics,
)
from uni_api.routing.core import RoutingPlan, select_provider_api_key_raw
from uni_api.upstream.policies import (
    NETWORK_ERRORS,
    CooldownPolicy,
    ProviderErrorClassifier,
    RetryPolicy,
)
from uni_api.upstream.transport_errors import (
    TransportErrorClassification,
    classify_httpx_transport_error,
)

UPSTREAM_NETWORK_ERRORS = NETWORK_ERRORS

_PROVIDER_ERROR_CLASSIFIER = ProviderErrorClassifier(safe_get=safe_get)
_RETRY_POLICY = RetryPolicy(_PROVIDER_ERROR_CLASSIFIER, get_engine=get_engine)
_COOLDOWN_POLICY = CooldownPolicy(_PROVIDER_ERROR_CLASSIFIER, get_engine=get_engine)

_ROUTING_ATTEMPT_HEAD = 16
_ROUTING_ATTEMPT_TAIL = 16
_ROUTING_ATTEMPT_LIMIT = _ROUTING_ATTEMPT_HEAD + _ROUTING_ATTEMPT_TAIL
_ROUTING_ERROR_HASH_TEXT_LIMIT = 4096
_ROUTING_ERROR_HASH_SCOPE = "unicode_text_prefix_4096_chars_utf8_v1"


def _clear_exception_traceback_chain(exc: BaseException) -> None:
    """Release completed failure frames after routing has consumed the error.

    A finished ``asyncio.Task`` retains its exception, and the exception retains
    every traceback frame.  When one of those frames also references the Task,
    large request payloads survive until cyclic GC happens.  Routing has already
    classified, logged, and made its retry decision before this helper is called,
    so the traceback is no longer part of the request contract.
    """

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)

        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)

        tb = current.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def _routing_error_sha256(value: Any) -> str:
    text = str(value)[:_ROUTING_ERROR_HASH_TEXT_LIMIT]
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def _observable_provider_name(value: Any) -> str:
    text = str(value or "")
    if text.startswith("sk-"):
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"local-api-key:{fingerprint}"
    return text[:256]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _extract_error_details_parts(details: Any) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    return _PROVIDER_ERROR_CLASSIFIER.details_parts(details)


def _is_retryable_rate_limit_error(status_code: int, details: Any) -> bool:
    return _PROVIDER_ERROR_CLASSIFIER.is_retryable_rate_limit_error(status_code, details)


def _extract_retry_after_seconds(details: Any) -> int:
    return _PROVIDER_ERROR_CLASSIFIER.retry_after_seconds(details)


def _get_rate_limit_cooling_time(provider: dict, status_code: int, details: Any) -> int:
    return _COOLDOWN_POLICY.rate_limit_cooling_time(provider, status_code, details)


def _is_quota_exhausted_error(status_code: int, details: str) -> bool:
    return _PROVIDER_ERROR_CLASSIFIER.is_quota_exhausted_error(status_code, details)


def _is_codex_chatgpt_model_unsupported_error(
    status_code: int,
    details: Any,
    provider: dict,
    endpoint: Optional[str],
    original_model: Optional[str],
) -> bool:
    return _RETRY_POLICY.is_codex_chatgpt_model_unsupported_error(
        status_code,
        details,
        provider,
        endpoint,
        original_model,
    )


def _is_missing_persisted_responses_item_error(status_code: int, details: Any) -> bool:
    return _RETRY_POLICY.is_missing_persisted_responses_item_error(status_code, details)


def _is_codex_permanent_auth_error(status_code: int, details: str) -> bool:
    return _PROVIDER_ERROR_CLASSIFIER.is_codex_permanent_auth_error(status_code, details)


def normalize_provider_exception(exc: Exception) -> tuple[int, str]:
    return _PROVIDER_ERROR_CLASSIFIER.normalize_exception(exc)


def remap_status_code_from_error(status_code: int, error_message: str) -> int:
    return _PROVIDER_ERROR_CLASSIFIER.remap_status_code(status_code, error_message)


def should_retry_provider(
    auto_retry: Any,
    status_code: int,
    provider: dict,
    *,
    error_message: Any = None,
    endpoint: Optional[str] = None,
    original_model: Optional[str] = None,
) -> bool:
    return _RETRY_POLICY.should_retry(
        auto_retry,
        status_code,
        provider,
        error_message=error_message,
        endpoint=endpoint,
        original_model=original_model,
    )


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
    if _COOLDOWN_POLICY.should_use_quota_cooldown(
        provider,
        status_code,
        error_message,
        endpoint=endpoint,
        original_model=original_model,
        retry_policy=_RETRY_POLICY,
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
    actual_model: str,
    status_code: int,
    error_message: str,
    *,
    provider: Optional[dict] = None,
    exclude_error_substrings: Optional[list[str]] = None,
    debug: bool = False,
) -> str | None:
    channel_manager = getattr(plan.app.state, "channel_manager", None)
    exclude_error_substrings = exclude_error_substrings or []
    if not channel_manager:
        return None
    excluded_from_legacy_cooldown = any(
        error in error_message for error in exclude_error_substrings
    )

    if status_code in (403, 404):
        record_failure = getattr(channel_manager, "record_model_failure", None)
        if callable(record_failure):
            opened = bool(
                await _maybe_await(
                    record_failure(provider_name, actual_model, status_code)
                )
            )
            if opened:
                logger.warning(
                    "Provider-model route circuit opened provider=%s model=%s "
                    "status_code=%s",
                    _observable_provider_name(provider_name),
                    str(actual_model or "")[:256],
                    status_code,
                )
                try:
                    await plan.refresh_matching_providers(debug=debug)
                except HTTPException as refresh_exc:
                    if refresh_exc.status_code != 503:
                        raise
                    return "opened_no_alternative"
                return "opened"

    if excluded_from_legacy_cooldown:
        return None
    provider_preferences = (
        provider.get("preferences", {})
        if isinstance(provider, dict)
        and isinstance(provider.get("preferences", {}), dict)
        else {}
    )
    effective_cooldown_period = provider_preferences.get(
        "cooldown_period",
        getattr(channel_manager, "cooldown_period", 0),
    )
    try:
        effective_cooldown_enabled = float(effective_cooldown_period) > 0
    except (TypeError, ValueError):
        effective_cooldown_enabled = False
    if (
        not effective_cooldown_enabled
        or plan.num_matching_providers <= 1
    ):
        return None
    await channel_manager.exclude_model(provider_name, actual_model)
    await plan.refresh_matching_providers(debug=debug)
    return None


async def maybe_reset_provider_model_circuit(
    plan: RoutingPlan,
    provider_name: str,
    actual_model: str,
) -> None:
    app = getattr(plan, "app", None)
    state = getattr(app, "state", None)
    channel_manager = getattr(state, "channel_manager", None)
    record_success = getattr(channel_manager, "record_model_success", None)
    if callable(record_success):
        await _maybe_await(record_success(provider_name, actual_model))


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
    routing_attempt_id: str = field(default_factory=lambda: uuid4().hex)
    provider_api_key_raw: Optional[str] = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpstreamAttemptResult:
    response: Any = None
    should_retry: bool = False
    finalize: bool = False
    local_admission_reason: Optional[str] = None
    local_admission_retry_after: Optional[int] = None
    status_origin: Optional[str] = None


def finalize_routing_attempt_entry(
    entry: Any,
    *,
    outcome: str,
    success: Optional[bool] = None,
    wire_status_code: Optional[int] = None,
    semantic_status_code: Optional[int] = None,
    terminal_event_type: Optional[str] = None,
    error_code: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Any = None,
) -> None:
    """Finalize a routing fact without retaining provider error contents."""

    if not isinstance(entry, dict):
        return
    started_monotonic = entry.pop("_started_monotonic", None)
    if isinstance(started_monotonic, (int, float)):
        entry["duration_ms"] = max(
            0,
            int((time.monotonic() - float(started_monotonic)) * 1000),
        )
    entry["outcome"] = str(outcome)[:80]
    if success is not None:
        entry["success"] = bool(success)
    if wire_status_code is not None:
        entry["wire_status_code"] = int(wire_status_code)
    if semantic_status_code is not None:
        entry["semantic_status_code"] = int(semantic_status_code)
    if terminal_event_type:
        entry["terminal_event_type"] = str(terminal_event_type)[:128]
    if error_code:
        entry["error_code"] = str(error_code)[:128]
    if error_type:
        entry["error_type"] = str(error_type)[:128]
    if error_message is not None:
        entry["error_message_sha256"] = _routing_error_sha256(error_message)
        entry["error_message_hash_scope"] = _ROUTING_ERROR_HASH_SCOPE


def finalize_routing_attempt(
    attempt: Any,
    **kwargs: Any,
) -> None:
    state = getattr(attempt, "state", None)
    if not isinstance(state, dict):
        return
    finalize_routing_attempt_entry(
        state.get("_routing_attempt_entry"),
        **kwargs,
    )


def finalize_latest_routing_attempt(
    current_info: Any,
    *,
    response_memory_lease: Any = None,
    **kwargs: Any,
) -> None:
    finalize_response_memory_attempt(
        outcome=str(kwargs.get("outcome") or "finished"),
        lease=response_memory_lease,
    )
    if not isinstance(current_info, dict):
        return
    attempts = current_info.get("routing_attempts")
    if not isinstance(attempts, list) or not attempts:
        return
    finalize_routing_attempt_entry(attempts[-1], **kwargs)


def finalize_response_memory_attempt(
    *,
    outcome: str,
    attempt: Any = None,
    lease: Any = None,
) -> None:
    """Finalize the response-buffer attempt when a stream really terminates."""

    state = getattr(attempt, "state", None)
    if lease is None and isinstance(state, dict):
        lease = state.get("_response_memory_lease")
    if lease is None:
        lease = get_request_admission_lease()
    if lease is None:
        return
    try:
        lease.finish_response_attempt(outcome=str(outcome)[:80])
    except Exception:
        # Observability finalization is fail-open by contract.
        return


class UpstreamRunner:
    def __init__(
        self,
        plan: RoutingPlan,
        *,
        endpoint: Optional[str] = None,
        debug: bool = False,
        provider_api_key_selector=None,
        clear_provider_auth_cache: Optional[Callable[[str], Any]] = None,
        observability_context: Optional[dict[str, Any]] = None,
    ):
        self.plan = plan
        self.endpoint = endpoint
        self.debug = debug
        self.provider_api_key_selector = provider_api_key_selector or select_provider_api_key_raw
        self.clear_provider_auth_cache = clear_provider_auth_cache
        self.observability_context = (
            observability_context
            if isinstance(observability_context, dict)
            else None
        )
        self.runtime_api_list = list(getattr(plan, "api_list", ()) or ())

    def _start_routing_attempt(
        self,
        attempt: UpstreamAttemptContext,
    ) -> Optional[dict[str, Any]]:
        info = self.observability_context
        response_memory_lease = get_request_admission_lease()
        attempt.state["_response_memory_lease"] = response_memory_lease
        if info is None:
            transport_entry: dict[str, Any] = {}
            transport_diagnostics = UpstreamTransportDiagnostics(transport_entry)
            attempt.state["_transport_diagnostics"] = transport_diagnostics
            attempt.state["_transport_context_token"] = (
                bind_upstream_transport_diagnostics(transport_diagnostics)
            )
            attempt.state["transport_diagnostics"] = transport_entry[
                "transport_diagnostics"
            ]
            if response_memory_lease is not None:
                response_memory_lease.begin_response_attempt(
                    None,
                    routing_attempt_id=attempt.routing_attempt_id,
                    routing_attempt_index=None,
                    provider=_observable_provider_name(attempt.provider_name),
                    request_model=str(
                        getattr(self.plan, "request_model_name", "") or ""
                    ),
                    actual_model=attempt.original_model,
                    transport_diagnostics=transport_diagnostics,
                )
            return None

        attempt_index = max(0, int(info.get("attempt_count") or 0)) + 1
        info["attempt_count"] = attempt_index
        entry: dict[str, Any] = {
            "index": attempt_index,
            "provider": _observable_provider_name(attempt.provider_name),
            "model": str(
                getattr(self.plan, "request_model_name", "") or ""
            )[:256],
            "actual_model": str(attempt.original_model or "")[:256],
            "routing_attempt_id": attempt.routing_attempt_id,
            "outcome": "started",
            "_started_monotonic": time.monotonic(),
        }
        started_at = info.get("start_time")
        if isinstance(started_at, (int, float)):
            entry["started_ms"] = max(
                0,
                int((time.time() - float(started_at)) * 1000),
            )

        attempts = info.get("routing_attempts")
        if not isinstance(attempts, list):
            attempts = []
            info["routing_attempts"] = attempts
        if len(attempts) < _ROUTING_ATTEMPT_LIMIT:
            attempts.append(entry)
        else:
            # Preserve the first failures and the most recent failures.  The
            # omitted middle is counted explicitly instead of silently lost.
            attempts.pop(_ROUTING_ATTEMPT_HEAD)
            attempts.append(entry)
            info["routing_attempts_omitted_count"] = max(
                0,
                int(info.get("routing_attempts_omitted_count") or 0),
            ) + 1
        attempt.state["_routing_attempt_entry"] = entry
        transport_diagnostics = UpstreamTransportDiagnostics(entry)
        attempt.state["_transport_diagnostics"] = transport_diagnostics
        attempt.state["_transport_context_token"] = (
            bind_upstream_transport_diagnostics(transport_diagnostics)
        )
        if response_memory_lease is not None:
            response_memory_lease.begin_response_attempt(
                entry,
                routing_attempt_id=attempt.routing_attempt_id,
                routing_attempt_index=attempt_index,
                provider=_observable_provider_name(attempt.provider_name),
                request_model=str(
                    getattr(self.plan, "request_model_name", "") or ""
                ),
                actual_model=attempt.original_model,
                transport_diagnostics=transport_diagnostics,
            )
        return entry

    @staticmethod
    def _finish_transport_diagnostics(
        attempt: UpstreamAttemptContext,
        *,
        outcome: str,
    ) -> None:
        diagnostics = attempt.state.get("_transport_diagnostics")
        if isinstance(diagnostics, UpstreamTransportDiagnostics):
            diagnostics.finalize(outcome)
        token = attempt.state.pop("_transport_context_token", None)
        if token is not None:
            try:
                reset_upstream_transport_diagnostics(token)
            except (LookupError, RuntimeError, ValueError):
                pass

    def _record_retry_transition(
        self,
        previous_entry: Optional[dict[str, Any]],
        next_entry: Optional[dict[str, Any]],
    ) -> None:
        info = self.observability_context
        if info is None or previous_entry is None or next_entry is None:
            return
        previous_entry["retry_transition_to_index"] = int(
            next_entry.get("index") or 0
        )
        info["retry_transition_count"] = max(
            0,
            int(info.get("retry_transition_count") or 0),
        ) + 1

    def _record_routing_failure(
        self,
        attempt: UpstreamAttemptContext,
        exc: Exception,
        *,
        status_code: int,
        error_message: Any,
        retry_decision: bool,
        local_admission_rejection: bool,
        status_origin: str,
    ) -> None:
        failure = {
            "wire_status_code": (
                getattr(exc, "wire_status_code", None)
                or attempt.state.get("routing_wire_status_code")
            ),
            "semantic_status_code": int(status_code),
            "terminal_event_type": str(
                getattr(exc, "event_type", None) or ""
            )[:128]
            or None,
            "error_code": str(
                getattr(exc, "error_code", None) or ""
            )[:128]
            or None,
            "error_type": str(
                getattr(exc, "error_type", None) or type(exc).__name__
            )[:128],
            "error_message_sha256": _routing_error_sha256(error_message),
            "error_message_hash_scope": _ROUTING_ERROR_HASH_SCOPE,
            "retry_decision": bool(retry_decision),
            "retry_reason": (
                f"http_{int(status_code)}:{type(exc).__name__}"
            )[:128],
            "local_admission_rejected": bool(local_admission_rejection),
            "status_origin": str(status_origin)[:64],
            "provider_model_circuit_opened": bool(
                attempt.state.get("provider_model_circuit_opened")
            ),
            "provider_model_circuit_blocks_retry": bool(
                attempt.state.get("provider_model_circuit_blocks_retry")
            ),
        }
        for key in (
            "transport_error_kind",
            "transport_error_owner",
            "transport_error_phase",
            "transport_error_status_code",
            "provider_penalty_eligible",
            "local_overload",
        ):
            if key in attempt.state:
                failure[key] = attempt.state[key]
        attempt.state["_routing_failure"] = failure
        if retry_decision and self.observability_context is not None:
            info = self.observability_context
            info["retry_decision_count"] = max(
                0,
                int(info.get("retry_decision_count") or 0),
            ) + 1
        if (
            attempt.state.get("transport_error_kind")
            and self.observability_context is not None
        ):
            info = self.observability_context
            info["transport_error_count"] = max(
                0,
                int(info.get("transport_error_count") or 0),
            ) + 1
            if attempt.state.get("local_overload"):
                info["local_overload_count"] = max(
                    0,
                    int(info.get("local_overload_count") or 0),
                ) + 1

    def _classify_transport_failure(
        self,
        attempt: UpstreamAttemptContext,
        exc: BaseException,
    ) -> TransportErrorClassification | None:
        diagnostics = attempt.state.get("_transport_diagnostics")
        failure_stage = None
        if isinstance(diagnostics, UpstreamTransportDiagnostics):
            failure_stage = diagnostics.facts.get("failure_stage")

        first_byte_observed: bool | None = None
        responses_diagnostics = attempt.state.get(
            "responses_stream_diagnostics_tracker"
        )
        responses_facts = getattr(responses_diagnostics, "facts", None)
        if isinstance(responses_facts, dict):
            failure_stage = (
                responses_facts.get("phase")
                or failure_stage
            )
            if "upstream_chunk_count" in responses_facts:
                first_byte_observed = bool(
                    int(responses_facts.get("upstream_chunk_count") or 0)
                )
        if (
            first_byte_observed is None
            and self.observability_context is not None
        ):
            first_byte_observed = bool(
                self.observability_context.get("_first_byte_observed")
            )

        classification = classify_httpx_transport_error(
            exc,
            failure_stage=failure_stage,
            first_byte_observed=first_byte_observed,
        )
        if classification is None:
            return None

        attempt.state.update(classification.observability_facts())
        if isinstance(diagnostics, UpstreamTransportDiagnostics):
            diagnostics.record_transport_failure(classification)
        return classification

    @staticmethod
    def _finish_routing_attempt(
        attempt: UpstreamAttemptContext,
        result: UpstreamAttemptResult,
    ) -> Optional[dict[str, Any]]:
        entry = attempt.state.get("_routing_attempt_entry")
        if not isinstance(entry, dict):
            failure = attempt.state.get("_routing_failure")
            response = result.response
            collected_stream_outcome = str(
                getattr(
                    response,
                    "_uni_api_response_attempt_terminal_outcome",
                    "",
                )
                or ""
            )[:80]
            if isinstance(failure, dict):
                outcome = (
                    "retry_decided"
                    if bool(failure.get("retry_decision"))
                    else "failed"
                )
            elif (
                response is not None
                and hasattr(response, "body_iterator")
                and not collected_stream_outcome
            ):
                outcome = "stream_pending"
            else:
                outcome = collected_stream_outcome or "finished"
            lease = attempt.state.get("_response_memory_lease")
            if lease is not None:
                lease.finish_response_attempt(
                    outcome=outcome,
                    keep_active=outcome == "stream_pending",
                )
            UpstreamRunner._finish_transport_diagnostics(
                attempt,
                outcome=outcome,
            )
            return None

        started_monotonic = entry.get("_started_monotonic")
        if isinstance(started_monotonic, (int, float)):
            entry["duration_ms"] = max(
                0,
                int((time.monotonic() - float(started_monotonic)) * 1000),
            )

        failure = attempt.state.get("_routing_failure")
        if isinstance(failure, dict):
            entry.update(failure)
            entry["success"] = False
            entry["outcome"] = (
                "retry_decided"
                if bool(failure.get("retry_decision"))
                else "failed"
            )
            entry.pop("_started_monotonic", None)
            lease = attempt.state.get("_response_memory_lease")
            if lease is not None:
                lease.finish_response_attempt(outcome=str(entry["outcome"]))
            UpstreamRunner._finish_transport_diagnostics(
                attempt,
                outcome=str(entry["outcome"]),
            )
            return entry

        response = result.response
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            entry["wire_status_code"] = int(response_status)
        collected_stream_outcome = str(
            getattr(
                response,
                "_uni_api_response_attempt_terminal_outcome",
                "",
            )
            or ""
        )[:80]
        if (
            response is not None
            and hasattr(response, "body_iterator")
            and not collected_stream_outcome
        ):
            entry["outcome"] = "stream_pending"
            lease = attempt.state.get("_response_memory_lease")
            if lease is not None:
                lease.finish_response_attempt(
                    outcome="stream_pending",
                    keep_active=True,
                )
            UpstreamRunner._finish_transport_diagnostics(
                attempt,
                outcome="stream_pending",
            )
            return entry

        success = bool(
            response is not None
            and isinstance(response_status, int)
            and response_status < 400
        )
        entry["success"] = success
        entry["outcome"] = (
            collected_stream_outcome
            or ("succeeded" if success else "finished")
        )
        entry.pop("_started_monotonic", None)
        lease = attempt.state.get("_response_memory_lease")
        if lease is not None:
            lease.finish_response_attempt(outcome=str(entry["outcome"]))
        UpstreamRunner._finish_transport_diagnostics(
            attempt,
            outcome=str(entry["outcome"]),
        )
        return entry

    def _runtime_api_list(self) -> list[str]:
        return list(self.runtime_api_list)

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
        retry_decider=None,
        max_attempts: Optional[int] = None,
        on_retry=None,
        on_cooldown=None,
    ) -> Any:
        final_local_admission_reason: Optional[str] = None
        final_local_admission_retry_after: Optional[int] = None
        final_status_origin: Optional[str] = None
        pending_retry_entry: Optional[dict[str, Any]] = None
        attempts_started = 0
        while True:
            if max_attempts is not None and attempts_started >= max(0, int(max_attempts)):
                if pending_retry_entry is not None:
                    pending_retry_entry["outcome"] = "retry_exhausted"
                break
            if before_next_attempt is not None:
                maybe_response = await _maybe_await(before_next_attempt())
                if maybe_response is not None:
                    if pending_retry_entry is not None:
                        pending_retry_entry["outcome"] = (
                            "retry_aborted_before_transition"
                        )
                    return maybe_response

            attempt = await self.next_attempt()
            if attempt is None:
                if pending_retry_entry is not None:
                    pending_retry_entry["outcome"] = "retry_exhausted"
                break

            attempt_entry = self._start_routing_attempt(attempt)
            attempts_started += 1
            self._record_retry_transition(pending_retry_entry, attempt_entry)
            pending_retry_entry = None

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
                retry_decider=retry_decider,
                on_retry=on_retry,
                on_cooldown=on_cooldown,
            )
            completed_entry = self._finish_routing_attempt(attempt, result)
            response_status = getattr(result.response, "status_code", None)
            if isinstance(response_status, int) and response_status < 400:
                await maybe_reset_provider_model_circuit(
                    self.plan,
                    attempt.provider_name,
                    attempt.original_model,
                )
            final_local_admission_reason = result.local_admission_reason
            final_local_admission_retry_after = (
                result.local_admission_retry_after
            )
            final_status_origin = result.status_origin
            if result.should_retry:
                pending_retry_entry = completed_entry
                continue
            if result.finalize:
                break
            if result.response is not None:
                self._apply_status_origin_header(
                    result.response,
                    status_origin=result.status_origin,
                )
                if self.observability_context is not None:
                    self.observability_context["status_origin"] = (
                        result.status_origin
                    )
                return result.response

        if build_final_response is not None:
            response = await _maybe_await(build_final_response(self.plan))
        else:
            response = JSONResponse(
                status_code=self.plan.status_code,
                content={"error": f"All {self.plan.request_model_name} error: {self.plan.error_message}"},
            )
        self._apply_local_admission_headers(
            response,
            reason=final_local_admission_reason,
            retry_after_seconds=final_local_admission_retry_after,
        )
        self._apply_status_origin_header(
            response,
            status_origin=final_status_origin,
        )
        if self.observability_context is not None:
            self.observability_context["status_origin"] = final_status_origin
        return response

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
        retry_decider=None,
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
                    try:
                        return await self._handle_failure(
                            attempt,
                            exc,
                            after_failure=after_failure,
                            exclude_error_substrings=exclude_error_substrings,
                            rollback_rate_limit_errors=rollback_rate_limit_errors,
                            should_cool_down=should_cool_down,
                            retry_decider=retry_decider,
                            on_retry=on_retry,
                            on_cooldown=on_cooldown,
                            prepare_failure=True,
                        )
                    finally:
                        _clear_exception_traceback_chain(exc)
            response = await execute_attempt(attempt)
            if isinstance(response, UpstreamAttemptResult):
                return response
            return UpstreamAttemptResult(response=response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                return await self._handle_failure(
                    attempt,
                    exc,
                    after_failure=after_failure,
                    build_error_response=build_error_response,
                    exclude_error_substrings=exclude_error_substrings,
                    rollback_rate_limit_errors=rollback_rate_limit_errors,
                    allow_channel_exclusion=allow_channel_exclusion,
                    should_cool_down=should_cool_down,
                    retry_decider=retry_decider,
                    on_retry=on_retry,
                    on_cooldown=on_cooldown,
                    prepare_failure=False,
                )
            finally:
                _clear_exception_traceback_chain(exc)

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
        retry_decider=None,
        on_retry=None,
        on_cooldown=None,
        prepare_failure: bool,
    ) -> UpstreamAttemptResult:
        transport_failure = self._classify_transport_failure(attempt, exc)
        status_code, error_message = normalize_provider_exception(exc)
        original_status_code = status_code
        status_code = remap_status_code_from_error(status_code, error_message)
        local_admission_rejection = bool(
            getattr(exc, "local_admission_rejection", False)
        )
        diagnostics = attempt.state.get("_transport_diagnostics")
        observed_wire_status = (
            diagnostics.facts.get("upstream_http_status_code")
            if isinstance(diagnostics, UpstreamTransportDiagnostics)
            else None
        )
        if local_admission_rejection:
            status_origin = "ember_local_admission"
        elif (
            observed_wire_status == original_status_code
            or observed_wire_status == status_code
        ):
            status_origin = "provider_http"
        elif bool(getattr(exc, "upstream_semantic_error", False)):
            status_origin = "provider_semantic"
        elif transport_failure is not None:
            status_origin = transport_failure.owner
        else:
            status_origin = "ember_upstream_processing"
        attempt.state["status_origin"] = status_origin
        if isinstance(diagnostics, UpstreamTransportDiagnostics):
            protocol_reason = str(
                diagnostics.facts.get("protocol_error_reason") or ""
            )
            if (
                isinstance(
                    exc,
                    (httpx.LocalProtocolError, httpx.RemoteProtocolError),
                )
                and protocol_reason
                and protocol_reason != "NOT_PROTOCOL_ERROR"
            ):
                prefix = (
                    "Local" if isinstance(exc, httpx.LocalProtocolError) else "Remote"
                )
                error_message = f"{prefix} protocol error: {protocol_reason}"
        self.plan.record_failure(status_code, error_message)
        local_admission_reason = (
            str(getattr(exc, "reason", "") or "upstream_overload")
            if local_admission_rejection
            else None
        )
        local_admission_retry_after = (
            max(1, int(getattr(exc, "retry_after_seconds", 1) or 1))
            if local_admission_rejection
            else None
        )
        if local_admission_rejection:
            attempt.state["local_admission_rejected"] = True
            attempt.state["track_channel_stats"] = False
        if (
            transport_failure is not None
            and not transport_failure.provider_penalty_eligible
        ):
            attempt.state["track_channel_stats"] = False

        request_scoped_failure = status_code in (400, 413) or (
            status_code == 404
            and _is_missing_persisted_responses_item_error(
                status_code,
                error_message,
            )
        )
        semantic_request_failure = bool(
            getattr(exc, "upstream_semantic_error", False)
            and status_code in (400, 413)
        )
        if request_scoped_failure:
            attempt.state["track_channel_stats"] = False
        if (
            allow_channel_exclusion
            and not prepare_failure
            and not local_admission_rejection
            and (
                transport_failure is None
                or transport_failure.provider_penalty_eligible
            )
            and not request_scoped_failure
        ):
            circuit_result = await maybe_exclude_failed_channel(
                self.plan,
                attempt.provider_name,
                attempt.original_model,
                status_code,
                error_message,
                provider=attempt.provider,
                exclude_error_substrings=exclude_error_substrings,
                debug=self.debug,
            )
            attempt.state["provider_model_circuit_opened"] = bool(
                circuit_result
            )
            attempt.state["provider_model_circuit_blocks_retry"] = (
                circuit_result == "opened_no_alternative"
            )

        force_cool_key = _is_codex_chatgpt_model_unsupported_error(
            status_code,
            error_message,
            attempt.provider,
            self.endpoint,
            attempt.original_model,
        )
        should_cool_key = bool(
            not local_admission_rejection
            and (
                transport_failure is None
                or transport_failure.provider_penalty_eligible
            )
            and (force_cool_key or not request_scoped_failure)
        )
        if (
            should_cool_down is not None
            and not local_admission_rejection
            and (
                transport_failure is None
                or transport_failure.provider_penalty_eligible
            )
        ):
            should_cool_key = force_cool_key or bool(
                not request_scoped_failure
                and await _maybe_await(
                    should_cool_down(
                        exc,
                        status_code,
                        error_message,
                        attempt,
                    )
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

        if not local_admission_rejection:
            await maybe_clear_provider_auth_cache(
                attempt,
                self.endpoint,
                status_code,
                self.clear_provider_auth_cache,
            )

        if after_failure is not None:
            await _maybe_await(after_failure(attempt, exc, status_code, error_message))

        if prepare_failure:
            retry_decision = bool(
                self.plan.auto_retry
                and not local_admission_rejection
                and not semantic_request_failure
            )
            if retry_decider is not None:
                retry_decision = bool(
                    await _maybe_await(
                        retry_decider(
                            exc,
                            status_code,
                            error_message,
                            attempt,
                            True,
                        )
                    )
                )
            self._record_routing_failure(
                attempt,
                exc,
                status_code=status_code,
                error_message=error_message,
                retry_decision=retry_decision,
                local_admission_rejection=local_admission_rejection,
                status_origin=status_origin,
            )
            if retry_decision:
                if on_retry is not None:
                    await _maybe_await(on_retry(attempt, status_code, error_message))
                return UpstreamAttemptResult(
                    should_retry=True,
                    local_admission_reason=local_admission_reason,
                    local_admission_retry_after=local_admission_retry_after,
                    status_origin=status_origin,
                )
            return UpstreamAttemptResult(
                finalize=True,
                local_admission_reason=local_admission_reason,
                local_admission_retry_after=local_admission_retry_after,
                status_origin=status_origin,
            )

        retry_decision = bool(
            not local_admission_rejection
            and not semantic_request_failure
            and not attempt.state.get("provider_model_circuit_blocks_retry")
            and should_retry_provider(
                self.plan.auto_retry,
                status_code,
                attempt.provider,
                error_message=error_message,
                endpoint=self.endpoint,
                original_model=attempt.original_model,
            )
        )
        if retry_decider is not None:
            retry_decision = bool(
                await _maybe_await(
                    retry_decider(
                        exc,
                        status_code,
                        error_message,
                        attempt,
                        False,
                    )
                )
            )
        self._record_routing_failure(
            attempt,
            exc,
            status_code=status_code,
            error_message=error_message,
            retry_decision=retry_decision,
            local_admission_rejection=local_admission_rejection,
            status_origin=status_origin,
        )
        if retry_decision:
            if on_retry is not None:
                await _maybe_await(on_retry(attempt, status_code, error_message))
            return UpstreamAttemptResult(
                should_retry=True,
                local_admission_reason=local_admission_reason,
                local_admission_retry_after=local_admission_retry_after,
                status_origin=status_origin,
            )

        if build_error_response is not None:
            response = await _maybe_await(build_error_response(status_code, error_message))
        else:
            response = JSONResponse(
                status_code=status_code,
                content={"error": f"Error: Current provider response failed: {error_message}"},
            )
        self._apply_local_admission_headers(
            response,
            reason=local_admission_reason,
            retry_after_seconds=local_admission_retry_after,
        )
        self._apply_status_origin_header(
            response,
            status_origin=status_origin,
        )
        return UpstreamAttemptResult(
            response=response,
            local_admission_reason=local_admission_reason,
            local_admission_retry_after=local_admission_retry_after,
            status_origin=status_origin,
        )

    @staticmethod
    def _apply_local_admission_headers(
        response: Any,
        *,
        reason: Optional[str],
        retry_after_seconds: Optional[int],
    ) -> None:
        if not reason or not hasattr(response, "headers"):
            return
        response.headers["retry-after"] = str(
            max(1, int(retry_after_seconds or 1))
        )
        response.headers["x-uni-api-admission-reason"] = reason

    @staticmethod
    def _apply_status_origin_header(
        response: Any,
        *,
        status_origin: Optional[str],
    ) -> None:
        if not status_origin or not hasattr(response, "headers"):
            return
        response.headers["x-uni-api-status-origin"] = str(status_origin)[:64]
