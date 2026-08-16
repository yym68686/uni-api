from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Callable

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from core.utils import get_engine, safe_get
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.idempotency import apply_oaix_routing_attempt_id
from uni_api.providers.header_passthrough import apply_provider_preference_headers
from uni_api.providers.payloads import force_codex_client_headers
from uni_api.routing.planner import (
    RoutingPlan,
    get_right_order_providers,
    select_provider_api_key_raw,
)
from uni_api.routing.request_rules import request_reasoning_effort
from uni_api.upstream.urls import normalize_alpha_search_upstream_url
from upstream import UpstreamRunner


ALPHA_SEARCH_ENDPOINT = "/v1/alpha/search"

_SENSITIVE_CLIENT_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
    "chatgpt-account-id",
}
_UNSAFE_RESPONSE_HEADERS = {
    "api-key",
    "authorization",
    "chatgpt-account-id",
    "connection",
    "content-encoding",
    "content-length",
    "cookie",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "set-cookie",
    "set-cookie2",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-api-key",
}
_SAFE_OAIX_RESPONSE_HEADERS = {
    "x-oaix-connection-id",
    "x-oaix-request-id",
}


class _RetryableAlphaSearchResponse(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        self.reason = f"alpha_search_upstream_status_{self.status_code}"
        super().__init__(self.reason)


def _safe_provider_label(provider_name: str) -> str:
    if provider_name.startswith("sk-"):
        digest = hashlib.sha256(provider_name.encode("utf-8")).hexdigest()[:16]
        return f"local-api-key:{digest}"
    return provider_name[:256]


def _safe_client_header_source(http_request: Request) -> SimpleNamespace:
    safe_headers: dict[str, str] = {}
    for name, value in (getattr(http_request, "headers", None) or {}).items():
        normalized = str(name).lower()
        if normalized in _SENSITIVE_CLIENT_HEADERS or normalized.startswith("x-oaix-"):
            continue
        safe_headers[str(name)] = str(value)
    return SimpleNamespace(headers=safe_headers)


def _set_header(headers: dict[str, Any], name: str, value: Any) -> None:
    normalized = name.lower()
    for existing_name in list(headers):
        if str(existing_name).lower() == normalized:
            headers.pop(existing_name, None)
    if value is not None and str(value) != "":
        headers[name] = str(value)


def _get_header(headers: Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)
    normalized = name.lower()
    for existing_name, value in (headers or {}).items():
        if str(existing_name).lower() == normalized:
            return str(value)
    return None


def _copy_alpha_search_response_headers(headers: Any) -> dict[str, str]:
    grouped: dict[str, tuple[str, list[str]]] = {}
    raw_headers = getattr(headers, "raw", None)
    pairs = raw_headers if raw_headers else (headers or {}).items()
    for raw_name, raw_value in pairs:
        name = (
            raw_name.decode("latin-1", errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        value = (
            raw_value.decode("latin-1", errors="replace")
            if isinstance(raw_value, bytes)
            else str(raw_value)
        ).strip(" \t")
        normalized = name.lower()
        if normalized in _UNSAFE_RESPONSE_HEADERS or not value:
            continue
        if (
            normalized.startswith("x-oaix-")
            and normalized not in _SAFE_OAIX_RESPONSE_HEADERS
        ):
            continue
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            continue
        if normalized not in grouped:
            grouped[normalized] = (name, [value])
        else:
            grouped[normalized][1].append(value)
    copied = {name: ", ".join(values) for name, values in grouped.values()}
    _set_header(copied, "Cache-Control", "no-store")
    return copied


def _json_error_response(status_code: int) -> JSONResponse:
    status = int(status_code)
    if status < 400 or status > 599:
        status = 502
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": "alpha/search upstream request failed",
                "type": "upstream_error",
                "code": "alpha_search_upstream_error",
            }
        },
        headers={"Cache-Control": "no-store"},
    )


class AlphaSearchRequestHandler:
    def __init__(
        self,
        *,
        app: Any,
        get_runtime_api_list: Callable[[], list[str]],
        api_key_has_model_rules: Callable[[Any, int], bool],
        resolve_codex_upstream_auth: Callable[..., Any],
        resolve_timeout: Callable[..., Any],
        add_trace_headers: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
        record_plan_observability: Callable[[dict[str, Any], RoutingPlan], Any] | None = None,
        record_retry_observability: Callable[..., Any] | None = None,
        provider_resolver: Callable[..., Any] = get_right_order_providers,
        debug: Callable[[], bool] | None = None,
    ) -> None:
        self.app = app
        self.get_runtime_api_list = get_runtime_api_list
        self.api_key_has_model_rules = api_key_has_model_rules
        self.resolve_codex_upstream_auth = resolve_codex_upstream_auth
        self.resolve_timeout = resolve_timeout
        self.add_trace_headers = add_trace_headers
        self.record_plan_observability = record_plan_observability
        self.record_retry_observability = record_retry_observability
        self.provider_resolver = provider_resolver
        self.debug = debug or (lambda: False)
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_search(
        self,
        *,
        http_request: Request,
        request_body: Any,
        api_index: int,
        background_tasks: BackgroundTasks | None = None,
    ) -> Response:
        _ = background_tasks
        request_model = request_body.get("model")
        api_list = list(self.get_runtime_api_list())
        if api_index < 0 or api_index >= len(api_list):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not self.api_key_has_model_rules(self.app, api_index):
            raise HTTPException(
                status_code=404,
                detail=f"No matching model found: {request_model}",
            )

        return await self._run(
            http_request=http_request,
            request_body=dict(request_body),
            request_model=request_model,
            api_index=api_index,
        )

    async def _run(
        self,
        *,
        http_request: Request,
        request_body: dict[str, Any],
        request_model: str,
        api_index: int,
    ) -> Response:
        encoded_request = await run_json_cpu(
            json.dumps,
            request_body,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            plan = await RoutingPlan.create(
                self.app,
                request_model,
                api_index,
                self.last_provider_indices,
                self.locks,
                endpoint=ALPHA_SEARCH_ENDPOINT,
                request_body_bytes=len(encoded_request.encode("utf-8")),
                reasoning_effort=request_reasoning_effort(request_body),
                debug=bool(self.debug()),
                provider_resolver=self.provider_resolver,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    status_code=503,
                    detail="No providers are available for /v1/alpha/search",
                ) from exc
            raise

        current_info = getattr(
            getattr(http_request, "state", None),
            "uni_api_request_info",
            None,
        )
        if not isinstance(current_info, dict):
            current_info = {}
        current_info["stream"] = False
        current_info["model"] = request_model
        if self.record_plan_observability is not None:
            result = self.record_plan_observability(current_info, plan)
            if inspect.isawaitable(result):
                await result

        execution = _AlphaSearchExecution(
            handler=self,
            http_request=http_request,
            request_body=request_body,
            request_model=request_model,
            plan=plan,
            current_info=current_info,
        )
        return await execution.run()


class _AlphaSearchExecution:
    def __init__(
        self,
        *,
        handler: AlphaSearchRequestHandler,
        http_request: Request,
        request_body: dict[str, Any],
        request_model: str,
        plan: RoutingPlan,
        current_info: dict[str, Any],
    ) -> None:
        self.handler = handler
        self.http_request = http_request
        self.request_body = request_body
        self.request_model = request_model
        self.plan = plan
        self.current_info = current_info
        self.last_retry_response: Response | None = None
        self.runner = UpstreamRunner(
            plan,
            endpoint=ALPHA_SEARCH_ENDPOINT,
            debug=bool(handler.debug()),
            provider_api_key_selector=select_provider_api_key_raw,
            observability_context=current_info,
        )

    async def run(self) -> Response:
        response = await self.runner.run(
            self._execute_attempt,
            prepare_attempt=self._prepare_attempt,
            after_failure=self._after_failure,
            build_error_response=self._build_error_response,
            build_final_response=self._build_final_response,
            allow_channel_exclusion=False,
            should_cool_down=lambda *_args: False,
            retry_decider=self._retry_decider,
            on_retry=self.handler.record_retry_observability,
        )
        if isinstance(response, Response):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def _prepare_attempt(self, attempt: Any) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        upstream_url = normalize_alpha_search_upstream_url(
            provider.get("base_url", "")
        )
        engine, _stream_mode = get_engine(
            provider,
            endpoint=ALPHA_SEARCH_ENDPOINT,
            original_model=original_model,
        )
        proxy = safe_get(self.handler.app.state.config, "preferences", "proxy")
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        attempt.provider_api_key_raw = await self.runner.select_provider_api_key(
            attempt
        )
        api_key = attempt.provider_api_key_raw
        codex_account_id = None
        if engine == "codex" and api_key:
            api_key, codex_account_id = await self.handler.resolve_codex_upstream_auth(
                provider_name,
                api_key,
                proxy,
            )
        timeout = self.handler.resolve_timeout(
            provider_name=provider_name,
            original_model=original_model,
            request_model=self.request_model,
            role=self.plan.role,
            engine=engine,
        )
        if inspect.isawaitable(timeout):
            timeout = await timeout
        attempt.state.update(
            {
                "upstream_url": upstream_url,
                "engine": engine,
                "proxy": proxy,
                "api_key": api_key,
                "codex_account_id": codex_account_id,
                "timeout": timeout,
            }
        )

    async def _execute_attempt(self, attempt: Any) -> Response:
        payload = dict(self.request_body)
        payload["model"] = attempt.original_model
        json_payload = await run_json_cpu(
            json.dumps,
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        headers = self._build_headers(attempt)
        if self.handler.add_trace_headers is not None:
            result = self.handler.add_trace_headers(headers, self.current_info)
            if inspect.isawaitable(result):
                await result

        async with self.handler.app.state.client_manager.get_client(
            attempt.state["upstream_url"],
            attempt.state["proxy"],
            http2=False if attempt.state["engine"] == "codex" else None,
        ) as client:
            upstream_response = await client.post(
                attempt.state["upstream_url"],
                headers=headers,
                content=json_payload,
                timeout=attempt.state["timeout"],
            )

        raw = bytes(upstream_response.content)
        attempt.state["routing_wire_status_code"] = int(
            upstream_response.status_code
        )
        downstream = Response(
            content=raw,
            status_code=upstream_response.status_code,
            headers=_copy_alpha_search_response_headers(
                upstream_response.headers
            ),
        )
        if not 200 <= upstream_response.status_code < 300:
            self.last_retry_response = downstream
            attempt.state["alpha_retry_response"] = downstream
            self.current_info["success"] = False
            if upstream_response.status_code == 429 or upstream_response.status_code >= 500:
                raise _RetryableAlphaSearchResponse(
                    upstream_response.status_code
                )
            return downstream

        self.current_info["success"] = True
        self.current_info["provider"] = _safe_provider_label(
            attempt.provider_name
        )
        self.current_info["actual_model"] = attempt.original_model
        return downstream

    def _build_headers(self, attempt: Any) -> dict[str, Any]:
        headers: dict[str, Any] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        apply_provider_preference_headers(
            headers,
            attempt.provider,
            http_request=_safe_client_header_source(self.http_request),
        )
        if attempt.state.get("api_key"):
            _set_header(
                headers,
                "Authorization",
                f"Bearer {attempt.state['api_key']}",
            )
        if attempt.state["engine"] == "codex":
            request_headers = getattr(self.http_request, "headers", None) or {}
            if _get_header(headers, "Openai-Beta") is None:
                _set_header(
                    headers,
                    "Openai-Beta",
                    _get_header(request_headers, "Openai-Beta")
                    or "responses=experimental",
                )
            if _get_header(headers, "Originator") is None:
                _set_header(
                    headers,
                    "Originator",
                    _get_header(request_headers, "Originator")
                    or "codex_cli_rs",
                )
            search_id = self.request_body.get("id")
            if isinstance(search_id, str) and search_id:
                _set_header(headers, "Session_id", search_id)
            if attempt.state.get("codex_account_id"):
                _set_header(
                    headers,
                    "Chatgpt-Account-Id",
                    attempt.state["codex_account_id"],
                )
            force_codex_client_headers(headers)
        _set_header(headers, "Content-Type", "application/json")
        _set_header(headers, "Accept", "application/json")
        apply_oaix_routing_attempt_id(
            headers,
            provider=attempt.provider,
            routing_attempt_id=attempt.routing_attempt_id,
        )
        return headers

    async def _retry_decider(
        self,
        exc: Exception,
        status_code: int,
        _error_message: Any,
        _attempt: Any,
        _prepare_failure: bool,
    ) -> bool:
        if not self.plan.auto_retry:
            return False
        if bool(getattr(exc, "local_admission_rejection", False)):
            return False
        return int(status_code) == 429 or 500 <= int(status_code) <= 599

    async def _after_failure(
        self,
        attempt: Any,
        _exc: Exception,
        _status_code: int,
        _error_message: Any,
    ) -> None:
        self.last_retry_response = attempt.state.get("alpha_retry_response")
        self.current_info["success"] = False

    async def _build_error_response(
        self,
        status_code: int,
        _error_message: Any,
    ) -> Response:
        return self.last_retry_response or _json_error_response(status_code)

    async def _build_final_response(self, plan: RoutingPlan) -> Response:
        return self.last_retry_response or _json_error_response(plan.status_code)
