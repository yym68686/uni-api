from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass
from time import time
from typing import Any, Awaitable, Callable, Optional

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from core.log_config import logger
from core.models import ModerationRequest
from core.utils import safe_get
from uni_api.serialization import json
from uni_api.observability.request_context import (
    RequestContext,
    get_request_info,
    reset_request_info,
    set_request_info,
)
from uni_api.observability.request_inspection import inspect_request_body
from uni_api.streaming.logging_response import LoggingStreamingResponse


@dataclass(frozen=True)
class StatsMiddlewareDependencies:
    app_state: Any
    database_disabled: bool
    runtime_gauges: Any
    trace_factory: Callable[..., Any]
    incoming_trace_context: Callable[[Any], dict[str, str]]
    get_api_key: Callable[[Request], Awaitable[Optional[str]]]
    get_client_ip: Callable[[Request], str]
    parse_request_body: Callable[[Request], Awaitable[Any]]
    message_role_summary: Callable[[Any], tuple[Optional[str], Optional[str]]]
    messages_request_last_text: Callable[[Any], Optional[str]]
    is_public_health_request: Callable[[Request], bool]
    is_video_or_asset_request_path: Callable[[str], bool]
    lingjing_request_model_for_openapi: Callable[[Optional[dict[str, Any]], Any], str]
    video_prompt_from_body: Callable[[dict[str, Any]], str]
    monitor_disconnect: Callable[[Request, asyncio.Event], Awaitable[None]]
    log_debug_request_headers: Callable[..., None]
    log_debug_request_body: Callable[..., None]
    mask_secret_for_log: Callable[[Any], str]
    update_stats: Callable[[dict[str, Any]], Awaitable[None]]
    emit_request_observability: Callable[[dict[str, Any]], None]
    mark_first_byte_observed: Callable[[dict[str, Any]], None]
    moderation_handler: Callable[[ModerationRequest, BackgroundTasks, int], Awaitable[Any]]
    logging_response_class: type[LoggingStreamingResponse] = LoggingStreamingResponse
    debug: bool = False


class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, dependencies: StatsMiddlewareDependencies):
        super().__init__(app)
        self.dependencies = dependencies

    def _forbidden_response(self, trace) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": "Invalid or missing API Key"},
            headers={"x-request-id": trace.trace_id},
        )

    def _api_index_for_token(self, token: str) -> int | None:
        api_key_index = getattr(self.dependencies.app_state, "api_key_index", None)
        if api_key_index is None:
            api_list = getattr(self.dependencies.app_state, "api_list", []) or []
            api_key_index = {api_key: index for index, api_key in enumerate(api_list)}
            self.dependencies.app_state.api_key_index = api_key_index
        return api_key_index.get(token)

    def _request_context(self, request: Request, *, trace, incoming_trace: dict[str, str], start_time: float, token: str) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        return RequestContext(
            request_id=request_id,
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            parent_span_id=trace.parent_span_id,
            trace_flags=trace.trace_flags,
            tracestate=trace.tracestate,
            x_request_id=incoming_trace.get("x_request_id"),
            start_time=start_time,
            endpoint=f"{request.method} {request.url.path}",
            client_ip=self.dependencies.get_client_ip(request),
            api_key=token,
            timing_spans=trace.snapshot(),
            extras={"trace": trace},
        ).to_dict()

    def _paid_key_enabled_response(self, token: str, api_index: int, request: Request, trace) -> JSONResponse | None:
        deps = self.dependencies
        if deps.database_disabled or request.url.path.startswith("/v1/token_usage"):
            return None
        check_api_key = safe_get(deps.app_state.config, "api_keys", api_index, "api")
        if safe_get(getattr(deps.app_state, "paid_api_keys_states", {}), check_api_key, "enabled", default=None) is not False:
            return None
        _ = token
        return JSONResponse(
            status_code=429,
            content={"error": "Balance is insufficient, please check your account."},
            headers={"x-request-id": trace.trace_id},
        )

    async def _parse_and_log_body(self, request: Request, request_id: str, trace) -> Any:
        deps = self.dependencies
        deps.log_debug_request_headers(
            "DEBUG client request headers",
            request.headers,
            method=request.method,
            endpoint=request.url.path,
            request_id=request_id,
        )
        parsed_body = await deps.parse_request_body(request)
        trace.mark("body_parsed")
        if parsed_body is not None:
            deps.log_debug_request_body(
                "DEBUG client request body",
                parsed_body,
                method=request.method,
                endpoint=request.url.path,
                request_id=request_id,
            )
        return parsed_body

    async def _start_disconnect_monitor(self, request: Request, current_info: dict[str, Any]) -> tuple[Optional[asyncio.Event], Optional[asyncio.Task]]:
        if request.method != "POST" or "application/json" not in request.headers.get("content-type", ""):
            return None, None
        disconnect_event = asyncio.Event()
        current_info["disconnect_event"] = disconnect_event
        return disconnect_event, asyncio.create_task(self.dependencies.monitor_disconnect(request, disconnect_event))

    async def _apply_body_policy(
        self,
        request: Request,
        *,
        parsed_body: Any,
        api_index: int,
        enable_moderation: bool,
        current_info: dict[str, Any],
        start_time: float,
    ) -> JSONResponse | None:
        if not parsed_body or request.url.path.startswith("/v1/api_config"):
            return None

        deps = self.dependencies
        final_api_key = deps.app_state.api_list[api_index]
        moderated_content = await self._rate_limit_and_extract_moderation_text(
            request,
            parsed_body=parsed_body,
            current_info=current_info,
            final_api_key=final_api_key,
        )
        if isinstance(moderated_content, JSONResponse):
            return moderated_content
        if enable_moderation and moderated_content:
            return await self._moderation_response_if_flagged(
                moderated_content,
                api_index=api_index,
                current_info=current_info,
                start_time=start_time,
            )
        return None

    async def _rate_limit_and_extract_moderation_text(
        self,
        request: Request,
        *,
        parsed_body: Any,
        current_info: dict[str, Any],
        final_api_key: str,
    ) -> str | JSONResponse | None:
        deps = self.dependencies
        if request.url.path.rstrip("/") == "/v1/messages":
            if isinstance(parsed_body, dict):
                model = str(parsed_body.get("model") or "").strip()
                if model:
                    current_info["model"] = model
                    limited_response = await self._rate_limit_response(final_api_key, model, current_info)
                    if limited_response is not None:
                        return limited_response
            return deps.messages_request_last_text(parsed_body)

        if deps.is_video_or_asset_request_path(request.url.path):
            model = deps.lingjing_request_model_for_openapi(
                parsed_body if isinstance(parsed_body, dict) else None,
                request.query_params,
            )
            current_info["model"] = model
            limited_response = await self._rate_limit_response(final_api_key, model, current_info)
            if limited_response is not None:
                return limited_response
            if isinstance(parsed_body, dict):
                moderated_content = str(safe_get(parsed_body, "taskParams", "input", "prompt", default="") or "").strip()
                return moderated_content or deps.video_prompt_from_body(parsed_body)
            return None

        inspection = inspect_request_body(parsed_body)
        model = inspection.model
        current_info["model"] = model
        if model:
            limited_response = await self._rate_limit_response(final_api_key, model, current_info)
            if limited_response is not None:
                return limited_response
        if inspection.request_type is None:
            logger.error("Unknown request type for middleware inspection: %s", request.url.path)
        return inspection.moderated_content

    async def _rate_limit_response(self, final_api_key: str, model: str, current_info: dict[str, Any]) -> JSONResponse | None:
        try:
            await self.dependencies.app_state.user_api_keys_rate_limit[final_api_key].next(model)
        except Exception:
            current_info["status_code"] = 429
            current_info["error_type"] = "rate_limited"
            return JSONResponse(status_code=429, content={"error": "Too many requests"})
        return None

    async def _moderation_response_if_flagged(
        self,
        moderated_content: str,
        *,
        api_index: int,
        current_info: dict[str, Any],
        start_time: float,
    ) -> JSONResponse | None:
        moderation_response = await self.moderate_content(moderated_content, api_index, BackgroundTasks())
        is_flagged = moderation_response.get("results", [{}])[0].get("flagged", False)
        if not is_flagged:
            return None
        logger.error("Content did not pass the moral check: %s", moderated_content)
        current_info["process_time"] = time() - start_time
        current_info["is_flagged"] = is_flagged
        current_info["text"] = moderated_content
        current_info["status_code"] = 400
        current_info["error_type"] = "moderation_flagged"
        await self.dependencies.update_stats(current_info)
        return JSONResponse(
            status_code=400,
            content={"error": "Content did not pass the moral check, please modify and try again."},
        )

    async def _wrap_response_for_observability(self, request: Request, response, current_info: dict[str, Any], trace):
        deps = self.dependencies
        trace.mark("downstream_response_start")
        response.headers["x-request-id"] = trace.trace_id
        current_info["status_code"] = getattr(response, "status_code", 0) or 0
        if not request.url.path.startswith("/v1") or deps.database_disabled:
            return response

        if isinstance(response, StarletteStreamingResponse) or type(response).__name__ == "_StreamingResponse":
            current_info["_defer_observability_until_stream_end"] = True
            return deps.logging_response_class(
                content=response.body_iterator,
                status_code=response.status_code,
                media_type=response.media_type,
                headers=response.headers,
                current_info=current_info,
                mark_first_byte_observed=deps.mark_first_byte_observed,
                emit_request_observability=deps.emit_request_observability,
                update_stats=deps.update_stats,
                trace_type=type(trace),
                debug=deps.debug,
            )
        if hasattr(response, "json"):
            logger.info("Response: %s", await response.json())
        else:
            logger.info(
                "Response: type=%s, status_code=%s, headers=%s",
                type(response).__name__,
                response.status_code,
                response.headers,
            )
        return response

    async def dispatch(self, request: Request, call_next):
        deps = self.dependencies
        runtime_gauges = deps.runtime_gauges

        if request.method == "OPTIONS":
            return await call_next(request)
        if deps.is_public_health_request(request):
            return await call_next(request)

        start_time = time()
        incoming_trace = deps.incoming_trace_context(request.headers)
        trace = deps.trace_factory(
            trace_id=incoming_trace["trace_id"],
            parent_span_id=incoming_trace.get("parent_span_id"),
            trace_flags=incoming_trace.get("trace_flags"),
            tracestate=incoming_trace.get("tracestate"),
        )
        if incoming_trace.get("x_request_id"):
            trace.set_tag("x_request_id", incoming_trace.get("x_request_id"))
        trace.mark("request_received")
        runtime_gauges.begin_inflight()
        await runtime_gauges.record_event_loop_lag()

        token = await deps.get_api_key(request)
        if not token:
            runtime_gauges.end_inflight()
            return self._forbidden_response(trace)

        api_index = self._api_index_for_token(token)
        if api_index is not None:
            enable_moderation = safe_get(deps.app_state.config, "api_keys", api_index, "preferences", "ENABLE_MODERATION", default=False)
            paid_key_response = self._paid_key_enabled_response(token, api_index, request, trace)
            if paid_key_response is not None:
                runtime_gauges.end_inflight()
                return paid_key_response
        else:
            runtime_gauges.end_inflight()
            return self._forbidden_response(trace)
        trace.mark("auth_done")

        request_info_data = self._request_context(
            request,
            trace=trace,
            incoming_trace=incoming_trace,
            start_time=start_time,
            token=token,
        )
        current_request_info = set_request_info(request_info_data)
        current_info = get_request_info()
        disconnect_task: Optional[asyncio.Task] = None
        try:
            parsed_body = await self._parse_and_log_body(request, current_info["request_id"], trace)
            if isinstance(parsed_body, dict):
                current_info["stream"] = parsed_body.get("stream")
                current_info["request_kind"] = request.url.path
                message_roles, role_counts = deps.message_role_summary(parsed_body)
                current_info["message_roles"] = message_roles
                current_info["role_counts"] = role_counts
            _, disconnect_task = await self._start_disconnect_monitor(request, current_info)
            policy_response = await self._apply_body_policy(
                request,
                parsed_body=parsed_body,
                api_index=api_index,
                enable_moderation=enable_moderation,
                current_info=current_info,
                start_time=start_time,
            )
            if policy_response is not None:
                return policy_response

            response = await call_next(request)
            return await self._wrap_response_for_observability(request, response, current_info, trace)

        except HTTPException as e:
            current_info["status_code"] = getattr(e, "status_code", 500)
            current_info["error_type"] = "http_exception"
            raise
        except ValidationError as e:
            logger.error("API key: %s, invalid request body: %s", deps.mask_secret_for_log(token), e.errors())
            content = await asyncio.to_thread(jsonable_encoder, {"detail": e.errors()})
            current_info["status_code"] = 422
            current_info["error_type"] = "validation_error"
            return JSONResponse(status_code=422, content=content)
        except Exception as e:
            if deps.debug:
                import traceback

                traceback.print_exc()
            logger.error("Error processing request: %s", e)
            current_info["status_code"] = 500
            current_info["error_type"] = type(e).__name__
            return JSONResponse(status_code=500, content={"error": f"Internal server error: {e}"})

        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            trace.mark("stream_end")
            current_info["process_time"] = time() - start_time
            current_info["timing_spans"] = trace.snapshot()
            logger.info(
                "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                current_info.get("trace_id"),
                current_info.get("request_id"),
                current_info.get("endpoint"),
                current_info.get("timing_spans"),
            )
            if not current_info.get("_defer_observability_until_stream_end"):
                deps.emit_request_observability(current_info)
            runtime_gauges.end_inflight()
            reset_request_info(current_request_info)

    async def moderate_content(self, content: str, api_index: int, background_tasks: BackgroundTasks) -> dict[str, Any]:
        response = await self.dependencies.moderation_handler(
            ModerationRequest(input=content),
            background_tasks,
            api_index,
        )

        moderation_result = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                moderation_result += chunk.encode("utf-8")
            else:
                moderation_result += chunk

        return json.loads(moderation_result.decode("utf-8"))
