import os
import io
import re
import json
import uuid
import codecs
import httpx
import string
import secrets
import tomllib
import asyncio
from asyncio import Semaphore
import contextvars
from time import time
from urllib.parse import urlparse
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Dict, Union, Optional, List, Any
from pydantic import ValidationError, BaseModel, field_serializer

from starlette.responses import Response
from starlette.types import Scope, Receive, Send
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from fastapi import FastAPI, HTTPException, Depends, Request, Body, BackgroundTasks, UploadFile, File, Form, Query

from core.log_config import logger, trace_logger
from core.request import (
    CODEX_CLI_VERSION,
    CODEX_USER_AGENT,
    apply_post_body_parameter_overrides,
    force_codex_client_headers,
    get_payload,
    strip_unsupported_codex_payload_fields,
)
from core.response import fetch_response, fetch_response_stream
from core.models import RequestModel, ResponsesRequest, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, TextToSpeechRequest, UnifiedRequest, EmbeddingRequest
from core.utils import (
    get_proxy,
    get_engine,
    parse_rate_limit,
    collect_openai_chat_completion_from_streaming_sse,
    ThreadSafeCircularList,
    provider_api_circular_list,
)
from routing import (
    RoutingPlan,
    build_api_key_models_map,
    estimate_request_total_tokens,
    get_right_order_providers,
    select_provider_api_key_raw,
)
from upstream import (
    UPSTREAM_NETWORK_ERRORS,
    UpstreamRunner,
    build_upstream_error_response,
)

from utils import (
    safe_get,
    load_config,
    update_config,
    post_all_models,
    InMemoryRateLimiter,
    error_handling_wrapper,
    query_channel_key_stats,
)

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, case, func, desc

from db import Base, RequestStat, ChannelStat, db_engine, async_session, DISABLE_DATABASE

DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
is_debug = bool(os.getenv("DEBUG", False))
logger.info("DISABLE_DATABASE: %s", DISABLE_DATABASE)

# 从 pyproject.toml 读取版本号
try:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
        VERSION = data['project']['version']
except Exception:
    VERSION = 'unknown'
logger.info("VERSION: %s", VERSION)

async def create_tables():
    if DISABLE_DATABASE:
        return
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 检查并添加缺失的列 - 扩展此简易迁移以支持 SQLite 和 PostgreSQL
        db_type = os.getenv("DB_TYPE", "sqlite").lower()
        if db_type in ["sqlite", "postgres"]:
            def check_and_add_columns(connection):
                inspector = inspect(connection)
                for table in [RequestStat, ChannelStat]:
                    table_name = table.__tablename__
                    existing_columns = {col['name'] for col in inspector.get_columns(table_name)}

                    for column_name, column in table.__table__.columns.items():
                        if column_name not in existing_columns:
                            # 适配 PostgreSQL 和 SQLite 的类型映射
                            col_type = column.type.compile(connection.dialect)
                            default = _get_default_sql(column.default) if db_type == "sqlite" else "" # PostgreSQL 的默认值处理更复杂，暂不处理

                            # 使用标准的 ALTER TABLE 语法
                            connection.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {col_type}{default}'))
                            logger.info(f"Added column '{column_name}' to table '{table_name}'.")

            await conn.run_sync(check_and_add_columns)

def _map_sa_type_to_sql_type(sa_type):
    type_map = {
        sqltypes.Integer: "INTEGER",
        sqltypes.String: "TEXT",
        sqltypes.Float: "REAL",
        sqltypes.Boolean: "BOOLEAN",
        sqltypes.DateTime: "DATETIME",
        sqltypes.Text: "TEXT"
    }
    return type_map.get(type(sa_type), "TEXT")

def _get_default_sql(default):
    if default is None:
        return ""
    if isinstance(default.arg, bool):
        return f" DEFAULT {str(default.arg).upper()}"
    if isinstance(default.arg, (int, float)):
        return f" DEFAULT {default.arg}"
    if isinstance(default.arg, str):
        return f" DEFAULT '{default.arg}'"
    return ""

def init_preference(all_config, preference_key, default_timeout=DEFAULT_TIMEOUT):
    # 存储超时配置
    preference_dict = {}
    preferences = safe_get(all_config, "preferences", default={})
    providers = safe_get(all_config, "providers", default=[])
    if preferences:
        if isinstance(preferences.get(preference_key), int):
            preference_dict["default"] = preferences.get(preference_key)
        else:
            for model_name, timeout_value in preferences.get(preference_key, {"default": default_timeout}).items():
                preference_dict[model_name] = timeout_value
            if "default" not in preferences.get(preference_key, {}):
                preference_dict["default"] = default_timeout

    result = defaultdict(lambda: defaultdict(lambda: default_timeout))
    for provider in providers:
        provider_preference_settings = safe_get(provider, "preferences", preference_key, default={})
        if provider_preference_settings:
            for model_name, timeout_value in provider_preference_settings.items():
                result[provider['provider']][model_name] = timeout_value

    result["global"] = preference_dict
    # print("result", json.dumps(result, indent=4))

    return result


def _build_user_api_keys_rate_limit(config: dict, api_list: list[str]) -> defaultdict:
    user_api_keys_rate_limit = defaultdict(ThreadSafeCircularList)
    for api_index, api_key in enumerate(api_list):
        user_api_keys_rate_limit[api_key] = ThreadSafeCircularList(
            [api_key],
            safe_get(config, "api_keys", api_index, "preferences", "rate_limit", default={"default": "999999/min"}),
            "round_robin",
        )
    return user_api_keys_rate_limit


def _build_admin_api_keys(api_keys_db: list[dict]) -> list[str]:
    admin_api_key = []
    for item in api_keys_db:
        if "admin" in item.get("role", ""):
            admin_api_key.append(item.get("api"))
    if admin_api_key:
        return admin_api_key
    if api_keys_db:
        return [api_keys_db[0].get("api")]

    from utils import yaml_error_message

    if yaml_error_message:
        raise HTTPException(
            status_code=500,
            detail={"error": yaml_error_message},
        )
    raise HTTPException(
        status_code=500,
        detail={"error": "No API key found in api.yaml"},
    )


async def refresh_runtime_state(app: FastAPI) -> None:
    config = getattr(app.state, "config", {}) or {}
    api_keys_db = getattr(app.state, "api_keys_db", []) or []
    api_list = getattr(app.state, "api_list", []) or []

    app.state.user_api_keys_rate_limit = _build_user_api_keys_rate_limit(config, api_list)
    app.state.global_rate_limit = parse_rate_limit(
        safe_get(config, "preferences", "rate_limit", default="999999/min")
    )
    app.state.admin_api_key = _build_admin_api_keys(api_keys_db)
    app.state.provider_timeouts = init_preference(config, "model_timeout", DEFAULT_TIMEOUT)
    app.state.keepalive_interval = init_preference(config, "keepalive_interval", 99999)
    app.state.models_list = build_api_key_models_map(config, api_list)

    if not DISABLE_DATABASE:
        app.state.paid_api_keys_states = {}
        for paid_key in api_list:
            await update_paid_api_keys_states(app, paid_key)


def get_runtime_api_list() -> list[str]:
    runtime_api_list = getattr(app.state, "api_list", None)
    if runtime_api_list:
        return runtime_api_list
    config = getattr(app.state, "config", {}) or {}
    return [item.get("api") for item in config.get("api_keys", []) if item.get("api")]

def get_current_model_prices(model_name: str):
    """
    根据当前配置偏好，返回指定模型的 prompt_price 和 completion_price（单位：$/M tokens）
    """
    try:
        model_price = safe_get(app.state.config, 'preferences', "model_price", default={})
        price_str = next((model_price[k] for k in model_price.keys() if model_name and model_name.startswith(k)), model_price.get("default", "0.3,1"))
        parts = [p.strip() for p in str(price_str).split(",")]
        prompt_price = float(parts[0]) if len(parts) > 0 and parts[0] != "" else 0.3
        completion_price = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 1.0
        return prompt_price, completion_price
    except Exception:
        return 0.3, 1.0

async def compute_total_cost_from_db(filter_api_key: Optional[str] = None, start_dt_obj: Optional[datetime] = None) -> float:
    """
    直接从数据库历史记录累计成本：
    sum((prompt_tokens*prompt_price + completion_tokens*completion_price)/1e6)
    """
    if DISABLE_DATABASE:
        return 0.0
    async with async_session() as session:
        expr = (func.coalesce(RequestStat.prompt_tokens, 0) * func.coalesce(RequestStat.prompt_price, 0.3) + func.coalesce(RequestStat.completion_tokens, 0) * func.coalesce(RequestStat.completion_price, 1.0)) / 1000000.0
        query = select(func.coalesce(func.sum(expr), 0.0))
        if filter_api_key:
            query = query.where(RequestStat.api_key == filter_api_key)
        if start_dt_obj:
            query = query.where(RequestStat.timestamp >= start_dt_obj)
        result = await session.execute(query)
        total_cost = result.scalar_one() or 0.0
        try:
            total_cost = float(total_cost)
        except Exception:
            total_cost = 0.0
        return total_cost

async def update_paid_api_keys_states(app, paid_key):
    """
    更新付费API密钥的状态

    参数:
    app - FastAPI应用实例
    check_index - API密钥在配置中的索引
    paid_key - 需要更新状态的API密钥
    """
    try:
        check_index = app.state.api_list.index(paid_key)
    except Exception:
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid or missing API Key"}
        )
    credits = safe_get(app.state.config, 'api_keys', check_index, "preferences", "credits", default=-1)
    created_at = safe_get(app.state.config, 'api_keys', check_index, "preferences", "created_at", default=datetime.now(timezone.utc) - timedelta(days=30))
    created_at = created_at.astimezone(timezone.utc)

    # 关键修改：总消耗改为从历史数据逐条累计当时价格
    total_cost = await compute_total_cost_from_db(filter_api_key=paid_key, start_dt_obj=created_at)

    if credits != -1:
        # 仍返回聚合的 token 统计，供前端展示
        all_tokens_info = await get_usage_data(filter_api_key=paid_key, start_dt_obj=created_at)

        app.state.paid_api_keys_states[paid_key] = {
            "credits": credits,
            "created_at": created_at,
            "all_tokens_info": all_tokens_info,
            "total_cost": total_cost,
            "enabled": True if total_cost <= credits else False
        }

    return credits, total_cost
        # logger.info(f"app.state.paid_api_keys_states {paid_key}: {json.dumps({k: v.isoformat() if k == 'created_at' else v for k, v in app.state.paid_api_keys_states[paid_key].items()}, indent=4)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    if not DISABLE_DATABASE:
        await create_tables()

    if app and not hasattr(app.state, 'config'):
        # logger.warning("Config not found, attempting to reload")
        app.state.config, app.state.api_keys_db, app.state.api_list = await load_config(app)
        # from ruamel.yaml.timestamp import TimeStamp
        # def json_default(obj):
        #     if isinstance(obj, TimeStamp):
        #         return obj.isoformat()
        #     raise TypeError
        # print("app.state.config", json.dumps(app.state.config, indent=4, ensure_ascii=False, default=json_default))

        await refresh_runtime_state(app)

    if app and not hasattr(app.state, 'client_manager'):

        default_config = {
            "headers": {
                "User-Agent": "curl/7.68.0",
                "Accept": "*/*",
                "Accept-Encoding": "identity",
            },
            "http2": True,
            "verify": True,
            "follow_redirects": True
        }

        # 初始化客户端管理器
        app.state.client_manager = ClientManager(pool_size=100)
        await app.state.client_manager.init(default_config)


    if app and not hasattr(app.state, "channel_manager"):
        if app.state.config and 'preferences' in app.state.config:
            COOLDOWN_PERIOD = app.state.config['preferences'].get('cooldown_period', 300)
        else:
            COOLDOWN_PERIOD = 300

        app.state.channel_manager = ChannelManager(cooldown_period=COOLDOWN_PERIOD)

    if app and not hasattr(app.state, "error_triggers"):
        if app.state.config and 'preferences' in app.state.config:
            ERROR_TRIGGERS = app.state.config['preferences'].get('error_triggers', [])
        else:
            ERROR_TRIGGERS = []
        app.state.error_triggers = ERROR_TRIGGERS

    yield
    # 关闭时的代码
    # await app.state.client.aclose()
    if hasattr(app.state, 'client_manager'):
        await app.state.client_manager.close()

app = FastAPI(lifespan=lifespan, debug=is_debug)

def generate_markdown_docs():
    openapi_schema = app.openapi()

    markdown = f"# {openapi_schema['info']['title']}\n\n"
    markdown += f"Version: {openapi_schema['info']['version']}\n\n"
    markdown += f"{openapi_schema['info'].get('description', '')}\n\n"

    markdown += "## API Endpoints\n\n"

    paths = openapi_schema['paths']
    for path, path_info in paths.items():
        for method, operation in path_info.items():
            markdown += f"### {method.upper()} {path}\n\n"
            markdown += f"{operation.get('summary', '')}\n\n"
            markdown += f"{operation.get('description', '')}\n\n"

            if 'parameters' in operation:
                markdown += "Parameters:\n"
                for param in operation['parameters']:
                    markdown += f"- {param['name']} ({param['in']}): {param.get('description', '')}\n"

            markdown += "\n---\n\n"

    return markdown

@app.get("/docs/markdown")
async def get_markdown_docs():
    markdown = generate_markdown_docs()
    return Response(
        content=markdown,
        media_type="text/markdown"
    )

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     error_messages = []
#     for error in exc.errors():
#         # 将字段路径转换为点分隔格式（例如 body.model -> model）
#         field = ".".join(str(loc) for loc in error["loc"] if loc not in ("body", "query", "path"))
#         error_type = error["type"]

#         # 生成更友好的错误消息
#         if error_type == "value_error.missing":
#             msg = f"字段 '{field}' 是必填项"
#         elif error_type == "type_error.integer":
#             msg = f"字段 '{field}' 必须是整数类型"
#         elif error_type == "type_error.str":
#             msg = f"字段 '{field}' 必须是字符串类型"
#         else:
#             msg = error["msg"]

#         error_messages.append({
#             "field": field,
#             "message": msg,
#             "type": error_type
#         })

#     return JSONResponse(
#         status_code=422,
#         content={"detail": error_messages},
#     )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        token = await get_api_key(request)
        logger.error(f"404 Error: {exc.detail} api_key: {token}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

request_info = contextvars.ContextVar('request_info', default={})

async def parse_request_body(request: Request):
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        try:
            body_bytes = await request.body()
            if not body_bytes:
                return None
            return await asyncio.to_thread(json.loads, body_bytes)
        except json.JSONDecodeError:
            return None
    return None

class ChannelManager:
    def __init__(self, cooldown_period=300):
        self._excluded_models = defaultdict(lambda: None)
        self.cooldown_period = cooldown_period

    async def exclude_model(self, provider: str, model: str):
        model_key = f"{provider}/{model}"
        self._excluded_models[model_key] = datetime.now()

    async def is_model_excluded(self, provider: str, model: str, cooldown_period=0) -> bool:
        model_key = f"{provider}/{model}"
        excluded_time = self._excluded_models[model_key]
        if not excluded_time:
            return False

        if datetime.now() - excluded_time > timedelta(seconds=cooldown_period):
            del self._excluded_models[model_key]
            return False
        return True

    async def get_available_providers(self, providers: list) -> list:
        """过滤出可用的providers，仅排除不可用的模型"""
        available_providers = []
        for provider in providers:
            provider_name = provider['provider']
            model_dict = provider['model'][0]  # 获取唯一的模型字典
            # source_model = list(model_dict.keys())[0]  # 源模型名称
            target_model = list(model_dict.values())[0]  # 目标模型名称
            cooldown_period = provider.get('preferences', {}).get('cooldown_period', self.cooldown_period)

            # 检查该模型是否被排除
            if not await self.is_model_excluded(provider_name, target_model, cooldown_period):
                available_providers.append(provider)

        return available_providers

# 根据数据库类型，动态创建信号量
# SQLite 需要严格的串行写入，而 PostgreSQL 可以处理高并发
if os.getenv("DB_TYPE", "sqlite").lower() == 'sqlite':
    db_semaphore = Semaphore(1)
    logger.info("Database semaphore configured for SQLite (1 concurrent writer).")
else: # For postgres
    # 允许50个并发写入操作，这对于PostgreSQL来说是合理的
    db_semaphore = Semaphore(50)
    logger.info("Database semaphore configured for PostgreSQL (50 concurrent writers).")

async def update_stats(current_info):
    if DISABLE_DATABASE:
        return

    # 在成功请求时，快照当前价格，写入数据库
    try:
        if current_info.get("success") and current_info.get("model"):
            prompt_price, completion_price = get_current_model_prices(current_info["model"])
            current_info["prompt_price"] = prompt_price
            current_info["completion_price"] = completion_price
    except Exception:
        pass

    try:
        # 等待获取数据库访问权限
        async with db_semaphore:
            async with async_session() as session:
                async with session.begin():
                    try:
                        columns = [column.key for column in RequestStat.__table__.columns]
                        filtered_info = {k: v for k, v in current_info.items() if k in columns}

                        # 清洗字符串中的 NUL 字符，防止 PostgreSQL 报错
                        for key, value in filtered_info.items():
                            if isinstance(value, str):
                                filtered_info[key] = value.replace('\x00', '')

                        new_request_stat = RequestStat(**filtered_info)
                        session.add(new_request_stat)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error updating stats: {str(e)}")
                        if is_debug:
                            import traceback
                            traceback.print_exc()

        check_key = current_info["api_key"]
        if check_key and check_key in app.state.paid_api_keys_states and current_info["total_tokens"] > 0:
            await update_paid_api_keys_states(app, check_key)
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()

async def update_channel_stats(request_id, provider, model, api_key, success, provider_api_key: str = None):
    if DISABLE_DATABASE:
        return

    try:
        async with db_semaphore:
            async with async_session() as session:
                async with session.begin():
                    try:
                        channel_stat = ChannelStat(
                            request_id=request_id,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            provider_api_key=provider_api_key,
                            success=success,
                        )
                        session.add(channel_stat)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error updating channel stats: {str(e)}")
                        if is_debug:
                            import traceback
                            traceback.print_exc()
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()

class LoggingStreamingResponse(Response):
    def __init__(self, content, status_code=200, headers=None, media_type=None, current_info=None):
        super().__init__(content=None, status_code=status_code, headers=headers, media_type=media_type)
        self.body_iterator = content
        self._closed = False
        self.current_info = current_info
        self._sse_buffer = ""

        # Remove Content-Length header if it exists
        if 'content-length' in self.headers:
            del self.headers['content-length']
        # Set Transfer-Encoding to chunked
        self.headers['transfer-encoding'] = 'chunked'

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send({
            'type': 'http.response.start',
            'status': self.status_code,
            'headers': self.raw_headers,
        })

        try:
            async for chunk in self._logging_iterator():
                await send({
                    'type': 'http.response.body',
                    'body': chunk,
                    'more_body': True,
                })
        except Exception as e:
            # 记录异常但不重新抛出，避免"Task exception was never retrieved"
            logger.error(f"Error in streaming response: {type(e).__name__}: {str(e)}")
            if is_debug:
                import traceback
                traceback.print_exc()
            # 发送错误消息给客户端（如果可能）
            try:
                error_data = json.dumps({"error": f"Streaming error: {str(e)}"})
                await send({
                    'type': 'http.response.body',
                    'body': f"data: {error_data}\n\n".encode('utf-8'),
                    'more_body': True,
                })
            except Exception as e:
                logger.error(f"Error sending error message: {str(e)}")
        finally:
            await send({
                'type': 'http.response.body',
                'body': b'',
                'more_body': False,
            })
            if hasattr(self.body_iterator, 'aclose') and not self._closed:
                await self.body_iterator.aclose()
                self._closed = True

            process_time = time() - self.current_info["start_time"]
            self.current_info["process_time"] = process_time
            await update_stats(self.current_info)

    async def _logging_iterator(self):
        async for chunk in self.body_iterator:
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            if self.current_info.get("endpoint").endswith("/v1/audio/speech"):
                yield chunk
                continue

            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                yield chunk
                continue

            if is_debug:
                try:
                    logger.info(text.encode("utf-8").decode("unicode_escape"))
                except Exception:
                    logger.info(text)

            # Stream may contain multiple SSE lines per chunk, and/or partial lines.
            self._sse_buffer += text
            while "\n" in self._sse_buffer:
                line, self._sse_buffer = self._sse_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue

                data = None
                if line.startswith("data:"):
                    data = line.removeprefix("data:").lstrip()
                elif line.startswith("{") or line.startswith("["):
                    data = line

                if not data:
                    continue
                if data.startswith("[DONE]") or data.startswith("OK"):
                    continue

                # Avoid parsing every delta event; only parse when usage is present.
                if "\"usage\"" not in data:
                    continue

                try:
                    resp = await asyncio.to_thread(json.loads, data)
                except Exception:
                    continue

                usage_obj = None
                if isinstance(resp, dict):
                    usage_obj = resp.get("usage") or safe_get(resp, "response", "usage", default=None) or safe_get(resp, "message", "usage", default=None)
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
                    total_tokens = int(total_tokens) if total_tokens is not None else (prompt_tokens + completion_tokens)
                except Exception:
                    total_tokens = prompt_tokens + completion_tokens

                self.current_info["prompt_tokens"] = prompt_tokens
                self.current_info["completion_tokens"] = completion_tokens
                self.current_info["total_tokens"] = total_tokens
            yield chunk

    async def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self.body_iterator, 'aclose'):
                await self.body_iterator.aclose()

async def get_api_key(request: Request):
    token = None
    if request.headers.get("x-api-key"):
        token = request.headers.get("x-api-key")
    elif request.headers.get("Authorization"):
        api_split_list = request.headers.get("Authorization").split(" ")
        if len(api_split_list) > 1:
            token = api_split_list[1]
    return token

def get_client_ip(request: Request) -> str:
    """
    获取客户端真实 IP 地址，支持代理场景
    优先级：X-Forwarded-For > X-Real-IP > CF-Connecting-IP > True-Client-IP > request.client.host
    """
    # 1. X-Forwarded-For: 最常用的代理头，格式为 "client, proxy1, proxy2"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 取第一个 IP（真实客户端 IP）
        return forwarded_for.split(",")[0].strip()

    # 2. X-Real-IP: nginx 常用
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 3. CF-Connecting-IP: Cloudflare 使用
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # 4. True-Client-IP: 部分 CDN 使用
    true_client_ip = request.headers.get("True-Client-IP")
    if true_client_ip:
        return true_client_ip.strip()

    # 5. 回退到直连 IP
    return request.client.host if request.client else "unknown"

async def monitor_disconnect(request: Request, disconnect_event: asyncio.Event) -> None:
    try:
        while not disconnect_event.is_set():
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                disconnect_event.set()
                return
    except asyncio.CancelledError:
        return
    except Exception:
        disconnect_event.set()

class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # 如果是 OPTIONS 请求，直接放行，由 CORSMiddleware 处理
        if request.method == "OPTIONS":
            return await call_next(request)

        start_time = time()

        # 根据token决定是否启用道德审查
        token = await get_api_key(request)
        if not token:
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid or missing API Key"}
            )

        enable_moderation = False  # 默认不开启道德审查
        config = app.state.config

        try:
            api_list = app.state.api_list
            api_index = api_list.index(token)
        except ValueError:
            # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
            # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
            api_index = None
            # token不在api_list中，使用默认值（不开启）

        if api_index is not None:
            enable_moderation = safe_get(config, 'api_keys', api_index, "preferences", "ENABLE_MODERATION", default=False)
            if not DISABLE_DATABASE:
                check_api_key = safe_get(config, 'api_keys', api_index, "api")
                # print("check_api_key", check_api_key)
                # logger.info(f"app.state.paid_api_keys_states {check_api_key}: {json.dumps({k: v.isoformat() if k == 'created_at' else v for k, v in app.state.paid_api_keys_states[check_api_key].items()}, indent=4)}")
                # print("app.state.paid_api_keys_states", safe_get(app.state.paid_api_keys_states, check_api_key, "enabled", default=None))
                if safe_get(app.state.paid_api_keys_states, check_api_key, "enabled", default=None) is False and \
                    not request.url.path.startswith("/v1/token_usage"):
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Balance is insufficient, please check your account."}
                    )
        else:
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid or missing API Key"}
            )

        # 在 app.state 中存储此请求的信息
        request_id = str(uuid.uuid4())

        # 初始化请求信息
        request_info_data = {
            "request_id": request_id,
            "start_time": start_time,
            "endpoint": f"{request.method} {request.url.path}",
            "client_ip": get_client_ip(request),
            "process_time": 0,
            "first_response_time": -1,
            "provider": None,
            "model": None,
            "success": False,
            "api_key": token,
            "is_flagged": False,
            "text": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # 设置请求信息到上下文
        current_request_info = request_info.set(request_info_data)
        current_info = request_info.get()
        disconnect_event: Optional[asyncio.Event] = None
        disconnect_task: Optional[asyncio.Task] = None
        try:
            parsed_body = await parse_request_body(request)
            if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
                disconnect_event = asyncio.Event()
                current_info["disconnect_event"] = disconnect_event
                disconnect_task = asyncio.create_task(monitor_disconnect(request, disconnect_event))
            if parsed_body and not request.url.path.startswith("/v1/api_config"):
                request_model = await asyncio.to_thread(UnifiedRequest.model_validate, parsed_body)
                request_model = request_model.data
                if is_debug:
                    logger.info("request_model: %s", json.dumps(request_model.model_dump(exclude_unset=True), indent=2, ensure_ascii=False))
                model = request_model.model
                current_info["model"] = model

                final_api_key = app.state.api_list[api_index]
                try:
                    await app.state.user_api_keys_rate_limit[final_api_key].next(model)
                except Exception:
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Too many requests"}
                    )

                moderated_content = None
                if request_model.request_type == "chat":
                    moderated_content = request_model.get_last_text_message()
                elif request_model.request_type == "image":
                    moderated_content = request_model.prompt
                elif request_model.request_type == "tts":
                    moderated_content = request_model.input
                elif request_model.request_type == "moderation":
                    pass
                elif request_model.request_type == "embedding":
                    if isinstance(request_model.input, list) and len(request_model.input) > 0 and isinstance(request_model.input[0], str):
                        moderated_content = "\n".join(request_model.input)
                    else:
                        moderated_content = request_model.input
                else:
                    logger.error(f"Unknown request type: {request_model.request_type}")

                if enable_moderation and moderated_content:
                    background_tasks_for_moderation = BackgroundTasks()
                    moderation_response = await self.moderate_content(moderated_content, api_index, background_tasks_for_moderation)
                    is_flagged = moderation_response.get('results', [{}])[0].get('flagged', False)

                    if is_flagged:
                        logger.error(f"Content did not pass the moral check: {moderated_content}")
                        process_time = time() - start_time
                        current_info["process_time"] = process_time
                        current_info["is_flagged"] = is_flagged
                        current_info["text"] = moderated_content  # 仅在标记时记录文本
                        await update_stats(current_info)
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Content did not pass the moral check, please modify and try again."}
                        )

            response = await call_next(request)

            if request.url.path.startswith("/v1") and not DISABLE_DATABASE:
                if isinstance(response, (FastAPIStreamingResponse, StarletteStreamingResponse)) or type(response).__name__ == '_StreamingResponse':
                    response = LoggingStreamingResponse(
                        content=response.body_iterator,
                        status_code=response.status_code,
                        media_type=response.media_type,
                        headers=response.headers,
                        current_info=current_info,
                    )
                elif hasattr(response, 'json'):
                    logger.info(f"Response: {await response.json()}")
                else:
                    logger.info(f"Response: type={type(response).__name__}, status_code={response.status_code}, headers={response.headers}")

            return response

        except HTTPException:
            # Let FastAPI's http_exception_handler format the response consistently.
            raise
        except ValidationError as e:
            logger.error(f"API key: {token}, Invalid request body: {json.dumps(parsed_body, indent=2, ensure_ascii=False)}, errors: {e.errors()}")
            content = await asyncio.to_thread(jsonable_encoder, {"detail": e.errors()})
            return JSONResponse(
                status_code=422,
                content=content
            )
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()
            logger.error(f"Error processing request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"}
            )

        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            # print("current_request_info", current_request_info)
            request_info.reset(current_request_info)

    async def moderate_content(self, content, api_index, background_tasks: BackgroundTasks):
        moderation_request = ModerationRequest(input=content)

        # 直接调用 moderations 函数
        response = await moderations(moderation_request, background_tasks, api_index)

        # 读取流式响应的内容
        moderation_result = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                moderation_result += chunk.encode('utf-8')
            else:
                moderation_result += chunk

        # 解码并解析 JSON
        moderation_data = json.loads(moderation_result.decode('utf-8'))

        return moderation_data

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部字段
)

app.add_middleware(StatsMiddleware)

@app.middleware("http")
async def ensure_config(request: Request, call_next):
    if app and app.state.api_keys_db and not hasattr(app.state, "models_list"):
        app.state.models_list = build_api_key_models_map(app.state.config, app.state.api_list)
    return await call_next(request)

class ClientManager:
    def __init__(self, pool_size=100):
        self.pool_size = pool_size
        self.clients = {}  # {host_timeout_proxy: AsyncClient}
        self._client_locks = defaultdict(asyncio.Lock)

    async def init(self, default_config):
        self.default_config = default_config

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2: Optional[bool] = None):
        # 从base_url中提取主机名
        parsed_url = urlparse(base_url)
        host = parsed_url.netloc

        # 创建唯一的客户端键
        client_key = f"{host}"
        if proxy:
            # 对代理URL进行规范化处理
            proxy_normalized = proxy.replace('socks5h://', 'socks5://')
            client_key += f"_{proxy_normalized}"
        if http2 is not None:
            client_key += f"_http2_{int(bool(http2))}"

        if client_key not in self.clients:
            async with self._client_locks[client_key]:
                if client_key not in self.clients:
                    timeout = httpx.Timeout(
                        connect=15.0,
                        read=None,
                        write=30.0,
                        pool=self.pool_size,
                    )
                    limits = httpx.Limits(max_connections=self.pool_size)

                    client_config = {
                        **self.default_config,
                        "timeout": timeout,
                        "limits": limits,
                    }

                    client_config = get_proxy(proxy, client_config)
                    if http2 is not None:
                        client_config["http2"] = bool(http2)

                    self.clients[client_key] = httpx.AsyncClient(**client_config)

        try:
            yield self.clients[client_key]
        except Exception as e:
            # if client_key in self.clients and "429" not in str(e):
            #     tmp_client = self.clients[client_key]
            #     del self.clients[client_key]  # 先删除引用
            #     await tmp_client.aclose()  # 然后关闭客户端
            # 仅在客户端主动关闭等严重错误时才考虑重建，暂时只将异常抛出
            # httpx的连接池会自动处理单个连接的失败
            # logger.warning(f"Exception with client {client_key}: {type(e).__name__}: {e}")
            raise e

    async def close(self):
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

rate_limiter = InMemoryRateLimiter()

async def rate_limit_dependency():
    if await rate_limiter.is_rate_limited("global", app.state.global_rate_limit):
        raise HTTPException(status_code=429, detail="Too many requests")

def get_preference_value(provider_timeouts, original_model):
    timeout_value = None
    original_model = original_model.lower()
    if original_model in provider_timeouts:
        timeout_value = provider_timeouts[original_model]
    else:
        # 尝试模糊匹配模型
        for timeout_model in provider_timeouts:
            if timeout_model != "default" and timeout_model.lower() in original_model.lower():
                timeout_value = provider_timeouts[timeout_model]
                break
        else:
            # 如果模糊匹配失败，使用渠道的默认值
            timeout_value = provider_timeouts.get("default", None)
    return timeout_value

def get_preference(preference_config, channel_id, original_request_model, default_value):
    original_model, request_model_name = original_request_model
    provider_timeouts = safe_get(preference_config, channel_id, default=preference_config["global"])
    timeout_value = get_preference_value(provider_timeouts, request_model_name)
    if timeout_value is None:
        timeout_value = get_preference_value(provider_timeouts, original_model)
    if timeout_value is None:
        timeout_value = get_preference_value(preference_config["global"], request_model_name)
    if timeout_value is None:
        timeout_value = get_preference_value(preference_config["global"], original_model)
    if timeout_value is None:
        timeout_value = preference_config["global"].get("default", default_value)
    # print("timeout_value", channel_id, timeout_value)
    return timeout_value

def _split_codex_api_key(raw_api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if raw_api_key is None:
        return None, None
    raw = str(raw_api_key).strip()
    if not raw:
        return None, None
    if "," not in raw:
        return None, raw
    account_id, token = raw.split(",", 1)
    account_id = account_id.strip() or None
    token = token.strip()
    if not token:
        raise ValueError("Invalid Codex API key format: expected 'account_id,refresh_token' (refresh_token missing)")
    return account_id, token

_CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CODEX_OAUTH_REFRESH_SKEW_SECONDS = 30

# provider_api_key_raw -> {"access_token": str, "refresh_token": str, "expires_at": float|None}
_codex_oauth_cache: dict[str, dict[str, Any]] = {}
_codex_oauth_locks: dict[str, asyncio.Lock] = {}

# provider_api_key_raw -> refresh_token
# NOTE: We intentionally key by the full raw config string (usually "account_id,refresh_token") so multiple
# Codex keys sharing the same account_id but having different refresh tokens won't overwrite each other.
_CODEX_REFRESH_TOKEN_STORE_PATH = os.getenv("CODEX_REFRESH_TOKEN_STORE_PATH", "./data/codex_refresh_tokens.json")
_codex_refresh_token_store: dict[str, str] = {}
_codex_refresh_token_store_loaded = False
_codex_refresh_token_store_lock = asyncio.Lock()

async def _ensure_codex_refresh_token_store_loaded() -> None:
    global _codex_refresh_token_store_loaded
    if _codex_refresh_token_store_loaded:
        return
    async with _codex_refresh_token_store_lock:
        if _codex_refresh_token_store_loaded:
            return
        try:
            with open(_CODEX_REFRESH_TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    key = str(k).strip()
                    val = str(v).strip()
                    if key and val:
                        _codex_refresh_token_store[key] = val
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to load Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)
        _codex_refresh_token_store_loaded = True

async def _reload_codex_refresh_token_store() -> None:
    global _codex_refresh_token_store_loaded
    async with _codex_refresh_token_store_lock:
        _codex_refresh_token_store.clear()
        try:
            with open(_CODEX_REFRESH_TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    key = str(k).strip()
                    val = str(v).strip()
                    if key and val:
                        _codex_refresh_token_store[key] = val
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to reload Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)
        _codex_refresh_token_store_loaded = True

async def _get_codex_refresh_token_from_store(provider_api_key_raw: Optional[str], *, force_reload: bool = False) -> Optional[str]:
    if provider_api_key_raw is None:
        return None
    key = str(provider_api_key_raw).strip()
    if not key:
        return None
    if force_reload:
        await _reload_codex_refresh_token_store()
    else:
        await _ensure_codex_refresh_token_store_loaded()
    token = _codex_refresh_token_store.get(key)
    return str(token) if token else None

async def _persist_codex_refresh_token(provider_api_key_raw: Optional[str], refresh_token: Optional[str]) -> None:
    if provider_api_key_raw is None:
        return
    key = str(provider_api_key_raw).strip()
    rt = str(refresh_token or "").strip()
    if not key or not rt:
        return
    await _ensure_codex_refresh_token_store_loaded()

    async with _codex_refresh_token_store_lock:
        if _codex_refresh_token_store.get(key) == rt:
            return
        _codex_refresh_token_store[key] = rt
        try:
            store_dir = os.path.dirname(_CODEX_REFRESH_TOKEN_STORE_PATH)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            tmp_path = f"{_CODEX_REFRESH_TOKEN_STORE_PATH}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(_codex_refresh_token_store, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _CODEX_REFRESH_TOKEN_STORE_PATH)
        except Exception as e:
            logger.warning("Failed to persist Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)

def _codex_oauth_lock(key: str) -> asyncio.Lock:
    lock = _codex_oauth_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _codex_oauth_locks[key] = lock
    return lock

def _codex_access_token_is_valid(entry: dict[str, Any]) -> bool:
    token = entry.get("access_token")
    if not token:
        return False
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return True
    try:
        return time() < float(expires_at) - _CODEX_OAUTH_REFRESH_SKEW_SECONDS
    except Exception:
        return True

async def _refresh_codex_access_token(refresh_token: str, proxy: Optional[str]) -> dict[str, Any]:
    rt = (refresh_token or "").strip()
    if not rt:
        raise HTTPException(status_code=401, detail="Codex refresh_token missing")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "client_id": _CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "scope": "openid profile email",
    }

    try:
        async with app.state.client_manager.get_client(_CODEX_OAUTH_TOKEN_URL, proxy) as client:
            resp = await client.post(_CODEX_OAUTH_TOKEN_URL, data=data, headers=headers, timeout=30)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Codex token refresh request failed: {type(e).__name__}: {e}")

    if resp.status_code != 200:
        body = (resp.text or "").strip()
        raise HTTPException(status_code=401, detail=f"Codex token refresh failed: status {resp.status_code}: {body}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise HTTPException(status_code=401, detail=f"Codex token refresh returned empty access_token: {resp.text}")

    new_refresh_token = str(payload.get("refresh_token") or "").strip() or None
    expires_in = payload.get("expires_in")
    expires_at = None
    try:
        expires_in_int = int(expires_in)
        if expires_in_int > 0:
            expires_at = time() + expires_in_int
    except Exception:
        expires_at = None

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "expires_at": expires_at,
    }

async def _get_codex_access_token(provider_name: str, provider_api_key_raw: str, proxy: Optional[str]) -> str:
    # provider_api_key_raw is the stable key-id we use for rate-limit/cooling/logging.
    account_id, refresh_token_from_config = _split_codex_api_key(provider_api_key_raw)
    if not refresh_token_from_config:
        raise HTTPException(status_code=401, detail="Codex refresh_token missing")

    persisted_refresh_token = await _get_codex_refresh_token_from_store(provider_api_key_raw)
    if persisted_refresh_token:
        refresh_token_from_config = persisted_refresh_token

    lock = _codex_oauth_lock(provider_api_key_raw)
    async with lock:
        entry = _codex_oauth_cache.get(provider_api_key_raw) or {}
        if _codex_access_token_is_valid(entry):
            return str(entry["access_token"])

        old_refresh_token = str(entry.get("refresh_token") or refresh_token_from_config).strip()
        try:
            refreshed = await _refresh_codex_access_token(old_refresh_token, proxy)
        except HTTPException as e:
            detail = str(getattr(e, "detail", "") or "")
            if "refresh_token_reused" in detail:
                latest = await _get_codex_refresh_token_from_store(provider_api_key_raw, force_reload=True)
                if latest and latest != old_refresh_token:
                    refreshed = await _refresh_codex_access_token(latest, proxy)
                    old_refresh_token = latest
                else:
                    raise
            raise

        updated_refresh_token = refreshed.get("refresh_token") or old_refresh_token
        _codex_oauth_cache[provider_api_key_raw] = {
            "access_token": refreshed["access_token"],
            "refresh_token": updated_refresh_token,
            "expires_at": refreshed.get("expires_at"),
        }
        await _persist_codex_refresh_token(provider_api_key_raw, updated_refresh_token)
        return str(refreshed["access_token"])


async def _resolve_codex_upstream_auth(
    provider_name: str,
    provider_api_key_raw: Optional[str],
    proxy: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if provider_api_key_raw is None:
        return None, None

    raw = str(provider_api_key_raw).strip()
    if not raw:
        return None, None

    # Support direct Codex-compatible proxies that only need a fixed Bearer token.
    if "," not in raw:
        return raw, None

    codex_account_id, _ = _split_codex_api_key(raw)
    api_key = await _get_codex_access_token(provider_name, raw, proxy)
    return api_key, codex_account_id

# 在 process_request 函数中更新成功和失败计数
async def process_request(
    request: Union[RequestModel, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest],
    provider: Dict,
    background_tasks: BackgroundTasks,
    endpoint=None,
    role=None,
    timeout_value=DEFAULT_TIMEOUT,
    keepalive_interval=None,
    provider_api_key_raw: Optional[str] = None,
):
    timeout_value = int(timeout_value)
    model_dict = provider["_model_dict_cache"]
    original_model = model_dict[request.model]
    if provider_api_key_raw is None:
        provider_api_key_raw = await select_provider_api_key_raw(
            provider,
            original_model,
            get_runtime_api_list(),
        )

    engine, stream_mode = get_engine(provider, endpoint, original_model)

    if stream_mode is not None:
        request.stream = stream_mode

    proxy = safe_get(app.state.config, "preferences", "proxy", default=None)  # global proxy
    proxy = safe_get(provider, "preferences", "proxy", default=proxy)  # provider proxy

    api_key = provider_api_key_raw
    codex_account_id = None
    if engine == "codex" and provider_api_key_raw:
        try:
            api_key, codex_account_id = await _resolve_codex_upstream_auth(
                provider["provider"],
                provider_api_key_raw,
                proxy,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Gemini preview TTS returns inline audio; force non-stream so we can return a single OpenAI-style JSON response.
    try:
        has_audio_modality = any(str(m).lower() == "audio" for m in (getattr(request, "modalities", None) or []))
    except Exception:
        has_audio_modality = False
    if engine in ["gemini", "vertex-gemini"] and (has_audio_modality or "preview-tts" in original_model.lower()):
        request.stream = False

    channel_id = f"{provider['provider']}"
    if engine != "moderation":
        logger.info(f"provider: {channel_id[:11]:<11} model: {request.model:<22} engine: {engine[:13]:<13} role: {role}")

    last_message_role = safe_get(request, "messages", -1, "role", default=None)
    url, headers, payload = await get_payload(request, engine, provider, api_key, endpoint=endpoint)
    if engine == "codex" and codex_account_id:
        headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))
    headers.update(safe_get(provider, "preferences", "headers", default={}))  # add custom headers
    if engine == "codex":
        force_codex_client_headers(headers)
    if is_debug:
        logger.info(url)
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        logger.info(json.dumps({k: v for k, v in payload.items() if k != 'file'}, indent=4, ensure_ascii=False))

    current_info = request_info.get()
    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(url, proxy, http2=False if engine == "codex" else None) as client:
            downstream_stream = bool(getattr(request, "stream", None))
            force_collect_codex_stream = engine == "codex" and not downstream_stream and endpoint is None

            if downstream_stream and not force_collect_codex_stream:
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model, timeout_value)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            elif force_collect_codex_stream:
                payload["stream"] = True
                headers["Accept"] = "text/event-stream"
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model, timeout_value)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                json_data = await collect_openai_chat_completion_from_streaming_sse(wrapped_generator, model=original_model)
                response = StarletteStreamingResponse(iter([json_data]), media_type="application/json")
            else:
                generator = fetch_response(client, url, headers, payload, engine, original_model, timeout_value)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, False, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)

                # 处理音频和其他二进制响应
                if endpoint == "/v1/audio/speech":
                    if isinstance(wrapped_generator, bytes):
                        response = Response(content=wrapped_generator, media_type="audio/mpeg")
                else:
                    first_element = await anext(wrapped_generator)
                    first_element = first_element.lstrip("data: ")
                    decoded_element = await asyncio.to_thread(json.loads, first_element)

                    # 提取 usage 信息到 current_info
                    usage_obj = decoded_element.get("usage") or {}
                    current_info["prompt_tokens"] = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
                    current_info["completion_tokens"] = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
                    current_info["total_tokens"] = usage_obj.get("total_tokens") or 0

                    encoded_element = await asyncio.to_thread(json.dumps, decoded_element)
                    response = StarletteStreamingResponse(iter([encoded_element]), media_type="application/json")

            # 更新成功计数和首次响应时间
            background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=True, provider_api_key=provider_api_key_raw)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = channel_id
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.LocalProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=False, provider_api_key=provider_api_key_raw)
        raise e

class ModelRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_model(
        self,
        request_data: Union[RequestModel, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest],
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint=None,
    ):
        config = app.state.config
        request_model_name = request_data.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_total_tokens = estimate_request_total_tokens(request_data)
        routing_endpoint = endpoint or "/v1/chat/completions"
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=routing_endpoint,
            request_total_tokens=request_total_tokens,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        exclude_error_rate_limit = [
            "BrokenResourceError",
            "Proxy connection timed out",
            "Unknown error: EndOfStream",
            "'status': 'INVALID_ARGUMENT'",
            "Unable to connect to service",
            "Connection closed unexpectedly",
            "Invalid JSON payload received. Unknown name ",
            "User location is not supported for the API use",
            "The model is overloaded. Please try again later.",
            "[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure (_ssl.c:1007)",
            "<title>Worker exceeded resource limits",
        ]
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            clear_provider_auth_cache=lambda provider_api_key_raw: _codex_oauth_cache.pop(provider_api_key_raw, None),
        )

        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                return Response(content="", status_code=499)
            return None

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model

            original_request_model = (original_model, request_data.model)
            local_api_list = get_runtime_api_list()
            if provider_name.startswith("sk-") and provider_name in local_api_list:
                local_provider_api_index = local_api_list.index(provider_name)
                local_provider_scheduling_algorithm = safe_get(
                    config,
                    "api_keys",
                    local_provider_api_index,
                    "preferences",
                    "SCHEDULING_ALGORITHM",
                    default="fixed_priority",
                )
                local_provider_matching_providers = await get_right_order_providers(
                    request_model_name,
                    config,
                    local_provider_api_index,
                    local_provider_scheduling_algorithm,
                    local_api_list,
                    app.state.models_list,
                    endpoint=routing_endpoint,
                    channel_manager=app.state.channel_manager,
                    request_total_tokens=request_total_tokens,
                    debug=is_debug,
                )
                local_timeout_value = 0
                for local_provider in local_provider_matching_providers:
                    local_provider_name = local_provider["provider"]
                    if not local_provider_name.startswith("sk-"):
                        original_request_model = (
                            local_provider["_model_dict_cache"][request_model_name],
                            request_data.model,
                        )
                        local_timeout_value += get_preference(
                            app.state.provider_timeouts,
                            local_provider_name,
                            original_request_model,
                            DEFAULT_TIMEOUT,
                        )
                local_provider_num_matching_providers = len(local_provider_matching_providers)
            else:
                local_timeout_value = get_preference(
                    app.state.provider_timeouts,
                    provider_name,
                    original_request_model,
                    DEFAULT_TIMEOUT,
                )
                local_provider_num_matching_providers = 1

            local_timeout_value = local_timeout_value * local_provider_num_matching_providers
            keepalive_interval = get_preference(
                app.state.keepalive_interval,
                provider_name,
                original_request_model,
                99999,
            )
            if keepalive_interval > local_timeout_value or provider_name.startswith("sk-"):
                keepalive_interval = None

            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)
            process_task = asyncio.create_task(
                process_request(
                    request_data,
                    provider,
                    background_tasks,
                    endpoint,
                    plan.role,
                    local_timeout_value,
                    keepalive_interval,
                    provider_api_key_raw=attempt.provider_api_key_raw,
                )
            )
            disconnect_task: Optional[asyncio.Task] = None
            try:
                if disconnect_event is not None:
                    disconnect_task = asyncio.create_task(disconnect_event.wait())
                    done, pending = await asyncio.wait(
                        [process_task, disconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnect_task in done and disconnect_event.is_set():
                        process_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await process_task
                        return Response(content="", status_code=499)

                return await process_task
            except asyncio.CancelledError:
                raise
            except Exception:
                if disconnect_event is not None and disconnect_event.is_set():
                    return Response(content="", status_code=499)
                raise
            finally:
                if disconnect_task is not None:
                    disconnect_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await disconnect_task

        def after_failure(attempt, exc, status_code, error_message):
            _ = exc
            request_model, actual_model = _log_model_names(request_data.model, attempt.original_model)
            logger.error(
                "Error %s with provider %s request_model=%s actual_model=%s API key: %s: %s",
                status_code,
                attempt.provider_name,
                request_model,
                actual_model,
                attempt.provider_api_key_raw,
                error_message,
            )
            if is_debug or status_code == 500:
                import traceback

                traceback.print_exc()

        def build_final_response(completed_plan):
            current_info = request_info.get()
            if isinstance(current_info, dict):
                current_info["first_response_time"] = -1
                current_info["success"] = False
                current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_data.model} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            before_next_attempt=before_next_attempt,
            after_failure=after_failure,
            build_final_response=build_final_response,
            exclude_error_substrings=exclude_error_rate_limit,
            rollback_rate_limit_errors=exclude_error_rate_limit,
            allow_channel_exclusion=True,
        )

def _normalize_responses_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    if engine != "codex":
        return base
    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return base
    return f"{base}/responses"

def _normalize_responses_compact_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")

    if base.endswith("/v1/responses/compact") or base.endswith("/responses/compact"):
        return base

    if engine == "codex":
        base = _normalize_responses_upstream_url(base, engine)

    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return f"{base}/compact"

    if base.endswith("/compact"):
        return base

    return f"{base}/compact"

def _log_model_names(request_model_name: Any, actual_model_name: Any = None) -> tuple[str, str]:
    request_model = str(request_model_name or "-")
    actual_model = str(actual_model_name or request_model)
    return request_model, actual_model

def _responses_request_id(current_info: Any) -> str:
    if isinstance(current_info, dict):
        request_id = current_info.get("request_id")
        if request_id:
            return str(request_id)
    return "-"

def _log_responses_downstream_disconnect(
    endpoint: str,
    current_info: Any,
    *,
    model_id: str,
    provider_name: Optional[str] = None,
    stage: str,
) -> None:
    trace_logger.info(
        "%s downstream disconnect stage=%s request_id=%s model=%s provider=%s",
        endpoint,
        stage,
        _responses_request_id(current_info),
        model_id,
        provider_name or "-",
    )

RESPONSES_STREAM_NETWORK_ERRORS = UPSTREAM_NETWORK_ERRORS

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

def _extract_responses_stream_event(raw_event: str) -> tuple[str, Any]:
    event_name = ""
    data_lines = []
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

def _responses_usage_from_payload(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    usage = safe_get(payload, "response", "usage", default=None)
    if not isinstance(usage, dict):
        usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None

def _responses_part_has_text(part: Any) -> bool:
    if not isinstance(part, dict):
        return False

    text = part.get("text")
    if isinstance(text, str) and text:
        return True

    refusal = part.get("refusal")
    return isinstance(refusal, str) and bool(refusal)

def _responses_item_has_substantive_output(item: Any) -> bool:
    if not isinstance(item, dict):
        return False

    content = item.get("content")
    if isinstance(content, list) and any(_responses_part_has_text(part) for part in content):
        return True

    item_type = str(item.get("type") or "")
    if item_type in {"function_call", "tool_call"}:
        return bool(item.get("name") or item.get("arguments") or item.get("call_id"))

    return False

def _responses_stream_event_has_real_output(event_type: str, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    if event_type.startswith("response.") and event_type.endswith(".delta"):
        return bool(str(payload.get("delta") or ""))

    if event_type in {"response.content_part.added", "response.content_part.done"}:
        return _responses_part_has_text(payload.get("part"))

    if event_type == "response.output_item.done":
        return _responses_item_has_substantive_output(payload.get("item"))

    if event_type.startswith("response.") and event_type.endswith(".done"):
        return bool(str(payload.get("text") or payload.get("refusal") or payload.get("arguments") or ""))

    return False

def _responses_stream_event_commits(event_type: str, payload: Any, commit_policy: str) -> bool:
    completed_with_usage = event_type == "response.completed" and _responses_usage_from_payload(payload) is not None
    if commit_policy == "completed_usage":
        return completed_with_usage
    return completed_with_usage or _responses_stream_event_has_real_output(event_type, payload)

def _responses_error_status_code(error_obj: Any) -> int:
    if isinstance(error_obj, dict):
        raw_status = error_obj.get("status_code") or error_obj.get("status")
        try:
            status_code = int(raw_status)
        except (TypeError, ValueError):
            status_code = None
        if status_code is not None and 100 <= status_code <= 599:
            return status_code

        error_code = str(error_obj.get("code") or "").strip().lower()
        if error_code in RESPONSES_FAILURE_STATUS_BY_CODE:
            return RESPONSES_FAILURE_STATUS_BY_CODE[error_code]

        error_type = str(error_obj.get("type") or "").strip().lower()
        if error_type in RESPONSES_FAILURE_STATUS_BY_TYPE:
            return RESPONSES_FAILURE_STATUS_BY_TYPE[error_type]

        message = str(error_obj.get("message") or "").lower()
        if "rate limit" in message or "too many requests" in message:
            return 429
        if "invalid" in message or "unsupported" in message:
            return 400
        if "not found" in message:
            return 404
        if "permission" in message or "forbidden" in message:
            return 403
        if "auth" in message or "api key" in message or "unauthorized" in message:
            return 401

    return 500

def _responses_failure_http_exception(payload: Any) -> Optional[HTTPException]:
    if not isinstance(payload, dict):
        return None

    error_obj = None
    response_status = str(safe_get(payload, "response", "status", default="") or "").strip().lower()
    payload_status = str(payload.get("status") or "").strip().lower()
    payload_type = str(payload.get("type") or "").strip().lower()

    if payload_type == "error" and payload.get("error") is not None:
        error_obj = payload.get("error")
    elif payload_type == "response.failed":
        error_obj = safe_get(payload, "response", "error", default=None)
    elif payload_status == "failed":
        error_obj = payload.get("error")
    elif response_status == "failed":
        error_obj = safe_get(payload, "response", "error", default=None)
    elif isinstance(payload.get("error"), dict):
        error_obj = payload.get("error")

    if error_obj is None and (payload_status == "failed" or response_status == "failed"):
        error_obj = {"message": "Responses upstream returned status=failed"}

    if error_obj is None:
        return None

    error_body = {"error": error_obj}
    return HTTPException(
        status_code=_responses_error_status_code(error_obj),
        detail=json.dumps(error_body, ensure_ascii=False),
    )

async def _prime_responses_upstream_stream(
    upstream_iter,
    *,
    disconnect_event: Optional[asyncio.Event] = None,
    commit_policy: str = "real_output",
) -> tuple[list[bytes], bool]:
    """
    Buffer structural Responses events so we can still fail over before a
    substantive output event or a completed response with usage is sent.
    """
    buffered_chunks: list[bytes] = []
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text_buffer = ""
    commit_policy = (commit_policy or "real_output").strip().lower()
    if commit_policy not in {"real_output", "completed_usage"}:
        commit_policy = "real_output"

    while True:
        if disconnect_event is not None and disconnect_event.is_set():
            return buffered_chunks, False

        try:
            chunk = await upstream_iter.__anext__()
        except StopAsyncIteration:
            if not buffered_chunks:
                raise HTTPException(status_code=502, detail="Upstream closed stream without data")
            if text_buffer.strip():
                raise HTTPException(status_code=502, detail="Upstream closed stream with an incomplete SSE event")
            raise HTTPException(status_code=502, detail="Responses upstream closed before substantive output")

        buffered_chunks.append(chunk)
        text_buffer += decoder.decode(chunk)

        while True:
            match = re.search(r"\r?\n\r?\n", text_buffer)
            if not match:
                break

            raw_event = text_buffer[:match.start()]
            text_buffer = text_buffer[match.end():]
            if not raw_event.strip():
                continue

            event_type, event_payload = _extract_responses_stream_event(raw_event)
            if event_type == "[DONE]":
                raise HTTPException(
                    status_code=502,
                    detail="Responses upstream ended before substantive output",
                )

            semantic_failure = _responses_failure_http_exception(event_payload)
            if semantic_failure is not None:
                raise semantic_failure

            if _responses_stream_event_commits(event_type, event_payload, commit_policy):
                return buffered_chunks, True

            continue

class ResponsesRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_responses(
        self,
        http_request: Request,
        request_data: ResponsesRequest,
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint: str = "/v1/responses",
    ):
        config = app.state.config
        request_model_name = request_data.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            clear_provider_auth_cache=lambda provider_api_key_raw: _codex_oauth_cache.pop(provider_api_key_raw, None),
        )

        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                _log_responses_downstream_disconnect(
                    endpoint,
                    current_info,
                    model_id=request_model_name,
                    stage="before-provider-select",
                )
                return Response(content="", status_code=499)
            return None

        async def prepare_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine, stream_mode = get_engine(provider, endpoint=endpoint, original_model=original_model)
            if stream_mode is not None:
                request_data.stream = stream_mode

            attempt.state["failure_stage"] = "validation"
            if engine not in ("gpt", "codex"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} only supports upstream engines: gpt/codex (got {engine})",
                )

            wants_compact = endpoint.rstrip("/").endswith("/compact")
            if wants_compact:
                upstream_url = _normalize_responses_compact_upstream_url(provider.get("base_url", ""), engine)
            else:
                upstream_url = _normalize_responses_upstream_url(provider.get("base_url", ""), engine)

            if engine == "gpt" and "v1/responses" not in upstream_url:
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} requires provider base_url ending with /v1/responses (got {upstream_url})",
                )
            if wants_compact and "compact" not in upstream_url:
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} requires provider base_url ending with /v1/responses/compact (got {upstream_url})",
                )

            proxy = safe_get(config, "preferences", "proxy", default=None)
            proxy = safe_get(provider, "preferences", "proxy", default=proxy)
            channel_id = f"{provider_name}"
            commit_policy = safe_get(
                provider,
                "preferences",
                "responses_stream_commit_policy",
                default="real_output",
            )
            attempt.state["upstream_url"] = upstream_url
            attempt.state["channel_id"] = channel_id
            attempt.state["engine"] = engine
            attempt.state["responses_stream_commit_policy"] = str(commit_policy or "real_output")
            attempt.state["failure_stage"] = "auth"

            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)
            api_key = attempt.provider_api_key_raw
            codex_account_id = None
            if engine == "codex" and attempt.provider_api_key_raw:
                api_key, codex_account_id = await _resolve_codex_upstream_auth(
                    provider_name,
                    attempt.provider_api_key_raw,
                    proxy,
                )

            timeout_value = get_preference(
                app.state.provider_timeouts,
                provider_name,
                (original_model, request_model_name),
                DEFAULT_TIMEOUT,
            )
            attempt.state["proxy"] = proxy
            attempt.state["api_key"] = api_key
            attempt.state["codex_account_id"] = codex_account_id
            attempt.state["wants_compact"] = wants_compact
            attempt.state["timeout_value"] = int(timeout_value)

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine = attempt.state["engine"]
            proxy = attempt.state["proxy"]
            api_key = attempt.state["api_key"]
            codex_account_id = attempt.state["codex_account_id"]
            wants_compact = attempt.state["wants_compact"]
            timeout_value = attempt.state["timeout_value"]
            upstream_url = attempt.state["upstream_url"]
            channel_id = attempt.state["channel_id"]
            commit_policy = attempt.state.get("responses_stream_commit_policy", "real_output")

            headers = {
                "Content-Type": "application/json",
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            if engine == "codex":
                headers.setdefault("Openai-Beta", http_request.headers.get("Openai-Beta") or "responses=experimental")
                headers.setdefault("Originator", http_request.headers.get("Originator") or "codex_cli_rs")
                headers.setdefault("Version", CODEX_CLI_VERSION)
                headers.setdefault("Session_id", http_request.headers.get("Session_id") or str(uuid.uuid4()))
                headers.setdefault("User-Agent", CODEX_USER_AGENT)
                headers.setdefault("Accept", "text/event-stream" if request_data.stream else "application/json")
                if codex_account_id:
                    headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))

            headers.update(safe_get(provider, "preferences", "headers", default={}) or {})
            if engine == "codex":
                force_codex_client_headers(headers)

            payload = request_data.model_dump(exclude_unset=True)
            payload["model"] = original_model
            if engine == "codex":
                payload.pop("previous_response_id", None)
                payload.pop("prompt_cache_retention", None)
                payload.pop("safety_identifier", None)
                payload.setdefault("instructions", "")

            apply_post_body_parameter_overrides(
                payload,
                provider,
                request_model_name,
                skip_keys={"translation_options"},
            )

            if engine == "codex":
                strip_unsupported_codex_payload_fields(payload, strip_store=wants_compact)

            logger.info(
                "provider: %-11s model: %-22s engine: %-13s role: %s",
                channel_id[:11],
                request_model_name,
                engine[:13],
                plan.role,
            )
            trace_logger.info(
                "endpoint=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
                endpoint,
                request_id,
                channel_id[:11],
                request_model_name,
                engine[:13],
                plan.role,
                upstream_url,
            )

            attempt.state["failure_stage"] = "upstream"
            attempt.state["track_channel_stats"] = True
            async with app.state.client_manager.get_client(upstream_url, proxy, http2=False if engine == "codex" else None) as client:
                json_payload = await asyncio.to_thread(json.dumps, payload)
                # json_payload = await asyncio.to_thread(json.dumps, payload, ensure_ascii=False)
                # if wants_compact:
                #     print("request /v1/responses/compact:", json_payload)
                if request_data.stream:
                    stream_cm = client.stream("POST", upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                    upstream_resp = await stream_cm.__aenter__()
                    if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                        raw = await upstream_resp.aread()
                        await stream_cm.__aexit__(None, None, None)
                        try:
                            error_message = raw.decode("utf-8", errors="replace")
                        except Exception:
                            error_message = str(raw)
                        raise HTTPException(status_code=upstream_resp.status_code, detail=error_message)

                    upstream_iter = upstream_resp.aiter_raw()
                    try:
                        buffered_chunks, stream_committed = await _prime_responses_upstream_stream(
                            upstream_iter,
                            disconnect_event=disconnect_event,
                            commit_policy=commit_policy,
                        )
                    except HTTPException:
                        await stream_cm.__aexit__(None, None, None)
                        raise
                    except RESPONSES_STREAM_NETWORK_ERRORS:
                        await stream_cm.__aexit__(None, None, None)
                        raise

                    if disconnect_event is not None and disconnect_event.is_set():
                        await stream_cm.__aexit__(None, None, None)
                        _log_responses_downstream_disconnect(
                            endpoint,
                            current_info,
                            model_id=request_model_name,
                            provider_name=provider_name,
                            stage="before-stream-commit",
                        )
                        return Response(content="", status_code=499)

                    async def proxy_stream():
                        completed_seen = False
                        usage_seen = False
                        output_seen = False
                        proxy_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                        proxy_text_buffer = ""

                        def track_responses_events(chunk: bytes) -> None:
                            nonlocal completed_seen, usage_seen, output_seen, proxy_text_buffer
                            proxy_text_buffer += proxy_decoder.decode(chunk)
                            while True:
                                match = re.search(r"\r?\n\r?\n", proxy_text_buffer)
                                if not match:
                                    break

                                raw_event = proxy_text_buffer[:match.start()]
                                proxy_text_buffer = proxy_text_buffer[match.end():]
                                if not raw_event.strip():
                                    continue

                                event_type, event_payload = _extract_responses_stream_event(raw_event)
                                if _responses_stream_event_has_real_output(event_type, event_payload):
                                    output_seen = True
                                if event_type == "response.completed":
                                    completed_seen = True
                                    if _responses_usage_from_payload(event_payload) is not None:
                                        usage_seen = True

                        try:
                            for chunk in buffered_chunks:
                                if disconnect_event is not None and disconnect_event.is_set():
                                    _log_responses_downstream_disconnect(
                                        endpoint,
                                        current_info,
                                        model_id=request_model_name,
                                        provider_name=provider_name,
                                        stage="after-stream-commit",
                                    )
                                    return
                                track_responses_events(chunk)
                                yield chunk
                            async for chunk in upstream_iter:
                                if disconnect_event is not None and disconnect_event.is_set():
                                    _log_responses_downstream_disconnect(
                                        endpoint,
                                        current_info,
                                        model_id=request_model_name,
                                        provider_name=provider_name,
                                        stage="after-stream-commit",
                                    )
                                    break
                                track_responses_events(chunk)
                                yield chunk
                        except RESPONSES_STREAM_NETWORK_ERRORS as e:
                            stream_stage = "post-commit" if stream_committed else "preflight"
                            error_text = str(e) or type(e).__name__
                            request_model, actual_model = _log_model_names(request_model_name, original_model)
                            trace_logger.warning(
                                "%s upstream stream aborted stage=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                                endpoint,
                                stream_stage,
                                type(e).__name__,
                                request_id,
                                request_model,
                                actual_model,
                                provider_name,
                                attempt.provider_api_key_raw,
                                upstream_url,
                                error_text,
                            )
                            if stream_committed:
                                yield b"data: [DONE]\n\n"
                        finally:
                            if not completed_seen or not usage_seen:
                                trace_logger.warning(
                                    "%s upstream stream finished without completed usage request_id=%s model=%s provider=%s output_seen=%s completed_seen=%s usage_seen=%s upstream_url=%s",
                                    endpoint,
                                    request_id,
                                    request_model_name,
                                    provider_name,
                                    output_seen,
                                    completed_seen,
                                    usage_seen,
                                    upstream_url,
                                )
                            await stream_cm.__aexit__(None, None, None)

                    background_tasks.add_task(
                        update_channel_stats,
                        current_info["request_id"],
                        channel_id,
                        request_model_name,
                        current_info["api_key"],
                        success=True,
                        provider_api_key=attempt.provider_api_key_raw,
                    )
                    current_info["first_response_time"] = 0
                    current_info["success"] = True
                    current_info["provider"] = channel_id
                    return StarletteStreamingResponse(proxy_stream(), media_type="text/event-stream")

                upstream_resp = await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                    raw = await upstream_resp.aread()
                    try:
                        error_message = raw.decode("utf-8", errors="replace")
                    except Exception:
                        error_message = str(raw)
                    raise HTTPException(status_code=upstream_resp.status_code, detail=error_message)

                data = upstream_resp.json()
                semantic_failure = _responses_failure_http_exception(data)
                if semantic_failure is not None:
                    raise semantic_failure

                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    channel_id,
                    request_model_name,
                    current_info["api_key"],
                    success=True,
                    provider_api_key=attempt.provider_api_key_raw,
                )
                current_info["first_response_time"] = 0
                current_info["success"] = True
                current_info["provider"] = channel_id
                return JSONResponse(status_code=upstream_resp.status_code, content=data)

        def after_failure(attempt, exc, status_code, error_message):
            if attempt.state.get("track_channel_stats"):
                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    attempt.state["channel_id"],
                    request_model_name,
                    current_info["api_key"],
                    success=False,
                    provider_api_key=attempt.provider_api_key_raw,
                )

            upstream_url = attempt.state.get("upstream_url", "")
            failure_stage = attempt.state.get("failure_stage")
            request_model, actual_model = _log_model_names(request_model_name, attempt.original_model)
            if failure_stage == "auth" and isinstance(exc, ValueError):
                trace_logger.error(
                    "%s invalid codex api key request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                    endpoint,
                    request_id,
                    request_model,
                    actual_model,
                    attempt.provider_name,
                    attempt.provider_api_key_raw,
                    upstream_url,
                    error_message,
                )
                return
            if failure_stage == "auth" and isinstance(exc, HTTPException):
                trace_logger.error(
                    "%s codex token refresh failed request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                    endpoint,
                    request_id,
                    request_model,
                    actual_model,
                    attempt.provider_name,
                    attempt.provider_api_key_raw,
                    upstream_url,
                    error_message,
                )
                return

            trace_logger.error(
                "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                endpoint,
                status_code,
                type(exc).__name__,
                request_id,
                request_model,
                actual_model,
                attempt.state.get("channel_id", attempt.provider_name),
                attempt.provider_api_key_raw,
                upstream_url,
                error_message,
            )

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = error_message, attempt
            return not isinstance(exc, ValueError) and status_code not in (400, 413)

        def build_error_response(status_code, error_message):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

        def build_final_response(completed_plan):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_model_name} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            prepare_attempt=prepare_attempt,
            before_next_attempt=before_next_attempt,
            after_failure=after_failure,
            build_error_response=build_error_response,
            build_final_response=build_final_response,
            should_cool_down=should_cool_down,
        )

model_handler = ModelRequestHandler()
responses_handler = ResponsesRequestHandler()

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    token = credentials.credentials
    api_index = None
    try:
        api_index = api_list.index(token)
    except ValueError:
        # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
        # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
        api_index = None
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_index

async def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    token = credentials.credentials
    api_index = None
    try:
        api_index = api_list.index(token)
    except ValueError:
        # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
        # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
        api_index = None
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    # for api_key in app.state.api_keys_db:
    #     if token.startswith(api_key['api']):
    if len(api_list) == 1:
        return token
    if "admin" not in app.state.api_keys_db[api_index].get('role', ''):
        raise HTTPException(status_code=403, detail="Permission denied")
    return token

@app.get("/search", dependencies=[Depends(rate_limit_dependency)])
@app.get("/v1/search", dependencies=[Depends(rate_limit_dependency)])
async def jina_search(
    request: Request,
    background_tasks: BackgroundTasks,
    q: str = Query("Jina+AI"),
    api_index: int = Depends(verify_api_key),
):
    """
    Config-driven search routed through the existing provider selection/rotation architecture.

    Usage:
      - Provider config must include model: search (e.g. provider: jina + model: [search, ...])
      - User api key must include a rule like: jina/search
    """
    search_request = RequestModel(
        model="search",
        messages=[{"role": "user", "content": q}],
        stream=False,
    )
    return await model_handler.request_model(search_request, api_index, background_tasks, endpoint=str(request.url.path))

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def chat_completions_route(request: RequestModel, background_tasks: BackgroundTasks, api_index: int = Depends(verify_api_key)):
    return await model_handler.request_model(request, api_index, background_tasks)

@app.post("/v1/responses", dependencies=[Depends(rate_limit_dependency)])
async def responses_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await responses_handler.request_responses(http_request, request, api_index, background_tasks)

@app.post("/v1/responses/compact", dependencies=[Depends(rate_limit_dependency)])
async def responses_compact_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    response = await responses_handler.request_responses(
        http_request,
        request,
        api_index,
        background_tasks,
        endpoint="/v1/responses/compact",
    )
    # response_body = getattr(response, "body", None)
    # if response_body is not None:
    #     print("print /v1/responses/compact:", response_body.decode("utf-8", errors="replace"))
    return response

# @app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
# async def options_handler():
#     return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(api_index: int = Depends(verify_api_key)):
    models = post_all_models(api_index, app.state.config, get_runtime_api_list(), app.state.models_list)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/generations")

def _is_form_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "file")

def _form_text(value: Any) -> Optional[str]:
    if value is None or _is_form_upload(value):
        return None
    text = str(value).strip()
    return text or None

def _form_bool(value: Any, default: bool = False) -> bool:
    text = _form_text(value)
    if text is None:
        return default
    return text.lower() in ("1", "true", "yes", "on")

async def _parse_image_edit_request(http_request: Request) -> ImageEditRequest:
    content_type = (http_request.headers.get("content-type") or "").strip().lower()
    if content_type.startswith("application/json"):
        try:
            body = await http_request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        request = ImageEditRequest(**body)
        request.request_type = "image"
        return request

    if content_type and not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail=f"Unsupported Content-Type {content_type!r}")

    try:
        form = await http_request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

    prompt = _form_text(form.get("prompt"))
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")

    model = _form_text(form.get("model")) or "gpt-image-2"
    multipart_data: list[tuple[str, Any]] = []
    multipart_files: list[tuple[str, Any]] = []
    form_items = form.multi_items() if hasattr(form, "multi_items") else (
        (key, value) for key in form.keys() for value in form.getlist(key)
    )
    for key, value in form_items:
        if _is_form_upload(value):
            try:
                file_content = await value.read()
            finally:
                try:
                    await value.close()
                except Exception:
                    pass
            multipart_files.append((
                key,
                (
                    value.filename or "upload",
                    file_content,
                    value.content_type or "application/octet-stream",
                ),
            ))
        else:
            multipart_data.append((key, str(value)))

    request = ImageEditRequest(
        prompt=prompt,
        model=model,
        stream=_form_bool(form.get("stream"), False),
        multipart_data=multipart_data,
        multipart_files=multipart_files,
    )
    request.request_type = "image"
    return request

@app.post("/v1/images/edits", dependencies=[Depends(rate_limit_dependency)])
async def images_edits(
    http_request: Request,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    request = await _parse_image_edit_request(http_request)
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/edits")

@app.post("/v1/embeddings", dependencies=[Depends(rate_limit_dependency)])
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/embeddings")

@app.post("/v1/audio/speech", dependencies=[Depends(rate_limit_dependency)])
async def audio_speech(
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks,
    api_index: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/audio/speech")

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/moderations")

@app.post("/v1/audio/transcriptions", dependencies=[Depends(rate_limit_dependency)])
async def audio_transcriptions(
    http_request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    api_index: int = Depends(verify_api_key)
):
    try:
        # Manually parse form data
        form_data = await http_request.form()
        # Use getlist to handle multiple values for the same key
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities: # If list is empty (parameter not sent)
            timestamp_granularities = None # Set to None to match Optional[List[str]]

        # 读取上传的文件内容 (file is still handled by FastAPI)
        content = await file.read()
        file_obj = io.BytesIO(content)

        # 创建AudioTranscriptionRequest对象
        request_obj = AudioTranscriptionRequest(
            file=(file.filename, file_obj, file.content_type),
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities
        )

        return await model_handler.request_model(request_obj, api_index, background_tasks, endpoint="/v1/audio/transcriptions")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid audio file encoding")
    except Exception as e:
        if is_debug:
            import traceback
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@app.get("/v1/generate-api-key", dependencies=[Depends(rate_limit_dependency)])
async def generate_api_key():
    # Define the character set (only alphanumeric)
    chars = string.ascii_letters + string.digits
    # Generate a random string of 36 characters
    random_string = ''.join(secrets.choice(chars) for _ in range(48))
    api_key = "sk-" + random_string
    return JSONResponse(content={"api_key": api_key})

# 在 /stats 路由中返回成功和失败百分比
@app.get("/v1/stats", dependencies=[Depends(rate_limit_dependency)])
async def get_stats(
    request: Request,
    token: str = Depends(verify_admin_api_key),
    hours: int = Query(default=24, ge=1, le=720, description="Number of hours to look back for stats (1-720)")
):
    '''
    ## 获取统计数据

    使用 `/v1/stats` 获取最近 24 小时各个渠道的使用情况统计。同时带上 自己的 uni-api 的 admin API key。

    数据包括：

    1. 每个渠道下面每个模型的成功率，成功率从高到低排序。
    2. 每个渠道总的成功率，成功率从高到低排序。
    3. 每个模型在所有渠道总的请求次数。
    4. 每个端点的请求次数。
    5. 每个ip请求的次数。

    `/v1/stats?hours=48` 参数 `hours` 可以控制返回最近多少小时的数据统计，不传 `hours` 这个参数，默认统计最近 24 小时的统计数据。

    还有其他统计数据，可以自己写sql在数据库自己查。其他数据包括：首字时间，每个请求的总处理时间，每次请求是否成功，每次请求是否符合道德审查，每次请求的文本内容，每次请求的 API key，每次请求的输入 token，输出 token 数量。
    '''
    if DISABLE_DATABASE:
        return JSONResponse(content={"stats": {}})
    async with async_session() as session:
        # 计算指定时间范围的开始时间
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # 1. 每个渠道下面每个模型的成功率
        channel_model_stats = await session.execute(
            select(
                ChannelStat.provider,
                ChannelStat.model,
                func.count().label('total'),
                func.sum(case((ChannelStat.success, 1), else_=0)).label('success_count')
            )
            .where(ChannelStat.timestamp >= start_time)
            .group_by(ChannelStat.provider, ChannelStat.model)
        )
        channel_model_stats = channel_model_stats.fetchall()

        # 2. 每个渠道总的成功率
        channel_stats = await session.execute(
            select(
                ChannelStat.provider,
                func.count().label('total'),
                func.sum(case((ChannelStat.success, 1), else_=0)).label('success_count')
            )
            .where(ChannelStat.timestamp >= start_time)
            .group_by(ChannelStat.provider)
        )
        channel_stats = channel_stats.fetchall()

        # 3. 每个模型在所有渠道总的请求次数
        model_stats = await session.execute(
            select(RequestStat.model, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.model)
            .order_by(desc('count'))
        )
        model_stats = model_stats.fetchall()

        # 4. 每个端点的请求次数
        endpoint_stats = await session.execute(
            select(RequestStat.endpoint, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.endpoint)
            .order_by(desc('count'))
        )
        endpoint_stats = endpoint_stats.fetchall()

        # 5. 每个ip请求的次数
        ip_stats = await session.execute(
            select(RequestStat.client_ip, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.client_ip)
            .order_by(desc('count'))
        )
        ip_stats = ip_stats.fetchall()

    # 处理统计数据并返回
    stats = {
        "time_range": f"Last {hours} hours",
        "channel_model_success_rates": [
            {
                "provider": stat.provider,
                "model": stat.model,
                "success_rate": stat.success_count / stat.total if stat.total > 0 else 0,
                "total_requests": stat.total
            } for stat in sorted(channel_model_stats, key=lambda x: x.success_count / x.total if x.total > 0 else 0, reverse=True)
        ],
        "channel_success_rates": [
            {
                "provider": stat.provider,
                "success_rate": stat.success_count / stat.total if stat.total > 0 else 0,
                "total_requests": stat.total
            } for stat in sorted(channel_stats, key=lambda x: x.success_count / x.total if x.total > 0 else 0, reverse=True)
        ],
        "model_request_counts": [
            {
                "model": stat.model,
                "count": stat.count
            } for stat in model_stats
        ],
        "endpoint_request_counts": [
            {
                "endpoint": stat.endpoint,
                "count": stat.count
            } for stat in endpoint_stats
        ],
        "ip_request_counts": [
            {
                "ip": stat.client_ip,
                "count": stat.count
            } for stat in ip_stats
        ]
    }

    return JSONResponse(content=stats)

@app.get("/", dependencies=[Depends(rate_limit_dependency)])
async def root():
    return RedirectResponse(url="https://uni-api-web.pages.dev", status_code=302)

# async def on_fetch(request, env):
#     import asgi
#     return await asgi.fetch(app, request, env)

@app.get("/v1/api_config", dependencies=[Depends(rate_limit_dependency)])
async def api_config(api_index: int = Depends(verify_admin_api_key)):
    encoded_config = jsonable_encoder(app.state.config)
    return JSONResponse(content={"api_config": encoded_config})

@app.post("/v1/api_config/update", dependencies=[Depends(rate_limit_dependency)])
async def api_config_update(api_index: int = Depends(verify_admin_api_key), config: dict = Body(...)):
    if "providers" in config:
        app.state.config["providers"] = config["providers"]
        app.state.config, app.state.api_keys_db, app.state.api_list = await update_config(
            app.state.config,
            use_config_url=False,
        )
        await refresh_runtime_state(app)
    return JSONResponse(content={"message": "API config updated"})

# Pydantic Models for Token Usage Response
class TokenUsageEntry(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int

class QueryDetails(BaseModel):
    model_config = {'protected_namespaces': ()}

    start_datetime: Optional[str] = None # e.g., "2023-10-27T10:00:00Z" or Unix timestamp
    end_datetime: Optional[str] = None   # e.g., "2023-10-28T12:30:45Z" or Unix timestamp
    api_key_filter: Optional[str] = None
    model_filter: Optional[str] = None
    credits: Optional[str] = None
    total_cost: Optional[str] = None
    balance: Optional[str] = None

class TokenUsageResponse(BaseModel):
    usage: List[TokenUsageEntry]
    query_details: QueryDetails


class ChannelKeyRanking(BaseModel):
    api_key: str
    success_count: int
    total_requests: int
    success_rate: float


class ChannelKeyRankingsResponse(BaseModel):
    rankings: List[ChannelKeyRanking]
    query_details: QueryDetails


async def query_token_usage(
    session: AsyncSession,
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None
) -> List[Dict]:
    """Queries the RequestStat table for aggregated token usage."""
    query = select(
        RequestStat.api_key,
        RequestStat.model,
        func.sum(RequestStat.prompt_tokens).label("total_prompt_tokens"),
        func.sum(RequestStat.completion_tokens).label("total_completion_tokens"),
        func.sum(RequestStat.total_tokens).label("total_tokens"),
        func.count(RequestStat.id).label("request_count")
    ).group_by(RequestStat.api_key, RequestStat.model)

    # Apply filters
    if filter_api_key:
        query = query.where(RequestStat.api_key == filter_api_key)
    if filter_model:
        query = query.where(RequestStat.model == filter_model)
    if start_dt:
        query = query.where(RequestStat.timestamp >= start_dt)
    if end_dt:
        # Make end_dt inclusive by adding one day
        query = query.where(RequestStat.timestamp < end_dt + timedelta(days=1))

    # Filter out entries with null or empty model if not specifically requested
    if not filter_model:
         query = query.where(RequestStat.model.isnot(None) & (RequestStat.model != ''))


    result = await session.execute(query)
    rows = result.mappings().all()

    # Process results: mask API key
    processed_usage = []
    for row in rows:
        usage_dict = dict(row)
        api_key = usage_dict.get("api_key", "")
        # Mask API key (show prefix like sk-...xyz)
        if api_key and len(api_key) > 7:
            prefix = api_key[:7]
            suffix = api_key[-4:]
            usage_dict["api_key_prefix"] = f"{prefix}...{suffix}"
        else:
            usage_dict["api_key_prefix"] = api_key # Show short keys as is or handle None
        del usage_dict["api_key"] # Remove original full key
        processed_usage.append(usage_dict)

    return processed_usage

async def get_usage_data(filter_api_key: Optional[str] = None, filter_model: Optional[str] = None,
                        start_dt_obj: Optional[datetime] = None, end_dt_obj: Optional[datetime] = None) -> List[Dict]:
    """
    查询数据库并获取令牌使用数据。
    这个函数封装了创建会话和查询令牌使用情况的逻辑。

    Args:
        filter_api_key: 可选的API密钥过滤器
        filter_model: 可选的模型过滤器
        start_dt_obj: 开始日期时间
        end_dt_obj: 结束日期时间

    Returns:
        包含令牌使用统计数据的列表
    """
    async with async_session() as session:
        usage_data = await query_token_usage(
            session=session,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj
        )
    return usage_data

@app.get("/v1/token_usage", response_model=TokenUsageResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_token_usage(
    request: Request, # Inject request to access app.state
    api_key_param: Optional[str] = None, # Query param for admin filtering
    model: Optional[str] = None,
    start_datetime: Optional[str] = None, # ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp
    end_datetime: Optional[str] = None,   # ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp
    last_n_days: Optional[int] = None,
    api_index: tuple = Depends(verify_api_key) # Use verify_api_key for auth and getting token/index
):
    """
    Retrieves aggregated token usage statistics based on API key and model,
    filtered by a specified time range.
    Admin users can filter by specific API keys.
    """
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    requesting_token = safe_get(app.state.config, 'api_keys', api_index, "api", default="") # verify_api_key returns the token directly now

    # Determine admin status
    is_admin = False
    # print("app.state.admin_api_key", app.state.admin_api_key, requesting_token, requesting_token in app.state.admin_api_key)
    if hasattr(app.state, "admin_api_key") and requesting_token in app.state.admin_api_key:
        is_admin = True

    # Determine API key filter
    filter_api_key = None
    api_key_filter_detail = "all" # For response details
    # print("api_key_param", is_admin, api_key_param)
    if is_admin:
        if api_key_param:
            filter_api_key = api_key_param
            api_key_filter_detail = api_key_param
        # else: filter_api_key remains None (all users)
    else:
        # Non-admin can only see their own stats
        filter_api_key = requesting_token
        api_key_filter_detail = "self"

    # Determine time range
    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None

    now = datetime.now(timezone.utc)

    def parse_datetime_input(dt_input: str) -> datetime:
        """Parses ISO 8601 string or Unix timestamp."""
        try:
            # Try parsing as Unix timestamp first
            return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
        except ValueError:
            # Try parsing as ISO 8601 format
            try:
                # Handle potential 'Z' for UTC timezone explicitly
                if dt_input.endswith('Z'):
                    dt_input = dt_input[:-1] + '+00:00'
                # Use fromisoformat for robust parsing
                dt_obj = datetime.fromisoformat(dt_input)
                # Ensure timezone is UTC if naive
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                # Convert to UTC if it has another timezone
                return dt_obj.astimezone(timezone.utc)
            except ValueError:
                raise ValueError(f"Invalid datetime format: {dt_input}. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp.")


    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(status_code=400, detail="Cannot use last_n_days with start_datetime or end_datetime.")
        if last_n_days <= 0:
            raise HTTPException(status_code=400, detail="last_n_days must be positive.")
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now # Use current time as end for last_n_days
        start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
        end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')
            # Basic validation: end datetime should not be before start datetime
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                 raise HTTPException(status_code=400, detail="end_datetime cannot be before start_datetime.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Default to last 30 days if no range specified
        start_dt_obj = now - timedelta(days=30)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
        end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')

    # 使用新的get_usage_data函数替代直接的数据库查询代码
    usage_data = await get_usage_data(
        filter_api_key=filter_api_key,
        filter_model=model,
        start_dt_obj=start_dt_obj,
        end_dt_obj=end_dt_obj
    )
    # print("usage_data", usage_data)

    if filter_api_key:
        credits, total_cost = await update_paid_api_keys_states(app, filter_api_key)
    else:
        credits, total_cost = None, None

    # Prepare response
    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=api_key_filter_detail,
        model_filter=model if model else "all",
        credits= "$" + str(credits),
        total_cost= "$" + str(total_cost),
        balance= "$" + str(float(credits) - float(total_cost)) if credits and total_cost else None
    )

    response_data = TokenUsageResponse(
        usage=[TokenUsageEntry(**item) for item in usage_data],
        query_details=query_details
    )

    return response_data


@app.get(
    "/v1/channel_key_rankings",
    response_model=ChannelKeyRankingsResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
async def get_channel_key_rankings(
    request: Request,
    provider_name: str,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
    token: str = Depends(verify_admin_api_key),
):
    """
    Retrieves the success rate ranking of API keys for a specific channel,
    filtered by a specified time range.
    """
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None

    now = datetime.now(timezone.utc)

    def parse_datetime_input(dt_input: str) -> datetime:
        """Parses ISO 8601 string or Unix timestamp."""
        try:
            return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
        except ValueError:
            try:
                if dt_input.endswith("Z"):
                    dt_input = dt_input[:-1] + "+00:00"
                dt_obj = datetime.fromisoformat(dt_input)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                return dt_obj.astimezone(timezone.utc)
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format: {dt_input}. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp."
                )

    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(
                status_code=400,
                detail="Cannot use last_n_days with start_datetime or end_datetime.",
            )
        if last_n_days <= 0:
            raise HTTPException(
                status_code=400, detail="last_n_days must be positive."
            )
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                raise HTTPException(
                    status_code=400, detail="end_datetime cannot be before start_datetime."
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Default to last 24 hours if no range specified
        start_dt_obj = now - timedelta(days=1)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")

    rankings_data = await query_channel_key_stats(
        provider_name=provider_name, start_dt=start_dt_obj, end_dt=end_dt_obj
    )

    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=provider_name,
    )

    response_data = ChannelKeyRankingsResponse(
        rankings=[ChannelKeyRanking(**item) for item in rankings_data],
        query_details=query_details,
    )

    return response_data


class TokenInfo(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int

class ApiKeyState(BaseModel):
    credits: float
    created_at: datetime
    all_tokens_info: List[Dict[str, Any]]
    total_cost: float
    enabled: bool

    @field_serializer('created_at')
    def serialize_dt(self, dt: datetime):
        return dt.isoformat()

class ApiKeysStatesResponse(BaseModel):
    api_keys_states: Dict[str, ApiKeyState]

@app.get("/v1/api_keys_states", dependencies=[Depends(rate_limit_dependency)])
async def api_keys_states(token: str = Depends(verify_admin_api_key)):
    # 转换原始状态数据为Pydantic模型
    states_dict = {}
    for key, state in app.state.paid_api_keys_states.items():
        # 创建ApiKeyState对象
        states_dict[key] = ApiKeyState(
            credits=state["credits"],
            created_at=state["created_at"],
            all_tokens_info=state["all_tokens_info"],
            total_cost=state["total_cost"],
            enabled=state["enabled"]
        )

    # 创建响应模型
    response = ApiKeysStatesResponse(api_keys_states=states_dict)

    # 返回JSON序列化结果
    return response

@app.post("/v1/add_credits", dependencies=[Depends(rate_limit_dependency)])
async def add_credits_to_api_key(
    request: Request, # Inject request to access app.state
    paid_key: str = Query(..., description="The API key to add credits to"),
    amount: float = Query(..., description="The amount of credits to add. Must be positive.", gt=0),
    token: str = Depends(verify_admin_api_key)
):
    if paid_key not in app.state.paid_api_keys_states:
        raise HTTPException(status_code=404, detail=f"API key '{paid_key}' not found in paid API keys states.")

    # The validation `amount > 0` is handled by `Query(..., gt=0)`

    # 更新 credits
    # Ensure 'amount' is treated as float, though Query should handle conversion.
    app.state.paid_api_keys_states[paid_key]["credits"] += float(amount)

    # 更新 enabled 状态
    current_credits = app.state.paid_api_keys_states[paid_key]["credits"]
    total_cost = app.state.paid_api_keys_states[paid_key]["total_cost"]
    app.state.paid_api_keys_states[paid_key]["enabled"] = current_credits >= total_cost

    logger.info(f"Credits for API key '{paid_key}' updated. Amount added: {amount}, New credits: {current_credits}, Enabled: {app.state.paid_api_keys_states[paid_key]['enabled']}")

    return JSONResponse(content={
        "message": f"Successfully added {amount} credits to API key '{paid_key}'.",
        "paid_key": paid_key,
        "new_credits": current_credits,
        "enabled": app.state.paid_api_keys_states[paid_key]["enabled"]
    })

# 添加静态文件挂载
app.mount("/", StaticFiles(directory="./static", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    import os
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        reload_dirs=["./"],
        reload_includes=["*.py", "api.yaml"],
        reload_excludes=["./data"],
        ws="none",
        # log_level="warning"
    )
