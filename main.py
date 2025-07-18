import os
import json
import httpx
import string
import secrets
from time import time
from urllib.parse import urlparse
from collections import defaultdict
from pydantic import ValidationError
from contextlib import asynccontextmanager
from typing import Dict, Union, Optional, List, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from fastapi import FastAPI, HTTPException, Depends, Request, Body, BackgroundTasks

from core.log_config import logger
from core.request import get_payload
from core.response import fetch_response, fetch_response_stream
from core.models import RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, TextToSpeechRequest, UnifiedRequest, EmbeddingRequest
from core.utils import (
    get_proxy,
    get_engine,
    parse_rate_limit,
    circular_list_encoder,
    ThreadSafeCircularList,
    provider_api_circular_list,
)

from utils import (
    safe_get,
    load_config,
    update_config,
    post_all_models,
    InMemoryRateLimiter,
    calculate_total_cost,
    error_handling_wrapper,
)


DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
is_debug = bool(os.getenv("DEBUG", False))
# is_debug = False

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes

# 添加新的环境变量检查
DISABLE_DATABASE = os.getenv("DISABLE_DATABASE", "false").lower() == "true"
logger.info("DISABLE_DATABASE: %s", DISABLE_DATABASE)

# 读取VERSION文件内容
try:
    with open('VERSION', 'r') as f:
        VERSION = f.read().strip()
except:
    VERSION = 'unknown'
logger.info("VERSION: %s", VERSION)

async def create_tables():
    if DISABLE_DATABASE:
        return
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 检查并添加缺失的列 - 此简易迁移仅针对 SQLite
        if os.getenv("DB_TYPE", "sqlite").lower() == "sqlite":
            def check_and_add_columns(connection):
                inspector = inspect(connection)
                for table in [RequestStat, ChannelStat]:
                    table_name = table.__tablename__
                    existing_columns = {col['name']: col['type'] for col in inspector.get_columns(table_name)}

                    for column_name, column in table.__table__.columns.items():
                        if column_name not in existing_columns:
                            col_type = _map_sa_type_to_sql_type(column.type)
                            default = _get_default_sql(column.default)
                            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {col_type}{default}"))

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

async def update_paid_api_keys_states(app, paid_key):
    """
    更新付费API密钥的状态

    参数:
    app - FastAPI应用实例
    check_index - API密钥在配置中的索引
    paid_key - 需要更新状态的API密钥
    """
    check_index = app.state.api_list.index(paid_key)
    credits = safe_get(app.state.config, 'api_keys', check_index, "preferences", "credits", default=-1)
    created_at = safe_get(app.state.config, 'api_keys', check_index, "preferences", "created_at", default=datetime.now(timezone.utc) - timedelta(days=30))
    model_price = safe_get(app.state.config, 'preferences', "model_price", default={})
    created_at = created_at.astimezone(timezone.utc)
    if credits != -1:
        all_tokens_info = await get_usage_data(filter_api_key=paid_key, start_dt_obj=created_at)
        total_cost = calculate_total_cost(all_tokens_info, model_price)
        app.state.paid_api_keys_states[paid_key] = {
            "credits": credits,
            "created_at": created_at,
            "all_tokens_info": all_tokens_info,
            "total_cost": total_cost,
            "enabled": True if total_cost <= credits else False
        }
        return credits, total_cost

    return credits, 0
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

        if app.state.api_list:
            app.state.user_api_keys_rate_limit = defaultdict(ThreadSafeCircularList)
            for api_index, api_key in enumerate(app.state.api_list):
                app.state.user_api_keys_rate_limit[api_key] = ThreadSafeCircularList(
                    [api_key],
                    safe_get(app.state.config, 'api_keys', api_index, "preferences", "rate_limit", default={"default": "999999/min"}),
                    "round_robin"
                )
        app.state.global_rate_limit = parse_rate_limit(safe_get(app.state.config, "preferences", "rate_limit", default="999999/min"))

        app.state.admin_api_key = []
        for item in app.state.api_keys_db:
            if item.get("role") == "admin":
                app.state.admin_api_key.append(item.get("api"))
        if app.state.admin_api_key == []:
            if len(app.state.api_keys_db) >= 1:
                app.state.admin_api_key = [app.state.api_keys_db[0].get("api")]
            else:
                from utils import yaml_error_message
                if yaml_error_message:
                    raise HTTPException(
                        status_code=500,
                        detail={"error": yaml_error_message}
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail={"error": "No API key found in api.yaml"}
                    )

        app.state.provider_timeouts = init_preference(app.state.config, "model_timeout", DEFAULT_TIMEOUT)
        app.state.keepalive_interval = init_preference(app.state.config, "keepalive_interval", 99999)
        # pprint(dict(app.state.provider_timeouts))
        # pprint(dict(app.state.keepalive_interval))
        # print("app.state.provider_timeouts", app.state.provider_timeouts)
        # print("app.state.keepalive_interval", app.state.keepalive_interval)
        if not DISABLE_DATABASE:
            app.state.paid_api_keys_states = {}
            for paid_key in app.state.api_list:
                await update_paid_api_keys_states(app, paid_key)

    if app and not hasattr(app.state, 'client_manager'):

        default_config = {
            "headers": {
                "User-Agent": "curl/7.68.0",
                "Accept": "*/*",
            },
            "http2": True,
            "verify": True,
            "follow_redirects": True
        }

        # 初始化客户端管理器
        app.state.client_manager = ClientManager(pool_size=200)
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

import uuid
import asyncio
import contextvars
request_info = contextvars.ContextVar('request_info', default={})

async def parse_request_body(request: Request):
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        try:
            return await request.json()
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

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, select, Boolean, Text
from sqlalchemy.sql import func
from sqlalchemy import event

# 定义数据库模型
Base = declarative_base()

class RequestStat(Base):
    __tablename__ = 'request_stats'
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    endpoint = Column(String)
    client_ip = Column(String)
    process_time = Column(Float)
    first_response_time = Column(Float)
    provider = Column(String, index=True)
    model = Column(String, index=True)
    # success = Column(Boolean, default=False)
    api_key = Column(String, index=True)
    is_flagged = Column(Boolean, default=False)
    text = Column(Text)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    # cost = Column(Float, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

class ChannelStat(Base):
    __tablename__ = 'channel_stats'
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    provider = Column(String, index=True)
    model = Column(String, index=True)
    api_key = Column(String)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


if not DISABLE_DATABASE:
    DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
    logger.info(f"Using {DB_TYPE} database.")

    if DB_TYPE == "postgres":
        # PostgreSQL-specific setup
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg is not installed. Please install it with 'pip install asyncpg' to use PostgreSQL.")

        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "postgres")

        db_url = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        db_engine = create_async_engine(db_url, echo=is_debug)

    elif DB_TYPE == "sqlite":
        # SQLite-specific setup
        # 获取数据库路径
        db_path = os.getenv('DB_PATH', './data/stats.db')

        # 确保 data 目录存在
        data_dir = os.path.dirname(db_path)
        os.makedirs(data_dir, exist_ok=True)

        # 创建异步引擎
        db_engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=is_debug)

        # 为 SQLite 设置 WAL 模式和 busy_timeout
        @event.listens_for(db_engine.sync_engine, "connect")
        def set_sqlite_pragma_on_connect(dbapi_connection, connection_record):
            """为每个 SQLite 连接开启 WAL 模式并设置 busy_timeout"""
            cursor = None
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA busy_timeout = 5000;") # 5000毫秒 = 5秒
            except Exception as e:
                logger.error(f"Failed to set PRAGMA for SQLite: {e}")
            finally:
                if cursor:
                    cursor.close()
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}. Please use 'sqlite' or 'postgres'.")

    # 创建会话 Session
    async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

from starlette.types import Scope, Receive, Send
from starlette.responses import Response

from asyncio import Semaphore

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

async def update_channel_stats(request_id, provider, model, api_key, success):
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
            except:
                pass  # 如果无法发送错误消息，则忽略
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
            if self.current_info.get("endpoint") == "/v1/audio/speech":
                yield chunk
                continue
            line = chunk.decode('utf-8')
            if is_debug:
                logger.info(f"{line.encode('utf-8').decode('unicode_escape')}")
            if line.startswith("data:"):
                line = line.lstrip("data: ")
            if not line.startswith("[DONE]") and not line.startswith("OK") and not line.startswith(": "):
                try:
                    resp: dict = json.loads(line)
                    input_tokens = safe_get(resp, "message", "usage", "input_tokens", default=0)
                    input_tokens = safe_get(resp, "usage", "prompt_tokens", default=0)
                    output_tokens = safe_get(resp, "usage", "completion_tokens", default=0)
                    total_tokens = input_tokens + output_tokens

                    self.current_info["prompt_tokens"] = input_tokens
                    self.current_info["completion_tokens"] = output_tokens
                    self.current_info["total_tokens"] = total_tokens
                except Exception as e:
                    logger.error(f"Error parsing response: {str(e)}, line: {repr(line)}")
                    continue
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
                if safe_get(app.state.paid_api_keys_states, check_api_key, default={}).get("enabled", None) == False and \
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
            "client_ip": request.client.host,
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
        try:
            parsed_body = await parse_request_body(request)
            if parsed_body and not request.url.path.startswith("/v1/api_config"):
                request_model = UnifiedRequest.model_validate(parsed_body).data
                if is_debug:
                    logger.info("request_model: %s", json.dumps(request_model.model_dump(exclude_unset=True), indent=2, ensure_ascii=False))
                model = request_model.model
                current_info["model"] = model

                final_api_key = app.state.api_list[api_index]
                try:
                    await app.state.user_api_keys_rate_limit[final_api_key].next(model)
                except Exception as e:
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

                if moderated_content:
                    current_info["text"] = moderated_content

                if enable_moderation and moderated_content:
                    background_tasks_for_moderation = BackgroundTasks()
                    moderation_response = await self.moderate_content(moderated_content, api_index, background_tasks_for_moderation)
                    is_flagged = moderation_response.get('results', [{}])[0].get('flagged', False)

                    if is_flagged:
                        logger.error(f"Content did not pass the moral check: %s", moderated_content)
                        process_time = time() - start_time
                        current_info["process_time"] = process_time
                        current_info["is_flagged"] = is_flagged
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

        except ValidationError as e:
            logger.error(f"Invalid request body: {json.dumps(parsed_body, indent=2, ensure_ascii=False)}, errors: {e.errors()}")
            return JSONResponse(
                status_code=422,
                content=jsonable_encoder({"detail": e.errors()})
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
        app.state.models_list = {}
        for item in app.state.api_keys_db:
            api_key_model_list = item.get("model", [])
            for provider_rule in api_key_model_list:
                provider_name = provider_rule.split("/")[0]
                if provider_name.startswith("sk-") and provider_name in app.state.api_list:
                    models_list = []
                    try:
                        # 构建请求头
                        headers = {
                            "Authorization": f"Bearer {provider_name}"
                        }
                        # 发送GET请求获取模型列表
                        base_url = "http://127.0.0.1:8000/v1/models"
                        async with app.state.client_manager.get_client(1, base_url) as client:
                            response = await client.get(
                                base_url,
                                headers=headers
                            )
                            if response.status_code == 200:
                                models_data = response.json()
                                # 将获取到的模型添加到models_list
                                for model in models_data.get("data", []):
                                    models_list.append(model["id"])
                    except Exception as e:
                        if str(e):
                            logger.error(f"获取模型列表失败: {str(e)}")
                    app.state.models_list[provider_name] = models_list
    return await call_next(request)

class ClientManager:
    def __init__(self, pool_size=100):
        self.pool_size = pool_size
        self.clients = {}  # {host_timeout_proxy: AsyncClient}

    async def init(self, default_config):
        self.default_config = default_config

    @asynccontextmanager
    async def get_client(self, timeout_value, base_url, proxy=None):
        # 直接获取或创建客户端,不使用锁
        timeout_value = int(timeout_value)

        # 从base_url中提取主机名
        parsed_url = urlparse(base_url)
        host = parsed_url.netloc

        # 创建唯一的客户端键
        client_key = f"{host}_{timeout_value}"
        if proxy:
            # 对代理URL进行规范化处理
            proxy_normalized = proxy.replace('socks5h://', 'socks5://')
            client_key += f"_{proxy_normalized}"

        if client_key not in self.clients:
            timeout = httpx.Timeout(
                connect=15.0,
                read=timeout_value,
                write=30.0,
                pool=self.pool_size
            )
            limits = httpx.Limits(max_connections=self.pool_size)

            client_config = {
                **self.default_config,
                "timeout": timeout,
                "limits": limits
            }

            client_config = get_proxy(proxy, client_config)

            self.clients[client_key] = httpx.AsyncClient(**client_config)

        try:
            yield self.clients[client_key]
        except Exception as e:
            if client_key in self.clients and "429" not in str(e):
                tmp_client = self.clients[client_key]
                del self.clients[client_key]  # 先删除引用
                await tmp_client.aclose()  # 然后关闭客户端
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
        timeout_value = get_preference_value(preference_config["global"], original_model)
    if timeout_value is None:
        timeout_value = preference_config["global"].get("default", default_value)
    # print("timeout_value", channel_id, timeout_value)
    return timeout_value

# 在 process_request 函数中更新成功和失败计数
async def process_request(request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], provider: Dict, background_tasks: BackgroundTasks, endpoint=None, role=None, timeout_value=DEFAULT_TIMEOUT, keepalive_interval=None):
    model_dict = provider["_model_dict_cache"]
    original_model = model_dict[request.model]
    if provider['provider'].startswith("sk-"):
        api_key = provider['provider']
    elif provider.get("api"):
        api_key = await provider_api_circular_list[provider['provider']].next(original_model)
    else:
        api_key = None

    engine, stream_mode = get_engine(provider, endpoint, original_model)

    if stream_mode != None:
        request.stream = stream_mode

    channel_id = f"{provider['provider']}"
    if engine != "moderation":
        logger.info(f"provider: {channel_id[:11]:<11} model: {request.model:<22} engine: {engine[:13]:<13} role: {role}")

    url, headers, payload = await get_payload(request, engine, provider, api_key)
    headers.update(safe_get(provider, "preferences", "headers", default={}))  # add custom headers
    if is_debug:
        logger.info(url)
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        logger.info(json.dumps({k: v for k, v in payload.items() if k != 'file'}, indent=4, ensure_ascii=False))

    current_info = request_info.get()

    proxy = safe_get(app.state.config, "preferences", "proxy", default=None)  # global proxy
    proxy = safe_get(provider, "preferences", "proxy", default=proxy)  # provider proxy
    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(timeout_value, url, proxy) as client:
            if request.stream:
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, request.stream, app.state.error_triggers, keepalive_interval=keepalive_interval)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:
                generator = fetch_response(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, request.stream, app.state.error_triggers, keepalive_interval=keepalive_interval)

                # 处理音频和其他二进制响应
                if endpoint == "/v1/audio/speech":
                    if isinstance(wrapped_generator, bytes):
                        response = Response(content=wrapped_generator, media_type="audio/mpeg")
                else:
                    first_element = await anext(wrapped_generator)
                    first_element = first_element.lstrip("data: ")
                    first_element = json.loads(first_element)
                    response = StarletteStreamingResponse(iter([json.dumps(first_element)]), media_type="application/json")

            # 更新成功计数和首次响应时间
            background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=True)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = channel_id
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=False)
        raise e

def weighted_round_robin(weights):
    provider_names = list(weights.keys())
    current_weights = {name: 0 for name in provider_names}
    num_selections = total_weight = sum(weights.values())
    weighted_provider_list = []

    for _ in range(num_selections):
        max_ratio = -1
        selected_letter = None

        for name in provider_names:
            current_weights[name] += weights[name]
            ratio = current_weights[name] / weights[name]

            if ratio > max_ratio:
                max_ratio = ratio
                selected_letter = name

        weighted_provider_list.append(selected_letter)
        current_weights[selected_letter] -= total_weight

    return weighted_provider_list

import random

def lottery_scheduling(weights):
    total_tickets = sum(weights.values())
    selections = []
    for _ in range(total_tickets):
        ticket = random.randint(1, total_tickets)
        cumulative = 0
        for provider, weight in weights.items():
            cumulative += weight
            if ticket <= cumulative:
                selections.append(provider)
                break
    return selections

async def get_provider_rules(model_rule, config, request_model):
    provider_rules = []
    if model_rule == "all":
        # 如模型名为 all，则返回所有模型
        for provider in config["providers"]:
            model_dict = provider["_model_dict_cache"]
            for model in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model)

    elif "/" in model_rule:
        if model_rule.startswith("<") and model_rule.endswith(">"):
            model_rule = model_rule[1:-1]
            # 处理带斜杠的模型名
            for provider in config['providers']:
                model_dict = provider["_model_dict_cache"]
                if model_rule in model_dict.keys():
                    provider_rules.append(provider['provider'] + "/" + model_rule)
        else:
            provider_name = model_rule.split("/")[0]
            model_name_split = "/".join(model_rule.split("/")[1:])
            models_list = []

            # api_keys 中 api 为 sk- 时，表示继承 api_keys，将 api_keys 中的 api key 当作 渠道
            if provider_name.startswith("sk-") and provider_name in app.state.api_list:
                if app.state.models_list.get(provider_name):
                    models_list = app.state.models_list[provider_name]
                else:
                    models_list = []
            else:
                for provider in config['providers']:
                    model_dict = provider["_model_dict_cache"]
                    if provider['provider'] == provider_name:
                        models_list.extend(list(model_dict.keys()))

            # print("models_list", models_list)
            # print("request_model", request_model)
            # print("model_name_split", model_name_split)
            # print("model", model)

            # api_keys 中 model 为 provider_name/* 时，表示所有模型都匹配
            if model_name_split == "*":
                if request_model in models_list:
                    provider_rules.append(provider_name + "/" + request_model)

                # 如果请求模型名： gpt-4* ，则匹配所有以模型名开头且不以 * 结尾的模型
                for models_list_model in models_list:
                    if request_model.endswith("*") and models_list_model.startswith(request_model.rstrip("*")):
                        provider_rules.append(provider_name + "/" + models_list_model)

            # api_keys 中 model 为 provider_name/model_name 时，表示模型名完全匹配
            elif model_name_split == request_model \
            or (request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*"))): # api_keys 中 model 为 provider_name/model_name 时，请求模型名： model_name*
                if model_name_split in models_list:
                    provider_rules.append(provider_name + "/" + model_name_split)

    else:
        for provider in config["providers"]:
            model_dict = provider["_model_dict_cache"]
            if model_rule in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model_rule)

    return provider_rules

def get_provider_list(provider_rules, config, request_model):
    provider_list = []
    # print("provider_rules", provider_rules)
    for item in provider_rules:
        provider_name = item.split("/")[0]
        if provider_name.startswith("sk-") and provider_name in app.state.api_list:
            provider_list.append({
                "provider": provider_name,
                "base_url": "http://127.0.0.1:8000/v1/chat/completions",
                "model": [{request_model: request_model}],
                "tools": True,
                "_model_dict_cache": {request_model: request_model}
            })
        else:
            for provider in config['providers']:
                model_dict = provider["_model_dict_cache"]
                if not model_dict:
                    continue
                model_name_split = "/".join(item.split("/")[1:])
                if "/" in item and provider['provider'] == provider_name and model_name_split in model_dict.keys():
                    if request_model in model_dict.keys() and model_name_split == request_model:
                        new_provider = {
                            "provider": provider["provider"],
                            "base_url": provider.get("base_url", ""),
                            "api": provider.get("api", None),
                            "model": [{model_dict[model_name_split]: request_model}],
                            "preferences": provider.get("preferences", {}),  # 可能也需要浅拷贝
                            "tools": provider.get("tools", False),
                            "_model_dict_cache": provider["_model_dict_cache"],
                            "project_id": provider.get("project_id", None),
                            "private_key": provider.get("private_key", None),
                            "client_email": provider.get("client_email", None),
                            "cf_account_id": provider.get("cf_account_id", None),
                            "aws_access_key": provider.get("aws_access_key", None),
                            "aws_secret_key": provider.get("aws_secret_key", None),
                            "engine":  provider.get("engine", None),
                        }
                        provider_list.append(new_provider)

                    elif request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*")):
                        new_provider = {
                            "provider": provider["provider"],
                            "base_url": provider.get("base_url", ""),
                            "api": provider.get("api", None),
                            "model": [{model_dict[model_name_split]: request_model}],
                            "preferences": provider.get("preferences", {}),  # 可能也需要浅拷贝
                            "tools": provider.get("tools", False),
                            "_model_dict_cache": provider["_model_dict_cache"],
                            "project_id": provider.get("project_id", None),
                            "private_key": provider.get("private_key", None),
                            "client_email": provider.get("client_email", None),
                            "cf_account_id": provider.get("cf_account_id", None),
                            "aws_access_key": provider.get("aws_access_key", None),
                            "aws_secret_key": provider.get("aws_secret_key", None),
                            "engine":  provider.get("engine", None),
                        }
                        provider_list.append(new_provider)
    return provider_list

async def get_matching_providers(request_model, config, api_index):
    provider_rules = []

    for model_rule in config['api_keys'][api_index]['model']:
        provider_rules.extend(await get_provider_rules(model_rule, config, request_model))

    provider_list = get_provider_list(provider_rules, config, request_model)

    # print("provider_list", provider_list)
    return provider_list

async def get_right_order_providers(request_model, config, api_index, scheduling_algorithm):
    matching_providers = await get_matching_providers(request_model, config, api_index)

    # # 检查是否所有key都被速率限制
    # available_providers = []
    # for provider in matching_providers:
    #     model_dict = get_model_dict(provider)
    #     original_model = model_dict[request_model]
    #     provider_name = provider['provider']
    #     # 如果是本地API密钥（以sk-开头）
    #     if provider_name.startswith("sk-") and provider_name in app.state.api_list:
    #         # 本地API密钥直接添加到可用列表中，因为它们的限制已在其他地方处理
    #         available_providers.append(provider)
    #     # 检查provider对应的API密钥列表是否都被速率限制
    #     elif not await provider_api_circular_list[provider_name].is_all_rate_limited(original_model):
    #         # 如果provider没有API密钥或至少有一个API密钥未被速率限制，则添加到可用列表
    #         available_providers.append(provider)
    #     else:
    #         logger.warning(f"Provider {provider_name}: all API keys are rate limited!")

    # # 使用筛选后的provider列表替换原始列表
    # matching_providers = available_providers

    # for provider in matching_providers:
    #     print(provider['provider'])

    if not matching_providers:
        raise HTTPException(status_code=404, detail=f"No available providers at the moment: {request_model}")

    num_matching_providers = len(matching_providers)
    # 如果某个渠道的一个模型报错，这个渠道会被排除
    if app.state.channel_manager.cooldown_period > 0 and num_matching_providers > 1:
        matching_providers = await app.state.channel_manager.get_available_providers(matching_providers)
        num_matching_providers = len(matching_providers)
        if not matching_providers:
            raise HTTPException(status_code=503, detail="No available providers at the moment")

    # for provider in matching_providers:
    #     print(provider['provider'])

    # 检查是否启用轮询
    if scheduling_algorithm == "random":
        matching_providers = random.sample(matching_providers, num_matching_providers)

    weights = safe_get(config, 'api_keys', api_index, "weights")

    if weights:
        intersection = None
        all_providers = set(provider['provider'] + "/" + request_model for provider in matching_providers)
        if all_providers:
            weight_keys = set(weights.keys())
            provider_rules = []
            for model_rule in weight_keys:
                provider_rules.extend(await get_provider_rules(model_rule, config, request_model))
            provider_list = get_provider_list(provider_rules, config, request_model)
            weight_keys = set([provider['provider'] + "/" + request_model for provider in provider_list])
            # print("all_providers", all_providers)
            # print("weights", weights)
            # print("weight_keys", weight_keys)

            # 步骤 3: 计算交集
            intersection = all_providers.intersection(weight_keys)
            # print("intersection", intersection)
            if len(intersection) == 1:
                intersection = None

        if intersection:
            filtered_weights = {k.split("/")[0]: v for k, v in weights.items() if k.split("/")[0] + "/" + request_model in intersection}
            # print("filtered_weights", filtered_weights)

            if scheduling_algorithm == "weighted_round_robin":
                weighted_provider_name_list = weighted_round_robin(filtered_weights)
            elif scheduling_algorithm == "lottery":
                weighted_provider_name_list = lottery_scheduling(filtered_weights)
            else:
                weighted_provider_name_list = list(filtered_weights.keys())
            # print("weighted_provider_name_list", weighted_provider_name_list)

            new_matching_providers = []
            for provider_name in weighted_provider_name_list:
                for provider in matching_providers:
                    if provider['provider'] == provider_name:
                        new_matching_providers.append(provider)
            matching_providers = new_matching_providers

    if is_debug:
        for provider in matching_providers:
            logger.info("available provider: %s", json.dumps(provider, indent=4, ensure_ascii=False, default=circular_list_encoder))

    return matching_providers

import asyncio
class ModelRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_model(self, request_data: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], api_index: int, background_tasks: BackgroundTasks, endpoint=None):
        config = app.state.config
        request_model_name = request_data.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        scheduling_algorithm = safe_get(config, 'api_keys', api_index, "preferences", "SCHEDULING_ALGORITHM", default="fixed_priority")

        matching_providers = await get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm)
        num_matching_providers = len(matching_providers)

        status_code = 500
        error_message = None

        start_index = 0
        if scheduling_algorithm != "fixed_priority":
            async with self.locks[request_model_name]:
                self.last_provider_indices[request_model_name] = (self.last_provider_indices[request_model_name] + 1) % num_matching_providers
                start_index = self.last_provider_indices[request_model_name]

        auto_retry = safe_get(config, 'api_keys', api_index, "preferences", "AUTO_RETRY", default=True)
        role = safe_get(config, 'api_keys', api_index, "role", default=safe_get(config, 'api_keys', api_index, "api", default="None")[:8])

        index = 0
        if num_matching_providers == 1 and (count := provider_api_circular_list[matching_providers[0]['provider']].get_items_count()) > 1:
            retry_count = count
        else:
            tmp_retry_count = sum(provider_api_circular_list[provider['provider']].get_items_count() for provider in matching_providers) * 2
            if tmp_retry_count < 10:
                retry_count = tmp_retry_count
            else:
                retry_count = 10

        while True:
            # print("start_index", start_index)
            # print("index", index)
            # print("num_matching_providers", num_matching_providers)
            # print("retry_count", retry_count)
            if index > num_matching_providers + retry_count:
                break
            current_index = (start_index + index) % num_matching_providers
            index += 1
            provider = matching_providers[current_index]

            provider_name = provider['provider']
            # print("current_index", current_index)
            # print("provider_name", provider_name)

            # 检查是否所有API密钥都被速率限制,如果被速率限制，则跳出循环
            model_dict = provider["_model_dict_cache"]
            original_model = model_dict[request_model_name]
            if await provider_api_circular_list[provider_name].is_all_rate_limited(original_model):
                # logger.warning(f"Provider {provider_name}: All API keys are rate limited and stop auto retry!")
                error_message = f"All API keys are rate limited and stop auto retry!"
                if num_matching_providers == 1:
                    break
                else:
                    continue

            original_request_model = (original_model, request_data.model)
            if provider_name.startswith("sk-") and provider_name in app.state.api_list:
                local_provider_api_index = app.state.api_list.index(provider_name)
                local_provider_scheduling_algorithm = safe_get(config, 'api_keys', local_provider_api_index, "preferences", "SCHEDULING_ALGORITHM", default="fixed_priority")
                local_provider_matching_providers = await get_right_order_providers(request_model_name, config, local_provider_api_index, local_provider_scheduling_algorithm)
                local_timeout_value = 0
                for local_provider in local_provider_matching_providers:
                    local_provider_name = local_provider['provider']
                    if not local_provider_name.startswith("sk-"):
                        local_timeout_value += get_preference(app.state.provider_timeouts, local_provider_name, original_request_model, DEFAULT_TIMEOUT)
                # print("local_timeout_value", provider_name, local_timeout_value)
                local_provider_num_matching_providers = len(local_provider_matching_providers)
            else:
                local_timeout_value = get_preference(app.state.provider_timeouts, provider_name, original_request_model, DEFAULT_TIMEOUT)
                local_provider_num_matching_providers = 1
                # print("local_timeout_value", provider_name, local_timeout_value)

            local_timeout_value = local_timeout_value * local_provider_num_matching_providers
            # print("local_timeout_value", provider_name, local_timeout_value)

            keepalive_interval = get_preference(app.state.keepalive_interval, provider_name, original_request_model, 99999)
            if keepalive_interval > local_timeout_value:
                keepalive_interval = None
            if provider_name.startswith("sk-"):
                keepalive_interval = None
            # print("keepalive_interval", provider_name, keepalive_interval)

            try:
                response = await process_request(request_data, provider, background_tasks, endpoint, role, local_timeout_value, keepalive_interval)
                return response
            except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:

                # 根据异常类型设置状态码和错误消息
                if isinstance(e, httpx.ReadTimeout):
                    status_code = 504  # Gateway Timeout
                    timeout_value = e.request.extensions.get('timeout', {}).get('read', -1)
                    error_message = f"Request timed out after {timeout_value} seconds"
                elif isinstance(e, httpx.ConnectError):
                    status_code = 503  # Service Unavailable
                    error_message = "Unable to connect to service"
                elif isinstance(e, httpx.ReadError):
                    status_code = 502  # Bad Gateway
                    error_message = "Network read error"
                elif isinstance(e, httpx.RemoteProtocolError):
                    status_code = 502  # Bad Gateway
                    error_message = "Remote protocol error"
                elif isinstance(e, asyncio.CancelledError):
                    status_code = 499  # Client Closed Request
                    error_message = "Request was cancelled"
                elif isinstance(e, HTTPException):
                    status_code = e.status_code
                    error_message = str(e.detail)
                else:
                    status_code = 500  # Internal Server Error
                    error_message = str(e) or f"Unknown error: {e.__class__.__name__}"

                exclude_error_rate_limit = [
                    # "Internal Server Error",
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
                ]

                channel_id = provider['provider']

                if app.state.channel_manager.cooldown_period > 0 and num_matching_providers > 1 \
                and all(error not in error_message for error in exclude_error_rate_limit):
                    # 获取源模型名称（实际配置的模型名）
                    # source_model = list(provider['model'][0].keys())[0]
                    await app.state.channel_manager.exclude_model(channel_id, request_model_name)
                    matching_providers = await get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm)
                    last_num_matching_providers = num_matching_providers
                    num_matching_providers = len(matching_providers)
                    if num_matching_providers != last_num_matching_providers:
                        index = 0

                cooling_time = safe_get(provider, "preferences", "api_key_cooldown_period", default=0)
                api_key_count = provider_api_circular_list[channel_id].get_items_count()
                current_api = await provider_api_circular_list[channel_id].after_next_current()

                if cooling_time > 0 and api_key_count > 1 \
                and all(error not in error_message for error in exclude_error_rate_limit):
                    await provider_api_circular_list[channel_id].set_cooling(current_api, cooling_time=cooling_time)

                # 有些错误并没有请求成功，所以需要删除请求记录
                if current_api and any(error in error_message for error in exclude_error_rate_limit) and provider_api_circular_list[provider_name].requests[current_api][original_model]:
                    provider_api_circular_list[provider_name].requests[current_api][original_model].pop()

                if "string_above_max_length" in error_message:
                    status_code = 413
                if "must be less than max_seq_len" in error_message:
                    status_code = 413
                if "Please reduce the length of the messages or completion" in error_message:
                    status_code = 413
                # gemini
                if "exceeds the maximum number of tokens allowed" in error_message:
                    status_code = 413
                if "'reason': 'API_KEY_INVALID'" in error_message or "API key not valid" in error_message:
                    status_code = 401
                if "User location is not supported for the API use." in error_message:
                    status_code = 403
                if "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in error_message:
                    status_code = 403


                logger.error(f"Error {status_code} with provider {channel_id} API key: {current_api}: {error_message}")
                if is_debug:
                    import traceback
                    traceback.print_exc()

                if auto_retry and (status_code not in [400, 413] or urlparse(provider.get('base_url', '')).netloc == 'models.inference.ai.azure.com'):
                    continue
                else:
                    return JSONResponse(
                        status_code=status_code,
                        content={"error": f"Error: Current provider response failed: {error_message}"}
                    )

        current_info = request_info.get()
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        return JSONResponse(
            status_code=status_code,
            content={"error": f"All {request_data.model} error: {error_message}"}
        )

model_handler = ModelRequestHandler()

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = app.state.api_list
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

def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = app.state.api_list
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
    if app.state.api_keys_db[api_index].get('role') != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    return token

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def chat_completions_route(request: RequestModel, background_tasks: BackgroundTasks, api_index: int = Depends(verify_api_key)):
    return await model_handler.request_model(request, api_index, background_tasks)

# @app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
# async def options_handler():
#     return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(api_index: int = Depends(verify_api_key)):
    models = post_all_models(api_index, app.state.config, app.state.api_list, app.state.models_list)
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

from fastapi import UploadFile, File, Form, HTTPException, Request
import io
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
def generate_api_key():
    # Define the character set (only alphanumeric)
    chars = string.ascii_letters + string.digits
    # Generate a random string of 36 characters
    random_string = ''.join(secrets.choice(chars) for _ in range(48))
    api_key = "sk-" + random_string
    return JSONResponse(content={"api_key": api_key})

# 在 /stats 路由中返回成功和失败百分比
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, desc, case
from fastapi import Query

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
                func.sum(case((ChannelStat.success == True, 1), else_=0)).label('success_count')
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
                func.sum(case((ChannelStat.success == True, 1), else_=0)).label('success_count')
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
        app.state.config, app.state.api_keys_db, app.state.api_list = update_config(app.state.config, use_config_url=False)
    return JSONResponse(content={"message": "API config updated"})

from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, field_serializer
from typing import Dict, Union, Optional, List, Any

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

from fastapi.staticfiles import StaticFiles
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