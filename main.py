from log_config import logger

import copy
import httpx
import secrets
from time import time
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from models import RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, TextToSpeechRequest, UnifiedRequest, EmbeddingRequest
from request import get_payload
from response import fetch_response, fetch_response_stream
from utils import (
    safe_get,
    get_proxy,
    get_engine,
    load_config,
    get_model_dict,
    post_all_models,
    circular_list_encoder,
    error_handling_wrapper,
    provider_api_circular_list,
    ThreadSafeCircularList,
)

from collections import defaultdict
from typing import Dict, Union
from urllib.parse import urlparse

import os
import string
import json

DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
is_debug = bool(os.getenv("DEBUG", False))
# is_debug = False

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes

# 添加新的环境变量检查
DISABLE_DATABASE = os.getenv("DISABLE_DATABASE", "false").lower() == "true"
IS_VERCEL = os.path.dirname(os.path.abspath(__file__)).startswith('/var/task')
logger.info("IS_VERCEL: %s", IS_VERCEL)
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

        # 检查并添加缺失的列
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    if not DISABLE_DATABASE:
        await create_tables()

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
        logger.error(f"404 Error: {exc.detail}")
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

    async def is_model_excluded(self, provider: str, model: str) -> bool:
        model_key = f"{provider}/{model}"
        excluded_time = self._excluded_models[model_key]
        if not excluded_time:
            return False

        if datetime.now() - excluded_time > timedelta(seconds=self.cooldown_period):
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

            # 检查该模型是否被排除
            if not await self.is_model_excluded(provider_name, target_model):
                available_providers.append(provider)

        return available_providers

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, select, Boolean, Text
from sqlalchemy.sql import func

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
    provider = Column(String)
    model = Column(String)
    # success = Column(Boolean, default=False)
    api_key = Column(String)
    is_flagged = Column(Boolean, default=False)
    text = Column(Text)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    # cost = Column(Float, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ChannelStat(Base):
    __tablename__ = 'channel_stats'
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    provider = Column(String)
    model = Column(String)
    api_key = Column(String)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


if not DISABLE_DATABASE:
    # 获取数据库路径
    db_path = os.getenv('DB_PATH', './data/stats.db')

    # 确保 data 目录存在
    data_dir = os.path.dirname(db_path)
    os.makedirs(data_dir, exist_ok=True)

    # 创建异步引擎和会话
    # db_engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=False)
    db_engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=is_debug)
    async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

from starlette.types import Scope, Receive, Send
from starlette.responses import Response

from asyncio import Semaphore

# 创建一个信号量来控制数据库访问
db_semaphore = Semaphore(1)  # 限制同时只有1个写入操作

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
                        new_request_stat = RequestStat(**filtered_info)
                        session.add(new_request_stat)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error updating stats: {str(e)}")
                        if is_debug:
                            import traceback
                            traceback.print_exc()
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
        try:
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
                if not line.startswith("[DONE]") and not line.startswith("OK"):
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
        except Exception as e:
            raise
        finally:
            logger.debug("_logging_iterator finished")

    async def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self.body_iterator, 'aclose'):
                await self.body_iterator.aclose()

class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time()

        enable_moderation = False  # 默认不开启道德审查

        config = app.state.config
        # 根据token决定是否启用道德审查
        if request.headers.get("x-api-key"):
            token = request.headers.get("x-api-key")
        elif request.headers.get("Authorization"):
            api_split_list = request.headers.get("Authorization").split(" ")
            if len(api_split_list) > 1:
                token = api_split_list[1]
            else:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid or missing API Key"}
                )
        else:
            token = None

        api_index = None
        if token:
            try:
                api_list = app.state.api_list
                api_index = api_list.index(token)
            except ValueError:
                # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
                api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
                # token不在api_list中，使用默认值（不开启）
                pass

            if api_index is not None:
                enable_moderation = safe_get(config, 'api_keys', api_index, "preferences", "ENABLE_MODERATION", default=False)
            else:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid or missing API Key"}
                )
        else:
            # 如果token为None，检查全局设置
            enable_moderation = config.get('ENABLE_MODERATION', False)

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
            # "cost": 0,
            "total_tokens": 0
        }

        # 设置请求信息到上下文
        current_request_info = request_info.set(request_info_data)
        current_info = request_info.get()

        parsed_body = await parse_request_body(request)
        if parsed_body and not request.url.path.startswith("/v1/api_config"):
            try:
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
                    moderation_response = await self.moderate_content(moderated_content, api_index)
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
            except RequestValidationError:
                logger.error(f"Invalid request body: {parsed_body}")
                pass
            except Exception as e:
                if is_debug:
                    import traceback
                    traceback.print_exc()

                logger.error(f"Error processing request or performing moral check: {str(e)}")

        try:
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
        finally:
            # print("current_request_info", current_request_info)
            request_info.reset(current_request_info)

    async def moderate_content(self, content, api_index):
        moderation_request = ModerationRequest(input=content)

        # 直接调用 moderations 函数
        response = await moderations(moderation_request, api_index)

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

        if client_key not in self.clients or IS_VERCEL:
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
            if client_key in self.clients:
                tmp_client = self.clients[client_key]
                del self.clients[client_key]  # 先删除引用
                await tmp_client.aclose()  # 然后关闭客户端
            raise e

    async def close(self):
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

@app.middleware("http")
async def ensure_config(request: Request, call_next):

    if app and not hasattr(app.state, 'config'):
        # logger.warning("Config not found, attempting to reload")
        app.state.config, app.state.api_keys_db, app.state.api_list = await load_config(app)

        if app.state.api_list:
            app.state.user_api_keys_rate_limit = defaultdict(ThreadSafeCircularList)
            for api_index, api_key in enumerate(app.state.api_list):
                app.state.user_api_keys_rate_limit[api_key] = ThreadSafeCircularList(
                    [api_key],
                    safe_get(app.state.config, 'api_keys', api_index, "preferences", "rate_limit", default={"default": "999999/min"}),
                    "round_robin"
                )

        for item in app.state.api_keys_db:
            if item.get("role") == "admin":
                app.state.admin_api_key = item.get("api")
        if not hasattr(app.state, "admin_api_key"):
            if len(app.state.api_keys_db) >= 1:
                app.state.admin_api_key = app.state.api_keys_db[0].get("api")
            else:
                from utils import yaml_error_message
                if yaml_error_message:
                    return JSONResponse(
                        status_code=500,
                        content={"error": yaml_error_message}
                    )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "No admin API key found"}
                    )

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

        # 存储超时配置
        app.state.timeouts = {}
        if app.state.config and 'preferences' in app.state.config:
            if isinstance(app.state.config['preferences'].get('model_timeout'), int):
                app.state.timeouts["default"] = app.state.config['preferences'].get('model_timeout')
            else:
                for model_name, timeout_value in app.state.config['preferences'].get('model_timeout', {"default": DEFAULT_TIMEOUT}).items():
                    app.state.timeouts[model_name] = timeout_value
                if "default" not in app.state.config['preferences'].get('model_timeout', {}):
                    app.state.timeouts["default"] = DEFAULT_TIMEOUT

        app.state.provider_timeouts = defaultdict(lambda: defaultdict(lambda: DEFAULT_TIMEOUT))
        for provider in app.state.config["providers"]:
            # print("provider", provider)
            provider_timeout_settings = safe_get(provider, "preferences", "model_timeout", default={})
            # print("provider_timeout_settings", provider_timeout_settings)
            if provider_timeout_settings:
                for model_name, timeout_value in provider_timeout_settings.items():
                    app.state.provider_timeouts[provider['provider']][model_name] = timeout_value

        app.state.provider_timeouts["global_time_out"] = app.state.timeouts

        # provider_timeouts_dict = {
        #     provider: dict(timeouts)
        #     for provider, timeouts in app.state.provider_timeouts.items()
        # }
        # print("app.state.provider_timeouts", provider_timeouts_dict)
        # print("ai" in app.state.provider_timeouts)

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

def get_timeout_value(provider_timeouts, original_model):
    timeout_value = None
    original_model = original_model.lower()
    if original_model in provider_timeouts:
        timeout_value = provider_timeouts[original_model]
    else:
        # 尝试模糊匹配模型
        for timeout_model in provider_timeouts:
            if timeout_model != "default" and timeout_model in original_model:
                timeout_value = provider_timeouts[timeout_model]
                break
        else:
            # 如果模糊匹配失败，使用渠道的默认值
            timeout_value = provider_timeouts.get("default")
    return timeout_value

# 在 process_request 函数中更新成功和失败计数
async def process_request(request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], provider: Dict, endpoint=None, role=None, num_matching_providers=1):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    engine, stream_mode = get_engine(provider, endpoint, original_model)

    if stream_mode != None:
        request.stream = stream_mode

    channel_id = f"{provider['provider']}"
    if engine != "moderation":
        logger.info(f"provider: {channel_id:<11} model: {request.model:<22} engine: {engine} role: {role}")

    url, headers, payload = await get_payload(request, engine, provider)
    headers.update(safe_get(provider, "preferences", "headers", default={}))  # add custom headers
    if is_debug:
        logger.info(url)
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        if payload.get("file"):
            pass
        else:
            logger.info(json.dumps(payload, indent=4, ensure_ascii=False))

    current_info = request_info.get()

    provider_timeouts = safe_get(app.state.provider_timeouts, channel_id, default=app.state.provider_timeouts["global_time_out"])
    timeout_value = get_timeout_value(provider_timeouts, original_model)
    if timeout_value is None:
        timeout_value = get_timeout_value(app.state.provider_timeouts["global_time_out"], original_model)
    if timeout_value is None:
        timeout_value = app.state.timeouts.get("default", DEFAULT_TIMEOUT)
    timeout_value = timeout_value * num_matching_providers
    # print("timeout_value", channel_id, timeout_value)

    proxy = safe_get(app.state.config, "preferences", "proxy", default=None)  # global proxy
    proxy = safe_get(provider, "preferences", "proxy", default=proxy)  # provider proxy
    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(timeout_value, url, proxy) as client:
            if request.stream:
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, request.stream, app.state.error_triggers)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:
                generator = fetch_response(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, request.stream, app.state.error_triggers)

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
            await update_channel_stats(current_info["request_id"], channel_id, request.model, current_info["api_key"], success=True)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = channel_id
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        await update_channel_stats(current_info["request_id"], channel_id, request.model, current_info["api_key"], success=False)
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
            model_dict = get_model_dict(provider)
            for model in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model)

    elif "/" in model_rule:
        if model_rule.startswith("<") and model_rule.endswith(">"):
            model_rule = model_rule[1:-1]
            # 处理带斜杠的模型名
            for provider in config['providers']:
                model_dict = get_model_dict(provider)
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
                    model_dict = get_model_dict(provider)
                    if provider['provider'] == provider_name:
                        models_list.extend(list(model_dict.keys()))

            # print("models_list", models_list)
            # print("model_name", model_name)
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
            model_dict = get_model_dict(provider)
            if model_rule in model_dict.keys():
                provider_rules.append(provider["provider"] + "/" + model_rule)

    return provider_rules

def get_provider_list(provider_rules, config, request_model):
    provider_list = []
    # print("provider_rules", provider_rules)
    for item in provider_rules:
        provider_name = item.split("/")[0]
        if provider_name.startswith("sk-") and provider_name in app.state.api_list:
            provider_list.append({"provider": provider_name, "base_url": "http://127.0.0.1:8000/v1/chat/completions", "model": [{request_model: request_model}], "tools": True})
        else:
            for provider in config['providers']:
                model_dict = get_model_dict(provider)
                model_name_split = "/".join(item.split("/")[1:])
                if "/" in item and provider['provider'] == provider_name and model_name_split in model_dict.keys():
                    if request_model in model_dict.keys() and model_name_split == request_model:
                        new_provider = copy.deepcopy(provider)
                        new_provider["model"] = [{model_dict[model_name_split]: request_model}]
                        provider_list.append(new_provider)

                    elif request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*")):
                        new_provider = copy.deepcopy(provider)
                        new_provider["model"] = [{model_dict[model_name_split]: request_model}]
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

    if not matching_providers:
        raise HTTPException(status_code=404, detail=f"No matching model found: {request_model}")

    num_matching_providers = len(matching_providers)
    if app.state.channel_manager.cooldown_period > 0 and num_matching_providers > 1:
        matching_providers = await app.state.channel_manager.get_available_providers(matching_providers)
        num_matching_providers = len(matching_providers)
        if not matching_providers:
            raise HTTPException(status_code=503, detail="No available providers at the moment")

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

    async def request_model(self, request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], api_index: int = None, endpoint=None):
        config = app.state.config
        request_model = request.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model}")

        scheduling_algorithm = safe_get(config, 'api_keys', api_index, "preferences", "SCHEDULING_ALGORITHM", default="fixed_priority")

        matching_providers = await get_right_order_providers(request_model, config, api_index, scheduling_algorithm)
        num_matching_providers = len(matching_providers)

        status_code = 500
        error_message = None

        start_index = 0
        if scheduling_algorithm != "fixed_priority":
            async with self.locks[request_model]:
                self.last_provider_indices[request_model] = (self.last_provider_indices[request_model] + 1) % num_matching_providers
                start_index = self.last_provider_indices[request_model]

        auto_retry = safe_get(config, 'api_keys', api_index, "preferences", "AUTO_RETRY", default=True)
        role = safe_get(config, 'api_keys', api_index, "role", default=safe_get(config, 'api_keys', api_index, "api", default="None")[:8])

        index = 0
        if num_matching_providers == 1 and (count := provider_api_circular_list[matching_providers[0]['provider']].get_items_count()) > 1:
            retry_count = count
        else:
            retry_count = int(auto_retry)

        while True:
            # print("start_index", start_index)
            # print("index", index)
            # print("num_matching_providers", num_matching_providers)
            # print("retry_count", retry_count)
            if index >= num_matching_providers + retry_count:
                break
            current_index = (start_index + index) % num_matching_providers
            index += 1
            provider = matching_providers[current_index]

            if provider['provider'].startswith("sk-") and provider['provider'] in app.state.api_list:
                local_provider_api_index = app.state.api_list.index(provider['provider'])
                local_provider_scheduling_algorithm = safe_get(config, 'api_keys', local_provider_api_index, "preferences", "SCHEDULING_ALGORITHM", default="fixed_priority")
                local_provider_matching_providers = await get_right_order_providers(request_model, config, local_provider_api_index, local_provider_scheduling_algorithm)
                local_provider_num_matching_providers = len(local_provider_matching_providers)
            else:
                local_provider_num_matching_providers = 1

            try:
                response = await process_request(request, provider, endpoint, role, local_provider_num_matching_providers)
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

                channel_id = f"{provider['provider']}"
                if app.state.channel_manager.cooldown_period > 0 and num_matching_providers > 1:
                    # 获取源模型名称（实际配置的模型名）
                    # source_model = list(provider['model'][0].keys())[0]
                    await app.state.channel_manager.exclude_model(channel_id, request_model)
                    matching_providers = await get_right_order_providers(request_model, config, api_index, scheduling_algorithm)
                    last_num_matching_providers = num_matching_providers
                    num_matching_providers = len(matching_providers)
                    if num_matching_providers != last_num_matching_providers:
                        index = 0

                cooling_time = safe_get(provider, "preferences", "api_key_cooldown_period", default=0)
                api_key_count = provider_api_circular_list[channel_id].get_items_count()
                current_api = await provider_api_circular_list[channel_id].after_next_current()
                if cooling_time > 0 and api_key_count > 1:
                    await provider_api_circular_list[channel_id].set_cooling(current_api, cooling_time=cooling_time)

                logger.error(f"Error {status_code} with provider {channel_id} API key: {current_api}: {error_message}")
                if is_debug:
                    import traceback
                    traceback.print_exc()
                if auto_retry and (status_code != 413 or urlparse(provider.get('base_url', '')).netloc == 'models.inference.ai.azure.com'):
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
            content={"error": f"All {request.model} error: {error_message}"}
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
        api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
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
        api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    # for api_key in app.state.api_keys_db:
    #     if token.startswith(api_key['api']):
    if app.state.api_keys_db[api_index].get('role') != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    return token

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, api_index: int = Depends(verify_api_key)):
    return await model_handler.request_model(request, api_index)

@app.options("/v1/chat/completions")
async def options_handler():
    return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models")
async def list_models(api_index: int = Depends(verify_api_key)):
    models = post_all_models(api_index, app.state.config, app.state.api_list, app.state.models_list)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.post("/v1/images/generations")
async def images_generations(
    request: ImageGenerationRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/images/generations")

@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/embeddings")

@app.post("/v1/audio/speech")
async def audio_speech(
    request: TextToSpeechRequest,
    api_index: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/audio/speech")

@app.post("/v1/moderations")
async def moderations(
    request: ModerationRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/moderations")

from fastapi import UploadFile, File, Form, HTTPException
import io
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    api_index: int = Depends(verify_api_key)
):
    try:
        # 读取上传的文件内容
        content = await file.read()
        file_obj = io.BytesIO(content)

        # 创建AudioTranscriptionRequest对象
        request = AudioTranscriptionRequest(
            file=(file.filename, file_obj, file.content_type),
            model=model
        )

        return await model_handler.request_model(request, api_index, endpoint="/v1/audio/transcriptions")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid audio file encoding")
    except Exception as e:
        if is_debug:
            import traceback
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@app.get("/v1/generate-api-key")
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

@app.get("/v1/stats")
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

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Hello, World!"})

# async def on_fetch(request, env):
#     import asgi
#     return await asgi.fetch(app, request, env)

@app.get("/v1/api_config")
async def api_config(api_index: int = Depends(verify_api_key)):
    return JSONResponse(content={"api_config": app.state.config})

@app.post("/v1/api_config/update")
async def api_config_update(api_index: int = Depends(verify_api_key), config: dict = Body(...)):
    if "providers" in config:
        app.state.config["providers"] = config["providers"]
    return JSONResponse(content={"message": "API config updated"})

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
        ws="none",
        # log_level="warning"
    )
