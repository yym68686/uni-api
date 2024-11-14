from log_config import logger

import copy
import httpx
import secrets
from time import time
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from models import RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, UnifiedRequest, EmbeddingRequest
from request import get_payload
from response import fetch_response, fetch_response_stream
from utils import (
    safe_get,
    load_config,
    save_api_yaml,
    get_model_dict,
    post_all_models,
    get_user_rate_limit,
    circular_list_encoder,
    error_handling_wrapper,
    rate_limiter,
    provider_api_circular_list,
)

from collections import defaultdict
from typing import List, Dict, Union
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
    # print("Main app routes:")
    # for route in app.routes:
    #     print(f"Route: {route.path}, methods: {route.methods}")

    # print("\nFrontend router routes:")
    # for route in frontend_router.routes:
    #     print(f"Route: {route.path}, methods: {route.methods}")

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

from decimal import Decimal, getcontext

# 设置全局精度
getcontext().prec = 17  # 设置为17是为了确保15位小数的精度

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    costs = {
        "gpt-4": {"input": Decimal('5.0') / Decimal('1000000'), "output": Decimal('15.0') / Decimal('1000000')},
        "claude-3-sonnet": {"input": Decimal('3.0') / Decimal('1000000'), "output": Decimal('15.0') / Decimal('1000000')}
    }

    if model not in costs:
        logger.error(f"Unknown model: {model}")
        return 0

    model_costs = costs[model]
    input_cost = Decimal(input_tokens) * model_costs["input"]
    output_cost = Decimal(output_tokens) * model_costs["output"]
    total_cost = input_cost + output_cost

    # 返回精确到15位小数的结果
    return total_cost.quantize(Decimal('0.000000000000001'))

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
        if parsed_body:
            try:
                request_model = UnifiedRequest.model_validate(parsed_body).data
                model = request_model.model
                current_info["model"] = model

                moderated_content = None
                if request_model.request_type == "chat":
                    moderated_content = request_model.get_last_text_message()
                elif request_model.request_type == "image":
                    moderated_content = request_model.prompt
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

            if request.url.path.startswith("/v1"):
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

            if proxy:
                # 解析代理URL
                parsed = urlparse(proxy)
                scheme = parsed.scheme.rstrip('h')

                if scheme == 'socks5':
                    try:
                        from httpx_socks import AsyncProxyTransport
                        proxy = proxy.replace('socks5h://', 'socks5://')
                        transport = AsyncProxyTransport.from_url(proxy)
                        client_config["transport"] = transport
                    except ImportError:
                        logger.error("httpx-socks package is required for SOCKS proxy support")
                        raise ImportError("Please install httpx-socks package for SOCKS proxy support: pip install httpx-socks")
                else:
                    client_config["proxies"] = {
                        "http://": proxy,
                        "https://": proxy
                    }

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

        for item in app.state.api_keys_db:
            if item.get("role") == "admin":
                app.state.admin_api_key = item.get("api")
        if not hasattr(app.state, "admin_api_key"):
            if len(app.state.api_keys_db) >= 1:
                app.state.admin_api_key = app.state.api_keys_db[0].get("api")
            else:
                raise Exception("No admin API key found")

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
            for model_name, timeout_value in app.state.config['preferences'].get('model_timeout', {}).items():
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

    return await call_next(request)

def get_timeout_value(provider_timeouts, original_model):
    timeout_value = None
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
async def process_request(request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], provider: Dict, endpoint=None):
    url = provider['base_url']
    parsed_url = urlparse(url)
    # print("parsed_url", parsed_url)
    engine = None
    if parsed_url.netloc == 'generativelanguage.googleapis.com':
        engine = "gemini"
    elif parsed_url.netloc == 'aiplatform.googleapis.com':
        engine = "vertex"
    elif parsed_url.netloc == 'api.cloudflare.com':
        engine = "cloudflare"
    elif parsed_url.netloc == 'api.anthropic.com' or parsed_url.path.endswith("v1/messages"):
        engine = "claude"
    elif parsed_url.netloc == 'openrouter.ai':
        engine = "openrouter"
    elif parsed_url.netloc == 'api.cohere.com':
        engine = "cohere"
        request.stream = True
    else:
        engine = "gpt"

    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    if "claude" not in original_model \
    and "gpt" not in original_model \
    and "gemini" not in original_model \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in original_model and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in original_model and engine == "vertex":
        engine = "vertex-gemini"

    if "o1-preview" in original_model or "o1-mini" in original_model:
        engine = "o1"
        request.stream = False

    if endpoint == "/v1/images/generations":
        engine = "dalle"
        request.stream = False

    if endpoint == "/v1/audio/transcriptions":
        engine = "whisper"
        request.stream = False

    if endpoint == "/v1/moderations":
        engine = "moderation"
        request.stream = False

    if endpoint == "/v1/embeddings":
        engine = "embedding"
        request.stream = False

    if provider.get("engine"):
        engine = provider["engine"]

    channel_id = f"{provider['provider']}"
    logger.info(f"provider: {channel_id:<11} model: {request.model:<22} engine: {engine}")

    url, headers, payload = await get_payload(request, engine, provider)
    if is_debug:
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
    # print("timeout_value", timeout_value)

    proxy = safe_get(provider, "preferences", "proxy", default=None)
    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(timeout_value, url, proxy) as client:
            # 打印client配置信息
            # logger.info(f"Client config - Timeout: {client.timeout}")
            # logger.info(f"Client config - Headers: {client.headers}")
            # if hasattr(client, '_transport'):
            #     if hasattr(client._transport, 'proxy_url'):
            #         logger.info(f"Client config - Proxy: {client._transport.proxy_url}")
            #     elif hasattr(client._transport, 'proxies'):
            #         logger.info(f"Client config - Proxies: {client._transport.proxies}")
            #     else:
            #         logger.info("Client config - No proxy configured")
            # else:
            #     logger.info("Client config - No transport configured")
            # logger.info(f"Client config - Follow Redirects: {client.follow_redirects}")
            if request.stream:
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:
                generator = fetch_response(client, url, headers, payload, engine, original_model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id)
                first_element = await anext(wrapped_generator)
                first_element = first_element.lstrip("data: ")
                # print("first_element", first_element)
                first_element = json.loads(first_element)
                response = StarletteStreamingResponse(iter([json.dumps(first_element)]), media_type="application/json")
                # response = JSONResponse(first_element)

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

def get_provider_rules(model_rule, config, request_model):
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
        for provider in config['providers']:
            model_dict = get_model_dict(provider)
            model_name_split = "/".join(item.split("/")[1:])
            if "/" in item and provider['provider'] == item.split("/")[0] and model_name_split in model_dict.keys():
                new_provider = copy.deepcopy(provider)
                # old: new
                # print("item", item)
                # print("model_dict", model_dict)
                # print("model_name_split", model_name_split)
                # print("request_model", request_model)
                new_provider["model"] = [{model_dict[model_name_split]: request_model}]
                if request_model in model_dict.keys() and model_name_split == request_model:
                    provider_list.append(new_provider)

                elif request_model.endswith("*") and model_name_split.startswith(request_model.rstrip("*")):
                    provider_list.append(new_provider)
    return provider_list

def get_matching_providers(request_model, config, api_index):
    provider_rules = []

    for model_rule in config['api_keys'][api_index]['model']:
        provider_rules.extend(get_provider_rules(model_rule, config, request_model))

    provider_list = get_provider_list(provider_rules, config, request_model)

    # print("provider_list", provider_list)
    return provider_list

async def get_right_order_providers(request_model, config, api_index, scheduling_algorithm):
    matching_providers = get_matching_providers(request_model, config, api_index)

    if not matching_providers:
        raise HTTPException(status_code=404, detail=f"No matching model found: {request_model}")

    num_matching_providers = len(matching_providers)
    if app.state.channel_manager.cooldown_period > 0 and num_matching_providers > 1:
        matching_providers = await app.state.channel_manager.get_available_providers(matching_providers)
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
                provider_rules.extend(get_provider_rules(model_rule, config, request_model))
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
            filtered_weights = {k.split("/")[0]: v for k, v in weights.items() if k in intersection}
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

        index = 0
        if num_matching_providers == 1 and (count := provider_api_circular_list[matching_providers[0]['provider']].get_items_count()) > 1:
            retry_count = count
        else:
            retry_count = 0

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
            try:
                response = await process_request(request, provider, endpoint)
                return response
            except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:

                # 根据异常类型设置状态码和错误消息
                if isinstance(e, httpx.ReadTimeout):
                    status_code = 504  # Gateway Timeout
                    error_message = "Request timed out"
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
                if auto_retry:
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

async def rate_limit_dependency(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials if credentials else None
    api_list = app.state.api_list
    try:
        api_index = api_list.index(token)
    except ValueError:
        # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
        api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
        if api_index is None:
            print("error: Invalid or missing API Key:", token)
            api_index = None
            token = None

    # 使用 IP 地址和 token（如果有）作为限制键
    client_ip = request.client.host
    rate_limit_key = f"{client_ip}:{token}" if token else client_ip

    limits = await get_user_rate_limit(app, api_index)
    if await rate_limiter.is_rate_limited(rate_limit_key, limits):
        raise HTTPException(status_code=429, detail="Too many requests")

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

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def request_model(request: RequestModel, api_index: int = Depends(verify_api_key)):
    return await model_handler.request_model(request, api_index)

@app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def options_handler():
    return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(api_index: int = Depends(verify_api_key)):
    models = post_all_models(api_index, app.state.config)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/images/generations")

@app.post("/v1/embeddings", dependencies=[Depends(rate_limit_dependency)])
async def embeddings(
    request: EmbeddingRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/embeddings")

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, endpoint="/v1/moderations")

from fastapi import UploadFile, File, Form, HTTPException
import io
@app.post("/v1/audio/transcriptions", dependencies=[Depends(rate_limit_dependency)])
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

@app.get("/v1/generate-api-key", dependencies=[Depends(rate_limit_dependency)])
def generate_api_key():
    # Define the character set (only alphanumeric)
    chars = string.ascii_letters + string.digits
    # Generate a random string of 36 characters
    random_string = ''.join(secrets.choice(chars) for _ in range(36))
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



from fastapi import FastAPI, Request
from fastapi import Form as FastapiForm, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from typing import Optional, List

from xue import HTML, Head, Body, Div, xue_initialize, Script, Ul, Li
from xue.components import input, dropdown, sheet, form, button, checkbox, sidebar, chart
from xue.components.model_config_row import model_config_row
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from components.provider_table import data_table

from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

frontend_router = APIRouter()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def get_api_key(request: Request, x_api_key: Optional[str] = Depends(api_key_header)):
    if not x_api_key:
        x_api_key = request.cookies.get("x_api_key") or request.query_params.get("x_api_key")
    # print(f"Cookie x_api_key: {request.cookies.get('x_api_key')}")  # 添加此行
    # print(f"Query param x_api_key: {request.query_params.get('x_api_key')}")  # 添加此行
    # print(f"Header x_api_key: {x_api_key}")  # 添加此行
    # logger.info(f"x_api_key: {x_api_key} {x_api_key == 'your_admin_api_key'}")

    if not hasattr(app.state, 'config'):
        await ensure_config(request, lambda: None)

    if x_api_key == app.state.admin_api_key:  # 替换为实际的管理员API密钥
        return x_api_key
    else:
        return None

async def frontend_rate_limit_dependency(request: Request, x_api_key: str = Depends(get_api_key)):
    token = x_api_key if x_api_key else None

    # 使用 IP 地址和 token（如果有）作为限制键
    client_ip = request.client.host
    rate_limit_key = f"{client_ip}:{token}" if token else client_ip

    limits = [(100, 60)]
    if await rate_limiter.is_rate_limited(rate_limit_key, limits):
        raise HTTPException(status_code=429, detail="Too many requests")

# def get_backend_router_api_list():
#     api_list = []
#     for route in frontend_router.routes:
#         api_list.append({
#             "path": f"/api{route.path}",  # 加上前缀
#             "method": route.methods,
#             "name": route.name,
#             "summary": route.summary
#         })
#     return api_list

# @app.get("/backend-router-api-list")
# async def backend_router_api_list():
#     return get_backend_router_api_list()

xue_initialize(tailwind=True)

data_table_columns = [
    # {"label": "Status", "value": "status", "sortable": True},
    {"label": "Provider", "value": "provider", "sortable": True},
    {"label": "Base url", "value": "base_url", "sortable": True},
    # {"label": "Engine", "value": "engine", "sortable": True},
    {"label": "Tools", "value": "tools", "sortable": True},
]

@frontend_router.get("/login", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def login_page():
    return HTML(
        Head(title="登录"),
        Body(
            Div(
                form.Form(
                    form.FormField("API Key", "x_api_key", type="password", placeholder="输入API密钥", required=True),
                    Div(id="error-message", class_="text-red-500 mt-2"),
                    Div(
                        button.button("提交", variant="primary", type="submit"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post="/verify-api-key",
                    hx_target="#error-message",
                    hx_swap="innerHTML",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-md"
            )
        )
    ).render()


@frontend_router.post("/verify-api-key", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def verify_api_key(x_api_key: str = FastapiForm(...)):
    if x_api_key == app.state.admin_api_key:  # 替换为实际的管理员API密钥
        response = JSONResponse(content={"success": True})
        response.headers["HX-Redirect"] = "/"  # 添加这一行
        response.set_cookie(
            key="x_api_key",
            value=x_api_key,
            httponly=True,
            max_age=1800,  # 30分钟
            secure=False,  # 在开发环境中设置为False，生产环境中使用HTTPS时设置为True
            samesite="lax"  # 改为"lax"以允许重定向时携带cookie
        )
        return response
    else:
        return Div("无效的API密钥", class_="text-red-500").render()

# 添加侧边栏配置
sidebar_items = [
    {
        "icon": "layout-dashboard",
        # "label": "仪表盘",
        "label": "Dashboard",
        "value": "dashboard",
        "hx": {"get": "/dashboard", "target": "#main-content"}
    },
    # {
    #     "icon": "settings",
    #     # "label": "设置",
    #     "label": "Settings",
    #     "value": "settings",
    #     "hx": {"get": "/settings", "target": "#main-content"}
    # },
    {
        "icon": "database",
        # "label": "数据",
        "label": "Data",
        "value": "data",
        "hx": {"get": "/data", "target": "#main-content"}
    },
    # {
    #     "icon": "scroll-text",
    #     # "label": "日志",
    #     "label": "Logs",
    #     "value": "logs",
    #     "hx": {"get": "/logs", "target": "#main-content"}
    # }
]

@frontend_router.get("/", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def root(x_api_key: str = Depends(get_api_key)):
    if not x_api_key:
        return RedirectResponse(url="/login", status_code=303)

    result = HTML(
        Head(
            Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    const filterInput = document.getElementById('users-table-filter');
                    filterInput.addEventListener('input', function() {
                        const filterValue = this.value;
                        htmx.ajax('GET', `/filter-table?filter=${filterValue}`, '#users-table');
                    });
                });
            """),
            title="uni-api"
        ),
        Body(
            Div(
                sidebar.Sidebar("zap", "uni-api", sidebar_items, is_collapsed=False, active_item="dashboard"),
                Div(
                    Div(
                        data_table(data_table_columns, app.state.config["providers"], "users-table"),
                        class_="p-4"
                    ),
                    Div(id="sheet-container"),  # sheet加载位置
                    id="main-content",
                    class_="ml-[200px] p-6 transition-[margin] duration-200 ease-in-out"
                ),
                class_="flex"
            ),
            class_="container mx-auto",
            id="body"
        )
    ).render()
    # print(result)
    return result

@frontend_router.get("/sidebar/toggle", response_class=HTMLResponse)
async def toggle_sidebar(is_collapsed: bool = False):
    return sidebar.Sidebar(
        "zap",
        "uni-api",
        sidebar_items,
        is_collapsed=not is_collapsed,
        active_item="dashboard"
    ).render()

@app.get("/sidebar/update/{active_item}", response_class=HTMLResponse)
async def update_sidebar(active_item: str):
    return sidebar.Sidebar(
        "zap",
        "uni-api",
        sidebar_items,
        is_collapsed=False,
        active_item=active_item
    ).render()

@frontend_router.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def data_page(x_api_key: str = Depends(get_api_key)):
    if not x_api_key:
        return RedirectResponse(url="/login", status_code=303)

    result = Div(
        Div(
            data_table(data_table_columns, app.state.config["providers"], "users-table"),
            class_="p-4"
        ),
        Div(id="sheet-container"),  # sheet加载位置
        id="main-content",
        class_="ml-[200px] p-6 transition-[margin] duration-200 ease-in-out"
    ).render()

    return result

@frontend_router.get("/data", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def data_page(x_api_key: str = Depends(get_api_key)):
    if not x_api_key:
        return RedirectResponse(url="/login", status_code=303)

    if DISABLE_DATABASE:
        return HTMLResponse("数据库已禁用")

    async with async_session() as session:
        # 计算过去24小时的开始时间
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)

        # 按小时统计每个模型的请求数据
        model_stats = await session.execute(
            select(
                func.strftime('%H', RequestStat.timestamp).label('hour'),
                RequestStat.model,
                func.count().label('count')
            )
            .where(RequestStat.timestamp >= start_time)
            .group_by('hour', RequestStat.model)
            .order_by('hour')
        )
        model_stats = model_stats.fetchall()

    # 获取所有唯一的模型名称
    models = list(set(stat.model for stat in model_stats))

    # 生成24小时的数据点
    chart_data = []
    current_hour = datetime.now().hour

    for i in range(24):
        # 计算小时标签(从当前小时往前推24小时)
        hour = (current_hour - i) % 24
        hour_str = f"{hour:02d}"

        # 创建该小时的数据点
        data_point = {"label": hour_str}

        # 添加每个模型在该小时的请求数
        for model in models:
            count = next(
                (stat.count for stat in model_stats
                 if stat.hour == f"{hour:02d}" and stat.model == model),
                0
            )
            data_point[model] = count

        chart_data.append(data_point)

    # 反转数据点顺序使其按时间正序显示
    chart_data.reverse()

    # 为每个模型配置显示属性
    chart_config = {
        model: {
            "label": model,
            "color": f"hsl({i * 360 / len(models)}, 70%, 50%)"  # 为每个模型生成不同的颜色
        }
        for i, model in enumerate(models)
    }

    result = HTML(
        Head(title="数据统计"),
        Body(
            Div(
                # 堆叠柱状图
                Div(
                    "模型使用统计 (24小时) - 按小时统计",
                    class_="text-2xl font-bold mb-4"
                ),
                Div(
                    chart.chart(
                        chart_data,
                        chart_config,
                        stacked=True,
                    ),
                    class_="mb-8 h-[400px]"  # 添加固定高度
                ),
                id="main-content",
                class_="container ml-[200px] mx-auto p-4"
            )
        )
    ).render()
    print(result)

    return result

@frontend_router.get("/dropdown-menu/{menu_id}/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def get_columns_menu(menu_id: str, row_id: str):
    columns = [
        {
            "label": "Edit",
            "value": "edit",
            "hx-get": f"/edit-sheet/{row_id}",
            "hx-target": "#sheet-container",
            "hx-swap": "innerHTML"
        },
        {
            "label": "Duplicate",
            "value": "duplicate",
            "hx-post": f"/duplicate/{row_id}",
            "hx-target": "body",
            "hx-swap": "outerHTML"
        },
        {
            "label": "Delete",
            "value": "delete",
            "hx-delete": f"/delete/{row_id}",
            "hx-target": "body",
            "hx-swap": "outerHTML",
            "hx-confirm": "Are you sure you want to delete this configuration?"
        },
    ]
    result = dropdown.dropdown_menu_content(menu_id, columns).render()
    print(result)
    return result

@frontend_router.get("/dropdown-menu/{menu_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def get_columns_menu(menu_id: str):
    result = dropdown.dropdown_menu_content(menu_id, data_table_columns).render()
    print(result)
    return result

@frontend_router.get("/filter-table", response_class=HTMLResponse)
async def filter_table(filter: str = ""):
    filtered_data = [
        (i, provider) for i, provider in enumerate(app.state.config["providers"])
        if filter.lower() in str(provider["provider"]).lower() or
           filter.lower() in str(provider["base_url"]).lower() or
           filter.lower() in str(provider["tools"]).lower()
    ]
    return data_table(data_table_columns, [p for _, p in filtered_data], "users-table", with_filter=False, row_ids=[i for i, _ in filtered_data]).render()

@frontend_router.post("/add-model", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def add_model():
    new_model_id = f"model{hash(str(time()))}"  # 生成一个唯一的ID
    new_model = model_config_row(new_model_id).render()
    return new_model

def render_api_keys(row_id, api_keys):
    return Ul(
        *[Li(
            Div(
                Div(
                    input.input(
                        type="text",
                        placeholder="Enter API key",
                        value=api_key,
                        name=f"api_key_{i}",
                        class_="flex-grow w-full"
                    ),
                    class_="flex-grow"
                ),
                button.button(
                    "Delete",
                    variant="outline",
                    type="button",
                    class_="ml-2",
                    hx_delete=f"/delete-api-key/{row_id}/{i}",
                    hx_target="#api-keys-container",
                    hx_swap="outerHTML"
                ),
                class_="flex items-center mb-2 w-full"
            )
        ) for i, api_key in enumerate(api_keys)],
        id="api-keys-container",
        class_="space-y-2 w-full"
    )

@frontend_router.get("/edit-sheet/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def get_edit_sheet(row_id: str, x_api_key: str = Depends(get_api_key)):
    row_data = get_row_data(row_id)
    print("row_data", row_data)

    model_list = []
    for index, model in enumerate(row_data["model"]):
        if isinstance(model, str):
            model_list.append(model_config_row(f"model{index}", model, "", True))
        if isinstance(model, dict):
            # print("model", model, list(model.items())[0])
            key, value = list(model.items())[0]
            model_list.append(model_config_row(f"model{index}", key, value, True))

    # 处理多个 API keys
    api_keys = row_data["api"] if isinstance(row_data["api"], list) else [row_data["api"]]
    api_key_inputs = render_api_keys(row_id, api_keys)

    sheet_id = "edit-sheet"
    edit_sheet_content = sheet.SheetContent(
        sheet.SheetHeader(
            sheet.SheetTitle("Edit Item"),
            sheet.SheetDescription("Make changes to your item here.")
        ),
        sheet.SheetBody(
            Div(
                form.Form(
                    form.FormField("Provider", "provider", value=row_data["provider"], placeholder="Enter provider name", required=True),
                    form.FormField("Base URL", "base_url", value=row_data["base_url"], placeholder="Enter base URL", required=True),
                    # form.FormField("API Key", "api_key", value=row_data["api"], type="text", placeholder="Enter API key"),
                    Div(
                        Div("API Keys", class_="text-lg font-semibold mb-2"),
                        api_key_inputs,
                        button.button(
                            "Add API Key",
                            class_="mt-2",
                            hx_post=f"/add-api-key/{row_id}",
                            hx_target="#api-keys-container",
                            hx_swap="outerHTML"
                        ),
                        class_="mb-4"
                    ),
                    Div(
                        Div("Models", class_="text-lg font-semibold mb-2"),
                        Div(
                            *model_list,
                            id="models-container",
                            class_="space-y-2 max-h-[40vh] overflow-y-auto"
                        ),
                        button.button(
                            "Add Model",
                            class_="mt-2",
                            hx_post="/add-model",
                            hx_target="#models-container",
                            hx_swap="beforeend"
                        ),
                        class_="mb-4"
                    ),
                    Div(
                        checkbox.checkbox("tools", "Enable Tools", checked=row_data["tools"], name="tools"),
                        class_="mb-4"
                    ),
                    form.FormField("Notes", "notes", value=row_data.get("notes", ""), placeholder="Enter any additional notes"),
                    Div(
                        button.button("Submit", variant="primary", type="submit"),
                        button.button("Cancel", variant="outline", type="button", class_="ml-2", onclick=f"toggleSheet('{sheet_id}')"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post=f"/submit/{row_id}",
                    hx_swap="outerHTML",
                    hx_target="body",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-2xl"
            )
        ),
        class_="max-h-[90vh] overflow-y-auto"
    )

    result = sheet.Sheet(
        sheet_id,
        Div(),
        edit_sheet_content,
        width="80%",
        max_width="800px"
    ).render()
    return result

@frontend_router.post("/add-api-key/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def add_api_key(row_id: str):
    row_data = get_row_data(row_id)
    api_keys = row_data["api"] if isinstance(row_data["api"], list) else [row_data["api"]]
    api_keys.append("")  # 添加一个空的API key

    api_key_inputs = render_api_keys(row_id, api_keys)

    return api_key_inputs.render()

@frontend_router.delete("/delete-api-key/{row_id}/{index}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def delete_api_key(row_id: str, index: int):
    row_data = get_row_data(row_id)
    api_keys = row_data["api"] if isinstance(row_data["api"], list) else [row_data["api"]]
    if len(api_keys) > 1:
        del api_keys[index]

    api_key_inputs = render_api_keys(row_id, api_keys)

    return api_key_inputs.render()

@frontend_router.get("/add-provider-sheet", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def get_add_provider_sheet():
    sheet_id = "add-provider-sheet"
    edit_sheet_content = sheet.SheetContent(
        sheet.SheetHeader(
            sheet.SheetTitle("Add New Provider"),
            sheet.SheetDescription("Enter details for the new provider.")
        ),
        sheet.SheetBody(
            Div(
                form.Form(
                    form.FormField("Provider", "provider", placeholder="Enter provider name", required=True),
                    form.FormField("Base URL", "base_url", placeholder="Enter base URL", required=True),
                    form.FormField("API Key", "api_key", type="text", placeholder="Enter API key"),
                    Div(
                        Div("Models", class_="text-lg font-semibold mb-2"),
                        Div(id="models-container"),
                        button.button(
                            "Add Model",
                            class_="mt-2",
                            hx_post="/add-model",
                            hx_target="#models-container",
                            hx_swap="beforeend"
                        ),
                        class_="mb-4"
                    ),
                    Div(
                        checkbox.checkbox("tools", "Enable Tools", name="tools"),
                        class_="mb-4"
                    ),
                    form.FormField("Notes", "notes", placeholder="Enter any additional notes"),
                    Div(
                        button.button("Submit", variant="primary", type="submit"),
                        button.button("Cancel", variant="outline", type="button", class_="ml-2", onclick=f"toggleSheet('{sheet_id}')"),
                        class_="flex justify-end mt-4"
                    ),
                    hx_post="/submit/new",
                    hx_swap="outerHTML",
                    hx_target="body",
                    class_="space-y-4"
                ),
                class_="container mx-auto p-4 max-w-2xl"
            )
        )
    )

    result = sheet.Sheet(
        sheet_id,
        Div(),
        edit_sheet_content,
        width="80%",
        max_width="800px"
    ).render()
    return result

def get_row_data(row_id):
    index = int(row_id)
    # print(app.state.config["providers"])
    return app.state.config["providers"][index]

def update_row_data(row_id, updated_data):
    print(row_id, updated_data)
    index = int(row_id)
    app.state.config["providers"][index] = updated_data

@frontend_router.post("/submit/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def submit_form(
    row_id: str,
    request: Request,
    provider: str = FastapiForm(...),
    base_url: str = FastapiForm(...),
    # api_key: Optional[str] = FastapiForm(None),
    tools: Optional[str] = FastapiForm(None),
    notes: Optional[str] = FastapiForm(None),
    x_api_key: str = Depends(get_api_key)
):
    form_data = await request.form()

    api_keys = [value for key, value in form_data.items() if key.startswith("api_key_") and value]

    # 收集模型数据
    models = []
    for key, value in form_data.items():
        if key.startswith("model_name_"):
            model_id = key.split("_")[-1]
            enabled = form_data.get(f"model_enabled_{model_id}") == "on"
            rename = form_data.get(f"model_rename_{model_id}")
            if value:
                if rename:
                    models.append({value: rename})
                else:
                    models.append(value)

    updated_data = {
        "provider": provider,
        "base_url": base_url,
        "api": api_keys[0] if len(api_keys) == 1 else api_keys,  # 如果只有一个 API key，就不使用列表
        "model": models,
        "tools": tools == "on",
        "notes": notes,
    }

    print("updated_data", updated_data)

    if row_id == "new":
        # 添加新提供者
        app.state.config["providers"].append(updated_data)
    else:
        # 更新现有提供者
        update_row_data(row_id, updated_data)

    # 保存更新后的配置
    if not DISABLE_DATABASE:
        save_api_yaml(app.state.config)

    return await root()

@frontend_router.post("/duplicate/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def duplicate_row(row_id: str):
    index = int(row_id)
    original_data = app.state.config["providers"][index]
    new_data = original_data.copy()
    new_data["provider"] += "-copy"
    app.state.config["providers"].insert(index + 1, new_data)

    # 保存更新后的配置
    if not DISABLE_DATABASE:
        save_api_yaml(app.state.config)

    return await root()

@frontend_router.delete("/delete/{row_id}", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def delete_row(row_id: str):
    index = int(row_id)
    del app.state.config["providers"][index]

    # 保存更新后的配置
    if not DISABLE_DATABASE:
        save_api_yaml(app.state.config)

    return await root()

app.include_router(frontend_router, tags=["frontend"])

# async def on_fetch(request, env):
#     import asgi
#     return await asgi.fetch(app, request, env)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        reload_includes=["*.py", "api.yaml"],
        ws="none",
        # log_level="warning"
    )