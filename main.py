from log_config import logger

import re
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
from utils import error_handling_wrapper, post_all_models, load_config, safe_get, circular_list_encoder, get_model_dict, save_api_yaml

from collections import defaultdict
from typing import List, Dict, Union
from urllib.parse import urlparse

import os
import string
import json

DEFAULT_TIMEOUT = float(os.getenv("TIMEOUT", 100))
is_debug = bool(os.getenv("DEBUG", False))
# is_debug = False

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes

# 添加新的环境变量检查
DISABLE_DATABASE = os.getenv("DISABLE_DATABASE", "false").lower() == "true"

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

async def update_stats(current_info):
    if DISABLE_DATABASE:
        return
    # 这里添加更新数据库的逻辑
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

async def update_channel_stats(request_id, provider, model, api_key, success):
    if DISABLE_DATABASE:
        return
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
            token = request.headers.get("Authorization").split(" ")[1]
        else:
            token = None
        if token:
            try:
                api_list = app.state.api_list
                api_index = api_list.index(token)
                enable_moderation = safe_get(config, 'api_keys', api_index, "preferences", "ENABLE_MODERATION", default=False)
            except ValueError:
                # token不在api_list中，使用默认值（不开启）
                pass
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

                if request_model.request_type == "chat":
                    moderated_content = request_model.get_last_text_message()
                elif request_model.request_type == "image":
                    moderated_content = request_model.prompt
                if moderated_content:
                    current_info["text"] = moderated_content


                if enable_moderation and moderated_content:
                    moderation_response = await self.moderate_content(moderated_content, token)
                    is_flagged = moderation_response.get('results', [{}])[0].get('flagged', False)

                    if is_flagged:
                        logger.error(f"Content did not pass the moral check: %s", moderated_content)
                        process_time = time() - start_time
                        current_info["process_time"] = process_time
                        current_info["is_flagged"] = is_flagged
                        await self.update_stats(current_info)
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

                logger.error(f"处理请求或进行道德检查时出错: {str(e)}")

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

    async def moderate_content(self, content, token):
        moderation_request = ModerationRequest(input=content)

        # 直接调用 moderations 函数
        response = await moderations(moderation_request, token)

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
        self.clients = {}  # {timeout_value: AsyncClient}
        self.locks = {}    # {timeout_value: Lock}

    async def init(self, default_config):
        self.default_config = default_config

    @asynccontextmanager
    async def get_client(self, timeout_value):
        # 对同一超时值的客户端加锁
        if timeout_value not in self.locks:
            self.locks[timeout_value] = asyncio.Lock()

        async with self.locks[timeout_value]:
            # 获取或创建指定超时值的客户端
            if timeout_value not in self.clients:
                timeout = httpx.Timeout(
                    connect=15.0,
                    read=timeout_value,
                    write=30.0,
                    pool=self.pool_size
                )
                self.clients[timeout_value] = httpx.AsyncClient(
                    timeout=timeout,
                    limits=httpx.Limits(max_connections=self.pool_size),
                    **self.default_config
                )

            try:
                yield self.clients[timeout_value]
            except Exception as e:
                # 如果客户端出现问题，关闭并重新创建
                await self.clients[timeout_value].aclose()
                del self.clients[timeout_value]
                raise e

    async def close(self):
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

@app.middleware("http")
async def ensure_config(request: Request, call_next):
    if not hasattr(app.state, 'config'):
        logger.warning("Config not found, attempting to reload")
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

        print("app.state.timeouts", app.state.timeouts)

    return await call_next(request)

# 在 process_request 函数中更新成功和失败计数
async def process_request(request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], provider: Dict, endpoint=None, token=None):
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
    if "claude" not in model_dict[request.model] \
    and "gpt" not in model_dict[request.model] \
    and "gemini" not in model_dict[request.model] \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in model_dict[request.model] and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in model_dict[request.model] and engine == "vertex":
        engine = "vertex-gemini"

    if "o1-preview" in model_dict[request.model] or "o1-mini" in model_dict[request.model]:
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

    logger.info(f"provider: {provider['provider']:<11} model: {request.model:<22} engine: {engine}")

    url, headers, payload = await get_payload(request, engine, provider)
    if is_debug:
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        if payload.get("file"):
            pass
        else:
            logger.info(json.dumps(payload, indent=4, ensure_ascii=False))

    current_info = request_info.get()
    model = model_dict[request.model]

    timeout_value = None
    # 先尝试精确匹配

    if model in app.state.timeouts:
        timeout_value = app.state.timeouts[model]
    else:
        # 如果没有精确匹配，尝试模糊匹配
        for timeout_model in app.state.timeouts:
            if timeout_model in model:
                timeout_value = app.state.timeouts[timeout_model]
                break

    # 如果都没匹配到，使用默认值
    if timeout_value is None:
        timeout_value = app.state.timeouts.get("default", DEFAULT_TIMEOUT)

    try:
        async with app.state.client_manager.get_client(timeout_value) as client:
            if request.stream:
                generator = fetch_response_stream(client, url, headers, payload, engine, model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:
                generator = fetch_response(client, url, headers, payload, engine, model)
                wrapped_generator, first_response_time = await error_handling_wrapper(generator)
                first_element = await anext(wrapped_generator)
                first_element = first_element.lstrip("data: ")
                # print("first_element", first_element)
                first_element = json.loads(first_element)
                response = StarletteStreamingResponse(iter([json.dumps(first_element)]), media_type="application/json")
                # response = JSONResponse(first_element)

            # 更新成功计数和首次响应时间
            await update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=True)
            # await app.middleware_stack.app.update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=True)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = provider['provider']
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
        await update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=False)
        # await app.middleware_stack.app.update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=False)

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
        # 如果模型名为 all，则返回所有模型
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

import asyncio
class ModelRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_model(self, request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest], token: str, endpoint=None):
        config = app.state.config
        api_list = app.state.api_list
        api_index = api_list.index(token)

        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail="No matching model found")

        request_model = request.model
        matching_providers = get_matching_providers(request_model, config, api_index)
        num_matching_providers = len(matching_providers)

        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")

        # 检查是否启用轮询
        scheduling_algorithm = safe_get(config, 'api_keys', api_index, "preferences", "SCHEDULING_ALGORITHM", default="fixed_priority")
        if scheduling_algorithm == "random":
            matching_providers = random.sample(matching_providers, num_matching_providers)

        weights = safe_get(config, 'api_keys', api_index, "weights")

        # 步骤 1: 提取 matching_providers 中的所有 provider 值
        # print("matching_providers", matching_providers)
        # print(type(matching_providers[0]['model'][0].keys()), list(matching_providers[0]['model'][0].keys())[0], matching_providers[0]['model'][0].keys())
        all_providers = set(provider['provider'] + "/" + list(provider['model'][0].keys())[0] for provider in matching_providers)

        intersection = None
        if weights and all_providers:
            weight_keys = set(weights.keys())
            provider_rules = []
            for model_rule in weight_keys:
                provider_rules.extend(get_provider_rules(model_rule, config, request_model))
            provider_list = get_provider_list(provider_rules, config, request_model)
            weight_keys = set([provider['provider'] + "/" + list(provider['model'][0].keys())[0] for provider in provider_list])
            # print("all_providers", all_providers)
            # print("weights", weights)
            # print("weight_keys", weight_keys)

            # 步骤 3: 计算交集
            intersection = all_providers.intersection(weight_keys)
            # print("intersection", intersection)

        if weights and intersection:
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

        status_code = 500
        error_message = None

        start_index = 0
        if scheduling_algorithm != "fixed_priority":
            async with self.locks[request_model]:
                self.last_provider_indices[request_model] = (self.last_provider_indices[request_model] + 1) % num_matching_providers
                start_index = self.last_provider_indices[request_model]

        auto_retry = safe_get(config, 'api_keys', api_index, "preferences", "AUTO_RETRY", default=True)

        for i in range(num_matching_providers + 1):
            current_index = (start_index + i) % num_matching_providers
            provider = matching_providers[current_index]
            try:
                response = await process_request(request, provider, endpoint, token)
                return response
            except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:

                # 根据异常类型设置状态码和错误消息
                if isinstance(e, httpx.ReadTimeout):
                    status_code = 504  # Gateway Timeout
                    error_message = "Request timed out"
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

                logger.error(f"Error {status_code} with provider {provider['provider']}: {error_message}")
                if is_debug:
                    import traceback
                    traceback.print_exc()
                if auto_retry:
                    continue
                else:
                    raise HTTPException(status_code=status_code, detail=f"Error: Current provider response failed: {error_message}")

        current_info = request_info.get()
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        raise HTTPException(status_code=status_code, detail=f"All {request.model} error: {error_message}")

model_handler = ModelRequestHandler()

def parse_rate_limit(limit_string):
    # 定义时间单位到秒的映射
    time_units = {
        's': 1, 'sec': 1, 'second': 1,
        'm': 60, 'min': 60, 'minute': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
        'mo': 2592000, 'month': 2592000,
        'y': 31536000, 'year': 31536000
    }

    # 使用正则表达式匹配数字和单位
    match = re.match(r'^(\d+)/(\w+)$', limit_string)
    if not match:
        raise ValueError(f"Invalid rate limit format: {limit_string}")

    count, unit = match.groups()
    count = int(count)

    # 转换单位到秒
    if unit not in time_units:
        raise ValueError(f"Unknown time unit: {unit}")

    seconds = time_units[unit]

    return (count, seconds)

class InMemoryRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    async def is_rate_limited(self, key: str, limit: int, period: int) -> bool:
        now = time()
        self.requests[key] = [req for req in self.requests[key] if req > now - period]
        if len(self.requests[key]) >= limit:
            return True
        self.requests[key].append(now)
        return False

rate_limiter = InMemoryRateLimiter()

async def get_user_rate_limit(api_index: str = None):
    # 这里应该实现根据 token 获取用户速率限制的逻辑
    # 示例： 返回 (次数， 秒数)
    config = app.state.config
    raw_rate_limit = safe_get(config, 'api_keys', api_index, "preferences", "RATE_LIMIT")

    if not api_index or not raw_rate_limit:
        return (30, 60)

    rate_limit = parse_rate_limit(raw_rate_limit)
    return rate_limit

security = HTTPBearer()

async def rate_limit_dependency(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials if credentials else None
    api_list = app.state.api_list
    try:
        api_index = api_list.index(token)
    except ValueError:
        print("error: Invalid or missing API Key:", token)
        api_index = None
        token = None
    limit, period = await get_user_rate_limit(api_index)

    # 使用 IP 地址和 token（如果有）作为限制键
    client_ip = request.client.host
    rate_limit_key = f"{client_ip}:{token}" if token else client_ip

    if await rate_limiter.is_rate_limited(rate_limit_key, limit, period):
        raise HTTPException(status_code=429, detail="Too many requests")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = app.state.api_list
    token = credentials.credentials
    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token

def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = app.state.api_list
    token = credentials.credentials
    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    for api_key in app.state.api_keys_db:
        if api_key['api'] == token:
            if api_key.get('role') != "admin":
                raise HTTPException(status_code=403, detail="Permission denied")
    return token

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

@app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def options_handler():
    return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(token: str = Depends(verify_api_key)):
    models = post_all_models(token, app.state.config, app.state.api_list)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    token: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, token, endpoint="/v1/images/generations")

@app.post("/v1/embeddings", dependencies=[Depends(rate_limit_dependency)])
async def embeddings(
    request: EmbeddingRequest,
    token: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, token, endpoint="/v1/embeddings")

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    token: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, token, endpoint="/v1/moderations")

from fastapi import UploadFile, File, Form, HTTPException
import io
@app.post("/v1/audio/transcriptions", dependencies=[Depends(rate_limit_dependency)])
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    token: str = Depends(verify_api_key)
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

        return await model_handler.request_model(request, token, endpoint="/v1/audio/transcriptions")
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
from xue.components.menubar import (
    Menubar, MenubarMenu, MenubarTrigger, MenubarContent,
    MenubarItem, MenubarSeparator
)
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

    if x_api_key == app.state.admin_api_key:  # 替换为实际的管理员API密钥
        return x_api_key
    else:
        return None

async def frontend_rate_limit_dependency(request: Request, x_api_key: str = Depends(get_api_key)):
    token = x_api_key if x_api_key else None
    limit, period = 100, 60

    # 使用 IP 地址和 token（如果有）作为限制键
    client_ip = request.client.host
    rate_limit_key = f"{client_ip}:{token}" if token else client_ip

    if await rate_limiter.is_rate_limited(rate_limit_key, limit, period):
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
                    class_="ml-[240px] p-6 transition-[margin] duration-200 ease-in-out"
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

@frontend_router.get("/data", response_class=HTMLResponse, dependencies=[Depends(frontend_rate_limit_dependency)])
async def data_page(x_api_key: str = Depends(get_api_key)):
    if not x_api_key:
        return RedirectResponse(url="/login", status_code=303)

    if DISABLE_DATABASE:
        return HTMLResponse("数据库已禁用")

    async with async_session() as session:
        # 计算过去24小时的开始时间
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)

        # 获取每个模型的请求数据
        model_stats = await session.execute(
            select(
                RequestStat.model,
                RequestStat.provider,
                func.count().label('count')
            )
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.model, RequestStat.provider)
            .order_by(desc('count'))
        )
        model_stats = model_stats.fetchall()

    # 处理数据以适配图表格式
    chart_data = []
    providers = list(set(stat.provider for stat in model_stats))
    models = list(set(stat.model for stat in model_stats))

    for model in models:
        data_point = {"model": model}
        for provider in providers:
            count = next(
                (stat.count for stat in model_stats
                 if stat.model == model and stat.provider == provider),
                0
            )
            data_point[provider] = count
        chart_data.append(data_point)

    # 定义图表系列
    series = [
        {"name": provider, "data_key": provider}
        for provider in providers
    ]

    # 图表配置
    chart_config = {
        "stacked": True,  # 堆叠柱状图
        "horizontal": False,
        "colors": [f"hsl({i * 360 / len(providers)}, 70%, 50%)" for i in range(len(providers))],  # 生成不同的颜色
        "grid": True,
        "legend": True,
        "tooltip": True
    }

    result = HTML(
        Head(title="数据统计"),
        Body(
            Div(
                Div(
                    "模型使用统计 (24小时)",
                    class_="text-2xl font-bold mb-4"
                ),
                Div(
                    chart.bar_chart("model-usage-chart", chart_data, "model", series, chart_config),
                    class_="h-[600px]"  # 设置图表高度
                ),
                class_="container mx-auto p-4"
            )
        )
    ).render()

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