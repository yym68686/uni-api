from log_config import logger

import re
import httpx
import secrets
import time as time_module
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from models import RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, UnifiedRequest
from request import get_payload
from response import fetch_response, fetch_response_stream
from utils import error_handling_wrapper, post_all_models, load_config, safe_get, circular_list_encoder

from collections import defaultdict
from typing import List, Dict, Union
from urllib.parse import urlparse

import os
import string
import json

is_debug = bool(os.getenv("DEBUG", False))
# is_debug = False

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes

async def create_tables():
    async with engine.begin() as conn:
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
    await create_tables()

    TIMEOUT = float(os.getenv("TIMEOUT", 100))
    timeout = httpx.Timeout(connect=15.0, read=TIMEOUT, write=30.0, pool=30.0)
    default_headers = {
        "User-Agent": "curl/7.68.0",  # 模拟 curl 的 User-Agent
        "Accept": "*/*",  # curl 的默认 Accept 头
    }
    app.state.client = httpx.AsyncClient(
        timeout=timeout,
        headers=default_headers,
        http2=True,  # 禁用 HTTP/2
        verify=True,  # 保持 SSL 验证（如需禁用，设为 False，但不建议）
        follow_redirects=True,  # 自动跟随重定向
    )
    # app.state.client = httpx.AsyncClient(timeout=timeout)
    app.state.config, app.state.api_keys_db, app.state.api_list = await load_config(app)
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan, debug=is_debug)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        logger.error(f"404 Error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

import uuid
import json
import asyncio
from time import time
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

# 获取数据库路径
db_path = os.getenv('DB_PATH', './data/stats.db')

# 确保 data 目录存在
data_dir = os.path.dirname(db_path)
os.makedirs(data_dir, exist_ok=True)

# 创建异步引擎和会话
# engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=False)
engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=is_debug)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

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
        await self.update_stats()

    async def update_stats(self):
        # 这里添加更新数据库的逻辑
        # print("current_info2")
        async with async_session() as session:
            async with session.begin():
                try:
                    columns = [column.key for column in RequestStat.__table__.columns]
                    filtered_info = {k: v for k, v in self.current_info.items() if k in columns}
                    new_request_stat = RequestStat(**filtered_info)
                    session.add(new_request_stat)
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Error updating stats: {str(e)}")

    async def _logging_iterator(self):
        try:
            async for chunk in self.body_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode('utf-8')
                line = chunk.decode()
                if is_debug:
                    logger.info(f"{line}")
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

    async def update_stats(self, current_info):
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

    async def update_channel_stats(self, request_id, provider, model, api_key, success):
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

# 在 process_request 函数中更新成功和失败计数
async def process_request(request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest], provider: Dict, endpoint=None, token=None):
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

    if "claude" not in provider['model'][request.model] \
    and "gpt" not in provider['model'][request.model] \
    and "gemini" not in provider['model'][request.model] \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in provider['model'][request.model] and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in provider['model'][request.model] and engine == "vertex":
        engine = "vertex-gemini"

    if "o1-preview" in provider['model'][request.model] or "o1-mini" in provider['model'][request.model]:
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

    if provider.get("engine"):
        engine = provider["engine"]

    logger.info(f"provider: {provider['provider']:<10} model: {request.model:<10} engine: {engine}")

    url, headers, payload = await get_payload(request, engine, provider)
    if is_debug:
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        if payload.get("file"):
            pass
        else:
            logger.info(json.dumps(payload, indent=4, ensure_ascii=False))
    current_info = request_info.get()
    try:
        if request.stream:
            model = provider['model'][request.model]
            generator = fetch_response_stream(app.state.client, url, headers, payload, engine, model)
            wrapped_generator, first_response_time = await error_handling_wrapper(generator)
            response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
        else:
            generator = fetch_response(app.state.client, url, headers, payload)
            wrapped_generator, first_response_time = await error_handling_wrapper(generator)
            first_element = await anext(wrapped_generator)
            first_element = first_element.lstrip("data: ")
            first_element = json.loads(first_element)
            response = StarletteStreamingResponse(iter([json.dumps(first_element)]), media_type="application/json")
            # response = JSONResponse(first_element)

        # 更新成功计数和首次响应时间
        await app.middleware_stack.app.update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=True)
        current_info["first_response_time"] = first_response_time
        current_info["success"] = True
        current_info["provider"] = provider['provider']

        return response
    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        await app.middleware_stack.app.update_channel_stats(current_info["request_id"], provider['provider'], request.model, token, success=False)

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

import asyncio
class ModelRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    def get_matching_providers(self, model_name, token):
        config = app.state.config
        # api_keys_db = app.state.api_keys_db
        api_list = app.state.api_list
        api_index = api_list.index(token)
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail="No matching model found")
        provider_rules = []

        for model in config['api_keys'][api_index]['model']:
            if model == "all":
                # 如果模型名为 *，则返回所有模型
                for provider in config["providers"]:
                    for model in provider["model"].keys():
                        provider_rules.append(provider["provider"] + "/" + model)
                break
            if "/" in model:
                if model.startswith("<") and model.endswith(">"):
                    model = model[1:-1]
                    # 处理带斜杠的模型名
                    for provider in config['providers']:
                        if model in provider['model'].keys():
                            provider_rules.append(provider['provider'] + "/" + model)
                else:
                    provider_name = model.split("/")[0]
                    model_name_split = "/".join(model.split("/")[1:])
                    models_list = []
                    for provider in config['providers']:
                        if provider['provider'] == provider_name:
                            models_list.extend(list(provider['model'].keys()))
                    # print("models_list", models_list)
                    # print("model_name", model_name)
                    # print("model_name_split", model_name_split)
                    # print("model", model)
                    if model_name_split == "*":
                        if model_name in models_list:
                            provider_rules.append(provider_name)
                    elif model_name_split == model_name:
                        if model_name in models_list:
                            provider_rules.append(provider_name)
            else:
                for provider in config['providers']:
                    if model in provider['model'].keys():
                        provider_rules.append(provider['provider'] + "/" + model)

        provider_list = []
        # print("provider_rules", provider_rules)
        for item in provider_rules:
            for provider in config['providers']:
                # print("provider", provider, provider['provider'] == item, item)
                if "/" in item:
                    if provider['provider'] == item.split("/")[0]:
                        if model_name in provider['model'].keys() and "/".join(item.split("/")[1:]) == model_name:
                            provider_list.append(provider)
                elif provider['provider'] == item:
                    if model_name in provider['model'].keys():
                        provider_list.append(provider)
                else:
                    pass

                # if provider['provider'] == item:
                #     if "/" in item:
                #         if item.split("/")[1] == model_name:
                #             provider_list.append(provider)
                #     else:
                #         if model_name in provider['model'].keys():
                #             provider_list.append(provider)
        if is_debug:
            for provider in provider_list:
                logger.info("available provider: %s", json.dumps(provider, indent=4, ensure_ascii=False, default=circular_list_encoder))
        return provider_list

    async def request_model(self, request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest], token: str, endpoint=None):
        config = app.state.config
        # api_keys_db = app.state.api_keys_db
        api_list = app.state.api_list

        model_name = request.model
        matching_providers = self.get_matching_providers(model_name, token)
        # import json
        # print("matching_providers", json.dumps(matching_providers, indent=4, ensure_ascii=False))
        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")
        # exit(0)
        # 检查是否启用轮询
        api_index = api_list.index(token)
        weights = safe_get(config, 'api_keys', api_index, "weights")
        if weights:
            # 步骤 1: 提取 matching_providers 中的所有 provider 值
            providers = set(provider['provider'] for provider in matching_providers)
            weight_keys = set(weights.keys())

            # 步骤 3: 计算交集
            intersection = providers.intersection(weight_keys)
            weights = dict(filter(lambda item: item[0] in intersection, weights.items()))
            weighted_provider_name_list = weighted_round_robin(weights)
            new_matching_providers = []
            for provider_name in weighted_provider_name_list:
                for provider in matching_providers:
                    if provider['provider'] == provider_name:
                        new_matching_providers.append(provider)
            matching_providers = new_matching_providers

        # import json
        # print("matching_providers", json.dumps(matching_providers, indent=4, ensure_ascii=False, default=circular_list_encoder))
        use_round_robin = True
        auto_retry = True
        if safe_get(config, 'api_keys', api_index, "preferences", "USE_ROUND_ROBIN") == False:
            use_round_robin = False
        if safe_get(config, 'api_keys', api_index, "preferences", "AUTO_RETRY") == False:
            auto_retry = False

        return await self.try_all_providers(request, matching_providers, use_round_robin, auto_retry, endpoint, token)

    # 在 try_all_providers 函数中处理失败的情况
    async def try_all_providers(self, request: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest], providers: List[Dict], use_round_robin: bool, auto_retry: bool, endpoint: str = None, token: str = None):
        status_code = 500
        error_message = None
        num_providers = len(providers)
        model_name = request.model

        if use_round_robin:
            async with self.locks[model_name]:
                self.last_provider_indices[model_name] = (self.last_provider_indices[model_name] + 1) % num_providers
                start_index = self.last_provider_indices[model_name]
        else:
            start_index = 0

        for i in range(num_providers + 1):
            current_index = (start_index + i) % num_providers
            provider = providers[current_index]
            try:
                response = await process_request(request, provider, endpoint, token)
                return response
            except HTTPException as e:
                logger.error(f"Error with provider {provider['provider']}: {str(e)}")
                status_code = e.status_code
                error_message = e.detail

                if auto_retry:
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Error: Current provider response failed: {error_message}")
            except (Exception, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                logger.error(f"Error with provider {provider['provider']}: {str(e)}")
                if is_debug:
                    import traceback
                    traceback.print_exc()
                error_message = str(e)
                if auto_retry:
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Error: Current provider response failed: {error_message}")

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
        now = time_module.time()
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

@app.get("/generate-api-key", dependencies=[Depends(rate_limit_dependency)])
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

@app.get("/stats", dependencies=[Depends(rate_limit_dependency)])
async def get_stats(
    request: Request,
    token: str = Depends(verify_admin_api_key),
    hours: int = Query(default=24, ge=1, le=720, description="Number of hours to look back for stats (1-720)")
):
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