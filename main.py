from log_config import logger

import re
import httpx
import secrets
import time as time_module
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError

from models import RequestModel, ImageGenerationRequest
from request import get_payload
from response import fetch_response, fetch_response_stream
from utils import error_handling_wrapper, post_all_models, load_config, safe_get, circular_list_encoder

from collections import defaultdict
from typing import List, Dict, Union
from urllib.parse import urlparse

import os
is_debug = os.getenv("DEBUG", False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码

    # # 启动事件
    # routes = [{"path": route.path, "name": route.name} for route in app.routes]
    # logger.info(f"Registered routes: {routes}")

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

import asyncio
from time import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from datetime import timedelta
import json
import aiofiles

async def parse_request_body(request: Request):
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        try:
            return await request.json()
        except json.JSONDecodeError:
            return None
    return None

class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exclude_paths=None, save_interval=3600, filename="stats.json"):
        super().__init__(app)
        self.request_counts = defaultdict(int)
        self.request_times = defaultdict(float)
        self.ip_counts = defaultdict(lambda: defaultdict(int))
        self.request_arrivals = defaultdict(list)
        self.channel_success_counts = defaultdict(int)
        self.model_counts = defaultdict(int)
        self.channel_failure_counts = defaultdict(int)
        self.lock = asyncio.Lock()
        self.exclude_paths = set(exclude_paths or [])
        self.save_interval = save_interval
        self.filename = filename
        self.last_save_time = time()

        # 启动定期保存和清理任务
        asyncio.create_task(self.periodic_save_and_cleanup())

    async def dispatch(self, request: Request, call_next):
        arrival_time = datetime.now()
        start_time = time()

        # 使用依赖注入获取预解析的请求体
        request.state.parsed_body = await parse_request_body(request)

        model = "unknown"
        if request.state.parsed_body:
            try:
                request_model = RequestModel(**request.state.parsed_body)
                model = request_model.model
            except RequestValidationError:
                pass
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")

        response = await call_next(request)
        process_time = time() - start_time

        endpoint = f"{request.method} {request.url.path}"
        client_ip = request.client.host

        if request.url.path not in self.exclude_paths:
            async with self.lock:
                self.request_counts[endpoint] += 1
                self.request_times[endpoint] += process_time
                self.ip_counts[endpoint][client_ip] += 1
                self.request_arrivals[endpoint].append(arrival_time)
                if model != "unknown":
                    self.model_counts[model] += 1

        return response

    async def periodic_save_and_cleanup(self):
        while True:
            await asyncio.sleep(self.save_interval)
            await self.save_stats()
            await self.cleanup_old_data()

    async def save_stats(self):
        current_time = time()
        if current_time - self.last_save_time < self.save_interval:
            return

        async with self.lock:
            stats = {
                "request_counts": dict(self.request_counts),
                "request_times": dict(self.request_times),
                "model_counts": dict(self.model_counts),
                "ip_counts": {k: dict(v) for k, v in self.ip_counts.items()},
                "request_arrivals": {k: [t.isoformat() for t in v] for k, v in self.request_arrivals.items()},
                "channel_success_counts": dict(self.channel_success_counts),
                "channel_failure_counts": dict(self.channel_failure_counts),
                "channel_success_percentages": self.calculate_success_percentages(),
                "channel_failure_percentages": self.calculate_failure_percentages()
            }

        filename = self.filename
        async with aiofiles.open(filename, mode='w') as f:
            await f.write(json.dumps(stats, indent=2))

        self.last_save_time = current_time

    def calculate_success_percentages(self):
        percentages = {}
        for channel, success_count in self.channel_success_counts.items():
            total_count = success_count + self.channel_failure_counts[channel]
            if total_count > 0:
                percentages[channel] = success_count / total_count * 100
            else:
                percentages[channel] = 0

        sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
        return sorted_percentages

    def calculate_failure_percentages(self):
        percentages = {}
        for channel, failure_count in self.channel_failure_counts.items():
            total_count = failure_count + self.channel_success_counts[channel]
            if total_count > 0:
                percentages[channel] = failure_count / total_count * 100
            else:
                percentages[channel] = 0

        sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
        return sorted_percentages

    async def cleanup_old_data(self):
        cutoff_time = datetime.now() - timedelta(hours=24)
        async with self.lock:
            for endpoint in list(self.request_arrivals.keys()):
                self.request_arrivals[endpoint] = [
                    t for t in self.request_arrivals[endpoint] if t > cutoff_time
                ]
                if not self.request_arrivals[endpoint]:
                    del self.request_arrivals[endpoint]
                    self.request_counts.pop(endpoint, None)
                    self.request_times.pop(endpoint, None)
                    self.ip_counts.pop(endpoint, None)

    async def cleanup(self):
        await self.save_stats()

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部字段
)

app.add_middleware(StatsMiddleware, exclude_paths=["/stats", "/generate-api-key"])

# 在 process_request 函数中更新成功和失败计数
async def process_request(request: Union[RequestModel, ImageGenerationRequest], provider: Dict, endpoint=None):
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

    if provider.get("engine"):
        engine = provider["engine"]

    logger.info(f"provider: {provider['provider']:<10} model: {request.model:<10} engine: {engine}")

    url, headers, payload = await get_payload(request, engine, provider)
    if is_debug:
        logger.info(json.dumps(headers, indent=4, ensure_ascii=False))
        logger.info(json.dumps(payload, indent=4, ensure_ascii=False))
    try:
        if request.stream:
            model = provider['model'][request.model]
            generator = fetch_response_stream(app.state.client, url, headers, payload, engine, model)
            wrapped_generator = await error_handling_wrapper(generator)
            response = StreamingResponse(wrapped_generator, media_type="text/event-stream")
        else:
            generator = fetch_response(app.state.client, url, headers, payload)
            wrapped_generator = await error_handling_wrapper(generator)
            first_element = await anext(wrapped_generator)
            first_element = first_element.lstrip("data: ")
            first_element = json.loads(first_element)
            response = JSONResponse(first_element)

        # 更新成功计数
        async with app.middleware_stack.app.lock:
            app.middleware_stack.app.channel_success_counts[provider['provider']] += 1

        return response
    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        # 更新失败计数
        async with app.middleware_stack.app.lock:
            app.middleware_stack.app.channel_failure_counts[provider['provider']] += 1

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
        self.last_provider_index = -1

    def get_matching_providers(self, model_name, token):
        config = app.state.config
        # api_keys_db = app.state.api_keys_db
        api_list = app.state.api_list
        api_index = api_list.index(token)
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail="No matching model found")
        provider_rules = []

        for model in config['api_keys'][api_index]['model']:
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
                    # print("model", model)
                    if (model_name_split and model_name in models_list) or (model_name_split == "*" and model_name in models_list):
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
            import json
            for provider in provider_list:
                print(json.dumps(provider, indent=4, ensure_ascii=False, default=circular_list_encoder))
        return provider_list

    async def request_model(self, request: Union[RequestModel, ImageGenerationRequest], token: str, endpoint=None):
        config = app.state.config
        # api_keys_db = app.state.api_keys_db
        api_list = app.state.api_list

        model_name = request.model
        matching_providers = self.get_matching_providers(model_name, token)
        # import json
        # print("matching_providers", json.dumps(matching_providers, indent=4, ensure_ascii=False))
        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")

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

        return await self.try_all_providers(request, matching_providers, use_round_robin, auto_retry, endpoint)

    # 在 try_all_providers 函数中处理失败的情况
    async def try_all_providers(self, request: Union[RequestModel, ImageGenerationRequest], providers: List[Dict], use_round_robin: bool, auto_retry: bool, endpoint: str = None):
        status_code = 500
        error_message = None
        num_providers = len(providers)
        start_index = self.last_provider_index + 1 if use_round_robin else 0
        for i in range(num_providers + 1):
            self.last_provider_index = (start_index + i) % num_providers
            provider = providers[self.last_provider_index]
            try:
                response = await process_request(request, provider, endpoint)
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
                error_message = str(e)
                if auto_retry:
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Error: Current provider response failed: {error_message}")

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
async def request_model(request: Union[RequestModel, ImageGenerationRequest], token: str = Depends(verify_api_key)):
    # logger.info(f"Request received: {request}")
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

@app.get("/generate-api-key", dependencies=[Depends(rate_limit_dependency)])
def generate_api_key():
    api_key = "sk-" + secrets.token_urlsafe(36)
    return JSONResponse(content={"api_key": api_key})

# 在 /stats 路由中返回成功和失败百分比
@app.get("/stats", dependencies=[Depends(rate_limit_dependency)])
async def get_stats(request: Request, token: str = Depends(verify_admin_api_key)):
    middleware = app.middleware_stack.app
    if isinstance(middleware, StatsMiddleware):
        async with middleware.lock:
            stats = {
                "channel_success_percentages": middleware.calculate_success_percentages(),
                "channel_failure_percentages": middleware.calculate_failure_percentages(),
                "model_counts": dict(middleware.model_counts),
                "request_counts": dict(middleware.request_counts),
                "request_times": dict(middleware.request_times),
                "ip_counts": {k: dict(v) for k, v in middleware.ip_counts.items()},
                "request_arrivals": {k: [t.isoformat() for t in v] for k, v in middleware.request_arrivals.items()},
                "channel_success_counts": dict(middleware.channel_success_counts),
                "channel_failure_counts": dict(middleware.channel_failure_counts),
            }
        return JSONResponse(content=stats)
    return {"error": "StatsMiddleware not found"}

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