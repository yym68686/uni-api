from log_config import logger

import httpx
import secrets
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import RequestModel
from utils import error_handling_wrapper, get_all_models, post_all_models, load_config
from request import get_payload
from response import fetch_response, fetch_response_stream

from typing import List, Dict
from urllib.parse import urlparse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    timeout = httpx.Timeout(connect=15.0, read=20.0, write=30.0, pool=30.0)
    app.state.client = httpx.AsyncClient(timeout=timeout)
    app.state.config, app.state.api_keys_db, app.state.api_list = await load_config(app)
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部字段
)

async def process_request(request: RequestModel, provider: Dict):
    url = provider['base_url']
    parsed_url = urlparse(url)
    # print(parsed_url)
    engine = None
    if parsed_url.netloc == 'generativelanguage.googleapis.com':
        engine = "gemini"
    elif parsed_url.netloc == 'api.anthropic.com' or parsed_url.path.endswith("v1/messages"):
        engine = "claude"
    elif parsed_url.netloc == 'openrouter.ai':
        engine = "openrouter"
    else:
        engine = "gpt"

    if "claude" not in provider['model'][request.model] \
    and "gpt" not in provider['model'][request.model] \
    and "gemini" not in provider['model'][request.model]:
        engine = "openrouter"

    if provider.get("engine"):
        engine = provider["engine"]
    logger.info(f"provider: {provider['provider']:<10} model: {request.model:<10} engine: {engine}")

    url, headers, payload = await get_payload(request, engine, provider)

    # request_info = {
    #     "url": url,
    #     "headers": headers,
    #     "payload": payload
    # }
    # import json
    # logger.info(f"Request details: {json.dumps(request_info, indent=4, ensure_ascii=False)}")

    if request.stream:
        model = provider['model'][request.model]
        generator = fetch_response_stream(app.state.client, url, headers, payload, engine, model)
        wrapped_generator = error_handling_wrapper(generator, status_code=500)
        return StreamingResponse(wrapped_generator, media_type="text/event-stream")
    else:
        return await fetch_response(app.state.client, url, headers, payload)

class ModelRequestHandler:
    def __init__(self):
        self.last_provider_index = -1

    def get_matching_providers(self, model_name, token):
        config = app.state.config
        # api_keys_db = app.state.api_keys_db
        api_list = app.state.api_list

        api_index = api_list.index(token)
        provider_rules = []

        for model in config['api_keys'][api_index]['model']:
            if "/" in model:
                provider_name = model.split("/")[0]
                model = model.split("/")[1]
                models_list = []
                for provider in config['providers']:
                    if provider['provider'] == provider_name:
                        models_list.extend(list(provider['model'].keys()))
                # print("models_list", models_list)
                # print("model_name", model_name)
                # print("model", model)
                if (model and model_name in models_list) or (model == "*" and model_name in models_list):
                    provider_rules.append(provider_name)
            else:
                for provider in config['providers']:
                    if model in provider['model'].keys():
                        provider_rules.append(provider['provider'] + "/" + model)

        provider_list = []
        # print("provider_rules", provider_rules)
        for item in provider_rules:
            for provider in config['providers']:
                if provider['provider'] in item:
                    if "/" in item:
                        if item.split("/")[1] == model_name:
                            provider_list.append(provider)
                    else:
                        if model_name in provider['model'].keys():
                            provider_list.append(provider)
        return provider_list

    async def request_model(self, request: RequestModel, token: str):
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
        use_round_robin = False
        auto_retry = False
        if config['api_keys'][api_index].get("preferences"):
            use_round_robin = config['api_keys'][api_index]["preferences"].get("USE_ROUND_ROBIN")
            auto_retry = config['api_keys'][api_index]["preferences"].get("AUTO_RETRY")

        return await self.try_all_providers(request, matching_providers, use_round_robin, auto_retry)

    async def try_all_providers(self, request: RequestModel, providers: List[Dict], use_round_robin: bool, auto_retry: bool):
        num_providers = len(providers)
        start_index = self.last_provider_index + 1 if use_round_robin else 0

        for i in range(num_providers + 1):
            self.last_provider_index = (start_index + i) % num_providers
            provider = providers[self.last_provider_index]
            try:
                response = await process_request(request, provider)
                return response
            except (Exception, HTTPException) as e:
                logger.error(f"Error with provider {provider['provider']}: {str(e)}")
                if auto_retry:
                    continue
                else:
                    raise HTTPException(status_code=500, detail="Error: Current provider response failed!")

        raise HTTPException(status_code=500, detail=f"All providers failed: {request.model}")

model_handler = ModelRequestHandler()

# 安全性依赖
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = app.state.api_list
    token = credentials.credentials
    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

@app.options("/v1/chat/completions")
async def options_handler():
    return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.post("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
    models = post_all_models(token, app.state.config, app.state.api_list)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.get("/v1/models")
async def list_models():
    models = get_all_models(config=app.state.config)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.get("/generate-api-key")
def generate_api_key():
    api_key = "sk-" + secrets.token_urlsafe(32)
    return JSONResponse(content={"api_key": api_key})

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
        ws="none",
        log_level="warning"
    )