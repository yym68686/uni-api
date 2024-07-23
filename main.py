import json
import traceback

import httpx
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends

from models import RequestModel
from utils import config, api_keys_db, api_list, error_handling_wrapper, get_all_models, verify_api_key, post_all_models
from request import get_payload
from response import fetch_response, fetch_response_stream

from typing import List, Dict
from urllib.parse import urlparse
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    timeout = httpx.Timeout(connect=15.0, read=10.0, write=30.0, pool=30.0)
    app.state.client = httpx.AsyncClient(timeout=timeout)
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
    print("provider: ", provider['provider'])
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
    print("engine", engine)

    url, headers, payload = await get_payload(request, engine, provider)

    # request_info = {
    #     "url": url,
    #     "headers": headers,
    #     "payload": payload
    # }
    # print(f"Request details: {json.dumps(request_info, indent=4, ensure_ascii=False)}")

    if request.stream:
        model = provider['model'][request.model]
        # try:
        generator = fetch_response_stream(app.state.client, url, headers, payload, engine, model)
        wrapped_generator = await error_handling_wrapper(generator, status_code=500)
        return StreamingResponse(wrapped_generator, media_type="text/event-stream")
        # except HTTPException as e:
        #     return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
        # except Exception as e:
        #     # 处理其他异常
        #     return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return await fetch_response(app.state.client, url, headers, payload)

class ModelRequestHandler:
    def __init__(self):
        self.last_provider_index = -1

    def get_matching_providers(self, model_name, token):
        # for provider in config:
        #     print("provider", model_name, list(provider['model'].keys()))
        #     if model_name in provider['model'].keys():
        #         print("provider", provider)
        api_index = api_list.index(token)
        provider_rules = []

        for model in config['api_keys'][api_index]['model']:
            if "/" in model:
                provider_name = model.split("/")[0]
                model = model.split("/")[1]
                for provider in config['providers']:
                    if provider['provider'] == provider_name:
                        models_list = provider['model'].keys()
                if (model and model_name in models_list) or (model == "*" and model_name in models_list):
                    provider_rules.append(provider_name)
            else:
                for provider in config['providers']:
                    if model in provider['model'].keys():
                        provider_rules.append(provider['provider'] + "/" + model)

        provider_list = []
        # print("provider_rules", provider_rules)
        for provider in config['providers']:
            for item in provider_rules:
                if provider['provider'] in item:
                    if "/" in item:
                        if item.split("/")[1] == model_name:
                            provider_list.append(provider)
                    else:
                        if model_name in provider['model'].keys():
                            provider_list.append(provider)
        return provider_list

    async def request_model(self, request: RequestModel, token: str):
        model_name = request.model
        matching_providers = self.get_matching_providers(model_name, token)
        # print("matching_providers", json.dumps(matching_providers, indent=4, ensure_ascii=False))
        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")

        # 检查是否启用轮询
        api_index = api_list.index(token)
        use_round_robin = False
        if config['api_keys'][api_index].get("preferences"):
            use_round_robin = config['api_keys'][api_index]["preferences"].get("USE_ROUND_ROBIN")

        return await self.try_all_providers(request, matching_providers, use_round_robin)

    async def try_all_providers(self, request: RequestModel, providers: List[Dict], use_round_robin: bool):
        num_providers = len(providers)
        start_index = self.last_provider_index + 1 if use_round_robin else 0

        for i in range(num_providers + 1):
            self.last_provider_index = (start_index + i) % num_providers
            provider = providers[self.last_provider_index]
            try:
                response = await process_request(request, provider)
                return response
            except (Exception, HTTPException) as e:
                print('\033[31m')
                print(f"Error with provider {provider['provider']}: {str(e)}")
                # traceback.print_exc()
                print('\033[0m')
                if use_round_robin:
                    continue
                else:
                    raise HTTPException(status_code=500, detail="Error: Current provider response failed!")

        raise HTTPException(status_code=500, detail="All providers failed")

model_handler = ModelRequestHandler()

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

@app.options("/v1/chat/completions")
async def options_handler():
    return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.post("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
    models = post_all_models(token)
    return {
        "object": "list",
        "data": models
    }

@app.get("/v1/models")
async def list_models():
    models = get_all_models()
    return {
        "object": "list",
        "data": models
    }

@app.get("/generate-api-key")
def generate_api_key():
    api_key = "sk-" + secrets.token_urlsafe(32)
    return {"api_key": api_key}

# async def on_fetch(request, env):
#     import asgi

#     return await asgi.fetch(app, request, env)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)