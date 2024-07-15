import json
import httpx
import logging
import yaml
import secrets
import traceback
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import RequestModel
from request import get_payload
from response import fetch_response, fetch_response_stream

from typing import List, Dict
from urllib.parse import urlparse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0)
    app.state.client = httpx.AsyncClient(timeout=timeout)
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

# 安全性依赖
security = HTTPBearer()

# 读取YAML配置文件
def load_config():
    try:
        with open('api.yaml', 'r') as f:
            conf = yaml.safe_load(f)
            for index, provider in enumerate(conf['providers']):
                model_dict = {}
                for model in provider['model']:
                    if type(model) == str:
                        model_dict[model] = model
                    if type(model) == dict:
                        model_dict.update({value: key for key, value in model.items()})
                provider['model'] = model_dict
                conf['providers'][index] = provider
            api_keys_db = conf['api_keys']
            api_list = [item["api"] for item in api_keys_db]
            # print(json.dumps(conf, indent=4, ensure_ascii=False))
            return conf, api_keys_db, api_list
    except FileNotFoundError:
        print("配置文件 'config.yaml' 未找到。请确保文件存在于正确的位置。")
        return []
    except yaml.YAMLError:
        print("配置文件 'config.yaml' 格式不正确。请检查YAML格式。")
        return []

config, api_keys_db, api_list = load_config()

async def error_handling_wrapper(generator, status_code=200):
    try:
        first_item = await generator.__anext__()
        if isinstance(first_item, dict) and "error" in first_item:
            # 如果第一个 yield 的项是错误信息，抛出 HTTPException
            raise HTTPException(status_code=status_code, detail=first_item)

        # 如果不是错误，创建一个新的生成器，首先yield第一个项，然后yield剩余的项
        async def new_generator():
            yield first_item
            async for item in generator:
                yield item

        return new_generator()
    except StopAsyncIteration:
        # 处理生成器为空的情况
        return []

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
        try:
            generator = fetch_response_stream(app.state.client, url, headers, payload, engine, model)
            wrapped_generator = await error_handling_wrapper(generator, status_code=500)
            return StreamingResponse(wrapped_generator, media_type="text/event-stream")
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"error": str(e.detail)})
        except Exception as e:
            # 处理其他异常
            return JSONResponse(status_code=500, content={"error": str(e)})
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
                if (model and model_name == model) or (model == "*" and model_name in models_list):
                    provider_rules.append(provider_name)
        provider_list = []
        for provider in config['providers']:
            if model_name in provider['model'].keys() and ((provider_rules and provider['provider'] in provider_rules) or provider_rules == []):
                provider_list.append(provider)
        return provider_list

    async def request_model(self, request: RequestModel, token: str):
        model_name = request.model
        matching_providers = self.get_matching_providers(model_name, token)
        print("matching_providers", json.dumps(matching_providers, indent=4, ensure_ascii=False))

        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")

        # 检查是否启用轮询
        use_round_robin = config["preferences"].get("USE_ROUND_ROBIN")

        return await self.try_all_providers(request, matching_providers, use_round_robin)

    async def try_all_providers(self, request: RequestModel, providers: List[Dict], use_round_robin: bool):
        num_providers = len(providers)

        for i in range(num_providers):
            if use_round_robin:
                # 始终从第一个提供者开始轮询
                self.last_provider_index = i % num_providers
            else:
                # 非轮询模式，按顺序尝试
                self.last_provider_index = i

            provider = providers[self.last_provider_index]
            try:
                response = await process_request(request, provider)
                return response
            except Exception as e:
                print('\033[31m')
                print(f"Error with provider {provider['provider']}: {str(e)}")
                traceback.print_exc()
                print('\033[0m')
                continue

        raise HTTPException(status_code=500, detail="All providers failed")

model_handler = ModelRequestHandler()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 打印请求信息
    logging.info(f"Request: {request.method} {request.url}")
    # 打印请求体（如果有）
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        logging.info(f"Request Body: {body.decode('utf-8')}")

    response = await call_next(request)
    return response

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

def get_all_models(token):
    all_models = []
    unique_models = set()

    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    api_index = api_list.index(token)
    if config['api_keys'][api_index]['model']:
        for model in config['api_keys'][api_index]['model']:
            if "/" in model:
                provider = model.split("/")[0]
                model = model.split("/")[1]
                if model == "*":
                    for provider_item in config["providers"]:
                        if provider_item['provider'] != provider:
                            continue
                        for model_item in provider_item['model'].keys():
                            if model_item not in unique_models:
                                unique_models.add(model_item)
                                model_info = {
                                    "id": model_item,
                                    "object": "model",
                                    "created": 1720524448858,
                                    "owned_by": provider_item['provider']
                                }
                                all_models.append(model_info)
                else:
                    for provider_item in config["providers"]:
                        if provider_item['provider'] != provider:
                            continue
                        for model_item in provider_item['model'].keys() :
                            if model_item not in unique_models and model_item == model:
                                unique_models.add(model_item)
                                model_info = {
                                    "id": model_item,
                                    "object": "model",
                                    "created": 1720524448858,
                                    "owned_by": provider_item['provider']
                                }
                                all_models.append(model_info)
                continue

            if model not in unique_models:
                unique_models.add(model)
                model_info = {
                    "id": model,
                    "object": "model",
                    "created": 1720524448858,
                    "owned_by": model
                }
                all_models.append(model_info)
    else:
        for provider in config["providers"]:
            for model in provider['model'].keys():
                if model not in unique_models:
                    unique_models.add(model)
                    model_info = {
                        "id": model,
                        "object": "model",
                        "created": 1720524448858,
                        "owned_by": provider['provider']
                    }
                    all_models.append(model_info)

    return all_models

@app.post("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
    models = get_all_models(token)
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