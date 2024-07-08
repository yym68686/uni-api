import os
import json
import httpx
import yaml
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
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
    app.state.client = httpx.AsyncClient()
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

# 模拟存储API Key的数据库
api_keys_db = {
    "sk-KjjI60Yf0JFcsvgRmXqFwgGmWUd9GZnmi3KlvowmRWpWpQRo": "user1",
    # 可以添加更多的API Key
}

# 安全性依赖
security = HTTPBearer()

# 读取YAML配置文件
def load_config():
    try:
        with open('api.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("配置文件 'config.yaml' 未找到。请确保文件存在于正确的位置。")
        return []
    except yaml.YAMLError:
        print("配置文件 'config.yaml' 格式不正确。请检查YAML格式。")
        return []

config = load_config()
# print(config)

async def process_request(request: RequestModel, provider: Dict):
    print("provider: ", provider['provider'])
    url = provider['base_url']
    parsed_url = urlparse(url)
    engine = None
    if parsed_url.netloc == 'generativelanguage.googleapis.com':
        engine = "gemini"
    elif parsed_url.netloc == 'api.anthropic.com':
        engine = "claude"
    else:
        engine = "gpt"

    url, headers, payload = await get_payload(request, engine, provider)

    request_info = {
        "url": url,
        "headers": headers,
        "payload": payload
    }
    print(f"Request details: {json.dumps(request_info, indent=2, ensure_ascii=False)}")

    if request.stream:
        return StreamingResponse(fetch_response_stream(app.state.client, url, headers, payload, engine, request.model), media_type="text/event-stream")
    else:
        return await fetch_response(app.state.client, url, headers, payload)

class ModelRequestHandler:
    def __init__(self):
        self.last_provider_index = -1

    def get_matching_providers(self, model_name):
        return [provider for provider in config if model_name in provider['model']]

    async def request_model(self, request: RequestModel, token: str):
        model_name = request.model
        matching_providers = self.get_matching_providers(model_name)
        # print("matching_providers", json.dumps(matching_providers, indent=2, ensure_ascii=False))

        if not matching_providers:
            raise HTTPException(status_code=404, detail="No matching model found")

        # 检查是否启用轮询
        use_round_robin = os.environ.get('USE_ROUND_ROBIN', 'false').lower() == 'true'

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

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in api_keys_db:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)