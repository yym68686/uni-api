import os
import json
import httpx
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

# 模拟存储API Key的数据库
api_keys_db = {
    "sk-KjjI60Yf0JFcsvgRmXqFwgGmWUd9GZnmi3KlvowmRWpWpQRo": "user1",
    # 可以添加更多的API Key
}

# 安全性依赖
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in api_keys_db:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    app.state.client = httpx.AsyncClient()
    yield
    # 关闭时的代码
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

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

# 定义 Function 参数模型
class FunctionParameter(BaseModel):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]

# 定义 Function 模型
class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameter

# 定义 Tool 模型
class Tool(BaseModel):
    type: str
    function: Function

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    name: Optional[str] = None
    content: Union[str, List[ContentItem]]

class RequestModel(BaseModel):
    model: str
    messages: List[Message]
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stream: Optional[bool] = None
    include_usage: Optional[bool] = None
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    user: Optional[str] = None
    tool_choice: Optional[str] = None
    tools: Optional[List[Tool]] = None

async def fetch_response_stream(client, url, headers, payload):
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        async for chunk in response.aiter_bytes():
            print(chunk.decode('utf-8'))
            yield chunk

async def fetch_response(client, url, headers, payload):
    response = await client.post(url, headers=headers, json=payload)
    return response.json()

async def process_request(request: RequestModel, provider: Dict):
    print("provider: ", provider['provider'])
    url = provider['base_url']
    headers = {
        'Authorization': f"Bearer {provider['api']}",
        'Content-Type': 'application/json'
    }

    # 转换消息格式
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": item.text})
                elif item.type == "image_url":
                    content.append({"type": "image_url", "image_url": item.image_url.dict()})
        else:
            content = msg.content
            name = msg.name
        if name:
            messages.append({"role": msg.role, "name": name, "content": content})
        else:
            messages.append({"role": msg.role, "content": content})


    payload = {
        "model": request.model,
        "messages": messages
    }

    for field, value in request.dict(exclude_unset=True).items():
        if field not in ['model', 'messages'] and value is not None:
            payload[field] = value

    request_info = {
        "url": url,
        "headers": headers,
        "payload": payload
    }
    print(f"Request details: {json.dumps(request_info, indent=2, ensure_ascii=False)}")
    if request.stream:
        return StreamingResponse(fetch_response_stream(app.state.client, url, headers, payload), media_type="text/event-stream")
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
                print(f"Error with provider {provider['provider']}: {str(e)}")
                continue

        raise HTTPException(status_code=500, detail="All providers failed")

model_handler = ModelRequestHandler()

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    return await model_handler.request_model(request, token)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)