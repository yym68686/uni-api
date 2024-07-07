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

class ContentItem(BaseModel):
    type: str
    text: str

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class RequestModel(BaseModel):
    model: str
    messages: List[Message]
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    stream: Optional[bool] = False
    include_usage: Optional[bool] = False

async def fetch_response_stream(client, url, headers, payload):
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        async for chunk in response.aiter_bytes():
            yield chunk

async def fetch_response(client, url, headers, payload):
    # request_info = {
    #     "url": url,
    #     "headers": headers,
    #     "payload": payload
    # }
    # print(f"Request details: {json.dumps(request_info, indent=2, ensure_ascii=False)}")

    response = await client.post(url, headers=headers, json=payload)
    # print(response.text)
    return response.json()

@app.post("/v1/chat/completions")
async def request_model(request: RequestModel, token: str = Depends(verify_api_key)):
    model_name = request.model

    for provider in config:
        if model_name in provider['model']:
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
                    content = " ".join([item.text for item in msg.content if item.type == "text"])
                else:
                    content = msg.content
                messages.append({"role": msg.role, "content": content})

            payload = {
                "model": model_name,
                "messages": messages
            }

            # 只有当相应参数存在且不为None时，才添加到payload中
            if request.stream is not None:
                payload["stream"] = request.stream
            if request.include_usage is not None:
                payload["include_usage"] = request.include_usage

            if provider['provider'] == 'anthropic':
                payload["max_tokens"] = 1000  # 您可能想让这个可配置
            else:
                if request.logprobs is not None:
                    payload["logprobs"] = request.logprobs
                if request.top_logprobs is not None:
                    payload["top_logprobs"] = request.top_logprobs

            try:
                if request.stream:
                    return StreamingResponse(fetch_response_stream(app.state.client, url, headers, payload), media_type="text/event-stream")
                else:
                    return await fetch_response(app.state.client, url, headers, payload)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error calling API: {str(e)}")

        raise HTTPException(status_code=404, detail="No matching model found")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)