import asyncio
import httpx
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

app = FastAPI()

# 读取JSON配置文件
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("配置文件 'config.json' 未找到。请确保文件存在于正确的位置。")
        return []
    except json.JSONDecodeError:
        print("配置文件 'config.json' 格式不正确。请检查JSON格式。")
        return []

config = load_config()

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

async def fetch_response(client, url, headers, payload):
    response = await client.post(url, headers=headers, json=payload)
    return response.json()

@app.post("/request_model")
async def request_model(request: RequestModel):
    model_name = request.model

    tasks = []
    async with httpx.AsyncClient() as client:
        for provider in config:
            if model_name in provider['model']:
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
                    "messages": messages,
                    "stream": request.stream,
                    "include_usage": request.include_usage
                }

                if provider['provider'] == 'anthropic':
                    payload["max_tokens"] = 1000  # 您可能想让这个可配置
                else:
                    if request.logprobs:
                        payload["logprobs"] = request.logprobs
                    if request.top_logprobs:
                        payload["top_logprobs"] = request.top_logprobs

                tasks.append(fetch_response(client, url, headers, payload))

        if not tasks:
            raise HTTPException(status_code=404, detail="No matching model found")

        try:
            responses = await asyncio.gather(*tasks)
            return responses
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling API: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)