import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import get_payload
import json

async def test_gemini_payload():
    # 构造测试请求
    request_data = {
        "model": "gemini-1.5-pro-latest",
        "messages": [
            {
                "role": "system",
                "content": "<plugins description=\"The plugins you can use below\">\n<collection name=\"Clock Time\">\n<collection.instructions>Display a clock to show current time</collection.instructions>\n<api identifier=\"clock-time____getCurrentTime____standalone\">获取当前时间</api>\n</collection>\n</plugins>"
            },
            {
                "role": "user",
                "content": "几点了？"
            }
        ],
        "stream": True,
        "temperature": 1.3,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "user": "d5a7516e-e919-45f0-81a1-42bad3da6125",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "clock-time____getCurrentTime____standalone",
                    "description": "获取当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    }

    request = RequestModel(**request_data)
    provider = {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "api": "your-api-key",  # 测试时替换为实际的 API key
        "model": ["gemini-1.5-pro-latest"],
        "project_id": "your-project-id"
    }

    url, headers, payload = await get_payload(request, "vertex-gemini", provider)
    # url, headers, payload = await get_payload(request, "gemini", provider)

    print("payload", json.dumps(payload, indent=4, ensure_ascii=False))

    # 验证生成的 payload 结构
    assert "contents" in payload
    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert "function_declarations" in payload["tools"][0]

    # 验证工具配置
    assert payload["tools"][0]["function_declarations"][0]["name"] == "clock-time____getCurrentTime____standalone"
    assert payload["tools"][0]["function_declarations"][0]["description"] == "获取当前时间"

    # 验证消息内容
    assert len(payload["contents"]) == 2
    assert payload["contents"][0]["role"] == "system"
    assert payload["contents"][1]["role"] == "user"

    # 验证其他参数
    assert payload["temperature"] == 1.3
    assert payload["top_p"] == 1.0

    # 验证安全设置
    assert "safetySettings" in payload
    assert len(payload["safetySettings"]) == 4

    # 验证工具配置
    assert "tool_config" in payload
    assert payload["tool_config"]["function_calling_config"]["mode"] == "AUTO"

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gemini_payload())