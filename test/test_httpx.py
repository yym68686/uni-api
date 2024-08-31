import httpx
import asyncio
import ssl
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def make_request():
    # SSL 上下文设置
    # ssl_context = ssl.create_default_context()
    # ssl_context.set_alpn_protocols(["h2", "http/1.1"])

    # 创建自定义传输
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        # verify=ssl_context,
        verify=False,
        retries=1
    )

    # 设置头部
    headers = {
        "User-Agent": "curl/8.7.1",
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-xxxxxxx"
    }

    # 请求数据
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "say test"
            }
        ],
        "stream": True
    }

    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.post(
                "https://api.xxxxxxxxxx.me/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )

            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Headers: {response.headers}")

            # 处理流式响应
            async for line in response.aiter_lines():
                if line:
                    print(line)

        except httpx.RequestError as e:
            logger.error(f"An error occurred while requesting {e.request.url!r}.")

# 运行异步函数
asyncio.run(make_request())