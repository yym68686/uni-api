import json
import httpx
from datetime import datetime


async def generate_sse_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, tokens_use=None, total_tokens=None):
    sample_data = {
        "id": "chatcmpl-9ijPeRHa0wtyA2G8wq5z8FC3wGMzc",
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "system_fingerprint": "fp_d576307f90",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "logprobs": None,
                "finish_reason": None
            }
        ],
        "usage": None
    }
    if function_call_content:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"arguments": function_call_content}}]}
    if tools_id and function_call_name:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"id":tools_id,"type":"function","function":{"name":function_call_name,"arguments":""}}]}
        # sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"id": tools_id, "name": function_call_name}}]}
    if role:
        sample_data["choices"][0]["delta"] = {"role": role, "content": ""}
    json_data = json.dumps(sample_data, ensure_ascii=False)

    # 构建SSE响应
    sse_response = f"data: {json_data}\n\n"

    return sse_response

async def fetch_gemini_response_stream(client, url, headers, payload, model):
    timestamp = datetime.timestamp(datetime.now())
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # print(line)
                if line and '\"text\": \"' in line:
                    try:
                        json_data = json.loads( "{" + line + "}")
                        content = json_data.get('text', '')
                        content = "\n".join(content.split("\\n"))
                        sse_string = await generate_sse_response(timestamp, model, content)
                        yield sse_string
                    except json.JSONDecodeError:
                        print(f"无法解析JSON: {line}")

        # # 处理缓冲区中剩余的内容
        # if buffer:
        #     # print(buffer)
        #     if '\"text\": \"' in buffer:
        #         try:
        #             json_data = json.loads(buffer)
        #             content = json_data.get('text', '')
        #             content = "\n".join(content.split("\\n"))
        #             sse_string = await generate_sse_response(timestamp, model, content)
        #             yield sse_string
        #         except json.JSONDecodeError:
        #             print(f"无法解析JSON: {buffer}")

async def fetch_gpt_response_stream(client, url, headers, payload):
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        # print("response.status_code", response.status_code)
        if response.status_code != 200:
            print("请求失败，状态码是", response.status_code)
            error_message = await response.aread()
            error_str = error_message.decode('utf-8', errors='replace')
            error_json = json.loads(error_str)
            print(json.dumps(error_json, indent=4, ensure_ascii=False))
            yield {"error": f"HTTP Error {response.status_code}", "details": error_json}
        buffer = ""
        async for chunk in response.aiter_text():
            # print(chunk)
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line + "\n"

async def fetch_claude_response_stream(client, url, headers, payload, model):
    timestamp = datetime.timestamp(datetime.now())
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        if response.status_code != 200:
            print('\033[31m')
            print(f"请求失败，状态码是{response.status_code}，错误信息：")
            error_message = await response.aread()
            error_str = error_message.decode('utf-8', errors='replace')
            error_json = json.loads(error_str)
            print(json.dumps(error_json, indent=4, ensure_ascii=False))
            print('\033[0m')
            yield {"error": f"HTTP Error {response.status_code}", "details": error_json}
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # print(line)

                if line.startswith("data:"):
                    line = line[6:]
                    resp: dict = json.loads(line)
                    message = resp.get("message")
                    if message:
                        tokens_use = resp.get("usage")
                        role = message.get("role")
                        if role:
                            sse_string = await generate_sse_response(timestamp, model, None, None, None, None, role)
                            yield sse_string
                        if tokens_use:
                            total_tokens = tokens_use["input_tokens"] + tokens_use["output_tokens"]
                            # print("\n\rtotal_tokens", total_tokens)
                    tool_use = resp.get("content_block")
                    tools_id = None
                    function_call_name = None
                    if tool_use and "tool_use" == tool_use['type']:
                        # print("tool_use", tool_use)
                        tools_id = tool_use["id"]
                        if "name" in tool_use:
                            function_call_name = tool_use["name"]
                            sse_string = await generate_sse_response(timestamp, model, None, tools_id, function_call_name, None)
                            yield sse_string
                    delta = resp.get("delta")
                    # print("delta", delta)
                    if not delta:
                        continue
                    if "text" in delta:
                        content = delta["text"]
                        sse_string = await generate_sse_response(timestamp, model, content, None, None)
                        yield sse_string
                    if "partial_json" in delta:
                        # {"type":"input_json_delta","partial_json":""}
                        function_call_content = delta["partial_json"]
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, function_call_content)
                        yield sse_string

async def fetch_response(client, url, headers, payload):
    for _ in range(2):
        try:
            response = await client.post(url, headers=headers, json=payload)
            return response.json()
        except httpx.ConnectError as e:
            print(f"fetch_response 连接错误： {e}")
            continue
        except httpx.ReadTimeout as e:
            print(f"fetch_response 读取响应超时： {e}")
            continue

async def fetch_response_stream(client, url, headers, payload, engine, model):
    for _ in range(2):
        try:
            if engine == "gemini":
                async for chunk in fetch_gemini_response_stream(client, url, headers, payload, model):
                    yield chunk
            elif engine == "claude":
                async for chunk in fetch_claude_response_stream(client, url, headers, payload, model):
                    yield chunk
            elif engine == "gpt":
                async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
                    yield chunk
            elif engine == "openrouter":
                async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
                    yield chunk
            else:
                raise ValueError("Unknown response")
            break
        except httpx.ConnectError as e:
            print(f"fetch_response_stream 连接错误： {e}")
            continue
        except httpx.ReadTimeout as e:
            print(f"fetch_response_stream 读取响应超时： {e}")
            continue