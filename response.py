from datetime import datetime
import json
import httpx

async def generate_sse_response(timestamp, model, content):
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
    json_data = json.dumps(sample_data, ensure_ascii=False)

    # 构建SSE响应
    sse_response = f"data: {json_data}\n\n"

    return sse_response

async def fetch_gemini_response_stream(client, url, headers, payload, model):
    try:
        timestamp = datetime.timestamp(datetime.now())
        async with client.stream('POST', url, headers=headers, json=payload) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    print(line)
                    if line and '\"text\": \"' in line:
                        try:
                            json_data = json.loads( "{" + line + "}")
                            content = json_data.get('text', '')
                            content = "\n".join(content.split("\\n"))
                            sse_string = await generate_sse_response(timestamp, model, content)
                            yield sse_string
                        except json.JSONDecodeError:
                            print(f"无法解析JSON: {line}")

            # 处理缓冲区中剩余的内容
            if buffer:
                # print(buffer)
                if '\"text\": \"' in buffer:
                    try:
                        json_data = json.loads(buffer)
                        content = json_data.get('text', '')
                        content = "\n".join(content.split("\\n"))
                        sse_string = await generate_sse_response(timestamp, model, content)
                        yield sse_string
                    except json.JSONDecodeError:
                        print(f"无法解析JSON: {buffer}")

            yield "data: [DONE]\n\n"
    except httpx.ConnectError as e:
        print(f"连接错误： {e}")

async def fetch_gpt_response_stream(client, url, headers, payload):
    try:
        async with client.stream('POST', url, headers=headers, json=payload) as response:
            async for chunk in response.aiter_bytes():
                print(chunk.decode('utf-8'))
                yield chunk
    except httpx.ConnectError as e:
        print(f"连接错误： {e}")

async def fetch_claude_response_stream(client, url, headers, payload, model):
    try:
        timestamp = datetime.timestamp(datetime.now())
        async with client.stream('POST', url, headers=headers, json=payload) as response:
            async for chunk in response.aiter_bytes():
                chunk_line = chunk.decode('utf-8').split("\n")
                for chunk in chunk_line:
                    if chunk.startswith("data:"):
                        line = chunk[6:]
                        # print(line)
                        resp: dict = json.loads(line)
                        message = resp.get("message")
                        if message:
                            tokens_use = resp.get("usage")
                            if tokens_use:
                                total_tokens = tokens_use["input_tokens"] + tokens_use["output_tokens"]
                                # print("\n\rtotal_tokens", total_tokens)
                        # tool_use = resp.get("content_block")
                        # if tool_use and "tool_use" == tool_use['type']:
                        #     # print("tool_use", tool_use)
                        #     tools_id = tool_use["id"]
                        #     need_function_call = True
                        #     if "name" in tool_use:
                        #         function_call_name = tool_use["name"]
                        delta = resp.get("delta")
                        # print("delta", delta)
                        if not delta:
                            continue
                        if "text" in delta:
                            content = delta["text"]
                            sse_string = await generate_sse_response(timestamp, model, content)
                            print(sse_string)
                            yield sse_string
                        # if "partial_json" in delta:
                        #     function_call_content = delta["partial_json"]
            yield "data: [DONE]\n\n"
    except httpx.ConnectError as e:
        print(f"连接错误： {e}")

async def fetch_response(client, url, headers, payload):
    response = await client.post(url, headers=headers, json=payload)
    return response.json()

async def fetch_response_stream(client, url, headers, payload, engine, model):
    if engine == "gemini":
        async for chunk in fetch_gemini_response_stream(client, url, headers, payload, model):
            yield chunk
    elif engine == "claude":
        async for chunk in fetch_claude_response_stream(client, url, headers, payload, model):
            yield chunk
    elif engine == "gpt":
        async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
            yield chunk
    else:
        raise ValueError("Unknown response")