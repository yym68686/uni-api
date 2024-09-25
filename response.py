import json
import httpx
from datetime import datetime
from io import BytesIO

from log_config import logger


async def generate_sse_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=0, prompt_tokens=0, completion_tokens=0):
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
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"id": tools_id,"type":"function","function":{"name": function_call_name, "arguments":""}}]}
        # sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"id": tools_id, "name": function_call_name}}]}
    if role:
        sample_data["choices"][0]["delta"] = {"role": role, "content": ""}
    if total_tokens:
        total_tokens = prompt_tokens + completion_tokens
        sample_data["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,"total_tokens": total_tokens}
        sample_data["choices"] = []
    json_data = json.dumps(sample_data, ensure_ascii=False)

    # 构建SSE响应
    sse_response = f"data: {json_data}\n\r\n"

    return sse_response

async def check_response(response, error_log):
    if response and response.status_code != 200:
        error_message = await response.aread()
        error_str = error_message.decode('utf-8', errors='replace')
        try:
            error_json = json.loads(error_str)
        except json.JSONDecodeError:
            error_json = error_str
        return {"error": f"{error_log} HTTP Error", "status_code": response.status_code, "details": error_json}
    return None

async def fetch_gemini_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_gemini_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        revicing_function_call = False
        function_full_response = "{"
        need_function_call = False
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
                        sse_string = await generate_sse_response(timestamp, model, content=content)
                        yield sse_string
                    except json.JSONDecodeError:
                        logger.error(f"无法解析JSON: {line}")

                if line and ('\"functionCall\": {' in line or revicing_function_call):
                    revicing_function_call = True
                    need_function_call = True
                    if ']' in line:
                        revicing_function_call = False
                        continue

                    function_full_response += line

        if need_function_call:
            function_call = json.loads(function_full_response)
            function_call_name = function_call["functionCall"]["name"]
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=function_call_name)
            yield sse_string
            function_full_response = json.dumps(function_call["functionCall"]["args"])
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=None, function_call_content=function_full_response)
            yield sse_string
        yield "data: [DONE]\n\r\n"

async def fetch_vertex_claude_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_vertex_claude_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        revicing_function_call = False
        function_full_response = "{"
        need_function_call = False
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(f"{line}")
                if line and '\"text\": \"' in line:
                    try:
                        json_data = json.loads( "{" + line + "}")
                        content = json_data.get('text', '')
                        content = "\n".join(content.split("\\n"))
                        sse_string = await generate_sse_response(timestamp, model, content=content)
                        yield sse_string
                    except json.JSONDecodeError:
                        logger.error(f"无法解析JSON: {line}")

                if line and ('\"type\": \"tool_use\"' in line or revicing_function_call):
                    revicing_function_call = True
                    need_function_call = True
                    if ']' in line:
                        revicing_function_call = False
                        continue

                    function_full_response += line

        if need_function_call:
            function_call = json.loads(function_full_response)
            function_call_name = function_call["name"]
            function_call_id = function_call["id"]
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=function_call_name)
            yield sse_string
            function_full_response = json.dumps(function_call["input"])
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=None, function_call_content=function_full_response)
            yield sse_string
        yield "data: [DONE]\n\r\n"

async def fetch_gpt_response_stream(client, url, headers, payload):
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_gpt_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and line != "data: " and line != "data:" and not line.startswith(": "):
                    yield line.strip() + "\n\r\n"

async def fetch_cloudflare_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_gpt_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line.startswith("data:"):
                    line = line.lstrip("data: ")
                    if line == "[DONE]":
                        yield "data: [DONE]\n\r\n"
                        return
                    resp: dict = json.loads(line)
                    message = resp.get("response")
                    if message:
                        sse_string = await generate_sse_response(timestamp, model, content=message)
                        yield sse_string

async def fetch_cohere_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_gpt_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                resp: dict = json.loads(line)
                if resp.get("is_finished") == True:
                    yield "data: [DONE]\n\r\n"
                    return
                if resp.get("event_type") == "text-generation":
                    message = resp.get("text")
                    sse_string = await generate_sse_response(timestamp, model, content=message)
                    yield sse_string

async def fetch_claude_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_claude_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        input_tokens = 0
        async for chunk in response.aiter_text():
            # logger.info(f"chunk: {repr(chunk)}")
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(line)

                if line.startswith("data:"):
                    line = line.lstrip("data: ")
                    resp: dict = json.loads(line)
                    message = resp.get("message")
                    if message:
                        role = message.get("role")
                        if role:
                            sse_string = await generate_sse_response(timestamp, model, None, None, None, None, role)
                            yield sse_string
                        tokens_use = message.get("usage")
                        if tokens_use:
                            input_tokens = tokens_use.get("input_tokens", 0)
                    usage = resp.get("usage")
                    if usage:
                        output_tokens = usage.get("output_tokens", 0)
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, total_tokens, input_tokens, output_tokens)
                        yield sse_string
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
        yield "data: [DONE]\n\r\n"

async def fetch_response(client, url, headers, payload):
    response = None
    if payload.get("file"):
        file = payload.pop("file")
        response = await client.post(url, headers=headers, data=payload, files={"file": file})
    else:
        response = await client.post(url, headers=headers, json=payload)
    error_message = await check_response(response, "fetch_response")
    if error_message:
        yield error_message
        return
    yield response.json()

async def fetch_response_stream(client, url, headers, payload, engine, model):
    try:
        if engine == "gemini" or engine == "vertex-gemini":
            async for chunk in fetch_gemini_response_stream(client, url, headers, payload, model):
                yield chunk
        elif engine == "claude" or engine == "vertex-claude":
            async for chunk in fetch_claude_response_stream(client, url, headers, payload, model):
                yield chunk
        elif engine == "gpt":
            async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
                yield chunk
        elif engine == "openrouter":
            async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
                yield chunk
        elif engine == "cloudflare":
            async for chunk in fetch_cloudflare_response_stream(client, url, headers, payload, model):
                yield chunk
        elif engine == "cohere":
            async for chunk in fetch_cohere_response_stream(client, url, headers, payload, model):
                yield chunk
        else:
            raise ValueError("Unknown response")
    except httpx.ConnectError as e:
        yield {"error": f"500", "details": "fetch_response_stream Connect Error"}
    except httpx.ReadTimeout as e:
        yield {"error": f"500", "details": "fetch_response_stream Read Response Timeout"}