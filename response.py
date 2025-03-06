import json
import random
import string
from datetime import datetime

from log_config import logger

from utils import safe_get, generate_sse_response, generate_no_stream_response, end_of_line

async def check_response(response, error_log):
    if response and not (200 <= response.status_code < 300):
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
        is_finish = False
        # line_index = 0
        # last_text_line = 0
        # if "thinking" in model:
        #     is_thinking = True
        # else:
        #     is_thinking = False
        async for chunk in response.aiter_text():
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # line_index += 1
                if line and '\"finishReason\": \"' in line:
                    is_finish = True
                    break
                # print(line)
                if line and '\"text\": \"' in line:
                    try:
                        json_data = json.loads( "{" + line + "}")
                        content = json_data.get('text', '')
                        content = "\n".join(content.split("\\n"))
                        # content = content.replace("\n", "\n\n")
                        # if last_text_line == 0 and is_thinking:
                        #     content = "> " + content.lstrip()
                        # if is_thinking:
                        #     content = content.replace("\n", "\n> ")
                        # if last_text_line == line_index - 3:
                        #     is_thinking = False
                        #     content = "\n\n\n" + content.lstrip()
                        sse_string = await generate_sse_response(timestamp, model, content=content)
                        yield sse_string
                    except json.JSONDecodeError:
                        logger.error(f"无法解析JSON: {line}")
                    # last_text_line = line_index

                if line and ('\"functionCall\": {' in line or revicing_function_call):
                    revicing_function_call = True
                    need_function_call = True
                    if ']' in line:
                        revicing_function_call = False
                        continue

                    function_full_response += line

            if is_finish:
                break

        if need_function_call:
            function_call = json.loads(function_full_response)
            function_call_name = function_call["functionCall"]["name"]
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=function_call_name)
            yield sse_string
            function_full_response = json.dumps(function_call["functionCall"]["args"])
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=None, function_call_content=function_full_response)
            yield sse_string
        yield "data: [DONE]" + end_of_line

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
        yield "data: [DONE]" + end_of_line

async def fetch_gpt_response_stream(client, url, headers, payload):
    timestamp = int(datetime.timestamp(datetime.now()))
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_gpt_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        enter_buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and line != "data: " and line != "data:" and not line.startswith(": "):
                    result = line.lstrip("data: ")
                    if result.strip() == "[DONE]":
                        yield "data: [DONE]" + end_of_line
                        return
                    line = json.loads(result)
                    line['id'] = f"chatcmpl-{random_str}"

                    # 处理 <think> 标签
                    content = safe_get(line, "choices", 0, "delta", "content", default="")
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        is_thinking = False
                        content = content.replace("</think>", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                        continue
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    # 处理 poe thinking 标签
                    if "Thinking..." in content and "\n> " in content:
                        is_thinking = True
                        content = content.replace("Thinking...", "").replace("\n> ", "")
                    if is_thinking and "\n\n" in content and not ark_tag:
                        is_thinking = False
                    if is_thinking and not ark_tag:
                        content = content.replace("\n> ", "")
                        if not has_send_thinking:
                            content = content.replace("\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    no_stream_content = safe_get(line, "choices", 0, "message", "content", default=None)
                    openrouter_reasoning = safe_get(line, "choices", 0, "delta", "reasoning", default="")
                    if openrouter_reasoning:
                        if openrouter_reasoning.endswith("\\\\"):
                            enter_buffer += openrouter_reasoning
                            continue
                        elif enter_buffer.endswith("\\\\") and openrouter_reasoning == 'n':
                            enter_buffer += "n"
                            continue
                        elif enter_buffer.endswith("\\\\n") and openrouter_reasoning == '\\\\n':
                            enter_buffer += "\\\\n"
                            continue
                        elif enter_buffer.endswith("\\\\n\\\\n"):
                            openrouter_reasoning = '\n\n' + openrouter_reasoning
                            enter_buffer = ""
                        elif enter_buffer:
                            openrouter_reasoning = enter_buffer + openrouter_reasoning
                            enter_buffer = ''

                        sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=openrouter_reasoning)
                        yield sse_string
                    elif no_stream_content and has_send_thinking == False:
                        sse_string = await generate_sse_response(safe_get(line, "created", default=None), safe_get(line, "model", default=None), content=no_stream_content)
                        yield sse_string
                    else:
                        if no_stream_content:
                            del line["choices"][0]["message"]
                        yield "data: " + json.dumps(line).strip() + end_of_line

async def fetch_azure_response_stream(client, url, headers, payload):
    timestamp = int(datetime.timestamp(datetime.now()))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_azure_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        sse_string = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and line != "data: " and line != "data:" and not line.startswith(": "):
                    result = line.lstrip("data: ")
                    if result.strip() == "[DONE]":
                        yield "data: [DONE]" + end_of_line
                        return
                    line = json.loads(result)
                    no_stream_content = safe_get(line, "choices", 0, "message", "content", default="")
                    content = safe_get(line, "choices", 0, "delta", "content", default="")

                    # 处理 <think> 标签
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        is_thinking = False
                        content = content.replace("</think>", "")
                        if not content:
                            continue
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    if no_stream_content or content or sse_string:
                        sse_string = await generate_sse_response(timestamp, safe_get(line, "model", default=None), content=no_stream_content or content)
                        yield sse_string
                    if no_stream_content:
                        yield "data: [DONE]" + end_of_line
                        return

async def fetch_cloudflare_response_stream(client, url, headers, payload, model):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload) as response:
        error_message = await check_response(response, "fetch_cloudflare_response_stream")
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
                        yield "data: [DONE]" + end_of_line
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
                    yield "data: [DONE]" + end_of_line
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
                    if "thinking" in delta and delta["thinking"]:
                        content = delta["thinking"]
                        sse_string = await generate_sse_response(timestamp, model, reasoning_content=content)
                        yield sse_string
                    if "partial_json" in delta:
                        # {"type":"input_json_delta","partial_json":""}
                        function_call_content = delta["partial_json"]
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, function_call_content)
                        yield sse_string
        yield "data: [DONE]" + end_of_line

async def fetch_response(client, url, headers, payload, engine, model):
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

    if engine == "tts":
        yield response.read()

    elif engine == "gemini" or engine == "vertex-gemini":
        response_json = response.json()

        if isinstance(response_json, str):
            import ast
            parsed_data = ast.literal_eval(str(response_json))
        elif isinstance(response_json, list):
            parsed_data = response_json
        else:
            logger.error(f"error fetch_response: Unknown response_json type: {type(response_json)}")
            parsed_data = response_json
        # print("parsed_data", json.dumps(parsed_data, indent=4, ensure_ascii=False))
        content = ""
        for item in parsed_data:
            chunk = safe_get(item, "candidates", 0, "content", "parts", 0, "text")
            # logger.info(f"chunk: {repr(chunk)}")
            if chunk:
                content += chunk

        usage_metadata = safe_get(parsed_data, -1, "usageMetadata")
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)

        role = safe_get(parsed_data, -1, "candidates", 0, "content", "role")
        if role == "model":
            role = "assistant"
        else:
            logger.error(f"Unknown role: {role}")
            role = "assistant"

        function_call_name = safe_get(parsed_data, -1, "candidates", 0, "content", "parts", 0, "functionCall", "name", default=None)
        function_call_content = safe_get(parsed_data, -1, "candidates", 0, "content", "parts", 0, "functionCall", "args", default=None)

        timestamp = int(datetime.timestamp(datetime.now()))
        yield await generate_no_stream_response(timestamp, model, content=content, tools_id=None, function_call_name=function_call_name, function_call_content=function_call_content, role=role, total_tokens=total_tokens, prompt_tokens=prompt_tokens, completion_tokens=candidates_tokens)

    elif engine == "claude":
        response_json = response.json()
        # print("response_json", json.dumps(response_json, indent=4, ensure_ascii=False))

        content = safe_get(response_json, "content", 0, "text")

        prompt_tokens = safe_get(response_json, "usage", "input_tokens")
        output_tokens = safe_get(response_json, "usage", "output_tokens")
        total_tokens = prompt_tokens + output_tokens

        role = safe_get(response_json, "role")

        function_call_name = safe_get(response_json, "content", 1, "name", default=None)
        function_call_content = safe_get(response_json, "content", 1, "input", default=None)
        tools_id = safe_get(response_json, "content", 1, "id", default=None)

        timestamp = int(datetime.timestamp(datetime.now()))
        yield await generate_no_stream_response(timestamp, model, content=content, tools_id=tools_id, function_call_name=function_call_name, function_call_content=function_call_content, role=role, total_tokens=total_tokens, prompt_tokens=prompt_tokens, completion_tokens=output_tokens)

    elif engine == "azure":
        response_json = response.json()
        # 删除 content_filter_results
        if "choices" in response_json:
            for choice in response_json["choices"]:
                if "content_filter_results" in choice:
                    del choice["content_filter_results"]

        # 删除 prompt_filter_results
        if "prompt_filter_results" in response_json:
            del response_json["prompt_filter_results"]

        yield response_json

    else:
        response_json = response.json()
        yield response_json

async def fetch_response_stream(client, url, headers, payload, engine, model):
    # try:
    if engine == "gemini" or engine == "vertex-gemini":
        async for chunk in fetch_gemini_response_stream(client, url, headers, payload, model):
            yield chunk
    elif engine == "claude" or engine == "vertex-claude":
        async for chunk in fetch_claude_response_stream(client, url, headers, payload, model):
            yield chunk
    elif engine == "gpt":
        async for chunk in fetch_gpt_response_stream(client, url, headers, payload):
            yield chunk
    elif engine == "azure":
        async for chunk in fetch_azure_response_stream(client, url, headers, payload):
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
    # except httpx.ConnectError as e:
    #     yield {"error": f"500", "details": "fetch_response_stream Connect Error"}
    # except httpx.ReadTimeout as e:
    #     yield {"error": f"500", "details": "fetch_response_stream Read Response Timeout"}