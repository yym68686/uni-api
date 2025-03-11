import os
import re
import io
import json
import httpx
import base64
import urllib.parse
from PIL import Image

from models import RequestModel
from utils import c35s, c3s, c3o, c3h, gemini1, gemini2, BaseAPI, get_model_dict, provider_api_circular_list, safe_get

def get_image_format(file_content):
    try:
        img = Image.open(io.BytesIO(file_content))
        return img.format.lower()
    except:
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
        img_format = get_image_format(file_content)
        if not img_format:
            raise ValueError("无法识别的图片格式")
        base64_encoded = base64.b64encode(file_content).decode('utf-8')

        if img_format == 'png':
            return f"data:image/png;base64,{base64_encoded}"
        elif img_format in ['jpg', 'jpeg']:
            return f"data:image/jpeg;base64,{base64_encoded}"
        else:
            raise ValueError(f"不支持的图片格式: {img_format}")

async def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        verify=False,
        retries=1
    )
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.get(
                url,
                timeout=30.0
            )
            with open(filename, 'wb') as f:
                f.write(response.content)

        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")

    return filename

async def get_encode_image(image_url):
    filename = await get_doc_from_url(image_url)
    image_path = os.getcwd() + "/" + filename
    base64_image = encode_image(image_path)
    os.remove(image_path)
    return base64_image

# from PIL import Image
# import io
# def validate_image(image_data, image_type):
#     try:
#         decoded_image = base64.b64decode(image_data)
#         image = Image.open(io.BytesIO(decoded_image))

#         # 检查图片格式是否与声明的类型匹配
#         # print("image.format", image.format)
#         if image_type == "image/png" and image.format != "PNG":
#             raise ValueError("Image is not a valid PNG")
#         elif image_type == "image/jpeg" and image.format not in ["JPEG", "JPG"]:
#             raise ValueError("Image is not a valid JPEG")

#         # 如果没有异常,则图片有效
#         return True
#     except Exception as e:
#         print(f"Image validation failed: {str(e)}")
#         return False

async def get_image_message(base64_image, engine = None):
    if base64_image.startswith("http"):
        base64_image = await get_encode_image(base64_image)
    colon_index = base64_image.index(":")
    semicolon_index = base64_image.index(";")
    image_type = base64_image[colon_index + 1:semicolon_index]

    if image_type == "image/webp":
        # 将webp转换为png

        # 解码base64获取图片数据
        image_data = base64.b64decode(base64_image.split(",")[1])

        # 使用PIL打开webp图片
        image = Image.open(io.BytesIO(image_data))

        # 转换为PNG格式
        png_buffer = io.BytesIO()
        image.save(png_buffer, format="PNG")
        png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

        # 返回PNG格式的base64
        base64_image = f"data:image/png;base64,{png_base64}"
        image_type = "image/png"

    if "gpt" == engine or "openrouter" == engine or "azure" == engine:
        return {
            "type": "image_url",
            "image_url": {
                "url": base64_image,
            }
        }
    if "claude" == engine or "vertex-claude" == engine:
        # if not validate_image(base64_image.split(",")[1], image_type):
        #     raise ValueError(f"Invalid image format. Expected {image_type}")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_type,
                "data": base64_image.split(",")[1],
            }
        }
    if "gemini" == engine or "vertex-gemini" == engine:
        return {
            "inlineData": {
                "mimeType": image_type,
                "data": base64_image.split(",")[1],
            }
        }
    raise ValueError("Unknown engine")

async def get_text_message(role, message, engine = None):
    if "gpt" == engine or "claude" == engine or "openrouter" == engine or "vertex-claude" == engine or "o1" == engine or "azure" == engine:
        return {"type": "text", "text": message}
    if "gemini" == engine or "vertex-gemini" == engine:
        return {"text": message}
    if engine == "cloudflare":
        return message
    if engine == "cohere":
        return message
    raise ValueError("Unknown engine")

async def get_gemini_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    gemini_stream = "streamGenerateContent"
    url = provider['base_url']
    parsed_url = urllib.parse.urlparse(url)
    # print("parsed_url", parsed_url)
    if parsed_url.path.endswith("/v1beta") or parsed_url.path.endswith("/v1"):
        api_version = parsed_url.path.split('/')[-1]  # 获取 v1 或 v1beta
    else:
        api_version = "v1beta"
    # https://generativelanguage.googleapis.com/v1beta/models/
    url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/models/{model}:{gemini_stream}?key={await provider_api_circular_list[provider['provider']].next(model)}"

    messages = []
    systemInstruction = None
    function_arguments = None
    for msg in request.messages:
        if msg.role == "assistant":
            msg.role = "model"
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = [{"text": msg.content}]
            tool_calls = msg.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_arguments = {
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
            }
            messages.append(
                {
                    "role": "model",
                    "parts": [function_arguments]
                }
            )
        elif msg.role == "tool":
            function_call_name = function_arguments["functionCall"]["name"]
            messages.append(
                {
                    "role": "function",
                    "parts": [{
                    "functionResponse": {
                        "name": function_call_name,
                        "response": {
                            "name": function_call_name,
                            "content": {
                                "result": msg.content,
                            }
                        }
                    }
                    }]
                }
            )
        elif msg.role != "system":
            messages.append({"role": msg.role, "parts": content})
        elif msg.role == "system":
            content[0]["text"] = re.sub(r"_+", "_", content[0]["text"])
            systemInstruction = {"parts": content}

    if "gemini-2.0-flash-exp" in model or "gemini-1.5" in model:
        safety_settings = "OFF"
    else:
        safety_settings = "BLOCK_NONE"

    payload = {
        "contents": messages or [{"role": "user", "parts": [{"text": "No messages"}]}],
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": safety_settings
            }
        ]
    }

    if systemInstruction:
        if api_version == "v1beta":
            payload["systemInstruction"] = systemInstruction
        if api_version == "v1":
            first_message = safe_get(payload, "contents", 0, "parts", 0, "text", default=None)
            system_instruction = safe_get(systemInstruction, "parts", 0, "text", default=None)
            if first_message and system_instruction:
                payload["contents"][0]["parts"][0]["text"] = system_instruction + "\n" + first_message

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'response_format'
    ]
    generation_config = {}

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools":
                # 处理每个工具的 function 定义
                processed_tools = []
                for tool in value:
                    function_def = tool["function"]
                    # 处理 parameters.properties 中的 default 字段
                    if safe_get(function_def, "parameters", "properties", default=None):
                        for prop_value in function_def["parameters"]["properties"].values():
                            if "default" in prop_value:
                                # 将 default 值添加到 description 中
                                default_value = prop_value["default"]
                                description = prop_value.get("description", "")
                                prop_value["description"] = f"{description}\nDefault: {default_value}"
                                # 删除 default 字段
                                del prop_value["default"]
                    if function_def["name"] != "googleSearch" and function_def["name"] != "googleSearch":
                        processed_tools.append({"function": function_def})

                if processed_tools:
                    payload.update({
                        "tools": [{
                            "function_declarations": [tool["function"] for tool in processed_tools]
                        }],
                        "tool_config": {
                            "function_calling_config": {
                                "mode": "AUTO"
                            }
                        }
                    })
            elif field == "temperature":
                generation_config["temperature"] = value
            elif field == "max_tokens":
                generation_config["maxOutputTokens"] = value
            elif field == "top_p":
                generation_config["topP"] = value
            else:
                payload[field] = value

    if generation_config:
        payload["generationConfig"] = generation_config
        if "maxOutputTokens" not in generation_config:
            payload["generationConfig"]["maxOutputTokens"] = 8192

    if request.model.endswith("-search"):
        if "tools" not in payload:
            payload["tools"] = [{
                "googleSearch": {}
            }]
        else:
            payload["tools"].append({
                "googleSearch": {}
            })

    return url, headers, payload

import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

def create_jwt(client_email, private_key):
    # JWT Header
    header = json.dumps({
        "alg": "RS256",
        "typ": "JWT"
    }).encode()

    # JWT Payload
    now = int(time.time())
    payload = json.dumps({
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now + 3600,
        "iat": now
    }).encode()

    # Encode header and payload
    segments = [
        base64.urlsafe_b64encode(header).rstrip(b'='),
        base64.urlsafe_b64encode(payload).rstrip(b'=')
    ]

    # Create signature
    signing_input = b'.'.join(segments)
    private_key = load_pem_private_key(private_key.encode(), password=None)
    signature = private_key.sign(
        signing_input,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    segments.append(base64.urlsafe_b64encode(signature).rstrip(b'='))
    return b'.'.join(segments).decode()

def get_access_token(client_email, private_key):
    jwt = create_jwt(client_email, private_key)

    with httpx.Client() as client:
        response = client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt
            },
            headers={'Content-Type': "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return response.json()["access_token"]

async def get_vertex_gemini_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    gemini_stream = "streamGenerateContent"
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    search_tool = None

    if "gemini-2.0" in model or "gemini-exp" in model:
        location = gemini2
        search_tool = {"googleSearch": {}}
    else:
        location = gemini1
        search_tool = {"googleSearchRetrieval": {}}

    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(LOCATION=await location.next(), PROJECT_ID=project_id, MODEL_ID=model, stream=gemini_stream)

    messages = []
    systemInstruction = None
    function_arguments = None
    for msg in request.messages:
        if msg.role == "assistant":
            msg.role = "model"
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = [{"text": msg.content}]
            tool_calls = msg.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_arguments = {
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
            }
            messages.append(
                {
                    "role": "model",
                    "parts": [function_arguments]
                }
            )
        elif msg.role == "tool":
            function_call_name = function_arguments["functionCall"]["name"]
            messages.append(
                {
                    "role": "function",
                    "parts": [{
                    "functionResponse": {
                        "name": function_call_name,
                        "response": {
                            "name": function_call_name,
                            "content": {
                                "result": msg.content,
                            }
                        }
                    }
                    }]
                }
            )
        elif msg.role != "system":
            messages.append({"role": msg.role, "parts": content})
        elif msg.role == "system":
            systemInstruction = {"parts": content}


    payload = {
        "contents": messages,
        # "safetySettings": [
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_NONE"
        #     }
        # ]
    }
    if systemInstruction:
        payload["system_instruction"] = systemInstruction

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs'
    ]
    generation_config = {}

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools":
                payload.update({
                    "tools": [{
                        "function_declarations": [tool["function"] for tool in value]
                    }],
                    "tool_config": {
                        "function_calling_config": {
                            "mode": "AUTO"
                        }
                    }
                })
            elif field == "temperature":
                generation_config["temperature"] = value
            elif field == "max_tokens":
                generation_config["max_output_tokens"] = value
            elif field == "top_p":
                generation_config["top_p"] = value
            else:
                payload[field] = value

    if generation_config:
        payload["generationConfig"] = generation_config
        if "max_output_tokens" not in generation_config:
            payload["generationConfig"]["max_output_tokens"] = 8192

    if request.model.endswith("-search"):
        if "tools" not in payload:
            payload["tools"] = [search_tool]
        else:
            payload["tools"].append(search_tool)

    return url, headers, payload

async def get_vertex_claude_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json',
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    if "claude-3-5-sonnet" in model or "claude-3-7-sonnet" in model:
        location = c35s
    elif "claude-3-opus" in model:
        location = c3o
    elif "claude-3-sonnet" in model:
        location = c3s
    elif "claude-3-haiku" in model:
        location = c3h

    claude_stream = "streamRawPredict"
    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:{stream}".format(LOCATION=await location.next(), PROJECT_ID=project_id, MODEL=model, stream=claude_stream)

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    model_dict = get_model_dict(provider)
    model = model_dict[request.model]

    if "claude-3-7-sonnet" in model:
        max_tokens = 20000
    elif "claude-3-5-sonnet" in model:
        max_tokens = 8192
    else:
        max_tokens = 4096

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    return url, headers, payload

async def get_gpt_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"

    elif provider['provider'].startswith("sk-"):
        headers['Authorization'] = f"Bearer {provider['provider']}"

    url = provider['base_url']

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True) and "o1-mini" not in model:
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            if msg.role == "system" and "o3-mini" in model and not content.startswith("Formatting re-enabled"):
                content = "Formatting re-enabled. " + content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            messages.append({"role": msg.role, "content": content})

    if ("o1-mini" in model or "o1-preview" in model) and len(messages) > 1 and messages[0]["role"] == "system":
        system_msg = messages.pop(0)
        messages[0]["content"] = system_msg["content"] + messages[0]["content"]

    payload = {
        "model": model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and ("o1" in model or "o3" in model):
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") == False or "o1" in model or "chatgpt-4o-latest" in model or "grok" in model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    if "models.inference.ai.azure.com" in url:
        payload["stream"] = False
        # request.stream = False
        payload.pop("stream_options", None)

    if "o3-mini" in model:
        if request.model.endswith("high"):
            payload["reasoning_effort"] = "high"
        elif request.model.endswith("low"):
            payload["reasoning_effort"] = "low"
        else:
            payload["reasoning_effort"] = "medium"

    if "o3-mini" in model or "o1" in model:
        if "temperature" in payload:
            payload.pop("temperature")

    if "deepseek-r" in model.lower():
        if "temperature" not in payload:
            payload["temperature"] = 0.6

    if request.model.endswith("-search") and "gemini" in request.model:
        if "tools" not in payload:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "googleSearch",
                    "description": "googleSearch"
                }
            }]
        else:
            if not any(tool["function"]["name"] == "googleSearch" for tool in payload["tools"]):
                payload["tools"].append({
                    "type": "function",
                    "function": {
                        "name": "googleSearch",
                        "description": "googleSearch"
                    }
                })

    return url, headers, payload

def build_azure_endpoint(base_url, deployment_id, api_version="2024-10-21"):
    # 移除base_url末尾的斜杠(如果有)
    base_url = base_url.rstrip('/')
    final_url = base_url

    if "models/chat/completions" not in final_url:
        # 构建路径
        path = f"/openai/deployments/{deployment_id}/chat/completions"
        # 使用urljoin拼接base_url和path
        final_url = urllib.parse.urljoin(base_url, path)

    if "?api-version=" not in final_url:
        # 添加api-version查询参数
        final_url = f"{final_url}?api-version={api_version}"

    return final_url

async def get_azure_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers['api-key'] = f"{await provider_api_circular_list[provider['provider']].next(model)}"

    url = build_azure_endpoint(
        base_url=provider['base_url'],
        deployment_id=model,
    )

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True) and "o1-mini" not in model:
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            messages.append({"role": msg.role, "content": content})

    payload = {
        "model": model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and "o1" in model:
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") == False or "o1" in model or "chatgpt-4o-latest" in model or "grok" in model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    return url, headers, payload

async def get_openrouter_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"

    elif provider['provider'].startswith("sk-"):
        headers['Authorization'] = f"Bearer {provider['provider']}"

    url = provider['base_url']

    messages = []
    for msg in request.messages:
        name = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            name = msg.name
        if name:
            messages.append({"role": msg.role, "name": name, "content": content})
        else:
            # print("content", content)
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        messages.append({"role": msg.role, "content": item["text"]})
                    elif item["type"] == "image_url":
                        messages.append({"role": msg.role, "content": [await get_image_message(item["image_url"]["url"], engine)]})
            else:
                messages.append({"role": msg.role, "content": content})

    payload = {
        "model": model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
        'n',
        'user',
        'include_usage',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def get_cohere_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"

    url = provider['base_url']

    role_map = {
        "user": "USER",
        "assistant" : "CHATBOT",
        "system": "SYSTEM"
    }

    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
        else:
            content = msg.content

        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    messages.append({"role": role_map[msg.role], "message": item["text"]})
        else:
            messages.append({"role": role_map[msg.role], "message": content})

    chat_history = messages[:-1]
    query = messages[-1].get("message")
    payload = {
        "model": model,
        "message": query,
    }

    if chat_history:
        payload["chat_history"] = chat_history

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs'
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def get_cloudflare_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"

    url = "https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{cf_model_id}".format(cf_account_id=provider['cf_account_id'], cf_model_id=model)

    msg = request.messages[-1]
    messages = []
    content = None
    if isinstance(msg.content, list):
        for item in msg.content:
            if item.type == "text":
                content = await get_text_message(msg.role, item.text, engine)
    else:
        content = msg.content
        name = msg.name

    payload = {
        "prompt": content,
    }

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs'
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def gpt2claude_tools_json(json_dict):
    import copy
    json_dict = copy.deepcopy(json_dict)

    # 处理 $ref 引用
    def resolve_refs(obj, defs):
        if isinstance(obj, dict):
            # 如果有 $ref 引用，替换为实际定义
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    # 完全替换为引用的对象
                    ref_obj = copy.deepcopy(defs[ref_name])
                    # 保留原始对象中的其他属性
                    for k, v in obj.items():
                        if k != "$ref":
                            ref_obj[k] = v
                    return ref_obj

            # 递归处理所有属性
            for key, value in list(obj.items()):
                obj[key] = resolve_refs(value, defs)

        elif isinstance(obj, list):
            # 递归处理列表中的每个元素
            for i, item in enumerate(obj):
                obj[i] = resolve_refs(item, defs)

        return obj

    # 提取 $defs 定义
    defs = {}
    if "parameters" in json_dict and "defs" in json_dict["parameters"]:
        defs = json_dict["parameters"]["defs"]
        # 从参数中删除 $defs，因为 Claude 不需要它
        del json_dict["parameters"]["defs"]

    # 解析所有引用
    json_dict = resolve_refs(json_dict, defs)

    # 继续原有的键名转换逻辑
    keys_to_change = {
        "parameters": "input_schema",
    }
    for old_key, new_key in keys_to_change.items():
        if old_key in json_dict:
            if new_key:
                if json_dict[old_key] == None:
                    json_dict[old_key] = {
                        "type": "object",
                        "properties": {}
                    }
                json_dict[new_key] = json_dict.pop(old_key)
            else:
                json_dict.pop(old_key)
    return json_dict

async def get_claude_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]

    if "claude-3-7-sonnet" in model:
        anthropic_beta = "output-128k-2025-02-19"
    elif "claude-3-5-sonnet" in model:
        anthropic_beta = "max-tokens-3-5-sonnet-2024-07-15"
    else:
        anthropic_beta = "tools-2024-05-16"

    headers = {
        "content-type": "application/json",
        "x-api-key": f"{await provider_api_circular_list[provider['provider']].next(model)}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": anthropic_beta,
    }
    url = provider['base_url']

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    model_dict = get_model_dict(provider)

    model = model_dict[request.model]

    if "claude-3-7-sonnet" in model:
        max_tokens = 20000
    elif "claude-3-5-sonnet" in model:
        max_tokens = 8192
    else:
        max_tokens = 4096

    payload = {
        "model": model,
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            # print("tool", type(tool), tool)
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "think" in request.model:
        payload["thinking"] = {
            "budget_tokens": 4096,
            "type": "enabled"
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        if request.model.split("-")[-1].isdigit():
            think_tokens = int(request.model.split("-")[-1])
            if think_tokens < max_tokens:
                payload["thinking"] = {
                    "budget_tokens": think_tokens,
                    "type": "enabled"
                }

    if request.thinking:
        payload["thinking"] = {
            "budget_tokens": request.thinking.budget_tokens,
            "type": request.thinking.type
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
    # print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    return url, headers, payload

async def get_dalle_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"
    url = provider['base_url']
    url = BaseAPI(url).image_url

    payload = {
        "model": model,
        "prompt": request.prompt,
        "n": request.n,
        "response_format": request.response_format,
        "size": request.size
    }

    return url, headers, payload

async def get_whisper_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers = {
        # "Content-Type": "multipart/form-data",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"
    url = provider['base_url']
    url = BaseAPI(url).audio_transcriptions

    payload = {
        "model": model,
        "file": request.file,
    }

    if request.prompt:
        payload["prompt"] = request.prompt
    if request.response_format:
        payload["response_format"] = request.response_format
    if request.temperature:
        payload["temperature"] = request.temperature
    if request.language:
        payload["language"] = request.language

    return url, headers, payload

async def get_moderation_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"
    url = provider['base_url']
    url = BaseAPI(url).moderations

    payload = {
        "input": request.input,
    }

    return url, headers, payload

async def get_embedding_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"
    url = provider['base_url']
    url = BaseAPI(url).embeddings

    payload = {
        "input": request.input,
        "model": model,
    }

    if request.encoding_format:
        if url.startswith("https://api.jina.ai"):
            payload["embedding_type"] = request.encoding_format
        else:
            payload["encoding_format"] = request.encoding_format

    return url, headers, payload

async def get_tts_payload(request, engine, provider):
    model_dict = get_model_dict(provider)
    model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {await provider_api_circular_list[provider['provider']].next(model)}"
    url = provider['base_url']
    url = BaseAPI(url).audio_speech

    payload = {
        "model": model,
        "input": request.input,
        "voice": request.voice,
    }

    if request.response_format:
        payload["response_format"] = request.response_format
    if request.speed:
        payload["speed"] = request.speed
    if request.stream is not None:
        payload["stream"] = request.stream

    return url, headers, payload


async def get_payload(request: RequestModel, engine, provider):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider)
    elif engine == "vertex-gemini":
        return await get_vertex_gemini_payload(request, engine, provider)
    elif engine == "vertex-claude":
        return await get_vertex_claude_payload(request, engine, provider)
    elif engine == "azure":
        return await get_azure_payload(request, engine, provider)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider)
    elif engine == "gpt":
        return await get_gpt_payload(request, engine, provider)
    elif engine == "openrouter":
        return await get_openrouter_payload(request, engine, provider)
    elif engine == "cloudflare":
        return await get_cloudflare_payload(request, engine, provider)
    elif engine == "cohere":
        return await get_cohere_payload(request, engine, provider)
    elif engine == "dalle":
        return await get_dalle_payload(request, engine, provider)
    elif engine == "whisper":
        return await get_whisper_payload(request, engine, provider)
    elif engine == "tts":
        return await get_tts_payload(request, engine, provider)
    elif engine == "moderation":
        return await get_moderation_payload(request, engine, provider)
    elif engine == "embedding":
        return await get_embedding_payload(request, engine, provider)
    else:
        raise ValueError("Unknown payload")