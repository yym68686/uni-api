import os
import re
import json
import httpx
import base64
import urllib.parse

from models import RequestModel
from utils import c35s, c3s, c3o, c3h, gem, BaseAPI

import imghdr

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
        file_type = imghdr.what(None, file_content)
        base64_encoded = base64.b64encode(file_content).decode('utf-8')

        if file_type == 'png':
            return f"data:image/png;base64,{base64_encoded}"
        elif file_type in ['jpeg', 'jpg']:
            return f"data:image/jpeg;base64,{base64_encoded}"
        else:
            raise ValueError(f"不支持的图片格式: {file_type}")

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

    if "gpt" == engine:
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
    if "gpt" == engine or "claude" == engine or "openrouter" == engine or "vertex-claude" == engine or "o1" == engine:
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
    model = provider['model'][request.model]
    gemini_stream = "streamGenerateContent"
    url = provider['base_url']
    if url.endswith("v1beta"):
        url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{stream}?key={api_key}".format(model=model, stream=gemini_stream, api_key=provider['api'].next())
    if url.endswith("v1"):
        url = "https://generativelanguage.googleapis.com/v1/models/{model}:{stream}?key={api_key}".format(model=model, stream=gemini_stream, api_key=provider['api'].next())

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
                elif item.type == "image_url":
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


    payload = {
        "contents": messages,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }
    if systemInstruction:
        payload["systemInstruction"] = systemInstruction

    miss_fields = [
        'model',
        'messages',
        'stream',
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
            else:
                payload[field] = value

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
    model = provider['model'][request.model]
    location = gem
    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(LOCATION=location.next(), PROJECT_ID=project_id, MODEL_ID=model, stream=gemini_stream)

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
                elif item.type == "image_url":
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
        "generationConfig": {
            "temperature": 0.5,
            "max_output_tokens": 8192,
            "top_k": 40,
            "top_p": 0.95
        },
    }
    if systemInstruction:
        payload["system_instruction"] = systemInstruction

    miss_fields = [
        'model',
        'messages',
        'stream',
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
            else:
                payload[field] = value

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

    model = provider['model'][request.model]
    if "claude-3-5-sonnet" in model:
        location = c35s
    elif "claude-3-opus" in model:
        location = c3o
    elif "claude-3-sonnet" in model:
        location = c3s
    elif "claude-3-haiku" in model:
        location = c3h

    claude_stream = "streamRawPredict"
    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:{stream}".format(LOCATION=location.next(), PROJECT_ID=project_id, MODEL=model, stream=claude_stream)

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
                elif item.type == "image_url":
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

    model = provider['model'][request.model]
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": 8192 if "claude-3-5-sonnet" in model else 4096,
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
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"
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
                elif item.type == "image_url":
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

    model = provider['model'][request.model]
    payload = {
        "model": model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages'
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    return url, headers, payload

async def get_openrouter_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"

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
                elif item.type == "image_url":
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
                        messages.append({"role": msg.role, "content": item["url"]})
            else:
                messages.append({"role": msg.role, "content": content})

    model = provider['model'][request.model]
    payload = {
        "model": model,
        "messages": messages,
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

async def get_cohere_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"

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

    model = provider['model'][request.model]
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
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"

    model = provider['model'][request.model]
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

    model = provider['model'][request.model]
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

async def get_o1_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"

    url = provider['base_url']

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

        if isinstance(content, list) and msg.role != "system":
            for item in content:
                if item["type"] == "text":
                    messages.append({"role": msg.role, "content": item["text"]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})

    model = provider['model'][request.model]
    payload = {
        "model": model,
        "messages": messages,
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
    model = provider['model'][request.model]
    headers = {
        "content-type": "application/json",
        "x-api-key": f"{provider['api'].next()}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" if "claude-3-5-sonnet" in model else "tools-2024-05-16",
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
                elif item.type == "image_url":
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

    model = provider['model'][request.model]
    payload = {
        "model": model,
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": 8192 if "claude-3-5-sonnet" in model else 4096,
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

    # print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    return url, headers, payload

async def get_dalle_payload(request, engine, provider):
    model = provider['model'][request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"
    url = provider['base_url']
    url = BaseAPI(url).image_url

    payload = {
        "model": model,
        "prompt": request.prompt,
        "n": request.n,
        "size": request.size
    }

    return url, headers, payload

async def get_whisper_payload(request, engine, provider):
    model = provider['model'][request.model]
    headers = {
        "Content-Type": "multipart/form-data",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"
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
    model = provider['model'][request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if provider.get("api"):
        headers['Authorization'] = f"Bearer {provider['api'].next()}"
    url = provider['base_url']
    url = BaseAPI(url).moderations

    payload = {
        "input": request.input,
    }

    return url, headers, payload

async def get_payload(request: RequestModel, engine, provider):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider)
    elif engine == "vertex-gemini":
        return await get_vertex_gemini_payload(request, engine, provider)
    elif engine == "vertex-claude":
        return await get_vertex_claude_payload(request, engine, provider)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider)
    elif engine == "gpt":
        return await get_gpt_payload(request, engine, provider)
    elif engine == "openrouter":
        return await get_openrouter_payload(request, engine, provider)
    elif engine == "cloudflare":
        return await get_cloudflare_payload(request, engine, provider)
    elif engine == "o1":
        return await get_o1_payload(request, engine, provider)
    elif engine == "cohere":
        return await get_cohere_payload(request, engine, provider)
    elif engine == "dalle":
        return await get_dalle_payload(request, engine, provider)
    elif engine == "whisper":
        return await get_whisper_payload(request, engine, provider)
    elif engine == "moderation":
        return await get_moderation_payload(request, engine, provider)
    else:
        raise ValueError("Unknown payload")