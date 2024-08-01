import json
from models import RequestModel
from log_config import logger

async def get_image_message(base64_image, engine = None):
    if "gpt" == engine:
        return {
            "type": "image_url",
            "image_url": {
                "url": base64_image,
            }
        }
    if "claude" == engine:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64_image.split(",")[1],
            }
        }
    if "gemini" == engine:
        return {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_image.split(",")[1],
            }
        }
    raise ValueError("Unknown engine")

async def get_text_message(role, message, engine = None):
    if "gpt" == engine or "claude" == engine or "openrouter" == engine:
        return {"type": "text", "text": message}
    if "gemini" == engine:
        return {"text": message}
    raise ValueError("Unknown engine")

async def get_gemini_payload(request, engine, provider):
    headers = {
        'Content-Type': 'application/json'
    }
    url = provider['base_url']
    model = provider['model'][request.model]
    if request.stream:
        gemini_stream = "streamGenerateContent"
    url = url.format(model=model, stream=gemini_stream, api_key=provider['api'])

    messages = []
    systemInstruction = None
    for msg in request.messages:
        if msg.role == "assistant":
            msg.role = "model"
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(msg.role, item.text, engine)
                    # print("text_message", text_message)
                    content.append(text_message)
                elif item.type == "image_url":
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = [{"text": msg.content}]
        if msg.role != "system":
            messages.append({"role": msg.role, "parts": content})
        if msg.role == "system":
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

async def get_gpt_payload(request, engine, provider):
    headers = {
        'Authorization': f"Bearer {provider['api']}",
        'Content-Type': 'application/json'
    }
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
        headers['Authorization'] = f"Bearer {provider['api']}"

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

async def gpt2claude_tools_json(json_dict):
    import copy
    json_dict = copy.deepcopy(json_dict)
    keys_to_change = {
        "parameters": "input_schema",
    }
    for old_key, new_key in keys_to_change.items():
        if old_key in json_dict:
            if new_key:
                json_dict[new_key] = json_dict.pop(old_key)
            else:
                json_dict.pop(old_key)
    # if "tools" in json_dict.keys():
    #     json_dict["tool_choice"] = {
    #         "type": "auto"
    #     }
    return json_dict

async def get_claude_payload(request, engine, provider):
    model = provider['model'][request.model]
    headers = {
        "content-type": "application/json",
        "x-api-key": f"{provider['api']}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" if "claude-3-5-sonnet" in model else "tools-2024-05-16",
    }
    url = provider['base_url']

    messages = []
    system_prompt = None
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
            arguments = msg.arguments
            if arguments:
                arguments = json.loads(arguments)
        if name:
            # messages.append({"role": "assistant", "name": name, "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01RofFmKHUKsEaZvqESG5Hwz",
                            "name": name,
                            "input": arguments,
                        }
                    ]
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01RofFmKHUKsEaZvqESG5Hwz",
                            "content": content
                        }
                    ]
                }
            )
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
    }

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
            payload["tool_choice"] = {
                "type": "auto"
            }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    # print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    return url, headers, payload

async def get_payload(request: RequestModel, engine, provider):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider)
    elif engine == "gpt":
        return await get_gpt_payload(request, engine, provider)
    elif engine == "openrouter":
        return await get_openrouter_payload(request, engine, provider)
    else:
        raise ValueError("Unknown payload")