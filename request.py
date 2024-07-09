from models import RequestModel

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
    if "gpt" == engine or "claude" == engine:
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
    for msg in request.messages:
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
            content = msg.content
        if msg.role != "system":
            messages.append({"role": msg.role, "parts": content})


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
    headers = {
        "content-type": "application/json",
        "x-api-key": f"{provider['api']}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024-05-16"
    }
    url = provider['base_url']

    messages = []
    for msg in request.messages:
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
            # messages.append({"role": "assistant", "name": name, "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01RofFmKHUKsEaZvqESG5Hwz",
                            "name": name,
                            "input": {"text": messages[-1]["content"][0]["text"]},
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

    model = provider['model'][request.model]
    payload = {
        "model": model,
        "messages": messages,
        "system": system_prompt,
    }
    # json_post = {
    #     "model": model or self.engine,
    #     "messages": self.conversation[convo_id] if pass_history else [{
    #         "role": "user",
    #         "content": prompt
    #     }],
    #     "temperature": kwargs.get("temperature", self.temperature),
    #     "top_p": kwargs.get("top_p", self.top_p),
    #     "max_tokens": model_max_tokens,
    #     "stream": True,
    # }

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

    if request.tools:
        tools = []
        for tool in request.tools:
            print("tool", type(tool), tool)

            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            payload["tool_choice"] = {
                "type": "auto"
            }
    import json
    print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    return url, headers, payload

async def get_payload(request: RequestModel, engine, provider):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider)
    elif engine == "gpt":
        return await get_gpt_payload(request, engine, provider)
    else:
        raise ValueError("Unknown payload")