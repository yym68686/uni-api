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
    if request.stream:
        gemini_stream = "streamGenerateContent"
    url = url.format(model=request.model, stream=gemini_stream, api_key=provider['api'])

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
    url = url.format(model=request.model, stream=request.stream, api_key=provider['api'])

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

    payload = {
        "model": request.model,
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

async def get_claude_payload(request, engine, provider):
    pass

async def get_payload(request: RequestModel, engine, provider):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider)
    elif engine == "gpt":
        return await get_gpt_payload(request, engine, provider)
    else:
        raise ValueError("Unknown payload")