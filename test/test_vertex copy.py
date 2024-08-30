import json
import base64
import time
import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 您的服务账号密钥（请将其保存在安全的地方，不要公开分享）
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

def ask_stream(prompt, client_email, private_key, project_id, engine):
    payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": prompt
                }
            ]
        }
    ],
    "system_instruction": {
        "parts": [
            {
                "text": "You are Gemini, a large language model trained by Google. Respond conversationally"
            }
        ]
    },
    # "safety_settings": [
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
    # ],
    "generationConfig": {
        "temperature": 0.5,
        "max_output_tokens": 256,
        "top_k": 40,
        "top_p": 0.95
    },
    "tools": [
        {
            "function_declarations": [
                {
                    "name": "get_search_results",
                    "description": "Search Google to enhance knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to search."
                            }
                        },
                        "required": [
                            "prompt"
                        ]
                    }
                },
                {
                    "name": "get_url_content",
                    "description": "Get the webpage content of a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "the URL to request"
                            }
                        },
                        "required": [
                            "url"
                        ]
                    }
                }
            ]
        }
    ],
    "tool_config": {
        "function_calling_config": {
            "mode": "AUTO"
        }
    }
}
    # payload = {
    #     "contents": [
    #         {
    #             "role": "user",
    #             "parts": [
    #                 {
    #                     "text": prompt
    #                 }
    #             ]
    #         },
    #     ],
    #     "generationConfig": {
    #         "temperature": 0.2,
    #         "maxOutputTokens": 256,
    #         "topK": 40,
    #         "topP": 0.95
    #     }
    # }

    access_token = get_access_token(client_email, private_key)
    headers = {
        'Authorization': f"Bearer {access_token}",
        'Content-Type': "application/json"
    }

    MODEL_ID = engine
    PROJECT_ID = project_id
    stream = "generateContent"
    with httpx.Client() as client:
        response = client.post(
            f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/{MODEL_ID}:{stream}",
            json=payload,
            headers=headers,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

# 使用示例
client_email, private_key, project_id = SERVICE_ACCOUNT_KEY["client_email"], SERVICE_ACCOUNT_KEY["private_key"], SERVICE_ACCOUNT_KEY["project_id"]
engine = "gemini-1.5-pro"
user_input = input("请输入您的问题： ")
result = ask_stream(user_input, client_email, private_key, project_id, engine)
print(json.dumps(result, ensure_ascii=False, indent=2))