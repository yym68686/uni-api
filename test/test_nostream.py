import requests
import base64
import json
import os
from datetime import datetime

# 設置API密鑰和自定義base URL
API_KEY = ''
BASE_URL = 'http://localhost:8000/v1'
SAVE_DIR = 'safe_output'  # 保存 JSON 輸出的目錄

def ensure_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(image_base64):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_underlined_text",
                "description": "從圖片中提取紅色下劃線的文字",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "underlined_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "紅色下劃線的文字列表"
                        }
                    },
                    "required": ["underlined_text"]
                }
            }
        }
    ]

    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "請仔細分析圖片，並提取所有使用紅色筆在單字、單詞或句子下方畫有橫線的文字。只提取有紅色下劃線的文字，忽略其他未標記的文字。將結果以 JSON 格式輸出，格式為 {\"underlined_text\": [\"文字1\", \"文字2\", ...]}。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        # "stream": True,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "extract_underlined_text"}},
        "max_tokens": 1000
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def save_json_output(data):
    ensure_save_directory()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{SAVE_DIR}/output_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename

def main(image_path):
    image_base64 = image_to_base64(image_path)

    response = get_model_response(image_base64)

    print("模型回應:")
    print(json.dumps(response, indent=2, ensure_ascii=False))

    if isinstance(response, str) and response.startswith("Error"):
        print(response)
        return

    if 'choices' in response and response['choices']:
        message = response['choices'][0]['message']
        if 'tool_calls' in message:
            tool_call = message['tool_calls'][0]
            if tool_call['function']['name'] == 'extract_underlined_text':
                function_args = json.loads(tool_call['function']['arguments'])
                print("\n提取的紅色下劃線文字:")
                print(json.dumps(function_args, indent=2, ensure_ascii=False))

                # 保存 JSON 輸出
                saved_file = save_json_output(function_args)
                print(f"\nJSON 輸出已保存至: {saved_file}")
            else:
                print("\n模型調用了未預期的函數。")
        else:
            print("\n模型沒有調用工具。")
    else:
        print("\n無法解析回應。")

if __name__ == "__main__":
    image_path = "1.jpg"  # 替換為您的圖像路徑
    main(image_path)