import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import load_config

provider_name = "linuxdoi"
model = "claude-3-5-sonnet"
import asyncio
config, api_keys_db, api_list = asyncio.run(load_config())
import json

print(json.dumps(api_keys_db, indent=2))
exit(0)
providers = config["providers"]
provider_config = None
for provider in providers:
    if provider["provider"] == provider_name:
        provider_config = provider
        break
if provider_config == None:
    print("Provider not found")
    sys.exit(1)
model_name = provider_config["model"][model]
# 定义请求的内容
request_content = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": {"text": "What is the meaning of life?"}}
    ],
    "stream": True
}

# 将请求内容转换为JSON字符串
request_json = json.dumps(request_content)

# 定义curl命令
curl_command = f"""
curl {provider_config["base_url"]} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {provider_config['api']}" \\
  -d '{request_json}'
"""

# 打印生成的curl命令
print(curl_command)