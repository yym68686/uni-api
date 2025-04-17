import httpx
import json
import os
import datetime
from datetime import timezone
import hashlib
import hmac
import urllib.parse
import base64 # 新增导入，用于解码流式响应块

# --- AWS Signature V4 Helper Functions ---

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    if not data:
        return default
    return data

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing

def get_signature(request_body, model_id, aws_access_key, aws_secret_key, aws_region, host, content_type, accept_header):
    request_body = json.dumps(request_body)
    SERVICE = "bedrock"
    canonical_querystring = ''
    method = 'POST'
    raw_path = f'/model/{model_id}/invoke-with-response-stream'
    canonical_uri = urllib.parse.quote(raw_path, safe='/-_.~')
    # Create a date for headers and the credential string
    t = datetime.datetime.now(timezone.utc)
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d') # Date YYYYMMDD

    # --- Task 1: Create a Canonical Request ---
    payload_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()

    canonical_headers = f'accept:{accept_header}\n' \
                        f'content-type:{content_type}\n' \
                        f'host:{host}\n' \
                        f'x-amz-bedrock-accept:{accept_header}\n' \
                        f'x-amz-content-sha256:{payload_hash}\n' \
                        f'x-amz-date:{amz_date}\n'
    # 注意：头名称需要按字母顺序排序

    signed_headers = 'accept;content-type;host;x-amz-bedrock-accept;x-amz-content-sha256;x-amz-date' # 按字母顺序排序

    canonical_request = f'{method}\n' \
                        f'{canonical_uri}\n' \
                        f'{canonical_querystring}\n' \
                        f'{canonical_headers}\n' \
                        f'{signed_headers}\n' \
                        f'{payload_hash}'

    # --- Task 2: Create the String to Sign ---
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date_stamp}/{aws_region}/{SERVICE}/aws4_request'
    string_to_sign = f'{algorithm}\n' \
                    f'{amz_date}\n' \
                    f'{credential_scope}\n' \
                    f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'

    # --- Task 3: Calculate the Signature ---
    signing_key = get_signature_key(aws_secret_key, date_stamp, aws_region, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    # --- Task 4: Add Signing Information to the Request ---
    authorization_header = f'{algorithm} Credential={aws_access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}'
    return amz_date, payload_hash, authorization_header

# Create request body (不需要 stream: true)
payload = {
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": "hi"}], # 修改了示例提示
  "anthropic_version": "bedrock-2023-05-31"
}

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')

# *** 修改点 4: 更新 Content-Type 和 Accept Header ***
CONTENT_TYPE = "application/json" # 发送内容仍是 JSON
ACCEPT_HEADER = "application/vnd.amazon.bedrock.payload+json" # 指定接受 Bedrock 流格式

AWS_REGION = "us-east-1"
# AWS_REGION = "us-west-2"
HOST = f"bedrock-runtime.{AWS_REGION}.amazonaws.com"
# MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
# MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
# MODEL_ID = "arn:aws:bedrock:us-east-1:390844780199:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
# *** 修改点 5: 更新 URL 指向流式端点 ***
url = f"https://{HOST}/model/{MODEL_ID}/invoke-with-response-stream"

# *** 修改点 6: 在调用 get_signature 时传入 accept_header ***
amz_date, payload_hash, authorization_header = get_signature(
    payload, MODEL_ID, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, HOST, CONTENT_TYPE, ACCEPT_HEADER
)

# *** 修改点 7: 更新请求头，添加 Accept 和 X-Amz-Bedrock-Accept ***
headers = {
    'Accept': ACCEPT_HEADER,
    'Content-Type': CONTENT_TYPE,
    'X-Amz-Date': amz_date,
    'X-Amz-Bedrock-Accept': ACCEPT_HEADER, # Bedrock 特定头
    'X-Amz-Content-Sha256': payload_hash,
    'Authorization': authorization_header,
    # Add 'X-Amz-Security-Token': SESSION_TOKEN if using temporary credentials
}

import re
# --- 发送流式请求并处理响应 ---
# *** 修改点 8: 使用 httpx.stream() 并处理流式响应 ***
try:
    with httpx.Client() as client:
        # 使用 stream() 方法发送请求
        with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
            response.raise_for_status() # 检查初始连接状态

            # print("Streaming response:")
            # 迭代处理返回的字节流
            for line in response.iter_lines(): # 继续使用 iter_lines，但加强处理
                if not line or \
                line.strip() == "" or\
                line.strip().startswith(':content-type') or \
                line.strip().startswith(':event-type'): # 过滤掉完全空的行或只有空白的行
                    continue

                # print(f"DEBUG Raw line: {line!r}") # 取消注释以查看原始行

                json_part = None # 用于存储尝试解析的 JSON 字符串
                try:
                    # 尝试找到 JSON 的起始位置 '{'
                    json_start_index = re.search(r'{.*?}', line)
                    if json_start_index:
                        # print(f"DEBUG json_start_index: {json_start_index.group(0)!r}")
                        json_part = json_start_index.group(0)
                        # 再次检查提取的部分是否为空或无效
                        if not json_part or not json_part.strip():
                            # print(f"\nWarning: Extracted JSON part is empty from line: {line!r}") # 取消注释以查看警告
                            continue

                        # 解析提取出的 JSON 部分
                        chunk_data = json.loads(json_part)
                    else:
                        continue

                    # --- 后续处理逻辑不变 ---
                    if "bytes" in chunk_data:
                        # 解码 Base64 编码的字节
                        decoded_bytes = base64.b64decode(chunk_data["bytes"])
                        # 将解码后的字节再次解析为 JSON
                        payload_chunk = json.loads(decoded_bytes.decode('utf-8'))
                        print(f"DEBUG payload_chunk: {payload_chunk!r}")

                        text = safe_get(payload_chunk, "delta", "text", default="")
                        output_tokens = safe_get(payload_chunk, "usage", "output_tokens", default="")
                        # if text:
                        #     print(text, end="", flush=True) # 打印文本块并立即刷新输出
                        # if output_tokens:
                        #     print(f"--- Output Tokens: {output_tokens} ---")

                    elif "internalServerException" in chunk_data or "modelStreamErrorException" in chunk_data:
                            print(f"\nError chunk received: {chunk_data}")

                    else:
                        # 如果行内没有 '{'，可能只是元数据行或格式不对
                        # print(f"\nSkipping line without JSON start '{{': {line!r}") # 取消注释以查看跳过的行
                        pass # 通常这些只是分隔符或头信息，可以安全忽略

                except json.JSONDecodeError as e:
                    # 提供更详细的错误信息，包括尝试解析的字符串
                    print(f"\nError decoding JSON. Original line: {line!r}. Attempted JSON part: {json_part!r}. Error: {e}")
                except base64.binascii.Error as e:
                    # 捕获 Base64 解码错误
                    chunk_info = chunk_data if 'chunk_data' in locals() else "Unavailable"
                    print(f"\nError decoding Base64. Original line: {line!r}. Chunk data before b64decode: {chunk_info}. Error: {e}")
                except UnicodeDecodeError as e:
                    # 捕获UTF-8解码错误
                    decoded_info = decoded_bytes if 'decoded_bytes' in locals() else "Unavailable"
                    print(f"\nError decoding UTF-8 after Base64. Original line: {line!r}. Decoded bytes: {decoded_info!r}. Error: {e}")
                except Exception as e:
                    # 捕获其他处理错误
                    print(f"\nError processing chunk. Original line: {line!r}. Error: {type(e).__name__}: {e}")

except httpx.RequestError as exc:
    print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
except httpx.HTTPStatusError as exc:
    print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
    try:
        # *** 修改点：使用同步方法读取错误响应体 ***
        # 尝试读取完整的响应体文本
        error_body = exc.response.read().decode('utf-8', errors='replace')
        print(f"Error Body:\n{error_body}")
        # 如果确定错误响应是JSON，可以尝试解析
        # try:
        #     print(f"Error JSON: {json.loads(error_body)}")
        # except json.JSONDecodeError:
        #     pass # 不是JSON也没关系，已经打印了文本
    except Exception as e:
        print(f"Could not read or decode error response body: {type(e).__name__}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {type(e).__name__}: {e}")
