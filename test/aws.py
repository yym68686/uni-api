import httpx
import json
from datetime import datetime
import hashlib
import hmac
import base64
import os

# AWS凭证和配置
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
REGION = 'us-east-1'
SERVICE = 'bedrock'
ENDPOINT = f'https://bedrock-runtime.{REGION}.amazonaws.com'

# 辅助函数用于生成AWS SigV4签名
def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, dateStamp, regionName, serviceName):
    kDate = sign(('AWS4' + key).encode('utf-8'), dateStamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'aws4_request')
    return kSigning

def create_authorization_header(method, url, payload, headers):
    t = datetime.utcnow()
    amzdate = t.strftime('%Y%m%dT%H%M%SZ')
    datestamp = t.strftime('%Y%m%d')

    canonical_uri = url.split(ENDPOINT)[1]
    canonical_querystring = ''
    canonical_headers = '\n'.join([f"{h.lower()}:{headers[h]}" for h in sorted(headers)]) + '\n'
    signed_headers = ';'.join([h.lower() for h in sorted(headers)])
    payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()

    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f"{datestamp}/{REGION}/{SERVICE}/aws4_request"
    string_to_sign = f"{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

    signing_key = getSignatureKey(AWS_SECRET_KEY, datestamp, REGION, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    authorization_header = (
        f"{algorithm} "
        f"Credential={AWS_ACCESS_KEY}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    return authorization_header

# 主函数
def invoke_bedrock_model():
    url = f"{ENDPOINT}/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke"

    payload = json.dumps({
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello, world"}],
        "anthropic_version": "bedrock-2023-05-31"
    })

    t = datetime.utcnow()
    amzdate = t.strftime('%Y%m%dT%H%M%SZ')

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Amz-Date': amzdate,
        'Host': f'bedrock-runtime.{REGION}.amazonaws.com'
    }

    authorization_header = create_authorization_header('POST', url, payload, headers)
    headers['Authorization'] = authorization_header

    with httpx.Client() as client:
        response = client.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        response_body = response.json()
        print(response_body.get("content"))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

# 运行函数
invoke_bedrock_model()