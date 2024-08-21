import requests
import json
from requests_auth_aws_sigv4 import AWSSigV4

REGION = 'us-east-1'
aws_auth = AWSSigV4('bedrock', region=REGION) # If not provided, check for AWS Credentials from Environment Variables

body = json.dumps({
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": "Hello, world"}],
  "anthropic_version": "bedrock-2023-05-31"
})
r = requests.request(
    'POST',
    f'https://bedrock-runtime.{REGION}.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke',
    data=body,
    auth=aws_auth)
print(r.text)