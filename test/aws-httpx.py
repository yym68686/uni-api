import httpx
import json
import os
import datetime
from datetime import timezone
import hashlib
import hmac
import urllib.parse

# --- AWS Signature V4 Helper Functions ---

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing

def get_signature(request_body, model_id, aws_access_key, aws_secret_key, aws_region, host, content_type):
    request_body = json.dumps(request_body)
    SERVICE = "bedrock"
    canonical_querystring = '' # No query parameters for invoke
    method = 'POST'
    raw_path = f'/model/{model_id}/invoke'
    canonical_uri = urllib.parse.quote(raw_path, safe='/-_.~')
    # Create a date for headers and the credential string
    t = datetime.datetime.now(timezone.utc)
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d') # Date YYYYMMDD

    # --- Task 1: Create a Canonical Request ---
    payload_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()

    canonical_headers = f'content-type:{content_type}\n' \
                        f'host:{host}\n' \
                        f'x-amz-content-sha256:{payload_hash}\n' \
                        f'x-amz-date:{amz_date}\n'
    # Note: Include other headers if needed, ensure they are lowercase and sorted alphabetically
    # For Bedrock invoke, these are usually sufficient. Add x-amz-security-token if using temporary credentials.

    signed_headers = 'content-type;host;x-amz-content-sha256;x-amz-date' # Semicolon-separated list of header names, lowercase, sorted.

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

# Create request body
payload = {
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": "Hello, world"}],
  "anthropic_version": "bedrock-2023-05-31"
}

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')

CONTENT_TYPE = "application/json"
AWS_REGION = "us-east-1"
HOST = f"bedrock-runtime.{AWS_REGION}.amazonaws.com"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
url = f"https://{HOST}/model/{MODEL_ID}/invoke"

amz_date, payload_hash, authorization_header = get_signature(payload, MODEL_ID, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, HOST, CONTENT_TYPE)
headers = {
    'Content-Type': CONTENT_TYPE,
    'X-Amz-Date': amz_date,
    'X-Amz-Content-Sha256': payload_hash, # Required for POST requests with body
    'Authorization': authorization_header,
    # Add 'X-Amz-Security-Token': SESSION_TOKEN if using temporary credentials
}

# --- Send the request using httpx ---
try:
    with httpx.Client() as client:
        response = client.post(
            url,
            headers=headers,
            json=payload,
            timeout=30.0 # Example timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        response_body = response.json()
        # Adjust based on actual Claude 3.5 response structure if needed
        print(response_body.get("content"))

except httpx.RequestError as exc:
    print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
except httpx.HTTPStatusError as exc:
    print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
    try:
        # Try to print the response body for more details on the error
        print(exc.response.json())
    except json.JSONDecodeError:
        print(exc.response.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
