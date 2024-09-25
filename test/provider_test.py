import os
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def api_key():
    return os.environ.get("API")

@pytest.fixture
def get_model():
    return os.environ.get("MODEL", "claude-3-5-sonnet")

def test_request_model(test_client, api_key, get_model):
    request_data = {
        "model": get_model,
        "messages": [
            {
                "role": "user",
                "content": "say test"
            }
        ],
        "max_tokens": 4096,
        "stream": True,
        "temperature": 0.5,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "n": 1,
        "user": "user",
        "tools": [
            {
                "type": "function",
                "function": {
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
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_url_content",
                    "description": "Get the webpage content of a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to request."
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }

    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }

    response = test_client.post("/v1/chat/completions", json=request_data, headers=headers)
    for line in response.iter_lines():
        print(line.lstrip("data: "))
    assert response.status_code == 200

if __name__ == "__main__":
    pytest.main(["-s", "test/provider_test.py"])