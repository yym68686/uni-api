# uni-api

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-KjjI60Yf0JFcsvgRmXqFwgGmWUd9GZnmi3KlvowmRWpWpQRo" \
-d '{"model": "gpt-4o","messages": [{"role": "user", "content": "Hello"}],"stream": true}'
```