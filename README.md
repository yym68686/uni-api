# uni-api

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-KjjI60Yf0JFcsvgRmXqFwgGmWUd9GZnmi3KlvowmRWpWpQRo" \
-d '{"model": "gpt-4o","messages": [{"role": "user", "content": "Hello"}],"stream": true}'
```

## Docker Local Deployment

Start the container

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-v ./api.yaml:/home/api.yaml \
-e USE_ROUND_ROBIN=True \
yym68686/uni-api:latest
```

Or if you want to use Docker Compose, here is a docker-compose.yml example:

```yaml
version: "3.5"
services:
  uni-api:
    container_name: uni-api
    image: yym68686/uni-api:latest
    environment:
      - USE_ROUND_ROBIN=True
    ports:
      - 8001:8000
    volumes:
      - ./api.yaml:/home/api.yaml
```

Run Docker Compose container in the background

```bash
docker-compose pull
docker-compose up -d
```

Docker build

```bash
docker build --no-cache -t uni-api:latest -f Dockerfile --platform linux/amd64 .
docker tag uni-api:latest yym68686/uni-api:latest
docker push yym68686/uni-api:latest
```

One-Click Restart Docker Image

```bash
set -eu
docker pull yym68686/uni-api:latest
docker rm -f uni-api
docker run --user root -p 8001:8000 -dit --name uni-api \
-v ./api.yaml:/home/api.yaml \
-e USE_ROUND_ROBIN=True \
yym68686/uni-api:latest
docker logs -f uni-api
```