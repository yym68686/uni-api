# uni-api

<p align="center">
  <a href="https://t.me/uni_api">
    <img src="https://img.shields.io/badge/Join Telegram Group-blue?&logo=telegram">
  </a>
   <a href="https://hub.docker.com/repository/docker/yym68686/uni-api">
    <img src="https://img.shields.io/docker/pulls/yym68686/uni-api?color=blue" alt="docker pull">
  </a>
</p>


## Introduction

这是一个统一管理大模型API的项目，可以通过一个统一的API接口调用多个后端服务，统一转换为 OpenAI 格式，支持负载均衡。目前支持的后端服务有：OpenAI、Anthropic、DeepBricks、OpenRouter、Gemini、Vertex 等。

## Features

- 统一管理多个后端服务
- 支持负载均衡
- 支持 OpenAI, Anthropic, Gemini, Vertex 函数调用
- 支持多个模型
- 支持多个 API Key

## Configuration

使用api.yaml配置文件，可以配置多个模型，每个模型可以配置多个后端服务，支持负载均衡。下面是 api.yaml 配置文件的示例：

```yaml
providers:
  - provider: provider_name # 服务提供商名称, 如 openai、anthropic、gemini、openrouter、deepbricks，随便取名字，必填
    base_url: https://api.your.com/v1/chat/completions # 后端服务的API地址，必填
    api: sk-YgS6GTi0b4bEabc4C # 提供商的API Key，必填
    model:
      - gpt-4o # 可以使用的模型名称，必填
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # 重命名模型，claude-3-5-sonnet-20240620 是服务商的模型名称，claude-3-5-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
    tools: true # 是否支持工具，如生成代码、生成文档等，默认是 true，选填
    url: https://openai.com/ # 服务商的网址，选填

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: sk-ant-api03-bNnAOJyA-xQw_twAA
    model:
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # 重命名模型，claude-3-5-sonnet-20240620 是服务商的模型名称，claude-3-5-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
    tools: true

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url 支持 v1beta/v1, 仅供 Gemini 模型使用，必填
    api: AIzaSyAN2k6IRdgw
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash
    tools: false

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx #    描述： 您的Google Cloud项目ID。格式： 字符串，通常由小写字母、数字和连字符组成。获取方式： 在Google Cloud Console的项目选择器中可以找到您的项目ID。
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # 描述： Google Cloud Vertex AI服务账号的私钥。格式： 一个JSON格式的字符串，包含服务账号的私钥信息。获取方式： 在Google Cloud Console中创建服务账号，生成JSON格式的密钥文件，然后将其内容设置为此环境变量的值。
    client_email: xxxxxxxxxx@developer.gserviceaccount.com # 描述： Google Cloud Vertex AI服务账号的电子邮件地址。格式： 通常是形如 "service-account-name@project-id.iam.gserviceaccount.com" 的字符串。获取方式： 在创建服务账号时生成，也可以在Google Cloud Console的"IAM与管理"部分查看服务账号详情获得。
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
    tools: true

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
    tools: false
    engine: openrouter # 强制使用某个消息格式，目前支持 gpt，claude，gemini，openrouter 原生格式，选填

api_keys:
  - api: sk-KjjI60Yf0JFWtfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key，用户使用本服务需要 API key，必填
    model: # 该 API Key 可以使用的模型，必填
      - gpt-4o # 可以使用的模型名称，可以使用所有提供商提供的 gpt-4o 模型
      - claude-3-5-sonnet # 可以使用的模型名称，可以使用所有提供商提供的 claude-3-5-sonnet 模型
      - gemini/* # 可以使用的模型名称，仅可以使用名为 gemini 提供商提供的所有模型，其中 gemini 是 provider 名称，* 代表所有模型
    role: admin

  - api: sk-pkhf60Yf0JGyJygRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # 可以使用的模型名称，仅可以使用名为 anthropic 提供商提供的 claude-3-5-sonnet 模型。其他提供商的 claude-3-5-sonnet 模型不可以使用。
    preferences:
      USE_ROUND_ROBIN: true # 是否使用轮询负载均衡，true 为使用，false 为不使用，默认为 true
      AUTO_RETRY: true # 是否自动重试，自动重试下一个提供商，true 为自动重试，false 为不自动重试，默认为 true
```

## Docker Local Deployment

Start the container

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-v ./api.yaml:/home/api.yaml \
yym68686/uni-api:latest
```

Or if you want to use Docker Compose, here is a docker-compose.yml example:

```yaml
services:
  uni-api:
    container_name: uni-api
    image: yym68686/uni-api:latest
    environment:
      - CONFIG_URL=http://file_url/api.yaml
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
-e CONFIG_URL=http://file_url/api.yaml \
-v ./api.yaml:/home/api.yaml \
yym68686/uni-api:latest
docker logs -f uni-api
```

RESTful curl test

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${API}" \
-d '{"model": "gpt-4o","messages": [{"role": "user", "content": "Hello"}],"stream": true}'
```

## Star History

<a href="https://github.com/yym68686/uni-api/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=yym68686/uni-api&type=Date">
</a>