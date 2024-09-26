# uni-api

<p align="center">
  <a href="https://t.me/uni_api">
    <img src="https://img.shields.io/badge/Join Telegram Group-blue?&logo=telegram">
  </a>
   <a href="https://hub.docker.com/repository/docker/yym68686/uni-api">
    <img src="https://img.shields.io/docker/pulls/yym68686/uni-api?color=blue" alt="docker pull">
  </a>
</p>

[英文](./README.md) | [中文](./README_CN.md)

## Introduction

如果个人使用的话，one/new-api 过于复杂，有很多个人不需要使用的商用功能，如果你不想要复杂的前端界面，有想要支持的模型多一点，可以试试 uni-api。这是一个统一管理大模型API的项目，可以通过一个统一的API接口调用多个后端服务，统一转换为 OpenAI 格式，支持负载均衡。目前支持的后端服务有：OpenAI、Anthropic、Gemini、Vertex、Cohere、Groq、Cloudflare、DeepBricks、OpenRouter 等。

## Features

- 无前端，纯配置文件配置 API 渠道。只要写一个文件就能运行起一个属于自己的 API 站，文档有详细的配置指南，小白友好。
- 统一管理多个后端服务，支持 OpenAI、Deepseek、DeepBricks、OpenRouter 等其他 API 是 OpenAI 格式的提供商。支持 OpenAI Dalle-3 图像生成。
- 同时支持 Anthropic、Gemini、Vertex AI、Cohere、Groq、Cloudflare。Vertex 同时支持 Claude 和 Gemini API。
- 支持 OpenAI、 Anthropic、Gemini、Vertex 原生 tool use 函数调用。
- 支持 OpenAI、Anthropic、Gemini、Vertex 原生识图 API。
- 支持四种负载均衡。
  1. 支持渠道级加权负载均衡，可以根据不同的渠道权重分配请求。默认不开启，需要配置渠道权重。
  2. 支持 Vertex 区域级负载均衡，支持 Vertex 高并发，最高可将 Gemini，Claude 并发提高 （API数量 * 区域数量） 倍。自动开启不需要额外配置。
  3. 除了 Vertex 区域级负载均衡，所有 API 均支持渠道级顺序负载均衡，提高沉浸式翻译体验。自动开启不需要额外配置。
  4. 支持单个渠道多个 API Key 自动开启 API key 级别的轮训负载均衡。
- 支持自动重试，当一个 API 渠道响应失败时，自动重试下一个 API 渠道。
- 支持细粒度的权限控制。支持使用通配符设置 API key 可用渠道的特定模型。
- 支持限流，可以设置每分钟最多请求次数，可以设置为整数，如 2/min，2 次每分钟、5/hour，5 次每小时、10/day，10 次每天，10/month，10 次每月，10/year，10 次每年。默认60/min。
- 支持多个标准 OpenAI 格式的接口：`/v1/chat/completions`，`/v1/images/generations`，`/v1/audio/transcriptions`，`/v1/moderations`，`/v1/models`。
- 支持 OpenAI moderation 道德审查，可以对用户的消息进行道德审查，如果发现不当的消息，会返回错误信息。降低后台 API 被提供商封禁的风险。

## Configuration

使用 api.yaml 配置文件，可以配置多个模型，每个模型可以配置多个后端服务，支持负载均衡。下面是 api.yaml 配置文件的示例：

```yaml
providers:
  - provider: provider_name # 服务提供商名称, 如 openai、anthropic、gemini、openrouter、deepbricks，随便取名字，必填
    base_url: https://api.your.com/v1/chat/completions # 后端服务的API地址，必填
    api: sk-YgS6GTi0b4bEabc4C # 提供商的API Key，必填
    model: # 至少填一个模型
      - gpt-4o # 可以使用的模型名称，必填
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # 重命名模型，claude-3-5-sonnet-20240620 是服务商的模型名称，claude-3-5-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # 支持多个 API Key，多个 key 自动开启轮训负载均衡，至少一个 key，必填
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # 重命名模型，claude-3-5-sonnet-20240620 是服务商的模型名称，claude-3-5-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
    tools: true # 是否支持工具，如生成代码、生成文档等，默认是 true，选填

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url 支持 v1beta/v1, 仅供 Gemini 模型使用，必填
    api: AIzaSyAN2k6IRdgw
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash # 重命名后，原来的模型名字 gemini-1.5-flash-exp-0827 无法使用，如果要使用原来的名字，可以在 model 中添加原来的名字，只要加上下面一行就可以使用原来的名字了
      - gemini-1.5-flash-exp-0827 # 加上这一行，gemini-1.5-flash-exp-0827 和 gemini-1.5-flash 都可以被请求
    tools: true

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx #    描述： 您的Google Cloud项目ID。格式： 字符串，通常由小写字母、数字和连字符组成。获取方式： 在Google Cloud Console的项目选择器中可以找到您的项目ID。
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # 描述： Google Cloud Vertex AI服务账号的私钥。格式： 一个JSON格式的字符串，包含服务账号的私钥信息。获取方式： 在Google Cloud Console中创建服务账号，生成JSON格式的密钥文件，然后将其内容设置为此环境变量的值。
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # 描述： Google Cloud Vertex AI服务账号的电子邮件地址。格式： 通常是形如 "service-account-name@project-id.iam.gserviceaccount.com" 的字符串。获取方式： 在创建服务账号时生成，也可以在Google Cloud Console的"IAM与管理"部分查看服务账号详情获得。
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
      - claude-3-5-sonnet@20240620: claude-3-5-sonnet
      - claude-3-opus@20240229: claude-3-opus
      - claude-3-sonnet@20240229: claude-3-sonnet
      - claude-3-haiku@20240307: claude-3-haiku
    tools: true
    notes: https://xxxxx.com/ # 可以放服务商的网址，备注信息，官方文档，选填

  - provider: cloudflare
    api: f42b3xxxxxxxxxxq4aoGAh # Cloudflare API Key，必填
    cf_account_id: 8ec0xxxxxxxxxxxxe721 # Cloudflare Account ID，必填
    model:
      - '@cf/meta/llama-3.1-8b-instruct': llama-3.1-8b # 重命名模型，@cf/meta/llama-3.1-8b-instruct 是服务商的原始的模型名称，必须使用引号包裹模型名，否则yaml语法错误，llama-3.1-8b 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
      - '@cf/meta/llama-3.1-8b-instruct' # 必须使用引号包裹模型名，否则yaml语法错误

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
      - anthropic/claude-3-5-sonnet
    tools: false
    engine: openrouter # 强制使用某个消息格式，目前支持 gpt，claude，gemini，openrouter 原生格式，选填

api_keys:
  - api: sk-KjjI60Yf0JFWxfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key，用户使用本服务需要 API key，必填
    model: # 该 API Key 可以使用的模型，必填
      - gpt-4o # 可以使用的模型名称，可以使用所有提供商提供的 gpt-4o 模型
      - claude-3-5-sonnet # 可以使用的模型名称，可以使用所有提供商提供的 claude-3-5-sonnet 模型
      - gemini/* # 可以使用的模型名称，仅可以使用名为 gemini 提供商提供的所有模型，其中 gemini 是 provider 名称，* 代表所有模型
    role: admin

  - api: sk-pkhf60Yf0JGyJxgRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # 可以使用的模型名称，仅可以使用名为 anthropic 提供商提供的 claude-3-5-sonnet 模型。其他提供商的 claude-3-5-sonnet 模型不可以使用。这种写法不会匹配到other-provider提供的名为anthropic/claude-3-5-sonnet的模型。
      - <anthropic/claude-3-5-sonnet> # 通过在模型名两侧加上尖括号，这样就不会去名为anthropic的渠道下去寻找claude-3-5-sonnet模型，而是将整个 anthropic/claude-3-5-sonnet 作为模型名称。这种写法可以匹配到other-provider提供的名为 anthropic/claude-3-5-sonnet 的模型。但不会匹配到anthropic下面的claude-3-5-sonnet模型。
      - openai-test/text-moderation-latest # 当开启消息道德审查后，可以使用名为 openai-test 渠道下的 text-moderation-latest 模型进行道德审查。
    preferences:
      USE_ROUND_ROBIN: true # 是否使用轮询负载均衡，true 为使用，false 为不使用，默认为 true。开启轮训后每次请求模型按照 model 配置的顺序依次请求。与 providers 里面原始的渠道顺序无关。因此你可以设置每个 API key 请求顺序不一样。
      AUTO_RETRY: true # 是否自动重试，自动重试下一个提供商，true 为自动重试，false 为不自动重试，默认为 true
      RATE_LIMIT: 2/min # 支持限流，每分钟最多请求次数，可以设置为整数，如 2/min，2 次每分钟、5/hour，5 次每小时、10/day，10 次每天，10/month，10 次每月，10/year，10 次每年。默认60/min，选填
      ENABLE_MODERATION: true # 是否开启消息道德审查，true 为开启，false 为不开启，默认为 false，当开启后，会对用户的消息进行道德审查，如果发现不当的消息，会返回错误信息。

  # 渠道级加权负载均衡配置示例
  - api: sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo
    model:
      - gcp1/*: 5 # 冒号后面就是权重，权重仅支持正整数。
      - gcp2/*: 3 # 数字的大小代表权重，数字越大，请求的概率越大。
      - gcp3/*: 2 # 在该示例中，所有渠道加起来一共有 10 个权重，及 10 个请求里面有 5 个请求会请求 gcp1/* 模型，2 个请求会请求 gcp2/* 模型，3 个请求会请求 gcp3/* 模型。

    preferences:
      USE_ROUND_ROBIN: true # 当 USE_ROUND_ROBIN 必须为 true 并且上面的渠道后面没有权重时，会按照原始的渠道顺序请求，如果有权重，会按照加权后的顺序请求。
      AUTO_RETRY: true
```

## 环境变量

- CONFIG_URL: 配置文件的下载地址，可以是本地文件，也可以是远程文件，选填
- TIMEOUT: 请求超时时间，默认为 100 秒，超时时间可以控制当一个渠道没有响应时，切换下一个渠道需要的时间。选填

## Docker Local Deployment

Start the container

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-e CONFIG_URL=http://file_url/api.yaml \ # 如果已经挂载了本地配置文件，不需要设置 CONFIG_URL
-v ./api.yaml:/home/api.yaml \ # 如果已经设置 CONFIG_URL，不需要挂载配置文件
-v ./uniapi_db:/home/data \ # 如果不想保存统计数据，不需要挂载该文件夹
yym68686/uni-api:latest
```

Or if you want to use Docker Compose, here is a docker-compose.yml example:

```yaml
services:
  uni-api:
    container_name: uni-api
    image: yym68686/uni-api:latest
    environment:
      - CONFIG_URL=http://file_url/api.yaml # 如果已经挂载了本地配置文件，不需要设置 CONFIG_URL
    ports:
      - 8001:8000
    volumes:
      - ./api.yaml:/home/api.yaml # 如果已经设置 CONFIG_URL，不需要挂载配置文件
      - ./uniapi_db:/home/data # 如果不想保存统计数据，不需要挂载该文件夹
```

CONFIG_URL 就是可以自动下载远程的配置文件。比如你在某个平台不方便修改配置文件，可以把配置文件传到某个托管服务，可以提供直链给 uni-api 下载，CONFIG_URL 就是这个直链。如果使用本地挂载的配置文件，不需要设置 CONFIG_URL。CONFIG_URL 是在不方便挂载配置文件的情况下使用。

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
-v ./uniapi_db:/home/data \
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