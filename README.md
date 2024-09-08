# uni-api

<p align="center">
<a href="https://t.me/uni_api">
    <img src="https://img.shields.io/badge/Join Telegram Group-blue?&logo=telegram">
  </a>
<a href="https://hub.docker.com/repository/docker/yym68686/uni-api">
    <img src="https://img.shields.io/docker/pulls/yym68686/uni-api?color=blue" alt="docker pull">
  </a>
</p>

[English](./README.md) | [Chinese](./README_CN.md)

## Introduction

If used personally, one/new-api is too complex and has many commercial features that individuals do not need. If you don't want a complicated frontend interface and want to support more models, you can try uni-api. This is a project that manages large model APIs uniformly. It allows you to call multiple backend services through a unified API interface, converting them uniformly to OpenAI format and supporting load balancing. The currently supported backend services include: OpenAI, Anthropic, Gemini, Vertex, DeepBricks, OpenRouter, etc.

## Features

- No front-end, purely configuration file to set up API channels. You can run your own API site just by writing a file. The documentation has detailed configuration guidelines, friendly for beginners.
- Unified management of multiple backend services, supporting OpenAI, Deepseek, DeepBricks, OpenRouter, and other API providers in the OpenAI format. Supports OpenAI Dalle-3 image generation.
- Supports Anthropic, Gemini, Vertex API simultaneously. Vertex supports both Claude and Gemini API.
- Support native tool use function calls for OpenAI, Anthropic, Gemini, Vertex.
- Supports OpenAI, Anthropic, Gemini, Vertex native image recognition API.
- Supports four types of load balancing.
  1. Supports channel-level weighted load balancing, which can allocate requests based on different channel weights. By default, it is not enabled and requires channel weight configuration.
  2. Supports Vertex region-level load balancing, supports Vertex high concurrency, and can increase Gemini and Claude concurrency by up to (number of APIs * number of regions) times. Automatically enabled without additional configuration.
  3. Except for Vertex region-level load balancing, all APIs support channel-level sequential load balancing, enhancing the immersive translation experience. Automatically enabled without additional configuration.
  4. Support automatic API key-level polling load balancing for multiple API Keys in a single channel.
- Supports automatic retry. When an API channel response fails, automatically retry the next API channel.
- Supports fine-grained access control. Supports using wildcards to set specific models available for API key channels.
- Supports rate limiting, can set the maximum number of requests per minute, can be set as an integer, such as 2/min, 2 times per minute, 5/hour, 5 times per hour, 10/day, 10 times per day, 10/month, 10 times per month, 10/year, 10 times per year. Default is 60/min.

## Configuration

Using the api.yaml configuration file, you can configure multiple models, and each model can configure multiple backend services, supporting load balancing. Below is an example of the api.yaml configuration file:

```yaml
providers:
  - provider: provider_name # Service provider name, such as openai, anthropic, gemini, openrouter, deepbricks, arbitrary name, required
    base_url: https://api.your.com/v1/chat/completions # Backend service API address, required
    api: sk-YgS6GTi0b4bEabc4C # Provider's API Key, required
    model: # At least one model
      - gpt-4o # Usable model name, required
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # Rename model, claude-3-5-sonnet-20240620 is the provider's model name, claude-3-5-sonnet is the renamed name, can use a short name instead of the original complex name, optional
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # Supports multiple API Keys, multiple keys automatically enable polling load balancing, at least one key, required
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # Rename model, claude-3-5-sonnet-20240620 is the provider's model name, claude-3-5-sonnet is the renamed name, can use a short name instead of the original complex name, optional
    tools: true # Whether to support tools, such as generating code, generating documents, etc., default is true, optional

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url supports v1beta/v1, only for Gemini models, required
    api: AIzaSyAN2k6IRdgw
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash # After renaming, the original model name gemini-1.5-flash-exp-0827 cannot be used, if you want to use the original name, you can add the original name in the model, just add the following line to use the original name
      - gemini-1.5-flash-exp-0827 # Add this line, both gemini-1.5-flash-exp-0827 and gemini-1.5-flash can be requested
    tools: true

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx # Description: Your Google Cloud project ID. Format: String, usually composed of lowercase letters, numbers, and hyphens. How to obtain: You can find your project ID in the project selector of the Google Cloud Console.
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # Description: The private key of the Google Cloud Vertex AI service account. Format: A JSON formatted string containing the private key information of the service account. How to obtain: Create a service account in the Google Cloud Console, generate a JSON formatted key file, and then set its content as the value of this environment variable.
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # Description: The email address of the Google Cloud Vertex AI service account. Format: Usually a string like "service-account-name@project-id.iam.gserviceaccount.com". How to obtain: Generated when creating the service account, you can also view the service account details in the "IAM & Admin" section of the Google Cloud Console.
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
      - claude-3-5-sonnet@20240620: claude-3-5-sonnet
      - claude-3-opus@20240229: claude-3-opus
      - claude-3-sonnet@20240229: claude-3-sonnet
      - claude-3-haiku@20240307: claude-3-haiku
    tools: true
    notes: https://xxxxx.com/ # Can put the provider's website, notes, official documentation, optional

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
    tools: false
    engine: openrouter # Force the use of a specific message format, currently supports gpt, claude, gemini, openrouter native format, optional

api_keys:
  - api: sk-KjjI60Yf0JFWtfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key, required for users to use this service
    model: # Models that this API Key can use, required
      - gpt-4o # Usable model name, can use all gpt-4o models provided by providers
      - claude-3-5-sonnet # Usable model name, can use all claude-3-5-sonnet models provided by providers
      - gemini/* # Usable model name, can only use all models provided by the provider named gemini, where gemini is the provider name, * represents all models
    role: admin

  - api: sk-pkhf60Yf0JGyJygRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # Usable model name, can only use the claude-3-5-sonnet model provided by the provider named anthropic. Models of other providers' claude-3-5-sonnet cannot be used.
    preferences:
      USE_ROUND_ROBIN: true # Whether to use polling load balancing, true to use, false to not use, default is true. When polling is enabled, each request model is requested in the order configured in the model. It is not related to the original channel order in providers. Therefore, you can set different request sequences for each API key.
      AUTO_RETRY: true # Whether to automatically retry, automatically retry the next provider, true to automatically retry, false to not automatically retry, default is true
      RATE_LIMIT: 2/min # Supports rate limiting, maximum number of requests per minute, can be set to an integer, such as 2/min, 2 times per minute, 5/hour, 5 times per hour, 10/day, 10 times per day, 10/month, 10 times per month, 10/year, 10 times per year. Default is 60/min, optional

  # Channel-level weighted load balancing configuration example
  - api: sk-KjjI60Yf0JFWtxxxxxxxxxxxxxxwmRWpWpQRo
    model:
      - gcp1/*: 5 # The number after the colon is the weight, weights only support positive integers.
      - gcp2/*: 3 # The larger the number, the greater the probability of being requested.
      - gcp3/*: 2 # In this example, there are a total of 10 weights for all channels, and 5 out of 10 requests will request the gcp1/* model, 2 requests will request the gcp2/* model, and 3 requests will request the gcp3/* model.

    preferences:
      USE_ROUND_ROBIN: true # When USE_ROUND_ROBIN must be true and there is no weight after the channels above, it will request in the original channel order, if there is weight, it will request in the weighted order.
      AUTO_RETRY: true
```

## Environment variables

- CONFIG_URL: The download address of the configuration file, it can be a local file or a remote file, optional
- TIMEOUT: Request timeout, default is 20 seconds. The timeout can control the time needed to switch to the next channel when a channel does not respond. Optional

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

CONFIG_URL is a direct link that can automatically download remote configuration files. For example, if you find it inconvenient to modify configuration files on a certain platform, you can upload the configuration file to a hosting service and provide a direct link for uni-api to download. CONFIG_URL is this direct link.

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