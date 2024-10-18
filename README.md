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

For personal use, one/new-api is too complex with many commercial features that individuals don't need. If you don't want a complicated frontend interface and prefer support for more models, you can try uni-api. This is a project that unifies the management of large language model APIs, allowing you to call multiple backend services through a single unified API interface, converting them all to OpenAI format, and supporting load balancing. Currently supported backend services include: OpenAI, Anthropic, Gemini, Vertex, Cohere, Groq, Cloudflare, DeepBricks, OpenRouter, and more.

## ✨ Features

- No front-end, pure configuration file to configure API channels. You can run your own API station just by writing a file, and the documentation has a detailed configuration guide, beginner-friendly.
- Unified management of multiple backend services, supporting providers such as OpenAI, Deepseek, DeepBricks, OpenRouter, and other APIs in OpenAI format. Supports OpenAI Dalle-3 image generation.
- Simultaneously supports Anthropic, Gemini, Vertex AI, Cohere, Groq, Cloudflare. Vertex simultaneously supports Claude and Gemini API.
- Support OpenAI, Anthropic, Gemini, Vertex native tool use function calls.
- Support OpenAI, Anthropic, Gemini, Vertex native image recognition API.
- Support four types of load balancing.
  1. Supports channel-level weighted load balancing, allowing requests to be distributed according to different channel weights. It is not enabled by default and requires configuring channel weights.
  2. Support Vertex regional load balancing and high concurrency, which can increase Gemini and Claude concurrency by up to (number of APIs * number of regions) times. Automatically enabled without additional configuration.
  3. Except for Vertex region-level load balancing, all APIs support channel-level sequential load balancing, enhancing the immersive translation experience. Automatically enabled without additional configuration.
  4. Support automatic API key-level round-robin load balancing for multiple API Keys in a single channel.
- Support automatic retry, when an API channel response fails, automatically retry the next API channel.
- Support fine-grained permission control. Support using wildcards to set specific models available for API key channels.
- Support rate limiting, you can set the maximum number of requests per minute as an integer, such as 2/min, 2 times per minute, 5/hour, 5 times per hour, 10/day, 10 times per day, 10/month, 10 times per month, 10/year, 10 times per year. Default is 60/min.
- Supports multiple standard OpenAI format interfaces: `/v1/chat/completions`, `/v1/images/generations`, `/v1/audio/transcriptions`, `/v1/moderations`, `/v1/models`.
- Support OpenAI moderation moral review, which can conduct moral reviews of user messages. If inappropriate messages are found, an error message will be returned. This reduces the risk of the backend API being banned by providers.

## Usage method

To start uni-api, a configuration file must be used. There are two ways to start with a configuration file:

1. The first method is to use the `CONFIG_URL` environment variable to fill in the configuration file URL, which will be automatically downloaded when uni-api starts.
2. The second method is to mount a configuration file named `api.yaml` into the container.

### Method 1: Mount the `api.yaml` configuration file to start uni-api

You must fill in the configuration file in advance to start `uni-api`, and you must use a configuration file named `api.yaml` to start `uni-api`, you can configure multiple models, each model can configure multiple backend services, and support load balancing. Below is an example of the minimum `api.yaml` configuration file that can be run:

```yaml
providers:
  - provider: provider_name # Service provider name, such as openai, anthropic, gemini, openrouter, deepbricks, can be any name, required
    base_url: https://api.your.com/v1/chat/completions # Backend service API address, required
    api: sk-YgS6GTi0b4bEabc4C # Provider's API Key, required, automatically uses base_url and api to get all available models through the /v1/models endpoint.
  # Multiple providers can be configured here, each provider can have multiple API Keys, and each API Key can have multiple models configured.
api_keys:
  - api: sk-Pkj60Yf8JFWxfgRmXQFWyGtWUddGZnmi3KlvowmRWpWpQxx # API Key, required for user requests to uni-api
    model: # Models that can be used by this API Key, required. Channel-level round-robin load balancing is enabled by default, and each request to the model follows the order configured in model. It is independent of the original channel order in providers. Therefore, you can set a different request order for each API key.
      - all # Can use all models from all channels set under providers, no need to add available channels one by one. If you don't want to set available channels for each api in api_keys, uni-api supports setting the api key to use all models from all channels under providers.
```

Detailed advanced configuration of `api.yaml`:

```yaml
providers:
  - provider: provider_name # Service provider name, such as openai, anthropic, gemini, openrouter, deepbricks, any name, required
    base_url: https://api.your.com/v1/chat/completions # Backend service API address, required
    api: sk-YgS6GTi0b4bEabc4C # Provider's API Key, required
    model: # Optional, if model is not configured, all available models will be automatically retrieved through base_url and api via the /v1/models endpoint.
      - gpt-4o # Usable model name, required
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # Rename model, claude-3-5-sonnet-20240620 is the provider's model name, claude-3-5-sonnet is the renamed name, you can use a simpler name instead of the original complex name, optional
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # Supports multiple API Keys, multiple keys automatically enable round-robin load balancing, at least one key, required
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # Rename model, claude-3-5-sonnet-20240620 is the provider's model name, claude-3-5-sonnet is the renamed name, you can use a simpler name instead of the original complex name, optional
    tools: true # Whether to support tools, such as code generation, document generation, etc., default is true, optional

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url supports v1beta/v1, only for Gemini models, required
    api: AIzaSyAN2k6IRdgw
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash # After renaming, the original model name gemini-1.5-flash-exp-0827 cannot be used, if you want to use the original name, you can add the original name in the model, just add the following line to use the original name
      - gemini-1.5-flash-exp-0827 # Add this line, both gemini-1.5-flash-exp-0827 and gemini-1.5-flash can be requested
    tools: true

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx #    Description: Your Google Cloud project ID. Format: String, usually composed of lowercase letters, numbers, and hyphens. How to obtain: You can find your project ID in the project selector of the Google Cloud Console.
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # Description: The private key of the Google Cloud Vertex AI service account. Format: A JSON-formatted string containing the private key information of the service account. How to obtain: Create a service account in the Google Cloud Console, generate a JSON-formatted key file, and then set its content as the value of this environment variable.
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # Description: The email address of the Google Cloud Vertex AI service account. Format: Usually a string like "service-account-name@project-id.iam.gserviceaccount.com". How to obtain: Generated when creating a service account, or can be obtained by viewing service account details in the "IAM & Admin" section of the Google Cloud Console.
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
      - claude-3-5-sonnet@20240620: claude-3-5-sonnet
      - claude-3-opus@20240229: claude-3-opus
      - claude-3-sonnet@20240229: claude-3-sonnet
      - claude-3-haiku@20240307: claude-3-haiku
    tools: true
    notes: https://xxxxx.com/ # Can include the provider's website, notes, official documentation, optional

  - provider: cloudflare
    api: f42b3xxxxxxxxxxq4aoGAh # Cloudflare API Key, required
    cf_account_id: 8ec0xxxxxxxxxxxxe721 # Cloudflare Account ID, required
    model:
      - '@cf/meta/llama-3.1-8b-instruct': llama-3.1-8b # Rename model, @cf/meta/llama-3.1-8b-instruct is the provider's original model name, must be enclosed in quotes to avoid YAML syntax error, llama-3.1-8b is the renamed name, you can use a simpler name instead of the original complex name, optional
      - '@cf/meta/llama-3.1-8b-instruct' # Must be enclosed in quotes to avoid YAML syntax error

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
      - anthropic/claude-3-5-sonnet
    tools: false
    engine: openrouter # Force use of a specific message format, currently supports gpt, claude, gemini, openrouter native format, optional

api_keys:
  - api: sk-KjjI60Yf0JFWxfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key, required for users to use this service
    model: # The models that this API Key can use, required. Channel-level round-robin load balancing is enabled by default, and each request model is requested in the order configured in the model. It is unrelated to the original channel order in providers. Therefore, you can set different request orders for each API key.
      - gpt-4o # Usable model name, can use all gpt-4o models provided by providers
      - claude-3-5-sonnet # Usable model name, can use all claude-3-5-sonnet models provided by providers
      - gemini/* # Usable model name, can only use all models provided by the provider named gemini, where gemini is the provider name, * represents all models
    role: admin

  - api: sk-pkhf60Yf0JGyJxgRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # Usable model name, can only use the claude-3-5-sonnet model provided by the provider named anthropic. Models named claude-3-5-sonnet from other providers cannot be used. This notation will not match the model named anthropic/claude-3-5-sonnet provided by other-provider.
      - <anthropic/claude-3-5-sonnet> # By adding angle brackets around the model name, it will not look for the claude-3-5-sonnet model under the channel named anthropic, but instead treat the entire anthropic/claude-3-5-sonnet as the model name. This notation can match the model named anthropic/claude-3-5-sonnet provided by other-provider. But it will not match the claude-3-5-sonnet model under anthropic.
      - openai-test/text-moderation-latest # When message moderation is enabled, the text-moderation-latest model under the channel named openai-test can be used for message moderation.
    preferences:
      SCHEDULING_ALGORITHM: fixed_priority # When SCHEDULING_ALGORITHM is fixed_priority, fixed priority scheduling is used, always executing the channel of the first model with a request. Modify the default channel round-robin load balancing. SCHEDULING_ALGORITHM options are: fixed_priority, weighted_round_robin, lottery, random.
      # When SCHEDULING_ALGORITHM is random, random round-robin load balancing is used, randomly requesting the channel of the model with a request.
      AUTO_RETRY: true # Whether to automatically retry, automatically retry the next provider, true for automatic retry, false for no automatic retry, default is true
      RATE_LIMIT: 2/min # Supports rate limiting, the maximum number of requests per minute, can be set as an integer, such as 2/min, 2 times per minute, 5/hour, 5 times per hour, 10/day, 10 times per day, 10/month, 10 times per month, 10/year, 10 times per year. Default is 60/min, optional
      ENABLE_MODERATION: true # Whether to enable message moderation, true to enable, false to disable, default is false, when enabled, user messages will be moderated, and if inappropriate messages are found, an error message will be returned.

  # Channel-level weighted load balancing configuration example
  - api: sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo
    model:
      - gcp1/*: 5 # The number after the colon is the weight, weight only supports positive integers.
      - gcp2/*: 3 # The larger the number, the greater the probability of the request.
      - gcp3/*: 2 # In this example, there are a total of 10 weights across all channels, and out of 10 requests, 5 requests will request the gcp1/* model, 2 requests will request the gcp2/* model, and 3 requests will request the gcp3/* model.

    preferences:
      SCHEDULING_ALGORITHM: weighted_round_robin # Only when SCHEDULING_ALGORITHM is weighted_round_robin and if the above channels have weights, requests will be made according to the weighted order. Use weighted round-robin load balancing, request the channel of the model with a request according to the weight order. When SCHEDULING_ALGORITHM is lottery, use lottery round-robin load balancing, request the channel of the model with a request according to the weight randomly.
      AUTO_RETRY: true
```

Mount the configuration file and start the uni-api docker container:

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-v ./api.yaml:/home/api.yaml \
yym68686/uni-api:latest
```

### Method two: Start uni-api using the `CONFIG_URL` environment variable

After writing the configuration file according to method one, upload it to the cloud disk, get the file's direct link, and then use the `CONFIG_URL` environment variable to start the uni-api docker container:

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-e CONFIG_URL=http://file_url/api.yaml \
yym68686/uni-api:latest
```

## Environment variable

- CONFIG_URL: The download address of the configuration file, which can be a local file or a remote file, optional
- TIMEOUT: Request timeout, default is 100 seconds. The timeout can control the time needed to switch to the next channel when one channel does not respond. Optional
- DISABLE_DATABASE: Whether to disable the database, default is false, optional

## Get statistical data

Use `/stats` to get the usage statistics of each channel for the past 24 hours. Also include your uni-api admin API key.

Data includes:

1. The success rate of each model under each channel, sorted from high to low.
2. The overall success rate of each channel, sorted from high to low.
3. The total number of requests for each model across all channels.
4. The number of requests for each endpoint.
5. The number of requests per IP.

The `hours` parameter in `/stats?hours=48` allows you to control how many hours of recent data statistics to return. If the `hours` parameter is not provided, it defaults to statistics for the last 24 hours.

There are other statistical data that you can query yourself by writing SQL in the database. Other data includes: first token time, total processing time for each request, whether each request was successful, whether each request passed content moderation, the text content of each request, the API key for each request, the number of input tokens, and the number of output tokens for each request.

## Vercel Deployment

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyym68686%2Funi-api%2Ftree%2Fmain&env=CONFIG_URL,DISABLE_DATABASE&project-name=uni-api-vercel&repository-name=uni-api-vercel)

## Docker local deployment

Start the container

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-e CONFIG_URL=http://file_url/api.yaml \ # If the local configuration file has already been mounted, there is no need to set CONFIG_URL
-v ./api.yaml:/home/api.yaml \ # If CONFIG_URL is already set, there is no need to mount the configuration file
-v ./uniapi_db:/home/data \ # If you do not want to save statistical data, there is no need to mount this folder
yym68686/uni-api:latest
```

Or if you want to use Docker Compose, here is a docker-compose.yml example:

```yaml
services:
  uni-api:
    container_name: uni-api
    image: yym68686/uni-api:latest
    environment:
      - CONFIG_URL=http://file_url/api.yaml # If a local configuration file is already mounted, there is no need to set CONFIG_URL
    ports:
      - 8001:8000
    volumes:
      - ./api.yaml:/home/api.yaml # If CONFIG_URL is already set, there is no need to mount the configuration file
      - ./uniapi_db:/home/data # If you do not want to save statistical data, there is no need to mount this folder
```

CONFIG_URL is the URL of the remote configuration file that can be automatically downloaded. For example, if you are not comfortable modifying the configuration file on a certain platform, you can upload the configuration file to a hosting service and provide a direct link to uni-api to download, which is the CONFIG_URL. If you are using a local mounted configuration file, there is no need to set CONFIG_URL. CONFIG_URL is used when it is not convenient to mount the configuration file.

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

## ⭐ Star History

<a href="https://github.com/yym68686/uni-api/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=yym68686/uni-api&type=Date">
</a>