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

For personal use, one/new-api is too complex with many commercial features that individuals don't need. If you don't want a complicated frontend interface and prefer support for more models, you can try uni-api. This is a project that unifies the management of large language model APIs, allowing you to call multiple backend services through a single unified API interface, converting them all to OpenAI format, and supporting load balancing. Currently supported backend services include: OpenAI, Anthropic, Gemini, Vertex, Azure, xai, Cohere, Groq, Cloudflare, OpenRouter, and more.

## ✨ Features

- No front-end, pure configuration file to configure API channels. You can run your own API station just by writing a file, and the documentation has a detailed configuration guide, beginner-friendly.
- Unified management of multiple backend services, supporting providers such as OpenAI, Deepseek, OpenRouter, and other APIs in OpenAI format. Supports OpenAI Dalle-3 image generation.
- Simultaneously supports Anthropic, Gemini, Vertex AI, Azure, xai, Cohere, Groq, Cloudflare. Vertex simultaneously supports Claude and Gemini API.
- Support OpenAI, Anthropic, Gemini, Vertex, Azure, xai native tool use function calls.
- Support OpenAI, Anthropic, Gemini, Vertex, Azure, xai native image recognition API.
- Support four types of load balancing.
  1. Supports channel-level weighted load balancing, allowing requests to be distributed according to different channel weights. It is not enabled by default and requires configuring channel weights.
  2. Support Vertex regional load balancing and high concurrency, which can increase Gemini and Claude concurrency by up to (number of APIs * number of regions) times. Automatically enabled without additional configuration.
  3. Except for Vertex region-level load balancing, all APIs support channel-level sequential load balancing, enhancing the immersive translation experience. It is not enabled by default and requires configuring `SCHEDULING_ALGORITHM` as `round_robin`.
  4. Support automatic API key-level round-robin load balancing for multiple API Keys in a single channel.
- Support automatic retry, when an API channel response fails, automatically retry the next API channel.
- Support channel cooling: When an API channel response fails, the channel will automatically be excluded and cooled for a period of time, and requests to the channel will be stopped. After the cooling period ends, the model will automatically be restored until it fails again, at which point it will be cooled again.
- Support fine-grained model timeout settings, allowing different timeout durations for each model.
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
  - provider: provider_name # Service provider name, such as openai, anthropic, gemini, openrouter, can be any name, required
    base_url: https://api.your.com/v1/chat/completions # Backend service API address, required
    api: sk-YgS6GTi0b4bEabc4C # Provider's API Key, required, automatically uses base_url and api to get all available models through the /v1/models endpoint.
  # Multiple providers can be configured here, each provider can configure multiple API Keys, and each provider can configure multiple models.
api_keys:
  - api: sk-Pkj60Yf8JFWxfgRmXQFWyGtWUddGZnmi3KlvowmRWpWpQxx # API Key, user request uni-api requires API key, required
  # This API Key can use all models, that is, it can use all models in all channels set under providers, without needing to add available channels one by one.
```

Detailed advanced configuration of `api.yaml`:

```yaml
providers:
  - provider: provider_name # Service provider name, such as openai, anthropic, gemini, openrouter, can be any name, required
    base_url: https://api.your.com/v1/chat/completions # Backend service API address, required
    api: sk-YgS6GTi0b4bEabc4C # Provider's API Key, required
    model: # Optional, if model is not configured, all available models will be automatically obtained through base_url and api via the /v1/models endpoint.
      - gpt-4o # Usable model name, required
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # Rename model, claude-3-5-sonnet-20240620 is the provider's model name, claude-3-5-sonnet is the renamed name, you can use a simple name to replace the original complex name, optional
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # Supports multiple API Keys, multiple keys automatically enable polling load balancing, at least one key, required
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-3-7-sonnet-20240620: claude-3-7-sonnet # Rename model, claude-3-7-sonnet-20240620 is the provider's model name, claude-3-7-sonnet is the renamed name, you can use a simple name to replace the original complex name, optional
      - claude-3-7-sonnet-20250219: claude-3-7-sonnet-think # Rename model, claude-3-7-sonnet-20250219 is the provider's model name, claude-3-7-sonnet-think is the renamed name, if "think" is in the renamed name, it will be automatically converted to claude think model, default think token limit is 4096. Optional
    tools: true # Whether to support tools, such as generating code, generating documents, etc., default is true, optional

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url supports v1beta/v1, only for Gemini model use, required
    api: # Supports multiple API Keys, multiple keys automatically enable polling load balancing, at least one key, required
      - AIzaSyAN2k6IRdgw123
      - AIzaSyAN2k6IRdgw456
      - AIzaSyAN2k6IRdgw789
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash # After renaming, the original model name gemini-1.5-flash-exp-0827 cannot be used, if you want to use the original name, you can add the original name in the model, just add the line below to use the original name
      - gemini-1.5-flash-exp-0827 # Add this line, both gemini-1.5-flash-exp-0827 and gemini-1.5-flash can be requested
    tools: true
    preferences:
      api_key_rate_limit: 15/min # Each API Key can request up to 15 times per minute, optional. The default is 999999/min. Supports multiple frequency constraints: 15/min,10/day
      # api_key_rate_limit: # You can set different frequency limits for each model
      #   gemini-1.5-flash: 15/min,1500/day
      #   gemini-1.5-pro: 2/min,50/day
      #   default: 4/min # If the model does not set the frequency limit, use the frequency limit of default
      api_key_cooldown_period: 60 # Each API Key will be cooled down for 60 seconds after encountering a 429 error. Optional, the default is 0 seconds. When set to 0, the cooling mechanism is not enabled. When there are multiple API keys, the cooling mechanism will take effect.
      api_key_schedule_algorithm: round_robin # Set the request order of multiple API Keys, optional. The default is round_robin, and the optional values are: round_robin, random, fixed_priority. It will take effect when there are multiple API keys. round_robin is polling load balancing, and random is random load balancing. fixed_priority is fixed priority scheduling, always use the first available API key.
      model_timeout: # Model timeout, in seconds, default 100 seconds, optional
        gemini-1.5-pro: 10 # Model gemini-1.5-pro timeout is 10 seconds
        gemini-1.5-flash: 10 # Model gemini-1.5-flash timeout is 10 seconds
        default: 10 # Model does not have a timeout set, use the default timeout of 10 seconds, when requesting a model not in model_timeout, the timeout is also 10 seconds, if default is not set, uni-api will use the default timeout set by the environment variable TIMEOUT, the default timeout is 100 seconds
      proxy: socks5://[username]:[password]@[ip]:[port] # Proxy address, optional. Supports socks5 and http proxies, default is not used.
      headers:  # Add custom http request headers, optional
        Custom-Header-1: Value-1
        Custom-Header-2: Value-2

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx # Description: Your Google Cloud project ID. Format: String, usually composed of lowercase letters, numbers, and hyphens. How to obtain: You can find your project ID in the project selector of the Google Cloud Console.
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # Description: Private key for Google Cloud Vertex AI service account. Format: A JSON formatted string containing the private key information of the service account. How to obtain: Create a service account in Google Cloud Console, generate a JSON formatted key file, and then set its content as the value of this environment variable.
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # Description: Email address of the Google Cloud Vertex AI service account. Format: Usually a string like "service-account-name@project-id.iam.gserviceaccount.com". How to obtain: Generated when creating a service account, or you can view the service account details in the "IAM and Admin" section of the Google Cloud Console.
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
      - gemini-1.5-pro: gemini-1.5-pro-search # Only supports using the gemini-1.5-pro-search model to request uni-api when using the Vertex Gemini API, to automatically use the Google official search tool.
      - claude-3-5-sonnet@20240620: claude-3-5-sonnet
      - claude-3-opus@20240229: claude-3-opus
      - claude-3-sonnet@20240229: claude-3-sonnet
      - claude-3-haiku@20240307: claude-3-haiku
    tools: true
    notes: https://xxxxx.com/ # You can put the provider's website, notes, official documentation, optional

  - provider: cloudflare
    api: f42b3xxxxxxxxxxq4aoGAh # Cloudflare API Key, required
    cf_account_id: 8ec0xxxxxxxxxxxxe721 # Cloudflare Account ID, required
    model:
      - '@cf/meta/llama-3.1-8b-instruct': llama-3.1-8b # Rename model, @cf/meta/llama-3.1-8b-instruct is the provider's original model name, must be enclosed in quotes, otherwise yaml syntax error, llama-3.1-8b is the renamed name, you can use a simple name to replace the original complex name, optional
      - '@cf/meta/llama-3.1-8b-instruct' # Must be enclosed in quotes, otherwise yaml syntax error

  - provider: azure
    base_url: https://your-endpoint.openai.azure.com
    api: your-api-key
    model:
      - gpt-4o

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
      - anthropic/claude-3-5-sonnet
    tools: false
    engine: openrouter # Force the use of a specific message format, currently supports gpt, claude, gemini, openrouter native format, optional

api_keys:
  - api: sk-KjjI60Yf0JFWxfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key, required for users to use this service
    model: # Models that can be used by this API Key, required. Default channel-level polling load balancing is enabled, and each request model is requested in sequence according to the model configuration. It is not related to the original channel order in providers. Therefore, you can set different request sequences for each API key.
      - gpt-4o # Usable model name, can use all gpt-4o models provided by providers
      - claude-3-5-sonnet # Usable model name, can use all claude-3-5-sonnet models provided by providers
      - gemini/* # Usable model name, can only use all models provided by providers named gemini, where gemini is the provider name, * represents all models
    role: admin # Set the alias of the API key, optional. The request log will display the alias of the API key. If role is admin, only this API key can request the v1/stats,/v1/generate-api-key endpoints. If all API keys do not have role set to admin, the first API key is set as admin and has permission to request the v1/stats,/v1/generate-api-key endpoints.

  - api: sk-pkhf60Yf0JGyJxgRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # Usable model name, can only use the claude-3-5-sonnet model provided by the provider named anthropic. Models with the same name from other providers cannot be used. This syntax will not match the model named anthropic/claude-3-5-sonnet provided by other-provider.
      - <anthropic/claude-3-5-sonnet> # By adding angle brackets on both sides of the model name, it will not search for the claude-3-5-sonnet model under the channel named anthropic, but will take the entire anthropic/claude-3-5-sonnet as the model name. This syntax can match the model named anthropic/claude-3-5-sonnet provided by other-provider. But it will not match the claude-3-5-sonnet model under anthropic.
      - openai-test/text-moderation-latest # When message moderation is enabled, the text-moderation-latest model under the channel named openai-test can be used for moderation.
      - sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo/* # Support using other API keys as channels
    preferences:
      SCHEDULING_ALGORITHM: fixed_priority # When SCHEDULING_ALGORITHM is fixed_priority, use fixed priority scheduling, always execute the channel of the first model with a request. Default is enabled, SCHEDULING_ALGORITHM default value is fixed_priority. SCHEDULING_ALGORITHM optional values are: fixed_priority, round_robin, weighted_round_robin, lottery, random.
      # When SCHEDULING_ALGORITHM is random, use random polling load balancing, randomly request the channel of the model with a request.
      # When SCHEDULING_ALGORITHM is round_robin, use polling load balancing, request the channel of the model used by the user in order.
      AUTO_RETRY: true # Whether to automatically retry, automatically retry the next provider, true for automatic retry, false for no automatic retry, default is true. Also supports setting a number, indicating the number of retries.
      rate_limit: 15/min # Supports rate limiting, each API Key can request up to 15 times per minute, optional. The default is 999999/min. Supports multiple frequency constraints: 15/min,10/day
      # rate_limit: # You can set different frequency limits for each model
      #   gemini-1.5-flash: 15/min,1500/day
      #   gemini-1.5-pro: 2/min,50/day
      #   default: 4/min # If the model does not set the frequency limit, use the frequency limit of default
      ENABLE_MODERATION: true # Whether to enable message moderation, true for enable, false for disable, default is false, when enabled, it will moderate the user's message, if inappropriate messages are found, an error message will be returned.

  # Channel-level weighted load balancing configuration example
  - api: sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo
    model:
      - gcp1/*: 5 # The number after the colon is the weight, weight only supports positive integers.
      - gcp2/*: 3 # The size of the number represents the weight, the larger the number, the greater the probability of the request.
      - gcp3/*: 2 # In this example, there are a total of 10 weights for all channels, and 10 requests will have 5 requests for the gcp1/* model, 2 requests for the gcp2/* model, and 3 requests for the gcp3/* model.

    preferences:
      SCHEDULING_ALGORITHM: weighted_round_robin # Only when SCHEDULING_ALGORITHM is weighted_round_robin and the above channel has weights, it will request according to the weighted order. Use weighted polling load balancing, request the channel of the model with a request according to the weight order. When SCHEDULING_ALGORITHM is lottery, use lottery polling load balancing, request the channel of the model with a request according to the weight randomly. Channels without weights automatically fall back to round_robin polling load balancing.
      AUTO_RETRY: true

preferences: # Global configuration
  model_timeout: # Model timeout, in seconds, default 100 seconds, optional
    gpt-4o: 10 # Model gpt-4o timeout is 10 seconds, gpt-4o is the model name, when requesting models like gpt-4o-2024-08-06, the timeout is also 10 seconds
    claude-3-5-sonnet: 10 # Model claude-3-5-sonnet timeout is 10 seconds, when requesting models like claude-3-5-sonnet-20240620, the timeout is also 10 seconds
    default: 10 # Model does not have a timeout set, use the default timeout of 10 seconds, when requesting a model not in model_timeout, the default timeout is 10 seconds, if default is not set, uni-api will use the default timeout set by the environment variable TIMEOUT, the default timeout is 100 seconds
    o1-mini: 30 # Model o1-mini timeout is 30 seconds, when requesting models starting with o1-mini, the timeout is 30 seconds
    o1-preview: 100 # Model o1-preview timeout is 100 seconds, when requesting models starting with o1-preview, the timeout is 100 seconds
  cooldown_period: 300 # Channel cooldown time, in seconds, default 300 seconds, optional. When a model request fails, the channel will be automatically excluded and cooled down for a period of time, and will not request the channel again. After the cooldown time ends, the model will be automatically restored until the request fails again, and it will be cooled down again. When cooldown_period is set to 0, the cooling mechanism is not enabled.
  error_triggers: # Error triggers, when the message returned by the model contains any of the strings in the error_triggers, the channel will return an error. Optional
    - The bot's usage is covered by the developer
    - process this request due to overload or policy
  proxy: socks5://[username]:[password]@[ip]:[port] # Proxy address, optional.
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

## Vercel remote deployment

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyym68686%2Funi-api%2Ftree%2Fmain&env=CONFIG_URL,DISABLE_DATABASE&project-name=uni-api-vercel&repository-name=uni-api-vercel)

After clicking the one-click deploy button above, set the environment variable `CONFIG_URL` to the direct link of the configuration file, `DISABLE_DATABASE` to true, and then click Create to create the project. After deployment, you need to manually set the Function Max Duration to 60 seconds in the Vercel project panel under Settings -> Functions, and then click the Deployments menu and click Redeploy to redeploy, which will set the timeout to 60 seconds. If you do not redeploy, the default timeout will remain at the original 10 seconds. Note that you should not delete the Vercel project and recreate it; instead, click redeploy in the Deployments menu within the currently deployed Vercel project to make the Function Max Duration modification take effect.

## Ubuntu deployment

In the warehouse Releases, find the latest version of the corresponding binary file, for example, a file named uni-api-linux-x86_64-0.0.99.pex. Download the binary file on the server and run it:

```bash
wget https://github.com/yym68686/uni-api/releases/download/v0.0.99/uni-api-linux-x86_64-0.0.99.pex
chmod +x uni-api-linux-x86_64-0.0.99.pex
./uni-api-linux-x86_64-0.0.99.pex
```

## Serv00 Remote Deployment (FreeBSD 14.0)

First, log in to the panel, in Additional services click on the tab Run your own applications to enable the option to run your own programs, then go to the panel Port reservation to randomly open a port.

If you don't have your own domain name, go to the panel WWW websites and delete the default domain name provided. Then create a new domain with the Domain being the one you just deleted. After clicking Advanced settings, set the Website type to Proxy domain, and the Proxy port should point to the port you just opened. Do not select Use HTTPS.

ssh login to the serv00 server, execute the following command:

```bash
git clone --depth 1 -b main --quiet https://github.com/yym68686/uni-api.git
cd uni-api
python -m venv uni-api
tmux new -A -s uni-api
source uni-api/bin/activate
export CFLAGS="-I/usr/local/include"
export CXXFLAGS="-I/usr/local/include"
export CC=gcc
export CXX=g++
export MAX_CONCURRENCY=1
export CPUCOUNT=1
export MAKEFLAGS="-j1"
CMAKE_BUILD_PARALLEL_LEVEL=1 cpuset -l 0 pip install -vv -r requirements.txt
cpuset -l 0 pip install -r -vv requirements.txt
```

ctrl+b d to exit tmux, wait a few hours for the installation to complete, and after the installation is complete, execute the following command:

```bash
tmux new -A -s uni-api
source uni-api/bin/activate
export CONFIG_URL=http://file_url/api.yaml
export DISABLE_DATABASE=true
# Modify the port, xxx is the port, modify it yourself, corresponding to the port opened in the panel Port reservation
sed -i '' 's/port=8000/port=xxx/' main.py
sed -i '' 's/reload=True/reload=False/' main.py
python main.py
```

Use ctrl+b d to exit tmux, allowing the program to run in the background. At this point, you can use uni-api in other chat clients. curl test script:

```bash
curl -X POST https://xxx.serv00.net/v1/chat/completions \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-xxx' \
-d '{"model": "gpt-4o","messages": [{"role": "user","content": "Hello"}]}'
```

Reference document:

https://docs.serv00.com/Python/

https://linux.do/t/topic/201181

https://linux.do/t/topic/218738

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

pex linux packaging:

```bash
VERSION=$(cat VERSION)
pex -D . -r requirements.txt \
    -c uvicorn \
    --inject-args 'main:app --host 0.0.0.0 --port 8000' \
    --platform linux_x86_64-cp-3.10.12-cp310 \
    --interpreter-constraint '==3.10.*' \
    --no-strip-pex-env \
    -o uni-api-linux-x86_64-${VERSION}.pex
```

macOS packaging:

```bash
VERSION=$(cat VERSION)
pex -r requirements.txt \
    -c uvicorn \
    --inject-args 'main:app --host 0.0.0.0 --port 8000' \
    -o uni-api-macos-arm64-${VERSION}.pex
```

## Sponsors

We thank the following sponsors for their support:
<!-- ¥2050 -->
- @PowerHunter: ¥2000
- @ioi：¥50

## How to sponsor us

If you would like to support our project, you can sponsor us in the following ways:

1. [PayPal](https://www.paypal.me/yym68686)

2. [USDT-TRC20](https://pb.yym68686.top/~USDT-TRC20), USDT-TRC20 wallet address: `TLFbqSv5pDu5he43mVmK1dNx7yBMFeN7d8`

3. [WeChat](https://pb.yym68686.top/~wechat)

4. [Alipay](https://pb.yym68686.top/~alipay)

Thank you for your support!

## FAQ

- Why does the error `Error processing request or performing moral check: 404: No matching model found` always appear?

Setting ENABLE_MODERATION to false will fix this issue. When ENABLE_MODERATION is true, the API must be able to use the text-moderation-latest model, and if you have not provided text-moderation-latest in the provider model settings, an error will occur indicating that the model cannot be found.

- How to prioritize requests for a specific channel, how to set the priority of a channel?

Directly set the channel order in the api_keys. No other settings are required. Sample configuration file:

```yaml
providers:
  - provider: ai1
    base_url: https://xxx/v1/chat/completions
    api: sk-xxx

  - provider: ai2
    base_url: https://xxx/v1/chat/completions
    api: sk-xxx

api_keys:
  - api: sk-1234
    model:
      - ai2/*
      - ai1/*
```

In this way, request ai2 first, and if it fails, request ai1.

- What is the behavior behind various scheduling algorithms? For example, fixed_priority, weighted_round_robin, lottery, random, round_robin?

All scheduling algorithms need to be enabled by setting api_keys.(api).preferences.SCHEDULING_ALGORITHM in the configuration file to any of the values: fixed_priority, weighted_round_robin, lottery, random, round_robin.

1. fixed_priority: Fixed priority scheduling. All requests are always executed by the channel of the model that first has a user request. In case of an error, it will switch to the next channel. This is the default scheduling algorithm.

2. weighted_round_robin: Weighted round-robin load balancing, requests channels with the user's requested model according to the weight order set in the configuration file api_keys.(api).model.

3. lottery: Draw round-robin load balancing, randomly request the channel of the model with user requests according to the weight set in the configuration file api_keys.(api).model.

4. round_robin: Round-robin load balancing, requests the channel that owns the model requested by the user according to the configuration order in the configuration file api_keys.(api).model. You can check the previous question on how to set the priority of channels.

- How should the base_url be filled in correctly?

Except for some special channels shown in the advanced configuration, all OpenAI format providers need to fill in the base_url completely, which means the base_url must end with /v1/chat/completions. If you are using GitHub models, the base_url should be filled in as https://models.inference.ai.azure.com/chat/completions, not Azure's URL.

For Azure channels, the base_url is compatible with the following formats: https://your-endpoint.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview and https://your-endpoint.services.ai.azure.com/models/chat/completions, https://your-endpoint.openai.azure.com, it is recommended to use the first format. If api-version is not explicitly specified, the default is 2024-10-21.

- How does the model timeout time work? What is the priority of the channel-level timeout setting and the global model timeout setting?

The channel-level timeout setting has higher priority than the global model timeout setting. The priority order is: channel-level model timeout setting > channel-level default timeout setting > global model timeout setting > global default timeout setting > environment variable TIMEOUT.

By adjusting the model timeout time, you can avoid the error of some channels timing out. If you encounter the error `{'error': '500', 'details': 'fetch_response_stream Read Response Timeout'}`, please try to increase the model timeout time.

- How does api_key_rate_limit work? How do I set the same rate limit for multiple models?

If you want to set the same frequency limit for the four models gemini-1.5-pro-latest, gemini-1.5-pro, gemini-1.5-pro-001, gemini-1.5-pro-002 simultaneously, you can set it like this:

```yaml
api_key_rate_limit:
  gemini-1.5-pro: 1000/min
```

This will match all models containing the gemini-1.5-pro string. The frequency limit for these four models, gemini-1.5-pro-latest, gemini-1.5-pro, gemini-1.5-pro-001, gemini-1.5-pro-002, will all be set to 1000/min. The logic for configuring the api_key_rate_limit field is as follows, here is a sample configuration file:

```yaml
api_key_rate_limit:
  gemini-1.5-pro: 1000/min
  gemini-1.5-pro-002: 500/min
```

At this time, if there is a request using the model gemini-1.5-pro-002.

First, the uni-api will attempt to precisely match the model in the api_key_rate_limit. If the rate limit for gemini-1.5-pro-002 is set, then the rate limit for gemini-1.5-pro-002 is 500/min. If the requested model at this time is not gemini-1.5-pro-002, but gemini-1.5-pro-latest, since the api_key_rate_limit does not have a rate limit set for gemini-1.5-pro-latest, it will look for any model with the same prefix as gemini-1.5-pro-latest that has been set, thus the rate limit for gemini-1.5-pro-latest will be set to 1000/min.

## ⭐ Star History

<a href="https://github.com/yym68686/uni-api/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=yym68686/uni-api&type=Date">
</a>