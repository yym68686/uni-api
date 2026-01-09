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

For personal use, one/new-api is too complex with many commercial features that individuals don't need. If you don't want a complicated frontend interface and prefer support for more models, you can try uni-api. This is a project that unifies the management of large language model APIs, allowing you to call multiple backend services through a single unified API interface, converting them all to OpenAI format, and supporting load balancing. Currently supported backend services include: OpenAI, Anthropic, Gemini, Vertex, Azure, AWS, xai, Cohere, Groq, Cloudflare, OpenRouter, [302.AI](https://share.302.ai/IRQYs0) and more.

## ✨ Features

- No front-end, pure configuration file to configure API channels. You can run your own API station just by writing a file, and the documentation has a detailed configuration guide, beginner-friendly.
- Unified management of multiple backend services, supporting providers such as OpenAI, Deepseek, OpenRouter, and other APIs in OpenAI format. Supports OpenAI Dalle-3 image generation.
- Simultaneously supports Anthropic, Gemini, Vertex AI, Azure, AWS, xai, Cohere, Groq, Cloudflare, [302.AI](https://share.302.ai/IRQYs0). Vertex simultaneously supports Claude and Gemini API.
- Support OpenAI, Anthropic, Gemini, Vertex, Azure, AWS, xai native tool use function calls.
- Support OpenAI, Anthropic, Gemini, Vertex, Azure, AWS, xai native image recognition API.
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
- Supports multiple standard OpenAI format interfaces: `/v1/chat/completions`, `/v1/images/generations`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/moderations`, `/v1/models`.
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
      - gpt-5.2 # Usable model name, required
      - claude-sonnet-4-5-20250929: claude-sonnet-4-5 # Rename model, claude-sonnet-4-5-20250929 is the provider's model name, claude-sonnet-4-5 is the renamed name, you can use a simple name to replace the original complex name, optional
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # Supports multiple API Keys, multiple keys automatically enable polling load balancing, at least one key, required
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-sonnet-4-5-20250929: claude-sonnet-4-5 # Rename model, claude-sonnet-4-5-20250929 is the provider's model name, claude-sonnet-4-5 is the renamed name, you can use a simple name to replace the original complex name, optional
      - claude-sonnet-4-5-20250929: claude-sonnet-4-5-think # Rename model, claude-sonnet-4-5-20250929 is the provider's model name, claude-sonnet-4-5-think is the renamed name, if "think" is in the renamed name, it will be automatically converted to claude think model, default think token limit is 4096. Optional
    tools: true # Whether to support tools, such as generating code, generating documents, etc., default is true, optional
    preferences:
      post_body_parameter_overrides: # Support customizing request body parameters
        claude-sonnet-4-5-think: # Add custom request body parameters to the model claude-sonnet-4-5-think
          tools:
            - type: code_execution_20250522 # Add code_execution tool to the model claude-sonnet-4-5-think
              name: code_execution
            - type: web_search_20250305 # Add web_search tool to the model claude-sonnet-4-5-think, max_uses means to use up to 5 times
              name: web_search
              max_uses: 5

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url supports v1beta/v1, only for Gemini model use, required
    api: # Supports multiple API Keys, multiple keys automatically enable polling load balancing, at least one key, required
      - AIzaSyAN2k6IRdgw123
      - AIzaSyAN2k6IRdgw456
      - AIzaSyAN2k6IRdgw789
    model:
      - gemini-3-pro-preview: gemini-3-pro
      - gemini-2.5-flash: gemini-2.5-flash # After renaming, the original model name gemini-2.5-flash cannot be used, if you want to use the original name, you can add the original name in the model, just add the line below to use the original name
      - gemini-2.5-flash
      - gemini-pro-latest: gemini-2.5-pro-search # To enable search for a model, rename it with the -search suffix and set custom request body parameters for this model in `post_body_parameter_overrides`.
      - gemini-2.5-flash: gemini-2.5-flash-think-24576-search # To enable search for a model, rename it with the -search suffix and set custom request body parameters for this model in post_body_parameter_overrides. Additionally, you can customize the inference budget using -think-number. These options can be used together or separately.
      - gemini-2.5-flash: gemini-2.5-flash-think-0 # Support to rename models with -think-number suffix to enable search, if the number is 0, it means to close the reasoning.
      - gemini-embedding-001
      - text-embedding-004
    tools: true
    preferences:
      api_key_rate_limit: 15/min # Each API Key can request up to 15 times per minute, optional. The default is 999999/min. Supports multiple frequency constraints: 15/min,10/day
      # api_key_rate_limit: # You can set different frequency limits for each model
      #   gemini-2.5-flash: 10/min,500/day
      #   gemini-2.5-pro: 5/min,25/day,1048576/tpr # 1048576/tpr means the token limit per request is 1,048,576 tokens.
      #   default: 4/min # If the model does not set the frequency limit, use the frequency limit of default
      api_key_cooldown_period: 60 # Each API Key will be cooled down for 60 seconds after encountering a 429 error. Optional, the default is 0 seconds. When set to 0, the cooling mechanism is not enabled. When there are multiple API keys, the cooling mechanism will take effect.
      api_key_schedule_algorithm: round_robin # Set the request order of multiple API Keys, optional. The default is round_robin, and the optional values are: round_robin, random, fixed_priority, smart_round_robin. It will take effect when there are multiple API keys. round_robin is polling load balancing, and random is random load balancing. fixed_priority is fixed priority scheduling, always use the first available API key. `smart_round_robin` is an intelligent scheduling algorithm based on historical success rates, see FAQ for details.
      model_timeout: # Model timeout, in seconds, default 100 seconds, optional
        gemini-2.5-pro: 500 # Model gemini-2.5-pro timeout is 500 seconds
        gemini-2.5-flash: 500 # Model gemini-2.5-flash timeout is 500 seconds
        default: 10 # Model does not have a timeout set, use the default timeout of 10 seconds, when requesting a model not in model_timeout, the timeout is also 10 seconds, if default is not set, uni-api will use the default timeout set by the environment variable TIMEOUT, the default timeout is 100 seconds
      keepalive_interval: # Heartbeat interval, in seconds, default 99999 seconds, optional. Suitable for when uni-api is hosted on cloudflare and uses inference models. Priority is higher than the global configuration keepalive_interval.
        gemini-2.5-pro: 50 # Model gemini-2.5-pro heartbeat interval is 50 seconds, this value must be less than the model_timeout set timeout, otherwise it will be ignored.
      proxy: socks5://[username]:[password]@[ip]:[port] # Proxy address, optional. Supports socks5 and http proxies, default is not used.
      headers:  # Add custom http request headers, optional
        Custom-Header-1: Value-1
        Custom-Header-2: Value-2
      post_body_parameter_overrides: # Support customizing request body parameters
        gemini-2.5-flash-search: # Add custom request body parameters to the model gemini-2.5-flash-search
          tools:
            - google_search: {} # Add google_search tool to the model gemini-2.5-flash-search
            - url_context: {} # Add url_context tool to the model gemini-2.5-flash-search

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx # Description: Your Google Cloud project ID. Format: String, usually composed of lowercase letters, numbers, and hyphens. How to obtain: You can find your project ID in the project selector of the Google Cloud Console.
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # Description: Private key for Google Cloud Vertex AI service account. Format: A JSON formatted string containing the private key information of the service account. How to obtain: Create a service account in Google Cloud Console, generate a JSON formatted key file, and then set its content as the value of this environment variable.
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # Description: Email address of the Google Cloud Vertex AI service account. Format: Usually a string like "service-account-name@project-id.iam.gserviceaccount.com". How to obtain: Generated when creating a service account, or you can view the service account details in the "IAM and Admin" section of the Google Cloud Console.
    model:
      - gemini-2.5-flash
      - gemini-3-pro-preview: gemini-3-pro
      - gemini-pro-latest: gemini-2.5-pro-search # To enable search for a model, rename it with the -search suffix and set custom request body parameters for this model in `post_body_parameter_overrides`. Not setting post_body_parameter_overrides will not enable search.
      - claude-sonnet-4-5@20250929: claude-sonnet-4-5
      - claude-opus-4-5@20251101: claude-opus-4-5
      - claude-haiku-4-5@20251001: claude-haiku-4-5
      - gemini-embedding-001
      - text-embedding-004
    tools: true
    notes: https://xxxxx.com/ # You can put the provider's website, notes, official documentation, optional
    preferences:
      post_body_parameter_overrides: # Support customizing request body parameters
        gemini-2.5-pro-search: # Add custom request body parameters to the model gemini-2.5-pro-search
          tools:
            - google_search: {} # Add google_search tool to the model gemini-2.5-pro-search
        gemini-2.5-flash:
          generationConfig:
            thinkingConfig:
              includeThoughts: True
              thinkingBudget: 24576
            maxOutputTokens: 65535
        gemini-2.5-flash-search:
          tools:
            - google_search: {}
            - url_context: {}

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
      - gpt-5.2
    preferences:
      post_body_parameter_overrides: # Support customizing request body parameters
        key1: value1 # Force the request to add "key1": "value1" parameter
        key2: value2 # Force the request to add "key2": "value2" parameter
        stream_options:
          include_usage: true # Force the request to add "stream_options": {"include_usage": true} parameter
      cooldown_period: 0 # When cooldown_period is set to 0, the cooling mechanism is not enabled, the priority is higher than the global configuration cooldown_period.

  - provider: databricks
    base_url: https://xxx.azuredatabricks.net
    api:
      - xxx
    model:
      - databricks-claude-sonnet-4: claude-sonnet-4
      - databricks-claude-opus-4: claude-opus-4
      - databricks-claude-sonnet-4-5: claude-sonnet-4-5

  - provider: aws
    base_url: https://bedrock-runtime.us-east-1.amazonaws.com
    aws_access_key: xxxxxxxx
    aws_secret_key: xxxxxxxx
    model:
      - anthropic.claude-sonnet-4-5-20250929-v1:0: claude-sonnet-4-5

  - provider: vertex-express
    base_url: https://aiplatform.googleapis.com/
    project_id:
      - xxx # project_id of key1
      - xxx # project_id of key2
    api:
      - xx.xxx # api of key1
      - xx.xxx # api of key2
    model:
      - gemini-3-pro-preview

  - provider: other-provider
    base_url: https://api.xxx.com/v1/messages
    api: sk-bNnAOJyA-xQw_twAA
    model:
      - causallm-35b-beta2ep-q6k: causallm-35b
      - anthropic/claude-sonnet-4-5
    tools: false
    engine: openrouter # Force the use of a specific message format, currently supports gpt, claude, gemini, openrouter native format, optional

  # Doubao (Volcengine Ark) Translation via /api/v3/responses
  - provider: doubao-translate
    base_url: https://ark.cn-beijing.volces.com/api/v3/responses
    api: xxxxxxxxxxxxxxxxxxxxxxxx
    model:
      - doubao-seed-translation
    preferences:
      post_body_parameter_overrides:
        doubao-seed-translation:
          translation_options:
            target_language: zh # Default target language (optional)
            # source_language: en # Optional

api_keys:
  - api: sk-KjjI60Yf0JFWxfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key, required for users to use this service
    model: # Models that can be used by this API Key, optional. Default channel-level polling load balancing is enabled, and each request model is requested in sequence according to the model configuration. It is not related to the original channel order in providers. Therefore, you can set different request sequences for each API key.
      - gpt-5.2 # Usable model name, can use all gpt-5.2 models provided by providers
      - claude-sonnet-4-5 # Usable model name, can use all claude-sonnet-4-5 models provided by providers
      - gemini/* # Usable model name, can only use all models provided by providers named gemini, where gemini is the provider name, * represents all models
    role: admin # Set the alias of the API key, optional. The request log will display the alias of the API key. If role is admin, only this API key can request the v1/stats,/v1/generate-api-key endpoints. If all API keys do not have role set to admin, the first API key is set as admin and has permission to request the v1/stats,/v1/generate-api-key endpoints.

  - api: sk-pkhf60Yf0JGyJxgRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-sonnet-4-5 # Usable model name, can only use the claude-sonnet-4-5 model provided by the provider named anthropic. Models with the same name from other providers cannot be used. This syntax will not match the model named anthropic/claude-sonnet-4-5 provided by other-provider.
      - <anthropic/claude-sonnet-4-5> # By adding angle brackets on both sides of the model name, it will not search for the claude-sonnet-4-5 model under the channel named anthropic, but will take the entire anthropic/claude-sonnet-4-5 as the model name. This syntax can match the model named anthropic/claude-sonnet-4-5 provided by other-provider. But it will not match the claude-sonnet-4-5 model under anthropic.
      - openai-test/omni-moderation-latest # When message moderation is enabled, the omni-moderation-latest model under the channel named openai-test can be used for moderation.
      - sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo/* # Support using other API keys as channels
    preferences:
      SCHEDULING_ALGORITHM: fixed_priority # When SCHEDULING_ALGORITHM is fixed_priority, use fixed priority scheduling, always execute the channel of the first model with a request. Default is enabled, SCHEDULING_ALGORITHM default value is fixed_priority. SCHEDULING_ALGORITHM optional values are: fixed_priority, round_robin, weighted_round_robin, lottery, random.
      # When SCHEDULING_ALGORITHM is random, use random polling load balancing, randomly request the channel of the model with a request.
      # When SCHEDULING_ALGORITHM is round_robin, use polling load balancing, request the channel of the model used by the user in order.
      AUTO_RETRY: true # Whether to automatically retry, automatically retry the next provider, true for automatic retry, false for no automatic retry, default is true. Also supports setting a number, indicating the number of retries.
      rate_limit: 15/min # Supports rate limiting, each API Key can request up to 15 times per minute, optional. The default is 999999/min. Supports multiple frequency constraints: 15/min,10/day
      # rate_limit: # You can set different frequency limits for each model
      #   gemini-2.5-flash: 10/min,500/day
      #   gemini-2.5-pro: 5/min,25/day
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
      credits: 10 # Supports setting balance, the number set here represents that the API Key can use 10 dollars, optional. The default is unlimited balance, when set to 0, the key cannot be used. When the user has used up the balance, subsequent requests will be blocked.
      created_at: 2024-01-01T00:00:00+08:00 # When the balance is set, created_at must be set, indicating that the usage cost starts from the time set in created_at. Optional. The default is 30 days before the current time.

preferences: # Global configuration
  model_timeout: # Model timeout, in seconds, default 100 seconds, optional
    gpt-5.2: 10 # Model gpt-5.2 timeout is 10 seconds, gpt-5.2 is the model name, when requesting models like gpt-5.2-2025-12-11, the timeout is also 10 seconds
    claude-sonnet-4-5: 10 # Model claude-sonnet-4-5 timeout is 10 seconds, when requesting models like claude-sonnet-4-5-20250929, the timeout is also 10 seconds
    default: 10 # Model does not have a timeout set, use the default timeout of 10 seconds, when requesting a model not in model_timeout, the default timeout is 10 seconds, if default is not set, uni-api will use the default timeout set by the environment variable TIMEOUT, the default timeout is 100 seconds
    gemini-3-pro: 30 # Model gemini-3-pro timeout is 30 seconds, when requesting models starting with gemini-3-pro, the timeout is 30 seconds
    gemini-3-pro-image: 100 # Model gemini-3-pro-image timeout is 100 seconds, when requesting models starting with gemini-3-pro-image, the timeout is 100 seconds
  cooldown_period: 300 # Channel cooldown time, in seconds, default 300 seconds, optional. When a model request fails, the channel will be automatically excluded and cooled down for a period of time, and will not request the channel again. After the cooldown time ends, the model will be automatically restored until the request fails again, and it will be cooled down again. When cooldown_period is set to 0, the cooling mechanism is not enabled.
  rate_limit: 999999/min # uni-api global rate limit, in times/minute, supports multiple frequency constraints, such as: 15/min,10/day. Default 999999/min, optional.
  keepalive_interval: # Heartbeat interval, in seconds, default 99999 seconds, optional. Suitable for when uni-api is hosted on cloudflare and uses inference models.
    gemini-2.5-pro: 50 # Model gemini-2.5-pro heartbeat interval is 50 seconds, this value must be less than the model_timeout set timeout, otherwise it will be ignored.
  error_triggers: # Error triggers, when the message returned by the model contains any of the strings in the error_triggers, the channel will return an error. Optional
    - The bot's usage is covered by the developer
    - process this request due to overload or policy
  proxy: socks5://[username]:[password]@[ip]:[port] # Proxy address, optional.
  model_price: # Model price, in dollars/M tokens, optional. Default price is 1,2, which means input 1 dollar/1M tokens, output 2 dollars/1M tokens.
    gpt-5.2: 1,2
    claude-sonnet-4-5: 0.12,0.48
    default: 1,2
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
- DEBUG: Whether to enable debug mode, default is false, optional. When enabled, more logs will be printed, which can be used when submitting issues.
- TIMEOUT: Request timeout, default is 100 seconds. The timeout can control the time needed to switch to the next channel when one channel does not respond. Optional
- DISABLE_DATABASE: Whether to disable the database, default is false, optional
- DB_TYPE: Database type, default is sqlite, optional. Supports sqlite and postgres.

When DB_TYPE is postgres, the following environment variables need to be set:

- DB_USER: Database user name, default is postgres, optional
- DB_PASSWORD: Database password, default is mysecretpassword, optional
- DB_HOST: Database host, default is localhost, optional
- DB_PORT: Database port, default is 5432, optional
- DB_NAME: Database name, default is postgres, optional

## Koyeb remote deployment

Click the button below to automatically use the built uni-api docker image to deploy:

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=uni-api&type=docker&image=docker.io%2Fyym68686%2Funi-api%3Alatest&instance_type=free&regions=was&instances_min=0&env%5BCONFIG_URL%5D=)

There are two ways to let Koyeb read the configuration file, choose one of them:

1. Fill in the environment variable `CONFIG_URL` with the direct link of the configuration file

2. Paste the api.yaml file content, if you paste the api.yaml file content directly into the Koyeb environment variable setting file, after pasting the text in the text box, enter the api.yaml path as `/home/api.yaml` in the path field.

Then click the Deploy button.

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
source uni-api/bin/activate
pip install --upgrade pip
cpuset -l 0 pip install -vv -r requirements.txt
```

From the start of installation to the completion of installation, it will take about 10 minutes. After the installation is complete, execute the following command:

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
-d '{"model": "gpt-5.2","messages": [{"role": "user","content": "Hello"}]}'
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

### api.yaml hot reload (minimal change) + frontend sync

`uni-api` reads `api.yaml` on startup. If you want “edit `api.yaml` in a frontend and have uni-api take effect immediately”, the minimal-change approach is:

- Mount the same `api.yaml` into both backend `uni-api` and a frontend (`uni-api-status`)
- Add a `config-watcher` service to watch `api.yaml` and `docker restart uni-api` on change

Here is a ready-to-use `docker-compose.yml` example (place `./api.yaml` in the same directory):

```yaml
services:
  uni-api:
    image: yym68686/uni-api:latest
    container_name: uni-api
    restart: unless-stopped
    ports:
      - "8001:8000"
    environment:
      - WATCHFILES_FORCE_POLLING=true
    volumes:
      - ./api.yaml:/home/api.yaml
      - ./uniapi_db:/home/data

  uniapi-frontend:
    image: ghcr.io/melosbot/uni-api-status:latest
    container_name: uni-api-frontend
    restart: unless-stopped
    ports:
      - "3700:3000"
    environment:
      - NODE_ENV=production
      - PORT=3000
      - API_YAML_PATH=/app/config/api.yaml
      - STATS_DB_PATH=/app/data/stats.db
    volumes:
      - ./api.yaml:/app/config/api.yaml
      - ./uniapi_db:/app/data:ro
    depends_on:
      - uni-api

  config-watcher:
    image: alpine:latest
    container_name: uni-api-config-watcher
    restart: unless-stopped
    volumes:
      - ./api.yaml:/watch/api.yaml:ro
      - /var/run/docker.sock:/var/run/docker.sock
    command: >
      sh -c "
      apk add --no-cache inotify-tools docker-cli &&
      while true; do
        inotifywait -e modify,close_write /watch/api.yaml &&
        echo 'api.yaml changed, restarting uni-api...' &&
        docker restart uni-api
      done
      "
```

Note: `config-watcher` mounts `/var/run/docker.sock` to restart containers; use it only on a trusted host/environment.

If you want to use a single domain for both frontend and API, here is a Caddy reverse proxy example (`Caddyfile`):

```caddyfile
yourdomain.com {
  encode gzip
  tls a@bc.com

  route /v1* {
    reverse_proxy localhost:8001 {
      header_up Host {host}
      header_up X-Real-IP {remote}
    }
  }

  route * {
    reverse_proxy localhost:3700 {
      header_up Host {host}
      header_up X-Real-IP {remote}
    }
  }
}
```

Now you can edit `api.yaml` via `yourdomain.com` (frontend). After saving, `uni-api` will be restarted and read the latest `api.yaml`.

Run Docker Compose container in the background

```bash
docker-compose pull
docker-compose up -d
```

Docker build

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t yym68686/uni-api:latest --push .
docker pull yym68686/uni-api:latest

# test image
docker buildx build --platform linux/amd64,linux/arm64 -t yym68686/uni-api:test -f Dockerfile.debug --push .
docker pull yym68686/uni-api:test
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
-d '{"model": "gpt-5.2","messages": [{"role": "user", "content": "Hello"}],"stream": true}'
```

Audio input examples (/v1/chat/completions):

```bash
curl -X POST 'https://xxx.xxx/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${API}" \
  --data '{
  "model": "gemini-2.5-flash",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Generate a transcript of the speech."
        },
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64 bytes here>",
            "format": "wav"
          }
        }
      ]
    }
  ]
}'
```

Use a URL for audio input:

```bash
curl -X POST 'https://xxx.xxx/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${API}" \
  --data '{
  "model": "gemini-2.5-flash",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Generate a transcript of the speech."
        },
        {
          "type": "input_audio",
          "input_audio": {
            "data": "https://www.youtube.com/watch?v=ku-N-eS1lgM",
            "format": "mp4"
          }
        }
      ]
    }
  ]
}'
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

## HuggingFace Space Remote Deployment

WARNING: Please be aware of the risk of key leakage in remote deployments. Do not abuse the service to avoid account suspension.

The Space repository requires three files: `Dockerfile`, `README.md`, and `entrypoint.sh`.
To run the program, you also need api.yaml (I'll use the example of storing it entirely in secrets, but you can also implement it via HTTP download). Access matching, model and channel configurations are all in the configuration file.

Operation Steps:

1. Visit https://huggingface.co/new-space to create a new space. It should be a public repository; the open source license/name/description can be filled as desired.

2. Visit your space's files page at https://huggingface.co/spaces/your-name/your-space-name/tree/main and upload the three files (`Dockerfile`, `README.md`, `entrypoint.sh`).

3. Visit your space's settings page at https://huggingface.co/spaces/your-name/your-space-name/settings, find the Secrets section and create a new secret called `API_YAML_CONTENT` (note the uppercase). Write your api.yaml locally, then copy it directly into the secret field using UTF-8 encoding.

4. Still in settings, find Factory rebuild and let it rebuild. If you modify secrets or files, or manually restart the Space, it may get stuck with no logs. Use this method to resolve such issues.

5. In the upper right corner of the settings page, find the three-dot button and select "Embed this Space" to get the public link for your Space. The format is https://(your-name)-(your-space-name).hf.space (remove the parentheses).

Related File Codes:

```Dockerfile
# Dockerfile,del this line
# Use the uni-api official image
FROM yym68686/uni-api:latest
# Create data directory and set permissions
RUN mkdir -p /data && chown -R 1000:1000 /data
# Set up user and working directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DISABLE_DATABASE=true
# Copy entrypoint script
COPY --chown=user entrypoint.sh /home/user/entrypoint.sh
RUN chmod +x /home/user/entrypoint.sh
# Ensure /home directory is writable (this is important!)
USER root
RUN chmod 777 /home
USER user
# Set working directory
WORKDIR /home/user
# Entry point
ENTRYPOINT ["/home/user/entrypoint.sh"]
```

```markdown
---
title: Uni API
emoji: 🌍
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: gpl-3.0
---
```

```shell
# entrypoint.sh,del this line
#!/bin/sh
set -e
CONFIG_FILE_PATH="/home/api.yaml"  # Note this is changed to /home/api.yaml
echo "DEBUG: Entrypoint script started."
# Check if Secret exists
if [ -z "$API_YAML_CONTENT" ]; then
  echo "ERROR: Secret 'API_YAML_CONTENT' does not exist or is empty. Exiting."
  exit 1
else
  echo "DEBUG: API_YAML_CONTENT secret found. Preparing to write..."
  printf '%s\n' "$API_YAML_CONTENT" > "$CONFIG_FILE_PATH"
  echo "DEBUG: Attempted to write to $CONFIG_FILE_PATH."

  if [ -f "$CONFIG_FILE_PATH" ]; then
    echo "DEBUG: File $CONFIG_FILE_PATH created successfully. Size: $(wc -c < "$CONFIG_FILE_PATH") bytes."
    # Display the first few lines for debugging (be careful not to display sensitive information)
    echo "DEBUG: First few lines (without sensitive info):"
    head -n 3 "$CONFIG_FILE_PATH" | grep -v "api:" | grep -v "password"
  else
    echo "ERROR: File $CONFIG_FILE_PATH was NOT created."
    exit 1
  fi
fi
echo "DEBUG: About to execute python main.py..."
# No need to use the --config parameter as the program has a default path
cd /home
exec python main.py "$@"
```

## uni-api frontend deployment

The frontend of uni-api can be deployed by yourself, address: https://github.com/yym68686/uni-api-web

You can also use the frontend I deployed, address: https://uni-api-web.pages.dev/

For explanations of the frontend environment variables, see the `uni-api-web` README: https://github.com/yym68686/uni-api-web

Here is a `docker-compose.yml` example (removed `mybot/publicbot/servicebot/addetect`; environment variables use `${VAR:-}` placeholders):

```yaml
services:
  web:
    image: yym68686/uni-api-frontend:main
    container_name: uni-api-frontend
    restart: unless-stopped
    depends_on:
      - api
    environment:
      # Inside Docker, use service-to-service networking (NOT localhost).
      API_BASE_URL: ${API_BASE_URL:-http://api:8000/v1}
      NEXT_TELEMETRY_DISABLED: ${NEXT_TELEMETRY_DISABLED:-1}
      NODE_ENV: ${NODE_ENV:-production}
      APP_NAME: ${APP_NAME:-UniAPI}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID:-}
      GOOGLE_REDIRECT_URI: ${GOOGLE_REDIRECT_URI:-}
    ports:
      - "8003:3000"

  db:
    image: postgres:17.6-alpine
    container_name: uni-api-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${DB_POSTGRES_USER:-uniapi}
      POSTGRES_PASSWORD: ${DB_POSTGRES_PASSWORD:-}
      POSTGRES_DB: ${DB_POSTGRES_DB:-uniapi}
    ports:
      - "5433:5432"
    volumes:
      - uniapi_pg_data:/var/lib/postgresql/data

  api:
    image: yym68686/uni-api-backend:main
    container_name: uni-api-backend
    restart: unless-stopped
    depends_on:
      - db
    environment:
      DATABASE_URL: ${DATABASE_URL:-}
      APP_ENV: ${APP_ENV:-dev}
      APP_NAME: ${BACKEND_APP_NAME:-Uni API Backend}
      API_PREFIX: ${API_PREFIX:-/v1}
      SESSION_TTL_DAYS: ${SESSION_TTL_DAYS:-7}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID:-}
      GOOGLE_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET:-}
      GOOGLE_REDIRECT_URI: ${GOOGLE_REDIRECT_URI:-}
      ADMIN_BOOTSTRAP_TOKEN: ${ADMIN_BOOTSTRAP_TOKEN:-}
      RESEND_API_KEY: ${RESEND_API_KEY:-}
      RESEND_FROM_EMAIL: ${RESEND_FROM_EMAIL:-}
      EMAIL_VERIFICATION_REQUIRED: ${EMAIL_VERIFICATION_REQUIRED:-true}
    ports:
      - "8002:8000"

  postgres:
    container_name: postgres
    image: postgres:17.6
    restart: always
    environment:
      POSTGRES_USER: ${UNIAPI_POSTGRES_USER:-root}
      POSTGRES_PASSWORD: ${UNIAPI_POSTGRES_PASSWORD:-}
      POSTGRES_DB: ${UNIAPI_POSTGRES_DB:-uniapi}
    ports:
      - "5432:5432"
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${UNIAPI_POSTGRES_USER:-root} -d ${UNIAPI_POSTGRES_DB:-uniapi}"]
      interval: 5s
      timeout: 5s
      retries: 5

  uni-api:
    container_name: uni-api
    image: yym68686/uni-api:latest
    environment:
      # CONFIG_URL: ${CONFIG_URL:-}
      TIMEOUT: ${TIMEOUT:-200}
      DB_TYPE: ${DB_TYPE:-postgres}
      DB_HOST: ${DB_HOST:-postgres}
      DB_PORT: ${DB_PORT:-5432}
      DB_USER: ${DB_USER:-root}
      DB_PASSWORD: ${DB_PASSWORD:-}
      DB_NAME: ${DB_NAME:-uniapi}
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8001:8000"
    volumes:
      - ./api-copy.yaml:/home/api.yaml
      - ./uniapi_db:/home/data
      - /etc/localtime:/etc/localtime:ro
    restart: unless-stopped

volumes:
  uniapi_pg_data:
```

## Sponsors

We thank the following sponsors for their support:
<!-- ¥2050 -->
- @PowerHunter: ¥2000
- @IM4O4: ¥100
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

Setting ENABLE_MODERATION to false will fix this issue. When ENABLE_MODERATION is true, the API must be able to use the omni-moderation-latest model, and if you have not provided omni-moderation-latest in the provider model settings, an error will occur indicating that the model cannot be found.

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

- What is the behavior behind various scheduling algorithms? For example, fixed_priority, weighted_round_robin, lottery, random, round_robin, smart_round_robin?

All scheduling algorithms need to be enabled by setting api_keys.(api).preferences.SCHEDULING_ALGORITHM in the configuration file to any of the values: fixed_priority, weighted_round_robin, lottery, random, round_robin, smart_round_robin.

1. fixed_priority: Fixed priority scheduling. All requests are always executed by the channel of the model that first has a user request. In case of an error, it will switch to the next channel. This is the default scheduling algorithm.

2. weighted_round_robin: Weighted round-robin load balancing, requests channels with the user's requested model according to the weight order set in the configuration file api_keys.(api).model.

3. lottery: Draw round-robin load balancing, randomly request the channel of the model with user requests according to the weight set in the configuration file api_keys.(api).model.

4. round_robin: Round-robin load balancing, requests the channel that owns the model requested by the user according to the configuration order in the configuration file api_keys.(api).model. You can check the previous question on how to set the priority of channels.

5. smart_round_robin: Intelligent success rate scheduling. This is an advanced scheduling algorithm designed for channels with a large number of API Keys (hundreds, thousands, or even tens of thousands). Its core mechanism is:
    - **Sorting based on historical success rate**: The algorithm dynamically sorts API Keys based on their actual request success rate over the past 72 hours.
    - **Intelligent grouping and load balancing**: To prevent traffic from always concentrating on a few "optimal" keys, the algorithm intelligently divides all keys (including unused ones) into several groups. It distributes the keys with the highest success rates to the beginning of each group, the next highest to the second position, and so on. This ensures that the load is evenly distributed among different tiers of keys and also guarantees that new or historically underperforming keys have a chance to be tried (exploration).
    - **Periodic automatic updates**: After all keys in a channel have been polled once, the system automatically triggers a re-sorting, pulling the latest success rate data from the database to generate a new, more optimal key sequence. The update frequency is adaptive: the larger the key pool and the lower the request volume, the longer the update cycle; and vice versa.
    - **Applicable scenarios**: It is highly recommended for users with a large number of API Keys to enable this algorithm to maximize the utilization of the key pool and the success rate of requests.

- How should the base_url be filled in correctly?

Except for some special channels shown in the advanced configuration, all OpenAI format providers need to fill in the base_url completely, which means the base_url must end with /v1/chat/completions or /v1/responses. If you are using GitHub models, the base_url should be filled in as https://models.inference.ai.azure.com/chat/completions, not Azure's URL.

For Azure channels, the base_url is compatible with the following formats: https://your-endpoint.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview and https://your-endpoint.services.ai.azure.com/models/chat/completions, https://your-endpoint.openai.azure.com, it is recommended to use the first format. If api-version is not explicitly specified, the default is 2024-10-21.

- How does the model timeout time work? What is the priority of the channel-level timeout setting and the global model timeout setting?

The channel-level timeout setting has higher priority than the global model timeout setting. The priority order is: channel-level model timeout setting > channel-level default timeout setting > global model timeout setting > global default timeout setting > environment variable TIMEOUT.

More specifically, `model_timeout` and `keepalive_interval` share the same matching and fallback rules (they are applied in the same way for global `preferences.model_timeout` / `preferences.keepalive_interval` and for each provider's `preferences.model_timeout` / `preferences.keepalive_interval`):

1. Define two names:
   - **Request model name**: the value you send in the request body `model` field, for example `gpt-5.2`, `claude-sonnet-4-5`.
   - **Upstream model name**: the original model ID on the provider side, i.e. the *left* side of your mapping in `providers.(provider).model`. For example:
     ```yaml
     providers:
       - provider: openai
         model:
           - gpt-5.2-2025-12-11: gpt-5.2   # left = upstream model name, right = request model alias
     ```
     In this case, the request model name is `gpt-5.2`, and the upstream model name is `gpt-5.2-2025-12-11`.

2. When resolving timeout / keepalive for a *specific provider* (its own `preferences.model_timeout` or `preferences.keepalive_interval`), uni-api tries the following 6 steps in order (it stops at the first match):

   1) Use the **request model name** to look for an exact key match in that provider's `model_timeout` / `keepalive_interval`.

   2) If there is no exact hit, try a **fuzzy match** with the request model name: check whether any key in `model_timeout` / `keepalive_interval` is contained as a substring of the request model name.
      For example, if you only configure:
      ```yaml
      model_timeout:
        gpt-5.2: 20
      ```
      then models like `gpt-5.2-2025-12-11` or `gpt-5-mini` will also match 20 seconds.

   3) If the request model name did not match anything for this provider, switch to the **upstream model name** and look for an exact key match in the same `model_timeout` / `keepalive_interval`.

   4) If the upstream model name still does not match exactly, try a **fuzzy match** with the upstream model name: check whether any key in `model_timeout` / `keepalive_interval` is a substring of the upstream model name.

   5) If none of the above matched, but the provider-level `model_timeout` / `keepalive_interval` defines a `default`, use this provider-level `default`.

   6) If this provider has no match at all (including no provider-level `default`), uni-api falls back to the **global** `preferences.model_timeout` / `preferences.keepalive_interval`:
      - It re-tries with the **request model name** against the global config using the same sequence: exact match → fuzzy match → global `default`.
      - If not found, it re-tries with the **upstream model name** against the global config using the same sequence: exact match → fuzzy match → global `default`.
      - If even the global config has no match, the final fallback is the environment variable `TIMEOUT` (default 100 seconds).

In practice, this means that the keys under `model_timeout` / `keepalive_interval` can be:

- The request alias you actually use (e.g. `gpt-5.2`, `claude-sonnet-4-5`);
- The upstream model ID (e.g. `gpt-5.2-2025-12-11`);
- Or a stable prefix / substring that is shared by a family of models (e.g. just `gpt-5.2` to cover `gpt-5.2-2025-12-11`, `gpt-5-mini`, etc.).

By tuning `model_timeout` and `keepalive_interval` based on this matching behavior, you can avoid unnecessary timeouts and better control how long uni-api waits on each provider. If you encounter the error `{'error': '500', 'details': 'fetch_response_stream Read Response Timeout'}`, try increasing the timeout for the corresponding model (or its prefix) instead of only changing the global TIMEOUT.

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

- I want to set channel 1 and channel 2 to random round-robin, and uni-api will request channel 3 after channel 1 and channel 2 failure. How do I set it?

uni-api supports api key as a channel, and can use this feature to manage channels by grouping them.

```yaml
api_keys:
  - api: sk-xxx1
    model:
      - sk-xxx2/* # channel 1 2 use random round-robin, request channel 3 after failure
      - aws/* # channel 3
    preferences:
      SCHEDULING_ALGORITHM: fixed_priority # always request api key: sk-xxx2 first, then request channel 3 after failure

  - api: sk-xxx2
    model:
      - anthropic/claude-sonnet-4-5 # channel 1
      - openrouter/claude-sonnet-4-5 # channel 2
    preferences:
      SCHEDULING_ALGORITHM: random # channel 1 2 use random round-robin
```

- I want to use Cloudflare AI Gateway, how should I fill in the base_url?

For gemini channels, the base_url for Cloudflare AI Gateway should be filled in as https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_name}/google-ai-studio/v1beta/openai/chat/completions , where {account_id} and {gateway_name} need to be replaced with your Cloudflare account ID and Gateway name.

For Vertex channels, the base_url for Cloudflare AI Gateway should be filled in as https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_name}/google-vertex-ai , where {account_id} and {gateway_name} need to be replaced with your Cloudflare account ID and Gateway name.

- When does the api key have management permissions?

1. When there is only one key, it means self-use, the only key has management permissions, and can see all channel sensitive information through the frontend.
2. When there are two or more keys, you must specify one or more keys to have the role of admin, only the keys with the role of admin have permission to access sensitive information. The reason for this design is to prevent another key user from also accessing sensitive information. Therefore, the design of forcing the key to set the role to admin has been added.

- When using koyeb to deploy uni-api, if the configuration file channel does not write the model field, the startup will report an error. How to solve it?

When deploying uni-api on koyeb, if the configuration file channel does not include the model field, it will report an error on startup. This is because the default permission of api.yaml on koyeb is 0644, and uni-api does not have write permission. When uni-api tries to obtain the model field, it will attempt to modify the configuration file, which will result in an error. You can resolve this by entering `chmod 0777 api.yaml` in the console to grant uni-api write permission.


- Why can't I get the user's real IP after using nginx as a proxy?

Add to
```xml
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Real-IP $remote_addr;

```

## Load Testing

Load testing tool: [locust](https://locust.io/)

Load testing script: [test/locustfile.py](test/locustfile.py)

mock_server: [test/mock_server.go](test/mock_server.go)

Start load testing:

```bash
go run test/mock_server.go
# 100 10 120s
locust -f test/locustfile.py
python main.py
```

Load testing result:

| Type | Name | 50% | 66% | 75% | 80% | 90% | 95% | 98% | 99% | 99.9% | 99.99% | 100% | # reqs |
|------|------|-----|-----|-----|-----|-----|-----|-----|-----|--------|---------|------|--------|
| POST | /v1/chat/completions (stream) | 18 | 23 | 29 | 35 | 83 | 120 | 140 | 160 | 220 | 270 | 270 | 6948 |
| | Aggregated | 18 | 23 | 29 | 35 | 83 | 120 | 140 | 160 | 220 | 270 | 270 | 6948 |

## Security

We take security seriously. If you discover any security issues, please contact us at [yym68686@outlook.com](mailto:yym68686@outlook.com).

**Acknowledgments:**

We would like to thank **@ryougishiki214** for reporting a security issue, which has been resolved in [v1.5.1](https://github.com/yym68686/uni-api/releases/tag/v1.5.1).

## ⭐ Star History

<a href="https://github.com/yym68686/uni-api/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=yym68686/uni-api&type=Date">
</a>
