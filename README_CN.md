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

## 介绍

如果个人使用的话，one/new-api 过于复杂，有很多个人不需要使用的商用功能，如果你不想要复杂的前端界面，又想要支持的模型多一点，可以试试 uni-api。这是一个统一管理大模型 API 的项目，可以通过一个统一的API 接口调用多种不同提供商的服务，统一转换为 OpenAI 格式，支持负载均衡。目前支持的后端服务有：OpenAI、Anthropic、Gemini、Vertex、Azure、xai、Cohere、Groq、Cloudflare、OpenRouter 等。

## ✨ 特性

- 无前端，纯配置文件配置 API 渠道。只要写一个文件就能运行起一个属于自己的 API 站，文档有详细的配置指南，小白友好。
- 统一管理多个后端服务，支持 OpenAI、Deepseek、OpenRouter 等其他 API 是 OpenAI 格式的提供商。支持 OpenAI Dalle-3 图像生成。
- 同时支持 Anthropic、Gemini、Vertex AI、Azure、xai、Cohere、Groq、Cloudflare。Vertex 同时支持 Claude 和 Gemini API。
- 支持 OpenAI、 Anthropic、Gemini、Vertex、Azure、xai 原生 tool use 函数调用。
- 支持 OpenAI、Anthropic、Gemini、Vertex、Azure、xai 原生识图 API。
- 支持四种负载均衡。
  1. 支持渠道级加权负载均衡，可以根据不同的渠道权重分配请求。默认不开启，需要配置渠道权重。
  2. 支持 Vertex 区域级负载均衡，支持 Vertex 高并发，最高可将 Gemini，Claude 并发提高 （API数量 * 区域数量） 倍。自动开启不需要额外配置。
  3. 除了 Vertex 区域级负载均衡，所有 API 均支持渠道级顺序负载均衡，提高沉浸式翻译体验。默认不开启，需要配置 `SCHEDULING_ALGORITHM` 为 `round_robin`。
  4. 支持单个渠道多个 API Key 自动开启 API key 级别的轮训负载均衡。
- 支持自动重试，当一个 API 渠道响应失败时，自动重试下一个 API 渠道。
- 支持渠道冷却，当一个 API 渠道响应失败时，会自动将该渠道排除冷却一段时间，不再请求该渠道，冷却时间结束后，会自动将该模型恢复，直到再次请求失败，会重新冷却。
- 支持细粒度的模型超时时间设置，可以为每个模型设置不同的超时时间。
- 支持细粒度的权限控制。支持使用通配符设置 API key 可用渠道的特定模型。
- 支持限流，可以设置每分钟最多请求次数，可以设置为整数，如 2/min，2 次每分钟、5/hour，5 次每小时、10/day，10 次每天，10/month，10 次每月，10/year，10 次每年。默认60/min。
- 支持多个标准 OpenAI 格式的接口：`/v1/chat/completions`，`/v1/images/generations`，`/v1/audio/transcriptions`，`/v1/moderations`，`/v1/models`。
- 支持 OpenAI moderation 道德审查，可以对用户的消息进行道德审查，如果发现不当的消息，会返回错误信息。降低后台 API 被提供商封禁的风险。

## 使用方法

启动 uni-api 必须使用配置文件，有两种方式可以启动配置文件：

1. 第一种是使用 `CONFIG_URL` 环境变量填写配置文件 URL，uni-api启动时会自动下载。
2. 第二种就是挂载名为 `api.yaml` 的配置文件到容器内。

### 方法一：挂载 `api.yaml` 配置文件启动 uni-api

必须事先填写完成配置文件才能启动 `uni-api`，必须使用名为 `api.yaml` 的配置文件才能启动 `uni-api`，可以配置多个模型，每个模型可以配置多个后端服务，支持负载均衡。下面是最小可运行的 `api.yaml` 配置文件的示例：

```yaml
providers:
  - provider: provider_name # 服务提供商名称, 如 openai、anthropic、gemini、openrouter，随便取名字，必填
    base_url: https://api.your.com/v1/chat/completions # 后端服务的API地址，必填
    api: sk-YgS6GTi0b4bEabc4C # 提供商的API Key，必填，自动使用 base_url 和 api 通过 /v1/models 端点获取可用的所有模型。
  # 这里可以配置多个提供商，每个提供商可以配置多个 API Key，每个提供商可以配置多个模型。
api_keys:
  - api: sk-Pkj60Yf8JFWxfgRmXQFWyGtWUddGZnmi3KlvowmRWpWpQxx # API Key，用户请求 uni-api 需要 API key，必填
  # 该 API Key 可以使用所有模型，即可以使用 providers 下面设置的所有渠道里面的所有模型，不需要一个个添加可用渠道。
```

`api.yaml` 详细的高级配置：

```yaml
providers:
  - provider: provider_name # 服务提供商名称, 如 openai、anthropic、gemini、openrouter，随便取名字，必填
    base_url: https://api.your.com/v1/chat/completions # 后端服务的API地址，必填
    api: sk-YgS6GTi0b4bEabc4C # 提供商的API Key，必填
    model: # 选填，如果不配置 model，会自动通过 base_url 和 api 通过 /v1/models 端点获取可用的所有模型。
      - gpt-4o # 可以使用的模型名称，必填
      - claude-3-5-sonnet-20240620: claude-3-5-sonnet # 重命名模型，claude-3-5-sonnet-20240620 是服务商的模型名称，claude-3-5-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
      - dall-e-3

  - provider: anthropic
    base_url: https://api.anthropic.com/v1/messages
    api: # 支持多个 API Key，多个 key 自动开启轮训负载均衡，至少一个 key，必填
      - sk-ant-api03-bNnAOJyA-xQw_twAA
      - sk-ant-api02-bNnxxxx
    model:
      - claude-3-7-sonnet-20240620: claude-3-7-sonnet # 重命名模型，claude-3-7-sonnet-20240620 是服务商的模型名称，claude-3-7-sonnet 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，选填
      - claude-3-7-sonnet-20250219: claude-3-7-sonnet-think # 重命名模型，claude-3-7-sonnet-20250219 是服务商的模型名称，claude-3-7-sonnet-think 是重命名后的名字，可以使用简洁的名字代替原来复杂的名称，如果重命名后的名字里面有think，则自动转换为 claude 思考模型，默认思考 token 限制为 4096。选填
    tools: true # 是否支持工具，如生成代码、生成文档等，默认是 true，选填

  - provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta # base_url 支持 v1beta/v1, 仅供 Gemini 模型使用，必填
    api: # 支持多个 API Key，多个 key 自动开启轮训负载均衡，至少一个 key，必填
      - AIzaSyAN2k6IRdgw123
      - AIzaSyAN2k6IRdgw456
      - AIzaSyAN2k6IRdgw789
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash-exp-0827: gemini-1.5-flash # 重命名后，原来的模型名字 gemini-1.5-flash-exp-0827 无法使用，如果要使用原来的名字，可以在 model 中添加原来的名字，只要加上下面一行就可以使用原来的名字了
      - gemini-1.5-flash-exp-0827 # 加上这一行，gemini-1.5-flash-exp-0827 和 gemini-1.5-flash 都可以被请求
      - gemini-1.5-pro: gemini-1.5-pro-search # 支持以 -search 后缀重命名模型启用搜索，使用 gemini-1.5-pro-search 模型请求 uni-api 时，表示 gemini-1.5-pro 模型自动使用 Google 官方搜索工具，支持全部 1.5/2.0 系列模型。
    tools: true
    preferences:
      api_key_rate_limit: 15/min # 每个 API Key 每分钟最多请求次数，选填。默认为 999999/min。支持多个频率约束条件：15/min,10/day
      # api_key_rate_limit: # 可以为每个模型设置不同的频率限制
      #   gemini-1.5-flash: 15/min,1500/day
      #   gemini-1.5-pro: 2/min,50/day
      #   default: 4/min # 如果模型没有设置频率限制，使用 default 的频率限制
      api_key_cooldown_period: 60 # 每个 API Key 遭遇 429 错误后的冷却时间，单位为秒，选填。默认为 0 秒, 当设置为 0 秒时，不启用冷却机制。当存在多个 API key 时才会生效。
      api_key_schedule_algorithm: round_robin # 设置多个 API Key 的请求顺序，选填。默认为 round_robin，可选值有：round_robin，random，fixed_priority。当存在多个 API key 时才会生效。round_robin 是轮询负载均衡，random 是随机负载均衡，fixed_priority 是固定优先级调度，永远使用第一个可用的 API key。
      model_timeout: # 模型超时时间，单位为秒，默认 100 秒，选填
        gemini-1.5-pro: 10 # 模型 gemini-1.5-pro 的超时时间为 10 秒
        gemini-1.5-flash: 10 # 模型 gemini-1.5-flash 的超时时间为 10 秒
        default: 10 # 模型没有设置超时时间，使用默认的超时时间 10 秒，当请求的不在 model_timeout 里面的模型时，超时时间默认是 10 秒，不设置 default，uni-api 会使用全局配置的模型超时时间。
      proxy: socks5://[用户名]:[密码]@[IP地址]:[端口] # 代理地址，选填。支持 socks5 和 http 代理，默认不使用代理。
      headers:  # 额外附加自定义HTTP请求头，选填。
        Custom-Header-1: Value-1
        Custom-Header-2: Value-2

  - provider: vertex
    project_id: gen-lang-client-xxxxxxxxxxxxxx #    描述： 您的Google Cloud项目ID。格式： 字符串，通常由小写字母、数字和连字符组成。获取方式： 在Google Cloud Console的项目选择器中可以找到您的项目ID。
    private_key: "-----BEGIN PRIVATE KEY-----\nxxxxx\n-----END PRIVATE" # 描述： Google Cloud Vertex AI服务账号的私钥。格式： 一个 JSON 格式的字符串，包含服务账号的私钥信息。获取方式： 在 Google Cloud Console 中创建服务账号，生成JSON格式的密钥文件，然后将其内容设置为此环境变量的值。
    client_email: xxxxxxxxxx@xxxxxxx.gserviceaccount.com # 描述： Google Cloud Vertex AI 服务账号的电子邮件地址。格式： 通常是形如 "service-account-name@project-id.iam.gserviceaccount.com" 的字符串。获取方式： 在创建服务账号时生成，也可以在 Google Cloud Console 的"IAM与管理"部分查看服务账号详情获得。
    model:
      - gemini-1.5-pro
      - gemini-1.5-flash
      - gemini-1.5-pro: gemini-1.5-pro-search # 仅支持在 vertex Gemini API 中，以 -search 后缀重命名模型后，使用 gemini-1.5-pro-search 模型请求 uni-api 时，表示 gemini-1.5-pro 模型自动使用 Google 官方搜索工具。
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
    engine: openrouter # 强制使用某个消息格式，目前支持 gpt，claude，gemini，openrouter 原生格式，选填

api_keys:
  - api: sk-KjjI60Yf0JFWxfgRmXqFWyGtWUd9GZnmi3KlvowmRWpWpQRo # API Key，用户使用本服务需要 API key，必填
    model: # 该 API Key 可以使用的模型，必填。默认开启渠道级轮询负载均衡，每次请求模型按照 model 配置的顺序依次请求。与 providers 里面原始的渠道顺序无关。因此你可以设置每个 API key 请求顺序不一样。
      - gpt-4o # 可以使用的模型名称，可以使用所有提供商提供的 gpt-4o 模型
      - claude-3-5-sonnet # 可以使用的模型名称，可以使用所有提供商提供的 claude-3-5-sonnet 模型
      - gemini/* # 可以使用的模型名称，仅可以使用名为 gemini 提供商提供的所有模型，其中 gemini 是 provider 名称，* 代表所有模型
    role: admin # 设置 API key 的别名，选填。请求日志会显示该 API key 的别名。如果 role 为 admin，则仅有此 API key 可以请求 v1/stats,/v1/generate-api-key 端点。如果所有 API key 都没有设置 role 为 admin，则默认第一个 API key 为 admin 拥有请求 v1/stats,/v1/generate-api-key 端点的权限。

  - api: sk-pkhf60Yf0JGyJxgRmXqFQyTgWUd9GZnmi3KlvowmRWpWqrhy
    model:
      - anthropic/claude-3-5-sonnet # 可以使用的模型名称，仅可以使用名为 anthropic 提供商提供的 claude-3-5-sonnet 模型。其他提供商的 claude-3-5-sonnet 模型不可以使用。这种写法不会匹配到other-provider提供的名为anthropic/claude-3-5-sonnet的模型。
      - <anthropic/claude-3-5-sonnet> # 通过在模型名两侧加上尖括号，这样就不会去名为anthropic的渠道下去寻找claude-3-5-sonnet模型，而是将整个 anthropic/claude-3-5-sonnet 作为模型名称。这种写法可以匹配到other-provider提供的名为 anthropic/claude-3-5-sonnet 的模型。但不会匹配到anthropic下面的claude-3-5-sonnet模型。
      - openai-test/text-moderation-latest # 当开启消息道德审查后，可以使用名为 openai-test 渠道下的 text-moderation-latest 模型进行道德审查。
      - sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo/* # 支持将其他 api key 当作渠道
    preferences:
      SCHEDULING_ALGORITHM: fixed_priority # 当 SCHEDULING_ALGORITHM 为 fixed_priority 时，使用固定优先级调度，永远执行第一个拥有请求的模型的渠道。默认开启，SCHEDULING_ALGORITHM 缺省值为 fixed_priority。SCHEDULING_ALGORITHM 可选值有：fixed_priority，round_robin，weighted_round_robin, lottery, random。
      # 当 SCHEDULING_ALGORITHM 为 random 时，使用随机轮训负载均衡，随机请求拥有请求的模型的渠道。
      # 当 SCHEDULING_ALGORITHM 为 round_robin 时，使用轮训负载均衡，按照顺序请求用户使用的模型的渠道。
      AUTO_RETRY: true # 是否自动重试，自动重试下一个提供商，true 为自动重试，false 为不自动重试，默认为 true。也可以设置为数字，表示重试次数。
      rate_limit: 15/min # 支持限流，每分钟最多请求次数，可以设置为整数，如 2/min，2 次每分钟、5/hour，5 次每小时、10/day，10 次每天，10/month，10 次每月，10/year，10 次每年。默认999999/min，选填。支持多个频率约束条件：15/min,10/day
      # rate_limit: # 可以为每个模型设置不同的频率限制
      #   gemini-1.5-flash: 15/min,1500/day
      #   gemini-1.5-pro: 2/min,50/day
      #   default: 4/min # 如果模型没有设置频率限制，使用 default 的频率限制
      ENABLE_MODERATION: true # 是否开启消息道德审查，true 为开启，false 为不开启，默认为 false，当开启后，会对用户的消息进行道德审查，如果发现不当的消息，会返回错误信息。

  # 渠道级加权负载均衡配置示例
  - api: sk-KjjI60Yd0JFWtxxxxxxxxxxxxxxwmRWpWpQRo
    model:
      - gcp1/*: 5 # 冒号后面就是权重，权重仅支持正整数。
      - gcp2/*: 3 # 数字的大小代表权重，数字越大，请求的概率越大。
      - gcp3/*: 2 # 在该示例中，所有渠道加起来一共有 10 个权重，及 10 个请求里面有 5 个请求会请求 gcp1/* 模型，2 个请求会请求 gcp2/* 模型，3 个请求会请求 gcp3/* 模型。

    preferences:
      SCHEDULING_ALGORITHM: weighted_round_robin # 仅当 SCHEDULING_ALGORITHM 为 weighted_round_robin 并且上面的渠道如果有权重，会按照加权后的顺序请求。使用加权轮训负载均衡，按照权重顺序请求拥有请求的模型的渠道。当 SCHEDULING_ALGORITHM 为 lottery 时，使用抽奖轮训负载均衡，按照权重随机请求拥有请求的模型的渠道。没设置权重的渠道自动回退到 round_robin 轮训负载均衡。
      AUTO_RETRY: true

preferences: # 全局配置
  model_timeout: # 模型超时时间，单位为秒，默认 100 秒，选填
    gpt-4o: 10 # 模型 gpt-4o 的超时时间为 10 秒,gpt-4o 是模型名称，当请求 gpt-4o-2024-08-06 等模型时，超时时间也是 10 秒
    claude-3-5-sonnet: 10 # 模型 claude-3-5-sonnet 的超时时间为 10 秒，当请求 claude-3-5-sonnet-20240620 等模型时，超时时间也是 10 秒
    default: 10 # 模型没有设置超时时间，使用默认的超时时间 10 秒，当请求的不在 model_timeout 里面的模型时，超时时间默认是 10 秒，不设置 default，uni-api 会使用 环境变量 TIMEOUT 设置的默认超时时间，默认超时时间是 100 秒
    o1-mini: 30 # 模型 o1-mini 的超时时间为 30 秒，当请求名字是 o1-mini 开头的模型时，超时时间是 30 秒
    o1-preview: 100 # 模型 o1-preview 的超时时间为 100 秒，当请求名字是 o1-preview 开头的模型时，超时时间是 100 秒
  cooldown_period: 300 # 渠道冷却时间，单位为秒，默认 300 秒，选填。当模型请求失败时，会自动将该渠道排除冷却一段时间，不再请求该渠道，冷却时间结束后，会自动将该模型恢复，直到再次请求失败，会重新冷却。当 cooldown_period 设置为 0 时，不启用冷却机制。
  error_triggers: # 错误触发器，当模型返回的消息包含错误触发器中的任意一个字符串时，该渠道会自动返回报错。选填
    - The bot's usage is covered by the developer
    - process this request due to overload or policy
  proxy: socks5://[username]:[password]@[ip]:[port] # 全局代理地址，选填。
```

挂载配置文件并启动 uni-api docker 容器：

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-v ./api.yaml:/home/api.yaml \
yym68686/uni-api:latest
```

### 方法二：使用 `CONFIG_URL` 环境变量启动 uni-api

按照方法一写完配置文件后，上传到云端硬盘，获取文件的直链，然后使用 `CONFIG_URL` 环境变量启动 uni-api docker 容器：

```bash
docker run --user root -p 8001:8000 --name uni-api -dit \
-e CONFIG_URL=http://file_url/api.yaml \
yym68686/uni-api:latest
```

## 环境变量

- CONFIG_URL: 配置文件的下载地址，可以是本地文件，也可以是远程文件，选填
- TIMEOUT: 请求超时时间，默认为 100 秒，超时时间可以控制当一个渠道没有响应时，切换下一个渠道需要的时间。选填
- DISABLE_DATABASE: 是否禁用数据库，默认为 false，选填

## Vercel 部署

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyym68686%2Funi-api%2Ftree%2Fmain&env=CONFIG_URL,DISABLE_DATABASE&project-name=uni-api-vercel&repository-name=uni-api-vercel)

点击上面的一键部署按钮后，设置环境变量 `CONFIG_URL` 为配置文件的直链， `DISABLE_DATABASE` 为 true，然后点击 Create 创建项目。部署完之后需要手动在 vercel 项目面板的 Settings -> Funcitons -> Function Max Duration 设置为 60 秒，然后点击 Deployments 菜单点击 Redeploy 重新部署，即可将超时时间设置为 60 秒，如果不重新部署，默认超时时间将是原来的 10 秒。注意不是删掉 vercel 项目重建，而是在当前部署好的 vercel 项目里面的 Deployments 菜单里面点 redeploy，这样才能让 Function Max Duration 的修改生效。

## Ubuntu 部署

在仓库 Releases 找到对应的二进制文件最新版本，例如名为 uni-api-linux-x86_64-0.0.99.pex 的文件。在服务器下载二进制文件并运行：

```bash
wget https://github.com/yym68686/uni-api/releases/download/v0.0.99/uni-api-linux-x86_64-0.0.99.pex
chmod +x uni-api-linux-x86_64-0.0.99.pex
./uni-api-linux-x86_64-0.0.99.pex
```

## serv00 远程部署（FreeBSD 14.0）

首先登录面板，Additional services 里面点击选项卡 Run your own applications 开启允许运行自己的程序，然后到面板 Port reservation 去随便开一个端口。

如果没有自己的域名，去面板 WWW websites 删掉默认给的域名，再新建一个域名 Domain 为刚才删掉的域名，点击 Advanced settings 后设置 Website type 为 Proxy 域名，Proxy port 指向你刚才开的端口，不要选中 Use HTTPS。

ssh 登陆到 serv00 服务器，执行下面的命令：

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

ctrl+b d 退出 tmux 等待几个小时安装完成，安装完成后执行下面的命令：

```bash
tmux new -A -s uni-api
source uni-api/bin/activate
export CONFIG_URL=http://file_url/api.yaml
export DISABLE_DATABASE=true
# 修改端口，xxx 为端口，自行修改，对应刚刚在面板 Port reservation 开的端口
sed -i '' 's/port=8000/port=xxx/' main.py
sed -i '' 's/reload=True/reload=False/' main.py
python main.py
```

使用 ctrl+b d 退出 tmux，即可让程序后台运行。此时就可以在其他聊天客户端使用 uni-api 了。curl 测试脚本：

```bash
curl -X POST https://xxx.serv00.net/v1/chat/completions \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-xxx' \
-d '{"model": "gpt-4o","messages": [{"role": "user","content": "你好"}]}'
```

参考文档：

https://docs.serv00.com/Python/

https://linux.do/t/topic/201181

https://linux.do/t/topic/218738

## Docker 本地部署

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

pex linux 打包：

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

macos 打包：

```bash
VERSION=$(cat VERSION)
pex -r requirements.txt \
    -c uvicorn \
    --inject-args 'main:app --host 0.0.0.0 --port 8000' \
    -o uni-api-macos-arm64-${VERSION}.pex
```

## 赞助商

我们感谢以下赞助商的支持：
<!-- ¥2050 -->
- @PowerHunter：¥2000
- @ioi：¥50

## 如何赞助我们

如果您想支持我们的项目，您可以通过以下方式赞助我们：

1. [PayPal](https://www.paypal.me/yym68686)

2. [USDT-TRC20](https://pb.yym68686.top/~USDT-TRC20)，USDT-TRC20 钱包地址：`TLFbqSv5pDu5he43mVmK1dNx7yBMFeN7d8`

3. [微信](https://pb.yym68686.top/~wechat)

4. [支付宝](https://pb.yym68686.top/~alipay)

感谢您的支持！

## 常见问题

- 为什么总是出现 `Error processing request or performing moral check: 404: No matching model found` 错误？

将 ENABLE_MODERATION 设置为 false 将修复这个问题。当 ENABLE_MODERATION 为 true 时，API 必须能够使用 text-moderation-latest 模型，如果你没有在提供商模型设置里面提供 text-moderation-latest，将会报错找不到模型。

- 怎么优先请求某个渠道，怎么设置渠道的优先级？

直接在api_keys里面通过设置渠道顺序即可。不需要做其他设置，示例配置文件：

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

这样设置则先请求 ai2，失败后请求 ai1。

- 各种调度算法背后的行为是怎样的？比如 fixed_priority，weighted_round_robin，lottery，random，round_robin？

所有调度算法需要通过在配置文件的 api_keys.(api).preferences.SCHEDULING_ALGORITHM 设置为 fixed_priority，weighted_round_robin，lottery，random，round_robin 中的任意值来开启。

1. fixed_priority：固定优先级调度。所有请求永远执行第一个拥有用户请求的模型的渠道。报错时，会切换下一个渠道。这是默认的调度算法。

2. weighted_round_robin：加权轮训负载均衡，按照配置文件 api_keys.(api).model 设定的权重顺序请求拥有用户请求的模型的渠道。

3. lottery：抽奖轮训负载均衡，按照配置文件 api_keys.(api).model 设置的权重随机请求拥有用户请求的模型的渠道。

4. round_robin：轮训负载均衡，按照配置文件 api_keys.(api).model 的配置顺序请求拥有用户请求的模型的渠道。可以查看上一个问题，如何设置渠道的优先级。

- 应该怎么正确填写 base_url？

除了高级配置里面所展示的一些特殊的渠道，所有 OpenAI 格式的提供商需要把 base_url 填完整，也就是说 base_url 必须以 /v1/chat/completions 结尾。如果你使用的 GitHub models，base_url 应该填写为 https://models.inference.ai.azure.com/chat/completions，而不是 Azure 的 URL。

对于 Azure 渠道，base_url 兼容以下几种写法：https://your-endpoint.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview 和 https://your-endpoint.services.ai.azure.com/models/chat/completions，https://your-endpoint.openai.azure.com，推荐使用第一种写法。如果不显式指定 api-version，默认使用 2024-10-21 版本。

- 模型超时时间是如何确认的？渠道级别的超时设置和全局模型超时设置的优先级是什么？

渠道级别的超时设置优先级高于全局模型超时设置。优先级顺序：渠道级别模型超时设置 > 渠道级别默认超时设置 > 全局模型超时设置 > 全局默认超时设置 > 环境变量 TIMEOUT。

通过调整模型超时时间，可以避免出现某些渠道请求超时报错的情况。如果你遇到 `{'error': '500', 'details': 'fetch_response_stream Read Response Timeout'}` 错误，请尝试增加模型超时时间。

- api_key_rate_limit 是怎么工作的？我如何给多个模型设置相同的频率限制？

如果你想同时给 gemini-1.5-pro-latest，gemini-1.5-pro，gemini-1.5-pro-001，gemini-1.5-pro-002 这四个模型设置相同的频率限制，可以这样设置：

```yaml
api_key_rate_limit:
  gemini-1.5-pro: 1000/min
```

这会匹配所有含有 gemini-1.5-pro 字符串的模型。gemini-1.5-pro-latest，gemini-1.5-pro，gemini-1.5-pro-001，gemini-1.5-pro-002 这四个模型频率限制都会设置为 1000/min。api_key_rate_limit 字段配置的逻辑如下，这是一个示例配置文件：

```yaml
api_key_rate_limit:
  gemini-1.5-pro: 1000/min
  gemini-1.5-pro-002: 500/min
```

此时如果有一个使用模型 gemini-1.5-pro-002 的请求。

首先，uni-api 会尝试精确匹配 api_key_rate_limit 的模型。如果刚好设置了 gemini-1.5-pro-002 的频率限制，则 gemini-1.5-pro-002 的频率限制则为 500/min，如果此时请求的模型不是 gemini-1.5-pro-002，而是 gemini-1.5-pro-latest，由于 api_key_rate_limit 没有设置 gemini-1.5-pro-latest 的频率限制，因此会寻找有没有前缀和 gemini-1.5-pro-latest 相同的模型被设置了，因此 gemini-1.5-pro-latest 的频率限制会被设置为 1000/min。

## ⭐ Star 历史

<a href="https://github.com/yym68686/uni-api/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=yym68686/uni-api&type=Date">
</a>