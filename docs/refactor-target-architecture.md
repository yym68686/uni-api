# uni-api 最终态重构设计

本文档描述 uni-api 完成系统性重构后的目标形态。内容只定义最终优化效果、模块边界、质量约束和可执行 TODO，不描述临时过渡状态。

## 目标

- 性能：请求热路径只执行必要计算，配置、路由、模型权限、权重、端点排除、Provider 能力均在运行时索引中 O(1) 或接近 O(1) 查询。
- 冗余：请求构造、响应解析、SSE 处理、流清理、错误归一化、统计埋点都只有一个权威实现。
- 灵活性：新增 Provider、Endpoint、模型特例、请求体覆盖规则时，只需要新增或扩展适配器，不修改核心路由和请求执行器。
- 鲁棒性：并发、取消、断流、重试、限流、冷却、认证刷新、数据库不可用、配置错误都有明确状态机和可测试行为。
- 可测试性：单元测试可离线稳定运行；集成测试按 marker 显式启用；核心行为有 fixture/golden case 覆盖。

## 最终架构

```text
uni_api/
  app.py
  app_state.py
  config/
    loader.py
    schema.py
    compiler.py
    runtime.py
  auth/
    api_keys.py
    codex_oauth.py
  routing/
    index.py
    planner.py
    scheduler.py
  providers/
    base.py
    registry.py
    openai.py
    codex.py
    gemini.py
    vertex_gemini.py
    anthropic.py
    vertex_anthropic.py
    aws_bedrock.py
    azure.py
    openrouter.py
    cloudflare.py
    cohere.py
    image.py
    audio.py
    search.py
    video/
      base.py
      registry.py
      deyunai.py
      callxyq.py
      lingjing.py
  upstream/
    client_pool.py
    runner.py
    errors.py
    retry_policy.py
  streaming/
    sse.py
    cleanup.py
    responses_events.py
    chat_completion_events.py
  rate_limit/
    policy.py
    state.py
    key_pool.py
  observability/
    request_context.py
    middleware.py
    telemetry.py
    stats.py
  api/
    chat.py
    responses.py
    messages.py
    images.py
    audio.py
    embeddings.py
    moderations.py
    models.py
    admin.py
    health.py
  persistence/
    models.py
    sessions.py
    repositories.py
  tests/
    unit/
    integration/
    fixtures/
```

## 核心设计

### 应用入口

`app.py` 只负责创建 FastAPI 应用、注册中间件、注册路由、注册 lifespan。它不包含业务逻辑、Provider 逻辑、数据库查询、流处理细节或配置变换。

`app_state.py` 暴露单一 `AppState` 对象，集中持有 `RuntimeConfig`、`ClientPool`、`ProviderRegistry`、`RateLimitStore`、`StatsRepository`、`ObservabilityClient`。所有 handler 通过依赖注入获取这些对象，不直接读写散落的 `app.state.*` 字段。

### 配置编译

配置加载分三层：

- `loader.py`：读取本地 YAML 或远程 `CONFIG_URL`，只负责 I/O。
- `schema.py`：用 Pydantic 模型校验配置结构、默认值、字段类型和枚举值。
- `compiler.py`：把配置编译为不可变 `RuntimeConfig`。

`RuntimeConfig` 包含以下索引：

- `api_key_by_token`
- `api_key_index_by_token`
- `provider_by_name`
- `models_by_provider`
- `providers_by_model`
- `api_key_allowed_models`
- `api_key_model_response_cache`
- `endpoint_exclusions_by_provider`
- `weights_by_api_key_model`
- `provider_preferences`
- `model_timeout_index`
- `keepalive_interval_index`
- `video_provider_configs`

运行时请求不再扫描原始 YAML 结构。

### 校验边界

Pydantic 只用于外部输入边界和冷路径，不用于请求内部热路径。

允许使用 Pydantic 的位置：

- 配置加载和配置更新：`api.yaml`、远程配置、管理接口提交的配置。
- FastAPI 路由入口：需要严格结构化的公开 API 请求。
- 测试 fixture 校验：确保 golden case 结构正确。

不允许使用 Pydantic 的位置：

- `RoutingPlanner` 内部。
- `RoutingPlan` 和 provider attempt 流转。
- Provider retry 循环。
- 限流、冷却、key pool 状态更新。
- SSE/stream chunk 解析。
- 已经由 FastAPI 校验过的请求对象再次完整 `model_validate`。

内部热路径使用 dataclass、TypedDict、不可变普通对象或经过编译的 dict 索引。大型 payload、图片、音频、长 messages、Responses input 只在边界做必要校验；进入 Provider 适配器后按字段访问和最小转换处理。

所有二次校验都必须有明确理由。业务中间件不能为了统计或限流而完整构造第二套 Pydantic 请求模型。需要读取模型名、stream、最后一条文本等少量字段时，直接从已解析的 raw dict 或已校验对象中提取。

### Provider 适配器

所有 Provider 统一实现 `ProviderAdapter`：

```python
class ProviderAdapter(Protocol):
    name: str
    supported_engines: set[str]

    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        ...

    async def parse_response(self, context: ProviderResponseContext) -> NormalizedResponse:
        ...

    async def parse_stream(self, context: ProviderStreamContext) -> AsyncIterator[bytes | str]:
        ...
```

核心执行器只认识：

- `UpstreamRequest`
- `NormalizedResponse`
- `ProviderError`
- `ProviderAdapter`

模型别名、请求体字段删除、请求体覆盖、工具调用转换、图片/音频格式转换、Provider 私有 header、Provider 私有 URL 规则都属于适配器职责。

新增 Provider 时，不允许修改核心执行器、路由计划器或全局响应解析函数。

### 路由计划

`RoutingPlanner` 输入：

- 请求模型名
- API key 身份
- Endpoint
- 估算 token 数
- 运行时配置索引
- 当前限流/冷却状态

输出不可变 `RoutingPlan`：

- ordered provider attempts
- selected original model
- selected endpoint behavior
- timeout
- keepalive interval
- retry budget
- retry policy
- observable plan metadata

`RoutingPlan` 不直接执行网络请求，只描述可执行计划。

### 调度和限流

调度策略独立于 key 池：

- `fixed_priority`
- `round_robin`
- `weighted_round_robin`
- `lottery`
- `smart_round_robin`

`ProviderKeyPool` 只管理 key 列表和当前游标。`RateLimitState` 管理滑动窗口、冷却和 TPR。`RateLimitPolicy` 管理解析后的规则。

所有限流窗口使用 deque，清理过期窗口后再判断。所有可变状态都有单一锁边界，不允许一个公开方法在持锁时调用另一个会获取同一把锁的公开方法。

### Upstream 执行

`UpstreamRunner` 是唯一重试状态机。它负责：

- 执行 `RoutingPlan`
- 请求前选择 provider key
- 统一捕获 Provider/HTTP/网络异常
- 判断是否重试
- 判断是否冷却 key
- 判断是否排除 channel
- 执行 rollback
- 输出最终响应或错误响应

业务 handler 不再手写重试循环。

### HTTP 客户端池

`ClientPool` 按以下维度复用 `httpx.AsyncClient`：

- scheme
- host
- port
- proxy
- http2 flag
- TLS verify
- redirect policy

每个 client key 有独立创建锁。client 生命周期由 lifespan 管理。健康指标暴露 client 数、连接池容量、最近 sweep 时间、最近 sweep 错误、关闭连接数。

### 流式处理

`streaming/` 是唯一 SSE 和流清理实现：

- `sse.py`：增量 SSE parser，支持 split chunk、comment frame、named event、`[DONE]`。
- `cleanup.py`：异步生成器关闭、response 关闭、stream context 关闭、取消保护、后台 cleanup 追踪。
- `responses_events.py`：Responses API 事件模型和 commit policy。
- `chat_completion_events.py`：Chat Completions chunk 构造。

所有流式 handler 都通过统一工具处理下游断开、上游关闭、首字节观测、keepalive、已提交输出后的错误策略。

### 观测和统计

`RequestContext` 是类型化对象，不再使用无结构 dict 传递请求状态。

包含字段：

- request_id
- trace_id
- span_id
- api_key
- endpoint
- model
- provider
- upstream_model
- stream
- retry_count
- cooldown_count
- status_code
- error_type
- token usage
- timing spans
- client ip

统计写入通过 repository 层完成。业务路径只提交结构化事件，不直接拼 SQL 或依赖 ORM 细节。

### 数据库

数据库层拆分为：

- `persistence/models.py`：SQLAlchemy model。
- `persistence/sessions.py`：engine/session 初始化。
- `persistence/repositories.py`：RequestStat/ChannelStat 查询和写入。

所有统计查询都在 repository 层命名化，handler 不直接构造复杂 SQL。

### API 路由

每个路由文件只做：

- 请求模型校验
- 依赖注入
- 调用 handler/use case
- 返回 Response

路由文件不包含 Provider 分支、网络请求、重试、数据库聚合或复杂流转换。

### 测试体系

测试分层：

- `tests/unit`：默认运行，离线、无外部凭证、无真实网络。
- `tests/integration`：需要 marker 和环境变量显式启用。
- `tests/fixtures`：Provider 请求/响应样例、SSE 样例、配置样例。

默认命令：

```bash
uv run pytest -q tests/unit
```

集成测试命令：

```bash
uv run pytest -q tests/integration --run-integration
```

## 性能目标

- `/v1/models` 对同一 API key 直接返回预编译缓存，不扫描 providers。
- 普通 chat/responses 请求路由选择不扫描完整配置。
- Provider 匹配基于 `providers_by_model` 和 endpoint exclusion set。
- 权重调度结果按规范化权重元组缓存。
- 限流判断只访问当前 key/model 的窗口。
- 非流响应 JSON 编解码只做一次必要转换。
- SSE 解析按 chunk 增量处理，不拼接全量流。
- 大型配置刷新发生在显式 reload/update，不在请求热路径隐式重建。
- Pydantic 校验只发生在配置加载、配置更新和 API 边界，不发生在内部路由、重试、限流和流式事件循环。
- 已经被 FastAPI 校验过的请求不再在中间件或 handler 中完整二次 `model_validate`。
- Stream chunk 和 SSE event 不构造 Pydantic 模型，只做增量解析和必要字段读取。

## 鲁棒性目标

- 下游断开时，上游 response、stream context、async iterator 必须关闭。
- 上游首包为空、语义错误、结构错误、网络错误分别归一化为明确错误类型。
- 输出已经提交给下游后，不再重试到下一个 Provider。
- 输出未提交前，可以按 retry policy 重试。
- 401/403 可以清除 Codex OAuth cache。
- 429 和可识别 quota/rate-limit 错误按 Provider/key 配置冷却。
- 数据库不可用或禁用时，请求主路径不失败。
- 远程配置加载失败时返回明确启动/健康状态，不产生半初始化运行时。
- 所有后台任务在 lifespan shutdown 时可等待或可观测。

## 可勾选 TODO

### 项目结构

- [x] 创建 `uni_api/` 包目录。
- [x] 创建 `uni_api/app.py`，只保留应用创建和注册逻辑。
- [x] 创建 `uni_api/app_state.py`，定义类型化 `AppState`。
- [x] 将根目录 `main.py` 变成薄入口或兼容导入层。
- [x] 将 `db.py` 拆入 `uni_api/persistence/`。
- [x] 将 `routing.py` 拆入 `uni_api/routing/`。
- [x] 将 `utils.py` 中配置、错误包装、模型列表、统计查询拆到对应包。
- [x] 将 `core/request.py` 中 Provider payload 构造拆到 `uni_api/providers/`。
- [x] 将 `core/response.py` 中 Provider response/stream 解析拆到 `uni_api/providers/` 和 `uni_api/streaming/`。
- [x] 删除业务代码对旧全局工具模块的新增依赖。

### 配置和运行时索引

- [x] 定义 `ConfigSchema`，覆盖 providers、api_keys、preferences、video_providers。
- [x] 定义 `RuntimeConfig`，所有字段不可变或只读。
- [x] 实现本地 YAML loader。
- [x] 实现远程 `CONFIG_URL` loader。
- [x] 实现环境变量展开规则。
- [x] 实现 provider model alias 编译。
- [x] 实现 api key 权限规则编译。
- [x] 实现 nested api key model 依赖解析。
- [x] 实现递归 api key 依赖检测。
- [x] 实现 endpoint exclusion set 编译。
- [x] 实现 model timeout index。
- [x] 实现 keepalive interval index。
- [x] 实现 weights index。
- [x] 实现 `/v1/models` response cache。
- [x] 实现配置 reload/update 后一次性替换 `RuntimeConfig`。
- [x] 为配置编译添加空配置测试。
- [x] 为配置编译添加 provider 缺失字段测试。
- [x] 为配置编译添加模型别名测试。
- [x] 为配置编译添加 wildcard 模型测试。
- [x] 为配置编译添加 nested api key 测试。
- [x] 为配置编译添加 endpoint exclusion 测试。
- [x] 为配置编译添加 weights 展开测试。

### 校验性能边界

- [x] 定义允许使用 Pydantic 的模块清单。
- [x] 定义禁止在热路径使用 Pydantic 的模块清单。
- [x] 确认配置加载只校验一次。
- [x] 确认配置更新只校验一次。
- [x] 确认 `RuntimeConfig` 是校验后的编译产物。
- [x] 移除中间件里的完整二次请求 `model_validate`。
- [x] 移除 Provider retry 循环中的完整请求模型重建。
- [x] 移除 stream chunk 到 Pydantic 模型的转换。
- [x] 将 `RoutingPlan` 改为 dataclass 或轻量不可变对象。
- [x] 将 Provider attempt 改为 dataclass 或轻量不可变对象。
- [x] 将 request context 改为 dataclass 或轻量可变对象。
- [x] 为大型 messages payload 添加校验开销 benchmark。
- [x] 为 Responses input payload 添加校验开销 benchmark。
- [x] 为中间件模型名提取添加 raw dict 轻量路径测试。
- [x] 为 FastAPI 已校验对象复用添加测试。
- [x] 为配置 Pydantic 校验失败添加错误消息测试。
- [x] 为热路径禁止二次校验添加回归测试或静态扫描检查。

### Provider 适配器

- [x] 定义 `ProviderAdapter` 协议。
- [x] 定义 `ProviderRequestContext`。
- [x] 定义 `ProviderResponseContext`。
- [x] 定义 `ProviderStreamContext`。
- [x] 定义 `UpstreamRequest`。
- [x] 定义 `NormalizedResponse`。
- [x] 定义 `ProviderError`。
- [x] 实现 `ProviderRegistry`。
- [x] 实现 OpenAI/GPT adapter。
- [x] 实现 Codex adapter。
- [x] 实现 Gemini adapter。
- [x] 实现 Vertex Gemini adapter。
- [x] 实现 Anthropic adapter。
- [x] 实现 Vertex Anthropic adapter。
- [x] 实现 AWS Bedrock adapter。
- [x] 实现 Azure adapter。
- [x] 实现 Azure Databricks adapter。
- [x] 实现 OpenRouter adapter。
- [x] 实现 Cloudflare adapter。
- [x] 实现 Cohere adapter。
- [x] 实现 image generation/edit adapter。
- [x] 实现 audio speech/transcription adapter。
- [x] 实现 moderation adapter。
- [x] 实现 embedding adapter。
- [x] 实现 search adapter。
- [x] 合并现有 video adapter 到新 registry。
- [x] 为每个 adapter 添加 request golden tests。
- [x] 为每个 adapter 添加 non-stream response golden tests。
- [x] 为每个支持流的 adapter 添加 stream golden tests。
- [x] 确认新增 Provider 不需要修改 `UpstreamRunner`。
- [x] 确认新增 Provider 不需要修改 `RoutingPlanner`。

### 请求体和响应归一化

- [x] 将 `post_body_parameter_overrides` 抽为独立策略。
- [x] 覆盖全局字段删除测试。
- [x] 覆盖模型级字段删除测试。
- [x] 覆盖深合并测试。
- [x] 覆盖 provider config 不被 mutation 测试。
- [x] 统一 OpenAI usage 构造。
- [x] 统一 Responses API usage 到 Chat Completions usage 转换。
- [x] 统一 tool calls 构造。
- [x] 统一 reasoning content 字段处理。
- [x] 统一 image base64/cache thought signature 处理。
- [x] 统一 audio inline data 到 OpenAI audio 对象处理。
- [x] 删除重复 JSON dumps/loads。
- [x] 删除 Provider 适配器外的模型特例分支。

### 路由和调度

- [x] 实现 `RoutingPlanner`。
- [x] 实现 `RoutingPlan` 不可变数据结构。
- [x] 实现 provider candidate 查询。
- [x] 实现 endpoint exclusion 过滤。
- [x] 实现 TPR 过滤。
- [x] 实现 channel cooldown 过滤。
- [x] 实现 fixed priority 调度。
- [x] 实现 round robin 调度。
- [x] 实现 weighted round robin 调度。
- [x] 实现 lottery 调度。
- [x] 实现 smart round robin 调度。
- [x] 实现 retry budget 计算。
- [x] 实现 provider key selection 只在 attempt 执行前发生。
- [x] 覆盖无 provider 的 404 行为测试。
- [x] 覆盖所有 provider 被限流的 429 行为测试。
- [x] 覆盖 request 超过 TPR 的 413 行为测试。
- [x] 覆盖权重和 endpoint exclusion 组合测试。
- [x] 覆盖 nested api key 路由测试。

### 限流、冷却和 key 池

- [x] 定义 `RateLimitPolicy`。
- [x] 定义 `RateLimitState`。
- [x] 定义 `ProviderKeyPool`。
- [x] 删除 `ThreadSafeCircularList` 的可变默认参数。
- [x] 删除公开方法之间的隐式锁依赖。
- [x] 实现 sliding window 限流。
- [x] 实现 TPR 限制。
- [x] 实现 key cooldown。
- [x] 实现 rollback last rate-limit record。
- [x] 实现 smart key reorder 使用 repository 查询。
- [x] 覆盖空 key pool 行为测试。
- [x] 覆盖多个窗口组合测试。
- [x] 覆盖 fuzzy model rate limit 测试。
- [x] 覆盖 cooldown 到期测试。
- [x] 覆盖 rollback 测试。
- [x] 覆盖并发 key selection 测试。

### Upstream 执行和错误策略

- [x] 实现唯一 `UpstreamRunner`。
- [x] 定义 `RetryPolicy`。
- [x] 定义 `CooldownPolicy`。
- [x] 定义 `ProviderErrorClassifier`。
- [x] 统一 HTTPException 到 ProviderError 转换。
- [x] 统一 httpx 网络异常转换。
- [x] 统一上游 JSON error 转换。
- [x] 实现已提交输出后禁止重试。
- [x] 实现未提交输出前允许重试。
- [x] 实现 prepare failure 的 retry 规则。
- [x] 实现 401/403 清 Codex auth cache。
- [x] 实现 429/key quota cooling。
- [x] 实现 channel exclusion。
- [x] 实现 rollback rate-limit record。
- [x] 覆盖 bad request 不重试测试。
- [x] 覆盖 semantic failure 重试测试。
- [x] 覆盖 network failure 重试测试。
- [x] 覆盖 auth failure 清 cache 测试。
- [x] 覆盖 cooldown observability 测试。

### HTTP ClientPool

- [x] 定义 client key 规范。
- [x] 按 host/proxy/http2/verify/redirect policy 复用 client。
- [x] 用 per-key lock 防止并发重复创建。
- [x] 实现 lifespan shutdown 关闭所有 clients。
- [x] 实现 idle connection sweep。
- [x] 实现 client pool snapshot。
- [x] 覆盖并发复用同一 client 测试。
- [x] 覆盖 Codex 强制 HTTP/1.1 测试。
- [x] 覆盖 proxy key 区分测试。
- [x] 覆盖 shutdown close 测试。

### Streaming

- [x] 实现唯一 `IncrementalSSEParser`。
- [x] 支持 chunk split event。
- [x] 支持 comment frame。
- [x] 支持 named event。
- [x] 支持 `[DONE]`。
- [x] 实现 Responses API event parser。
- [x] 实现 Responses stream 到 Chat Completions stream 转换。
- [x] 实现 Chat Completions chunk builder。
- [x] 实现 keepalive 策略。
- [x] 实现 first-byte 标记。
- [x] 实现下游断开检测。
- [x] 实现 async iterator safe close。
- [x] 实现 response safe close。
- [x] 实现 stream context safe close。
- [x] 实现 cleanup background task tracking。
- [x] 删除 main/core response 中重复 cleanup 代码。
- [x] 覆盖 split SSE 测试。
- [x] 覆盖 keepalive 不 commit 测试。
- [x] 覆盖 stream retry before output 测试。
- [x] 覆盖 output committed 后不 retry 测试。
- [x] 覆盖 downstream close cleanup 测试。
- [x] 覆盖 upstream response close 测试。

### 认证和 Codex OAuth

- [x] 将 API key 验证移入 `auth/api_keys.py`。
- [x] 将 admin key 验证移入 `auth/api_keys.py`。
- [x] 将 Codex key split 移入 `auth/codex_oauth.py`。
- [x] 将 Codex refresh token store 移入独立类。
- [x] 实现 refresh token store 原子写入。
- [x] 实现 per-account OAuth lock。
- [x] 实现 access token cache TTL。
- [x] 实现 401/403 清 cache。
- [x] 覆盖 plain bearer passthrough 测试。
- [x] 覆盖 account/refresh 格式测试。
- [x] 覆盖 refresh token 持久化测试。
- [x] 覆盖并发 refresh 单飞测试。

### Observability 和统计

- [x] 定义类型化 `RequestContext`。
- [x] 将 contextvar 封装到 `request_context.py`。
- [x] 将统计中间件移到 `observability/middleware.py`。
- [x] 将 Fugue telemetry 适配到 `observability/telemetry.py`。
- [x] 将 request/channel stats 写入移到 repository。
- [x] 将 token usage 查询移到 repository。
- [x] 将 channel key rankings 查询移到 repository。
- [x] 将 paid key states 计算移到 service。
- [x] 统一 trace/span 字段命名。
- [x] 统一错误类型分类。
- [x] 覆盖 healthz 不鉴权测试。
- [x] 覆盖 x-request-id/traceparent 测试。
- [x] 覆盖 streaming 结束后写统计测试。
- [x] 覆盖数据库禁用时主路径不失败测试。

### API 层

- [x] 拆分 `/v1/chat/completions` 路由。
- [x] 拆分 `/v1/responses` 路由。
- [x] 拆分 `/v1/responses/compact` 路由。
- [x] 拆分 `/v1/messages` 路由。
- [x] 拆分 `/v1/models` 路由。
- [x] 拆分 image generation/edit 路由。
- [x] 拆分 audio speech/transcription 路由。
- [x] 拆分 embeddings 路由。
- [x] 拆分 moderations 路由。
- [x] 拆分 video/assets 路由。
- [x] 拆分 admin config 路由。
- [x] 拆分 stats/token usage/rankings 路由。
- [x] 拆分 health/observability 路由。
- [x] 每个路由文件只保留参数、依赖和 handler 调用。

### 测试和 CI

- [x] 创建 pytest 配置。
- [x] 将默认 testpaths 指向离线 unit tests。
- [x] 增加 `integration` marker。
- [x] 增加 `network` marker。
- [x] 增加 `credentials` marker。
- [x] 修复 `core/test` 相对导入收集问题。
- [x] 将脚本型测试移出默认 testpaths。
- [x] 将 matplotlib 示例测试标记为 optional。
- [x] 将 requests 真实网络测试标记为 integration。
- [x] 将 Vertex 凭证测试标记为 credentials。
- [x] 添加 `uv run pytest -q tests/unit` CI。
- [x] 添加 lint/type 检查命令。
- [x] 添加 adapter golden fixture。
- [x] 添加 SSE fixture。
- [x] 添加 config fixture。
- [x] 添加 benchmark fixture。

### 性能验证

- [x] 添加 `/v1/models` 缓存命中测试。
- [x] 添加 large config 路由 benchmark。
- [x] 添加 100 providers x 100 models 配置编译 benchmark。
- [x] 添加 1000 API keys 权限解析 benchmark。
- [x] 添加 weighted scheduling cache benchmark。
- [x] 添加 SSE parser chunk throughput benchmark。
- [x] 添加 non-stream response parse benchmark。
- [x] 添加 client pool concurrency benchmark。
- [x] 记录重构后基线数据。

重构后已验证基线：

- 请求中间件轻量 inspection benchmark，命令：
  `uv run python tools/benchmark_request_inspection.py --iterations 2000 --rounds 5 --messages 32`
- Chat messages payload，32 条 messages、2000 次、5 轮：完整 Pydantic 校验 best 0.775856s；轻量 inspection best 0.001605s；best speedup 约 483.41x。
- Responses input payload，32 条 input items、2000 次、5 轮：完整 Pydantic 校验 best 0.007914s；轻量 inspection best 0.002860s；best speedup 约 2.77x。
- 配置编译 benchmark，命令：
  `uv run python tools/benchmark_config_compile.py --providers 100 --models-per-provider 100 --api-keys 1000 --rounds 5`
- 100 providers x 100 models x 1000 API keys，5 轮：best 3.417933s；mean 3.941323s。该路径是显式 reload/update 冷路径，不在请求热路径执行。
- 路由 benchmark，命令：
  `uv run python tools/benchmark_routing.py --providers 100 --models-per-provider 100 --iterations 1000 --rounds 5`
- 100 providers x 100 models，固定优先级路由 1000 次、5 轮：best 2.055642s；mean 2.105607s。weighted scheduling cache 1000 次、5 轮：best 0.009259s；mean 0.009729s。
- Streaming benchmark，命令：
  `uv run python tools/benchmark_streaming.py --events 5000 --rounds 5`
- SSE parser 5000 events、5 轮：best 0.004292s；mean 0.004451s。Responses stream parse/Chat chunk conversion 5000 events、5 轮：best 0.052280s；mean 0.053509s。
- Client pool concurrency benchmark，命令：
  `uv run python tools/benchmark_client_pool.py --concurrency 100 --acquisitions 100 --rounds 5`
- 100 并发 worker x 100 次获取、5 轮：best 0.150901s；mean 0.179258s。
- API key lookup benchmark，命令：
  `uv run python tools/benchmark_api_key_lookup.py --api-keys 10000 --lookups 1000 --rounds 5`
- 10000 API keys、1000 次查找、5 轮：`list.index` best 0.040428s；dict index best 0.000022s；best speedup 约 1862.35x。请求鉴权/统计路径使用运行时编译出的 `api_key_index_by_token`，避免每次请求线性扫描。
- 本轮结构约束验证，命令：
  `python3 - <<'PY' ... ast scan ... PY`、`rg -n "app\\.state|main\\.app\\.state" uni_api/providers -g '*.py'`、`python3 - <<'PY' ... uni_api/api route decorator scan ... PY`
- 验证结果：全仓无 200 行以上函数；`uni_api/providers` 无全局 `app.state` 访问；`uni_api/api/*.py` 无 FastAPI 路由装饰器，路由入口在 runtime 中只做参数/依赖绑定和 handler 调用。

### 删除和收敛

- [x] 删除重复 `safe cleanup` 实现。
- [x] 删除重复 response usage 构造。
- [x] 删除重复 SSE comment 判断。
- [x] 删除重复 BaseAPI URL 归一化。
- [x] 删除 handler 内手写重试循环。
- [x] 删除 handler 内直接 SQL 查询。
- [x] 删除业务路径对原始 YAML dict 的扫描。
- [x] 删除 Provider 逻辑中的全局 `app.state` 访问。
- [x] 删除测试对全局 mutable state 的隐式依赖。

## 完成标准

- [x] 默认单元测试离线稳定通过。
- [x] 所有 Provider adapter 有 request/response/stream golden tests。
- [x] `/v1/models`、路由选择、限流、重试、stream cleanup 都有专项测试。
- [x] 新增 Provider 不需要修改核心执行器。
- [x] 新增 Endpoint 不需要修改 Provider registry 以外的共享热路径。
- [x] `main.py` 不再包含业务实现。
- [x] 没有 200 行以上的业务函数。
- [x] 请求热路径不扫描完整配置。
- [x] 流式断开不会泄漏上游连接。
- [x] 数据库禁用时核心 API 可用。
- [x] 全部后台任务可在 shutdown 时关闭或观测。
