import json
from fastapi import HTTPException
import httpx
from urllib.parse import urlparse

from log_config import logger

import re
from time import time
def parse_rate_limit(limit_string):
    # 定义时间单位到秒的映射
    time_units = {
        's': 1, 'sec': 1, 'second': 1,
        'm': 60, 'min': 60, 'minute': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
        'mo': 2592000, 'month': 2592000,
        'y': 31536000, 'year': 31536000
    }

    # 处理多个限制条件
    limits = []
    for limit in limit_string.split(','):
        limit = limit.strip()
        # 使用正则表达式匹配数字和单位
        match = re.match(r'^(\d+)/(\w+)$', limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {limit}")

        count, unit = match.groups()
        count = int(count)

        # 转换单位到秒
        if unit not in time_units:
            raise ValueError(f"Unknown time unit: {unit}")

        seconds = time_units[unit]
        limits.append((count, seconds))

    return limits

import asyncio
from collections import defaultdict

class ThreadSafeCircularList:
    def __init__(self, items = [], rate_limit={"default": "999999/min"}, schedule_algorithm="round_robin"):
        if schedule_algorithm == "random":
            import random
            self.items = random.sample(items, len(items))
            self.schedule_algorithm = "random"
        elif schedule_algorithm == "round_robin":
            self.items = items
            self.schedule_algorithm = "round_robin"
        elif schedule_algorithm == "fixed_priority":
            self.items = items
            self.schedule_algorithm = "fixed_priority"
        else:
            self.items = items
            logger.warning(f"Unknown schedule algorithm: {schedule_algorithm}, use (round_robin, random, fixed_priority) instead")
            self.schedule_algorithm = "round_robin"
        self.index = 0
        self.lock = asyncio.Lock()
        # 修改为二级字典，第一级是item，第二级是model
        self.requests = defaultdict(lambda: defaultdict(list))
        self.cooling_until = defaultdict(float)
        self.rate_limits = {}
        if isinstance(rate_limit, dict):
            for rate_limit_model, rate_limit_value in rate_limit.items():
                self.rate_limits[rate_limit_model] = parse_rate_limit(rate_limit_value)
        elif isinstance(rate_limit, str):
            self.rate_limits["default"] = parse_rate_limit(rate_limit)
        else:
            logger.error(f"Error ThreadSafeCircularList: Unknown rate_limit type: {type(rate_limit)}, rate_limit: {rate_limit}")

    async def set_cooling(self, item: str, cooling_time: int = 60):
        """设置某个 item 进入冷却状态

        Args:
            item: 需要冷却的 item
            cooling_time: 冷却时间(秒)，默认60秒
        """
        if item == None:
            return
        now = time()
        async with self.lock:
            self.cooling_until[item] = now + cooling_time
            # 清空该 item 的请求记录
            # self.requests[item] = []
            logger.warning(f"API key {item} 已进入冷却状态，冷却时间 {cooling_time} 秒")

    async def is_rate_limited(self, item, model: str = None) -> bool:
        now = time()
        # 检查是否在冷却中
        if now < self.cooling_until[item]:
            return True

        # 获取适用的速率限制

        if model:
            model_key = model
        else:
            model_key = "default"

        rate_limit = None
        # 先尝试精确匹配
        if model and model in self.rate_limits:
            rate_limit = self.rate_limits[model]
        else:
            # 如果没有精确匹配，尝试模糊匹配
            for limit_model in self.rate_limits:
                if limit_model != "default" and model and limit_model in model:
                    rate_limit = self.rate_limits[limit_model]
                    break

        # 如果都没匹配到，使用默认值
        if rate_limit is None:
            rate_limit = self.rate_limits.get("default", [(999999, 60)])  # 默认限制

        # 检查所有速率限制条件
        for limit_count, limit_period in rate_limit:
            # 使用特定模型的请求记录进行计算
            recent_requests = sum(1 for req in self.requests[item][model_key] if req > now - limit_period)
            if recent_requests >= limit_count:
                logger.warning(f"API key {item} 对模型 {model_key} 已达到速率限制 ({limit_count}/{limit_period}秒)")
                return True

        # 清理太旧的请求记录
        max_period = max(period for _, period in rate_limit)
        self.requests[item][model_key] = [req for req in self.requests[item][model_key] if req > now - max_period]

        # 记录新的请求
        self.requests[item][model_key].append(now)
        return False

    async def next(self, model: str = None):
        async with self.lock:
            if self.schedule_algorithm == "fixed_priority":
                self.index = 0
            start_index = self.index
            while True:
                item = self.items[self.index]
                self.index = (self.index + 1) % len(self.items)

                if not await self.is_rate_limited(item, model):
                    return item

                # 如果已经检查了所有的 API key 都被限制
                if self.index == start_index:
                    logger.warning(f"All API keys are rate limited!")
                    raise HTTPException(status_code=429, detail="Too many requests")

    async def after_next_current(self):
        # 返回当前取出的 API，因为已经调用了 next，所以当前API应该是上一个
        if len(self.items) == 0:
            return None
        async with self.lock:
            item = self.items[(self.index - 1) % len(self.items)]
            return item

    def get_items_count(self) -> int:
        """返回列表中的项目数量

        Returns:
            int: items列表的长度
        """
        return len(self.items)

def get_proxy(proxy, client_config = {}):
    if proxy:
        # 解析代理URL
        parsed = urlparse(proxy)
        scheme = parsed.scheme.rstrip('h')

        if scheme == 'socks5':
            try:
                from httpx_socks import AsyncProxyTransport
                proxy = proxy.replace('socks5h://', 'socks5://')
                transport = AsyncProxyTransport.from_url(proxy)
                client_config["transport"] = transport
                # print("proxy", proxy)
            except ImportError:
                logger.error("httpx-socks package is required for SOCKS proxy support")
                raise ImportError("Please install httpx-socks package for SOCKS proxy support: pip install httpx-socks")
        else:
            client_config["proxies"] = {
                "http://": proxy,
                "https://": proxy
            }
    return client_config

def circular_list_encoder(obj):
    if isinstance(obj, ThreadSafeCircularList):
        return obj.to_dict()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

provider_api_circular_list = defaultdict(ThreadSafeCircularList)

def get_model_dict(provider):
    model_dict = {}
    for model in provider['model']:
        if type(model) == str:
            model_dict[model] = model
        if isinstance(model, dict):
            model_dict.update({new: old for old, new in model.items()})
    return model_dict

def update_initial_model(provider):
    try:
        engine, stream_mode = get_engine(provider, endpoint=None, original_model="")
        # print("engine", engine, provider)
        api_url = provider['base_url']
        api = provider['api']
        proxy = safe_get(provider, "preferences", "proxy", default=None)
        client_config = get_proxy(proxy)
        if engine == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            params = {"key": api}

            with httpx.Client(**client_config) as client:
                response = client.get(url, params=params)

            original_models = response.json()
            if original_models.get("error"):
                raise Exception({"error": original_models.get("error"), "endpoint": url, "api": api})

            models = {"data": []}
            for model in original_models["models"]:
                models["data"].append({
                    "id": model["name"].split("models/")[-1],
                })
        else:
            endpoint = BaseAPI(api_url=api_url)
            endpoint_models_url = endpoint.v1_models
            if isinstance(api, list):
                api = api[0]
            headers = {"Authorization": f"Bearer {api}"}
            response = httpx.get(
                endpoint_models_url,
                headers=headers,
                **client_config
            )
            models = response.json()
            if models.get("error"):
                logger.error({"error": models.get("error"), "endpoint": endpoint_models_url, "api": api})
                return []

        # print(models)
        models_list = models["data"]
        models_id = [model["id"] for model in models_list]
        set_models = set()
        for model_item in models_id:
            set_models.add(model_item)
        models_id = list(set_models)
        # print(models_id)
        return models_id
    except Exception as e:
        # print("error:", e)
        import traceback
        traceback.print_exc()
        return []

from ruamel.yaml import YAML, YAMLError
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

API_YAML_PATH = "./api.yaml"
yaml_error_message = None

def save_api_yaml(config_data):
    with open(API_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

def update_config(config_data, use_config_url=False):
    for index, provider in enumerate(config_data['providers']):
        if provider.get('project_id'):
            provider['base_url'] = 'https://aiplatform.googleapis.com/'
        if provider.get('cf_account_id'):
            provider['base_url'] = 'https://api.cloudflare.com/'

        if isinstance(provider['provider'], int):
            provider['provider'] = str(provider['provider'])

        provider_api = provider.get('api', None)
        if provider_api:
            if isinstance(provider_api, int):
                provider_api = str(provider_api)
            if isinstance(provider_api, str):
                provider_api_circular_list[provider['provider']] = ThreadSafeCircularList(
                    [provider_api],
                    safe_get(provider, "preferences", "api_key_rate_limit", default={"default": "999999/min"}),
                    safe_get(provider, "preferences", "api_key_schedule_algorithm", default="round_robin")
                )
            if isinstance(provider_api, list):
                provider_api_circular_list[provider['provider']] = ThreadSafeCircularList(
                    provider_api,
                    safe_get(provider, "preferences", "api_key_rate_limit", default={"default": "999999/min"}),
                    safe_get(provider, "preferences", "api_key_schedule_algorithm", default="round_robin")
                )

        if "models.inference.ai.azure.com" in provider['base_url'] and not provider.get("model"):
            provider['model'] = [
                "gpt-4o",
                "gpt-4o-mini",
                "o1-mini",
                "o1-preview",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ]

        if not provider.get("model"):
            model_list = update_initial_model(provider)
            if model_list:
                provider["model"] = model_list
                if not use_config_url:
                    save_api_yaml(config_data)

        if provider.get("tools") == None:
            provider["tools"] = True

        config_data['providers'][index] = provider

    api_keys_db = config_data['api_keys']

    for index, api_key in enumerate(config_data['api_keys']):
        weights_dict = {}
        models = []
        if api_key.get('model'):
            for model in api_key.get('model'):
                if isinstance(model, dict):
                    key, value = list(model.items())[0]
                    provider_name = key.split("/")[0]
                    model_name = key.split("/")[1]

                    for provider_item in config_data["providers"]:
                        if provider_item['provider'] != provider_name:
                            continue
                        model_dict = get_model_dict(provider_item)
                        if model_name in model_dict.keys():
                            weights_dict.update({provider_name + "/" + model_name: int(value)})
                        elif model_name == "*":
                            weights_dict.update({provider_name + "/" + model_name: int(value) for model_item in model_dict.keys()})

                    models.append(key)
                if isinstance(model, str):
                    models.append(model)
            if weights_dict:
                config_data['api_keys'][index]['weights'] = weights_dict
            config_data['api_keys'][index]['model'] = models
            api_keys_db[index]['model'] = models
        else:
            # Default to all models if 'model' field is not set
            config_data['api_keys'][index]['model'] = ["all"]
            api_keys_db[index]['model'] = ["all"]

    api_list = [item["api"] for item in api_keys_db]
    # logger.info(json.dumps(config_data, indent=4, ensure_ascii=False))
    return config_data, api_keys_db, api_list

# 读取YAML配置文件
async def load_config(app=None):
    import os
    try:
        with open(API_YAML_PATH, 'r', encoding='utf-8') as file:
            conf = yaml.load(file)

        if conf:
            config, api_keys_db, api_list = update_config(conf, use_config_url=False)
        else:
            logger.error("配置文件 'api.yaml' 为空。请检查文件内容。")
            config, api_keys_db, api_list = {}, {}, []
    except FileNotFoundError:
        if not os.environ.get('CONFIG_URL'):
            logger.error("'api.yaml' not found. Please check the file path.")
        config, api_keys_db, api_list = {}, {}, []
    except YAMLError as e:
        logger.error("配置文件 'api.yaml' 格式不正确。请检查 YAML 格式。%s", e)
        global yaml_error_message
        yaml_error_message = "配置文件 'api.yaml' 格式不正确。请检查 YAML 格式。"
        config, api_keys_db, api_list = {}, {}, []
    except OSError as e:
        logger.error(f"open 'api.yaml' failed: {e}")
        config, api_keys_db, api_list = {}, {}, []

    if config != {}:
        return config, api_keys_db, api_list

    # 新增： 从环境变量获取配置URL并拉取配置
    config_url = os.environ.get('CONFIG_URL')
    if config_url:
        try:
            default_config = {
                "headers": {
                    "User-Agent": "curl/7.68.0",
                    "Accept": "*/*",
                },
                "http2": True,
                "verify": True,
                "follow_redirects": True
            }
            # 初始化客户端管理器
            timeout = httpx.Timeout(
                connect=15.0,
                read=100,
                write=30.0,
                pool=200
            )
            client = httpx.AsyncClient(
                timeout=timeout,
                **default_config
            )
            response = await client.get(config_url)
            # logger.info(f"Fetching config from {response.text}")
            response.raise_for_status()
            config_data = yaml.load(response.text)
            # 更新配置
            # logger.info(config_data)
            if config_data:
                config, api_keys_db, api_list = update_config(config_data, use_config_url=True)
            else:
                logger.error(f"Error fetching or parsing config from {config_url}")
                config, api_keys_db, api_list = {}, {}, []
        except Exception as e:
            logger.error(f"Error fetching or parsing config from {config_url}: {str(e)}")
            config, api_keys_db, api_list = {}, {}, []
    return config, api_keys_db, api_list

def ensure_string(item):
    if isinstance(item, (bytes, bytearray)):
        return item.decode("utf-8")
    elif isinstance(item, str):
        return item
    elif isinstance(item, dict):
        return f"data: {json.dumps(item)}\n\n"
    else:
        return str(item)

def identify_audio_format(file_bytes):
    # 读取开头的字节
    if file_bytes.startswith(b'\xFF\xFB') or file_bytes.startswith(b'\xFF\xF3'):
        return "MP3"
    elif file_bytes.startswith(b'ID3'):
        return "MP3 with ID3"
    elif file_bytes.startswith(b'OpusHead'):
        return "OPUS"
    elif file_bytes.startswith(b'ADIF'):
        return "AAC (ADIF)"
    elif file_bytes.startswith(b'\xFF\xF1') or file_bytes.startswith(b'\xFF\xF9'):
        return "AAC (ADTS)"
    elif file_bytes.startswith(b'fLaC'):
        return "FLAC"
    elif file_bytes.startswith(b'RIFF') and file_bytes[8:12] == b'WAVE':
        return "WAV"
    return "Unknown/PCM"

import asyncio
import time as time_module
async def error_handling_wrapper(generator, channel_id, engine, stream, error_triggers):
    start_time = time_module.time()
    try:
        first_item = await generator.__anext__()
        first_response_time = time_module.time() - start_time
        first_item_str = first_item
        # logger.info("first_item_str: %s :%s", type(first_item_str), first_item_str)
        if isinstance(first_item_str, (bytes, bytearray)):
            if identify_audio_format(first_item_str) in ["MP3", "MP3 with ID3", "OPUS", "AAC (ADIF)", "AAC (ADTS)", "FLAC", "WAV"]:
                return first_item, first_response_time
            else:
                first_item_str = first_item_str.decode("utf-8")
        if isinstance(first_item_str, str):
            if first_item_str.startswith("data:"):
                first_item_str = first_item_str.lstrip("data: ")
            if first_item_str.startswith("[DONE]"):
                logger.error(f"provider: {channel_id:<11} error_handling_wrapper [DONE]!")
                raise StopAsyncIteration
            try:
                encode_first_item_str = first_item_str.encode().decode('unicode-escape')
            except UnicodeDecodeError:
                encode_first_item_str = first_item_str
                logger.error(f"provider: {channel_id:<11} error UnicodeDecodeError: %s", first_item_str)
            if any(x in encode_first_item_str for x in error_triggers):
                logger.error(f"provider: {channel_id:<11} error const string: %s", encode_first_item_str)
                raise StopAsyncIteration
            try:
                first_item_str = json.loads(first_item_str)
            except json.JSONDecodeError:
                logger.error(f"provider: {channel_id:<11} error_handling_wrapper JSONDecodeError! {repr(first_item_str)}")
                raise StopAsyncIteration
        if isinstance(first_item_str, dict) and 'error' in first_item_str and first_item_str.get('error') != {"message": "","type": "","param": "","code": None}:
            # 如果第一个 yield 的项是错误信息，抛出 HTTPException
            status_code = first_item_str.get('status_code', 500)
            detail = first_item_str.get('details', f"{first_item_str}")
            raise HTTPException(status_code=status_code, detail=f"{detail}"[:300])

        if isinstance(first_item_str, dict) and safe_get(first_item_str, "choices", 0, "error", default=None):
            # 如果第一个 yield 的项是错误信息，抛出 HTTPException
            status_code = safe_get(first_item_str, "choices", 0, "error", "code", default=500)
            detail = safe_get(first_item_str, "choices", 0, "error", "message", default=f"{first_item_str}")
            raise HTTPException(status_code=status_code, detail=f"{detail}"[:300])

        if isinstance(first_item_str, dict) and engine not in ["tts", "embedding", "dalle", "moderation", "whisper"] and stream == False:
            if any(x in str(first_item_str) for x in error_triggers):
                logger.error(f"provider: {channel_id:<11} error const string: %s", first_item_str)
                raise StopAsyncIteration
            content = safe_get(first_item_str, "choices", 0, "message", "content", default=None)
            tool_calls = safe_get(first_item_str, "choices", 0, "message", "tool_calls", default=None)
            if (content == "" or content is None) and (tool_calls == "" or tool_calls is None):
                raise StopAsyncIteration

        # 如果不是错误，创建一个新的生成器，首先yield第一个项，然后yield剩余的项
        async def new_generator():
            # print("type(first_item)", type(first_item))
            # print("first_item", ensure_string(first_item))
            yield ensure_string(first_item)
            try:
                async for item in generator:
                    yield ensure_string(item)
            except asyncio.CancelledError:
                # 客户端断开连接是正常行为，不需要记录错误日志
                logger.debug(f"provider: {channel_id:<11} Stream cancelled by client")
                return
            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                # 只记录真正的网络错误
                # logger.error(f"provider: {channel_id:<11} Network error in new_generator: {e}")
                raise HTTPException(status_code=502, detail=f"Network error in new_generator: {e}")

        return new_generator(), first_response_time

    except StopAsyncIteration:
        raise HTTPException(status_code=400, detail="data: {'error': 'No data returned'}")

def post_all_models(api_index, config, api_list, models_list):
    all_models = []
    unique_models = set()

    if config['api_keys'][api_index]['model']:
        for model in config['api_keys'][api_index]['model']:
            if model == "all":
                # 如果模型名为 all，则返回所有模型
                all_models = get_all_models(config)
                return all_models
            if "/" in model:
                provider = model.split("/")[0]
                model = model.split("/")[1]
                if model == "*":
                    if provider.startswith("sk-") and provider in api_list:
                        for model_item in models_list[provider]:
                            if model_item not in unique_models:
                                unique_models.add(model_item)
                                model_info = {
                                    "id": model_item,
                                    "object": "model",
                                    "created": 1720524448858,
                                    "owned_by": "uni-api"
                                }
                                all_models.append(model_info)
                    else:
                        for provider_item in config["providers"]:
                            if provider_item['provider'] != provider:
                                continue
                            model_dict = get_model_dict(provider_item)
                            for model_item in model_dict.keys():
                                if model_item not in unique_models:
                                    unique_models.add(model_item)
                                    model_info = {
                                        "id": model_item,
                                        "object": "model",
                                        "created": 1720524448858,
                                        "owned_by": "uni-api"
                                        # "owned_by": provider_item['provider']
                                    }
                                    all_models.append(model_info)
                else:
                    if provider.startswith("sk-") and provider in api_list:
                        if model in models_list[provider] and model not in unique_models:
                            unique_models.add(model)
                            model_info = {
                                "id": model,
                                "object": "model",
                                "created": 1720524448858,
                                "owned_by": "uni-api"
                            }
                            all_models.append(model_info)
                    else:
                        for provider_item in config["providers"]:
                            if provider_item['provider'] != provider:
                                continue
                            model_dict = get_model_dict(provider_item)
                            for model_item in model_dict.keys():
                                if model_item not in unique_models and model_item == model:
                                    unique_models.add(model_item)
                                    model_info = {
                                        "id": model_item,
                                        "object": "model",
                                        "created": 1720524448858,
                                        "owned_by": "uni-api"
                                    }
                                    all_models.append(model_info)
                continue

            if model.startswith("sk-") and model in api_list:
                continue

            if model not in unique_models:
                unique_models.add(model)
                model_info = {
                    "id": model,
                    "object": "model",
                    "created": 1720524448858,
                    "owned_by": "uni-api"
                }
                all_models.append(model_info)

    return all_models

def get_all_models(config):
    all_models = []
    unique_models = set()

    for provider in config["providers"]:
        model_dict = get_model_dict(provider)
        for model in model_dict.keys():
            if model not in unique_models:
                unique_models.add(model)
                model_info = {
                    "id": model,
                    "object": "model",
                    "created": 1720524448858,
                    "owned_by": "uni-api"
                }
                all_models.append(model_info)

    return all_models

# 【GCP-Vertex AI 目前有這些區域可用】 https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude?hl=zh_cn
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations?hl=zh-cn#available-regions

# c3.5s
# us-east5
# europe-west1

# c3s
# us-east5
# us-central1
# asia-southeast1

# c3o
# us-east5

# c3h
# us-east5
# us-central1
# europe-west1
# europe-west4


c35s = ThreadSafeCircularList(["us-east5", "europe-west1"])
c3s = ThreadSafeCircularList(["us-east5", "us-central1", "asia-southeast1"])
c3o = ThreadSafeCircularList(["us-east5"])
c3h = ThreadSafeCircularList(["us-east5", "us-central1", "europe-west1", "europe-west4"])
gemini1 = ThreadSafeCircularList(["us-central1", "us-east4", "us-west1", "us-west4", "europe-west1", "europe-west2"])
gemini2 = ThreadSafeCircularList(["us-central1"])

class BaseAPI:
    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        if api_url == "":
            api_url = "https://api.openai.com/v1/chat/completions"
        self.source_api_url: str = api_url
        from urllib.parse import urlparse, urlunparse
        parsed_url = urlparse(self.source_api_url)
        # print("parsed_url", parsed_url)
        if parsed_url.scheme == "":
            raise Exception("Error: API_URL is not set")
        if parsed_url.path != '/':
            before_v1 = parsed_url.path.split("chat/completions")[0]
        else:
            before_v1 = ""
        self.base_url: str = urlunparse(parsed_url[:2] + ("",) + ("",) * 3)
        self.v1_url: str = urlunparse(parsed_url[:2]+ (before_v1,) + ("",) * 3)
        self.v1_models: str = urlunparse(parsed_url[:2] + (before_v1 + "models",) + ("",) * 3)
        self.chat_url: str = urlunparse(parsed_url[:2] + (before_v1 + "chat/completions",) + ("",) * 3)
        self.image_url: str = urlunparse(parsed_url[:2] + (before_v1 + "images/generations",) + ("",) * 3)
        self.audio_transcriptions: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/transcriptions",) + ("",) * 3)
        self.moderations: str = urlunparse(parsed_url[:2] + (before_v1 + "moderations",) + ("",) * 3)
        self.embeddings: str = urlunparse(parsed_url[:2] + (before_v1 + "embeddings",) + ("",) * 3)
        self.audio_speech: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/speech",) + ("",) * 3)

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    if not data:
        return default
    return data


# end_of_line = "\n\r\n"
# end_of_line = "\r\n"
# end_of_line = "\n\r"
end_of_line = "\n\n"
# end_of_line = "\r"
# end_of_line = "\n"

import random
import string
async def generate_sse_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=0, prompt_tokens=0, completion_tokens=0, reasoning_content=None):
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))

    delta_content = {"role": "assistant", "content": content} if content else {}
    if reasoning_content:
        delta_content = {"role": "assistant", "content": "", "reasoning_content": reasoning_content}

    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta_content,
                "logprobs": None,
                "finish_reason": None if content else "stop"
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_d576307f90",
    }
    if function_call_content:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"arguments": function_call_content}}]}
    if tools_id and function_call_name:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"id": tools_id,"type":"function","function":{"name": function_call_name, "arguments":""}}]}
        # sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"id": tools_id, "name": function_call_name}}]}
    if role:
        sample_data["choices"][0]["delta"] = {"role": role, "content": ""}
    if total_tokens:
        total_tokens = prompt_tokens + completion_tokens
        sample_data["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
        sample_data["choices"] = []
    json_data = json.dumps(sample_data, ensure_ascii=False)

    # 构建SSE响应
    sse_response = f"data: {json_data}" + end_of_line

    return sse_response

async def generate_no_stream_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=0, prompt_tokens=0, completion_tokens=0):
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                    "refusal": None
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_a7d06e42a7"
    }

    if function_call_name:
        if not tools_id:
            tools_id = f"call_{random_str}"
        sample_data = {
            "id": f"chatcmpl-{random_str}",
            "object": "chat.completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tools_id,
                            "type": "function",
                            "function": {
                                "name": function_call_name,
                                "arguments": json.dumps(function_call_content, ensure_ascii=False)
                            }
                        }
                    ],
                    "refusal": None
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": None,
            "service_tier": "default",
            "system_fingerprint": "fp_4691090a87"
        }

    if total_tokens:
        total_tokens = prompt_tokens + completion_tokens
        sample_data["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}

    json_data = json.dumps(sample_data, ensure_ascii=False)

    return json_data

def get_engine(provider, endpoint=None, original_model=""):
    parsed_url = urlparse(provider['base_url'])
    # print("parsed_url", parsed_url)
    engine = None
    stream = None
    if parsed_url.path.endswith("/v1beta") or parsed_url.path.endswith("/v1"):
        engine = "gemini"
    elif parsed_url.netloc == 'aiplatform.googleapis.com':
        engine = "vertex"
    elif parsed_url.netloc.rstrip('/').endswith('openai.azure.com') or parsed_url.netloc.rstrip('/').endswith('services.ai.azure.com'):
        engine = "azure"
    elif parsed_url.netloc == 'api.cloudflare.com':
        engine = "cloudflare"
    elif parsed_url.netloc == 'api.anthropic.com' or parsed_url.path.endswith("v1/messages"):
        engine = "claude"
    elif parsed_url.netloc == 'api.cohere.com':
        engine = "cohere"
        stream = True
    else:
        engine = "gpt"

    original_model = original_model.lower()
    if original_model \
    and "claude" not in original_model \
    and "gpt" not in original_model \
    and "deepseek" not in original_model \
    and "o1" not in original_model \
    and "o3" not in original_model \
    and "gemini" not in original_model \
    and "learnlm" not in original_model \
    and "grok" not in original_model \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in original_model and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in original_model and engine == "vertex":
        engine = "vertex-gemini"

    if provider.get("engine"):
        engine = provider["engine"]

    if endpoint == "/v1/images/generations" or "stable-diffusion" in original_model:
        engine = "dalle"
        stream = False

    if endpoint == "/v1/audio/transcriptions":
        engine = "whisper"
        stream = False

    if endpoint == "/v1/moderations":
        engine = "moderation"
        stream = False

    if endpoint == "/v1/embeddings":
        engine = "embedding"

    if endpoint == "/v1/audio/speech":
        engine = "tts"
        stream = False

    return engine, stream