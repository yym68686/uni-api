import json
import httpx
import asyncio
import h2.exceptions
from time import time
from fastapi import HTTPException
from collections import defaultdict

from core.log_config import logger
from core.utils import (
    safe_get,
    get_model_dict,
    update_initial_model,
    ThreadSafeCircularList,
    provider_api_circular_list,
)

class InMemoryRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)

    async def is_rate_limited(self, key: str, limits) -> bool:
        now = time()

        # 检查所有速率限制条件
        for limit, period in limits:
            # 计算在当前时间窗口内的请求数量
            recent_requests = sum(1 for req in self.requests[key] if req > now - period)
            if recent_requests >= limit:
                return True

        # 清理太旧的请求记录（比最长时间窗口还要老的记录）
        max_period = max(period for _, period in limits)
        self.requests[key] = [req for req in self.requests[key] if req > now - max_period]

        # 记录新的请求
        self.requests[key].append(now)
        return False

from ruamel.yaml import YAML, YAMLError
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

API_YAML_PATH = "./api.yaml"
yaml_error_message = None

def save_api_yaml(config_data):
    with open(API_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

async def update_config(config_data, use_config_url=False):
    for index, provider in enumerate(config_data['providers']):
        if provider.get('project_id'):
            if "google-vertex-ai" not in provider.get("base_url", ""):
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
                "gpt-4.1",
                "gpt-4o-mini",
                "o4-mini",
                "o3",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ]

        if not provider.get("model"):
            model_list = await update_initial_model(provider)
            if model_list:
                provider["model"] = model_list
                if not use_config_url:
                    save_api_yaml(config_data)

        if provider.get("tools") == None:
            provider["tools"] = True

        provider["_model_dict_cache"] = get_model_dict(provider)

        config_data['providers'][index] = provider

    for index, api_key in enumerate(config_data['api_keys']):
        if "api" in api_key:
            config_data['api_keys'][index]["api"] = str(api_key["api"])

    api_keys_db = config_data['api_keys']

    for index, api_key in enumerate(config_data['api_keys']):
        weights_dict = {}
        models = []

        # 确保api字段为字符串类型
        if "api" in api_key:
            config_data['api_keys'][index]["api"] = str(api_key["api"])

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
            config, api_keys_db, api_list = await update_config(conf, use_config_url=False)
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
                config, api_keys_db, api_list = await update_config(config_data, use_config_url=True)
            else:
                logger.error(f"Error fetching or parsing config from {config_url}")
                config, api_keys_db, api_list = {}, {}, []
        except Exception as e:
            logger.error(f"Error fetching or parsing config from {config_url}: {str(e)}")
            config, api_keys_db, api_list = {}, {}, []
    return config, api_keys_db, api_list

async def ensure_string(item):
    if isinstance(item, (bytes, bytearray)):
        return item.decode("utf-8")
    elif isinstance(item, str):
        return item
    elif isinstance(item, dict):
        json_str = await asyncio.to_thread(json.dumps, item)
        return f"data: {json_str}\n\n"
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

async def wait_for_timeout(wait_for_thing, timeout = 3, wait_task=None):
    # 创建一个任务来获取第一个响应，但不直接中断生成器
    if wait_task is None:
        first_response_task = asyncio.create_task(wait_for_thing.__anext__())
    else:
        first_response_task = wait_task

    # 创建一个超时任务
    timeout_task = asyncio.create_task(asyncio.sleep(timeout))

    # 等待任意一个任务完成
    done, pending = await asyncio.wait(
        [first_response_task, timeout_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    # 成功返回
    if first_response_task in done:
        # 取消超时任务
        timeout_task.cancel()
        return first_response_task.result(), "success"

    # 超时返回
    else:
        return first_response_task, "timeout"

import asyncio
import time as time_module
async def error_handling_wrapper(generator, channel_id, engine, stream, error_triggers, keepalive_interval=None):

    async def new_generator(first_item=None, with_keepalive=False, wait_task=None, timeout=3):
        # print("type(first_item)", type(first_item))
        # print("first_item", ensure_string(first_item))
        if first_item:
            yield await ensure_string(first_item)

        # 如果需要心跳机制但不使用嵌套生成器方式
        if with_keepalive:
            yield f": keepalive\n\n"
            while True:
                try:
                    item, status = await wait_for_timeout(generator, timeout=timeout, wait_task=wait_task)
                    if status == "timeout":
                        yield f": keepalive\n\n"
                    else:
                        yield await ensure_string(item)
                        wait_task = None
                except asyncio.CancelledError:
                    # 处理客户端断开连接
                    logger.debug(f"provider: {channel_id:<11} Stream cancelled by client in main loop")
                    break
                except Exception as e:
                    # 捕获任何其他异常
                    # import traceback
                    # error_stack = traceback.format_exc()
                    # error_message = error_stack.split("\n")[-2]
                    # logger.info(f"provider: {channel_id:<11} keepalive loop: {error_message}")
                    break
        else:
            # 原始的逻辑，当不需要心跳时
            try:
                async for item in generator:
                    yield await ensure_string(item)
            except asyncio.CancelledError:
                # 客户端断开连接是正常行为，不需要记录错误日志
                logger.debug(f"provider: {channel_id:<11} Stream cancelled by client")
                return
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.WriteError, httpx.ProtocolError, h2.exceptions.ProtocolError) as e:
                # 网络错误
                logger.error(f"provider: {channel_id:<11} Network error in new_generator: {e}")
                yield "data: [DONE]\n\n"
                return

    start_time = time_module.time()
    try:
        # 创建一个任务来获取第一个响应，但不直接中断生成器
        if keepalive_interval and stream == True:
            first_item, status = await wait_for_timeout(generator, timeout=keepalive_interval)
            if status == "timeout":
                return new_generator(None, with_keepalive=True, wait_task=first_item, timeout=keepalive_interval), 3.1415
        else:
            first_item = await generator.__anext__()

        first_response_time = time_module.time() - start_time
        # 对第一个响应项进行原有的处理逻辑
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
                first_item_str = await asyncio.to_thread(json.loads, first_item_str)
            except json.JSONDecodeError:
                logger.error(f"provider: {channel_id:<11} error_handling_wrapper JSONDecodeError! {repr(first_item_str)}")
                raise StopAsyncIteration

            # minimax
            status_code = safe_get(first_item_str, 'base_resp', 'status_code', default=200)
            if status_code != 200:
                if status_code == 2013:
                    status_code = 400
                if status_code == 1008:
                    status_code = 429
                detail = safe_get(first_item_str, 'base_resp', 'status_msg', default="no error returned")
                raise HTTPException(status_code=status_code, detail=f"{detail}"[:300])

        # minimax
        if isinstance(first_item_str, dict) and safe_get(first_item_str, "base_resp", "status_msg", default=None) == "success":
            full_audio_hex = safe_get(first_item_str, "data", "audio", default=None)
            audio_bytes = bytes.fromhex(full_audio_hex)
            return audio_bytes, first_response_time

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

        finish_reason = safe_get(first_item_str, "choices", 0, "finish_reason", default=None)
        if isinstance(first_item_str, dict) and finish_reason == "PROHIBITED_CONTENT":
            raise HTTPException(status_code=400, detail=f"PROHIBITED_CONTENT")

        if isinstance(first_item_str, dict) and finish_reason == "stop" and not safe_get(first_item_str, "choices", 0, "message", "content", default=None):
            raise StopAsyncIteration

        if isinstance(first_item_str, dict) and engine not in ["tts", "embedding", "dalle", "moderation", "whisper"] and stream == False:
            if any(x in str(first_item_str) for x in error_triggers):
                logger.error(f"provider: {channel_id:<11} error const string: %s", first_item_str)
                raise StopAsyncIteration
            content = safe_get(first_item_str, "choices", 0, "message", "content", default=None)
            reasoning_content = safe_get(first_item_str, "choices", 0, "message", "reasoning_content", default=None)
            b64_json = safe_get(first_item_str, "data", 0, "b64_json", default=None)
            tool_calls = safe_get(first_item_str, "choices", 0, "message", "tool_calls", default=None)
            if (content == "" or content is None) and (tool_calls == "" or tool_calls is None) and (reasoning_content == "" or reasoning_content is None) and b64_json == None:
                raise StopAsyncIteration

        return new_generator(first_item), first_response_time

    except StopAsyncIteration:
        # 502 Bad Gateway 是一个更合适的状态码，因为它表明作为代理或网关的服务器从上游服务器收到了无效的响应。
        raise HTTPException(status_code=502, detail="Upstream server returned an empty response.")

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
        model_dict = provider["_model_dict_cache"]
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

def calculate_total_cost(all_tokens_info, model_price):
    """
    计算所有模型使用的总金额

    参数:
        all_tokens_info: 列表，包含各模型的代币使用情况
        model_price: 字典，包含各模型的价格设置，格式为 "prompt_price,completion_price"，单位为 $/M tokens

    返回:
        float: 总金额（美元）
    """
    total_cost = 0.0

    for token_info in all_tokens_info:
        model_name = token_info["model"]
        prompt_tokens = token_info["total_prompt_tokens"]
        completion_tokens = token_info["total_completion_tokens"]

        # 获取模型价格，如果模型不存在则使用默认价格
        price_str = next((model_price[config_model_name] for config_model_name in model_price.keys() if model_name.startswith(config_model_name)), model_price.get("default", "1,2"))
        # print("price_str", price_str)

        # 解析价格字符串
        price_parts = price_str.split(",")
        prompt_price = float(price_parts[0])
        completion_price = float(price_parts[1])

        # 计算当前模型的费用 ($/M tokens 转换为 $)
        model_cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000000

        total_cost += model_cost

    return total_cost