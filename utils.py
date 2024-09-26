import json
from fastapi import HTTPException
import httpx

from log_config import logger

def update_config(config_data):
    for index, provider in enumerate(config_data['providers']):
        model_dict = {}
        for model in provider['model']:
            if type(model) == str:
                model_dict[model] = model
            if type(model) == dict:
                model_dict.update({new: old for old, new in model.items()})
        provider['model'] = model_dict
        if provider.get('project_id'):
            provider['base_url'] = 'https://aiplatform.googleapis.com/'
        if provider.get('cf_account_id'):
            provider['base_url'] = 'https://api.cloudflare.com/'

        if provider.get('api'):
            if isinstance(provider.get('api'), str):
                provider['api'] = CircularList([provider.get('api')])
            if isinstance(provider.get('api'), list):
                provider['api'] = CircularList(provider.get('api'))

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
                    if "/" in key:
                        weights_dict.update({provider_name: int(value)})
                    models.append(key)
                if isinstance(model, str):
                    models.append(model)
            config_data['api_keys'][index]['weights'] = weights_dict
            config_data['api_keys'][index]['model'] = models
            api_keys_db[index]['model'] = models

    api_list = [item["api"] for item in api_keys_db]
    # logger.info(json.dumps(config_data, indent=4, ensure_ascii=False, default=circular_list_encoder))
    return config_data, api_keys_db, api_list

# 读取YAML配置文件
async def load_config(app=None):
    import yaml
    try:
        # with open('./api.yaml', 'r') as f:
        #     tokens = yaml.scan(f)
        #     for token in tokens:
        #         if isinstance(token, yaml.ScalarToken):
        #             value = token.value
        #             # 如果plain为False，表示字符串被引号包裹
        #             is_quoted = not token.plain
        #             print(f"值: {value}, 是否被引号包裹: {is_quoted}")

        with open('./api.yaml', 'r') as f:
            # 判断是否为空文件
            conf = yaml.safe_load(f)
            # conf = None
            if conf:
                config, api_keys_db, api_list = update_config(conf)
            else:
                # logger.error("配置文件 'api.yaml' 为空。请检查文件内容。")
                config, api_keys_db, api_list = [], [], []
    except FileNotFoundError:
        logger.error("'api.yaml' not found. Please check the file path.")
        config, api_keys_db, api_list = [], [], []
    except yaml.YAMLError:
        logger.error("配置文件 'api.yaml' 格式不正确。请检查 YAML 格式。")
        config, api_keys_db, api_list = [], [], []
    except OSError as e:
        logger.error(f"open 'api.yaml' failed: {e}")
        config, api_keys_db, api_list = [], [], []

    if config != []:
        return config, api_keys_db, api_list

    import os
    # 新增： 从环境变量获取配置URL并拉取配置
    config_url = os.environ.get('CONFIG_URL')
    if config_url:
        try:
            response = await app.state.client.get(config_url)
            # logger.info(f"Fetching config from {response.text}")
            response.raise_for_status()
            config_data = yaml.safe_load(response.text)
            # 更新配置
            # logger.info(config_data)
            if config_data:
                config, api_keys_db, api_list = update_config(config_data)
            else:
                logger.error(f"Error fetching or parsing config from {config_url}")
                config, api_keys_db, api_list = [], [], []
        except Exception as e:
            logger.error(f"Error fetching or parsing config from {config_url}: {str(e)}")
            config, api_keys_db, api_list = [], [], []
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

import asyncio
import time as time_module
async def error_handling_wrapper(generator):
    start_time = time_module.time()
    try:
        first_item = await generator.__anext__()
        first_response_time = time_module.time() - start_time
        first_item_str = first_item
        # logger.info("first_item_str: %s", first_item_str)
        if isinstance(first_item_str, (bytes, bytearray)):
            first_item_str = first_item_str.decode("utf-8")
        if isinstance(first_item_str, str):
            if first_item_str.startswith("data:"):
                first_item_str = first_item_str.lstrip("data: ")
            if first_item_str.startswith("[DONE]"):
                logger.error("error_handling_wrapper [DONE]!")
                raise StopAsyncIteration
            if "The bot's usage is covered by the developer" in first_item_str:
                logger.error("error const string!")
                raise StopAsyncIteration
            try:
                first_item_str = json.loads(first_item_str)
            except json.JSONDecodeError:
                logger.error("error_handling_wrapper JSONDecodeError!" + repr(first_item_str))
                raise StopAsyncIteration
        if isinstance(first_item_str, dict) and 'error' in first_item_str:
            # 如果第一个 yield 的项是错误信息，抛出 HTTPException
            status_code = first_item_str.get('status_code', 500)
            detail = first_item_str.get('details', f"{first_item_str}")
            raise HTTPException(status_code=status_code, detail=f"{detail}"[:300])

        # 如果不是错误，创建一个新的生成器，首先yield第一个项，然后yield剩余的项
        async def new_generator():
            yield ensure_string(first_item)
            try:
                async for item in generator:
                    yield ensure_string(item)
            except (httpx.ReadError, asyncio.CancelledError, httpx.RemoteProtocolError) as e:
                logger.error(f"Network error in new_generator: {e}")
                raise

        return new_generator(), first_response_time

    except StopAsyncIteration:
        raise HTTPException(status_code=400, detail="data: {'error': 'No data returned'}")

def post_all_models(token, config, api_list):
    all_models = []
    unique_models = set()

    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    api_index = api_list.index(token)
    if config['api_keys'][api_index]['model']:
        for model in config['api_keys'][api_index]['model']:
            if "/" in model:
                provider = model.split("/")[0]
                model = model.split("/")[1]
                if model == "*":
                    for provider_item in config["providers"]:
                        if provider_item['provider'] != provider:
                            continue
                        for model_item in provider_item['model'].keys():
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
                    for provider_item in config["providers"]:
                        if provider_item['provider'] != provider:
                            continue
                        for model_item in provider_item['model'].keys() :
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

            if model not in unique_models:
                unique_models.add(model)
                model_info = {
                    "id": model,
                    "object": "model",
                    "created": 1720524448858,
                    "owned_by": model
                }
                all_models.append(model_info)

    return all_models

def get_all_models(config):
    all_models = []
    unique_models = set()

    for provider in config["providers"]:
        for model in provider['model'].keys():
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

def circular_list_encoder(obj):
    if isinstance(obj, CircularList):
        return obj.to_dict()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

from collections import deque
class CircularList:
    def __init__(self, items):
        self.queue = deque(items)

    def next(self):
        if not self.queue:
            return None
        item = self.queue.popleft()
        self.queue.append(item)
        return item

    def to_dict(self):
        return {
            'queue': list(self.queue)
        }



c35s = CircularList(["us-east5", "europe-west1"])
c3s = CircularList(["us-east5", "us-central1", "asia-southeast1"])
c3o = CircularList(["us-east5"])
c3h = CircularList(["us-east5", "us-central1", "europe-west1", "europe-west4"])
gem = CircularList(["us-central1", "us-east4", "us-west1", "us-west4", "europe-west1", "europe-west2"])

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
        if parsed_url.scheme == "":
            raise Exception("Error: API_URL is not set")
        if parsed_url.path != '/':
            before_v1 = parsed_url.path.split("/v1")[0]
        else:
            before_v1 = ""
        self.base_url: str = urlunparse(parsed_url[:2] + (before_v1,) + ("",) * 3)
        self.v1_url: str = urlunparse(parsed_url[:2]+ (before_v1 + "/v1",) + ("",) * 3)
        self.v1_models: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/models",) + ("",) * 3)
        if parsed_url.netloc == "api.deepseek.com":
            self.chat_url: str = urlunparse(parsed_url[:2] + ("/chat/completions",) + ("",) * 3)
        else:
            self.chat_url: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/chat/completions",) + ("",) * 3)
        self.image_url: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/images/generations",) + ("",) * 3)
        self.audio_transcriptions: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/audio/transcriptions",) + ("",) * 3)
        self.moderations: str = urlunparse(parsed_url[:2] + (before_v1 + "/v1/moderations",) + ("",) * 3)

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    return data