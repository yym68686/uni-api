import yaml
import json
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 读取YAML配置文件
def load_config():
    try:
        with open('./api.yaml', 'r') as f:
            conf = yaml.safe_load(f)
            for index, provider in enumerate(conf['providers']):
                model_dict = {}
                for model in provider['model']:
                    if type(model) == str:
                        model_dict[model] = model
                    if type(model) == dict:
                        model_dict.update({new: old for old, new in model.items()})
                        model_dict.update({old: old for old, new in model.items()})
                provider['model'] = model_dict
                conf['providers'][index] = provider
            api_keys_db = conf['api_keys']
            api_list = [item["api"] for item in api_keys_db]
            # print(json.dumps(conf, indent=4, ensure_ascii=False))
            return conf, api_keys_db, api_list
    except FileNotFoundError:
        print("配置文件 'api.yaml' 未找到。请确保文件存在于正确的位置。")
        return [], [], []
    except yaml.YAMLError:
        print("配置文件 'api.yaml' 格式不正确。请检查 YAML 格式。")
        return [], [], []

config, api_keys_db, api_list = load_config()

async def error_handling_wrapper(generator, status_code=200):
    try:
        first_item = await generator.__anext__()
        first_item_str = first_item
        if isinstance(first_item_str, (bytes, bytearray)):
            first_item_str = first_item_str.decode("utf-8")
        if isinstance(first_item_str, str):
            if first_item_str.startswith("data: "):
                first_item_str = first_item_str[6:]
            elif first_item_str.startswith("data:"):
                first_item_str = first_item_str[5:]
            first_item_str = json.loads(first_item_str)
        if isinstance(first_item_str, dict) and 'error' in first_item_str:
            # 如果第一个 yield 的项是错误信息，抛出 HTTPException
            raise HTTPException(status_code=status_code, detail=f"{first_item_str}"[:300])

        # 如果不是错误，创建一个新的生成器，首先yield第一个项，然后yield剩余的项
        async def new_generator():
            yield first_item
            async for item in generator:
                yield item

        return new_generator()
    except StopAsyncIteration:
        # 处理生成器为空的情况
        return "data: {'error': 'No data returned'}\n\n"

def post_all_models(token):
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

def get_all_models():
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

# 安全性依赖
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in api_list:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return token