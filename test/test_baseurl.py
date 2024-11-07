import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BaseAPI

def print_base_api(url):
    base_api = BaseAPI(url)
    print("base_url            ", base_api.base_url)
    print("v1_url              ", base_api.v1_url)
    print("chat_url            ", base_api.chat_url)
    print("image_url           ", base_api.image_url)
    print("audio_transcriptions", base_api.audio_transcriptions)
    print("moderations         ", base_api.moderations)
    print("embeddings          ", base_api.embeddings)
    print("-"*50)


print_base_api("https://api.openai.com/v1/chat/completions")
print_base_api("https://api.deepseek.com/chat/completions")
print_base_api("https://models.inference.ai.azure.com/chat/completions")
print_base_api("https://open.bigmodel.cn/api/paas/v4/chat/completions")
