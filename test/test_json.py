import ast
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import safe_get
# 读取文件内容
with open('test/states.json', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用ast.literal_eval解析非标准JSON
parsed_data = ast.literal_eval(content)

for item in parsed_data:
    print(safe_get(item, "candidates", 0, "content", "parts", 0, "text"))
    print(safe_get(item, "candidates", 0, "content", "role"))

# 将解析后的数据转换为标准JSON
standard_json = json.dumps(parsed_data, ensure_ascii=False, indent=2)

# 将标准JSON写入新文件
with open('test/standard_states.json', 'w', encoding='utf-8') as file:
    file.write(standard_json)

print("转换完成，标准JSON已保存到 'test/standard_states.json'")