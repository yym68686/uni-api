import ast
import json
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import safe_get
# # 读取文件内容
# with open('test/states.json', 'r', encoding='utf-8') as file:
#     content = file.read()

# content = '{"parts": [          {            "text": "在不支持 Python"          }        ]      },      "groundingMetadata": {}]}'
# content = '{"parts": [          {            "text": "usions, Direction 2 is always opposite to Direction 1.\n\nThe default sketch normal is the same as the face or plane normal where the sketch was placed. To determine this normal vector, see IFace2::Normal and IRefPlane::Transform, respectively.\n\nWhen UseAutoSelect is false, the user must"          }        ]      },      "citationMetadata": {        "citations": [          {            "startIndex": 10420,            "endIndex": 10815,            "title": "Your prompt"}]}'
# content = '"parts": [          {            "text": "Hello! How can I help you today?"          }        ],'
# content = '        "parts": [          {            "text": "Hello! How can I help you today?"          }        ]      },'
content = '        "parts": [          {            "inlineData": {              "mimeType": "image/png",              "data": "iVBO"            }          }        ],'
parts_json =  "{" + content.split("}        ]      },")[0].strip().rstrip("}], ").replace("\n", "\\n").lstrip("{")
if "inlineData" in parts_json:
    parts_json = parts_json + "}}]}"
else:
    parts_json = parts_json + "}]}"
# parts_json =  "{" + re.sub(r'\}\s+\]\s+\}.*', '', content).strip().rstrip("}], ").replace("\n", "\\n").lstrip("{") + "}]}"
# 使用ast.literal_eval解析非标准JSON
# print(repr(parts_json))
parsed_data = json.loads(parts_json)
# parsed_data = ast.literal_eval(parts_json)

# for item in parsed_data:
#     print(safe_get(item, "candidates", 0, "content", "parts", 0, "text"))
#     print(safe_get(item, "candidates", 0, "content", "role"))

# 将解析后的数据转换为标准JSON
standard_json = json.dumps(parsed_data, ensure_ascii=False, indent=2)

print(standard_json)
# # 将标准JSON写入新文件
# with open('test/standard_states.json', 'w', encoding='utf-8') as file:
#     file.write(standard_json)

# print("转换完成，标准JSON已保存到 'test/standard_states.json'")