import json
import base64

# 读取JSON文件
with open('test/response.json', 'r') as f:
    data = json.load(f)

# 获取base64编码的图片数据
b64_data = data['data'][0]['b64_json']

# 解码base64数据
image_data = base64.b64decode(b64_data)

# 保存为PNG图片
with open('test/output.png', 'wb') as f:
    f.write(image_data)
