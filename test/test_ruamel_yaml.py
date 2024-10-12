from ruamel.yaml import YAML

# 假设我们有以下 YAML 内容
yaml_content = """
# 这是顶级注释
key1: value1  # 行尾注释
key2: value2

# 这是嵌套结构的注释
nested:
  subkey1: subvalue1
  subkey2: subvalue2  # 嵌套的行尾注释

# 列表的注释
list_key:
  - item1
  - item2  # 列表项的注释
"""

# 创建 YAML 对象
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open('api.yaml', 'r', encoding='utf-8') as file:
    data = yaml.load(file)

# data = yaml.load(yaml_content)
# 加载 YAML 数据
print(data)

# # 修改数据
# data['key1'] = 'new_value1'
# data['nested']['subkey1'] = 'new_subvalue1'
# data['list_key'].append('new_item')

# 将修改后的数据写回文件（这里我们使用 StringIO 来模拟文件操作）
# from io import StringIO
# output = StringIO()
# yaml.dump(data, output)
# print(output.getvalue())

with open('formatted.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(data, file)