class MarkdownEntity:
    def __init__(self, content: str, entity_type: str):
        self.content = content
        self.entity_type = entity_type

    def __repr__(self):
        return f'<{self.entity_type}: {self.content}>'

class Title(MarkdownEntity):
    def __init__(self, content: str, level: int):
        super().__init__(content, 'Title')
        self.level = level

class CodeBlock(MarkdownEntity):
    def __init__(self, content: str, language: str = 'python'):
        super().__init__(content, 'CodeBlock')
        self.language = language

class ListItem(MarkdownEntity):
    def __init__(self, content: str):
        super().__init__(content, 'ListItem')

class Link(MarkdownEntity):
    def __init__(self, content: str, url: str):
        super().__init__(content, 'Link')
        self.url = url

class EmptyLine(MarkdownEntity):
    def __init__(self, content: str):
        super().__init__(content, 'EmptyLine')

class Paragraph(MarkdownEntity):
    def __init__(self, content: str):
        super().__init__(content, 'Paragraph')

def parse_markdown(lines, delimiter='\n\n'):
    entities = []
    current_code_block = []
    in_code_block = False
    language = None

    for line in lines:
        # line = line.strip()

        if line.startswith('#'):
            level = line.count('#')
            title_content = line[level:].strip()
            entities.append(Title(title_content, level))

        elif line.startswith('```'):
            if in_code_block and language:
                entities.append(CodeBlock(''.join(current_code_block), language))
                current_code_block = []
                in_code_block = False
                language = None
            else:
                in_code_block = True
                language = line.lstrip('`').strip()

        elif in_code_block:
            current_code_block.append(line)

        elif '[' in line and ']' in line and '(' in line and ')' in line and line.count('[') == 1:
            start = line.index('[') + 1
            end = line.index(']')
            url_start = line.index('(') + 1
            url_end = line.index(')')
            link_text = line[start:end].strip()
            link_url = line[url_start:url_end].strip()
            entities.append(Link(link_text, link_url))

        elif line == delimiter:
            entities.append(EmptyLine(line))

        elif line:
            entities.append(Paragraph(line))

    return entities

def convert_entities_to_text(entities):
    result = []
    for entity in entities:
        if isinstance(entity, Title):
            result.append(f"{'#' * entity.level} {entity.content}")
        elif isinstance(entity, CodeBlock):
            code = entity.content.lstrip('\n').rstrip('\n')
            result.append(f"```{entity.language}\n{code}\n```")
        elif isinstance(entity, ListItem):
            result.append(f"- {entity.content}")
        elif isinstance(entity, Link):
            result.append(f"[{entity.content}]({entity.url})")
        elif isinstance(entity, EmptyLine):
            result.append(f"{entity.content}")
        elif isinstance(entity, Paragraph):
            result.append(f"{entity.content}")
    return ''.join(result)

def save_text_to_file(text: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def process_markdown_entities_and_save(entities, file_path, raw_text=None):
    # Step 1: Convert entities to text
    text_output = convert_entities_to_text(entities)
    if raw_text and raw_text != text_output:
        raise ValueError("The text output does not match the raw text input.")
    # Step 2: Save to file
    save_text_to_file(text_output, file_path)

def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_markdown(text, delimiter='\n\n'):
    # 使用两个换行符作为分割标记，分割段落
    # 创建一个新的列表来存储结果
    paragraphs = text.split(delimiter)
    result = []

    # 遍历分割后的段落，在它们之间插入空行实体
    for i, paragraph in enumerate(paragraphs):
        if i > 0:
            # 在非第一段之前插入空行实体
            result.append(delimiter)

        # 添加当前段落
        result.append(paragraph)

    return result

def get_entities_from_markdown_file(file_path, delimiter='\n\n'):
    # 读取 Markdown 文件
    markdown_text = read_markdown_file(file_path)

    # 分割 Markdown 文档
    paragraphs = split_markdown(markdown_text, delimiter=delimiter)

    # 解析 Markdown 文档
    return parse_markdown(paragraphs, delimiter=delimiter)

if __name__ == '__main__':
    markdown_file_path = "README_CN.md"  # 替换为你的 Markdown 文件路径

    # 读取 Markdown 文件
    delimiter = '\n'
    markdown_text = read_markdown_file(markdown_file_path)
    paragraphs = split_markdown(markdown_text, delimiter=delimiter)
    parsed_entities = parse_markdown(paragraphs, delimiter=delimiter)

    # # 显示解析结果
    # result = [str(entity) for entity in parsed_entities]
    # for idx, entity in enumerate(result):
    #     print(f"段落 {idx + 1} 解析：{entity}\n")

    # 保存到文件
    output_file_path = "output.md"
    process_markdown_entities_and_save(parsed_entities, output_file_path, raw_text=markdown_text)

    print(f"Markdown 文档已保存到 {output_file_path}")