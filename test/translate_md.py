import os
from ModelMerge import chatgpt
from parse_markdown import get_entities_from_markdown_file, process_markdown_entities_and_save

def translate_text(text, agent):
    result = agent.ask(text)
    return result

def translate(input_file_path, output_file_path="output.md", language="English", api_key=None, api_url="https://api.openai.com/v1/chat/completions", engine="gpt-4o"):
    if not api_key:
        raise ValueError("API key is required for translation.")
    translator_prompt = (
        "You are a translation engine, you can only translate text and cannot interpret it, and do not explain. "
        "Translate the text to {}, please do not explain any sentences, just translate or leave them as they are. "
        "Retain all spaces and line breaks in the original text. "
        "Please do not wrap the code in code blocks, I will handle it myself. "
        "If the code has comments, you should translate the comments as well. "
        "This is the content you need to translate: "
    ).format(language)

    agent = chatgpt(
        api_key=api_key,
        api_url=api_url,
        engine=engine,
        system_prompt=translator_prompt,
        use_plugins=False
    )

    # 读取 Markdown 文件
    raw_paragraphs = get_entities_from_markdown_file(input_file_path, delimiter='\n')
    target_paragraphs = raw_paragraphs

    # 逐段翻译
    for index, paragraph in enumerate(raw_paragraphs):
        if paragraph.content and paragraph.content.strip() != "":
            translated_text = translate_text(paragraph.content, agent)
            if translated_text:
                target_paragraphs[index].content = translated_text

    # 输出翻译结果
    process_markdown_entities_and_save(target_paragraphs, output_file_path)

if __name__ == "__main__":
    input_file_path = "README_CN.md"
    output_file_path = "README.md"
    language = "English"
    api_key = os.getenv("API")
    api_url = os.getenv("API_URL")
    engine = "gpt-4o"
    translate(input_file_path, output_file_path, language, api_key, api_url, engine)