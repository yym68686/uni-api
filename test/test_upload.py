import asyncio
import sys
import os

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import upload_image_to_0x0st

async def main():
    # 一个 1x1 红色像素的 PNG 图片的 base64 编码
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC1lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="

    print("正在上传图片到 0x0.st...")
    try:
        image_url = await upload_image_to_0x0st(base64_image)
        print(f"图片上传成功！ URL: {image_url}")
    except Exception as e:
        print(f"图片上传失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
