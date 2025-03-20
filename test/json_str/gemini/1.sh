API_KEY=${GEMINI_API_KEY}

# 首先下载图片到本地
IMAGE_TEMP_FILE="temp_image.jpg"
echo "正在下载图片..."
curl -s -o "${IMAGE_TEMP_FILE}" "${IMAGE_URL}"
if [ ! -f "${IMAGE_TEMP_FILE}" ]; then
  echo "图片下载失败，请检查 IMAGE_URL"
  exit 1
fi

# 将图片转换为base64
echo "正在准备图片数据..."
IMAGE_BASE64=$(base64 -i "${IMAGE_TEMP_FILE}")

# 调用Gemini API进行图像生成
echo "正在调用Gemini API处理图像..."

# 创建临时JSON文件
JSON_FILE="request.json"
cat > "${JSON_FILE}" << EOF
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "inline_data": {
            "mime_type": "image/jpeg",
            "data": "${IMAGE_BASE64}"
          }
        },
        {
          "text": "删除手中的书"
        }
      ]
    }
  ],
  "safetySettings": [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "OFF"
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "OFF"
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "OFF"
    },
    {
      "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
      "threshold": "OFF"
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "OFF"
    }
  ],
  "generationConfig": {
    "temperature": 1,
    "topK": 40,
    "topP": 0.95,
    "maxOutputTokens": 8192,
    "responseMimeType": "text/plain"
  }
}
EOF

# 发送请求
curl -s -X POST \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key=${API_KEY}" \
  -H "Content-Type: application/json" \
  -d @"${JSON_FILE}"

# 清理临时文件
rm -f "${IMAGE_TEMP_FILE}" "${JSON_FILE}"