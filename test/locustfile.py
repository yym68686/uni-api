from locust import HttpUser, task, between

class MockApiUser(HttpUser):
    wait_time = between(0.1, 0.5) # 较短的等待时间，模拟 API 调用

    # host 可以直接在启动时通过 --host 参数指定
    host = "http://localhost:8000"

    @task
    def get_chat_completions_stream(self):
        headers = {
            "Accept": "text/event-stream",
            "Authorization": "Bearer sk-xxx", # 如果需要认证
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ],
            "stream": True
        }

        # 使用 stream=True 来处理流式响应
        # 使用 catch_response=True 允许我们自己判断成功或失败
        with self.client.post("/v1/chat/completions", headers=headers, json=payload, stream=True, name="/v1/chat/completions (stream)", catch_response=True) as response:
            if response.status_code == 200:
                # 对于流式响应，内容会分块到达
                # 你可以在这里迭代处理流的内容，但通常在 Locust 中，
                # 我们主要关注请求是否成功建立以及整体的响应时间（直到流结束）
                # 例如，我们可以简单地读取一些内容来确保流中有数据
                content_received = False
                for chunk in response.iter_lines(): # 迭代内容块
                    if chunk:
                        content_received = True
                        # print(f"Received chunk: {chunk[:50]}...") # 调试用，生产压测时避免打印
                        # break # 简单演示，实际可能需要完整读取或更复杂的验证
                if content_received:
                    response.success()
                else:
                    response.failure("Stream was empty or connection closed early")
            else:
                response.failure(f"Status code was {response.status_code}")

    # 你可以添加更多 @task 来模拟不同的 API 调用或场景
    # @task
    # def post_some_data(self):
    #     payload = {"key": "value"}
    #     self.client.post("/v1/some_endpoint", json=payload)