package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

// 预定义的SSE数据块
// 每个元素是 "data: <json_payload>" 或 "data: [DONE]"
var predefinedDataChunks = []string{
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "", "refusal": null}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": "Hello"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": "!"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " How"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " can"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " I"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " assist"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " you"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": " today"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {"content": "?"}, "logprobs": null, "finish_reason": null}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop"}], "usage": null}`,
	`data: {"id": "chatcmpl-fGeeCsNG1gMGU08ZnpDGtXmEGVxXH", "object": "chat.completion.chunk", "created": 1746713188, "model": "gpt-4o-mini-2024-07-18", "service_tier": "default", "system_fingerprint": "fp_dbaca60df0", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 10, "total_tokens": 18, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}}`,
	`data: [DONE]`,
}

func mockChatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	// log.Printf("Received request for %s from %s\n", r.URL.Path, r.RemoteAddr)

	// 设置 SSE 头部
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// 如果需要CORS，可以取消注释下一行并按需修改
	// w.Header().Set("Access-Control-Allow-Origin", "*")

	// 获取 Flusher接口，用于将数据块刷新到客户端
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
		return
	}

	// 模拟流式发送数据
	for _, chunk := range predefinedDataChunks {
		_, err := fmt.Fprintf(w, "%s\n\n", chunk) // SSE 格式要求每个消息后有两个换行符
		if err != nil {
			log.Printf("Error writing to response: chunk: %s, err: %v", chunk, err)
			// 客户端可能已经断开连接
			return
		}
		flusher.Flush() // 将数据块发送到客户端
		time.Sleep(100 * time.Millisecond) // 模拟真实流的处理延迟
	}

	// log.Printf("Finished streaming response for %s\n", r.URL.Path)
}

func main() {
	port := "8080" // 您可以根据需要更改端口

	http.HandleFunc("/v1/chat/completions", mockChatCompletionsHandler)

	log.Printf("Starting mock server on port %s...\n", port)
	log.Printf("Listening for requests on http://localhost:%s/v1/chat/completions\n", port)

	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatalf("Failed to start server: %s\n", err)
	}
}