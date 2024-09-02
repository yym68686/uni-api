from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int
    size: str
    stream: bool = False

class FunctionParameter(BaseModel):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]

# 定义 Function 模型
class Function(BaseModel):
    name: str
    description: str
    parameters: Optional[FunctionParameter] = Field(default=None, exclude=None)

# 定义 Tool 模型
class Tool(BaseModel):
    type: str
    function: Function

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    name: Optional[str] = None
    arguments: Optional[str] = None
    content: Optional[Union[str, List[ContentItem]]] = None
    tool_calls: Optional[List[ToolCall]] = None

class Message(BaseModel):
    role: str
    name: Optional[str] = None
    content: Optional[Union[str, List[ContentItem]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    class Config:
        extra = "allow"  # 允许额外的字段

class RequestModel(BaseModel):
    model: str
    messages: List[Message]
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stream: Optional[bool] = None
    include_usage: Optional[bool] = None
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    user: Optional[str] = None
    tool_choice: Optional[str] = None
    tools: Optional[List[Tool]] = None