"""LLM Provider 抽象层 - 支持 Anthropic (Claude) 和 OpenAI 兼容 API (DeepSeek/GPT/GLM)。

通过统一接口屏蔽不同 SDK 的差异，Agent 层只需调用 provider.chat() 即可。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """统一的工具调用结构。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """统一的 LLM 响应结构。"""
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    stop_reason: str = ""  # "end_turn" or "tool_use"
    input_tokens: int = 0
    output_tokens: int = 0
    raw_content: Any = None  # 原始 assistant content，用于回填 messages


class LLMProvider(ABC):
    """LLM Provider 抽象基类。"""

    @abstractmethod
    def chat(
        self,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict],
        messages: list[dict],
    ) -> LLMResponse:
        ...

    @abstractmethod
    def build_tool_result_message(
        self, tool_call_id: str, content: str
    ) -> dict:
        """构建工具结果消息，格式因 provider 而异。"""
        ...

    @abstractmethod
    def build_assistant_message(self, raw_content: Any) -> dict:
        """构建 assistant 消息，用于追加到 messages。"""
        ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider。"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        import anthropic
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def chat(self, model, max_tokens, system, tools, messages) -> LLMResponse:
        # Anthropic 的 tools 格式用 input_schema
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        resp = LLMResponse(
            raw_content=response.content,
        )

        if hasattr(response, "usage") and response.usage:
            resp.input_tokens = response.usage.input_tokens
            resp.output_tokens = response.usage.output_tokens

        if response.stop_reason == "end_turn":
            resp.stop_reason = "end_turn"
            parts = [b.text for b in response.content if b.type == "text"]
            resp.text = "\n".join(parts)
        else:
            resp.stop_reason = "tool_use"
            resp.tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    resp.tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))

        return resp

    def build_tool_result_message(self, tool_call_id, content):
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }

    def build_assistant_message(self, raw_content):
        return {"role": "assistant", "content": raw_content}


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容 API provider（支持 DeepSeek、GPT、GLM、Qwen 等）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(self, model, max_tokens, system, tools, messages) -> LLMResponse:
        import time as _time

        # 转换 tools 格式：Anthropic input_schema → OpenAI parameters
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })

        # OpenAI 格式：system 作为第一条消息
        oai_messages = [{"role": "system", "content": system}] + messages

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        # 自动重试（处理 429 限流）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning("限流，%ds 后重试 (%d/%d): %s", wait, attempt + 1, max_retries, e)
                    _time.sleep(wait)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

        # 某些网关可能返回非标准格式，加容错
        if isinstance(response, str):
            logger.warning("LLM 返回了字符串而非对象，尝试解析: %s", response[:200])
            return LLMResponse(stop_reason="end_turn", text=response)

        choice = response.choices[0]

        resp = LLMResponse()

        if hasattr(response, "usage") and response.usage:
            resp.input_tokens = response.usage.prompt_tokens or 0
            resp.output_tokens = response.usage.completion_tokens or 0

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            resp.stop_reason = "tool_use"
            resp.raw_content = choice.message
            resp.tool_calls = []
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}
                resp.tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        else:
            resp.stop_reason = "end_turn"
            resp.text = choice.message.content or ""
            resp.raw_content = choice.message

        return resp

    def build_tool_result_message(self, tool_call_id, content):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def build_assistant_message(self, raw_content):
        # raw_content 是 OpenAI 的 ChatCompletionMessage 对象
        return {"role": "assistant", "content": raw_content.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (raw_content.tool_calls or [])
                ] if raw_content.tool_calls else None}


def create_provider(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """工厂函数，根据 provider 名称创建对应的 LLM Provider。

    Args:
        provider: "anthropic" 或 "openai_compatible"
        api_key: API 密钥（可选，默认从环境变量读取）
        base_url: API 地址（仅 openai_compatible 需要）
    """
    if provider == "anthropic":
        return AnthropicProvider(api_key=api_key, base_url=base_url)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(api_key=api_key, base_url=base_url)
    raise ValueError(f"不支持的 provider: {provider}。可选: anthropic, openai_compatible")
