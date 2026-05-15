"""对话状态管理。"""

import json
from dataclasses import dataclass, field

RECENT_TURNS_KEEP = 3
MAX_COMPRESSED_TURNS = 5


@dataclass
class ConversationState:
    """管理对话历史、置顶消息和最近查询上下文。"""

    history: list[dict] = field(default_factory=list)
    pinned_messages: list[dict] = field(default_factory=list)
    last_query_context: dict | None = None
    locked_business: str = ""

    def clear_history(self) -> None:
        """清空对话历史，保留置顶消息。"""
        self.history.clear()

    def pin_message(self, content: str) -> None:
        """置顶一条消息。"""
        self.pinned_messages.append({"role": "user", "content": f"[置顶] {content}"})

    def trim_history(self) -> None:
        """修剪历史并保留最近几轮完整上下文。

        确保截断后不会出现孤立的 tool 消息（没有对应 assistant tool_calls）。
        """
        recent_count = RECENT_TURNS_KEEP * 2

        if len(self.history) <= recent_count:
            self._prepend_pinned()
            return

        max_total = (RECENT_TURNS_KEEP + MAX_COMPRESSED_TURNS) * 2
        if len(self.history) > max_total:
            self.history = self.history[-max_total:]
            self._repair_orphaned_tool_messages()

        if len(self.history) <= recent_count:
            self._prepend_pinned()
            return

        recent = self.history[-recent_count:]
        older = self.history[:-recent_count]

        compressed = []
        for msg in older:
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                text = self.extract_text_from_content(content)
                if text:
                    first_line = text.split("\n", 1)[0]
                    if len(first_line) > 500:
                        summary = first_line[:500] + "..."
                    elif len(text) > 500:
                        summary = first_line + "\n..."
                    else:
                        summary = text
                    compressed.append({"role": "assistant", "content": f"[历史] {summary}"})
            elif role == "user" and isinstance(content, str):
                compressed.append({"role": "user", "content": content})

        self.history = compressed + recent
        self._repair_orphaned_tool_messages()
        self._prepend_pinned()

    def _prepend_pinned(self) -> None:
        """将置顶消息放回历史开头。"""
        if not self.pinned_messages:
            return
        self.history = [
            m for m in self.history
            if not (isinstance(m.get("content"), str) and m["content"].startswith("[置顶] "))
        ]
        self.history = self.pinned_messages + self.history

    def _repair_orphaned_tool_messages(self) -> None:
        """修复消息序列中的工具调用不一致问题。

        处理两种情况：
        1. 孤立的 tool 消息：删除没有对应 assistant(tool_calls) 的 tool 消息
        2. 缺失的 tool 结果：assistant(tool_calls) 后缺少对应 tool 消息时，补充 stub 结果

        截断/压缩历史后可能出现：
        - tool 消息前没有 assistant 消息包含 tool_calls → 删除
        - assistant(tool_calls) 后缺少部分或全部 tool 结果消息 → 补充 stub
        """
        cleaned: list[dict] = []
        i = 0

        while i < len(self.history):
            msg = self.history[i]
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                # 收集期望的 tool_call_id
                expected_ids: set[str] = set()
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        expected_ids.add(tc_id)

                # 同时检查 Anthropic 格式的 content 中的 tool_use
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tc_id = block.get("id", "")
                            if tc_id:
                                expected_ids.add(tc_id)

                cleaned.append(msg)

                # 收集紧随其后的 tool 消息
                found_ids: set[str] = set()
                j = i + 1
                while j < len(self.history) and self.history[j].get("role") == "tool":
                    tool_msg = self.history[j]
                    tool_id = tool_msg.get("tool_call_id", "")
                    if tool_id in expected_ids:
                        found_ids.add(tool_id)
                        cleaned.append(tool_msg)
                    # else: 孤立的 tool 消息（ID 不匹配当前 assistant），跳过
                    j += 1

                # 为缺失的 tool_call_id 补充 stub 结果
                for missing_id in expected_ids - found_ids:
                    cleaned.append({
                        "role": "tool",
                        "tool_call_id": missing_id,
                        "content": json.dumps(
                            {"success": False, "error_message": "工具结果因历史裁剪丢失"},
                            ensure_ascii=False,
                        ),
                    })

                i = j

            elif role == "tool":
                # 孤立的 tool 消息（没有对应的 assistant(tool_calls)），跳过
                i += 1

            else:
                cleaned.append(msg)
                i += 1

        self.history = cleaned

    @staticmethod
    def extract_text_from_content(content) -> str:
        """从 assistant content 中提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts)
        return ""
