"""对话状态管理。"""

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
        """修剪历史并保留最近几轮完整上下文。"""
        recent_count = RECENT_TURNS_KEEP * 2

        if len(self.history) <= recent_count:
            self._prepend_pinned()
            return

        max_total = (RECENT_TURNS_KEEP + MAX_COMPRESSED_TURNS) * 2
        if len(self.history) > max_total:
            self.history = self.history[-max_total:]

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
                    compressed.append({"role": "assistant", "content": f"[历史] {text[:200]}"})
            elif role == "user" and isinstance(content, str):
                compressed.append({"role": "user", "content": content})

        self.history = compressed + recent
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
