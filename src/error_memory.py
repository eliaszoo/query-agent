"""错误记忆模块 - 持久化记录 Agent 犯过的错误，避免重复犯错。

将查询过程中的错误（SQL 被拒绝、查询失败、用户纠正等）记录到本地 JSON 文件，
下次查询时自动注入到 System Prompt 中，让 Agent 从历史错误中学习。
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = "./error_memory.json"
MAX_MEMORY_ENTRIES = 50  # 最多保留的错误记录数


@dataclass
class ErrorEntry:
    """单条错误记录。"""

    timestamp: str
    user_query: str
    error_type: str  # "SQL_REJECTED", "QUERY_FAILED", "USER_CORRECTION", "WRONG_TABLE"
    bad_sql: Optional[str] = None
    error_message: Optional[str] = None
    corrected_sql: Optional[str] = None
    lesson: str = ""  # Agent 从这次错误中学到的经验


@dataclass
class ErrorMemory:
    """错误记忆存储。"""

    entries: list[ErrorEntry] = field(default_factory=list)


class ErrorMemoryManager:
    """管理错误记忆的持久化读写和 prompt 生成。"""

    def __init__(self, memory_path: str = DEFAULT_MEMORY_PATH):
        self._path = memory_path
        self._memory = self._load()

    def _load(self) -> ErrorMemory:
        """从文件加载错误记忆。"""
        if not os.path.exists(self._path):
            return ErrorMemory()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = [ErrorEntry(**e) for e in data.get("entries", [])]
            return ErrorMemory(entries=entries)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("错误记忆文件解析失败，将重新创建: %s", e)
            return ErrorMemory()

    def _save(self) -> None:
        """持久化错误记忆到文件。"""
        data = {"entries": [asdict(e) for e in self._memory.entries]}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_error(
        self,
        user_query: str,
        error_type: str,
        bad_sql: Optional[str] = None,
        error_message: Optional[str] = None,
        corrected_sql: Optional[str] = None,
        lesson: str = "",
    ) -> None:
        """记录一条错误。"""
        entry = ErrorEntry(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            error_type=error_type,
            bad_sql=bad_sql,
            error_message=error_message,
            corrected_sql=corrected_sql,
            lesson=lesson,
        )
        self._memory.entries.append(entry)

        # 超过上限时淘汰最旧的记录
        if len(self._memory.entries) > MAX_MEMORY_ENTRIES:
            self._memory.entries = self._memory.entries[-MAX_MEMORY_ENTRIES:]

        self._save()

    def get_entries(self) -> list[ErrorEntry]:
        """获取所有错误记录。"""
        return list(self._memory.entries)

    def build_memory_prompt(self) -> str:
        """将错误记忆转换为可注入 System Prompt 的文本。

        只取最近的记录，按错误类型分组，生成简洁的经验总结。
        """
        entries = self._memory.entries
        if not entries:
            return ""

        lines = ["\n## 历史错误经验（请务必避免重复犯错）\n"]

        for i, entry in enumerate(entries[-20:], 1):  # 最多注入最近 20 条
            lines.append(f"### 经验 {i}")
            lines.append(f"- 用户查询: {entry.user_query}")
            if entry.bad_sql:
                lines.append(f"- 错误 SQL: `{entry.bad_sql}`")
            if entry.error_message:
                lines.append(f"- 错误原因: {entry.error_message}")
            if entry.corrected_sql:
                lines.append(f"- 正确 SQL: `{entry.corrected_sql}`")
            if entry.lesson:
                lines.append(f"- 教训: {entry.lesson}")
            lines.append("")

        return "\n".join(lines)

    def clear(self) -> None:
        """清空所有错误记忆。"""
        self._memory = ErrorMemory()
        self._save()
