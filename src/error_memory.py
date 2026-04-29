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

MAX_MEMORY_ENTRIES = 50  # 最多保留的错误记录数
MAX_MEMORY_TOKENS = 500  # 错误记忆的 token 预算
CHARS_PER_TOKEN = 1.5    # 中文粗略估算：1 字 ≈ 1.5 token


@dataclass
class ErrorEntry:
    """单条错误记录。"""

    timestamp: str
    user_query: str
    error_type: str  # "SQL_REJECTED", "QUERY_FAILED", "USER_CORRECTION", "USER_FEEDBACK", "WRONG_TABLE"
    business: str = ""  # 关联的业务标识（空字符串表示通用经验）
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

    def __init__(self, memory_path: str):
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
        """持久化错误记忆到文件（原子写入，避免并发写冲突）。"""
        data = {"entries": [asdict(e) for e in self._memory.entries]}
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)  # 原子替换，跨平台安全

    def add_error(
        self,
        user_query: str,
        error_type: str,
        business: str = "",
        bad_sql: Optional[str] = None,
        error_message: Optional[str] = None,
        corrected_sql: Optional[str] = None,
        lesson: str = "",
    ) -> None:
        """记录一条错误（自动去重）。"""
        # 去重：相同 business + lesson 的经验不重复记录
        if lesson:
            for existing in self._memory.entries:
                if existing.business == business and existing.lesson == lesson:
                    logger.debug("经验已存在，跳过: %s", lesson)
                    return

        entry = ErrorEntry(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            error_type=error_type,
            business=business,
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

    def build_memory_prompt(self, current_business: str = "") -> str:
        """将错误记忆转换为可注入 System Prompt 的文本。

        只注入经验教训（lesson），不注入原始查询和 SQL，减少 token 消耗。
        按 token 预算控制总量，从最新的经验开始倒序填充。

        Args:
            current_business: 当前查询的业务标识，为空则返回全部经验。
        """
        entries = self._memory.entries
        if not entries:
            return ""

        # 过滤业务：只保留通用经验（business 为空）和当前业务的经验
        if current_business:
            entries = [e for e in entries if not e.business or e.business == current_business]

        # 只注入有 lesson 的条目
        lessons = [e for e in entries if e.lesson]
        if not lessons:
            return ""

        budget_chars = int(MAX_MEMORY_TOKENS * CHARS_PER_TOKEN)
        lines = ["\n## 历史经验（请避免重复犯错）\n"]
        total_chars = 0

        # 从最新的开始，倒序填充
        for entry in reversed(lessons):
            prefix = f"[{entry.business}] " if entry.business else ""
            line = f"- {prefix}{entry.lesson}"

            if total_chars + len(line) > budget_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if len(lines) <= 1:
            return ""

        # 恢复正序（最早在前）
        lines = [lines[0]] + list(reversed(lines[1:]))
        return "\n".join(lines)

    def clear(self, business: str = "") -> None:
        """清空错误记忆。

        Args:
            business: 指定业务则只清除该业务的记忆，为空则清空全部。
        """
        if not business:
            self._memory = ErrorMemory()
        else:
            self._memory.entries = [
                e for e in self._memory.entries if e.business != business
            ]
        self._save()

    def get_businesses(self) -> list[str]:
        """获取所有有错误记忆的业务标识列表（去重）。"""
        return list(dict.fromkeys(e.business for e in self._memory.entries if e.business))
