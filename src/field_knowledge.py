"""字段知识模块 - 持久化记录查询过程中发现的字段含义。

将查询中明确的字段语义（如 tb_voice.origin: 1=自研,2=阿里云）记录到本地 JSON 文件，
下次查询时自动注入到 System Prompt 中，避免 LLM 重复探索字段含义。
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FieldEntry:
    """单条字段知识记录。"""

    business: str        # 业务标识，如 "digitalhuman"
    table: str           # 表名，如 "tb_voice"
    column: str          # 字段名，如 "origin"
    description: str     # 字段含义，如 "音色来源: 1=自研,2=阿里云,3=腾讯云,5=火山引擎"
    timestamp: str = ""  # 记录时间


@dataclass
class FieldKnowledge:
    """字段知识存储。"""

    entries: list[FieldEntry] = field(default_factory=list)


class FieldKnowledgeManager:
    """管理字段知识的持久化读写和 prompt 生成。"""

    def __init__(self, knowledge_path: str):
        self._path = knowledge_path
        self._knowledge = self._load()
        self._schema_cache: dict[tuple[str, str], list[dict]] = {}

    def _load(self) -> FieldKnowledge:
        """从文件加载字段知识。"""
        if not os.path.exists(self._path):
            return FieldKnowledge()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = []
            for item in data.get("entries", []):
                if "business" not in item:
                    item = {"business": "", **item}
                entries.append(FieldEntry(**item))
            return FieldKnowledge(entries=entries)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("字段知识文件解析失败，将重新创建: %s", e)
            return FieldKnowledge()

    def _save(self) -> None:
        """持久化字段知识到文件（原子写入）。"""
        data = {"entries": [asdict(e) for e in self._knowledge.entries]}
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def add_field(self, business: str, table: str, column: str, description: str) -> None:
        """添加或更新一条字段知识。"""
        # 去重：相同 business + table + column 则更新
        for entry in self._knowledge.entries:
            if (
                entry.business == business
                and entry.table == table
                and entry.column == column
            ):
                entry.description = description
                entry.timestamp = datetime.now().isoformat()
                self._save()
                return

        self._knowledge.entries.append(FieldEntry(
            business=business,
            table=table,
            column=column,
            description=description,
            timestamp=datetime.now().isoformat(),
        ))
        self._save()

    def remove_field(self, business: str, table: str, column: str) -> bool:
        """删除一条字段知识，返回是否找到并删除。"""
        before = len(self._knowledge.entries)
        self._knowledge.entries = [
            e for e in self._knowledge.entries
            if not (
                e.business == business and e.table == table and e.column == column
            )
        ]
        if len(self._knowledge.entries) < before:
            self._save()
            return True
        return False

    def get_entries(self, business: str = "") -> list[FieldEntry]:
        """获取字段知识。business 为空时返回全部。"""
        if not business:
            return list(self._knowledge.entries)
        return [e for e in self._knowledge.entries if e.business == business]

    def build_field_prompt(self, business: str = "") -> str:
        """将字段知识转换为可注入 System Prompt 的文本。"""
        entries = self.get_entries(business=business)
        if not entries:
            return ""

        # 按表分组
        table_fields: dict[str, list[FieldEntry]] = {}
        for entry in entries:
            if entry.table not in table_fields:
                table_fields[entry.table] = []
            table_fields[entry.table].append(entry)

        lines = ["\n## 字段含义（已确认，直接使用，无需再查询确认）\n"]
        for table, fields in sorted(table_fields.items()):
            for f in fields:
                lines.append(f"- {f.table}.{f.column}: {f.description}")

        return "\n".join(lines)

    def clear(self) -> None:
        """清空所有字段知识。"""
        self._knowledge = FieldKnowledge()
        self._schema_cache.clear()
        self._save()

    def clear_business(self, business: str) -> None:
        """清除指定业务的字段知识和 schema 缓存。"""
        self._knowledge.entries = [
            e for e in self._knowledge.entries if e.business != business
        ]
        self._schema_cache = {
            key: columns
            for key, columns in self._schema_cache.items()
            if key[0] != business
        }
        self._save()

    # ---- 表结构缓存（内存级，不持久化，每次会话内有效） ----

    def cache_table_schema(self, business: str, table: str, columns: list[dict]) -> None:
        """缓存一张表的列结构。columns 格式: [{"name": "id", "type": "varchar"}, ...]"""
        self._schema_cache[(business, table)] = columns

    def get_cached_schema(self, business: str, table: str) -> list[dict] | None:
        """获取缓存的表结构，无缓存返回 None。"""
        return self._schema_cache.get((business, table))

    def build_schema_prompt(self, business: str = "") -> str:
        """将缓存的表结构转换为可注入 System Prompt 的文本。"""
        if not self._schema_cache:
            return ""

        lines = ["\n## 已知表结构（无需再调用 get_table_schema 查询这些表）\n"]
        filtered_items = self._schema_cache.items()
        if business:
            filtered_items = [
                (key, value) for key, value in filtered_items if key[0] == business
            ]

        if not filtered_items:
            return ""

        for (_, table), columns in sorted(filtered_items):
            col_strs = [f"{c['name']}({c.get('type', '?')})" for c in columns]
            lines.append(f"- {table}: {', '.join(col_strs)}")

        return "\n".join(lines)
