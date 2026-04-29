"""知识与本地状态存储聚合。"""

import json
import logging
import re

from src.error_memory import ErrorMemoryManager
from src.field_knowledge import FieldKnowledgeManager

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """聚合错误记忆、字段知识以及相关本地写入逻辑。"""

    TABLE_FIELD_PATTERN = re.compile(
        r'(tb_\w+)\.(\w+)\s*[：:]\s*'
        r'((?:\d+\s*[（(]\s*[^）)]+\s*[）)]\s*[，,]?\s*)+)',
        re.UNICODE,
    )
    FIELD_ENUM_PATTERN = re.compile(
        r'(\w+)\s*[：:]\s*'
        r'((?:\d+\s*[（(]\s*[^）)]+\s*[）)]\s*[，,]?\s*)+)',
        re.UNICODE,
    )
    FIELD_EQ_PATTERN = re.compile(
        r'\*{0,2}(?:\S+?\s*)?\((\w+)\)\*{0,2}\s*[：:]\s*'
        r'((?:\d+\s*=\s*[^,，\n]+(?:\s*[，,]\s*|\s*))+)',
        re.UNICODE,
    )
    FIELD_KNOWLEDGE_TAG = re.compile(
        r'<!--\s*FIELD_KNOWLEDGE:\s*(\[[\s\S]*?\])\s*-->',
        re.UNICODE,
    )

    def __init__(
        self,
        error_memory: ErrorMemoryManager,
        field_knowledge: FieldKnowledgeManager,
        mark_prompt_dirty,
    ) -> None:
        self.error_memory = error_memory
        self.field_knowledge = field_knowledge
        self._mark_prompt_dirty = mark_prompt_dirty

    def clear_business(self, business: str) -> None:
        self.field_knowledge.clear_business(business)
        self._mark_prompt_dirty()

    def set_error_memory(self, error_memory: ErrorMemoryManager) -> None:
        self.error_memory = error_memory

    def set_field_knowledge(self, field_knowledge: FieldKnowledgeManager) -> None:
        self.field_knowledge = field_knowledge

    def add_field_knowledge(
        self, business: str, table: str, column: str, description: str
    ) -> None:
        self.field_knowledge.add_field(business, table, column, description)
        self._mark_prompt_dirty()

    def remove_field_knowledge(self, business: str, table: str, column: str) -> bool:
        removed = self.field_knowledge.remove_field(business, table, column)
        if removed:
            self._mark_prompt_dirty()
        return removed

    def list_field_knowledge(self, business: str = ""):
        return self.field_knowledge.get_entries(business=business)

    def clear_error_memory(self, business: str = "") -> None:
        self.error_memory.clear(business=business)
        self._mark_prompt_dirty()

    def get_error_memory_entries(self):
        return self.error_memory.get_entries()

    def get_error_memory_businesses(self):
        return self.error_memory.get_businesses()

    def record_feedback(
        self, original_query: str, business: str, user_feedback: str, lesson: str
    ) -> None:
        self.error_memory.add_error(
            user_query=original_query,
            error_type="USER_FEEDBACK",
            business=business,
            error_message=user_feedback,
            lesson=lesson,
        )
        self._mark_prompt_dirty()

    def check_and_record_error(
        self,
        user_query: str,
        tool_input: dict,
        result_text: str,
        business: str = "",
        is_stdio_mode: bool = False,
        lesson_builder=None,
    ) -> None:
        try:
            result = json.loads(result_text)
        except (ValueError, TypeError):
            return

        if not isinstance(result, dict) or result.get("success", True):
            return

        error_type = result.get("error_type", "UNKNOWN")
        error_message = result.get("error_message", "")

        skip_error_types = {
            "CONNECTION_ERROR",
            "CONFIG_ERROR",
            "POOL_ERROR",
            "TIMEOUT_ERROR",
            "QUERY_ERROR",
        }
        if error_type in skip_error_types:
            logger.debug("跳过错误，不记录: %s - %s", error_type, error_message)
            return

        bad_sql = tool_input.get("sql", "") if isinstance(tool_input, dict) else ""
        if not business and is_stdio_mode:
            business = "default"

        lesson = lesson_builder(error_type, error_message, bad_sql) if lesson_builder else ""
        self.error_memory.add_error(
            user_query=user_query,
            error_type=error_type,
            business=business,
            bad_sql=bad_sql,
            error_message=error_message,
            lesson=lesson,
        )
        self._mark_prompt_dirty()
        logger.info("已记录错误到记忆: %s - %s", error_type, lesson)

    def auto_extract_field_knowledge(
        self,
        response_text: str,
        business: str,
        sql: str,
    ) -> None:
        dirty = False

        for match in self.FIELD_KNOWLEDGE_TAG.finditer(response_text):
            try:
                items = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                items = []
            for item in items:
                table = item.get("table", "")
                field = item.get("field", "")
                values = item.get("values", "")
                if table and field and values:
                    self.field_knowledge.add_field(business, table, field, values)
                    dirty = True
                    logger.info("提取字段知识(结构化): %s.%s: %s", table, field, values)
            if dirty:
                self._mark_prompt_dirty()
                return

        self._auto_extract_field_knowledge_fallback(response_text, business, sql)

    def _auto_extract_field_knowledge_fallback(
        self, response_text: str, business: str, sql: str
    ) -> None:
        dirty = False

        for match in self.FIELD_EQ_PATTERN.finditer(response_text):
            column = match.group(1)
            raw_values = match.group(2)
            description = self.parse_eq_values(raw_values)
            if description:
                table = self.infer_table_from_sql(sql)
                if table:
                    self.field_knowledge.add_field(business, table, column, description)
                    dirty = True
                    logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        for match in self.TABLE_FIELD_PATTERN.finditer(response_text):
            table = match.group(1)
            column = match.group(2)
            raw_values = match.group(3)
            description = self.parse_enum_values(raw_values)
            if description:
                self.field_knowledge.add_field(business, table, column, description)
                dirty = True
                logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        if not dirty:
            for match in self.FIELD_ENUM_PATTERN.finditer(response_text):
                column = match.group(1)
                raw_values = match.group(2)
                description = self.parse_enum_values(raw_values)
                if description and not column.startswith("tb_"):
                    table = self.infer_table_from_sql(sql)
                    if table:
                        self.field_knowledge.add_field(business, table, column, description)
                        dirty = True
                        logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        if dirty:
            self._mark_prompt_dirty()

    @staticmethod
    def infer_table_from_sql(sql: str) -> str:
        if not sql:
            return ""
        table_match = re.search(r"FROM\s+(tb_\w+)", sql, re.IGNORECASE)
        return table_match.group(1) if table_match else ""

    @staticmethod
    def parse_enum_values(raw: str) -> str:
        parts = re.findall(r"(\d+)\s*[（(]\s*([^）)]+)\s*[）)]", raw)
        if not parts:
            return ""
        return ", ".join(f"{num}={label.strip()}" for num, label in parts)

    @staticmethod
    def parse_eq_values(raw: str) -> str:
        parts = re.findall(r"(\d+)\s*=\s*([^,，\n]+)", raw)
        if not parts:
            return ""
        return ", ".join(f"{num}={label.strip()}" for num, label in parts)
