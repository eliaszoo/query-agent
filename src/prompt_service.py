"""System prompt 构建与缓存管理。"""

from src.config import BusinessKnowledge
from src.prompts import build_system_prompt


class PromptService:
    """封装 system prompt 的组装、缓存和失效。

    缓存策略：记录构建时的 current_business，如果 business 变化则自动重建。
    显式 mark_dirty() 用于知识/记忆/规则变化时强制重建。
    """

    def __init__(self) -> None:
        self._cached_prompt = ""
        self._cached_business = None  # 构建时的 current_business
        self._dirty = True

    def mark_dirty(self) -> None:
        """标记 prompt 需要重建。"""
        self._dirty = True

    def build(
        self,
        businesses: list,
        is_stdio_mode: bool,
        current_business: str,
        configured_business_knowledge: BusinessKnowledge,
        field_knowledge_manager,
        error_memory_manager,
        preference_rules_manager,
    ) -> str:
        """构建动态 system prompt。"""
        # 业务变化时自动标记 dirty
        if not self._dirty and self._cached_business != current_business:
            self._dirty = True

        if not self._dirty and self._cached_prompt:
            return self._cached_prompt

        prompt = self._do_build(
            businesses, is_stdio_mode, current_business,
            configured_business_knowledge,
            field_knowledge_manager, error_memory_manager,
            preference_rules_manager,
        )

        self._cached_prompt = prompt
        self._cached_business = current_business
        self._dirty = False
        return prompt

    def build_for_business(
        self,
        business_entry,
        configured_business_knowledge: BusinessKnowledge,
        field_knowledge_manager,
        error_memory_manager,
        preference_rules_manager,
    ) -> str:
        """为单个业务构建 prompt，不使用共享缓存。"""
        knowledge = (
            business_entry.knowledge
            if business_entry and business_entry.knowledge is not None
            else configured_business_knowledge
        )
        prompt = build_system_prompt(
            [business_entry] if business_entry else [],
            {business_entry.name: knowledge} if business_entry and knowledge else {},
        )

        business_name = business_entry.name if business_entry else "default"

        field_prompt = field_knowledge_manager.build_field_prompt(business_name)
        if field_prompt:
            prompt += field_prompt

        schema_prompt = field_knowledge_manager.build_schema_prompt(business_name)
        if schema_prompt:
            prompt += schema_prompt

        memory_prompt = error_memory_manager.build_memory_prompt(business_name)
        if memory_prompt:
            prompt += memory_prompt

        rules_prompt = preference_rules_manager.build_rules_prompt(business_name)
        if rules_prompt:
            prompt += rules_prompt

        return prompt

    @staticmethod
    def _do_build(
        businesses: list,
        is_stdio_mode: bool,
        current_business: str,
        configured_business_knowledge: BusinessKnowledge,
        field_knowledge_manager,
        error_memory_manager,
        preference_rules_manager,
    ) -> str:
        """实际构建 prompt 的逻辑。"""
        knowledge_map: dict[str, BusinessKnowledge] = {}
        for entry in businesses:
            if entry.knowledge is not None:
                knowledge_map[entry.name] = entry.knowledge

        if is_stdio_mode and configured_business_knowledge.description:
            knowledge_map["default"] = configured_business_knowledge

        prompt = build_system_prompt(businesses, knowledge_map)

        field_prompt = field_knowledge_manager.build_field_prompt(current_business)
        if field_prompt:
            prompt += field_prompt

        schema_prompt = field_knowledge_manager.build_schema_prompt(current_business)
        if schema_prompt:
            prompt += schema_prompt

        memory_prompt = error_memory_manager.build_memory_prompt(current_business)
        if memory_prompt:
            prompt += memory_prompt

        rules_prompt = preference_rules_manager.build_rules_prompt(current_business)
        if rules_prompt:
            prompt += rules_prompt

        return prompt
