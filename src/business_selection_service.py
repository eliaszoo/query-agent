"""业务选择服务。"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BusinessSelectionResult:
    """业务选择结果。"""

    business: object | None
    strategy: str
    reason: str = ""


class BusinessSelectionService:
    """封装启发式与 LLM 兜底的业务选择逻辑。"""

    def __init__(self, provider, model: str, registry) -> None:
        self._provider = provider
        self._model = model
        self._registry = registry

    async def select_business(self, user_input: str):
        """根据用户输入选择目标业务。"""
        businesses = self._registry.list_businesses()
        if len(businesses) <= 1:
            return BusinessSelectionResult(
                business=businesses[0] if businesses else None,
                strategy="single" if businesses else "fallback_all",
                reason="只有一个可用业务，直接使用" if businesses else "没有可用业务，回退到全业务模式",
            )

        heuristic_match = self._heuristic_select(user_input, businesses)
        if heuristic_match is not None:
            return BusinessSelectionResult(
                business=heuristic_match,
                strategy="heuristic",
                reason=f"用户输入命中业务名或显示名：{heuristic_match.display_name}",
            )

        try:
            llm_match = self._select_with_llm(user_input, businesses)
            if llm_match is not None:
                return BusinessSelectionResult(
                    business=llm_match,
                    strategy="llm",
                    reason=f"LLM 判断最相关业务为：{llm_match.display_name}",
                )
        except Exception:
            logger.debug("LLM 业务选择失败，回退到全业务模式", exc_info=True)

        return BusinessSelectionResult(
            business=None,
            strategy="fallback_all",
            reason="未能唯一识别目标业务，回退到全业务模式",
        )

    def _heuristic_select(self, user_input: str, businesses: list):
        text = user_input.lower()
        matches = []
        for entry in businesses:
            candidates = {entry.name.lower(), entry.display_name.lower()}
            if entry.knowledge and entry.knowledge.description:
                candidates.add(entry.knowledge.description.lower())
            if any(candidate and candidate in text for candidate in candidates):
                matches.append(entry)

        if len(matches) == 1:
            return matches[0]
        return None

    def _select_with_llm(self, user_input: str, businesses: list):
        business_lines = []
        for entry in businesses:
            description = (
                entry.knowledge.description
                if entry.knowledge and entry.knowledge.description
                else entry.display_name
            )
            business_lines.append(
                f"- {entry.name}: {entry.display_name} | {description}"
            )

        prompt = (
            "请根据用户查询，从以下业务中选择最相关的一个。\n"
            "如果无法判断或查询明显跨业务，请回答 NONE。\n\n"
            "业务列表：\n"
            f"{chr(10).join(business_lines)}\n\n"
            f"用户查询：{user_input}\n\n"
            "只回答业务 name 或 NONE，不要输出其他内容。"
        )
        response = self._provider.chat(
            model=self._model,
            max_tokens=50,
            system="你是一个业务路由助手，只做业务识别。",
            tools=[],
            messages=[{"role": "user", "content": prompt}],
        )
        selected_name = response.text.strip()
        if not selected_name or selected_name.upper() == "NONE":
            return None

        normalized = selected_name.strip("`").strip()
        for entry in businesses:
            if entry.name == normalized:
                return entry
        return None
