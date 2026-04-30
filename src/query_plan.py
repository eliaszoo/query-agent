"""查询计划结构。"""

from dataclasses import dataclass, field


@dataclass
class QueryPlan:
    """单次查询的计划预览。"""

    user_input: str
    business: str = ""
    business_display_name: str = ""
    business_strategy: str = ""
    business_reason: str = ""
    locked_business: str = ""
    default_cluster: str = ""
    active_rules: list[str] = field(default_factory=list)
    overridden_rules: list[str] = field(default_factory=list)
