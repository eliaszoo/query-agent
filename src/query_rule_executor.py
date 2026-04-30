"""默认规则执行器。"""

import re
from dataclasses import dataclass, field


@dataclass
class RuleApplication:
    """单条规则命中结果。"""

    business: str
    rule_type: str
    description: str
    applied: bool = True
    overridden: bool = False
    override_reason: str = ""


@dataclass
class QueryRuleResult:
    """规则应用结果。"""

    applications: list[RuleApplication] = field(default_factory=list)
    arguments: dict = field(default_factory=dict)


class QueryRuleExecutor:
    """根据用户输入与默认规则计算命中结果。"""

    AVAILABLE_OVERRIDE_HINTS = ("查全部", "全部数据", "包含禁用", "包含已删除", "所有数据")

    @classmethod
    def apply(cls, user_input: str, business: str, rules: list, arguments: dict | None = None) -> QueryRuleResult:
        result = QueryRuleResult(arguments=dict(arguments or {}))
        normalized = (user_input or "").strip()
        override_available_only = any(hint in normalized for hint in cls.AVAILABLE_OVERRIDE_HINTS)

        for rule in rules:
            rule_type = getattr(rule, "rule_type", "")
            rule_text = getattr(rule, "rule", "")
            payload = getattr(rule, "payload", None) or {}

            if rule_type == "available_only":
                if not override_available_only and result.arguments.get("sql"):
                    result.arguments["sql"] = cls._apply_available_only_sql(
                        result.arguments["sql"], payload
                    )
                result.applications.append(
                    RuleApplication(
                        business=business,
                        rule_type=rule_type,
                        description=rule_text or "默认只查可用数据",
                        applied=not override_available_only,
                        overridden=override_available_only,
                        override_reason="用户本次明确要求查看全部/禁用/已删除数据" if override_available_only else "",
                    )
                )
                continue

            if rule_type == "default_cluster_test":
                if not result.arguments.get("cluster"):
                    result.arguments["cluster"] = payload.get("cluster", "test")
                result.applications.append(
                    RuleApplication(
                        business=business,
                        rule_type=rule_type,
                        description=rule_text or "默认优先查询测试环境",
                        applied=True,
                    )
                )
                continue

            result.applications.append(
                RuleApplication(
                    business=business,
                    rule_type=rule_type or "natural_language",
                    description=rule_text,
                    applied=True,
                )
            )

        return result

    @staticmethod
    def _apply_available_only_sql(sql: str, payload: dict) -> str:
        if not sql:
            return sql

        working = sql.strip().rstrip(";")
        clauses = []
        if payload.get("deleted_at_is_null") and "deleted_at" not in working.lower():
            clauses.append("deleted_at IS NULL")
        forbidden_status_value = payload.get("forbidden_status")
        if forbidden_status_value is not None and "forbidden_status" not in working.lower():
            clauses.append(f"forbidden_status = {forbidden_status_value}")

        if not clauses:
            return working

        order_match = re.search(r"\bORDER\s+BY\b|\bGROUP\s+BY\b|\bLIMIT\b", working, re.IGNORECASE)
        if re.search(r"\bWHERE\b", working, re.IGNORECASE):
            injection = " AND " + " AND ".join(clauses)
            if order_match:
                return working[:order_match.start()] + injection + " " + working[order_match.start():]
            return working + injection

        where_clause = " WHERE " + " AND ".join(clauses)
        if order_match:
            return working[:order_match.start()] + where_clause + " " + working[order_match.start():]
        return working + where_clause
