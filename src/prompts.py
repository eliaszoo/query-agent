"""System Prompt 构建 - 通用框架 + 多业务领域知识动态注入。"""

from src.config import BusinessKnowledge
from src.business_registry import BusinessEntry


def build_system_prompt(
    businesses: list[BusinessEntry] | None = None,
    knowledge_map: dict[str, BusinessKnowledge] | None = None,
) -> str:
    """构建 system prompt，支持多业务模式。

    Args:
        businesses: 已注册的业务列表。
        knowledge_map: 业务名 → 业务知识映射。

    Returns:
        完整的 system prompt 字符串。
    """
    businesses = businesses or []
    knowledge_map = knowledge_map or {}
    is_multi = len(businesses) > 1

    if is_multi:
        return _build_multi_business_prompt(businesses, knowledge_map)

    # 单业务模式（向后兼容）
    bk = next(iter(knowledge_map.values()), None) if knowledge_map else None
    return _build_single_business_prompt(bk)


def _build_single_business_prompt(business_knowledge: BusinessKnowledge | None = None) -> str:
    """构建单业务 system prompt。"""
    bk = business_knowledge or BusinessKnowledge()
    description = bk.description or "业务"

    parts = [
        f"你是一个{description}数据查询助手。你可以帮助运营人员查询{description}的业务数据。",
        "",
        "## 你的能力",
        "- 将自然语言转换为 SQL 查询",
        f"- 查询{description}相关业务数据",
        "- 支持多集群查询（测试/生产环境）",
    ]

    if bk.term_mappings:
        parts.append("")
        parts.append("## 业务术语映射")
        for term, mapping in bk.term_mappings.items():
            parts.append(f'- "{term}" → {mapping}')

    if bk.table_relationships:
        parts.append("")
        parts.append("## 核心表关系")
        for rel in bk.table_relationships:
            parts.append(f"- {rel}")

    if bk.status_codes:
        parts.append("")
        parts.append("## 常用状态码")
        for code in bk.status_codes:
            parts.append(f"- {code}")

    parts.append("")
    parts.append("## 查询规则")
    parts.append("1. 始终使用 execute_readonly_sql 工具执行查询")
    parts.append("2. 首次查询某张表时，必须先调用 get_table_schema 了解列名，不要猜测列名")
    parts.append("3. 使用 get_cluster_list 查看可用集群")
    parts.append("4. 默认查询测试集群，用户明确指定时才查询生产集群")
    parts.append("5. 注意 deleted_at IS NULL 条件过滤已删除数据")
    parts.append("6. 结果以表格形式呈现，附带中文字段说明")
    parts.append("7. 回复末尾用 HTML 注释声明字段含义，格式：")
    parts.append("   <!-- FIELD_KNOWLEDGE: [{\"table\":\"tb_xxx\",\"field\":\"yyy\",\"values\":\"1=正常,2=禁用\"}] -->")
    parts.append("   仅声明枚举/状态类字段，普通字段无需声明。table 取自 SQL 的 FROM 表名。")

    if bk.custom_rules:
        for i, rule in enumerate(bk.custom_rules, 7):
            parts.append(f"{i}. {rule}")

    return "\n".join(parts)


def _build_multi_business_prompt(
    businesses: list[BusinessEntry],
    knowledge_map: dict[str, BusinessKnowledge],
) -> str:
    """构建多业务 system prompt。"""
    parts = [
        "你是一个多业务数据查询助手。你可以帮助运营人员查询不同业务的数据。",
        "",
        "## 可查询的业务",
    ]

    for entry in businesses:
        bk = knowledge_map.get(entry.name)
        desc = bk.description if bk and bk.description else entry.display_name
        parts.append(f"- **{entry.name}** ({entry.display_name}): {desc}")

    parts.append("")
    parts.append("查询时请在工具调用中指定 business 参数选择目标业务。")

    # 每个业务的领域知识
    for entry in businesses:
        bk = knowledge_map.get(entry.name)
        if not bk:
            continue

        has_content = bk.term_mappings or bk.table_relationships or bk.status_codes or bk.custom_rules
        if not has_content:
            continue

        parts.append("")
        parts.append(f"## {entry.display_name} 业务知识")

        if bk.term_mappings:
            parts.append("")
            parts.append("### 业务术语映射")
            for term, mapping in bk.term_mappings.items():
                parts.append(f'- "{term}" → {mapping}')

        if bk.table_relationships:
            parts.append("")
            parts.append("### 核心表关系")
            for rel in bk.table_relationships:
                parts.append(f"- {rel}")

        if bk.status_codes:
            parts.append("")
            parts.append("### 常用状态码")
            for code in bk.status_codes:
                parts.append(f"- {code}")

        if bk.custom_rules:
            parts.append("")
            parts.append("### 自定义规则")
            for rule in bk.custom_rules:
                parts.append(f"- {rule}")

    # 查询规则
    parts.append("")
    parts.append("## 查询规则")
    parts.append("1. 始终使用 execute_readonly_sql 工具执行查询，必须指定 business 参数")
    parts.append("2. 首次查询某张表时，必须先调用 get_table_schema 了解列名（需指定 business 参数），不要猜测列名")
    parts.append("3. 使用 get_cluster_list 查看可用集群，必须指定 business 参数")
    parts.append("4. 默认查询测试集群，用户明确指定时才查询生产集群")
    parts.append("5. 注意 deleted_at IS NULL 条件过滤已删除数据")
    parts.append("6. 结果以表格形式呈现，附带中文字段说明")
    parts.append("7. 回复末尾用 HTML 注释声明字段含义，格式：")
    parts.append("   <!-- FIELD_KNOWLEDGE: [{\"table\":\"tb_xxx\",\"field\":\"yyy\",\"values\":\"1=正常,2=禁用\"}] -->")
    parts.append("   仅声明枚举/状态类字段，普通字段无需声明。table 取自 SQL 的 FROM 表名。")

    return "\n".join(parts)
