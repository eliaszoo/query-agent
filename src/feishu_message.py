"""飞书消息格式化 — 将 QueryAgent 输出转换为飞书卡片消息 JSON。"""

import json
import logging
import re

from src.agent import QueryMetrics

logger = logging.getLogger(__name__)

# 匹配 Markdown 表格：| col1 | col2 | \n |---|---|\n | v1 | v2 |
_MD_TABLE_PATTERN = re.compile(
    r"(\|.+\|)\s*\n(\|[-:\s|]+\|)\s*\n((?:\|.+\|\s*\n?)+)",
    re.MULTILINE,
)


def _parse_markdown_table(md_text: str) -> list[list[str]] | None:
    """从 Markdown 文本中提取第一个表格的数据。

    Returns:
        二维列表 [[header1, header2, ...], [row1_col1, row1_col2, ...], ...]
        或 None（未找到表格）。
    """
    match = _MD_TABLE_PATTERN.search(md_text)
    if not match:
        return None

    def _split_row(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    header = _split_row(match.group(1))
    rows = []
    for row_line in match.group(3).strip().split("\n"):
        row_line = row_line.strip()
        if row_line:
            rows.append(_split_row(row_line))

    return [header] + rows


def build_query_card(result: str, metrics: QueryMetrics | None = None, context: dict | None = None) -> str:
    """将查询结果构建为飞书交互卡片 JSON 字符串。

    Args:
        result: QueryAgent 的文本输出。
        metrics: 查询元信息（可选）。
        context: 查询上下文（含 cluster 等，可选）。

    Returns:
        飞书卡片消息的 content JSON 字符串。
    """
    # 尝试解析 Markdown 表格
    table_data = _parse_markdown_table(result)

    if table_data and len(table_data) >= 2:
        card = _build_table_card(table_data, result, metrics, context)
    else:
        card = _build_text_card(result, metrics, context)

    return json.dumps(card, ensure_ascii=False)


def _build_metrics_header(title: str, metrics: QueryMetrics | None, context: dict | None = None) -> dict:
    """构建卡片 header，包含查询元信息。"""
    if metrics:
        subtitle_parts = []
        if metrics.selected_business:
            biz = metrics.selected_business
            if metrics.business_selection_strategy:
                biz += f" via {metrics.business_selection_strategy}"
            subtitle_parts.append(f"业务: {biz}")
        if context and context.get("cluster"):
            subtitle_parts.append(f"集群: {context['cluster']}")
        if metrics.duration_seconds:
            subtitle_parts.append(f"耗时: {metrics.duration_seconds}s")
        if metrics.input_tokens or metrics.output_tokens:
            subtitle_parts.append(f"{metrics.input_tokens}+ {metrics.output_tokens}-")
        if metrics.tool_calls:
            subtitle_parts.append(f"{metrics.tool_calls} tool calls")
        if metrics.model:
            subtitle_parts.append(f"模型: {metrics.model}")
        subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""
    else:
        subtitle = ""

    header = {
        "title": {"tag": "plain_text", "content": title},
        "template": "blue",
    }
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": subtitle}

    return header


def _build_table_card(
    table_data: list[list[str]], raw_result: str, metrics: QueryMetrics | None,
    context: dict | None = None,
) -> dict:
    """构建包含表格的飞书卡片。"""
    header_row = table_data[0]
    data_rows = table_data[1:]
    col_count = len(header_row)

    # 限制展示行数
    max_display_rows = 50
    truncated = len(data_rows) > max_display_rows
    display_rows = data_rows[:max_display_rows]

    elements: list[dict] = []

    # 表格使用 column_set 布局
    # 先添加表头行
    column_set = {"tag": "column_set", "columns": []}
    for i, col_name in enumerate(header_row):
        column_set["columns"].append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{col_name}**",
                    },
                }
            ],
        })
    elements.append(column_set)

    # 数据行
    for row in display_rows:
        row_set = {"tag": "column_set", "columns": []}
        for i in range(col_count):
            cell_value = row[i] if i < len(row) else ""
            row_set["columns"].append({
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": cell_value,
                        },
                    }
                ],
            })
        elements.append(row_set)

    # 截断提示
    if truncated:
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"_仅展示前 {max_display_rows} 行，共 {len(data_rows)} 行_",
            },
        })

    # 表格外的额外文本（表格之前或之后的说明）
    table_match = _MD_TABLE_PATTERN.search(raw_result)
    if table_match:
        before_text = raw_result[: table_match.start()].strip()
        after_text = raw_result[table_match.end() :].strip()

        extra_parts = []
        if before_text:
            extra_parts.append(before_text)
        if after_text:
            extra_parts.append(after_text)
        if extra_parts:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n\n".join(extra_parts),
                },
            })

    return {
        "config": {"wide_screen_mode": True},
        "header": _build_metrics_header("查询结果", metrics, context),
        "elements": elements,
    }


def _build_text_card(result: str, metrics: QueryMetrics | None, context: dict | None = None) -> dict:
    """构建纯文本飞书卡片。"""
    return {
        "config": {"wide_screen_mode": True},
        "header": _build_metrics_header("查询结果", metrics, context),
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": result,
                },
            }
        ],
    }


def build_command_card(title: str, content: str) -> str:
    """构建命令结果卡片。

    Args:
        title: 卡片标题（如 "MCP Servers & Businesses"）。
        content: Markdown 格式的命令输出。

    Returns:
        飞书卡片消息的 content JSON 字符串。
    """
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "green",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": content,
                },
            }
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def build_error_card(error_message: str) -> str:
    """构建错误提示卡片。

    Args:
        error_message: 错误信息。

    Returns:
        飞书卡片消息的 content JSON 字符串。
    """
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "查询出错"},
            "template": "red",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": error_message,
                },
            }
        ],
    }
    return json.dumps(card, ensure_ascii=False)


def build_thinking_card() -> str:
    """构建"正在查询"提示卡片。

    Returns:
        飞书卡片消息的 content JSON 字符串。
    """
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "正在查询..."},
            "template": "blue",
        },
        "elements": [],
    }
    return json.dumps(card, ensure_ascii=False)


def build_sql_preview_card(
    sql: str, cluster: str = "",
    risk_level: str = "", risk_reasons: list[str] | None = None,
) -> str:
    """构建 SQL 预览卡片。

    Args:
        sql: 将要执行的 SQL 语句。
        cluster: 目标集群。
        risk_level: 风险等级 ("high"/"medium"/"")。
        risk_reasons: 风险原因列表。

    Returns:
        飞书卡片消息的 content JSON 字符串。
    """
    # 颜色根据风险等级
    template = {"high": "red", "medium": "orange"}.get(risk_level, "blue")
    title = {"high": "高风险 SQL 预览", "medium": "SQL 预览（中风险）"}.get(
        risk_level, "SQL 预览"
    )

    elements: list[dict] = []

    # SQL 代码块
    sql_display = sql[:2000]  # 限制长度
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"**集群**: {cluster or '默认'}\n```sql\n{sql_display}\n```",
        },
    })

    # 风险提示
    if risk_reasons:
        risk_lines = "\n".join(f"- {r}" for r in risk_reasons)
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**风险 [{risk_level}]**:\n{risk_lines}",
            },
        })

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": elements,
    }, ensure_ascii=False)


def build_risk_confirm_card(
    sql: str, cluster: str = "",
    risk_level: str = "high", risk_reasons: list[str] | None = None,
    confirm_id: str = "",
) -> str:
    """构建高风险 SQL 确认卡片（带按钮）。

    Args:
        sql: 将要执行的 SQL 语句。
        cluster: 目标集群。
        risk_level: 风险等级。
        risk_reasons: 风险原因列表。
        confirm_id: 确认 ID，用于按钮回调路由。

    Returns:
        飞书交互卡片消息的 content JSON 字符串。
    """
    elements: list[dict] = []

    # SQL 代码块
    sql_display = sql[:2000]
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"**集群**: {cluster or '默认'}\n```sql\n{sql_display}\n```",
        },
    })

    # 风险提示
    if risk_reasons:
        risk_lines = "\n".join(f"- {r}" for r in risk_reasons)
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**风险 [{risk_level}]**:\n{risk_lines}",
            },
        })

    # 确认按钮
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "继续执行"},
                "type": "danger",
                "value": {"confirm_id": confirm_id, "approved": True},
            },
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "取消"},
                "type": "default",
                "value": {"confirm_id": confirm_id, "approved": False},
            },
        ],
    })

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "高风险 SQL 确认"},
            "template": "red",
        },
        "elements": elements,
    }, ensure_ascii=False)
