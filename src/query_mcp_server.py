"""MCP Server - 提供 execute_readonly_sql、get_cluster_list、get_table_schema、get_business_knowledge 工具。

使用 FastMCP 创建 MCP Server，通过 MCP 协议暴露数据库查询能力。
所有工具返回统一的结构化结果，错误时返回 {success, error_type, error_message}。
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

from src.config import load_config, AppConfig
from src.sql_validator import SQLValidator
from src.db_pool import ConnectionPoolManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server & shared state
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "query-agent",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8765")),
)

_config: AppConfig | None = None
_pool_manager: ConnectionPoolManager | None = None
_validator: SQLValidator | None = None


def _error_response(error_type: str, error_message: str) -> dict:
    """构建统一的错误返回。"""
    return {
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
    }


async def _ensure_initialized() -> tuple[AppConfig, ConnectionPoolManager, SQLValidator]:
    """确保配置、连接池和验证器已初始化，返回 (config, pool_manager, validator)。"""
    global _config, _pool_manager, _validator

    if _config is None:
        config_path = os.environ.get("CONFIG_PATH", "./config.yaml")
        _config = load_config(config_path)

    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager(_config.clusters)
        await _pool_manager.initialize()

    if _validator is None:
        _validator = SQLValidator(allowed_tables=_config.sql_security.allowed_tables)

    return _config, _pool_manager, _validator


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def execute_readonly_sql(
    cluster: str,
    sql: str,
    max_rows: int = 100,
) -> dict:
    """在指定集群上执行只读 SQL 查询。

    Args:
        cluster: 集群名称，如 "test", "production"。
        sql: 要执行的 SQL 查询语句。
        max_rows: 最大返回行数，默认 100。

    Returns:
        成功: {"success": true, "columns": [...], "rows": [...], "row_count": N, "truncated": bool}
        失败: {"success": false, "error_type": "...", "error_message": "..."}
    """
    try:
        config, pool_manager, validator = await _ensure_initialized()
    except Exception as exc:
        return _error_response("CONNECTION_ERROR", f"初始化失败: {exc}")

    # 1. 验证集群是否存在
    if not pool_manager.cluster_configured(cluster):
        available = list(config.clusters.keys())
        return _error_response(
            "INVALID_CLUSTER",
            f"集群 '{cluster}' 不存在。可用集群: {', '.join(available)}",
        )

    if not pool_manager.has_cluster(cluster):
        return _error_response(
            "CONNECTION_ERROR",
            f"集群 '{cluster}' 连接池未就绪，请检查集群状态。",
        )

    # 2. SQL 安全验证
    result = validator.validate(sql)
    if not result.is_valid:
        return _error_response(
            result.error_type or "UNSAFE_SQL",
            result.error_message or "SQL 验证失败",
        )

    # 3. 确保 LIMIT
    safe_sql = validator.ensure_limit(result.sanitized_sql or sql, max_rows)

    # 4. 执行查询
    try:
        async with pool_manager.get_connection(cluster) as conn:
            async with conn.cursor() as cur:
                await cur.execute(safe_sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = await cur.fetchall()

                # 将每行 tuple 转为 list 以便 JSON 序列化
                rows_list = [list(row) for row in rows]
                truncated = len(rows_list) >= max_rows

                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows_list,
                    "row_count": len(rows_list),
                    "truncated": truncated,
                }
    except Exception as exc:
        return _error_response("QUERY_ERROR", f"查询执行失败: {exc}")


@mcp.tool()
async def get_cluster_list() -> dict:
    """获取所有已配置的数据库集群列表。

    Returns:
        {"clusters": [{"name": "...", "description": "...", "database": "...", "status": "..."}]}
    """
    try:
        config, pool_manager, _ = await _ensure_initialized()
    except Exception as exc:
        return _error_response("CONNECTION_ERROR", f"初始化失败: {exc}")

    clusters = []
    for name, cluster_cfg in config.clusters.items():
        clusters.append({
            "name": name,
            "description": cluster_cfg.description,
            "database": cluster_cfg.database,
            "status": pool_manager.get_pool_status(name),
        })

    return {"clusters": clusters}


@mcp.tool()
async def get_table_schema(
    cluster: str,
    table_name: str | None = None,
) -> dict:
    """获取指定集群的表结构信息。

    当 table_name 为空时，返回所有允许查询的业务表列表。
    当指定 table_name 时，执行 SHOW COLUMNS FROM table_name 返回表的详细 schema。

    Args:
        cluster: 集群名称。
        table_name: 表名，为空则返回所有表列表。

    Returns:
        表列表: {"tables": ["..."]}
        表详情: {"table_name": "...", "columns": [{"name": "...", "type": "...", ...}]}
    """
    try:
        config, pool_manager, validator = await _ensure_initialized()
    except Exception as exc:
        return _error_response("CONNECTION_ERROR", f"初始化失败: {exc}")

    # 验证集群
    if not pool_manager.cluster_configured(cluster):
        available = list(config.clusters.keys())
        return _error_response(
            "INVALID_CLUSTER",
            f"集群 '{cluster}' 不存在。可用集群: {', '.join(available)}",
        )

    if not pool_manager.has_cluster(cluster):
        return _error_response(
            "CONNECTION_ERROR",
            f"集群 '{cluster}' 连接池未就绪，请检查集群状态。",
        )

    # 无 table_name → 返回白名单表列表
    if table_name is None:
        return {"tables": list(validator._allowed_tables)}

    # 验证 table_name 在白名单中
    allowed_set = set(validator._allowed_tables)
    if allowed_set and table_name not in allowed_set:
        return _error_response(
            "FORBIDDEN_TABLE",
            f"表 '{table_name}' 不在允许查询的表列表中。",
        )

    # 查询表结构
    try:
        async with pool_manager.get_connection(cluster) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SHOW COLUMNS FROM `{table_name}`")
                rows = await cur.fetchall()

                columns = []
                for row in rows:
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "key": row[3] or "",
                        "default": row[4],
                        "extra": row[5] or "",
                    })

                return {
                    "table_name": table_name,
                    "columns": columns,
                }
    except Exception as exc:
        return _error_response("QUERY_ERROR", f"获取表结构失败: {exc}")


@mcp.tool()
async def get_table_indexes(
    cluster: str,
    table_name: str | None = None,
) -> dict:
    """获取表的索引信息。

    当 table_name 为空时，返回所有白名单表的索引信息。
    当指定 table_name 时，返回该表的索引详情。

    Args:
        cluster: 集群名称。
        table_name: 表名，为空则返回所有白名单表的索引信息。

    Returns:
        {"indexes": [{"table": "...", "name": "...", "columns": [...], "unique": bool, "type": "..."}]}
    """
    try:
        config, pool_manager, validator = await _ensure_initialized()
    except Exception as exc:
        return _error_response("CONNECTION_ERROR", f"初始化失败: {exc}")

    # 验证集群
    if not pool_manager.cluster_configured(cluster):
        available = list(config.clusters.keys())
        return _error_response(
            "INVALID_CLUSTER",
            f"集群 '{cluster}' 不存在。可用集群: {', '.join(available)}",
        )

    if not pool_manager.has_cluster(cluster):
        return _error_response(
            "CONNECTION_ERROR",
            f"集群 '{cluster}' 连接池未就绪，请检查集群状态。",
        )

    # 确定要查询的表列表
    if table_name:
        tables_to_query = [table_name]
    else:
        tables_to_query = list(validator._allowed_tables)

    if not tables_to_query:
        return {"indexes": []}

    indexes = []
    try:
        async with pool_manager.get_connection(cluster) as conn:
            async with conn.cursor() as cur:
                for tbl in tables_to_query:
                    try:
                        await cur.execute(f"SHOW INDEX FROM `{tbl}`")
                        rows = await cur.fetchall()

                        # SHOW INDEX 返回列顺序:
                        # Table, Non_unique, Key_name, Seq_in_index, Column_name,
                        # Collation, Cardinality, Sub_part, Packed, Null,
                        # Index_type, Comment, Index_comment, Visible, Expression
                        # 按 Key_name 分组，合并同索引的多列
                        idx_map: dict[str, dict] = {}
                        for row in rows:
                            key_name = row[2]
                            seq = int(row[3])
                            col_name = row[4]
                            non_unique = int(row[1])
                            idx_type = row[10] if len(row) > 10 else "BTREE"

                            if key_name not in idx_map:
                                idx_map[key_name] = {
                                    "table": tbl,
                                    "name": key_name,
                                    "columns": [],
                                    "unique": non_unique == 0,
                                    "type": idx_type,
                                }
                            # 按 Seq_in_index 顺序插入
                            idx_map[key_name]["columns"].append((seq, col_name))

                        # 排序列并转为最终格式
                        for idx_data in idx_map.values():
                            idx_data["columns"] = [
                                col for _, col in sorted(idx_data["columns"])
                            ]
                            indexes.append(idx_data)
                    except Exception as exc:
                        logger.warning("获取表 '%s' 索引信息失败: %s", tbl, exc)

        return {"indexes": indexes}
    except Exception as exc:
        return _error_response("QUERY_ERROR", f"获取索引信息失败: {exc}")


@mcp.tool()
async def get_business_knowledge() -> dict:
    """获取当前配置的业务领域知识。

    Agent 在首次连接时调用此工具获取业务术语映射、表关系、状态码等知识，
    补充到 system prompt 中，无需在 agent 侧硬编码业务知识。

    Returns:
        {"description": "...", "term_mappings": {...}, "table_relationships": [...], "status_codes": [...], "custom_rules": [...]}
    """
    try:
        config, _, _ = await _ensure_initialized()
    except Exception as exc:
        return _error_response("CONNECTION_ERROR", f"初始化失败: {exc}")

    bk = config.business_knowledge
    return {
        "description": bk.description,
        "term_mappings": bk.term_mappings,
        "table_relationships": bk.table_relationships,
        "status_codes": bk.status_codes,
        "custom_rules": bk.custom_rules,
    }


# ---------------------------------------------------------------------------
# Entry point — 支持 stdio（本地）和 SSE HTTP（远程）两种传输模式
#
# stdio 模式（默认）: python3 -m src.query_mcp_server
# SSE 模式（远程）:   MCP_HOST=0.0.0.0 MCP_PORT=8765 python3 -m src.query_mcp_server --transport sse
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="查询 MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="传输模式: stdio（本地子进程）或 sse（HTTP 远程）",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
