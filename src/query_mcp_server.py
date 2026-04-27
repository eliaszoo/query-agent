"""MCP Server - 提供 execute_readonly_sql、get_cluster_list、get_table_schema、get_business_knowledge 工具。

使用 FastMCP 创建 MCP Server，通过 MCP 协议暴露数据库查询能力。
所有工具返回统一的结构化结果，错误时返回 {success, error_type, error_message}。
"""

import asyncio
import logging
import os
import re
import secrets

from mcp.server.fastmcp import FastMCP

from src.config import load_config, AppConfig
from src.sql_validator import SQLValidator
from src.db_pool import ConnectionPoolManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 鉴权中间件
# ---------------------------------------------------------------------------


class BearerAuthMiddleware:
    """Starlette ASGI 中间件，校验 Bearer Token。

    对所有 HTTP 请求校验 Authorization: Bearer <api_key>，
    不匹配则返回 401/403。非 HTTP 请求直接放行。
    """

    def __init__(self, app, api_key: str):
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        # 对 HTTP 和 WebSocket 都做鉴权
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        from starlette.responses import Response

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        if not auth_header.startswith("Bearer "):
            if scope["type"] == "http":
                response = Response(
                    "Unauthorized",
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 4001})
            return

        token = auth_header[7:]  # Strip "Bearer "
        if not secrets.compare_digest(token, self.api_key):
            if scope["type"] == "http":
                response = Response("Forbidden", status_code=403)
                await response(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 4003})
            return

        await self.app(scope, receive, send)


def _get_api_key() -> str:
    """获取 API Key，优先从配置文件 auth.api_key，其次从环境变量 MCP_API_KEY。"""
    try:
        config_path = os.environ.get("CONFIG_PATH", "./config.yaml")
        config = load_config(config_path)
        if config.auth.api_key:
            return config.auth.api_key
    except Exception:
        pass
    return os.environ.get("MCP_API_KEY", "")


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
_init_lock = asyncio.Lock()

# 表名合法字符正则：纵深防御，防止反引号逃逸等 SQL 注入
_TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


def _validate_table_name(table_name: str) -> bool:
    """校验表名只包含合法字符，防止 SQL 注入。"""
    return bool(_TABLE_NAME_PATTERN.match(table_name))


def _error_response(error_type: str, error_message: str) -> dict:
    """构建统一的错误返回。"""
    return {
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
    }


async def _ensure_initialized() -> tuple[AppConfig, ConnectionPoolManager, SQLValidator]:
    """确保配置、连接池和验证器已初始化，返回 (config, pool_manager, validator)。

    使用 double-check locking 防止并发重复初始化。
    """
    global _config, _pool_manager, _validator

    # 快速路径：已初始化直接返回
    if _config is not None and _pool_manager is not None and _validator is not None:
        return _config, _pool_manager, _validator

    async with _init_lock:
        # double-check
        if _config is not None and _pool_manager is not None and _validator is not None:
            return _config, _pool_manager, _validator

        config_path = os.environ.get("CONFIG_PATH", "./config.yaml")
        _config = load_config(config_path)
        _pool_manager = ConnectionPoolManager(_config.clusters)
        await _pool_manager.initialize()
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
    """在指定集群上执行只读 SQL 查询。"""
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
                # 设置查询超时（MySQL 5.7.8+，仅对 SELECT 生效）
                timeout_ms = config.sql_security.query_timeout * 1000
                await cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {timeout_ms}")

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
        error_msg = str(exc)
        if "MAX_EXECUTION_TIME" in error_msg or "Query execution was interrupted" in error_msg:
            return _error_response(
                "TIMEOUT_ERROR",
                f"查询超时（{config.sql_security.query_timeout}s），请优化查询条件",
            )
        return _error_response("QUERY_ERROR", f"查询执行失败: {exc}")


@mcp.tool()
async def get_cluster_list() -> dict:
    """获取所有已配置的数据库集群列表。"""
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
    """获取表结构。不指定 table_name 则返回所有允许查询的表列表。"""
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

    # 校验表名合法性（纵深防御，防止 SQL 注入）
    if not _validate_table_name(table_name):
        return _error_response("INVALID_INPUT", f"非法表名: {table_name}")

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
    """获取表的索引信息。不指定 table_name 则返回所有白名单表的索引。"""
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
        if not _validate_table_name(table_name):
            return _error_response("INVALID_INPUT", f"非法表名: {table_name}")
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
    """获取当前配置的业务领域知识。"""
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
    import anyio
    import uvicorn

    parser = argparse.ArgumentParser(description="查询 MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="传输模式: stdio（本地子进程）或 sse（HTTP 远程）",
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径（默认: 环境变量 CONFIG_PATH 或 ./config.yaml）",
    )
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        # 构建带鉴权的 Starlette app
        app = mcp.sse_app()
        api_key = _get_api_key()
        if api_key:
            app = BearerAuthMiddleware(app, api_key)
            logger.info("SSE server: API Key authentication enabled")
        else:
            logger.warning("SSE server: No API Key configured, authentication disabled")

        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8765"))
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        anyio.run(server.serve)
