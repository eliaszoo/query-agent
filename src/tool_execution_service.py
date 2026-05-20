"""工具执行相关服务。"""

import json
import logging

from mcp import ClientSession

from src.business_registry import BusinessRegistry
from src.sql_risk_checker import SQLRiskChecker, IndexInfo

logger = logging.getLogger(__name__)


class ToolExecutionService:
    """封装工具路由、执行前检查以及索引/schema 相关缓存。"""

    MAX_CELL_LENGTH = 200

    def __init__(
        self,
        registry: BusinessRegistry,
        risk_checker: SQLRiskChecker,
        confirm_callback,
        is_stdio_mode: bool,
        field_knowledge_manager,
        sql_preview_callback=None,
    ) -> None:
        self._registry = registry
        self._risk_checker = risk_checker
        self._confirm_callback = confirm_callback
        self._is_stdio_mode = is_stdio_mode
        self._field_knowledge = field_knowledge_manager
        self._sql_preview_callback = sql_preview_callback
        self._indexes_loaded = False

    @property
    def indexes_loaded(self) -> bool:
        return self._indexes_loaded

    @indexes_loaded.setter
    def indexes_loaded(self, value: bool) -> None:
        self._indexes_loaded = value

    def set_field_knowledge_manager(self, field_knowledge_manager) -> None:
        self._field_knowledge = field_knowledge_manager

    def set_sql_preview_callback(self, callback) -> None:
        """设置 SQL 预览回调。"""
        self._sql_preview_callback = callback

    async def ensure_indexes_loaded(self) -> None:
        """确保索引信息已加载到风险检测器中（多业务 SSE 模式）。"""
        if self._indexes_loaded:
            return

        if not self._is_stdio_mode:
            for entry in self._registry.list_businesses():
                try:
                    # 使用聚合的集群路由表
                    for cluster_name in entry.cluster_routing:
                        try:
                            result_text = await self._registry.call_tool(
                                entry.name, "get_table_indexes",
                                {"cluster": cluster_name, "business": entry.name}
                            )
                            self.parse_and_cache_indexes(
                                business=entry.name, cluster=cluster_name, result_text=result_text
                            )
                        except Exception:
                            logger.warning(
                                "获取业务 '%s' 集群 '%s' 索引信息失败",
                                entry.name, cluster_name, exc_info=True,
                            )
                except Exception:
                    logger.warning("获取业务 '%s' 索引信息失败", entry.name, exc_info=True)

        cached_keys = list(self._risk_checker._index_cache.keys())
        logger.warning(
            "索引加载完成: %d 个表, keys=%s",
            len(cached_keys),
            [(b, c, t) for b, c, t in cached_keys],
        )
        self._indexes_loaded = True

    async def ensure_indexes_loaded_stdio(self, session: ClientSession) -> None:
        """确保索引信息已加载到风险检测器中（stdio 模式）。"""
        if self._indexes_loaded:
            return

        await self.load_indexes_from_session(session)
        self._indexes_loaded = True

    async def load_indexes_from_session(self, session: ClientSession) -> None:
        """从 stdio MCP session 加载索引信息。"""
        try:
            clusters_result = await session.call_tool("get_cluster_list", {})
            clusters_text = self.serialize_tool_result(clusters_result)
            clusters_data = json.loads(clusters_text)
            cluster_list = [c["name"] for c in clusters_data.get("clusters", [])]

            if cluster_list:
                cluster = cluster_list[0]
                result = await session.call_tool(
                    "get_table_indexes", {"cluster": cluster}
                )
                result_text = self.serialize_tool_result(result)
                self.parse_and_cache_indexes(
                    business="default", cluster=cluster, result_text=result_text
                )
                self._indexes_loaded = True
        except Exception:
            logger.debug("从 MCP Server 获取索引信息失败", exc_info=True)

    def parse_and_cache_indexes(self, business: str, cluster: str, result_text: str) -> None:
        """解析索引信息 JSON 并缓存到风险检测器。"""
        try:
            data = json.loads(result_text)
            if isinstance(data, dict) and not data.get("success", True):
                return
            indexes = data.get("indexes", [])
            table_indexes: dict[str, list[IndexInfo]] = {}
            for idx in indexes:
                table = idx.get("table", "")
                if table not in table_indexes:
                    table_indexes[table] = []
                table_indexes[table].append(IndexInfo(
                    table=table,
                    name=idx.get("name", ""),
                    columns=idx.get("columns", []),
                    unique=idx.get("unique", False),
                    index_type=idx.get("type", "BTREE"),
                ))
            for table, idx_list in table_indexes.items():
                self._risk_checker.update_indexes(business, cluster, table, idx_list)
        except (ValueError, TypeError):
            logger.debug("解析索引信息失败", exc_info=True)

    async def pre_execute_check(self, tool_name: str, arguments: dict) -> str | None:
        """执行前检查：发送 SQL 预览，检测性能风险，等待确认。"""
        if tool_name != "execute_readonly_sql":
            return None

        sql = arguments.get("sql", "") if isinstance(arguments, dict) else ""
        cluster = arguments.get("cluster", "") if isinstance(arguments, dict) else ""
        business = arguments.get("business", "") if isinstance(arguments, dict) else ""
        risk_note = arguments.get("risk_note", "") if isinstance(arguments, dict) else ""

        # 风险评估：始终运行静态检测，LLM risk_note 作为补充
        await self.ensure_indexes_loaded()
        risk_result = self._risk_checker.check(business or "default", cluster, sql)
        if risk_result.has_risk:
            logger.warning(
                "静态风险检测: business=%s cluster=%s sql=%.80s → %s %s",
                business, cluster, sql, risk_result.risk_level, risk_result.risk_reasons,
            )
        static_risk_level = risk_result.risk_level
        static_reasons = risk_result.risk_reasons

        if risk_note:
            note_risk_level, note_reasons = self.parse_risk_note(risk_note)
        else:
            note_risk_level, note_reasons = "", []

        # 静态检测有索引信息时以其为准；无索引信息时用 risk_note 兜底
        if static_reasons:
            risk_level = static_risk_level
            reasons = static_reasons
        elif note_reasons:
            risk_level = note_risk_level
            reasons = note_reasons
        else:
            risk_level = ""
            reasons = []

        # 发送 SQL 预览
        # 飞书模式：有风险+确认回调时跳过单独预览（确认卡片已含 SQL+风险+按钮）
        # stdio 模式：始终先 print SQL，再确认
        skip_preview = self._sql_preview_callback and reasons and self._confirm_callback
        if not skip_preview:
            if self._sql_preview_callback:
                await self._sql_preview_callback(
                    sql=sql, cluster=cluster,
                    risk_level=risk_level, risk_reasons=reasons,
                )
            else:
                print(f"  SQL ({cluster}): {sql}")
                if reasons:
                    print(f"  Risk [{risk_level}]:")
                    for reason in reasons:
                        print(f"    - {reason}")

        # 高风险确认
        if reasons and self._confirm_callback:
            confirmed = await self._confirm_callback(
                sql=sql, cluster=cluster,
                risk_level=risk_level, risk_reasons=reasons,
            )
            if not confirmed:
                return json.dumps({
                    "success": False,
                    "error_type": "USER_CANCELLED",
                    "error_message": "用户取消了查询执行",
                }, ensure_ascii=False)

        return None

    async def route_tool_call(self, tool_name: str, arguments: dict, current_business: str = "") -> str:
        """路由工具调用到对应业务的 MCP Server。

        优先从 arguments 中获取 business 参数；如果没有则使用 current_business 自动填充。
        BusinessRegistry 根据 cluster 参数路由到对应地域的 MCP Server。
        business 参数会自动注入到 arguments 中传递给 MCP Server 工具。

        Args:
            tool_name: 工具名称。
            arguments: 工具参数（可能不含 business）。
            current_business: 当前选定的业务名（自动填充用）。

        Returns:
            工具结果的 JSON 字符串。
        """
        business = arguments.get("business", "") or current_business

        if not business:
            return json.dumps({
                "success": False,
                "error_type": "MISSING_BUSINESS",
                "error_message": "未指定目标业务，请在工具调用中提供 business 参数",
            }, ensure_ascii=False)

        if not self._registry.has_business(business):
            available = [e.name for e in self._registry.list_businesses()]
            return json.dumps({
                "success": False,
                "error_type": "INVALID_BUSINESS",
                "error_message": f"业务 '{business}' 不存在，可用业务: {available}",
            }, ensure_ascii=False)

        # 自动填充 business 参数（MCP Server 工具需要）
        args_with_business = dict(arguments)
        args_with_business["business"] = business

        try:
            return await self._registry.call_tool(business, tool_name, args_with_business)
        except Exception as e:
            logger.error("业务 '%s' 工具调用失败: %s", business, e, exc_info=True)
            return json.dumps({
                "success": False,
                "error_type": "TOOL_CALL_ERROR",
                "error_message": f"业务 '{business}' 工具调用失败: {e}",
            }, ensure_ascii=False)

    def cache_schema_from_result(self, arguments: dict, result_text: str, business: str) -> None:
        """从 get_table_schema 结果中缓存表结构。"""
        try:
            data = json.loads(result_text)
            if isinstance(data, dict) and "columns" in data:
                table_name = data.get("table_name") or (
                    arguments.get("table_name") if isinstance(arguments, dict) else ""
                )
                if table_name:
                    columns = [
                        {"name": c.get("name"), "type": c.get("type")}
                        for c in data.get("columns", [])
                    ]
                    self._field_knowledge.cache_table_schema(business, table_name, columns)
                    logger.info("已缓存表结构: %s (%d 列)", table_name, len(columns))
        except (ValueError, TypeError):
            pass

    @staticmethod
    def parse_risk_note(note: str) -> tuple[str, list[str]]:
        """解析 LLM 在 risk_note 中声明的风险分析。"""
        note_lower = note.lower()
        reasons = []
        risk_level = ""

        is_index_driven = "索引驱动" in note_lower or ("索引" in note_lower and "驱动" in note_lower)

        if "全表扫描" in note_lower:
            reasons.append("WHERE 条件无法命中索引，可能全表扫描")
            risk_level = "high"

        if not is_index_driven:
            if "select *" in note_lower:
                reasons.append("SELECT * 返回全列")
                risk_level = risk_level or "medium"

            if "like" in note_lower and "%" in note:
                reasons.append("LIKE 前导通配符，无法使用索引")
                risk_level = risk_level or "medium"

        if not reasons and not is_index_driven:
            reasons.append(note)
            risk_level = "medium"

        return risk_level, reasons

    @staticmethod
    def serialize_tool_result(result) -> str:
        """将 MCP CallToolResult 序列化为字符串。"""
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                texts.append(str(item))
        return "\n".join(texts) if texts else ""

    @staticmethod
    def summarize_tool_result(tool_name: str, result_text: str) -> str:
        """精简工具结果，减少回传给 LLM 的 token 消耗。"""
        try:
            data = json.loads(result_text)
        except (ValueError, TypeError):
            return result_text

        if tool_name == "get_table_schema" and isinstance(data, dict) and "columns" in data:
            simplified_columns = []
            for col in data.get("columns", []):
                simplified_columns.append({
                    "name": col.get("name"),
                    "type": col.get("type"),
                    "nullable": col.get("nullable"),
                    "key": col.get("key", ""),
                })
            data["columns"] = simplified_columns
            return json.dumps(data, ensure_ascii=False)

        if tool_name == "execute_readonly_sql" and isinstance(data, dict) and "rows" in data:
            rows = data.get("rows", [])

            if len(rows) > 10:
                rows = rows[:10]
                data["row_count"] = len(data.get("rows", []))
                data["truncated"] = True
                data["note"] = f"共 {data['row_count']} 行，仅展示前 10 行"

            truncated_rows = []
            for row in rows:
                if isinstance(row, list):
                    truncated_row = []
                    for cell in row:
                        cell_str = str(cell) if cell is not None else None
                        if cell_str and len(cell_str) > ToolExecutionService.MAX_CELL_LENGTH:
                            cell_str = cell_str[:ToolExecutionService.MAX_CELL_LENGTH] + "..."
                        truncated_row.append(cell_str)
                    truncated_rows.append(truncated_row)
                else:
                    truncated_rows.append(row)

            data["rows"] = truncated_rows
            return json.dumps(data, ensure_ascii=False)

        return result_text
