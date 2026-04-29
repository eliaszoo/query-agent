"""Agent 核心 - 支持多 LLM Provider + 多业务 MCP Client 的通用查询 Agent。

支持 Anthropic (Claude)、OpenAI (GPT)、DeepSeek、GLM 等模型，
通过 LLM Provider 抽象层统一接口。
通过 BusinessRegistry 管理多个业务的 MCP Server 连接，LLM 自动路由。
"""

import json as _json
import logging
import os
import re as _re
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

from src.business_registry import BusinessRegistry
from src.config import load_config, AppConfig, BusinessKnowledge, BusinessEntryConfig
from src.error_memory import ErrorMemoryManager
from src.field_knowledge import FieldKnowledgeManager
from src.llm_provider import LLMProvider, create_provider
from src.prompts import build_system_prompt
from src.sql_risk_checker import SQLRiskChecker, IndexInfo

logger = logging.getLogger(__name__)

# 完整保留的最近轮数
RECENT_TURNS_KEEP = 3
# 压缩保留的最大轮数（超出直接丢弃）
MAX_COMPRESSED_TURNS = 5


def _sanitize_args_for_log(args: dict) -> dict:
    """脱敏工具参数，用于日志输出。"""
    if not isinstance(args, dict):
        return args
    sanitized = {}
    for k, v in args.items():
        if k in ("password", "api_key", "token", "secret"):
            sanitized[k] = "***"
        elif k == "sql" and isinstance(v, str) and len(v) > 200:
            sanitized[k] = v[:200] + "..."
        else:
            sanitized[k] = v
    return sanitized


@dataclass
class QueryMetrics:
    """单次查询的元信息。"""

    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    model: str = ""


def _convert_mcp_tools_to_anthropic(mcp_tools: list) -> list[dict]:
    """将 MCP Tool 列表转换为 Anthropic API 的 tools 格式。

    Args:
        mcp_tools: MCP ClientSession.list_tools() 返回的工具列表。

    Returns:
        Anthropic messages API 所需的 tools 参数列表。
    """
    tools = []
    for tool in mcp_tools:
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        })
    return tools


def _merge_tools_with_business_param(
    tools_per_business: dict[str, list[dict]],
) -> list[dict]:
    """合并多个业务的工具定义，注入 business 参数。

    每个业务暴露相同的工具集，合并后给 LLM 呈现统一接口：
    - execute_readonly_sql(business, cluster, sql, max_rows)
    - get_cluster_list(business)
    - get_table_schema(business, cluster, table_name)

    get_business_knowledge 不暴露给 LLM（仅在初始化时内部调用）。

    Args:
        tools_per_business: 业务名 → 该业务的工具列表。

    Returns:
        合并后的工具定义列表（包含 business 参数）。
    """
    # 用第一个业务的工具作为模板
    first_tools = next(iter(tools_per_business.values()), [])
    if not first_tools:
        return []

    merged = []
    available_businesses = list(tools_per_business.keys())
    biz_enum = {
        "type": "string",
        "enum": available_businesses,
        "description": "目标业务标识",
    }

    for tool in first_tools:
        name = tool["name"]

        # get_business_knowledge 不暴露给 LLM
        if name == "get_business_knowledge":
            continue

        # 深拷贝 schema 并注入 business 参数
        schema = _json.loads(_json.dumps(tool.get("input_schema", {})))
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 在最前面插入 business 参数
        new_properties = {"business": biz_enum}
        new_properties.update(properties)
        schema["properties"] = new_properties
        schema["required"] = ["business"] + required

        merged.append({
            "name": name,
            "description": tool["description"],
            "input_schema": schema,
        })

    return merged


class QueryAgent:
    """通用查询 Agent，通过 MCP 协议连接多个业务的数据查询工具。

    支持：
    - 多业务动态路由（BusinessRegistry + LLM 自动选择 business 参数）
    - 多轮对话上下文（理解"再查一下"等引用）
    - 查询元信息追踪（耗时、token 用量）
    - 错误记忆持久化
    - 业务领域知识从 MCP Server 动态加载
    - 向后兼容单业务模式（stdio 本地子进程）
    """

    def __init__(self, config_path: str = "./config.yaml", confirm_callback=None):
        self.config: AppConfig = load_config(config_path)
        self.provider: LLMProvider = create_provider(
            provider=self.config.agent.provider,
            api_key=self.config.agent.api_key or None,
            base_url=self.config.agent.base_url or None,
        )
        self.error_memory = ErrorMemoryManager()
        self.field_knowledge = FieldKnowledgeManager()
        self.last_metrics: Optional[QueryMetrics] = None

        # 多轮对话历史
        self._conversation_history: list[dict] = []
        self._pinned_messages: list[dict] = []

        # SQL 性能风险检测器
        self._risk_checker = SQLRiskChecker()
        self._indexes_loaded = False

        # System prompt 缓存
        self._cached_system_prompt: str = ""
        self._prompt_dirty: bool = True

        # 最近查询上下文（用于反馈检测）
        self._last_query_context: dict | None = None  # {"business", "query", "response", "sql"}

        # 用户确认回调（默认用 input，测试时可以注入 mock）
        self._confirm_callback = confirm_callback or self._default_confirm

        # 多业务注册表
        self.registry = BusinessRegistry()

        # 从配置加载初始业务列表
        for name, entry_cfg in self.config.businesses.items():
            self.registry.register(name, entry_cfg.mcp_server_url, entry_cfg.display_name, api_key=entry_cfg.api_key)

        # 本地 MCP Server 子进程启动参数（stdio 模式，仅当无远程 URL 时使用）
        self.mcp_server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.query_mcp_server"],
            env={"CONFIG_PATH": config_path, **os.environ},
        )

        # 业务知识：优先使用本地配置，远程模式下会在首次连接时尝试从 MCP Server 获取
        self._business_knowledge: BusinessKnowledge = self.config.business_knowledge

        # 标记是否为单业务 stdio 模式（向后兼容）
        self._is_stdio_mode = not self.config.businesses and not self.config.agent.mcp_server_url

    def _build_system_prompt(self) -> str:
        """构建包含多业务知识和错误记忆的动态 System Prompt（带缓存）。"""
        if not self._prompt_dirty and self._cached_system_prompt:
            return self._cached_system_prompt

        businesses = self.registry.list_businesses()
        knowledge_map: dict[str, BusinessKnowledge] = {}

        for entry in businesses:
            if entry.knowledge is not None:
                knowledge_map[entry.name] = entry.knowledge

        # 单业务 stdio 模式：使用本地配置的业务知识
        if self._is_stdio_mode and self._business_knowledge.description:
            knowledge_map["default"] = self._business_knowledge

        prompt = build_system_prompt(businesses, knowledge_map)

        # 字段知识：查询过程中确认的字段含义
        field_prompt = self.field_knowledge.build_field_prompt()
        if field_prompt:
            prompt += field_prompt

        # 已缓存的表结构：已知列名，无需再查 schema
        schema_prompt = self.field_knowledge.build_schema_prompt()
        if schema_prompt:
            prompt += schema_prompt

        # 错误记忆：单业务模式按业务过滤，多业务模式注入全部
        current_business = ""
        if self._is_stdio_mode:
            current_business = "default"
        memory_prompt = self.error_memory.build_memory_prompt(current_business)
        if memory_prompt:
            prompt += memory_prompt

        self._cached_system_prompt = prompt
        self._prompt_dirty = False
        return prompt

    def _mark_prompt_dirty(self) -> None:
        """标记 system prompt 需要重建（在知识或记忆变化时调用）。"""
        self._prompt_dirty = True

    def clear_history(self) -> None:
        """清空对话历史（保留置顶消息）。"""
        self._conversation_history.clear()

    def pin_message(self, content: str) -> None:
        """置顶一条重要上下文消息，压缩时不会被截断。"""
        self._pinned_messages.append({"role": "user", "content": f"[置顶] {content}"})

    def _trim_history(self) -> None:
        """修剪对话历史，保留最近 N 轮，早期轮次压缩为摘要。

        最近 RECENT_TURNS_KEEP 轮保留完整内容，更早的轮次只保留
        用户消息和 assistant 文本摘要，去掉工具调用详情和工具结果。
        置顶消息始终保留在最前面。
        """
        recent_count = RECENT_TURNS_KEEP * 2

        if len(self._conversation_history) <= recent_count:
            # 即使不压缩，也要确保置顶消息在前面
            self._prepend_pinned()
            return

        # 超过最大上限，先截断
        max_total = (RECENT_TURNS_KEEP + MAX_COMPRESSED_TURNS) * 2
        if len(self._conversation_history) > max_total:
            self._conversation_history = self._conversation_history[-max_total:]

        # 保留最近 RECENT_TURNS_KEEP 轮完整，更早的压缩
        if len(self._conversation_history) <= recent_count:
            self._prepend_pinned()
            return

        recent = self._conversation_history[-recent_count:]
        older = self._conversation_history[:-recent_count]

        compressed = []
        for msg in older:
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                text = self._extract_text_from_content(content)
                if text:
                    compressed.append({"role": "assistant", "content": f"[历史] {text[:200]}"})
            elif role == "user" and isinstance(content, str):
                compressed.append({"role": "user", "content": content})
            # 跳过 tool result 消息和 tool_calls 格式的 assistant 消息

        self._conversation_history = compressed + recent
        self._prepend_pinned()

    def _prepend_pinned(self) -> None:
        """将置顶消息插入对话历史最前面（去重）。"""
        if not self._pinned_messages:
            return
        # 移除已有的置顶消息，再重新插入
        self._conversation_history = [
            m for m in self._conversation_history
            if not (isinstance(m.get("content"), str) and m["content"].startswith("[置顶] "))
        ]
        self._conversation_history = self._pinned_messages + self._conversation_history

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """从 assistant content 中提取纯文本（兼容 Anthropic 和 OpenAI 格式）。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Anthropic 格式：content 是 block 列表
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts)
        return ""

    async def _ensure_knowledge_loaded(self) -> None:
        """确保所有已注册业务的领域知识已加载（用于构建 system prompt）。"""
        for entry in self.registry.list_businesses():
            if entry.knowledge is None:
                try:
                    await self.registry.fetch_business_knowledge(entry.name)
                    self._mark_prompt_dirty()
                except Exception:
                    logger.warning("获取业务 '%s' 知识失败，跳过", entry.name, exc_info=True)

    async def _build_merged_tools(self) -> list[dict]:
        """构建合并了 business 参数的工具列表。"""
        businesses = self.registry.list_businesses()

        if not businesses:
            return []

        # 单业务 stdio 模式：直接从本地 MCP server 获取工具
        if self._is_stdio_mode:
            async with stdio_client(self.mcp_server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return _convert_mcp_tools_to_anthropic(tools_result.tools)

        # 多业务 SSE 模式：收集每个业务的工具定义，合并注入 business 参数
        tools_per_business: dict[str, list[dict]] = {}
        for entry in businesses:
            try:
                tools = await self.registry.fetch_tools_schema(entry.name)
                tools_per_business[entry.name] = tools
            except Exception:
                logger.warning("获取业务 '%s' 工具列表失败，跳过", entry.name, exc_info=True)

        if not tools_per_business:
            return []

        return _merge_tools_with_business_param(tools_per_business)

    async def run_query(self, user_input: str) -> str:
        """发送用户查询并返回 Agent 的最终文本响应。

        根据业务模式选择执行方式：
        - 多业务 SSE 模式 → BusinessRegistry 路由
        - 单业务 stdio 模式 → 本地子进程（向后兼容）

        Args:
            user_input: 用户的自然语言查询。

        Returns:
            Agent 的最终文本响应。
        """
        start_time = time.time()
        metrics = QueryMetrics(model=self.config.agent.model)

        if self._is_stdio_mode:
            result = await self._run_query_stdio(user_input, metrics)
        else:
            result = await self._run_query_multi_business(user_input, metrics)

        metrics.duration_seconds = round(time.time() - start_time, 2)
        self.last_metrics = metrics
        return result

    async def _run_query_stdio(self, user_input: str, metrics: QueryMetrics) -> str:
        """单业务 stdio 模式（向后兼容）。"""
        async with stdio_client(self.mcp_server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = _convert_mcp_tools_to_anthropic(tools_result.tools)
                await self._try_fetch_business_knowledge(session)

                # stdio 模式：确保索引在对话开始前加载（需要 session）
                await self._ensure_indexes_loaded_stdio(session)

                async def execute_tool(name: str, args: dict, _business: str) -> tuple[str, str]:
                    result = await session.call_tool(name, args)
                    return self._serialize_tool_result(result), "default"

                return await self._conversation_loop_core(
                    tools, user_input, metrics, system_prompt=None, execute_tool=execute_tool,
                )

    async def _run_query_multi_business(
        self, user_input: str, metrics: QueryMetrics
    ) -> str:
        """多业务 SSE 模式。"""
        # 确保业务知识已加载
        await self._ensure_knowledge_loaded()

        # 构建合并的工具列表
        tools = await self._build_merged_tools()

        if not tools:
            return "当前没有可用的业务，请先使用 /add 命令添加业务。"

        # 构建包含多业务知识的 system prompt
        system_prompt = self._build_system_prompt()

        # 执行对话循环（无需传入 session，工具调用通过 registry 路由）
        result = await self._multi_business_conversation_loop(
            tools, user_input, metrics, system_prompt
        )
        return result

    async def _try_fetch_business_knowledge(self, session: ClientSession) -> None:
        """从 MCP Server 获取业务领域知识，补充到本地配置中。

        仅当本地配置中 business_knowledge 为空时才从 server 获取。
        （用于单业务 stdio 模式）
        """
        bk = self._business_knowledge
        if bk.description or bk.term_mappings or bk.table_relationships or bk.status_codes:
            return  # 本地已有业务知识，无需从 server 获取

        try:
            result = await session.call_tool("get_business_knowledge", {})
            result_text = self._serialize_tool_result(result)
            data = _json.loads(result_text)

            if isinstance(data, dict) and data.get("description"):
                self._business_knowledge = BusinessKnowledge(
                    description=data.get("description", ""),
                    term_mappings=data.get("term_mappings", {}),
                    table_relationships=data.get("table_relationships", []),
                    status_codes=data.get("status_codes", []),
                    custom_rules=data.get("custom_rules", []),
                )
                logger.info("从 MCP Server 获取到业务知识: %s", self._business_knowledge.description)
        except Exception:
            logger.debug("从 MCP Server 获取业务知识失败，使用本地配置", exc_info=True)

    async def _ensure_indexes_loaded(self) -> None:
        """确保索引信息已加载到风险检测器中（多业务 SSE 模式）。"""
        if self._indexes_loaded:
            return

        if not self._is_stdio_mode:
            # 多业务模式：从每个 MCP server 获取索引信息
            for entry in self.registry.list_businesses():
                try:
                    # 先获取集群列表，再用真实集群名获取索引
                    clusters_text = await self.registry.call_tool(
                        entry.name, "get_cluster_list", {}
                    )
                    clusters_data = _json.loads(clusters_text)
                    cluster_list = [
                        c["name"] for c in clusters_data.get("clusters", [])
                        if c.get("status") == "connected"
                    ]
                    if cluster_list:
                        result_text = await self.registry.call_tool(
                            entry.name, "get_table_indexes", {"cluster": cluster_list[0]}
                        )
                        self._parse_and_cache_indexes(result_text)
                except Exception:
                    logger.warning("获取业务 '%s' 索引信息失败", entry.name, exc_info=True)

        self._indexes_loaded = True

    async def _ensure_indexes_loaded_stdio(self, session: ClientSession) -> None:
        """确保索引信息已加载到风险检测器中（stdio 模式，需要 session）。"""
        if self._indexes_loaded:
            return

        await self._load_indexes_from_session(session)
        self._indexes_loaded = True

    async def _load_indexes_from_session(self, session: ClientSession) -> None:
        """从 stdio MCP session 加载索引信息。"""
        try:
            # 获取集群列表
            clusters_result = await session.call_tool("get_cluster_list", {})
            clusters_text = self._serialize_tool_result(clusters_result)
            clusters_data = _json.loads(clusters_text)
            cluster_list = [c["name"] for c in clusters_data.get("clusters", [])]

            # 从第一个可用集群获取索引信息
            if cluster_list:
                result = await session.call_tool(
                    "get_table_indexes", {"cluster": cluster_list[0]}
                )
                result_text = self._serialize_tool_result(result)
                self._parse_and_cache_indexes(result_text)
                self._indexes_loaded = True
        except Exception:
            logger.debug("从 MCP Server 获取索引信息失败", exc_info=True)

    def _parse_and_cache_indexes(self, result_text: str) -> None:
        """解析索引信息 JSON 并缓存到风险检测器。"""
        try:
            data = _json.loads(result_text)
            # 处理可能的错误响应
            if isinstance(data, dict) and not data.get("success", True):
                return
            indexes = data.get("indexes", [])
            # 按 table 分组
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
                self._risk_checker.update_indexes(table, idx_list)
        except (ValueError, TypeError):
            logger.debug("解析索引信息失败", exc_info=True)

    async def _pre_execute_check(
        self, tool_name: str, arguments: dict
    ) -> str | None:
        """执行前检查：打印 SQL，检测性能风险，等待确认。

        优先使用 LLM 在 risk_note 参数中声明的风险分析，
        无 risk_note 时回退到静态索引检查。

        Args:
            tool_name: 工具名称。
            arguments: 工具参数。

        Returns:
            None: 允许执行。
            str: 拒绝执行的错误消息 JSON。
        """
        if tool_name != "execute_readonly_sql":
            return None

        sql = arguments.get("sql", "") if isinstance(arguments, dict) else ""
        cluster = arguments.get("cluster", "") if isinstance(arguments, dict) else ""
        risk_note = arguments.get("risk_note", "") if isinstance(arguments, dict) else ""

        # 1. 打印 SQL
        print(f"  SQL ({cluster}): {sql}")

        # 2. 风险分析：优先用 LLM 的 risk_note，否则回退静态检查
        if risk_note:
            risk_level, reasons = self._parse_risk_note(risk_note)
        else:
            await self._ensure_indexes_loaded()
            risk_result = self._risk_checker.check(sql)
            risk_level = risk_result.risk_level
            reasons = risk_result.risk_reasons

        if reasons:
            print(f"  Risk [{risk_level}]:")
            for reason in reasons:
                print(f"    - {reason}")

            # 3. 等待用户确认
            if not self._confirm_callback("是否继续执行？(y/N): "):
                return _json.dumps({
                    "success": False,
                    "error_type": "USER_CANCELLED",
                    "error_message": "用户取消了查询执行",
                }, ensure_ascii=False)

        return None

    @staticmethod
    def _parse_risk_note(note: str) -> tuple[str, list[str]]:
        """解析 LLM 在 risk_note 中声明的风险分析。

        格式示例:
        - "索引驱动: app_id" → 无风险，索引驱动查询高效
        - "全表扫描风险" → high, ["WHERE 条件无法命中索引，可能全表扫描"]
        - "SELECT * 返回全列" → medium, ["SELECT * 返回全列"]
        - "LIKE 前导通配符" → medium, ["LIKE 前导通配符，无法使用索引"]
        """
        note_lower = note.lower()
        reasons = []
        risk_level = ""

        # 索引驱动 → 查询高效，不构成风险
        is_index_driven = "索引驱动" in note_lower or ("索引" in note_lower and "驱动" in note_lower)

        if "全表扫描" in note_lower:
            reasons.append("WHERE 条件无法命中索引，可能全表扫描")
            risk_level = "high"

        if not is_index_driven:
            # 非索引驱动的其他风险
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
    def _default_confirm(prompt: str) -> bool:
        """默认确认回调，使用标准输入。"""
        answer = input(f"   {prompt}").strip().lower()
        return answer == "y"

    async def _conversation_loop_core(
        self,
        tools: list[dict],
        user_input: str,
        metrics: QueryMetrics,
        system_prompt: str | None,
        execute_tool: "Callable[[str, dict, str], Awaitable[tuple[str, str]]]",
    ) -> str:
        """通用对话循环，工具执行通过 execute_tool 回调注入。

        Args:
            tools: LLM 可用的工具列表。
            user_input: 用户输入。
            metrics: 查询元信息。
            system_prompt: 系统提示词，为 None 时自动构建。
            execute_tool: 工具执行器，签名 (tool_name, arguments, business) -> (result_text, business)。
        """
        self._trim_history()
        messages = list(self._conversation_history)
        messages.append({"role": "user", "content": user_input})

        if system_prompt is None:
            system_prompt = self._build_system_prompt()

        while True:
            response = self.provider.chat(
                model=self.config.agent.model,
                max_tokens=self.config.agent.max_tokens,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            metrics.input_tokens += response.input_tokens
            metrics.output_tokens += response.output_tokens

            if response.stop_reason == "end_turn":
                # 提取字段含义（在剥离注释前）
                self._auto_extract_field_knowledge(response.text)

                # 剥离 FIELD_KNOWLEDGE 注释，不展示给用户
                display_text = self._FIELD_KNOWLEDGE_TAG.sub('', response.text).rstrip()

                # 保存到对话历史（保留注释供后续提取）
                self._conversation_history.append(
                    {"role": "user", "content": user_input}
                )
                self._conversation_history.append(
                    {"role": "assistant", "content": response.text}
                )

                return display_text

            # 处理工具调用
            messages.append(
                self.provider.build_assistant_message(response.raw_content)
            )

            tool_results = []
            for tc in (response.tool_calls or []):
                metrics.tool_calls += 1

                # 提前保存 business（多业务模式下 _route_tool_call 会 pop 掉）
                tc_business = tc.arguments.get("business", "") if isinstance(tc.arguments, dict) else ""

                # 执行前检查（打印 SQL、性能风险检测、用户确认）
                cancel_result = await self._pre_execute_check(tc.name, tc.arguments)
                if cancel_result is not None:
                    # 用户拒绝执行，直接中断对话循环
                    self._conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    self._conversation_history.append(
                        {"role": "assistant", "content": "查询已被用户取消。"}
                    )
                    return "查询已被用户取消。"
                else:
                    result_text, resolved_business = await execute_tool(tc.name, tc.arguments, tc_business)
                    if not tc_business and resolved_business:
                        tc_business = resolved_business

                logger.info("调用工具: %s(%s)", tc.name, _sanitize_args_for_log(tc.arguments))

                # 缓存 get_table_schema 的结果，避免重复查询
                if tc.name == "get_table_schema":
                    self._cache_schema_from_result(tc.arguments, result_text)
                    self._mark_prompt_dirty()

                self._check_and_record_error(
                    user_input, tc.name, tc.arguments, result_text, tc_business
                )

                # 追踪最近查询上下文（用于反馈检测）
                if tc.name == "execute_readonly_sql":
                    self._last_query_context = {
                        "business": tc_business or "default",
                        "query": user_input,
                        "sql": tc.arguments.get("sql", "") if isinstance(tc.arguments, dict) else "",
                    }

                tool_results.append(
                    self.provider.build_tool_result_message(
                        tc.id, self._summarize_tool_result(tc.name, result_text)
                    )
                )

            # Anthropic: tool_results 作为 user message 的 content 列表
            # OpenAI: tool_results 是独立的 tool role 消息
            if self.config.agent.provider == "anthropic":
                messages.append({"role": "user", "content": tool_results})
            else:
                messages.extend(tool_results)

    async def _multi_business_conversation_loop(
        self,
        tools: list[dict],
        user_input: str,
        metrics: QueryMetrics,
        system_prompt: str,
    ) -> str:
        """执行消息循环（多业务模式），通过 BusinessRegistry 路由工具调用。"""
        async def execute_tool(name: str, args: dict, _business: str) -> tuple[str, str]:
            result_text = await self._route_tool_call(name, args)
            return result_text, _business

        return await self._conversation_loop_core(
            tools, user_input, metrics, system_prompt=system_prompt, execute_tool=execute_tool,
        )

    async def _route_tool_call(self, tool_name: str, arguments: dict) -> str:
        """路由工具调用到对应的业务 MCP Server。

        从 arguments 中提取 business 参数，路由到对应业务的 MCP session。

        Args:
            tool_name: 工具名称。
            arguments: 工具参数（包含 business 字段）。

        Returns:
            工具结果的 JSON 字符串。
        """
        # 提取 business 参数
        business = arguments.pop("business", None)

        if not business:
            return _json.dumps({
                "success": False,
                "error_type": "MISSING_BUSINESS",
                "error_message": "未指定目标业务，请在工具调用中提供 business 参数",
            }, ensure_ascii=False)

        if not self.registry.has_business(business):
            available = [e.name for e in self.registry.list_businesses()]
            return _json.dumps({
                "success": False,
                "error_type": "INVALID_BUSINESS",
                "error_message": f"业务 '{business}' 不存在，可用业务: {available}",
            }, ensure_ascii=False)

        try:
            return await self.registry.call_tool(business, tool_name, arguments)
        except Exception as e:
            logger.error("业务 '%s' 工具调用失败: %s", business, e, exc_info=True)
            return _json.dumps({
                "success": False,
                "error_type": "TOOL_CALL_ERROR",
                "error_message": f"业务 '{business}' 工具调用失败: {e}",
            }, ensure_ascii=False)

    def _check_and_record_error(
        self,
        user_query: str,
        tool_name: str,
        tool_input: dict,
        result_text: str,
        business: str = "",
    ) -> None:
        """检测工具返回的错误并记录到错误记忆。

        跳过环境/基础设施错误和 SQL 语法错误，这些不是 Agent 可学习的经验。
        """
        try:
            result = _json.loads(result_text)
        except (ValueError, TypeError):
            return

        if not isinstance(result, dict) or result.get("success", True):
            return

        error_type = result.get("error_type", "UNKNOWN")
        error_message = result.get("error_message", "")

        # 跳过环境/基础设施错误，Agent 无法从这些错误中学习
        _SKIP_ERROR_TYPES = {
            "CONNECTION_ERROR",    # 数据库连接失败
            "CONFIG_ERROR",       # 配置缺失或无效
            "POOL_ERROR",        # 连接池错误
            "TIMEOUT_ERROR",     # 连接超时
            "QUERY_ERROR",       # SQL 语法错误 — Agent 应先 get_table_schema 了解表结构
        }
        if error_type in _SKIP_ERROR_TYPES:
            logger.debug("跳过错误，不记录: %s - %s", error_type, error_message)
            return

        bad_sql = tool_input.get("sql", "") if isinstance(tool_input, dict) else ""

        # 使用传入的 business（多业务模式下 _route_tool_call 会 pop 掉 arguments 中的 business）
        if not business and self._is_stdio_mode:
            business = "default"

        # 根据错误类型生成经验教训
        lesson = self._generate_lesson(error_type, error_message, bad_sql)

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

    @staticmethod
    def _generate_lesson(error_type: str, error_message: str, bad_sql: str) -> str:
        """根据错误类型自动生成经验教训。"""
        if error_type == "UNSAFE_SQL":
            return f"不要生成包含写操作或锁子句的 SQL。错误 SQL: {bad_sql}"
        if error_type == "FORBIDDEN_TABLE":
            return f"只能查询白名单中的表。{error_message}"
        if error_type == "INVALID_CLUSTER":
            return f"使用正确的集群名称。{error_message}"
        if error_type == "QUERY_ERROR":
            return f"SQL 语法或逻辑有误，需要修正。{error_message}"
        if error_type == "MULTI_STATEMENT":
            return "不要在一次调用中执行多条 SQL 语句"
        if error_type == "MISSING_BUSINESS":
            return "每次工具调用必须指定 business 参数"
        if error_type == "INVALID_BUSINESS":
            return f"使用正确的业务标识。{error_message}"
        if error_type == "USER_CANCELLED":
            return "查询被用户取消，需要优化查询条件或获取用户确认"
        if error_type == "USER_FEEDBACK":
            return f"用户反馈: {error_message}"
        return f"{error_type}: {error_message}"

    async def extract_feedback_lesson(
        self,
        original_query: str,
        agent_response: str,
        user_feedback: str,
    ) -> str | None:
        """用 LLM 从用户反馈中提取经验教训。

        Args:
            original_query: 用户的原始查询。
            agent_response: Agent 的回答。
            user_feedback: 用户的后续输入。

        Returns:
            经验教训字符串，如果不是反馈则返回 None。
        """
        prompt = (
            f'用户刚才问了: "{original_query}"\n\n'
            f'Agent 回答了: "{agent_response[:500]}"\n\n'
            f'用户接下来输入了: "{user_feedback}"\n\n'
            "请判断用户接下来的输入是否是对 Agent 回答的反馈/纠正/补充？\n"
            "- 如果是全新的、不相关的查询 → 回答: NONE\n"
            "- 如果是反馈/纠正 → 提取一条简洁的经验教训（一句话），以\"应该\"开头\n\n"
            "只回答 NONE 或一条经验教训，不要其他内容。"
        )

        try:
            response = self.provider.chat(
                model=self.config.agent.model,
                max_tokens=200,
                system="你是一个经验提取助手，从用户反馈中提取简洁的经验教训。",
                tools=[],
                messages=[{"role": "user", "content": prompt}],
            )

            lesson = response.text.strip()
            if lesson.upper() == "NONE" or not lesson:
                return None
            return lesson
        except Exception:
            logger.debug("提取反馈经验失败", exc_info=True)
            return None

    # 字段含义自动提取的正则
    # 模式1: "tb_voice.origin: 1(自研), 2(阿里云)" — 表名.字段名 + 括号枚举
    _TABLE_FIELD_PATTERN = _re.compile(
        r'(tb_\w+)\.(\w+)\s*[：:]\s*'
        r'((?:\d+\s*[（(]\s*[^）)]+\s*[）)]\s*[，,]?\s*)+)',
        _re.UNICODE
    )
    # 模式2: "origin: 1(自研), 2(阿里云)" — 字段名 + 括号枚举（需从 SQL 推断表名）
    _FIELD_ENUM_PATTERN = _re.compile(
        r'(\w+)\s*[：:]\s*'
        r'((?:\d+\s*[（(]\s*[^）)]+\s*[）)]\s*[，,]?\s*)+)',
        _re.UNICODE
    )
    # 模式3: "**来源(origin)**: 5 = 火山" 或 "禁用状态(forbidden_status): 1 = 正常, 2 = 禁用"
    # — 中文名(字段名): 数字 = 含义（支持 markdown ** 粗体）
    _FIELD_EQ_PATTERN = _re.compile(
        r'\*{0,2}(?:\S+?\s*)?\((\w+)\)\*{0,2}\s*[：:]\s*'  # **中文(field_name)**:
        r'((?:\d+\s*=\s*[^,，\n]+(?:\s*[，,]\s*|\s*))+)',   # 5 = 火山, 1 = 正常, 2 = 禁用
        _re.UNICODE
    )

        # HTML 注释中的结构化字段知识声明
    # 格式: <!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山,1=自研"}] -->
    _FIELD_KNOWLEDGE_TAG = _re.compile(
        r'<!--\s*FIELD_KNOWLEDGE:\s*(\[[\s\S]*?\])\s*-->',
        _re.UNICODE
    )

    def _auto_extract_field_knowledge(self, response_text: str) -> None:
        """从 LLM 回复中提取字段含义并持久化到 field_knowledge。

        优先解析结构化 HTML 注释声明：
          <!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山,1=自研"}] -->

        若无结构化声明，回退到正则匹配自由文本中的字段含义。
        """
        dirty = False

        # 优先：结构化 HTML 注释
        for match in self._FIELD_KNOWLEDGE_TAG.finditer(response_text):
            try:
                items = _json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                items = []
            for item in items:
                table = item.get("table", "")
                field = item.get("field", "")
                values = item.get("values", "")
                if table and field and values:
                    self.field_knowledge.add_field(table, field, values)
                    dirty = True
                    logger.info("提取字段知识(结构化): %s.%s: %s", table, field, values)
            if dirty:
                self._mark_prompt_dirty()
                return  # 结构化声明已处理，无需回退

        # 回退：正则匹配自由文本
        self._auto_extract_field_knowledge_fallback(response_text)

    def _auto_extract_field_knowledge_fallback(self, response_text: str) -> None:
        """回退：从自由文本中正则匹配字段含义。"""
        dirty = False
        sql = (self._last_query_context or {}).get("sql", "")

        # 模式: 中文名(字段名): 数字 = 含义  — 如 **来源(origin)**: 5 = 火山
        for match in self._FIELD_EQ_PATTERN.finditer(response_text):
            column = match.group(1)
            raw_values = match.group(2)
            description = self._parse_eq_values(raw_values)
            if description:
                table = self._infer_table_from_sql(sql)
                if table:
                    self.field_knowledge.add_field(table, column, description)
                    dirty = True
                    logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        # 模式: 表名.字段名: 括号枚举 — 如 tb_voice.origin: 1(自研), 2(阿里云)
        for match in self._TABLE_FIELD_PATTERN.finditer(response_text):
            table = match.group(1)
            column = match.group(2)
            raw_values = match.group(3)
            description = self._parse_enum_values(raw_values)
            if description:
                self.field_knowledge.add_field(table, column, description)
                dirty = True
                logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        # 模式: 字段名: 括号枚举 — 如 origin: 1(自研), 2(阿里云)
        if not dirty:
            for match in self._FIELD_ENUM_PATTERN.finditer(response_text):
                column = match.group(1)
                raw_values = match.group(2)
                description = self._parse_enum_values(raw_values)
                if description and not column.startswith('tb_'):
                    table = self._infer_table_from_sql(sql)
                    if table:
                        self.field_knowledge.add_field(table, column, description)
                        dirty = True
                        logger.info("提取字段知识(回退): %s.%s: %s", table, column, description)

        if dirty:
            self._mark_prompt_dirty()

    @staticmethod
    def _infer_table_from_sql(sql: str) -> str:
        """从 SQL 中提取表名，优先取 FROM 子句中的 tb_ 表。"""
        if not sql:
            return ""
        table_match = _re.search(r'FROM\s+(tb_\w+)', sql, _re.IGNORECASE)
        return table_match.group(1) if table_match else ""

    @staticmethod
    def _parse_enum_values(raw: str) -> str:
        """解析 "1(自研), 2(阿里云), 3(腾讯云)" 格式为 "1=自研, 2=阿里云, 3=腾讯云"。"""
        parts = _re.findall(r'(\d+)\s*[（(]\s*([^）)]+)\s*[）)]', raw)
        if not parts:
            return ""
        return ", ".join(f"{num}={label.strip()}" for num, label in parts)

    @staticmethod
    def _parse_eq_values(raw: str) -> str:
        """解析 "5 = 火山, 1 = 正常, 2 = 禁用" 格式为 "5=火山, 1=正常, 2=禁用"。"""
        parts = _re.findall(r'(\d+)\s*=\s*([^,，\n]+)', raw)
        if not parts:
            return ""
        return ", ".join(f"{num}={label.strip()}" for num, label in parts)

    def _cache_schema_from_result(self, arguments: dict, result_text: str) -> None:
        """从 get_table_schema 结果中缓存表结构，注入 prompt 避免重复查询。"""
        try:
            data = _json.loads(result_text)
            if isinstance(data, dict) and "columns" in data:
                table_name = data.get("table_name") or (arguments.get("table_name") if isinstance(arguments, dict) else "")
                if table_name:
                    columns = [{"name": c.get("name"), "type": c.get("type")} for c in data.get("columns", [])]
                    self.field_knowledge.cache_table_schema(table_name, columns)
                    logger.info("已缓存表结构: %s (%d 列)", table_name, len(columns))
        except (ValueError, TypeError):
            pass

    @staticmethod
    def _serialize_tool_result(result) -> str:
        """将 MCP CallToolResult 序列化为字符串。"""
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                texts.append(str(item))
        return "\n".join(texts) if texts else ""

    MAX_CELL_LENGTH = 200  # 单个单元格最大字符数

    @staticmethod
    def _summarize_tool_result(tool_name: str, result_text: str) -> str:
        """精简工具结果，减少回传给 LLM 的 token 消耗。

        - get_table_schema: 去掉 default、extra 等低价值字段
        - execute_readonly_sql: 超过 10 行时只保留前 10 行，单元格超长截断
        """
        try:
            data = _json.loads(result_text)
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
            return _json.dumps(data, ensure_ascii=False)

        if tool_name == "execute_readonly_sql" and isinstance(data, dict) and "rows" in data:
            rows = data.get("rows", [])

            # 截断行数
            if len(rows) > 10:
                rows = rows[:10]
                data["row_count"] = len(data.get("rows", []))
                data["truncated"] = True
                data["note"] = f"共 {data['row_count']} 行，仅展示前 10 行"

            # 截断列宽
            truncated_rows = []
            for row in rows:
                if isinstance(row, list):
                    truncated_row = []
                    for cell in row:
                        cell_str = str(cell) if cell is not None else None
                        if cell_str and len(cell_str) > QueryAgent.MAX_CELL_LENGTH:
                            cell_str = cell_str[:QueryAgent.MAX_CELL_LENGTH] + "..."
                        truncated_row.append(cell_str)
                    truncated_rows.append(truncated_row)
                else:
                    truncated_rows.append(row)

            data["rows"] = truncated_rows
            return _json.dumps(data, ensure_ascii=False)

        return result_text
