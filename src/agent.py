"""Agent 核心 - 支持多 LLM Provider + 多业务 MCP Client 的通用查询 Agent。

支持 Anthropic (Claude)、OpenAI (GPT)、DeepSeek、GLM 等模型，
通过 LLM Provider 抽象层统一接口。
通过 BusinessRegistry 管理多个业务的 MCP Server 连接，LLM 自动路由。
"""

import json as _json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

from src.business_registry import BusinessRegistry
from src.business_selection_service import (
    BusinessSelectionResult,
    BusinessSelectionService,
)
from src.config import (
    load_config,
    AppConfig,
    BusinessKnowledge,
    BusinessEntryConfig,
)
from src.conversation_state import ConversationState
from src.error_memory import ErrorMemoryManager
from src.field_knowledge import FieldKnowledgeManager
from src.knowledge_store import KnowledgeStore
from src.llm_provider import LLMProvider, create_provider
from src.preference_rules import PreferenceRulesManager
from src.prompt_service import PromptService
from src.query_plan import QueryPlan
from src.query_rule_executor import QueryRuleExecutor
from src.tool_execution_service import ToolExecutionService
from src.sql_risk_checker import SQLRiskChecker, IndexInfo

logger = logging.getLogger(__name__)


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
    selected_business: str = ""
    business_selection_strategy: str = ""
    business_selection_reason: str = ""
    applied_rules: list[str] | None = None
    overridden_rules: list[str] | None = None
    query_plan: QueryPlan | None = None


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
        self.last_metrics: Optional[QueryMetrics] = None

        self._conversation = ConversationState()

        # SQL 性能风险检测器
        self._risk_checker = SQLRiskChecker()

        self._prompt_service = PromptService()

        # 用户确认回调（默认用 input，测试时可以注入 mock）
        self._confirm_callback = confirm_callback or self._default_confirm

        # 每业务独立的知识存储
        self._knowledge_stores: dict[str, KnowledgeStore] = {}
        self._preference_rules_managers: dict[str, PreferenceRulesManager] = {}

        # 多业务注册表
        self.registry = BusinessRegistry()

        # 从配置加载初始业务列表并初始化存储
        for name, entry_cfg in self.config.businesses.items():
            self.registry.register(name, entry_cfg.mcp_server_url, entry_cfg.display_name, api_key=entry_cfg.api_key)
            self._ensure_business_storage(name)

        # 加载动态添加的业务（重启后恢复 /add 添加的业务）
        self._load_dynamic_businesses()

        # 数据迁移：将旧的单目录存储拆分到每业务目录
        self._migrate_legacy_storage()

        # 清理旧的 hash 命名空间空目录
        self._cleanup_empty_config_dirs()

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

        # stdio 模式下需要初始化 "default" 业务存储
        if self._is_stdio_mode:
            self._ensure_business_storage("default")

        # ToolExecutionService 需要一个 field_knowledge_manager 用于 schema 缓存
        # 使用第一个业务的 manager，或 stdio 模式的 "default"
        first_biz = ""
        if self._is_stdio_mode:
            first_biz = "default"
        elif self.config.businesses:
            first_biz = next(iter(self.config.businesses))
        if first_biz:
            default_fk = self._get_field_knowledge_manager(first_biz)
        else:
            default_fk = FieldKnowledgeManager(knowledge_path=os.path.join(".query-agent", "_shared", "field_knowledge.json"))
        self._business_selector = BusinessSelectionService(
            provider=self.provider,
            model=self.config.agent.model,
            registry=self.registry,
        )
        self._tool_execution = ToolExecutionService(
            registry=self.registry,
            risk_checker=self._risk_checker,
            confirm_callback=self._confirm_callback,
            is_stdio_mode=self._is_stdio_mode,
            field_knowledge_manager=default_fk,
        )

    def _ensure_business_storage(self, business: str) -> None:
        """懒初始化业务的知识存储目录和 manager。"""
        if business in self._knowledge_stores:
            return
        biz_dir = os.path.join(".query-agent", business)
        os.makedirs(biz_dir, exist_ok=True)
        self._knowledge_stores[business] = KnowledgeStore(
            error_memory=ErrorMemoryManager(
                memory_path=os.path.join(biz_dir, "error_memory.json")
            ),
            field_knowledge=FieldKnowledgeManager(
                knowledge_path=os.path.join(biz_dir, "field_knowledge.json")
            ),
            mark_prompt_dirty=self._mark_prompt_dirty,
        )
        self._preference_rules_managers[business] = PreferenceRulesManager(
            rules_path=os.path.join(biz_dir, "preference_rules.json")
        )

    def _get_knowledge_store(self, business: str) -> KnowledgeStore:
        """获取指定业务的知识存储，自动初始化。"""
        self._ensure_business_storage(business)
        return self._knowledge_stores[business]

    def _get_preference_rules_manager(self, business: str) -> PreferenceRulesManager:
        """获取指定业务的偏好规则管理器，自动初始化。"""
        self._ensure_business_storage(business)
        return self._preference_rules_managers[business]

    def _get_field_knowledge_manager(self, business: str) -> FieldKnowledgeManager:
        """获取指定业务的字段知识管理器。"""
        return self._get_knowledge_store(business).field_knowledge

    def _get_error_memory_manager(self, business: str) -> ErrorMemoryManager:
        """获取指定业务的错误记忆管理器。"""
        return self._get_knowledge_store(business).error_memory

    def _save_dynamic_businesses(self) -> None:
        """将动态添加的业务保存到文件，重启后可恢复。"""
        configured = set(self.config.businesses.keys())
        dynamic = {}
        for entry in self.registry.list_businesses():
            if entry.name not in configured:
                dynamic[entry.name] = {
                    "mcp_server_url": entry.mcp_server_url,
                    "display_name": entry.display_name,
                    "api_key": entry.api_key or "",
                }
        path = os.path.join(".query-agent", "dynamic_businesses.json")
        os.makedirs(".query-agent", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            _json.dump(dynamic, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _load_dynamic_businesses(self) -> None:
        """从文件加载动态添加的业务。"""
        path = os.path.join(".query-agent", "dynamic_businesses.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            configured = set(self.config.businesses.keys())
            for name, info in data.items():
                if name not in configured and not self.registry.has_business(name):
                    self.registry.register(
                        name,
                        info.get("mcp_server_url", ""),
                        info.get("display_name", name),
                        api_key=info.get("api_key", ""),
                    )
                    self._ensure_business_storage(name)
        except (_json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("动态业务文件解析失败，跳过: %s", exc)

    def _migrate_legacy_storage(self) -> None:
        """将旧的单目录存储（可能包含多业务数据）拆分到每业务目录。"""
        # 查找旧的存储目录（可能是 hash 命名空间或业务名称命名的单目录）
        if not os.path.isdir(".query-agent"):
            return

        # 对每个可能包含多业务数据的旧目录进行迁移
        for dirname in os.listdir(".query-agent"):
            dirpath = os.path.join(".query-agent", dirname)
            if not os.path.isdir(dirpath) or dirname.startswith("config-"):
                continue  # 跳过 hash 目录和动态业务文件

            # 检查是否有旧的单文件存储需要迁移
            for filename in ("field_knowledge.json", "error_memory.json", "preference_rules.json"):
                filepath = os.path.join(dirpath, filename)
                if not os.path.exists(filepath):
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = _json.load(f)
                except (_json.JSONDecodeError, TypeError):
                    continue

                entries = data.get("entries", [])
                if not entries:
                    continue

                # 按业务分组
                groups: dict[str, list] = {}
                for entry in entries:
                    biz = entry.get("business", "") or dirname
                    if biz not in groups:
                        groups[biz] = []
                    groups[biz].append(entry)

                # 如果只有一个业务且与目录名匹配，无需迁移
                if len(groups) == 1 and dirname in groups:
                    continue

                # 将各业务的数据写入对应目录
                for biz, biz_entries in groups.items():
                    self._ensure_business_storage(biz)
                    biz_path = os.path.join(".query-agent", biz, filename)
                    # 读取目标文件已有的数据，合并
                    existing = []
                    if os.path.exists(biz_path):
                        try:
                            with open(biz_path, "r", encoding="utf-8") as f:
                                existing_data = _json.load(f)
                            existing = existing_data.get("entries", [])
                        except (_json.JSONDecodeError, TypeError):
                            existing = []

                    # 合并：以目标文件已有数据为主，补充旧文件中缺失的
                    existing_keys = {
                        (e.get("business", ""), e.get("table", ""), e.get("column", ""))
                        if filename == "field_knowledge.json"
                        else (e.get("business", ""), e.get("error_type", ""), e.get("timestamp", ""))
                        if filename == "error_memory.json"
                        else (e.get("business", ""), e.get("rule", ""), e.get("rule_type", ""))
                        for e in existing
                    }
                    merged = list(existing)
                    for e in biz_entries:
                        key = (
                            (e.get("business", ""), e.get("table", ""), e.get("column", ""))
                            if filename == "field_knowledge.json"
                            else (e.get("business", ""), e.get("error_type", ""), e.get("timestamp", ""))
                            if filename == "error_memory.json"
                            else (e.get("business", ""), e.get("rule", ""), e.get("rule_type", ""))
                        )
                        if key not in existing_keys:
                            merged.append(e)

                    tmp_path = biz_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        _json.dump({"entries": merged}, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, biz_path)

                # 迁移完成，删除旧文件
                os.remove(filepath)
                logger.info("已迁移 %s 中的多业务数据到每业务目录", filepath)

    def _cleanup_empty_config_dirs(self) -> None:
        """清理旧的 hash 命名空间空目录（config-* 模式）。"""
        base = ".query-agent"
        if not os.path.isdir(base):
            return
        for d in os.listdir(base):
            if d.startswith("config-"):
                dp = os.path.join(base, d)
                if os.path.isdir(dp) and not os.listdir(dp):
                    os.rmdir(dp)
                    logger.info("已清理空目录: %s", dp)

    def _build_system_prompt(self) -> str:
        """构建包含多业务知识和错误记忆的动态 System Prompt（带缓存）。"""
        current_business = self._get_current_prompt_business()
        # 当无明确业务时，使用第一个已注册业务或 "default"
        if not current_business:
            if self._is_stdio_mode:
                current_business = "default"
            elif self._knowledge_stores:
                current_business = next(iter(self._knowledge_stores))
        return self._prompt_service.build(
            businesses=self.registry.list_businesses(),
            is_stdio_mode=self._is_stdio_mode,
            current_business=current_business,
            configured_business_knowledge=self._business_knowledge,
            field_knowledge_manager=self._get_field_knowledge_manager(current_business or "default"),
            error_memory_manager=self._get_error_memory_manager(current_business or "default"),
            preference_rules_manager=self._get_preference_rules_manager(current_business or "default"),
        )

    def _mark_prompt_dirty(self) -> None:
        """标记 system prompt 需要重建（在知识或记忆变化时调用）。"""
        self._prompt_service.mark_dirty()

    def _get_current_prompt_business(self) -> str:
        """获取当前 prompt 应使用的业务视图。"""
        if self._is_stdio_mode:
            return "default"
        if self._conversation.last_query_context:
            return self._conversation.last_query_context.get("business", "")
        return ""

    def clear_history(self) -> None:
        """清空对话历史（保留置顶消息）。"""
        self._conversation.clear_history()

    def pin_message(self, content: str) -> None:
        """置顶一条重要上下文消息，压缩时不会被截断。"""
        self._conversation.pin_message(content)

    def get_locked_business(self) -> str:
        """获取当前会话锁定的业务。"""
        return self._conversation.locked_business

    def lock_business(self, business: str) -> None:
        """锁定当前会话业务。"""
        if business and not self.registry.has_business(business):
            raise KeyError(f"业务 '{business}' 不存在")
        self._conversation.locked_business = business

    def clear_locked_business(self) -> None:
        """清除当前会话业务锁定。"""
        self._conversation.locked_business = ""

    def list_businesses(self):
        """列出当前已注册业务。"""
        return self.registry.list_businesses()

    def add_business(
        self, name: str, mcp_server_url: str, display_name: str = "", api_key: str = ""
    ) -> None:
        """注册新业务、初始化存储并持久化。"""
        self.registry.register(name, mcp_server_url, display_name, api_key=api_key)
        self._ensure_business_storage(name)
        self._save_dynamic_businesses()
        self._mark_prompt_dirty()

    async def remove_business(self, name: str) -> None:
        """移除业务并清除相关缓存。"""
        await self.registry.remove(name)
        if name in self._knowledge_stores:
            self._knowledge_stores[name].clear_business(name)
        if self._conversation.locked_business == name:
            self._conversation.locked_business = ""
        if (
            self._conversation.last_query_context
            and self._conversation.last_query_context.get("business") == name
        ):
            self._conversation.last_query_context = None
        self._save_dynamic_businesses()

    def add_field_knowledge(
        self, business: str, table: str, column: str, description: str
    ) -> None:
        """添加字段知识并失效 prompt 缓存。"""
        self._get_knowledge_store(business).add_field_knowledge(business, table, column, description)

    def remove_field_knowledge(self, business: str, table: str, column: str) -> bool:
        """删除字段知识并在成功时失效 prompt 缓存。"""
        return self._get_knowledge_store(business).remove_field_knowledge(business, table, column)

    def list_field_knowledge(self, business: str = ""):
        """列出字段知识。"""
        if business:
            return self._get_knowledge_store(business).list_field_knowledge(business=business)
        # 聚合所有业务的字段知识
        result = []
        for biz, store in self._knowledge_stores.items():
            result.extend(store.list_field_knowledge(business=""))
        return result

    def clear_error_memory(self, business: str = "") -> None:
        """清空错误记忆并失效 prompt 缓存。"""
        if business:
            self._get_knowledge_store(business).clear_error_memory(business=business)
        else:
            for store in self._knowledge_stores.values():
                store.clear_error_memory(business="")

    def add_preference_rule(self, business: str, rule: str, source: str = "") -> None:
        """添加默认查询规则并失效 prompt 缓存。"""
        parsed = self.extract_explicit_feedback_rule(rule)
        self._get_preference_rules_manager(business).add_rule(
            business,
            rule,
            source,
            rule_type=parsed.get("rule_type", "") if parsed else "",
            payload=parsed.get("payload") if parsed else None,
        )
        self._mark_prompt_dirty()

    def list_preference_rules(self, business: str = ""):
        """列出默认查询规则。"""
        if business:
            return self._get_preference_rules_manager(business).get_rules(business)
        # 聚合所有业务的规则
        result = []
        for mgr in self._preference_rules_managers.values():
            result.extend(mgr.get_rules(business=""))
        return result

    def clear_preference_rules(self, business: str = "") -> None:
        """清空默认查询规则并失效 prompt 缓存。"""
        if business:
            self._get_preference_rules_manager(business).clear(business)
        else:
            for mgr in self._preference_rules_managers.values():
                mgr.clear(business="")
        self._mark_prompt_dirty()

    def get_error_memory_entries(self):
        """获取错误记忆条目。"""
        result = []
        for store in self._knowledge_stores.values():
            result.extend(store.get_error_memory_entries())
        return result

    def get_error_memory_businesses(self):
        """获取拥有错误记忆的业务标识。"""
        result = set()
        for store in self._knowledge_stores.values():
            result.update(store.get_error_memory_businesses())
        return list(result)

    def get_last_business(self) -> str:
        """获取最近一次查询所属业务。"""
        if not self._conversation.last_query_context:
            return "default" if self._is_stdio_mode else ""
        return self._conversation.last_query_context.get("business", "")

    def record_feedback(
        self, original_query: str, business: str, user_feedback: str, lesson: str
    ) -> None:
        """记录用户反馈经验并失效 prompt 缓存。"""
        parsed = self.extract_explicit_feedback_rule(user_feedback)
        if parsed is not None:
            self.add_preference_rule(business, parsed["rule"], source=user_feedback)
            return

        self._get_knowledge_store(business).record_feedback(
            original_query, business, user_feedback, lesson
        )

    @staticmethod
    def extract_explicit_feedback_lesson(user_feedback: str) -> str | None:
        """从明确的用户指令中直接提取经验，不依赖 LLM。"""
        parsed = QueryAgent.extract_explicit_feedback_rule(user_feedback)
        if parsed is not None:
            return parsed["rule"]

        normalized = user_feedback.strip()
        if not normalized:
            return None

        hints = ("记住", "默认", "以后都", "后续查询", "优先过滤")
        if not any(hint in normalized for hint in hints):
            return None

        return f"后续查询遵循用户偏好：{normalized}"

    @staticmethod
    def extract_explicit_feedback_rule(user_feedback: str) -> dict | None:
        """从用户显式反馈中解析结构化默认规则。"""
        normalized = user_feedback.strip()
        if not normalized:
            return None

        hints = ("记住", "默认", "以后都", "后续查询", "优先过滤")
        if not any(hint in normalized for hint in hints):
            return None

        if (
            "可用" in normalized
            or "未禁用" in normalized
            or "禁用" in normalized
            or "过滤" in normalized
        ):
            return {
                "rule": "默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。",
                "rule_type": "available_only",
                "payload": {"deleted_at_is_null": True, "forbidden_status": 1},
            }

        if "测试环境" in normalized or "test集群" in normalized:
            return {
                "rule": "默认优先查询测试环境，除非用户明确指定生产环境。",
                "rule_type": "default_cluster_test",
                "payload": {"cluster": "test"},
            }

        return {
            "rule": f"后续查询遵循用户偏好：{normalized}",
            "rule_type": "natural_language",
            "payload": None,
        }

    def _collect_rule_applications(self, user_input: str, business: str, arguments: dict | None = None) -> tuple[list[str], list[str], dict]:
        rules = self.list_preference_rules(business)
        result = QueryRuleExecutor.apply(user_input, business, rules, arguments=arguments)
        applied = [item.description for item in result.applications if item.applied]
        overridden = [
            f"{item.description}（已覆盖: {item.override_reason}）"
            for item in result.applications if item.overridden
        ]
        return applied, overridden, result.arguments

    async def _select_business_for_query(
        self, user_input: str
    ) -> BusinessSelectionResult:
        """为查询选择业务，优先使用会话锁定。"""
        locked_business = self.get_locked_business()
        if locked_business:
            if self.registry.has_business(locked_business):
                entry = self.registry.get_entry(locked_business)
                return BusinessSelectionResult(
                    business=entry,
                    strategy="locked",
                    reason=f"当前会话已锁定业务：{entry.display_name} ({entry.name})",
                )
            self.clear_locked_business()

        return await self._business_selector.select_business(user_input)

    async def build_query_plan(self, user_input: str) -> QueryPlan:
        """构建查询计划预览。"""
        if self._is_stdio_mode:
            plan = QueryPlan(
                user_input=user_input,
                business="default",
                business_display_name="default",
                business_strategy="single",
                business_reason="单业务 stdio 模式",
                locked_business="",
                default_cluster=self.config.agent.default_cluster,
            )
        else:
            selection = await self._select_business_for_query(user_input)
            business_name = selection.business.name if selection.business else ""
            display_name = selection.business.display_name if selection.business else ""
            plan = QueryPlan(
                user_input=user_input,
                business=business_name,
                business_display_name=display_name,
                business_strategy=selection.strategy,
                business_reason=selection.reason,
                locked_business=self.get_locked_business(),
                default_cluster=self.config.agent.default_cluster,
            )

        rule_business = plan.business or ("default" if self._is_stdio_mode else "")
        applied, overridden, _ = self._collect_rule_applications(
            user_input, rule_business, arguments={"cluster": self.config.agent.default_cluster}
        )
        plan.active_rules = applied
        plan.overridden_rules = overridden
        return plan

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

    async def _build_business_tools(self, business_name: str) -> list[dict]:
        """为单个业务构建工具视图，仅暴露该业务。"""
        if self._is_stdio_mode:
            async with stdio_client(self.mcp_server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return _convert_mcp_tools_to_anthropic(tools_result.tools)

        tools = await self.registry.fetch_tools_schema(business_name)
        return _merge_tools_with_business_param({business_name: tools})

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
            metrics.selected_business = "default"
            metrics.business_selection_strategy = "single"
            metrics.business_selection_reason = "单业务 stdio 模式"
            applied, overridden, _ = self._collect_rule_applications(user_input, "default")
            metrics.applied_rules = applied
            metrics.overridden_rules = overridden
            metrics.query_plan = await self.build_query_plan(user_input)
            result = await self._run_query_stdio(user_input, metrics)
        else:
            metrics.query_plan = await self.build_query_plan(user_input)
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
                await self._tool_execution.ensure_indexes_loaded_stdio(session)

                async def execute_tool(name: str, args: dict, _business: str) -> tuple[str, str]:
                    result = await session.call_tool(name, args)
                    return ToolExecutionService.serialize_tool_result(result), "default"

                return await self._conversation_loop_core(
                    tools, user_input, metrics, system_prompt=None, execute_tool=execute_tool,
                )

    async def _run_query_multi_business(
        self, user_input: str, metrics: QueryMetrics
    ) -> str:
        """多业务 SSE 模式。"""
        # 确保业务知识已加载
        await self._ensure_knowledge_loaded()

        selection = await self._select_business_for_query(user_input)
        metrics.business_selection_strategy = selection.strategy
        metrics.business_selection_reason = selection.reason

        if selection.business:
            metrics.selected_business = selection.business.name
            applied, overridden, _ = self._collect_rule_applications(user_input, selection.business.name)
            metrics.applied_rules = applied
            metrics.overridden_rules = overridden
            tools = await self._build_business_tools(selection.business.name)
            business_name = selection.business.name
            system_prompt = self._prompt_service.build_for_business(
                business_entry=selection.business,
                configured_business_knowledge=self._business_knowledge,
                field_knowledge_manager=self._get_field_knowledge_manager(business_name),
                error_memory_manager=self._get_error_memory_manager(business_name),
                preference_rules_manager=self._get_preference_rules_manager(business_name),
            )
            return await self._multi_business_conversation_loop(
                tools, user_input, metrics, system_prompt
            )

        # 构建合并的工具列表
        tools = await self._build_merged_tools()

        if not tools:
            return "当前没有可用的业务，请先使用 /add 命令添加业务。"

        metrics.selected_business = "all"
        metrics.applied_rules = []
        metrics.overridden_rules = []

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
            result_text = ToolExecutionService.serialize_tool_result(result)
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
        self._conversation.trim_history()
        messages = list(self._conversation.history)
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
                last_biz = self.get_last_business() or "default"
                self._get_knowledge_store(last_biz).auto_extract_field_knowledge(
                    response_text=response.text,
                    business=last_biz,
                    sql=(self._conversation.last_query_context or {}).get("sql", ""),
                )

                # 剥离 FIELD_KNOWLEDGE 注释，不展示给用户
                display_text = KnowledgeStore.FIELD_KNOWLEDGE_TAG.sub('', response.text).rstrip()

                # 将完整的 messages（包含工具调用过程）同步回对话历史
                # messages 已包含 user_input + 工具调用过程 + 最终回复
                messages.append({"role": "assistant", "content": response.text})
                self._conversation.history = messages

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
                tool_args = dict(tc.arguments) if isinstance(tc.arguments, dict) else {}
                current_business = tc_business or self.get_last_business() or "default"
                applied_rules, overridden_rules, effective_args = self._collect_rule_applications(
                    user_input, current_business, arguments=tool_args
                )
                if applied_rules:
                    metrics.applied_rules = list(dict.fromkeys((metrics.applied_rules or []) + applied_rules))
                if overridden_rules:
                    metrics.overridden_rules = list(dict.fromkeys((metrics.overridden_rules or []) + overridden_rules))
                tool_args = effective_args or tool_args

                # 执行前检查（打印 SQL、性能风险检测、用户确认）
                cancel_result = await self._tool_execution.pre_execute_check(tc.name, tool_args)
                if cancel_result is not None:
                    # 用户拒绝执行，保存已有的对话过程到历史
                    messages.append({"role": "assistant", "content": "查询已被用户取消。"})
                    self._conversation.history = messages
                    return "查询已被用户取消。"
                else:
                    result_text, resolved_business = await execute_tool(tc.name, tool_args, tc_business)
                    if not tc_business and resolved_business:
                        tc_business = resolved_business

                logger.info("调用工具: %s(%s)", tc.name, _sanitize_args_for_log(tool_args))

                # 缓存 get_table_schema 的结果，避免重复查询
                if tc.name == "get_table_schema":
                    self._tool_execution.cache_schema_from_result(
                        tool_args, result_text, tc_business or "default"
                    )
                    self._mark_prompt_dirty()

                self._get_knowledge_store(tc_business or "default").check_and_record_error(
                    user_query=user_input,
                    tool_input=tool_args,
                    result_text=result_text,
                    business=tc_business,
                    is_stdio_mode=self._is_stdio_mode,
                    lesson_builder=self._generate_lesson,
                )

                # 追踪最近查询上下文（用于反馈检测）
                if tc.name == "execute_readonly_sql":
                    self._conversation.last_query_context = {
                        "business": tc_business or "default",
                        "query": user_input,
                        "cluster": tool_args.get("cluster", "") if isinstance(tool_args, dict) else "",
                        "sql": tool_args.get("sql", "") if isinstance(tool_args, dict) else "",
                    }

                tool_results.append(
                    self.provider.build_tool_result_message(
                        tc.id, ToolExecutionService.summarize_tool_result(tc.name, result_text)
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
        return await self._tool_execution.route_tool_call(tool_name, arguments)

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
