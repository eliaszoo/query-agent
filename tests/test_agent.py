"""Agent 核心模块的单元测试。

测试 _convert_mcp_tools_to_anthropic、_serialize_tool_result、
_merge_tools_with_business_param、_route_tool_call
以及 QueryAgent 的初始化和消息循环逻辑。
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import QueryAgent, QueryMetrics, _convert_mcp_tools_to_anthropic, _merge_tools_with_business_param
from src.business_selection_service import BusinessSelectionService
from src.conversation_state import ConversationState
from src.error_memory import ErrorMemoryManager
from src.knowledge_store import KnowledgeStore
from src.llm_provider import LLMResponse
from src.query_rule_executor import QueryRuleExecutor
from src.tool_execution_service import ToolExecutionService


# ---------------------------------------------------------------------------
# _convert_mcp_tools_to_anthropic
# ---------------------------------------------------------------------------


class FakeMCPTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class TestConvertMCPTools:
    def test_empty_list(self):
        assert _convert_mcp_tools_to_anthropic([]) == []

    def test_single_tool(self):
        tool = FakeMCPTool(
            name="execute_readonly_sql",
            description="执行只读 SQL",
            input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
        )
        result = _convert_mcp_tools_to_anthropic([tool])
        assert len(result) == 1
        assert result[0]["name"] == "execute_readonly_sql"
        assert result[0]["description"] == "执行只读 SQL"
        assert result[0]["input_schema"]["type"] == "object"

    def test_none_description_becomes_empty_string(self):
        tool = FakeMCPTool(name="t", description=None, input_schema={})
        result = _convert_mcp_tools_to_anthropic([tool])
        assert result[0]["description"] == ""

    def test_multiple_tools(self):
        tools = [
            FakeMCPTool("a", "desc_a", {}),
            FakeMCPTool("b", "desc_b", {}),
            FakeMCPTool("c", "desc_c", {}),
        ]
        result = _convert_mcp_tools_to_anthropic(tools)
        assert [t["name"] for t in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _merge_tools_with_business_param
# ---------------------------------------------------------------------------


class TestMergeToolsWithBusinessParam:
    def test_empty_input(self):
        assert _merge_tools_with_business_param({}) == []

    def test_single_business(self):
        tools_per_business = {
            "digitalhuman": [
                {
                    "name": "execute_readonly_sql",
                    "description": "执行只读 SQL",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "cluster": {"type": "string"},
                            "sql": {"type": "string"},
                        },
                        "required": ["cluster", "sql"],
                    },
                },
            ],
        }
        merged = _merge_tools_with_business_param(tools_per_business)
        assert len(merged) == 1
        assert merged[0]["name"] == "execute_readonly_sql"
        props = merged[0]["input_schema"]["properties"]
        assert "business" in props
        assert props["business"]["enum"] == ["digitalhuman"]
        assert merged[0]["input_schema"]["required"] == ["business", "cluster", "sql"]

    def test_multiple_businesses(self):
        tools_per_business = {
            "digitalhuman": [
                {
                    "name": "execute_readonly_sql",
                    "description": "执行只读 SQL",
                    "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
                },
            ],
            "order": [
                {
                    "name": "execute_readonly_sql",
                    "description": "执行只读 SQL",
                    "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
                },
            ],
        }
        merged = _merge_tools_with_business_param(tools_per_business)
        assert len(merged) == 1
        assert merged[0]["input_schema"]["properties"]["business"]["enum"] == ["digitalhuman", "order"]

    def test_get_business_knowledge_excluded(self):
        tools_per_business = {
            "digitalhuman": [
                {
                    "name": "get_business_knowledge",
                    "description": "获取业务知识",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "execute_readonly_sql",
                    "description": "执行只读 SQL",
                    "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
                },
            ],
        }
        merged = _merge_tools_with_business_param(tools_per_business)
        assert [t["name"] for t in merged] == ["execute_readonly_sql"]


# ---------------------------------------------------------------------------
# _serialize_tool_result (static methods)
# ---------------------------------------------------------------------------


class FakeToolResultContent:
    def __init__(self, text=None):
        self.text = text


class TestSerializeToolResult:
    def test_text_content(self):
        result = MagicMock()
        result.content = [FakeToolResultContent(text='{"success": true}')]
        assert ToolExecutionService.serialize_tool_result(result) == '{"success": true}'

    def test_no_text_attr_falls_back_to_str(self):
        item = 42  # no .text attribute
        result = MagicMock()
        result.content = [item]
        assert ToolExecutionService.serialize_tool_result(result) == "42"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []
        assert ToolExecutionService.serialize_tool_result(result) == ""


# ---------------------------------------------------------------------------
# QueryAgent.__init__
# ---------------------------------------------------------------------------


class TestAgentInit:
    def test_init_loads_config(self, tmp_path):
        """Agent 初始化时应加载配置并创建 provider。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "claude-sonnet-4-20250514"
  max_tokens: 2048
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        assert agent.config.agent.model == "claude-sonnet-4-20250514"
        assert agent.config.agent.max_tokens == 2048
        assert "test" in agent.config.clusters
        assert agent.provider is not None
        assert agent.mcp_server_params.command is not None

    def test_init_with_businesses_config(self, tmp_path):
        """Agent 初始化时从配置加载业务列表。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # 配置中的 2 个业务（可能还有动态加载的业务）
        entries = agent.registry.list_businesses()
        names = {e.name for e in entries}
        assert "digitalhuman" in names
        assert "order" in names

    def test_init_uses_business_name_as_namespace(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://localhost:8765/sse"
agent:
  model: "claude-sonnet-4-20250514"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        assert "digitalhuman" in agent._knowledge_stores
        assert "digitalhuman" in agent._get_knowledge_store("digitalhuman").error_memory._path

    def test_init_uses_description_as_namespace(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
business_knowledge:
  description: "数字人平台"
agent:
  model: "claude-sonnet-4-20250514"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # stdio mode with description: "default" business storage should be initialized
        assert "default" in agent._knowledge_stores

    def test_init_falls_back_to_path_hash(self, tmp_path):
        """无 businesses、无 description、无显式 namespace 时，使用 default 存储。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "claude-sonnet-4-20250514"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # stdio mode: "default" business storage should be initialized
        assert "default" in agent._knowledge_stores

    def test_init_uses_explicit_storage_namespace(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
storage:
  namespace: "prod-order"
agent:
  model: "claude-sonnet-4-20250514"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # stdio mode: "default" business storage initialized with explicit namespace not used
        # since we now use per-business directories
        assert "default" in agent._knowledge_stores

    def test_init_creates_preference_rules_storage(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "claude-sonnet-4-20250514"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        agent.add_preference_rule("default", "默认只查可用数据", source="test")
        rules = agent.list_preference_rules("default")
        assert len(rules) == 1
        assert rules[0].rule == "默认只查可用数据"
        assert rules[0].rule_type == "available_only"

    def test_lock_and_clear_business(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        agent.lock_business("digitalhuman")
        assert agent.get_locked_business() == "digitalhuman"

        agent.clear_locked_business()
        assert agent.get_locked_business() == ""


class TestQueryPlan:
    @pytest.mark.asyncio
    async def test_build_query_plan_stdio_mode(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        plan = await agent.build_query_plan("查询所有数据")
        assert plan.business == "default"
        assert plan.business_strategy == "single"
        assert plan.business_reason == "单业务 stdio 模式"

    @pytest.mark.asyncio
    async def test_build_query_plan_locked_business(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        agent.lock_business("digitalhuman")
        plan = await agent.build_query_plan("查询订单")
        assert plan.business == "digitalhuman"
        assert plan.business_strategy == "locked"
        assert "已锁定业务" in plan.business_reason
        assert plan.locked_business == "digitalhuman"

    @pytest.mark.asyncio
    async def test_build_query_plan_uses_auto_selection(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        plan = await agent.build_query_plan("查询数字人的数据")
        assert plan.business == "digitalhuman"
        assert plan.business_strategy == "heuristic"
        assert "命中业务名或显示名" in plan.business_reason

    @pytest.mark.asyncio
    async def test_run_query_multi_business_uses_locked_business(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        agent.lock_business("order")
        agent._ensure_knowledge_loaded = AsyncMock()
        agent._build_business_tools = AsyncMock(return_value=[])
        agent._multi_business_conversation_loop = AsyncMock(return_value="ok")
        agent.provider.chat = MagicMock(return_value=LLMResponse(stop_reason="end_turn", text="ok"))

        await agent.run_query("查询数字人的数据")

        assert agent.last_metrics is not None
        assert agent.last_metrics.selected_business == "order"
        assert agent.last_metrics.business_selection_strategy == "locked"
        assert "已锁定业务" in agent.last_metrics.business_selection_reason
        agent._build_business_tools.assert_awaited_once_with("order")


# ---------------------------------------------------------------------------
# _route_tool_call
# ---------------------------------------------------------------------------


class TestRouteToolCall:
    @pytest.mark.asyncio
    async def test_missing_business_parameter(self, tmp_path):
        """未指定 business 参数时返回 MISSING_BUSINESS 错误。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        result = await agent._route_tool_call("execute_readonly_sql", {"sql": "SELECT 1"})
        assert "MISSING_BUSINESS" in result

    @pytest.mark.asyncio
    async def test_invalid_business(self, tmp_path):
        """指定不存在的业务时返回 INVALID_BUSINESS 错误。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        result = await agent._route_tool_call("execute_readonly_sql", {"business": "nonexistent", "sql": "SELECT 1"})
        assert "INVALID_BUSINESS" in result


# ---------------------------------------------------------------------------
# _conversation_loop_core (unified conversation loop)
# ---------------------------------------------------------------------------


class TestConversationLoop:
    @pytest.mark.asyncio
    async def test_direct_text_response(self, tmp_path):
        """当 LLM 直接返回文本（无工具调用）时，应立即返回。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file), confirm_callback=lambda _: True)

        # Mock Anthropic response: direct text, no tool use
        from src.llm_provider import LLMResponse
        mock_response = LLMResponse(
            stop_reason="end_turn",
            text="共有 42 条记录",
        )
        agent.provider.chat = MagicMock(return_value=mock_response)

        tools = [{"name": "execute_readonly_sql", "description": "...", "input_schema": {}}]

        async def execute_tool(name, args, business):
            return '{"success": true}', "default"

        from src.agent import QueryMetrics
        metrics = QueryMetrics(model="test-model")
        result = await agent._conversation_loop_core(
            tools, "有多少条记录？", metrics, None, execute_tool
        )
        assert result == "共有 42 条记录"

    @pytest.mark.asyncio
    async def test_tool_use_then_text(self, tmp_path):
        """当 LLM 先调用工具再返回文本时，应正确处理消息循环。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file), confirm_callback=lambda _: True)

        from src.llm_provider import LLMResponse, ToolCall

        # 跳过索引加载
        agent._tool_execution.indexes_loaded = True

        # First response: tool_use
        resp1 = LLMResponse(
            stop_reason="tool_use",
            tool_calls=[ToolCall(
                id="tool_123",
                name="execute_readonly_sql",
                arguments={"cluster": "test", "sql": "SELECT COUNT(*) FROM tb_scene"},
            )],
            raw_content=MagicMock(),
        )

        # Second response: final text
        resp2 = LLMResponse(
            stop_reason="end_turn",
            text="查询结果：42",
        )

        agent.provider.chat = MagicMock(side_effect=[resp1, resp2])
        agent.provider.build_assistant_message = MagicMock(
            return_value={"role": "assistant", "content": "mock"}
        )
        agent.provider.build_tool_result_message = MagicMock(
            side_effect=lambda tid, content: {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": content,
            }
        )

        # Mock execute_tool callback
        async def execute_tool(name, args, business):
            return '{"success": true, "row_count": 1}', "default"

        tools = [{"name": "execute_readonly_sql", "description": "...", "input_schema": {}}]

        from src.agent import QueryMetrics
        metrics = QueryMetrics(model="test-model")
        result = await agent._conversation_loop_core(
            tools, "有多少条记录？", metrics, None, execute_tool
        )

        assert result == "查询结果：42"
        assert agent.provider.chat.call_count == 2


# ---------------------------------------------------------------------------
# _pre_execute_check
# ---------------------------------------------------------------------------


class TestPreExecuteCheck:
    @pytest.mark.asyncio
    async def test_non_sql_tool_passes(self, tmp_path):
        """非 execute_readonly_sql 工具直接放行。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        result = await agent._tool_execution.pre_execute_check(
            "get_cluster_list", {"cluster": "test"}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_user_cancel_returns_error(self, tmp_path):
        """用户拒绝执行时返回 USER_CANCELLED 错误。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        # confirm_callback 返回 False（用户拒绝）
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file), confirm_callback=lambda _: False)

        # 注入索引信息使 SQL 有风险
        from src.sql_risk_checker import IndexInfo
        agent._risk_checker.update_indexes("default", "test", "tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._tool_execution.indexes_loaded = True

        result = await agent._tool_execution.pre_execute_check(
            "execute_readonly_sql",
            {"cluster": "test", "sql": "SELECT * FROM tb_scene"},
        )
        assert result is not None
        assert "USER_CANCELLED" in result

    @pytest.mark.asyncio
    async def test_user_confirmed_allows_execution(self, tmp_path):
        """用户确认执行时返回 None（允许执行）。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        # confirm_callback 返回 True（用户确认）
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file), confirm_callback=lambda _: True)

        from src.sql_risk_checker import IndexInfo
        agent._risk_checker.update_indexes("default", "test", "tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._tool_execution.indexes_loaded = True

        result = await agent._tool_execution.pre_execute_check(
            "execute_readonly_sql",
            {"cluster": "test", "sql": "SELECT * FROM tb_scene"},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_risk_allows_execution(self, tmp_path):
        """无性能风险时直接允许执行，不调用 confirm。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        confirm_called = False

        def track_confirm(prompt):
            nonlocal confirm_called
            confirm_called = True
            return True

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file), confirm_callback=track_confirm)

        from src.sql_risk_checker import IndexInfo
        agent._risk_checker.update_indexes("default", "test", "tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._tool_execution.indexes_loaded = True

        result = await agent._tool_execution.pre_execute_check(
            "execute_readonly_sql",
            {"cluster": "test", "sql": "SELECT id FROM tb_scene WHERE id = 1"},
        )
        assert result is None
        assert not confirm_called


class TestCheckAndRecordErrorBusiness:
    async def test_records_business_from_tool_input(self, tmp_path):
        """多业务模式下从 tool_input 提取 business。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # 使用临时目录避免测试间共享 error_memory
        biz_store = agent._get_knowledge_store("digitalhuman")
        biz_store.set_error_memory(ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        ))

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "FORBIDDEN_TABLE",
            "error_message": "表 xxx 不在白名单中",
        })

        biz_store.check_and_record_error(
            user_query="查数字人",
            tool_input={"business": "digitalhuman", "sql": "SELECT * FROM xxx"},
            result_text=result_text,
            business="digitalhuman",
            lesson_builder=agent._generate_lesson,
        )

        entries = biz_store.get_error_memory_entries()
        assert len(entries) == 1
        assert entries[0].business == "digitalhuman"

    async def test_records_business_default_for_stdio(self, tmp_path):
        """stdio 模式下 business 默认为 'default'。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        default_store = agent._get_knowledge_store("default")
        default_store.set_error_memory(ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        ))

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "UNSAFE_SQL",
            "error_message": "包含写操作",
        })

        default_store.check_and_record_error(
            user_query="查数据",
            tool_input={"sql": "DELETE FROM tb_scene"},
            result_text=result_text,
            is_stdio_mode=agent._is_stdio_mode,
            lesson_builder=agent._generate_lesson,
        )

        entries = default_store.get_error_memory_entries()
        assert len(entries) == 1
        assert entries[0].business == "default"


class TestGenerateLessonUserFeedback:
    def test_user_feedback_lesson(self):
        lesson = QueryAgent._generate_lesson("USER_FEEDBACK", "不对，应该只查测试环境", "")
        assert "用户反馈" in lesson


class TestSkipUnrecordedErrors:
    async def test_connection_error_not_recorded(self, tmp_path):
        """CONNECTION_ERROR 不记录到错误记忆。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))
        default_store = agent._get_knowledge_store("default")
        default_store.set_error_memory(ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        ))

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "CONNECTION_ERROR",
            "error_message": "集群 'test' 连接失败",
        })

        default_store.check_and_record_error(
            user_query="查数据",
            tool_input={"sql": "SELECT 1"},
            result_text=result_text,
            lesson_builder=agent._generate_lesson,
        )

        entries = agent.get_error_memory_entries()
        assert len(entries) == 0

    async def test_config_error_not_recorded(self, tmp_path):
        """CONFIG_ERROR 不记录到错误记忆。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))
        default_store = agent._get_knowledge_store("default")
        default_store.set_error_memory(ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        ))

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "CONFIG_ERROR",
            "error_message": "环境变量未设置",
        })

        default_store.check_and_record_error(
            user_query="查数据",
            tool_input={"sql": "SELECT 1"},
            result_text=result_text,
            lesson_builder=agent._generate_lesson,
        )

        entries = agent.get_error_memory_entries()
        assert len(entries) == 0

    async def test_query_error_is_recorded(self, tmp_path):
        """QUERY_ERROR（SQL 语法错误）应被记录，避免 Agent 重复犯错。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))
        default_store = agent._get_knowledge_store("default")
        default_store.set_error_memory(ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        ))

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "QUERY_ERROR",
            "error_message": "语法错误",
        })

        default_store.check_and_record_error(
            user_query="查数据",
            tool_input={"sql": "SELECT name FROM tb_scene"},
            result_text=result_text,
            lesson_builder=agent._generate_lesson,
        )

        entries = agent.get_error_memory_entries()
        assert len(entries) == 1
        assert entries[0].error_type == "QUERY_ERROR"


class TestExtractFeedbackLesson:
    def test_extract_explicit_feedback_lesson_for_available_data_rule(self):
        lesson = QueryAgent.extract_explicit_feedback_lesson("记住，后续查询默认只查可用的数据")
        assert lesson == "默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。"

    def test_extract_explicit_feedback_rule_for_available_data(self):
        rule = QueryAgent.extract_explicit_feedback_rule("记住，后续查询默认只查可用的数据")
        assert rule is not None
        assert rule["rule_type"] == "available_only"
        assert rule["payload"]["forbidden_status"] == 1

    def test_query_rule_executor_applies_available_only_sql(self):
        rule = MagicMock(
            rule="默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。",
            rule_type="available_only",
            payload={"deleted_at_is_null": True, "forbidden_status": 1},
        )
        result = QueryRuleExecutor.apply(
            "查两个公共音色",
            "default",
            [rule],
            arguments={"sql": "SELECT * FROM tb_voice LIMIT 2"},
        )
        assert "deleted_at IS NULL" in result.arguments["sql"]
        assert "forbidden_status = 1" in result.arguments["sql"]

    def test_query_rule_executor_overrides_available_only_when_user_requests_all(self):
        rule = MagicMock(
            rule="默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。",
            rule_type="available_only",
            payload={"deleted_at_is_null": True, "forbidden_status": 1},
        )
        result = QueryRuleExecutor.apply(
            "查全部数据",
            "default",
            [rule],
            arguments={"sql": "SELECT * FROM tb_voice LIMIT 2"},
        )
        assert "deleted_at IS NULL" not in result.arguments["sql"]
        assert any(item.overridden for item in result.applications)

    def test_query_rule_executor_fills_default_cluster(self):
        rule = MagicMock(
            rule="默认优先查询测试环境，除非用户明确指定生产环境。",
            rule_type="default_cluster_test",
            payload={"cluster": "test"},
        )
        result = QueryRuleExecutor.apply(
            "查两个公共音色",
            "default",
            [rule],
            arguments={"sql": "SELECT * FROM tb_voice LIMIT 2"},
        )
        assert result.arguments["cluster"] == "test"

    def test_extract_explicit_feedback_lesson_returns_none_for_plain_query(self):
        lesson = QueryAgent.extract_explicit_feedback_lesson("帮我查两个公共音色")
        assert lesson is None

    async def test_returns_none_for_new_query(self, tmp_path):
        """当用户输入是新查询时返回 None。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # Mock provider.chat to return NONE
        mock_response = MagicMock()
        mock_response.text = "NONE"
        agent.provider.chat = MagicMock(return_value=mock_response)

        result = await agent.extract_feedback_lesson(
            original_query="查数字人数量",
            agent_response="共有 100 个数字人",
            user_feedback="查一下训练任务数量",
        )
        assert result is None

    async def test_returns_lesson_for_feedback(self, tmp_path):
        """当用户输入是反馈时返回经验教训。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        # Mock provider.chat to return a lesson
        mock_response = MagicMock()
        mock_response.text = "应该只查询测试环境的数据，不要查生产环境"
        agent.provider.chat = MagicMock(return_value=mock_response)

        result = await agent.extract_feedback_lesson(
            original_query="查数字人数量",
            agent_response="生产环境共有 100 个数字人",
            user_feedback="不对，我只要测试环境的",
        )
        assert result == "应该只查询测试环境的数据，不要查生产环境"

    async def test_returns_none_on_exception(self, tmp_path):
        """LLM 调用失败时返回 None。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        agent.provider.chat = MagicMock(side_effect=Exception("API error"))

        result = await agent.extract_feedback_lesson(
            original_query="test",
            agent_response="response",
            user_feedback="feedback",
        )
        assert result is None


class TestSummarizeToolResult:
    """Test _summarize_tool_result method."""

    def test_schema_columns_simplified(self):
        result_text = json.dumps({
            "table_name": "tb_scene",
            "columns": [
                {"name": "id", "type": "bigint", "nullable": False, "key": "PRI",
                 "default": None, "extra": "auto_increment"},
                {"name": "name", "type": "varchar(100)", "nullable": True, "key": "",
                 "default": "", "extra": ""},
            ],
        })
        result = ToolExecutionService.summarize_tool_result("get_table_schema", result_text)
        data = json.loads(result)
        for col in data["columns"]:
            assert "default" not in col
            assert "extra" not in col
            assert "name" in col
            assert "type" in col

    def test_sql_result_truncated_over_10_rows(self):
        rows = [[i, f"name_{i}"] for i in range(20)]
        result_text = json.dumps({
            "success": True,
            "columns": ["id", "name"],
            "rows": rows,
            "row_count": 20,
            "truncated": False,
        })
        result = ToolExecutionService.summarize_tool_result("execute_readonly_sql", result_text)
        data = json.loads(result)
        assert len(data["rows"]) == 10
        assert data["row_count"] == 20
        assert data["truncated"] is True
        assert "仅展示前 10 行" in data["note"]

    def test_sql_result_not_truncated_under_10_rows(self):
        rows = [[i, f"name_{i}"] for i in range(5)]
        result_text = json.dumps({
            "success": True,
            "columns": ["id", "name"],
            "rows": rows,
            "row_count": 5,
            "truncated": False,
        })
        result = ToolExecutionService.summarize_tool_result("execute_readonly_sql", result_text)
        data = json.loads(result)
        assert len(data["rows"]) == 5
        assert "note" not in data

    def test_error_result_passthrough(self):
        result_text = json.dumps({"success": False, "error_type": "CONNECTION_ERROR", "error_message": "fail"})
        result = ToolExecutionService.summarize_tool_result("execute_readonly_sql", result_text)
        assert result == result_text

    def test_non_json_passthrough(self):
        result = ToolExecutionService.summarize_tool_result("execute_readonly_sql", "not json")
        assert result == "not json"


class TestTrimHistory:
    """Test conversation history compression."""

    def _make_agent(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
""")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryAgent(config_path=str(config_file))

    def test_short_history_not_compressed(self, tmp_path):
        agent = self._make_agent(tmp_path)
        # 6 messages = 3 turns, should not compress
        agent._conversation.history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        agent._conversation.trim_history()
        assert len(agent._conversation.history) == 6

    def test_long_history_compresses_older(self, tmp_path):
        agent = self._make_agent(tmp_path)
        # 10 messages = 5 turns, older should compress
        history = []
        for i in range(1, 6):
            history.append({"role": "user", "content": f"query {i}"})
            history.append({"role": "assistant", "content": f"answer {i}"})
        agent._conversation.history = history
        agent._conversation.trim_history()

        # Recent 3 turns (6 messages) should be intact
        recent = agent._conversation.history[-6:]
        assert recent[0]["content"] == "query 3"
        assert recent[1]["content"] == "answer 3"

        # Older turns should be compressed (only user + [历史] assistant)
        older = agent._conversation.history[:-6]
        assert any("[历史]" in m.get("content", "") for m in older)

    def test_extract_text_from_string_content(self):
        assert ConversationState.extract_text_from_content("hello") == "hello"

    def test_extract_text_from_block_list(self):
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert ConversationState.extract_text_from_content(content) == "hello \nworld"

    def test_extract_text_from_empty(self):
        assert ConversationState.extract_text_from_content(None) == ""


class TestFieldKnowledgeAutoExtract:
    """字段含义自动提取测试。"""

    def _make_agent(self, tmp_path):
        """创建带 field_knowledge 的 agent（使用临时目录避免文件污染）。"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "root"
    password: "pass"
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
"""
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))
        agent._tool_execution.indexes_loaded = True
        # 使用临时目录的 field_knowledge 文件，避免测试间共享
        fk_path = str(tmp_path / "field_knowledge.json")
        from src.field_knowledge import FieldKnowledgeManager
        field_knowledge = FieldKnowledgeManager(knowledge_path=fk_path)
        default_store = agent._get_knowledge_store("default")
        default_store.set_field_knowledge(field_knowledge)
        agent._tool_execution.set_field_knowledge_manager(field_knowledge)
        return agent

    @staticmethod
    def _extract(agent: QueryAgent, response_text: str) -> None:
        default_store = agent._get_knowledge_store("default")
        default_store.auto_extract_field_knowledge(
            response_text=response_text,
            business=agent.get_last_business() or "default",
            sql=(agent._conversation.last_query_context or {}).get("sql", ""),
        )

    def test_parse_enum_values(self):
        """解析括号枚举格式。"""
        result = KnowledgeStore.parse_enum_values("1(自研), 2(阿里云), 3(腾讯云)")
        assert result == "1=自研, 2=阿里云, 3=腾讯云"

    def test_parse_eq_values(self):
        """解析等号枚举格式。"""
        result = KnowledgeStore.parse_eq_values("5 = 火山, 1 = 正常, 2 = 禁用")
        assert result == "5=火山, 1=正常, 2=禁用"

    def test_infer_table_from_sql(self):
        """从 SQL 推断表名。"""
        assert KnowledgeStore.infer_table_from_sql("SELECT * FROM tb_voice WHERE id=1") == "tb_voice"
        assert KnowledgeStore.infer_table_from_sql("") == ""
        assert KnowledgeStore.infer_table_from_sql("SELECT 1") == ""

    def test_structured_field_knowledge_tag(self, tmp_path):
        """结构化 HTML 注释声明：优先解析。"""
        agent = self._make_agent(tmp_path)
        response = (
            "查询结果：...\n\n"
            '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山,1=自研,2=阿里云"},'
            '{"table":"tb_voice","field":"forbidden_status","values":"1=正常,2=禁用"}] -->'
        )
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert any(e.table == "tb_voice" and e.column == "origin" and "5=火山" in e.description for e in entries)
        assert any(e.table == "tb_voice" and e.column == "forbidden_status" and "1=正常" in e.description for e in entries)
        assert all(e.business == "default" for e in entries)

    def test_structured_tag_strips_from_display(self, tmp_path):
        """结构化注释从展示文本中剥离。"""
        import re
        tag_pattern = KnowledgeStore.FIELD_KNOWLEDGE_TAG
        response = (
            "查询结果：42\n"
            '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山"}] -->'
        )
        display = tag_pattern.sub('', response).rstrip()
        assert "FIELD_KNOWLEDGE" not in display
        assert "查询结果：42" in display

    def test_extract_from_markdown_table(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent._conversation.last_query_context = {"sql": "SELECT origin, visibility FROM tb_voice"}
        response = (
            "| 字段 | 音色1 | 音色2 |\n"
            "|---|---|---|\n"
            "| **来源(origin)** | 5（火山） | 1（自研） |\n"
            "| **可见性(visibility)** | 1（公用） | 2（私有） |"
        )
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert any(e.column == "origin" and "5=火山" in e.description and "1=自研" in e.description for e in entries)
        assert any(e.column == "visibility" and "1=公用" in e.description and "2=私有" in e.description for e in entries)

    def test_fallback_field_eq_pattern(self, tmp_path):
        """回退模式: **来源(origin)**: 5 = 火山。"""
        agent = self._make_agent(tmp_path)
        agent._conversation.last_query_context = {"sql": "SELECT origin FROM tb_voice WHERE deleted_at IS NULL"}
        response = "- **来源(origin)**: 5 = 火山\n- **可见性(visibility)**: 1 = 公用"
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert any(e.column == "origin" and "5=火山" in e.description for e in entries)
        assert any(e.column == "visibility" and "1=公用" in e.description for e in entries)

    def test_fallback_field_enum_pattern(self, tmp_path):
        """回退模式: origin: 1(自研), 2(阿里云) — 需 SQL 推断表名。"""
        agent = self._make_agent(tmp_path)
        agent._conversation.last_query_context = {"sql": "SELECT * FROM tb_voice"}
        response = "origin: 1(自研), 2(阿里云), 3(腾讯云)"
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert any(e.table == "tb_voice" and e.column == "origin" for e in entries)

    def test_fallback_table_field_enum_pattern(self, tmp_path):
        """回退模式: tb_voice.origin: 1(自研), 2(阿里云) — 直接提取。"""
        agent = self._make_agent(tmp_path)
        response = "tb_voice.origin: 1(自研), 2(阿里云)"
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert any(e.table == "tb_voice" and e.column == "origin" and "1=自研" in e.description for e in entries)

    def test_no_extract_without_sql_for_fallback(self, tmp_path):
        """回退模式无 SQL 上下文时不提取。"""
        agent = self._make_agent(tmp_path)
        agent._conversation.last_query_context = {}
        response = "origin: 1(自研), 2(阿里云)"
        self._extract(agent, response)
        entries = agent.list_field_knowledge()
        assert not any(e.column == "origin" for e in entries)

    def test_mark_prompt_dirty_on_extract(self, tmp_path):
        """提取字段知识后标记 prompt 需重建。"""
        agent = self._make_agent(tmp_path)
        agent._prompt_service._cached_prompt = "old prompt"
        agent._prompt_service._dirty = False
        response = '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山"}] -->'
        self._extract(agent, response)
        assert agent._prompt_service._dirty is True
        response = "tb_voice.origin: 5 = 火山"
        self._extract(agent, response)
        assert agent._prompt_service._dirty is True

    def test_field_knowledge_isolated_by_business(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.add_field_knowledge("digitalhuman", "tb_voice", "origin", "1=自研")
        agent.add_field_knowledge("order", "tb_voice", "origin", "1=订单")

        dh_entries = agent.list_field_knowledge("digitalhuman")
        order_entries = agent.list_field_knowledge("order")

        assert len(dh_entries) == 1
        assert dh_entries[0].description == "1=自研"
        assert len(order_entries) == 1
        assert order_entries[0].description == "1=订单"

    @pytest.mark.asyncio
    async def test_remove_business_clears_business_field_knowledge(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = QueryAgent(config_path=str(config_file))

        from src.field_knowledge import FieldKnowledgeManager
        # Use per-business temp files
        dh_fk_path = str(tmp_path / "dh_field_knowledge.json")
        order_fk_path = str(tmp_path / "order_field_knowledge.json")
        dh_fk = FieldKnowledgeManager(knowledge_path=dh_fk_path)
        order_fk = FieldKnowledgeManager(knowledge_path=order_fk_path)
        agent._get_knowledge_store("digitalhuman").set_field_knowledge(dh_fk)
        agent._get_knowledge_store("order").set_field_knowledge(order_fk)
        agent.add_field_knowledge("digitalhuman", "tb_voice", "origin", "1=自研")
        agent.add_field_knowledge("order", "tb_voice", "origin", "1=订单")

        await agent.remove_business("digitalhuman")

        assert not agent.list_field_knowledge("digitalhuman")
        assert len(agent.list_field_knowledge("order")) == 1


class TestRiskNoteParse:
    """_parse_risk_note 风险声明解析测试。"""

    def test_index_driven_no_risk(self):
        """索引驱动 → 无风险。"""
        level, reasons = ToolExecutionService.parse_risk_note("索引驱动: app_id")
        assert level == ""
        assert reasons == []

    def test_full_scan(self):
        """全表扫描 → high。"""
        level, reasons = ToolExecutionService.parse_risk_note("全表扫描风险")
        assert level == "high"
        assert any("全表扫描" in r for r in reasons)

    def test_select_star(self):
        """SELECT * → medium。"""
        level, reasons = ToolExecutionService.parse_risk_note("SELECT * 返回全列")
        assert level == "medium"
        assert any("SELECT *" in r for r in reasons)

    def test_like_wildcard(self):
        """LIKE 前导通配符 → medium。"""
        level, reasons = ToolExecutionService.parse_risk_note("LIKE 前导通配符")
        assert level == "medium"
        assert any("LIKE" in r for r in reasons)

    def test_full_scan_plus_index_driven(self):
        """全表扫描 + 索引驱动同时出现 → high 优先（全表扫描是真实风险）。"""
        level, reasons = ToolExecutionService.parse_risk_note("全表扫描风险，索引驱动: id")
        assert level == "high"

    def test_index_driven_ignores_select_star(self):
        """索引驱动时 SELECT * 不报风险（查询已高效，返回列多只是信息提示）。"""
        level, reasons = ToolExecutionService.parse_risk_note("索引驱动: id, SELECT * 返回全列")
        assert level == ""
        assert reasons == []

    def test_unknown_note_defaults_medium(self):
        """未知 risk_note → medium。"""
        level, reasons = ToolExecutionService.parse_risk_note("其他风险提示")
        assert level == "medium"
        assert reasons == ["其他风险提示"]


class TestBusinessSelection:
    def _make_multi_business_agent(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryAgent(config_path=str(config_file))

    @pytest.mark.asyncio
    async def test_select_business_by_name(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        selection = await agent._business_selector.select_business("帮我查 digitalhuman 的数据")
        assert selection.business is not None
        assert selection.business.name == "digitalhuman"
        assert selection.strategy == "heuristic"

    @pytest.mark.asyncio
    async def test_select_business_by_display_name(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        selection = await agent._business_selector.select_business("帮我查订单业务的数据")
        assert selection.business is not None
        assert selection.business.name == "order"
        assert selection.strategy == "heuristic"

    @pytest.mark.asyncio
    async def test_select_business_returns_none_when_ambiguous(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        mock_response = MagicMock()
        mock_response.text = "NONE"
        agent.provider.chat = MagicMock(return_value=mock_response)
        agent._business_selector = BusinessSelectionService(
            provider=agent.provider,
            model=agent.config.agent.model,
            registry=agent.registry,
        )
        selection = await agent._business_selector.select_business("帮我查一下数据")
        assert selection.business is None
        assert selection.strategy == "fallback_all"

    @pytest.mark.asyncio
    async def test_select_business_falls_back_to_llm(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        mock_response = MagicMock()
        mock_response.text = "order"
        agent.provider.chat = MagicMock(return_value=mock_response)
        agent._business_selector = BusinessSelectionService(
            provider=agent.provider,
            model=agent.config.agent.model,
            registry=agent.registry,
        )
        selection = await agent._business_selector.select_business("帮我看一下退款进度")

        assert selection.business is not None
        assert selection.business.name == "order"
        assert selection.strategy == "llm"

    @pytest.mark.asyncio
    async def test_run_query_multi_business_prefers_selected_business(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        agent._ensure_knowledge_loaded = AsyncMock()
        agent._build_business_tools = AsyncMock(return_value=[{"name": "execute_readonly_sql"}])
        agent._build_merged_tools = AsyncMock(return_value=[{"name": "execute_readonly_sql"}])
        agent._multi_business_conversation_loop = AsyncMock(return_value="ok")
        metrics = QueryMetrics()

        result = await agent._run_query_multi_business("查询数字人的数据", metrics)

        assert result == "ok"
        agent._build_business_tools.assert_awaited_once_with("digitalhuman")
        agent._build_merged_tools.assert_not_called()
        assert metrics.selected_business == "digitalhuman"
        assert metrics.business_selection_strategy == "heuristic"

    @pytest.mark.asyncio
    async def test_run_query_multi_business_falls_back_when_not_selected(self, tmp_path):
        agent = self._make_multi_business_agent(tmp_path)
        agent._ensure_knowledge_loaded = AsyncMock()
        agent._build_business_tools = AsyncMock(return_value=[{"name": "execute_readonly_sql"}])
        agent._build_merged_tools = AsyncMock(return_value=[{"name": "execute_readonly_sql"}])
        agent._multi_business_conversation_loop = AsyncMock(return_value="ok")
        metrics = QueryMetrics()

        result = await agent._run_query_multi_business("查询所有数据", metrics)

        assert result == "ok"
        agent._build_merged_tools.assert_awaited_once()
        assert metrics.selected_business == "all"
        assert metrics.business_selection_strategy == "fallback_all"
