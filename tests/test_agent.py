"""Agent 核心模块的单元测试。

测试 _convert_mcp_tools_to_anthropic、_serialize_tool_result、
_merge_tools_with_business_param、_route_tool_call
以及 QueryAgent 的初始化和消息循环逻辑。
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import QueryAgent, _convert_mcp_tools_to_anthropic, _merge_tools_with_business_param


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
        assert QueryAgent._serialize_tool_result(result) == '{"success": true}'

    def test_no_text_attr_falls_back_to_str(self):
        item = 42  # no .text attribute
        result = MagicMock()
        result.content = [item]
        assert QueryAgent._serialize_tool_result(result) == "42"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []
        assert QueryAgent._serialize_tool_result(result) == ""


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

        assert len(agent.registry.list_businesses()) == 2
        entries = agent.registry.list_businesses()
        names = {e.name for e in entries}
        assert "digitalhuman" in names
        assert "order" in names


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
# _conversation_loop (single business stdio mode)
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

        session = AsyncMock()
        tools = [{"name": "execute_readonly_sql", "description": "...", "input_schema": {}}]

        from src.agent import QueryMetrics
        metrics = QueryMetrics(model="test-model")
        result = await agent._conversation_loop(session, tools, "有多少条记录？", metrics)
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
        agent._indexes_loaded = True

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

        # Mock MCP session.call_tool
        session = AsyncMock()
        tool_result = MagicMock()
        tool_result.content = [FakeToolResultContent(text='{"success": true, "row_count": 1}')]
        session.call_tool = AsyncMock(return_value=tool_result)

        tools = [{"name": "execute_readonly_sql", "description": "...", "input_schema": {}}]

        from src.agent import QueryMetrics
        metrics = QueryMetrics(model="test-model")
        result = await agent._conversation_loop(session, tools, "有多少条记录？", metrics)

        assert result == "查询结果：42"
        session.call_tool.assert_called_once_with(
            "execute_readonly_sql",
            {"cluster": "test", "sql": "SELECT COUNT(*) FROM tb_scene"},
        )
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

        result = await agent._pre_execute_check(
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
        agent._risk_checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._indexes_loaded = True

        result = await agent._pre_execute_check(
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
        agent._risk_checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._indexes_loaded = True

        result = await agent._pre_execute_check(
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
        agent._risk_checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        agent._indexes_loaded = True

        result = await agent._pre_execute_check(
            "execute_readonly_sql",
            {"cluster": "test", "sql": "SELECT id FROM tb_scene WHERE id = 1"},
        )
        assert result is None
        assert not confirm_called
