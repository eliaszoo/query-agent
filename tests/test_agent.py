"""Agent 核心模块的单元测试。

测试 _convert_mcp_tools_to_anthropic、_serialize_tool_result、
_merge_tools_with_business_param、_route_tool_call
以及 QueryAgent 的初始化和消息循环逻辑。
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import QueryAgent, _convert_mcp_tools_to_anthropic, _merge_tools_with_business_param
from src.error_memory import ErrorMemoryManager


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
        agent.error_memory = ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        )

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "FORBIDDEN_TABLE",
            "error_message": "表 xxx 不在白名单中",
        })

        agent._check_and_record_error(
            user_query="查数字人",
            tool_name="execute_readonly_sql",
            tool_input={"business": "digitalhuman", "sql": "SELECT * FROM xxx"},
            result_text=result_text,
            business="digitalhuman",
        )

        entries = agent.error_memory.get_entries()
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

        agent.error_memory = ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        )

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "UNSAFE_SQL",
            "error_message": "包含写操作",
        })

        agent._check_and_record_error(
            user_query="查数据",
            tool_name="execute_readonly_sql",
            tool_input={"sql": "DELETE FROM tb_scene"},
            result_text=result_text,
        )

        entries = agent.error_memory.get_entries()
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
        agent.error_memory = ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        )

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "CONNECTION_ERROR",
            "error_message": "集群 'test' 连接失败",
        })

        agent._check_and_record_error(
            user_query="查数据",
            tool_name="execute_readonly_sql",
            tool_input={"sql": "SELECT 1"},
            result_text=result_text,
        )

        entries = agent.error_memory.get_entries()
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
        agent.error_memory = ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        )

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "CONFIG_ERROR",
            "error_message": "环境变量未设置",
        })

        agent._check_and_record_error(
            user_query="查数据",
            tool_name="execute_readonly_sql",
            tool_input={"sql": "SELECT 1"},
            result_text=result_text,
        )

        entries = agent.error_memory.get_entries()
        assert len(entries) == 0

    async def test_query_error_not_recorded(self, tmp_path):
        """QUERY_ERROR（SQL 语法错误）不记录 — Agent 应先查表结构。"""
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
        agent.error_memory = ErrorMemoryManager(
            memory_path=str(tmp_path / "error_memory.json")
        )

        import json
        result_text = json.dumps({
            "success": False,
            "error_type": "QUERY_ERROR",
            "error_message": "语法错误",
        })

        agent._check_and_record_error(
            user_query="查数据",
            tool_name="execute_readonly_sql",
            tool_input={"sql": "SELECT name FROM tb_scene"},
            result_text=result_text,
        )

        entries = agent.error_memory.get_entries()
        assert len(entries) == 0


class TestExtractFeedbackLesson:
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
        result = QueryAgent._summarize_tool_result("get_table_schema", result_text)
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
        result = QueryAgent._summarize_tool_result("execute_readonly_sql", result_text)
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
        result = QueryAgent._summarize_tool_result("execute_readonly_sql", result_text)
        data = json.loads(result)
        assert len(data["rows"]) == 5
        assert "note" not in data

    def test_error_result_passthrough(self):
        result_text = json.dumps({"success": False, "error_type": "CONNECTION_ERROR", "error_message": "fail"})
        result = QueryAgent._summarize_tool_result("execute_readonly_sql", result_text)
        assert result == result_text

    def test_non_json_passthrough(self):
        result = QueryAgent._summarize_tool_result("execute_readonly_sql", "not json")
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
        agent._conversation_history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        agent._trim_history()
        assert len(agent._conversation_history) == 6

    def test_long_history_compresses_older(self, tmp_path):
        agent = self._make_agent(tmp_path)
        # 10 messages = 5 turns, older should compress
        history = []
        for i in range(1, 6):
            history.append({"role": "user", "content": f"query {i}"})
            history.append({"role": "assistant", "content": f"answer {i}"})
        agent._conversation_history = history
        agent._trim_history()

        # Recent 3 turns (6 messages) should be intact
        recent = agent._conversation_history[-6:]
        assert recent[0]["content"] == "query 3"
        assert recent[1]["content"] == "answer 3"

        # Older turns should be compressed (only user + [历史] assistant)
        older = agent._conversation_history[:-6]
        assert any("[历史]" in m.get("content", "") for m in older)

    def test_extract_text_from_string_content(self):
        assert QueryAgent._extract_text_from_content("hello") == "hello"

    def test_extract_text_from_block_list(self):
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert QueryAgent._extract_text_from_content(content) == "hello \nworld"

    def test_extract_text_from_empty(self):
        assert QueryAgent._extract_text_from_content(None) == ""


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
        agent._indexes_loaded = True
        # 使用临时目录的 field_knowledge 文件，避免测试间共享
        fk_path = str(tmp_path / "field_knowledge.json")
        from src.field_knowledge import FieldKnowledgeManager
        agent.field_knowledge = FieldKnowledgeManager(knowledge_path=fk_path)
        return agent

    def test_parse_enum_values(self):
        """解析括号枚举格式。"""
        result = QueryAgent._parse_enum_values("1(自研), 2(阿里云), 3(腾讯云)")
        assert result == "1=自研, 2=阿里云, 3=腾讯云"

    def test_parse_eq_values(self):
        """解析等号枚举格式。"""
        result = QueryAgent._parse_eq_values("5 = 火山, 1 = 正常, 2 = 禁用")
        assert result == "5=火山, 1=正常, 2=禁用"

    def test_infer_table_from_sql(self):
        """从 SQL 推断表名。"""
        assert QueryAgent._infer_table_from_sql("SELECT * FROM tb_voice WHERE id=1") == "tb_voice"
        assert QueryAgent._infer_table_from_sql("") == ""
        assert QueryAgent._infer_table_from_sql("SELECT 1") == ""

    def test_structured_field_knowledge_tag(self, tmp_path):
        """结构化 HTML 注释声明：优先解析。"""
        agent = self._make_agent(tmp_path)
        response = (
            "查询结果：...\n\n"
            '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山,1=自研,2=阿里云"},'
            '{"table":"tb_voice","field":"forbidden_status","values":"1=正常,2=禁用"}] -->'
        )
        agent._auto_extract_field_knowledge(response)
        entries = agent.field_knowledge.get_entries()
        assert any(e.table == "tb_voice" and e.column == "origin" and "5=火山" in e.description for e in entries)
        assert any(e.table == "tb_voice" and e.column == "forbidden_status" and "1=正常" in e.description for e in entries)

    def test_structured_tag_strips_from_display(self, tmp_path):
        """结构化注释从展示文本中剥离。"""
        import re
        tag_pattern = QueryAgent._FIELD_KNOWLEDGE_TAG
        response = (
            "查询结果：42\n"
            '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山"}] -->'
        )
        display = tag_pattern.sub('', response).rstrip()
        assert "FIELD_KNOWLEDGE" not in display
        assert "查询结果：42" in display

    def test_fallback_field_eq_pattern(self, tmp_path):
        """回退模式: **来源(origin)**: 5 = 火山。"""
        agent = self._make_agent(tmp_path)
        agent._last_query_context = {"sql": "SELECT origin FROM tb_voice WHERE deleted_at IS NULL"}
        response = "- **来源(origin)**: 5 = 火山\n- **可见性(visibility)**: 1 = 公用"
        agent._auto_extract_field_knowledge(response)
        entries = agent.field_knowledge.get_entries()
        assert any(e.column == "origin" and "5=火山" in e.description for e in entries)
        assert any(e.column == "visibility" and "1=公用" in e.description for e in entries)

    def test_fallback_field_enum_pattern(self, tmp_path):
        """回退模式: origin: 1(自研), 2(阿里云) — 需 SQL 推断表名。"""
        agent = self._make_agent(tmp_path)
        agent._last_query_context = {"sql": "SELECT * FROM tb_voice"}
        response = "origin: 1(自研), 2(阿里云), 3(腾讯云)"
        agent._auto_extract_field_knowledge(response)
        entries = agent.field_knowledge.get_entries()
        assert any(e.table == "tb_voice" and e.column == "origin" for e in entries)

    def test_fallback_table_field_enum_pattern(self, tmp_path):
        """回退模式: tb_voice.origin: 1(自研), 2(阿里云) — 直接提取。"""
        agent = self._make_agent(tmp_path)
        response = "tb_voice.origin: 1(自研), 2(阿里云)"
        agent._auto_extract_field_knowledge(response)
        entries = agent.field_knowledge.get_entries()
        assert any(e.table == "tb_voice" and e.column == "origin" and "1=自研" in e.description for e in entries)

    def test_no_extract_without_sql_for_fallback(self, tmp_path):
        """回退模式无 SQL 上下文时不提取。"""
        agent = self._make_agent(tmp_path)
        agent._last_query_context = {}
        response = "origin: 1(自研), 2(阿里云)"
        agent._auto_extract_field_knowledge(response)
        entries = agent.field_knowledge.get_entries()
        assert not any(e.column == "origin" for e in entries)

    def test_mark_prompt_dirty_on_extract(self, tmp_path):
        """提取字段知识后标记 prompt 需重建。"""
        agent = self._make_agent(tmp_path)
        agent._cached_system_prompt = "old prompt"
        agent._prompt_dirty = False
        response = '<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"5=火山"}] -->'
        agent._auto_extract_field_knowledge(response)
        assert agent._prompt_dirty is True
        response = "tb_voice.origin: 5 = 火山"
        agent._auto_extract_field_knowledge(response)
        assert agent._prompt_dirty is True


class TestRiskNoteParse:
    """_parse_risk_note 风险声明解析测试。"""

    def test_index_driven_no_risk(self):
        """索引驱动 → 无风险。"""
        level, reasons = QueryAgent._parse_risk_note("索引驱动: app_id")
        assert level == ""
        assert reasons == []

    def test_full_scan(self):
        """全表扫描 → high。"""
        level, reasons = QueryAgent._parse_risk_note("全表扫描风险")
        assert level == "high"
        assert any("全表扫描" in r for r in reasons)

    def test_select_star(self):
        """SELECT * → medium。"""
        level, reasons = QueryAgent._parse_risk_note("SELECT * 返回全列")
        assert level == "medium"
        assert any("SELECT *" in r for r in reasons)

    def test_like_wildcard(self):
        """LIKE 前导通配符 → medium。"""
        level, reasons = QueryAgent._parse_risk_note("LIKE 前导通配符")
        assert level == "medium"
        assert any("LIKE" in r for r in reasons)

    def test_full_scan_plus_index_driven(self):
        """全表扫描 + 索引驱动同时出现 → high 优先（全表扫描是真实风险）。"""
        level, reasons = QueryAgent._parse_risk_note("全表扫描风险，索引驱动: id")
        assert level == "high"

    def test_index_driven_ignores_select_star(self):
        """索引驱动时 SELECT * 不报风险（查询已高效，返回列多只是信息提示）。"""
        level, reasons = QueryAgent._parse_risk_note("索引驱动: id, SELECT * 返回全列")
        assert level == ""
        assert reasons == []

    def test_unknown_note_defaults_medium(self):
        """未知 risk_note → medium。"""
        level, reasons = QueryAgent._parse_risk_note("其他风险提示")
        assert level == "medium"
        assert reasons == ["其他风险提示"]
