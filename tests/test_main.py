"""Tests for the CLI entry point (main.py)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.main import main, main_entry, EXIT_COMMANDS, _build_welcome_message, _likely_feedback
from src.business_registry import BusinessEntry


class TestLikelyFeedback:
    """Test the _likely_feedback heuristic."""

    def test_short_with_keyword_is_feedback(self):
        assert _likely_feedback("不对，应该查全部") is True

    def test_short_with_correction_keyword(self):
        assert _likely_feedback("错了") is True

    def test_short_with_should_keyword(self):
        assert _likely_feedback("应该用test集群") is True

    def test_long_input_not_feedback(self):
        assert _likely_feedback("帮我查一下数字人平台中训练成功的形象有多少个，按模型分组统计") is False

    def test_short_without_keyword_not_feedback(self):
        assert _likely_feedback("查数字人") is False

    def test_medium_without_keyword_not_feedback(self):
        assert _likely_feedback("查询所有训练成功的数字人形象列表") is False

    def test_medium_with_keyword_is_feedback(self):
        assert _likely_feedback("不对，应该只查测试环境的") is True

    def test_memory_style_feedback_is_feedback(self):
        assert _likely_feedback("记住，后续查询默认只查可用的数据") is True

    def test_default_rule_feedback_is_feedback(self):
        assert _likely_feedback("以后都优先过滤禁用数据") is True


class TestExitCommands:
    """Test that exit commands are correctly defined."""

    def test_exit_commands_contains_expected(self):
        assert "exit" in EXIT_COMMANDS
        assert "quit" in EXIT_COMMANDS
        assert "q" in EXIT_COMMANDS

    def test_exit_commands_count(self):
        assert len(EXIT_COMMANDS) == 3


class TestWelcomeMessage:
    """Test welcome message generation."""

    def test_default_welcome_contains_title(self):
        msg = _build_welcome_message()
        assert "query-agent" in msg

    def test_single_business_welcome(self):
        businesses = [BusinessEntry(name="default", display_name="数字人平台", mcp_server_url="http://a/sse")]
        msg = _build_welcome_message(businesses)
        assert "query-agent" in msg
        assert "数字人平台" not in msg

    def test_multi_business_welcome(self):
        businesses = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
            BusinessEntry(name="order", display_name="订单", mcp_server_url="http://b/sse"),
        ]
        msg = _build_welcome_message(businesses)
        assert "multi-business" in msg
        assert "digitalhuman" in msg
        assert "order" in msg

    def test_welcome_contains_slash_commands(self):
        msg = _build_welcome_message()
        assert "/add" in msg
        assert "/business" in msg
        assert "/list" in msg
        assert "/memory" in msg
        assert "/plan" in msg
        assert "/remember" in msg
        assert "/rules" in msg
        assert "/clear" in msg
        assert "/new" in msg


class TestMainLoop:
    """Test the async main loop behavior."""

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_exit_command_exits_loop(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="exit"):
            await main()
        # Exits cleanly (no exception)

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_quit_command_exits_loop(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="quit"):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_q_command_exits_loop(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="q"):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_exit_case_insensitive(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="EXIT"):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_empty_input_continues(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        inputs = iter(["", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_successful_query(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(return_value="查询结果：共3条记录")
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["查询所有数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.run_query.assert_called_once_with("查询所有数据")
        captured = capsys.readouterr()
        assert "查询结果：共3条记录" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_explicit_feedback_bypasses_llm_and_saves_lesson(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(return_value="查询结果：共3条记录")
        mock_agent.extract_explicit_feedback_lesson.return_value = "默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。"
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["查询所有数据", "记住，后续查询默认只查可用的数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.extract_explicit_feedback_lesson.assert_called_once_with("记住，后续查询默认只查可用的数据")
        mock_agent.extract_feedback_lesson.assert_not_called()
        mock_agent.record_feedback.assert_called_once()
        captured = capsys.readouterr()
        assert "Saved lesson" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_query_error_allows_continue(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(side_effect=Exception("API 连接失败"))
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["查询数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "API 连接失败" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_eof_exits_gracefully(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", side_effect=EOFError):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_keyboard_interrupt_exits_gracefully(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_add_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/add order http://host:8765/sse 订单", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.add_business.assert_called_once_with("order", "http://host:8765/sse", "订单", api_key="")
        captured = capsys.readouterr()
        assert "Added" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_remove_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.remove_business = AsyncMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/remove order", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.remove_business.assert_called_once_with("order")
        captured = capsys.readouterr()
        assert "Removed" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_list_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.list_businesses.return_value = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
        ]
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/list", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "digitalhuman" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_remember_command_saves_rule(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_last_business.return_value = "digitalhuman"
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/remember 默认只查可用数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.add_preference_rule.assert_called_once_with(
            "digitalhuman", "默认只查可用数据", source="manual"
        )
        captured = capsys.readouterr()
        assert "Saved default rule" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_rules_command_prints_rules(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.list_preference_rules.return_value = [
            MagicMock(business="digitalhuman", rule="默认只查可用数据")
        ]
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/rules", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "默认只查可用数据" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_memory_command_hides_user_feedback_entries(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_error_memory_entries.return_value = [
            SimpleNamespace(business="digitalhuman", error_type="USER_FEEDBACK", lesson="默认只查可用数据"),
            SimpleNamespace(business="digitalhuman", error_type="FORBIDDEN_TABLE", lesson="只能查询白名单中的表"),
        ]
        mock_agent.get_error_memory_businesses.return_value = ["digitalhuman"]
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/memory", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "FORBIDDEN_TABLE" in captured.out
        assert "USER_FEEDBACK" not in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_query_prints_applied_and_overridden_rules(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(return_value="查询结果：共3条记录")
        mock_agent.list_preference_rules.side_effect = [[MagicMock(rule="默认只查可用数据")], [MagicMock(rule="默认只查可用数据")]]
        mock_agent.last_metrics = SimpleNamespace(
            duration_seconds=1.2,
            input_tokens=10,
            output_tokens=5,
            tool_calls=1,
            business_selection_strategy="single",
            business_selection_reason="单业务 stdio 模式",
            selected_business="default",
            applied_rules=["默认只查可用数据"],
            overridden_rules=["默认只查可用数据（已覆盖: 用户本次明确要求查看全部/禁用/已删除数据）"],
        )
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["查询所有数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "Applying default rules" in captured.out
        assert "Applied rules" in captured.out
        assert "Overridden rules" in captured.out
        assert "Route" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_clear_command_all(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/clear", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.clear_error_memory.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_clear_command_specific_business(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/clear digitalhuman", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.clear_error_memory.assert_called_once_with(business="digitalhuman")
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_business_current_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_locked_business.return_value = "digitalhuman"
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/business current", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "Locked business" in captured.out
        assert "digitalhuman" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_business_set_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/business set digitalhuman", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.lock_business.assert_called_once_with("digitalhuman")
        captured = capsys.readouterr()
        assert "Locked business" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_business_clear_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/business clear", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.clear_locked_business.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_plan_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.build_query_plan = AsyncMock(
            return_value=SimpleNamespace(
                business="digitalhuman",
                business_display_name="数字人",
                business_strategy="heuristic",
                business_reason="用户输入命中业务名或显示名：数字人",
                locked_business="",
                default_cluster="test",
                active_rules=["默认只查可用数据"],
                overridden_rules=[],
            )
        )
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/plan 查询数字人的数据", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.build_query_plan.assert_awaited_once_with("查询数字人的数据")
        captured = capsys.readouterr()
        assert "Query Plan" in captured.out
        assert "digitalhuman" in captured.out
        assert "默认只查可用数据" in captured.out


class TestMainEntry:
    """Test the sync entry point wrapper."""

    @patch("src.main.main", new_callable=AsyncMock)
    def test_main_entry_calls_asyncio_run(self, mock_main):
        with patch("src.main.asyncio") as mock_asyncio:
            with patch("sys.argv", ["query-agent"]):
                main_entry()
            mock_asyncio.run.assert_called_once()
