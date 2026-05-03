"""Tests for the CLI entry point (main.py)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.main import (
    main, main_entry, EXIT_COMMANDS, SLASH_COMMANDS,
    _build_welcome_message, _likely_feedback,
    SlashCommandCompleter, GhostTextProcessor,
)
from src.business_registry import BusinessEntry


def _make_prompt_session(inputs):
    """Create a mock PromptSession whose .prompt() returns values from inputs iterable."""
    mock_session = MagicMock()
    mock_session.prompt = MagicMock(side_effect=inputs)
    return mock_session


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


class TestSlashCommands:
    """Test SLASH_COMMANDS structure for usage hints."""

    def test_all_entries_have_three_elements(self):
        for cmd, entry in SLASH_COMMANDS.items():
            assert len(entry) == 3, f"{cmd} should have 3 elements (completion, desc, usage), got {len(entry)}"

    def test_commands_with_args_have_usage(self):
        """Commands whose completion text ends with space (meaning they take args) must have non-empty usage."""
        for cmd, (completion, desc, usage) in SLASH_COMMANDS.items():
            if completion.endswith(" "):
                assert usage, f"{cmd} takes arguments but has no usage hint"

    def test_no_arg_commands_have_empty_usage(self):
        """Commands that don't take arguments should have empty usage."""
        for cmd, (completion, desc, usage) in SLASH_COMMANDS.items():
            if not completion.endswith(" "):
                assert usage == "", f"{cmd} takes no arguments but has usage: {usage}"


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
    @patch("src.main.PromptSession")
    async def test_exit_command_exits_loop(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["exit"])
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_quit_command_exits_loop(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["quit"])
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_q_command_exits_loop(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["q"])
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_exit_case_insensitive(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["EXIT"])
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_empty_input_continues(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["", "exit"])
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_successful_query(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(return_value="查询结果：共3条记录")
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["查询所有数据", "exit"])

        await main()

        mock_agent.run_query.assert_called_once_with("查询所有数据")
        captured = capsys.readouterr()
        assert "查询结果：共3条记录" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_explicit_feedback_bypasses_llm_and_saves_lesson(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(return_value="查询结果：共3条记录")
        mock_agent.extract_explicit_feedback_lesson.return_value = "默认优先查询可用数据：过滤已删除和已禁用记录，除非用户明确要求查看全部或包含禁用数据。"
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["查询所有数据", "记住，后续查询默认只查可用的数据", "exit"])

        await main()

        mock_agent.extract_explicit_feedback_lesson.assert_called_once_with("记住，后续查询默认只查可用的数据")
        mock_agent.extract_feedback_lesson.assert_not_called()
        mock_agent.record_feedback.assert_called_once()
        captured = capsys.readouterr()
        assert "Saved lesson" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_query_error_allows_continue(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.run_query = AsyncMock(side_effect=Exception("API 连接失败"))
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["查询数据", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "API 连接失败" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_eof_exits_gracefully(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=EOFError)
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_keyboard_interrupt_exits_gracefully(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=KeyboardInterrupt)
        await main()

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_add_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/add order http://host:8765/sse 订单", "exit"])

        await main()

        mock_agent.add_business.assert_called_once_with("order", "http://host:8765/sse", "订单", api_key="")
        captured = capsys.readouterr()
        assert "Added" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_remove_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.remove_business = AsyncMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/remove order", "exit"])

        await main()

        mock_agent.remove_business.assert_called_once_with("order")
        captured = capsys.readouterr()
        assert "Removed" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_list_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.list_businesses.return_value = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
        ]
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/list", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "digitalhuman" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_remember_command_saves_rule(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_last_business.return_value = "digitalhuman"
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/remember 默认只查可用数据", "exit"])

        await main()

        mock_agent.add_preference_rule.assert_called_once_with(
            "digitalhuman", "默认只查可用数据", source="manual"
        )
        captured = capsys.readouterr()
        assert "Saved default rule" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_rules_command_prints_rules(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.list_preference_rules.return_value = [
            MagicMock(business="digitalhuman", rule="默认只查可用数据")
        ]
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/rules", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "默认只查可用数据" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_memory_command_hides_user_feedback_entries(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_error_memory_entries.return_value = [
            SimpleNamespace(business="digitalhuman", error_type="USER_FEEDBACK", lesson="默认只查可用数据"),
            SimpleNamespace(business="digitalhuman", error_type="FORBIDDEN_TABLE", lesson="只能查询白名单中的表"),
        ]
        mock_agent.get_error_memory_businesses.return_value = ["digitalhuman"]
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/memory", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "FORBIDDEN_TABLE" in captured.out
        assert "USER_FEEDBACK" not in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_query_prints_applied_and_overridden_rules(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
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
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["查询所有数据", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "Applying default rules" in captured.out
        assert "Applied rules" in captured.out
        assert "Overridden rules" in captured.out
        assert "Route" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_clear_command_all(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/clear", "exit"])

        await main()

        mock_agent.clear_error_memory.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_clear_command_specific_business(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/clear digitalhuman", "exit"])

        await main()

        mock_agent.clear_error_memory.assert_called_once_with(business="digitalhuman")
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_business_current_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.get_locked_business.return_value = "digitalhuman"
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/business current", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "Locked business" in captured.out
        assert "digitalhuman" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_business_set_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/business set digitalhuman", "exit"])

        await main()

        mock_agent.lock_business.assert_called_once_with("digitalhuman")
        captured = capsys.readouterr()
        assert "Locked business" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_business_clear_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/business clear", "exit"])

        await main()

        mock_agent.clear_locked_business.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_plan_command(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
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
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/plan 查询数字人的数据", "exit"])

        await main()

        mock_agent.build_query_plan.assert_awaited_once_with("查询数字人的数据")
        captured = capsys.readouterr()
        assert "Query Plan" in captured.out
        assert "digitalhuman" in captured.out
        assert "默认只查可用数据" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_unknown_slash_command_with_similar(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/li", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        assert "/list" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    @patch("src.main.PromptSession")
    async def test_unknown_slash_command_no_similar(self, mock_session_cls, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        mock_session_cls.return_value.prompt_async = AsyncMock(side_effect=["/xyz", "exit"])

        await main()

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        assert "Tab" in captured.out


class TestMainEntry:
    """Test the sync entry point wrapper."""

    @patch("src.main.main", new_callable=AsyncMock)
    def test_main_entry_calls_asyncio_run(self, mock_main):
        with patch("src.main.asyncio") as mock_asyncio:
            with patch("sys.argv", ["query-agent"]):
                main_entry()
            mock_asyncio.run.assert_called_once()


class TestGhostTextProcessor:
    """Test GhostTextProcessor inline gray hints."""

    def _make_ti(self, text, lineno=0, line_count=1):
        """Create a minimal TransformationInput mock."""
        from prompt_toolkit.document import Document
        doc = Document(text, cursor_position=len(text))
        ti = MagicMock()
        ti.document = doc
        ti.lineno = lineno
        ti.fragments = [("", text)]  # single line, no style
        return ti

    def test_add_command_shows_usage_hint(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        ti = self._make_ti("/add ")
        result = proc.apply_transformation(ti)
        # fragments should include ghost text
        fragments = result.fragments
        text = "".join(f[1] for f in fragments)
        assert "<name>" in text

    def test_add_with_one_arg_shrinks_hint(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        ti = self._make_ti("/add order ")
        result = proc.apply_transformation(ti)
        fragments = result.fragments
        text = "".join(f[1] for f in fragments)
        assert "<sse_url>" in text
        assert "<name>" not in text

    def test_no_arg_command_no_hint(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        ti = self._make_ti("/list ")
        result = proc.apply_transformation(ti)
        # Should be unchanged (no ghost text added)
        fragments = result.fragments
        text = "".join(f[1] for f in fragments)
        assert text == "/list "

    def test_non_slash_input_no_hint(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        ti = self._make_ti("查询数据")
        result = proc.apply_transformation(ti)
        fragments = result.fragments
        text = "".join(f[1] for f in fragments)
        assert text == "查询数据"

    def test_all_args_filled_no_hint(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        # /add has 4 usage tokens; provide 4 args
        ti = self._make_ti("/add order http://x/sse display key")
        result = proc.apply_transformation(ti)
        fragments = result.fragments
        text = "".join(f[1] for f in fragments)
        assert "<name>" not in text
        assert "<sse_url>" not in text

    def test_non_last_line_passthrough(self):
        proc = GhostTextProcessor(SLASH_COMMANDS)
        # Use a mock document where line_count > 1, so lineno=0 is not the last line
        ti = MagicMock()
        ti.document = MagicMock()
        ti.document.text = "/add "
        ti.document.line_count = 3
        ti.lineno = 0
        ti.fragments = [("", "/add ")]
        result = proc.apply_transformation(ti)
        # Non-last line should pass through unchanged
        assert result.fragments == ti.fragments


class TestSlashCommandCompleter:
    """Test SlashCommandCompleter command and argument completions."""

    def test_command_name_completion(self):
        completer = SlashCommandCompleter()
        from prompt_toolkit.document import Document
        doc = Document("/a", cursor_position=2)
        completions = list(completer.get_completions(doc, MagicMock()))
        texts = [c.text for c in completions]
        assert "/add" in texts

    def test_no_completion_for_non_slash(self):
        completer = SlashCommandCompleter()
        from prompt_toolkit.document import Document
        doc = Document("查询", cursor_position=2)
        completions = list(completer.get_completions(doc, MagicMock()))
        assert len(completions) == 0

    def test_business_subcommand_completion(self):
        mock_agent = MagicMock()
        mock_agent.registry.list_businesses.return_value = []
        completer = SlashCommandCompleter(mock_agent)
        from prompt_toolkit.document import Document
        doc = Document("/business ", cursor_position=10)
        completions = list(completer.get_completions(doc, MagicMock()))
        texts = [c.text for c in completions]
        assert "current" in texts
        assert "set" in texts
        assert "clear" in texts
