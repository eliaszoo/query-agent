"""Tests for the CLI entry point (main.py)."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.main import main, main_entry, EXIT_COMMANDS, _build_welcome_message
from src.business_registry import BusinessEntry


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
        assert "查询 Agent" in msg

    def test_single_business_welcome(self):
        businesses = [BusinessEntry(name="default", display_name="数字人平台", mcp_server_url="http://a/sse")]
        msg = _build_welcome_message(businesses)
        assert "数字人平台" in msg

    def test_multi_business_welcome(self):
        businesses = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
            BusinessEntry(name="order", display_name="订单", mcp_server_url="http://b/sse"),
        ]
        msg = _build_welcome_message(businesses)
        assert "多业务模式" in msg
        assert "digitalhuman" in msg
        assert "order" in msg

    def test_welcome_contains_exit_hint(self):
        msg = _build_welcome_message()
        assert "exit/quit/q" in msg

    def test_welcome_contains_add_command(self):
        msg = _build_welcome_message()
        assert "/add" in msg

    def test_welcome_contains_list_command(self):
        msg = _build_welcome_message()
        assert "/list" in msg


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
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_quit_command_exits_loop(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="quit"):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_q_command_exits_loop(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="q"):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_exit_case_insensitive(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", return_value="EXIT"):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_empty_input_continues(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        inputs = iter(["", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

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
        assert "查询中" in captured.out
        assert "查询结果：共3条记录" in captured.out

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
        assert "查询出错" in captured.out
        assert "API 连接失败" in captured.out
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_eof_exits_gracefully(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", side_effect=EOFError):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_keyboard_interrupt_exits_gracefully(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent_cls.return_value = MagicMock()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await main()
        captured = capsys.readouterr()
        assert "再见" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_add_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.registry = MagicMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/add order http://host:8765/sse 订单", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.registry.register.assert_called_once_with("order", "http://host:8765/sse", "订单")
        captured = capsys.readouterr()
        assert "已添加业务" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_remove_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.registry = MagicMock()
        mock_agent.registry.remove = AsyncMock()
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/remove order", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        mock_agent.registry.remove.assert_called_once_with("order")
        captured = capsys.readouterr()
        assert "已移除业务" in captured.out

    @pytest.mark.asyncio
    @patch("src.main.load_config")
    @patch("src.main.QueryAgent")
    async def test_list_command(self, mock_agent_cls, mock_load_config, capsys):
        mock_load_config.return_value = MagicMock(business_knowledge=MagicMock(description=""), businesses={})
        mock_agent = MagicMock()
        mock_agent.registry = MagicMock()
        mock_agent.registry.list_businesses.return_value = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
        ]
        mock_agent_cls.return_value = mock_agent

        inputs = iter(["/list", "exit"])
        with patch("builtins.input", side_effect=inputs):
            await main()

        captured = capsys.readouterr()
        assert "digitalhuman" in captured.out


class TestMainEntry:
    """Test the sync entry point wrapper."""

    @patch("src.main.main", new_callable=AsyncMock)
    def test_main_entry_calls_asyncio_run(self, mock_main):
        with patch("src.main.asyncio") as mock_asyncio:
            with patch("sys.argv", ["query-agent"]):
                main_entry()
            mock_asyncio.run.assert_called_once()
