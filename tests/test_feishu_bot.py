"""飞书机器人模块测试 — 会话管理、消息格式化、文本提取。"""

import json
import time

import pytest

from src.feishu_session import FeishuSessionManager, SessionEntry
from src.feishu_message import (
    build_query_card,
    build_error_card,
    build_thinking_card,
    _parse_markdown_table,
)
from src.feishu_bot import _extract_text_from_event
from src.agent import QueryMetrics


# ---------------------------------------------------------------------------
# FeishuSessionManager 测试
# ---------------------------------------------------------------------------


class TestFeishuSessionManager:
    """会话管理器测试。"""

    @staticmethod
    def _write_config(tmp_path):
        """写入包含 businesses 的测试配置。"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "agent:\n  provider: anthropic\n"
            "businesses:\n  test:\n    name: test\n    display_name: test\n    mcp_server_url: http://localhost:8765/sse\n"
        )
        return str(config_path)

    def test_get_agent_creates_new_session(self, tmp_path):
        """首次获取用户 agent 应创建新会话。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=600)

        agent = mgr.get_agent("user_001")
        assert agent is not None
        assert mgr.active_session_count == 1

    def test_get_agent_returns_same_instance(self, tmp_path):
        """同一用户多次获取应返回同一 agent 实例。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=600)

        agent1 = mgr.get_agent("user_001")
        agent2 = mgr.get_agent("user_001")
        assert agent1 is agent2
        assert mgr.active_session_count == 1

    def test_different_users_get_different_sessions(self, tmp_path):
        """不同用户应获得不同的 agent 实例。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=600)

        agent1 = mgr.get_agent("user_001")
        agent2 = mgr.get_agent("user_002")
        assert agent1 is not agent2
        assert mgr.active_session_count == 2

    def test_cleanup_expired(self, tmp_path):
        """超时会话应被清理。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=10)

        agent = mgr.get_agent("user_001")
        assert mgr.active_session_count == 1

        # 模拟超时：修改 last_active
        mgr._sessions["user_001"].last_active = time.time() - 100

        cleaned = mgr.cleanup_expired()
        assert cleaned == 1
        assert mgr.active_session_count == 0

    def test_cleanup_not_expired(self, tmp_path):
        """未超时的会话不应被清理。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=600)

        mgr.get_agent("user_001")
        cleaned = mgr.cleanup_expired()
        assert cleaned == 0
        assert mgr.active_session_count == 1

    def test_get_lock_returns_same_lock(self, tmp_path):
        """同一用户多次获取锁应返回同一实例。"""
        mgr = FeishuSessionManager(config_path=self._write_config(tmp_path), session_ttl=600)

        lock1 = mgr.get_lock("user_001")
        lock2 = mgr.get_lock("user_001")
        assert lock1 is lock2


# ---------------------------------------------------------------------------
# 消息格式化测试
# ---------------------------------------------------------------------------


class TestFeishuMessage:
    """飞书消息格式化测试。"""

    def test_parse_markdown_table_basic(self):
        """基本 Markdown 表格解析。"""
        md = "| 名称 | 数量 |\n|---|---|\n| 苹果 | 10 |\n| 橘子 | 20 |"
        result = _parse_markdown_table(md)
        assert result is not None
        assert result[0] == ["名称", "数量"]
        assert result[1] == ["苹果", "10"]
        assert result[2] == ["橘子", "20"]

    def test_parse_markdown_table_no_table(self):
        """无表格时返回 None。"""
        result = _parse_markdown_table("这是一段普通文本")
        assert result is None

    def test_parse_markdown_table_with_alignment(self):
        """带对齐标记的表格。"""
        md = "| 名称 | 数量 |\n|:---|---:|\n| 苹果 | 10 |"
        result = _parse_markdown_table(md)
        assert result is not None
        assert result[0] == ["名称", "数量"]

    def test_build_query_card_text(self):
        """纯文本结果应生成有效卡片 JSON。"""
        card_str = build_query_card("查询完成，共 5 条记录")
        card = json.loads(card_str)
        assert card["header"]["title"]["content"] == "查询结果"
        assert card["header"]["template"] == "blue"
        assert len(card["elements"]) > 0

    def test_build_query_card_with_table(self):
        """包含 Markdown 表格的结果应生成表格卡片。"""
        md = "查询结果：\n\n| 名称 | 数量 |\n|---|---|\n| 苹果 | 10 |\n\n共 1 条记录"
        card_str = build_query_card(md)
        card = json.loads(card_str)
        assert card["header"]["title"]["content"] == "查询结果"
        # 应包含 column_set 元素
        has_column_set = any(
            e.get("tag") == "column_set" for e in card["elements"]
        )
        assert has_column_set

    def test_build_query_card_with_metrics(self):
        """带 metrics 的卡片应包含元信息。"""
        metrics = QueryMetrics(
            duration_seconds=1.5,
            selected_business="digitalhuman",
            model="claude-sonnet-4-20250514",
        )
        card_str = build_query_card("结果", metrics)
        card = json.loads(card_str)
        header = card["header"]
        assert "subtitle" in header
        assert "digitalhuman" in header["subtitle"]["content"]

    def test_build_error_card(self):
        """错误卡片应为红色模板。"""
        card_str = build_error_card("查询超时")
        card = json.loads(card_str)
        assert card["header"]["template"] == "red"
        assert card["header"]["title"]["content"] == "查询出错"

    def test_build_thinking_card(self):
        """思考中卡片应为蓝色模板。"""
        card_str = build_thinking_card()
        card = json.loads(card_str)
        assert card["header"]["template"] == "blue"
        assert "正在查询" in card["header"]["title"]["content"]


# ---------------------------------------------------------------------------
# 文本提取测试
# ---------------------------------------------------------------------------


class TestExtractText:
    """飞书消息文本提取测试。"""

    def test_extract_plain_text(self):
        """普通文本消息提取。"""
        event_data = {"content": json.dumps({"text": "查询可用数字人"})}
        text = _extract_text_from_event(event_data)
        assert text == "查询可用数字人"

    def test_extract_text_with_mention(self):
        """去掉 @机器人 mention。"""
        event_data = {"content": json.dumps({"text": "@_user_1 查询可用数字人"})}
        text = _extract_text_from_event(event_data)
        assert text == "查询可用数字人"

    def test_extract_text_with_multiple_mentions(self):
        """去掉多个 mention。"""
        event_data = {"content": json.dumps({"text": "@_user_1 @_user_2 查询数字人"})}
        text = _extract_text_from_event(event_data)
        assert text == "查询数字人"

    def test_extract_text_empty(self):
        """空消息返回空字符串。"""
        event_data = {"content": "{}"}
        text = _extract_text_from_event(event_data)
        assert text == ""

    def test_extract_text_invalid_json(self):
        """无效 JSON 返回空字符串。"""
        event_data = {"content": "not json"}
        text = _extract_text_from_event(event_data)
        assert text == ""
