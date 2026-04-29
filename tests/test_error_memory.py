"""错误记忆模块的单元测试。"""

import json
import os
import tempfile

import pytest

from src.error_memory import ErrorMemoryManager, ErrorEntry


class TestErrorEntryBusiness:
    def test_default_business_empty(self):
        entry = ErrorEntry(
            timestamp="2024-01-01T00:00:00",
            user_query="test",
            error_type="SQL_REJECTED",
        )
        assert entry.business == ""

    def test_business_field(self):
        entry = ErrorEntry(
            timestamp="2024-01-01T00:00:00",
            user_query="test",
            error_type="SQL_REJECTED",
            business="digitalhuman",
        )
        assert entry.business == "digitalhuman"


class TestErrorMemoryManagerBusiness:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "error_memory.json")
        self.manager = ErrorMemoryManager(memory_path=self._path)

    def test_add_error_with_business(self):
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            bad_sql="DELETE FROM tb_scene",
            lesson="不要生成写操作 SQL",
        )
        entries = self.manager.get_entries()
        assert len(entries) == 1
        assert entries[0].business == "digitalhuman"

    def test_add_error_without_business(self):
        self.manager.add_error(
            user_query="test",
            error_type="QUERY_ERROR",
            lesson="SQL 语法错误",
        )
        entries = self.manager.get_entries()
        assert entries[0].business == ""

    def test_build_memory_prompt_filter_by_business(self):
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="数字人经验",
        )
        self.manager.add_error(
            user_query="查订单",
            error_type="SQL_REJECTED",
            business="order",
            lesson="订单经验",
        )
        self.manager.add_error(
            user_query="通用查询",
            error_type="QUERY_ERROR",
            business="",
            lesson="通用经验",
        )

        # 过滤 digitalhuman 业务：应包含 digitalhuman 经验 + 通用经验
        prompt = self.manager.build_memory_prompt(current_business="digitalhuman")
        assert "数字人经验" in prompt
        assert "通用经验" in prompt
        assert "订单经验" not in prompt

        # 过滤 order 业务：应包含 order 经验 + 通用经验
        prompt = self.manager.build_memory_prompt(current_business="order")
        assert "订单经验" in prompt
        assert "通用经验" in prompt
        assert "数字人经验" not in prompt

    def test_build_memory_prompt_no_filter(self):
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="数字人经验",
        )
        self.manager.add_error(
            user_query="查订单",
            error_type="SQL_REJECTED",
            business="order",
            lesson="订单经验",
        )

        # 不过滤：应包含所有经验
        prompt = self.manager.build_memory_prompt()
        assert "数字人经验" in prompt
        assert "订单经验" in prompt

    def test_persistence_with_business(self):
        self.manager.add_error(
            user_query="test",
            error_type="USER_FEEDBACK",
            business="digitalhuman",
            lesson="应该只查测试环境",
        )

        # 重新加载
        manager2 = ErrorMemoryManager(memory_path=self._path)
        entries = manager2.get_entries()
        assert len(entries) == 1
        assert entries[0].business == "digitalhuman"
        assert entries[0].error_type == "USER_FEEDBACK"

    def test_user_feedback_type(self):
        self.manager.add_error(
            user_query="查训练状态",
            error_type="USER_FEEDBACK",
            business="digitalhuman",
            error_message="不对，应该只查训练成功的",
            lesson="应该只查 status=2 的记录",
        )
        entries = self.manager.get_entries()
        assert entries[0].error_type == "USER_FEEDBACK"
        assert entries[0].lesson == "应该只查 status=2 的记录"

        # 确认出现在 prompt 中
        prompt = self.manager.build_memory_prompt(current_business="digitalhuman")
        assert "应该只查 status=2 的记录" in prompt


class TestErrorDeduplication:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "error_memory.json")
        self.manager = ErrorMemoryManager(memory_path=self._path)

    def test_dedup_same_business_same_lesson(self):
        """相同 business + lesson 不重复记录。"""
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="不要生成写操作 SQL",
        )
        self.manager.add_error(
            user_query="再查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="不要生成写操作 SQL",
        )
        entries = self.manager.get_entries()
        assert len(entries) == 1

    def test_different_business_same_lesson_both_kept(self):
        """不同 business 的相同 lesson 各自保留。"""
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="不要生成写操作 SQL",
        )
        self.manager.add_error(
            user_query="查订单",
            error_type="SQL_REJECTED",
            business="order",
            lesson="不要生成写操作 SQL",
        )
        entries = self.manager.get_entries()
        assert len(entries) == 2

    def test_different_lesson_both_kept(self):
        """不同 lesson 各自保留。"""
        self.manager.add_error(
            user_query="查数字人",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="不要生成写操作 SQL",
        )
        self.manager.add_error(
            user_query="查数字人",
            error_type="QUERY_ERROR",
            business="digitalhuman",
            lesson="SQL 语法有误",
        )
        entries = self.manager.get_entries()
        assert len(entries) == 2

    def test_empty_lesson_not_deduped(self):
        """空 lesson 不做去重（避免意外吞掉无 lesson 的错误）。"""
        self.manager.add_error(
            user_query="test1",
            error_type="QUERY_ERROR",
            business="digitalhuman",
        )
        self.manager.add_error(
            user_query="test2",
            error_type="QUERY_ERROR",
            business="digitalhuman",
        )
        entries = self.manager.get_entries()
        assert len(entries) == 2


class TestClearByBusiness:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "error_memory.json")
        self.manager = ErrorMemoryManager(memory_path=self._path)

    def test_clear_specific_business(self):
        """只清除指定业务的记忆。"""
        self.manager.add_error(
            user_query="q1", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="经验1",
        )
        self.manager.add_error(
            user_query="q2", error_type="SQL_REJECTED",
            business="order", lesson="经验2",
        )
        self.manager.add_error(
            user_query="q3", error_type="QUERY_ERROR",
            business="", lesson="通用经验",
        )

        self.manager.clear(business="digitalhuman")
        entries = self.manager.get_entries()
        assert len(entries) == 2
        assert all(e.business != "digitalhuman" for e in entries)

    def test_clear_all(self):
        """不指定 business 清除全部。"""
        self.manager.add_error(
            user_query="q1", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="经验1",
        )
        self.manager.add_error(
            user_query="q2", error_type="SQL_REJECTED",
            business="order", lesson="经验2",
        )

        self.manager.clear()
        assert len(self.manager.get_entries()) == 0

    def test_clear_nonexistent_business_noop(self):
        """清除不存在的业务不影响其他记忆。"""
        self.manager.add_error(
            user_query="q1", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="经验1",
        )
        self.manager.clear(business="nonexistent")
        assert len(self.manager.get_entries()) == 1

    def test_clear_persists(self):
        """清除后持久化生效。"""
        self.manager.add_error(
            user_query="q1", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="经验1",
        )
        self.manager.add_error(
            user_query="q2", error_type="SQL_REJECTED",
            business="order", lesson="经验2",
        )

        self.manager.clear(business="digitalhuman")

        manager2 = ErrorMemoryManager(memory_path=self._path)
        entries = manager2.get_entries()
        assert len(entries) == 1
        assert entries[0].business == "order"


class TestGetBusinesses:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "error_memory.json")
        self.manager = ErrorMemoryManager(memory_path=self._path)

    def test_returns_unique_businesses(self):
        self.manager.add_error(
            user_query="q1", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="a",
        )
        self.manager.add_error(
            user_query="q2", error_type="SQL_REJECTED",
            business="digitalhuman", lesson="b",
        )
        self.manager.add_error(
            user_query="q3", error_type="SQL_REJECTED",
            business="order", lesson="c",
        )
        self.manager.add_error(
            user_query="q4", error_type="QUERY_ERROR",
            business="", lesson="d",
        )

        businesses = self.manager.get_businesses()
        assert businesses == ["digitalhuman", "order"]

    def test_empty_when_no_entries(self):
        assert self.manager.get_businesses() == []


class TestNamespaceIsolation:
    def test_different_paths_are_isolated(self):
        tmpdir = tempfile.mkdtemp()
        path_a = os.path.join(tmpdir, "a", "error_memory.json")
        path_b = os.path.join(tmpdir, "b", "error_memory.json")
        os.makedirs(os.path.dirname(path_a), exist_ok=True)
        os.makedirs(os.path.dirname(path_b), exist_ok=True)

        manager_a = ErrorMemoryManager(memory_path=path_a)
        manager_b = ErrorMemoryManager(memory_path=path_b)

        manager_a.add_error(
            user_query="q1",
            error_type="SQL_REJECTED",
            business="digitalhuman",
            lesson="经验A",
        )
        manager_b.add_error(
            user_query="q2",
            error_type="SQL_REJECTED",
            business="order",
            lesson="经验B",
        )

        assert len(manager_a.get_entries()) == 1
        assert manager_a.get_entries()[0].lesson == "经验A"
        assert len(manager_b.get_entries()) == 1
        assert manager_b.get_entries()[0].lesson == "经验B"
