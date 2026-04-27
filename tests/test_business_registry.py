"""BusinessRegistry 单元测试。"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.business_registry import BusinessRegistry, BusinessEntry
from src.config import BusinessKnowledge


class TestRegister:
    def test_register_adds_entry(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        entries = registry.list_businesses()
        assert len(entries) == 1
        assert entries[0].name == "digitalhuman"
        assert entries[0].display_name == "数字人"
        assert entries[0].mcp_server_url == "http://host:8765/sse"

    def test_register_default_display_name(self):
        registry = BusinessRegistry()
        registry.register("order", "http://host:8765/sse")
        assert registry.list_businesses()[0].display_name == "order"

    def test_register_with_api_key(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人", api_key="secret123")
        entry = registry.get_entry("digitalhuman")
        assert entry.api_key == "secret123"

    def test_register_no_api_key_default_empty(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse")
        entry = registry.get_entry("digitalhuman")
        assert entry.api_key == ""


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_existing(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        await registry.remove("digitalhuman")
        assert not registry.has_business("digitalhuman")

    @pytest.mark.asyncio
    async def test_remove_nonexistent_raises(self):
        registry = BusinessRegistry()
        with pytest.raises(KeyError, match="不存在"):
            await registry.remove("nonexistent")


class TestHasBusiness:
    def test_has_business_true(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse")
        assert registry.has_business("digitalhuman")

    def test_has_business_false(self):
        registry = BusinessRegistry()
        assert not registry.has_business("nonexistent")


class TestGetEntry:
    def test_get_entry_existing(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        entry = registry.get_entry("digitalhuman")
        assert entry.name == "digitalhuman"

    def test_get_entry_nonexistent_raises(self):
        registry = BusinessRegistry()
        with pytest.raises(KeyError, match="不存在"):
            registry.get_entry("nonexistent")


class TestFetchBusinessKnowledge:
    @pytest.mark.asyncio
    async def test_cached_knowledge_returned(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse")
        entry = registry.get_entry("digitalhuman")
        entry.knowledge = BusinessKnowledge(description="数字人平台")

        result = await registry.fetch_business_knowledge("digitalhuman")
        assert result.description == "数字人平台"

    @pytest.mark.asyncio
    async def test_fetch_from_mcp_server(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse")

        mock_result = MagicMock()
        mock_result.content = [MagicMock(text=json.dumps({
            "description": "数字人平台",
            "term_mappings": {"模型": "tb_model 表"},
            "table_relationships": ["tb_scene.model_id → tb_model.id"],
            "status_codes": ["status: 1=活跃"],
            "custom_rules": ["不要使用子查询"],
        }))]

        with patch.object(registry, "call_tool", new_callable=AsyncMock, return_value=json.dumps({
            "description": "数字人平台",
            "term_mappings": {"模型": "tb_model 表"},
            "table_relationships": ["tb_scene.model_id → tb_model.id"],
            "status_codes": ["status: 1=活跃"],
            "custom_rules": ["不要使用子查询"],
        })):
            result = await registry.fetch_business_knowledge("digitalhuman")

        assert result.description == "数字人平台"
        assert result.term_mappings == {"模型": "tb_model 表"}

    @pytest.mark.asyncio
    async def test_fetch_failure_uses_default(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")

        with patch.object(registry, "call_tool", new_callable=AsyncMock, side_effect=Exception("连接失败")):
            result = await registry.fetch_business_knowledge("digitalhuman")

        assert result.description == "数字人"


class TestCloseAll:
    @pytest.mark.asyncio
    async def test_close_all_clears_entries(self):
        registry = BusinessRegistry()
        registry.register("a", "http://a/sse")
        registry.register("b", "http://b/sse")
        await registry.close_all()
        assert not registry.list_businesses()


class TestSerializeToolResult:
    def test_text_content(self):
        result = MagicMock()
        result.content = [MagicMock(text="hello")]
        assert BusinessRegistry._serialize_tool_result(result) == "hello"

    def test_empty_content(self):
        result = MagicMock()
        result.content = []
        assert BusinessRegistry._serialize_tool_result(result) == ""

    def test_no_text_attr_falls_back_to_str(self):
        result = MagicMock()
        result.content = [42]
        assert BusinessRegistry._serialize_tool_result(result) == "42"
