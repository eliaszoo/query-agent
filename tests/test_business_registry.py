"""BusinessRegistry 单元测试。"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.business_registry import BusinessRegistry, BusinessEntry
from src.config import BusinessKnowledge, MCPServerEndpoint


class TestRegister:
    def test_register_with_mcp_server_url(self):
        """向后兼容：传入 mcp_server_url 自动转为 servers 列表。"""
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        entries = registry.list_businesses()
        assert len(entries) == 1
        assert entries[0].name == "digitalhuman"
        assert entries[0].display_name == "数字人"
        assert len(entries[0].servers) == 1
        assert entries[0].servers[0].url == "http://host:8765/sse"

    def test_register_default_display_name(self):
        registry = BusinessRegistry()
        registry.register("order", "http://host:8765/sse")
        assert registry.list_businesses()[0].display_name == "order"

    def test_register_with_api_key(self):
        """向后兼容：传入 api_key 自动关联到第一个 server。"""
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人", api_key="secret123")
        entry = registry.get_entry("digitalhuman")
        assert entry.servers[0].api_key == "secret123"

    def test_register_no_api_key_default_empty(self):
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse")
        entry = registry.get_entry("digitalhuman")
        assert entry.servers[0].api_key == ""

    def test_register_with_servers_list(self):
        """新格式：传入 servers 列表支持多 MCP Server。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse", api_key="key_sh"),
            MCPServerEndpoint(url="http://bj:8765/sse", api_key="key_bj"),
        ]
        registry.register("digitalhuman", servers=servers, display_name="数字人")
        entry = registry.get_entry("digitalhuman")
        assert len(entry.servers) == 2
        assert entry.servers[0].url == "http://sh:8765/sse"
        assert entry.servers[1].url == "http://bj:8765/sse"


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

    @pytest.mark.asyncio
    async def test_remove_closes_cached_session(self):
        """移除业务时关闭其所有 server 的缓存 session。"""
        registry = BusinessRegistry()
        registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        # session cache 现在按 server URL 键控
        registry._session_cache["http://host:8765/sse"] = (MagicMock(), MagicMock())

        with patch.object(registry, "_close_cached_session", new_callable=AsyncMock) as mock_close:
            await registry.remove("digitalhuman")

        mock_close.assert_awaited_once_with("http://host:8765/sse")

    @pytest.mark.asyncio
    async def test_remove_closes_all_server_sessions(self):
        """多 server 业务移除时关闭所有 server 的缓存 session。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse"),
            MCPServerEndpoint(url="http://bj:8765/sse"),
        ]
        registry.register("digitalhuman", servers=servers, display_name="数字人")
        registry._session_cache["http://sh:8765/sse"] = (MagicMock(), MagicMock())
        registry._session_cache["http://bj:8765/sse"] = (MagicMock(), MagicMock())

        with patch.object(registry, "_close_cached_session", new_callable=AsyncMock) as mock_close:
            await registry.remove("digitalhuman")

        assert mock_close.await_count == 2


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


class TestClusterRouting:
    @pytest.mark.asyncio
    async def test_fetch_cluster_routing(self):
        """从多个 MCP Server 获取集群列表，构建路由表。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse"),
            MCPServerEndpoint(url="http://bj:8765/sse"),
        ]
        registry.register("digitalhuman", servers=servers, display_name="数字人")

        # 上海 MCP Server 返回 huangpu 集群
        sh_clusters = json.dumps({"clusters": [{"name": "huangpu"}]})
        # 北京 MCP Server 返回 tiantan 集群
        bj_clusters = json.dumps({"clusters": [{"name": "tiantan"}]})

        with patch.object(registry, "call_tool_on_server", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [sh_clusters, bj_clusters]
            routing = await registry.fetch_cluster_routing("digitalhuman")

        assert "huangpu" in routing
        assert "tiantan" in routing
        assert routing["huangpu"].url == "http://sh:8765/sse"
        assert routing["tiantan"].url == "http://bj:8765/sse"

    @pytest.mark.asyncio
    async def test_call_tool_routes_by_cluster(self):
        """call_tool 根据 cluster 参数路由到对应 MCP Server。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse"),
            MCPServerEndpoint(url="http://bj:8765/sse"),
        ]
        registry.register("digitalhuman", servers=servers, display_name="数字人")

        # 手动设置路由表
        entry = registry.get_entry("digitalhuman")
        entry.cluster_routing = {
            "huangpu": servers[0],
            "tiantan": servers[1],
        }

        with patch.object(registry, "call_tool_on_server", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"success": true}'
            await registry.call_tool("digitalhuman", "execute_readonly_sql", {"cluster": "huangpu", "sql": "SELECT 1"})

        # 验证路由到了上海 server
        mock_call.assert_awaited_once()
        called_server = mock_call.call_args[0][0]
        assert called_server.url == "http://sh:8765/sse"

    @pytest.mark.asyncio
    async def test_call_tool_fallback_to_first_server(self):
        """cluster 参数不在路由表中时，回退到第一个 server。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse"),
        ]
        registry.register("digitalhuman", servers=servers, display_name="数字人")

        with patch.object(registry, "call_tool_on_server", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"success": true}'
            await registry.call_tool("digitalhuman", "get_cluster_list", {"business": "digitalhuman"})

        called_server = mock_call.call_args[0][0]
        assert called_server.url == "http://sh:8765/sse"


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

    @pytest.mark.asyncio
    async def test_close_sessions_keeps_entries(self):
        registry = BusinessRegistry()
        registry.register("a", "http://a/sse")
        # session cache 按 server URL 键控
        registry._session_cache["http://a/sse"] = (MagicMock(), MagicMock())

        with patch.object(registry, "_close_cached_session", new_callable=AsyncMock) as mock_close:
            await registry.close_sessions()

        mock_close.assert_awaited_once_with("http://a/sse")
        assert registry.has_business("a")


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


class TestDiscoverFromServers:
    """discover_from_servers 测试：从 MCP Server 动态发现业务。"""

    @pytest.mark.asyncio
    async def test_discovers_businesses_from_servers(self):
        """从多个 MCP Server 发现业务并自动注册。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://sh:8765/sse"),
            MCPServerEndpoint(url="http://bj:8765/sse"),
        ]

        # 上海 server 返回数字人和版权音乐
        sh_result = json.dumps({
            "businesses": [
                {"name": "digitalhuman", "display_name": "数字人", "clusters": ["huangpu"]},
                {"name": "copyright_music", "display_name": "版权音乐", "clusters": ["huangpu"]},
            ]
        })
        # 北京 server 也返回数字人和版权音乐
        bj_result = json.dumps({
            "businesses": [
                {"name": "digitalhuman", "display_name": "数字人", "clusters": ["tiantan"]},
                {"name": "copyright_music", "display_name": "版权音乐", "clusters": ["tiantan"]},
            ]
        })

        with patch.object(registry, "call_tool_on_server", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [sh_result, bj_result]
            discovered = await registry.discover_from_servers(servers)

        assert set(discovered) == {"digitalhuman", "copyright_music"}
        assert registry.has_business("digitalhuman")
        assert registry.has_business("copyright_music")

        # 每个业务关联了两个 server
        dh_entry = registry.get_entry("digitalhuman")
        assert len(dh_entry.servers) == 2
        cm_entry = registry.get_entry("copyright_music")
        assert len(cm_entry.servers) == 2

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing(self):
        """已存在的业务不被覆盖，但会合并新 server。"""
        registry = BusinessRegistry()
        existing_server = MCPServerEndpoint(url="http://existing:8765/sse")
        registry.register("digitalhuman", servers=[existing_server], display_name="数字人")

        servers = [MCPServerEndpoint(url="http://sh:8765/sse")]
        sh_result = json.dumps({
            "businesses": [
                {"name": "digitalhuman", "display_name": "数字人", "clusters": ["huangpu"]},
            ]
        })

        with patch.object(registry, "call_tool_on_server", new_callable=AsyncMock, return_value=sh_result):
            discovered = await registry.discover_from_servers(servers)

        # digitalhuman 已存在，不算新发现
        assert discovered == []
        # 但新 server 被合并
        entry = registry.get_entry("digitalhuman")
        assert len(entry.servers) == 2

    @pytest.mark.asyncio
    async def test_server_failure_does_not_block_others(self):
        """一个 server 失败不影响其他 server 的发现。"""
        registry = BusinessRegistry()
        servers = [
            MCPServerEndpoint(url="http://bad:8765/sse"),
            MCPServerEndpoint(url="http://good:8765/sse"),
        ]

        good_result = json.dumps({
            "businesses": [
                {"name": "digitalhuman", "display_name": "数字人", "clusters": ["test"]},
            ]
        })

        async def mock_call(server, tool_name, args):
            if "bad" in server.url:
                raise Exception("连接失败")
            return good_result

        with patch.object(registry, "call_tool_on_server", side_effect=mock_call):
            discovered = await registry.discover_from_servers(servers)

        assert discovered == ["digitalhuman"]