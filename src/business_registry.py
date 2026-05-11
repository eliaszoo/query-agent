"""多业务 MCP Server 连接管理器。

管理多个业务的 MCP Server SSE 连接，支持：
- 多 MCP Server：一个业务可连接多个地域 MCP Server
- 集群路由：根据 cluster 名称路由到对应地域的 MCP Server
- 懒连接：注册时不立即连接，首次使用时建立
- 会话缓存：已建立的会话复用，避免每次查询重连
- 业务知识获取：首次连接后自动调用 get_business_knowledge
- 动态增减：运行时添加/移除业务
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.config import BusinessKnowledge, MCPServerEndpoint

logger = logging.getLogger(__name__)


@dataclass
class BusinessEntry:
    """一个业务的注册信息。"""

    name: str  # 业务标识，如 "digitalhuman"
    display_name: str  # 显示名，如 "数字人"
    servers: list[MCPServerEndpoint] = field(default_factory=list)  # 多 MCP Server 端点
    knowledge: Optional[BusinessKnowledge] = None  # 从 MCP server 获取的业务知识
    # 集群路由表：cluster_name → MCPServerEndpoint
    # 在调用 fetch_cluster_routing() 后填充
    cluster_routing: dict[str, MCPServerEndpoint] = field(default_factory=dict)
    _connected: bool = field(default=False, repr=False, init=False)


class BusinessRegistry:
    """管理多个业务的 MCP Server 连接。

    使用方式：
        registry = BusinessRegistry()
        registry.register("digitalhuman", servers=[MCPServerEndpoint(url="http://sh:8765/sse"), ...])
        await registry.fetch_cluster_routing("digitalhuman")  # 聚合集群路由
        result = await registry.call_tool("digitalhuman", "execute_readonly_sql", {...})
        await registry.close_all()
    """

    def __init__(self) -> None:
        self._entries: dict[str, BusinessEntry] = {}
        # 缓存 session：server_url → (session, ctx)
        self._session_cache: dict[str, tuple[ClientSession, Any]] = {}

    def register(
        self,
        name: str,
        mcp_server_url: str = "",
        display_name: str = "",
        api_key: str = "",
        servers: list[MCPServerEndpoint] | None = None,
    ) -> None:
        """注册一个业务（不立即连接）。

        向后兼容：支持传入单个 mcp_server_url + api_key，
        或传入 servers 列表。

        Args:
            name: 业务标识。
            mcp_server_url: 单个 MCP Server SSE URL（向后兼容）。
            display_name: 显示名称。
            api_key: 单个 MCP Server 鉴权密钥（向后兼容）。
            servers: 多 MCP Server 端点列表。
        """
        if servers is None:
            servers = []
        # 向后兼容：mcp_server_url 自动转为 servers
        if mcp_server_url and not servers:
            servers = [MCPServerEndpoint(url=mcp_server_url, api_key=api_key)]

        self._entries[name] = BusinessEntry(
            name=name,
            display_name=display_name or name,
            servers=servers,
        )
        logger.info(
            "已注册业务: %s (%s) -> %d server(s)",
            name, display_name, len(servers),
        )

    async def remove(self, name: str) -> None:
        """移除一个业务（断开连接）。"""
        if name not in self._entries:
            raise KeyError(f"业务 '{name}' 不存在")
        entry = self._entries[name]
        for server in entry.servers:
            await self._close_cached_session(server.url)
        del self._entries[name]
        logger.info("已移除业务: %s", name)

    def list_businesses(self) -> list[BusinessEntry]:
        """列出所有已注册的业务。"""
        return list(self._entries.values())

    def has_business(self, name: str) -> bool:
        """检查业务是否已注册。"""
        return name in self._entries

    def get_entry(self, name: str) -> BusinessEntry:
        """获取业务条目。"""
        if name not in self._entries:
            raise KeyError(f"业务 '{name}' 不存在")
        return self._entries[name]

    async def fetch_cluster_routing(self, name: str) -> dict[str, MCPServerEndpoint]:
        """获取并聚合某业务所有 MCP Server 的集群，构建路由表。

        对每个 MCP Server 调用 get_cluster_list(business=name)，
        将返回的集群名映射到对应的 MCPServerEndpoint。

        Args:
            name: 业务标识。

        Returns:
            集群路由表 {cluster_name: MCPServerEndpoint}。
        """
        entry = self.get_entry(name)
        routing: dict[str, MCPServerEndpoint] = {}

        for server in entry.servers:
            try:
                result_text = await self.call_tool_on_server(
                    server, "get_cluster_list", {"business": name}
                )
                data = json.loads(result_text)
                for cluster_info in data.get("clusters", []):
                    cluster_name = cluster_info.get("name", "")
                    if cluster_name:
                        if cluster_name in routing:
                            logger.warning(
                                "集群 '%s' 在多个 MCP Server 中存在，使用后注册的 server",
                                cluster_name,
                            )
                        routing[cluster_name] = server
            except Exception:
                logger.warning(
                    "从 MCP Server %s 获取集群列表失败", server.url, exc_info=True
                )

        entry.cluster_routing = routing
        logger.info(
            "业务 '%s' 集群路由: %s",
            name,
            {k: v.url for k, v in routing.items()},
        )
        return routing

    async def fetch_all_cluster_routings(self) -> None:
        """获取所有业务的集群路由表。"""
        for name in list(self._entries.keys()):
            try:
                await self.fetch_cluster_routing(name)
            except Exception:
                logger.warning("获取业务 '%s' 集群路由失败", name, exc_info=True)

    async def discover_from_servers(self, servers: list[MCPServerEndpoint]) -> list[str]:
        """从 MCP Server 动态发现业务并自动注册。

        对每个 server 调用 get_business_list，
        发现业务后构建 business → servers 映射，
        自动调用 register() 注册新发现的业务。

        Args:
            servers: 顶层 mcp_servers 列表。

        Returns:
            新发现的业务名称列表。
        """
        # biz_name → [MCPServerEndpoint, ...] 收集
        biz_servers_map: dict[str, list[MCPServerEndpoint]] = {}
        # biz_name → display_name 收集
        biz_display_names: dict[str, str] = {}

        for server in servers:
            try:
                result_text = await self.call_tool_on_server(
                    server, "get_business_list", {}
                )
                data = json.loads(result_text)
                for biz_info in data.get("businesses", []):
                    biz_name = biz_info.get("name", "")
                    if not biz_name:
                        continue
                    biz_servers_map.setdefault(biz_name, []).append(server)
                    # 取第一个有 display_name 的
                    if biz_name not in biz_display_names or not biz_display_names[biz_name]:
                        biz_display_names[biz_name] = biz_info.get("display_name", "")
            except Exception:
                logger.warning(
                    "从 MCP Server %s 获取业务列表失败", server.url, exc_info=True
                )

        # 注册新发现的业务（不覆盖已存在的）
        discovered = []
        for biz_name, biz_servers in biz_servers_map.items():
            if not self.has_business(biz_name):
                self.register(
                    biz_name,
                    display_name=biz_display_names.get(biz_name, ""),
                    servers=biz_servers,
                )
                discovered.append(biz_name)
            else:
                # 已存在：合并新的 server（可能是之前不知道的 server）
                entry = self.get_entry(biz_name)
                existing_urls = {s.url for s in entry.servers}
                for s in biz_servers:
                    if s.url not in existing_urls:
                        entry.servers.append(s)
                        existing_urls.add(s.url)

        if discovered:
            logger.info("从 MCP Server 发现 %d 个业务: %s", len(discovered), discovered)
        return discovered

    @asynccontextmanager
    async def get_session(self, name: str) -> AsyncIterator[ClientSession]:
        """获取业务的 MCP 会话（使用第一个 server，懒连接）。"""
        entry = self.get_entry(name)
        if not entry.servers:
            raise ValueError(f"业务 '{name}' 没有配置 MCP Server")
        server = entry.servers[0]
        async with self._get_server_session(server) as session:
            yield session

    @asynccontextmanager
    async def get_session_for_cluster(
        self, name: str, cluster: str
    ) -> AsyncIterator[ClientSession]:
        """根据集群名路由到对应的 MCP Server 获取会话。

        Args:
            name: 业务标识。
            cluster: 集群名称。

        Yields:
            已初始化的 ClientSession。
        """
        entry = self.get_entry(name)
        server = entry.cluster_routing.get(cluster)
        if server is None:
            # 回退到第一个 server
            if entry.servers:
                server = entry.servers[0]
                logger.debug(
                    "集群 '%s' 未在路由表中，回退到第一个 server: %s",
                    cluster, server.url,
                )
            else:
                raise ValueError(f"业务 '{name}' 没有配置 MCP Server")

        async with self._get_server_session(server) as session:
            yield session

    @asynccontextmanager
    async def _get_server_session(
        self, server: MCPServerEndpoint
    ) -> AsyncIterator[ClientSession]:
        """获取指定 MCP Server 的会话（每次新建连接，用完关闭）。"""
        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"
        async with sse_client(server.url, headers=headers or None) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    @asynccontextmanager
    async def get_cached_session(self, name: str) -> AsyncIterator[ClientSession]:
        """获取缓存的 session（使用第一个 server），同一业务复用连接。"""
        entry = self.get_entry(name)
        if not entry.servers:
            raise ValueError(f"业务 '{name}' 没有配置 MCP Server")
        server = entry.servers[0]
        async with self._get_cached_server_session(server) as session:
            yield session

    @asynccontextmanager
    async def _get_cached_server_session(
        self, server: MCPServerEndpoint
    ) -> AsyncIterator[ClientSession]:
        """获取缓存的 MCP Server session。"""
        if server.url in self._session_cache:
            session, _ = self._session_cache[server.url]
            try:
                yield session
                return
            except Exception:
                await self._close_cached_session(server.url)

        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        ctx = sse_client(server.url, headers=headers or None)
        read_stream, write_stream = await ctx.__aenter__()
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()

        self._session_cache[server.url] = (session, ctx)
        try:
            yield session
        except Exception:
            await self._close_cached_session(server.url)
            raise

    async def _close_cached_session(self, server_url: str) -> None:
        """关闭并清除指定 server 的缓存 session。"""
        if server_url not in self._session_cache:
            return
        session, ctx = self._session_cache.pop(server_url)
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            logger.warning("关闭 session 失败: %s", server_url, exc_info=True)
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("关闭 SSE context 失败: %s", server_url, exc_info=True)

    async def call_tool(self, name: str, tool_name: str, arguments: dict) -> str:
        """在指定业务上调用 MCP 工具（使用缓存的 SSE 连接）。

        如果 arguments 中包含 cluster 且业务有集群路由表，
        则路由到对应地域的 MCP Server。

        Args:
            name: 业务标识。
            tool_name: 工具名称。
            arguments: 工具参数。

        Returns:
            工具结果的 JSON 字符串。
        """
        entry = self.get_entry(name)
        cluster = arguments.get("cluster", "")

        # 根据集群路由选择 server
        server = None
        if cluster and cluster in entry.cluster_routing:
            server = entry.cluster_routing[cluster]
        elif entry.servers:
            server = entry.servers[0]

        if server is None:
            raise ValueError(f"业务 '{name}' 没有可用的 MCP Server")

        return await self.call_tool_on_server(server, tool_name, arguments)

    async def call_tool_on_server(
        self, server: MCPServerEndpoint, tool_name: str, arguments: dict
    ) -> str:
        """在指定 MCP Server 上调用工具（复用缓存连接）。"""
        async with self._get_cached_server_session(server) as session:
            result = await session.call_tool(tool_name, arguments)
            return self._serialize_tool_result(result)

    async def fetch_business_knowledge(self, name: str) -> BusinessKnowledge:
        """从 MCP Server 获取业务领域知识并缓存。"""
        entry = self.get_entry(name)
        if entry.knowledge is not None:
            return entry.knowledge

        try:
            result_text = await self.call_tool(name, "get_business_knowledge", {"business": name})
            data = json.loads(result_text)

            if isinstance(data, dict) and data.get("description"):
                entry.knowledge = BusinessKnowledge(
                    description=data.get("description", ""),
                    term_mappings=data.get("term_mappings", {}),
                    table_relationships=data.get("table_relationships", []),
                    status_codes=data.get("status_codes", []),
                    custom_rules=data.get("custom_rules", []),
                )
                logger.info("从 MCP Server 获取到业务知识: %s -> %s", name, entry.knowledge.description)
            else:
                entry.knowledge = BusinessKnowledge(description=entry.display_name)
        except Exception:
            logger.warning("从 MCP Server 获取业务知识失败: %s，使用默认值", name, exc_info=True)
            entry.knowledge = BusinessKnowledge(description=entry.display_name)

        return entry.knowledge

    async def fetch_all_knowledge(self) -> None:
        """获取所有已注册业务的领域知识。"""
        for name in list(self._entries.keys()):
            try:
                await self.fetch_business_knowledge(name)
            except Exception:
                logger.warning("获取业务 '%s' 知识失败，跳过", name, exc_info=True)

    async def fetch_tools_schema(self, name: str) -> list[dict]:
        """从 MCP Server 获取工具列表。"""
        async with self.get_session(name) as session:
            tools_result = await session.list_tools()
            tools = []
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })
            return tools

    async def close_sessions(self) -> None:
        """关闭所有缓存的 session，保留业务注册。"""
        for url in list(self._session_cache.keys()):
            await self._close_cached_session(url)

    async def close_all(self) -> None:
        """关闭所有缓存的 session 并清空注册。"""
        await self.close_sessions()
        self._entries.clear()

    @staticmethod
    def _serialize_tool_result(result) -> str:
        """将 MCP CallToolResult 序列化为字符串。"""
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                texts.append(str(item))
        return "\n".join(texts) if texts else ""
