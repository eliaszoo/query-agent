"""多业务 MCP Server 连接管理器。

管理多个业务的 MCP Server SSE 连接，支持：
- 懒连接：注册时不立即连接，首次使用时建立
- 会话缓存：已建立的会话复用，避免每次查询重连
- 业务知识获取：首次连接后自动调用 get_business_knowledge
- 动态增减：运行时添加/移除业务
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.config import BusinessKnowledge

logger = logging.getLogger(__name__)


@dataclass
class BusinessEntry:
    """一个业务的注册信息。"""

    name: str  # 业务标识，如 "digitalhuman"
    display_name: str  # 显示名，如 "数字人"
    mcp_server_url: str  # SSE URL
    api_key: str = ""  # MCP Server 鉴权密钥
    knowledge: Optional[BusinessKnowledge] = None  # 从 MCP server 获取的业务知识
    _connected: bool = field(default=False, repr=False, init=False)


class BusinessRegistry:
    """管理多个业务的 MCP Server 连接。

    使用方式：
        registry = BusinessRegistry()
        await registry.register("digitalhuman", "http://host:8765/sse", "数字人")
        result = await registry.call_tool("digitalhuman", "execute_readonly_sql", {...})
        await registry.close_all()
    """

    def __init__(self) -> None:
        self._entries: dict[str, BusinessEntry] = {}
        self._session_cache: dict[str, tuple[ClientSession, Any]] = {}  # name → (session, ctx)

    def register(self, name: str, mcp_server_url: str, display_name: str = "", api_key: str = "") -> None:
        """注册一个业务（不立即连接）。

        Args:
            name: 业务标识。
            mcp_server_url: MCP Server 的 SSE URL。
            display_name: 显示名称。
            api_key: MCP Server 鉴权密钥。
        """
        self._entries[name] = BusinessEntry(
            name=name,
            display_name=display_name or name,
            mcp_server_url=mcp_server_url,
            api_key=api_key,
        )
        logger.info("已注册业务: %s (%s) -> %s", name, display_name, mcp_server_url)

    async def remove(self, name: str) -> None:
        """移除一个业务（断开连接）。

        Args:
            name: 业务标识。

        Raises:
            KeyError: 业务不存在。
        """
        if name not in self._entries:
            raise KeyError(f"业务 '{name}' 不存在")
        await self._close_cached_session(name)
        del self._entries[name]
        logger.info("已移除业务: %s", name)

    def list_businesses(self) -> list[BusinessEntry]:
        """列出所有已注册的业务。"""
        return list(self._entries.values())

    def has_business(self, name: str) -> bool:
        """检查业务是否已注册。"""
        return name in self._entries

    def get_entry(self, name: str) -> BusinessEntry:
        """获取业务条目。

        Raises:
            KeyError: 业务不存在。
        """
        if name not in self._entries:
            raise KeyError(f"业务 '{name}' 不存在")
        return self._entries[name]

    @asynccontextmanager
    async def get_session(self, name: str) -> AsyncIterator[ClientSession]:
        """获取业务的 MCP 会话（懒连接，自动建立和关闭）。

        每次调用都会建立新的 SSE 连接，用完自动关闭。
        MCP SSE 协议不支持连接复用，因此采用用完即关的策略。

        Args:
            name: 业务标识。

        Yields:
            已初始化的 ClientSession。
        """
        entry = self.get_entry(name)
        headers = {}
        if entry.api_key:
            headers["Authorization"] = f"Bearer {entry.api_key}"
        async with sse_client(entry.mcp_server_url, headers=headers or None) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                entry._connected = True
                yield session

    @asynccontextmanager
    async def get_cached_session(self, name: str) -> AsyncIterator[ClientSession]:
        """获取缓存的 session，同一业务复用连接，异常时自动清除缓存并重连。"""
        if name in self._session_cache:
            session, _ = self._session_cache[name]
            try:
                yield session
                return
            except Exception:
                # 连接异常时清除缓存，下面会重建
                await self._close_cached_session(name)

        entry = self.get_entry(name)
        headers = {}
        if entry.api_key:
            headers["Authorization"] = f"Bearer {entry.api_key}"

        ctx = sse_client(entry.mcp_server_url, headers=headers or None)
        read_stream, write_stream = await ctx.__aenter__()
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()
        entry._connected = True

        self._session_cache[name] = (session, ctx)
        try:
            yield session
        except Exception:
            await self._close_cached_session(name)
            raise

    async def _close_cached_session(self, name: str) -> None:
        """关闭并清除指定业务的缓存 session。"""
        if name not in self._session_cache:
            return
        session, ctx = self._session_cache.pop(name)
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            logger.warning("关闭 session 失败: %s", name, exc_info=True)
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("关闭 SSE context 失败: %s", name, exc_info=True)

    async def call_tool(self, name: str, tool_name: str, arguments: dict) -> str:
        """在指定业务上调用 MCP 工具（复用缓存的 SSE 连接）。

        Args:
            name: 业务标识。
            tool_name: 工具名称。
            arguments: 工具参数。

        Returns:
            工具结果的 JSON 字符串。
        """
        async with self.get_cached_session(name) as session:
            result = await session.call_tool(tool_name, arguments)
            return self._serialize_tool_result(result)

    async def fetch_business_knowledge(self, name: str) -> BusinessKnowledge:
        """从 MCP Server 获取业务领域知识并缓存。

        Args:
            name: 业务标识。

        Returns:
            BusinessKnowledge 实例。
        """
        entry = self.get_entry(name)

        # 已缓存则直接返回
        if entry.knowledge is not None:
            return entry.knowledge

        try:
            result_text = await self.call_tool(name, "get_business_knowledge", {})
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
        """从 MCP Server 获取工具列表（复用缓存的 SSE 连接）。

        Args:
            name: 业务标识。

        Returns:
            工具定义列表。
        """
        async with self.get_cached_session(name) as session:
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
        for name in list(self._session_cache.keys()):
            await self._close_cached_session(name)

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
