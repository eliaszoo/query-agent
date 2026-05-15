"""飞书会话管理 — 每用户独立 QueryAgent 实例，支持超时清理。"""

import asyncio
import logging
import time
from dataclasses import dataclass

from src.agent import QueryAgent

logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    """单个用户的会话条目。"""

    agent: QueryAgent
    last_active: float  # 上次活动时间戳
    last_query: str = ""  # 最近一次查询文本
    last_response: str = ""  # 最近一次查询回复


class FeishuSessionManager:
    """飞书用户会话管理器。

    每个飞书用户（open_id）维护一个独立的 QueryAgent 实例，
    QueryAgent 内含 ConversationState，天然支持多轮对话。

    会话超时自动清理：超过 session_ttl 未活动的会话，删除 agent 实例。
    使用 asyncio.Lock 防止同一用户并发查询。
    """

    def __init__(self, config_path: str, session_ttl: int = 1800):
        self._sessions: dict[str, SessionEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._config_path = config_path
        self._session_ttl = session_ttl

    def get_agent(self, user_id: str) -> QueryAgent:
        """获取或创建用户的 QueryAgent 会话。

        Args:
            user_id: 飞书用户 open_id。

        Returns:
            该用户的 QueryAgent 实例。
        """
        now = time.time()

        if user_id in self._sessions:
            entry = self._sessions[user_id]
            entry.last_active = now
            return entry.agent

        # 创建新会话
        async def _confirm(**_):
            return True

        agent = QueryAgent(config_path=self._config_path, confirm_callback=_confirm)
        self._sessions[user_id] = SessionEntry(agent=agent, last_active=now)
        logger.info("为用户 %s 创建新会话，当前总会话数: %d", user_id, len(self._sessions))
        return agent

    def get_session(self, user_id: str) -> SessionEntry | None:
        """获取用户的 SessionEntry（含 last_query/last_response）。"""
        return self._sessions.get(user_id)

    def update_last_query(self, user_id: str, query: str, response: str) -> None:
        """更新用户最近一次查询和回复。"""
        entry = self._sessions.get(user_id)
        if entry:
            entry.last_query = query
            entry.last_response = response

    def get_lock(self, user_id: str) -> asyncio.Lock:
        """获取用户的并发锁，防止同一用户同时发起多个查询。

        Args:
            user_id: 飞书用户 open_id。

        Returns:
            该用户的 asyncio.Lock 实例。
        """
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def cleanup_expired(self) -> int:
        """清理超时会话。

        Returns:
            清理的会话数量。
        """
        now = time.time()
        expired_users = [
            user_id
            for user_id, entry in self._sessions.items()
            if now - entry.last_active > self._session_ttl
        ]

        for user_id in expired_users:
            del self._sessions[user_id]
            self._locks.pop(user_id, None)

        if expired_users:
            logger.info("清理 %d 个超时会话，剩余: %d", len(expired_users), len(self._sessions))

        return len(expired_users)

    @property
    def active_session_count(self) -> int:
        """当前活跃会话数。"""
        return len(self._sessions)
