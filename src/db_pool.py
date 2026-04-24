"""数据库连接池管理 - 基于 aiomysql 为每个集群创建独立连接池。"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiomysql

from src.config import ClusterConfig

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """为每个集群维护独立的 aiomysql 连接池。

    各集群连接池独立，一个集群故障不影响其他集群。
    """

    def __init__(self, cluster_configs: dict[str, ClusterConfig]) -> None:
        self._cluster_configs = cluster_configs
        self._pools: dict[str, aiomysql.Pool] = {}

    async def initialize(self) -> None:
        """为所有配置的集群创建连接池。

        单个集群连接失败不影响其他集群的初始化。
        """
        for name, cfg in self._cluster_configs.items():
            try:
                pool = await aiomysql.create_pool(
                    host=cfg.host,
                    port=cfg.port,
                    db=cfg.database,
                    user=cfg.user,
                    password=cfg.password,
                    charset=cfg.charset,
                    maxsize=cfg.max_connections,
                    connect_timeout=cfg.connect_timeout,
                    autocommit=True,
                )
                self._pools[name] = pool
                logger.info("集群 '%s' 连接池创建成功", name)
            except Exception:
                logger.warning("集群 '%s' 连接池创建失败，已跳过", name, exc_info=True)

    async def close(self) -> None:
        """关闭所有连接池。"""
        for name, pool in list(self._pools.items()):
            try:
                pool.close()
                await pool.wait_closed()
                logger.info("集群 '%s' 连接池已关闭", name)
            except Exception:
                logger.exception("关闭集群 '%s' 连接池时出错", name)
        self._pools.clear()

    def has_cluster(self, cluster: str) -> bool:
        """检查集群连接池是否已就绪。"""
        return cluster in self._pools

    def cluster_configured(self, cluster: str) -> bool:
        """检查集群是否在配置中（不论连接池是否就绪）。"""
        return cluster in self._cluster_configs

    def get_pool_status(self, cluster: str) -> str:
        """获取集群连接状态。"""
        if cluster in self._pools:
            return "connected"
        if cluster in self._cluster_configs:
            return "disconnected"
        return "unknown"

    @asynccontextmanager
    async def get_connection(self, cluster: str) -> AsyncIterator[aiomysql.Connection]:
        """获取指定集群的数据库连接（异步上下文管理器）。

        Args:
            cluster: 集群名称。

        Yields:
            aiomysql.Connection 实例。

        Raises:
            ValueError: 集群不存在或连接池未就绪。
        """
        if cluster not in self._cluster_configs:
            available = sorted(self._cluster_configs.keys())
            raise ValueError(
                f"无效的集群名称: '{cluster}'。可用集群: {', '.join(available)}"
            )

        if cluster not in self._pools:
            raise ValueError(
                f"集群 '{cluster}' 的连接池不可用（可能初始化失败）"
            )

        pool = self._pools[cluster]
        async with pool.acquire() as conn:
            yield conn
