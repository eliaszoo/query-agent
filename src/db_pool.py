"""数据库连接池管理 - 基于 aiomysql 为每个集群创建独立连接池。

支持两种索引模式：
- 扁平模式（向后兼容）：cluster_name → Pool
- 多业务模式：(business, cluster) → Pool

扁平模式时所有 pool 存入 _pools[cluster] 和 _pools_by_business[("default", cluster)]。
多业务模式时存入 _pools_by_business[(business, cluster)] 和 _pools[cluster]。
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiomysql

from src.config import ClusterConfig

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """为每个集群维护独立的 aiomysql 连接池。

    支持两种初始化方式：
    1. 扁平模式（向后兼容）：传入 dict[str, ClusterConfig]
    2. 多业务模式：传入 dict[str, dict[str, ClusterConfig]]

    各集群连接池独立，一个集群故障不影响其他集群。
    """

    def __init__(
        self,
        cluster_configs: dict[str, ClusterConfig] | None = None,
        business_clusters: dict[str, dict[str, ClusterConfig]] | None = None,
    ) -> None:
        # 两种初始化方式可共存
        self._cluster_configs: dict[str, ClusterConfig] = cluster_configs or {}
        self._business_clusters: dict[str, dict[str, ClusterConfig]] = business_clusters or {}

        # 如果只传了 cluster_configs（扁平模式），自动放入 default 业务
        if cluster_configs and not business_clusters:
            self._business_clusters["default"] = dict(cluster_configs)

        # 如果只传了 business_clusters，构建扁平视图
        if business_clusters and not cluster_configs:
            for biz_clusters in business_clusters.values():
                self._cluster_configs.update(biz_clusters)

        self._pools: dict[str, aiomysql.Pool] = {}  # cluster_name → Pool（扁平视图）
        self._pools_by_business: dict[tuple[str, str], aiomysql.Pool] = {}  # (business, cluster) → Pool

    async def initialize(self) -> None:
        """为所有配置的集群创建连接池。

        单个集群连接失败不影响其他集群的初始化。
        """
        for business, cluster_name, cfg in self._iter_all_configs():
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
                self._pools[cluster_name] = pool
                self._pools_by_business[(business, cluster_name)] = pool
                logger.info("集群 '%s' (业务: %s) 连接池创建成功", cluster_name, business)
            except Exception:
                logger.warning("集群 '%s' (业务: %s) 连接池创建失败，已跳过", cluster_name, business, exc_info=True)

    def _iter_all_configs(self):
        """迭代所有 (business, cluster_name, ClusterConfig) 配置项。"""
        for business, biz_clusters in self._business_clusters.items():
            for cluster_name, cfg in biz_clusters.items():
                yield business, cluster_name, cfg

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
        self._pools_by_business.clear()

    def has_cluster(self, cluster: str) -> bool:
        """检查集群连接池是否已就绪。"""
        return cluster in self._pools

    def cluster_configured(self, cluster: str) -> bool:
        """检查集群是否在配置中（不论连接池是否就绪）。"""
        return cluster in self._cluster_configs

    def has_business_cluster(self, business: str, cluster: str) -> bool:
        """检查指定业务的集群连接池是否已就绪。"""
        return (business, cluster) in self._pools_by_business

    def business_cluster_configured(self, business: str, cluster: str) -> bool:
        """检查指定业务的集群是否在配置中。"""
        return business in self._business_clusters and cluster in self._business_clusters[business]

    def get_business_cluster_list(self, business: str) -> list[str]:
        """获取指定业务的所有已配置集群名称。"""
        biz_clusters = self._business_clusters.get(business, {})
        return list(biz_clusters.keys())

    def get_business_list(self) -> list[str]:
        """获取所有已配置的业务名称。"""
        return list(self._business_clusters.keys())

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

    @asynccontextmanager
    async def get_business_connection(self, business: str, cluster: str) -> AsyncIterator[aiomysql.Connection]:
        """获取指定业务集群的数据库连接。

        Args:
            business: 业务名称。
            cluster: 集群名称。

        Yields:
            aiomysql.Connection 实例。

        Raises:
            ValueError: 业务或集群不存在或连接池未就绪。
        """
        key = (business, cluster)
        if key not in self._pools_by_business:
            available = self.get_business_cluster_list(business)
            raise ValueError(
                f"业务 '{business}' 集群 '{cluster}' 不存在。可用集群: {', '.join(available)}"
            )

        pool = self._pools_by_business[key]
        async with pool.acquire() as conn:
            yield conn