"""ConnectionPoolManager 单元测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.config import ClusterConfig
from src.db_pool import ConnectionPoolManager


def _make_cluster(name: str = "test", host: str = "127.0.0.1") -> ClusterConfig:
    return ClusterConfig(
        name=name,
        description=f"{name}环境",
        host=host,
        port=3306,
        database=f"db_{name}",
        user="readonly",
        password="secret",
        charset="utf8mb4",
        max_connections=5,
        connect_timeout=10,
    )


class TestConnectionPoolManagerInit:
    """测试 ConnectionPoolManager 初始化。"""

    def test_accepts_cluster_configs(self):
        clusters = {"test": _make_cluster("test")}
        mgr = ConnectionPoolManager(clusters)
        assert mgr._cluster_configs == clusters
        assert mgr._pools == {}


class TestInitialize:
    """测试连接池初始化。"""

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_creates_pool_for_each_cluster(self, mock_create_pool):
        mock_pool = MagicMock()
        mock_create_pool.return_value = mock_pool

        clusters = {
            "test": _make_cluster("test"),
            "prod": _make_cluster("prod", host="prod-host"),
        }
        mgr = ConnectionPoolManager(clusters)
        await mgr.initialize()

        assert mock_create_pool.call_count == 2
        assert "test" in mgr._pools
        assert "prod" in mgr._pools

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_passes_correct_params(self, mock_create_pool):
        mock_create_pool.return_value = MagicMock()
        cfg = _make_cluster("test")
        mgr = ConnectionPoolManager({"test": cfg})
        await mgr.initialize()

        mock_create_pool.assert_called_once_with(
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

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_one_cluster_failure_does_not_block_others(self, mock_create_pool):
        """一个集群连接失败不影响其他集群。"""
        good_pool = MagicMock()

        async def side_effect(**kwargs):
            if kwargs["host"] == "bad-host":
                raise ConnectionRefusedError("Connection refused")
            return good_pool

        mock_create_pool.side_effect = side_effect

        clusters = {
            "good": _make_cluster("good", host="good-host"),
            "bad": _make_cluster("bad", host="bad-host"),
        }
        mgr = ConnectionPoolManager(clusters)
        await mgr.initialize()

        assert "good" in mgr._pools
        assert "bad" not in mgr._pools


class TestGetConnection:
    """测试 get_connection 上下文管理器。"""

    @pytest.mark.asyncio
    async def test_invalid_cluster_raises_valueerror(self):
        mgr = ConnectionPoolManager({"test": _make_cluster("test")})
        with pytest.raises(ValueError, match="无效的集群名称: 'nonexistent'"):
            async with mgr.get_connection("nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_unavailable_pool_raises_valueerror(self):
        """集群配置存在但连接池未创建（初始化失败）时报错。"""
        mgr = ConnectionPoolManager({"test": _make_cluster("test")})
        # 不调用 initialize()，所以 _pools 为空
        with pytest.raises(ValueError, match="连接池不可用"):
            async with mgr.get_connection("test"):
                pass

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_yields_connection_from_pool(self, mock_create_pool):
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        # pool.acquire() 返回一个异步上下文管理器
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_create_pool.return_value = mock_pool

        mgr = ConnectionPoolManager({"test": _make_cluster("test")})
        await mgr.initialize()

        async with mgr.get_connection("test") as conn:
            assert conn is mock_conn

    @pytest.mark.asyncio
    async def test_error_message_lists_available_clusters(self):
        clusters = {
            "test": _make_cluster("test"),
            "prod": _make_cluster("prod"),
        }
        mgr = ConnectionPoolManager(clusters)
        with pytest.raises(ValueError, match="可用集群") as exc_info:
            async with mgr.get_connection("staging"):
                pass
        # 确保错误消息包含可用集群名
        assert "test" in str(exc_info.value)
        assert "prod" in str(exc_info.value)


class TestClose:
    """测试连接池关闭。"""

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_closes_all_pools(self, mock_create_pool):
        mock_pool = MagicMock()
        mock_pool.wait_closed = AsyncMock()
        mock_create_pool.return_value = mock_pool

        clusters = {"a": _make_cluster("a"), "b": _make_cluster("b")}
        mgr = ConnectionPoolManager(clusters)
        await mgr.initialize()
        await mgr.close()

        assert mock_pool.close.call_count == 2
        assert mock_pool.wait_closed.call_count == 2
        assert mgr._pools == {}


class TestGetPoolStatus:
    """测试连接池状态查询。"""

    @pytest.mark.asyncio
    @patch("src.db_pool.aiomysql.create_pool", new_callable=AsyncMock)
    async def test_connected_status(self, mock_create_pool):
        mock_create_pool.return_value = MagicMock()
        mgr = ConnectionPoolManager({"test": _make_cluster("test")})
        await mgr.initialize()
        assert mgr.get_pool_status("test") == "connected"

    def test_disconnected_status(self):
        mgr = ConnectionPoolManager({"test": _make_cluster("test")})
        assert mgr.get_pool_status("test") == "disconnected"

    def test_unknown_cluster_returns_unknown(self):
        mgr = ConnectionPoolManager({})
        assert mgr.get_pool_status("unknown") == "unknown"
