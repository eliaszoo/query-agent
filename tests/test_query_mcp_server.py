"""MCP Server 鉴权中间件的单元测试。"""

import pytest

from src.query_mcp_server import BearerAuthMiddleware


class TestBearerAuthMiddleware:
    """测试 BearerAuthMiddleware 中间件。"""

    @staticmethod
    def _make_scope(headers: dict[str, str] | None = None, scope_type: str = "http") -> dict:
        """构造 ASGI scope 对象。"""
        raw_headers = []
        if headers:
            for k, v in headers.items():
                raw_headers.append((k.encode(), v.encode()))
        return {
            "type": scope_type,
            "headers": raw_headers,
            "method": "GET",
            "path": "/sse",
        }

    @pytest.mark.asyncio
    async def test_valid_bearer_token_passes_through(self):
        """有效的 Bearer token 请求通过鉴权。"""
        api_key = "my-secret-key"
        received_scope = None

        async def mock_app(scope, receive, send):
            nonlocal received_scope
            received_scope = scope

        middleware = BearerAuthMiddleware(app=mock_app, api_key=api_key)

        scope = self._make_scope(headers={"authorization": f"Bearer {api_key}"})
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        assert received_scope is not None

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self):
        """缺少 Authorization header 返回 401。"""
        async def mock_app(scope, receive, send):
            pass  # Should not be called

        middleware = BearerAuthMiddleware(app=mock_app, api_key="secret")

        scope = self._make_scope()
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        status = send_messages[0]["status"] if send_messages else None
        assert status == 401

    @pytest.mark.asyncio
    async def test_wrong_token_returns_403(self):
        """错误的 token 返回 403。"""
        async def mock_app(scope, receive, send):
            pass  # Should not be called

        middleware = BearerAuthMiddleware(app=mock_app, api_key="correct-key")

        scope = self._make_scope(headers={"authorization": "Bearer wrong-key"})
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        status = send_messages[0]["status"] if send_messages else None
        assert status == 403

    @pytest.mark.asyncio
    async def test_non_bearer_scheme_returns_401(self):
        """非 Bearer scheme 返回 401。"""
        async def mock_app(scope, receive, send):
            pass

        middleware = BearerAuthMiddleware(app=mock_app, api_key="secret")

        scope = self._make_scope(headers={"authorization": "Basic dXNlcjpwYXNz"})
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        status = send_messages[0]["status"] if send_messages else None
        assert status == 401

    @pytest.mark.asyncio
    async def test_websocket_requires_auth(self):
        """WebSocket scope 也需要鉴权，无 Bearer token 时关闭连接。"""
        received_scope = None

        async def mock_app(scope, receive, send):
            nonlocal received_scope
            received_scope = scope

        middleware = BearerAuthMiddleware(app=mock_app, api_key="secret")

        scope = self._make_scope(scope_type="websocket")
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        # 未提供 Bearer token，应关闭 websocket 连接
        assert received_scope is None
        assert len(send_messages) == 1
        assert send_messages[0]["type"] == "websocket.close"
        assert send_messages[0]["code"] == 4001

    @pytest.mark.asyncio
    async def test_websocket_valid_token_passes(self):
        """WebSocket scope 提供有效 Bearer token 时通过。"""
        received_scope = None

        async def mock_app(scope, receive, send):
            nonlocal received_scope
            received_scope = scope

        middleware = BearerAuthMiddleware(app=mock_app, api_key="secret")

        scope = self._make_scope(scope_type="websocket")
        scope["headers"] = [(b"authorization", b"Bearer secret")]
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)
        assert received_scope is not None

    @pytest.mark.asyncio
    async def test_401_includes_www_authenticate_header(self):
        """401 响应包含 WWW-Authenticate header。"""
        async def mock_app(scope, receive, send):
            pass

        middleware = BearerAuthMiddleware(app=mock_app, api_key="secret")

        scope = self._make_scope()
        send_messages = []

        async def send(msg):
            send_messages.append(msg)

        async def receive():
            return {}

        await middleware(scope, receive, send)

        # Find the response start message with headers
        for msg in send_messages:
            if msg.get("type") == "http.response.start":
                headers = dict(msg.get("headers", []))
                assert b"www-authenticate" in headers
                assert headers[b"www-authenticate"] == b"Bearer"
