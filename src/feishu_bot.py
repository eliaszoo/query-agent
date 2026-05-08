"""飞书机器人服务 — FastAPI app、事件回调处理、消息收发。

通过飞书开放平台事件回调接收群聊中 @机器人 的消息，
调用 QueryAgent 处理查询，将结果以卡片形式回复到群聊。
"""

import asyncio
import json
import logging
import re

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from fastapi import FastAPI, Request, Response

from src.config import load_config
from src.feishu_session import FeishuSessionManager
from src.feishu_message import build_query_card, build_error_card

logger = logging.getLogger(__name__)


def _extract_text_from_event(event_data: dict) -> str:
    """从飞书消息事件中提取纯文本内容，去掉 @机器人 mention。

    Args:
        event_data: 事件消息体中的 message 字段。

    Returns:
        去掉 @mention 后的纯文本。
    """
    content_str = event_data.get("content", "{}")
    try:
        content = json.loads(content_str) if isinstance(content_str, str) else content_str
    except json.JSONDecodeError:
        return ""

    # 文本消息
    text = content.get("text", "")

    # 去掉 @user_id 的 mention（格式：@_user_1 ）
    # 飞书文本消息中 @机器人 会产生 @_user_1 这样的 mention
    text = re.sub(r"@_user_\d+\s*", "", text).strip()

    return text


class FeishuBotService:
    """飞书机器人服务。

    管理飞书事件回调、用户会话、消息处理和回复。
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config = load_config(config_path)
        feishu_cfg = self._config.feishu

        if not feishu_cfg.app_id or not feishu_cfg.app_secret:
            raise ValueError("飞书配置缺少 app_id 或 app_secret")

        # lark-oapi 客户端
        self._client = lark.Client.builder() \
            .app_id(feishu_cfg.app_id) \
            .app_secret(feishu_cfg.app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

        # 会话管理器
        self._session_manager = FeishuSessionManager(
            config_path=config_path,
            session_ttl=feishu_cfg.session_ttl,
        )

        # 事件处理器
        self._event_handler = lark.EventDispatcherHandler.builder(
            feishu_cfg.verification_token,
            feishu_cfg.encrypt_key,
        ).register_p2_im_message_receive_v1(self._on_message_receive).build()

        # FastAPI app
        self._app = FastAPI(title="query-agent-feishu-bot")
        self._setup_routes()

        # 定时清理任务
        self._cleanup_task: asyncio.Task | None = None

    @property
    def app(self) -> FastAPI:
        """返回 FastAPI 应用实例。"""
        return self._app

    def _setup_routes(self) -> None:
        """注册 FastAPI 路由。"""

        @self._app.post("/webhook/event")
        async def handle_event(request: Request):
            """处理飞书事件回调。"""
            headers = dict(request.headers)
            body = await request.body()

            try:
                self._event_handler.handle(bytes(body), headers)
            except Exception as e:
                logger.error("处理飞书事件失败: %s", e, exc_info=True)

            # 飞书要求 3 秒内返回 HTTP 200
            return Response(status_code=200)

        @self._app.get("/health")
        async def health_check():
            """健康检查端点。"""
            return {
                "status": "ok",
                "active_sessions": self._session_manager.active_session_count,
            }

    def _on_message_receive(self, ctx: lark.Context, event: lark.im.v1.P2ImMessageReceiveV1) -> None:
        """处理 im.message.receive_v1 事件。

        从事件中提取用户消息，异步调用 QueryAgent 处理。
        """
        try:
            event_data = event.event.message
            sender = event.event.sender

            user_id = sender.sender_id.open_id if sender and sender.sender_id else ""
            chat_id = event_data.chat_id if event_data else ""
            message_type = event_data.message_type if event_data else ""

            # 只处理文本消息
            if message_type != "text":
                logger.debug("忽略非文本消息: type=%s", message_type)
                return

            # 提取消息文本（去掉 @mention）
            message_dict = {}
            if event_data:
                message_dict = {
                    "content": event_data.content or "{}",
                }
            text = _extract_text_from_event(message_dict)

            if not text:
                logger.debug("消息文本为空，忽略")
                return

            logger.info("收到飞书消息: user=%s chat=%s text=%s", user_id, chat_id, text[:100])

            # 异步处理查询，不阻塞事件回调
            asyncio.create_task(self._process_query(user_id, chat_id, text))

        except Exception as e:
            logger.error("处理消息事件失败: %s", e, exc_info=True)

    async def _process_query(self, user_id: str, chat_id: str, text: str) -> None:
        """异步处理用户查询并发送回复。

        Args:
            user_id: 飞书用户 open_id。
            chat_id: 群聊 chat_id。
            text: 用户查询文本。
        """
        lock = self._session_manager.get_lock(user_id)
        async with lock:
            try:
                agent = self._session_manager.get_agent(user_id)
                result = await agent.run_query(text)
                metrics = agent.last_metrics

                # 构建卡片消息
                card_content = build_query_card(result, metrics)
                await self._send_card_message(chat_id, card_content)

            except Exception as e:
                logger.error("查询处理失败: user=%s error=%s", user_id, e, exc_info=True)
                error_card = build_error_card(f"查询处理失败: {e}")
                await self._send_card_message(chat_id, error_card)

    async def _send_card_message(self, chat_id: str, card_content: str) -> None:
        """发送卡片消息到飞书群聊。

        Args:
            chat_id: 群聊 chat_id。
            card_content: 卡片 JSON 字符串。
        """
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(card_content)
                    .build()) \
                .build()

            response = self._client.im.v1.message.create(request)

            if not response.success():
                logger.error(
                    "发送飞书消息失败: code=%s msg=%s",
                    response.code,
                    response.msg,
                )
        except Exception as e:
            logger.error("发送飞书消息异常: %s", e, exc_info=True)

    async def start_cleanup_loop(self) -> None:
        """启动会话超时清理定时任务。"""
        while True:
            await asyncio.sleep(60)  # 每 60 秒清理一次
            self._session_manager.cleanup_expired()

    def run(self, host: str | None = None, port: int | None = None) -> None:
        """启动飞书机器人服务。

        Args:
            host: 监听地址，默认使用配置值。
            port: 监听端口，默认使用配置值。
        """
        import uvicorn

        feishu_cfg = self._config.feishu
        host = host or feishu_cfg.host
        port = port or feishu_cfg.port

        logger.info("启动飞书机器人服务: %s:%d", host, port)

        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        # 在同一个事件循环中启动清理任务
        async def _run_with_cleanup():
            cleanup_task = asyncio.create_task(self.start_cleanup_loop())
            try:
                await server.serve()
            finally:
                cleanup_task.cancel()

        import anyio
        anyio.run(_run_with_cleanup)
