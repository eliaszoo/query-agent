"""飞书机器人服务 — FastAPI app、事件回调处理、消息收发。

通过飞书开放平台事件回调接收群聊中 @机器人 的消息，
调用 QueryAgent 处理查询，将结果以卡片形式回复到群聊。
支持话题回复、SQL 预览、高风险确认卡片。
"""

import asyncio
import json
import logging
import re
from uuid import uuid4

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from fastapi import FastAPI, Request, Response

from src.config import load_config
from src.feishu_session import FeishuSessionManager
from src.feishu_message import (
    build_query_card, build_error_card, build_command_card,
    build_sql_preview_card, build_risk_confirm_card,
)

# 飞书签名相关 header，encrypt_key 未配置时需移除以绕过 SDK 签名校验
_LARK_SIGN_HEADERS = {"x-lark-signature", "x-lark-request-timestamp", "x-lark-request-nonce"}

logger = logging.getLogger(__name__)


def _strip_lark_sign_headers(headers) -> dict:
    """移除飞书签名相关 header，用于 encrypt_key 未配置时绕过 SDK 签名校验。"""
    if hasattr(headers, "items"):
        return {k: v for k, v in headers.items() if k.lower() not in _LARK_SIGN_HEADERS}
    return headers


def _extract_text_from_event(event_data: dict) -> str:
    """从飞书消息事件中提取纯文本内容，去掉 @机器人 mention。"""
    content_str = event_data.get("content", "{}")
    try:
        content = json.loads(content_str) if isinstance(content_str, str) else content_str
    except json.JSONDecodeError:
        return ""

    text = content.get("text", "")
    text = re.sub(r"@_user_\d+\s*", "", text).strip()
    return text


class FeishuBotService:
    """飞书机器人服务。

    管理飞书事件回调、用户会话、消息处理和回复。
    支持话题回复、SQL 预览卡片、高风险确认按钮。
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

        # encrypt_key 是否已配置
        self._encrypt_key_configured = bool(feishu_cfg.encrypt_key)

        # 高风险 SQL 确认：confirm_id → Future
        self._pending_confirms: dict[str, asyncio.Future] = {}

        # 事件处理器（仅用于消息事件）
        self._event_handler = lark.EventDispatcherHandler.builder(
            feishu_cfg.encrypt_key or "",
            feishu_cfg.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_receive
        ).build()

        # 卡片交互处理器（独立路由）
        self._card_handler = lark.CardActionHandler.builder(
            feishu_cfg.encrypt_key or "",
            feishu_cfg.verification_token or "",
        ).register(self._on_card_action).build()

        # FastAPI app
        self._app = FastAPI(title="query-agent-feishu-bot")
        self._setup_routes()

        # 定时清理任务
        self._cleanup_task: asyncio.Task | None = None

    @property
    def app(self) -> FastAPI:
        return self._app

    def _setup_routes(self) -> None:
        @self._app.post("/webhook/event")
        async def handle_event(request: Request):
            headers = request.headers
            body = await request.body()

            try:
                raw_req = lark.RawRequest()
                raw_req.uri = str(request.url)
                raw_req.headers = headers
                raw_req.body = bytes(body)

                if not self._encrypt_key_configured:
                    raw_req.headers = _strip_lark_sign_headers(raw_req.headers)

                raw_resp = self._event_handler.do(raw_req)
                return Response(
                    content=raw_resp.content,
                    status_code=raw_resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                logger.error("处理飞书事件失败: %s", e, exc_info=True)
                return Response(status_code=500)

        @self._app.get("/health")
        async def health_check():
            return {
                "status": "ok",
                "active_sessions": self._session_manager.active_session_count,
                "pending_confirms": len(self._pending_confirms),
            }

        @self._app.post("/webhook/card")
        async def handle_card_action(request: Request):
            headers = request.headers
            body = await request.body()

            try:
                raw_req = lark.RawRequest()
                raw_req.uri = str(request.url)
                raw_req.body = bytes(body)
                if not self._encrypt_key_configured:
                    raw_req.headers = _strip_lark_sign_headers(headers)
                else:
                    raw_req.headers = headers

                raw_resp = self._card_handler.do(raw_req)
                return Response(
                    content=raw_resp.content,
                    status_code=raw_resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                logger.error("处理卡片交互失败: %s", e, exc_info=True)
                return Response(status_code=500)

    # ── 事件回调 ──

    def _on_message_receive(self, event: lark.im.v1.P2ImMessageReceiveV1) -> None:
        """处理 im.message.receive_v1 事件。"""
        try:
            event_data = event.event.message
            sender = event.event.sender

            user_id = sender.sender_id.open_id if sender and sender.sender_id else ""
            chat_id = event_data.chat_id if event_data else ""
            message_type = event_data.message_type if event_data else ""
            message_id = event_data.message_id if event_data else ""

            if message_type != "text":
                logger.debug("忽略非文本消息: type=%s", message_type)
                return

            message_dict = {}
            if event_data:
                message_dict = {"content": event_data.content or "{}"}
            text = _extract_text_from_event(message_dict)

            if not text:
                logger.debug("消息文本为空，忽略")
                return

            logger.info("收到飞书消息: user=%s chat=%s msg_id=%s text=%s",
                        user_id, chat_id, message_id, text[:100])

            task = asyncio.create_task(
                self._process_query(user_id, chat_id, text, message_id)
            )
            task.add_done_callback(self._on_task_done)

        except Exception as e:
            logger.error("处理消息事件失败: %s", e, exc_info=True)

    def _on_card_action(self, card: lark.Card) -> None:
        """处理卡片交互按钮回调（高风险确认）。"""
        try:
            action = card.action
            value = (action.value or {}) if action else {}
            confirm_id = value.get("confirm_id", "")
            approved = value.get("approved", False)

            if not confirm_id:
                return

            future = self._pending_confirms.pop(confirm_id, None)
            if future and not future.done():
                future.set_result(approved)
                logger.info("卡片按钮回调: confirm_id=%s approved=%s", confirm_id, approved)
            else:
                logger.warning("未找到或已完成的确认: confirm_id=%s", confirm_id)

        except Exception as e:
            logger.error("处理卡片交互失败: %s", e, exc_info=True)

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            logger.warning("查询任务被取消")
            return
        exc = task.exception()
        if exc:
            logger.error("查询任务未捕获异常: %s", exc, exc_info=exc)

    # ── 查询处理 ──

    async def _add_reaction(self, message_id: str, emoji: str = "👀") -> None:
        """给消息添加表情反馈。"""
        if not message_id:
            return
        try:
            request = CreateMessageReactionRequestBuilder() \
                .message_id(message_id) \
                .request_body(CreateMessageReactionRequestBodyBuilder() \
                    .reaction_type(EmojiBuilder().emoji_type(emoji).build()) \
                    .build()) \
                .build()
            response = self._client.im.v1.message_reaction.create(request)
            if not response.success():
                logger.warning("添加表情失败: code=%s msg=%s", response.code, response.msg)
        except Exception as e:
            logger.debug("添加表情异常: %s", e)

    async def _process_query(
        self, user_id: str, chat_id: str, text: str, message_id: str = ""
    ) -> None:
        """异步处理用户查询并发送回复。"""
        lock = self._session_manager.get_lock(user_id)
        async with lock:
            try:
                # 即时反馈：给用户消息加表情
                await self._add_reaction(message_id, "👀")

                # Slash 命令
                if text.startswith("/"):
                    card_content = await self._handle_slash_command(text, user_id)
                    if card_content:
                        await self._send_card_message(chat_id, card_content, message_id=message_id)
                        return

                agent = self._session_manager.get_agent(user_id)

                # 注入飞书确认回调（替换默认的 input 确认）
                original_confirm = agent._confirm_callback
                agent._confirm_callback = self._make_confirm_callback(
                    agent, user_id, chat_id, message_id
                )
                # 同步到 ToolExecutionService
                agent._tool_execution._confirm_callback = agent._confirm_callback

                try:
                    result = await agent.run_query(
                        text,
                        on_sql_preview=self._make_sql_preview_callback(chat_id, message_id),
                    )
                finally:
                    # 恢复原始确认回调
                    agent._confirm_callback = original_confirm
                    agent._tool_execution._confirm_callback = original_confirm

                metrics = agent.last_metrics
                card_content = build_query_card(result, metrics)
                await self._send_card_message(chat_id, card_content, message_id=message_id)

            except Exception as e:
                logger.error("查询处理失败: user=%s error=%s", user_id, e, exc_info=True)
                error_card = build_error_card(f"查询处理失败: {e}")
                await self._send_card_message(chat_id, error_card, message_id=message_id)

    def _make_sql_preview_callback(self, chat_id: str, message_id: str):
        """创建 SQL 预览回调，发送预览卡片到话题。"""
        async def _on_sql_preview(sql, cluster, risk_level, risk_reasons):
            card = build_sql_preview_card(sql, cluster, risk_level, risk_reasons)
            await self._send_card_message(chat_id, card, message_id=message_id)
        return _on_sql_preview

    def _make_confirm_callback(self, agent, user_id: str, chat_id: str, message_id: str):
        """创建飞书模式的高风险确认回调。

        发送带按钮的确认卡片，用 asyncio.Future 等待用户点击。
        无风险时直接通过（返回 True）。
        """
        async def _feishu_confirm(**kwargs):
            risk_level = kwargs.get("risk_level", "")
            risk_reasons = kwargs.get("risk_reasons", [])
            sql = kwargs.get("sql", "")
            cluster = kwargs.get("cluster", "")

            if not risk_reasons:
                return True

            # 创建 Future 等待用户点击
            confirm_id = f"{user_id}:{uuid4().hex[:8]}"
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._pending_confirms[confirm_id] = future

            # 发送确认卡片
            card = build_risk_confirm_card(
                sql, cluster, risk_level, risk_reasons, confirm_id
            )
            await self._send_card_message(chat_id, card, message_id=message_id)

            # 等待用户点击（5 分钟超时）
            try:
                result = await asyncio.wait_for(future, timeout=300)
                return bool(result)
            except asyncio.TimeoutError:
                self._pending_confirms.pop(confirm_id, None)
                logger.warning("高风险确认超时: confirm_id=%s", confirm_id)
                return False

        return _feishu_confirm

    # ── 消息发送 ──

    async def _send_card_message(
        self, chat_id: str, card_content: str, message_id: str = ""
    ) -> None:
        """发送卡片消息。有 message_id 时回复到话题中。"""
        try:
            if message_id:
                # 话题回复
                request = ReplyMessageRequest.builder() \
                    .message_id(message_id) \
                    .request_body(ReplyMessageRequestBody.builder()
                        .msg_type("interactive")
                        .content(card_content)
                        .reply_in_thread(True)
                        .build()) \
                    .build()
                response = self._client.im.v1.message.reply(request)
            else:
                # 新消息
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
                logger.error("发送飞书消息失败: code=%s msg=%s", response.code, response.msg)
        except Exception as e:
            logger.error("发送飞书消息异常: %s", e, exc_info=True)

    # ── Slash 命令 ──

    async def _handle_slash_command(self, text: str, user_id: str) -> str | None:
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]
        agent = self._session_manager.get_agent(user_id)

        if cmd == "/list":
            return await self._cmd_list(agent)
        if cmd == "/memory":
            return self._cmd_memory(agent)
        if cmd == "/clear":
            return self._cmd_clear(agent, args)
        if cmd == "/new":
            agent.clear_history()
            return build_command_card("新会话", "已开始新的对话。")
        if cmd == "/business":
            return self._cmd_business(agent, args)
        if cmd == "/fields":
            return self._cmd_fields(agent)
        if cmd == "/field":
            return self._cmd_field(agent, args)
        if cmd == "/field_rm":
            return self._cmd_field_rm(agent, args)
        if cmd == "/rules":
            return self._cmd_rules(agent)
        if cmd == "/rules_clear":
            return self._cmd_rules_clear(agent, args)
        if cmd == "/remember":
            return self._cmd_remember(agent, args, user_id)
        if cmd == "/prompt":
            prompt = agent._last_system_prompt or agent._build_system_prompt()
            return build_command_card("System Prompt", f"```\n{prompt}\n```")
        if cmd == "/add":
            return await self._cmd_add(agent, args)
        if cmd == "/remove":
            return await self._cmd_remove(agent, args)

        return None

    async def _cmd_list(self, agent) -> str:
        if not agent.list_businesses() and agent.mcp_servers:
            try:
                await agent._ensure_knowledge_loaded()
            except Exception:
                pass

        lines = []
        servers = agent.mcp_servers
        if servers:
            lines.append("**MCP Servers:**")
            for s in servers:
                lines.append(f"  - {s.name} ({s.url})")

        businesses = agent.list_businesses()
        if businesses:
            lines.append("\n**Businesses:**")
            for b in businesses:
                status = "loaded" if b.knowledge else "pending"
                cluster_labels = [
                    f"{c} ({b.cluster_descriptions[c]})" if b.cluster_descriptions.get(c) else c
                    for c in b.cluster_routing.keys()
                ] if b.cluster_routing else ["pending"]
                clusters = ", ".join(cluster_labels)
                lines.append(f"  - **{b.name}** ({b.display_name}) [{status}]")
                lines.append(f"    clusters: {clusters}")

        if not servers and not businesses:
            lines.append("No MCP Servers configured. Use /add to add one.")

        return build_command_card("MCP Servers & Businesses", "\n".join(lines))

    def _cmd_memory(self, agent) -> str:
        entries = [e for e in agent.get_error_memory_entries() if e.error_type != "USER_FEEDBACK"]
        if not entries:
            return build_command_card("Error Memory", "No error memory.")

        lines = []
        businesses = agent.get_error_memory_businesses()
        for biz in businesses:
            biz_entries = [e for e in entries if e.business == biz]
            lines.append(f"**{biz}** ({len(biz_entries)}):")
            for i, e in enumerate(biz_entries, 1):
                lines.append(f"  {i}. [{e.error_type}] {e.lesson}")
        general = [e for e in entries if not e.business]
        if general:
            lines.append(f"\n**general** ({len(general)}):")
            for i, e in enumerate(general, 1):
                lines.append(f"  {i}. [{e.error_type}] {e.lesson}")

        return build_command_card("Error Memory", "\n".join(lines))

    def _cmd_clear(self, agent, args: list[str]) -> str:
        if args:
            biz_name = args[0]
            agent.clear_error_memory(business=biz_name)
            return build_command_card("Clear Memory", f"Cleared memory for business '{biz_name}'")
        agent.clear_error_memory()
        return build_command_card("Clear Memory", "Cleared all error memory")

    def _cmd_business(self, agent, args: list[str]) -> str:
        if not args or args[0] == "current":
            current = agent.get_locked_business()
            if current:
                return build_command_card("Business", f"Locked: {current}")
            return build_command_card("Business", "No locked business in current session.")

        if args[0] == "set":
            if len(args) < 2:
                return build_command_card("Business", "Usage: /business set <name>")
            try:
                agent.lock_business(args[1])
                return build_command_card("Business", f"Locked: {args[1]}")
            except KeyError as e:
                return build_error_card(str(e))

        if args[0] == "clear":
            agent.clear_locked_business()
            return build_command_card("Business", "Cleared locked business")

        return build_command_card("Business", "Usage: /business current | set <name> | clear")

    def _cmd_fields(self, agent) -> str:
        entries = agent.list_field_knowledge()
        if not entries:
            return build_command_card("Field Knowledge", "No field knowledge recorded.")

        lines = []
        table_groups: dict[str, list] = {}
        for e in entries:
            table_groups.setdefault(e.table, []).append(e)
        for table, fields in sorted(table_groups.items()):
            lines.append(f"**{table}:**")
            for f in fields:
                lines.append(f"  {f.column}: {f.description}")

        return build_command_card("Field Knowledge", "\n".join(lines))

    def _cmd_field(self, agent, args: list[str]) -> str:
        if len(args) < 2:
            return build_command_card("Field Knowledge", "Usage: /field <table>.<column> <description>")
        field_key = args[0]
        if "." not in field_key:
            return build_error_card("Field key must be in table.column format")
        table, column = field_key.split(".", 1)
        description = " ".join(args[1:])
        business = agent.get_last_business()
        agent.add_field_knowledge(business, table, column, description)
        return build_command_card("Field Knowledge", f"Added: {table}.{column}: {description}")

    def _cmd_field_rm(self, agent, args: list[str]) -> str:
        if len(args) < 1:
            return build_command_card("Field Knowledge", "Usage: /field_rm <table>.<column>")
        field_key = args[0]
        if "." not in field_key:
            return build_error_card("Field key must be in table.column format")
        table, column = field_key.split(".", 1)
        business = agent.get_last_business()
        removed = agent.remove_field_knowledge(business, table, column)
        if removed:
            return build_command_card("Field Knowledge", f"Removed: {table}.{column}")
        return build_command_card("Field Knowledge", f"Not found: {table}.{column}")

    def _cmd_rules(self, agent) -> str:
        rules = agent.list_preference_rules()
        if not rules:
            return build_command_card("Default Rules", "No default query rules.")

        lines = []
        for idx, rule in enumerate(rules, 1):
            biz = rule.business or "general"
            lines.append(f"{idx}. [{biz}] {rule.rule}")

        return build_command_card("Default Rules", "\n".join(lines))

    def _cmd_rules_clear(self, agent, args: list[str]) -> str:
        if args:
            biz_name = args[0]
            agent.clear_preference_rules(biz_name)
            return build_command_card("Clear Rules", f"Cleared default rules for business '{biz_name}'")
        agent.clear_preference_rules()
        return build_command_card("Clear Rules", "Cleared all default rules")

    def _cmd_remember(self, agent, args: list[str], user_id: str) -> str:
        rule = " ".join(args).strip()
        if not rule:
            return build_command_card("Remember", "Usage: /remember <default query rule>")
        business = agent.get_last_business()
        if not business:
            return build_command_card("Remember", "No business context. Run a query first or use /business set <name>.")
        agent.add_preference_rule(business, rule, source="feishu")
        return build_command_card("Remember", f"Saved default rule: [{business}] {rule}")

    async def _cmd_add(self, agent, args: list[str]) -> str:
        if len(args) < 2:
            return build_command_card("Add Server", "Usage: /add <server_name> <sse_url> [api_key]")

        name = args[0]
        url = args[1]
        api_key = args[2] if len(args) > 2 else ""

        await agent.add_mcp_server(name, url, api_key=api_key)

        businesses = agent.list_businesses()
        lines = [f"Added MCP Server: {name} ({url})"]
        if businesses:
            lines.append("\nDiscovered businesses:")
            for b in businesses:
                cluster_labels = [
                    f"{c} ({b.cluster_descriptions[c]})" if b.cluster_descriptions.get(c) else c
                    for c in b.cluster_routing.keys()
                ] if b.cluster_routing else ["pending"]
                clusters = ", ".join(cluster_labels)
                lines.append(f"  - {b.name} ({b.display_name}) clusters: {clusters}")

        return build_command_card("Add Server", "\n".join(lines))

    async def _cmd_remove(self, agent, args: list[str]) -> str:
        if len(args) < 1:
            return build_command_card("Remove Server", "Usage: /remove <server_name>")

        name = args[0]
        server_names = [s.name for s in agent.mcp_servers]
        if name not in server_names:
            return build_error_card(f"MCP Server '{name}' not found. Available: {', '.join(server_names) or 'none'}")

        await agent.remove_mcp_server(name)
        return build_command_card("Remove Server", f"Removed MCP Server: {name}")

    # ── 生命周期 ──

    async def start_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._session_manager.cleanup_expired()

    def run(self, host: str | None = None, port: int | None = None) -> None:
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

        async def _run_with_cleanup():
            cleanup_task = asyncio.create_task(self.start_cleanup_loop())
            try:
                await server.serve()
            finally:
                cleanup_task.cancel()

        import anyio
        anyio.run(_run_with_cleanup)
