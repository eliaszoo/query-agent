"""命令行交互入口 - 支持多轮对话的异步 REPL"""

import argparse
import asyncio
import sys

from src.agent import QueryAgent
from src.config import load_config

EXIT_COMMANDS = {"exit", "quit", "q"}


def _build_welcome_message(businesses: list = None) -> str:
    """根据业务列表构建欢迎信息。"""
    businesses = businesses or []

    if len(businesses) > 1:
        biz_list = "\n".join(f"  - {b.name} ({b.display_name})" for b in businesses)
        return f"""\
🤖 查询 Agent (多业务模式)
====================
已注册业务：
{biz_list}

输入自然语言查询数据，Agent 会自动路由到对应业务。

特殊命令：
  - /add <name> <sse_url> [显示名]  添加业务
  - /remove <name>                  移除业务
  - /list                           列出所有业务
  - /memory                         查看 Agent 已学到的错误经验
  - /clear                          清空错误记忆
  - /new                            开始新对话（清空上下文）
  - exit/quit/q                     退出
"""
    description = businesses[0].display_name if businesses else "业务"
    return f"""\
🤖 查询 Agent ({description})
====================
输入自然语言查询{description}数据，例如：
  - 帮我查一下测试环境的数据
  - 查询生产环境的记录数

特殊命令：
  - /add <name> <sse_url> [显示名]  添加业务
  - /remove <name>                  移除业务
  - /list                           列出所有业务
  - /memory                         查看 Agent 已学到的错误经验
  - /clear                          清空错误记忆
  - /new                            开始新对话（清空上下文）
  - exit/quit/q                     退出
"""


async def _handle_add(agent: QueryAgent, args: list[str]) -> None:
    """处理 /add 命令。"""
    if len(args) < 2:
        print("用法: /add <name> <sse_url> [显示名]")
        return

    name = args[0]
    url = args[1]
    display_name = args[2] if len(args) > 2 else name

    agent.registry.register(name, url, display_name)
    print(f"✅ 已添加业务: {name} ({display_name}) -> {url}")


async def _handle_remove(agent: QueryAgent, args: list[str]) -> None:
    """处理 /remove 命令。"""
    if len(args) < 1:
        print("用法: /remove <name>")
        return

    name = args[0]
    try:
        await agent.registry.remove(name)
        print(f"🗑️ 已移除业务: {name}")
    except KeyError as e:
        print(f"❌ {e}")


def _handle_list(agent: QueryAgent) -> None:
    """处理 /list 命令。"""
    businesses = agent.registry.list_businesses()
    if not businesses:
        print("📝 暂无已注册业务，使用 /add 添加")
        return

    print(f"📋 共 {len(businesses)} 个业务：")
    for b in businesses:
        knowledge_status = "✅ 已加载" if b.knowledge else "⏳ 未加载"
        print(f"  - {b.name} ({b.display_name}) -> {b.mcp_server_url} [{knowledge_status}]")


async def main(config_path: str = "./config.yaml") -> None:
    """异步主函数，运行交互式查询循环。"""
    config = load_config(config_path)
    businesses = []
    if config.businesses:
        from src.business_registry import BusinessEntry
        businesses = [
            BusinessEntry(
                name=name,
                display_name=cfg.display_name,
                mcp_server_url=cfg.mcp_server_url,
            )
            for name, cfg in config.businesses.items()
        ]

    print(_build_welcome_message(businesses))

    agent = QueryAgent(config_path=config_path)

    while True:
        try:
            user_input = input("\n🧑 请输入查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("👋 再见！")
            break

        # 特殊命令：查看错误记忆
        if user_input.lower() == "/memory":
            entries = agent.error_memory.get_entries()
            if not entries:
                print("📝 暂无错误记忆")
            else:
                print(f"📝 共 {len(entries)} 条错误记忆：")
                for i, e in enumerate(entries, 1):
                    print(f"  {i}. [{e.error_type}] {e.lesson}")
            continue

        # 特殊命令：清空错误记忆
        if user_input.lower() == "/clear":
            agent.error_memory.clear()
            print("🗑️ 错误记忆已清空")
            continue

        # 特殊命令：开始新对话
        if user_input.lower() == "/new":
            agent.clear_history()
            print("🔄 已开始新对话")
            continue

        # 特殊命令：添加业务
        if user_input.lower().startswith("/add"):
            parts = user_input.split()
            await _handle_add(agent, parts[1:])
            continue

        # 特殊命令：移除业务
        if user_input.lower().startswith("/remove"):
            parts = user_input.split()
            await _handle_remove(agent, parts[1:])
            continue

        # 特殊命令：列出业务
        if user_input.lower() == "/list":
            _handle_list(agent)
            continue

        print("⏳ 查询中...")
        try:
            response = await agent.run_query(user_input)
            print(f"\n🤖 {response}")

            # 展示查询元信息
            if agent.last_metrics:
                m = agent.last_metrics
                print(
                    f"\n📊 耗时 {m.duration_seconds}s | "
                    f"Token: {m.input_tokens}↑ {m.output_tokens}↓ | "
                    f"工具调用: {m.tool_calls}次"
                )
        except Exception as e:
            # 打印完整异常链，方便排查
            import traceback
            print(f"\n❌ 查询出错: {e}")
            traceback.print_exc()
            print("请重新输入查询，或输入 exit 退出。")


def main_entry() -> None:
    """同步入口，供 pyproject.toml entry point 使用。"""
    parser = argparse.ArgumentParser(description="查询 Agent")
    parser.add_argument(
        "--config", default="./config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(config_path=args.config))
    except KeyboardInterrupt:
        print("\n👋 再见！")
        sys.exit(0)


if __name__ == "__main__":
    main_entry()
