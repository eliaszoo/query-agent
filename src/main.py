"""命令行交互入口 - 支持多轮对话的异步 REPL"""

import argparse
import asyncio
import sys

from src.agent import QueryAgent
from src.config import load_config

EXIT_COMMANDS = {"exit", "quit", "q"}

# 反馈检测关键词
_FEEDBACK_KEYWORDS = {"不对", "错了", "不是", "应该", "缺少", "遗漏", "多了", "少了", "不要", "不能"}

# ANSI 样式
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _likely_feedback(text: str) -> bool:
    """启发式判断用户输入是否可能是对前次结果的反馈。

    短输入 + 包含否定/纠正关键词 → 可能是反馈。
    长输入大概率是新查询，不触发 LLM 判断。
    """
    if len(text) > 100:
        return False
    return any(kw in text for kw in _FEEDBACK_KEYWORDS)


def _build_welcome_message(businesses: list = None) -> str:
    """根据业务列表构建欢迎信息。"""
    businesses = businesses or []

    if len(businesses) > 1:
        biz_list = "\n".join(f"  - {b.name} ({b.display_name})" for b in businesses)
        return f"""\
{_BOLD}query-agent{_RESET} {_DIM}(multi-business){_RESET}

Registered businesses:
{biz_list}

{_DIM}Slash commands:{_RESET}
  {_CYAN}/add{_RESET} <name> <sse_url> [display] [key]   Add business
  {_CYAN}/remove{_RESET} <name>                    Remove business
  {_CYAN}/list{_RESET}                             List businesses
  {_CYAN}/memory{_RESET}                           Show error memory (grouped by business)
  {_CYAN}/clear{_RESET} [business]                  Clear memory (all if no business specified)
  {_CYAN}/new{_RESET}                              Start new conversation
  {_CYAN}/pin{_RESET} <message>                     Pin important context (survives history compression)
"""
    description = businesses[0].display_name if businesses else "business"
    return f"""\
{_BOLD}query-agent{_RESET} {_DIM}({description}){_RESET}

{_DIM}Slash commands:{_RESET}
  {_CYAN}/add{_RESET} <name> <sse_url> [display] [key]   Add business
  {_CYAN}/remove{_RESET} <name>                    Remove business
  {_CYAN}/list{_RESET}                             List businesses
  {_CYAN}/memory{_RESET}                           Show error memory (grouped by business)
  {_CYAN}/clear{_RESET} [business]                  Clear memory (all if no business specified)
  {_CYAN}/new{_RESET}                              Start new conversation
  {_CYAN}/pin{_RESET} <message>                     Pin important context (survives history compression)
"""


async def _handle_add(agent: QueryAgent, args: list[str]) -> None:
    """处理 /add 命令。"""
    if len(args) < 2:
        print(f"  {_DIM}Usage: /add <name> <sse_url> [display_name] [api_key]{_RESET}")
        return

    name = args[0]
    url = args[1]
    display_name = args[2] if len(args) > 2 else name
    api_key = args[3] if len(args) > 3 else ""

    agent.registry.register(name, url, display_name, api_key=api_key)
    print(f"  {_GREEN}Added{_RESET} business: {name} ({display_name}) -> {url}")


async def _handle_remove(agent: QueryAgent, args: list[str]) -> None:
    """处理 /remove 命令。"""
    if len(args) < 1:
        print(f"  {_DIM}Usage: /remove <name>{_RESET}")
        return

    name = args[0]
    try:
        await agent.registry.remove(name)
        print(f"  {_GREEN}Removed{_RESET} business: {name}")
    except KeyError as e:
        print(f"  {_RED}Error:{_RESET} {e}")


def _handle_list(agent: QueryAgent) -> None:
    """处理 /list 命令。"""
    businesses = agent.registry.list_businesses()
    if not businesses:
        print(f"  {_DIM}No businesses registered. Use /add to add one.{_RESET}")
        return

    for b in businesses:
        status = f"{_GREEN}loaded{_RESET}" if b.knowledge else f"{_YELLOW}pending{_RESET}"
        print(f"  - {b.name} ({b.display_name}) -> {b.mcp_server_url} [{status}]")


def _handle_memory(agent: QueryAgent) -> None:
    """处理 /memory 命令。"""
    entries = agent.error_memory.get_entries()
    if not entries:
        print(f"  {_DIM}No error memory.{_RESET}")
        return

    # 按业务分组显示
    businesses = agent.error_memory.get_businesses()
    for biz in businesses:
        biz_entries = [e for e in entries if e.business == biz]
        print(f"\n  {_BOLD}{biz}{_RESET} ({len(biz_entries)}):")
        for i, e in enumerate(biz_entries, 1):
            print(f"    {i}. [{_YELLOW}{e.error_type}{_RESET}] {e.lesson}")
    # 通用经验
    general = [e for e in entries if not e.business]
    if general:
        print(f"\n  {_BOLD}general{_RESET} ({len(general)}):")
        for i, e in enumerate(general, 1):
            print(f"    {i}. [{_YELLOW}{e.error_type}{_RESET}] {e.lesson}")
    print(f"\n  {_DIM}Use /clear [business] to clear (all if no business specified){_RESET}")


def _handle_clear(agent: QueryAgent, args: list[str]) -> None:
    """处理 /clear 命令。"""
    if args:
        biz_name = args[0]
        agent.error_memory.clear(business=biz_name)
        print(f"  {_GREEN}Cleared{_RESET} memory for business '{biz_name}'")
    else:
        agent.error_memory.clear()
        print(f"  {_GREEN}Cleared{_RESET} all error memory")


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
                api_key=cfg.api_key,
            )
            for name, cfg in config.businesses.items()
        ]

    print(_build_welcome_message(businesses))

    agent = QueryAgent(config_path=config_path)

    last_query = ""
    last_response = ""

    while True:
        try:
            user_input = input(f"\n{_BOLD}>{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            break

        # Slash commands
        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd == "/memory":
            _handle_memory(agent)
            continue

        if cmd == "/clear":
            _handle_clear(agent, parts[1:])
            continue

        if cmd == "/new":
            agent.clear_history()
            last_query = ""
            last_response = ""
            print(f"  {_GREEN}Started new conversation.{_RESET}")
            continue

        if cmd == "/pin":
            pin_content = " ".join(parts[1:])
            if pin_content:
                agent.pin_message(pin_content)
                print(f"  {_GREEN}Pinned: {pin_content}{_RESET}")
            else:
                print(f"  Usage: /pin <message>")
            continue

        if cmd == "/add":
            await _handle_add(agent, parts[1:])
            continue

        if cmd == "/remove":
            await _handle_remove(agent, parts[1:])
            continue

        if cmd == "/list":
            _handle_list(agent)
            continue

        # 检测用户反馈：如果上一轮有查询结果，判断当前输入是否是对前次结果的纠正
        # 反馈只记录经验，不触发查询
        if last_query and last_response and _likely_feedback(user_input):
            feedback_lesson = await agent.extract_feedback_lesson(
                original_query=last_query,
                agent_response=last_response,
                user_feedback=user_input,
            )
            if feedback_lesson:
                business = agent._last_query_context.get("business", "") if agent._last_query_context else ""
                agent.error_memory.add_error(
                    user_query=last_query,
                    error_type="USER_FEEDBACK",
                    business=business,
                    error_message=user_input,
                    lesson=feedback_lesson,
                )
                agent._mark_prompt_dirty()
                print(f"  {_YELLOW}Saved lesson:{_RESET} {feedback_lesson}")
            else:
                print(f"  {_DIM}No actionable lesson extracted from feedback.{_RESET}")
            continue

        # 正常查询流程
        try:
            response = await agent.run_query(user_input)
            print(f"\n{response}")

            # 展示查询元信息
            if agent.last_metrics:
                m = agent.last_metrics
                print(
                    f"{_DIM}({m.duration_seconds}s | "
                    f"{m.input_tokens}+ {m.output_tokens}- | "
                    f"{m.tool_calls} tool calls){_RESET}"
                )

            last_query = user_input
            last_response = response

        except Exception as e:
            import traceback
            print(f"\n  {_RED}Error:{_RESET} {e}")
            traceback.print_exc()


def main_entry() -> None:
    """同步入口，供 pyproject.toml entry point 使用。"""
    parser = argparse.ArgumentParser(description="query-agent")
    parser.add_argument(
        "--config", default="./config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(config_path=args.config))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main_entry()
