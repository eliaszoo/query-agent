"""命令行交互入口 - 支持多轮对话的异步 REPL"""

import argparse
import asyncio
import readline
import sys
import threading
import time

from src.agent import QueryAgent
from src.config import load_config

EXIT_COMMANDS = {"exit", "quit", "q"}

# Slash 命令定义：命令名 → (补全文本, 简短说明)
SLASH_COMMANDS = {
    "/add":      ("/add ",      "Add business"),
    "/clear":    ("/clear ",    "Clear error memory"),
    "/exit":     ("/exit",      "Exit"),
    "/field":    ("/field ",    "Add field knowledge"),
    "/field_rm": ("/field_rm ", "Remove field knowledge"),
    "/fields":   ("/fields",    "List field knowledge"),
    "/list":     ("/list",      "List businesses"),
    "/memory":   ("/memory",    "Show error memory"),
    "/new":      ("/new",       "New conversation"),
    "/pin":      ("/pin ",      "Pin context message"),
    "/quit":     ("/quit",      "Exit"),
    "/remove":   ("/remove ",   "Remove business"),
}


def _slash_completer(text: str, state: int):
    """readline 补全函数：输入 / 时列出匹配的命令，显示简短说明。"""
    # 只对以 / 开头的输入进行补全
    line = readline.get_line_buffer().lstrip()
    if not line.startswith("/"):
        return None

    matches = [cmd for cmd in SLASH_COMMANDS if cmd.startswith(text)]
    if state < len(matches):
        # 返回补全文本（带空格的命令表示需要参数）
        return SLASH_COMMANDS[matches[state]][0]
    return None


def _display_hook(substitutions, matches, longest_hit_len):
    """自定义补全显示：每条命令附带说明。"""
    if not matches:
        return

    # matches 可能是补全文本（带空格），我们需要映射回命令名
    cmd_to_desc = {v[0]: v[1] for v in SLASH_COMMANDS.values()}

    lines = []
    for m in matches:
        desc = cmd_to_desc.get(m, "")
        lines.append(f"  {_CYAN}{m}{_RESET}  {_DIM}{desc}{_RESET}")

    print()
    print("\n".join(lines))
    # 重新打印 prompt
    prompt = f"\n{_BOLD}>{_RESET} "
    readline.redisplay()


def _setup_readline():
    """配置 readline 补全行为。"""
    readline.set_completer_delims(" \t\n")
    readline.set_completer(_slash_completer)
    readline.set_completion_display_matches_hook(_display_hook)
    # macOS libedit 和 GNU readline 用不同绑定
    try:
        readline.parse_and_bind("tab: menu-complete")
    except Exception:
        pass
    try:
        readline.parse_and_bind("bind ^I rl_complete")
    except Exception:
        pass

# 反馈检测关键词
_FEEDBACK_KEYWORDS = {
    "不对", "错了", "不是", "应该", "缺少", "遗漏", "多了", "少了", "不要", "不能",
    "记住", "默认", "以后都", "后续查询", "优先过滤",
}

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ANSI 样式
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


class Spinner:
    """终端 spinner 动画，在后台线程中循环显示进度指示。

    支持 pause/resume，在等待用户输入时暂停动画避免覆盖输入。

    用法:
        with Spinner("Thinking"):
            await some_long_operation()
    """

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._paused.set()  # 解除 pause 阻塞，让线程能退出
        if self._thread:
            self._thread.join()
        # 清除 spinner 行并恢复光标
        sys.stdout.write(f"\r{' ' * (len(self._message) + 4)}\r")
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
        return False

    def pause(self):
        """暂停 spinner，清除当前行，恢复光标（用于等待用户输入）。"""
        self._paused.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        sys.stdout.write(f"\r{' ' * (len(self._message) + 4)}\r")
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    def resume(self):
        """恢复 spinner 动画。"""
        self._paused.clear()
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._stop.clear()
        self._thread.start()

    def _spin(self):
        idx = 0
        while not self._stop.is_set():
            if self._paused.is_set():
                return  # pause 时线程退出，resume 时新建
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            sys.stdout.write(f"\r {_CYAN}{frame}{_RESET} {self._message}")
            sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)


# 当前活跃的 spinner 实例（模块级，供 confirm 回调使用）
_active_spinner: Spinner | None = None


def _confirm_with_spinner(prompt: str) -> bool:
    """确认回调：暂停 spinner → 等待用户输入 → 恢复 spinner。"""
    if _active_spinner:
        _active_spinner.pause()
    try:
        answer = input(f"   {prompt}").strip().lower()
        return answer == "y"
    finally:
        if _active_spinner:
            _active_spinner.resume()


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

{_DIM}Slash commands (Tab to autocomplete):{_RESET}
  {_CYAN}/add{_RESET} <name> <sse_url> [display] [key]   Add business
  {_CYAN}/remove{_RESET} <name>                    Remove business
  {_CYAN}/list{_RESET}                             List businesses
  {_CYAN}/memory{_RESET}                           Show error memory (grouped by business)
  {_CYAN}/clear{_RESET} [business]                  Clear memory (all if no business specified)
  {_CYAN}/new{_RESET}                              Start new conversation
  {_CYAN}/pin{_RESET} <message>                     Pin important context (survives history compression)
  {_CYAN}/field{_RESET} <table>.<col> <desc>         Add field knowledge
  {_CYAN}/field_rm{_RESET} <table>.<col>             Remove field knowledge
  {_CYAN}/fields{_RESET}                             List all field knowledge
"""
    return f"""\
{_BOLD}query-agent{_RESET}

{_DIM}Slash commands (Tab to autocomplete):{_RESET}
  {_CYAN}/add{_RESET} <name> <sse_url> [display] [key]   Add business
  {_CYAN}/remove{_RESET} <name>                    Remove business
  {_CYAN}/list{_RESET}                             List businesses
  {_CYAN}/memory{_RESET}                           Show error memory (grouped by business)
  {_CYAN}/clear{_RESET} [business]                  Clear memory (all if no business specified)
  {_CYAN}/new{_RESET}                              Start new conversation
  {_CYAN}/pin{_RESET} <message>                     Pin important context (survives history compression)
  {_CYAN}/field{_RESET} <table>.<col> <desc>         Add field knowledge
  {_CYAN}/field_rm{_RESET} <table>.<col>             Remove field knowledge
  {_CYAN}/fields{_RESET}                             List all field knowledge
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

    agent.add_business(name, url, display_name, api_key=api_key)
    print(f"  {_GREEN}Added{_RESET} business: {name} ({display_name}) -> {url}")


async def _handle_remove(agent: QueryAgent, args: list[str]) -> None:
    """处理 /remove 命令。"""
    if len(args) < 1:
        print(f"  {_DIM}Usage: /remove <name>{_RESET}")
        return

    name = args[0]
    try:
        await agent.remove_business(name)
        print(f"  {_GREEN}Removed{_RESET} business: {name}")
    except KeyError as e:
        print(f"  {_RED}Error:{_RESET} {e}")


def _handle_list(agent: QueryAgent) -> None:
    """处理 /list 命令。"""
    businesses = agent.list_businesses()
    if not businesses:
        print(f"  {_DIM}No businesses registered. Use /add to add one.{_RESET}")
        return

    for b in businesses:
        status = f"{_GREEN}loaded{_RESET}" if b.knowledge else f"{_YELLOW}pending{_RESET}"
        print(f"  - {b.name} ({b.display_name}) -> {b.mcp_server_url} [{status}]")


def _handle_memory(agent: QueryAgent) -> None:
    """处理 /memory 命令。"""
    entries = agent.get_error_memory_entries()
    if not entries:
        print(f"  {_DIM}No error memory.{_RESET}")
        return

    # 按业务分组显示
    businesses = agent.get_error_memory_businesses()
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
        agent.clear_error_memory(business=biz_name)
        print(f"  {_GREEN}Cleared{_RESET} memory for business '{biz_name}'")
    else:
        agent.clear_error_memory()
        print(f"  {_GREEN}Cleared{_RESET} all error memory")


async def main(config_path: str = "./config.yaml") -> None:
    """异步主函数，运行交互式查询循环。"""
    global _active_spinner
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

    _setup_readline()

    agent = QueryAgent(config_path=config_path, confirm_callback=_confirm_with_spinner)

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

        if cmd == "/field":
            if len(parts) < 3:
                print(f"  {_DIM}Usage: /field <table>.<column> <description>{_RESET}")
                print(f"  {_DIM}Example: /field tb_voice.origin 1=自研,2=阿里云,3=腾讯云,5=火山引擎{_RESET}")
                continue
            field_key = parts[1]
            if "." not in field_key:
                print(f"  {_RED}Error:{_RESET} Field key must be in table.column format (e.g. tb_voice.origin)")
                continue
            table, column = field_key.split(".", 1)
            description = " ".join(parts[2:])
            business = agent.get_last_business()
            agent.add_field_knowledge(business, table, column, description)
            print(f"  {_GREEN}Added field knowledge:{_RESET} {table}.{column}: {description}")
            continue

        if cmd == "/field_rm":
            if len(parts) < 2:
                print(f"  {_DIM}Usage: /field_rm <table>.<column>{_RESET}")
                continue
            field_key = parts[1]
            if "." not in field_key:
                print(f"  {_RED}Error:{_RESET} Field key must be in table.column format")
                continue
            table, column = field_key.split(".", 1)
            business = agent.get_last_business()
            removed = agent.remove_field_knowledge(business, table, column)
            if removed:
                print(f"  {_GREEN}Removed:{_RESET} {table}.{column}")
            else:
                print(f"  {_YELLOW}Not found:{_RESET} {table}.{column}")
            continue

        if cmd == "/fields":
            entries = agent.list_field_knowledge()
            if not entries:
                print(f"  {_DIM}No field knowledge recorded.{_RESET}")
            else:
                # 按表分组显示
                table_groups: dict[str, list] = {}
                for e in entries:
                    if e.table not in table_groups:
                        table_groups[e.table] = []
                    table_groups[e.table].append(e)
                for table, fields in sorted(table_groups.items()):
                    print(f"\n  {_BOLD}{table}{_RESET}:")
                    for f in fields:
                        print(f"    {f.column}: {f.description}")
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
            feedback_lesson = agent.extract_explicit_feedback_lesson(user_input)
            if feedback_lesson is None:
                spinner = Spinner("Analyzing feedback")
                _active_spinner = spinner
                with spinner:
                    feedback_lesson = await agent.extract_feedback_lesson(
                        original_query=last_query,
                        agent_response=last_response,
                        user_feedback=user_input,
                    )
                _active_spinner = None
            if feedback_lesson:
                business = agent.get_last_business()
                agent.record_feedback(last_query, business, user_input, feedback_lesson)
                print(f"  {_YELLOW}Saved lesson:{_RESET} {feedback_lesson}")
            else:
                print(f"  {_DIM}No actionable lesson extracted from feedback.{_RESET}")
            continue

        # 正常查询流程
        try:
            spinner = Spinner("Thinking")
            _active_spinner = spinner
            with spinner:
                response = await agent.run_query(user_input)
            _active_spinner = None
            print(f"\n{response}")

            # 展示查询元信息
            if agent.last_metrics:
                m = agent.last_metrics
                selection_info = ""
                if m.business_selection_strategy:
                    selection_info = (
                        f" | business={m.selected_business or 'unknown'}"
                        f" via {m.business_selection_strategy}"
                    )
                print(
                    f"{_DIM}({m.duration_seconds}s | "
                    f"{m.input_tokens}+ {m.output_tokens}- | "
                    f"{m.tool_calls} tool calls"
                    f"{selection_info}){_RESET}"
                )

            last_query = user_input
            last_response = response

        except Exception as e:
            import traceback
            print(f"\n  {_RED}Error:{_RESET} {e}")
            traceback.print_exc()

    # 清理 SSE 连接，避免 asyncio 退出时报 cancel scope 错误
    try:
        await agent.registry.close_sessions()
    except Exception:
        pass


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
