# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Query Agent (通用查询 Agent) — a configurable NL-to-SQL query system. Users ask questions in natural language; the agent translates them to SQL, executes read-only snapshot queries against MySQL, and returns results. Built with Python 3.10+, MCP protocol, and multi-LLM support. Supports **multi-business dynamic routing**: a single agent instance can query multiple business domains, each backed by a separate MCP Server. Domain knowledge (term mappings, table relationships, status codes) is configured via YAML or fetched from MCP Servers, making the agent adaptable to any business domain.

## Build & Run Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the interactive agent (local MCP server via stdio)
python -m src.main --config config.yaml

# Run the MCP server standalone (stdio mode)
python -m src.query_mcp_server

# Run the MCP server standalone (SSE/HTTP mode for remote deployment)
MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m src.query_mcp_server --transport sse

# Run all tests
pytest

# Run a single test file
pytest tests/test_sql_validator.py

# Run a single test by name
pytest tests/test_sql_validator.py::TestValidate::test_for_update_rejected
```

## Architecture

The system has two deployable components connected via MCP protocol:

1. **Agent (client side)** — `src/main.py` → `src/agent.py`
   - Interactive REPL that accepts natural language queries
   - Uses LLM (via `src/llm_provider.py`) to translate NL → SQL tool calls
   - **Multi-business routing**: `BusinessRegistry` manages multiple MCP Server connections; LLM auto-selects target business via `business` parameter in tool calls
   - Connects to MCP Servers via **SSE** (remote) for multi-business mode, or **stdio** (local) for backward-compatible single-business mode
   - Maintains multi-turn conversation history (max 10 turns)
   - Error memory system (`src/error_memory.py`) persists past mistakes into `error_memory.json` and injects lessons into the system prompt
   - Business knowledge loaded from config and/or fetched from MCP Servers via `get_business_knowledge` tool
   - REPL commands: `/add`, `/remove`, `/list` for dynamic business management

2. **MCP Server (server side)** — `src/query_mcp_server.py`
   - Exposes four MCP tools: `execute_readonly_sql`, `get_cluster_list`, `get_table_schema`, `get_business_knowledge`
   - Manages per-cluster MySQL connection pools (`src/db_pool.py`) via aiomysql
   - All SQL passes through `src/sql_validator.py` (whitelist tables, forbid writes, forbid lock clauses, strip comments, enforce LIMIT)

### Key modules

| Module | Role |
|--------|------|
| `src/main.py` | CLI entry point, async REPL loop with `/add`/`/remove`/`/list` commands |
| `src/agent.py` | `QueryAgent` — multi-business MCP routing + conversation loop + business knowledge fetching |
| `src/business_registry.py` | `BusinessRegistry` — manages multiple MCP Server SSE connections with lazy connect and knowledge caching |
| `src/llm_provider.py` | Abstract `LLMProvider` with `AnthropicProvider` and `OpenAICompatibleProvider` implementations |
| `src/config.py` | YAML config loading with `${ENV_VAR}` substitution, includes `BusinessKnowledge`, `BusinessEntryConfig`, `businesses` config section |
| `src/db_pool.py` | `ConnectionPoolManager` — per-cluster aiomysql pools |
| `src/sql_validator.py` | `SQLValidator` — SELECT-only, configurable table whitelist, lock clause rejection, comment stripping, multi-statement rejection, LIMIT enforcement |
| `src/prompts.py` | `build_system_prompt(businesses, knowledge_map)` — generic framework + multi-business domain knowledge injection |
| `src/error_memory.py` | `ErrorMemoryManager` — JSON-backed error persistence, auto-lesson generation |
| `src/query_mcp_server.py` | FastMCP server with four tools, lazy init, stdio/SSE transport |

### LLM Provider abstraction

`src/llm_provider.py` provides a unified `chat()` interface. The `create_provider()` factory selects between:
- `anthropic` — direct Anthropic SDK (Claude)
- `openai_compatible` — OpenAI SDK pointed at any compatible endpoint (DeepSeek, GPT, GLM, Qwen, Zego gateway, etc.)

Tool result message formats differ between providers (Anthropic uses `type: tool_result` in user content; OpenAI uses `role: tool` messages). The provider handles this translation.

### Configuration

Three config files for different deployment scenarios:
- `config.yaml` — full config with DB credentials via `${ENV_VAR}` placeholders, `business_knowledge` section, and agent settings
- `config-local.yaml` — agent-only config with `businesses` section pointing to remote MCP Server SSE endpoints, with LLM provider settings
- `config-server.yaml` — MCP server-only config (DB + SQL security + business_knowledge, no agent section)

**Multi-business mode**: When `businesses` section is present, the agent connects to multiple MCP Servers via SSE. Each business is registered in `BusinessRegistry` and tools are merged with a `business` parameter for LLM auto-routing.

**Backward compatibility**: If `agent.mcp_server_url` is set but no `businesses` section, the agent auto-creates a "default" business entry, preserving single-business behavior. If neither `businesses` nor `mcp_server_url` is set, the agent uses local stdio mode.

### Multi-business routing

The agent supports querying multiple business domains through a single instance:
1. Each business has its own MCP Server (SSE endpoint) registered in `BusinessRegistry`
2. Tool definitions from all businesses are merged with a `business` parameter injected into each tool's input schema
3. The LLM receives a system prompt listing available businesses and auto-selects the target by specifying `business` in tool calls
4. The agent's `_route_tool_call()` extracts the `business` parameter and routes the call to the corresponding MCP Server
5. Business knowledge is fetched lazily from each MCP Server's `get_business_knowledge` tool on first use

### Business Knowledge configuration

The `business_knowledge` section in config.yaml drives the system prompt:
- `description`: business domain name (e.g., "数字人平台")
- `term_mappings`: natural language → table name mappings
- `table_relationships`: foreign key relationship descriptions
- `status_codes`: enum value descriptions
- `custom_rules`: additional query rules appended after the standard 6 rules

When using SSE mode with a remote MCP server, the agent fetches business knowledge at startup via `get_business_knowledge` tool (only if local config has no business_knowledge).

### SQL Security Pipeline

Every SQL query passes through: comment stripping → multi-statement check → sqlparse type check (SELECT only) → forbidden keyword check → **lock clause rejection** (FOR UPDATE / LOCK IN SHARE MODE / FOR SHARE) → table whitelist check → LIMIT enforcement. The allowed tables list is configured via `sql_security.allowed_tables` and passed to `SQLValidator` constructor. Connection pools use `autocommit=True` to ensure no transaction locks are held.

## Testing

Tests use pytest with `pytest-asyncio` (asyncio_mode = "auto"). The `tests/` directory mirrors `src/` with `test_` prefixed files. No database is required — all DB interactions are mocked.
