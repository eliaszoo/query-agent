# CLAUDE.md

This repository is a natural-language query agent for MySQL built on MCP.

## What the system does

- Converts user questions into read-only SQL via LLM
- Supports local `stdio` mode and remote multi-business `SSE` mode
- Routes queries across businesses with heuristic/LLM/locked selection
- Persists field knowledge, error memory, and default preference rules
- Detects SQL performance risks before execution (index-based + LLM risk_note)
- Auto-extracts field enum meanings from LLM output

## Run commands

```bash
pip install -e ".[dev]"

# Agent (stdio mode)
python -m src.main --config config.yaml
query-agent --config config.yaml

# MCP Server (SSE mode)
MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m src.query_mcp_server --transport sse --config config-server.yaml

# Tests
pytest
pytest tests/test_agent.py tests/test_main.py -q
pytest tests/test_prompts.py tests/test_config.py tests/test_business_registry.py -q
```

## Architecture

```text
main.py (REPL) → agent.py (QueryAgent)
  ├── ConversationState        — history, pinned messages, compression
  ├── BusinessSelectionService — heuristic + LLM business routing
  ├── PromptService            — prompt assembly + cache
  ├── ToolExecutionService     — tool routing, risk check, index/schema cache
  ├── KnowledgeStore           — error memory + field knowledge coordination
  ├── QueryRuleExecutor        — default rule matching and argument rewrite
  └── PreferenceRulesManager   — user preference rule persistence

MCP Server (query_mcp_server.py):
  execute_readonly_sql, get_cluster_list, get_table_schema,
  get_table_indexes, get_business_knowledge
  → SQLValidator → ConnectionPoolManager → MySQL
```

## Main modules

| File | Purpose |
|---|---|
| `src/main.py` | Interactive REPL, slash commands, spinner, feedback detection |
| `src/agent.py` | `QueryAgent`, conversation loop, business selection, learning integration |
| `src/llm_provider.py` | LLM abstraction (Anthropic / OpenAI Compatible) |
| `src/prompts.py` | System prompt templates (single + multi-business) |
| `src/config.py` | YAML config loading, `${ENV_VAR}` substitution, validation |
| `src/business_registry.py` | Multi-business SSE session management, lazy connect, knowledge caching |
| `src/business_selection_service.py` | Business selection with heuristic/LLM/locked strategies |
| `src/conversation_state.py` | Conversation history, pinned messages, trim/compress |
| `src/prompt_service.py` | Prompt assembly with caching and dirty-flag invalidation |
| `src/tool_execution_service.py` | Tool routing, pre-execution risk check, index/schema caching |
| `src/knowledge_store.py` | Error memory + field knowledge coordination and auto-extraction |
| `src/field_knowledge.py` | Field knowledge persistence and prompt rendering |
| `src/error_memory.py` | Error lesson persistence with token-budget prompt generation |
| `src/preference_rules.py` | Default rule persistence |
| `src/query_rule_executor.py` | Rule matching and tool argument rewriting |
| `src/query_plan.py` | Query plan preview data structure |
| `src/query_mcp_server.py` | FastMCP server with Bearer auth, SQL tools, lazy init |
| `src/sql_validator.py` | SQL safety validation (SELECT-only, whitelist, lock rejection) |
| `src/sql_risk_checker.py` | Index-based SQL performance risk analysis |
| `src/db_pool.py` | Per-cluster aiomysql connection pool management |

## Important current behaviors

### Business routing

- Multi-business selection goes through `BusinessSelectionService.select_business()`
- Strategies: `single` → `heuristic` → `llm` → `fallback_all`
- Selection result includes strategy name, shown to user in query metadata
- `build_query_plan()` and actual execution must stay consistent

### Learning system

Three persisted knowledge channels — do not mix casually:

1. **`error_memory.json`** — failure lessons (SQL rejected, wrong table, etc.)
2. **`field_knowledge.json`** — enum/value meanings extracted from model output
3. **`preference_rules.json`** — reusable user preferences (e.g., "默认只查可用数据")

Feedback like "以后默认只查可用数据" should become a preference rule, not generic error memory.

### FIELD_KNOWLEDGE contract

- Prompts instruct the model to emit: `<!-- FIELD_KNOWLEDGE: [...] -->`
- `KnowledgeStore` strips that tag from user-visible output
- Auto-extraction also has Markdown-table fallback regex patterns
- If you change prompt wording or output parsing, preserve this contract

### Risk detection (dual-track)

1. **LLM risk_note** (priority) — LLM declares risk via `risk_note` tool parameter
   - "索引驱动: app_id" → no risk (index-driven query)
   - "全表扫描风险" → high risk
   - "SELECT * 返回全列" → medium risk
2. **Static index analysis** (fallback) — `SQLRiskChecker` checks WHERE columns against indexes
   - Driving column (index prefix hit) → non-indexed columns are filtered, not flagged
   - No driving column → high risk

### Default rules

Structured rule types: `available_only`, `default_cluster_test`, `natural_language`

Rule application: `QueryRuleExecutor.apply()` → rewrites tool arguments before execution.

To add a new rule type, update:
- `src/agent.py` explicit feedback parsing
- `src/query_rule_executor.py`
- prompt text and tests

### Query execution safety

Do not bypass:
- `ToolExecutionService.pre_execute_check()`
- `SQLValidator`
- `SQLRiskChecker`

High-risk queries pause for user confirmation.

## CLI surface

Slash commands (Tab-completable):

- `/add`, `/remove`, `/list` — business management
- `/memory`, `/clear` — error memory
- `/field`, `/field_rm`, `/fields` — field knowledge
- `/remember`, `/rules`, `/rules_clear` — preference rules
- `/new`, `/pin` — conversation control

If you change command behavior, update both `README.md` and `tests/test_main.py`.

## Persistence model

Local state stored under `.query-agent/<namespace>/`:

- `namespace` from `storage.namespace` config, or auto-derived from config path
- Different configs → different namespaces → no cross-contamination

Files: `error_memory.json`, `field_knowledge.json`, `preference_rules.json`

## Configuration facts

- `businesses` section → multi-business SSE mode
- `agent.mcp_server_url` without `businesses` → auto-creates `default` business
- Neither → local `stdio` mode (MCP server as subprocess)
- `auth.api_key` → Bearer token auth for MCP Server SSE mode
- `business_knowledge` → domain knowledge injected into system prompt

## Testing expectations

Before finishing a change:

```bash
# Routing, CLI, or learning behavior
pytest tests/test_agent.py tests/test_main.py -q

# Prompt or config changes
pytest tests/test_prompts.py tests/test_config.py tests/test_business_registry.py -q

# SQL safety or risk detection
pytest tests/test_sql_validator.py tests/test_sql_risk_checker.py -q
```

## Editing guidance

- Keep changes incremental; this codebase has behavior coverage around P0/P1 work
- Preserve current service boundaries; don't push more logic into `main.py`
- When adding user-visible behavior, include or update tests
- When modifying business routing, ensure all selection strategies are covered
- When modifying prompts, check that FIELD_KNOWLEDGE contract and rule numbering stay consistent
- `arguments.pop("business")` in `ToolExecutionService.route_tool_call()` mutates the input dict — be careful when logging or re-accessing arguments after this call
