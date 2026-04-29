# Query Agent

通用 MySQL 查询 Agent，用户输入自然语言，LLM 自动生成 SQL 并执行只读查询。支持多业务路由、多 LLM Provider、索引级风险检测。

## 核心能力

- 自然语言 → SQL，支持多轮对话上下文
- 多业务动态路由：单 Agent 连接多个 MCP Server，LLM 自动选择目标业务
- 多 LLM Provider：Anthropic Claude / OpenAI GPT / DeepSeek / GLM / Qwen
- SQL 安全管道：白名单表 → SELECT-only → 禁止锁子句 → 强制 LIMIT → 注释剥离
- 索引级性能风险检测：基于 `SHOW INDEX` 信息判断查询是否命中索引
- 业务领域知识注入：术语映射、表关系、状态码，通过 YAML 或 MCP 工具动态加载
- 字段知识持久化：自动从 LLM 回复中提取枚举字段含义，后续查询直接使用无需重复探索
- 错误记忆：自动从失败中学习，将教训注入后续 system prompt

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                     Agent (客户端)                        │
│                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌────────────┐    │
│   │ CLI REPL │───→│ LLM Provider │───→│ Tool Router │    │
│   └──────────┘    └──────────────┘    └─────┬──────┘    │
│                                              │           │
│                                    ┌─────────┴──────┐   │
│                                    │BusinessRegistry │   │
│                                    └──┬─────────┬───┘   │
│                                       │         │       │
│                              ┌────────┴──┐ ┌────┴─────┐ │
│                              │RiskChecker │ │ErrMemory  │ │
│                              └───────────┘ │FieldKnow  │ │
│                                            └───────────┘ │
└──────────────────────────────┼───────────────────────────┘
                                 │
                          stdio / SSE
                                 │
┌────────────────────────────────┼────────────────────────┐
│                        MCP Server (服务端)               │
│                                │                        │
│   ┌────────────────────────────┴──────────────────┐     │
│   │  execute_readonly_sql  │  get_table_schema     │     │
│   │  get_cluster_list      │  get_table_indexes    │     │
│   │  get_business_knowledge                        │     │
│   └────────────────────────┬──────────────────────┘     │
│                            │                            │
│               ┌────────────┴────────────┐               │
│               │  SQL Validator          │               │
│               │  Connection Pool        │               │
│               └────────────┬────────────┘               │
│                            │                            │
│                     MySQL (只读)                         │
└─────────────────────────────────────────────────────────┘
```

两种部署模式：

| 模式 | Agent | MCP Server | 传输 | 场景 |
|------|-------|------------|------|------|
| stdio 一体 | 本地 | 本地子进程 | stdin/stdout | 开发调试、单业务 |
| SSE 分离 | 本地 | 远程（靠近 DB） | HTTP/SSE | 生产部署、多业务 |

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 本地 stdio 模式

Agent 和 MCP Server 在同一进程，适合开发调试：

```bash
export ANTHROPIC_API_KEY=sk-xxx
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx

python -m src.main --config config.yaml
```

### 远程 SSE 模式

MCP Server 部署在数据库所在机器，Agent 通过 SSE 远程连接：

```bash
# 1. 启动 MCP Server（数据库机器）
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx
export MCP_API_KEY=your-secret

python -m src.query_mcp_server --transport sse --config config-server.yaml

# 2. 启动 Agent（本地）
export LLM_API_KEY=xxx
export MCP_API_KEY=your-secret

python -m src.main --config config-local.yaml
```

## 配置

配置文件使用 YAML 格式，支持 `${ENV_VAR}` 环境变量替换。

### 全量配置（config.yaml）— stdio 一体模式

```yaml
clusters:
  test:
    description: "测试环境"
    host: "${DB_TEST_HOST}"
    port: 3306
    database: "mydb"
    user: "readonly_user"
    password: "${DB_TEST_PASSWORD}"

sql_security:
  max_rows: 100
  query_timeout: 30
  allowed_tables:
    - "tb_user"
    - "tb_order"

business_knowledge:
  description: "业务平台"
  term_mappings:
    "用户": "tb_user 表"
  table_relationships:
    - "tb_order.user_id → tb_user.id"
  status_codes:
    - "tb_order.status: 1=待支付, 2=已支付"
  custom_rules: []

agent:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  default_cluster: "test"
```

### 多业务配置（config-local.yaml）— SSE 分离模式

```yaml
agent:
  provider: "openai_compatible"
  model: "glm-5.1"
  max_tokens: 4096
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL}"

businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://mcp-host:8765/sse"
    api_key: "${MCP_API_KEY}"
  order:
    display_name: "订单"
    mcp_server_url: "http://other-host:8765/sse"
```

### LLM Provider 选项

| Provider | provider 值 | 示例 model | 需要 base_url |
|----------|-------------|-----------|--------------|
| Anthropic Claude | `anthropic` | `claude-sonnet-4-20250514` | 否 |
| OpenAI GPT | `openai_compatible` | `gpt-4o` | `https://api.openai.com/v1` |
| DeepSeek | `openai_compatible` | `deepseek-chat` | `https://api.deepseek.com` |
| GLM (智谱) | `openai_compatible` | `glm-5.1` | `https://open.bigmodel.cn/api/paas/v4` |
| Qwen (通义千问) | `openai_compatible` | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

## REPL 命令

输入 `/` 后按 Tab 可自动补全命令。

| 命令 | 说明 |
|------|------|
| `/add <name> <url> [显示名] [api_key]` | 动态添加业务 |
| `/remove <name>` | 移除业务 |
| `/list` | 列出已注册业务 |
| `/memory` | 查看错误记忆（按业务分组） |
| `/clear [business]` | 清空错误记忆 |
| `/new` | 新对话（清空上下文） |
| `/pin <message>` | 置顶上下文（压缩历史后保留） |
| `/field <table>.<col> <desc>` | 添加字段知识 |
| `/field_rm <table>.<col>` | 删除字段知识 |
| `/fields` | 列出所有字段知识 |
| `exit` / `quit` / `q` | 退出 |

## SQL 安全管道

每条 SQL 经过以下检查链：

```
注释剥离 → 多语句拒绝 → SELECT-only 校验 → 禁止关键字检查
→ 禁止锁子句（FOR UPDATE / LOCK IN SHARE MODE）→ 表白名单 → LIMIT 强制
```

- 只允许 SELECT 语句，拒绝所有写操作
- 禁止 `FOR UPDATE`、`LOCK IN SHARE MODE`、`FOR SHARE`，确保快照读
- 白名单外的表直接拒绝
- 无 LIMIT 自动追加 `LIMIT {max_rows}`，超限自动降低
- 数据库连接使用 `readonly_user` + `autocommit=True`

## 性能风险检测

Agent 首次查询时从 MCP Server 获取白名单表的索引信息，后续每条 SQL 执行前自动分析：

| 风险等级 | 触发条件 |
|---------|---------|
| high | WHERE 列不在任何索引中；无 WHERE 且无 LIMIT |
| medium | WHERE 列只命中索引非前缀列；SELECT *；LIKE '%...'；派生表 |

检测到风险时打印提示，等待用户确认后才执行。

## 字段知识持久化

LLM 回复中发现的枚举/状态字段含义会自动提取并持久化到 `field_knowledge.json`，下次查询时注入 system prompt，避免重复调用 `get_table_schema` 探索字段含义。

提取方式（按优先级）：

1. **结构化声明**（推荐）：LLM 在回复末尾输出 HTML 注释，格式固定、解析可靠
   ```
   <!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"1=自研,2=阿里云,5=火山"}] -->
   ```
2. **回退正则匹配**：从自由文本中匹配 `来源(origin): 5 = 火山` 等模式

也支持手动管理：`/field tb_voice.origin 1=自研,2=阿里云`。

## 错误记忆

Agent 自动记录查询过程中的错误（SQL 被拒绝、表不在白名单、用户纠正等），持久化到 `error_memory.json`。下次查询时自动注入 system prompt，避免重复犯错。

支持用户反馈学习：当用户输入短文本且包含纠正关键词（"不对"、"应该"等），Agent 用 LLM 提取经验教训并记录。

## 项目结构

```
src/
├── main.py              # CLI 入口，REPL 循环，Tab 补全，反馈检测
├── agent.py             # QueryAgent 核心：对话循环、工具路由、字段知识提取
├── llm_provider.py      # LLM 抽象层（Anthropic / OpenAI Compatible）
├── business_registry.py # 多业务 MCP 连接管理（SSE 懒连接 + 会话缓存）
├── prompts.py           # System Prompt 构建（单业务/多业务 + 字段知识注入）
├── config.py            # YAML 配置加载 + 环境变量替换
├── db_pool.py           # aiomysql 连接池管理
├── sql_validator.py     # SQL 安全验证器
├── sql_risk_checker.py  # 索引级性能风险分析
├── error_memory.py      # 错误记忆持久化（字符预算 + 原子写入）
├── field_knowledge.py   # 字段知识持久化 + 表结构缓存
└── query_mcp_server.py  # MCP Server（FastMCP + Bearer Auth + 查询超时）
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 运行单个测试
pytest tests/test_sql_validator.py -v
```

## 技术栈

- Python 3.10+
- [MCP Protocol](https://modelcontextprotocol.io/) — stdio / SSE 传输
- Anthropic SDK + OpenAI SDK — 多 LLM Provider
- aiomysql — 异步 MySQL 连接池
- sqlparse — SQL 解析
- FastMCP — MCP Server 框架

## License

MIT
