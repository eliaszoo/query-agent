# Query Agent

通用业务数据查询 Agent — 基于 LLM + MCP Server + MySQL 的智能查询系统。

用户输入自然语言，Agent 自动翻译为 SQL，执行只读快照查询并返回结果。支持多业务动态路由，单实例可同时查询多个业务域。

## 特性

- **自然语言转 SQL** — 用户用中文提问，LLM 自动生成并执行 SQL 查询
- **多业务动态路由** — 单 Agent 实例连接多个 MCP Server，自动路由到目标业务
- **多 LLM 支持** — Anthropic Claude / OpenAI GPT / DeepSeek / GLM / Qwen 等任意 OpenAI 兼容模型
- **SQL 安全管道** — 白名单表、SELECT-only、禁止锁子句、强制 LIMIT、注释剥离、多语句拒绝
- **索引级性能风险检测** — 基于实际索引信息分析 SQL 是否能命中索引，执行前打印 SQL 并提示风险
- **业务领域知识注入** — 术语映射、表关系、状态码等通过 YAML 或 MCP 工具动态注入
- **错误记忆系统** — 自动从失败中学习，将教训注入后续对话的 system prompt
- **多集群支持** — 测试/生产环境隔离，默认查询测试集群
- **两种部署模式** — 本地 stdio 一体模式 / 远程 SSE 分离模式

## 架构

```
┌─────────────────────────────────────────────────┐
│                   Agent (客户端)                  │
│                                                   │
│  REPL ←→ LLM ←→ Tool Router ←→ BusinessRegistry  │
│                        │              │           │
│                   Risk Checker    Error Memory     │
└────────────────────────┼──────────────┼───────────┘
                         │              │
                    stdio/SSE      SSE (远程)
                         │              │
┌────────────────────────┼──────────────┼───────────┐
│                MCP Server (服务端)      │           │
│                                        │           │
│  execute_readonly_sql  get_table_schema│           │
│  get_cluster_list      get_table_indexes│          │
│  get_business_knowledge                │           │
│                    │                               │
│            SQL Validator                │           │
│            Connection Pool              │           │
│                    │                               │
│               MySQL (只读)                          │
└─────────────────────────────────────────────────────┘
```

**部署模式：**

| 模式 | Agent | MCP Server | 传输 | 适用场景 |
|------|-------|------------|------|---------|
| stdio 一体 | 本地 | 本地（同进程） | stdin/stdout | 开发调试、单业务 |
| SSE 分离 | 本地/远程 | 远程（靠近数据库） | HTTP/SSE | 生产部署、多业务 |

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 配置

复制并编辑配置文件，填入数据库连接信息和 LLM API Key：

```bash
cp config.yaml config.yaml.local
# 编辑 config.yaml.local，设置环境变量或直接填入值
```

配置中使用 `${ENV_VAR}` 引用环境变量：

```yaml
clusters:
  test:
    host: "${DB_TEST_HOST}"
    password: "${DB_TEST_PASSWORD}"
```

### 启动

**本地 stdio 模式（Agent + MCP Server 一体）：**

```bash
export ANTHROPIC_API_KEY=sk-xxx
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx

python -m src.main --config config.yaml
```

**远程 SSE 模式：**

1. 在数据库所在机器启动 MCP Server：

```bash
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx

MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m src.query_mcp_server --transport sse --config config-server.yaml
```

2. 在本地启动 Agent：

```bash
export ANTHROPIC_API_KEY=sk-xxx

python -m src.main --config config-local.yaml
```

### 使用

```
🤖 查询 Agent (数字人平台)
====================
输入自然语言查询数字人平台数据，例如：
  - 帮我查一下测试环境的数据
  - 查询生产环境的记录数

🧑 请输入查询: 查一下训练成功的形象有多少个

📝 即将执行 SQL (集群: test):
   SELECT COUNT(*) FROM tb_scene WHERE status = 2

🤖 训练成功的形象共有 1,234 个。

📊 耗时 2.3s | Token: 856↑ 312↓ | 工具调用: 2次
```

**性能风险提示示例：**

```
📝 即将执行 SQL (集群: test):
   SELECT * FROM tb_scene WHERE name LIKE '%测试%'

⚠️  性能风险 [high]:
   - 表 tb_scene: name 不在任何索引中，可能无法高效使用索引
   - LIKE 使用前导通配符，无法使用索引: %测试%
   是否继续执行？(y/N): n

🤖 用户取消了查询执行
```

## 配置参考

### 全量配置（config.yaml）

Agent + MCP Server 一体模式，适用于本地开发：

```yaml
# 数据库集群
clusters:
  test:
    description: "测试环境"
    host: "${DB_TEST_HOST}"
    port: 3306
    database: "mydb"
    user: "readonly_user"
    password: "${DB_TEST_PASSWORD}"
  production:
    description: "生产环境"
    host: "${DB_PROD_HOST}"
    port: 3306
    database: "mydb"
    user: "readonly_user"
    password: "${DB_PROD_PASSWORD}"

# SQL 安全策略
sql_security:
  max_rows: 100
  query_timeout: 30
  allowed_tables:
    - "tb_user"
    - "tb_order"

# 业务领域知识
business_knowledge:
  description: "业务平台"
  term_mappings:
    "用户": "tb_user 表"
    "订单": "tb_order 表"
  table_relationships:
    - "tb_order.user_id → tb_user.id"
  status_codes:
    - "tb_order.status: 1=待支付, 2=已支付, 3=已取消"
  custom_rules: []

# Agent 配置
agent:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  default_cluster: "test"
```

### Agent 本地配置（config-local.yaml）

连接远程 MCP Server，无需数据库配置：

```yaml
agent:
  default_cluster: "test"

businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://mcp-host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other-host:8765/sse"
```

### MCP Server 配置（config-server.yaml）

部署在数据库所在机器，包含数据库连接和安全策略：

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
  table_relationships: []
  status_codes: []
  custom_rules: []
```

### LLM Provider 配置

在 `agent` 节点下配置 LLM：

```yaml
# Anthropic Claude
agent:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  # api_key 留空则从 ANTHROPIC_API_KEY 环境变量读取

# OpenAI GPT
agent:
  provider: "openai_compatible"
  model: "gpt-4o"
  max_tokens: 4096
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"

# DeepSeek
agent:
  provider: "openai_compatible"
  model: "deepseek-chat"
  max_tokens: 4096
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com"

# GLM (智谱)
agent:
  provider: "openai_compatible"
  model: "glm-4-plus"
  max_tokens: 4096
  api_key: "${ZHIPU_API_KEY}"
  base_url: "https://open.bigmodel.cn/api/paas/v4"

# Qwen (通义千问)
agent:
  provider: "openai_compatible"
  model: "qwen-plus"
  max_tokens: 4096
  api_key: "${DASHSCOPE_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/add <name> <sse_url> [显示名]` | 动态添加业务 |
| `/remove <name>` | 移除业务 |
| `/list` | 列出所有已注册业务 |
| `/memory` | 查看 Agent 已学到的错误经验 |
| `/clear` | 清空错误记忆 |
| `/new` | 开始新对话（清空上下文） |
| `exit` / `quit` / `q` | 退出 |

## SQL 安全

每条 SQL 查询经过以下安全管道：

```
注释剥离 → 多语句检查 → SELECT-only → 禁止锁子句 → 表白名单 → LIMIT 强制
```

- **禁止锁子句**：拒绝 `FOR UPDATE`、`LOCK IN SHARE MODE`、`FOR SHARE`
- **表白名单**：只允许查询 `sql_security.allowed_tables` 中配置的表
- **LIMIT 强制**：无 LIMIT 的查询自动追加 `LIMIT max_rows`
- **只读连接**：使用 `readonly_user` + `autocommit=True`，确保不会写入或持有锁

## 性能风险检测

Agent 在执行 SQL 前自动检测性能风险：

1. 首次查询时从 MCP Server 获取所有白名单表的索引信息（`SHOW INDEX`）
2. 解析 SQL 提取表名和 WHERE 条件列
3. 逐表检查索引覆盖情况
4. 如有风险，打印提示并等待用户确认

**风险等级：**

| 等级 | 场景 |
|------|------|
| **high** | WHERE 列不在任何索引中 / 无 WHERE 且无 LIMIT（全表扫描） |
| **medium** | WHERE 列只命中索引非前缀列 / SELECT * / LIKE '%...' 前导通配符 / 派生表 |

## MCP 工具

MCP Server 暴露以下工具：

| 工具 | 说明 |
|------|------|
| `execute_readonly_sql` | 执行只读 SQL 查询 |
| `get_cluster_list` | 获取可用集群列表 |
| `get_table_schema` | 获取表结构（DESCRIBE） |
| `get_table_indexes` | 获取表索引信息（SHOW INDEX） |
| `get_business_knowledge` | 获取业务领域知识 |

## 项目结构

```
query-agent/
├── src/
│   ├── main.py              # CLI 入口，REPL 循环
│   ├── agent.py             # QueryAgent 核心逻辑
│   ├── business_registry.py # 多业务 MCP 连接管理
│   ├── config.py            # YAML 配置加载
│   ├── db_pool.py           # MySQL 连接池管理
│   ├── llm_provider.py      # LLM Provider 抽象层
│   ├── prompts.py           # System Prompt 构建
│   ├── sql_validator.py     # SQL 安全验证器
│   ├── sql_risk_checker.py  # SQL 性能风险分析器
│   ├── error_memory.py      # 错误记忆持久化
│   └── query_mcp_server.py  # MCP Server 实现
├── tests/
│   ├── test_agent.py
│   ├── test_business_registry.py
│   ├── test_config.py
│   ├── test_db_pool.py
│   ├── test_main.py
│   ├── test_prompts.py
│   ├── test_sql_validator.py
│   └── test_sql_risk_checker.py
├── config.yaml              # 全量配置（stdio 模式）
├── config-local.yaml        # Agent 本地配置（SSE 模式）
├── config-server.yaml       # MCP Server 配置
├── pyproject.toml
└── requirements.txt
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_sql_validator.py

# 运行单个测试
pytest tests/test_sql_validator.py::TestValidate::test_for_update_rejected
```

## 技术栈

- **Python 3.10+**
- **MCP Protocol** — [Model Context Protocol](https://modelcontextprotocol.io/)，stdio/SSE 传输
- **Anthropic SDK** / **OpenAI SDK** — 多 LLM Provider 支持
- **aiomysql** — 异步 MySQL 连接池
- **sqlparse** — SQL 解析与验证
- **FastMCP** — MCP Server 框架

## License

MIT
