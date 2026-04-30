# Query Agent

通用业务数据查询 Agent — 用户输入自然语言，Agent 调用 LLM 生成 SQL，通过 MCP Server 执行只读查询并返回结果。

## 特性

- **自然语言转 SQL** — 用户无需写 SQL，用中文描述即可查询
- **多业务路由** — 单个 Agent 实例可同时对接多个业务系统，自动识别并路由
- **双部署模式** — 本地 stdio 模式（开发调试）和远程 SSE 模式（生产部署）
- **多 LLM 支持** — Anthropic Claude / OpenAI 兼容接口（DeepSeek、GLM、Qwen 等）
- **智能学习** — 错误记忆、字段知识自动提取、用户偏好规则持久化
- **SQL 安全保障** — 只读查询、白名单表、禁止锁子句、LIMIT 强制
- **性能风险检测** — 基于索引信息分析 SQL 执行风险，高危查询需用户确认
- **多轮对话** — 支持上下文引用（"再查一下"、"换成生产环境"等）

## 架构

```text
┌──────────────┐
│   CLI / REPL │  src/main.py
└──────┬───────┘
       │
┌──────▼───────┐
│  QueryAgent  │  src/agent.py
│              │
│  ┌───────────────────────────────────┐
│  │ ConversationState                 │
│  │ BusinessSelectionService          │
│  │ PromptService                     │
│  │ ToolExecutionService              │
│  │ KnowledgeStore                    │
│  │ QueryRuleExecutor                 │
│  └───────────────────────────────────┘
└──────┬───────┘
       │
   ┌───┴────────────────────┐
   │                        │
┌──▼──────────┐  ┌─────────▼────────┐
│ stdio 模式   │  │ SSE 模式          │
│ 本地子进程    │  │ BusinessRegistry  │
│             │  │ → MCP Server A    │
│             │  │ → MCP Server B    │
└──┬──────────┘  └─────────┬────────┘
   │                        │
┌──▼────────────────────────▼──┐
│     MCP Server               │  src/query_mcp_server.py
│                              │
│  execute_readonly_sql        │
│  get_cluster_list            │
│  get_table_schema            │
│  get_table_indexes           │
│  get_business_knowledge      │
│                              │
│  SQLValidator → DB Pool      │
└──────────────────────────────┘
```

## 安装

```bash
pip install -e ".[dev]"
```

安装后可直接使用入口命令：

```bash
query-agent --config config.yaml
```

## 快速开始

### 本地 stdio 模式

适合开发调试和单业务场景。Agent 自动拉起本地 MCP Server 子进程。

```bash
# 设置环境变量
export ANTHROPIC_API_KEY=sk-xxx
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx

# 启动
python -m src.main --config config.yaml
```

### 远程 SSE 模式

适合生产部署和多业务场景。MCP Server 独立部署，Agent 通过 SSE 连接。

**1. 启动 MCP Server**（在数据库所在机器上）：

```bash
export MCP_API_KEY=your-secret
export DB_TEST_HOST=127.0.0.1
export DB_TEST_PASSWORD=xxx

MCP_HOST=0.0.0.0 MCP_PORT=8765 python -m src.query_mcp_server --transport sse --config config-server.yaml
```

**2. 启动 Agent**：

```bash
export LLM_API_KEY=xxx

python -m src.main --config config-local.yaml
```

## 配置

配置文件为 YAML，支持 `${ENV_VAR}` 环境变量替换。

### 单业务本地模式

完整配置示例（Agent + MCP Server 合一）：

```yaml
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

sql_security:
  max_rows: 100
  query_timeout: 30
  allowed_tables:
    - "tb_user"
    - "tb_order"
    - "tb_product"

business_knowledge:
  description: "电商系统"
  term_mappings:
    "用户": "tb_user 表"
    "订单": "tb_order 表"
    "商品": "tb_product 表"
  table_relationships:
    - "tb_order.user_id → tb_user.id（订单所属用户）"
    - "tb_order.product_id → tb_product.id（订单关联商品）"
  status_codes:
    - "tb_order.status: 1=待支付, 2=已支付, 3=已发货, 4=已完成"
    - "tb_user.status: 1=活跃, 0=禁用"
  custom_rules:
    - "查询订单时默认按创建时间倒序排列"

agent:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  default_cluster: "test"

storage:
  namespace: "local-dev"
```

### 多业务远程模式

Agent 端配置，每个业务指向独立的 MCP Server：

```yaml
agent:
  provider: "openai_compatible"
  model: "glm-5.1"
  max_tokens: 4096
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL}"
  default_cluster: "test"

businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://mcp-host-a:8765/sse"
    api_key: "${MCP_API_KEY}"
  order:
    display_name: "订单"
    mcp_server_url: "http://mcp-host-b:8765/sse"
    api_key: "${MCP_API_KEY}"

storage:
  namespace: "prod-agent"
```

### MCP Server 独立配置

部署在数据库所在机器上的配置文件：

```yaml
clusters:
  test:
    description: "测试环境"
    host: "${DB_TEST_HOST}"
    port: 3306
    database: "${DB_TEST_DB}"
    user: "${DB_TEST_USER}"
    password: "${DB_TEST_PASSWORD}"

sql_security:
  max_rows: 100
  query_timeout: 30
  allowed_tables:
    - "tb_scene"
    - "tb_model"

business_knowledge:
  description: "数字人平台"
  term_mappings:
    "模型": "tb_model 表"
    "形象/数字人": "tb_scene 表"
  table_relationships:
    - "tb_scene.model_id → tb_model.id（形象属于某个模型）"
  status_codes:
    - "tb_scene.status: 1=训练中, 2=训练成功, 3=训练失败"
  custom_rules: []

auth:
  api_key: "${MCP_API_KEY}"
```

### 兼容说明

如果只配置 `agent.mcp_server_url` 而不配置 `businesses`，系统会自动创建名为 `default` 的业务，兼容旧配置。

### Provider 配置

| Provider | `agent.provider` 值 | 说明 |
|---|---|---|
| Anthropic Claude | `anthropic` | 直接使用 Anthropic SDK |
| DeepSeek / GLM / Qwen / GPT | `openai_compatible` | 使用 OpenAI 兼容接口，需配 `base_url` |

### 业务领域知识生成

可使用 AI 辅助生成 `business_knowledge` 配置。将数据库 DDL（`SHOW CREATE TABLE` 输出）提供给 AI，结合以下提示词：

```
你是一个数据库业务领域知识提取专家。请根据提供的数据库表结构，生成 YAML 格式的业务领域知识配置。

输出格式：
business_knowledge:
  description: "一句话业务描述"
  term_mappings:        # 用户常用术语 → 表名映射
    "术语": "表名 表"
  table_relationships:  # 外键关系
    - "表A.列 → 表B.列（关系说明）"
  status_codes:         # 状态/枚举字段
    - "表.列: 0=含义, 1=含义"
  custom_rules: []      # 业务特有查询规则
```

## REPL 命令

输入 `/` 后按 Tab 可补全命令。

| 命令 | 说明 |
|---|---|
| `/add <name> <url> [display] [api_key]` | 动态添加业务 |
| `/remove <name>` | 移除业务 |
| `/list` | 列出已注册业务 |
| `/new` | 开始新对话 |
| `/pin <message>` | 置顶重要上下文（压缩时保留） |
| `/field <table>.<col> <desc>` | 添加字段知识 |
| `/field_rm <table>.<col>` | 删除字段知识 |
| `/fields` | 列出所有字段知识 |
| `/remember <rule>` | 保存默认查询规则 |
| `/rules` | 查看默认规则 |
| `/rules_clear [business]` | 清空默认规则 |
| `/memory` | 查看错误记忆 |
| `/clear [business]` | 清空错误记忆 |
| `exit` / `quit` / `q` | 退出 |

## 查询执行流程

一次查询的完整链路：

```text
用户输入自然语言
  │
  ▼
业务选择 (single / heuristic / llm / fallback_all)
  │
  ▼
构建 System Prompt
  ├── 基础查询规则
  ├── 业务领域知识（术语映射、表关系、状态码）
  ├── 字段知识（自动提取的枚举含义）
  ├── 错误记忆（历史失败经验）
  └── 默认规则（用户偏好）
  │
  ▼
LLM 生成工具调用（get_table_schema / execute_readonly_sql）
  │
  ▼
执行前检查
  ├── 打印 SQL
  ├── 性能风险检测（索引分析 / LLM risk_note）
  └── 高风险需用户确认
  │
  ▼
调用 MCP 工具执行查询
  │
  ▼
返回结果 → 自动提取字段知识 → 更新对话上下文
```

### 业务选择策略

| 策略 | 说明 |
|---|---|
| `single` | 只有一个业务，直接使用 |
| `heuristic` | 用户输入命中业务名或显示名 |
| `llm` | 通过 LLM 判断最相关业务 |
| `fallback_all` | 无法唯一识别，合并所有业务的工具 |

## SQL 安全

### 安全管道

每条 SQL 都经过多重检查：

```text
注释剥离 → 多语句拒绝 → 只允许 SELECT → 禁止危险关键字
→ 禁止锁子句 → 白名单表校验 → LIMIT 强制
```

关键规则：
- 只允许 `SELECT` 查询，禁止 `INSERT`/`UPDATE`/`DELETE`/`DDL`
- 禁止 `FOR UPDATE` / `LOCK IN SHARE MODE` / `FOR SHARE`
- 表必须在 `sql_security.allowed_tables` 白名单中
- 无 `LIMIT` 自动补齐

### MCP Server 鉴权

SSE 模式支持 Bearer Token 鉴权：
- 优先读取配置文件 `auth.api_key`
- 其次读取环境变量 `MCP_API_KEY`
- 未配置 API Key 时不鉴权（仅适合可信内网）

### 性能风险检测

执行 SQL 前会分析性能风险：

| 风险级别 | 触发条件 | 处理方式 |
|---|---|---|
| **high** | WHERE 列无索引覆盖（全表扫描）| 需用户确认 |
| **medium** | SELECT *、LIKE 前导通配符 | 需用户确认 |
| **无风险** | 有索引驱动列（如主键查询）| 直接执行 |

双轨检测机制：
1. **LLM risk_note**（优先）— LLM 在调用工具时通过 `risk_note` 参数声明风险分析
2. **静态索引分析**（兜底）— 基于 `get_table_indexes` 获取的索引信息做静态检查

当 WHERE 条件有索引驱动列时（如 `WHERE app_id = 1 AND deleted_at IS NULL`，`app_id` 有索引），视为无风险，不提示。

## 学习与记忆

### 默认规则

用户偏好持久化，自动应用到后续查询。

**创建方式**：
- `/remember 默认只查可用数据`
- 对话中包含"记住/默认/以后都/后续查询/优先过滤"等关键词的反馈

**持久化位置**：`.query-agent/<namespace>/preference_rules.json`

### 错误记忆

Agent 自动记录失败经验（SQL 被拒绝、表不在白名单等），注入后续 prompt 避免重犯。

- 查看：`/memory`
- 清空：`/clear [business]`

**持久化位置**：`.query-agent/<namespace>/error_memory.json`

### 字段知识

Agent 从回答中自动提取字段枚举含义，后续查询可直接复用，无需重复查表。

提取依赖 LLM 回复中的隐藏标记：

```html
<!-- FIELD_KNOWLEDGE: [{"table":"tb_voice","field":"origin","values":"1=自研,5=火山"}] -->
```

也支持从 Markdown 表格回退提取。

- 手工添加：`/field tb_voice.origin 1=自研,2=阿里云,5=火山引擎`
- 查看：`/fields`
- 删除：`/field_rm tb_voice.origin`

**持久化位置**：`.query-agent/<namespace>/field_knowledge.json`

### 持久化目录

所有学习数据存放在 `.query-agent/<namespace>/`，其中 `<namespace>`：
- 优先使用 `storage.namespace` 配置
- 否则根据配置文件路径自动推导

不同配置文件使用不同 namespace，互不干扰。

## 代码结构

```text
src/
  main.py                       # CLI 入口，异步 REPL 循环
  agent.py                      # QueryAgent 核心，对话循环 + 业务路由
  llm_provider.py               # LLM Provider 抽象（Anthropic / OpenAI Compatible）
  prompts.py                    # System Prompt 构建
  config.py                     # YAML 配置加载，环境变量替换
  business_registry.py          # 多业务 MCP Server 连接管理（SSE）
  business_selection_service.py # 业务选择（启发式 + LLM 兜底）
  conversation_state.py         # 对话历史管理 + 压缩
  prompt_service.py             # Prompt 组装与缓存
  tool_execution_service.py     # 工具路由、风险检测、索引加载
  knowledge_store.py            # 知识聚合（错误记忆 + 字段知识）
  field_knowledge.py            # 字段知识持久化
  error_memory.py               # 错误记忆持久化
  preference_rules.py           # 默认规则持久化
  query_rule_executor.py        # 规则命中与参数改写
  query_plan.py                 # 查询计划数据结构
  sql_validator.py              # SQL 安全验证
  sql_risk_checker.py           # SQL 性能风险分析（索引级）
  query_mcp_server.py           # MCP Server（stdio / SSE）
  db_pool.py                    # MySQL 连接池管理
```

## 测试

```bash
# 全量测试
pytest

# 单文件
pytest tests/test_agent.py -v

# 单用例
pytest tests/test_sql_validator.py::TestValidate::test_for_update_rejected
```

测试使用 pytest + pytest-asyncio，所有数据库交互已 mock，无需真实数据库。
