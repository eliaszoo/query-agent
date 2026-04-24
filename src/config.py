"""配置加载模块。

解析 YAML 配置文件，支持环境变量替换和配置完整性验证。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


class ConfigError(Exception):
    """配置错误异常。"""


@dataclass
class ClusterConfig:
    """数据库集群配置。"""

    name: str
    description: str
    host: str
    port: int
    database: str
    user: str
    password: str
    charset: str = "utf8mb4"
    max_connections: int = 5
    connect_timeout: int = 10


@dataclass
class SQLSecurityConfig:
    """SQL 安全配置。"""

    max_rows: int = 100
    query_timeout: int = 30
    allowed_tables: list[str] = field(default_factory=list)


@dataclass
class BusinessKnowledge:
    """业务领域知识，注入到 system prompt。"""

    description: str = ""  # 业务描述，如"数字人平台"
    term_mappings: dict[str, str] = field(default_factory=dict)  # 术语→表映射
    table_relationships: list[str] = field(default_factory=list)  # 表关系说明
    status_codes: list[str] = field(default_factory=list)  # 状态码说明
    custom_rules: list[str] = field(default_factory=list)  # 额外查询规则


@dataclass
class AgentConfig:
    """Agent 配置。"""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    default_cluster: str = "test"
    mcp_server_url: str = ""  # 远程 MCP Server SSE URL，为空则使用本地 stdio
    provider: str = "anthropic"  # "anthropic" 或 "openai_compatible"
    api_key: str = ""  # API 密钥，为空则从环境变量读取
    base_url: str = ""  # API 地址，仅 openai_compatible 需要


@dataclass
class BusinessEntryConfig:
    """单个业务的配置（用于多业务模式）。"""

    name: str  # 业务标识，如 "digitalhuman"
    display_name: str  # 显示名，如 "数字人"
    mcp_server_url: str  # MCP Server SSE URL


@dataclass
class AppConfig:
    """应用顶层配置。"""

    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    sql_security: SQLSecurityConfig = field(default_factory=SQLSecurityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    business_knowledge: BusinessKnowledge = field(default_factory=BusinessKnowledge)
    businesses: dict[str, BusinessEntryConfig] = field(default_factory=dict)


# 环境变量占位符正则：匹配 ${VAR_NAME}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# 集群必填字段
_REQUIRED_CLUSTER_FIELDS = ("host", "port", "database", "user", "password")


def _substitute_env_vars(value: Any) -> Any:
    """递归替换配置值中的 ${VAR_NAME} 环境变量占位符。

    Args:
        value: 配置值，可以是 str、dict、list 或其他类型。

    Returns:
        替换后的值。
    """
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                return match.group(0)  # 保留原始占位符
            return env_value

        return _ENV_VAR_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    return value


def _validate_config(raw: dict) -> None:
    """验证配置完整性。

    Args:
        raw: 解析后的原始配置字典。

    Raises:
        ConfigError: 配置缺失或无效时抛出。
    """
    if not isinstance(raw, dict):
        raise ConfigError("配置文件格式无效，应为 YAML 字典")

    # 验证 clusters（当使用 businesses 多业务模式或有 mcp_server_url 时，clusters 可以为空）
    clusters = raw.get("clusters")
    has_businesses = raw.get("businesses") and isinstance(raw.get("businesses"), dict)
    has_mcp_url = isinstance(raw.get("agent"), dict) and raw.get("agent", {}).get("mcp_server_url")
    if not clusters or not isinstance(clusters, dict):
        if not has_businesses and not has_mcp_url:
            raise ConfigError("配置缺少 'clusters' 或 clusters 为空")
    else:
        for name, cluster in clusters.items():
            if not isinstance(cluster, dict):
                raise ConfigError(f"集群 '{name}' 配置格式无效")
            for req_field in _REQUIRED_CLUSTER_FIELDS:
                if req_field not in cluster or cluster[req_field] is None:
                    raise ConfigError(
                        f"集群 '{name}' 缺少必填字段: '{req_field}'"
                    )
            # 验证环境变量占位符是否已被替换
            for str_field in ("host", "password"):
                val = cluster.get(str_field, "")
                if isinstance(val, str) and _ENV_VAR_PATTERN.search(val):
                    raise ConfigError(
                        f"集群 '{name}' 的 '{str_field}' 包含未解析的环境变量: {val}"
                    )

    # 验证 sql_security（可选但如果存在需要合法）
    sql_sec = raw.get("sql_security")
    if sql_sec is not None and not isinstance(sql_sec, dict):
        raise ConfigError("'sql_security' 配置格式无效")

    # 验证 agent（可选但如果存在需要合法）
    agent = raw.get("agent")
    if agent is not None and not isinstance(agent, dict):
        raise ConfigError("'agent' 配置格式无效")

    # 验证 business_knowledge（可选但如果存在需要合法）
    bk = raw.get("business_knowledge")
    if bk is not None and not isinstance(bk, dict):
        raise ConfigError("'business_knowledge' 配置格式无效")

    # 验证 businesses（可选但如果存在需要合法）
    businesses = raw.get("businesses")
    if businesses is not None and not isinstance(businesses, dict):
        raise ConfigError("'businesses' 配置格式无效")


def _build_app_config(raw: dict) -> AppConfig:
    """从原始字典构建类型安全的 AppConfig。

    Args:
        raw: 经过验证的配置字典。

    Returns:
        AppConfig 实例。
    """
    # 构建集群配置（当使用 businesses 多业务模式时，clusters 可以为空）
    clusters: dict[str, ClusterConfig] = {}
    for name, c in raw.get("clusters", {}).items():
        clusters[name] = ClusterConfig(
            name=name,
            description=c.get("description", ""),
            host=c["host"],
            port=int(c["port"]),
            database=c["database"],
            user=c["user"],
            password=c["password"],
            charset=c.get("charset", "utf8mb4"),
            max_connections=int(c.get("max_connections", 5)),
            connect_timeout=int(c.get("connect_timeout", 10)),
        )

    # 构建 SQL 安全配置
    sql_sec_raw = raw.get("sql_security", {})
    sql_security = SQLSecurityConfig(
        max_rows=int(sql_sec_raw.get("max_rows", 100)),
        query_timeout=int(sql_sec_raw.get("query_timeout", 30)),
        allowed_tables=sql_sec_raw.get("allowed_tables", []),
    )

    # 构建 Agent 配置
    agent_raw = raw.get("agent", {})
    agent = AgentConfig(
        model=agent_raw.get("model", "claude-sonnet-4-20250514"),
        max_tokens=int(agent_raw.get("max_tokens", 4096)),
        default_cluster=agent_raw.get("default_cluster", "test"),
        mcp_server_url=agent_raw.get("mcp_server_url", ""),
        provider=agent_raw.get("provider", "anthropic"),
        api_key=agent_raw.get("api_key", ""),
        base_url=agent_raw.get("base_url", ""),
    )

    # 构建业务知识配置
    bk_raw = raw.get("business_knowledge", {})
    business_knowledge = BusinessKnowledge(
        description=bk_raw.get("description", ""),
        term_mappings=bk_raw.get("term_mappings", {}),
        table_relationships=bk_raw.get("table_relationships", []),
        status_codes=bk_raw.get("status_codes", []),
        custom_rules=bk_raw.get("custom_rules", []),
    )

    # 构建多业务配置
    businesses: dict[str, BusinessEntryConfig] = {}
    businesses_raw = raw.get("businesses", {})
    for biz_name, biz_cfg in businesses_raw.items():
        businesses[biz_name] = BusinessEntryConfig(
            name=biz_name,
            display_name=biz_cfg.get("display_name", biz_name),
            mcp_server_url=biz_cfg.get("mcp_server_url", ""),
        )

    # 向后兼容：如果配置了 agent.mcp_server_url 但没有 businesses，
    # 自动转为名为 "default" 的单业务条目
    if not businesses and agent.mcp_server_url:
        businesses["default"] = BusinessEntryConfig(
            name="default",
            display_name=business_knowledge.description or "默认业务",
            mcp_server_url=agent.mcp_server_url,
        )

    return AppConfig(
        clusters=clusters,
        sql_security=sql_security,
        agent=agent,
        business_knowledge=business_knowledge,
        businesses=businesses,
    )


def load_config(path: str) -> AppConfig:
    """加载并解析 YAML 配置文件。

    1. 读取 YAML 文件
    2. 递归替换 ${VAR_NAME} 环境变量占位符
    3. 验证配置完整性（必填字段检查）
    4. 返回类型安全的 AppConfig 实例

    Args:
        path: YAML 配置文件路径。

    Returns:
        AppConfig 实例。

    Raises:
        ConfigError: 配置文件不存在、格式错误或缺少必填字段。
    """
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析失败: {e}") from e

    if raw is None:
        raise ConfigError("配置文件为空")

    # 环境变量替换
    raw = _substitute_env_vars(raw)

    # 验证配置
    _validate_config(raw)

    # 构建类型安全配置
    return _build_app_config(raw)
