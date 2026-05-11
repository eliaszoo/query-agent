"""配置加载模块。

解析 YAML 配置文件，支持环境变量替换和配置完整性验证。
"""

import hashlib
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
    business_allowed_tables: dict[str, list[str]] = field(default_factory=dict)


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
class MCPServerEndpoint:
    """单个 MCP Server 端点配置。"""

    name: str = ""  # server 标识，如 "shanghai"（顶层 mcp_servers 引用用）
    url: str = ""  # MCP Server SSE URL
    api_key: str = ""  # 鉴权密钥，为空则不传鉴权 Header


@dataclass
class BusinessEntryConfig:
    """单个业务的配置（用于多业务模式）。"""

    name: str  # 业务标识，如 "digitalhuman"
    display_name: str  # 显示名，如 "数字人"
    servers: list[str] = field(default_factory=list)  # server 名称引用列表（对应 mcp_servers 中的 name）
    # 向后兼容字段
    mcp_server_url: str = ""  # 单 MCP Server SSE URL（旧格式）
    api_key: str = ""  # 单 MCP Server 鉴权密钥（旧格式）


@dataclass
class AuthConfig:
    """MCP Server 鉴权配置。"""

    api_key: str = ""  # 鉴权密钥，为空则不启用鉴权


@dataclass
class StorageConfig:
    """本地持久化存储配置。"""

    namespace: str = ""  # 持久化命名空间；为空则由 config path 自动推导


@dataclass
class FeishuConfig:
    """飞书机器人配置。"""

    app_id: str = ""  # 飞书应用 App ID
    app_secret: str = ""  # 飞书应用 App Secret
    encrypt_key: str = ""  # 事件加密 key（可选）
    verification_token: str = ""  # 事件验证 token
    host: str = "0.0.0.0"  # 监听地址
    port: int = 8080  # 监听端口
    session_ttl: int = 1800  # 会话超时（秒），默认 30 分钟


@dataclass
class AppConfig:
    """应用顶层配置。"""

    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    business_clusters: dict[str, dict[str, ClusterConfig]] = field(default_factory=dict)
    mcp_servers: list[MCPServerEndpoint] = field(default_factory=list)  # 顶层 MCP Server 列表
    sql_security: SQLSecurityConfig = field(default_factory=SQLSecurityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    business_knowledge: BusinessKnowledge = field(default_factory=BusinessKnowledge)
    business_knowledges: dict[str, BusinessKnowledge] = field(default_factory=dict)
    businesses: dict[str, BusinessEntryConfig] = field(default_factory=dict)
    auth: AuthConfig = field(default_factory=AuthConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    feishu: FeishuConfig = field(default_factory=FeishuConfig)


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


def _validate_cluster_entry(name: str, cluster: dict) -> None:
    """验证单个集群配置项。"""
    for req_field in _REQUIRED_CLUSTER_FIELDS:
        if req_field not in cluster or cluster[req_field] is None:
            raise ConfigError(f"集群 '{name}' 缺少必填字段: '{req_field}'")
    for str_field in ("host", "password"):
        val = cluster.get(str_field, "")
        if isinstance(val, str) and _ENV_VAR_PATTERN.search(val):
            raise ConfigError(
                f"集群 '{name}' 的 '{str_field}' 包含未解析的环境变量: {val}"
            )


def _validate_clusters(clusters: dict) -> None:
    """验证 clusters 配置，支持扁平（单业务）和多业务嵌套两种格式。

    扁平格式：clusters = {cluster_name: {host, port, ...}}
    嵌套格式：clusters = {business_name: {cluster_name: {host, port, ...}}}

    通过检测第一个值是否包含 _REQUIRED_CLUSTER_FIELDS 来区分格式。
    """
    # 检测是否为嵌套格式（多业务）
    first_value = next(iter(clusters.values()), None)
    if isinstance(first_value, dict):
        # 如果第一个值包含集群必填字段 → 扁平格式
        if any(k in first_value for k in _REQUIRED_CLUSTER_FIELDS):
            for name, cluster in clusters.items():
                if not isinstance(cluster, dict):
                    raise ConfigError(f"集群 '{name}' 配置格式无效")
                _validate_cluster_entry(name, cluster)
        else:
            # 嵌套格式：business_name → {cluster_name → config}
            for biz_name, biz_clusters in clusters.items():
                if not isinstance(biz_clusters, dict):
                    raise ConfigError(f"业务 '{biz_name}' 集群配置格式无效")
                for cluster_name, cluster in biz_clusters.items():
                    if not isinstance(cluster, dict):
                        raise ConfigError(f"业务 '{biz_name}' 集群 '{cluster_name}' 配置格式无效")
                    _validate_cluster_entry(cluster_name, cluster)


def _validate_config(raw: dict) -> None:
    """验证配置完整性。

    Args:
        raw: 解析后的原始配置字典。

    Raises:
        ConfigError: 配置缺失或无效时抛出。
    """
    if not isinstance(raw, dict):
        raise ConfigError("配置文件格式无效，应为 YAML 字典")

    # 验证 clusters（支持扁平和多业务嵌套两种格式）
    clusters = raw.get("clusters")
    has_businesses = raw.get("businesses") and isinstance(raw.get("businesses"), dict)
    has_mcp_url = isinstance(raw.get("agent"), dict) and raw.get("agent", {}).get("mcp_server_url")
    has_feishu = raw.get("feishu") and isinstance(raw.get("feishu"), dict) and raw.get("feishu", {}).get("app_id")
    if not clusters or not isinstance(clusters, dict):
        if not has_businesses and not has_mcp_url and not has_feishu:
            raise ConfigError("配置缺少 'clusters' 或 clusters 为空")
    else:
        _validate_clusters(clusters)

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

    # 验证 auth（可选但如果存在需要合法）
    auth = raw.get("auth")
    if auth is not None and not isinstance(auth, dict):
        raise ConfigError("'auth' 配置格式无效")

    storage = raw.get("storage")
    if storage is not None and not isinstance(storage, dict):
        raise ConfigError("'storage' 配置格式无效")

    # 验证 feishu（可选但如果存在需要合法）
    feishu = raw.get("feishu")
    if feishu is not None and not isinstance(feishu, dict):
        raise ConfigError("'feishu' 配置格式无效")


def _build_cluster_config(name: str, c: dict) -> ClusterConfig:
    """从字典构建单个 ClusterConfig。"""
    return ClusterConfig(
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


def _build_business_knowledge(bk_raw: dict) -> BusinessKnowledge:
    """从原始字典构建 BusinessKnowledge。"""
    return BusinessKnowledge(
        description=bk_raw.get("description", ""),
        term_mappings=bk_raw.get("term_mappings", {}),
        table_relationships=bk_raw.get("table_relationships", []),
        status_codes=bk_raw.get("status_codes", []),
        custom_rules=bk_raw.get("custom_rules", []),
    )


def _build_app_config(raw: dict) -> AppConfig:
    """从原始字典构建类型安全的 AppConfig。

    Args:
        raw: 经过验证的配置字典。

    Returns:
        AppConfig 实例。
    """
    # 构建集群配置（支持扁平和多业务嵌套两种格式）
    clusters: dict[str, ClusterConfig] = {}
    business_clusters: dict[str, dict[str, ClusterConfig]] = {}
    clusters_raw = raw.get("clusters", {})
    if clusters_raw:
        # 检测格式：嵌套 or 扁平
        first_value = next(iter(clusters_raw.values()), None)
        is_nested = (
            isinstance(first_value, dict)
            and not any(k in first_value for k in _REQUIRED_CLUSTER_FIELDS)
        )
        if is_nested:
            # 嵌套格式：{business: {cluster: config}}
            for biz_name, biz_clusters in clusters_raw.items():
                biz_cc: dict[str, ClusterConfig] = {}
                for c_name, c in biz_clusters.items():
                    cc = _build_cluster_config(c_name, c)
                    biz_cc[c_name] = cc
                    clusters[c_name] = cc  # 扁平视图，向后兼容
                business_clusters[biz_name] = biz_cc
        else:
            # 扁平格式：{cluster: config}，放入 "default" 业务下
            for name, c in clusters_raw.items():
                cc = _build_cluster_config(name, c)
                clusters[name] = cc
            if clusters:
                business_clusters["default"] = dict(clusters)

    # 构建 SQL 安全配置
    sql_sec_raw = raw.get("sql_security", {})
    business_allowed_tables: dict[str, list[str]] = {}
    bat_raw = sql_sec_raw.get("business_allowed_tables", {})
    if isinstance(bat_raw, dict):
        for biz_name, tables in bat_raw.items():
            if isinstance(tables, list):
                business_allowed_tables[biz_name] = tables
    sql_security = SQLSecurityConfig(
        max_rows=int(sql_sec_raw.get("max_rows", 100)),
        query_timeout=int(sql_sec_raw.get("query_timeout", 30)),
        allowed_tables=sql_sec_raw.get("allowed_tables", []),
        business_allowed_tables=business_allowed_tables,
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
    business_knowledge = _build_business_knowledge(bk_raw)

    # 构建多业务知识配置（business_knowledges）
    business_knowledges: dict[str, BusinessKnowledge] = {}
    # 如果 business_knowledge 是嵌套格式（多个业务），解析为 business_knowledges
    if bk_raw and "description" not in bk_raw:
        # 嵌套格式：{business_name: {description, ...}}
        for biz_name, biz_bk in bk_raw.items():
            if isinstance(biz_bk, dict):
                business_knowledges[biz_name] = _build_business_knowledge(biz_bk)
    elif bk_raw:
        # 单业务格式，放入 "default" 下
        business_knowledges["default"] = business_knowledge

    # 构建顶层 MCP Server 列表
    mcp_servers: list[MCPServerEndpoint] = []
    mcp_servers_raw = raw.get("mcp_servers", [])
    if isinstance(mcp_servers_raw, list):
        for i, s in enumerate(mcp_servers_raw):
            if isinstance(s, dict):
                mcp_servers.append(MCPServerEndpoint(
                    name=s.get("name", f"server-{i}"),
                    url=s.get("url", ""),
                    api_key=s.get("api_key", ""),
                ))
            elif isinstance(s, str):
                # 简写：直接是 URL 字符串
                mcp_servers.append(MCPServerEndpoint(
                    name=f"server-{i}",
                    url=s,
                    api_key="",
                ))

    # 构建多业务配置
    businesses: dict[str, BusinessEntryConfig] = {}
    businesses_raw = raw.get("businesses", {})
    for biz_name, biz_cfg in businesses_raw.items():
        servers_raw = biz_cfg.get("servers", [])

        # servers 支持两种格式：
        # 1. 名称引用列表: ["shanghai", "beijing"]  → list[str]
        # 2. 旧格式 dict 列表: [{url, api_key}]     → 自动提取到 mcp_servers，替换为名称引用
        server_refs: list[str] = []
        if isinstance(servers_raw, list):
            for s in servers_raw:
                if isinstance(s, str):
                    # 名称引用格式
                    server_refs.append(s)
                elif isinstance(s, dict):
                    # 旧格式：自动提取为顶层 mcp_server
                    auto_name = f"auto-{biz_name}-{len(mcp_servers)}"
                    mcp_servers.append(MCPServerEndpoint(
                        name=auto_name,
                        url=s.get("url", ""),
                        api_key=s.get("api_key", ""),
                    ))
                    server_refs.append(auto_name)

        businesses[biz_name] = BusinessEntryConfig(
            name=biz_name,
            display_name=biz_cfg.get("display_name", biz_name),
            servers=server_refs,
            mcp_server_url=biz_cfg.get("mcp_server_url", ""),
            api_key=biz_cfg.get("api_key", ""),
        )

    # 向后兼容：如果配置了 agent.mcp_server_url 但没有 businesses 和 mcp_servers，
    # 自动创建 default server 和 default 业务
    if not businesses and not mcp_servers and agent.mcp_server_url:
        default_server = MCPServerEndpoint(
            name="default",
            url=agent.mcp_server_url,
            api_key=agent_raw.get("mcp_api_key", ""),
        )
        mcp_servers.append(default_server)
        businesses["default"] = BusinessEntryConfig(
            name="default",
            display_name=business_knowledge.description or "默认业务",
            servers=["default"],
            mcp_server_url=agent.mcp_server_url,
            api_key=agent_raw.get("mcp_api_key", ""),
        )

    # 构建鉴权配置
    auth_raw = raw.get("auth", {})
    auth = AuthConfig(
        api_key=auth_raw.get("api_key", ""),
    )

    storage_raw = raw.get("storage", {})
    storage = StorageConfig(
        namespace=storage_raw.get("namespace", ""),
    )

    # 构建飞书配置
    feishu_raw = raw.get("feishu", {})
    feishu = FeishuConfig(
        app_id=feishu_raw.get("app_id", ""),
        app_secret=feishu_raw.get("app_secret", ""),
        encrypt_key=feishu_raw.get("encrypt_key", ""),
        verification_token=feishu_raw.get("verification_token", ""),
        host=feishu_raw.get("host", "0.0.0.0"),
        port=int(feishu_raw.get("port", 8080)),
        session_ttl=int(feishu_raw.get("session_ttl", 1800)),
    )

    return AppConfig(
        clusters=clusters,
        business_clusters=business_clusters,
        mcp_servers=mcp_servers,
        sql_security=sql_security,
        agent=agent,
        business_knowledge=business_knowledge,
        business_knowledges=business_knowledges,
        businesses=businesses,
        auth=auth,
        storage=storage,
        feishu=feishu,
    )


def derive_storage_namespace(config_path: str) -> str:
    """从配置文件路径推导稳定的存储命名空间。"""
    normalized = os.path.abspath(config_path)
    stem = os.path.splitext(os.path.basename(normalized))[0] or "config"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}"


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
