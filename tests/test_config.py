"""配置加载模块的单元测试。"""

import os
import tempfile

import pytest

from src.config import (
    AppConfig,
    ClusterConfig,
    ConfigError,
    SQLSecurityConfig,
    AgentConfig,
    BusinessKnowledge,
    BusinessEntryConfig,
    AuthConfig,
    StorageConfig,
    derive_storage_namespace,
    load_config,
    _substitute_env_vars,
)


def _write_yaml(content: str) -> str:
    """写入临时 YAML 文件并返回路径。"""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


MINIMAL_VALID_YAML = """\
clusters:
  test:
    description: "测试环境"
    host: "127.0.0.1"
    port: 3306
    database: "testdb"
    user: "root"
    password: "secret"
"""


class TestLoadConfig:
    """测试 load_config 函数。"""

    def test_load_valid_config(self):
        path = _write_yaml(MINIMAL_VALID_YAML)
        try:
            cfg = load_config(path)
            assert isinstance(cfg, AppConfig)
            assert "test" in cfg.clusters
            cluster = cfg.clusters["test"]
            assert cluster.host == "127.0.0.1"
            assert cluster.port == 3306
            assert cluster.database == "testdb"
            assert cluster.user == "root"
            assert cluster.password == "secret"
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(ConfigError, match="配置文件不存在"):
            load_config("/nonexistent/path.yaml")

    def test_empty_file(self):
        path = _write_yaml("")
        try:
            with pytest.raises(ConfigError, match="配置文件为空"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_invalid_yaml(self):
        path = _write_yaml("{{invalid: yaml: [")
        try:
            with pytest.raises(ConfigError, match="YAML 解析失败"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_missing_clusters_without_businesses(self):
        path = _write_yaml("agent:\n  model: test\n")
        try:
            with pytest.raises(ConfigError, match="缺少 'clusters'"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_empty_clusters_without_businesses(self):
        path = _write_yaml("clusters: {}\n")
        try:
            with pytest.raises(ConfigError, match="缺少 'clusters'"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_missing_required_field(self):
        yaml_content = """\
clusters:
  test:
    host: "127.0.0.1"
    port: 3306
    database: "testdb"
    user: "root"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="缺少必填字段: 'password'"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_full_config(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
    charset: "utf8"
    max_connections: 10
    connect_timeout: 5
sql_security:
  max_rows: 50
  query_timeout: 15
  allowed_tables:
    - "tb_scene"
    - "tb_model"
agent:
  model: "claude-sonnet-4-20250514"
  max_tokens: 2048
  default_cluster: "test"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.clusters["test"].charset == "utf8"
            assert cfg.clusters["test"].max_connections == 10
            assert cfg.sql_security.max_rows == 50
            assert cfg.sql_security.query_timeout == 15
            assert cfg.sql_security.allowed_tables == ["tb_scene", "tb_model"]
            assert cfg.agent.model == "claude-sonnet-4-20250514"
            assert cfg.agent.max_tokens == 2048
        finally:
            os.unlink(path)

    def test_defaults_for_optional_sections(self):
        path = _write_yaml(MINIMAL_VALID_YAML)
        try:
            cfg = load_config(path)
            # sql_security and agent should have defaults
            assert isinstance(cfg.sql_security, SQLSecurityConfig)
            assert isinstance(cfg.agent, AgentConfig)
            assert isinstance(cfg.business_knowledge, BusinessKnowledge)
            assert isinstance(cfg.storage, StorageConfig)
            assert cfg.sql_security.max_rows == 100
            assert cfg.agent.default_cluster == "test"
            assert cfg.business_knowledge.description == ""
            assert cfg.storage.namespace == ""
        finally:
            os.unlink(path)

    def test_business_knowledge_config(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
business_knowledge:
  description: "数字人平台"
  term_mappings:
    "模型": "tb_model 表"
    "形象": "tb_scene 表"
  table_relationships:
    - "tb_scene.model_id → tb_model.id"
  status_codes:
    - "tb_scene.status: 1=训练中"
  custom_rules:
    - "不要使用子查询"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            bk = cfg.business_knowledge
            assert bk.description == "数字人平台"
            assert bk.term_mappings == {"模型": "tb_model 表", "形象": "tb_scene 表"}
            assert bk.table_relationships == ["tb_scene.model_id → tb_model.id"]
            assert bk.status_codes == ["tb_scene.status: 1=训练中"]
            assert bk.custom_rules == ["不要使用子查询"]
        finally:
            os.unlink(path)

    def test_business_knowledge_invalid_type(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
business_knowledge: "invalid"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="'business_knowledge' 配置格式无效"):
                load_config(path)
        finally:
            os.unlink(path)


class TestBusinessesConfig:
    """测试多业务配置。"""

    def test_businesses_config(self):
        yaml_content = """\
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
  order:
    display_name: "订单"
    mcp_server_url: "http://other:8765/sse"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert len(cfg.businesses) == 2
            assert cfg.businesses["digitalhuman"].display_name == "数字人"
            assert cfg.businesses["digitalhuman"].mcp_server_url == "http://host:8765/sse"
            assert cfg.businesses["order"].display_name == "订单"
        finally:
            os.unlink(path)

    def test_businesses_with_api_key(self):
        yaml_content = """\
agent:
  model: "test-model"
  max_tokens: 1024
  default_cluster: "test"
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
    api_key: "secret123"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.businesses["digitalhuman"].api_key == "secret123"
        finally:
            os.unlink(path)

    def test_businesses_without_clusters_ok(self):
        """有 businesses 时，clusters 可以为空。"""
        yaml_content = """\
agent:
  model: "test-model"
  max_tokens: 1024
businesses:
  digitalhuman:
    display_name: "数字人"
    mcp_server_url: "http://host:8765/sse"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert len(cfg.businesses) == 1
            assert len(cfg.clusters) == 0
        finally:
            os.unlink(path)

    def test_backward_compat_mcp_server_url_creates_default_business(self):
        """向后兼容：有 agent.mcp_server_url 但无 businesses 时，自动创建 default 业务。"""
        yaml_content = """\
agent:
  model: "test-model"
  max_tokens: 1024
  mcp_server_url: "http://host:8765/sse"
  default_cluster: "test"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert "default" in cfg.businesses
            assert cfg.businesses["default"].mcp_server_url == "http://host:8765/sse"
        finally:
            os.unlink(path)

    def test_backward_compat_with_business_knowledge_description(self):
        """向后兼容：自动创建 default 业务时使用 business_knowledge.description 作为 display_name。"""
        yaml_content = """\
agent:
  model: "test-model"
  max_tokens: 1024
  mcp_server_url: "http://host:8765/sse"
  default_cluster: "test"
business_knowledge:
  description: "数字人平台"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.businesses["default"].display_name == "数字人平台"
        finally:
            os.unlink(path)

    def test_businesses_invalid_type(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
businesses: "invalid"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="'businesses' 配置格式无效"):
                load_config(path)
        finally:
            os.unlink(path)


class TestEnvVarSubstitution:
    """测试环境变量替换。"""

    def test_substitute_string(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        result = _substitute_env_vars("${MY_VAR}")
        assert result == "hello"

    def test_substitute_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "db.example.com")
        result = _substitute_env_vars("jdbc://${HOST}:3306")
        assert result == "jdbc://db.example.com:3306"

    def test_unset_env_var_preserved(self):
        # Ensure the var is not set
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        result = _substitute_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_substitute_in_dict(self, monkeypatch):
        monkeypatch.setenv("DB_PASS", "s3cret")
        data = {"password": "${DB_PASS}", "port": 3306}
        result = _substitute_env_vars(data)
        assert result == {"password": "s3cret", "port": 3306}

    def test_substitute_in_list(self, monkeypatch):
        monkeypatch.setenv("ITEM", "value")
        result = _substitute_env_vars(["${ITEM}", "static"])
        assert result == ["value", "static"]

    def test_non_string_passthrough(self):
        assert _substitute_env_vars(42) == 42
        assert _substitute_env_vars(True) is True
        assert _substitute_env_vars(None) is None

    def test_load_config_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("TEST_HOST", "env-host.example.com")
        monkeypatch.setenv("TEST_PASS", "env-password")
        yaml_content = """\
clusters:
  test:
    host: "${TEST_HOST}"
    port: 3306
    database: "testdb"
    user: "root"
    password: "${TEST_PASS}"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.clusters["test"].host == "env-host.example.com"
            assert cfg.clusters["test"].password == "env-password"
        finally:
            os.unlink(path)

    def test_unresolved_env_var_raises_error(self):
        os.environ.pop("UNSET_HOST_VAR", None)
        yaml_content = """\
clusters:
  test:
    host: "${UNSET_HOST_VAR}"
    port: 3306
    database: "testdb"
    user: "root"
    password: "plain_password"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="未解析的环境变量"):
                load_config(path)
        finally:
            os.unlink(path)


class TestAuthConfig:
    """测试鉴权配置。"""

    def test_auth_api_key_from_config(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
auth:
  api_key: "my-secret-key"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.auth.api_key == "my-secret-key"
        finally:
            os.unlink(path)


class TestStorageConfig:
    def test_storage_namespace_from_config(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
storage:
  namespace: "prod-digitalhuman"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.storage.namespace == "prod-digitalhuman"
        finally:
            os.unlink(path)

    def test_storage_invalid_type(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
storage: "invalid"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="'storage' 配置格式无效"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_derive_storage_namespace_is_stable(self):
        namespace1 = derive_storage_namespace("/tmp/a/config.yaml")
        namespace2 = derive_storage_namespace("/tmp/a/config.yaml")
        namespace3 = derive_storage_namespace("/tmp/b/config.yaml")

        assert namespace1 == namespace2
        assert namespace1 != namespace3

    def test_auth_empty_by_default(self):
        path = _write_yaml(MINIMAL_VALID_YAML)
        try:
            cfg = load_config(path)
            assert cfg.auth.api_key == ""
        finally:
            os.unlink(path)

    def test_auth_api_key_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MCP_API_KEY", "env-secret")
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
auth:
  api_key: "${MCP_API_KEY}"
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.auth.api_key == "env-secret"
        finally:
            os.unlink(path)

    def test_auth_invalid_type(self):
        yaml_content = """\
clusters:
  test:
    description: "测试"
    host: "localhost"
    port: 3306
    database: "testdb"
    user: "user"
    password: "pass"
auth: "invalid"
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ConfigError, match="'auth' 配置格式无效"):
                load_config(path)
        finally:
            os.unlink(path)
