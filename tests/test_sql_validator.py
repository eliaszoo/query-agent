"""SQL 验证器单元测试。"""

import pytest
from src.sql_validator import SQLValidator, ValidationResult


# 使用通用白名单，不依赖数字人表名
_TEST_ALLOWED_TABLES = ["tb_scene", "tb_model", "tb_user", "tb_order"]


@pytest.fixture
def validator():
    return SQLValidator(allowed_tables=_TEST_ALLOWED_TABLES)


class TestValidate:
    """validate() 方法测试。"""

    def test_valid_select(self, validator):
        result = validator.validate("SELECT * FROM tb_scene")
        assert result.is_valid is True
        assert result.error_type is None
        assert result.sanitized_sql is not None

    def test_valid_select_with_where(self, validator):
        result = validator.validate(
            "SELECT id, title FROM tb_scene WHERE status = 2"
        )
        assert result.is_valid is True

    def test_valid_join(self, validator):
        sql = (
            "SELECT s.id, m.title FROM tb_scene s "
            "JOIN tb_model m ON s.model_id = m.id"
        )
        result = validator.validate(sql)
        assert result.is_valid is True

    def test_empty_sql(self, validator):
        result = validator.validate("")
        assert result.is_valid is False
        assert result.error_type == "PARSE_ERROR"

    def test_none_sql(self, validator):
        result = validator.validate(None)
        assert result.is_valid is False
        assert result.error_type == "PARSE_ERROR"

    def test_whitespace_only(self, validator):
        result = validator.validate("   ")
        assert result.is_valid is False
        assert result.error_type == "PARSE_ERROR"

    # --- Statement type checks ---

    def test_insert_rejected(self, validator):
        result = validator.validate("INSERT INTO tb_scene (id) VALUES ('x')")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    def test_update_rejected(self, validator):
        result = validator.validate("UPDATE tb_scene SET title='x'")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    def test_delete_rejected(self, validator):
        result = validator.validate("DELETE FROM tb_scene WHERE id='x'")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    def test_drop_rejected(self, validator):
        result = validator.validate("DROP TABLE tb_scene")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    # --- Forbidden keywords ---

    def test_forbidden_keyword_truncate(self, validator):
        result = validator.validate("TRUNCATE TABLE tb_scene")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    def test_forbidden_keyword_in_subquery(self, validator):
        # SELECT with a forbidden keyword embedded
        result = validator.validate(
            "SELECT * FROM tb_scene WHERE id IN "
            "(SELECT id FROM tb_scene) INTO OUTFILE '/tmp/x'"
        )
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    # --- Lock clause checks (snapshot read guarantee) ---

    def test_for_update_rejected(self, validator):
        result = validator.validate("SELECT * FROM tb_scene FOR UPDATE")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"
        assert "FOR UPDATE" in result.error_message

    def test_lock_in_share_mode_rejected(self, validator):
        result = validator.validate("SELECT * FROM tb_scene LOCK IN SHARE MODE")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"
        assert "LOCK IN SHARE MODE" in result.error_message

    def test_for_share_rejected(self, validator):
        result = validator.validate("SELECT * FROM tb_scene FOR SHARE")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    def test_for_update_nowait_rejected(self, validator):
        result = validator.validate("SELECT * FROM tb_scene FOR UPDATE NOWAIT")
        assert result.is_valid is False
        assert result.error_type == "UNSAFE_SQL"

    # --- Table whitelist ---

    def test_forbidden_table(self, validator):
        result = validator.validate("SELECT * FROM users")
        assert result.is_valid is False
        assert result.error_type == "FORBIDDEN_TABLE"

    def test_forbidden_table_in_join(self, validator):
        result = validator.validate(
            "SELECT * FROM tb_scene s JOIN secret_table t ON s.id = t.id"
        )
        assert result.is_valid is False
        assert result.error_type == "FORBIDDEN_TABLE"

    # --- Empty whitelist allows all tables ---

    def test_empty_whitelist_allows_any_table(self):
        v = SQLValidator(allowed_tables=[])
        result = v.validate("SELECT * FROM any_table")
        assert result.is_valid is True

    # --- Comment injection ---

    def test_comment_stripped_dash(self, validator):
        result = validator.validate(
            "SELECT * FROM tb_scene -- this is a comment"
        )
        assert result.is_valid is True
        assert "--" not in (result.sanitized_sql or "")

    def test_comment_stripped_block(self, validator):
        result = validator.validate(
            "SELECT * FROM tb_scene /* block comment */"
        )
        assert result.is_valid is True
        assert "/*" not in (result.sanitized_sql or "")

    def test_only_comments(self, validator):
        result = validator.validate("-- just a comment")
        assert result.is_valid is False
        assert result.error_type == "PARSE_ERROR"

    # --- Multi-statement ---

    def test_multi_statement_rejected(self, validator):
        result = validator.validate(
            "SELECT * FROM tb_scene; DROP TABLE tb_scene"
        )
        assert result.is_valid is False
        assert result.error_type == "MULTI_STATEMENT"

    def test_trailing_semicolon_ok(self, validator):
        result = validator.validate("SELECT * FROM tb_scene;")
        assert result.is_valid is True


class TestEnsureLimit:
    """ensure_limit() 方法测试。"""

    def test_no_limit_appends(self, validator):
        sql = "SELECT * FROM tb_scene"
        result = validator.ensure_limit(sql, 100)
        assert result == "SELECT * FROM tb_scene LIMIT 100"

    def test_limit_under_max_kept(self, validator):
        sql = "SELECT * FROM tb_scene LIMIT 50"
        result = validator.ensure_limit(sql, 100)
        assert result == "SELECT * FROM tb_scene LIMIT 50"

    def test_limit_equal_max_kept(self, validator):
        sql = "SELECT * FROM tb_scene LIMIT 100"
        result = validator.ensure_limit(sql, 100)
        assert result == "SELECT * FROM tb_scene LIMIT 100"

    def test_limit_over_max_capped(self, validator):
        sql = "SELECT * FROM tb_scene LIMIT 500"
        result = validator.ensure_limit(sql, 100)
        assert result == "SELECT * FROM tb_scene LIMIT 100"

    def test_trailing_semicolon_stripped(self, validator):
        sql = "SELECT * FROM tb_scene;"
        result = validator.ensure_limit(sql, 100)
        assert result == "SELECT * FROM tb_scene LIMIT 100"

    def test_custom_max_rows(self, validator):
        sql = "SELECT * FROM tb_scene"
        result = validator.ensure_limit(sql, 50)
        assert result == "SELECT * FROM tb_scene LIMIT 50"
