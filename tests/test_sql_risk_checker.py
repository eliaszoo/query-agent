"""SQL 性能风险分析器的单元测试。"""

import pytest

from src.sql_risk_checker import SQLRiskChecker, IndexInfo, RiskCheckResult


class TestIndexInfo:
    def test_create_index_info(self):
        idx = IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE")
        assert idx.table == "tb_scene"
        assert idx.columns == ["id"]
        assert idx.unique is True


class TestSQLRiskChecker:
    def setup_method(self):
        self.checker = SQLRiskChecker()

    # --- 索引缓存 ---

    def test_update_indexes(self):
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        assert self.checker.has_indexes("tb_scene")

    def test_no_indexes(self):
        assert not self.checker.has_indexes("tb_scene")

    # --- 无 WHERE 条件 ---

    def test_no_where_no_limit_high_risk(self):
        """无 WHERE 且无 LIMIT → high risk。"""
        result = self.checker.check("SELECT * FROM tb_scene")
        assert result.has_risk
        assert result.risk_level == "high"

    def test_no_where_with_limit_still_risk(self):
        """无 WHERE 但有 LIMIT → 仍有 SELECT * 风险。"""
        result = self.checker.check("SELECT * FROM tb_scene LIMIT 10")
        assert result.has_risk
        # SELECT * 是 medium risk
        assert "medium" in result.risk_level or result.risk_level == "medium"

    # --- 索引命中 ---

    def test_where_hits_index_prefix_no_risk(self):
        """WHERE 条件命中索引前缀列 → 无索引风险。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
            IndexInfo(table="tb_scene", name="idx_model_id", columns=["model_id"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE model_id = 1")
        assert not result.has_risk

    def test_where_hits_composite_index_prefix_no_risk(self):
        """WHERE 条件命中复合索引前缀列 → 无索引风险。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_status_created", columns=["status", "created_at"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE status = 1")
        assert not result.has_risk

    def test_where_non_prefix_column_medium_risk(self):
        """WHERE 条件只命中索引非前缀列 → medium risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_status_created", columns=["status", "created_at"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE created_at > '2024-01-01'")
        assert result.has_risk
        assert result.risk_level == "medium"
        assert any("非前缀列" in r for r in result.risk_reasons)

    def test_where_column_not_in_any_index(self):
        """WHERE 条件列不在任何索引中 → high risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE name = 'test'")
        assert result.has_risk
        assert result.risk_level == "high"

    def test_mixed_prefix_and_no_index(self):
        """部分列命中索引前缀，部分不在任何索引中 → high risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_model_id", columns=["model_id"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check(
            "SELECT id FROM tb_scene WHERE model_id = 1 AND name = 'test'"
        )
        assert result.has_risk
        assert result.risk_level == "high"

    def test_mixed_prefix_and_non_prefix(self):
        """部分列命中索引前缀，部分命中非前缀列 → medium risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_model_id", columns=["model_id"], unique=False, index_type="BTREE"),
            IndexInfo(table="tb_scene", name="idx_status_created", columns=["status", "created_at"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check(
            "SELECT id FROM tb_scene WHERE model_id = 1 AND created_at > '2024-01-01'"
        )
        assert result.has_risk
        assert result.risk_level == "medium"

    # --- SELECT * ---

    def test_select_star_medium_risk(self):
        """SELECT * → medium risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT * FROM tb_scene WHERE id = 1")
        assert result.has_risk
        assert any("SELECT *" in r for r in result.risk_reasons)

    # --- LIKE 前导通配符 ---

    def test_leading_wildcard_like(self):
        """LIKE '%...' 前导通配符 → medium risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_name", columns=["name"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE name LIKE '%test%'")
        assert result.has_risk
        assert any("前导通配符" in r for r in result.risk_reasons)

    def test_trailing_wildcard_like_no_risk(self):
        """LIKE 'test%' 后缀通配符 → 无 LIKE 风险。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="idx_name", columns=["name"], unique=False, index_type="BTREE"),
        ])
        result = self.checker.check("SELECT id FROM tb_scene WHERE name LIKE 'test%'")
        assert not any("前导通配符" in r for r in result.risk_reasons)

    # --- 派生表 ---

    def test_derived_table_medium_risk(self):
        """FROM 子查询（派生表）→ medium risk。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        result = self.checker.check(
            "SELECT id FROM (SELECT id FROM tb_scene WHERE id = 1) AS sub"
        )
        assert result.has_risk
        assert any("派生表" in r for r in result.risk_reasons)

    # --- 无索引信息 ---

    def test_no_index_info_no_where_high_risk(self):
        """没有索引信息且无 WHERE → high risk（全表扫描）。"""
        result = self.checker.check("SELECT id FROM tb_scene")
        assert result.has_risk
        assert result.risk_level == "high"

    def test_no_index_info_with_where_no_risk(self):
        """没有索引信息但有 WHERE → 无法判断，不报索引风险。"""
        result = self.checker.check("SELECT id FROM tb_scene WHERE id = 1")
        assert not result.has_risk

    # --- 空输入 ---

    def test_empty_sql(self):
        result = self.checker.check("")
        assert not result.has_risk

    # --- 多表查询 ---

    def test_multi_table_join(self):
        """多表 JOIN 时逐表检查索引覆盖。"""
        self.checker.update_indexes("tb_scene", [
            IndexInfo(table="tb_scene", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        self.checker.update_indexes("tb_model", [
            IndexInfo(table="tb_model", name="PRIMARY", columns=["id"], unique=True, index_type="BTREE"),
        ])
        result = self.checker.check(
            "SELECT s.id FROM tb_scene s JOIN tb_model m ON s.model_id = m.id WHERE s.id = 1"
        )
        assert not result.has_risk
