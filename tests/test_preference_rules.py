"""默认规则模块的单元测试。"""

import json
import os

import pytest

from src.preference_rules import PreferenceRulesManager


class TestPreferenceRulesManager:
    """PreferenceRulesManager 测试。"""

    def test_general_rules_included_for_business(self, tmp_path):
        """查询特定业务规则时，通用规则（business 为空）也应返回。"""
        rules_path = str(tmp_path / "rules.json")
        manager = PreferenceRulesManager(rules_path=rules_path)

        # 添加通用规则
        manager.add_rule(business="", rule="默认只查可用数据", rule_type="available_only")
        # 添加业务规则
        manager.add_rule(business="digitalhuman", rule="默认查测试环境", rule_type="default_cluster_test")

        # 查询特定业务时，应同时返回通用规则
        rules = manager.get_rules(business="digitalhuman")
        assert len(rules) == 2
        rule_texts = [r.rule for r in rules]
        assert "默认只查可用数据" in rule_texts
        assert "默认查测试环境" in rule_texts

    def test_general_rules_only_returned_for_empty_business(self, tmp_path):
        """查询全部规则时返回所有条目。"""
        rules_path = str(tmp_path / "rules.json")
        manager = PreferenceRulesManager(rules_path=rules_path)

        manager.add_rule(business="", rule="通用规则")
        manager.add_rule(business="order", rule="订单规则")

        rules = manager.get_rules(business="")
        assert len(rules) == 2

    def test_no_cross_business_rules(self, tmp_path):
        """查询业务 A 时不应返回业务 B 的规则。"""
        rules_path = str(tmp_path / "rules.json")
        manager = PreferenceRulesManager(rules_path=rules_path)

        manager.add_rule(business="digitalhuman", rule="数字人规则")
        manager.add_rule(business="order", rule="订单规则")

        rules = manager.get_rules(business="digitalhuman")
        assert len(rules) == 1
        assert rules[0].rule == "数字人规则"
