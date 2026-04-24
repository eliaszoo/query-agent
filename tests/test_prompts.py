"""System Prompt 构建模块的单元测试。"""

import pytest

from src.config import BusinessKnowledge
from src.business_registry import BusinessEntry
from src.prompts import build_system_prompt


class TestBuildSystemPromptSingleBusiness:
    """单业务模式 build_system_prompt() 函数测试。"""

    def test_default_prompt_with_no_knowledge(self):
        """无业务知识时生成通用 prompt。"""
        prompt = build_system_prompt()
        assert "业务数据查询助手" in prompt
        assert "将自然语言转换为 SQL 查询" in prompt
        assert "execute_readonly_sql" in prompt
        assert "业务术语映射" not in prompt
        assert "核心表关系" not in prompt
        assert "常用状态码" not in prompt

    def test_prompt_with_description(self):
        """有 description 时替换到 prompt 中。"""
        bk = BusinessKnowledge(description="数字人平台")
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "数字人平台数据查询助手" in prompt
        assert "查询数字人平台相关业务数据" in prompt

    def test_prompt_with_term_mappings(self):
        """有 term_mappings 时生成术语映射节。"""
        bk = BusinessKnowledge(
            term_mappings={"模型": "tb_model 表", "用户": "tb_user 表"},
        )
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "## 业务术语映射" in prompt
        assert '"模型" → tb_model 表' in prompt
        assert '"用户" → tb_user 表' in prompt

    def test_prompt_with_table_relationships(self):
        """有 table_relationships 时生成表关系节。"""
        bk = BusinessKnowledge(
            table_relationships=["tb_user.id → tb_order.user_id"],
        )
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "## 核心表关系" in prompt
        assert "tb_user.id → tb_order.user_id" in prompt

    def test_prompt_with_status_codes(self):
        """有 status_codes 时生成状态码节。"""
        bk = BusinessKnowledge(
            status_codes=["tb_user.status: 1=活跃, 0=禁用"],
        )
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "## 常用状态码" in prompt
        assert "tb_user.status: 1=活跃, 0=禁用" in prompt

    def test_prompt_with_custom_rules(self):
        """有 custom_rules 时追加到查询规则中。"""
        bk = BusinessKnowledge(
            custom_rules=["不要使用子查询"],
        )
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "7. 不要使用子查询" in prompt

    def test_full_business_knowledge(self):
        """完整业务知识时所有节都出现。"""
        bk = BusinessKnowledge(
            description="电商",
            term_mappings={"商品": "tb_product 表"},
            table_relationships=["tb_order.product_id → tb_product.id"],
            status_codes=["tb_order.status: 1=待支付"],
            custom_rules=["优先使用索引列查询"],
        )
        prompt = build_system_prompt(knowledge_map={"default": bk})
        assert "电商数据查询助手" in prompt
        assert "## 业务术语映射" in prompt
        assert "## 核心表关系" in prompt
        assert "## 常用状态码" in prompt
        assert "7. 优先使用索引列查询" in prompt

    def test_query_rules_always_present(self):
        """查询规则始终存在。"""
        prompt = build_system_prompt()
        assert "1. 始终使用 execute_readonly_sql 工具执行查询" in prompt
        assert "2. 查询前可使用 get_table_schema 了解表结构" in prompt
        assert "3. 使用 get_cluster_list 查看可用集群" in prompt
        assert "4. 默认查询测试集群" in prompt
        assert "5. 注意 deleted_at IS NULL" in prompt
        assert "6. 结果以表格形式呈现" in prompt


class TestBuildSystemPromptMultiBusiness:
    """多业务模式 build_system_prompt() 函数测试。"""

    def test_multi_business_prompt_has_business_list(self):
        """多业务 prompt 包含业务列表。"""
        businesses = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
            BusinessEntry(name="order", display_name="订单", mcp_server_url="http://b/sse"),
        ]
        prompt = build_system_prompt(businesses=businesses)
        assert "多业务数据查询助手" in prompt
        assert "digitalhuman" in prompt
        assert "order" in prompt
        assert "数字人" in prompt
        assert "订单" in prompt
        assert "business 参数" in prompt

    def test_multi_business_with_knowledge(self):
        """多业务 prompt 包含各业务领域知识。"""
        businesses = [
            BusinessEntry(name="digitalhuman", display_name="数字人", mcp_server_url="http://a/sse"),
            BusinessEntry(name="order", display_name="订单", mcp_server_url="http://b/sse"),
        ]
        knowledge_map = {
            "digitalhuman": BusinessKnowledge(
                description="数字人平台",
                term_mappings={"模型": "tb_model 表"},
            ),
            "order": BusinessKnowledge(
                description="订单系统",
                status_codes=["tb_order.status: 1=待支付"],
            ),
        }
        prompt = build_system_prompt(businesses=businesses, knowledge_map=knowledge_map)
        assert "数字人 业务知识" in prompt
        assert "订单 业务知识" in prompt
        assert "tb_model 表" in prompt
        assert "tb_order.status: 1=待支付" in prompt

    def test_multi_business_rules_require_business_param(self):
        """多业务模式的查询规则要求指定 business 参数。"""
        businesses = [
            BusinessEntry(name="a", display_name="A", mcp_server_url="http://a/sse"),
            BusinessEntry(name="b", display_name="B", mcp_server_url="http://b/sse"),
        ]
        prompt = build_system_prompt(businesses=businesses)
        assert "必须指定 business 参数" in prompt
