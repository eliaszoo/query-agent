"""BusinessSelectionService 单元测试。"""

import pytest

from src.business_registry import BusinessEntry
from src.business_selection_service import BusinessSelectionService, BusinessSelectionResult
from src.config import BusinessKnowledge, MCPServerEndpoint


def _make_entry(name: str, display_name: str = "", knowledge: BusinessKnowledge | None = None) -> BusinessEntry:
    return BusinessEntry(
        name=name,
        display_name=display_name or name,
        servers=[MCPServerEndpoint(url=f"http://{name}/sse")],
        knowledge=knowledge,
    )


class TestHeuristicSelect:
    """测试 _heuristic_select 逻辑。"""

    def test_match_by_name(self):
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman"),
            _make_entry("copyright_music"),
        ]
        result = service._heuristic_select("查询 digitalhuman 数据", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_match_by_display_name(self):
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", display_name="数字人"),
            _make_entry("copyright_music", display_name="版权音乐"),
        ]
        result = service._heuristic_select("查数字人", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_match_by_knowledge_description(self):
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", knowledge=BusinessKnowledge(description="数字人平台")),
            _make_entry("copyright_music", knowledge=BusinessKnowledge(description="版权音乐平台")),
        ]
        result = service._heuristic_select("查数字人平台数据", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_match_by_term_mappings_single_word(self):
        """term_mappings 中单个关键词匹配。"""
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", knowledge=BusinessKnowledge(
                description="dh",
                term_mappings={"模型": "tb_model 表"},
            )),
            _make_entry("copyright_music", knowledge=BusinessKnowledge(description="cm")),
        ]
        result = service._heuristic_select("查模型数据", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_match_by_term_mappings_slash_separated(self):
        """term_mappings 中用 / 分隔的关键词匹配。如 "形象/数字人" → "数字人" 可匹配。"""
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", knowledge=BusinessKnowledge(
                description="dh",
                term_mappings={"形象/数字人": "tb_scene 表"},
            )),
            _make_entry("copyright_music", knowledge=BusinessKnowledge(description="cm")),
        ]
        result = service._heuristic_select("查2个数字人", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_match_by_term_mappings_slash_first_word(self):
        """/ 分隔的第一个词也能匹配。如 "形象/数字人" → "形象" 可匹配。"""
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", knowledge=BusinessKnowledge(
                description="dh",
                term_mappings={"形象/数字人": "tb_scene 表"},
            )),
            _make_entry("copyright_music", knowledge=BusinessKnowledge(description="cm")),
        ]
        result = service._heuristic_select("查形象列表", businesses)
        assert result is not None
        assert result.name == "digitalhuman"

    def test_no_match_returns_none(self):
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman"),
            _make_entry("copyright_music"),
        ]
        result = service._heuristic_select("查订单数据", businesses)
        assert result is None

    def test_multiple_matches_returns_none(self):
        """多个业务匹配时返回 None（无法唯一确定）。"""
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman", display_name="数字人", knowledge=BusinessKnowledge(
                description="数字人平台",
                term_mappings={"形象/数字人": "tb_scene 表"},
            )),
            _make_entry("copyright_music", display_name="版权音乐", knowledge=BusinessKnowledge(
                description="版权音乐平台",
                term_mappings={"数字人": "tb_voice 表"},
            )),
        ]
        # "数字人" 同时匹配 digitalhuman 的 term_mappings 和 copyright_music 的 term_mappings
        result = service._heuristic_select("查数字人数据", businesses)
        assert result is None

    def test_term_mappings_no_knowledge_skipped(self):
        """没有 knowledge 时不崩溃。"""
        service = BusinessSelectionService(provider=None, model="", registry=None)
        businesses = [
            _make_entry("digitalhuman"),
            _make_entry("copyright_music"),
        ]
        result = service._heuristic_select("查数字人", businesses)
        assert result is None
