"""默认规则模块 - 持久化记录用户明确声明的查询偏好。"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PreferenceRule:
    """单条默认规则。"""

    business: str
    rule: str
    rule_type: str = ""
    payload: dict | None = None
    source: str = ""
    timestamp: str = ""


@dataclass
class PreferenceRuleStore:
    """默认规则存储。"""

    entries: list[PreferenceRule] = field(default_factory=list)


class PreferenceRulesManager:
    """管理默认规则的持久化读写和 prompt 生成。"""

    def __init__(self, rules_path: str):
        self._path = rules_path
        self._store = self._load()

    def _load(self) -> PreferenceRuleStore:
        if not os.path.exists(self._path):
            return PreferenceRuleStore()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PreferenceRuleStore(
                entries=[PreferenceRule(**item) for item in data.get("entries", [])]
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("默认规则文件解析失败，将重新创建: %s", exc)
            return PreferenceRuleStore()

    def _save(self) -> None:
        data = {"entries": [asdict(entry) for entry in self._store.entries]}
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def add_rule(
        self,
        business: str,
        rule: str,
        source: str = "",
        rule_type: str = "",
        payload: dict | None = None,
    ) -> None:
        for entry in self._store.entries:
            if (
                entry.business == business
                and entry.rule == rule
                and entry.rule_type == rule_type
            ):
                entry.source = source or entry.source
                entry.payload = payload or entry.payload
                entry.timestamp = datetime.now().isoformat()
                self._save()
                return

        self._store.entries.append(
            PreferenceRule(
                business=business,
                rule=rule,
                rule_type=rule_type,
                payload=payload,
                source=source,
                timestamp=datetime.now().isoformat(),
            )
        )
        self._save()

    def get_rules(self, business: str = "") -> list[PreferenceRule]:
        if not business:
            return list(self._store.entries)
        # 同时返回通用规则（business 为空）和指定业务的规则
        return [
            entry for entry in self._store.entries
            if not entry.business or entry.business == business
        ]

    def clear(self, business: str = "") -> None:
        if not business:
            self._store = PreferenceRuleStore()
        else:
            self._store.entries = [
                entry for entry in self._store.entries if entry.business != business
            ]
        self._save()

    def build_rules_prompt(self, current_business: str = "") -> str:
        entries = self.get_rules(current_business)
        if not entries:
            return ""

        lines = ["\n## 默认查询规则（用户明确指定，优先遵循）\n"]
        for entry in entries:
            prefix = f"[{entry.business}] " if entry.business else ""
            lines.append(f"- {prefix}{entry.rule}")
        return "\n".join(lines)
