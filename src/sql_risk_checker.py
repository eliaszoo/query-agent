"""SQL 性能风险分析器 - 基于索引信息判断查询是否存在性能风险。"""

import re
from dataclasses import dataclass, field

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis, Where
from sqlparse.tokens import Keyword, DML


@dataclass
class IndexInfo:
    """单个索引的信息。"""

    table: str  # 表名
    name: str  # 索引名（PRIMARY, idx_xxx 等）
    columns: list[str]  # 索引列列表（按顺序）
    unique: bool  # 是否唯一索引
    index_type: str  # 索引类型（BTREE, HASH 等）


@dataclass
class RiskCheckResult:
    """风险检测结果。"""

    has_risk: bool
    risk_level: str = ""  # "high" / "medium"
    risk_reasons: list[str] = field(default_factory=list)


class SQLRiskChecker:
    """基于索引信息分析 SQL 查询的潜在性能风险。

    使用方式：
        checker = SQLRiskChecker()
        checker.update_indexes("tb_scene", [IndexInfo(...), ...])
        result = checker.check("SELECT * FROM tb_scene WHERE name = 'x'")
        if result.has_risk:
            print(result.risk_level, result.risk_reasons)
    """

    def __init__(self) -> None:
        self._index_cache: dict[str, list[IndexInfo]] = {}

    def update_indexes(self, table: str, indexes: list[IndexInfo]) -> None:
        """更新表的索引信息缓存。

        Args:
            table: 表名。
            indexes: 该表的索引列表。
        """
        self._index_cache[table] = indexes

    def has_indexes(self, table: str) -> bool:
        """检查某张表是否有索引信息。"""
        return table in self._index_cache

    def check(self, sql: str) -> RiskCheckResult:
        """分析 SQL 的性能风险。

        Args:
            sql: 待分析的 SQL 查询语句。

        Returns:
            RiskCheckResult 实例。
        """
        reasons: list[str] = []
        risk_level = ""

        sql_stripped = sql.strip()
        if not sql_stripped:
            return RiskCheckResult(has_risk=False)

        # 提取表名
        tables = self._extract_tables(sql_stripped)

        # 提取 WHERE 条件中引用的列
        where_columns = self._extract_where_columns(sql_stripped)

        # 检查是否有 WHERE 子句
        has_where = bool(where_columns)

        # 检查是否有 LIMIT
        has_limit = bool(re.search(r"\bLIMIT\s+\d+", sql_stripped, re.IGNORECASE))

        # 1. 逐表检查索引覆盖
        for table in tables:
            indexes = self._index_cache.get(table)
            if indexes is None:
                # 没有索引信息，跳过索引分析
                continue

            table_where_cols = {
                col for col in where_columns
                if not col.startswith(("func:", "expr:"))
            }

            if not table_where_cols:
                # 无 WHERE 条件且无 LIMIT → 全表扫描
                if not has_limit:
                    reasons.append(f"表 {table} 无 WHERE 条件且无 LIMIT，可能返回大量数据（全表扫描）")
                    risk_level = "high"
                continue

            # 检查 WHERE 列是否能命中某个索引的前缀列
            hit_index = False
            non_prefix_hits: list[str] = []
            no_index_hits: list[str] = []

            # 先判断 WHERE 整体是否有驱动列（命中某索引前缀）
            has_driving_column = any(
                col == idx.columns[0]
                for col in table_where_cols
                for idx in indexes
                if idx.columns
            )

            for col in table_where_cols:
                col_hit_prefix = False
                col_hit_non_prefix = False
                for idx in indexes:
                    if not idx.columns:
                        continue
                    # 前缀列命中
                    if col == idx.columns[0]:
                        col_hit_prefix = True
                        hit_index = True
                        break
                    # 非前缀列命中
                    if col in idx.columns:
                        non_prefix_hits.append(
                            f"{col} 是索引 {idx.name}({', '.join(idx.columns)}) 的非前缀列"
                        )
                        col_hit_non_prefix = True
                        break

                if not col_hit_prefix and not col_hit_non_prefix:
                    if not has_driving_column:
                        # 无驱动列，此列导致全表扫描
                        no_index_hits.append(f"{col} 不在任何索引中")

            # 有驱动列时，非索引列是附加过滤，不构成风险 — 不加入 reasons
            if no_index_hits:
                for hint in no_index_hits:
                    reasons.append(f"表 {table}: {hint}，可能无法高效使用索引")
                risk_level = "high" if not risk_level or risk_level == "high" else risk_level
            if non_prefix_hits and not no_index_hits:
                for hint in non_prefix_hits:
                    reasons.append(f"表 {table}: {hint}")
                if not risk_level or risk_level == "medium":
                    risk_level = "medium"
            elif non_prefix_hits and hit_index:
                for hint in non_prefix_hits:
                    reasons.append(f"表 {table}: {hint}")
                if not risk_level:
                    risk_level = "medium"

        # 2. SELECT * 检查
        if self._has_select_star(sql_stripped):
            reasons.append("使用 SELECT *，返回所有列")
            if not risk_level:
                risk_level = "medium"

        # 3. LIKE '%...' 前导通配符检查
        leading_wildcard = self._check_leading_wildcard_like(sql_stripped)
        if leading_wildcard:
            reasons.append(f"LIKE 使用前导通配符，无法使用索引: {leading_wildcard}")
            if not risk_level:
                risk_level = "medium"

        # 4. 无 WHERE 且无 LIMIT（如果前面没有因为全表扫描标记过）
        if not has_where and not has_limit and not any("全表扫描" in r for r in reasons):
            # 没有索引信息时也标记
            reasons.append("无 WHERE 条件且无 LIMIT，可能返回大量数据")
            risk_level = "high"

        # 5. 派生表（子查询在 FROM 中）
        if self._has_derived_table(sql_stripped):
            reasons.append("使用派生表（FROM 子查询），可能产生临时表")
            if not risk_level:
                risk_level = "medium"

        has_risk = bool(reasons)
        return RiskCheckResult(
            has_risk=has_risk,
            risk_level=risk_level,
            risk_reasons=reasons,
        )

    # -----------------------------------------------------------------------
    # 内部方法
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_tables(sql: str) -> set[str]:
        """提取 SQL 中引用的表名。"""
        tables: set[str] = set()
        parsed = sqlparse.parse(sql)
        if not parsed:
            return tables

        for stmt in parsed:
            SQLRiskChecker._extract_tables_from_tokens(stmt.tokens, tables)

        return tables

    @staticmethod
    def _extract_tables_from_tokens(tokens, tables: set[str]) -> None:
        """递归提取表名。"""
        from_seen = False
        join_seen = False

        for token in tokens:
            if token.ttype is Keyword and token.normalized.upper() in (
                "FROM", "INTO", "UPDATE", "TABLE",
            ):
                from_seen = True
                join_seen = False
                continue

            if token.ttype is Keyword and "JOIN" in token.normalized.upper():
                join_seen = True
                from_seen = False
                continue

            if from_seen or join_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        name = SQLRiskChecker._get_table_name(identifier)
                        if name:
                            tables.add(name)
                    from_seen = False
                    join_seen = False
                elif isinstance(token, Identifier):
                    # 跳过子查询（派生表）
                    if not token.ttype and any(
                        t.ttype is DML and t.normalized.upper() == "SELECT"
                        for t in token.tokens
                    ):
                        from_seen = False
                        join_seen = False
                        continue
                    name = SQLRiskChecker._get_table_name(token)
                    if name:
                        tables.add(name)
                    from_seen = False
                    join_seen = False
                elif token.ttype not in (
                    sqlparse.tokens.Whitespace,
                    sqlparse.tokens.Punctuation,
                ):
                    from_seen = False
                    join_seen = False

            # 递归子查询
            if isinstance(token, Parenthesis):
                SQLRiskChecker._extract_tables_from_tokens(token.tokens, tables)
            elif hasattr(token, "tokens") and not isinstance(token, Identifier):
                SQLRiskChecker._extract_tables_from_tokens(token.tokens, tables)

    @staticmethod
    def _get_table_name(identifier) -> str | None:
        """从 Identifier 提取表名。"""
        if isinstance(identifier, Identifier):
            name = identifier.get_real_name()
            if name:
                return name.strip("`'\"")
        return None

    @staticmethod
    def _extract_where_columns(sql: str) -> set[str]:
        """提取 WHERE 子句中引用的列名（去除表别名前缀）。"""
        columns: set[str] = set()
        parsed = sqlparse.parse(sql)
        if not parsed:
            return columns

        for stmt in parsed:
            # 找到 WHERE 子句
            where_seen = False
            for token in stmt.tokens:
                if isinstance(token, Where):
                    SQLRiskChecker._extract_columns_from_expression(token, columns)
                    break
                if token.ttype is Keyword and token.normalized.upper() == "WHERE":
                    where_seen = True
                    continue
                if where_seen:
                    # 处理 WHERE 后到 ORDER BY / GROUP BY / LIMIT 之前的条件
                    if token.ttype is Keyword and token.normalized.upper() in (
                        "ORDER", "GROUP", "LIMIT", "HAVING",
                    ):
                        break
                    SQLRiskChecker._extract_columns_from_expression(token, columns)

        # 去除表别名前缀：s.id → id, tb_scene.id → id
        cleaned = set()
        for col in columns:
            if "." in col:
                cleaned.add(col.split(".", 1)[1])
            else:
                cleaned.add(col)
        return cleaned

    @staticmethod
    def _extract_columns_from_expression(token, columns: set[str]) -> None:
        """从表达式 token 中提取列名。"""
        if hasattr(token, "tokens"):
            for t in token.tokens:
                if isinstance(t, Identifier):
                    # 获取完整列名（可能包含表别名前缀，如 s.id）
                    val = t.value.strip().strip("`")
                    # 排除纯数字和函数调用
                    if val and not val.upper() in (
                        "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
                        "BETWEEN", "EXISTS", "ASC", "DESC", "TRUE", "FALSE",
                        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT",
                        "INNER", "OUTER", "ON", "AS", "GROUP", "ORDER", "HAVING",
                        "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT",
                    ):
                        columns.add(val)
                elif t.ttype is sqlparse.tokens.Name:
                    # 独立列名
                    val = t.value.strip("`")
                    val_upper = val.upper()
                    if val_upper not in (
                        "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
                        "BETWEEN", "EXISTS", "ASC", "DESC", "TRUE", "FALSE",
                        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT",
                        "INNER", "OUTER", "ON", "AS", "GROUP", "ORDER", "HAVING",
                        "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT",
                    ):
                        columns.add(val)
                elif hasattr(t, "tokens"):
                    SQLRiskChecker._extract_columns_from_expression(t, columns)

    @staticmethod
    def _has_select_star(sql: str) -> bool:
        """检查 SQL 是否使用 SELECT *。"""
        pattern = re.compile(r"\bSELECT\s+\*\s+FROM\b", re.IGNORECASE)
        return bool(pattern.search(sql))

    @staticmethod
    def _check_leading_wildcard_like(sql: str) -> str | None:
        """检查 LIKE 是否使用前导通配符。

        Returns:
            匹配到的 LIKE 模式，或 None。
        """
        pattern = re.compile(r"LIKE\s+'(%[^']*)'", re.IGNORECASE)
        match = pattern.search(sql)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _has_derived_table(sql: str) -> bool:
        """检查 SQL 是否使用派生表（FROM 子查询）。"""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False

        for stmt in parsed:
            from_seen = False
            for token in stmt.tokens:
                if token.ttype is Keyword and token.normalized.upper() == "FROM":
                    from_seen = True
                    continue
                if from_seen and isinstance(token, Identifier):
                    # 检查 identifier 内是否有 Parenthesis 包含子查询
                    for sub_token in token.tokens:
                        if isinstance(sub_token, Parenthesis):
                            # 检查 Parenthesis 内是否有 SELECT
                            inner_sql = sub_token.value.strip()
                            if re.search(r"\bSELECT\b", inner_sql, re.IGNORECASE):
                                return True
                    from_seen = False
        return False
