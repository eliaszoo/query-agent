"""SQL 安全验证器 - 确保只执行只读查询，验证表白名单。"""

import re
from dataclasses import dataclass
from typing import Optional

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis
from sqlparse.tokens import Keyword, DML


@dataclass
class ValidationResult:
    """SQL 验证结果。"""

    is_valid: bool
    error_type: Optional[str] = None  # "UNSAFE_SQL", "FORBIDDEN_TABLE", "PARSE_ERROR", "MULTI_STATEMENT"
    error_message: Optional[str] = None
    sanitized_sql: Optional[str] = None


class SQLValidator:
    """SQL 安全验证器，确保只执行只读快照查询。

    安全检查：
    - 只允许 SELECT 语句
    - 禁止写操作关键字（INSERT/UPDATE/DELETE 等）
    - 禁止锁相关语法（FOR UPDATE / LOCK IN SHARE MODE / FOR SHARE）确保快照读
    - 禁止注释注入和多语句
    - 表白名单验证（白名单通过构造函数注入）
    - 强制 LIMIT 子句
    """

    FORBIDDEN_KEYWORDS = [
        "INSERT", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "REPLACE", "GRANT", "REVOKE", "UNLOCK",
        "CALL", "EXEC", "EXECUTE", "SET", "LOAD", "INTO OUTFILE",
        "INTO DUMPFILE",
    ]

    # 禁止的锁相关后缀，确保只做快照读不产生行锁
    FORBIDDEN_LOCK_CLAUSES = [
        "FOR UPDATE",
        "LOCK IN SHARE MODE",
        "FOR SHARE",
        "FOR SHARE NOWAIT",
        "FOR SHARE SKIP LOCKED",
        "FOR UPDATE NOWAIT",
        "FOR UPDATE SKIP LOCKED",
    ]

    # Pre-compiled patterns
    _COMMENT_PATTERN = re.compile(
        r"--[^\n]*|/\*.*?\*/", re.DOTALL | re.IGNORECASE
    )
    _LIMIT_PATTERN = re.compile(
        r"\bLIMIT\s+(\d+)\s*$", re.IGNORECASE
    )
    _MULTI_STATEMENT_PATTERN = re.compile(
        r";\s*\S", re.DOTALL
    )

    def __init__(self, allowed_tables: list[str] | None = None) -> None:
        self._allowed_tables: list[str] = allowed_tables if allowed_tables is not None else []

    def _strip_comments(self, sql: str) -> str:
        """Remove SQL comments (-- and /* */)."""
        return self._COMMENT_PATTERN.sub("", sql).strip()

    def _check_comments(self, sql: str) -> bool:
        """Return True if SQL contains comments."""
        return bool(self._COMMENT_PATTERN.search(sql))

    def _check_multi_statement(self, sql: str) -> bool:
        """Return True if SQL contains multiple statements separated by semicolons."""
        # Strip trailing whitespace/semicolons first
        cleaned = sql.strip().rstrip(";").strip()
        return bool(self._MULTI_STATEMENT_PATTERN.search(cleaned))

    def _check_forbidden_keywords(self, sql: str) -> Optional[str]:
        """Check for forbidden keywords. Returns the matched keyword or None."""
        sql_upper = sql.upper()
        for kw in self.FORBIDDEN_KEYWORDS:
            # Use word boundary for single-word keywords
            if " " in kw:
                # Multi-word keywords like "INTO OUTFILE"
                pattern = re.compile(r"\b" + kw.replace(" ", r"\s+") + r"\b", re.IGNORECASE)
            else:
                pattern = re.compile(r"\b" + kw + r"\b", re.IGNORECASE)
            if pattern.search(sql_upper):
                return kw
        return None

    def _check_lock_clauses(self, sql: str) -> Optional[str]:
        """检查是否包含锁相关子句，确保只做快照读。

        Returns:
            匹配到的锁子句，或 None。
        """
        sql_upper = sql.upper().rstrip(";").rstrip()
        for clause in self.FORBIDDEN_LOCK_CLAUSES:
            pattern = re.compile(
                r"\b" + clause.replace(" ", r"\s+") + r"\b", re.IGNORECASE
            )
            if pattern.search(sql_upper):
                return clause
        return None

    def _check_statement_type(self, sql: str) -> Optional[str]:
        """Check that the SQL is a SELECT statement. Returns error message or None."""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return "无法解析 SQL 语句"
        stmt = parsed[0]
        stmt_type = stmt.get_type()
        if stmt_type != "SELECT":
            return f"仅允许 SELECT 查询，检测到: {stmt_type or '未知'}"
        return None

    def _extract_table_names(self, sql: str) -> set[str]:
        """Extract all table names referenced in the SQL."""
        tables: set[str] = set()
        parsed = sqlparse.parse(sql)
        if not parsed:
            return tables

        for stmt in parsed:
            self._extract_tables_from_tokens(stmt.tokens, tables)

        return tables

    def _extract_tables_from_tokens(self, tokens, tables: set[str]) -> None:
        """Recursively extract table names from token list."""
        from_seen = False
        join_seen = False

        for token in tokens:
            # Track FROM and JOIN keywords
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
                        name = self._get_table_name(identifier)
                        if name:
                            tables.add(name)
                    from_seen = False
                    join_seen = False
                elif isinstance(token, Identifier):
                    name = self._get_table_name(token)
                    if name:
                        tables.add(name)
                    from_seen = False
                    join_seen = False
                elif token.ttype is not sqlparse.tokens.Whitespace:
                    # If it's a plain name token (not whitespace)
                    if token.ttype in (
                        sqlparse.tokens.Name,
                        sqlparse.tokens.Literal.String.Single,
                    ):
                        tables.add(token.value.strip("`'\""))
                    from_seen = False
                    join_seen = False

            # Recurse into subqueries / parenthesized expressions
            if isinstance(token, Parenthesis):
                self._extract_tables_from_tokens(token.tokens, tables)
            elif hasattr(token, "tokens") and not isinstance(token, Identifier):
                self._extract_tables_from_tokens(token.tokens, tables)

    @staticmethod
    def _get_table_name(identifier) -> Optional[str]:
        """Extract the real table name from an Identifier token."""
        if isinstance(identifier, Identifier):
            # Handle schema.table or just table
            name = identifier.get_real_name()
            if name:
                return name.strip("`'\"")
        return None

    def validate(self, sql: str) -> ValidationResult:
        """验证 SQL 安全性。

        检查步骤：
        1. 使用 sqlparse 解析 SQL，确认语句类型为 SELECT
        2. 正则检查是否包含禁止关键字
        3. 检查是否包含锁相关子句（FOR UPDATE 等），确保快照读
        4. 提取所有表名，验证是否在白名单中
        5. 检查是否包含注释注入（-- 或 /* */）
        6. 检查是否包含多语句

        Args:
            sql: 待验证的 SQL 字符串。

        Returns:
            ValidationResult 实例。
        """
        if not sql or not sql.strip():
            return ValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message="SQL 语句为空",
            )

        # Step 5 (early): Check and strip comments
        has_comments = self._check_comments(sql)
        working_sql = self._strip_comments(sql) if has_comments else sql.strip()

        if not working_sql:
            return ValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message="SQL 语句为空（仅包含注释）",
            )

        # Step 6: Check multi-statement
        if self._check_multi_statement(working_sql):
            return ValidationResult(
                is_valid=False,
                error_type="MULTI_STATEMENT",
                error_message="不允许执行多条 SQL 语句",
            )

        # Step 1: Check statement type via sqlparse
        type_error = self._check_statement_type(working_sql)
        if type_error:
            return ValidationResult(
                is_valid=False,
                error_type="UNSAFE_SQL",
                error_message=type_error,
            )

        # Step 2: Check forbidden keywords
        forbidden = self._check_forbidden_keywords(working_sql)
        if forbidden:
            return ValidationResult(
                is_valid=False,
                error_type="UNSAFE_SQL",
                error_message=f"SQL 包含禁止的操作关键字: {forbidden}",
            )

        # Step 3: Check lock clauses (ensure snapshot read)
        lock_clause = self._check_lock_clauses(working_sql)
        if lock_clause:
            return ValidationResult(
                is_valid=False,
                error_type="UNSAFE_SQL",
                error_message=f"不允许使用锁子句，只允许快照读: {lock_clause}",
            )

        # Step 4: Extract and validate table names
        tables = self._extract_table_names(working_sql)
        allowed_set = set(self._allowed_tables)
        if not allowed_set:
            # 白名单为空时跳过表名验证（由配置决定是否启用）
            pass
        else:
            forbidden_tables = tables - allowed_set
            if forbidden_tables:
                return ValidationResult(
                    is_valid=False,
                    error_type="FORBIDDEN_TABLE",
                    error_message=f"SQL 引用了非白名单表: {', '.join(sorted(forbidden_tables))}",
                )

        return ValidationResult(
            is_valid=True,
            sanitized_sql=working_sql,
        )

    def ensure_limit(self, sql: str, max_rows: int = 100) -> str:
        """确保 SQL 包含 LIMIT 子句。

        - 如果 SQL 没有 LIMIT，追加 LIMIT max_rows
        - 如果 SQL 的 LIMIT > max_rows，替换为 max_rows
        - 如果 SQL 的 LIMIT <= max_rows，保留原值

        Args:
            sql: SQL 查询语句（应已通过 validate 验证）。
            max_rows: 最大返回行数。

        Returns:
            带有 LIMIT 子句的 SQL。
        """
        sql = sql.strip().rstrip(";").strip()

        match = self._LIMIT_PATTERN.search(sql)
        if match:
            current_limit = int(match.group(1))
            if current_limit > max_rows:
                # Replace the existing LIMIT with max_rows
                sql = sql[: match.start()] + f"LIMIT {max_rows}"
            # else: keep original limit
            return sql

        # No LIMIT found — append one
        return f"{sql} LIMIT {max_rows}"
