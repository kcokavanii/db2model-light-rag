"""Context serialization, routing, and SQL safety for the MCP server.

The automatic policy intentionally uses only baseline, compact, and structural
contexts.  ``semantic_v3`` remains available as an explicit/manual ablation,
but its long semantic descriptions are never selected by ``auto``.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlglot import Dialects, exp, parse
from sqlglot.errors import ParseError


FK_KEYWORDS = "foreign key references join"
VALUE_DESCRIPTION_PATTERN = re.compile(
    r"^VALUE\s+(?P<value>.+?)\s+in column\s+(?P<column>[^.]+\.[^.]+)\.$",
    re.IGNORECASE,
)


class ContextMode(StrEnum):
    """Schema-context modes exposed by the MCP tool."""

    AUTO = "auto"
    BASELINE = "baseline"
    COMPACT = "compact"
    STRUCTURAL = "structural"
    SEMANTIC_V3 = "semantic_v3"


@dataclass(frozen=True)
class RoutingDecision:
    """A deterministic automatic routing decision and its audit reason."""

    mode: ContextMode
    reason: str


def _result_data(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data")
    if not isinstance(data, dict):
        raise ValueError("LightRAG result has no structured data")
    return data


def _entities(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entities = _result_data(result).get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("LightRAG entities must be a list")
    if not all(isinstance(entity, dict) for entity in raw_entities):
        raise ValueError("Every LightRAG entity must be a dictionary")
    return raw_entities


def _relationships(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_relationships = _result_data(result).get("relationships", [])
    if not isinstance(raw_relationships, list):
        raise ValueError("LightRAG relationships must be a list")
    if not all(isinstance(item, dict) for item in raw_relationships):
        raise ValueError("Every LightRAG relationship must be a dictionary")
    return raw_relationships


def _parse_column_id(entity_id: str) -> tuple[str, str]:
    full_name = entity_id.removeprefix("COL:")
    table_name, separator, column_name = full_name.partition(".")
    if not separator or not table_name or not column_name:
        raise ValueError(f"Invalid column entity ID: {entity_id!r}")
    return table_name, column_name


def _table_names(result: dict[str, Any]) -> set[str]:
    table_names: set[str] = set()
    for entity in _entities(result):
        entity_type = str(entity.get("entity_type") or "").casefold()
        entity_name = str(entity.get("entity_name") or "")
        if entity_type == "table":
            table_name = entity_name.removeprefix("TABLE:")
            if table_name:
                table_names.add(table_name)
        elif entity_type == "column":
            table_name, _ = _parse_column_id(entity_name)
            table_names.add(table_name)
    return table_names


def choose_auto_mode(
    full_schema_tokens: int,
    retrieval_result: dict[str, Any] | None,
    threshold: int = 150,
) -> RoutingDecision:
    """Choose baseline, compact, or structural without an LLM router.

    ``semantic_v3`` is deliberately absent from this policy.  A schema at or
    below the threshold is cheap enough to send in full.  Larger schemas need
    one retrieval result before the policy can distinguish a simple one-table
    context from a context that needs FK paths or literal values.
    """

    if full_schema_tokens < 0:
        raise ValueError("full_schema_tokens must be non-negative")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if full_schema_tokens <= threshold:
        return RoutingDecision(
            mode=ContextMode.BASELINE,
            reason=(
                f"full schema has {full_schema_tokens} tokens, "
                f"not more than threshold {threshold}"
            ),
        )

    if retrieval_result is None:
        raise ValueError("retrieval_result is required above the schema threshold")

    entities = _entities(retrieval_result)
    relationships = _relationships(retrieval_result)
    table_names = _table_names(retrieval_result)
    value_count = sum(
        str(entity.get("entity_type") or "").casefold() == "value"
        for entity in entities
    )
    fk_count = sum(
        str(item.get("keywords") or "").casefold() == FK_KEYWORDS
        for item in relationships
    )

    if not table_names:
        raise ValueError("LightRAG returned no table or column entities")

    if len(table_names) > 1 or fk_count > 0 or value_count > 0:
        return RoutingDecision(
            mode=ContextMode.STRUCTURAL,
            reason=(
                "retrieval needs structural context: "
                f"tables={len(table_names)}, fk_paths={fk_count}, "
                f"literal_values={value_count}"
            ),
        )

    return RoutingDecision(
        mode=ContextMode.COMPACT,
        reason="retrieval contains one table and no FK path or literal value",
    )


def format_compact_context(result: dict[str, Any]) -> str:
    """Reuse the benchmarked compact serializer without importing it eagerly."""

    from scripts.query_lightrag import format_baseline_subgraph_context

    return format_baseline_subgraph_context(result)


def format_semantic_context(result: dict[str, Any]) -> str:
    """Reuse the benchmarked V3 serializer for explicit/manual mode only."""

    from scripts.query_lightrag import format_subgraph_context

    return format_subgraph_context(result)


def _knowledge_tables(knowledge: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tables = knowledge.get("tables")
    if isinstance(raw_tables, list):
        tables: dict[str, dict[str, Any]] = {}
        for table in raw_tables:
            if not isinstance(table, dict):
                raise ValueError("Every db_knowledge table must be a dictionary")
            table_name = str(table.get("table_name") or "")
            if not table_name:
                raise ValueError("db_knowledge table has no table_name")
            tables[table_name] = table
        return tables

    if isinstance(raw_tables, dict):
        tables = {}
        for table_name, table in raw_tables.items():
            if not isinstance(table, dict):
                raise ValueError("Every db_knowledge table must be a dictionary")
            normalized = dict(table)
            normalized.setdefault("table_name", str(table_name))
            tables[str(table_name)] = normalized
        return tables

    raise ValueError("db_knowledge must contain a tables list or dictionary")


def _column_metadata(
    tables: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    columns: dict[tuple[str, str], dict[str, Any]] = {}
    for table_name, table in tables.items():
        raw_columns = table.get("columns", [])
        if not isinstance(raw_columns, list):
            raise ValueError(f"Columns for table {table_name!r} must be a list")
        for column in raw_columns:
            if not isinstance(column, dict):
                raise ValueError("Every db_knowledge column must be a dictionary")
            column_name = str(column.get("column_name") or "")
            if not column_name:
                raise ValueError(f"A column in table {table_name!r} has no name")
            columns[(table_name, column_name)] = column
    return columns


def _parse_literal(value_repr: str) -> str | int | float | bool | None:
    try:
        value = ast.literal_eval(value_repr)
    except (SyntaxError, ValueError):
        return value_repr
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value_repr


def _literal_value(
    entity: dict[str, Any],
) -> tuple[str, str, str | int | float | bool | None] | None:
    description = str(entity.get("description") or "").strip()
    description_match = VALUE_DESCRIPTION_PATTERN.match(description)
    if description_match is not None:
        column = description_match.group("column")
        table_name, separator, column_name = column.partition(".")
        if separator:
            value_repr = description_match.group("value")
            value = _parse_literal(value_repr)
            return table_name, column_name, value

    entity_name = str(entity.get("entity_name") or "")
    raw_value_and_column = entity_name.removeprefix("VAL:")
    raw_value, separator, full_column = raw_value_and_column.rpartition("@")
    if not separator:
        return None
    table_name, column_separator, column_name = full_column.partition(".")
    if not column_separator:
        return None
    decoded_value = raw_value.replace("_", " ")
    value = _parse_literal(decoded_value)
    return table_name, column_name, value


def format_structural_context(
    result: dict[str, Any],
    knowledge: dict[str, Any],
) -> str:
    """Serialize V3 structured data without long semantic descriptions.

    The serializer uses canonical types and constraints from ``db_knowledge``.
    It includes retrieved columns, primary keys for selected tables, returned
    FK join paths, and already-filtered relevant VALUE entities.
    """

    entities = _entities(result)
    relationships = _relationships(result)
    tables = _knowledge_tables(knowledge)
    columns = _column_metadata(tables)

    selected_tables = _table_names(result)
    selected_columns: dict[str, set[str]] = defaultdict(set)

    for entity in entities:
        if str(entity.get("entity_type") or "").casefold() != "column":
            continue
        table_name, column_name = _parse_column_id(
            str(entity.get("entity_name") or "")
        )
        selected_columns[table_name].add(column_name)

    join_paths: set[tuple[str, str, str, str]] = set()
    for relationship in relationships:
        if str(relationship.get("keywords") or "").casefold() != FK_KEYWORDS:
            continue
        left_table, left_column = _parse_column_id(
            str(relationship.get("src_id") or "")
        )
        right_table, right_column = _parse_column_id(
            str(relationship.get("tgt_id") or "")
        )
        selected_tables.update((left_table, right_table))
        selected_columns[left_table].add(left_column)
        selected_columns[right_table].add(right_column)
        join_paths.add((left_table, left_column, right_table, right_column))

    for table_name in selected_tables:
        table = tables.get(table_name)
        if table is None:
            raise ValueError(f"Table {table_name!r} is absent from db_knowledge")
        raw_primary_keys = table.get("primary_keys", [])
        if not isinstance(raw_primary_keys, list):
            raise ValueError(f"Primary keys for {table_name!r} must be a list")
        selected_columns[table_name].update(str(item) for item in raw_primary_keys)

    if not selected_tables:
        raise ValueError("Structural context has no selected tables")

    lines: list[str] = []
    for table_name in sorted(selected_tables):
        if lines:
            lines.append("")
        lines.append(f"TABLE {table_name}")

        table = tables[table_name]
        primary_keys = {str(item) for item in table.get("primary_keys", [])}
        for column_name in sorted(selected_columns[table_name]):
            column = columns.get((table_name, column_name))
            if column is None:
                raise ValueError(
                    f"Column {table_name}.{column_name} is absent from db_knowledge"
                )
            data_type = str(column.get("data_type") or "").upper()
            if not data_type:
                raise ValueError(f"Column {table_name}.{column_name} has no type")
            pk_suffix = " [PK]" if column_name in primary_keys else ""
            lines.append(f"  - {column_name} ({data_type}){pk_suffix}")

    if join_paths:
        lines.extend(("", "JOIN PATHS"))
        for left_table, left_column, right_table, right_column in sorted(join_paths):
            lines.append(
                f"  - {left_table}.{left_column} -> "
                f"{right_table}.{right_column}"
            )

    literal_values: set[tuple[str, str, str | int | float | bool | None]] = set()
    for entity in entities:
        if str(entity.get("entity_type") or "").casefold() != "value":
            continue
        literal = _literal_value(entity)
        if literal is not None and literal[0] in selected_tables:
            literal_values.add(literal)

    if literal_values:
        lines.extend(("", "RELEVANT VALUES"))
        for table_name, column_name, value in sorted(
            literal_values,
            key=lambda item: (item[0], item[1], repr(item[2])),
        ):
            lines.append(f"  - {table_name}.{column_name} = {value!r}")

    return "\n".join(lines)


def validate_read_only_sql(sql: str) -> None:
    """Require exactly one PostgreSQL read-only query expression.

    This is a structural guard, not a proof that arbitrary user-defined
    functions are side-effect free.  The server therefore returns SQL rather
    than executing it and uses a read-only transaction for optional EXPLAIN.
    """

    if not sql.strip():
        raise ValueError("SQL query is empty")

    try:
        statements = [
            statement
            for statement in parse(sql, dialect=Dialects.POSTGRES)
            if statement is not None
        ]
    except ParseError as error:
        raise ValueError(f"Invalid PostgreSQL SQL: {error}") from error

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed")

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ValueError("Only SELECT queries and read-only query expressions are allowed")

    for node in statement.walk():
        if isinstance(node, (exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock)):
            raise ValueError(f"Data-changing SQL is forbidden: {node.key.upper()}")
