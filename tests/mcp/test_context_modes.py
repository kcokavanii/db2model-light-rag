from __future__ import annotations

import copy
import unittest
from typing import Any

from src.adv_text2sql.mcp_servers.text2sql_tool.src.context_modes import (
    ContextMode,
    choose_auto_mode,
    format_compact_context,
    format_semantic_context,
    format_structural_context,
)


FK_KEYWORDS = "foreign key references join"
VALUE_KEYWORDS = "example lookup value column"


def make_retrieval_result() -> dict[str, Any]:
    """Return a small but realistic post-processed LightRAG result."""
    return {
        "status": "success",
        "data": {
            "entities": [
                {
                    "entity_name": "TABLE:accounts",
                    "entity_type": "table",
                    "description": (
                        "TABLE accounts. Long semantic description of customer "
                        "bank accounts that must not leak into structural mode."
                    ),
                },
                {
                    "entity_name": "COL:accounts.balance",
                    "entity_type": "column",
                    "description": (
                        "COLUMN accounts.balance. Data type: numeric. "
                        "Long semantic description of the current balance."
                    ),
                },
                {
                    "entity_name": "COL:accounts.district_id",
                    "entity_type": "column",
                    "description": (
                        "COLUMN accounts.district_id. Data type: bigint. "
                        "Long semantic description of the account district."
                    ),
                },
                {
                    "entity_name": "TABLE:districts",
                    "entity_type": "table",
                    "description": (
                        "TABLE districts. Long semantic description of regions."
                    ),
                },
                {
                    "entity_name": "COL:districts.district_id",
                    "entity_type": "column",
                    "description": (
                        "COLUMN districts.district_id. Data type: bigint. "
                        "Constraints: PRIMARY KEY, NOT NULL."
                    ),
                },
                {
                    "entity_name": "COL:districts.name",
                    "entity_type": "column",
                    "description": (
                        "COLUMN districts.name. Data type: text. "
                        "Long semantic description of the district name."
                    ),
                },
                {
                    "entity_name": "VAL:east_Bohemia@districts.name",
                    "entity_type": "value",
                    "description": "VALUE 'east Bohemia' in column districts.name.",
                },
            ],
            "relationships": [
                {
                    "src_id": "COL:accounts.district_id",
                    "tgt_id": "COL:districts.district_id",
                    "keywords": FK_KEYWORDS,
                    "description": (
                        "Column accounts.district_id references column "
                        "districts.district_id through a foreign key."
                    ),
                },
                {
                    "src_id": "VAL:east_Bohemia@districts.name",
                    "tgt_id": "COL:districts.name",
                    "keywords": VALUE_KEYWORDS,
                    "description": (
                        "Value 'east Bohemia' is an example value of column "
                        "districts.name."
                    ),
                },
            ],
            "chunks": [],
        },
    }


def make_knowledge() -> dict[str, Any]:
    """Return the subset of db_knowledge needed by the structural formatter."""
    return {
        "tables": [
            {
                "table_name": "accounts",
                "primary_keys": ["account_id"],
                "columns": [
                    {"column_name": "account_id", "data_type": "bigint"},
                    {"column_name": "district_id", "data_type": "bigint"},
                    {"column_name": "balance", "data_type": "numeric"},
                    {"column_name": "internal_note", "data_type": "text"},
                ],
                "foreign_keys": [
                    {
                        "column_name": "district_id",
                        "foreign_table": "districts",
                        "foreign_column": "district_id",
                    }
                ],
            },
            {
                "table_name": "districts",
                "primary_keys": ["district_id"],
                "columns": [
                    {"column_name": "district_id", "data_type": "bigint"},
                    {"column_name": "name", "data_type": "text"},
                ],
                "foreign_keys": [],
            },
            {
                "table_name": "audit_log",
                "primary_keys": ["event_id"],
                "columns": [
                    {"column_name": "event_id", "data_type": "bigint"},
                ],
                "foreign_keys": [],
            },
        ]
    }


def make_single_table_result(*, include_value: bool = False) -> dict[str, Any]:
    entities: list[dict[str, Any]] = [
        {
            "entity_name": "TABLE:accounts",
            "entity_type": "table",
            "description": "TABLE accounts.",
        },
        {
            "entity_name": "COL:accounts.balance",
            "entity_type": "column",
            "description": "COLUMN accounts.balance. Data type: numeric.",
        },
    ]
    relationships: list[dict[str, Any]] = []

    if include_value:
        entities.append(
            {
                "entity_name": "VAL:100@accounts.balance",
                "entity_type": "value",
                "description": "VALUE '100' in column accounts.balance.",
            }
        )
        relationships.append(
            {
                "src_id": "VAL:100@accounts.balance",
                "tgt_id": "COL:accounts.balance",
                "keywords": VALUE_KEYWORDS,
                "description": (
                    "Value '100' is an example value of column accounts.balance."
                ),
            }
        )

    return {
        "status": "success",
        "data": {
            "entities": entities,
            "relationships": relationships,
            "chunks": [],
        },
    }


class ContextSerializerTests(unittest.TestCase):
    def test_compact_contains_only_retrieved_tables_columns_and_types(self) -> None:
        context = format_compact_context(make_retrieval_result())

        self.assertEqual(
            context,
            "\n".join(
                [
                    "TABLE accounts",
                    "  - balance (NUMERIC)",
                    "  - district_id (BIGINT)",
                    "",
                    "TABLE districts",
                    "  - district_id (BIGINT)",
                    "  - name (TEXT)",
                ]
            ),
        )
        self.assertNotIn("east Bohemia", context)
        self.assertNotIn("foreign key", context.casefold())
        self.assertNotIn("semantic description", context.casefold())

    def test_semantic_keeps_descriptions_relationships_and_values(self) -> None:
        context = format_semantic_context(make_retrieval_result())

        self.assertIn("Long semantic description", context)
        self.assertIn("Foreign-key relationships:", context)
        self.assertIn("accounts.district_id", context)
        self.assertIn("districts.district_id", context)
        self.assertIn("Relevant database values:", context)
        self.assertIn("east Bohemia", context)

    def test_structural_keeps_structure_and_values_without_semantics(self) -> None:
        context = format_structural_context(
            make_retrieval_result(),
            make_knowledge(),
        )

        self.assertIn("TABLE accounts", context)
        self.assertIn("account_id (BIGINT) [PK]", context)
        self.assertIn("district_id (BIGINT)", context)
        self.assertIn("balance (NUMERIC)", context)
        self.assertIn("TABLE districts", context)
        self.assertIn("district_id (BIGINT) [PK]", context)
        self.assertIn("name (TEXT)", context)
        self.assertIn("accounts.district_id -> districts.district_id", context)
        self.assertIn("districts.name", context)
        self.assertIn("east Bohemia", context)

        self.assertNotIn("Long semantic description", context)
        self.assertNotIn("internal_note", context)
        self.assertNotIn("audit_log", context)

    def test_structural_output_is_deterministic(self) -> None:
        result = make_retrieval_result()
        knowledge = make_knowledge()
        expected = format_structural_context(result, knowledge)

        shuffled_result = copy.deepcopy(result)
        shuffled_result["data"]["entities"].reverse()
        shuffled_result["data"]["relationships"].reverse()
        shuffled_knowledge = copy.deepcopy(knowledge)
        shuffled_knowledge["tables"].reverse()

        self.assertEqual(
            format_structural_context(shuffled_result, shuffled_knowledge),
            expected,
        )

    def test_structural_rejects_column_absent_from_knowledge(self) -> None:
        result = make_single_table_result()
        result["data"]["entities"].append(
            {
                "entity_name": "COL:accounts.hallucinated_column",
                "entity_type": "column",
                "description": (
                    "COLUMN accounts.hallucinated_column. Data type: text."
                ),
            }
        )

        with self.assertRaises(ValueError):
            format_structural_context(result, make_knowledge())


class AutoRoutingTests(unittest.TestCase):
    def test_threshold_is_inclusive_for_baseline(self) -> None:
        for token_count in (0, 149, 150):
            with self.subTest(token_count=token_count):
                decision = choose_auto_mode(
                    full_schema_tokens=token_count,
                    retrieval_result=None,
                    threshold=150,
                )
                self.assertEqual(decision.mode, ContextMode.BASELINE)
                self.assertTrue(decision.reason)

    def test_large_single_table_result_uses_compact(self) -> None:
        decision = choose_auto_mode(
            full_schema_tokens=151,
            retrieval_result=make_single_table_result(),
            threshold=150,
        )

        self.assertEqual(decision.mode, ContextMode.COMPACT)
        self.assertTrue(decision.reason)

    def test_large_multi_table_result_uses_structural(self) -> None:
        result = make_single_table_result()
        result["data"]["entities"].append(
            {
                "entity_name": "TABLE:districts",
                "entity_type": "table",
                "description": "TABLE districts.",
            }
        )

        decision = choose_auto_mode(151, result, threshold=150)

        self.assertEqual(decision.mode, ContextMode.STRUCTURAL)
        self.assertTrue(decision.reason)

    def test_large_result_with_relevant_value_uses_structural(self) -> None:
        decision = choose_auto_mode(
            full_schema_tokens=151,
            retrieval_result=make_single_table_result(include_value=True),
            threshold=150,
        )

        self.assertEqual(decision.mode, ContextMode.STRUCTURAL)
        self.assertTrue(decision.reason)

    def test_large_result_with_fk_uses_structural(self) -> None:
        result = make_single_table_result()
        result["data"]["relationships"].append(
            {
                "src_id": "COL:accounts.district_id",
                "tgt_id": "COL:districts.district_id",
                "keywords": FK_KEYWORDS,
                "description": "Foreign-key relationship.",
            }
        )

        decision = choose_auto_mode(151, result, threshold=150)

        self.assertEqual(decision.mode, ContextMode.STRUCTURAL)

    def test_large_schema_without_retrieval_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            choose_auto_mode(
                full_schema_tokens=151,
                retrieval_result=None,
                threshold=150,
            )

    def test_auto_never_selects_semantic_v3(self) -> None:
        cases = [
            (88, None),
            (568, make_single_table_result()),
            (568, make_single_table_result(include_value=True)),
            (568, make_retrieval_result()),
        ]

        for full_schema_tokens, retrieval_result in cases:
            with self.subTest(full_schema_tokens=full_schema_tokens):
                decision = choose_auto_mode(
                    full_schema_tokens=full_schema_tokens,
                    retrieval_result=retrieval_result,
                    threshold=150,
                )
                self.assertNotEqual(decision.mode, ContextMode.SEMANTIC_V3)
                self.assertIn(
                    decision.mode,
                    {
                        ContextMode.BASELINE,
                        ContextMode.COMPACT,
                        ContextMode.STRUCTURAL,
                    },
                )


if __name__ == "__main__":
    unittest.main()
