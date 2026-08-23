from __future__ import annotations

import asyncio
import unittest
from typing import Any

from src.adv_text2sql.mcp_servers.text2sql_tool.src.context_modes import (
    ContextMode,
)
from src.adv_text2sql.mcp_servers.text2sql_tool.src.service import (
    AdaptiveText2SQLService,
    Attempt,
)


class ZeroUsageClient:
    def get_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class FailingCascadeService(AdaptiveText2SQLService):
    """Exercise orchestration without constructing DB, retriever, or LLM clients."""

    def __init__(self) -> None:
        self.db_name = "test_database"
        self.target_client = ZeroUsageClient()
        self.schema_token_threshold = 150
        self.validate_with_explain = False
        self._initialized = True
        self._full_schema_tokens = 568
        self._retriever = None
        self._request_lock = asyncio.Lock()

        self.retrieval_result: dict[str, Any] = {
            "status": "success",
            "data": {
                "entities": [
                    {
                        "entity_name": "TABLE:accounts",
                        "entity_type": "table",
                        "description": "TABLE accounts.",
                    },
                    {
                        "entity_name": "COL:accounts.balance",
                        "entity_type": "column",
                        "description": (
                            "COLUMN accounts.balance. Data type: numeric."
                        ),
                    },
                ],
                "relationships": [],
                "chunks": [],
            },
        }
        self.retrieve_calls = 0
        self.attempt_modes: list[ContextMode] = []
        self.attempt_retrieval_ids: list[int | None] = []

    async def initialize(self) -> None:
        return None

    async def _retrieve(self, query: str) -> dict[str, Any]:
        self.retrieve_calls += 1
        return self.retrieval_result

    async def _attempt(
        self,
        *,
        mode: ContextMode,
        user_query: str,
        retrieval_result: dict[str, Any] | None,
    ) -> Attempt:
        self.attempt_modes.append(mode)
        self.attempt_retrieval_ids.append(
            id(retrieval_result) if retrieval_result is not None else None
        )
        return Attempt(
            mode=mode,
            status="error",
            query=None,
            failure_reason="synthetic generation failure",
            context_tokens_cl100k=10,
        )


class FallbackPolicyTests(unittest.TestCase):
    def test_fallback_modes_are_bounded_and_never_semantic(self) -> None:
        expected = {
            ContextMode.BASELINE: ContextMode.STRUCTURAL,
            ContextMode.COMPACT: ContextMode.STRUCTURAL,
            ContextMode.STRUCTURAL: ContextMode.BASELINE,
            ContextMode.SEMANTIC_V3: None,
        }

        for initial_mode, fallback_mode in expected.items():
            with self.subTest(initial_mode=initial_mode):
                self.assertEqual(
                    AdaptiveText2SQLService._fallback_mode(initial_mode),
                    fallback_mode,
                )
                self.assertNotEqual(fallback_mode, ContextMode.SEMANTIC_V3)


class AutoCascadeTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_uses_one_retrieval_and_at_most_two_attempts(self) -> None:
        service = FailingCascadeService()

        result = await service.generate_sql(
            question="Show account balances",
            mode=ContextMode.AUTO,
        )

        self.assertEqual(service.retrieve_calls, 1)
        self.assertEqual(
            service.attempt_modes,
            [ContextMode.COMPACT, ContextMode.STRUCTURAL],
        )
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(
            service.attempt_retrieval_ids,
            [id(service.retrieval_result), id(service.retrieval_result)],
        )
        self.assertNotIn(ContextMode.SEMANTIC_V3, service.attempt_modes)

        self.assertEqual(result["requested_mode"], ContextMode.AUTO.value)
        self.assertEqual(result["initial_mode"], ContextMode.COMPACT.value)
        self.assertEqual(result["effective_mode"], ContextMode.STRUCTURAL.value)
        self.assertTrue(result["fallback_used"])
        self.assertTrue(result["metadata"]["retrieval_attempted"])
        self.assertFalse(result["metadata"]["semantic_v3_allowed_in_auto"])


if __name__ == "__main__":
    unittest.main()
