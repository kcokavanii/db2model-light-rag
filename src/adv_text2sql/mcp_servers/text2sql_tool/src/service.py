"""Adaptive Text-to-SQL application service used by the MCP entrypoint."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from .context_modes import (
    ContextMode,
    RoutingDecision,
    choose_auto_mode,
    format_compact_context,
    format_semantic_context,
    format_structural_context,
    validate_read_only_sql,
)
from .prompts import MCP_SQL_PROMPT_TEMPLATE, MCP_SYSTEM_PROMPT_TEMPLATE
from .text2sql_implementation import Text2SQLGenerator


log = logging.getLogger(__name__)
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class UsageTrackingClient:
    """Track usage while preserving the AutoGen model-client interface."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.total_usage = {key: 0 for key in TOKEN_KEYS}

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        response = await self.client.create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is None:
            return response

        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = usage.get("total_tokens")
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0))
            completion = int(getattr(usage, "completion_tokens", 0))
            total = getattr(usage, "total_tokens", None)

        self.total_usage["prompt_tokens"] += prompt
        self.total_usage["completion_tokens"] += completion
        self.total_usage["total_tokens"] += int(
            total if total is not None else prompt + completion
        )
        return response

    def get_usage(self) -> dict[str, int]:
        return dict(self.total_usage)

    async def close(self) -> None:
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_result = close_client()
            if inspect.isawaitable(close_result):
                await close_result


def _usage(client: Any) -> dict[str, int]:
    get_usage = getattr(client, "get_usage", None)
    if not callable(get_usage):
        return {key: 0 for key in TOKEN_KEYS}
    raw_usage = get_usage()
    if not isinstance(raw_usage, Mapping):
        return {key: 0 for key in TOKEN_KEYS}
    return {key: int(raw_usage.get(key, 0)) for key in TOKEN_KEYS}


def _usage_delta(
    after: dict[str, int],
    before: dict[str, int],
) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in TOKEN_KEYS
    }


@dataclass
class Attempt:
    """One target-model attempt with a concrete schema-context mode."""

    mode: ContextMode
    status: str
    query: str | None
    failure_reason: str | None
    context_tokens_cl100k: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "context_tokens_cl100k": self.context_tokens_cl100k,
        }


class AdaptiveText2SQLService:
    """Generate SQL using explicit modes or a cheap-first adaptive cascade.

    Automatic mode has a strict ceiling at ``structural``.  Long semantic V3
    descriptions are available only when a caller explicitly requests
    ``semantic_v3``.
    """

    def __init__(
        self,
        *,
        db_uri: str,
        db_name: str,
        target_client: Any,
        retrieval_model_name: str,
        retrieval_base_url: str,
        retrieval_api_key: str,
        lightrag_storage_dir: Path,
        db_knowledge_path: Path,
        schema_token_threshold: int = 150,
        validate_with_explain: bool = True,
        explain_timeout_ms: int = 3000,
    ) -> None:
        if schema_token_threshold < 0:
            raise ValueError("schema_token_threshold must be non-negative")
        if explain_timeout_ms <= 0:
            raise ValueError("explain_timeout_ms must be positive")

        self.db_name = db_name
        self.target_client = target_client
        self.generator = Text2SQLGenerator(
            db_uri=db_uri,
            llm_client=target_client,
            system_prompt_template=MCP_SYSTEM_PROMPT_TEMPLATE,
            sql_prompt_template=MCP_SQL_PROMPT_TEMPLATE,
        )
        self.retrieval_model_name = retrieval_model_name
        self.retrieval_base_url = retrieval_base_url
        self.retrieval_api_key = retrieval_api_key
        self.lightrag_storage_dir = lightrag_storage_dir.resolve()
        self.db_knowledge_path = db_knowledge_path.resolve()
        self.schema_token_threshold = schema_token_threshold
        self.validate_with_explain = validate_with_explain
        self.explain_timeout_ms = explain_timeout_ms

        self._request_lock = asyncio.Lock()
        self._initialized = False
        self._full_schema = ""
        self._full_schema_tokens = 0
        self._knowledge: dict[str, Any] | None = None
        self._retriever: Any | None = None
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def full_schema_tokens(self) -> int:
        if not self._initialized:
            raise RuntimeError("AdaptiveText2SQLService is not initialized")
        return self._full_schema_tokens

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self.generator.build)
        self._full_schema = self.generator.db_schema
        self._full_schema_tokens = len(self._encoding.encode(self._full_schema))
        self._initialized = True
        log.info(
            "Initialized MCP Text-to-SQL service for %s: full schema=%d cl100k tokens",
            self.db_name,
            self._full_schema_tokens,
        )

    def _load_knowledge(self) -> dict[str, Any]:
        if self._knowledge is not None:
            return self._knowledge
        if not self.db_knowledge_path.is_file():
            raise FileNotFoundError(
                f"Database knowledge artifact not found: {self.db_knowledge_path}"
            )
        with self.db_knowledge_path.open(encoding="utf-8") as file:
            raw_knowledge: Any = json.load(file)
        if not isinstance(raw_knowledge, dict):
            raise ValueError("Database knowledge artifact must be a JSON object")
        self._knowledge = raw_knowledge
        return raw_knowledge

    def _create_retriever(self) -> Any:
        from scripts.query_lightrag import LightRAGRetriever

        return LightRAGRetriever(
            db_name=self.db_name,
            llm_model_name=self.retrieval_model_name,
            llm_base_url=self.retrieval_base_url,
            llm_api_key=self.retrieval_api_key,
            working_dir=self.lightrag_storage_dir,
            verbose=False,
        )

    async def _get_retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = await asyncio.to_thread(self._create_retriever)
        return self._retriever

    async def _retrieve(self, query: str) -> dict[str, Any]:
        retriever = await self._get_retriever()
        result = await retriever.retrieve(query)
        if not isinstance(result, dict):
            raise RuntimeError("LightRAG returned a non-dictionary result")
        return result

    def _retrieval_usage(self) -> dict[str, int]:
        retriever = self._retriever
        if retriever is None:
            return {key: 0 for key in TOKEN_KEYS}
        return _usage(retriever.token_tracker)

    def _context_for_mode(
        self,
        mode: ContextMode,
        retrieval_result: dict[str, Any] | None,
    ) -> str:
        if mode is ContextMode.BASELINE:
            return self._full_schema
        if retrieval_result is None:
            raise ValueError(f"Mode {mode.value!r} requires a retrieval result")
        if mode is ContextMode.COMPACT:
            return format_compact_context(retrieval_result)
        if mode is ContextMode.STRUCTURAL:
            return format_structural_context(
                retrieval_result,
                self._load_knowledge(),
            )
        if mode is ContextMode.SEMANTIC_V3:
            return format_semantic_context(retrieval_result)
        raise ValueError(f"Cannot build context for mode {mode.value!r}")

    def _explain(self, sql: str) -> None:
        timeout = int(self.explain_timeout_ms)
        with self.generator.engine.begin() as connection:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql(
                f"SET LOCAL statement_timeout = '{timeout}ms'"
            )
            connection.exec_driver_sql(f"EXPLAIN (FORMAT JSON) {sql}")

    async def _attempt(
        self,
        *,
        mode: ContextMode,
        user_query: str,
        retrieval_result: dict[str, Any] | None,
    ) -> Attempt:
        try:
            context = self._context_for_mode(mode, retrieval_result)
            self.generator.set_schema_context(context)
            result = await self.generator.query(user_query, max_retries=1)
        except Exception as error:
            log.exception("Text-to-SQL attempt failed in mode %s", mode.value)
            return Attempt(
                mode=mode,
                status="error",
                query=None,
                failure_reason=str(error),
                context_tokens_cl100k=0,
            )

        context_tokens = len(self._encoding.encode(context))
        status = str(result.get("status") or "error")
        if status != "success":
            return Attempt(
                mode=mode,
                status=status,
                query=None,
                failure_reason=(
                    str(
                        result.get("clarification_needed")
                        or "ambiguity check rejected the request"
                    )
                    if status == "ambiguous"
                    else str(result.get("error") or "target generation failed")
                ),
                context_tokens_cl100k=context_tokens,
            )

        sql = str(result.get("query") or "").strip()
        try:
            validate_read_only_sql(sql)
            if self.validate_with_explain:
                await asyncio.to_thread(self._explain, sql)
        except Exception as error:
            return Attempt(
                mode=mode,
                status="error",
                query=None,
                failure_reason=str(error),
                context_tokens_cl100k=context_tokens,
            )

        return Attempt(
            mode=mode,
            status="success",
            query=sql,
            failure_reason=None,
            context_tokens_cl100k=context_tokens,
        )

    @staticmethod
    def _fallback_mode(initial_mode: ContextMode) -> ContextMode | None:
        if initial_mode in {ContextMode.BASELINE, ContextMode.COMPACT}:
            return ContextMode.STRUCTURAL
        if initial_mode is ContextMode.STRUCTURAL:
            return ContextMode.BASELINE
        return None

    @staticmethod
    def _query_with_evidence(question: str, evidence: str | None) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if evidence is None or not evidence.strip():
            return normalized_question
        return (
            f"question: {normalized_question}, "
            f"evidence (may be empty): {evidence.strip()}"
        )

    async def generate_sql(
        self,
        *,
        question: str,
        mode: ContextMode | str = ContextMode.AUTO,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        """Generate a validated SQL query without executing it."""

        requested_mode = ContextMode(mode)
        user_query = self._query_with_evidence(question, evidence)

        async with self._request_lock:
            await self.initialize()
            target_before = _usage(self.target_client)
            retrieval_before = self._retrieval_usage()
            retrieval_result: dict[str, Any] | None = None
            retrieval_attempted = False
            retrieval_error: str | None = None

            async def retrieve_once() -> dict[str, Any] | None:
                nonlocal retrieval_attempted, retrieval_error, retrieval_result
                if retrieval_attempted:
                    return retrieval_result
                retrieval_attempted = True
                try:
                    retrieval_result = await self._retrieve(user_query)
                except Exception as error:
                    retrieval_error = str(error)
                    log.exception("LightRAG retrieval failed for %s", self.db_name)
                return retrieval_result

            routing_decision: RoutingDecision
            if requested_mode is ContextMode.AUTO:
                if self._full_schema_tokens <= self.schema_token_threshold:
                    routing_decision = choose_auto_mode(
                        self._full_schema_tokens,
                        None,
                        self.schema_token_threshold,
                    )
                else:
                    await retrieve_once()
                    if retrieval_result is None:
                        routing_decision = RoutingDecision(
                            mode=ContextMode.BASELINE,
                            reason=(
                                "retrieval failed above the threshold; "
                                "using full-schema coverage"
                            ),
                        )
                    else:
                        try:
                            routing_decision = choose_auto_mode(
                                self._full_schema_tokens,
                                retrieval_result,
                                self.schema_token_threshold,
                            )
                        except ValueError as error:
                            retrieval_error = str(error)
                            routing_decision = RoutingDecision(
                                mode=ContextMode.BASELINE,
                                reason=(
                                    "retrieval was structurally unusable; "
                                    "using full-schema coverage"
                                ),
                            )
            else:
                routing_decision = RoutingDecision(
                    mode=requested_mode,
                    reason=f"mode {requested_mode.value!r} was requested explicitly",
                )
                if requested_mode in {
                    ContextMode.COMPACT,
                    ContextMode.STRUCTURAL,
                    ContextMode.SEMANTIC_V3,
                }:
                    await retrieve_once()

            attempts = [
                await self._attempt(
                    mode=routing_decision.mode,
                    user_query=user_query,
                    retrieval_result=retrieval_result,
                )
            ]

            if requested_mode is ContextMode.AUTO and attempts[0].status != "success":
                fallback_mode = self._fallback_mode(routing_decision.mode)
                if fallback_mode is ContextMode.STRUCTURAL:
                    await retrieve_once()
                    if retrieval_result is None:
                        fallback_mode = None
                if fallback_mode is not None:
                    attempts.append(
                        await self._attempt(
                            mode=fallback_mode,
                            user_query=user_query,
                            retrieval_result=retrieval_result,
                        )
                    )

            final_attempt = attempts[-1]
            target_after = _usage(self.target_client)
            retrieval_after = self._retrieval_usage()
            target_usage = _usage_delta(target_after, target_before)
            retrieval_usage = _usage_delta(retrieval_after, retrieval_before)
            data = (
                retrieval_result.get("data", {})
                if isinstance(retrieval_result, dict)
                else {}
            )
            if not isinstance(data, dict):
                data = {}

            return {
                "status": final_attempt.status,
                "query": final_attempt.query,
                "database": self.db_name,
                "requested_mode": requested_mode.value,
                "initial_mode": routing_decision.mode.value,
                "effective_mode": final_attempt.mode.value,
                "fallback_used": len(attempts) > 1,
                "routing_reason": routing_decision.reason,
                "attempts": [attempt.as_dict() for attempt in attempts],
                "error": (
                    final_attempt.failure_reason
                    if final_attempt.status != "success"
                    else None
                ),
                "metadata": {
                    "router_version": "adaptive_v1_no_semantic_auto",
                    "target_prompt_version": "mcp_v1",
                    "schema_token_threshold": self.schema_token_threshold,
                    "full_schema_tokens_cl100k": self._full_schema_tokens,
                    "retrieval_attempted": retrieval_attempted,
                    "retrieval_error": retrieval_error,
                    "retrieved_entities": len(data.get("entities", [])),
                    "retrieved_relationships": len(
                        data.get("relationships", [])
                    ),
                    "target_usage": target_usage,
                    "retrieval_usage": retrieval_usage,
                    "combined_usage": {
                        key: target_usage[key] + retrieval_usage[key]
                        for key in TOKEN_KEYS
                    },
                    "sql_executed": False,
                    "explain_validated": (
                        self.validate_with_explain
                        and final_attempt.status == "success"
                    ),
                    "semantic_v3_allowed_in_auto": False,
                },
            }

    async def close(self) -> None:
        if self._retriever is not None:
            await self._retriever.close()
        close_target = getattr(self.target_client, "close", None)
        if callable(close_target):
            close_result = close_target()
            if inspect.isawaitable(close_result):
                await close_result
        await asyncio.to_thread(self.generator.engine.dispose)
