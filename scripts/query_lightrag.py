import argparse
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations, islice
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.base import BaseGraphStorage, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

try:
    from scripts.load_to_lightrag import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL_NAME,
        LIGHTRAG_DIR,
        RELATION_KEYWORDS,
    )
except ModuleNotFoundError:
    from load_to_lightrag import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL_NAME,
        LIGHTRAG_DIR,
        RELATION_KEYWORDS,
    )


log = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b")
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.])")
MATCH_SEPARATOR_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
COLUMN_DATA_TYPE_PATTERN = re.compile(
    r"\bData type:\s*(?P<data_type>[^.]+)\.",
    re.IGNORECASE,
)


class LightRAGTokenTracker:
    """Accumulate token usage reported by LightRAG's OpenAI adapter."""

    def __init__(self) -> None:
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def add_usage(self, usage: dict[str, int]) -> None:
        for key in self.total_usage:
            self.total_usage[key] += int(usage.get(key, 0))

    def get_usage(self) -> dict[str, int]:
        return dict(self.total_usage)


def normalize_match_text(text: str) -> str:
    """Normalize text for exact value and schema identifier matching"""
    normalized = MATCH_SEPARATOR_PATTERN.sub(
        " ",
        text.replace("_", " ").casefold(),
    )
    return " ".join(normalized.split())


def contains_normalized_phrase(
    normalized_query: str,
    phrase: str,
) -> bool:
    """Check that a phrase occurs as complete normalized tokens"""
    normalized_phrase = normalize_match_text(phrase)
    if not normalized_phrase:
        return False

    return f" {normalized_phrase} " in f" {normalized_query} "


def filter_value_noise(
    query: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only VALUE entities explicitly mentioned in the query."""
    normalized_query = normalize_match_text(query)
    query_without_dates = DATE_PATTERN.sub(" ", query)

    query_numbers: set[Decimal] = set()
    for match in NUMBER_PATTERN.finditer(query_without_dates):
        try:
            query_numbers.add(Decimal(match.group().replace(",", ".")))
        except InvalidOperation:
            continue

    removed_entity_names: set[str] = set()

    for entity in entities:
        if str(entity.get("entity_type")).lower() != "value":
            continue

        entity_name = str(entity.get("entity_name") or "")
        if not entity_name.startswith("VAL:"):
            continue

        raw_value = entity_name.removeprefix("VAL:").rsplit("@", 1)[0]

        try:
            numeric_value = Decimal(raw_value.replace(",", "."))
        except InvalidOperation:
            value_is_relevant = contains_normalized_phrase(
                normalized_query,
                raw_value,
            )
        else:
            value_is_relevant = numeric_value in query_numbers

        if not value_is_relevant:
            removed_entity_names.add(entity_name)

    filtered_entities = [
        entity
        for entity in entities
        if entity["entity_name"] not in removed_entity_names
    ]
    filtered_relationships = [
        relationship
        for relationship in relationships
        if str(relationship.get("src_id")) not in removed_entity_names
        and str(relationship.get("tgt_id")) not in removed_entity_names
    ]

    return filtered_entities, filtered_relationships


def identifier_is_mentioned(
    normalized_query: str,
    identifier: str,
) -> bool:
    """Check an identifier, including a simple English plural form."""
    normalized_identifier = normalize_match_text(identifier)
    if not normalized_identifier:
        return False

    variants = {
        normalized_identifier,
        f"{normalized_identifier}s",
        f"{normalized_identifier}es",
    }
    return any(f" {variant} " in f" {normalized_query} " for variant in variants)


def find_core_tables(
    query: str,
    entities: list[dict[str, Any]],
    table_by_column: dict[str, str],
    column_by_value: dict[str, str],
    max_fallback_tables: int = 3,
) -> set[str]:
    """Find tables directly supported by the query or retrieval ranking"""
    normalized_query = normalize_match_text(query)
    core_tables: set[str] = set()

    for entity in entities:
        entity_name = str(entity.get("entity_name") or "")
        entity_type = str(entity.get("entity_type") or "").lower()

        if entity_type == "table":
            table_name = entity_name.removeprefix("TABLE:")
            if identifier_is_mentioned(
                normalized_query,
                table_name,
            ):
                core_tables.add(entity_name)

        elif entity_type == "column":
            column_name = entity_name.rsplit(".", 1)[-1]
            if identifier_is_mentioned(
                normalized_query,
                column_name,
            ):
                table_id = table_by_column.get(entity_name)
                if table_id is not None:
                    core_tables.add(table_id)

        elif entity_type == "value":
            column_id = column_by_value.get(entity_name)
            if column_id is not None:
                table_id = table_by_column.get(column_id)
                if table_id is not None:
                    core_tables.add(table_id)

    if core_tables:
        return core_tables

    # Semantic fallback after LightRAG retrieval
    for entity in entities:
        entity_name = str(entity.get("entity_name") or "")
        entity_type = str(entity.get("entity_type") or "").lower()

        if entity_type == "table":
            candidate = entity_name
        elif entity_type == "column":
            candidate = table_by_column.get(entity_name)
        else:
            candidate = None

        if candidate is not None:
            core_tables.add(candidate)

        if len(core_tables) >= max_fallback_tables:
            break

    return core_tables


def entity_from_storage(
    node: dict[str, Any],
) -> dict[str, Any]:
    """Convert a stored LightRAG node to a query-result entity"""
    entity = {
        key: value for key, value in node.items() if key not in {"id", "entity_id"}
    }
    entity["entity_name"] = node["id"]
    return entity


def relationship_from_storage(
    edge: dict[str, Any],
) -> dict[str, Any]:
    """Convert a stored LightRAG edge to a query-result relationship"""
    relationship = {
        key: value for key, value in edge.items() if key not in {"source", "target"}
    }
    relationship["src_id"] = edge["source"]
    relationship["tgt_id"] = edge["target"]
    return relationship


def relationship_key(
    relationship: dict[str, Any],
) -> tuple[str, str]:
    """Return an order-independent relationship key"""
    src_id, tgt_id = sorted(
        (
            str(relationship["src_id"]),
            str(relationship["tgt_id"]),
        )
    )
    return src_id, tgt_id


async def expand_fk_join_paths(
    graph_storage: BaseGraphStorage,
    query: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    max_table_hops: int = 3,
    max_paths_per_pair: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select relevant tables and add their shortest FK paths"""
    stored_nodes = await graph_storage.get_all_nodes()
    stored_edges = await graph_storage.get_all_edges()

    nodes_by_id = {str(node["id"]): node for node in stored_nodes}

    def node_type(node_id: str) -> str:
        node = nodes_by_id.get(node_id, {})
        return str(node.get("entity_type") or "").lower()

    has_column_keyword = RELATION_KEYWORDS["HAS_COLUMN"]
    value_in_keyword = RELATION_KEYWORDS["VALUE_IN"]
    fk_keyword = RELATION_KEYWORDS["FK_REFERENCES"]

    table_by_column: dict[str, str] = {}
    has_column_by_column: dict[str, dict[str, Any]] = {}
    column_by_value: dict[str, str] = {}
    value_in_by_value: dict[str, dict[str, Any]] = {}

    for edge in stored_edges:
        keywords = edge.get("keywords")
        if keywords not in {
            has_column_keyword,
            value_in_keyword,
        }:
            continue

        left = str(edge["source"])
        right = str(edge["target"])

        if keywords == has_column_keyword:
            if node_type(left) == "table" and node_type(right) == "column":
                table_id, column_id = left, right
            elif node_type(right) == "table" and node_type(left) == "column":
                table_id, column_id = right, left
            else:
                continue

            table_by_column[column_id] = table_id
            has_column_by_column[column_id] = edge

        else:
            if node_type(left) == "value" and node_type(right) == "column":
                value_id, column_id = left, right
            elif node_type(right) == "value" and node_type(left) == "column":
                value_id, column_id = right, left
            else:
                continue

            column_by_value[value_id] = column_id
            value_in_by_value[value_id] = edge

    table_graph = nx.Graph()
    table_graph.add_nodes_from(
        sorted(node_id for node_id in nodes_by_id if node_type(node_id) == "table")
    )

    fk_edges_by_table_pair: dict[
        frozenset[str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for edge in sorted(
        stored_edges,
        key=lambda item: (
            str(item["source"]),
            str(item["target"]),
        ),
    ):
        if edge.get("keywords") != fk_keyword:
            continue

        left_column = str(edge["source"])
        right_column = str(edge["target"])

        left_table = table_by_column.get(left_column)
        right_table = table_by_column.get(right_column)

        if left_table is None or right_table is None or left_table == right_table:
            continue

        table_pair = frozenset((left_table, right_table))
        fk_edges_by_table_pair[table_pair].append(edge)
        table_graph.add_edge(left_table, right_table)

    seed_tables = find_core_tables(
        query=query,
        entities=entities,
        table_by_column=table_by_column,
        column_by_value=column_by_value,
    )

    if not seed_tables:
        return entities, relationships

    selected_tables = set(seed_tables)
    selected_table_pairs: set[frozenset[str]] = set()

    for left_table, right_table in combinations(
        sorted(seed_tables),
        2,
    ):
        try:
            paths = islice(
                nx.all_shortest_paths(
                    table_graph,
                    source=left_table,
                    target=right_table,
                ),
                max_paths_per_pair,
            )

            for path in paths:
                if len(path) - 1 > max_table_hops:
                    continue

                selected_tables.update(path)

                for path_left, path_right in zip(
                    path,
                    path[1:],
                ):
                    selected_table_pairs.add(frozenset((path_left, path_right)))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    entities_by_name = {str(entity["entity_name"]): entity for entity in entities}
    relationships_by_key = {
        relationship_key(relationship): relationship for relationship in relationships
    }

    def add_stored_entity(entity_id: str) -> None:
        if entity_id in entities_by_name:
            return

        stored_node = nodes_by_id.get(entity_id)
        if stored_node is not None:
            entities_by_name[entity_id] = entity_from_storage(stored_node)

    def add_stored_relationship(edge: dict[str, Any]) -> None:
        relationship = relationship_from_storage(edge)
        relationships_by_key.setdefault(
            relationship_key(relationship),
            relationship,
        )

    for table_id in sorted(selected_tables):
        add_stored_entity(table_id)

    for table_pair in sorted(
        selected_table_pairs,
        key=lambda pair: tuple(sorted(pair)),
    ):
        for fk_edge in fk_edges_by_table_pair[table_pair]:
            add_stored_relationship(fk_edge)

            for column_id in (
                str(fk_edge["source"]),
                str(fk_edge["target"]),
            ):
                add_stored_entity(column_id)

                has_column_edge = has_column_by_column.get(column_id)
                if has_column_edge is not None:
                    add_stored_relationship(has_column_edge)

    for entity_id, entity in list(entities_by_name.items()):
        if str(entity.get("entity_type")).lower() != "value":
            continue

        column_id = column_by_value.get(entity_id)
        if column_id is None:
            continue

        table_id = table_by_column.get(column_id)
        if table_id is None or table_id not in selected_tables:
            continue

        add_stored_entity(column_id)

        value_in_edge = value_in_by_value.get(entity_id)
        if value_in_edge is not None:
            add_stored_relationship(value_in_edge)

    for entity_id, entity in list(entities_by_name.items()):
        if str(entity.get("entity_type")).lower() != "column":
            continue

        table_id = table_by_column.get(entity_id)
        if table_id not in selected_tables:
            continue

        add_stored_entity(table_id)

        has_column_edge = has_column_by_column.get(entity_id)
        if has_column_edge is not None:
            add_stored_relationship(has_column_edge)

    allowed_entity_names: set[str] = set()

    for entity_id, entity in entities_by_name.items():
        entity_type = str(entity.get("entity_type") or "").lower()

        if entity_type == "table" and entity_id in selected_tables:
            allowed_entity_names.add(entity_id)

        elif (
            entity_type == "column"
            and table_by_column.get(entity_id) in selected_tables
        ):
            allowed_entity_names.add(entity_id)

        elif entity_type == "value":
            column_id = column_by_value.get(entity_id)
            if (
                column_id is not None
                and table_by_column.get(column_id) in selected_tables
            ):
                allowed_entity_names.add(entity_id)

    pruned_entities = [
        entities_by_name[entity_id] for entity_id in sorted(allowed_entity_names)
    ]

    pruned_relationships = [
        relationship
        for relationship in relationships_by_key.values()
        if str(relationship["src_id"]) in allowed_entity_names
        and str(relationship["tgt_id"]) in allowed_entity_names
    ]
    pruned_relationships.sort(
        key=lambda relationship: (
            str(relationship["src_id"]),
            str(relationship["tgt_id"]),
            str(relationship.get("keywords") or ""),
        )
    )

    return pruned_entities, pruned_relationships


def format_subgraph_context(result: dict[str, Any]) -> str:
    """Format a retrieved subgraph as compact schema context for Text-to-SQL."""
    data = result.get("data")
    if not isinstance(data, dict):
        raise ValueError("LightRAG result has no structured data")

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    table_descriptions: dict[str, str] = {}
    columns_by_table: dict[str, list[str]] = defaultdict(list)
    ungrouped_columns: list[str] = []
    value_descriptions: list[str] = []

    for entity in entities:
        entity_name = str(entity.get("entity_name") or "")
        entity_type = str(entity.get("entity_type") or "").lower()
        description = str(entity.get("description") or entity_name).strip()

        if entity_type == "table":
            table_name = entity_name.removeprefix("TABLE:")
            table_descriptions[table_name] = description
        elif entity_type == "column":
            full_name = entity_name.removeprefix("COL:")
            table_name, separator, _ = full_name.partition(".")
            if separator:
                columns_by_table[table_name].append(description)
            else:
                ungrouped_columns.append(description)
        elif entity_type == "value":
            value_descriptions.append(description)

    table_names = sorted(set(table_descriptions) | set(columns_by_table))
    lines = ["Retrieved database schema:"]

    for table_name in table_names:
        lines.append("")
        lines.append(
            table_descriptions.get(
                table_name,
                f"TABLE {table_name}.",
            )
        )
        for column_description in sorted(columns_by_table[table_name]):
            lines.append(f"  - {column_description}")

    if ungrouped_columns:
        lines.extend(("", "Other retrieved columns:"))
        lines.extend(f"  - {description}" for description in sorted(ungrouped_columns))

    fk_descriptions = sorted(
        {
            str(relationship.get("description") or "").strip()
            for relationship in relationships
            if relationship.get("keywords") == RELATION_KEYWORDS["FK_REFERENCES"]
            and str(relationship.get("description") or "").strip()
        }
    )
    if fk_descriptions:
        lines.extend(("", "Foreign-key relationships:"))
        lines.extend(f"  - {description}" for description in fk_descriptions)

    if value_descriptions:
        lines.extend(("", "Relevant database values:"))
        lines.extend(
            f"  - {description}" for description in sorted(set(value_descriptions))
        )

    return "\n".join(lines).strip()


def _parse_retrieved_column(
    entity: dict[str, Any],
) -> tuple[str, str, str]:
    """Return table name, column name and SQL type from a column entity."""
    entity_name = str(entity.get("entity_name") or "")
    full_name = entity_name.removeprefix("COL:")

    table_name, separator, column_name = full_name.partition(".")
    if not separator or not table_name or not column_name:
        raise ValueError(f"Invalid LightRAG column entity name: {entity_name!r}")

    description = str(entity.get("description") or "")
    type_match = COLUMN_DATA_TYPE_PATTERN.search(description)
    if type_match is None:
        raise ValueError(f"Column entity has no data type: {entity_name!r}")

    data_type = type_match.group("data_type").strip().upper()
    if not data_type:
        raise ValueError(f"Column entity has an empty data type: {entity_name!r}")

    return table_name, column_name, data_type


def format_baseline_subgraph_context(result: dict[str, Any]) -> str:
    """Format retrieved entities like the full-schema baseline.

    Only retrieved tables and columns with their SQL types are included.
    Semantic descriptions, constraints, values and FK descriptions are
    intentionally excluded because the baseline context does not contain them.
    """
    data = result.get("data")
    if not isinstance(data, dict):
        raise ValueError("LightRAG result has no structured data")

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("LightRAG entities must be a list")

    retrieved_tables: set[str] = set()
    columns_by_table: dict[str, dict[str, str]] = defaultdict(dict)

    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("LightRAG entity must be a dictionary")

        entity_name = str(entity.get("entity_name") or "")
        entity_type = str(entity.get("entity_type") or "").casefold()

        if entity_type == "table":
            table_name = entity_name.removeprefix("TABLE:")
            if not table_name:
                raise ValueError(
                    f"Invalid LightRAG table entity name: {entity_name!r}"
                )
            retrieved_tables.add(table_name)
        elif entity_type == "column":
            table_name, column_name, data_type = _parse_retrieved_column(entity)
            retrieved_tables.add(table_name)

            existing_type = columns_by_table[table_name].get(column_name)
            if existing_type is not None and existing_type != data_type:
                raise ValueError(
                    "Conflicting types for retrieved column "
                    f"{table_name}.{column_name}: "
                    f"{existing_type!r} and {data_type!r}"
                )

            columns_by_table[table_name][column_name] = data_type

    if not retrieved_tables:
        raise ValueError("LightRAG returned no table or column entities")

    lines: list[str] = []

    for table_name in sorted(retrieved_tables):
        if lines:
            lines.append("")

        lines.append(f"TABLE {table_name}")

        for column_name, data_type in sorted(columns_by_table[table_name].items()):
            lines.append(f"  - {column_name} ({data_type})")

    return "\n".join(lines)


class LightRAGRetriever:
    """Reuse one LightRAG instance for multiple retrieval queries."""

    def __init__(
        self,
        db_name: str,
        llm_model_name: str,
        llm_base_url: str,
        llm_api_key: str,
        working_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.db_name = db_name
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.working_dir = (working_dir or LIGHTRAG_DIR / db_name).resolve()
        self.verbose = verbose
        self.token_tracker = LightRAGTokenTracker()
        self._initialized = False

        if not self.working_dir.exists():
            raise FileNotFoundError(
                "LightRAG storage does not exist. "
                "Run load_to_lightrag.py first: "
                f"{self.working_dir}"
            )

        log.info(
            "Loading SentenceTransformer model from %s",
            EMBEDDING_MODEL_NAME,
        )
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
        )

        async def embedding_func(texts: list[str]):
            return await asyncio.to_thread(
                self.embedding_model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs.pop("token_tracker", None)
            return await openai_complete_if_cache(
                model=self.llm_model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                token_tracker=self.token_tracker,
                **kwargs,
            )

        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                model_name=EMBEDDING_MODEL_NAME,
                func=embedding_func,
            ),
            llm_model_func=llm_model_func,
            llm_model_name=self.llm_model_name,
            embedding_func_max_async=1,
            default_embedding_timeout=300,
        )

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.rag.initialize_storages()
        self._initialized = True

    async def retrieve(self, query: str) -> dict[str, Any]:
        """Retrieve, post-process and validate a database subgraph."""
        await self.initialize()

        result = await self.rag.aquery_data(
            query=query,
            param=QueryParam(
                mode="hybrid",
                top_k=15,
                chunk_top_k=8,
                max_entity_tokens=4000,
                max_relation_tokens=5000,
                max_total_tokens=12000,
                enable_rerank=False,
            ),
        )

        if result.get("status") != "success":
            raise RuntimeError(
                f"LightRAG retrieval failed: {result.get('message', 'unknown error')}"
            )

        data = result.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("LightRAG returned no structured retrieval data")

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        chunks = data.get("chunks", [])

        retrieved_entity_count = len(entities)
        retrieved_relationship_count = len(relationships)

        entities, relationships = filter_value_noise(
            query=query,
            entities=entities,
            relationships=relationships,
        )
        entities, relationships = await expand_fk_join_paths(
            graph_storage=self.rag.chunk_entity_relation_graph,
            query=query,
            entities=entities,
            relationships=relationships,
            max_table_hops=3,
            max_paths_per_pair=3,
        )

        data["entities"] = entities
        data["relationships"] = relationships

        empty_entity_names = [
            entity
            for entity in entities
            if not str(entity.get("entity_name") or "").strip()
        ]
        if empty_entity_names:
            raise RuntimeError("Retrieved subgraph contains empty entity names")

        relationship_endpoints = {
            str(endpoint)
            for relationship in relationships
            for endpoint in (
                relationship.get("src_id"),
                relationship.get("tgt_id"),
            )
            if endpoint
        }

        returned_entity_names = {str(entity["entity_name"]) for entity in entities}
        missing_returned_endpoints = sorted(
            relationship_endpoints - returned_entity_names
        )
        if missing_returned_endpoints:
            raise RuntimeError(
                "Retrieved relationships reference entities "
                "missing from the returned subgraph: "
                f"{missing_returned_endpoints[:5]}"
            )

        existing_endpoints = await self.rag.chunk_entity_relation_graph.has_nodes_batch(
            sorted(relationship_endpoints)
        )
        missing_endpoints = sorted(relationship_endpoints - existing_endpoints)
        if missing_endpoints:
            raise RuntimeError(
                "Retrieved relationships reference entities "
                f"missing from storage: {missing_endpoints[:5]}"
            )

        if self.verbose:
            print("LightRAG retrieval post-processing completed:")
            print(
                "  entities: "
                f"{retrieved_entity_count} retrieved, "
                f"{len(entities)} after filtering and FK expansion"
            )
            print(
                "  relationships: "
                f"{retrieved_relationship_count} retrieved, "
                f"{len(relationships)} after filtering and FK expansion"
            )
            print("LightRAG retrieval validation passed:")
            print(f"  entities: {len(entities)}")
            print(f"  relationships: {len(relationships)}")
            print(f"  chunks: {len(chunks)}")

            subgraph = {
                "entities": entities,
                "relationships": relationships,
            }
            print("\nRetrieved subgraph:")
            print(
                json.dumps(
                    subgraph,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

        return result

    async def close(self) -> None:
        if not self._initialized:
            return
        await self.rag.finalize_storages()
        self._initialized = False


async def query_lightrag(
    db_name: str,
    query: str,
) -> dict[str, Any]:
    """Retrieve and validate a database subgraph from LightRAG storage."""
    load_dotenv(override=True)

    required_env_vars = (
        "LLM_MODEL_NAME",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    )
    missing_env_vars = [name for name in required_env_vars if not os.getenv(name)]
    if missing_env_vars:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}"
        )

    retriever = LightRAGRetriever(
        db_name=db_name,
        llm_model_name=os.environ["LLM_MODEL_NAME"],
        llm_base_url=os.environ["LLM_BASE_URL"],
        llm_api_key=os.environ["LLM_API_KEY"],
        verbose=True,
    )
    try:
        return await retriever.retrieve(query)
    finally:
        await retriever.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Retrieve a relevant database subgraph from LightRAG storage")
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Target database name, for example financial",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language database question",
    )
    args = parser.parse_args()

    asyncio.run(
        query_lightrag(
            db_name=args.db,
            query=args.query,
        )
    )


if __name__ == "__main__":
    main()
