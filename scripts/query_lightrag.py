import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any, cast

import networkx as nx
from dotenv import load_dotenv
from networkx.algorithms.shortest_paths.generic import shortest_path
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.base import BaseGraphStorage, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from load_to_lightrag import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    LIGHTRAG_DIR,
    RELATION_KEYWORDS,
)


DATE_PATTERN = re.compile(
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"
)
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.])"
)


def filter_numeric_value_noise(
    query: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove numeric value entities not explicitly mentioned in the query"""
    query_without_dates = DATE_PATTERN.sub(" ", query)

    query_numbers: set[Decimal] = set()
    for match in NUMBER_PATTERN.finditer(query_without_dates):
        try:
            query_numbers.add(
                Decimal(match.group().replace(",", "."))
            )
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
            continue

        if numeric_value not in query_numbers:
            removed_entity_names.add(entity_name)

    filtered_entities = [
        entity
        for entity in entities
        if entity["entity_name"] not in removed_entity_names
    ]
    filtered_relationships = [
        relationship
        for relationship in relationships
        if relationship["src_id"] not in removed_entity_names
        and relationship["tgt_id"] not in removed_entity_names
    ]

    return filtered_entities, filtered_relationships


def entity_from_storage(
    node: dict[str, Any],
) -> dict[str, Any]:
    """Convert a stored LightRAG node to a query-result entity"""
    entity = {
        key: value
        for key, value in node.items()
        if key not in {"id", "entity_id"}
    }
    entity["entity_name"] = node["id"]
    return entity


def relationship_from_storage(
    edge: dict[str, Any],
) -> dict[str, Any]:
    """Convert a stored LightRAG edge to a query-result relationship"""
    relationship = {
        key: value
        for key, value in edge.items()
        if key not in {"source", "target"}
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
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    max_table_hops: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add shortest FK paths between tables found by semantic retrieval"""
    stored_nodes = await graph_storage.get_all_nodes()
    stored_edges = await graph_storage.get_all_edges()

    nodes_by_id = {
        str(node["id"]): node
        for node in stored_nodes
    }

    def node_type(node_id: str) -> str:
        node = nodes_by_id.get(node_id, {})
        return str(node.get("entity_type") or "").lower()

    has_column_keyword = RELATION_KEYWORDS["HAS_COLUMN"]
    fk_keyword = RELATION_KEYWORDS["FK_REFERENCES"]

    table_by_column: dict[str, str] = {}
    has_column_by_column: dict[str, dict[str, Any]] = {}

    for edge in stored_edges:
        if edge.get("keywords") != has_column_keyword:
            continue

        left = str(edge["source"])
        right = str(edge["target"])

        if node_type(left) == "table" and node_type(right) == "column":
            table_id, column_id = left, right
        elif node_type(right) == "table" and node_type(left) == "column":
            table_id, column_id = right, left
        else:
            continue

        table_by_column[column_id] = table_id
        has_column_by_column[column_id] = edge

    table_graph = nx.Graph()
    table_graph.add_nodes_from(
        sorted(
            node_id
            for node_id in nodes_by_id
            if node_type(node_id) == "table"
        )
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

        if (
            left_table is None
            or right_table is None
            or left_table == right_table
        ):
            continue

        table_pair = frozenset((left_table, right_table))
        fk_edges_by_table_pair[table_pair].append(edge)
        table_graph.add_edge(left_table, right_table)

    seed_tables = {
        str(entity["entity_name"])
        for entity in entities
        if str(entity.get("entity_type")).lower() == "table"
    }

    if len(seed_tables) < 2:
        seed_tables.update(
            table_by_column[column_id]
            for entity in entities
            if str(entity.get("entity_type")).lower() == "column"
            if (column_id := str(entity["entity_name"]))
            in table_by_column
        )

    selected_tables = set(seed_tables)
    selected_table_pairs: set[frozenset[str]] = set()

    for left_table, right_table in combinations(
        sorted(seed_tables),
        2,
    ):
        try:
            path = cast(
                list[str],
                shortest_path(
                    table_graph,
                    source=left_table,
                    target=right_table,
                ),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if len(path) - 1 > max_table_hops:
            continue

        selected_tables.update(path)

        for path_left, path_right in zip(path, path[1:]):
            selected_table_pairs.add(
                frozenset((path_left, path_right))
            )

    entities_by_name = {
        str(entity["entity_name"]): entity
        for entity in entities
    }
    relationships_by_key = {
        relationship_key(relationship): relationship
        for relationship in relationships
    }

    def add_stored_entity(entity_id: str) -> None:
        if entity_id in entities_by_name:
            return

        stored_node = nodes_by_id.get(entity_id)
        if stored_node is not None:
            entities_by_name[entity_id] = entity_from_storage(
                stored_node
            )

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

    for relationship in relationships_by_key.values():
        add_stored_entity(str(relationship["src_id"]))
        add_stored_entity(str(relationship["tgt_id"]))

    return (
        list(entities_by_name.values()),
        list(relationships_by_key.values()),
    )


async def query_lightrag(
    db_name: str,
    query: str,
) -> dict[str, Any]:
    """Retrieve and validate a database subgraph from LightRAG storage"""
    load_dotenv(override=True)

    required_env_vars = (
        "LLM_MODEL_NAME",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    )
    missing_env_vars = [
        name
        for name in required_env_vars
        if not os.getenv(name)
    ]
    if missing_env_vars:
        raise RuntimeError(
            "Missing required environment variables: "
            f"{', '.join(missing_env_vars)}"
        )

    llm_model_name = os.environ["LLM_MODEL_NAME"]
    llm_base_url = os.environ["LLM_BASE_URL"]
    llm_api_key = os.environ["LLM_API_KEY"]

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
    )

    async def embedding_func(texts: list[str]):
        return await asyncio.to_thread(
            embedding_model.encode,
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
        return await openai_complete_if_cache(
            model=llm_model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=llm_api_key,
            base_url=llm_base_url,
            **kwargs,
        )

    working_dir = LIGHTRAG_DIR / db_name
    if not working_dir.exists():
        raise FileNotFoundError(
            "LightRAG storage does not exist. "
            "Run load_to_lightrag.py first: "
            f"{working_dir}"
        )

    rag = LightRAG(
        working_dir=str(working_dir),
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            model_name=EMBEDDING_MODEL_NAME,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func,
        llm_model_name=llm_model_name,
    )

    await rag.initialize_storages()

    try:
        result = await rag.aquery_data(
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
                "LightRAG retrieval failed: "
                f"{result.get('message', 'unknown error')}"
            )

        data = result.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(
                "LightRAG returned no structured retrieval data"
            )

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        chunks = data.get("chunks", [])

        retrieved_entity_count = len(entities)
        retrieved_relationship_count = len(relationships)

        entities, relationships = filter_numeric_value_noise(
            query=query,
            entities=entities,
            relationships=relationships,
        )
        entities, relationships = await expand_fk_join_paths(
            graph_storage=rag.chunk_entity_relation_graph,
            entities=entities,
            relationships=relationships,
            max_table_hops=3,
        )

        data["entities"] = entities
        data["relationships"] = relationships

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

        empty_entity_names = [
            entity
            for entity in entities
            if not str(entity.get("entity_name") or "").strip()
        ]
        if empty_entity_names:
            raise RuntimeError(
                "Retrieved subgraph contains empty entity names"
            )

        relationship_endpoints = {
            str(endpoint)
            for relationship in relationships
            for endpoint in (
                relationship.get("src_id"),
                relationship.get("tgt_id"),
            )
            if endpoint
        }

        returned_entity_names = {
            str(entity["entity_name"])
            for entity in entities
        }
        missing_returned_endpoints = sorted(
            relationship_endpoints - returned_entity_names
        )
        if missing_returned_endpoints:
            raise RuntimeError(
                "Retrieved relationships reference entities "
                "missing from the returned subgraph: "
                f"{missing_returned_endpoints[:5]}"
            )

        existing_endpoints = (
            await rag.chunk_entity_relation_graph.has_nodes_batch(
                sorted(relationship_endpoints)
            )
        )
        missing_endpoints = sorted(
            relationship_endpoints - existing_endpoints
        )
        if missing_endpoints:
            raise RuntimeError(
                "Retrieved relationships reference entities "
                f"missing from storage: {missing_endpoints[:5]}"
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

    finally:
        await rag.finalize_storages()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve a relevant database subgraph "
            "from LightRAG storage"
        )
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
