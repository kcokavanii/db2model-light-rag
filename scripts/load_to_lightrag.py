import pickle
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
from sentence_transformers import SentenceTransformer

import asyncio
import logging
import time

import networkx as nx
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


BASE_ARTIFACTS_DIR = Path("artifacts")
GRAPH_DIR = BASE_ARTIFACTS_DIR / "graphs"
LIGHTRAG_DIR = BASE_ARTIFACTS_DIR / "lightrag"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

SUPPORTED_RELATIONS = {
    "HAS_COLUMN",
    "FK_REFERENCES",
    "VALUE_IN",
    "SEMANTICALLY_RELATED",
}

RELATION_KEYWORDS = {
    "HAS_COLUMN": "schema table column has column",
    "FK_REFERENCES": "foreign key references join",
    "VALUE_IN": "example lookup value column",
    "SEMANTICALLY_RELATED": "semantic similarity related columns",
}
LIGHTRAG_EXCLUDED_RELATIONS = {
    "SEMANTICALLY_RELATED",
}


def should_load_relationship(
    edge_data: Mapping[str, Any],
) -> bool:
    """Return whether a graph relationship should be loaded into LightRAG"""
    relation = str(edge_data.get("relation") or "").upper()
    return relation not in LIGHTRAG_EXCLUDED_RELATIONS


def create_description_on_type_node(node_data: Mapping[str, Any]) -> str:
    """Create a searchable text description for a database graph node"""
    node_type = str(node_data.get("type", "")).lower()
    semantic_description = str(node_data.get("description") or "").strip()

    if node_type == "table":
        table_name = str(node_data.get("name") or "<unknown>")
        parts = [f"TABLE {table_name}."]

        if semantic_description:
            parts.append(semantic_description)

        return " ".join(parts)

    if node_type == "column":
        full_name = str(
            node_data.get("full_name")
            or node_data.get("name")
            or "<unknown>"
        )
        data_type = str(node_data.get("data_type") or "unknown").lower()

        constraints: list[str] = []

        if node_data.get("is_pk") is True:
            constraints.append("PRIMARY KEY")

        is_nullable = node_data.get("is_nullable")
        if is_nullable is False:
            constraints.append("NOT NULL")
        elif is_nullable is True:
            constraints.append("NULLABLE")

        parts = [
            f"COLUMN {full_name}.",
            f"Data type: {data_type}.",
        ]

        if constraints:
            parts.append(f"Constraints: {', '.join(constraints)}.")

        is_temporal = (
            data_type == "date"
            or data_type.startswith("timestamp")
        )
        min_date = node_data.get("min_date")
        max_date = node_data.get("max_date")

        if is_temporal and min_date and max_date:
            parts.append(
                "Approximate date range from database statistics: "
                f"{min_date} to {max_date}."
            )

        if semantic_description:
            parts.append(semantic_description)

        return " ".join(parts)

    if node_type == "value":
        value = node_data.get("value")
        column = str(node_data.get("column") or "<unknown>")

        if value is None:
            value_text = "<unknown>"
        elif value == "":
            value_text = "<empty string>"
        else:
            value_text = str(value)

        return f"VALUE {value_text!r} in column {column}."

    raise ValueError(f"Unsupported node type: {node_type!r}")


def get_table_name(node_data: Mapping[str, Any]) -> str:
    """Return the owning table name for a table, column, or value node"""
    node_type = str(node_data.get("type", "")).lower()

    if node_type == "table":
        table_name = node_data.get("name")
    elif node_type == "column":
        full_name = str(node_data.get("full_name") or "")
        table_name, separator, _ = full_name.partition(".")
        if not separator:
            table_name = ""
    elif node_type == "value":
        column = str(node_data.get("column") or "")
        table_name, separator, _ = column.partition(".")
        if not separator:
            table_name = ""
    else:
        raise ValueError(
            f"Unsupported node type for source mapping: {node_type!r}"
        )

    table_name = str(table_name or "").strip()
    if not table_name:
        raise ValueError(
            f"Cannot determine table for node data: {dict(node_data)!r}"
        )

    return table_name


def create_source_id(
    db_name: str,
    node_data: Mapping[str, Any],
) -> str:
    """Create the ID of the table chunk owning the node"""
    db_name = db_name.strip()
    if not db_name:
        raise ValueError("db_name must not be empty")

    table_name = get_table_name(node_data)
    return f"{db_name}:table:{table_name}"


def node_display_name(
    node_id: str,
    node_data: Mapping[str, Any],
) -> str:
    """Return a human-readable node name for relationship descriptions"""
    node_type = str(node_data.get("type", "")).lower()

    if node_type == "table":
        return str(node_data.get("name") or node_id)

    if node_type == "column":
        return str(
            node_data.get("full_name")
            or node_data.get("name")
            or node_id
        )

    if node_type == "value":
        value = node_data.get("value")
        if value is None:
            return "<unknown>"
        if value == "":
            return "<empty string>"
        return str(value)

    return node_id


def create_relationship_description(
    src_id: str,
    tgt_id: str,
    edge_data: Mapping[str, Any],
    graph: nx.DiGraph,
) -> str:
    """Create a searchable description for a graph relationship"""
    relation = str(edge_data.get("relation", "")).upper()
    src_data = graph.nodes[src_id]
    tgt_data = graph.nodes[tgt_id]
    src_name = node_display_name(src_id, src_data)
    tgt_name = node_display_name(tgt_id, tgt_data)

    if relation == "HAS_COLUMN":
        return f"Table {src_name} contains column {tgt_name}."

    if relation == "FK_REFERENCES":
        return (
            f"Column {src_name} references column {tgt_name} "
            "through a foreign key."
        )

    if relation == "VALUE_IN":
        return (
            f"Value {src_name!r} is an example value "
            f"of column {tgt_name}."
        )

    if relation == "SEMANTICALLY_RELATED":
        description = (
            f"Column {src_name} is semantically related "
            f"to column {tgt_name}."
        )
        reason = (
            str(edge_data.get("reason") or "")
            .replace("_", " ")
            .strip()
        )
        if reason:
            description += f" Detection reason: {reason}."
        return description

    raise ValueError(f"Unsupported relationship type: {relation!r}")


def get_relationship_weight(edge_data: Mapping[str, Any]) -> float:
    """Return a structural or semantic relationship weight"""
    relation = str(edge_data.get("relation", "")).upper()
    if relation != "SEMANTICALLY_RELATED":
        return 1.0

    raw_weight = edge_data.get(
        "confidence",
        edge_data.get("similarity", 1.0),
    )
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid semantic relationship weight: {raw_weight!r}"
        ) from exc

    if weight < 0:
        raise ValueError(
            f"Relationship weight must be non-negative, got {weight}"
        )

    return weight


def create_relationship(
    src_id: str,
    tgt_id: str,
    edge_data: Mapping[str, Any],
    graph: nx.DiGraph,
    db_name: str,
    file_path: str,
) -> dict[str, Any]:
    """Convert one NetworkX edge into a LightRAG relationship"""
    relation = str(edge_data.get("relation", "")).upper()
    if relation not in SUPPORTED_RELATIONS:
        raise ValueError(f"Unsupported relationship type: {relation!r}")

    return {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "description": create_relationship_description(
            src_id=src_id,
            tgt_id=tgt_id,
            edge_data=edge_data,
            graph=graph,
        ),
        "keywords": RELATION_KEYWORDS[relation],
        "weight": get_relationship_weight(edge_data),
        "source_id": create_source_id(db_name, graph.nodes[src_id]),
        "file_path": file_path,
    }


def create_table_chunks(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    db_name: str,
    file_path: str,
) -> list[dict[str, Any]]:
    """Aggregate entities and relationships into one chunk per table"""
    grouped_entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_relationships: dict[str, list[dict[str, Any]]] = defaultdict(list)
    table_source_ids: set[str] = set()

    for entity in entities:
        source_id = entity["source_id"]
        grouped_entities[source_id].append(entity)
        if str(entity["entity_type"]).lower() == "table":
            table_source_ids.add(source_id)

    for relationship in relationships:
        grouped_relationships[relationship["source_id"]].append(
            relationship
        )

    chunks: list[dict[str, Any]] = []
    entity_type_order = {
        "table": 0,
        "column": 1,
        "value": 2,
    }

    for source_id in sorted(table_source_ids):
        source_entities = sorted(
            grouped_entities[source_id],
            key=lambda entity: (
                entity_type_order.get(
                    str(entity["entity_type"]).lower(),
                    99,
                ),
                entity["entity_name"],
            ),
        )
        source_relationships = sorted(
            grouped_relationships[source_id],
            key=lambda relationship: (
                relationship["src_id"],
                relationship["tgt_id"],
                relationship["keywords"],
            ),
        )

        lines = [
            f"Database: {db_name}",
            f"Source: {source_id}",
            "",
            "Entities:",
        ]
        lines.extend(
            f"- {entity['description']}" for entity in source_entities
        )

        if source_relationships:
            lines.extend(("", "Relationships:"))
            lines.extend(
                f"- {relationship['description']}"
                for relationship in source_relationships
            )

        chunks.append(
            {
                "content": "\n".join(lines),
                "source_id": source_id,
                "file_path": file_path,
            }
        )

    return chunks


def validate_custom_kg(
    custom_kg: Mapping[str, list[dict[str, Any]]],
    graph: nx.DiGraph,
    db_name: str,
) -> dict[str, int]:
    """Validate custom KG integrity and return unique object counts"""
    entities = custom_kg["entities"]
    relationships = custom_kg["relationships"]
    chunks = custom_kg["chunks"]
    errors: list[str] = []

    entity_names = [
        str(entity.get("entity_name") or "").strip()
        for entity in entities
    ]
    entity_name_set = set(entity_names)

    if any(not name for name in entity_names):
        errors.append("empty entity_name found")
    if len(entity_names) != len(entity_name_set):
        errors.append("duplicate entity_name found")
    if entity_name_set != set(graph.nodes):
        errors.append("entity names do not match graph nodes")

    relationship_pairs = {
        (
            str(relationship.get("src_id") or ""),
            str(relationship.get("tgt_id") or ""),
        )
        for relationship in relationships
    }
    expected_relationship_pairs = {
        (str(src_id), str(tgt_id))
        for src_id, tgt_id, edge_data in graph.edges(data=True)
        if should_load_relationship(edge_data)
    }

    if len(relationships) != len(relationship_pairs):
        errors.append("duplicate relationships found")
    if relationship_pairs != expected_relationship_pairs:
        errors.append("relationships do not match included graph edges")

    missing_endpoints = sorted(
        {
            endpoint
            for pair in relationship_pairs
            for endpoint in pair
            if endpoint not in entity_name_set
        }
    )
    if missing_endpoints:
        errors.append(
            "relationship endpoints missing from entities: "
            f"{missing_endpoints[:5]}"
        )

    chunk_source_ids = [
        str(chunk.get("source_id") or "").strip() for chunk in chunks
    ]
    chunk_source_id_set = set(chunk_source_ids)

    if any(not source_id for source_id in chunk_source_ids):
        errors.append("empty chunk source_id found")
    if len(chunk_source_ids) != len(chunk_source_id_set):
        errors.append("duplicate table chunk source_id found")

    expected_chunk_ids = {
        create_source_id(db_name, node_data)
        for _, node_data in graph.nodes(data=True)
        if str(node_data.get("type", "")).lower() == "table"
    }
    if chunk_source_id_set != expected_chunk_ids:
        errors.append("table chunks do not match graph table nodes")

    referenced_source_ids = {
        str(item.get("source_id") or "").strip()
        for item in [*entities, *relationships]
    }
    if "" in referenced_source_ids:
        errors.append("empty entity or relationship source_id found")

    missing_source_ids = sorted(
        referenced_source_ids - chunk_source_id_set
    )
    if missing_source_ids:
        errors.append(
            "source_id has no matching chunk: "
            f"{missing_source_ids[:5]}"
        )

    if len(entities) != graph.number_of_nodes():
        errors.append("entity count does not match NetworkX graph")
    if len(relationships) != len(expected_relationship_pairs):
        errors.append(
            "relationship count does not match included graph edges"
        )
    if len(chunks) != len(expected_chunk_ids):
        errors.append("table chunk count does not match NetworkX graph")

    if errors:
        raise ValueError("Invalid custom_kg: " + "; ".join(errors))

    return {
        "unique_entities": len(entity_name_set),
        "unique_relationships": len(relationship_pairs),
        "unique_table_chunks": len(chunk_source_id_set),
    }


def convert_graph_to_custom_kg(
    graph: nx.DiGraph,
    db_name: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, int],
]:
    """Convert a database NetworkX graph into LightRAG custom KG"""
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(
            f"Expected nx.DiGraph, got {type(graph).__name__}"
        )

    file_path = (GRAPH_DIR / f"{db_name}_graph.pkl").as_posix()
    entities: list[dict[str, Any]] = []

    for node_id, node_data in sorted(
        graph.nodes(data=True),
        key=lambda item: item[0],
    ):
        entities.append(
            {
                "entity_name": node_id,
                "entity_type": str(node_data.get("type") or "unknown"),
                "description": create_description_on_type_node(node_data),
                "source_id": create_source_id(db_name, node_data),
                "file_path": file_path,
            }
        )

    relationships = [
        create_relationship(
            src_id=src_id,
            tgt_id=tgt_id,
            edge_data=edge_data,
            graph=graph,
            db_name=db_name,
            file_path=file_path,
        )
        for src_id, tgt_id, edge_data in sorted(
            graph.edges(data=True),
            key=lambda item: (
                item[0],
                item[1],
                str(item[2].get("relation", "")),
            ),
        )
        if should_load_relationship(edge_data)
    ]

    chunks = create_table_chunks(
        entities=entities,
        relationships=relationships,
        db_name=db_name,
        file_path=file_path,
    )
    custom_kg = {
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
    }
    validation_summary = validate_custom_kg(
        custom_kg=custom_kg,
        graph=graph,
        db_name=db_name,
    )

    return custom_kg, validation_summary


async def load_custom_kg_into_lightrag(
    custom_kg: dict[str, list[dict[str, Any]]],
    db_name: str,
) -> None:
    """Load a pre-built database knowledge graph into LightRAG storage"""
    total_started_at = time.perf_counter()

    log.info(
        "Starting custom KG load for database '%s': "
        "%d chunks, %d entities, %d relationships",
        db_name,
        len(custom_kg["chunks"]),
        len(custom_kg["entities"]),
        len(custom_kg["relationships"]),
    )

    model_started_at = time.perf_counter()
    log.info(
        "Loading embedding model '%s'",
        EMBEDDING_MODEL_NAME,
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
    )

    log.info(
        "Embedding model loaded on device '%s' in %.1f seconds",
        embedding_model.device,
        time.perf_counter() - model_started_at,
    )

    embedding_batch_number = 0

    async def embedding_func(texts: list[str]):
        nonlocal embedding_batch_number
        embedding_batch_number += 1
        batch_started_at = time.perf_counter()

        log.info(
            "Embedding batch %d started: %d texts",
            embedding_batch_number,
            len(texts),
        )

        try:
            embeddings = await asyncio.to_thread(
                embedding_model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            log.exception(
                "Embedding batch %d failed after %.1f seconds",
                embedding_batch_number,
                time.perf_counter() - batch_started_at,
            )
            raise

        log.info(
            "Embedding batch %d completed: %d texts in %.1f seconds",
            embedding_batch_number,
            len(texts),
            time.perf_counter() - batch_started_at,
        )

        return embeddings

    async def llm_model_func_not_used(
        *_args: Any,
        **_kwargs: Any,
    ) -> str:
        raise RuntimeError(
            "LLM must not be called while loading a custom KG"
        )

    working_dir = LIGHTRAG_DIR / db_name

    log.info(
        "Creating LightRAG instance with working directory '%s'",
        working_dir,
    )

    rag = LightRAG(
        working_dir=str(working_dir),
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            model_name=EMBEDDING_MODEL_NAME,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func_not_used,
        embedding_func_max_async=1,
        default_embedding_timeout=300,
    )

    initialization_started_at = time.perf_counter()
    log.info("Initializing LightRAG storages")

    await rag.initialize_storages()

    log.info(
        "LightRAG storages initialized in %.1f seconds",
        time.perf_counter() - initialization_started_at,
    )

    try:
        insertion_started_at = time.perf_counter()
        log.info("Inserting custom KG into LightRAG storages")

        await rag.ainsert_custom_kg(custom_kg)

        log.info(
            "Custom KG inserted successfully in %.1f seconds",
            time.perf_counter() - insertion_started_at,
        )
    except Exception:
        log.exception(
            "Custom KG load failed for database '%s'",
            db_name,
        )
        raise
    finally:
        finalization_started_at = time.perf_counter()
        log.info("Finalizing LightRAG storages")

        await rag.finalize_storages()

        log.info(
            "LightRAG storages finalized in %.1f seconds",
            time.perf_counter() - finalization_started_at,
        )

    log.info(
        "Custom KG load completed for database '%s' in %.1f seconds",
        db_name,
        time.perf_counter() - total_started_at,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a database graph to LightRAG custom KG")
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help=(
            "Target database name. Requires "
            "artifacts/graphs/{db}_graph.pkl to exist."
        ),
    )
    args = parser.parse_args()

    db_name = args.db
    graph_file = GRAPH_DIR / f"{db_name}_graph.pkl"

    if not graph_file.exists():
        raise FileNotFoundError(f"Database graph not found: {graph_file}")

    with graph_file.open("rb") as file:
        graph = pickle.load(file)

    custom_kg, summary = convert_graph_to_custom_kg(
        graph=graph,
        db_name=db_name,
    )

    print("Custom KG validation passed:")
    print(f"  unique entities: {summary['unique_entities']}")
    print(f"  unique relationships: {summary['unique_relationships']}")
    print(f"  unique table chunks: {summary['unique_table_chunks']}")
    print(
        "  prepared objects: "
        f"{len(custom_kg['entities'])} entities, "
        f"{len(custom_kg['relationships'])} relationships, "
        f"{len(custom_kg['chunks'])} chunks"
    )

    asyncio.run(
        load_custom_kg_into_lightrag(
            custom_kg=custom_kg,
            db_name=db_name,
        )
    )


if __name__ == "__main__":
    main()
