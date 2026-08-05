"""
build_db_graph.py

Builds a database knowledge graph from `db_knowledge.json` and `m_schema` files.
The graph contains:
- Table nodes (TABLE:*)
- Column nodes (COL:table.column)
- Value nodes (VAL:value@table.column) - only for lookup/dictionary columns
- Edges: HAS_COLUMN, FK_REFERENCES, VALUE_IN, SEMANTICALLY_RELATED

Uses bge-m3 for embeddings and a Teacher-LLM for one-time semantic 
edge labeling.

Example usage:
    python build_db_graph.py --db financial
    python build_db_graph.py --db financial --no-llm --inspect
"""

import json
import argparse
import sys
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-m3"

SEMANTIC_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_HIGH_CONFIDENCE = 0.92  

BASE_ARTIFACTS_DIR = Path("artifacts")
KNOWLEDGE_DIR = BASE_ARTIFACTS_DIR / "db_knowledge"
SCHEMA_DIR = BASE_ARTIFACTS_DIR / "m_schemas"
OUTPUT_DIR = BASE_ARTIFACTS_DIR / "graphs"


def parse_semantic_schema(semantic_file: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse semantic.txt, created by Teacher-LLM.
    Returns dictionary: {table_name: {"_table": desc, "col_name": desc, ...}}    
    Expected format:
        [TABLE] table_name
        Table description (can be multi-line)
        
        [COLUMN] column_name
        Column description (can be multi-line)
        
        [COLUMN] another_column
        Description of the other column
    """
    descriptions: Dict[str, Dict[str, str]] = {}
    
    if not semantic_file.exists():
        log.warning(f"No semantic scheme found: {semantic_file}")
        return descriptions
    
    content = semantic_file.read_text(encoding="utf-8")
    
    current_table = None
    current_column = None
    # Buffer to store multi-line description
    pending_description_lines: List[str] = []
    
    def flush_pending_description():
        """Saves the cumulative description to the dictionary cell """
        nonlocal pending_description_lines
        if not pending_description_lines or not current_table:
            pending_description_lines = []
            return
        
        desc = " ".join(line.strip() for line in pending_description_lines if line.strip())
        if not desc:
            pending_description_lines = []
            return
        
        if current_column:
            descriptions[current_table][current_column] = desc
        else:
            descriptions[current_table]["_table"] = desc
        
        pending_description_lines = []
    
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        
        if line.startswith("#"):
            continue
        
        if line.startswith("[TABLE]"):
            flush_pending_description()
            current_table = line.replace("[TABLE]", "", 1).strip()
            current_column = None
            descriptions.setdefault(current_table, {"_table": ""})
            
        elif line.startswith("[COLUMN]"):
            flush_pending_description()
            current_column = line.replace("[COLUMN]", "", 1).strip()
            if current_table:
                descriptions[current_table].setdefault(current_column, "")
                
        elif line and current_table:
            pending_description_lines.append(line)
    
    # Last description in file
    flush_pending_description()
    
    if not descriptions:
        log.warning(f"Parser found no descriptions in {semantic_file}")
    else:
        n_tables = len(descriptions)
        n_cols = sum(len(v) - 1 for v in descriptions.values())  # -1 after "_table"
        n_with_desc = sum(
            1 for t_data in descriptions.values()
            for k, v in t_data.items()
            if k != "_table" and v
        )
        log.info(
            f"  Semantic parsing: {n_tables} tables, "
            f"{n_cols} columns, {n_with_desc} with disctiptions"
        )
    
    return descriptions


# ---- Value nodes logic ----

def is_date_or_timestamp(data_type: str | None) -> bool:
    normalized_type = (data_type or "").strip().lower()
    return (
        normalized_type == "date"
        or normalized_type.startswith("timestamp")
    )

def should_create_value_nodes(col_stats: Dict[str, Any], col_name: str, data_type: str) -> bool:
    """
    Determines whether to create value nodes for a given column.
    Skips IDs, dates/timestamps, and high-cardinality numeric columns.
    """
    n_distinct_fraction = col_stats.get("n_distinct_fraction")
    col_lower = col_name.lower()
    
    if col_lower.endswith("_id") or col_lower == "id":
        return False
    
    if is_date_or_timestamp(data_type):
        return False
    
    if n_distinct_fraction is not None:
        if n_distinct_fraction <= 0.05: 
            return True
        elif n_distinct_fraction <= 0.20:
            return bool(col_stats.get("most_common_vals"))
    
    return False


def get_values_to_include(col_stats: Dict[str, Any]) -> List[str]:
    """
    Determines how many values to include as nodes.
    """
    n_distinct_fraction = col_stats.get("n_distinct_fraction")
    most_common_vals = col_stats.get("most_common_vals") or []
    
    # Remove only null values
    most_common_vals = [v for v in most_common_vals if v is not None]
    
    if not most_common_vals:
        return []
    
    if n_distinct_fraction is None:
        return most_common_vals[:10]
    elif n_distinct_fraction <= 0.05:
        return most_common_vals  
    elif n_distinct_fraction <= 0.20:
        return most_common_vals[:20]
    else:
        return most_common_vals[:10]

def sanitize_value_for_id(value: str) -> str:
    """
    Sanitizes a value to be safe for use in XML node ID.
    Handles empty strings, spaces, and special characters.
    """
    if value == "":
        return "EMPTY_STRING"
    elif value.strip() == "":
        return "WHITESPACE"
    else:
        # Replace problematic characters
        return value.replace(" ", "_").replace('"', "'").replace("<", "_").replace(">", "_")

def create_value_node(table_name: str, col_name: str, value: Any, col_type: str) -> Dict:
    """
    Create a value node with safe ID.
    Handles empty strings, spaces, and special characters.
    Preserves the original value for LLM context.
    """
    str_value = "" if value is None else str(value)
    
    safe_id_suffix = sanitize_value_for_id(str_value)
    
    node_id = f"VAL:{safe_id_suffix}@{table_name}.{col_name}"
    full_col_name = f"{table_name}.{col_name}"
    
    return {
        "id": node_id,
        "type": "value",
        "value": str_value,          
        "column": full_col_name, # Full name for schema linking
        "data_type": col_type,
    }

# ---- Build base graph ----

def build_base_graph(
    db_knowledge: Dict[str, Any],
    db_name: str,
    semantic_descriptions: Dict[str, Dict[str, str]],
) -> nx.DiGraph:
    """
    Builds the base graph: tables, columns, values, and foreign keys (FK)
    """
    G = nx.DiGraph()
    G.graph["db_name"] = db_name
    G.graph["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    tables = db_knowledge.get("tables", [])
    
    for table in tables:
        table_name = table["table_name"]
        
        # ---- Table Node ----
        table_node = f"TABLE:{table_name}"
        table_desc = semantic_descriptions.get(table_name, {}).get("_table", "")
        
        G.add_node(
            table_node,
            type="table",
            name=table_name,
            description=table_desc,
            table_type=table.get("table_type", "BASE TABLE"),
        )
        
        # ---- Column Nodes ----
        columns = table.get("columns", [])
        col_stats = table.get("column_statistics", {})
        
        for col in columns:
            col_name = col["column_name"]
            col_node = f"COL:{table_name}.{col_name}"
            col_desc = semantic_descriptions.get(table_name, {}).get(col_name, "")
            
            stats = col_stats.get(col_name, {})
            
            G.add_node(
                col_node,
                type="column",
                name=col_name,
                full_name=f"{table_name}.{col_name}",
                data_type=col.get("data_type", ""),
                is_nullable=col.get("is_nullable", True),
                is_pk=col_name in table.get("primary_keys", []),
                description=col_desc,
                n_distinct=stats.get("n_distinct"),
            )
            
            # TABLE-HAS-COLUMN edge
            G.add_edge(table_node, col_node, relation="HAS_COLUMN")
            
            # ---- Value Nodes (lookups only) ----
            if should_create_value_nodes(stats, col_name, col.get("data_type", "")):
                values_to_include = get_values_to_include(stats)
                
                for val in values_to_include:
                    if val is None:
                        continue

                    node_data = create_value_node(
                        table_name=table_name,
                        col_name=col_name,
                        value=val,
                        col_type=col.get("data_type", ""),
                    )
                    
                    
                    if not G.has_node(node_data["id"]):
                        G.add_node(
                            node_data["id"],
                            type=node_data["type"],
                            value=node_data["value"],
                            column=node_data["column"],
                            data_type=node_data["data_type"],
                        )
                    
                    G.add_edge(node_data["id"], col_node, relation="VALUE_IN")

            # ---- Metadata enrichment for dates ----
            if is_date_or_timestamp(col.get("data_type")):
                histogram = stats.get("histogram_bounds")
                if histogram and isinstance(histogram, str) and histogram.startswith("{"):
                    inner = histogram.strip("{}")
                    if inner:
                        bounds = inner.split(",")
                        if len(bounds) >= 2:
                            min_date = bounds[0].strip().strip('"')
                            max_date = bounds[-1].strip().strip('"')

                            G.nodes[col_node]["min_date"] = min_date
                            G.nodes[col_node]["max_date"] = max_date
                            try:
                                min_dt = datetime.fromisoformat(bounds[0].strip().strip('"'))
                                max_dt = datetime.fromisoformat(bounds[-1].strip().strip('"'))
                                G.nodes[col_node]["date_range_years"] = round((max_dt - min_dt).days / 365.25, 2)
                            except Exception:
                                log.warning(
                                    "Could not parse temporal bounds for %s: %r, %r",
                                    col_node,
                                    min_date,
                                    max_date,
                                )

    # ---- FK Edges ----
    for table in tables:
        table_name = table["table_name"]
        for fk in table.get("foreign_keys", []):
            from_col = f"COL:{table_name}.{fk['column_name']}"
            to_col = f"COL:{fk['foreign_table']}.{fk['foreign_column']}"
            
            if G.has_node(from_col) and G.has_node(to_col):
                G.add_edge(from_col, to_col, relation="FK_REFERENCES")
            else:
                log.warning(f"FK skipped: {from_col} -> {to_col} (node missing)")
    
    return G



def find_semantic_edges_by_name(G: nx.DiGraph) -> List[Tuple[str, str, Dict]]:
    """Find semantic edges by identical column names across different tables."""
    edges = []

    # Blacklist
    GENERIC_NAMES = {"type", "name", "status", "value", "code", "description"}
    
    columns = [
        (node, data) for node, data in G.nodes(data=True)
        if data.get("type") == "column"
    ]
    
    for i, (node1, data1) in enumerate(columns):
        col_name1 = data1["name"]
        table1 = node1.split(".")[0].replace("COL:", "")

        if col_name1.lower() in GENERIC_NAMES:
            continue

        for node2, data2 in columns[i+1:]:
            col_name2 = data2["name"]
            table2 = node2.split(".")[0].replace("COL:", "")
            
            if col_name1 == col_name2 and table1 != table2:
                # Skip if FK already exists
                if G.has_edge(node1, node2) or G.has_edge(node2, node1):
                    continue
                
                edges.append((
                    node1, node2,
                    {
                        "relation": "SEMANTICALLY_RELATED",
                        "reason": "same_column_name",
                        "confidence": 0.9,
                        "source": "heuristic",
                    }
                ))
    
    return edges


def find_semantic_edges_by_embeddings(G: nx.DiGraph, embedding_model: Any,) -> List[Tuple[str, str, Dict]]:
    """Finds semantic edges via column description embeddings (bge-m3)"""
    edges = []
    
    columns_with_desc = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "column":
            desc = data.get("description", "")
            if desc:
                columns_with_desc.append((node, data, desc))
    
    if len(columns_with_desc) < 2:
        return edges
    
    log.info(f"  Computing embeddings for {len(columns_with_desc)} columns...")
    descriptions = [c[2] for c in columns_with_desc]
    embeddings = embedding_model.encode(
        descriptions,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    # Find similar pairs
    n = len(columns_with_desc)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            
            if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                node1, data1, _ = columns_with_desc[i]
                node2, data2, _ = columns_with_desc[j]

                table1 = node1.split(".")[0].replace("COL:", "")
                table2 = node2.split(".")[0].replace("COL:", "")
                col1 = data1["name"]
                col2 = data2["name"]

                # Skip connections inside one table 
                if table1 == table2:
                    continue
                
                # Skip if FK already exists
                if G.has_edge(node1, node2) or G.has_edge(node2, node1):
                    continue

                if col1 != col2:
                    # Skip id
                    if col1.endswith("_id") and col2.endswith("_id"):
                        continue
                    
                    # Approve if the similarity is very high
                    if similarity < 0.95:
                        continue
                
                edges.append((
                    node1, node2,
                    {
                        "relation": "SEMANTICALLY_RELATED",
                        "reason": "semantic_similarity",
                        "similarity": round(similarity, 4),
                        "confidence": round(similarity, 4),
                        "source": "embedding",
                    }
                ))
    
    return edges


def verify_semantic_edges_with_llm(
    uncertain_edges: List[Tuple[str, str, Dict]],
    G: nx.DiGraph,
    client: OpenAI,
    model: str,
) -> List[Tuple[str, str, Dict]]:
    """Teacher-LLM verifies uncertain semantic edges"""
    if not uncertain_edges:
        return []
    
    log.info(f"  Teacher-LLM verifying {len(uncertain_edges)} uncertain edges...")
    
    pairs_info = []
    for node1, node2, data in uncertain_edges:
        d1 = G.nodes[node1]
        d2 = G.nodes[node2]
        
        pairs_info.append({
            "pair_id": len(pairs_info) + 1,
            "column1": d1.get("full_name", node1),
            "description1": d1.get("description", ""),
            "column2": d2.get("full_name", node2),
            "description2": d2.get("description", ""),
            "similarity": data.get("similarity", 0),
        })
    
    prompt = f"""Ты — эксперт по базам данных. Проверь, действительно ли \
следующие пары колонок семантически связаны (описывают одно и то же понятие).

Пары колонок:
{json.dumps(pairs_info, indent=2, ensure_ascii=False)}

Для каждой пары определи:
- is_related: true, если колонки описывают одно и то же
- reason: краткое объяснение

Верни ТОЛЬКО JSON-массив:
[
  {{"pair_id": 1, "is_related": true, "reason": "..."}},
  {{"pair_id": 2, "is_related": false, "reason": "..."}}
]
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — эксперт по БД."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=180,
        )
        
        content = response.choices[0].message.content or ""
        json_match = re.search(r"\[[\s\S]*\]", content)
        
        if not json_match:
            log.warning("Failed to extract JSON from Teacher-LLM response")
            return uncertain_edges
        
        verdicts = json.loads(json_match.group())
        verdict_map = {v["pair_id"]: v for v in verdicts}
        
        verified = []
        for idx, (node1, node2, data) in enumerate(uncertain_edges):
            verdict = verdict_map.get(idx + 1, {})
            
            if verdict.get("is_related", True):
                data["verified_by_llm"] = True
                data["llm_reason"] = verdict.get("reason", "")
                verified.append((node1, node2, data))
            else:
                log.debug(
                    f"  LLM rejected: {node1} <-> {node2}: "
                    f"{verdict.get('reason', '')}"
                )
        
        log.info(f"  LLM verified: {len(verified)}/{len(uncertain_edges)}")
        return verified
        
    except Exception as e:
        log.warning(f"Teacher-LLM error: {e}")
        return uncertain_edges


def add_semantic_edges(G: nx.DiGraph, client: OpenAI = None, llm_model: str = None, verify_all: bool = False, ) -> Dict[str, int]:
    """
    Adds semantic edges using a hybrid approach:
    1. Name heuristics
    2. bge-m3 embeddings
    3. Teacher-LLM verification for uncertain cases
    """
    stats = {"heuristic": 0, "embedding": 0, "llm_verified": 0, "total": 0}
    
    # Heuristics
    log.info("Semantic edges: name heuristics...")
    heuristic_edges = find_semantic_edges_by_name(G)
    G.add_edges_from(heuristic_edges)
    stats["heuristic"] = len(heuristic_edges)
    
    # Embeddings
    log.info("Semantic edges: embeddings (bge-m3)...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_edges = find_semantic_edges_by_embeddings(G, embedding_model)
        stats["embedding"] = len(embedding_edges)
    except Exception as e:
        log.error(f"Embedding error: {e}")
    
    all_candidates = heuristic_edges + embedding_edges
    log.info(f"Total semantic candidates: {len(all_candidates)}")
    
    if verify_all and client and llm_model and all_candidates:
        log.info("Teacher-LLM verifying ALL semantic edges...")
        verified = verify_semantic_edges_with_llm(
            all_candidates, G, client, llm_model
        )
        G.add_edges_from(verified)
        stats["llm_verified"] = len(verified)
        stats["total"] = len(verified)
    else:
        G.add_edges_from(all_candidates)
        stats["total"] = len(all_candidates)
    
    return stats


# ---- Main graph building function ----

def build_db_graph(
    knowledge_file: Path,
    m_schema_file: Path,
    semantic_file: Path,
    output_dir: Path,
    use_llm: bool = True,
) -> nx.DiGraph:
    """Main function to build the graph for a single database."""
    db_name = knowledge_file.stem.replace("_knowledge", "")
    log.info(f"\n{'='*60}")
    log.info(f"Database: {db_name}")
    log.info(f"{'='*60}")
    
    log.info("Loading data...")
    with knowledge_file.open("r", encoding="utf-8") as f:
        db_knowledge = json.load(f)
    
    # Parse semantic descriptions
    log.info("Parsing semantic descriptions...")
    semantic_descriptions = parse_semantic_schema(semantic_file)
    n_desc = sum(len(v) - 1 for v in semantic_descriptions.values())
    log.info(f"  Found: {len(semantic_descriptions)} tables, {n_desc} columns with descriptions")
    
    # Base graph
    log.info("Building base graph...")
    G = build_base_graph(db_knowledge, db_name, semantic_descriptions)
    
    n_tables = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "table")
    n_columns = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "column")
    n_values = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "value")
    n_fk = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "FK_REFERENCES")
    n_val_in = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "VALUE_IN")
    
    log.info(f"  Tables: {n_tables}")
    log.info(f"  Columns: {n_columns}")
    log.info(f"  Value nodes: {n_values}")
    log.info(f"  FK edges: {n_fk}")
    log.info(f"  VALUE_IN edges: {n_val_in}")
    log.info(f"  Total nodes: {G.number_of_nodes()}")
    log.info(f"  Total edges: {G.number_of_edges()}")
    
    # Semantic edges
    log.info("Building semantic edges...")    
    client = None
    llm_model = None    
    if use_llm:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        llm_model = os.getenv("LLM_MODEL_NAME")
        
        if api_key and base_url and llm_model:
            client = OpenAI(api_key=api_key, base_url=base_url)
            log.info(f"  Teacher-LLM: {llm_model}")
        else:
            log.warning("  Teacher-LLM unavailable (missing environment variables)")
    
    semantic_stats = add_semantic_edges(G, client, llm_model)
    
    log.info("  Semantic edges summary:")
    log.info(f"    - heuristics: {semantic_stats['heuristic']}")
    log.info(f"    - embeddings: {semantic_stats['embedding']}")
    log.info(f"    - LLM verified: {semantic_stats['llm_verified']}")
    log.info(f"    - total: {semantic_stats['total']}")

    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pickle (fast loading for LightRAG)
    pickle_file = output_dir / f"{db_name}_graph.pkl"
    with pickle_file.open("wb") as f:
        pickle.dump(G, f)
    log.info(f"Saved pickle: {pickle_file}")
    
    # for Gephi / visualization)
    graphml_file = output_dir / f"{db_name}_graph.graphml"
    try:
        G_clean = nx.DiGraph()
        for node, data in G.nodes(data=True):
            G_clean.add_node(node, **{k: v for k, v in data.items() if v is not None})
        for u, v, data in G.edges(data=True):
            G_clean.add_edge(u, v, **{k: v for k, v in data.items() if v is not None})
        nx.write_graphml(G_clean, graphml_file)
        log.info(f"Saved GraphML: {graphml_file}")
    except Exception as e:
        log.warning(f"GraphML not saved: {e}")
    
    stats = {
        "db_name": db_name,
        "nodes": {
            "total": G.number_of_nodes(),
            "tables": n_tables,
            "columns": n_columns,
            "values": n_values,
        },
        "edges": {
            "total": G.number_of_edges(),
            "has_column": n_columns,
            "fk_references": n_fk,
            "value_in": n_val_in,
            "semantically_related": semantic_stats["total"],
        },
        "semantic_breakdown": semantic_stats,
    }
    
    stats_file = output_dir / f"{db_name}_graph_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Saved statistics: {stats_file}")
    
    return G


# ---- Quick Graph Verification ----

def inspect_graph(G: nx.DiGraph, n_examples: int = 5):
    """Prints a brief summary of the graph for inspection."""
    print(f"\n📊 DB Graph: {G.graph.get('db_name', '?')}")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    
    # Count by type
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"   Nodes by type: {type_counts}")
    
    rel_counts = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    
    print(f"   Edges by type: {rel_counts}")
    
    # Examples of each node type
    print(f"\n   Examples (up to {n_examples}):")
    for node_type in ["table", "column", "value"]:
        examples = [
            (n, d) for n, d in G.nodes(data=True)
            if d.get("type") == node_type
        ][:n_examples]
        
        if examples:
            print(f"\n   [{node_type.upper()}]:")
            for node, data in examples:
                desc = data.get("description", "")[:60]
                if desc:
                    print(f"     - {node}: {desc}...")
                else:
                    print(f"     - {node}")



def main():
    parser = argparse.ArgumentParser(description="Builds a database knowledge graph.")
    parser.add_argument(
        "--db", type=str, required=True,
        help="Target database name (e.g., financial). Requires artifacts/db_knowledge/{db}_knowledge.json to exist."
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Do not use Teacher-LLM for semantic edge verification (fast debug mode, saves tokens)."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print example graph nodes after building."
    )
    parser.add_argument(
        "--verify-all", action="store_true",
        help="Use Teacher-LLM to verify ALL semantic edges (heuristics + embeddings). "
            "Costs more tokens but gives cleaner graph."
    )
    args = parser.parse_args()

    db_name = args.db
    use_llm = not args.no_llm
    do_inspect = args.inspect

    knowledge_file = KNOWLEDGE_DIR / f"{db_name}_knowledge.json"
    m_schema_file = SCHEMA_DIR / f"{db_name}_m_schema.txt"
    semantic_file = SCHEMA_DIR / f"{db_name}_semantic.txt"

    log.info("=" * 60)
    log.info(f"Building knowledge graph for DB: {db_name}")
    log.info(f"Knowledge : {knowledge_file.resolve()}")
    log.info(f"M-Schema  : {m_schema_file.resolve()}")
    log.info(f"Semantic  : {semantic_file.resolve()}")
    log.info(f"Output    : {OUTPUT_DIR.resolve()}")
    log.info(f"LLM       : {'Yes' if use_llm else 'No (heuristics + embeddings only)'}")

    missing = []
    if not knowledge_file.exists():
        missing.append(str(knowledge_file))
    if not m_schema_file.exists():
        missing.append(str(m_schema_file))
    if not semantic_file.exists():
        missing.append(str(semantic_file))

    if missing:
        log.error("Missing required files:")
        for m in missing:
            log.error(f"   - {m}")
        log.error("Tip: First run: python scripts/run_week2_exploration.py --db %s", db_name)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    G = build_db_graph(
        knowledge_file=knowledge_file,
        m_schema_file=m_schema_file,
        semantic_file=semantic_file,
        output_dir=OUTPUT_DIR,
        use_llm=use_llm,
    )

    if do_inspect:
        inspect_graph(G)

    log.info("=" * 60)
    log.info(f"Graph for '{db_name}' built successfully!")
    log.info(f"   Nodes: {G.number_of_nodes()}")
    log.info(f"   Edges: {G.number_of_edges()}")
    log.info(f"   Files saved in: {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
