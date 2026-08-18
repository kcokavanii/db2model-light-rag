from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.engine.reflection import Inspector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "bird_large.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "schema_profiles"
    / "bird_large_schema_profile.csv"
)

FIELDNAMES = (
    "db_id",
    "question_count",
    "tables",
    "columns",
    "foreign_key_constraints",
    "foreign_key_column_pairs",
    "largest_fk_component_tables",
    "schema_chars",
    "context_tokens_cl100k_per_query",
    "schema_sha256",
    "transaction_read_only",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile BIRD PostgreSQL schemas without reading table rows or "
            "calling an LLM."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="BIRD JSON dataset used to discover DB IDs and count questions.",
    )
    parser.add_argument(
        "--db",
        nargs="+",
        dest="db_ids",
        help="Optional subset of database IDs.",
    )
    parser.add_argument(
        "--db-url",
        default=f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5444')}",
        help="PostgreSQL host and port, for example localhost:5444.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=5,
        help="Connection timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_db_url(db_url: str) -> tuple[str, int]:
    host, separator, port_text = db_url.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError(f"Invalid --db-url {db_url!r}; expected format host:port")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid port in --db-url {db_url!r}") from exc

    return host, port


def load_question_counts(dataset_path: Path) -> Counter[str]:
    with dataset_path.open(encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    counts: Counter[str] = Counter()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} is not an object")

        db_id = item.get("db_id")
        if not isinstance(db_id, str) or not db_id:
            raise ValueError(f"Dataset item {index} has invalid db_id")

        counts[db_id] += 1

    return counts


def reflect_baseline_schema(
    inspector: Inspector,
) -> tuple[str, list[str], int]:
    """Mirror Text2SQLGenerator._get_db_schema_light() exactly."""
    schema_parts: list[str] = []

    # Do not sort here: baseline currently uses reflection order.
    tables = inspector.get_table_names(schema="public")
    if not tables:
        raise RuntimeError("No tables found in public schema")

    column_count = 0

    for table in tables:
        schema_parts.append(f"TABLE {table}")

        columns = inspector.get_columns(table, schema="public")
        column_count += len(columns)

        for column in columns:
            column_type = str(column["type"])
            schema_parts.append(f"  - {column['name']} ({column_type})")

        schema_parts.append("")

    return "\n".join(schema_parts).strip(), tables, column_count


def collect_fk_metrics(
    inspector: Inspector,
    tables: list[str],
) -> tuple[int, int, int]:
    table_names = set(tables)
    adjacency = {table: set() for table in tables}
    constraint_count = 0
    column_pair_count = 0

    for table in tables:
        foreign_keys = inspector.get_foreign_keys(table, schema="public")
        constraint_count += len(foreign_keys)

        for foreign_key in foreign_keys:
            constrained_columns = foreign_key.get("constrained_columns") or []
            referred_columns = foreign_key.get("referred_columns") or []
            column_pair_count += min(
                len(constrained_columns),
                len(referred_columns),
            )

            referred_table = foreign_key.get("referred_table")
            if referred_table in table_names:
                adjacency[table].add(referred_table)
                adjacency[referred_table].add(table)

    visited: set[str] = set()
    largest_component = 0

    for start_table in tables:
        if start_table in visited:
            continue

        pending = [start_table]
        visited.add(start_table)
        component_size = 0

        while pending:
            table = pending.pop()
            component_size += 1

            for neighbour in adjacency[table]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)

        largest_component = max(largest_component, component_size)

    return constraint_count, column_pair_count, largest_component


def profile_database(
    *,
    db_id: str,
    question_count: int,
    host: str,
    port: int,
    username: str,
    password: str,
    connect_timeout: int,
    encoding: tiktoken.Encoding,
) -> dict[str, object]:
    url = URL.create(
        "postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=db_id,
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": connect_timeout,
            "options": (
                "-c default_transaction_read_only=on "
                "-c statement_timeout=30000"
            ),
        },
    )

    try:
        # All reflection calls below use this same read-only connection.
        with engine.connect() as connection:
            read_only = connection.execute(
                text("SHOW transaction_read_only")
            ).scalar_one()

            if read_only != "on":
                raise RuntimeError("PostgreSQL connection is not read-only")

            inspector = inspect(connection)
            schema, tables, column_count = reflect_baseline_schema(inspector)
            (
                fk_constraints,
                fk_column_pairs,
                largest_component,
            ) = collect_fk_metrics(inspector, tables)
    finally:
        engine.dispose()

    return {
        "db_id": db_id,
        "question_count": question_count,
        "tables": len(tables),
        "columns": column_count,
        "foreign_key_constraints": fk_constraints,
        "foreign_key_column_pairs": fk_column_pairs,
        "largest_fk_component_tables": largest_component,
        "schema_chars": len(schema),
        "context_tokens_cl100k_per_query": len(encoding.encode(schema)),
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "transaction_read_only": True,
        "error": "",
    }


def make_error_row(
    db_id: str,
    question_count: int,
    error: Exception,
) -> dict[str, object]:
    return {
        "db_id": db_id,
        "question_count": question_count,
        "tables": "",
        "columns": "",
        "foreign_key_constraints": "",
        "foreign_key_column_pairs": "",
        "largest_fk_component_tables": "",
        "schema_chars": "",
        "context_tokens_cl100k_per_query": "",
        "schema_sha256": "",
        "transaction_read_only": "",
        "error": f"{type(error).__name__}: {error}",
    }


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    load_dotenv(override=True)
    args = parse_args()

    username = required_env("DB_USER")
    password = required_env("DB_PASS")
    host, port = parse_db_url(args.db_url)

    question_counts = load_question_counts(args.dataset)

    if args.db_ids:
        db_ids = list(dict.fromkeys(args.db_ids))
        unknown = sorted(set(db_ids) - set(question_counts))
        if unknown:
            raise ValueError(
                f"Requested databases are absent from the dataset: {unknown}"
            )
    else:
        db_ids = sorted(question_counts)

    encoding = tiktoken.get_encoding("cl100k_base")
    rows: list[dict[str, object]] = []

    for db_id in db_ids:
        print(f"Profiling {db_id}...")

        try:
            row = profile_database(
                db_id=db_id,
                question_count=question_counts[db_id],
                host=host,
                port=port,
                username=username,
                password=password,
                connect_timeout=args.connect_timeout,
                encoding=encoding,
            )
        except Exception as error:
            row = make_error_row(
                db_id,
                question_counts[db_id],
                error,
            )
            print(f"  ERROR: {row['error']}", file=sys.stderr)
        else:
            print(
                "  "
                f"tables={row['tables']}, "
                f"columns={row['columns']}, "
                f"schema_tokens={row['context_tokens_cl100k_per_query']}"
            )

        rows.append(row)

    # Successful databases first, largest schema first.
    rows.sort(
        key=lambda row: (
            bool(row["error"]),
            -int(row["context_tokens_cl100k_per_query"] or -1),
            str(row["db_id"]),
        )
    )
    write_csv(args.output, rows)

    failed = [row["db_id"] for row in rows if row["error"]]

    print(f"Saved profile: {args.output.resolve()}")

    if failed:
        print(f"Profiling failed for: {failed}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
