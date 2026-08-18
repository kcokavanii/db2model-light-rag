import argparse
import asyncio
import csv
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import tiktoken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate_bird import print_evaluation_report, save_manual_check
from src.adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "bird_large_filtered.json"
DB_URL = "localhost:5444"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-schema BIRD baseline evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Source BIRD JSON dataset.",
    )
    parser.add_argument(
        "--db",
        help="Optional single database ID; omit to run every DB in the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for all run artifacts. Omit to preserve the current "
            "working-directory behavior."
        ),
    )
    return parser.parse_args()


def env_value(primary: str, fallback: str | None = None) -> str:
    """Read a required environment value with an optional fallback."""
    value = os.getenv(primary)
    if not value and fallback is not None:
        value = os.getenv(fallback)
    if not value:
        names = f"{primary} or {fallback}" if fallback else primary
        raise RuntimeError(f"Missing required environment variable: {names}")
    return value


def load_selected_questions(
    dataset_path: Path,
    db_id: str | None,
) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    selected: list[dict[str, Any]] = []
    available_databases: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} is not an object")

        item_db_id = item.get("db_id")
        if not isinstance(item_db_id, str) or not item_db_id:
            raise ValueError(f"Dataset item {index} has invalid db_id")

        available_databases.add(item_db_id)

        if db_id is None or item_db_id == db_id:
            selected.append(item)

    if db_id is not None and db_id not in available_databases:
        raise ValueError(
            f"Database {db_id!r} is absent from {dataset_path}. "
            f"Available databases: {sorted(available_databases)}"
        )

    if not selected:
        raise ValueError("Dataset selection contains no questions")

    seen_question_ids: set[str] = set()
    duplicate_question_ids: set[str] = set()

    for index, item in enumerate(selected):
        if "question_id" not in item:
            raise ValueError(f"Selected dataset item {index} has no question_id")

        question_id = str(item["question_id"])
        if question_id in seen_question_ids:
            duplicate_question_ids.add(question_id)
        seen_question_ids.add(question_id)

    if duplicate_question_ids:
        raise ValueError(
            "Selected dataset contains duplicate question_id values: "
            f"{sorted(duplicate_question_ids)}"
        )

    return selected


def save_selected_questions(
    output_path: Path,
    questions: list[dict[str, Any]],
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(questions, file, ensure_ascii=False, indent=2)


def prepare_output_directory(output_dir: Path | None) -> Path:
    if output_dir is None:
        return Path.cwd()

    resolved_output_dir = output_dir.resolve()

    if resolved_output_dir.exists():
        if not resolved_output_dir.is_dir():
            raise ValueError(
                f"Output path is not a directory: {resolved_output_dir}"
            )
        if any(resolved_output_dir.iterdir()):
            raise ValueError(
                f"Output directory is not empty: {resolved_output_dir}"
            )
    else:
        resolved_output_dir.mkdir(parents=True)

    return resolved_output_dir


@contextmanager
def working_directory(directory: Path) -> Iterator[None]:
    original_directory = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(original_directory)


def save_baseline_csv(
    *,
    report: dict[str, Any],
    benchmark: BenchmarkBIRD,
    total_usage: dict[str, int],
) -> None:
    # db_schemas.json was created by this benchmark run from the exact schema
    # strings injected into the baseline target prompts.
    with open("db_schemas.json", encoding="utf-8") as file:
        schemas_by_database: dict[str, str] = json.load(file)

    schema_encoding = tiktoken.get_encoding("cl100k_base")
    schema_tokens_by_database = {
        database_id: len(schema_encoding.encode(schema))
        for database_id, schema in schemas_by_database.items()
    }

    database_ids = sorted(report["ex_by_database"])
    queries_by_database = {
        database_id: sum(
            result.get("db_id") == database_id for result in report["results"]
        )
        for database_id in database_ids
    }

    missing_schemas = set(database_ids) - set(schema_tokens_by_database)
    missing_usage = set(database_ids) - set(
        benchmark.target_prompt_tokens_by_database
    )
    if missing_schemas or missing_usage:
        raise RuntimeError(
            f"Missing schemas: {sorted(missing_schemas)}; "
            f"missing token usage: {sorted(missing_usage)}"
        )

    total_queries = int(report["total"])
    if total_queries <= 0:
        raise RuntimeError("Benchmark returned no queries")

    if sum(queries_by_database.values()) != total_queries:
        raise RuntimeError("Per-database query counts do not match report total")

    attributed_prompt_tokens = sum(
        benchmark.target_prompt_tokens_by_database.values()
    )
    if attributed_prompt_tokens != total_usage["prompt_tokens"]:
        raise RuntimeError(
            "Per-database prompt usage does not match global token usage: "
            f"{attributed_prompt_tokens} != {total_usage['prompt_tokens']}"
        )

    overall_target_prompt_per_query = attributed_prompt_tokens / total_queries
    overall_schema_tokens = (
        sum(
            schema_tokens_by_database[database_id]
            * queries_by_database[database_id]
            for database_id in database_ids
        )
        / total_queries
    )

    # Keep target API usage and the cl100k schema-size proxy as separate metrics.
    with open("baseline.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "db_id",
                "EX_percent",
                "VES_percent",
                "target_prompt_tokens_total",
                "target_prompt_tokens_per_query",
                "context_tokens_cl100k_per_query",
                "total_queries",
            ]
        )

        for database_id in database_ids:
            queries_count = queries_by_database[database_id]
            target_prompt_total = benchmark.target_prompt_tokens_by_database[
                database_id
            ]
            target_prompt_per_query = target_prompt_total / queries_count

            writer.writerow(
                [
                    database_id,
                    f"{report['ex_by_database'][database_id]:.2f}",
                    f"{report['ves_by_database'][database_id]:.2f}",
                    target_prompt_total,
                    f"{target_prompt_per_query:.0f}",
                    schema_tokens_by_database[database_id],
                    queries_count,
                ]
            )

        writer.writerow(
            [
                "OVERALL",
                f"{report['overall_ex']:.2f}",
                f"{report['overall_ves']:.2f}",
                attributed_prompt_tokens,
                f"{overall_target_prompt_per_query:.0f}",
                f"{overall_schema_tokens:.0f}",
                total_queries,
            ]
        )


async def run_benchmark(args: argparse.Namespace) -> None:
    dataset_path = args.dataset.resolve()
    selected_questions = load_selected_questions(dataset_path, args.db)

    target_model_name = env_value("TARGET_LLM_MODEL_NAME", "LLM_MODEL_NAME")
    target_base_url = env_value("TARGET_LLM_BASE_URL", "LLM_BASE_URL")
    target_api_key = env_value("TARGET_LLM_API_KEY", "LLM_API_KEY")
    output_dir = prepare_output_directory(args.output_dir)

    target_client = OpenAIChatCompletionClient(
        model=target_model_name,
        base_url=target_base_url,
        api_key=target_api_key,
        temperature=0,
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    print(f"Dataset: {dataset_path}")
    print(f"Database filter: {args.db or 'all'}")
    print(f"Questions: {len(selected_questions)}")
    print(f"Target model: {target_model_name}")
    print(f"Output directory: {output_dir}")

    try:
        with TemporaryDirectory(prefix="adv_text2sql_baseline_") as temp_dir:
            selected_dataset_path = Path(temp_dir) / "bird_selected.json"
            save_selected_questions(selected_dataset_path, selected_questions)

            with working_directory(output_dir):
                # Passing the same selected snapshot to generation and evaluation
                # keeps the question set aligned. Generation reads question/evidence;
                # gold SQL is consumed only by the evaluator.
                benchmark = BenchmarkBIRD(
                    db_url=DB_URL,
                    query_file=str(selected_dataset_path),
                    answer_file=str(selected_dataset_path),
                    use_evidence=True,
                    llm_client=target_client,
                )

                report = await benchmark.run(Text2SQLGenerator)

                print_evaluation_report(report)

                if benchmark.llm_client is None:
                    raise RuntimeError("Benchmark did not initialize its LLM client")
                total_usage = benchmark.llm_client.get_usage()
                print("Report on the tokens spent: ", total_usage)

                save_manual_check(report, output_path="manual_check.json")
                save_baseline_csv(
                    report=report,
                    benchmark=benchmark,
                    total_usage=total_usage,
                )
    finally:
        await target_client.close()


def main() -> None:
    args = parse_args()
    load_dotenv(override=True)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
