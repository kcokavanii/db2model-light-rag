"""Run the BIRD Text-to-SQL benchmark with LightRAG schema retrieval.

The script reuses the baseline target-model prompts and evaluator. LightRAG
only replaces the full database schema with a retrieved subgraph context.

Examples:
    uv run --env-file .env python scripts/run_week3_lightrag.py \
        --db financial --question-ids 91 113 180
    uv run --env-file .env python scripts/run_week3_lightrag.py \
        --db financial
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import tiktoken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bird import BenchmarkBIRD  # noqa: E402
from benchmarks.evaluate_bird import (  # noqa: E402
    print_evaluation_report,
    run_evaluation,
    save_manual_check,
)
from scripts.query_lightrag import (  # noqa: E402
    LIGHTRAG_DIR,
    LightRAGRetriever,
    format_subgraph_context,
)
from src.adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (  
    Text2SQLGenerator,
)


log = logging.getLogger(__name__)
DEFAULT_DATASET = PROJECT_ROOT / "data" / "bird_large_filtered.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "benchmarks" / "lightrag"
KEYWORD_CACHE_FILE = "kv_store_llm_response_cache.json"
BASELINE_TARGET_MODEL = "Qwen2.5-Coder-7B-Instruct"


def usage_delta(
    after: dict[str, int],
    before: dict[str, int],
) -> dict[str, int]:
    """Return a non-negative token-usage delta."""
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
    }


def get_usage(client: Any) -> dict[str, int]:
    """Read token usage from a tracking client when available."""
    if hasattr(client, "get_usage"):
        return client.get_usage()
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


class LightRAGText2SQLGenerator(Text2SQLGenerator):
    """Generate SQL with the baseline model using a LightRAG schema context."""

    def __init__(
        self,
        db_uri: str,
        llm_client: Any,
        storage_dir: Path,
        retrieval_model_name: str,
        retrieval_base_url: str,
        retrieval_api_key: str,
        verbose_retrieval: bool = False,
    ) -> None:
        super().__init__(db_uri=db_uri, llm_client=llm_client)
        self.db_name = make_url(db_uri).database or "unknown"
        self.storage_dir = storage_dir
        self.retrieval_model_name = retrieval_model_name
        self.retrieval_base_url = retrieval_base_url
        self.retrieval_api_key = retrieval_api_key
        self.verbose_retrieval = verbose_retrieval
        self.retriever: LightRAGRetriever | None = None
        self.query_records: list[dict[str, Any]] = []

    def build(self) -> None:
        """Prepare one reusable LightRAG session for this database."""
        self.db_schema = "LightRAG retrieves a database-specific schema for each query."
        self.system_prompt = self._create_system_prompt()
        self.retriever = LightRAGRetriever(
            db_name=self.db_name,
            llm_model_name=self.retrieval_model_name,
            llm_base_url=self.retrieval_base_url,
            llm_api_key=self.retrieval_api_key,
            working_dir=self.storage_dir,
            verbose=self.verbose_retrieval,
        )

    async def query(self, user_query: str) -> dict[str, Any]:
        """Retrieve schema context, then run the unchanged baseline query flow."""
        retriever = self.retriever
        if retriever is None:
            raise RuntimeError("LightRAGText2SQLGenerator.build() was not called")

        retrieval_before = retriever.token_tracker.get_usage()
        target_before = get_usage(self.llm_client)

        try:
            retrieval_result = await retriever.retrieve(user_query)
            schema_context = format_subgraph_context(retrieval_result)

            self.db_schema = schema_context
            self.system_prompt = self._create_system_prompt()
            result = await super().query(user_query)
        except Exception as error:
            log.exception(
                "LightRAG Text-to-SQL query failed for database '%s'",
                self.db_name,
            )
            retrieval_result = {"data": {}}
            schema_context = ""
            result = {
                "status": "error",
                "query": "error",
                "error": str(error),
            }

        retrieval_after = retriever.token_tracker.get_usage()
        target_after = get_usage(self.llm_client)
        data = retrieval_result.get("data")
        if not isinstance(data, dict):
            data = {}

        self.query_records.append(
            {
                "db_id": self.db_name,
                "user_query": user_query,
                "status": result.get("status"),
                "generated_query": result.get("query"),
                "error": result.get("error"),
                "schema_context": schema_context,
                "schema_context_chars": len(schema_context),
                "schema_context_tokens_cl100k": len(
                    tiktoken.get_encoding("cl100k_base").encode(schema_context)
                ),
                "entities": len(data.get("entities", [])),
                "relationships": len(data.get("relationships", [])),
                "chunks": len(data.get("chunks", [])),
                "retrieval_usage": usage_delta(
                    retrieval_after,
                    retrieval_before,
                ),
                "target_usage": usage_delta(
                    target_after,
                    target_before,
                ),
            }
        )
        return result

    async def close(self) -> None:
        retriever = self.retriever
        if retriever is not None:
            await retriever.close()


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Temporarily run the unchanged evaluator inside an artifact directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ArtifactBenchmarkBIRD(BenchmarkBIRD):
    """Store the unchanged BIRD evaluator outputs outside baseline files."""

    output_dir: str

    def _dump_db_schemas_json(
        self,
        tool_dict: dict[str, Any],
        output_path: str = "db_schemas.json",
    ) -> None:
        del output_path
        super()._dump_db_schemas_json(
            tool_dict,
            str(Path(self.output_dir) / "db_schemas.json"),
        )

    async def evaluate(self, predictions: dict[str, str]) -> dict[str, Any]:
        output_dir = Path(self.output_dir)
        self._save_predictions(
            predictions,
            str(output_dir / "query_results.json"),
        )
        with working_directory(output_dir):
            return run_evaluation(
                predictions,
                self.answer_file,
                self.db_url,
            )


def load_selected_questions(
    dataset_path: Path,
    db_name: str,
    question_ids: list[int] | None,
) -> list[dict[str, Any]]:
    """Select one database and, optionally, explicit BIRD question IDs."""
    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    database_items = [item for item in dataset if item.get("db_id") == db_name]
    if not database_items:
        raise ValueError(f"Database {db_name!r} is absent from {dataset_path}")

    if not question_ids:
        return database_items

    items_by_id = {int(item["question_id"]): item for item in database_items}
    missing_ids = [
        question_id for question_id in question_ids if question_id not in items_by_id
    ]
    if missing_ids:
        raise ValueError(f"Question IDs are absent for {db_name!r}: {missing_ids}")

    return [items_by_id[question_id] for question_id in question_ids]


def create_output_dir(db_name: str, requested: Path | None) -> Path:
    """Create a new non-overwriting directory for benchmark artifacts."""
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / db_name / timestamp
    else:
        output_dir = requested.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_storage(
    source_storage: Path,
    output_dir: Path,
    reuse_keyword_cache: bool,
) -> Path:
    """Use source storage or create a run-local copy without the LLM cache."""
    source_storage = source_storage.resolve()
    if not source_storage.exists():
        raise FileNotFoundError(f"LightRAG storage does not exist: {source_storage}")

    if reuse_keyword_cache:
        return source_storage

    run_storage = output_dir / "lightrag_storage"
    shutil.copytree(
        source_storage,
        run_storage,
        ignore=shutil.ignore_patterns(KEYWORD_CACHE_FILE, "*.lock"),
    )
    return run_storage


def env_value(primary: str, fallback: str | None = None) -> str:
    """Read a required environment value with an optional fallback."""
    value = os.getenv(primary)
    if not value and fallback is not None:
        value = os.getenv(fallback)
    if not value:
        fallback_note = f" or {fallback}" if fallback else ""
        raise RuntimeError(
            f"Missing required environment variable: {primary}{fallback_note}"
        )
    return value


def git_commit() -> str:
    """Return the checked-out commit for the run configuration."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def git_is_dirty() -> bool:
    """Return whether the benchmark is running with uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def save_summary_csv(
    path: Path,
    db_name: str,
    report: dict[str, Any],
    target_usage: dict[str, int],
    retrieval_usage: dict[str, int],
    query_records: list[dict[str, Any]],
) -> None:
    total_queries = int(report["total"])
    divisor = total_queries if total_queries > 0 else 1
    mean_context_tokens = (
        sum(int(record["schema_context_tokens_cl100k"]) for record in query_records)
        / divisor
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "db_id",
                "EX_percent",
                "VES_percent",
                "target_prompt_tokens_per_query",
                "retrieval_prompt_tokens_per_query",
                "total_prompt_tokens_per_query",
                "context_tokens_cl100k_per_query",
                "total_queries",
            ]
        )
        target_prompt = target_usage["prompt_tokens"] / divisor
        retrieval_prompt = retrieval_usage["prompt_tokens"] / divisor
        writer.writerow(
            [
                db_name,
                f"{report['overall_ex']:.2f}",
                f"{report['overall_ves']:.2f}",
                f"{target_prompt:.0f}",
                f"{retrieval_prompt:.0f}",
                f"{target_prompt + retrieval_prompt:.0f}",
                f"{mean_context_tokens:.0f}",
                total_queries,
            ]
        )


async def run_benchmark(args: argparse.Namespace) -> Path:
    load_dotenv(override=True)

    target_model_name = env_value(
        "TARGET_LLM_MODEL_NAME",
        "LLM_MODEL_NAME",
    )
    target_base_url = env_value(
        "TARGET_LLM_BASE_URL",
        "LLM_BASE_URL",
    )
    target_api_key = env_value(
        "TARGET_LLM_API_KEY",
        "LLM_API_KEY",
    )
    retrieval_model_name = os.getenv("LIGHTRAG_LLM_MODEL_NAME") or target_model_name
    retrieval_base_url = os.getenv("LIGHTRAG_LLM_BASE_URL") or target_base_url
    retrieval_api_key = os.getenv("LIGHTRAG_LLM_API_KEY") or target_api_key

    if not target_model_name.casefold().endswith(BASELINE_TARGET_MODEL.casefold()):
        log.warning(
            "Target model '%s' does not match the recorded baseline model "
            "'%s'. Results will not be a controlled baseline comparison.",
            target_model_name,
            BASELINE_TARGET_MODEL,
        )

    if retrieval_model_name != target_model_name:
        log.warning(
            "Retrieval model '%s' differs from target model '%s'. "
            "Record this as a separate-model system comparison.",
            retrieval_model_name,
            target_model_name,
        )

    dataset_path = args.dataset.resolve()
    selected_questions = load_selected_questions(
        dataset_path=dataset_path,
        db_name=args.db,
        question_ids=args.question_ids,
    )
    output_dir = create_output_dir(args.db, args.output_dir)
    selected_dataset_path = output_dir / "bird_selected.json"
    save_json(selected_dataset_path, selected_questions)

    source_storage = (
        args.storage_dir.resolve()
        if args.storage_dir is not None
        else (PROJECT_ROOT / LIGHTRAG_DIR / args.db).resolve()
    )
    run_storage = prepare_storage(
        source_storage=source_storage,
        output_dir=output_dir,
        reuse_keyword_cache=args.reuse_keyword_cache,
    )

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "database": args.db,
        "source_dataset": str(dataset_path),
        "selected_question_ids": [
            int(item["question_id"]) for item in selected_questions
        ],
        "use_evidence": True,
        "target_model": target_model_name,
        "retrieval_model": retrieval_model_name,
        "embedding_model": "BAAI/bge-m3",
        "source_storage": str(source_storage),
        "run_storage": str(run_storage),
        "keyword_cache": (
            "reused" if args.reuse_keyword_cache else "cold run-local copy"
        ),
        "retrieval": {
            "mode": "hybrid",
            "top_k": 15,
            "chunk_top_k": 8,
            "max_entity_tokens": 4000,
            "max_relation_tokens": 5000,
            "max_total_tokens": 12000,
            "rerank": False,
        },
    }
    save_json(output_dir / "run_config.json", config)

    print(f"Database: {args.db}")
    print(f"Questions: {len(selected_questions)}")
    print(f"Target model: {target_model_name}")
    print(f"Retrieval model: {retrieval_model_name}")
    print(f"Keyword cache: {config['keyword_cache']}")
    print(f"Artifacts: {output_dir}")

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
    benchmark = ArtifactBenchmarkBIRD(
        db_url=args.db_url,
        query_file=str(selected_dataset_path),
        answer_file=str(selected_dataset_path),
        use_evidence=True,
        llm_client=target_client,
        output_dir=str(output_dir),
    )
    tool_factory = partial(
        LightRAGText2SQLGenerator,
        storage_dir=run_storage,
        retrieval_model_name=retrieval_model_name,
        retrieval_base_url=retrieval_base_url,
        retrieval_api_key=retrieval_api_key,
        verbose_retrieval=args.verbose_retrieval,
    )

    tools: dict[str, LightRAGText2SQLGenerator] = {}
    try:
        tools = await benchmark.build(tool_factory)
        predictions = await benchmark.predict(tools)
        report = await benchmark.evaluate(predictions)

        tool = tools[args.db]
        for item, record in zip(
            selected_questions,
            tool.query_records,
        ):
            record["question_id"] = int(item["question_id"])

        target_usage = get_usage(benchmark.llm_client)
        retrieval_usage = (
            tool.retriever.token_tracker.get_usage()
            if tool.retriever is not None
            else get_usage(None)
        )

        print_evaluation_report(report)
        save_manual_check(
            report,
            output_path=str(output_dir / "manual_check.json"),
        )
        save_json(output_dir / "contexts.json", tool.query_records)
        save_json(
            output_dir / "token_usage.json",
            {
                "target": target_usage,
                "retrieval": retrieval_usage,
                "combined": {
                    key: target_usage[key] + retrieval_usage[key]
                    for key in target_usage
                },
                "context_tokenizer": (
                    "cl100k_base proxy; target API usage is authoritative"
                ),
            },
        )
        save_summary_csv(
            path=output_dir / "lightrag.csv",
            db_name=args.db,
            report=report,
            target_usage=target_usage,
            retrieval_usage=retrieval_usage,
            query_records=tool.query_records,
        )
    finally:
        await asyncio.gather(
            *(tool.close() for tool in tools.values()),
            return_exceptions=True,
        )
        await target_client.close()

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run BIRD evaluation with LightRAG-retrieved schema context.")
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Target BIRD database, for example financial",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Source BIRD JSON dataset",
    )
    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="*",
        help="Optional explicit BIRD IDs for a smoke run",
    )
    parser.add_argument(
        "--db-url",
        default="localhost:5444",
        help="PostgreSQL host and port used by the existing evaluator",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        help="Source LightRAG storage; defaults to artifacts/lightrag/<db>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New artifact directory; defaults to a timestamped path",
    )
    parser.add_argument(
        "--reuse-keyword-cache",
        action="store_true",
        help=(
            "Use source storage directly, including cached keyword results. "
            "The default makes a run-local copy without the keyword cache."
        ),
    )
    parser.add_argument(
        "--verbose-retrieval",
        action="store_true",
        help="Print every retrieved entity and relationship",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    output_dir = asyncio.run(run_benchmark(parse_args()))
    print(f"LightRAG benchmark completed: {output_dir}")


if __name__ == "__main__":
    main()
