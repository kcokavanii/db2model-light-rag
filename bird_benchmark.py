import asyncio
import csv
import json
import os

import tiktoken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate_bird import print_evaluation_report, save_manual_check
from src.adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

load_dotenv(override=True)


def env_value(primary: str, fallback: str | None = None) -> str:
    """Read a required environment value with an optional fallback."""
    value = os.getenv(primary)
    if not value and fallback is not None:
        value = os.getenv(fallback)
    if not value:
        names = f"{primary} or {fallback}" if fallback else primary
        raise RuntimeError(f"Missing required environment variable: {names}")
    return value


db_url = "localhost:5444"

target_model_name = env_value("TARGET_LLM_MODEL_NAME", "LLM_MODEL_NAME")
target_base_url = env_value("TARGET_LLM_BASE_URL", "LLM_BASE_URL")
target_api_key = env_value("TARGET_LLM_API_KEY", "LLM_API_KEY")

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

print(f"Target model: {target_model_name}")

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/bird_large_filtered.json",
    answer_file="./data/bird_large_filtered.json",
    use_evidence=True,
    llm_client=target_client,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

total_usage = benchmark.llm_client.get_usage()
print("Report on the tokens spent: ", total_usage)

# Save manual_check.json for manual verification of EX
save_manual_check(report, output_path="manual_check.json")

# db_schemas.json was created by this benchmark run from the exact schema
# strings injected into the baseline target prompts.
with open("db_schemas.json", encoding="utf-8") as file:
    schemas_by_database: dict[str, str] = json.load(file)

schema_encoding = tiktoken.get_encoding("cl100k_base")
schema_tokens_by_database = {
    db_id: len(schema_encoding.encode(schema))
    for db_id, schema in schemas_by_database.items()
}

database_ids = sorted(report["ex_by_database"])
queries_by_database = {
    db_id: sum(result.get("db_id") == db_id for result in report["results"])
    for db_id in database_ids
}

missing_schemas = set(database_ids) - set(schema_tokens_by_database)
missing_usage = set(database_ids) - set(benchmark.target_prompt_tokens_by_database)
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

attributed_prompt_tokens = sum(benchmark.target_prompt_tokens_by_database.values())
if attributed_prompt_tokens != total_usage["prompt_tokens"]:
    raise RuntimeError(
        "Per-database prompt usage does not match global token usage: "
        f"{attributed_prompt_tokens} != {total_usage['prompt_tokens']}"
    )

overall_target_prompt_per_query = attributed_prompt_tokens / total_queries
overall_schema_tokens = (
    sum(
        schema_tokens_by_database[db_id] * queries_by_database[db_id]
        for db_id in database_ids
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

    for db_id in database_ids:
        queries_count = queries_by_database[db_id]
        target_prompt_total = benchmark.target_prompt_tokens_by_database[db_id]
        target_prompt_per_query = target_prompt_total / queries_count

        writer.writerow(
            [
                db_id,
                f"{report['ex_by_database'][db_id]:.2f}",
                f"{report['ves_by_database'][db_id]:.2f}",
                target_prompt_total,
                f"{target_prompt_per_query:.0f}",
                schema_tokens_by_database[db_id],
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
