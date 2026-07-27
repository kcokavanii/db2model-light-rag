import os
import asyncio
import csv


from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate_bird import print_evaluation_report, save_manual_check
from src.adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

from dotenv import load_dotenv

load_dotenv(override=True)

db_url = "localhost:5444"

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/bird_large_filtered.json",
    answer_file="./data/bird_large_filtered.json",
    use_evidence=True,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())

# Save manual_check.json for manual verification of EX
save_manual_check(report, output_path="manual_check.json")

# Save baseline.csv with EX, VES and tokens were spent
with open("baseline.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["db_id", "EX_percent", "VES_percent", "prompt_tokens_per_query", "total_queries"])
    
    total_tokens = benchmark.llm_client.get_usage()["prompt_tokens"]
    tokens_per_query = total_tokens / report["total"] if report["total"] > 0 else 0
    
    for db_id in sorted(report["ex_by_database"].keys()):
        ex_percent = report["ex_by_database"][db_id]
        ves_percent = report["ves_by_database"][db_id]
        queries_count = len([r for r in report["results"] if r.get("db_id") == db_id])
        writer.writerow([db_id, f"{ex_percent:.2f}", f"{ves_percent:.2f}", f"{tokens_per_query:.0f}", queries_count])
    
    writer.writerow([
        "OVERALL", 
        f"{report['overall_ex']:.2f}", 
        f"{report['overall_ves']:.2f}", 
        f"{tokens_per_query:.0f}", 
        report["total"]
    ])