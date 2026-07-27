import os
import json
from typing import Dict
from benchmarks.evaluate_bird import run_evaluation, print_evaluation_report, save_manual_check

db_url = os.getenv("BENCHMARK_DB_URL")

def _load_queries(filepath: str = "query_results.json") -> Dict[str, str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

predictions = _load_queries()

report = run_evaluation(predictions, "./data/bird_large_filtered.json", db_url, iterate_num=5)

print_evaluation_report(report)
