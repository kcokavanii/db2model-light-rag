import os
import json
import logging
import sqlglot
import decimal
import re
import time 
import math
import numpy as np

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict

logger = logging.getLogger(__name__)


def sqlite_to_postgres(query: str) -> str:
    # Strip ```sql``` and comments first if needed
    query = re.sub(r"```sql|```", "", query, flags=re.IGNORECASE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL).strip()

    # Transpile using sqlglot
    try:
        query_pg = sqlglot.transpile(query, read='sqlite', write='postgres')[0]
    except Exception as e:
        print("SQLGlot parse error:", e)
        query_pg = query

    # Fix division by zero
    query_pg = re.sub(r"(\b\w+\b)\s*/\s*(\b\w+\b)", r"\1 / NULLIF(\2,0)", query_pg)

    return query_pg

def clean_abnormal(diff_list: list) -> list:
    """
    Remove outliers from a list using the 3-sigma rule.
    Values deviating from the mean by more than 3 standard deviations are filtered out.
    """
    if len(diff_list) < 3:
        return diff_list
    mean = np.mean(diff_list)
    stdev = np.std(diff_list)
    return [x for x in diff_list if abs(x - mean) <= 3 * stdev]

def save_manual_check(report: dict, output_path: str = "manual_check.json"):
    """
    Saves a human-readable version of the evaluation results 
    for manual verification of EX
    """
    check_data = []
    for r in report["results"]: 
        check_data.append({
            "question_id": r["question_id"],
            "db_id": r.get("db_id", "unknown"),
            "difficulty": r["difficulty"],
            "score": r["score"],
            "ves_reward": r.get("ves_reward", 0.0),
            "time_ratio": r.get("time_ratio", 0.0),
            "gold_sql": r["gold_sql"],
            "predicted_sql": r["predicted_sql"],
            "error": r.get("error", None)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(check_data, f, indent=2, ensure_ascii=False)    
    print(f"Manual check file saved to {output_path}")


def run_evaluation(predictions: Dict[str, str], answer_file: str, db_url: str, iterate_num: int = 5):
    with open(answer_file, "r") as f:
        answer_file = json.load(f)

    gold_queries = {str(item["question_id"]): item for item in answer_file}

    # provide prediction keys to strings for security
    #predictions = {str(k): v for k, v in predictions.items()}

    results = []

    db_username = os.environ["DB_USER"]
    db_password = os.environ["DB_PASS"]

    all_predicted = {}
    all_gold = {}

    for question_id, predicted_sql in predictions.items():
        gold_query = gold_queries[question_id]

        gold_sql = gold_query["SQL"]
        db_id = gold_query["db_id"]
        difficulty = gold_query["difficulty"]

        # ---- ambiguous request processing ----
        if gold_sql == "ambiguous" or predicted_sql == "ambiguous":
            if gold_sql == "ambiguous" and predicted_sql == "ambiguous":
                score = 1
            else:
                score = 0

            results.append(
                {
                    "db_id": db_id,
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                    "ves_reward": 0.0
                }
            )
            continue

        # ---- SQL execution and comparison with gold results ----
        db_uri = (
            f"postgresql+psycopg://{db_username}:{db_password}@{db_url}/{db_id}"
        )

        try:
            # predicted_sql = sqlite_to_postgres(predicted_sql)
            gold_sql = sqlite_to_postgres(gold_sql)
            engine = create_engine(db_uri)

            # ----Isolated calculation EX----
            with engine.connect() as conn:
                # abort queries exceeding 3 seconds to prevent long-running operations
                conn.execute(text("SET statement_timeout = 3000"))
                gold_res = conn.execute(text(gold_sql)).fetchall()
                pred_res = conn.execute(text(predicted_sql)).fetchall()

            is_correct = set(pred_res) == set(gold_res)
            ex_score = 1 if is_correct else 0

            all_gold[question_id] = [list(row) for row in gold_res]
            all_predicted[question_id] = [list(row) for row in pred_res]

            ves_reward = 0.0
            time_ratio = 0.0
          

            # ----Isolated calculation R-VES only if the query is correct ----
            if is_correct:
                try:
                    diff_list = []
                    for _ in range(iterate_num):
                        with engine.connect() as conn:
                            # Abort queries exceeding 3 seconds to prevent long-running operations
                            conn.execute(text("SET statement_timeout = 3000"))
                            
                            start_p = time.perf_counter()
                            conn.execute(text(predicted_sql)).fetchall()
                            t_pred = time.perf_counter() - start_p
                            
                            start_g = time.perf_counter()
                            conn.execute(text(gold_sql)).fetchall()
                            t_gold = time.perf_counter() - start_g
                            
                            # Avoid division by zero
                            safe_t_pred = max(t_pred, 0.0001)
                            diff_list.append(t_gold / safe_t_pred)
                    
                    # Remove outliers and calculate average time ratio
                    processed_diff_list = clean_abnormal(diff_list)
                    if processed_diff_list:
                        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
                    
                    # Discrete reward system (R-VES) from the official repository
                    if time_ratio == 0:
                        ves_reward = 0.0
                    elif time_ratio >= 2:
                        ves_reward = 1.25
                    elif time_ratio >= 1:
                        ves_reward = 1.0
                    elif time_ratio >= 0.5:
                        ves_reward = 0.75
                    elif time_ratio >= 0.25:
                        ves_reward = 0.5
                    else:
                        ves_reward = 0.25

                except SQLAlchemyError as ves_e:
                    # If the time freeze has dropped, we don’t break EX, just VES = 0
                    logger.warning(f"VES measurement failed for {question_id} (EX is still valid). Error: {ves_e}")
                    ves_reward = 0.0
                    time_ratio = 0.0

            results.append(
                {
                    "db_id": db_id, 
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": ex_score,
                    "ves_reward": ves_reward,
                    "time_ratio": round(time_ratio, 4),
                    "difficulty": difficulty,
                }
            )

        except SQLAlchemyError as e:
            # only if the initial request has crashed (which determines EX)
            logger.info(f"Failed to process sql query for question {question_id}: '{predicted_sql}'")
            results.append(
                {
                    "db_id": db_id, 
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": 0,
                    "ves_reward": 0.0,
                    "time_ratio": 0.0,
                    "difficulty": difficulty,
                    "error": str(e),
                }
            )

            

    with open("all_predicted_results.json", "w", encoding="utf-8") as f:
        json.dump(all_predicted, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))

    with open("all_gold_results.json", "w", encoding="utf-8") as f:
        json.dump(all_gold, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))


    # ---- accuracy calculation ----

    # ---- Metric calculation ----
    def calc_metric(rows, metric_key):
        if not rows:
            return 0.0
        return 100.0 * sum(r[metric_key] for r in rows) / len(rows)

    total_ex = calc_metric(results, "score")
    total_ves = calc_metric(results, "ves_reward")

    by_difficulty_ex, by_difficulty_ves = {}, {}
    by_database_ex, by_database_ves = {}, {}

    for diff in set(r["difficulty"] for r in results):
        subset = [r for r in results if r["difficulty"] == diff]
        by_difficulty_ex[diff] = calc_metric(subset, "score")
        by_difficulty_ves[diff] = calc_metric(subset, "ves_reward")

    for db in set(r.get("db_id", "unknown") for r in results):
        subset = [r for r in results if r.get("db_id") == db]
        by_database_ex[db] = calc_metric(subset, "score")
        by_database_ves[db] = calc_metric(subset, "ves_reward")

    # ---- Ambiguity analysis ----
    # False ambiguous: model refused to answer, but gold SQL exists
    false_ambiguous = sum(
        1 for r in results 
        if r["predicted_sql"] == "ambiguous" and r["gold_sql"] != "ambiguous"
    )
    
    # True ambiguous: both model and gold agree question is ambiguous
    true_ambiguous = sum(
        1 for r in results 
        if r["predicted_sql"] == "ambiguous" and r["gold_sql"] == "ambiguous"
    )
    
    # Missed ambiguous: gold was ambiguous, but model generated SQL anyway
    missed_ambiguous = sum(
        1 for r in results 
        if r["predicted_sql"] != "ambiguous" and r["gold_sql"] == "ambiguous"
    )

    false_ambiguous_rate = 100.0 * false_ambiguous / len(results) if results else 0.0
    true_ambiguous_rate = 100.0 * true_ambiguous / len(results) if results else 0.0
    missed_ambiguous_rate = 100.0 * missed_ambiguous / len(results) if results else 0.0

    print(results)
    report = {
        "overall_ex": total_ex,
        "overall_ves": total_ves,
        "ex_by_difficulty": by_difficulty_ex,
        "ves_by_difficulty": by_difficulty_ves,
        "ex_by_database": by_database_ex,
        "ves_by_database": by_database_ves,
        "false_ambiguous": false_ambiguous,
        "false_ambiguous_rate": false_ambiguous_rate,
        "true_ambiguous": true_ambiguous,
        "true_ambiguous_rate": true_ambiguous_rate,
        "missed_ambiguous": missed_ambiguous,
        "missed_ambiguous_rate": missed_ambiguous_rate,
        "total": len(results),
        "results": results,
    }

    return report

def print_evaluation_report(report: dict):
    print("\n================ BIRD Benchmark Results ====================\n")

    # ---- overall ----
    print(f"Overall EX (Accuracy): {report['overall_ex']:.2f}%")
    print(f"Overall VES (Efficiency): {report['overall_ves']:.2f}%")
    print(f" Total queries : {report['total']}")

    # ---- By difficulty ----
    print("\nMetrics by difficulty:")
    for diff in sorted(report["ex_by_difficulty"].keys()):
        ex = report["ex_by_difficulty"][diff]
        ves = report["ves_by_difficulty"][diff]
        print(f"  {diff:<12}: EX = {ex:>5.2f}% | VES = {ves:>5.2f}%")

    # ---- By database ----
    print("\nMetrics by database:")
    for db in sorted(report["ex_by_database"].keys()):
        ex = report["ex_by_database"][db]
        ves = report["ves_by_database"][db]
        print(f"  {db:<20}: EX = {ex:>5.2f}% | VES = {ves:>5.2f}%")
    print()

    # ---- Ambiguity analysis ----
    print("\nAmbiguity detection analysis:")
    print(f"  False ambiguous (model refused, but gold exists): {report['false_ambiguous']} ({report['false_ambiguous_rate']:.2f}%)")
    print(f"  True ambiguous (both agree question is ambiguous): {report['true_ambiguous']} ({report['true_ambiguous_rate']:.2f}%)")
    print(f"  Missed ambiguous (gold was ambiguous, model answered anyway): {report['missed_ambiguous']} ({report['missed_ambiguous_rate']:.2f}%)")


    print("============================================================\n")
