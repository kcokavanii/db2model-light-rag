import os
import json
import logging
import sqlglot
import decimal
import re

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


def run_evaluation(predictions: Dict[str, str], answer_file: str, db_url: str):
    with open(answer_file, "r") as f:
        answer_file = json.load(f)

    gold_queries = {str(item["question_id"]): item for item in answer_file}

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

            with engine.connect() as conn:

                gold_res = conn.execute(text(gold_sql)).fetchall()
                all_gold[question_id] = [list(row) for row in gold_res]

                pred_res = conn.execute(text(predicted_sql)).fetchall()
                all_predicted[question_id] = [list(row) for row in pred_res]

            score = set(pred_res) == set(gold_res)

            results.append(
                {
                    "db_id": db_id, 
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                }
            )

        except SQLAlchemyError as e:
            logger.info(
                f"Failed to process sql query for question {question_id}: '{predicted_sql}'"
            )
            results.append(
                {
                    "db_id": db_id, 
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": 0,
                    "difficulty": difficulty,
                    "error": str(e),
                }
            )

    with open("all_predicted_results.json", "w", encoding="utf-8") as f:
        json.dump(all_predicted, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))

    with open("all_gold_results.json", "w", encoding="utf-8") as f:
        json.dump(all_gold, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))


    # ---- accuracy calculation ----

    def accuracy(rows):
        if not rows:
            return 0.0
        return 100.0 * sum(r["score"] for r in rows) / len(rows)

    total_acc = accuracy(results)

    by_difficulty = {}
    for diff in set(r["difficulty"] for r in results):
        subset = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = accuracy(subset)

    # Группировка по БД
    by_database = {}
    for db in set(r.get("db_id", "unknown") for r in results):
        subset = [r for r in results if r.get("db_id") == db]
        by_database[db] = accuracy(subset)

    false_ambiguous = sum(
        1 for r in results if r["predicted_sql"] == "ambiguous"
    )

    false_ambiguous_rate = (
        100.0 * false_ambiguous / len(results) if results else 0.0
    )

    print(results)
    report = {
        "overall_accuracy": total_acc,
        "accuracy_by_difficulty": by_difficulty,
        "accuracy_by_database": by_database,
        "false_ambiguous": false_ambiguous,
        "false_ambiguous_rate": false_ambiguous_rate,
        "total": len(results),
        "results": results,
    }

    return report

def print_evaluation_report(report: dict):
    print("\n================ BIRD Benchmark Results ====================\n")

    # ---- overall ----
    print(f"Overall accuracy: {report['overall_accuracy']:.2f}%")
    print(f" Total queries : {report['total']}")

    # ---- difficulty ----
    print()
    print("Accuracy by difficulty:")
    for diff, acc in sorted(report["accuracy_by_difficulty"].items()):
        print(f"  {diff:<12}: {acc:.2f}%")
    print()
    print("\nAccuracy by database:")
    for db, acc in sorted(report["accuracy_by_database"].items()):
        print(f"  {db:<20}: {acc:.2f}%")
    print()

    print(f"False ambiguous predicted : {report['false_ambiguous']}")
    print(f"False ambiguous rate      : {report['false_ambiguous_rate']:.2f}%")

    print("============================================================\n")
