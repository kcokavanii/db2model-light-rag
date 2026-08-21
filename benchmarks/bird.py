import json
from typing import Any, Dict, List

from pydantic import Field

from .base import BenchmarkBase
from .evaluate_bird import run_evaluation
from .response import ToolResponse


class BenchmarkBIRD(BenchmarkBase):
    use_evidence: bool = False
    """Dump evidence directly into the `user_query` instead of the prompt
    to help the model. (debatable, consult teacher)"""
    target_prompt_tokens_by_database: Dict[str, int] = Field(
        default_factory=dict,
        exclude=True,
    )

    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    async def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Returns predictions in the expected format."""
        queries = self._load_queries()
        predictions = {}
        self.target_prompt_tokens_by_database = {}

        for item in queries:
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]
            evidence = item["evidence"]

            tool = tool_dict[db_id]

            if self.use_evidence:
                question = f"question: {question}, evidence (may be empty): {evidence}"

            # predict() is sequential, so this global-usage delta belongs to
            # the current question and can be attributed to its database.
            prompt_tokens_before = self.llm_client.get_usage()["prompt_tokens"]
            try:
                result = await tool.query(question)
            finally:
                prompt_tokens_after = self.llm_client.get_usage()["prompt_tokens"]
                prompt_tokens_delta = prompt_tokens_after - prompt_tokens_before
                if prompt_tokens_delta < 0:
                    raise RuntimeError("Token counter decreased during prediction")

                self.target_prompt_tokens_by_database[db_id] = (
                    self.target_prompt_tokens_by_database.get(db_id, 0)
                    + prompt_tokens_delta
                )

            print(result)

            result = ToolResponse.model_validate(result)

            if result.status == "ambiguous":
                sql_query = "ambiguous"
            elif result.status == "success":
                sql_query = result.query
            else:
                sql_query = "error"

            predictions[str(qid)] = sql_query

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    async def evaluate(self, predictions: Dict[str, str]) -> dict:
        self._save_predictions(predictions, "./query_results.json")
        report = run_evaluation(predictions, self.answer_file, self.db_url)

        return report
