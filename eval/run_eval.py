"""Evaluate the agent with MLflow GenAI judges and code-level checks.

Runs the real agent (PP_ENV=databricks: gateway + Vector Search) on eval/eval_set.jsonl and scores:
  * Safety, RelevanceToQuery, RetrievalGroundedness: Databricks-hosted LLM judges
  * citation_coverage: every claim in a non-refusal answer carries a [n] marker (code)
  * refusal_behavior: refuses exactly when the eval set expects a refusal (code)

Results land in an MLflow experiment. This signals quality; it does not gate deployment unless
--fail-under is passed.

    python eval/run_eval.py --experiment /Shared/policypilot-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety, scorer  # noqa: E402

from agent.citations import verify_answer  # noqa: E402
from agent.graph import run_agent  # noqa: E402
from agent.nodes import is_refusal_text  # noqa: E402


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(question: str) -> str:
    return run_agent(question)["answer"]


@scorer
def citation_coverage(outputs: str) -> Feedback:
    if is_refusal_text(outputs):
        return Feedback(value="yes", rationale="Refusal: nothing to cite.")
    result = verify_answer(outputs, num_sources=10_000, strict_numbers=False)
    if result.ok:
        return Feedback(value="yes", rationale=f"All {len(result.claims)} claims cite a source.")
    return Feedback(value="no", rationale="; ".join(result.failures)[:500])


@scorer
def refusal_behavior(outputs: str, expectations: dict[str, Any]) -> Feedback:
    expected = bool(expectations.get("should_refuse", False))
    actual = is_refusal_text(outputs)
    ok = expected == actual
    return Feedback(
        value="yes" if ok else "no",
        rationale=f"expected refusal={expected}, got refusal={actual}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).with_name("eval_set.jsonl")))
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/policypilot-eval"))
    parser.add_argument("--fail-under", type=float, default=None, help="optional minimum for every */mean metric, e.g. 0.8")
    args = parser.parse_args()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(args.experiment)

    results = mlflow.genai.evaluate(
        data=load_eval_set(Path(args.data)),
        predict_fn=predict,
        scorers=[Safety(), RelevanceToQuery(), RetrievalGroundedness(), citation_coverage, refusal_behavior],
    )

    # Judge and scorer results are aggregated into run metrics such as "safety/mean".
    summary = {k: float(v) for k, v in sorted(results.metrics.items()) if isinstance(v, (int, float))}

    lines = ["| metric | value |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in summary.items()]
    report = "\n".join(lines)
    print(report)
    print(f"MLflow run: {results.run_id}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"## PolicyPilot agent-eval\n\n{report}\n\nMLflow run `{results.run_id}`\n")

    if args.fail_under is not None:
        failing = {k: v for k, v in summary.items() if k.endswith("/mean") and v < args.fail_under}
        if failing:
            raise SystemExit(f"Below {args.fail_under}: {failing}")


if __name__ == "__main__":
    main()
