"""Command-line interface for live runs and offline score reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import FeTaQA10, legacy_predictions_path
from .metrics import evaluate_rouge


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _evaluate(path: Path) -> dict:
    dataset = FeTaQA10()
    predictions = _read_jsonl(path)
    by_id = {int(row["id"]): row["prediction"] for row in predictions}
    examples = dataset.examples(ids=sorted(by_id))
    missing = [example.id for example in examples if example.id not in by_id]
    if missing:
        raise ValueError(f"Missing predictions for ids: {missing}")
    metrics = evaluate_rouge(
        [example.answer for example in examples],
        [by_id[example.id] for example in examples],
    )
    return {"count": len(examples), "metrics": metrics}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal AgenticTQA FeTaQA release")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a JSONL prediction file")
    evaluate.add_argument(
        "--predictions",
        type=Path,
        default=legacy_predictions_path(),
        help="JSONL with id and prediction fields (default: archived 10-question run)",
    )

    run = subparsers.add_parser("run", help="Run the live agentic pipeline")
    run.add_argument("--ids", type=int, nargs="*", default=None, help="Question ids (21-30)")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument(
        "--retrieval",
        choices=("cached", "oracle"),
        default="cached",
        help="cached replays open-domain retrieval; oracle is for pipeline debugging only",
    )
    run.add_argument("--max-sql-retries", type=int, default=2)
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument(
        "--strict-grounded",
        action="store_true",
        help="Disable the original parametric-knowledge fallback when table evidence is insufficient",
    )
    run.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        print(json.dumps(_evaluate(args.predictions), indent=2, ensure_ascii=False))
        return

    from .llm import LLM, LLMConfig
    from .pipeline import AgenticTQAPipeline

    dataset = FeTaQA10()
    examples = dataset.examples(ids=args.ids, limit=args.limit)
    llm = LLM(LLMConfig.from_env(model=args.model, base_url=args.base_url))
    pipeline = AgenticTQAPipeline(
        dataset,
        llm,
        top_k=args.top_k,
        max_sql_retries=args.max_sql_retries,
        allow_parametric_fallback=not args.strict_grounded,
        retrieval_mode=args.retrieval,
    )
    results = []
    for example in examples:
        result = pipeline.run(example)
        row = result.to_dict()
        results.append(row)
        print(f"[{example.id}] {row['prediction']}")

    _write_jsonl(args.output, results)
    scored = _evaluate(args.output)
    print(json.dumps(scored, indent=2, ensure_ascii=False))
    print(f"Saved auditable traces to {args.output}")
