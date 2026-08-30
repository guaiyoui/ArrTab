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
        choices=("cached", "live"),
        default="cached",
        help="replay the open-domain cache or execute the released retriever",
    )
    run.add_argument(
        "--retrieval-cache",
        type=Path,
        default=None,
        help="optional question-to-table JSONL cache used by --retrieval cached",
    )
    run.add_argument(
        "--assets-root",
        type=Path,
        default=Path("retriever_assets"),
        help="local root containing models/ and index/ (not included in this repository)",
    )
    run.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets_agentic_tqa"),
        help="local dataset root containing FeTaQA/labels/tables_numid.jsonl",
    )
    run.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=None,
        help="FeTaQA BNN fusion-reranker checkpoint required by --retrieval live",
    )
    run.add_argument(
        "--retriever-work-dir",
        type=Path,
        default=Path("outputs/retriever"),
        help="directory for non-destructive live-retrieval intermediate files",
    )
    run.add_argument(
        "--write-retrieval-cache",
        type=Path,
        default=None,
        help="write live ranked table IDs as a reusable JSONL cache",
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
    from .retriever import (
        CachedOpenDomainRetriever,
        LegacyOpenDomainRetriever,
        LegacyRetrieverConfig,
    )

    if args.retrieval == "live":
        fusion_checkpoint = args.fusion_checkpoint or (
            args.assets_root / "checkpoints" / "fetaqa_fusion.pt"
        )
        retriever = LegacyOpenDomainRetriever(
            LegacyRetrieverConfig(
                assets_root=args.assets_root,
                dataset_root=args.dataset_root,
                fusion_checkpoint=fusion_checkpoint,
                work_dir=args.retriever_work_dir,
            )
        )
        dataset = FeTaQA10(
            table_catalog_path=args.dataset_root / "FeTaQA" / "labels" / "tables_numid.jsonl"
        )
    else:
        retriever = CachedOpenDomainRetriever(args.retrieval_cache)
        dataset = FeTaQA10()
    examples = dataset.examples(ids=args.ids, limit=args.limit)
    retriever.prepare(examples, args.top_k)
    if args.write_retrieval_cache:
        if not isinstance(retriever, LegacyOpenDomainRetriever):
            raise ValueError("--write-retrieval-cache is only valid with --retrieval live")
        retriever.write_cache(args.write_retrieval_cache, examples)
    llm = LLM(LLMConfig.from_env(model=args.model, base_url=args.base_url))
    pipeline = AgenticTQAPipeline(
        dataset,
        llm,
        retriever,
        top_k=args.top_k,
        max_sql_retries=args.max_sql_retries,
        allow_parametric_fallback=not args.strict_grounded,
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
