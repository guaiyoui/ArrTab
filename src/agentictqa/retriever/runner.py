"""Configurable wrapper around the original AgenticTQA retriever implementation."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data import Example


@dataclass(frozen=True)
class LegacyRetrieverConfig:
    """Paths and inference settings for the original FeTaQA retriever."""

    assets_root: Path
    dataset_root: Path
    fusion_checkpoint: Path
    work_dir: Path = Path("outputs/retriever")
    dense_top_n: int = 2000
    fusion_contexts: int = 250
    min_tables: int = 5
    max_vectors: int = 10000
    bnn_samples: int = 6
    no_fp16: bool = False

    @property
    def student_model(self) -> Path:
        return self.assets_root / "models" / "student_tqa_retriever_step_29500"

    @property
    def teacher_model(self) -> Path:
        return self.assets_root / "models" / "tqa_retriever"

    @property
    def reader_model(self) -> Path:
        return self.assets_root / "models" / "tqa_reader_base"

    @property
    def index_dir(self) -> Path:
        return self.assets_root / "index" / "on_disk_index_FeTaQA_rel_graph"

    @property
    def table_catalog(self) -> Path:
        return self.dataset_root / "FeTaQA" / "labels" / "tables_numid.jsonl"

    def validate(self) -> None:
        required = [
            self.student_model,
            self.teacher_model,
            self.reader_model,
            self.index_dir / "populated.index",
            self.index_dir / "merged_index.ivfdata",
            self.index_dir / "passages.jsonl",
            self.table_catalog,
            self.fusion_checkpoint,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(f"Missing retriever assets:\n  - {joined}")


class LegacyOpenDomainRetriever:
    """Run dense retrieval, teacher reranking, and BNN fusion reranking."""

    source = "live-open-domain"

    def __init__(self, config: LegacyRetrieverConfig) -> None:
        self.config = config
        self._table_ids: dict[int, list[str]] = {}
        self.last_run_dir: Path | None = None

    def prepare(self, examples: list[Example], top_k: int) -> None:
        missing = [example for example in examples if example.id not in self._table_ids]
        if not missing:
            return
        rows, run_dir = run_legacy_retriever(missing, self.config)
        self.last_run_dir = run_dir
        for example, table_ids in zip(missing, rows, strict=True):
            self._table_ids[example.id] = [str(table_id) for table_id in table_ids]
        for example in missing:
            self.retrieve_ids(example, top_k)

    def retrieve_ids(self, example: Example, top_k: int) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if example.id not in self._table_ids:
            self.prepare([example], top_k)
        return self._table_ids[example.id][:top_k]

    def write_cache(self, path: Path, examples: list[Example]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                row = {
                    "id": example.id,
                    "question": example.question,
                    "table_ids": self._table_ids[example.id],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _activate_legacy_imports() -> None:
    """Expose the original module layout without changing the vendored source."""

    legacy_root = Path(__file__).resolve().parent / "legacy"
    paths = [legacy_root / "table2txt", legacy_root / "relevance", legacy_root]
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _write_queries(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = {
                "id": str(example.id),
                "question": example.question,
                "table_id_lst": [],
                "answers": ["placeholder"],
                "ctxs": [{"title": "", "text": "placeholder"}],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_table_catalog(path: Path) -> dict[str, dict]:
    tables: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                tables[str(row["tableId"])] = row
    return tables


def run_legacy_retriever(
    examples: list[Example],
    config: LegacyRetrieverConfig,
) -> tuple[list[list[str]], Path]:
    """Execute the three original retrieval stages for a batch of questions."""

    if not examples:
        return [], config.work_dir
    config.validate()
    _activate_legacy_imports()

    # Imported lazily so cached runs do not require torch, transformers, or FAISS.
    import finetune_table_retr as fusion_reranker  # type: ignore[import-not-found]
    import passage_ondisk_retrieval as dense_retriever  # type: ignore[import-not-found]
    from table2txt.retr_utils import process_dev  # type: ignore[import-not-found]

    run_dir = config.work_dir / f"run-{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    query_path = run_dir / "fusion_query.jsonl"
    dense_path = run_dir / "fusion_retrieved.jsonl"
    tagged_path = run_dir / "fusion_retrieved_tagged.jsonl"
    _write_queries(query_path, examples)

    dense_args = argparse.Namespace(
        student_model_path=str(config.student_model),
        teacher_model_path=str(config.teacher_model),
        index_dir=str(config.index_dir),
        index_file=str(config.index_dir / "populated.index"),
        passage_file=str(config.index_dir / "passages.jsonl"),
        data=str(query_path),
        output_path=str(dense_path),
        n_docs=config.dense_top_n,
        min_tables=config.min_tables,
        max_retr=config.max_vectors,
        question_maxlength=50,
        no_fp16=config.no_fp16,
    )
    dense_retriever.main(dense_args)

    with dense_path.open(encoding="utf-8") as handle:
        dense_rows = [json.loads(line) for line in handle if line.strip()]
    tagged_rows = process_dev(
        dense_rows,
        config.fusion_contexts,
        _load_table_catalog(config.table_catalog),
        "rel_graph",
        config.min_tables,
    )
    with tagged_path.open("w", encoding="utf-8") as handle:
        for row in tagged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fusion_args = argparse.Namespace(
        sql_batch_no=None,
        do_train=False,
        model_path=str(config.reader_model),
        fusion_retr_model=str(config.fusion_checkpoint),
        train_data=None,
        eval_data=str(tagged_path),
        n_context=config.fusion_contexts,
        per_gpu_batch_size=4,
        per_gpu_eval_batch_size=1,
        cuda=0,
        name="fusion",
        checkpoint_dir=str(run_dir),
        bnn=1,
        prior_model=None,
        text_maxlength=300,
        bnn_num_eval_sample=config.bnn_samples,
        multi_model_eval=0,
        debug=0,
    )
    selected = fusion_reranker.main(fusion_args)
    if len(selected) != len(examples):
        raise RuntimeError(f"Retriever returned {len(selected)} rows for {len(examples)} questions")
    return selected, run_dir
