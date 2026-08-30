"""Load the bundled FeTaQA-10 questions, retrieval cache, and CSV tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "fetaqa_10"


@dataclass(frozen=True)
class Example:
    id: int
    question: str
    answer: str
    candidate_ids: tuple[str, ...]
    reference_table_ids: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    id: str
    title: str
    frame: pd.DataFrame


class FeTaQA10:
    """Self-contained 10-question slice used by the original experiment."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else RESOURCE_DIR
        self.tables_dir = self.root / "tables"
        self._examples = self._load_examples(self.root / "questions.jsonl")

    @staticmethod
    def _load_examples(path: Path) -> dict[int, Example]:
        examples: dict[int, Example] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                example = Example(
                    id=int(row["id"]),
                    question=row["question"],
                    answer=row["answer"],
                    candidate_ids=tuple(str(value) for value in row["candidate_ids"]),
                    reference_table_ids=tuple(str(value) for value in row["reference_table_ids"]),
                )
                if example.id in examples:
                    raise ValueError(f"Duplicate question id: {example.id}")
                examples[example.id] = example
        return examples

    def examples(self, ids: list[int] | None = None, limit: int | None = None) -> list[Example]:
        selected_ids = ids if ids else sorted(self._examples)
        unknown = [question_id for question_id in selected_ids if question_id not in self._examples]
        if unknown:
            raise KeyError(f"Unknown question ids: {unknown}; available ids are 21-30")
        rows = [self._examples[question_id] for question_id in selected_ids]
        return rows[:limit] if limit is not None else rows

    def retrieve(
        self,
        example: Example,
        top_k: int = 5,
        mode: str = "cached",
    ) -> list[Table]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if mode == "cached":
            table_ids = example.candidate_ids[:top_k]
        elif mode == "oracle":
            table_ids = example.reference_table_ids
        else:
            raise ValueError("retrieval mode must be 'cached' or 'oracle'")
        return [self.load_table(table_id) for table_id in table_ids]

    def load_table(self, table_id: str) -> Table:
        matches = list(self.tables_dir.glob(f"{table_id}_*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one CSV for table id {table_id}, found {len(matches)}"
            )
        path = matches[0]
        return Table(id=table_id, title=path.stem, frame=pd.read_csv(path))


def legacy_predictions_path() -> Path:
    return RESOURCE_DIR / "legacy_predictions.jsonl"
