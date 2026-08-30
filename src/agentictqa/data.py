"""Load the bundled FeTaQA-10 questions and table data."""

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


@dataclass(frozen=True)
class Table:
    id: str
    title: str
    frame: pd.DataFrame


class FeTaQA10:
    """Self-contained 10-question slice used by the original experiment."""

    def __init__(
        self,
        root: Path | None = None,
        table_catalog_path: Path | None = None,
    ) -> None:
        self.root = Path(root) if root else RESOURCE_DIR
        self.tables_dir = self.root / "tables"
        self.table_catalog_path = Path(table_catalog_path) if table_catalog_path else None
        self._table_catalog: dict[str, dict] | None = None
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

    def load_table(self, table_id: str) -> Table:
        matches = list(self.tables_dir.glob(f"{table_id}_*.csv"))
        if len(matches) == 1:
            path = matches[0]
            return Table(id=table_id, title=path.stem, frame=pd.read_csv(path))
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Expected one CSV for table id {table_id}, found {len(matches)}"
            )

        row = self._catalog().get(str(table_id))
        if row is None:
            hint = (
                " Pass --dataset-root when using live retrieval."
                if not self.table_catalog_path
                else ""
            )
            raise FileNotFoundError(f"Table id {table_id} is not available.{hint}")
        columns = [column["text"] for column in row["columns"]]
        records = [
            [cell.get("text", "") for cell in table_row["cells"]] for table_row in row["rows"]
        ]
        frame = pd.DataFrame(records, columns=columns)
        return Table(id=str(table_id), title=row.get("documentTitle", str(table_id)), frame=frame)

    def _catalog(self) -> dict[str, dict]:
        if self._table_catalog is not None:
            return self._table_catalog
        self._table_catalog = {}
        if self.table_catalog_path is None:
            return self._table_catalog
        with self.table_catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self._table_catalog[str(row["tableId"])] = row
        return self._table_catalog


def legacy_predictions_path() -> Path:
    return RESOURCE_DIR / "legacy_predictions.jsonl"


def retrieval_cache_path() -> Path:
    return RESOURCE_DIR / "retrieval_cache.jsonl"
