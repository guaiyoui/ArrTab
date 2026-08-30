"""Replay the committed outputs of the original open-domain retriever."""

from __future__ import annotations

import json
from pathlib import Path

from ..data import Example, retrieval_cache_path


class CachedOpenDomainRetriever:
    """Question-keyed open-domain cache produced by the released retriever."""

    source = "cached-open-domain"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else retrieval_cache_path()
        self._rows = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> dict[int, dict]:
        rows: dict[int, dict] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = int(row["id"])
                if question_id in rows:
                    raise ValueError(f"Duplicate retrieval-cache id: {question_id}")
                rows[question_id] = row
        return rows

    def prepare(self, examples: list[Example], top_k: int) -> None:
        for example in examples:
            self.retrieve_ids(example, top_k)

    def retrieve_ids(self, example: Example, top_k: int) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        row = self._rows.get(example.id)
        if row is None:
            raise KeyError(f"Question {example.id} is not present in {self.path}")
        if row["question"] != example.question:
            raise ValueError(f"Question text mismatch for retrieval-cache id {example.id}")
        return [str(table_id) for table_id in row["table_ids"][:top_k]]
