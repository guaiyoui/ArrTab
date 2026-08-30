"""Dependency-free reproduction of the original AgenticTQA ROUGE metric."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from statistics import fmean

# This intentionally preserves the lowercased Unicode tokenizer used for the
# archived 2025-09-28 run. Changing it silently changes the reported scores.
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _prf(overlap: int, candidate_count: int, reference_count: int) -> dict[str, float]:
    precision = overlap / candidate_count if candidate_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(reference: str, candidate: str, n: int) -> dict[str, float]:
    if n < 1:
        raise ValueError("n must be at least 1")
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    reference_ngrams = [
        tuple(reference_tokens[index : index + n]) for index in range(len(reference_tokens) - n + 1)
    ]
    candidate_ngrams = [
        tuple(candidate_tokens[index : index + n]) for index in range(len(candidate_tokens) - n + 1)
    ]
    overlap = sum((Counter(reference_ngrams) & Counter(candidate_ngrams)).values())
    return _prf(overlap, len(candidate_ngrams), len(reference_ngrams))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, candidate: str) -> dict[str, float]:
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    overlap = _lcs_length(reference_tokens, candidate_tokens)
    return _prf(overlap, len(candidate_tokens), len(reference_tokens))


def evaluate_rouge(
    references: Iterable[str], candidates: Iterable[str]
) -> dict[str, dict[str, float]]:
    reference_list = list(references)
    candidate_list = list(candidates)
    if len(reference_list) != len(candidate_list):
        raise ValueError("references and candidates must have the same length")
    if not reference_list:
        raise ValueError("at least one prediction is required")

    per_example = [
        {
            "ROUGE-1": rouge_n(reference, candidate, 1),
            "ROUGE-2": rouge_n(reference, candidate, 2),
            "ROUGE-L": rouge_l(reference, candidate),
        }
        for reference, candidate in zip(reference_list, candidate_list)
    ]
    return {
        metric: {
            field: fmean(row[metric][field] for row in per_example)
            for field in ("precision", "recall", "f1")
        }
        for metric in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
    }
