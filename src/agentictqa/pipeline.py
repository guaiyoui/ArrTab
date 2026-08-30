"""ArrTab pipeline: retrieve, integrate, extract, and answer."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import pandas as pd

from .data import Example, FeTaQA10, Table
from .llm import LLM


class Retriever(Protocol):
    source: str

    def retrieve_ids(self, example: Example, top_k: int) -> list[str]: ...


@dataclass
class RunResult:
    id: int
    question: str
    prediction: str
    retrieval_source: str
    candidate_table_ids: list[str]
    selected_table_ids: list[str]
    plan: dict[str, Any]
    sql: str | None
    sql_errors: list[str]
    evidence: list[dict[str, Any]]
    parametric_fallback_allowed: bool
    llm_calls: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_answer(answer: str) -> str:
    return re.sub(r"^\s*(?:final\s+)?answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()


def _table_context(tables: dict[str, Table], rows: int = 5) -> str:
    blocks = []
    for alias, table in tables.items():
        sample = table.frame.head(rows).where(pd.notna(table.frame.head(rows)), None)
        blocks.append(
            f"{alias} (retrieved table id={table.id}, title={table.title})\n"
            f"columns={list(table.frame.columns)}\n"
            f"rows={sample.to_dict(orient='records')}"
        )
    return "\n\n".join(blocks)


def _validate_read_only_sql(sql: str) -> str:
    statement = sql.strip().removesuffix(";").strip()
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed")
    if ";" in statement:
        raise ValueError("Multiple SQL statements are not allowed")
    forbidden = re.compile(
        r"\b(attach|detach|alter|create|delete|drop|insert|pragma|reindex|replace|update|vacuum)\b",
        flags=re.IGNORECASE,
    )
    if forbidden.search(statement):
        raise ValueError("Only read-only SQL is allowed")
    direct_string_literal = re.compile(
        r"(?:\bselect|,)\s*'[^']*'\s+(?:as\s+[\w\"']+|,|from\b)",
        flags=re.IGNORECASE,
    )
    if direct_string_literal.search(statement):
        raise ValueError("SELECT evidence from table columns; do not fabricate string literals")
    return statement


class AgenticTQAPipeline:
    def __init__(
        self,
        dataset: FeTaQA10,
        llm: LLM,
        retriever: Retriever,
        *,
        top_k: int = 5,
        max_sql_retries: int = 2,
        allow_parametric_fallback: bool = True,
    ) -> None:
        self.dataset = dataset
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_sql_retries = max_sql_retries
        self.allow_parametric_fallback = allow_parametric_fallback

    def _plan(self, question: str, candidates: dict[str, Table]) -> dict[str, Any]:
        plan = self.llm.json(
            system="You are a table retrieval and integration planner. Return valid JSON only.",
            prompt=f"""Select only the retrieved tables needed to answer the question.

Question: {question}

Retrieved table schemas and sample rows:
{_table_context(candidates, rows=6)}

Return exactly these fields:
{{
  "operation": "single|join|union|none",
  "selected_tables": ["t0"],
  "matched_columns": [["t0.column", "t1.column"]]
}}
Use the aliases shown above. Do not invent a table or column.""",
        )
        selected = plan.get("selected_tables", [])
        if not isinstance(selected, list):
            selected = []
        selected = list(dict.fromkeys(alias for alias in selected if alias in candidates))
        if not selected:
            selected = [next(iter(candidates))]
        plan["selected_tables"] = selected
        return plan

    def _generate_sql(
        self,
        question: str,
        plan: dict[str, Any],
        selected: dict[str, Table],
        error: str | None = None,
    ) -> str:
        correction = f"\nPrevious execution error: {error}\nFix it." if error else ""
        response = self.llm.json(
            system="You write executable SQLite SELECT queries. Return valid JSON only.",
            prompt=f"""Question: {question}
Integration plan: {plan}

SQLite tables:
{_table_context(selected, rows=20)}
{correction}

Return {{"sql": "..."}}. Use only the aliases above. Quote unusual column names.
Select the smallest evidence needed for answering; never modify the database.
Every selected answer value must come from a table column or an explicit calculation over
table columns. Never type a remembered answer value as a SELECT literal.
For ratios, differences, or aggregates, also SELECT the raw numerator, denominator, or
supporting values so the answer can state both the calculation and its source values.
Good pattern: SELECT "Candidate.1", "Party", "%" FROM t0 WHERE "Party" LIKE '%name%'.
Bad pattern: SELECT 'remembered name' AS winner FROM t0.""",
        )
        sql = response.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Model did not return a SQL string")
        return _validate_read_only_sql(sql)

    @staticmethod
    def _execute_sql(sql: str, selected: dict[str, Table]) -> pd.DataFrame:
        connection = sqlite3.connect(":memory:")
        try:
            for alias, table in selected.items():
                table.frame.to_sql(alias, connection, index=False, if_exists="replace")
            return pd.read_sql_query(sql, connection)
        finally:
            connection.close()

    def _answer(
        self,
        question: str,
        selected: dict[str, Table],
        evidence: pd.DataFrame,
    ) -> str:
        evidence_rows = evidence.head(30).where(pd.notna(evidence.head(30)), None)
        if self.allow_parametric_fallback:
            answer_policy = (
                "First use the supplied evidence. If it is insufficient or irrelevant, answer "
                "directly from your best internal knowledge. Never mention missing or insufficient "
                "data and never refuse. This fallback reproduces the original AgenticTQA behavior."
            )
        else:
            answer_policy = (
                "Use only the supplied evidence and provide the best direct answer supported by it. "
                "Do not discuss evidence sufficiency and do not add unsupported facts."
            )
        response = self.llm.json(
            system="You are a precise table question-answering agent. Return valid JSON only.",
            prompt=f"""Question: {question}

Selected tables:
{_table_context(selected, rows=3)}

Table titles and section names are evidence too; they may encode context such as a
deanery, competition, or award that is shared by every row.

SQL evidence:
columns={list(evidence.columns)}
rows={evidence_rows.to_dict(orient="records")}

Return {{"answer": "one direct sentence"}}.
Directly answer every part of the question. Keep exact entity names, years, and numbers.
Be concise and prefer exact table wording. Do not repeat background already stated in the
question. For a requested ratio, state the raw form (for example, "151 of 273 seats") and
optionally its percentage. For multiple records, include their key years and labels.
Answer policy: {answer_policy}""",
        )
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model did not return an answer string")
        return clean_answer(answer)

    def run(self, example: Example) -> RunResult:
        started = time.perf_counter()
        calls_before = self.llm.calls

        table_ids = self.retriever.retrieve_ids(example, self.top_k)
        if not table_ids:
            raise RuntimeError(f"Retriever returned no tables for question {example.id}")
        retrieved = [self.dataset.load_table(table_id) for table_id in table_ids]
        candidates = {f"t{index}": table for index, table in enumerate(retrieved)}
        plan = self._plan(example.question, candidates)
        selected = {alias: candidates[alias] for alias in plan["selected_tables"]}

        sql = None
        sql_errors: list[str] = []
        evidence = pd.DataFrame()
        error = None
        for _ in range(self.max_sql_retries):
            try:
                sql = self._generate_sql(example.question, plan, selected, error)
                evidence = self._execute_sql(sql, selected)
                break
            # This is the retry boundary for model, validation, and SQLite errors.
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                sql_errors.append(error)
        else:
            # The final answer still gets useful evidence if SQL generation repeatedly fails.
            evidence = pd.concat(
                [table.frame.head(20).add_prefix(f"{alias}.") for alias, table in selected.items()],
                axis=1,
            )

        prediction = self._answer(example.question, selected, evidence)
        evidence_rows = evidence.head(30).where(pd.notna(evidence.head(30)), None)
        return RunResult(
            id=example.id,
            question=example.question,
            prediction=prediction,
            retrieval_source=self.retriever.source,
            candidate_table_ids=[table.id for table in retrieved],
            selected_table_ids=[table.id for table in selected.values()],
            plan=plan,
            sql=sql,
            sql_errors=sql_errors,
            evidence=evidence_rows.to_dict(orient="records"),
            parametric_fallback_allowed=self.allow_parametric_fallback,
            llm_calls=self.llm.calls - calls_before,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
