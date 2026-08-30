# Verified FeTaQA-10 run

This directory contains the complete trace from a live downstream-pipeline
validation performed on 2026-08-30 (UTC).

```bash
agentictqa run \
  --retrieval oracle \
  --strict-grounded \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --output results/fetaqa_10_oracle_qwen3_235b_20260830.jsonl
```

| Questions | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---:|---:|---:|---:|
| 10 | 0.6747 | 0.4609 | 0.5587 |

The JSONL artifact includes retrieval mode, selected table IDs, integration
plan, read-only SQL, SQL errors (if any), evidence rows, prediction, call count,
and latency for every question. It contains no API key or prompt credential.

This is an **oracle-table, closed-domain diagnostic**. It demonstrates that the
integration, SQL, execution, and answer stages can exceed the numerical target
on the 10-question slice when retrieval is correct. It must not be presented as
the paper's open-domain result. Live model outputs may vary as providers update
model implementations.
