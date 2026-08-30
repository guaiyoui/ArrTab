# ArrTab

Minimal release of the AgenticTQA open-domain FeTaQA pipeline. The repository
contains the original retriever implementation, its cached output for the 10
FeTaQA examples used by the experiment, and a compact table integration and
answering pipeline. Every committed result uses open-domain retrieval.

## Reproduce the cached open-domain score

```bash
git clone https://github.com/guaiyoui/ArrTab.git
cd ArrTab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agentictqa evaluate
```

Expected macro F1 over FeTaQA IDs 21–30:

| Artifact | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Archived open-domain run | 0.6183 | 0.4318 | 0.5161 |
| Paper-reported result | 0.6490 | 0.4503 | 0.5331 |

The first row is exactly reproducible from the committed per-question
predictions. The second row is retained as the number reported in the paper;
the original working directory does not contain the matching prediction file,
so this release does not relabel it as reproduced.

## Run with the cached open-domain retrieval

```bash
cp .env.example .env
agentictqa run --retrieval cached --output outputs/fetaqa_10.jsonl
agentictqa evaluate --predictions outputs/fetaqa_10.jsonl
```

The cache is the ranked output of the original open-domain retriever, not gold
table injection. `--top-k 5` is the default used by the original agent; use
`--top-k 10` to expose every cached candidate.

## Recompute retrieval

Retriever code is included under `src/agentictqa/retriever/legacy/`. Models,
the FeTaQA FAISS index, and the dataset catalog are intentionally not committed.
Place those local assets in this layout:

```text
retriever_assets/
├── checkpoints/fetaqa_fusion.pt
├── index/on_disk_index_FeTaQA_rel_graph/
│   ├── merged_index.ivfdata
│   ├── passages.jsonl
│   └── populated.index
└── models/
    ├── student_tqa_retriever_step_29500/
    ├── tqa_reader_base/
    └── tqa_retriever/

datasets_agentic_tqa/
└── FeTaQA/labels/tables_numid.jsonl
```

Then run the actual three-stage retriever and optionally save its ranked output
as a reusable cache:

```bash
python -m pip install -e '.[retriever]'
CUDA_VISIBLE_DEVICES=0 agentictqa run \
  --retrieval live \
  --assets-root retriever_assets \
  --dataset-root datasets_agentic_tqa \
  --fusion-checkpoint retriever_assets/checkpoints/fetaqa_fusion.pt \
  --write-retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_live.jsonl
```

The stages are student dense retrieval against the on-disk FAISS index,
teacher passage reranking, and BNN fusion table reranking. Each live run writes
its queries, dense candidates, tagged candidates, and fusion traces to a new
directory under `outputs/retriever/`; it never deletes an existing run. See
[RETRIEVER.md](RETRIEVER.md) for the exact execution flow and path mapping.

## Pipeline

1. Retrieve ranked table IDs from the committed open-domain cache or live retriever.
2. Select relevant tables and plan a single-table, join, or union operation.
3. Generate and validate read-only SQLite, then execute it with retry-on-error.
4. Produce one concise answer from the table and SQL evidence.

The OpenAI-compatible API key is read only from the environment and is never
written to traces. `--strict-grounded` disables the original implementation's
parametric fallback.

## Layout

```text
src/agentictqa/
├── cli.py
├── data.py
├── metrics.py
├── pipeline.py
├── resources/fetaqa_10/       # questions, open-domain cache, predictions, tables
└── retriever/
    ├── cache.py               # cached open-domain replay
    ├── runner.py              # configurable, non-destructive execution wrapper
    └── legacy/                # original dense + teacher + BNN retriever source
```

Run offline checks with:

```bash
python -m unittest discover -s tests -v
```

## Scope

- Ten examples are a reproducibility slice, not a full benchmark estimate.
- Cached retrieval is deterministic; live LLM answers depend on the serving model.
- The legacy live retriever expects CUDA and downloads the standard
  `bert-base-uncased` and `t5-base` tokenizers if they are not already cached.
- Original ArrTab code is MIT licensed. Bundled FeTaQA-derived data is CC BY-SA
  4.0, while vendored FiD-derived files retain CC BY-NC 4.0 terms; see
  [DATA_LICENSE.md](DATA_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
