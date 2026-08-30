# MuRe: self-supervised multi-table retrieval

MuRe is the retrieval component of ArrTab. The implementation in
`src/agentictqa/retriever/legacy/` follows the inference path used in the
FeTaQA experiments, while `src/agentictqa/retriever/runner.py` provides
configurable paths, batch execution, and non-destructive trace handling.

## Retrieval pipeline

```text
question batch
  → student query encoder
  → FeTaQA relation-graph FAISS index (top 2,000 passages)
  → teacher passage reranker
  → relation-graph tagging (top 250 passages, at least 5 tables)
  → FiD cross-attention features
  → FeTaQA BNN fusion reranker
  → top 10 unique table IDs
```

The BNN fusion reranker combines passage and table representations with the
FiD-derived feature representation. Its trained FeTaQA state dict is included
at `checkpoints/fetaqa_fusion.pt` and is the default for `--retrieval live`.

## Required assets

Only the trained FeTaQA fusion checkpoint is included. Supply the following
base-model directories, index files, and table catalog for live retrieval:

| Component | Default path |
|---|---|
| Student query encoder | `<assets-root>/models/student_tqa_retriever_step_29500` |
| Teacher passage reranker | `<assets-root>/models/tqa_retriever` |
| FiD reader | `<assets-root>/models/tqa_reader_base` |
| Relation-graph FAISS index | `<assets-root>/index/on_disk_index_FeTaQA_rel_graph` |
| FeTaQA fusion checkpoint | `checkpoints/fetaqa_fusion.pt` (included) |
| FeTaQA table catalog | `<dataset-root>/FeTaQA/labels/tables_numid.jsonl` |

Use `--fusion-checkpoint PATH` to evaluate a different fusion checkpoint.

## Live retrieval

```bash
python -m pip install -e '.[retriever]'
CUDA_VISIBLE_DEVICES=0 agentictqa run \
  --retrieval live \
  --assets-root retriever_assets \
  --dataset-root datasets_agentic_tqa \
  --top-k 5 \
  --write-retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_live.jsonl
```

The wrapper batches all requested questions so models and the index are loaded
once. Each invocation creates a unique directory under `outputs/retriever/` and
keeps the query, dense candidate, tagged candidate, and fusion files for
inspection. The legacy model code addresses CUDA device 0 internally; expose
the desired GPU with `CUDA_VISIBLE_DEVICES`.

## Cached retrieval

`src/agentictqa/resources/fetaqa_10/retrieval_cache.jsonl` contains the ranked
MuRe output for the released FeTaQA examples. Replay any compatible cache
without loading the retrieval stack:

```bash
agentictqa run \
  --retrieval cached \
  --retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_replay.jsonl
```

Each cache row is keyed by question and contains retrieved table IDs only.
