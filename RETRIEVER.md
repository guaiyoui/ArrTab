# Open-domain retriever

This release includes the inference path called by the original
`AgenticTQA/agent_team/agents.py`. The old entry point called
`tools.data_acquision.tester.main(query)`; the release wrapper in
`src/agentictqa/retriever/runner.py` preserves the same stages while replacing
machine-specific paths and destructive temporary-directory cleanup.

## Execution flow

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

The committed `retrieval_cache.jsonl` stores the last line of this process for
the 10 release questions. It is an open-domain cache: it contains only retrieved
IDs and has no reference/gold-table field.

## Original-to-release path mapping

| Original path | Release CLI path |
|---|---|
| `models/student_tqa_retriever_step_29500` | `<assets-root>/models/student_tqa_retriever_step_29500` |
| `models/tqa_retriever` | `<assets-root>/models/tqa_retriever` |
| `models/tqa_reader_base` | `<assets-root>/models/tqa_reader_base` |
| `index/on_disk_index_FeTaQA_rel_graph` | `<assets-root>/index/on_disk_index_FeTaQA_rel_graph` |
| best FeTaQA `sql_1_...pt` | `--fusion-checkpoint` |
| `datasets_agentic_tqa/FeTaQA/labels/tables_numid.jsonl` | `<dataset-root>/FeTaQA/labels/tables_numid.jsonl` |

The original FeTaQA `best_metric_info.json` points to
`sql_1_epoc_5_step_1614_model_8_14_2025.pt`. Copy or symlink that file to the
path supplied with `--fusion-checkpoint`.

## Commands

Validate the asset layout as part of a live run and recompute the 10-question
cache:

```bash
python -m pip install -e '.[retriever]'
CUDA_VISIBLE_DEVICES=0 agentictqa run \
  --retrieval live \
  --assets-root retriever_assets \
  --dataset-root datasets_agentic_tqa \
  --fusion-checkpoint retriever_assets/checkpoints/fetaqa_fusion.pt \
  --top-k 5 \
  --write-retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_live.jsonl
```

Replay a newly generated cache without loading any retrieval model:

```bash
agentictqa run \
  --retrieval cached \
  --retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_replay.jsonl
```

The live wrapper batches all requested questions so model and index loading is
not repeated per example. A unique run directory is created for every
invocation, retaining the intermediate JSONL files for inspection. The legacy
model code addresses CUDA device 0 internally, so expose the desired single GPU
with `CUDA_VISIBLE_DEVICES` as shown above.
