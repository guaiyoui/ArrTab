# FeTaQA-10 evaluation slice

- Original FeTaQA label indices: 21 through 30.
- Retrieval candidates: the first 10 IDs stored in the original
  `AgenticTQA/data/FeTaQA/test_our/previous_results.json` cache.
- Live-run default: top 5 candidates, matching the original agent pipeline.
- `legacy_predictions.jsonl`: predictions transcribed from the complete archived
  run `AgenticTQA/logs/FeTaQA/index_v3.log` dated 2025-09-28.
- Metric: macro average of the original lowercased Unicode ROUGE-1, ROUGE-2, and
  ROUGE-L implementation.

The sample exposes retrieval failures instead of adding missing oracle tables.
For example, the relevant table for question 28 is not among its cached top-10
candidates. This is intentional: this slice is an open-domain cached-retrieval
evaluation, not an oracle-table evaluation.

Five gold tables absent from all cached candidate lists are bundled solely for
the explicitly selected `--retrieval oracle` pipeline-debugging mode. They are
never added to cached retrieval.
