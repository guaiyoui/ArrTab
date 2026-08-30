# FeTaQA-10 evaluation slice

- Original FeTaQA label indices: 21 through 30.
- `retrieval_cache.jsonl`: the first 10 open-domain IDs stored in the original
  `AgenticTQA/data/FeTaQA/test_our/previous_results.json` cache.
- Live-run default: top 5 candidates, matching the original agent pipeline.
- `legacy_predictions.jsonl`: predictions transcribed from the complete archived
  run `AgenticTQA/logs/FeTaQA/index_v3.log` dated 2025-09-28.
- Metric: macro average of the original lowercased Unicode ROUGE-1, ROUGE-2, and
  ROUGE-L implementation.

The sample preserves natural retrieval failures without substituting manually
selected tables. For example, the relevant table for question 28 is not among
its cached top-10 candidates. This is intentional for open-domain evaluation.
