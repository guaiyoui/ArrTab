# FeTaQA MuRe checkpoint

`fetaqa_fusion.pt` is the trained Bayesian neural network (BNN) table-fusion
reranker used by MuRe for the FeTaQA live-retrieval pipeline.

## Model details

| Field | Value |
|---|---|
| Task | Open-domain table retrieval on FeTaQA |
| Component | MuRe BNN table-fusion reranker |
| Format | PyTorch state dict |
| Parameters | 16 tensors; posterior mean and rho for 8 parameter groups |
| Input / hidden size | 2,304 / 768 |
| File size | 37,780,125 bytes |
| SHA-256 | `533d583a1a4357bd5464aebf77da0e4571cd1c366d187ac6098edfe7593da047` |

The original checkpoint-selection record identifies epoch 5 as the best saved
state on 2,000 FeTaQA training-evaluation instances, with P@1 = 0.6500 and P@5
= 0.6865. These retrieval checkpoint-selection scores are distinct from the
end-to-end ROUGE results reported in the main README.

## Usage

The ArrTab CLI selects this file automatically:

```bash
agentictqa run \
  --retrieval live \
  --assets-root retriever_assets \
  --dataset-root datasets_agentic_tqa
```

The checkpoint contains only the trained fusion-reranker weights. Live
retrieval also requires the external student, teacher, and FiD base models plus
the FeTaQA FAISS index documented in [RETRIEVER.md](../RETRIEVER.md).

To verify the file after cloning:

```bash
sha256sum checkpoints/fetaqa_fusion.pt
```
