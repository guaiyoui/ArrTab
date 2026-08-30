# ArrTab

## Agentic Open-Domain Tabular Question Answering with Self-Supervised Multi-Table Retriever

Official implementation of **ArrTab** (**A**gentic **r**etrieval and **r**easoning
on **Tab**les). ArrTab answers questions over heterogeneous table collections by
combining **MuRe**, a self-supervised multi-table retriever, with agents for table
integration, information extraction, and answer generation.

This repository provides the FeTaQA experiment pipeline, the original MuRe
retrieval implementation, a trained FeTaQA MuRe fusion checkpoint, and a
10-question reproducibility bundle with ranked table candidates and answer
traces.

## Method

```text
Question
  │
  ├─ MuRe table retrieval
  │    student dense retrieval → teacher passage reranking → BNN table fusion
  │
  ├─ Table integration
  │    select tables → single-table / join / union plan
  │
  ├─ Information extraction
  │    generate, validate, and execute read-only SQL
  │
  └─ Answer generation
       grounded answer from the retrieved tables and SQL evidence
```

MuRe is trained with self-supervision: table-derived SQL is translated into
natural-language questions, providing retrieval supervision without manual
question-to-table labels. At inference time, ArrTab retrieves from the full
table collection rather than receiving a preselected table.

## Released artifacts

| Artifact | Location | Description |
|---|---|---|
| ArrTab pipeline | `src/agentictqa/` | Retrieval, table integration, SQL execution, and answering |
| MuRe implementation | `src/agentictqa/retriever/legacy/` | Student, teacher, FiD feature, and BNN fusion stages |
| FeTaQA MuRe checkpoint | `checkpoints/fetaqa_fusion.pt` | Trained BNN table-fusion reranker, 37.8 MB |
| FeTaQA-10 bundle | `src/agentictqa/resources/fetaqa_10/` | Questions, 95 retrieved candidate tables, retrieval cache, and predictions |

The repository includes exactly one trained model: the FeTaQA-specific MuRe
fusion checkpoint used by live retrieval. Base encoders and the FAISS index are
not redistributed.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/guaiyoui/ArrTab.git
cd ArrTab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the additional retrieval dependencies only when recomputing MuRe
retrieval:

```bash
python -m pip install -e '.[retriever]'
```

## FeTaQA evaluation

### Reproduce the released predictions

The following command is fully offline and deterministically evaluates the
committed 10-question prediction file:

```bash
agentictqa evaluate
```

Expected macro-F1 values (the CLI also reports precision and recall):

```json
{
  "count": 10,
  "metrics": {
    "ROUGE-1": {"f1": 0.6183023105416839},
    "ROUGE-2": {"f1": 0.4318393726319961},
    "ROUGE-L": {"f1": 0.5160935719888268}
  }
}
```

### Run ArrTab with cached retrieval

Configure an OpenAI-compatible model endpoint, then replay the ranked table IDs
produced by MuRe:

```bash
cp .env.example .env
agentictqa run \
  --retrieval cached \
  --output outputs/fetaqa_10.jsonl
agentictqa evaluate --predictions outputs/fetaqa_10.jsonl
```

`--top-k 5` is the default used by the original pipeline. Set `--top-k 10` to
use every stored candidate. `--strict-grounded` disables the original
parametric fallback when the retrieved evidence is insufficient.

### Run live MuRe retrieval

Live retrieval uses the included `checkpoints/fetaqa_fusion.pt` by default. It
also requires the external student/teacher/FiD base models, FeTaQA table
catalog, and FAISS index:

```text
retriever_assets/
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

```bash
CUDA_VISIBLE_DEVICES=0 agentictqa run \
  --retrieval live \
  --assets-root retriever_assets \
  --dataset-root datasets_agentic_tqa \
  --write-retrieval-cache outputs/fetaqa_10_retrieval.jsonl \
  --output outputs/fetaqa_10_live.jsonl
```

The live wrapper loads the retrieval stack once per batch and preserves dense
candidates, reranking inputs, and fusion traces under `outputs/retriever/`.
See [RETRIEVER.md](RETRIEVER.md) for the stage-by-stage execution path and
custom asset options.

## Results

Open-domain FeTaQA results use ROUGE F1.

| Evaluation | Questions | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| ArrTab, paper result | Full experiment | **0.6490** | **0.4503** | **0.5331** |
| Released prediction trace | 10 | 0.6183 | 0.4318 | 0.5161 |

The 10-question trace is provided for code-path verification and is not a
replacement for the full benchmark estimate.

## Repository structure

```text
ArrTab/
├── checkpoints/
│   ├── README.md
│   └── fetaqa_fusion.pt
├── src/agentictqa/
│   ├── cli.py
│   ├── data.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── resources/fetaqa_10/
│   └── retriever/
│       ├── cache.py
│       ├── runner.py
│       └── legacy/
├── tests/
├── RETRIEVER.md
└── pyproject.toml
```

Run the release checks with:

```bash
python -m unittest discover -s tests -v
```

## License

ArrTab code is released under the MIT License. The bundled FeTaQA-derived data
is available under CC BY-SA 4.0. Vendored FiD-derived files retain CC BY-NC 4.0
terms. See [DATA_LICENSE.md](DATA_LICENSE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
