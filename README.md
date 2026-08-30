# ArrTab — minimal AgenticTQA FeTaQA release

This folder is a self-contained, auditable release of the original AgenticTQA
FeTaQA experiment. It keeps the method's essential stages while removing
training code, large indexes, checkpoints, logs, duplicate dataset adapters,
and machine-specific paths.

The release contains:

- the exact 10 questions used by the original script (FeTaQA IDs 21–30);
- the original cached top-10 retrieval results, 95 candidate tables, and 5
  separately isolated oracle-debug tables;
- a small retrieve → integration-plan → SQL → answer pipeline;
- the original macro ROUGE implementation used by the archived run;
- per-question predictions from the complete 2025-09-28 run;
- offline tests and a live OpenAI-compatible API entry point.

## Reproduce the archived result (no API required)

```bash
git clone https://github.com/guaiyoui/ArrTab.git
cd ArrTab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agentictqa evaluate
```

Expected F1 scores:

| Evidence | Questions | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| Archived predictions (`index_v3.log`) | 10 | 0.6183 | 0.4318 | 0.5161 |
| Paper-reported target (artifact unavailable) | 10 (assumed) | 0.6490 | 0.4503 | 0.5331 |
| Verified oracle diagnostic (Qwen3-235B) | 10 | 0.6747 | 0.4609 | 0.5587 |

The first row is reproducible from committed per-question predictions. The
second row is retained only as a paper-reported target: the original repository
contains the numbers in LaTeX and plotting files, but no matching prediction
artifact. This release therefore does **not** present that row as independently
reproduced. The third row verifies the downstream implementation but uses gold
tables, so it is not an open-domain comparison; see [results/README.md](results/README.md).

The metric intentionally preserves the archived run's lowercased Unicode
tokenizer. This matters: a later file in the working repository uses a different
ASCII-oriented tokenizer and cannot reproduce the logged values for names such
as `Chávez` and `Årdal`.

## Run the live pipeline

Copy the environment template and add a key for an OpenAI-compatible service:

```bash
cp .env.example .env
agentictqa run --ids 21 22 --output outputs/smoke.jsonl
```

To run all 10 questions:

```bash
agentictqa run --output outputs/fetaqa_10.jsonl
agentictqa evaluate --predictions outputs/fetaqa_10.jsonl
```

Configuration can be supplied in `.env` or as CLI flags:

```bash
AGENTICTQA_API_KEY=... \
agentictqa run \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --base-url https://api.studio.nebius.com/v1/ \
  --top-k 5
```

The API key is read only from the environment; it is never accepted as a CLI
argument or written to the output. Each JSONL output row includes the retrieval
mode, candidate tables, selected tables, integration plan, SQL, evidence,
answer, call count, and latency.

## What the pipeline implements

1. Replay the original open-domain retriever's cached ranked table IDs.
2. Ask an LLM to select tables and choose `single`, `join`, or `union`.
3. Ask an LLM for read-only SQLite, validate it, execute it, and retry on error.
4. Generate one grounded answer sentence from the selected tables and SQL result.

Like the original `AgenticTQA/agent_team/agents.py`, the default answer policy
falls back to the LLM's internal knowledge when retrieved evidence is
insufficient and forbids refusal text. Use `--strict-grounded` to disable this
behavior for applications where provenance matters more than matching the
original experiment.

The bundled sample defaults to top-5, matching the original agent code. Use
`--top-k 10` to expose all cached candidates. Missing relevant tables are not
silently added: questions 26, 28, and others preserve genuine retrieval errors.

An oracle-table smoke test is available for debugging the downstream agents:

```bash
agentictqa run --retrieval oracle --strict-grounded \
  --output outputs/fetaqa_10_oracle.jsonl
```

Oracle mode is a closed-domain diagnostic and must not be compared to or
reported as the paper's open-domain result.

## Repository layout

```text
AgenticTQA_release/
├── pyproject.toml
├── README.md
├── DATA_LICENSE.md
├── results/            # verified live-run trace and configuration
├── src/agentictqa/
│   ├── cli.py          # run / evaluate commands
│   ├── data.py         # FeTaQA-10 loader and cached retrieval
│   ├── llm.py          # OpenAI-compatible JSON client
│   ├── metrics.py      # exact archived macro ROUGE
│   ├── pipeline.py     # agentic orchestration and safe SQL
│   └── resources/      # 10 questions, predictions, 100 small tables
└── tests/
```

Run the offline test suite:

```bash
python -m unittest discover -s tests -v
```

## Scope and limitations

- Ten examples are a smoke/reproducibility slice, not a statistically reliable
  estimate of full-test-set performance.
- Retrieval is replayed from the original cache. The 1.5 GB index and multi-GB
  checkpoints are deliberately excluded, so this release does not recompute
  dense retrieval.
- Live answers depend on the model and serving backend. Only the archived
  prediction artifact is deterministic.
- The default parametric fallback can improve answer coverage after retrieval
  failures, but those answers are not grounded in the bundled tables. Use
  `--strict-grounded` when every answer must be traceable to table evidence.
- The default model reflects what was available on the configured Nebius
  endpoint when this release was validated; override it when providers rotate
  model IDs.
- The original metric is included for exact comparison; it is not claimed to be
  interchangeable with every third-party `rouge-score` implementation.

## License and data attribution

Code is MIT licensed. The bundled FeTaQA-derived sample is separately licensed
under CC BY-SA 4.0; see [DATA_LICENSE.md](DATA_LICENSE.md). FeTaQA was introduced
by Nan et al., *Transactions of the Association for Computational Linguistics*,
2022. Please cite the original dataset when using this sample.
