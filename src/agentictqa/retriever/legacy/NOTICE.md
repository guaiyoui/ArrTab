# Vendored retriever source

This directory contains the inference-time retrieval source from the original
AgenticTQA working repository: dense student retrieval, teacher reranking,
relation-graph passage processing, and the BNN fusion reranker. The wrapper in
`../runner.py` supplies configurable paths and a non-destructive run directory;
the legacy modules are otherwise kept recognizable for auditability.

Several files retain their upstream Facebook/FiD copyright headers. Those files
remain under FiD's CC BY-NC 4.0 license; see the repository-level
`THIRD_PARTY_NOTICES.md` for attribution and the license link.
