# Local reranker experiment

Used `cross-encoder/ms-marco-MiniLM-L6-v2` to rerank the top 100 candidates from the exact MLX 512d float16 retriever for each of the existing 100 cached queries. The cross-encoder received only the query and passage text, not relevance judgments. No API calls were made.

## Observed results

- Hit@1: 19% → 28%.
- Hit@5: 49% → 59% (15 newly hit queries; 5 previously hit queries lost).
- Hit@10: 65% → 70% (11 newly hit queries; 6 previously hit queries lost).
- MRR@10: 0.32245 → 0.40162.
- Candidate Hit@100: 92% before and after, as expected because candidate membership is unchanged.
- Mean retrieval time for top 100: 13.46 ms.
- Mean additional reranker time: 313.77 ms, including tokenization, inference, synchronization and score transfer.
- Sum of mean stage times: 327.24 ms; p95 of per-query summed stage times: 498.28 ms. These are sums of separately measured stages, not a deployed endpoint benchmark.

The observed reranker improves final ordering while adding substantial latency. It is a local MPS cross-encoder baseline, not a fully optimized deployment. The application backend was not changed.

## Reproduction

Use an isolated environment with existing NumPy/MLX/PyTorch plus `transformers>=4.45,<5`. Run `diagnostics/benchmark_reranker.py` with access to Metal GPU and the first time, network access to download the public model. The script caches model files and candidate results under `data/reranker/`, which should remain excluded from Git. The 100 query vectors and corpus files are local prerequisites and are not published.

Model revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Device: PyTorch MPS; batch size 16; maximum pair length 512 tokens; one reranker warmup excluded. GPU candidates are 512d float16; the reranker is a separate model whose parameter memory is additional to the 1.024 GB vector matrix.

## Limits

This is the same selected development sample reused in earlier experiments, not fresh held-out validation. The model is trained on MS MARCO; training overlap with the expanded dev sample has not been independently audited. Long text pairs are truncated. Total production memory, concurrent throughput, an integrated endpoint and browser visuals have not been tested.

Official model card: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

Raw output: `reranker-results.json`; interactive summary: `embedding-comparison-dashboard.html`.
