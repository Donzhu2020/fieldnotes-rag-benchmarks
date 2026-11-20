# Fieldnotes — Local Semantic Search

A local search website over English MS MARCO passages. No search index, no approximate nearest neighbors, and no vector database: every query scans the complete vector matrix in RAM.

[Open the self-contained experiment dashboard](diagnostics/embedding-comparison-dashboard.html) by downloading the HTML file and opening it in a browser.

## What it does

- Embeds passages and queries with `text-embedding-3-small` through Build Space.
- Loads all prepared passage vectors into memory as 512-dimensional float32 arrays.
- Computes exact cosine similarity over every row in bounded chunks, then sorts all scores.
- Shows the original passage, passage ID, similarity, scan time, embedding time, and total time.
- Supports up to 5, 10, or 20 results, copying passages, and keyboard submission with Cmd/Ctrl+Enter.
- Suggestion buttons run a search immediately. The three examples cover Dvorak, the Manhattan Project, and roasting squash seeds; each was verified against the current corpus.
- Uses a warm white and forest green interface that adapts to mobile screens.

This is the retrieval component of a RAG workflow. It does not generate answers.

## Current scope

The local corpus contains the first **1,000,000 passages** from MS MARCO Passage Ranking v1. All **1,000,000 passages** now have embeddings and participate in search; the first 50,000 vectors were reused and the remaining 950,000 were prepared in resumable batches.

The full vector matrix occupies 2,048,000,000 bytes (1.91 GiB), excluding passage text and other process memory. The complete MS MARCO v1 archive contains 8,841,823 passages. Original passage IDs are preserved. The subset uses archive order, not random sampling.

## Requirements

- Python 3.11 or later on macOS or Linux. Budget locking uses the Unix `fcntl` module.
- A Build Space API token and network access for embeddings.
- The MS MARCO corpus, downloaded separately. Data files and credentials are excluded from Git.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export AI_BUILDER_TOKEN='your-build-space-token'
```

Set credentials through the environment variable shown above. Credentials are never returned to the browser.

### Prepare data on a fresh checkout

The existing workspace already has the data and prepared vectors. On a fresh checkout, download the official passage archive:

```bash
mkdir -p data/msmarco
curl -fL --retry 3 -o data/msmarco/collection.tar.gz \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
```

Create the one-million-passage source file without extracting the complete corpus:

```bash
python3 - <<'PY'
import itertools
import tarfile
from pathlib import Path

root = Path('data/msmarco')
with tarfile.open(root / 'collection.tar.gz', 'r:gz') as archive:
    source = archive.extractfile('collection.tsv')
    with (root / 'collection.first1m.tsv').open('xb') as target:
        target.writelines(itertools.islice(source, 1_000_000))
PY
```

Generate the one-million-passage vectors, then start the website:

```bash
python3 prepare.py
python3 app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Try **What was the Manhattan Project?**

Preparation makes paid API requests and requests compact base64 responses, which are decoded into float32 vectors locally. Completed 100-passage batches are saved and reused if preparation is interrupted. The target is one million passages; requests are grouped under a 200,000-token ceiling. The active corpus is replaced only after every batch has been validated, and the previous corpus is backed up. Final assembly truncates the original 1536d vectors to the first 512 coordinates and L2-normalizes them, storing float32. Query vectors use the same truncation and normalization before scoring.

## How retrieval works

1. `np.load` loads the complete `data/search/vectors.npy` array without memory mapping.
2. The query is embedded with the same model and normalized.
3. Matrix multiplication scores **every passage** by cosine similarity.
4. A full, stable sort orders the scores; the top requested passages are returned without a fixed score cutoff.

Query embeddings are cached for up to 128 queries in the running process. Restarting the server clears this cache but preserves passage vectors. The 5/10/20 selection returns the top-ranked passages; low cosine scores can still be weak matches. Similarity is a ranking score, not answer accuracy.

The website listens only on `127.0.0.1:8765`. Queries and passage text are sent to the Build Space embedding endpoint; the interface runs locally, but embedding is not offline.

## API cost controls

Embedding preparation and uncached queries use a paid API. The code maintains a local reservation ledger, blocks calls over its configured cap, and retains failed-call reservations without automatic retries. Review `BUDGET`, `RESERVE_PER_TOKEN`, and `EXPANSION_BUDGET` before running preparation. Rates are configuration assumptions, not current provider quotes. Account balances, transaction history, and local billing records are excluded from this release.

## Tests and observations

```bash
python3 -m unittest discover -s tests -v
```

Nine tests cover exact ranking, stable ties, request validation, local access, cross-site request rejection, budget blocking, failed-call reservations, vector self-matching, and query caching. Preparation tests also verify reuse, resuming saved batches, and backing up the previous corpus. Embedding tests verify compact response decoding. The real-matrix test skips when local vectors are absent.

A real browser query over **50,000 passages** took approximately **28.26 ms for the full scan** and **0.71 s in total**, including query embedding. On the one-million-passage corpus, a live local API search scored all one million passages, returned five results, and took **1,132.86 ms for the full scan** and **2,013.06 ms total**, including an uncached query embedding. The leading passage ID was `3`. Each is a single observation, not a benchmark or performance guarantee. The one-million-passage check used the API route; no browser visual review or repeated load test was done after expansion.

An offline 100-query dashboard compares 1536d float32/float16 and truncated 512d float32/float16 against official MS MARCO dev qrels. It shows Hit@5, Hit@10, MRR@10, matrix/RSS memory, and scan time. See [the visual dashboard](diagnostics/embedding-comparison-dashboard.html) and [raw benchmark results](diagnostics/dimensions-100-results.json).

A selected 100-query qrels evaluation is documented in the dashboard. No fresh held-out evaluation, concurrent-load test, or screen-reader test has been completed. Some original passages contain encoding artifacts; their text is preserved without silent rewriting.

## Project files

- `app.py`: local Flask server and exact in-memory retrieval.
- `embeddings.py`: authenticated embedding requests and budget ledger.
- `prepare.py`: resumable preparation of the one-million-passage corpus.
- `templates/` and `static/`: English-only website.
- `tests/`: offline tests; no paid API requests.
- `PUBLICATION_REVIEW.md`: release scope, privacy review, and validation limits.

## Dataset

Source: [Microsoft MS MARCO datasets](https://microsoft.github.io/msmarco/Datasets). This repository does not redistribute the corpus or API credentials. Consult the dataset's official terms before using or redistributing its contents.

## Similarity scores

The app previously used a 0.60 cosine cutoff based on eight manually chosen examples from the 50,000-passage corpus. It was added to honor the request to avoid returning weak, unrelated passages. It was not an MS MARCO or OpenAI standard, and it was never calibrated on the one-million-passage corpus. The cutoff has been removed; the app now returns the highest-ranked passages even when their scores are low. Similarity is a ranking score, not a relevance guarantee. The system scores all 1,000,000 vectors in RAM and does not use an index or vector database.

## Current application versus benchmark recommendation

The application uses NumPy with a 512d float32 matrix and stable full sorting. Native MLX 512d float16 is the recommended benchmark configuration but has not been deployed into the application. MLX benchmark scripts require Apple Silicon and an accessible Metal GPU; install MLX separately in an appropriate environment. Raw passage data and embedding arrays are not distributed. Benchmark scripts need the corresponding locally prepared vectors and cached query vectors.

## Interpreting retrieval quality

Our 49% Hit@5 and 65% Hit@10 are strict qrels-ID metrics on a selected 100-query sample, not answer accuracy. The same ranking reaches Hit@100=92%. See [the diagnosis](diagnostics/RETRIEVAL_QUALITY.md) for metric comparability, sparse judgments, inspected failure cases, and proposed next evaluations.
