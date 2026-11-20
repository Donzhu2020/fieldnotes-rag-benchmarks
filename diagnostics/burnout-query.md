# Burnout query investigation

Query: `How to deal with burnout`

## Reproduction

The local endpoint returned five passages, scoring every one of the 50,000 vectors. The scores were:

| Passage ID | Cosine similarity | Topic |
| --- | ---: | --- |
| 39725 | 0.371594 | Taking short brain breaks |
| 7362 | 0.371037 | Triggers of post-acute withdrawal symptoms |
| 18410 | 0.351821 | Employee participation in wellness programs |
| 49652 | 0.350271 | Job-related article headlines |
| 7359 | 0.345053 | Triggers of post-acute withdrawal symptoms |

These are loosely related concepts, not a reliable set of evidence addressing burnout management. Similarity values are not probabilities or a calibrated relevance threshold.

## Corpus coverage

The searchable corpus consists of the first 50,000 rows in archive order, not a representative sample of topics.

- No exact `burnout` or `burn-out` occurrence was found within those rows.
- A case-insensitive `burn[ -]?out` search found one passage, ID 20892, concerning dehydration and muscles that "burn out".
- No `burned out` or `burnt out` phrase was found in those rows.
- The full one-million-row local file contains 61 `burn[ -]?out` matches, including irrelevant word senses.
- Passage 97985, outside the searchable subset, explicitly discusses workplace flexibility and burnout. Its existence shows relevant source material is available locally but has not yet been embedded for search. It does not establish clinical validity or complete coverage of the query.
- Keyword absence does not prove the absence of all semantically relevant material.

## Retrieval integrity

Checked all 50,000 passage IDs and texts against the source file. Checked every 100-row saved batch against the final vector matrix. The matrix is float32, shape (50,000, 1,536), finite, and row-normalized. No text/vector alignment discrepancy was found.

`rank_all` calculates every score and always returns the requested top-k. It does not assess whether enough evidence exists to answer the query. The interface's "matching passages" label can therefore overstate relevance.

## Conclusion and next steps

Corpus coverage is a supported explanation for this failure; no vector-alignment defect was found. A vector database or index would not create missing evidence.

For better evaluation, use queries with known relevant passages in the searchable subset, and include out-of-corpus queries such as this one. For broader user questions, improve corpus coverage with a representative or topic-specific selection. Calibrate abstention on labeled examples rather than inventing a universal cosine threshold. Consider labeling results as "nearest passages" to avoid implying they answer the question.

No retrieval code, corpus membership, or UI behavior was changed during this investigation. Reproducing the query required one small embedding call because that exact query string was not cached; it used the existing approved budget.

## Fix applied

Added a minimum cosine similarity of 0.60 after full-scan scoring and before returning results. The result count is now an upper limit; the site shows "No relevant passages found" when every candidate is below the cutoff. No weak passages are inserted to fill remaining slots.

Eight local calibration queries separated into four source-supported questions (top scores 0.664–0.804) and four unsupported examples (0.339–0.544). This is a small development set, not a held-out evaluation or a general guarantee. The local calibration vectors and observations are stored under data/search/ and excluded from Git.
