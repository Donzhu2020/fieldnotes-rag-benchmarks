# Memory Format and Retrieval Quality Comparison

## Summary

This offline comparison tests four storage representations against the existing 1,000,000-passage collection. It uses the same exact full scan for each option and makes no embedding API calls.

| Representation | Matrix size | Relative to 1536d float32 | Known relevant passage in top 5 | Top-5 overlap vs baseline | Top-1 same as baseline |
|---|---:|---:|---:|---:|---:|
| 1536d float32 (baseline) | 6.144 GB | 100% | 4/4 positive cases | — | — |
| 1536d float16 values | 3.072 GB | 50% | 4/4 | 100% mean | 8/8 |
| 512d float32 | 2.048 GB | 33.3% | 4/4 | 87.5% mean | 4/8 |
| 512d float16 values | 1.024 GB | 16.7% | 4/4 | 87.5% mean | 4/8 |

“Known relevant passage” means the passage ID manually recorded for each of the four positive smoke cases remains somewhere in the top five. This does not mean every result in the top five is relevant.

## What changed in the results

- **1536d float16:** all eight queries kept the same top result and the same top-five passage IDs as the float32 baseline in this sample. It halves vector-matrix storage with no observed ranking change in these cases.
- **512 dimensions:** the four known passages remained in the top five, but only four of eight top results matched the baseline. Rankings changed for Dvorak and lactose, among others. This is a measurable ranking change; the sample is too small to say whether it harms user-perceived relevance.
- **512d float16:** had the same retrieval metrics as 512d float32 on these eight cases. It uses one-sixth of baseline matrix storage.

The existing score cutoff is 0.60. For “How to deal with burnout,” the full-collection baseline returns passage 710448 at 0.579, while both 512d variants score it above 0.60 (about 0.616). The source passage is actually about burnout, so the old “out” label—created against a smaller collection—is invalid here. “What is the weather in Boston right now?” returns a dated 2014 forecast headline at about 0.668; that is topically related but cannot answer “right now.” This shows why a similarity cutoff alone and labels carried over from the 50,000-passage collection cannot establish answer quality on the million-passage collection.

## Method and limits

- Eight cached query vectors were scanned against all one million passages: four manually labeled “in” cases and four “out” cases.
- The 512d variants take the first 512 dimensions of each stored embedding and re-normalize vectors before cosine scoring.
- The float16 simulation rounds stored vector values to float16, converts them back to float32 for scoring, then re-normalizes. It tests quantization effects on rankings, not the speed of native float16 matrix multiplication.
- Timings cover all eight queries on one local run; they are not a stable latency benchmark.
- There is no judged MS MARCO qrels set or independent relevance annotation in this local evaluation. The eight examples are a smoke set, not a statistically meaningful accuracy test.

## Recommendation

If the goal is a low-risk memory reduction, test **1536d float16 storage** first: this sample shows half the matrix size and unchanged top-five rankings. If the goal is the smallest matrix, **512d float16** reaches one-sixth of baseline size, but its changed rankings require a proper judged evaluation and a separately calibrated relevance cutoff before claiming quality is preserved.

The vectors currently used by the site were generated at 1536 dimensions. This comparison does not modify the active site or its stored vectors. OpenAI's Embeddings API supports requesting a reduced `dimensions` value; that is a separate embedding-generation choice from truncating already-generated vectors. See the [Embeddings API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create) and [embeddings guide](https://developers.openai.com/api/docs/guides/embeddings).
