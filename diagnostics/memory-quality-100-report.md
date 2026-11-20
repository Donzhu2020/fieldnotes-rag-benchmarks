# 100-Query Float32 vs Float16 Evaluation

> Historical result from before removal of the manual 0.60 score cutoff. The current app returns the top-ranked results without that filter. Use `embedding-comparison-dashboard.html` for the new 512d/1536d rank-only comparison.

## Results

Both configurations searched the same one-million-passage collection through the local Flask API with the same 100 cached query vectors and the existing 0.60 cosine cutoff. The 100 queries were sampled with seed `20261119` from official dev queries whose official qrels include at least one relevant passage inside the indexed corpus (1,780 eligible queries in total).

| Metric | 1536d float32 | 1536d float16 |
|---|---:|---:|
| Vector matrix | 6,144,000,000 bytes | 3,072,000,000 bytes |
| Peak process RSS, including passage text | 8.74 GB | 5.58 GB |
| Hit@5 | 41/100 (41.0%) | 41/100 (41.0%) |
| Hit@10 | 54/100 (54.0%) | 54/100 (54.0%) |
| MRR@10 | 0.3091 | 0.3091 |
| Mean full-scan time/query | 183.40 ms | 865.82 ms |
| Median full-scan time/query | 164.10 ms | 828.60 ms |
| 95th-percentile scan time | 201.93 ms | 999.62 ms |
| Total wall time for 100 searches | 18.64 s | 86.71 s |

Hit@5 was **41/100 for both**. Increasing the window to ten results raised Hit@10 to **54/100 for both**, and every query had the same hit/miss outcome across precisions. MRR@10 was also unchanged. The returned rankings were not identical: mean top-five ID overlap was 79.4%, and the top result matched in 90/100 queries.

## Interpretation

The float16 matrix uses half the vector memory. Peak process RSS fell by 3.15 GB (36.1%) in these separate-process runs. On this implementation, float16 full scans were about 4.72× slower on average because each scan converts and normalizes float16 blocks before scoring.

This sample shows no observed Hit@5 or MRR@10 loss, but it does show different ranked passages and a substantial latency cost. The current float16 implementation trades process memory for query speed.

## Method limits

- The sample is 100 queries drawn from the 1,780 official dev queries with at least one judged relevant passage in the indexed first-million subset. It is not a random sample of all MS MARCO queries.
- Hit@5 and MRR@10 measure matches against the provided qrels. These labels are not exhaustive judgments of every passage, so they are not absolute answer-accuracy measures.
- The corpus contains the first 1,000,000 of 8,841,823 MS MARCO passages; results do not describe retrieval over the full collection.
- Each format was measured once in a separate local process. Timing is a single-run observation.
- Peak RSS includes loaded passage text and Python overhead, not only the vector matrix.
- The same query vectors were generated once using `text-embedding-3-small` and reused for both runs. Query embeddings were cached and reused; local billing information is excluded.

Detailed per-query results are in `memory-quality-100-results.json`; the fixed sample and cached query vectors are in `msmarco-precision-queries.json` and `msmarco-precision-query-vectors.npy`.

## Effect of the 0.60 score cutoff

On the original float32 matrix, repeating the same 100 scans with the cutoff disabled raised Hit@5 from 41 to 46 queries and Hit@10 from 54 to 64. Thus the 0.60 cutoff accounts for 5 and 10 missed judged hits, respectively, in this sample. Even without the cutoff, 36 queries had no qrels-positive passage in the top 10. The cutoff contributes to the low score, while ranking quality and incomplete corpus coverage also matter.

Only 1,780 of the 55,578 dev qids with qrels had a judged relevant passage in the indexed first-million subset. The 100-query benchmark intentionally sampled from those 1,780 eligible qids; it does not represent the full dev query set.
