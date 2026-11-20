# Understanding the retrieval scores

## What the reported numbers mean

The reported 49% Hit@5 and 65% Hit@10 are matches against MS MARCO qrels-positive passage IDs on a selected 100-query sample. They are not passage precision, generated-answer accuracy, or a percentage of all relevant content found. The system has no answer-generation stage.

A new exact scan of the same cached queries using 512d float32 measured:

- Hit@1: 19/100
- Hit@5: 49/100
- Hit@10: 65/100
- Hit@20: 75/100
- Hit@50: 88/100
- Hit@100: 92/100
- Hit@1000: 98/100

Changing K alone can turn 65% into 92% without improving a single ranking. A claimed score above 90% is comparable only when the metric, K, corpus, query split, relevance labels, and candidate-generation task match. We have not inspected the specific high-scoring system mentioned in the discussion.

## Evidence from this project

**Sparse judged IDs:** 98 of the 100 sampled queries have exactly one qrels-positive ID inside the indexed corpus; one has two and one has three. A different passage that answers the question is still counted as a miss when its ID is not judged positive. This is a limitation of this evaluation, not proof that every miss is actually relevant.

**Ranking mistakes are real:** for the query asking for the longest vein in the body, the judged passage ranked sixth while leading results discussed the largest veins. Semantic relatedness can miss a crucial qualifier.

**Some strict-ID misses have plausible answer coverage:** the cooked-turkey storage query has a judged passage at rank 15, while the leading returned passages cover the same type of storage question. Movie-ticket results also contain closely related answer material at higher ranks than the judged ID. These are qualitative source inspections, not an independent human relevance reannotation or domain validation.

**Compression is not the primary observed cause:** unfiltered 1536d float32 Hit@5/10 was 46%/64%; 512d float32 was 49%/65%. Native MLX 512d float16 retained 49%/65% in the sample. Reducing to 256d did lower these values to 41%/61%.

**This is a limited retrieval baseline:** it uses a general-purpose embedding model and exact cosine ranking. There is no lexical fusion, cross-encoder reranker, query-specific temporal interpretation, or task-specific model tuning. Additional stages may help but have not been tested here. Exact exhaustive scoring already searches every stored vector; swapping the search library does not by itself fix semantic rankings.

**Corpus coverage does not explain these selected misses by itself:** every sampled query has at least one judged positive in the indexed first-million corpus by construction. The subset and eligibility filtering do limit comparison with full-corpus benchmarks. The first-million subset is not a random sample of the complete corpus.

## Next evidence to collect

Keep the current 100 queries as development data, then evaluate on a fresh held-out sample. Inspect a blinded sample of misses with explicit relevance criteria, recording both strict-ID hit rates and human answer coverage. Retrieve a candidate pool of 50–100 passages and test reranking to a final five; Hit@100=92% shows a candidate ceiling in this particular sample, not a guarantee of 92% final Hit@5. Compare lexical/semantic fusion and alternative embedding models under the same fixed protocol before claiming an improvement.

## Sources and limitations

- Official dataset and qrels: https://microsoft.github.io/msmarco/Datasets
- Official MS MARCO passage ranking evaluator (MRR@10): https://github.com/microsoft/MSMARCO-Passage-Ranking/blob/master/ms_marco_eval.py
- Microsoft's distinct MS MARCO Web Search benchmark reports different Recall@K columns; its scores are not directly comparable with this subset experiment: https://github.com/microsoft/MS-MARCO-Web-Search
- Raw diagnostic results: `retrieval-diagnosis.json` (public query text, passage IDs and ranks only; no corpus text).

One selected 100-query sample has been reused throughout development. No fresh held-out evaluation, exhaustive human relevance labeling, or reranking experiment has been completed. The extra rank-depth scan used existing cached vectors and made no API calls.
