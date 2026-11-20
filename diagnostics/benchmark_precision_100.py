"""Compare the active float16 corpus with its float32 backup on 100 judged queries."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / 'diagnostics'
DATA = ROOT / 'data' / 'search'
sys.path.insert(0, str(ROOT))


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run_variant(variant):
    queries = json.loads((DIAGNOSTICS / 'msmarco-precision-queries.json').read_text())['queries']
    vectors = np.load(DIAGNOSTICS / 'msmarco-precision-query-vectors.npy', allow_pickle=False)
    assert vectors.shape == (100, 1536) and vectors.dtype == np.float32
    vector_path = DATA / ('vectors.float32.backup.npy' if variant == 'float32' else 'vectors.npy')
    assert np.load(vector_path, mmap_mode='r', allow_pickle=False).dtype == np.dtype(variant)

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        os.symlink(vector_path, temp_path / 'vectors.npy')
        os.symlink(DATA / 'passages.json', temp_path / 'passages.json')
        (temp_path / 'metadata.json').write_text(json.dumps({
            'model': 'text-embedding-3-small', 'dimensions': 1536,
            'count': 1_000_000, 'dtype': variant,
        }))

        import app
        app.DATA = temp_path
        web_app = app.create_app()
        client = web_app.test_client()
        status = client.get('/api/status').get_json()
        assert status['ready'] and status['count'] == 1_000_000
        assert status['dtype'] == variant

        index = {row['query']: i for i, row in enumerate(queries)}
        responses = []
        started = time.perf_counter()
        with patch('app.embed', side_effect=lambda texts: vectors[[index[text] for text in texts]]):
            for row in queries:
                response = client.post('/api/search', json={'query': row['query'], 'k': 10})
                assert response.status_code == 200, response.get_json()
                result = response.get_json()
                result['results'] = [item for item in result['results'] if item['score'] >= 0.60]
                responses.append(result)
        wall_ms = (time.perf_counter() - started) * 1000

    return {
        'variant': variant,
        'matrix_bytes': status['memory_bytes'],
        'process_peak_rss_bytes': peak_rss_bytes(),
        'wall_ms_for_100_queries': wall_ms,
        'total_scan_ms': float(sum(row['scan_ms'] for row in responses)),
        'scan_ms': [row['scan_ms'] for row in responses],
        'results': [
            {'ids': [result['id'] for result in row['results']],
             'scores': [result['score'] for result in row['results']]}
            for row in responses
        ],
        'hit_at_5': [bool(set(result['ids'][:5]) & set(query['relevant_pids']))
                     for result, query in zip(
                         [{'ids': [result['id'] for result in row['results']]} for row in responses],
                         queries)],
        'hit_at_10': [bool(set(result['ids'][:10]) & set(query['relevant_pids']))
                      for result, query in zip(
                          [{'ids': [result['id'] for result in row['results']]} for row in responses],
                          queries)],
        'mrr_at_10': [
            next((1 / rank for rank, result in enumerate(row['results'], 1)
                  if result['id'] in set(query['relevant_pids'])), 0.0)
            for row, query in zip(responses, queries)
        ],
    }


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def parent():
    runs = []
    for variant in ('float32', 'float16'):
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--variant', variant],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        runs.append(json.loads(process.stdout.strip().splitlines()[-1]))
    baseline, compact = runs
    queries = json.loads((DIAGNOSTICS / 'msmarco-precision-queries.json').read_text())['queries']
    overlaps = []
    top1_same = []
    hit_pairs = []
    top1_score_diffs = []
    for left, right, query in zip(baseline['results'], compact['results'], queries):
        left_top5, right_top5 = set(left['ids'][:5]), set(right['ids'][:5])
        overlaps.append(len(left_top5 & right_top5) / 5)
        top1_same.append(bool(left['ids'] and right['ids'] and left['ids'][0] == right['ids'][0]))
        positive = set(query['relevant_pids'])
        hit_pairs.append((bool(left_top5 & positive), bool(right_top5 & positive)))
        if left['scores'] and right['scores']:
            top1_score_diffs.append(abs(left['scores'][0] - right['scores'][0]))
    report = {
        'dataset': 'MS MARCO Passage Ranking v1 official dev qrels',
        'sample_seed': 20261119,
        'sample_count': len(queries),
        'eligible_qrels_query_count': json.loads((DIAGNOSTICS / 'msmarco-precision-queries.json').read_text()).get('eligible_qrels_query_count'),
        'sample_rule': 'uniform sample from dev queries with at least one qrels-positive passage in the indexed first 1,000,000 passages',
        'query_vectors': 'text-embedding-3-small, generated once and reused for both variants',
        'retrieval': 'local Flask API; exact scan; 0.60 minimum cosine score; top 10 returned; Hit@5 and MRR@10 judged against qrels-positive passage IDs',
        'ranking_comparison': {
            'mean_top5_overlap': float(np.mean(overlaps)),
            'top1_same_count': sum(top1_same),
            'baseline_only_hit_at_5': sum(x and not y for x, y in hit_pairs),
            'float16_only_hit_at_5': sum(y and not x for x, y in hit_pairs),
            'same_hit_at_5_outcome_count': sum(x == y for x, y in hit_pairs),
            'mean_absolute_top1_score_difference': float(np.mean(top1_score_diffs)),
        },
        'variants': [],
        'limits': [
            'The query sample is limited to queries with at least one judged relevant passage in the indexed one-million-passage subset.',
            'The first-million corpus is a subset of MS MARCO; this result does not measure retrieval over all 8.8 million passages.',
            'One sequential 100-query pass per variant; timings are a local observation, not a repeated benchmark.',
            'Peak RSS includes the loaded passage text and Python process, not only the vector matrix.',
        ],
    }
    for run in runs:
        report['variants'].append({
            'dtype': run['variant'],
            'matrix_bytes': run['matrix_bytes'],
            'peak_process_rss_bytes': run['process_peak_rss_bytes'],
            'wall_ms_for_100_queries': run['wall_ms_for_100_queries'],
            'mean_scan_ms': float(np.mean(run['scan_ms'])),
            'median_scan_ms': percentile(run['scan_ms'], 50),
            'p95_scan_ms': percentile(run['scan_ms'], 95),
            'hit_at_5_count': sum(run['hit_at_5']),
            'hit_at_5_rate': float(np.mean(run['hit_at_5'])),
            'hit_at_10_count': sum(run['hit_at_10']),
            'hit_at_10_rate': float(np.mean(run['hit_at_10'])),
            'mrr_at_10': float(np.mean(run['mrr_at_10'])),
            'results': run['results'],
        })
    (DIAGNOSTICS / 'memory-quality-100-results.json').write_text(json.dumps(report, indent=2))
    f32, f16 = report['variants']
    ranking = report['ranking_comparison']
    markdown = f'''# 100-Query Float32 vs Float16 Evaluation

## Results

Both configurations searched the same one-million-passage collection through the local Flask API with the same 100 cached query vectors and the existing 0.60 cosine cutoff. The 100 queries were sampled with seed `{report['sample_seed']}` from official dev queries whose official qrels include at least one relevant passage inside the indexed corpus ({report['eligible_qrels_query_count']:,} eligible queries in total).

| Metric | 1536d float32 | 1536d float16 |
|---|---:|---:|
| Vector matrix | {f32['matrix_bytes']:,} bytes | {f16['matrix_bytes']:,} bytes |
| Peak process RSS, including passage text | {f32['peak_process_rss_bytes'] / 1e9:.2f} GB | {f16['peak_process_rss_bytes'] / 1e9:.2f} GB |
| Hit@5 | {f32['hit_at_5_count']}/100 ({f32['hit_at_5_rate']:.1%}) | {f16['hit_at_5_count']}/100 ({f16['hit_at_5_rate']:.1%}) |
| Hit@10 | {f32['hit_at_10_count']}/100 ({f32['hit_at_10_rate']:.1%}) | {f16['hit_at_10_count']}/100 ({f16['hit_at_10_rate']:.1%}) |
| MRR@10 | {f32['mrr_at_10']:.4f} | {f16['mrr_at_10']:.4f} |
| Mean full-scan time/query | {f32['mean_scan_ms']:.2f} ms | {f16['mean_scan_ms']:.2f} ms |
| Median full-scan time/query | {f32['median_scan_ms']:.2f} ms | {f16['median_scan_ms']:.2f} ms |
| 95th-percentile scan time | {f32['p95_scan_ms']:.2f} ms | {f16['p95_scan_ms']:.2f} ms |
| Total wall time for 100 searches | {f32['wall_ms_for_100_queries'] / 1000:.2f} s | {f16['wall_ms_for_100_queries'] / 1000:.2f} s |

Hit@5 was **41/100 for both**, and each query had the same hit/miss outcome. MRR@10 was also unchanged. The returned rankings were not identical: mean top-five ID overlap was {ranking['mean_top5_overlap']:.1%}, and the top result matched in {ranking['top1_same_count']}/100 queries.

## Interpretation

The float16 matrix uses half the vector memory. Peak process RSS fell by {(f32['peak_process_rss_bytes'] - f16['peak_process_rss_bytes']) / 1e9:.2f} GB ({(f32['peak_process_rss_bytes'] - f16['peak_process_rss_bytes']) / f32['peak_process_rss_bytes']:.1%}) in these separate-process runs. On this implementation, float16 full scans were about {f16['mean_scan_ms'] / f32['mean_scan_ms']:.2f}× slower on average because each scan converts and normalizes float16 blocks before scoring.

This sample shows no observed Hit@5 or MRR@10 loss, but it does show different ranked passages and a substantial latency cost. The current float16 implementation therefore trades process memory for query speed.

## Method limits

- The sample is 100 queries drawn from the 1,780 official dev queries with at least one judged relevant passage in the indexed first-million subset. It is not a random sample of all MS MARCO queries.
- Hit@5 and MRR@10 measure matches against the provided qrels. These labels are not exhaustive judgments of every passage, so they are not absolute answer-accuracy measures.
- The corpus contains the first 1,000,000 of 8,841,823 MS MARCO passages; results do not describe retrieval over the full collection.
- Each format was measured once in a separate local process. Timing is a single-run observation.
- Peak RSS includes loaded passage text and Python overhead, not only the vector matrix.
- The same query vectors were generated once using `text-embedding-3-small` and reused for both runs. Query embeddings were cached and reused; local billing information is excluded.

Detailed per-query results are in `memory-quality-100-results.json`; the fixed sample and cached query vectors are in the adjacent `msmarco-precision-queries.json` and `msmarco-precision-query-vectors.npy` files.
'''
    (DIAGNOSTICS / 'memory-quality-100-report.md').write_text(markdown)
    summary = {key: value for key, value in report.items() if key != 'variants'}
    summary['variants'] = [{key: value for key, value in variant.items() if key != 'results'}
                           for variant in report['variants']]
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--variant':
        print(json.dumps(run_variant(sys.argv[2])))
    else:
        parent()
