"""Compare local embedding memory variants using the saved eight-query smoke set."""
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/search'
QUERIES = DATA / 'relevance-check-queries.npy'
CASES = DATA / 'relevance-check.json'
COUNT = 1_000_000
K = 5
CHUNK = 100_000


def search_all(matrix, queries, dimensions, half):
    queries = queries[:, :dimensions].astype(np.float32, copy=True)
    if half:
        queries = queries.astype(np.float16).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    best_scores = [np.empty(0, np.float32) for _ in queries]
    best_ids = [np.empty(0, np.int64) for _ in queries]
    for start in range(0, COUNT, CHUNK):
        end = min(start + CHUNK, COUNT)
        values = matrix[start:end, :dimensions]
        if half or dimensions < 1536:
            values = values.astype(np.float32, copy=True)
            if half:
                values = values.astype(np.float16).astype(np.float32)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        scores = values @ queries.T
        for j in range(len(queries)):
            column = scores[:, j]
            local = np.argpartition(column, -K)[-K:]
            candidate_scores = np.concatenate((best_scores[j], column[local]))
            candidate_ids = np.concatenate((best_ids[j], local.astype(np.int64) + start))
            order = np.lexsort((candidate_ids, -candidate_scores))[:K]
            best_scores[j] = candidate_scores[order]
            best_ids[j] = candidate_ids[order]
    return [(ids.tolist(), scores.tolist()) for ids, scores in zip(best_ids, best_scores)]


def main():
    matrix = np.load(DATA / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    queries = np.load(QUERIES, allow_pickle=False)
    cases = json.loads(CASES.read_text())
    assert matrix.shape == (COUNT, 1536) and len(queries) == len(cases) == 8
    configs = [
        ('1536d_float32', 1536, False, 4),
        ('1536d_float16_values', 1536, True, 2),
        ('512d_float32', 512, False, 4),
        ('512d_float16_values', 512, True, 2),
    ]
    results = {}
    for name, dims, half, itemsize in configs:
        start = time.perf_counter()
        per_query = []
        ranked = search_all(matrix, queries, dims, half)
        for case, (ids, scores) in zip(cases, ranked):
            per_query.append({
                'query': case['query'],
                'label': case['label'],
                'top_ids': [str(i) for i in ids],
                'top_scores': [round(float(s), 6) for s in scores],
                'accepted_at_existing_0_60_cutoff': bool(scores and scores[0] >= 0.60),
                'known_passage_in_top5': case['top_id'] in {str(i) for i in ids} if case['label'] == 'in' else None,
            })
        elapsed = (time.perf_counter() - start) * 1000
        baseline = results.get('1536d_float32')
        for i, row in enumerate(per_query):
            row['top5_overlap_with_1536_float32'] = None if baseline is None else len(set(row['top_ids']) & set(baseline['queries'][i]['top_ids']))
            row['top1_matches_1536_float32'] = None if baseline is None else row['top_ids'][0] == baseline['queries'][i]['top_ids'][0]
        results[name] = {
            'dimensions': dims,
            'stored_dtype': 'float16' if half else 'float32',
            'matrix_bytes': COUNT * dims * itemsize,
            'elapsed_ms_for_eight_queries': round(elapsed, 1),
            'queries': per_query,
        }
    report = {
        'method': 'offline exact scans; saved query vectors; truncated vectors are re-normalized; float16 is simulated by rounding vectors to float16, converting back to float32 for scoring, and re-normalizing',
        'query_set_size': len(cases),
        'query_set_limits': 'Eight manually labeled local smoke cases (four in, four out); not a judged MS MARCO test set and not a general accuracy estimate.',
        'results': results,
    }
    out = ROOT / 'diagnostics/memory-quality-comparison.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, result in results.items():
        rows = result['queries']
        in_rows = [r for r in rows if r['label'] == 'in']
        out_rows = [r for r in rows if r['label'] == 'out']
        overlap = [r['top5_overlap_with_1536_float32'] for r in rows if r['top5_overlap_with_1536_float32'] is not None]
        match1 = [r['top1_matches_1536_float32'] for r in rows if r['top1_matches_1536_float32'] is not None]
        print(json.dumps({
            'configuration': name,
            'matrix_bytes': result['matrix_bytes'],
            'known_passage_hits_at_5': sum(r['known_passage_in_top5'] for r in in_rows),
            'positive_queries_accepted_at_0_60': sum(r['accepted_at_existing_0_60_cutoff'] for r in in_rows),
            'negative_queries_accepted_at_0_60': sum(r['accepted_at_existing_0_60_cutoff'] for r in out_rows),
            'mean_top5_overlap_with_baseline': None if not overlap else round(sum(overlap) / (5 * len(overlap)), 3),
            'top1_matches_with_baseline': None if not match1 else sum(match1),
            'elapsed_ms_for_eight_queries': result['elapsed_ms_for_eight_queries'],
        }))
    print('report:', out)


if __name__ == '__main__':
    main()
