"""Compare 1536d and truncated 512d vectors on the fixed MS MARCO qrels sample."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / 'diagnostics'
DATA = ROOT / 'data' / 'search'
sys.path.insert(0, str(ROOT))
FORMATS = ('1536_float32', '1536_float16', '512_float32', '512_float16')


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def prepare_512(output):
    source = np.load(DATA / 'vectors.float32.backup.npy', mmap_mode='r', allow_pickle=False)
    assert source.shape == (1_000_000, 1536) and source.dtype == np.float32
    f32_path, f16_path = output / 'vectors-512-f32.npy', output / 'vectors-512-f16.npy'
    f32 = np.lib.format.open_memmap(f32_path, mode='w+', dtype=np.float32, shape=(len(source), 512))
    f16 = np.lib.format.open_memmap(f16_path, mode='w+', dtype=np.float16, shape=(len(source), 512))
    for start in range(0, len(source), 8192):
        block = np.array(source[start:start + 8192, :512], dtype=np.float32, copy=True)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        f32[start:start + len(block)] = block
        f16[start:start + len(block)] = block.astype(np.float16)
    f32.flush()
    f16.flush()
    del f32, f16
    queries = np.load(DIAGNOSTICS / 'msmarco-precision-query-vectors.npy', allow_pickle=False)
    reduced = queries[:, :512].copy()
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
    q512_path = output / 'queries-512.npy'
    np.save(q512_path, reduced, allow_pickle=False)
    return f32_path, f16_path, q512_path


def run_variant(name, matrix_path, query_path):
    import app

    matrix = np.load(matrix_path, allow_pickle=False)
    queries = json.loads((DIAGNOSTICS / 'msmarco-precision-queries.json').read_text())['queries']
    query_vectors = np.load(query_path, allow_pickle=False)
    dimensions, dtype = name.split('_')
    dimensions = int(dimensions)
    assert matrix.shape == (1_000_000, dimensions) and matrix.dtype == np.dtype(dtype)
    assert query_vectors.shape == (100, dimensions) and query_vectors.dtype == np.float32
    scan_ms, ranked_rows = [], []
    started = time.perf_counter()
    for vector in query_vectors:
        scan_start = time.perf_counter()
        ranked = app.rank_all(matrix, vector, 10)
        scan_ms.append((time.perf_counter() - scan_start) * 1000)
        ranked_rows.append({'ids': [str(i) for i, _ in ranked], 'scores': [s for _, s in ranked]})
    wall_ms = (time.perf_counter() - started) * 1000

    hit5, hit10, gated5, gated10, reciprocal = [], [], [], [], []
    for row, query in zip(ranked_rows, queries):
        relevant = set(query['relevant_pids'])
        ids, scores = row['ids'], row['scores']
        hit5.append(bool(relevant & set(ids[:5])))
        hit10.append(bool(relevant & set(ids[:10])))
        gated5.append(bool(relevant & {pid for pid, score in zip(ids[:5], scores[:5]) if score >= 0.60}))
        gated10.append(bool(relevant & {pid for pid, score in zip(ids[:10], scores[:10]) if score >= 0.60}))
        reciprocal.append(next((1 / rank for rank, pid in enumerate(ids, 1) if pid in relevant), 0.0))

    return {
        'name': name, 'dimensions': dimensions, 'dtype': dtype,
        'matrix_bytes': matrix.nbytes, 'peak_process_rss_bytes': peak_rss_bytes(),
        'wall_ms_for_100_queries': wall_ms, 'mean_scan_ms': float(np.mean(scan_ms)),
        'median_scan_ms': float(np.percentile(scan_ms, 50)),
        'p95_scan_ms': float(np.percentile(scan_ms, 95)),
        'hit_at_5': sum(hit5), 'hit_at_10': sum(hit10),
        'mrr_at_10': float(np.mean(reciprocal)),
        'hit_at_5_with_060_gate': sum(gated5), 'hit_at_10_with_060_gate': sum(gated10),
        'results': ranked_rows,
    }


def parent():
    sample = json.loads((DIAGNOSTICS / 'msmarco-precision-queries.json').read_text())
    q1536 = DIAGNOSTICS / 'msmarco-precision-query-vectors.npy'
    runs = []
    with tempfile.TemporaryDirectory(prefix='msmarco-512-') as folder:
        temp = Path(folder)
        f32_512, f16_512, q512 = prepare_512(temp)
        inputs = {
            '1536_float32': (DATA / 'vectors.float32.backup.npy', q1536),
            '1536_float16': (DATA / 'vectors.npy', q1536),
            '512_float32': (f32_512, q512),
            '512_float16': (f16_512, q512),
        }
        for name in FORMATS:
            matrix_path, query_path = inputs[name]
            command = [sys.executable, str(Path(__file__).resolve()), '--variant', name,
                       str(matrix_path), str(query_path)]
            process = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            runs.append(json.loads(process.stdout.strip().splitlines()[-1]))

    base = runs[0]
    for run in runs:
        overlap, same_top1 = [], []
        for a, b in zip(base['results'], run['results']):
            overlap.append(len(set(a['ids'][:5]) & set(b['ids'][:5])) / 5)
            same_top1.append(bool(a['ids'] and b['ids'] and a['ids'][0] == b['ids'][0]))
        run['top5_overlap_vs_1536_float32'] = float(np.mean(overlap))
        run['top1_match_count_vs_1536_float32'] = sum(same_top1)

    report = {
        'sample_count': len(sample['queries']), 'seed': sample['seed'],
        'eligible_qrels_query_count': sample['eligible_qrels_query_count'],
        'selection': 'Uniform fixed-seed sample from official dev qids with a qrels-positive passage in the indexed first million.',
        'query_vectors': 'The same 100 cached text-embedding-3-small vectors were used; no new API call.',
        '512_method': 'Take the first 512 coordinates of each existing 1536d vector, then L2-normalize; do the same to queries. Float16 variants round the normalized values to float16. This is an offline truncation simulation, not a fresh API-dimension embedding run.',
        'corpus': 'First 1,000,000 MS MARCO Passage Ranking v1 passages.',
        'hit_metrics': 'Primary Hit@5 and Hit@10 use top-k ranking without a score threshold. Historical 0.60 filtered rates are included separately.',
        'retrieval': 'Exact full scan; 100 queries, top 10; float16 chunks are converted to float32 and row-normalized during scoring.',
        'formats': runs,
        'limitations': [
            'The sample comes from 1,780 of 55,578 qrels dev qids with a judged relevant passage in the indexed first-million subset.',
            'Qrels are incomplete relevance judgments, not exhaustive ground truth.',
            'One sequential run per format; latency varies with CPU load and OS cache.',
            'Peak RSS includes the matrix and benchmark process overhead, but not passage text.',
            'The 512d vectors are locally truncated and normalized; a production API-dimension run should be validated separately.',
        ],
    }
    (DIAGNOSTICS / 'dimensions-100-results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'formats'} | {
        'formats': [{key: value for key, value in run.items() if key != 'results'} for run in runs]
    }, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--variant':
        print(json.dumps(run_variant(sys.argv[2], sys.argv[3], sys.argv[4])))
    else:
        parent()
