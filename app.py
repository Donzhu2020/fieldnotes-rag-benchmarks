import json
import threading
import time

from flask import Flask, jsonify, render_template, request
import numpy as np
from embeddings import DATA, MODEL, embed

SCAN_CHUNK_SIZE = 8192


def rank_all(matrix, query, k):
    """Exact cosine search: every row participates, no index or ANN."""
    if matrix.dtype == np.float16:
        scores = np.empty(len(matrix), dtype=np.float32)
        query = np.asarray(query, dtype=np.float32)
        for start in range(0, len(matrix), SCAN_CHUNK_SIZE):
            block = matrix[start:start + SCAN_CHUNK_SIZE].astype(np.float32)
            norms = np.linalg.norm(block, axis=1)
            block /= norms[:, None]
            scores[start:start + len(block)] = block @ query
    else:
        scores = matrix @ query
    order = np.argsort(-scores, kind='stable')[:k]
    return [(int(i), float(scores[i])) for i in order]


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8192
    matrix = None
    passages = []
    if (DATA / 'vectors.npy').exists():
        meta = json.loads((DATA / 'metadata.json').read_text())
        if meta['model'] != MODEL or meta['dimensions'] not in (512, 1536) or meta.get('dtype') not in ('float16', 'float32'):
            raise RuntimeError('The passage and query embedding models do not match.')
        # np.load without mmap: the entire matrix lives in RAM.
        matrix = np.load(DATA / 'vectors.npy', allow_pickle=False)
        passages = json.loads((DATA / 'passages.json').read_text())
        if matrix.shape != (len(passages), meta['dimensions']) or len(passages) != meta['count'] or matrix.dtype != np.dtype(meta['dtype']) or not np.isfinite(matrix).all():
            raise RuntimeError('Vector and passage files are inconsistent or corrupted.')
    cache = {}
    search_lock = threading.Lock()

    @app.before_request
    def local_only():
        if request.host.split(':')[0] not in ('127.0.0.1', 'localhost'):
            return jsonify(error='Only local access is allowed.'), 403
        if request.method == 'POST' and request.headers.get('Origin') not in (None, 'http://' + request.host):
            return jsonify(error='Cross-site requests are not allowed.'), 403

    @app.after_request
    def headers(response):
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/status')
    def status():
        return jsonify(ready=matrix is not None, count=len(passages), dimensions=matrix.shape[1] if matrix is not None else 512, dtype=str(matrix.dtype) if matrix is not None else None, model=MODEL, memory_bytes=matrix.nbytes if matrix is not None else 0, corpus_count=1000000)

    @app.post('/api/search')
    def search():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('query'), str):
            return jsonify(error='Please enter a text query.'), 400
        query = body['query'].strip()
        k = body.get('k', 5)
        if not query or len(query) > 1000 or type(k) is not int or not 1 <= k <= 20:
            return jsonify(error='Queries must contain 1–1000 characters. The result count must be between 1 and 20.'), 400
        if matrix is None:
            return jsonify(error='Vectors are not ready. Run python3 prepare.py first.'), 503
        started = time.perf_counter()
        if not search_lock.acquire(blocking=False):
            return jsonify(error='Another search is in progress. Please try again shortly.'), 429
        try:
            cached = query in cache
            embedding_start = time.perf_counter()
            if not cached:
                value = embed([query])[0][:matrix.shape[1]].astype(np.float32)
                value /= np.linalg.norm(value)
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
                cache[query] = value
            embedding_ms = (time.perf_counter() - embedding_start) * 1000
            scan_start = time.perf_counter()
            ranked = rank_all(matrix, cache[query], k)
            scan_ms = (time.perf_counter() - scan_start) * 1000
            return jsonify(results=[dict(passages[i], score=score) for i, score in ranked], scanned=len(passages), scan_ms=scan_ms, embedding_ms=embedding_ms, total_ms=(time.perf_counter() - started) * 1000, cached=cached, query=query)
        except (RuntimeError, ValueError) as e:
            return jsonify(error=str(e)), 502
        except Exception:
            app.logger.exception('Search failed')
            return jsonify(error='Search failed. Please check the local server logs.'), 502
        finally:
            search_lock.release()

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=8765, debug=False)
