"""Build Space embeddings; credentials stay in the backend process."""
import base64
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import requests
import tiktoken

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'search'
MODEL = 'text-embedding-3-small'
BASE = 'https://space.ai-builders.com/backend/v1'
BUDGET = 0.25
# New calls reserve twice the previously observed $0.02/M rate; never auto-retry.
RESERVE_PER_TOKEN = 0.04 / 1_000_000


def token():
    value = os.environ.get('AI_BUILDER_TOKEN')
    if not value:
        raise RuntimeError('Build Space credentials not found. Set AI_BUILDER_TOKEN before starting the app.')
    return value


def embed(texts, *, data_dir=None, budget=None):
    data_dir = DATA if data_dir is None else Path(data_dir)
    budget = BUDGET if budget is None else budget
    if not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget < 0:
        raise ValueError("The budget must be a finite nonnegative amount.")
    enc = tiktoken.get_encoding('cl100k_base')
    counts = [len(enc.encode(t, disallowed_special=())) for t in texts]
    if not counts or any(n == 0 or n > 8191 for n in counts) or sum(counts) > 250000:
        raise ValueError('Text is empty or exceeds the embedding input limit.')
    data_dir.mkdir(parents=True, exist_ok=True)
    # Lock also covers the request so concurrent searches cannot overspend the ledger.
    with (data_dir / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = data_dir / 'budget.json'
        ledger = json.loads(path.read_text()) if path.exists() else {'reserved_usd': 0, 'tokens_reserved': 0, 'requests': 0}
        reserve = sum(counts) * RESERVE_PER_TOKEN
        if ledger['reserved_usd'] + reserve > budget + 1e-12:
            raise RuntimeError(f'The authorized ${budget:.2f} budget is exhausted. No API request was made.')
        credential = token()
        ledger['reserved_usd'] += reserve
        ledger['tokens_reserved'] += sum(counts)
        ledger['requests'] += 1
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(path)
        try:
            response = requests.post(BASE + '/embeddings', headers={'Authorization': 'Bearer ' + credential}, json={'model': MODEL, 'input': texts, 'encoding_format': 'base64'}, timeout=(10, 90))
        except requests.RequestException:
            raise RuntimeError('Could not connect to the embedding service. No automatic retry was attempted to avoid duplicate charges.') from None
        if not response.ok:
            raise RuntimeError(f'Embedding service returned HTTP {response.status_code}. No automatic retry was attempted.')
        payload = response.json()
        rows = sorted(payload.get('data', []), key=lambda r: r['index'])
        if payload.get('model') != MODEL or [r['index'] for r in rows] != list(range(len(texts))):
            raise RuntimeError('The embedding service returned an unexpected model or passage order.')
        vectors = np.asarray([np.frombuffer(base64.b64decode(r['embedding'], validate=True), dtype='<f4') for r in rows], dtype=np.float32)
        if vectors.shape != (len(texts), 1536) or not np.isfinite(vectors).all():
            raise RuntimeError('The embedding service returned invalid vector dimensions or values.')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError('The embedding service returned a zero vector.')
        return vectors / norms
