"""Prepare one million MS MARCO passages, reusing saved 100-passage batches."""
import itertools
import json
import shutil
import numpy as np
import tiktoken
from embeddings import DATA, MODEL, ROOT, embed

TARGET_COUNT = 1_000_000
BATCH_SIZE = 100
REQUEST_SIZE = 500
MAX_REQUEST_TOKENS = 200_000
EXPANSION_BUDGET = 2.88


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / 'data/msmarco/collection.first1m.tsv'
    enc = tiktoken.get_encoding('cl100k_base')
    with source_path.open() as source:
        for start in range(0, TARGET_COUNT, REQUEST_SIZE):
            pending = []
            pending_tokens = 0

            def flush():
                nonlocal pending, pending_tokens
                if not pending:
                    return
                texts = [text for _, block in pending for text in block]
                vectors = embed(texts, data_dir=DATA / 'expansion_1m', budget=EXPANSION_BUDGET)
                cursor = 0
                for offset, block in pending:
                    path = DATA / f'batch-{offset:04}.npy'
                    temp = path.with_suffix('.tmp')
                    with temp.open('wb') as f:
                        np.save(f, vectors[cursor:cursor + len(block)])
                    temp.replace(path)
                    cursor += len(block)
                pending = []
                pending_tokens = 0

            end = min(start + REQUEST_SIZE, TARGET_COUNT)
            for offset in range(start, end, BATCH_SIZE):
                rows = []
                for _ in range(min(BATCH_SIZE, end - offset)):
                    fields = next(source).rstrip('\n').split('\t', 1)
                    if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
                        raise ValueError(f'Invalid source passage at row {offset + len(rows) + 1}.')
                    rows.append((fields[0], fields[1]))
                path = DATA / f'batch-{offset:04}.npy'
                if path.exists():
                    continue
                texts = [text for _, text in rows]
                token_count = sum(len(enc.encode(text, disallowed_special=())) for text in texts)
                if pending and pending_tokens + token_count > MAX_REQUEST_TOKENS:
                    flush()
                if token_count > MAX_REQUEST_TOKENS:
                    raise ValueError(f'Passage batch at row {offset + 1} exceeds the request token budget.')
                pending.append((offset, texts))
                pending_tokens += token_count
            flush()
            print(f'{end:,}/{TARGET_COUNT:,} passages prepared', flush=True)

    vectors_tmp = DATA / 'vectors.tmp'
    matrix = np.lib.format.open_memmap(vectors_tmp, mode='w+', dtype=np.float32, shape=(TARGET_COUNT, 512))
    for start in range(0, TARGET_COUNT, BATCH_SIZE):
        batch = np.load(DATA / f'batch-{start:04}.npy', allow_pickle=False)
        size = min(BATCH_SIZE, TARGET_COUNT - start)
        if batch.shape != (size, 1536) or not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-4):
            raise ValueError(f'Invalid saved vector batch at passage {start}.')
        reduced = batch[:, :512].copy()
        reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
        matrix[start:start + size] = reduced
    matrix.flush()
    del matrix

    with source_path.open() as source, (DATA / 'passages.tmp').open('w') as target:
        target.write('[')
        for i, line in enumerate(itertools.islice(source, TARGET_COUNT)):
            fields = line.rstrip('\n').split('\t', 1)
            if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
                raise ValueError(f'Invalid source passage at row {i + 1}.')
            if i:
                target.write(',')
            json.dump({'id': fields[0], 'text': fields[1]}, target, ensure_ascii=False, separators=(',', ':'))
        target.write(']')

    (DATA / 'metadata.tmp').write_text(json.dumps({'model': MODEL, 'dimensions': 512, 'count': TARGET_COUNT, 'source': 'MS MARCO Passage Ranking v1', 'selection': f'first {TARGET_COUNT} rows', 'dtype': 'float32', 'reduction': 'first 512 coordinates, L2-normalized'}, indent=2))
    if (DATA / 'metadata.json').exists():
        old_count = json.loads((DATA / 'metadata.json').read_text())['count']
        backup = DATA / f'backup-{old_count}'
        if old_count != TARGET_COUNT and not backup.exists():
            backup.mkdir()
            for name in ['vectors.npy', 'passages.json', 'metadata.json']:
                shutil.copy2(DATA / name, backup / name)
    for staged, active in [('vectors.tmp', 'vectors.npy'), ('passages.tmp', 'passages.json'), ('metadata.tmp', 'metadata.json')]:
        (DATA / staged).replace(DATA / active)
    print('Ready:', TARGET_COUNT, 'passages;', TARGET_COUNT * 512 * 4, 'vector bytes (float32)')


if __name__ == '__main__':
    main()
