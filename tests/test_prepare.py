import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import prepare


class PrepareTest(unittest.TestCase):
    def test_reuses_batches_resumes_and_preserves_old_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'prepared'
            data.mkdir()
            source = root / 'data/msmarco'
            source.mkdir(parents=True)
            (source / 'collection.first1m.tsv').write_text(''.join(f'{i}\tPassage {i}\n' for i in range(6)))
            vectors = np.zeros((6, 1536), dtype=np.float32)
            vectors[:, 0] = 1
            np.save(data / 'batch-0000.npy', vectors[:2])
            np.save(data / 'vectors.npy', vectors[:2])
            (data / 'passages.json').write_text('[]')
            (data / 'metadata.json').write_text(json.dumps({'count': 2}))
            with patch.multiple(prepare, ROOT=root, DATA=data, TARGET_COUNT=6, BATCH_SIZE=2, REQUEST_SIZE=4), patch('prepare.embed', side_effect=[vectors[2:4], vectors[4:]]) as call:
                prepare.main()
                self.assertEqual(call.call_args_list[0].args[0], ['Passage 2', 'Passage 3'])
                self.assertEqual(call.call_count, 2)
                prepare.main()
                self.assertEqual(call.call_count, 2)
            self.assertEqual(np.load(data / 'vectors.npy').shape, (6, 1536))
            self.assertEqual(json.loads((data / 'metadata.json').read_text())['count'], 6)
            self.assertEqual(len(json.loads((data / 'passages.json').read_text())), 6)
            self.assertEqual(np.load(data / 'backup-2/vectors.npy').shape, (2, 1536))
