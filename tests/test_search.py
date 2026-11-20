import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app import create_app, rank_all


class SearchTest(unittest.TestCase):
    def test_exact_search_every_row_and_ties(self):
        matrix = np.array([[0, 1], [1, 0], [1, 0], [-1, 0]], dtype=np.float32)
        result = rank_all(matrix, np.array([1, 0], dtype=np.float32), 4)
        self.assertEqual([i for i, _ in result], [1, 2, 0, 3])
        np.testing.assert_allclose([s for _, s in result], [1, 1, 0, -1])

    def test_returns_top_k_even_when_scores_are_weak(self):
        matrix = np.array([[.20, 0], [.80, 0], [.59, 0], [.60, 0], [.95, 0]], dtype=np.float32)
        query = np.array([1, 0], dtype=np.float32)
        self.assertEqual([i for i, _ in rank_all(matrix, query, 5)], [4, 1, 3, 2, 0])
        self.assertEqual([i for i, _ in rank_all(matrix, query, 1)], [4])
        self.assertEqual(rank_all(matrix[:1], query, 5), [(0, .20)])

    def test_validation_and_local_access(self):
        client = create_app().test_client()
        for body in [None, [], {}, {'query': ''}, {'query': 3}, {'query': 'a', 'k': True}, {'query': 'a', 'k': 21}, {'query': 'x' * 1001}]:
            self.assertEqual(client.post('/api/search', json=body).status_code, 400)
        self.assertEqual(client.get('/api/status', headers={'Host': 'evil.example'}).status_code, 403)
        self.assertEqual(client.post('/api/search', json={'query': 'x'}, headers={'Origin': 'https://evil.example'}).status_code, 403)

    def test_budget_blocks_api_before_request(self):
        import embeddings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'budget.json').write_text(json.dumps({'reserved_usd': embeddings.BUDGET, 'tokens_reserved': 100000, 'requests': 1}))
            with patch.object(embeddings, 'DATA', path), patch.object(embeddings.requests, 'post') as call:
                with self.assertRaisesRegex(RuntimeError, 'budget'):
                    embeddings.embed(['hello'])
                call.assert_not_called()

    def test_failed_call_still_reserves_budget(self):
        import embeddings
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch.object(embeddings, 'DATA', path), patch.object(embeddings, 'token', return_value='test-only'), patch.object(embeddings.requests, 'post', side_effect=embeddings.requests.Timeout) as call:
                with self.assertRaisesRegex(RuntimeError, 'No automatic retry'):
                    embeddings.embed(['hello'])
                self.assertEqual(call.call_count, 1)
                self.assertGreater(json.loads((path / 'budget.json').read_text())['reserved_usd'], 0)

    def test_compact_embedding_response(self):
        import base64
        import embeddings
        raw = np.zeros(1536, dtype='<f4')
        raw[0] = 2
        response = unittest.mock.Mock(ok=True)
        response.json.return_value = {'model': embeddings.MODEL, 'data': [{'index': 0, 'embedding': base64.b64encode(raw.tobytes()).decode()}]}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(embeddings, 'DATA', Path(folder)), patch.object(embeddings, 'token', return_value='test-only'), patch.object(embeddings.requests, 'post', return_value=response) as call:
                vector = embeddings.embed(['hello'])
                self.assertEqual(call.call_args.kwargs['json']['encoding_format'], 'base64')
                self.assertEqual(vector.shape, (1, 1536))
                self.assertEqual(vector.dtype, np.float32)
                self.assertEqual(vector[0, 0], 1)
                self.assertEqual(float(np.linalg.norm(vector)), 1)

    def test_real_matrix_with_mock_query_and_cache(self):
        client = create_app().test_client()
        if not client.get('/api/status').json['ready']:
            self.skipTest('Run prepare.py first')
        matrix = np.load('data/search/vectors.npy')
        with patch('app.embed', return_value=matrix[:1]) as call:
            first = client.post('/api/search', json={'query': 'self-match'}).json
            second = client.post('/api/search', json={'query': 'self-match'}).json
        self.assertAlmostEqual(first['results'][0]['score'], 1, places=5)
        self.assertEqual(first['scanned'], len(matrix))
        self.assertTrue(second['cached'])
        self.assertEqual(call.call_count, 1)
        with patch('app.embed', return_value=-matrix[:1]):
            response = client.post('/api/search', json={'query': 'unrelated-query'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['results']), 5)
        scores = [row['score'] for row in response.json['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(response.json['scanned'], len(matrix))

    def test_saved_queries_always_return_ranked_candidates(self):
        queries_path = Path('data/search/relevance-check-queries.npy')
        if not queries_path.exists():
            self.skipTest('Local query vectors are not present')
        vectors = np.load(queries_path)
        matrix = np.load('data/search/vectors.npy')
        for index, query in enumerate(vectors):
            with self.subTest(query=index):
                result = rank_all(matrix, query, 5)
                self.assertEqual(len(result), 5)
                self.assertEqual([score for _, score in result], sorted((score for _, score in result), reverse=True))


if __name__ == '__main__':
    unittest.main()
