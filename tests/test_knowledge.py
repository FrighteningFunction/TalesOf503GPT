import unittest
from unittest.mock import patch, mock_open
from app.repositories.knowledge import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):

    @patch('os.listdir', return_value=['document1.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data="This is a test document.")
    def test_load_and_chunk_documents(self, mock_file, mock_listdir):
        kb = KnowledgeBase(directory='test_directory')
        self.assertEqual(len(kb.chunks), 1)
        self.assertEqual(kb.chunks[0][1], 'document1.txt')
        self.assertEqual(kb.chunks[0][2], 'This is a test document.')

    @patch('os.listdir', return_value=['document1.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data="This is a test document.")
    @patch('openai.Embedding.create', return_value={'data': [{'embedding': [0.1, 0.2, 0.3]}]})
    def test_generate_embeddings(self, mock_embedding, mock_file, mock_listdir):
        kb = KnowledgeBase(directory='test_directory')
        kb.generate_embeddings()
        self.assertEqual(len(kb.embeddings), 1)
        self.assertIn('document1.txt_chunk_0', kb.embeddings)
        self.assertEqual(kb.embeddings['document1.txt_chunk_0'], [0.1, 0.2, 0.3])

    @patch('os.listdir', return_value=['document1.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data="This is a test document.")
    @patch('openai.Embedding.create', side_effect=[
        {'data': [{'embedding': [0.1, 0.2, 0.3]}]},
        {'data': [{'embedding': [0.4, 0.5, 0.6]}]}
    ])
    def test_search(self, mock_embedding, mock_file, mock_listdir):
        kb = KnowledgeBase(directory='test_directory')
        kb.generate_embeddings()
        results = kb.search(query="test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 'document1.txt')
        self.assertEqual(results[0][2], 'This is a test document.')

    def test_cosine_similarity(self):
        vec1 = [1, 0, -1]
        vec2 = [-1, 0, 1]
        similarity = KnowledgeBase._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, -1.0)

    # more advanced tests with edge cases

    @patch('os.listdir', return_value=[])
    def test_empty_directory(self, mock_listdir):
        kb = KnowledgeBase(directory='test_directory')
        self.assertEqual(len(kb.chunks), 0)
        self.assertEqual(len(kb.embeddings), 0)
    

    @patch('os.listdir', return_value=['empty.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data="")
    def test_empty_file(self, mock_file, mock_listdir):
        kb = KnowledgeBase(directory='test_directory')
        self.assertEqual(len(kb.chunks), 0)

    @patch('os.listdir', return_value=['large_document.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data="A" * 1500)  # Large file content
    def test_large_file_chunking(self, mock_file, mock_listdir):
        kb = KnowledgeBase(directory='test_directory', chunk_size=500)
        self.assertEqual(len(kb.chunks), 3)
        self.assertEqual(kb.chunks[0][2], "A" * 500)
        self.assertEqual(kb.chunks[1][2], "A" * 500)
        self.assertEqual(kb.chunks[2][2], "A" * 500)

if __name__ == '__main__':
    unittest.main()
