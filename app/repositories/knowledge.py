import os
import json
from openai import OpenAI
import logging
from typing import List, Tuple
from app.app_logging.logging_config import setup_logging

setup_logging()

class KnowledgeBase:
    def __init__(self, documents_dir: str, saved_embeddings_dir: str, embedding_model: str = "text-embedding-3-small", chunk_size: int = 800):
        """
        Initialize the knowledge base.
        :param documents_dir: Path to the directory containing .txt files for the chosen database.
        :param saved_embeddings_dir: Path to the directory where embeddings will be saved.
        :param embedding_model: OpenAI embedding model to use for semantic search.
        :param chunk_size: Maximum number of characters in each chunk.
        """
        self.documents_dir = documents_dir
        self.saved_embeddings_dir = saved_embeddings_dir

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size

        self.chunks = []  # List to store (chunk_id, filename, text)
        self.embeddings = {}  # Dictionary to store chunk embeddings

        self.logger = logging.getLogger(__name__)
        self.logger.info("KnowledgeBase initialized")

        self.client = OpenAI()
        self.database_name = os.path.basename(os.path.normpath(documents_dir))
        self.embeddings_file = os.path.join(saved_embeddings_dir, f"{self.database_name}_embeddings.json")
        self._load_existing_embeddings()

        self._load_and_chunk_documents()

    def _load_existing_embeddings(self):
        """
        Load existing embeddings from the JSON file if it exists.
        """
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.embeddings = data.get("embeddings", {})
                self.chunks = data.get("chunks", [])
            self.logger.info("Loaded existing embeddings from file.")
        else:
            self.logger.info("No existing embeddings file found.")

    def _load_and_chunk_documents(self):
        """
        Load all .txt files from the directory and split them into chunks.
        """
        for filename in os.listdir(self.documents_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Split content into chunks
                for i in range(0, len(content), self.chunk_size):
                    chunk_text = content[i:i + self.chunk_size]
                    chunk_id = f"{filename}_chunk_{i // self.chunk_size}"
                    self.chunks.append((chunk_id, filename, chunk_text))

        self.logger.info(f"Loaded and chunked {len(self.chunks)} sections from files.")

    def generate_embeddings(self):
        """
        Generate and store embeddings for all chunks in the knowledge base.
        """
        new_embeddings = {}
        for chunk_id, filename, chunk_text in self.chunks:
            if chunk_id not in self.embeddings:
                response = self.client.embeddings.create(
                    input=chunk_text,
                    model=self.embedding_model
                )
                new_embeddings[chunk_id] = response.data[0].embedding
                self.embeddings[chunk_id] = response.data[0].embedding

        if new_embeddings:
            self._save_embeddings()
            self.logger.info(f"Embeddings generated for {len(new_embeddings)} new chunks.")
        else:
            self.logger.info("No new embeddings were generated.")

    def _save_embeddings(self):
        """
        Save the embeddings and chunks to a JSON file.
        """
        data = {
            "embeddings": self.embeddings,
            "chunks": self.chunks
        }
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
        with open(self.embeddings_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        self.logger.info("Embeddings saved to file.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str, float]]:
        """
        Search for the most relevant chunks to a query.
        :param query: User's query.
        :param top_k: Number of top results to return.
        :return: List of tuples (chunk_id, filename, text, similarity_score).
        """
        # Check if embeddings exist
        if not self.embeddings:
            self.logger.info("No embeddings found. Generating embeddings...")
            self.generate_embeddings()

        # Generate query embedding
        query_embedding = self.client.embeddings.create(
            input=query,
            model=self.embedding_model
        ).data[0].embedding

        # Compute cosine similarity with all chunk embeddings
        results = []
        for chunk_id, (_, filename, text) in zip(self.embeddings.keys(), self.chunks):
            similarity = self._cosine_similarity(query_embedding, self.embeddings[chunk_id])
            results.append((chunk_id, filename, text, similarity))

        # Sort by similarity and return top_k results, but filtered
        results = sorted(results, key=lambda x: x[3], reverse=True)
        filtered_results = [result for result in results if result[3] >= 0.5]
        return filtered_results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """
        Compute the cosine similarity between two vectors.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude = (sum(a**2 for a in vec1) ** 0.5) * (sum(b**2 for b in vec2) ** 0.5)
        return dot_product / magnitude if magnitude else 0.0