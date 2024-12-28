import logging
from app.repositories.knowledge import KnowledgeBase

def main():
    # Initialize the knowledge base
    kb = KnowledgeBase(documents_dir='app/repositories/documents/database1', saved_embeddings_dir='app/repositories/saved_embeddings')

    # Generate embeddings for the documents
    kb.generate_embeddings()

    # Perform a search
    query = "Keresés megoldások"
    results = kb.search(query=query, top_k=3)

    # Print the search results
    for chunk_id, filename, text, similarity in results:
        print(f"Filename: {filename}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Text: {text}")
        print(f"Similarity: {similarity}")
        print("-" * 40)

if __name__ == '__main__':
    main()
