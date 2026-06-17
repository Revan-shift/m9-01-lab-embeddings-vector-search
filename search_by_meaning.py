"""
Lab | Search by Meaning, by Hand
---------------------------------
This script embeds a small knowledge base, embeds a set of test queries,
and finds the best-matching passages by computing cosine similarity
BY HAND with NumPy. No vector store, no built-in search function -
just a dot product and a couple of norms.

Usage:
    export GOOGLE_API_KEY="your-free-gemini-key"
    python search_by_meaning.py
"""

import json
import os
import numpy as np
from google import genai

# ---------------------------------------------------------------------
# 1) SET UP THE GEMINI CLIENT
# ---------------------------------------------------------------------
# The client automatically reads the GOOGLE_API_KEY environment
# variable. We never hardcode the key here - that's how we keep it
# out of the repo (see the "no API key committed" requirement).
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Sends a list of texts to the Gemini embedding model and returns
    their vectors as a single NumPy array of shape (N, D).

    N = number of texts, D = embedding dimensionality (e.g. 3072).
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    vectors = [np.array(e.values, dtype=np.float32) for e in response.embeddings]
    return np.vstack(vectors)


# ---------------------------------------------------------------------
# 2) COSINE SIMILARITY - WRITTEN BY HAND (NumPy, no shortcuts)
# ---------------------------------------------------------------------
def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity formula:

        cos(theta) = (A . B) / (||A|| * ||B||)

    Where:
      - A . B        -> dot product of the two vectors
      - ||A||, ||B|| -> Euclidean norm (length) of each vector

    query_vec:  embedding of a single query, shape (D,)
    doc_matrix: embeddings of every passage, shape (N, D)

    Returns: one similarity score per passage, shape (N,)
    """
    # Dot product between the query and every row (every passage) at once.
    # (N, D) @ (D,) -> (N,)
    dot_products = doc_matrix @ query_vec

    # Length of each passage vector
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    # Length of the query vector
    query_norm = np.linalg.norm(query_vec)

    # Small epsilon avoids division by zero
    similarities = dot_products / (doc_norms * query_norm + 1e-10)
    return similarities


# ---------------------------------------------------------------------
# 3) LOAD THE KNOWLEDGE BASE
# ---------------------------------------------------------------------
def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    knowledge_base = load_knowledge_base(kb_path)

    print(f"Loaded knowledge base: {len(knowledge_base)} passages.\n")

    passage_texts = [item["text"] for item in knowledge_base]

    print("Embedding passages (Gemini gemini-embedding-001)...")
    passage_vectors = embed_texts(passage_texts)
    print(f"Embedding matrix shape: {passage_vectors.shape}  "
          f"({passage_vectors.shape[0]} passages x {passage_vectors.shape[1]} dimensions)\n")

    # ---------------------------------------------------------------
    # 4) TEST QUERIES
    # ---------------------------------------------------------------
    test_queries = [
        "my laptop won't switch on",
        "how do I stop being billed every month?",
        "access denied error when saving a file",
        "where do I leave my car in the evening?",
    ]

    print("=" * 70)
    print("SEARCH RESULTS (top 3 passages per query)")
    print("=" * 70)

    for query in test_queries:
        query_vector = embed_texts([query])[0]
        scores = cosine_similarity(query_vector, passage_vectors)

        top_3_indices = np.argsort(scores)[::-1][:3]

        print(f"\nQuery: \"{query}\"")
        print("-" * 70)
        for rank, idx in enumerate(top_3_indices, start=1):
            item = knowledge_base[idx]
            print(f"  {rank}. [{item['id']} | {item['source']}] "
                  f"score={scores[idx]:.4f}")
            print(f"     \"{item['text']}\"")

    # ---------------------------------------------------------------
    # 5) OPTIONAL STRETCH - a query NOT covered by the knowledge base
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STRETCH: out-of-scope query")
    print("=" * 70)

    out_of_scope_query = "what's the wifi password?"
    query_vector = embed_texts([out_of_scope_query])[0]
    scores = cosine_similarity(query_vector, passage_vectors)
    top_idx = np.argmax(scores)
    best_item = knowledge_base[top_idx]

    print(f"\nQuery: \"{out_of_scope_query}\"")
    print(f"Best match: [{best_item['id']}] score={scores[top_idx]:.4f}")
    print(f"  \"{best_item['text']}\"")
    print("\n(This score is typically lower than the top score for the other")
    print(" queries - a similarity threshold could be used to flag")
    print(" 'we don't actually have an answer for this' cases.)")


if __name__ == "__main__":
    main()
