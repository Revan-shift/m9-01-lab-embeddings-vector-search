"""
Lab | Search by Meaning, by Hand
---------------------------------
Bu script bir "knowledge base"-i embedding-lərə çevirir, sonra test
sualları üçün cosine similarity-i ÖZÜMÜZ NumPy ilə hesablayaraq
ən uyğun passage-ləri tapır. Hazır vector store axtarış funksiyası
işlədilmir - bütün riyaziyyat bizim öz əlimizlə yazılıb.

İşlətmək üçün:
    export GOOGLE_API_KEY="sənin-gemini-key-in"
    python search_by_meaning.py
"""

import json
import os
import numpy as np
from google import genai

# ---------------------------------------------------------------------
# 1) GEMINI CLIENT-İ QURMAQ
# ---------------------------------------------------------------------
# Client, environment-dəki GOOGLE_API_KEY dəyişənini avtomatik oxuyur.
# Key-i kod içində YAZMIRIQ (best practice - heç vaxt repo-ya commit etmə).
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Bir neçə mətni Gemini embedding modelinə göndərib,
    onların vektorlarını (N x D ölçülü NumPy array) qaytarır.

    N = mətnlərin sayı, D = embedding-in ölçüsü (məsələn 3072).
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    # Hər bir embedding obyektindən .values siyahısını çıxarırıq
    vectors = [np.array(e.values, dtype=np.float32) for e in response.embeddings]
    return np.vstack(vectors)


# ---------------------------------------------------------------------
# 2) COSINE SIMILARITY - ÖZÜMÜZ YAZIRIQ (numpy ilə, hazır funksiya yox)
# ---------------------------------------------------------------------
def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity formulu:

        cos(theta) = (A . B) / (||A|| * ||B||)

    Burada:
      - A . B          -> iki vektorun dot product-i (skalyar hasil)
      - ||A||, ||B||   -> vektorların uzunluğu (Euclidean norm)

    query_vec:  tək bir sualın embedding vektoru, ölçü (D,)
    doc_matrix: bütün passage-lərin embedding-ləri, ölçü (N, D)

    Qaytarır: hər passage üçün bir oxşarlıq skoru, ölçü (N,)
    """
    # Dot product: sualın vektoru ilə HƏR sətrin (hər passage-in) hasili
    # doc_matrix @ query_vec -> (N, D) @ (D,) = (N,) ölçülü nəticə
    dot_products = doc_matrix @ query_vec

    # Hər passage vektorunun norması (uzunluğu)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    # Sualın vektorunun norması (tək ədəd)
    query_norm = np.linalg.norm(query_vec)

    # Formula: dot / (norm_a * norm_b)
    # 1e-10 əlavə edirik ki, sıfıra bölünmə xətası baş verməsin
    similarities = dot_products / (doc_norms * query_norm + 1e-10)
    return similarities


# ---------------------------------------------------------------------
# 3) KNOWLEDGE BASE-İ YÜKLƏMƏK VƏ EMBED ETMƏK
# ---------------------------------------------------------------------
def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    knowledge_base = load_knowledge_base(kb_path)

    print(f"Knowledge base yükləndi: {len(knowledge_base)} passage tapıldı.\n")

    # Bütün passage mətnlərini bir siyahıya çıxarırıq
    passage_texts = [item["text"] for item in knowledge_base]

    print("Passage-lər embed edilir (Gemini gemini-embedding-001)...")
    passage_vectors = embed_texts(passage_texts)
    print(f"Embedding ölçüsü: {passage_vectors.shape}  "
          f"({passage_vectors.shape[0]} passage x {passage_vectors.shape[1]} dimension)\n")

    # ---------------------------------------------------------------
    # 4) TEST SUALLARI
    # ---------------------------------------------------------------
    test_queries = [
        "my laptop won't switch on",
        "how do I stop being billed every month?",
        "access denied error when saving a file",
        "where do I leave my car in the evening?",
    ]

    print("=" * 70)
    print("AXTARIŞ NƏTİCƪLƏRİ (hər sual üçün TOP 3 passage)")
    print("=" * 70)

    for query in test_queries:
        query_vector = embed_texts([query])[0]  # tək sualın vektoru
        scores = cosine_similarity(query_vector, passage_vectors)

        # Skorları böyükdən kiçiyə sıralayıb ən yaxşı 3-nü götürürük
        top_3_indices = np.argsort(scores)[::-1][:3]

        print(f"\nSual: \"{query}\"")
        print("-" * 70)
        for rank, idx in enumerate(top_3_indices, start=1):
            item = knowledge_base[idx]
            print(f"  {rank}. [{item['id']} | {item['source']}] "
                  f"score={scores[idx]:.4f}")
            print(f"     \"{item['text']}\"")

    # ---------------------------------------------------------------
    # 5) STRETCH (optional) - korpusda olmayan sual
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STRETCH: Korpusda olmayan sual")
    print("=" * 70)

    out_of_scope_query = "what's the wifi password?"
    query_vector = embed_texts([out_of_scope_query])[0]
    scores = cosine_similarity(query_vector, passage_vectors)
    top_idx = np.argmax(scores)
    best_item = knowledge_base[top_idx]

    print(f"\nSual: \"{out_of_scope_query}\"")
    print(f"Ən yaxşı uyğunluq: [{best_item['id']}] score={scores[top_idx]:.4f}")
    print(f"  \"{best_item['text']}\"")
    print("\n(Bu skor adətən digər sorğulardakı top skordan aşağı olur -")
    print(" bu fərq, threshold (həddi) qoyub 'cavabımız yoxdur' demək üçün")
    print(" istifadə edilə bilər.)")


if __name__ == "__main__":
    main()
