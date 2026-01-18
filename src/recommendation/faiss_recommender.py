from pathlib import Path
import faiss
import joblib
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Load PyTorch embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_text(text):
    """
    Generate embedding for a single query text
    """
    with torch.no_grad():
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = model(**encoded)

        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.cpu().numpy()


# -----------------------------
# Load FAISS + metadata
# -----------------------------
def load_faiss():
    project_root = Path(__file__).resolve().parents[2]

    index = faiss.read_index(
        str(project_root / "data" / "embeddings" / "faiss.index")
    )

    metadata = joblib.load(
        project_root / "data" / "embeddings" / "metadata.pkl"
    )

    return index, metadata


# -----------------------------
# Recommend books (COSINE SIM)
# -----------------------------
def recommend_books_semantic(query_text, top_k=5):
    index, metadata = load_faiss()

    # Generate query embedding
    query_embedding = embed_text(query_text)

    # 🔥 CRITICAL FIX: normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    similarity_scores, indices = index.search(query_embedding, top_k)

    results = metadata.iloc[indices[0]].copy()

    # Rename distance → similarity_score (cosine similarity)
    results["similarity_score"] = similarity_scores[0]

    return results[["title", "authors", "genres", "similarity_score"]]


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    query = """
    Harry Potter wizard school magic friendship dark lord
    """

    recs = recommend_books_semantic(query, top_k=5)

    print("\n📚 Semantic Recommendations:\n")
    for _, row in recs.iterrows():
        print(
            f"• {row['title']} by {row['authors']} "
            f"(similarity: {row['similarity_score']:.2f})"
        )
