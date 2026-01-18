from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import joblib
import torch

from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Load HF model (PyTorch only)
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_texts(texts, batch_size=16):
    """
    Generate sentence embeddings using mean pooling
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            outputs = model(**encoded)

            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def build_faiss_index():
    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "processed" / "books_with_genres.csv"
    index_path = project_root / "data" / "embeddings" / "faiss.index"
    meta_path = project_root / "data" / "embeddings" / "metadata.pkl"

    index_path.parent.mkdir(parents=True, exist_ok=True)

    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)

    texts = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    ).tolist()

    print("🧠 Generating embeddings (PyTorch only)...")
    embeddings = embed_texts(texts)

    # 🔥 CRITICAL FIX 1: Normalize embeddings
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print(f"📐 Embedding dimension: {dim}")

    # 🔥 CRITICAL FIX 2: Use Inner Product (Cosine Similarity)
    print("📦 Building FAISS index (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    joblib.dump(df, meta_path)

    print("✅ FAISS index built successfully")
    print(f"🔢 Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    build_faiss_index()
