from pathlib import Path
import faiss
import joblib
import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_query(text):
    with torch.no_grad():
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = model(**encoded)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    faiss.normalize_L2(emb)
    return emb


def get_project_root():
    """
    Robust project root detection
    """
    current = Path(__file__).resolve()
    while current.name != "book-genre-rag":
        current = current.parent
    return current


def retrieve_books(query, top_k=5):
    project_root = get_project_root()

    index_path = project_root / "data" / "embeddings" / "faiss.index"
    meta_path = project_root / "data" / "embeddings" / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(index_path))
    metadata = joblib.load(meta_path)

    query_emb = embed_query(query)
    scores, indices = index.search(query_emb, top_k)

    results = metadata.iloc[indices[0]].copy()
    results["similarity"] = scores[0]
    max_similarity = scores[0][0]
    return results, max_similarity

