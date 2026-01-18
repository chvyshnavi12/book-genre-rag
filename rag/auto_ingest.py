import requests
import faiss
import joblib
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Load embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# -----------------------------
# Helper: normalize query
# -----------------------------
def normalize_query(query: str) -> str:
    """
    Normalize user query into title-like form
    Example: 'four wings' -> 'Four Wings'
    """
    return " ".join(word.capitalize() for word in query.split())


# -----------------------------
# Embedding function (FIX 2)
# -----------------------------
def embed_text(text: str):
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


# -----------------------------
# Google Books fetch (FILTERED)
# -----------------------------
def fetch_from_google(query: str):
    url = "https://www.googleapis.com/books/v1/volumes"

    params = {
        "q": f"intitle:{query}",
        "maxResults": 10,
        "printType": "books",
        "orderBy": "relevance"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if "items" not in data:
        return None

    bad_keywords = [
        "summary", "guide", "analysis", "study",
        "notes", "calendar", "prophecy", "dissertation"
    ]

    for item in data["items"]:
        info = item["volumeInfo"]

        title = info.get("title", "")
        description = info.get("description", "")

        # 🚫 Reject junk books
        if any(bad in title.lower() for bad in bad_keywords):
            continue

        if len(description) < 80:
            continue

        return {
            "title": title,
            "authors": ", ".join(info.get("authors", [])),
            "description": description,
            "genres": ", ".join(info.get("categories", [])),
            "categories": ", ".join(info.get("categories", [])),
            "published_year": info.get("publishedDate", "")[:4],
        }

    return None


# -----------------------------
# Auto inject logic (FINAL)
# -----------------------------
def auto_inject_if_needed(query: str, similarity_threshold=0.35):
    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "data" / "embeddings" / "faiss.index"
    meta_path = project_root / "data" / "embeddings" / "metadata.pkl"

    index = faiss.read_index(str(index_path))
    metadata = joblib.load(meta_path)

    normalized_query = normalize_query(query)

    # 🚫 Prevent duplicates
    if metadata["title"].str.lower().str.contains(normalized_query.lower()).any():
        return False

    print(f"🌐 Searching Google Books for: {normalized_query}")
    book = fetch_from_google(normalized_query)

    if not book:
        print("❌ No valid canonical book found")
        return False

    # 🧠 Structured semantic text (FIX 2)
    text = (
        f"Title: {book['title']}. "
        f"Description: {book['description']}. "
        f"Genres: {book['genres']}. "
        f"Authors: {book['authors']}."
    )

    emb = embed_text(text)

    index.add(emb)
    metadata = pd.concat([metadata, pd.DataFrame([book])], ignore_index=True)

    faiss.write_index(index, str(index_path))
    joblib.dump(metadata, meta_path)

    print(f"✅ Auto-added canonical book: {book['title']}")
    print(f"📚 Total books in index: {index.ntotal}")

    return True
