import requests
import pandas as pd
import faiss
import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_text(text):
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


def fetch_book_from_google(query):
    url = "https://www.googleapis.com/books/v1/volumes"

    params = {
        "q": query,
        "maxResults": 10,
        "printType": "books",
        "orderBy": "relevance"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if "items" not in data:
        return None

    for item in data["items"]:
        info = item["volumeInfo"]

        title = info.get("title", "")
        description = info.get("description", "")

        # 🚫 Reject junk
        bad_keywords = [
            "summary", "guide", "analysis", "study",
            "calendar", "prophecy", "notes"
        ]

        if any(bad in title.lower() for bad in bad_keywords):
            continue

        if len(description) < 80:
            continue

        return {
            "title": title,
            "authors": ", ".join(info.get("authors", [])),
            "description": description,
            "genres": ", ".join(info.get("categories", [])),
            "published_year": info.get("publishedDate", "")[:4],
        }

    return None


def inject_new_book(book_name):
    project_root = Path(__file__).resolve().parents[2]

    index_path = project_root / "data" / "embeddings" / "faiss.index"
    meta_path = project_root / "data" / "embeddings" / "metadata.pkl"

    print(f"🔍 Searching Google Books for: {book_name}")
    book = fetch_book_from_google(book_name)

    if not book:
        print("❌ Book not found in Google Books")
        return

    metadata = joblib.load(meta_path)

    # Avoid duplicates
    if metadata["title"].str.lower().eq(book["title"].lower()).any():
        print("⚠️ Book already exists in dataset")
        return

    text = (
        book["title"] + " " +
        book["description"] + " " +
        book["genres"]
    )

    emb = embed_text(text)

    index = faiss.read_index(str(index_path))
    index.add(emb)

    metadata = pd.concat([metadata, pd.DataFrame([book])], ignore_index=True)

    faiss.write_index(index, str(index_path))
    joblib.dump(metadata, meta_path)

    print(f"✅ Injected new book: {book['title']}")
    print(f"📚 Total books now: {index.ntotal}")
