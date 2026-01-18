from pathlib import Path
import pandas as pd
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from similarity import compute_similarity


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Load dataset
# -----------------------------
def load_books():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "books_with_genres.csv"

    df = pd.read_csv(data_path)

    if "description" not in df.columns:
        raise ValueError("Dataset must contain description column")

    df["clean_description"] = df["description"].apply(clean_text)

    return df


# -----------------------------
# Build TF-IDF matrix
# -----------------------------
def build_tfidf_matrix(descriptions):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer


# -----------------------------
# Recommendation function
# -----------------------------
def recommend_books(book_description, top_k=5):
    df = load_books()

    tfidf_matrix, vectorizer = build_tfidf_matrix(df["clean_description"])

    query_clean = clean_text(book_description)
    query_vector = vectorizer.transform([query_clean])

    similarities = compute_similarity(tfidf_matrix, query_vector)

    df["similarity_score"] = similarities

    recommendations = (
        df.sort_values(by="similarity_score", ascending=False)
        .head(top_k)
    )

    return recommendations[["title", "authors", "genres", "similarity_score"]]


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    sample_description = """
     Harry Potter wizard school magic friendship dark lord
    """

    recs = recommend_books(sample_description, top_k=5)

    print("\n📚 Recommended Books:\n")
    for _, row in recs.iterrows():
        print(f"• {row['title']} by {row['authors']} (score: {row['similarity_score']:.2f})")
