from pathlib import Path
import pandas as pd

# -----------------------------
# Genre keyword mapping
# -----------------------------
GENRE_KEYWORDS = {
    "Fantasy": ["fantasy", "magic", "wizard"],
    "Romance": ["romance", "love", "relationship"],
    "Sci-Fi": ["science fiction", "sci fi", "space", "future"],
    "Thriller": ["thriller", "suspense", "crime"],
    "Mystery": ["mystery", "detective"],
    "Horror": ["horror", "ghost", "terror"],
    "Classic": ["classic", "literature"],
    "Young Adult": ["young adult", "ya"],
    "Adventure": ["adventure", "journey"],
    "Drama": ["drama"]
}


def extract_genres_from_categories(categories: str):
    """
    Extract genres from category text
    """
    categories = str(categories).lower()
    genres = []

    for genre, keywords in GENRE_KEYWORDS.items():
        if any(word in categories for word in keywords):
            genres.append(genre)

    return genres


def add_genre_labels():
    # Resolve project root safely
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "books_cleaned.csv"
    output_path = project_root / "data" / "processed" / "books_with_genres.csv"

    print(f"📂 Reading from: {input_path}")

    df = pd.read_csv(input_path)
    print(f"📑 Columns found: {list(df.columns)}")

    # ✅ Use categories column
    if "categories" not in df.columns:
        raise ValueError("❌ 'categories' column not found in dataset")

    # Extract genres
    df["genres"] = df["categories"].fillna("").apply(extract_genres_from_categories)

    # Remove rows without genres
    df = df[df["genres"].map(len) > 0]

    # Save result
    df.to_csv(output_path, index=False)
    print(f"✅ Genre-labeled data saved to: {output_path}")
    print(f"📊 Final dataset shape: {df.shape}")


if __name__ == "__main__":
    add_genre_labels()
