import pandas as pd
import re
from pathlib import Path


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_csv_safely(path):
    """
    Tries common delimiters automatically
    """
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding="utf-8"
            )
            if df.shape[1] > 1:
                print(f"✅ Detected delimiter: '{sep}'")
                return df
        except Exception:
            pass

    raise ValueError("❌ Could not detect CSV delimiter")


def preprocess_books():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "books_raw.csv"
    output_path = project_root / "data" / "processed" / "books_cleaned.csv"

    print(f"📂 Reading data from: {input_path}")

    df = load_csv_safely(input_path)
    print(f"📘 Original dataset shape: {df.shape}")
    print(f"📑 Columns found: {list(df.columns)}")

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Column aliases (dataset-agnostic)
    COLUMN_MAP = {
        "title": ["title", "book_title", "name"],
        "authors": ["authors", "author", "writer"],
        "description": ["description", "summary", "book_description"],
        "popular_shelves": ["popular_shelves", "shelves", "genres"],
        "average_rating": ["average_rating", "rating"],
        "ratings_count": ["ratings_count", "num_ratings"]
    }

    def find_column(possible_names):
        for col in possible_names:
            if col in df.columns:
                return col
        return None

    # Resolve actual column names
    resolved = {}
    for key, aliases in COLUMN_MAP.items():
        resolved[key] = find_column(aliases)

    # Validate mandatory fields
    if not resolved["title"] or not resolved["description"]:
        raise ValueError(
            "❌ Dataset must contain title and description columns.\n"
            f"Found columns: {df.columns.tolist()}"
        )

    # Rename columns to standard names
    df = df.rename(columns={v: k for k, v in resolved.items() if v})

    # Keep only resolved columns
    df = df[[k for k, v in resolved.items() if v]]

    # Drop invalid rows
    df = df.dropna(subset=["title", "description"])

    # Clean text
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)

    # Fill missing values
    for col in df.columns:
        df[col] = df[col].fillna("")

    # Remove duplicates
    df = df.drop_duplicates(subset=["clean_title"])

    print(f"✅ Cleaned dataset shape: {df.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"💾 Processed data saved to: {output_path}")


if __name__ == "__main__":
    preprocess_books()
