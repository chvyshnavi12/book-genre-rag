from pathlib import Path
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


# -----------------------------
# Text cleaning (fallback)
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_genre_model():
    # -----------------------------
    # Resolve paths
    # -----------------------------
    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "processed" / "books_with_genres.csv"
    model_dir = project_root / "models" / "genre_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading dataset from: {data_path}")

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(data_path)

    if "genres" not in df.columns:
        raise ValueError("❌ 'genres' column not found in dataset")

    # ✅ Ensure clean_description exists
    if "clean_description" in df.columns:
        X = df["clean_description"]
        print("✅ Using existing clean_description column")
    elif "description" in df.columns:
        print("⚠️ clean_description not found, cleaning description on the fly")
        X = df["description"].apply(clean_text)
    else:
        raise ValueError("❌ No description column found for training")

    # Convert genres from string → list
    y = df["genres"].apply(eval)

    print(f"📊 Total samples: {len(df)}")

    # -----------------------------
    # Encode labels
    # -----------------------------
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y)

    print(f"🎯 Genres learned: {list(mlb.classes_)}")

    # -----------------------------
    # Train / Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # Vectorization
    # -----------------------------
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -----------------------------
    # Model
    # -----------------------------
    model = OneVsRestClassifier(
        LogisticRegression(max_iter=500)
    )

    print("🚀 Training model...")
    model.fit(X_train_vec, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test_vec)

    print("\n📈 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    # -----------------------------
    # Save artifacts
    # -----------------------------
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    joblib.dump(mlb, model_dir / "label_binarizer.pkl")

    print("\n✅ Model training complete")
    print(f"💾 Saved to: {model_dir}")


if __name__ == "__main__":
    train_genre_model()
