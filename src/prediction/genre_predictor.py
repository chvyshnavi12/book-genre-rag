from pathlib import Path
import joblib
import re
import pandas as pd
import numpy as np


# -----------------------------
# Text cleaning (same as training)
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Load trained artifacts
# -----------------------------
def load_genre_model():
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models" / "genre_classifier"

    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    label_binarizer = joblib.load(model_dir / "label_binarizer.pkl")

    return model, vectorizer, label_binarizer


# -----------------------------
# Predict genres
# -----------------------------
def predict_genre(description: str, threshold: float = 0.3):
    """
    description: book description text
    threshold: minimum probability to include genre
    """
    model, vectorizer, mlb = load_genre_model()

    description = clean_text(description)

    # Vectorize
    X_vec = vectorizer.transform([description])

    # Predict probabilities
    probs = model.predict_proba(X_vec)[0]

    results = {}
    for genre, prob in zip(mlb.classes_, probs):
        results[genre] = round(prob * 100, 2)

    # If nothing crosses threshold, return top genre
    if not results:
        top_idx = np.argmax(probs)
        results[mlb.classes_[top_idx]] = round(probs[top_idx] * 100, 2)

    return results


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    sample_description = """
    A young wizard discovers his magical powers and attends a school of magic,
    where he faces dark forces and learns about friendship and courage.
    """

    predictions = predict_genre(sample_description)

    print("\n📚 Predicted Genres:")
    for genre, confidence in predictions.items():
        print(f"• {genre}: {confidence}%")
