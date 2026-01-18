from pathlib import Path
import joblib
import re
import pandas as pd

from src.prediction.feature_matcher import match_features


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_genre_model():
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models" / "genre_classifier"

    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    mlb = joblib.load(model_dir / "label_binarizer.pkl")

    return model, vectorizer, mlb


def predict_book(description: str, top_k: int = 3):
    model, vectorizer, mlb = load_genre_model()

    description_clean = clean_text(description)
    X_vec = vectorizer.transform([description_clean])
    probs = model.predict_proba(X_vec)[0]

    top_indices = probs.argsort()[-top_k:][::-1]
    genre_predictions = {
        mlb.classes_[i]: round(probs[i] * 100, 2)
        for i in top_indices
    }

    feature_matches = match_features(description)

    return {
        "predicted_genres": genre_predictions,
        "feature_matches": feature_matches
    }


def predict_genre_with_scores(book_text: str) -> dict:
    result = predict_book(book_text)
    return result["predicted_genres"]
