import re
import pandas as pd


# -----------------------------
# Feature keyword definitions
# -----------------------------
THEMES = {
    "Magic": ["magic", "wizard", "spell"],
    "Love & Relationships": ["love", "relationship", "romance"],
    "Friendship": ["friendship", "friends"],
    "Social Issues": ["society", "justice", "racism", "inequality"],
    "War & Conflict": ["war", "battle", "conflict"],
    "Crime": ["crime", "murder", "investigation"]
}

TONE = {
    "Light": ["fun", "humor", "comedy"],
    "Serious": ["serious", "dark", "intense"],
    "Emotional": ["emotional", "heartbreaking"],
    "Suspenseful": ["suspense", "thriller"]
}

WRITING_STYLE = {
    "Narrative Prose": ["story", "narrative"],
    "Descriptive": ["descriptive", "detailed"],
    "Fast-Paced": ["fast", "action", "quick"]
}

PLOT_ELEMENTS = {
    "Coming-of-Age": ["growing up", "coming of age"],
    "Hero’s Journey": ["hero", "journey", "quest"],
    "Human Relationships": ["family", "relationships"]
}

TARGET_AUDIENCE = {
    "Children": ["children", "kids"],
    "Young Adult": ["young adult", "teen"],
    "Adult": ["adult", "mature"]
}


# -----------------------------
# Utility
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def calculate_match(text, keyword_dict):
    """
    Calculates percentage match for a feature category
    """
    text = clean_text(text)
    scores = {}

    for feature, keywords in keyword_dict.items():
        count = sum(1 for k in keywords if k in text)
        if count > 0:
            scores[feature] = min(100, count * 30)  # scale score
    return scores


# -----------------------------
# Main Feature Matcher
# -----------------------------
def match_features(description: str):
    return {
        "Themes": calculate_match(description, THEMES),
        "Tone": calculate_match(description, TONE),
        "Writing Style": calculate_match(description, WRITING_STYLE),
        "Plot Elements": calculate_match(description, PLOT_ELEMENTS),
        "Target Audience": calculate_match(description, TARGET_AUDIENCE),
    }


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    sample_description = """
    A serious and emotional story about human relationships,
    love, and social injustice in a changing society.
    """

    features = match_features(sample_description)

    print("\n📊 Feature Matching Results:\n")
    for category, values in features.items():
        if values:
            print(f"{category}:")
            for k, v in values.items():
                print(f"  • {k}: {v}%")
