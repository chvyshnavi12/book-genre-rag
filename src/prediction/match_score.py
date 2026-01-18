def compute_preference_match(predictions: dict, preferences: list) -> float:
    """
    predictions: {genre: percentage}
    preferences: list of preferred genres
    """
    score = 0.0

    for genre, value in predictions.items():
        if genre in preferences:
            score += value

    return round(score, 2)
