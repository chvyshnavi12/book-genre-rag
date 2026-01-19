def compute_preference_match(predictions: dict, preferences: list) -> int:
    if not predictions or not preferences:
        return 0

    scores = [predictions.get(p, 0) for p in preferences]

    if max(scores) == 0:
        return 0

    # Normalize relative to model confidence range
    normalized_scores = [
        (s / max(scores)) * 100 for s in scores
    ]

    final_score = sum(normalized_scores) / len(preferences)

    return int(round(final_score))
