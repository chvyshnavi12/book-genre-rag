from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(matrix, query_vector):
    """
    Compute cosine similarity between query and all books
    """
    similarities = cosine_similarity(query_vector, matrix)
    return similarities.flatten()
