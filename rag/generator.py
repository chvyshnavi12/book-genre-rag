def generate_response(query, retrieved_books):
    response = []
    response.append("📚 Recommended Books Based on Your Query:\n")

    for _, row in retrieved_books.iterrows():
        explanation = (
            f"• **{row['title']}** by {row['authors']}\n"
            f"  Reason: This book matches your interest through similar themes "
            f"and concepts found in its description and genre "
            f"({row.get('genres', 'N/A')}).\n"
        )
        response.append(explanation)

    response.append(
        "\nThese recommendations were generated using semantic similarity "
        "search and retrieval-augmented generation (RAG)."
    )

    return "\n".join(response)
