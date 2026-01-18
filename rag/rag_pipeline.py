from rag.retriever import retrieve_books
from rag.generator import generate_response
from rag.auto_ingest import auto_inject_if_needed


def run_rag(query):
    # 🔄 Always try auto-inject for missing books
    auto_inject_if_needed(query)

    retrieved,_ = retrieve_books(query, top_k=5)
    return generate_response(query, retrieved)


if __name__ == "__main__":
    query = input("\n🔍 Enter your book query: ")
    print("\n🧠 RAG RESPONSE\n")
    print(run_rag(query))
