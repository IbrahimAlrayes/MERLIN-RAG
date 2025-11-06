import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings


# ---- Setup once ----
PERSIST_DIR = "chroma_multilang_db_1103"
COLLECTION_NAME = "multilingual_entities"

# Load embedding model (must match what was used for indexing)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return model.encode(input, normalize_embeddings=True).tolist()
# def embed(texts):
#     return model.encode(texts, normalize_embeddings=True).tolist()


# ---- Core function ----
def query_entities(query_text: str, language: str | None = None, top_k: int = 5):
    """Query Chroma for similar summaries, optionally filtered by language."""
    # connect to existing DB
    client = chromadb.PersistentClient(
        path=PERSIST_DIR
    )

    # load existing collection
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=MyEmbeddingFunction()
    )

    # prepare filter
    where_filter = {"language": language} if language else None

    # run query
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where_filter
    )

    # format readable output
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "entity_name": meta.get("entity_name"),
            "language": meta.get("language"),
            "wikidata_id": meta.get("wikidata_id"),
            "article_title": meta.get("article_title"),
            "summary": doc,
            "distance": dist
        })

    return output


# ---- Example usage ----
if __name__ == "__main__":
    results = query_entities(
        query_text="Vietnamese economist and academic",
        language="vietnamese",
        top_k=3
    )

    for r in results:
        print(f"🔹 Entity: {r['entity_name']}")
        print(f"   Lang: {r['language']} | Dist: {r['distance']:.4f}")
        print(f"   Title: {r['article_title']}")
        print(f"   Summary: {r['summary'][:200]}...\n")
