import os, json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
import uuid



def load_json_files(data_dir: str):
    """Load multilingual JSON files and extract summaries, metadata, and IDs."""
    all_docs, all_metas, all_ids = [], [], []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        lang_file = filename.split(".")[0]  # e.g., 'vietnamese'
        path = os.path.join(data_dir, filename)

        with open(path, "r") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping {filename}: invalid JSON ({e})")
                continue

        for rec in records:
            summary = rec.get("summary")
            if not summary or summary.strip() == "":
                continue
            
            unique_id = str(uuid.uuid4())

            all_docs.append(summary)
            all_metas.append({
                "language": lang_file,
                "entity_name": rec.get("Entity Name"),
                "wikidata_id": rec.get("Wikidata_ID"),
                "article_title": rec.get("Article Title"),
                "pageviews": rec.get("wikipedia_pageviews_90d", 0)
            })
            all_ids.append(unique_id)

    return all_docs, all_metas, all_ids


def create_embedder(model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    """Load a SentenceTransformer embedding model."""
    print(f"🔹 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            # embed the documents somehow
            return model.encode(input, normalize_embeddings=True).tolist()

    return MyEmbeddingFunction()


def create_chroma_client(persist_dir: str = "./chroma_multilang_db"):
    """Initialize Chroma client with persistence."""
    # client = chromadb.Client(Settings(
    #     chroma_db_impl="duckdb+parquet",
    #     persist_directory=persist_dir
    # ))
    client = chromadb.PersistentClient(
        path=persist_dir,
        )

    return client


def create_or_get_collection(client, name: str, embedding_fn):
    """Create or get a Chroma collection with embedding function."""
    collection = client.create_collection(
        name=name,
        embedding_function=embedding_fn
    )
    return collection


def populate_chroma(collection, docs, metas, ids, batch_size: int = 100):
    """Add documents, metadata, and IDs to Chroma in batches."""
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing documents"):
        batch_docs = docs[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"[ERROR] Batch {i//batch_size}: {e}")


def main():
    data_dir = "./test_data_100_1103"
    persist_dir = "chroma_multilang_db_1103"
    collection_name = "multilingual_entities"

    docs, metas, ids = load_json_files(data_dir)
    print(f"Loaded {len(docs)} summaries from {data_dir}")

    embed_fn = create_embedder("Qwen/Qwen3-Embedding-0.6B")

    client = create_chroma_client(persist_dir)

    collection = create_or_get_collection(client, collection_name, embed_fn)

    populate_chroma(collection, docs, metas, ids)

    print(f"🎉 Done! Total documents in Chroma: {collection.count()}")


if __name__ == "__main__":
    main()
