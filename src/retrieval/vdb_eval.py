import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

# ---------- Config ----------
DATA_DIR = "./test_data_100_1103"
PERSIST_DIR = "chroma_multilang_db_1103"
COLLECTION_NAME = "multilingual_entities"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 1
# ----------------------------


def load_json_files(data_dir: str):
    """Load multilingual JSON files and extract summaries, metadata, and IDs."""
    lang_data = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
        lang = filename.split(".")[0]
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
    return lang_data


def create_chroma_client(persist_dir: str = PERSIST_DIR):
    return chromadb.PersistentClient(
        path=persist_dir
    )


def create_embedder(model_name: str = MODEL_NAME):
    model = SentenceTransformer(model_name)
    # def embed(texts):
    #     return model.encode(texts, normalize_embeddings=True).tolist()
    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            # embed the documents somehow
            return model.encode(input, normalize_embeddings=True).tolist()
    return MyEmbeddingFunction()


def evaluate_retrieval(client, embed_fn, lang_data, collection_name: str, top_k: int = 1):
    """Evaluate top-K retrieval accuracy by comparing Wikidata IDs."""
    collection = client.get_collection(collection_name, embedding_function=embed_fn)

    results = {}
    overall_correct = 0
    overall_total = 0

    for lang, records in lang_data.items():
        correct = 0
        total = 0

        print(f"\n🌍 Evaluating {lang} ({len(records)} samples)")
        for record in tqdm(records, desc=f"{lang}", unit="docs"):
            query_text = record.get("Article Title")
            if not query_text:
                continue

            try:
                retrieved = collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    where={"language": lang}
                )

                # Compare top-1 match
                # retrieved_id = retrieved["ids"][0][0] if retrieved["ids"][0] else None
                if record.get("summary") in retrieved['documents'][0]:
                    correct += 1
            except Exception as e:
                print(f"[WARN] Error querying {lang}: {e}")

            total += 1

        acc = correct / total if total > 0 else 0.0
        results[lang] = {"correct": correct, "total": total, "accuracy": acc}

        overall_correct += correct
        overall_total += total

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    results["overall"] = {"correct": overall_correct, "total": overall_total, "accuracy": overall_acc}

    return results


def main():
    # Load true data
    lang_data = load_json_files(DATA_DIR)
    print(f"📂 Loaded {len(lang_data)} language files.")

    # Prepare Chroma + embedding model
    client = create_chroma_client()
    embed_fn = create_embedder()

    # Run evaluation
    stats = evaluate_retrieval(client, embed_fn, lang_data, COLLECTION_NAME, TOP_K)

    # Print results
    print("\n==================== 📊 Retrieval Accuracy ====================")
    for lang, res in stats.items():
        print(f"{lang.capitalize():<12} | {res['correct']:>4}/{res['total']:<4} | {res['accuracy']*100:5.2f}%")
    print("===============================================================")

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("💾 Saved results to eval_results/retrieval_eval.json")


if __name__ == "__main__":
    main()
