# weaviate_setup.py

import os
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, Optional

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

load_dotenv()
# --- Connect (env: WEAVIATE_ENDPOINT, OPENAI_API_KEY) ---
def connect_weaviate() -> weaviate.WeaviateClient:
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["X-OpenAI-Api-Key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("HUGGINGFACE_API_KEY"):
        headers["X-HuggingFace-Api-Key"] = os.environ["HUGGINGFACE_API_KEY"]

    http_host = os.environ.get("WEAVIATE_ENDPOINT", "127.0.0.1")
    http_port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=http_host,
        grpc_port=grpc_port,
        grpc_secure=False,
        headers=headers or None,
        # auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY", "")) if os.environ.get("WEAVIATE_API_KEY") else None,
    )
    return client

# --- Create collection with server-side vectorizer  ---
def ensure_collection(client: weaviate.WeaviateClient, name: str = "WikipediaArticles") -> None:
    existing = client.collections.list_all()

    # Extract collection names and normalize to lowercase
    if isinstance(existing, dict):
        collection_names = [n.lower() for n in existing.keys()]
    else:
        collection_names = [c.name.lower() for c in existing] if existing else []

    if name.lower() in collection_names:
        print(f"✅ Collection '{name}' already exists — skipping creation.")
        return



    print(f"🚀 Creating new collection '{name}' ...")

    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.text2vec_huggingface(model="Qwen/Qwen3-Embedding-0.6B"),
        properties=[
            Property(name="title", data_type=DataType.TEXT, vectorize=False),
            Property(name="lang", data_type=DataType.TEXT, vectorize=False),
            Property(name="qid", data_type=DataType.TEXT, vectorize=False),
            Property(name="summary", data_type=DataType.TEXT, vectorize=True),  # only this embedded
            Property(name="content", data_type=DataType.TEXT, vectorize=False),

            # numeric metadata (non-vectorized)
            Property(name="wikipedia_pageviews_90d", data_type=DataType.INT, vectorize=False),
            Property(name="wikipedia_backlinks", data_type=DataType.INT, vectorize=False),
            Property(name="article_size_bytes", data_type=DataType.INT, vectorize=False),
            Property(name="reference_count", data_type=DataType.INT, vectorize=False),
            Property(name="revision_count", data_type=DataType.INT, vectorize=False),
            Property(name="unique_editors", data_type=DataType.INT, vectorize=False),
            Property(name="category_count", data_type=DataType.INT, vectorize=False),
            Property(name="image_count", data_type=DataType.INT, vectorize=False),
            Property(name="external_links", data_type=DataType.INT, vectorize=False),

            # Wikidata metrics
            Property(name="wikidata_incoming_links", data_type=DataType.INT, vectorize=False),
            Property(name="wikidata_outgoing_links", data_type=DataType.INT, vectorize=False),
            Property(name="language_editions", data_type=DataType.INT, vectorize=False),
            Property(name="statement_count", data_type=DataType.INT, vectorize=False),
            Property(name="has_wikidata_image", data_type=DataType.BOOL, vectorize=False),
            Property(name="qualifier_count", data_type=DataType.INT, vectorize=False),
            Property(name="entity_age_days", data_type=DataType.INT, vectorize=False),
            Property(name="entity_age_years", data_type=DataType.NUMBER, vectorize=False),
        ],
    )
