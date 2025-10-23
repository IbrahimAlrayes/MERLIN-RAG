# weaviate_setup.py

import os
import uuid
from typing import Dict, Any, Optional

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

# --- Connect (env: WEAVIATE_ENDPOINT, OPENAI_API_KEY) ---
def connect_weaviate() -> weaviate.WeaviateClient:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_ENDPOINT"],
        auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY", "")),
        headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY", "")},  # for text2vec-openai
    )
    return client

# --- Create collection with server-side vectorizer (OpenAI) ---
def ensure_collection(client: weaviate.WeaviateClient, name: str = "WikipediaArticles") -> None:
    if name in [c.name for c in client.collections.list_all()]:
        return

    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.text2vec_openai(model="text-embedding-3-large"),
        # You can also use Azure OpenAI with text2vec-azure-openai if preferred.
        # Or a self-hosted multilingual model via text2vec-huggingface.
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="lang", data_type=DataType.TEXT),
            Property(name="qid", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),

            # Wikipedia metrics
            Property(name="wikipedia_pageviews_90d", data_type=DataType.INT),
            Property(name="wikipedia_backlinks", data_type=DataType.INT),
            Property(name="article_size_bytes", data_type=DataType.INT),
            Property(name="reference_count", data_type=DataType.INT),
            Property(name="revision_count", data_type=DataType.INT),
            Property(name="unique_editors", data_type=DataType.INT),
            Property(name="category_count", data_type=DataType.INT),
            Property(name="image_count", data_type=DataType.INT),
            Property(name="external_links", data_type=DataType.INT),

            # Wikidata metrics
            Property(name="wikidata_incoming_links", data_type=DataType.INT),
            Property(name="wikidata_outgoing_links", data_type=DataType.INT),
            Property(name="language_editions", data_type=DataType.INT),
            Property(name="statement_count", data_type=DataType.INT),
            Property(name="has_wikidata_image", data_type=DataType.BOOL),
            Property(name="qualifier_count", data_type=DataType.INT),
            Property(name="entity_age_days", data_type=DataType.INT),
            Property(name="entity_age_years", data_type=DataType.NUMBER),
        ],
        # Vectorize only the summary by default (best for fast search). You can switch to content if needed.
        vectorizer_config_settings=Configure.NamedVectors.text2vec_openai(  # optional named vectors
            name="summary_vec",
            source_properties=["summary"],
            model="text-embedding-3-large",
        ),
    )
