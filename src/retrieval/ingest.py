# ingest.py

from typing import Dict, Any, Optional
import uuid
import weaviate

from weaviate_setup import connect_weaviate, ensure_collection
import langid

def _resolve_lang_from_title(title: str) -> str:
    lang, _ = langid.classify(title)
    return lang if lang in {"id", "hi", "ja", "ta", "vi"} else "en"

# Import your earlier function from your other file:
from wikipedia_setup import wikipedia_profile_by_title, wikipedia_profile_by_qid

def upsert_wikipedia_article(
    client: weaviate.WeaviateClient,
    input_record: Dict[str, Any],
    collection_name: str = "WikipediaArticles",
    lang_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    input_record example:
      {
        "Article Title": "कलाम-सैट: इसरो ने लॉन्च किया 1.2 किलो का उपग्रह",
        "Entity Name": "कलाम - सैट",
        "Wikidata ID": "Q29964022",
        "English Wikipedia Title": "NIL",
        "Image Name": "hindi_575.jpg",
        "wikidata_incoming_links": 0
      }
    """
    title = input_record.get("Article Title")
    qid = input_record.get("Wikidata ID") or input_record.get("wikidata_id")
    if not title and not qid:
        raise ValueError("Need at least 'Article Title' or 'Wikidata ID'")

    lang = lang_hint or (title and _resolve_lang_from_title(title)) or "en"

    # Fetch full profile (summary, content, metrics) from your earlier module
    try:
        prof: Dict[str, Any]
        if qid:
            prof = wikipedia_profile_by_qid(qid, lang=lang)
            if not prof.get("summary") and title:
                # If the language lacks a sitelink, fall back to title-based lookup.
                fallback = wikipedia_profile_by_title(title, lang=lang)
                for key, value in fallback.items():
                    if prof.get(key) in {None, ""} and value not in {None, ""}:
                        prof[key] = value
        else:
            prof = wikipedia_profile_by_title(title, lang=lang)
    except Exception as e:
        print(f"Warning: Failed to fetch Wikipedia profile: {e}")
        prof = {}

    # Helper function to safely get values - NEVER return empty strings for text fields
    def safe_text(key: str, default: str = "N/A") -> str:
        val = prof.get(key)
        # If value is None or empty string, use default
        return val if val and isinstance(val, str) and val.strip() else default
    
    def safe_int(key: str, default: int = 0) -> int:
        val = prof.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    
    def safe_float(key: str, default: float = 0.0) -> float:
        val = prof.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Construct properties for Weaviate
    # CRITICAL: Never use empty strings - use "N/A" or None for missing text
    props = {
        "title": safe_text("Input_Title") or title or qid or "Untitled",
        "lang": safe_text("Input_Lang") or lang,
        "qid": safe_text("Wikidata_ID") or qid or "N/A",
        "summary": safe_text("summary", "No summary available"),
        "content": safe_text("content", "No content available"),

        # Numeric fields - always use 0 instead of None
        "wikipedia_pageviews_90d": safe_int("wikipedia_pageviews_90d"),
        "wikipedia_backlinks": safe_int("wikipedia_backlinks"),
        "article_size_bytes": safe_int("article_size_bytes"),
        "reference_count": safe_int("reference_count"),
        "revision_count": safe_int("revision_count"),
        "unique_editors": safe_int("unique_editors"),
        "category_count": safe_int("category_count"),
        "image_count": safe_int("image_count"),
        "external_links": safe_int("external_links"),

        "wikidata_incoming_links": safe_int("wikidata_incoming_links"),
        "wikidata_outgoing_links": safe_int("wikidata_outgoing_links"),
        "language_editions": safe_int("language_editions"),
        "statement_count": safe_int("statement_count"),
        "has_wikidata_image": bool(prof.get("has_wikidata_image", False)),
        "qualifier_count": safe_int("qualifier_count"),
        "entity_age_days": safe_int("entity_age_days"),
        "entity_age_years": safe_float("entity_age_years"),
    }

    # Use deterministic UUID5 to avoid duplicates (proper upsert behavior)
    namespace_key = f"{props['lang']}::{props['title']}"
    oid = uuid.uuid5(uuid.NAMESPACE_URL, namespace_key)

    col = client.collections.get(collection_name)
    
    # Try to insert - if UUID exists, Weaviate will replace it
    try:
        col.data.insert(properties=props, uuid=str(oid))
        print(f"✅ Inserted/Updated: {props['title']}")
    except Exception as e:
        print(f"❌ Failed to insert {props['title']}")
        print(f"   Error: {e}")
        print(f"   Properties: {props}")
        raise
    
    return {"id": str(oid), "title": props["title"], "lang": props["lang"]}

if __name__ == "__main__":
    client = connect_weaviate()
    
    # Recreate the collection with proper config
    if client.collections.exists('New'):
        print("Deleting old 'New' collection...")
        client.collections.delete('New')
    
    ensure_collection(client, 'New_2')

    example = {
        "Article Title": "कलाम-सैट: इसरो ने लॉन्च किया 1.2 किलो का उपग्रह",
        "Entity Name": "कलाम - सैट",
        "Wikidata ID": "Q29964022",
        "English Wikipedia Title": "NIL",
        "Image Name": "hindi_575.jpg",
        "wikidata_incoming_links": 0
    }
    result = upsert_wikipedia_article(client, example, lang_hint="hi", collection_name='New_2')
    print(f"Result: {result}")
    client.close()