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
from wikipedia_setup import wikipedia_profile_by_title

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
    if not title:
        raise ValueError("Missing 'Article Title'")

    lang = lang_hint or _resolve_lang_from_title(title)

    # Fetch full profile (summary, content, metrics) from your earlier module
    prof = wikipedia_profile_by_title(title, lang=lang)

    # Construct properties for Weaviate
    props = {
        "title": prof.get("Input_Title") or title,
        "lang": prof.get("Input_Lang") or lang,
        "qid": prof.get("Wikidata_ID"),
        "summary": prof.get("summary"),
        "content": prof.get("content"),

        "wikipedia_pageviews_90d": prof.get("wikipedia_pageviews_90d"),
        "wikipedia_backlinks": prof.get("wikipedia_backlinks"),
        "article_size_bytes": prof.get("article_size_bytes"),
        "reference_count": prof.get("reference_count"),
        "revision_count": prof.get("revision_count"),
        "unique_editors": prof.get("unique_editors"),
        "category_count": prof.get("category_count"),
        "image_count": prof.get("image_count"),
        "external_links": prof.get("external_links"),

        "wikidata_incoming_links": prof.get("wikidata_incoming_links"),
        "wikidata_outgoing_links": prof.get("wikidata_outgoing_links"),
        "language_editions": prof.get("language_editions"),
        "statement_count": prof.get("statement_count"),
        "has_wikidata_image": prof.get("has_wikidata_image"),
        "qualifier_count": prof.get("qualifier_count"),
        "entity_age_days": prof.get("entity_age_days"),
        "entity_age_years": prof.get("entity_age_years"),
    }

    # We’ll use the title+lang as a deterministic UUID5 to avoid duplicates.
    oid = uuid.uuid5(uuid.NAMESPACE_URL, f"{props['lang']}::{props['title']}")

    col = client.collections.get(collection_name)
    col.data.insert(properties=props, uuid=str(oid))
    return {"id": str(oid), "title": props["title"], "lang": props["lang"]}

if __name__ == "__main__":
    client = connect_weaviate()
    ensure_collection(client)

    example = {
        "Article Title": "कलाम-सैट: इसरो ने लॉन्च किया 1.2 किलो का उपग्रह",
        "Entity Name": "कलाम - सैट",
        "Wikidata ID": "Q29964022",
        "English Wikipedia Title": "NIL",
        "Image Name": "hindi_575.jpg",
        "wikidata_incoming_links": 0
    }
    print(upsert_wikipedia_article(client, example, lang_hint="hi"))
    client.close()
