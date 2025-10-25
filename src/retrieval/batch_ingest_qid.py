import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

from ingest import upsert_wikipedia_article
from weaviate_setup import connect_weaviate, ensure_collection

LANG_NAME_TO_CODE = {
    "hindi": "hi",
    "indonesian": "id",
    "japanese": "ja",
    "tamil": "ta",
    "vietnamese": "vi",
}


def _iter_records(data_dir: Path) -> Iterable[Tuple[str, Dict[str, Any], Path]]:
    for path in sorted(data_dir.glob("*.json")):
        lang_key = path.stem.lower()
        lang = LANG_NAME_TO_CODE.get(lang_key)
        if not lang:
            print(f"[WARN] No language mapping for {path.name}; skipping.")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        if not isinstance(payload, list):
            print(f"[WARN] Expected list in {path}; skipping.")
            continue
        for record in payload:
            if isinstance(record, dict):
                yield lang, record, path
            else:
                print(f"[WARN] Non-dict record in {path}; skipping entry.")


def ingest_directory(
    data_dir: Path,
    collection_name: str = "WikipediaArticles",
) -> None:
    client = connect_weaviate()
    ensure_collection(client, collection_name)
    try:
        for lang, record, source in _iter_records(data_dir):
            try:
                result = upsert_wikipedia_article(
                    client,
                    record,
                    collection_name=collection_name,
                    lang_hint=lang,
                )
                print(f"[OK] {result['lang']}::{result['title']} (from {source.name})")
            except Exception as exc:
                qid = record.get("Wikidata ID") or record.get("wikidata_id")
                print(f"[ERR] {qid} from {source.name}: {exc}")
    finally:
        client.close()


def _default_data_dir() -> Path:
    base = Path(__file__).resolve().parents[2]
    return base / "cultural-mllm" / "test_data_100"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest Wikipedia summaries by QID.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Directory containing JSON files of cultural entities.",
    )
    parser.add_argument(
        "--collection",
        default="WikipediaArticles",
        help="Weaviate collection name.",
    )
    args = parser.parse_args()
    ingest_directory(args.data_dir, collection_name=args.collection)
