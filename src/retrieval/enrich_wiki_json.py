import json
import time
import os
import argparse
from typing import List, Dict
from tqdm import tqdm

from wikipedia_setup import wikipedia_profile_by_qid

LANG_MAP = {
    "vietnamese": "vi",
    "tamil": "ta",
    "japanese": "ja",
    "indonesian": "id",
    "hindi": "hi",
}


def enrich_file(lang_key: str, input_dir: str, output_dir: str):
    lang = LANG_MAP[lang_key]
    input_file = os.path.join(input_dir, f"{lang_key}.json")
    output_file = os.path.join(output_dir, f"{lang_key}_enriched.json")

    # Load dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    # Resume if output exists
    enriched = []
    if os.path.isfile(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            enriched = json.load(f)

    start_index = len(enriched)
    print(f"\n🌍 Processing {lang_key} ({lang})")
    print(f"⏩ Resume at: {start_index}/{len(data)} rows")

    for i in tqdm(range(start_index, len(data)), desc=f"{lang_key}", dynamic_ncols=True):
        row = data[i]
        qid = row.get("Wikidata ID")

        if qid:
            try:
                profile = wikipedia_profile_by_qid(qid, lang)
                row.update(profile)
            except Exception as e:
                tqdm.write(f"[WARN] QID={qid}, row={i}, err={e}")

        enriched.append(row)

        # Save progress incrementally
        if i % 5 == 0:
            with open(output_file, "w", encoding="utf-8") as fw:
                json.dump(enriched, fw, ensure_ascii=False, indent=2)

        time.sleep(0.3)  # safety for API rate-limit

    # Final save
    with open(output_file, "w", encoding="utf-8") as fw:
        json.dump(enriched, fw, ensure_ascii=False, indent=2)

    print(f"✅ Done: saved → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Add Wikipedia/Wikidata metadata to JSON files.")
    parser.add_argument("-i", "--input", required=True, help="Input directory path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for lang_key in tqdm(LANG_MAP.keys(), desc="🌐 Languages", dynamic_ncols=True):
        enrich_file(lang_key, args.input, args.output)


if __name__ == "__main__":
    main()
