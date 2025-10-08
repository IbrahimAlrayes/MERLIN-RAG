import json, time, argparse, csv
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional
from tqdm import tqdm

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKI_API = "https://en.wikipedia.org/w/api.php"

REQUEST_DELAY = 0.1
MAX_RETRIES = 3

def call_api_with_retry(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, params=params, headers={"User-Agent":"LowFreqEntities/1.0"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[WARN] API failed after retries: {e}")
                return None
            time.sleep(2 ** attempt)

def count_wikidata_incoming_links(qid: str, threshold: int = 2000) -> int:
    """Count items that link to QID on Wikidata (like Special:WhatLinksHere)."""
    params = {
        "action": "query",
        "format": "json",
        "titles": qid,          # QID is used as the 'title' on wikidata.org
        "prop": "linkshere",
        "lhlimit": "max",
    }
    total = 0
    while True:
        data = call_api_with_retry(WIKIDATA_API, params)
        if not data:
            break
        for p in data.get("query",{}).get("pages",{}).values():
            links = p.get("linkshere", [])
            total += len(links)
            if total >= threshold:
                return total
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return total

def get_enwiki_title_from_qid(qid: str) -> Optional[str]:
    """Fetch English Wikipedia sitelink title for the QID."""
    params = {"action":"wbgetentities","ids":qid,"props":"sitelinks","format":"json"}
    data = call_api_with_retry(WIKIDATA_API, params)
    if not data:
        return None
    ent = data.get("entities", {}).get(qid, {})
    sl = ent.get("sitelinks", {}).get("enwiki")
    return sl["title"] if sl else None

def count_enwiki_backlinks(title: str, threshold: int = 3000) -> int:
    """Count backlinks on English Wikipedia."""
    if not title:
        return 0
    params = {"action":"query","list":"backlinks","bltitle":title,"bllimit":"max","format":"json"}
    total = 0
    while True:
        data = call_api_with_retry(WIKI_API, params)
        if not data:
            break
        links = data.get("query", {}).get("backlinks", [])
        total += len(links)
        if total >= threshold:
            return total
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return total

def enrich_records(records: List[Dict[str, Any]], compute_backlinks: bool=False) -> List[Dict[str, Any]]:
    """Add wikidata_incoming_links (+ optional enwiki_backlinks) to each record."""
    out = []
    for rec in tqdm(records, desc="Enriching", unit="item"):
        qid = rec.get("Wikidata ID")
        enriched = rec.copy()
        incoming = count_wikidata_incoming_links(qid) if qid else 0
        enriched["wikidata_incoming_links"] = incoming

        if compute_backlinks:
            title = get_enwiki_title_from_qid(qid) if qid else None
            enriched["English Wikipedia Title (resolved)"] = title
            enriched["enwiki_backlinks"] = count_enwiki_backlinks(title) if title else 0

        out.append(enriched)
    return out


def process_one_file(inp: Path, outdir: Path, bottom_k: int, metric: str, compute_backlinks: bool):
    outdir.mkdir(parents=True, exist_ok=True)

    records = json.loads(inp.read_text(encoding="utf-8"))
    assert isinstance(records, list), f"Input JSON must be a list of objects: {inp}"

    enriched = enrich_records(records, compute_backlinks=compute_backlinks)

    # write "<stem>_enriched.json" (array)
    json_path = outdir / f"{inp.stem}_enriched.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # rank by metric ascending
    def keyfn(r): return r.get(metric, float("inf"))
    ranked = sorted(enriched, key=keyfn)

    # write ranked CSV
    csv_path = outdir / f"{inp.stem}_ranked_by_{metric}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(ranked)

    # write bottom-K json
    k = min(bottom_k, len(ranked))
    lowest_json_path = outdir / f"{inp.stem}_lowest_{k}_by_{metric}.json"
    with lowest_json_path.open("w", encoding="utf-8") as f:
        json.dump(ranked[:k], f, ensure_ascii=False, indent=2)

    # console summary
    print(f"\nProcessed: {inp.name}")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    print(f"  -> {lowest_json_path}")
    print(f"Lowest {k} by {metric}:")
    for i, r in enumerate(ranked[:k], 1):
        print(f"{i:>3}. {r.get('Entity Name')} | QID={r.get('Wikidata ID')} | {metric}={r.get(metric)}")

def main():
    ap = argparse.ArgumentParser(description="Find lowest-frequency entities for one file or a folder of JSON files.")
    ap.add_argument("--input_path", required=True, help="Path to a JSON file (list of dicts) or a folder of JSON files.")
    ap.add_argument("--output_dir", required=True, help="Where to write outputs.")
    ap.add_argument("--bottom_k", type=int, default=100, help="How many lowest-frequency rows to export per file.")
    ap.add_argument("--metric", choices=["wikidata_incoming_links", "enwiki_backlinks"],
                    default="wikidata_incoming_links", help="Ranking metric.")
    ap.add_argument("--compute_backlinks", action="store_true",
                    help="Also compute English Wikipedia backlinks (slower).")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    outdir = Path(args.output_dir)

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {input_path}")
        process_one_file(input_path, outdir, args.bottom_k, args.metric, args.compute_backlinks)

    elif input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No .json files found in {input_path}")
        print(f"Found {len(files)} JSON files in {input_path}")
        for p in files:
            process_one_file(p, outdir, args.bottom_k, args.metric, args.compute_backlinks)

    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

if __name__ == "__main__":
    main()
    
# def main():
#     ap = argparse.ArgumentParser(description="Find lowest-frequency entities based on link counts.")
#     ap.add_argument("--input", required=True, help="Path to input JSON file (list of dicts).")
#     ap.add_argument("--output_dir", required=True, help="Where to write enriched outputs.")
#     ap.add_argument("--bottom_k", type=int, default=50, help="How many lowest-frequency to show.")
#     ap.add_argument("--metric", choices=["wikidata_incoming_links","enwiki_backlinks"], default="wikidata_incoming_links",
#                     help="Which metric to rank by (default: wikidata_incoming_links).")
#     ap.add_argument("--compute_backlinks", action="store_true", help="Also compute English Wikipedia backlinks.")
#     args = ap.parse_args()

#     inp = Path(args.input)
#     outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

#     records = json.loads(inp.read_text(encoding="utf-8"))
#     assert isinstance(records, list), "Input JSON must be a list of objects."

#     enriched = enrich_records(records, compute_backlinks=args.compute_backlinks)

#     # dump JSONL and CSV (sorted ascending by metric)
#     # jsonl_path = outdir / "enriched.jsonl"
#     json_path = outdir / f"{inp.stem}_enriched.json"
#     with json_path.open("w", encoding="utf-8") as f:
#         json.dump(enriched, f, ensure_ascii=False, indent=2)


#     sort_key = args.metric
#     missing = [r for r in enriched if sort_key not in r]
#     if missing:
#         print(f"[WARN] {len(missing)} records missing metric {sort_key}; they will sort as high values.")

#     def keyfn(r): return r.get(sort_key, float("inf"))
#     ranked = sorted(enriched, key=keyfn)

#     # csv_path = outdir / f"ranked_by_{sort_key}.csv"
#     csv_path = outdir / f"{inp.stem}_ranked_by_{sort_key}.csv"
#     with csv_path.open("w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
#         w.writeheader()
#         w.writerows(ranked)

#     # show bottom-K
#     k = min(args.bottom_k, len(ranked))
#     print(f"\nLowest {k} by {sort_key}:")
#     for i, r in enumerate(ranked[:k], 1):
#         print(f"{i:>3}. {r.get('Entity Name')} | QID={r.get('Wikidata ID')} | {sort_key}={r.get(sort_key)}")
        
#     bottom = ranked[:k]
#     lowest_json_path = outdir / f"{inp.stem}_lowest_{k}_by_{sort_key}.json"
#     with lowest_json_path.open("w", encoding="utf-8") as f:
#         json.dump(bottom, f, ensure_ascii=False, indent=2)

#     print(f"Wrote: {lowest_json_path}")
#     print(f"\nWrote: {json_path}")
#     print(f"Wrote: {csv_path}")

# if __name__ == "__main__":
#     main()
