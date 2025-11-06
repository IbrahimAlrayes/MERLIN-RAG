import requests, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from functools import lru_cache

HEADERS = {"User-Agent": "AgenticRAG/1.0 (hemoner1@gmail.com)"}
REQUEST_DELAY = 0.20
MAX_RETRIES = 3

def _get(url: str, params: Dict[str, Any] = None, *, accept_json: bool = True) -> Optional[Dict[str, Any]]:
    """GET with retries + minimal JSON safety."""
    params = params or {}
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json() if accept_json else {"text": r.text}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[WARN] GET failed: {url} params={params} err={e}")
                return None
            time.sleep(2 ** attempt)

# ------------------------ Wikipedia (per-language) ------------------------

def get_qid_from_title(title: str, lang: str) -> Optional[str]:
    """Resolve Wikidata QID from a Wikipedia page via pageprops.wikibase_item."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    data = _get(api, {"action":"query","titles":title,"prop":"pageprops","format":"json"})
    try:
        page = next(iter(data["query"]["pages"].values()))
        return page["pageprops"]["wikibase_item"]
    except Exception:
        return None  # page not found or no sitelink

def wikipedia_pageviews_90d(title: str, lang: str) -> Optional[int]:
    """Sum daily pageviews over the last 90 days for {lang}.wikipedia.org."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=90)
    base = "https://wikimedia.org/api/rest_v1"
    path = f"/metrics/pageviews/per-article/{lang}.wikipedia.org/all-access/all-agents/{title.replace(' ','_')}/daily/{start:%Y%m%d}/{end:%Y%m%d}"
    data = _get(base + path, {})
    if not data or "items" not in data: return None
    return sum(int(i.get("views",0)) for i in data["items"])  # Wikimedia REST. :contentReference[oaicite:4]{index=4}

def wikipedia_backlinks(title: str, lang: str) -> int:
    """Count backlinks on the target language Wikipedia."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    total = 0; params = {"action":"query","list":"backlinks","bltitle":title,"bllimit":"max","format":"json"}
    while True:
        data = _get(api, params)
        if not data: break
        total += len(data.get("query",{}).get("backlinks",[]))
        if "continue" in data: params.update(data["continue"])
        else: break
    return total  # MediaWiki Action API backlinks. :contentReference[oaicite:5]{index=5}

def wikipedia_structural_counts(title: str, lang: str) -> Dict[str, Any]:
    """Counts: size, revisions, unique editors, categories, images, external links."""
    api = f"https://{lang}.wikipedia.org/w/api.php"

    # size (prop=info)
    info = _get(api, {"action":"query","prop":"info","titles":title,"format":"json"})  # :contentReference[oaicite:6]{index=6}
    pages = info.get("query",{}).get("pages",{}) if info else {}
    pageid, page = (next(iter(pages.items())) if pages else (None, {}))
    out = {"article_size_bytes": page.get("length")}

    # revisions (rv) & contributors (pc) for robust unique editor count
    def _paged(prop: str, key: str, extra: Dict[str,Any]) -> List[Dict[str,Any]]:
        p = {"action":"query","prop":prop,"titles":title,"format":"json", **extra}
        items = []
        while True:
            data = _get(api, p)
            if not data: break
            pg = data.get("query",{}).get("pages",{}).get(str(pageid), {})
            items.extend(pg.get(key, []))
            if "continue" in data: p.update(data["continue"])
            else: break
        return items

    revs = _paged("revisions", "revisions", {"rvprop":"user","rvlimit":"max"})  # :contentReference[oaicite:7]{index=7}
    contribs = _paged("contributors", "contributors", {"pclimit":"max"})       # :contentReference[oaicite:8]{index=8}
    unique_editors = {r.get("user") for r in revs if "user" in r}
    unique_contribs = {c.get("name") or c.get("userid") for c in contribs}

    cats = _paged("categories", "categories", {"cllimit":"max"})               # :contentReference[oaicite:9]{index=9}
    imgs = _paged("images", "images", {"imlimit":"max"})                       # :contentReference[oaicite:10]{index=10}
    exts = _paged("extlinks", "extlinks", {"ellimit":"max"})                   # :contentReference[oaicite:11]{index=11}

    out.update({
        "revision_count": len(revs),
        "unique_editors": max(len(unique_editors), len(unique_contribs)),
        "category_count": len(cats),
        "image_count": len(imgs),
        "external_links": len(exts),
    })
    return out

def wikipedia_reference_count(title: str, lang: str) -> Optional[int]:
    """Proxy for number of references by counting <ref> tags from parsed HTML."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    data = _get(api, {"action":"parse","page":title,"prop":"text","format":"json"})  # :contentReference[oaicite:12]{index=12}
    try:
        html = data["parse"]["text"]["*"]
        return html.count("<ref")
    except Exception:
        return None

def wikipedia_text(title: str, lang: str, include_full: bool = True) -> Dict[str, Optional[str]]:
    """Summary (intro) + full plain text content via TextExtracts."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    q = {"action":"query","prop":"extracts","titles":title,"format":"json","explaintext":1,"exintro":1}
    intro = _get(api, q)  # TextExtracts. :contentReference[oaicite:13]{index=13}
    page_intro = next(iter(intro.get("query",{}).get("pages",{}).values()), {}) if intro else {}
    summary = page_intro.get("extract")
    content = None
    if include_full:
        full = _get(api, {"action":"query","prop":"extracts","titles":title,"format":"json","explaintext":1})
        page_full = next(iter(full.get("query",{}).get("pages",{}).values()), {}) if full else {}
        content = page_full.get("extract")
    return {"summary": summary, "content": content}

# ------------------------ Wikidata (language-agnostic) ------------------------

def wd_entity_age_days(qid: str) -> Optional[int]:
    """Age (days) since first revision on the Wikidata item."""
    data = _get("https://www.wikidata.org/w/api.php",
                {"action":"query","prop":"revisions","titles":qid,"rvlimit":1,"rvdir":"newer","rvprop":"timestamp","format":"json"})
    if not data: return None
    page = next(iter(data.get("query",{}).get("pages",{}).values()), {})
    revs = page.get("revisions", [])
    if not revs: return None
    t0 = datetime.fromisoformat(revs[0]["timestamp"].replace("Z","+00:00"))
    return (datetime.now(timezone.utc) - t0).days  # :contentReference[oaicite:14]{index=14}

def wd_claims_and_sitelinks(qid: str) -> Dict[str, Any]:
    """Return statement_count, qualifier_count, has_wikidata_image (P18), language_editions."""
    data = _get("https://www.wikidata.org/w/api.php",
                {"action":"wbgetentities","ids":qid,"props":"claims|sitelinks","format":"json"})  # :contentReference[oaicite:15]{index=15}
    if not data: return {}
    ent = data["entities"].get(qid, {})
    claims = ent.get("claims", {})
    statement_count = sum(len(lst) for lst in claims.values())
    qualifier_count = 0
    for lst in claims.values():
        for st in lst:
            qs = st.get("qualifiers", {})
            qualifier_count += sum(len(v) for v in qs.values())
    return {
        "statement_count": statement_count,
        "qualifier_count": qualifier_count,
        "has_wikidata_image": "P18" in claims,
        "language_editions": len(ent.get("sitelinks", {})),
    }

def wd_incoming_links(qid: str, threshold: int = 100000) -> int:
    """Count pages that link to the item (WhatLinksHere)."""
    params = {"action":"query","prop":"linkshere","titles":qid,"lhlimit":"max","format":"json"}
    total = 0
    while True:
        data = _get("https://www.wikidata.org/w/api.php", params)  # :contentReference[oaicite:16]{index=16}
        if not data: break
        for page in data.get("query",{}).get("pages",{}).values():
            links = page.get("linkshere", [])
            total += len(links)
            if total >= threshold: return total
        if "continue" in data: params.update(data["continue"])
        else: break
    return total

def wd_outgoing_links_distinct_props(qid: str) -> Optional[int]:
    """Count distinct wdt: predicates used by the item (SPARQL)."""
    query = f"""
    SELECT (COUNT(DISTINCT ?p) AS ?count) WHERE {{
      VALUES ?item {{ wd:{qid} }}
      ?item ?p ?o .
      FILTER(STRSTARTS(STR(?p), STR(wdt:)))
    }}
    """
    try:
        r = requests.get("https://query.wikidata.org/sparql",
                         params={"query": query, "format": "json"},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        return int(r.json()["results"]["bindings"][0]["count"]["value"])
    except Exception as e:
        print("[WARN] WDQS failed:", e)
        return None  # WDQS SPARQL endpoint. :contentReference[oaicite:17]{index=17}

# ------------------------ Orchestrator ------------------------

def get_title_from_qid(qid: str, lang: str = "en") -> Optional[str]:
    """Resolve the Wikipedia title for a given QID and language edition."""
    site_key = f"{lang}wiki"
    data = _get("https://www.wikidata.org/w/api.php",
                {"action": "wbgetentities", "ids": qid, "props": "sitelinks", "format": "json"})
    try:
        ent = data["entities"][qid]
        return ent["sitelinks"][site_key]["title"]
    except Exception:
        return None

@lru_cache(maxsize=None)
def get_title_from_qid_cached(qid: str, lang: str) -> Optional[str]:
    return get_title_from_qid(qid, lang)


def wikipedia_profile_by_title(title: str, lang: str = "en", *, qid: Optional[str] = None) -> Dict[str, Any]:
    """
    Return ALL requested columns + summary + full content for a Wikipedia page
    identified by (title, lang). Also resolves the linked Wikidata item.
    """
    qid = qid or get_qid_from_title(title, lang)  # prop=pageprops.wikibase_item
    profile = {
        "Input_Lang": lang,
        "Input_Title": title,
        "English_Wikipedia_Title": None,  # we can fill via sitelinks if you want later
        "Wikidata_ID": qid,
    }

    # Wikipedia-side metrics/content for this language
    pv = wikipedia_pageviews_90d(title, lang)
    bl = wikipedia_backlinks(title, lang)
    struct = wikipedia_structural_counts(title, lang)
    refs = wikipedia_reference_count(title, lang)
    text = wikipedia_text(title, lang, include_full=True)

    profile.update({
        "wikipedia_pageviews_90d": pv,
        "wikipedia_backlinks": bl,
        **struct,
        "reference_count": refs,
        "summary": text["summary"],
        "content": text["content"],
    })

    # Wikidata-side metrics (if we resolved QID)
    if qid:
        age_days = wd_entity_age_days(qid)
        claims = wd_claims_and_sitelinks(qid)
        profile.update({
            "wikidata_incoming_links": wd_incoming_links(qid),
            "wikidata_outgoing_links": wd_outgoing_links_distinct_props(qid),
            "language_editions": claims.get("language_editions"),
            "statement_count": claims.get("statement_count"),
            "has_wikidata_image": claims.get("has_wikidata_image"),
            "qualifier_count": claims.get("qualifier_count"),
            "entity_age_days": age_days,
            "entity_age_years": (round(age_days/365.25, 2) if isinstance(age_days, int) else None),
        })
    else:
        profile.update({
            "wikidata_incoming_links": None,
            "wikidata_outgoing_links": None,
            "language_editions": None,
            "statement_count": None,
            "has_wikidata_image": None,
            "qualifier_count": None,
            "entity_age_days": None,
            "entity_age_years": None,
        })

    return profile


# def wikipedia_profile_by_qid(qid: str, lang: str = "en") -> Dict[str, Any]:
#     """
#     Return the same profile as `wikipedia_profile_by_title`, but keyed off a Wikidata QID.
#     Falls back to Wikidata-only metrics if the language has no associated article.
#     """
#     title = get_title_from_qid(qid, lang)
#     if title:
#         return wikipedia_profile_by_title(title, lang, qid=qid)

#     # No sitelink for the requested language; still surface Wikidata stats.
#     profile = {
#         "Input_Lang": lang,
#         "Input_Title": None,
#         "English_Wikipedia_Title": None,
#         "Wikidata_ID": qid,
#         "wikipedia_pageviews_90d": None,
#         "wikipedia_backlinks": None,
#         "article_size_bytes": None,
#         "revision_count": None,
#         "unique_editors": None,
#         "category_count": None,
#         "image_count": None,
#         "external_links": None,
#         "reference_count": None,
#         "summary": None,
#         "content": None,
#     }
#     age_days = wd_entity_age_days(qid)
#     claims = wd_claims_and_sitelinks(qid)
#     profile.update({
#         "wikidata_incoming_links": wd_incoming_links(qid),
#         "wikidata_outgoing_links": wd_outgoing_links_distinct_props(qid),
#         "language_editions": claims.get("language_editions"),
#         "statement_count": claims.get("statement_count"),
#         "has_wikidata_image": claims.get("has_wikidata_image"),
#         "qualifier_count": claims.get("qualifier_count"),
#         "entity_age_days": age_days,
#         "entity_age_years": (round(age_days/365.25, 2) if isinstance(age_days, int) else None),
#     })
#     return profile

def wikipedia_profile_by_qid(qid: str, lang: str = "en") -> Dict[str, Any]:
    """
    Enhanced version:
    1) Try language-specific sitelink first
    2) If missing, fallback to English sitelink
    3) If still missing, fallback to Wikidata-only stats
    """

    # 1️⃣ Initial attempt with target language
    title = get_title_from_qid_cached(qid, lang)

    # ✅ Fallback to English Wikipedia if missing sitelink in target language
    if not title:
        en_title = get_title_from_qid_cached(qid, "en")
        if en_title:
            print(f"[FALLBACK] {qid}: No {lang}wiki sitelink. Using English page: {en_title}")
            profile = wikipedia_profile_by_title(en_title, "en", qid=qid)
            profile["English_Wikipedia_Title"] = en_title
            profile["fallback_lang"] = "en"
            return profile

        # ❌ No Wikipedia page in requested language OR English → Wikidata only
        print(f"[MISS] {qid}: No Wikipedia sitelinks in {lang} or en")
        profile = {
            "Input_Lang": lang,
            "Input_Title": None,
            "English_Wikipedia_Title": None,
            "Wikidata_ID": qid,
            "fallback_lang": None,
            "wikipedia_pageviews_90d": None,
            "wikipedia_backlinks": None,
            "article_size_bytes": None,
            "revision_count": None,
            "unique_editors": None,
            "category_count": None,
            "image_count": None,
            "external_links": None,
            "reference_count": None,
            "summary": None,
            "content": None,
        }

        # ✅ Still include Wikidata enrichment
        age_days = wd_entity_age_days(qid)
        claims = wd_claims_and_sitelinks(qid)
        profile.update({
            "wikidata_incoming_links": wd_incoming_links(qid),
            "wikidata_outgoing_links": wd_outgoing_links_distinct_props(qid),
            "language_editions": claims.get("language_editions"),
            "statement_count": claims.get("statement_count"),
            "has_wikidata_image": claims.get("has_wikidata_image"),
            "qualifier_count": claims.get("qualifier_count"),
            "entity_age_days": age_days,
            "entity_age_years": (round(age_days / 365.25, 2) if isinstance(age_days, int) else None),
        })
        return profile

    # ✅ Full success path (language-specific Wikipedia exists)
    print(f"[OK] {qid}: Found {lang}wiki page: {title}")
    profile = wikipedia_profile_by_title(title, lang, qid=qid)
    profile["English_Wikipedia_Title"] = get_title_from_qid_cached(qid, "en")  # Fill for downstream code
    profile["fallback_lang"] = None
    return profile

if __name__ == "__main__":
    # title = "तेजिंदर पाल सिंह बग्गा: 'हमलावर' से बीजेपी उम्मीदवार तक" 
    # lang = "hi"
    # qid = get_qid_from_title(title, lang)
    # if qid:
    #     profile = wikipedia_profile_by_qid(qid, lang)
    #     for k, v in profile.items():
    #         print(f"{k}: {v}")
    # else:
    #     print("QID not found for sample title.")
    
    i = wikipedia_profile_by_qid('Q29476895', 'hi')
    print(i)
