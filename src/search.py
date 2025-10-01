from __future__ import annotations
import re
import os
import logging
from typing import Dict, List, Any, Tuple
import time
import requests
from requests.exceptions import Timeout, ConnectionError, ChunkedEncodingError

from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
import math
from langchain_community.utilities import SearxSearchWrapper

from dotenv import load_dotenv


########################
# Env Loading
########################

load_dotenv()


########################
# Configuration
########################
SEARX_HOST = os.getenv("SEARXNG_HOST", "http://localhost:8080").rstrip("/")
READER_HOST = os.getenv("READER_HOST", "http://localhost:3001").rstrip("/")
READER_MODE = os.getenv("READER_MODE", "proxy")  # "proxy" or "reader"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
MIN_CONTENT_CHARS = int(os.getenv("MIN_CONTENT_CHARS", "400"))
MIN_SOURCES = int(os.getenv("MIN_SOURCES", "2"))
LANG = os.getenv("LANG", "auto")

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "7"))
RETRY_TIMEOUT = int(os.getenv("RETRY_TIMEOUT", "10"))
RETRY_ON_FAILURE = int(os.getenv("RETRY_ON_FAILURE", "1"))

FETCH_WORKERS = int(os.getenv("FETCH_WORKERS", "8"))

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
log = logging.getLogger("merlin-rag.search")


##########################
## Orchestrator
##########################


def search_and_fetch(query: str, k: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """
    Orchestrator:
      searx_search -> dedupe_by_link -> fetch_urls_with_proxy -> normalize_records
    Returns a clean list of dicts ready for display.
    """
    # 1) search
    raw_hits = searx_search(query, k)
    searched = len(raw_hits)

    # 2) dedupe
    hits = dedupe_by_link(raw_hits)
    deduped = len(hits)

    # 3) fetch content via your proxy
    hits = fetch_urls_with_proxy(hits)
    fetched = len(hits)

    # 4) normalize shape
    clean = normalize_records(hits)

    try:
        kept, dropped = filter_failed_and_tiny(clean, min_chars=MIN_CONTENT_CHARS)
        successes = len(kept)
        failures = len(dropped)
        snippet_only = successes < MIN_SOURCES
    except Exception as _:
        # If filter function isn’t present for some reason, fall back to simple counts
        successes = sum(
            1
            for it in clean
            if isinstance(it.get("content"), str)
            and not it["content"].strip().startswith(("Request failed:", "Error "))
            and len(_clean(it["content"])) >= MIN_CONTENT_CHARS
        )
        failures = len(clean) - successes
        snippet_only = successes < MIN_SOURCES

    _log_run_summary(
        searched=searched,
        deduped=deduped,
        fetched=fetched,
        successes=successes,
        failures=failures,
        snippet_only=snippet_only,
    )
    # --- end stats ---

    return clean


def _log_run_summary(
    searched: int,
    deduped: int,
    fetched: int,
    successes: int,
    failures: int,
    snippet_only: bool,
) -> None:
    log.info(
        "search_and_fetch summary | searched=%d, deduped=%d, fetched=%d, successes=%d, failures=%d, snippet_only=%s",
        searched,
        deduped,
        fetched,
        successes,
        failures,
        str(snippet_only),
    )


###########
## URLs
###########
def dedupe_by_link(items: List[Dict]) -> List[Dict]:
    """
    Keep only the first occurrence of each unique 'link'.
    Items without a 'link' are kept (can't dedupe them reliably).
    """
    seen = set()
    out: List[Dict] = []
    for it in items:
        link = it.get("link")
        if not link:
            out.append(it)
            continue
        if link in seen:
            continue
        seen.add(link)
        out.append(it)
    return out


######################
## Search (SearXNG)
#####################
def searx_search(query: str, k: int = 5) -> List[Dict]:
    """
    Run a SearXNG search via LangChain and return a list of result dicts
    shaped like: {title, link, snippet, engines, category}.

    Args:
        query: Search query string.
        k:     Max number of results to return.

    Returns:
        List[Dict]: Each item contains keys:
            - title (str)
            - link (str)
            - snippet (str | None)
            - engines (list[str] | None)
            - category (str | None)
    """
    s = SearxSearchWrapper(searx_host=SEARX_HOST)

    try:
        raw = s.results(query=query, num_results=k)
        return raw
    except Exception as e:
        print(f"[searx_search] Error: {e}")
        return []


##########################
## HTTP Fetch via Proxy
##########################
def fetch_urls_with_proxy(
    data: List[Dict[str, Any]],
    proxy_base: str = READER_HOST,
    timeout: int = TIMEOUT,
    retry_timeout: int = RETRY_TIMEOUT,
    retry_delay: float = 0.75,
    max_retries: int = RETRY_ON_FAILURE,
    max_workers: int = FETCH_WORKERS,
) -> List[Dict[str, Any]]:
    """
    Fast, concurrent fetch through a proxy with minimal retry logic.

    - Reuses a single requests.Session for connection pooling / keep-alive.
    - Parallelizes network I/O with a thread pool (I/O-bound).
    - One or more retries on transient errors, controlled by `max_retries`.
    - Returns items in the same order as `data`.
    """
    proxy_base = proxy_base.rstrip("/") + "/"
    transient_status = {429, 500, 502, 503, 504}
    transient_errors = (Timeout, ConnectionError, ChunkedEncodingError)

    # Sensible default for I/O-bound concurrency
    if max_workers is None:
        # Scale with batch size but cap to avoid thrashing
        max_workers = min(16, max(4, math.ceil(len(data) / 2)))

    # Prepare a pooled session (keep-alive, larger pool)
    session = requests.Session()
    session.headers.update({"User-Agent": "MERLIN-RAG/0.1"})
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def fetch_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, item = idx_item
        url = item.get("link")
        if not url:
            item["content"] = None
            return idx, item

        proxied_url = proxy_base + url

        attempts = 0
        while True:
            try:
                t = timeout if attempts == 0 else retry_timeout
                resp = session.get(proxied_url, timeout=t)
                if resp.status_code == 200:
                    item["content"] = resp.text
                else:
                    item["content"] = f"Error {resp.status_code}"
                return idx, item
            except transient_errors as e:
                attempts += 1
                if attempts > max_retries:
                    item["content"] = f"Request failed: {e}"
                    return idx, item
                time.sleep(retry_delay * (1.25 ** (attempts - 1)))  # tiny backoff
            except Exception as e:
                item["content"] = f"Request failed: {e}"
                return idx, item

    # Submit tasks
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, item in enumerate(data):
            futures.append(ex.submit(fetch_one, (idx, item.copy())))

        # Collect in an index-addressable buffer to preserve order
        results_buffer: List[Tuple[int, Dict[str, Any]] | None] = [None] * len(data)
        for fut in as_completed(futures):
            idx, out_item = fut.result()
            results_buffer[idx] = (idx, out_item)

    # Flatten buffer (preserving original order)
    results: List[Dict[str, Any]] = [
        pair[1] for pair in results_buffer if pair is not None
    ]

    return results


##########################
## Record Normalization
##########################
def normalize_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each record has the expected keys and simple, safe types.
    Keys: title, link, snippet, engines, category, content
    """
    normalized: List[Dict[str, Any]] = []
    for it in items:
        normalized.append(
            {
                "title": (it.get("title") or "")
                if isinstance(it.get("title"), str)
                else "",
                "link": it.get("link"),
                "snippet": it.get("snippet")
                if isinstance(it.get("snippet"), str)
                else None,
                "engines": it.get("engines")
                if isinstance(it.get("engines"), list)
                else None,
                "category": it.get("category")
                if isinstance(it.get("category"), str)
                else None,
                ## ADDING content field from fetch step
                "content": it.get("content")
                if isinstance(it.get("content"), str)
                else None,
            }
        )
    return normalized


##########################
## Cleaning & Filtering
##########################
def _truncate(text: str, limit: int) -> str:
    """Truncate to ~limit chars, trying to end on a sentence boundary."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # try to cut at last sentence-ish boundary in the window
    m = re.search(r"(?s).*[.!?…]\s+", cut)
    if m:
        return cut[: m.end()].rstrip()
    return cut.rstrip() + "…"


def _clean(text: str) -> str:
    """Light cleanup for prompt-friendliness."""
    if text is None:
        return ""
    # collapse extreme whitespace, remove control chars
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def is_failed_content(text: str | None) -> bool:
    """
    Return True if the page 'content' looks like a fetch error.
    We treat None/empty as failed too.
    """
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    # cheap checks that match your current error shapes
    prefixes = ("Request failed:", "Error ")
    return s.startswith(prefixes)


def filter_failed_and_tiny(
    items: List[Dict[str, Any]],
    min_chars: int = 400,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Keep only items that:
      - do NOT look like failures, and
      - have non-trivial content length after cleanup (>= min_chars)

    Returns:
      kept, dropped
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for it in items:
        content = it.get("content") or ""
        if is_failed_content(content):
            dropped.append(it)
            continue

        cleaned = _clean(content)
        if len(cleaned) >= min_chars:
            kept.append(it)
        else:
            dropped.append(it)

    return kept, dropped


#############################
## Formatting for the LLM
#############################
def to_snippet_only(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the minimal fields for snippet-only fallback.
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "title": (it.get("title") or "")
                if isinstance(it.get("title"), str)
                else "",
                "link": it.get("link"),
                "snippet": it.get("snippet")
                if isinstance(it.get("snippet"), str)
                else None,
            }
        )
    return out


def format_results_for_llm_snippet_only(
    results: List[Dict[str, Any]],
    notice: str = "Some sources could not be fetched, so only snippets are available. Use them cautiously.",
) -> str:
    """
    Build a prompt that includes ONLY title + URL + snippet per source,
    and prepends a clear notice for the model.
    """
    lines: List[str] = []
    lines.append(notice)
    lines.append("")
    lines.append("SOURCES (snippet-only):")

    for i, item in enumerate(results, start=1):
        title = _clean(item.get("title") or "") or "(Untitled)"
        link = (item.get("link") or "").strip()
        snippet = _clean(item.get("snippet") or "")

        lines.append(f"[S{i}] {title}")
        if link:
            lines.append(f"URL: {link}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("-" * 60)

    return "\n".join(lines)


def format_results_for_llm(
    results: List[Dict[str, Any]],
) -> str:
    """
    Turn `search_and_fetch(...)` results into a single prompt context string.

    Each source is labeled [S#] so you can ask the LLM to cite as [S1], [S2], etc.

    Args:
        results: list of dicts with keys: title, link, snippet, content, engines, category
        query:   the user question (placed at the end as instruction context)
        max_context_chars: total budget for all source contents combined
        min_per_doc: minimum chars allocated for each doc's content

    Returns:
        A single string suitable to pass as the LLM's context.
    """

    lines: List[str] = []
    lines.append(
        "You are a careful assistant. Use the SOURCES below to answer the QUESTION."
    )
    lines.append("")
    lines.append("SOURCES:")

    for i, item in enumerate(results, start=1):
        title = _clean(item.get("title") or "") or "(Untitled)"
        link = (item.get("link") or "").strip()
        snippet = _clean(item.get("snippet") or "")
        category = item.get("category") or None
        content = _clean(item.get("content") or "")

        header_bits = [f"[S{i}] {title}"]
        if category:
            header_bits.append(f"(category: {category})")
        lines.append(" ".join(header_bits))
        if link:
            lines.append(f"URL: {link}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("Content:")
        lines.append(content if content else "(empty)")
        lines.append("-" * 60)

    return "\n".join(lines)
