# app.py
from __future__ import annotations
import os, json, io
import pandas as pd
import streamlit as st

# --- import your backend (put the code you pasted in search_backend.py OR same folder as a module) ---
# If your functions live in the same file/project (e.g., merlin_rag/search.py), adjust the import below.

from search import (  # rename to your actual module path if different
    search_and_fetch,
    filter_failed_and_tiny,
    to_snippet_only,
    format_results_for_llm_snippet_only,
    format_results_for_llm,
    SEARX_HOST, READER_HOST, MIN_CONTENT_CHARS, MIN_SOURCES, MAX_RESULTS,
)


st.set_page_config(page_title="Web Scraper UI", page_icon="🌐", layout="wide")

# -------------------- SIDEBAR: Settings --------------------
st.sidebar.title("Settings")
st.sidebar.caption("These mirror your backend env/config.")
st.sidebar.write(f"**SearXNG**: {SEARX_HOST}")
st.sidebar.write(f"**Reader Proxy**: {READER_HOST}")

k = st.sidebar.slider("Max results (k)", min_value=1, max_value=20, value=min(MAX_RESULTS, 5), step=1)
min_chars = st.sidebar.number_input("Min content chars (filter)", min_value=0, value=MIN_CONTENT_CHARS, step=100)
min_sources = st.sidebar.number_input("Min sources (for full mode)", min_value=0, value=MIN_SOURCES, step=1)
truncate_chars = st.sidebar.slider("Preview truncate length", 200, 5000, 1200, 100)
st.sidebar.divider()
export_include_html = st.sidebar.checkbox("Include raw HTML in export", value=False,
                                          help="Exports full raw content; disable to keep JSON smaller.")

# -------------------- HEADER --------------------
st.title("🌐 Friendly Web Scraper (SearXNG + Proxy Reader)")
st.caption("Type a query, fetch via your proxy, preview results, and copy an LLM-ready prompt.")

# -------------------- INPUT --------------------
query = st.text_input("Query", placeholder="e.g. best practices for RAG routing, 2025")
go = st.button("Search & Fetch", type="primary", use_container_width=True)

# Storage for last results
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "kept" not in st.session_state:
    st.session_state.kept = []
if "dropped" not in st.session_state:
    st.session_state.dropped = []

# -------------------- RUN --------------------
if go and query.strip():
    with st.status("Running search → dedupe → fetch → normalize …", expanded=False) as status:
        try:
            print("Running search_and_fetch with query:", query, "k:", k)
            results = search_and_fetch(query.strip(), k=k)
            print(f"Fetched {len(results)} results.")
            kept, dropped = filter_failed_and_tiny(results, min_chars=min_chars)
            st.session_state.last_results = results
            st.session_state.kept = kept
            st.session_state.dropped = dropped
            status.update(label="Done ✅", state="complete")
        except Exception as e:
            status.update(label="Failed ❌", state="error")
            st.error(f"Error while fetching: {e}")

# -------------------- RESULTS --------------------
results = st.session_state.last_results
kept = st.session_state.kept
dropped = st.session_state.dropped

if results:
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total returned", len(results))
    c2.metric("Usable (kept)", len(kept))
    c3.metric("Dropped / failed", len(dropped))
    c4.metric("Snippet-only fallback?", "Yes" if len(kept) < min_sources else "No")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Table", "LLM Prompt", "Export"])

    # --- PREVIEW ---
    with tab1:
        st.subheader("Preview")
        st.caption("Expand to view snippet and truncated content.")
        for i, item in enumerate(results, start=1):
            title = (item.get("title") or "").strip() or "(Untitled)"
            link = (item.get("link") or "").strip()
            engines = item.get("engines")
            snippet = item.get("snippet") or ""
            content = item.get("content") or ""
            clean_preview = (content[:truncate_chars] + "…") if len(content) > truncate_chars else content

            color = "✅" if item in kept else "⚠️"
            with st.expander(f"{color} [S{i}] {title}"):
                if link:
                    st.markdown(f"**URL:** {link}")
                if engines:
                    st.write("Engines:", ", ".join(engines))
                if snippet:
                    st.markdown("**Snippet**")
                    st.write(snippet)
                st.markdown("**Content (truncated preview)**")
                st.code(clean_preview or "(empty)", language="html")

    # --- TABLE ---
    with tab2:
        st.subheader("Results (table)")
        df_rows = []
        for i, it in enumerate(results, start=1):
            df_rows.append({
                "S#": i,
                "title": (it.get("title") or "").strip(),
                "link": it.get("link"),
                "snippet": (it.get("snippet") or "").strip(),
                "category": it.get("category"),
                "engines": ", ".join(it.get("engines") or []) if isinstance(it.get("engines"), list) else None,
                "content_len": len(it.get("content") or ""),
                "kept": it in kept
            })
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # --- LLM PROMPT ---
    with tab3:
        st.subheader("LLM Prompt")
        mode = "Snippet-only" if len(kept) < min_sources else "Full"
        st.caption(f"Mode: **{mode}** (threshold={min_sources})")

        if len(kept) < min_sources:
            # snippet-only prompt
            minimal = to_snippet_only(results)
            prompt = format_results_for_llm_snippet_only(minimal)
        else:
            # full prompt with content
            prompt = format_results_for_llm(kept)

        st.text_area("Copy this into your LLM context:", value=prompt, height=320)
        st.download_button("Download prompt (.txt)", data=prompt.encode("utf-8"),
                           file_name="llm_prompt.txt", mime="text/plain")

    # --- EXPORT ---
    with tab4:
        st.subheader("Export")
        # Clean export object
        def _clean_for_export(items):
            out = []
            for it in items:
                rec = {
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("snippet"),
                    "category": it.get("category"),
                    "engines": it.get("engines"),
                }
                if export_include_html:
                    rec["content"] = it.get("content")
                else:
                    # smaller file: include length only
                    rec["content_len"] = len(it.get("content") or "")
                out.append(rec)
            return out

        export_all = _clean_for_export(results)
        export_kept = _clean_for_export(kept)
        export_dropped = _clean_for_export(dropped)

        colA, colB, colC = st.columns(3)
        colA.download_button("Download ALL (.json)", json.dumps(export_all, ensure_ascii=False, indent=2),
                             "results_all.json", "application/json", use_container_width=True)
        colB.download_button("Download KEPT (.json)", json.dumps(export_kept, ensure_ascii=False, indent=2),
                             "results_kept.json", "application/json", use_container_width=True)
        colC.download_button("Download DROPPED (.json)", json.dumps(export_dropped, ensure_ascii=False, indent=2),
                             "results_dropped.json", "application/json", use_container_width=True)

        # CSV (no content html column)
        df_csv = pd.DataFrame(export_all)
        buf = io.StringIO()
        df_csv.to_csv(buf, index=False)
        st.download_button("Download ALL (.csv)", buf.getvalue(), "results_all.csv", "text/csv", use_container_width=True)

else:
    st.info("Enter a query and click **Search & Fetch** to begin.")
