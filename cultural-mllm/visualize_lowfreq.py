#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_records(input_path: Path) -> Tuple[pd.DataFrame, List[Path]]:
    """Load one or more JSON files (each a list of dicts) and add 'language' column."""
    files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {input_path}")
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No .json files found in {input_path}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{fp} does not contain a JSON array")
        language = fp.stem
        for row in data:
            row = dict(row)
            row.setdefault("language", language)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    expected_cols = [
        "Article Title", "Entity Name", "Wikidata ID", "English Wikipedia Title",
        "Image Name", "wikidata_incoming_links", "language",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["wikidata_incoming_links"] = pd.to_numeric(df["wikidata_incoming_links"], errors="coerce").fillna(0).astype(int)
    return df, files


def plot_histogram(df: pd.DataFrame, outdir: Path) -> Path:
    """Distribution of wikidata_incoming_links (log x-axis)."""
    fig, ax = plt.subplots()
    vals = df["wikidata_incoming_links"].values
    bins = np.logspace(0, math.log10(max(1, vals.max() + 1)), 30) if vals.max() > 0 else np.arange(2) - 0.5
    ax.hist(vals, bins=bins)
    ax.set_xscale("log" if vals.max() > 0 else "linear")
    ax.set_xlabel("Wikidata incoming links (log scale)")
    ax.set_ylabel("Count of entities")
    ax.set_title("Distribution of entity frequency (all languages)")
    fig.tight_layout()
    out = outdir / "hist_incoming_links.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_box_by_language(df: pd.DataFrame, outdir: Path) -> Path:
    """Boxplot of incoming links per language."""
    grouped = [g["wikidata_incoming_links"].values for _, g in df.groupby("language")]
    labels = [name for name, _ in df.groupby("language")]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    ax.boxplot(grouped, showfliers=False)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Wikidata incoming links")
    ax.set_title("Incoming links by language (boxplot, outliers hidden)")
    fig.tight_layout()
    out = outdir / "box_incoming_by_language.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_ecdf(df: pd.DataFrame, outdir: Path) -> Path:
    """ECDF (cumulative %) for incoming links; helps pick a rarity cutoff."""
    vals = np.sort(df["wikidata_incoming_links"].values)
    y = np.arange(1, len(vals) + 1) / len(vals)
    fig, ax = plt.subplots()
    ax.plot(vals, y, drawstyle="steps-post")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Wikidata incoming links (symlog)")
    ax.set_ylabel("Cumulative fraction of entities")
    ax.set_title("ECDF of entity frequency")
    fig.tight_layout()
    out = outdir / "ecdf_incoming_links.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bottom_k_bar(df: pd.DataFrame, outdir: Path, k: int = 20) -> Path:
    """Bar chart of bottom-k rarest entities (ascending)."""
    bottom = df.sort_values("wikidata_incoming_links", ascending=True).head(k)
    labels = [f"{r['Entity Name']}" for _, r in bottom.iterrows()]
    values = bottom["wikidata_incoming_links"].tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wikidata incoming links")
    ax.set_title(f"Bottom {k} rarest entities (overall)")
    fig.tight_layout()
    out = outdir / f"bar_bottom_{k}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# -------------------------------------------------------------------------
# 🔍 NEW: Cross-language comparison plots
# -------------------------------------------------------------------------
def plot_language_hist(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    langs = sorted(df["language"].unique())
    for lang in langs:
        subset = df[df["language"] == lang]["wikidata_incoming_links"]
        plt.hist(subset, bins=20, alpha=0.5, label=lang)
    plt.xlabel("Wikidata incoming links")
    plt.ylabel("Count")
    plt.title("Distribution of Wikidata incoming links per language")
    plt.legend()
    plt.tight_layout()
    out = outdir / "language_hist_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_avg_incoming(df: pd.DataFrame, outdir: Path) -> Path:
    stats = df.groupby("language")["wikidata_incoming_links"].agg(["mean", "median", "std", "count"]).reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(stats["language"], stats["mean"], yerr=stats["std"], capsize=4)
    plt.ylabel("Mean incoming links ± std")
    plt.title("Average Wikidata incoming links per language")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = outdir / "avg_incoming_links_per_language.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out, stats


def plot_avg_title_len(df: pd.DataFrame, outdir: Path) -> Path:
    df["title_len"] = df["English Wikipedia Title"].astype(str).str.len()
    avg_len = df.groupby("language")["title_len"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(avg_len["language"], avg_len["title_len"], color="teal")
    plt.ylabel("Average title length (chars)")
    plt.title("Average English Wikipedia title length per language")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = outdir / "avg_title_length_per_language.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out, avg_len


def plot_scatter(df: pd.DataFrame, outdir: Path) -> Path:
    df["title_len"] = df["English Wikipedia Title"].astype(str).str.len()
    grp = df.groupby("language").agg(
        mean_links=("wikidata_incoming_links", "mean"),
        mean_len=("title_len", "mean"),
    ).reset_index()
    plt.figure(figsize=(6, 5))
    plt.scatter(grp["mean_links"], grp["mean_len"], s=60)
    for _, row in grp.iterrows():
        plt.text(row["mean_links"], row["mean_len"], row["language"], fontsize=9, ha="center", va="bottom")
    plt.xlabel("Mean Wikidata incoming links")
    plt.ylabel("Mean title length")
    plt.title("Popularity vs Title length by language")
    plt.tight_layout()
    out = outdir / "scatter_popularity_vs_length.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out, grp


def summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    overall = {
        "count": int(df.shape[0]),
        "min": int(df["wikidata_incoming_links"].min() if not df.empty else 0),
        "p10": int(df["wikidata_incoming_links"].quantile(0.10) if not df.empty else 0),
        "median": int(df["wikidata_incoming_links"].median() if not df.empty else 0),
        "p90": int(df["wikidata_incoming_links"].quantile(0.90) if not df.empty else 0),
        "max": int(df["wikidata_incoming_links"].max() if not df.empty else 0),
    }
    by_lang = (
        df.groupby("language")["wikidata_incoming_links"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return {"overall": overall, "by_language": by_lang}


def main():
    ap = argparse.ArgumentParser(description="Visualize and compare entity frequency across languages.")
    ap.add_argument("--input_path", required=True, help="Path to JSON file or folder.")
    ap.add_argument("--output_dir", required=True, help="Directory to save charts and tables.")
    ap.add_argument("--bottom_k", type=int, default=20, help="How many rarest entities to chart.")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, files = load_records(input_path)
    if df.empty:
        raise SystemExit("No records loaded.")

    # Save merged data
    merged_json = outdir / "merged_all.json"
    merged_json.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    # Existing plots
    paths = [
        plot_histogram(df, outdir),
        plot_box_by_language(df, outdir),
        plot_ecdf(df, outdir),
        plot_bottom_k_bar(df, outdir, k=args.bottom_k),
    ]

    # New comparative plots
    paths.append(plot_language_hist(df, outdir))
    p_avg, avg_links = plot_avg_incoming(df, outdir)
    paths.append(p_avg)
    p_len, avg_len = plot_avg_title_len(df, outdir)
    paths.append(p_len)
    p_scat, scatter = plot_scatter(df, outdir)
    paths.append(p_scat)

    # Merge stats for CSV
    summary = avg_links.merge(avg_len, on="language", how="left")
    summary.to_csv(outdir / "LANGUAGE_COMPARISON_STATS.csv", index=False)

    print("Loaded files:")
    for fp in files:
        print(f" - {fp.name}")
    print("\nCharts generated:")
    for p in paths:
        print(f" - {p}")
    print(f"\nSaved: LANGUAGE_COMPARISON_STATS.csv")

if __name__ == "__main__":
    main()
