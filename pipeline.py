# -*- coding: utf-8 -*-
"""POPABOT pipeline — LLM-only (3-file project)

Files:
  - tg_scraper.py        → collects & writes CSVs (new_messages.csv, latest_combined.csv)
  - llm_report_writer.py → analyzes CSV and writes Markdown (LLM-only, Markdown output)
  - pipeline.py          → orchestrates scrape → load → plot → LLM report

Behavior:
  - ALWAYS uses the LLM writer (no CPU fallback).
  - Scraper is optional (POPABOT_USE_SCRAPER=1). In CI, keep it off.
  - Robust CSV loading: uses data/new_messages.csv; if missing, falls back to data/latest_combined.csv.
  - Plots keyword trend image to POPABOT_TREND_PNG.
  - Passes user keywords into the LLM writer.

Env vars:
  - POPABOT_USE_SCRAPER (default 0)
  - POPABOT_SAMPLE_N (e.g., 50 to speed up local runs)
  - POPABOT_OUT_MD (default analytics/output/summary.md)
  - POPABOT_TREND_PNG (default analytics/output/keyword_mentions.png)
  - POPABOT_USE_LLM is ignored here (we are LLM-only by design)
"""

from __future__ import annotations
import os
import re
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: scraper
try:
    from tg_scraper import run_scraper
    SCRAPER_AVAILABLE = True
except Exception:
    SCRAPER_AVAILABLE = False

# LLM writer (required)
from llm_report_writer import write_markdown_report as write_md_llm

# ---------- Text utils for auto keywords ----------
STOPWORDS = set(
    """a an and are as at be by for from has have i in is it its of on or our that the their them they this to was were will with you your we he she not if but than then so such""".split()
)


def _kw_pattern(kw: str) -> str:
    core = re.sub(r"\s+", r"\\s+", re.escape(kw.lower()))
    return rf"\b{core}\b"


def _tokenize(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    # strip URLs and punctuation
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9_#@]+", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) >= 3]
    return toks


def _simple_tfidf(texts: list[str]) -> list[tuple[str, float]]:
    docs = [t for t in texts if isinstance(t, str) and t.strip()]
    if not docs:
        return []
    import math
    from collections import Counter
    N = len(docs)
    tfs: list[Counter] = []
    df: Counter = Counter()
    for d in docs:
        toks = _tokenize(d)
        tf = Counter(toks)
        tfs.append(tf)
        for tok in set(tf):
            df[tok] += 1
    scores: Counter = Counter()
    for tf in tfs:
        for tok, f in tf.items():
            idf = math.log((N + 1) / (df[tok] + 1)) + 1
            scores[tok] += f * idf
    for tok in list(scores):
        scores[tok] /= N
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def auto_top_keywords(df: pd.DataFrame, top_n: int = 6) -> list[tuple[str, float]]:
    terms = _simple_tfidf(list(df["text"].astype(str)))
    terms = [(t, s) for t, s in terms if t not in STOPWORDS][:top_n]
    return terms

# ---------- IO helpers ----------

def load_new_data(new_path='data/new_messages.csv', fallback_path='data/latest_combined.csv') -> pd.DataFrame:
    """Load the input CSV. If new_path is missing, try the fallback.
    Ensures required columns & UTC datetimes; returns frame sorted by date.
    """
    path_used = None
    if os.path.exists(new_path):
        path_used = new_path
    elif os.path.exists(fallback_path):
        path_used = fallback_path
        print(f"ℹ️ Using fallback CSV: {fallback_path}")
    else:
        raise FileNotFoundError(f"Input CSV not found: {new_path} (no fallback {fallback_path})")

    df = pd.read_csv(path_used)
    must_have = {"message_id", "group", "date", "text", "url"}
    missing = must_have - set(df.columns)
    if missing:
        raise ValueError(f"CSV must include columns: {sorted(missing)}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.sort_values('date')

    # If we ended up loading from fallback, also mirror it to new_path for downstream convenience
    if path_used == fallback_path and not os.path.exists(new_path):
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        try:
            shutil.copyfile(fallback_path, new_path)
        except Exception:
            pass

    return df


def load_keywords(keywords_path='keywords/') -> list[str]:
    if not os.path.isdir(keywords_path):
        print(f"ℹ️ Keywords folder not found: {keywords_path} (continuing without user keywords)")
        return []
    out = set()
    for fname in os.listdir(keywords_path):
        path = os.path.join(keywords_path, fname)
        if not os.path.isfile(path):
            continue
        if not any(fname.lower().endswith(ext) for ext in (".txt", ".csv", ".tsv", ".list", ".keywords", "")):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    kw = line.strip()
                    if not kw or kw.startswith("#"):
                        continue
                    out.add(kw.strip(" '\"").lower())
        except Exception as e:
            print(f"⚠️ Failed to read keywords from {path}: {e}")
    return sorted(out)

# ---------- Plot ----------

def _choose_plot_keywords(df: pd.DataFrame, user_keywords: list[str], max_k=6) -> list[str]:
    if user_keywords:
        texts = df["text"].astype(str).str.lower()
        totals = []
        for kw in user_keywords:
            pat = _kw_pattern(kw)
            try:
                cnt = int(texts.str.count(pat, flags=re.IGNORECASE).sum())
            except TypeError:
                cnt = int(texts.apply(lambda t: len(re.findall(pat, t, flags=re.IGNORECASE))).sum())
            totals.append((kw, cnt))
        nonzero = [(k, c) for k, c in totals if c > 0]
        if nonzero:
            nonzero.sort(key=lambda x: x[1], reverse=True)
            return [k for k, _ in nonzero[:max_k]]
    auto = auto_top_keywords(df, top_n=max_k)
    return [kw for kw, _ in auto]


def plot_keyword_mentions(df: pd.DataFrame, keywords: list[str], save_path='analytics/output/keyword_mentions.png') -> None:
    if df.empty:
        print("⚠️ No data to plot.")
        return
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce", utc=True)
    s = s[pd.notna(s["date"])]
    if s.empty:
        print("⚠️ No valid dates to plot.")
        return
    if getattr(s["date"].dt, "tz", None) is not None:
        s["day"] = s["date"].dt.tz_convert(None).dt.date
    else:
        s["day"] = s["date"].dt.date

    plot_kws = _choose_plot_keywords(s, keywords, max_k=6)
    if not plot_kws:
        print("ℹ️ No suitable keywords found for plotting (skipping keyword trend plot).")
        return

    days = pd.date_range(start=min(s["day"]), end=max(s["day"]), freq="D").date
    frame = pd.DataFrame(index=days)

    texts = s["text"].astype(str).str.lower()
    for kw in frame.columns:
        # placeholder to initialize columns
        frame[kw] = 0
    for kw in plot_kws:
        pat = _kw_pattern(kw)
        try:
            counts = texts.str.count(pat, flags=re.IGNORECASE)
        except TypeError:
            counts = texts.apply(lambda t: len(re.findall(pat, t, flags=re.IGNORECASE)))
        per_day = counts.groupby(s["day"]).sum()
        frame[kw] = per_day
    frame = frame.fillna(0)

    if (frame.sum(axis=0) == 0).all():
        print("ℹ️ Keywords had zero mentions across the period (skipping plot).")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    for kw in frame.columns:
        plt.plot(frame.index, frame[kw].values, marker='o', label=kw)
    plt.title("Keyword Mentions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mentions")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved keyword trend plot → {save_path}")

# ---------- Pipeline ----------

def main() -> None:
    use_scraper = os.getenv("POPABOT_USE_SCRAPER", "0") == "1"
    sample_n = int(os.getenv("POPABOT_SAMPLE_N", "0"))
    out_md = os.getenv("POPABOT_OUT_MD", "analytics/output/summary.md")
    trend_png = os.path.join(os.path.dirname(out_md) or ".", "keyword_mentions.png")

    if use_scraper and SCRAPER_AVAILABLE:
        print("🛰️ Running scraper…")
        try:
            run_scraper()
        except Exception as e:
            print(f"⚠️ Scraper failed (continuing with existing CSV): {e}")

    # Load data (with fallback)
    df = load_new_data('data/new_messages.csv', 'data/latest_combined.csv')
    if df.empty:
        print("No new messages to analyze.")
        return

    if sample_n > 0:
        df = df.sort_values("date").tail(sample_n)
        print(f"🔎 Sampling last {sample_n} rows for faster iteration.")

    keywords = load_keywords('keywords/')

    # Deterministic trend plot
    plot_keyword_mentions(df, keywords, save_path=trend_png)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # LLM-only path
    write_md_llm(
        df=df,
        out_path=out_md,
        user_keywords=list(keywords),
        trend_png_path=trend_png,   # NEW: hand the path to the writer
    )
    print("🧠 Wrote LLM-first report.")
    print("✅ Pipeline complete. Outputs saved to 'analytics/output/'.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise