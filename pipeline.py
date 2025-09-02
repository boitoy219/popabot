# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: your scraper
try:
    from tg_scraper import run_scraper
    SCRAPER_AVAILABLE = True
except Exception:
    SCRAPER_AVAILABLE = False

from report_writer import write_markdown_report, auto_top_keywords

# ---------- Helpers ----------
def _kw_pattern(kw: str) -> str:
    core = re.sub(r"\s+", r"\\s+", re.escape(kw.lower()))
    return rf"\b{core}\b"

def load_new_data(new_path='data/new_messages.csv'):
    if not os.path.exists(new_path):
        raise FileNotFoundError(f"Input CSV not found: {new_path}")
    df = pd.read_csv(new_path)
    must_have = {"message_id", "group", "date", "text", "url"}
    missing = must_have - set(df.columns)
    if missing:
        raise ValueError(f"CSV must include columns: {sorted(missing)}")
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    return df.sort_values('date')

def load_keywords(keywords_path='keywords/'):
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

def _choose_plot_keywords(df: pd.DataFrame, user_keywords, max_k=6):
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

def plot_keyword_mentions(df, keywords, save_path='analytics/output/keyword_mentions.png'):
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
def main():
    use_scraper = os.getenv("POPABOT_USE_SCRAPER", "0") == "1"
    sample_n = int(os.getenv("POPABOT_SAMPLE_N", "0"))
    out_md = os.getenv("POPABOT_OUT_MD", "analytics/output/summary.md")
    trend_png = os.getenv("POPABOT_TREND_PNG", "analytics/output/keyword_mentions.png")

    if use_scraper and SCRAPER_AVAILABLE:
        print("🛰️ Running scraper...")
        try:
            run_scraper()
        except Exception as e:
            print(f"⚠️ Scraper failed (continuing with existing CSV): {e}")

    df = load_new_data('data/new_messages.csv')
    if df.empty:
        print("No new messages to analyze.")
        return

    if sample_n > 0:
        df = df.sort_values("date").tail(sample_n)
        print(f"🔎 Sampling last {sample_n} rows for faster iteration.")

    keywords = load_keywords('keywords/')
    plot_keyword_mentions(df, keywords, save_path=trend_png)

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    write_markdown_report(
        df=df,
        out_path=out_md,
        user_keywords=list(keywords),
        make_cloud=False  # keep off; it wasn’t useful
    )
    print("✅ Pipeline complete. Outputs saved to 'analytics/output/'.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise
