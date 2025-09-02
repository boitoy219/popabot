# -*- coding: utf-8 -*-
"""
llm_report_writer.py — LLM-first report writer (CPU-friendly, GH Actions-safe)

- Prefers local Ollama LLM via llm_writer.py to generate analytic sections.
- Graceful fallback to heuristic summaries if the LLM isn’t available.
- No wordcloud (removed by design).
- Includes keyword trend plot if pipeline created it.

Backwards compatible with old pipeline signature:
  write_markdown_report(df, topic_model, keywords)
and with the refactored one:
  write_markdown_report(df, user_keywords=keywords)
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

# Light NLP
import nltk
from nltk.tokenize import sent_tokenize

# Heuristic summarization fallback
try:
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    SUMY_OK = True
except Exception:
    SUMY_OK = False

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional near-duplicate merge
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# LLM helper
try:
    from llm_writer import try_llm_analytic_report, enrich_stats_for_llm
except Exception:
    # If llm_writer isn’t present, we’ll just skip LLM use
    try_llm_analytic_report = None
    enrich_stats_for_llm = None

# ---------- Version & NLTK ----------
if sys.version_info >= (3, 13):
    raise RuntimeError("Python 3.13+ not supported here. Use Python 3.10 or 3.11.")
nltk.download("punkt", quiet=True)

# ---------- Stopwords ----------
RU_SW = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да",
    "ты","к","у","же","вы","за","бы","по","ее","мне","было","вот","от","меня","еще","нет","о","из",
    "ему","теперь","когда","даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него",
    "до","вас","нибудь","опять","уж","вам","ведь","там","потом","себя","ничего","ей","может","они",
    "тут","где","есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто",
    "чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому","этого","какой",
    "совсем","ним","здесь","этом","один","почти","мой","тем","чтобы","нее","кажется","сейчас","были",
    "куда","зачем","всех","никогда","можно","при","наконец","два","об","другой","хоть","после",
    "над","больше","тот","через","эти","нас","про","всего","них","какая","много","разве","три","эту",
    "моя","впрочем","хорошо","свою","этой","перед","иногда","лучше","чуть","том","нельзя","такой",
    "им","более","всегда","конечно","всю","между","лишь"
}
EN_SW = {
    "the","a","an","and","or","but","if","to","of","for","in","on","at","by","from","with","as","is","are",
    "was","were","be","been","being","it","its","this","that","these","those","there","their","we","you",
    "he","she","they","i","me","my","our","your","his","her","their","them","us","not","no","yes","do","does",
    "did","so","than","then","also","such","into","over","under","out","up","down","about","more","most","any",
    "all","some","many","few"
}
GEN_SW = {
    "—","–","-","•","…","—","–","—","️","►","▪","🔴","⚡️","⭐️","🟢","🔸","🔹","#","@","http","https","t.me",
    "telegram","📣","📌","📍","📱","👉","🌟","💬","✉️","📩","📨","📧","💌","📥","🎼","🌐","__","**"
}
STOPWORDS = RU_SW | EN_SW | GEN_SW

# ---------- Cleaning ----------
URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WS_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001F6FF" "\U0001F1E0-\U0001F1FF" "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF" "\U00002600-\U000026FF" "]+", flags=re.UNICODE
)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\u200b", " ")
    s = MD_LINK_RE.sub(r"\1", s)
    s = URL_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s.encode("utf-8", "ignore").decode("utf-8")

def normalize_for_hash(s: str) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = WS_RE.sub(" ", s).strip()
    return s

def sent_split(text: str) -> list:
    try:
        return sent_tokenize(text, language="russian")
    except Exception:
        try:
            return sent_tokenize(text, language="english")
        except Exception:
            return re.split(r"(?<=[.!?])\s+", text)

# ---------- Summarization Fallback ----------
def textrank_summary(text: str, n_sentences: int = 7, lang: str = "russian") -> str:
    text = clean_text(text)
    if not text:
        return ""
    if SUMY_OK:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(lang))
        except Exception:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        max_sents = max(1, min(n_sentences, len(parser.document.sentences)))
        summary = summarizer(parser.document, max_sents)
        return " ".join(str(s) for s in summary).strip()
    # very light fallback: take first N high-information sentences
    sents = sent_split(text)[:max(1, n_sentences)]
    return " ".join(sents).strip()

# ---------- TF-IDF ----------
def extract_tfidf_terms(texts, top_n=14, ngram_range=(1,2), min_df=2, max_features=6000):
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w\-]+\b",
        lowercase=True,
        stop_words=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).ravel()
    items = list(zip(terms, scores))
    def keep(term):
        parts = term.split()
        if any((p in STOPWORDS) or (len(p) <= 2) for p in parts):
            return False
        return True
    items = [(t, s) for (t, s) in items if keep(t)]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]

# ---------- Duplicates ----------
def identify_duplicates(df, use_fuzzy=True, fuzzy_threshold=94):
    buckets: Dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        buckets[normalize_for_hash(row["text"])].append(row)

    clusters = []
    for _, rows in buckets.items():
        texts = [clean_text(r["text"]) for r in rows]
        canonical = max(texts, key=len)
        srcs = [f'{r["group"]} [{r["url"]}]' for r in rows]
        clusters.append({"text": canonical, "sources": list(dict.fromkeys(srcs))})

    if use_fuzzy and RAPIDFUZZ_OK and len(clusters) > 1:
        merged = []
        used = set()
        for i in range(len(clusters)):
            if i in used:
                continue
            base = clusters[i]
            for j in range(i+1, len(clusters)):
                if j in used:
                    continue
                score = fuzz.token_set_ratio(base["text"], clusters[j]["text"])
                if score >= fuzzy_threshold:
                    base["sources"].extend(clusters[j]["sources"])
                    base["sources"] = list(dict.fromkeys(base["sources"]))
                    if len(clusters[j]["text"]) > len(base["text"]):
                        base["text"] = clusters[j]["text"]
                    used.add(j)
            merged.append(base)
        clusters = merged
    return clusters

# ---------- Keywords ----------
def count_keywords(df, keywords):
    out = {}
    texts = df["text"].astype(str).str.lower()
    for kw in keywords:
        pattern = rf"\b{re.escape(str(kw).lower())}\b"
        out[kw] = int(texts.str.count(pattern).sum())
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))

def auto_top_keywords(df, top_n=35):
    counts = Counter()
    for s in df["text"].astype(str):
        s = clean_text(s).lower()
        tokens = re.findall(r"(?u)\b\w[\w\-]+\b", s)
        for t in tokens:
            if (t in STOPWORDS) or (len(t) <= 2):
                continue
            counts[t] += 1
    return counts.most_common(top_n)

# ---------- Propaganda ----------
PROP_CUES = [
    "nato","provocation","biolab","nazis","aggressor","strike","escalation","false flag",
    "disinformation","psychological","mobilization","hybrid",
    "нато","провокац","биолаб","наци","агресс","удар","эскалац","ложн","дезинформ","психолог",
    "мобилизац","гибридн"
]
def detect_propaganda_patterns(df):
    def hit(text):
        t = text.lower()
        return any(cue in t for cue in PROP_CUES)
    mask = df["text"].astype(str).apply(hit)
    matches = df[mask].copy()
    return int(matches.shape[0]), matches[["text","url","group"]]

# ---------- Actors ----------
DEFAULT_ACTORS = [
    "путин","лукашенко","нато","зеленский","европа","польша","литва","латвия","эстония",
    "сша","одкб","шос","ес","украина","киев","москва","минск","калининград","гродно","брест",
    "ревенко","хренин","муравейко","сердюков","кадыров","мид","минобороны"
]
def summarize_actors(df, actors=DEFAULT_ACTORS):
    texts = df["text"].astype(str).str.lower()
    res = []
    for a in actors:
        res.append((a, int(texts.str.count(rf"\b{re.escape(a)}\b").sum())))
    res.sort(key=lambda x: x[1], reverse=True)
    return res

# ---------- Observables ----------
OBSERVABLES = [
    ("перемещ",  "Troop movement"),
    ("мобилизац","Mobilization references"),
    ("гродно",   "Grodno deployments"),
    ("брест",    "Brest deployments"),
    ("сувалк",   "Suwalki corridor"),
    ("учени",    "Military exercises"),
    ("удар",     "Strike implication"),
    ("ядер",     "Nuclear context"),
    ("орешник",  "Oreshnik system"),
    ("quadriga", "Quadriga-2025 (NATO)"),
    ("żelazny",  "Żelazny Obrońca (PL)"),
    ("namejs",   "Namejs-2025 (LV)"),
    ("iron",     "Iron Spear / Shield (LT)"),
]
def tally_observables(df):
    obs_lines = []
    texts = df["text"].astype(str).str.lower()
    for k, note in OBSERVABLES:
        count = int(texts.str.count(k).sum())
        if count > 0:
            obs_lines.append((note, k, count))
    return obs_lines

# ---------- Core Writer ----------
def write_markdown_report(df: pd.DataFrame, *args, out_path: str = "analytics/output/summary.md",
                          user_keywords: Any = None, **kwargs):
    """
    Compatible with both:
      write_markdown_report(df, topic_model, keywords)
      write_markdown_report(df, user_keywords=keywords)

    We ignore 'topic_model' (no BERTopic here).
    """
    # Back-compat: infer user_keywords from positional args if needed
    if user_keywords is None:
        # Old: (df, topic_model, keywords)
        if len(args) >= 2 and isinstance(args[1], (list, set, tuple, dict)):
            user_keywords = args[1]
        # Sometimes pipeline may pass only keywords as 1st arg
        elif len(args) == 1 and isinstance(args[0], (list, set, tuple, dict)):
            user_keywords = args[0]
        else:
            user_keywords = []
    # Normalize KW container
    if isinstance(user_keywords, dict):
        # Already a dict of counts? keep keys as the set we’ll count anyway
        user_keywords = list(user_keywords.keys())
    elif isinstance(user_keywords, (set, tuple)):
        user_keywords = list(user_keywords)
    elif not isinstance(user_keywords, list):
        user_keywords = []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Validate & clean df
    for col in ("text","url","group","date"):
        if col not in df.columns:
            raise ValueError(f"Input DataFrame missing required column: {col}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["group"] = df["group"].astype(str)
    df["url"] = df["url"].astype(str)

    # Basic stats
    message_count = len(df)
    date_min = df["date"].min()
    date_max = df["date"].max()
    start_date = date_min.strftime("%B %d, %Y") if pd.notna(date_min) else "N/A"
    end_date = date_max.strftime("%B %d, %Y") if pd.notna(date_max) else "N/A"
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Deduplicate clusters
    deduped_msgs = identify_duplicates(df, use_fuzzy=True, fuzzy_threshold=94)
    dedup_text = "\n".join(m["text"] for m in deduped_msgs)

    # Heuristic artifacts
    tfidf_terms = extract_tfidf_terms(df["text"].tolist(), top_n=14)
    propaganda_count, propaganda_examples = detect_propaganda_patterns(df)
    actor_summary = summarize_actors(df)
    observables = tally_observables(df)
    auto_keywords = auto_top_keywords(df, top_n=35)
    keyword_counts = count_keywords(df, user_keywords) if user_keywords else {}

    # Try LLM analytics
    llm_sections = {}
    if try_llm_analytic_report and enrich_stats_for_llm:
        stats_for_llm = enrich_stats_for_llm(
            df,
            tfidf_terms=tfidf_terms,
            auto_keywords=auto_keywords,
            user_keywords=keyword_counts,
            observables=observables,
            actor_summary=actor_summary,
            propaganda_examples=propaganda_examples,
            deduped_msgs=deduped_msgs,
        )
        llm_sections = try_llm_analytic_report(
            df,
            stats=stats_for_llm,
            temperature=0.2,
            max_context_chars=12000,
            lang_hint="auto",
        ) or {}

    # Fallback executive summary if LLM missing
    if not llm_sections.get("exec_summary"):
        llm_sections["exec_summary"] = textrank_summary(dedup_text[:120000], n_sentences=8, lang="russian")

    # ---- Build Markdown ----
    lines: List[str] = []
    lines.append(f"# CIA-Style Intelligence Report — {date_str}\n")
    lines.append(f"**Source Dataset:** {message_count} Telegram messages\n")
    lines.append(f"**Date Range:** {start_date} — {end_date}\n")

    # Executive / LLM analytics
    lines.append("\n## 🧭 Executive Summary\n")
    lines.append((llm_sections.get("exec_summary") or "_No summary available._") + "\n")

    def maybe_section(title_md: str, key: str):
        val = llm_sections.get(key, "").strip()
        if val:
            lines.append(title_md + "\n")
            lines.append(val + "\n")

    maybe_section("## 🧵 Narrative Assessment", "narrative_assessment")
    maybe_section("## 🧠 Propaganda / Influence Ops Assessment", "propaganda_assessment")
    maybe_section("## ⚠️ Risk Outlook (7–21 days)", "risk_outlook")
    maybe_section("## 🎯 Likely Adversary Objectives", "adversary_objectives")
    maybe_section("## 🕳️ Collection Gaps & Next Tasks", "collection_gaps")
    maybe_section("## 📝 Method Notes", "method_notes")

    # Dominant terms
    lines.append("## 🔹 Dominant Terms (TF-IDF)\n")
    lines.append("| Term | Weight |\n|---|---|")
    for term, score in tfidf_terms:
        lines.append(f"| {term} | {round(float(score), 2)} |")

    # User keywords (if any)
    if user_keywords:
        lines.append("\n## 🔑 Tracked Keywords (User-Defined)\n")
        lines.append("| Keyword | Mentions |\n|---|---|")
        for kw, cnt in keyword_counts.items():
            lines.append(f"| {kw} | {cnt} |")

    # Auto “repetitive & important” keywords
    lines.append("\n## ♻️ Repetitive High-Signal Keywords (Auto)\n")
    lines.append("| Keyword | Mentions |\n|---|---|")
    for kw, cnt in auto_keywords:
        lines.append(f"| {kw} | {cnt} |")

    # Trend image (from pipeline)
    trend_path = os.path.join(os.path.dirname(out_path), "keyword_mentions.png")
    if os.path.exists(trend_path):
        lines.append("\n## 📊 Keyword Mention Trends\n")
        lines.append(f"![Keyword Mentions]({os.path.basename(trend_path)})\n")

    # Propaganda indicators
    lines.append("\n## 🧠 Information Warfare / Propaganda Indicators\n")
    lines.append(f"Detected **{propaganda_count}** messages containing propaganda cues.\n")
    if propaganda_count > 0:
        examples = propaganda_examples.head(10).itertuples(index=False)
        for row in examples:
            snippet = clean_text(getattr(row, "text", ""))[:300].replace("\n", " ").strip()
            link = getattr(row, "url", "")
            lines.append(f"> {snippet} ([source]({link}))")

    # Observables
    if observables:
        lines.append("\n## ⏱️ Time-Sensitive Observables\n")
        for note, key, count in observables:
            lines.append(f"- **{note}** (`{key}`) mentioned **{count}** times")
    else:
        lines.append("\n## ⏱️ Time-Sensitive Observables\n_No salient observables detected._")

    # Actor mentions
    lines.append("\n## 👤 Actor Mentions\n")
    lines.append("| Actor | Mentions |\n|---|---|")
    for actor, cnt in actor_summary:
        lines.append(f"| {actor} | {cnt} |")

    # Cross-posted items (dedup clusters)
    lines.append("\n## 🔎 Source-Corroborated Items (Cross-posted / Duplicates)\n")
    for msg in deduped_msgs[:25]:
        quote = msg["text"][:280].replace("\n", " ").strip()
        sources = ", ".join(msg["sources"][:6])
        if len(msg["sources"]) > 6:
            sources += ", …"
        lines.append(f"- {quote} _(seen in: {sources})_")

    lines.append("\n---\n_Generated by LLM-first pipeline (with CPU fallbacks)_\n")

    md = "\n".join(lines)
    md = md.encode("utf-8", "ignore").decode("utf-8")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"📝 Markdown report saved to: {out_path}")

# ---- Optional CLI (for ad-hoc runs) ----
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"message_id","group","date","text","url"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df

def parse_keywords(arg: str):
    if not arg:
        return []
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LLM-first Telegram report builder")
    p.add_argument("--input", "-i", required=True, help="CSV with columns: message_id,group,date,text,url")
    p.add_argument("--output", "-o", default="analytics/output/summary.md", help="Markdown output path")
    p.add_argument("--keywords", "-k", default="", help="Comma-separated keywords OR path to a .txt list")
    args = p.parse_args()
    df = load_csv(args.input)
    kws = parse_keywords(args.keywords)
    write_markdown_report(df, user_keywords=kws, out_path=args.output)
