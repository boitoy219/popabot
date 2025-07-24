# report_writer.py — Optimized Narrative Summary + Duplicate Consolidation

import os
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
from bertopic import BERTopic
from transformers import pipeline
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

import sys
if sys.version_info >= (3, 13):
    raise RuntimeError("Python 3.13 is not supported. Use Python 3.10 or 3.11 instead.")

try:
    summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", tokenizer="csebuetnlp/mT5_multilingual_XLSum", use_fast=False)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load summarizer model: {e}")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub(r"\s+", " ", text)     # Normalize whitespace
    return text.strip()

def summarize_text_block(text_block, max_tokens=512):
    sentences = sent_tokenize(text_block)
    batches = []
    current_batch = ""
    for sent in sentences:
        if len(current_batch + sent) < max_tokens:
            current_batch += sent + " "
        else:
            batches.append(current_batch.strip())
            current_batch = sent + " "
    if current_batch:
        batches.append(current_batch.strip())

    summaries = []
    for chunk in batches:
        try:
            summary = summarizer(chunk, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"\u26a0\ufe0f Summarization error: {e}")
    return "\n".join(summaries)

def summarize_narratives(df, top_n=10):
    model = BERTopic(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    topics, _ = model.fit_transform(df['text'].astype(str))
    top_topics = model.get_topic_info().head(top_n + 1).iloc[1:]
    return top_topics, model

def detect_propaganda_patterns(df):
    cues = ["nato", "provocation", "biolabs", "nazis", "угроза", "сша", "провокация", "удар", "реакция"]
    propaganda_hits = df['text'].str.lower().apply(lambda t: any(cue in t for cue in cues))
    matches = df[propaganda_hits].copy()
    return len(matches), matches[['text', 'url']]

def summarize_actors(df):
    actors = ["путин", "лукашенко", "нато", "зеленский", "европа", "польша", "сша", "эстония"]
    mentions = Counter()
    for actor in actors:
        mentions[actor] = df['text'].str.lower().str.count(actor).sum()
    return sorted(mentions.items(), key=lambda x: x[1], reverse=True)

def get_date_range(df):
    return df['date'].min().strftime("%B %d, %Y"), df['date'].max().strftime("%B %d, %Y")

def identify_duplicates(df):
    seen = {}
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        key = row['text'].strip()
        url = row['url']
        group = row['group']
        if key in seen:
            grouped[key].append(f"{group} [{url}]")
        else:
            seen[key] = f"{group} [{url}]"
            grouped[key].insert(0, f"{group} [{url}]")
    deduped = [{'text': text, 'sources': grouped[text]} for text in grouped]
    return deduped

def write_markdown_report(df, topic_model, out_path="analytics/output/summary.md"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df['text'] = df['text'].astype(str).apply(clean_text)

    date_str = datetime.now().strftime("%Y-%m-%d")
    start_date, end_date = get_date_range(df)
    message_count = len(df)

    topics_df, _ = summarize_narratives(df)
    propaganda_count, propaganda_examples = detect_propaganda_patterns(df)
    actor_summary = summarize_actors(df)

    deduped_msgs = identify_duplicates(df)
    full_text = "\n".join(msg['text'] for msg in deduped_msgs)[:6000]
    summary_narrative = summarize_text_block(full_text)

    report_lines = []
    report_lines.append(f"# Strategic Telegram Analysis Report — {date_str}\n")
    report_lines.append(f"**Source Dataset:**\n\n{message_count} messages from multiple Russian or pro-Russian Telegram channels\n")
    report_lines.append(f"**Date Range:** {start_date} — {end_date}\n")

    report_lines.append("\n## \ud83e\udde9 Narrative and Messaging Themes\n")
    report_lines.append(summary_narrative + "\n")

    report_lines.append("### \ud83d\udd39 Dominant Topics:\n")
    report_lines.append("| Term | Count | Notes |\n|------|--------|-------|")
    for _, row in topics_df.iterrows():
        term = row['Name'].split(':')[1].strip() if ':' in row['Name'] else row['Name']
        count = row['Count']
        report_lines.append(f"| {term} | {count} |  |")

    report_lines.append("\n## \ud83e\udde0 Information Warfare and Propaganda Indicators\n")
    report_lines.append(f"Detected **{propaganda_count}** potential propaganda-aligned messages.\n")
    for _, row in propaganda_examples.iterrows():
        text = row['text'][:300].replace('\n', ' ').strip()
        link = row['url']
        report_lines.append(f"> {text} ([source]({link}))\n")

    report_lines.append("\n## \u23f1\ufe0f Time-Sensitive Observables\n")
    observables = [
        ("перемещение", "Troop movement"),
        ("мобилизация", "Mobilization references"),
        ("гродно", "Grodno deployments"),
        ("брест", "Brest deployments"),
        ("учения", "Military exercises"),
        ("удар", "Strike implication")
    ]
    for k, note in observables:
        count = df['text'].str.lower().str.count(k).sum()
        if count > 0:
            report_lines.append(f"- **{note}** (`{k}`) mentioned {int(count)} times\n")

    report_lines.append("\n## \ud83d\udc64 Actor Analysis\n")
    report_lines.append("| Actor | Mentions | Notes |\n|--------|----------|-------|")
    for actor, count in actor_summary:
        report_lines.append(f"| {actor} | {int(count)} |  |")

    report_lines.append("\n## \ud83d\udd1a Strategic Takeaways for Intel Community\n")
    for msg in deduped_msgs:
        quote = msg['text'][:300].replace('\n', ' ').strip()
        sources = ", ".join(msg['sources'])
        report_lines.append(f"- {quote} (seen in: {sources})\n")

    if os.path.exists("analytics/output/keyword_mentions.png"):
        report_lines.append("\n## \ud83d\udcc8 Keyword Mention Trends\n")
        report_lines.append("![Keyword Mentions](keyword_mentions.png)\n")

    report_lines.append("\n---\n_Generated by automated pipeline_\n")

    markdown_text = "\n".join(report_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"\ud83d\udcdd Markdown report saved to: {out_path}")
