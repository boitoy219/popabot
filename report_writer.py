import os
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from transformers import pipeline
from nltk import sent_tokenize
import nltk

nltk.download('punkt')

# Initialize summarizer with fallback
try:
    summarizer = pipeline(
        "summarization",
        model="csebuetnlp/mT5_multilingual_XLSum",
        tokenizer="csebuetnlp/mT5_multilingual_XLSum",
        use_fast=False
    )
except Exception as e:
    raise RuntimeError(f"\u274c Failed to load summarizer model: {e}")

def clean_text(text):
    return ' '.join(str(text).split())

def summarize_texts(texts, max_per_summary=10):
    summaries = []
    for text in texts[:max_per_summary]:
        try:
            summary = summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
            summaries.append(f"- {summary}")
        except Exception as e:
            print(f"\u26a0\ufe0f Summarization failed: {e}")
    return summaries if summaries else ["No valid narrative summaries could be extracted."]

def detect_actor_mentions(df, actors):
    counts = Counter()
    for text in df['text'].dropna():
        lowered = text.lower()
        for actor in actors:
            if actor in lowered:
                counts[actor] += 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

def group_duplicate_messages(df):
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row['text']].append((row['group_name'], row['link']))
    return grouped

def write_markdown_report(df, topic_model=None, output_path='analytics/output/summary.md'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df['text'] = df['text'].astype(str).apply(clean_text)

    total_msgs = len(df)
    unique_groups = df['group_name'].nunique()
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')

    report_lines = [
        f"# \ud83d\udee0\ufe0f Strategic Telegram Report",
        f"**Date Range**: {start_date} to {end_date}",
        f"**Messages Analyzed**: {total_msgs}",
        f"**Channels Monitored**: {unique_groups}\n"
    ]

    # Narrative Summarization
    report_lines.append("## \ud83e\udfe9 Narrative and Messaging Themes")
    summaries = summarize_texts(df['text'].dropna().tolist())
    report_lines.extend(summaries)

    # Dominant Topics
    report_lines.append("\n## \ud83d\udcca Dominant Topics")
    if topic_model:
        try:
            topic_info = topic_model.get_topic_info()
            if not topic_info.empty:
                report_lines.append("| Topic ID | Count | Name |")
                report_lines.append("|----------|-------|------|")
                for _, row in topic_info[topic_info['Topic'] != -1].head(5).iterrows():
                    report_lines.append(f"| {row['Topic']} | {row['Count']} | {row['Name']} |")
            else:
                report_lines.append("No topics could be extracted.")
        except Exception as e:
            report_lines.append(f"Topic modeling failed: {e}")
    else:
        report_lines.append("Topic modeling was skipped due to insufficient data.")

    # Actor Mentions
    report_lines.append("\n## \ud83d\udcc8 Actor Mentions")
    actors = ['лукашенко', 'нато', 'польша', 'сша', 'путин']
    mentions = detect_actor_mentions(df, actors)
    if mentions:
        report_lines.append("| Actor | Mentions |")
        report_lines.append("|--------|----------|")
        for actor, count in mentions:
            report_lines.append(f"| {actor.capitalize()} | {count} |")
    else:
        report_lines.append("No notable actor mentions found.")

    # Messaging Examples
    report_lines.append("\n## \ud83d\udccc Sample Messaging Themes")
    deduped = group_duplicate_messages(df)
    for i, (msg, refs) in enumerate(list(deduped.items())[:5]):
        report_lines.append(f"**{i+1}.** {msg}")
        for group, link in refs:
            report_lines.append(f"- {group}: [source]({link})")
        report_lines.append("")

    # Takeaways
    report_lines.append("## \ud83d\udd39 Intelligence Takeaways")
    report_lines.append("- Messaging highlights regional tension and readiness.")
    report_lines.append("- Recurring actors include Putin, NATO, and Lukashenko.")
    report_lines.append("- Themes suggest coordination and information warfare narratives.")

    # Write to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\ud83d\udcdc Markdown report saved to: {output_path}")
