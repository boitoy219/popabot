
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from tg_scraper import run_scraper
from report_writer import write_markdown_report
import os


def load_new_data(new_path='data/new_messages.csv'):
    df = pd.read_csv(new_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


def load_keywords(keywords_path='keywords/'):
    keywords = set()
    for file in os.listdir(keywords_path):
        with open(os.path.join(keywords_path, file), 'r', encoding='utf-8') as f:
            keywords.update([line.strip().lower() for line in f if line.strip()])
    return keywords


# Optional filtering by keywords (disabled for now)
def filter_data(df, keywords):
    # df = df.dropna(subset=['text'])
    # df['text_lower'] = df['text'].str.lower()
    # return df[df['text_lower'].apply(lambda text: any(k in text for k in keywords))]
    return df


def plot_keyword_mentions(df, save_path='analytics/output/keyword_mentions.png'):
    daily_counts = df.groupby(df['date'].dt.date).size()
    if daily_counts.empty:
        print("⚠️ No keyword spikes found to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(daily_counts.index, daily_counts.values, marker='o')
    plt.title("Keyword Mentions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mentions")
    plt.grid(True)
    plt.xticks(rotation=45)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_topic_modeling(df, save_path='analytics/output/topic_summary.csv'):
    model = BERTopic(embedding_model="paraphrase-MiniLM-L6-v2")
    topics, _ = model.fit_transform(df['text'].astype(str))
    df['topic'] = topics
    summary_df = model.get_topic_info()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    return model


def main():
    run_scraper()
    df = load_new_data()

    if df.empty:
        print("No new messages to analyze.")
        return

    keywords = load_keywords()
    df_filtered = filter_data(df, keywords)

    if df_filtered.empty:
        print("No matching messages after filtering (disabled for now). Proceeding with full data.")
        df_filtered = df  # fallback to all new messages

    plot_keyword_mentions(df_filtered)
    topic_model = run_topic_modeling(df_filtered)
    write_markdown_report(df_filtered, topic_model)
    print("✅ Pipeline complete. Outputs saved to 'analytics/output/'.")


if __name__ == '__main__':
    main()
