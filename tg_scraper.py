import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from telethon.sync import TelegramClient
import pandas as pd


def run_scraper(env_path=".env", group_dir="groups/", keyword_dir="keywords/", run_dir="data/runs/"):
    load_dotenv(env_path)
    api_id = int(os.getenv("API_ID"))
    api_hash = os.getenv("API_HASH")
    session_name = os.getenv("SESSION_NAME", "session")

    today = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(run_dir, f"matched_messages_{today}.csv")
    latest_file = "data/latest_combined.csv"
    new_file = "data/new_messages.csv"

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load keywords
    keywords = set()
    for file in os.listdir(keyword_dir):
        with open(os.path.join(keyword_dir, file), 'r', encoding='utf-8') as f:
            keywords.update([line.strip().lower() for line in f if line.strip()])

    # Load groups
    group_identifiers = []
    for file in os.listdir(group_dir):
        with open(os.path.join(group_dir, file), 'r', encoding='utf-8') as f:
            group_identifiers.extend([line.strip() for line in f if line.strip()])

    def match_keywords(text):
        return any(k in text.lower() for k in keywords)

    rows = []
    with TelegramClient(session_name, api_id, api_hash) as client:
        for group in group_identifiers:
            try:
                entity = client.get_entity(group)
                group_name = entity.username or entity.title or str(entity.id)
                print(f"🔍 Reading messages from: {group_name}")

                for message in client.iter_messages(entity, limit=500):
                    if message.text and match_keywords(message.text):
                        message_id = f"{group_name}_{message.id}"
                        url = f"https://t.me/{entity.username}/{message.id}" if getattr(entity, 'username', None) else "(private group - no URL)"
                        rows.append([message_id, group_name, message.date, message.text.strip(), url])
            except Exception as e:
                print(f"⚠️ Error in group '{group}': {e}")

    # Save today's run
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message_id", "group", "date", "text", "url"])
        writer.writerows(rows)

    # Deduplicate vs prior runs
    new_df = pd.read_csv(output_file)
    if os.path.exists(latest_file):
        prev_df = pd.read_csv(latest_file)
        combined_df = pd.concat([prev_df, new_df]).drop_duplicates(subset="message_id", keep="first")
        new_only = combined_df[~combined_df["message_id"].isin(prev_df["message_id"])]
    else:
        combined_df = new_df
        new_only = new_df

    combined_df.to_csv(latest_file, index=False)
    new_only.to_csv(new_file, index=False)
    print(f"✅ Scrape complete. New messages: {len(new_only)}")


if __name__ == "__main__":
    run_scraper()
