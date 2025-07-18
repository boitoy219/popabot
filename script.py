import os
import csv
from telethon.sync import TelegramClient
from telethon.tl.functions.contacts import ResolveUsernameRequest
from dotenv import load_dotenv

# Load .env
load_dotenv()
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
session_name = os.getenv("SESSION_NAME", "session")

# Load keywords
with open("keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip().lower() for line in f if line.strip()]

# Load groups
with open("groups.txt", "r", encoding="utf-8") as f:
    group_identifiers = [line.strip() for line in f if line.strip()]

# Output CSV
output_file = "matched_messages.csv"
csv_headers = ["group", "date", "text", "url"]

# Create file with headers if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

def match_keywords(text):
    return any(k in text.lower() for k in keywords)

with TelegramClient(session_name, api_id, api_hash) as client:
    for group in group_identifiers:
        try:
            entity = client.get_entity(group)
            group_name = entity.username or entity.title or str(entity.id)

            print(f"🔍 Reading messages from: {group_name}")

            for message in client.iter_messages(entity, limit=1000):  # Adjust limit as needed
                if message.text and match_keywords(message.text):
                    # Construct URL if possible
                    if hasattr(entity, 'username') and entity.username:
                        url = f"https://t.me/{entity.username}/{message.id}"
                    else:
                        url = "(private group - no URL)"

                    print(f"\n📌 Match in {group_name}")
                    print(f"🕒 {message.date}")
                    print(f"🔗 {url}")
                    print(f"✉️ {message.text[:300]}...")

                    # Append to CSV
                    with open(output_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([group_name, message.date, message.text.strip(), url])
        except Exception as e:
            print(f"⚠️ Error in group '{group}': {e}")
