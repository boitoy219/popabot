import os, requests, json
ollama = os.getenv("OLLAMA_HOST","http://127.0.0.1:11434")
r = requests.post(f"{ollama}/api/generate", json={"model":"llama3.1:8b-instruct-q4_K_M","prompt":"Say hi in one short line."}, timeout=120)
print(r.status_code, r.text[:200])