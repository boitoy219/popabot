import os, json, sys
import requests

HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
LLM   = os.getenv("POPABOT_LLM", "").lower()

print("[diag] POPABOT_LLM =", LLM or "<unset>")
print("[diag] OLLAMA_HOST  =", HOST)
print("[diag] OLLAMA_MODEL =", MODEL)

if LLM not in {"ollama", "on"}:
    print("[diag] FAIL: POPABOT_LLM not set to 'ollama' or 'on'")
    sys.exit(2)

try:
    r = requests.get(f"{HOST}/api/tags", timeout=5)
    print("[diag] /api/tags status:", r.status_code)
    print("[diag] /api/tags body:", r.text[:300], "...")
    r.raise_for_status()
    tags = r.json()
except Exception as e:
    print("[diag] FAIL: cannot reach Ollama:", e)
    sys.exit(3)

have_model = any((m.get("name") == MODEL) for m in tags.get("models", []))
print("[diag] model present:", have_model)

try:
    payload = {"model": MODEL, "prompt": "say 'ok'", "stream": False}
    r = requests.post(f"{HOST}/api/generate", json=payload, timeout=30)
    print("[diag] /api/generate status:", r.status_code)
    print("[diag] /api/generate body:", r.text[:300], "...")
    r.raise_for_status()
    j = r.json()
    print("[diag] response:", j.get("response", "")[:120])
except Exception as e:
    print("[diag] FAIL: generate call failed:", e)
    sys.exit(4)

print("[diag] OK ✅")
