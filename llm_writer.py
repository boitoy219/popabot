# -*- coding: utf-8 -*-
"""
llm_writer.py — Local LLM-assisted analytic sections with robust JSON fallbacks.

Strategy:
  A) Single-shot JSON with schema (format:'json').
  B) If invalid, run a "repair" pass.
  C) If still invalid OR LLM_FORCE_MICRO=1, fetch sections via per-field JSON micro-calls and assemble.

Env:
  OLLAMA_HOST            (default http://127.0.0.1:11434)
  OLLAMA_MODEL / POPABOT_LLM
  OLLAMA_TIMEOUT         (default 600)
  LLM_CTX                (default 2048)
  LLM_MAX_TOKENS         (default 450)
  LLM_MAX_CONTEXT_CHARS  (default 6000)
  LLM_TEMPERATURE        (default 0.2)
  LLM_LOG_LEVEL          (default INFO)
  LLM_RETRIES            (default 2)
  LLM_FORCE_MICRO        (default 0)  # set to 1 to skip A/B and go straight to micro-calls
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

try:
    import requests
except Exception:
    requests = None


# ----------------------- Env helpers -----------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


# ----------------------- Logging -----------------------
LOG = logging.getLogger("llm_writer")
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[llm_writer] %(levelname)s: %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(getattr(logging, os.environ.get("LLM_LOG_LEVEL", "INFO").upper(), logging.INFO))


# ----------------------- Utilities -----------------------
def _is_cyrillic_heavy(s: str) -> bool:
    if not s:
        return False
    cyr = sum(1 for ch in s if "\u0400" <= ch <= "\u04FF")
    return (cyr / max(1, len(s))) > 0.3

def detect_language(df: pd.DataFrame, hint: str = "auto") -> str:
    if hint and hint.lower() in ("ru", "russian"):
        return "ru"
    if hint and hint.lower() in ("en", "english"):
        return "en"
    sample = " ".join(df["text"].astype(str).head(50).tolist())
    return "ru" if _is_cyrillic_heavy(sample) else "en"

def truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)].rstrip() + "..."

def compact(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.replace("\u200b", " ").strip()

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _preview(label: str, s: str, n=260):
    if LOG.level <= logging.DEBUG:
        msg = (s or "")[:n].replace("\n", " ")
        LOG.debug("%s: %s", label, msg if msg else "<empty>")


# ----------------------- Ollama client -----------------------
def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

def _ollama_model() -> Optional[str]:
    m = os.environ.get("OLLAMA_MODEL") or os.environ.get("POPABOT_LLM")
    if m and m.strip().lower() == "ollama":
        return None
    return m

def _ollama_available(timeout=3.0) -> bool:
    if requests is None:
        return False
    host = _ollama_host()
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        return bool(r and r.ok)
    except Exception:
        return False

def _ollama_chat(
    messages,
    model: str,
    *,
    options: Optional[dict] = None,
    response_format: Optional[str] = None,
    timeout: int = 120
) -> str:
    if requests is None:
        raise RuntimeError("requests not installed; cannot call Ollama.")
    host = _ollama_host()
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    if response_format:
        payload["format"] = response_format

    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    try:
        r.raise_for_status()
    except Exception as e:
        body = r.text[:400] if hasattr(r, "text") else "<no body>"
        raise RuntimeError(f"Ollama /api/chat HTTP {getattr(r, 'status_code', '?')}: {e}\nBody: {body}")
    data = r.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    return content

def _ollama_warmup(model: str, timeout: int = 45) -> None:
    try:
        _ = _ollama_chat(
            messages=[{"role": "user", "content": "Return OK."}],
            model=model,
            options={"num_predict": 2},
            response_format=None,
            timeout=timeout,
        )
    except Exception:
        pass


# ----------------------- Prompt & Schema -----------------------
SYS_RU = (
    "Вы — аналитик разведсообщений (OSINT). Пишите сдержанно, аналитично, без лозунгов.\n"
    "ОТВЕЧАЙТЕ ТОЛЬКО СТРОГИМ JSON без Markdown.\n"
    "Строгий JSON-требования: все ключи и строки в двойных кавычках; никаких одинарных кавычек для ключей; без запятых на конце; без комментариев.\n"
    "Если в ЗНАЧЕНИЯХ нужны кавычки, используйте одинарные кавычки внутри текста или экранируйте двойные как \\\".\n"
    "Пишите на русском.\n"
)
SYS_EN = (
    "You are an OSINT analyst. Be concise, analytical, decision-focused.\n"
    "Reply with STRICT JSON only (no Markdown).\n"
    "Strict JSON rules: all keys and strings use double quotes; never single-quote keys; no trailing commas; no comments.\n"
    "If a VALUE needs quotes, prefer single quotes inside or escape double quotes as \\\".\n"
    "Write in English.\n"
)

TEMPLATE = """
CONTEXT:
- Time window: {date_range}
- Total messages: {n_msgs}

Dominant terms (TF-IDF):
{top_terms}

User-tracked keywords (counts):
{user_kws}

Auto high-signal keywords (counts):
{auto_kws}

Actors (top):
{actors}

Time-sensitive observables (count | token):
{observables}

Cross-posted items (evidence of corroboration):
{cross_posts}

Propaganda-like messages (sampled):
{prop_samples}

TASK:
Produce a JSON object with these keys:
- "exec_summary": 4-8 sentences; focus on what matters now; avoid slogans.
- "narrative_assessment": 5-10 bullets; main narratives; proponents; convergence/divergence.
- "propaganda_assessment": 3-6 bullets; techniques, objectives, effects, indicators.
- "risk_outlook": 3-6 bullets; 7–21 day outlook; include Low/Med/High likelihood markers.
- "adversary_objectives": 3-6 bullets; inferred goals and likely courses of action.
- "collection_gaps": 3-6 bullets; unknowns + specific next collection tasks.
- "method_notes": 2-4 bullets; caveats/limits of the data.

IMPORTANT:
- Valid strict JSON parsable by Python json.loads.
- No markdown; plain text values.
- Keep each bullet short (<= 24 words).
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "exec_summary",
        "narrative_assessment",
        "propaganda_assessment",
        "risk_outlook",
        "adversary_objectives",
        "collection_gaps",
        "method_notes",
    ],
    "properties": {
        "exec_summary": {"type": "string"},
        "narrative_assessment": {"type": "array", "items": {"type": "string"}},
        "propaganda_assessment": {"type": "array", "items": {"type": "string"}},
        "risk_outlook": {"type": "array", "items": {"type": "string"}},
        "adversary_objectives": {"type": "array", "items": {"type": "string"}},
        "collection_gaps": {"type": "array", "items": {"type": "string"}},
        "method_notes": {"type": "array", "items": {"type": "string"}},
    },
}

FIELD_SCHEMAS = {
    "exec_summary": {"type": "object", "required": ["exec_summary"], "properties": {"exec_summary": {"type": "string"}}},
    "narrative_assessment": {
        "type": "object",
        "required": ["narrative_assessment"],
        "properties": {"narrative_assessment": {"type": "array", "items": {"type": "string"}}},
    },
    "propaganda_assessment": {
        "type": "object",
        "required": ["propaganda_assessment"],
        "properties": {"propaganda_assessment": {"type": "array", "items": {"type": "string"}}},
    },
    "risk_outlook": {
        "type": "object",
        "required": ["risk_outlook"],
        "properties": {"risk_outlook": {"type": "array", "items": {"type": "string"}}},
    },
    "adversary_objectives": {
        "type": "object",
        "required": ["adversary_objectives"],
        "properties": {"adversary_objectives": {"type": "array", "items": {"type": "string"}}},
    },
    "collection_gaps": {
        "type": "object",
        "required": ["collection_gaps"],
        "properties": {"collection_gaps": {"type": "array", "items": {"type": "string"}}},
    },
    "method_notes": {
        "type": "object",
        "required": ["method_notes"],
        "properties": {"method_notes": {"type": "array", "items": {"type": "string"}}},
    },
}


def _fmt_terms(terms: List[Tuple[str, float]], k: int = 12) -> str:
    out = []
    for t, s in (terms or [])[:k]:
        out.append(f"- {t} ({round(float(s), 2)})")
    return "\n".join(out) or "- (none)"

def _fmt_counts(counts: Dict[str, int], k: int = 12) -> str:
    if not counts:
        return "- (none)"
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return "\n".join(f"- {w}: {c}" for w, c in items) or "- (none)"

def _fmt_pairs(pairs: List[Tuple[str, int]], k: int = 14) -> str:
    out = []
    for w, c in (pairs or [])[:k]:
        out.append(f"- {w}: {c}")
    return "\n".join(out) or "- (none)"

def _fmt_observables(obss: List[Tuple[str, str, int]], k: int = 10) -> str:
    out = []
    for note, key, cnt in (obss or [])[:k]:
        out.append(f"- {cnt} | {key} | {note}")
    return "\n".join(out) or "- (none)"

def _fmt_actors(actors: List[Tuple[str, int]], k: int = 12) -> str:
    out = []
    for a, c in (actors or [])[:k]:
        out.append(f"- {a}: {c}")
    return "\n".join(out) or "- (none)"

def _fmt_cross_posts(deduped: List[Dict[str, Any]], k: int = 6) -> str:
    out = []
    for item in (deduped or [])[:k]:
        text = truncate(compact(item.get("text", "")), 280)
        srcs = ", ".join((item.get("sources") or [])[:4])
        out.append(f"- {text} | sources: {srcs}")
    return "\n".join(out) or "- (none)"

def _fmt_prop_samples(df: pd.DataFrame, k: int = 6) -> str:
    if df is None or df.empty:
        return "- (none)"
    out = []
    for row in df.head(k).itertuples(index=False):
        snippet = truncate(compact(getattr(row, "text", "")), 240)
        grp = getattr(row, "group", "")
        out.append(f"- {snippet} | ch:{grp}")
    return "\n".join(out) or "- (none)"


# ----------------------- JSON helpers -----------------------
def _extract_json_block(text: str) -> Optional[dict]:
    if not text:
        return None
    text = _strip_code_fences(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = candidate.replace("\t", " ").replace("\r", " ")
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def _bullets_to_str(val: Any) -> str:
    if isinstance(val, list):
        return "\n".join(f"- {compact(str(x))}" for x in val if str(x).strip())
    if isinstance(val, dict):
        return "\n".join(f"- {k}: {compact(str(v))}" for k, v in val.items())
    if val is None:
        return ""
    return compact(str(val))


# ----------------------- Core public API -----------------------
def try_llm_analytic_report(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    *,
    temperature: float = 0.2,
    max_context_chars: int = 12000,
    lang_hint: str = "auto",
    timeout: int = None
) -> Dict[str, str]:

    model = _ollama_model()
    max_context_chars = _env_int("LLM_MAX_CONTEXT_CHARS", 4900)
    num_predict       = _env_int("LLM_MAX_TOKENS", 1300)
    num_ctx           = _env_int("LLM_CTX", 2048)
    temperature       = _env_float("LLM_TEMPERATURE", temperature)
    timeout           = timeout or _env_int("OLLAMA_TIMEOUT", 600)
    retries           = _env_int("LLM_RETRIES", 1)
    force_micro       = bool(int(os.environ.get("LLM_FORCE_MICRO", "0")))

    if not model or not _ollama_available():
        LOG.info("Local LLM not available (missing requests/model/host). Returning empty sections.")
        return {}

    lang = detect_language(df, lang_hint)
    system_prompt = SYS_RU if lang == "ru" else SYS_EN

    # Dates
    date_min = stats.get("date_min")
    date_max = stats.get("date_max")
    date_range = "N/A"
    try:
        if pd.notna(date_min) and pd.notna(date_max):
            date_range = f"{pd.to_datetime(date_min).strftime('%b %d, %Y')} — {pd.to_datetime(date_max).strftime('%b %d, %Y')}"
    except Exception:
        pass

    # Context
    user_prompt = TEMPLATE.format(
        date_range=date_range,
        n_msgs=int(df.shape[0]),
        top_terms=_fmt_terms(stats.get("tfidf_terms") or [], k=12),
        user_kws=_fmt_counts(stats.get("user_keywords") or {}, k=12),
        auto_kws=_fmt_pairs(stats.get("auto_keywords") or [], k=14),
        actors=_fmt_actors(stats.get("actor_summary") or [], k=12),
        observables=_fmt_observables(stats.get("observables") or [], k=10),
        cross_posts=_fmt_cross_posts(stats.get("deduped_msgs") or [], k=6),
        prop_samples=_fmt_prop_samples(stats.get("propaganda_examples"), k=6),
    )
    user_prompt = truncate(user_prompt, max_context_chars)

    base_options = {
        "temperature": float(temperature),
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_ctx": int(num_ctx),
        "num_predict": int(num_predict),
    }

    _ollama_warmup(model, timeout=min(60, timeout))

    # ---------- C-first: Force micro (if set) ----------
    if force_micro:
        LOG.info("LLM_FORCE_MICRO=1 → using per-field micro-calls directly.")
        return _micro_sections(model, system_prompt, base_options, user_prompt, timeout, retries)

    # ---------- A) Single-shot JSON with schema ----------
    content = ""
    for attempt in range(retries + 1):
        try:
            content = _ollama_chat(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                model=model,
                options={**base_options, "json_schema": JSON_SCHEMA},
                response_format="json",
                timeout=timeout,
            )
            _preview("single-shot(schema) content", content)
            content = _strip_code_fences(content)
            data = json.loads(content)
            return _normalize_sections_dict(data)
        except Exception as e:
            LOG.warning("Strict JSON/schema call failed (try %d/%d): %s", attempt + 1, retries + 1, e)

    # ---------- B) Repair pass ----------
    try:
        repair_inst = (
            "Your previous output was not valid JSON. Return ONLY valid strict JSON conforming to the schema. "
            "No markdown, no comments."
        )
        fixed = _ollama_chat(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt},
                      {"role": "assistant", "content": content or "{}"},
                      {"role": "user", "content": repair_inst}],
            model=model,
            options={**base_options, "json_schema": JSON_SCHEMA},
            response_format="json",
            timeout=timeout,
        )
        _preview("repair content", fixed)
        fixed = _strip_code_fences(fixed)
        data = _extract_json_block(fixed) or {}
        if isinstance(data, dict) and data:
            return _normalize_sections_dict(data)
    except Exception as e:
        LOG.warning("Repair attempt failed: %s", e)

    # ---------- C) Per-field micro-calls ----------
    LOG.warning("Falling back to per-field JSON micro-calls.")
    return _micro_sections(model, system_prompt, base_options, user_prompt, timeout, retries)


def _micro_sections(model, system_prompt, base_options, user_prompt, timeout, retries) -> Dict[str, str]:
    sections = {}
    for key in (
        "exec_summary",
        "narrative_assessment",
        "propaganda_assessment",
        "risk_outlook",
        "adversary_objectives",
        "collection_gaps",
        "method_notes",
    ):
        sections[key] = _micro_call(
            model=model,
            system_prompt=system_prompt,
            base_options=base_options,
            field=key,
            field_schema=FIELD_SCHEMAS[key],
            context=user_prompt,
            timeout=timeout,
            retries=retries,
        )
    return sections


def _normalize_sections_dict(data: dict) -> Dict[str, str]:
    out = {}
    for k in (
        "exec_summary",
        "narrative_assessment",
        "propaganda_assessment",
        "risk_outlook",
        "adversary_objectives",
        "collection_gaps",
        "method_notes",
    ):
        out[k] = _bullets_to_str(data.get(k))
    return out


def _micro_call(
    *,
    model: str,
    system_prompt: str,
    base_options: dict,
    field: str,
    field_schema: dict,
    context: str,
    timeout: int,
    retries: int,
) -> str:
    lang_hint = "Напишите по-русски." if "Пишите на русском" in system_prompt else "Write in English."
    field_task = {
        "exec_summary": "Return only JSON with key \"exec_summary\" containing 4-8 concise sentences.",
        "narrative_assessment": "Return only JSON with key \"narrative_assessment\" as 5-10 short bullets.",
        "propaganda_assessment": "Return only JSON with key \"propaganda_assessment\" as 3-6 short bullets.",
        "risk_outlook": "Return only JSON with key \"risk_outlook\" as 3-6 short bullets with Low/Med/High markers.",
        "adversary_objectives": "Return only JSON with key \"adversary_objectives\" as 3-6 short bullets.",
        "collection_gaps": "Return only JSON with key \"collection_gaps\" as 3-6 short bullets with specific collection asks.",
        "method_notes": "Return only JSON with key \"method_notes\" as 2-4 short bullets.",
    }[field]

    prompt = (
        f"{context}\n\nMICRO-TASK:\n{field_task}\n"
        "Strict JSON only (double-quoted keys and strings). No markdown. "
        "Use single quotes inside values if you need quotes or escape \\\". "
        f"{lang_hint}"
    )

    for attempt in range(retries + 1):
        try:
            content = _ollama_chat(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
                model=model,
                options={**base_options, "json_schema": field_schema},
                response_format="json",
                timeout=timeout,
            )
            _preview(f"micro({field}) content", content)
            content = _strip_code_fences(content)
            data = _extract_json_block(content) or {}
            return _bullets_to_str(data.get(field, ""))
        except Exception as e:
            LOG.warning("micro-call '%s' failed (try %d/%d): %s", field, attempt + 1, retries + 1, e)

    return ""


# ----------------------- Glue for report_writer -----------------------
def enrich_stats_for_llm(
    df: pd.DataFrame,
    *,
    tfidf_terms: List[Tuple[str, float]] = None,
    auto_keywords: List[Tuple[str, int]] = None,
    user_keywords: Dict[str, int] = None,
    observables: List[Tuple[str, str, int]] = None,
    actor_summary: List[Tuple[str, int]] = None,
    propaganda_examples: pd.DataFrame = None,
    deduped_msgs: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return dict(
        tfidf_terms=tfidf_terms or [],
        auto_keywords=auto_keywords or [],
        user_keywords=user_keywords or {},
        observables=observables or [],
        actor_summary=actor_summary or [],
        propaganda_examples=propaganda_examples if isinstance(propaganda_examples, pd.DataFrame) else None,
        deduped_msgs=deduped_msgs or [],
        date_min=pd.to_datetime(df["date"]).min() if "date" in df.columns else None,
        date_max=pd.to_datetime(df["date"]).max() if "date" in df.columns else None,
    )


# ----------------------- Minimal self-test -----------------------
if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "text": [
                "Беларусь и Россия проведут учения Запад-2025. Минобороны заявило о прозрачности.",
                "НАТО усиливает активность у границ. Литва и Польша проводят учения.",
            ],
            "date": ["2025-08-01", "2025-08-05"],
            "group": ["modmilby", "news_lv"],
            "url": ["https://t.me/modmilby/1", "https://t.me/news_lv/1"],
        }
    )
    stats = enrich_stats_for_llm(
        sample,
        tfidf_terms=[("учения", 2.3), ("беларусь", 1.9)],
        auto_keywords=[("учения", 5), ("нато", 3)],
        user_keywords={"запад-2025": 4},
        observables=[("Military exercises", "учени", 9)],
        actor_summary=[("нато", 3), ("минобороны", 2)],
        propaganda_examples=pd.DataFrame(
            {"text": ["Минск гасит искры — Запад раздувает истерию."], "url": [""], "group": ["runeurope"]}
        ),
        deduped_msgs=[{"text": "Запад-2025 пройдет в сентябре.", "sources": ["modmilby", "runeurope"]}],
    )

    sections = try_llm_analytic_report(sample, stats, temperature=0.2, max_context_chars=6000)
    print(json.dumps(sections, ensure_ascii=False, indent=2) if sections else "LLM not available or failed — no sections.")
