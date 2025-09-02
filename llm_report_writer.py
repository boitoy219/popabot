# llm_report_writer.py — Markdown-only LLM writer (no JSON)
# - Clustering of reposts
# - URL & text dedupe
# - Prompt budgeter
# - Incomplete-output detection + brief retry
# - Topics derived deterministically (n-grams from TF-IDF)
# - Writes both OUT_MD and YYYY-MM-DD_summary.md
# - Default model switched to qwen2.5:7b-instruct-q4_K_M (tighter Markdown discipline)

from __future__ import annotations
import os
import re
import json
import math
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import datetime as _dt

try:
    import requests
except Exception:
    requests = None

RE_URL = re.compile(r"https?://[^\s>]+", re.I)
RE_WS = re.compile(r"\s+")
STOPWORDS = set(
    '''a an and are as at be by for from has have i in is it its of on or our that the their them they this to was were will with you your we he she not if but than then so such'''.split()
)

# ---------------------
# Public entry point
# ---------------------

def write_markdown_report(df, out_path, user_keywords=None, **kwargs):
    """LLM-first writer that expects Markdown output directly (no JSON).
    - Builds deterministic topics & evidence (clustered, deduped, budgeted)
    - Calls Ollama once (with retry-on-incomplete)
    - Appends deterministic appendices
    - Writes to out_path and YYYY-MM-DD_summary.md
    - Raises RuntimeError on LLM failure to allow CPU fallback
    """
    if df is None:
        raise RuntimeError("llm_report_writer: dataframe is None")

    use_llm = os.getenv("POPABOT_USE_LLM", "1") == "1"
    if not use_llm:
        raise RuntimeError("POPABOT_USE_LLM=0 -> skip LLM writer")
    if requests is None:
        raise RuntimeError("requests not available for Ollama call")

    # Env
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip('/')
    # Default model switched per user guidance
    model = os.getenv("POPABOT_LLM", "qwen2.5:7b-instruct-q4_K_M")
    temp = float(os.getenv("POPABOT_LLM_TEMP", "0.2"))
    max_tokens = int(os.getenv("POPABOT_LLM_MAX_TOKENS", "1536"))
    num_ctx = int(os.getenv("POPABOT_LLM_CTX", "8192"))
    trend_png = kwargs.get("trend_png_path")
    if not trend_png:
        trend_png = os.path.join(os.path.dirname(out_path) or ".", "keyword_mentions.png")

    # Deterministic signals & inputs
    det = _build_deterministic(df, user_keywords=user_keywords)
    topics = _derive_topics(det, top_k=8)

    # Evidence selection (cluster reposts, prefer URLs, dedupe, budget)
    evidence = _cluster_and_select_evidence(df, max_items=12)
    evidence = _budget_snippets(evidence, max_chars=int(os.getenv("POPABOT_SNIPPET_CHAR_BUDGET", "8000")))
    if not evidence:
        # As a last resort, pick from raw rows
        evidence = _select_representative_snippets(df, max_snippets=8, min_snippets=4)

    # Compose a strict Markdown prompt (no JSON)
    md = _call_ollama_markdown(
        host=host,
        model=model,
        temp=temp,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        evidence=evidence,
        topics=topics,
        det=det,
        user_keywords=user_keywords,
        force_brief=False,
        num_predict_boost=1.0,
    )

    # If incomplete, retry with briefer instructions and more tokens
    if _looks_incomplete(md):
        md = _call_ollama_markdown(
            host=host,
            model=model,
            temp=temp,
            max_tokens=int(max_tokens * 1.5),
            num_ctx=num_ctx,
            evidence=evidence,
            topics=topics,
            det=det,
            user_keywords=user_keywords,
            force_brief=True,
            num_predict_boost=1.5,
        )

    if not isinstance(md, str) or not md.strip():
        raise RuntimeError("LLM returned empty content")

    # Compose final document: LLM sections + deterministic appendices + footer
    out_md = []
    now_str = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out_md.append(f"# POPABOT Analytical Report\n\n_Generated {now_str}_\n")

    # NEW: show model attribution
    out_md.append(f"_LLM model: {model}_\n")

    # If the LLM forgot Evidence, add a compact auto-evidence block
    if ("Evidence" not in md) and ("Citations" not in md):
        out_md.append("\n## 🔗 Evidence & Citations (auto)\n")
        for ev in evidence[:10]:
            quote = ev.get("text", "").replace('\n', ' ').strip()
            url = ev.get("url", "").strip()
            if not quote and not url:
                continue
            tail = f" ([source]({url}))" if url else ""
            out_md.append(f"> “{quote[:240]}{'…' if len(quote)>240 else ''}”{tail}")
        out_md.append("")

    # Deterministic appendices
    out_md.append("---\n\n# Appendices — Deterministic Metrics\n")

    # TF-IDF table
    out_md.append("## TF-IDF — Top Terms")
    if det["tfidf"]:
        out_md.append("\n| term | tf-idf |\n|---|---:|")
        for term, score in det["tfidf"][:30]:
            out_md.append(f"| {term} | {score:.4f} |")
        out_md.append("")
    else:
        out_md.append("_No textual content available to compute TF-IDF._\n")

    # Tracked & Auto Keywords
    out_md.append("## Tracked & Auto Keywords")
    if det["tracked_keyword_counts"]:
        out_md.append("### Tracked (from `keywords/`)\n")
        out_md.append("| keyword | mentions |\n|---|---:|")
        for k, c in det["tracked_keyword_counts"]:
            out_md.append(f"| {k} | {c} |")
        out_md.append("")
    else:
        out_md.append("_No tracked keywords provided or no matches._\n")

    if det["auto_keywords"]:
        out_md.append("### Auto (top tf-idf candidates)\n")
        out_md.append(", ".join([f"`{k}`" for k in det["auto_keywords"][:25]]))
        out_md.append("")

    # Trend image (if present)
    out_md.append("## 📊 Keyword Trend")
    if os.path.exists(trend_png):
        out_md.append(f"![Keyword mentions over time]({trend_png})\n")
    else:
        out_md.append("_Trend image not found._\n")

    # Observables
    out_md.append("## ⏱️ Observables")
    if det["observables_table"]:
        out_md.append("| observable | count |\n|---|---:|")
        for obs, c in det["observables_table"]:
            out_md.append(f"| {obs} | {c} |")
        out_md.append("")
    else:
        out_md.append("_No observables found._\n")

    # Actors
    out_md.append("## 👤 Actors")
    if det["actors_table"]:
        out_md.append("| actor (group) | messages |\n|---|---:|")
        for grp, c in det["actors_table"]:
            out_md.append(f"| {grp} | {c} |")
        out_md.append("")
    else:
        out_md.append("_No actors found._\n")

    # Cross-posted items
    out_md.append("## 🔎 Cross-posted Items")
    any_x = False
    if det["crosspost_urls"]:
        any_x = True
        out_md.append("### Same URL across multiple groups")
        for url, groups in det["crosspost_urls"]:
            groups_str = ", ".join(sorted(groups))
            out_md.append(f"- {url} — **{groups_str}**")
        out_md.append("")
    if det["crosspost_texts"]:
        any_x = True
        out_md.append("### Exact same text seen in multiple groups (top)")
        for snippet, groups in det["crosspost_texts"][:20]:
            groups_str = ", ".join(sorted(groups))
            out_md.append(f"- “{snippet}” — **{groups_str}**")
        out_md.append("")
    if not any_x:
        out_md.append("_No cross-posted items detected._\n")

    # Footer
    out_md.append("---\n_Generated by Hybrid pipeline (LLM + deterministic metrics)_\n")

    final_md = "\n".join(out_md).strip() + "\n"

    # Primary write
    _write_file(out_path, final_md)

    # Dated copy (YYYY-MM-DD_summary.md) alongside out_path
    date_tag = _local_datestr()
    dated_dir = os.path.dirname(out_path) or "."
    dated_path = os.path.join(dated_dir, f"{date_tag}_summary.md")
    _write_file(dated_path, final_md)


# ---------------------
# LLM plumbing (Markdown output)
# ---------------------

def _call_ollama_markdown(host: str, model: str, temp: float, max_tokens: int, num_ctx: int,
                          evidence: List[Dict[str, str]], topics: List[str], det: Dict[str, Any],
                          user_keywords: List[str] | None, force_brief: bool, num_predict_boost: float) -> str:
    url = f"{host}/api/chat"

    system = (
        "You are an intelligence analyst. Produce a concise, decision-ready Markdown report. "
        "Use only the supplied EVIDENCE URLs for quotes. Avoid speculation. Keep bullets ≤ 24 words."
    )

    def compact(s: str, lim: int = 360) -> str:
        s = RE_WS.sub(" ", (s or "")).strip()
        return (s[: lim - 1] + "…") if len(s) > lim else s

    ev_items = [
        {
            "text": compact(sn.get("text")),
            "url": sn.get("url") or "",
            "date": sn.get("date") or "",
            "group": sn.get("group") or "",
        }
        for sn in evidence[:12]
    ]

    tracked_kw = ", ".join(sorted(set((user_keywords or []))))

    # Deterministic signals snapshot for model context
    tfidf_terms = ", ".join([t for t, _ in det.get("tfidf", [])[:25]])
    observed_domains = ", ".join([d for d, _ in det.get("observables_table", [])[:10]])

    # Strict section order and headings (LLM should output exactly this skeleton)
    skeleton = (
        f"## 🧭 Executive Summary (LLM — {model})\n\n"
        "## 🔎 Topics to Explore (from data)\n\n"
        "## 📚 Key Narratives (LLM)\n\n"
        "## 🧠 Information Operations / Propaganda (LLM)\n\n"
        "## ⚠️ Near-term Risk Outlook (LLM)\n\n"
        "## 🎯 Adversary Objectives (LLM)\n\n"
        "## 🔍 Collection Gaps & Next Steps (LLM)\n\n"
        "## 🧪 Method Notes (LLM)\n\n"
        "## 🔗 Evidence & Citations (LLM)\n"
    )

    brief_clause = ("\nTOTAL LENGTH ≤ 600 words. Prefer bullets; compress phrasing.\n"
                    if force_brief else "")

    content = (
        "TASK: Write the following Markdown sections in the exact order and headings shown below.\n"
        "Do not add extra intro/outro lines.\n\n"
        f"SECTION SKELETON:\n{skeleton}\n\n"
        f"TRACKED_KEYWORDS: {tracked_kw}\n"
        f"SUGGESTED_TOPICS (deterministic): {json.dumps(topics, ensure_ascii=False)}\n"
        f"TFIDF_TERMS: {tfidf_terms}\n"
        f"OBSERVED_DOMAINS: {observed_domains}\n\n"
        "EVIDENCE_INPUT (max 12):\n"
        f"{json.dumps(ev_items, ensure_ascii=False)}\n\n"
        "GUIDELINES:\n"
        "- Executive Summary: 4–8 sentences.\n"
        "- Risk bullets labeled Low / Medium / High.\n"
        "- Cite only from EVIDENCE_INPUT using > quote + ([source](URL)).\n"
        "- Keep everything tight and non-redundant; avoid hedging and speculation.\n"
        f"{brief_clause}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "stream": False,
        # Markdown output (no JSON format)
        "options": {
            "temperature": temp,
            "num_predict": int(max_tokens * num_predict_boost),
            "num_ctx": int(num_ctx),
        },
    }

    last_err = None
    for _ in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            md = (data.get("message", {}) or {}).get("content", "").strip()
            return md
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"LLM call failed: {last_err}")


def _looks_incomplete(md: str) -> bool:
    if not md or not isinstance(md, str):
        return True
    must = [
        "## 🧭 Executive Summary (LLM)",
        "## 📚 Key Narratives (LLM)",
        "## 🔗 Evidence & Citations (LLM)",
    ]
    missing = any(h not in md for h in must)
    # Truncation heuristic: ends with unfinished last line (no newline) or dangling bullet
    ends_odd = not md.endswith("\n") or md.rstrip().endswith(('-','*'))
    return bool(missing or ends_odd)


# ---------------------
# Deterministic appendices & inputs
# ---------------------

def _build_deterministic(df, user_keywords=None) -> Dict[str, Any]:
    texts = list(df["text"])  # pandas Series compatible
    groups = list(df["group"])  # type: ignore
    urls = list(df["url"])  # type: ignore

    tfidf = _simple_tfidf(texts)

    # Tracked keyword counts
    tracked_kw = [k.strip() for k in (user_keywords or []) if k and k.strip()]
    tracked_counts = []
    if tracked_kw:
        lc_texts = [t.lower() if isinstance(t, str) else "" for t in texts]
        for kw in sorted(set(tracked_kw)):
            patt = re.compile(rf"\b{re.escape(kw.lower())}\b")
            c = sum(1 for t in lc_texts if patt.search(t))
            tracked_counts.append((kw, c))
        tracked_counts.sort(key=lambda x: (-x[1], x[0]))

    # Auto keywords from tf-idf
    auto_keywords = [t for t, _ in tfidf if t not in STOPWORDS][:50]

    # Observables: domains
    domains = []
    for u in urls:
        if not isinstance(u, str):
            continue
        m = re.match(r"https?://([^/]+)/?", u)
        if m:
            domains.append(m.group(1).lower())
    obs_counts = Counter(domains).most_common(25)

    # Actors
    grp_counts = Counter([g for g in groups if isinstance(g, str)]).most_common(25)

    # Cross-posts (URL across groups)
    url_groups: Dict[str, set] = defaultdict(set)
    for u, g in zip(urls, groups):
        if isinstance(u, str) and isinstance(g, str) and u:
            url_groups[u].add(g)
    crosspost_urls = [(u, sorted(gs)) for u, gs in url_groups.items() if len(gs) > 1]
    crosspost_urls.sort(key=lambda x: (-len(x[1]), x[0]))

    # Cross-posts (exact same text across groups)
    text_groups: Dict[str, set] = defaultdict(set)
    for t, g in zip(texts, groups):
        if isinstance(t, str) and isinstance(g, str):
            key = _normalize_text(t)
            if key:
                text_groups[key].add(g)
    cross_texts = [
        (key[:160] + ("…" if len(key) > 160 else ""), sorted(gs))
        for key, gs in text_groups.items() if len(gs) > 1
    ]
    cross_texts.sort(key=lambda x: (-len(x[1]), x[0]))

    return {
        "tfidf": tfidf,
        "tracked_keyword_counts": tracked_counts,
        "auto_keywords": auto_keywords,
        "observables_table": obs_counts,
        "actors_table": grp_counts,
        "crosspost_urls": crosspost_urls,
        "crosspost_texts": cross_texts,
    }


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = RE_WS.sub(" ", s.strip())
    return s


def _tokenize(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9_#@]+", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) >= 3]
    return toks


def _simple_tfidf(texts: List[str]) -> List[Tuple[str, float]]:
    docs = [t for t in texts if isinstance(t, str) and t.strip()]
    if not docs:
        return []
    N = len(docs)
    from collections import Counter as C
    tfs: List[C] = []
    df: C = C()
    for d in docs:
        toks = _tokenize(d)
        tf = C(toks)
        tfs.append(tf)
        for tok in set(tf):
            df[tok] += 1
    scores: C = C()
    for tf in tfs:
        for tok, f in tf.items():
            idf = math.log((N + 1) / (df[tok] + 1)) + 1  # smoothed idf
            scores[tok] += f * idf
    for tok in list(scores):
        scores[tok] /= N
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _derive_topics(det: Dict[str, Any], top_k: int = 8) -> List[str]:
    """Deterministically derive compact topics using TF-IDF anchors + n-gram counts."""
    terms = [t for t, _ in det.get("tfidf", [])[:120]]
    if not terms:
        return []

    def is_stop(tok: str) -> bool:
        return (tok in STOPWORDS) or (len(tok) < 3)

    grams: Counter = Counter()
    pseudo = " ".join(terms)
    toks = [t for t in re.sub(r"[^a-z0-9_#@]+", " ", pseudo.lower()).split() if not is_stop(t)]

    for a, b in zip(toks, toks[1:]):
        if is_stop(a) or is_stop(b):
            continue
        grams[f"{a} {b}"] += 1
    for a, b, c in zip(toks, toks[1:], toks[2:]):
        if is_stop(a) or is_stop(b) or is_stop(c):
            continue
        grams[f"{a} {b} {c}"] += 1

    top_anchor = set([t for t, _ in det.get("tfidf", [])[:30]])
    scored: List[Tuple[str, float]] = []
    for g, f in grams.items():
        anchor_bonus = 1.0 if any(tok in top_anchor for tok in g.split()) else 0.0
        scored.append((g, f + anchor_bonus))
    scored.sort(key=lambda x: (-x[1], x[0]))

    topics: List[str] = [g for g, _ in scored[:top_k]]

    if len(topics) < top_k:
        for t in terms:
            if t not in topics and not is_stop(t):
                topics.append(t)
            if len(topics) >= top_k:
                break

    return topics[:top_k]


# ---------------------
# Evidence selection (clustering + budgeting)
# ---------------------

def _cluster_and_select_evidence(df, max_items: int = 12) -> List[Dict[str, str]]:
    """Cluster near-duplicate posts & cross-posts; return representative evidence rows.
    Clustering strategy:
      1) Hard group by URL (exact match) first.
      2) For rows without URL duplicates, cluster by Jaccard ≥ 0.7 over token sets.
      3) Choose a representative per cluster (prefer most recent with a URL),
         and produce a concise snippet (<= 360 chars).
    """
    import pandas as _pd

    s = df.copy()
    s["date"] = _pd.to_datetime(s["date"], errors="coerce", utc=True)

    # 1) Primary: group by exact URL
    url_groups: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(s["url"].astype(str).tolist()):
        url_groups[u].append(i)

    visited = set()
    clusters: List[List[int]] = []

    # same-URL clusters
    for u, idxs in url_groups.items():
        if len(idxs) > 1:
            clusters.append(idxs)
            visited.update(idxs)

    # 2) Jaccard clustering for remaining
    def toks(text: str) -> set:
        text = str(text or "").lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^a-z0-9_#@]+", " ", text)
        return set([t for t in text.split() if t and t not in STOPWORDS])

    remaining = [i for i in range(len(s)) if i not in visited]
    text_sets = {i: toks(s.iloc[i]["text"]) for i in remaining}
    used = set()
    for i in remaining:
        if i in used:
            continue
        group = [i]
        Si = text_sets[i]
        for j in remaining:
            if j in used or j == i:
                continue
            Sj = text_sets[j]
            if not Si or not Sj:
                continue
            jac = len(Si & Sj) / max(1, len(Si | Sj))
            if jac >= 0.7:
                group.append(j)
                used.add(j)
        used.add(i)
        clusters.append(group)

    # 3) Representative per cluster
    reps: List[Dict[str, str]] = []
    for idxs in clusters:
        sub = s.iloc[idxs].sort_values("date", ascending=False)
        # prefer rows that have a real URL
        with_url = sub[sub["url"].astype(str).str.startswith("http")]
        rep = with_url.iloc[0] if not with_url.empty else sub.iloc[0]
        reps.append({
            "message_id": str(rep["message_id"]),
            "group": str(rep["group"]) if rep["group"] is not None else "",
            "date": str(rep["date"]) if rep["date"] is not None else "",
            "text": _normalize_text(str(rep["text"]))[:360],
            "url": str(rep["url"]) if rep["url"] is not None else "",
        })

    # Add a few singletons (recent, url-first) to reach max_items
    if len(reps) < max_items:
        singles = s[~s.index.isin([i for grp in clusters for i in grp])]  # type: ignore
        singles = singles.sort_values(["date"], ascending=False)
        for _, row in singles.iterrows():
            reps.append({
                "message_id": str(row["message_id"]),
                "group": str(row["group"]) if row["group"] is not None else "",
                "date": str(row["date"]) if row["date"] is not None else "",
                "text": _normalize_text(str(row["text"]))[:360],
                "url": str(row["url"]) if row["url"] is not None else "",
            })
            if len(reps) >= max_items:
                break

    # Final URL dedupe (just in case) and cap length
    seen_urls = set()
    out: List[Dict[str, str]] = []
    for r in reps:
        u = r.get("url", "")
        if u and u in seen_urls:
            continue
        seen_urls.add(u)
        r["text"] = r.get("text", "")[:360]
        out.append(r)
        if len(out) >= max_items:
            break

    return out


def _select_representative_snippets(df, max_snippets=12, min_snippets=6) -> List[Dict[str, str]]:
    """Fallback snippet selector (diverse by group, recent, URL-first, dedupes URLs)."""
    import pandas as _pd

    s = df.copy()
    s["date"] = _pd.to_datetime(s["date"], errors="coerce", utc=True)
    s["url_ok"] = s["url"].astype(str).str.startswith("http")
    s = s.sort_values(["url_ok", "date"], ascending=[False, False])

    buckets = {g: gdf for g, gdf in s.groupby("group")}
    order = sorted(buckets.keys(), key=lambda g: -len(buckets[g]))

    chosen: List[Dict[str, str]] = []
    seen_urls = set()
    while len(chosen) < max_snippets:
        made = False
        for g in order:
            gdf = buckets[g]
            idx = len([r for r in chosen if r["group"] == g])
            if idx < len(gdf):
                row = gdf.iloc[idx]
                url = str(row["url"]) if row["url"] is not None else ""
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                chosen.append({
                    "message_id": str(row["message_id"]),
                    "group": str(row["group"]) if row["group"] is not None else "",
                    "date": str(row["date"]) if row["date"] is not None else "",
                    "text": _normalize_text(str(row["text"]))[:360],
                    "url": url,
                })
                made = True
                if len(chosen) >= max_snippets:
                    break
        if not made:
            break

    if len(chosen) < min_snippets:
        for _, row in s.iterrows():
            if len(chosen) >= min_snippets:
                break
            url = str(row["url"]) if row["url"] is not None else ""
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            chosen.append({
                "message_id": str(row["message_id"]),
                "group": str(row["group"]) if row["group"] is not None else "",
                "date": str(row["date"]) if row["date"] is not None else "",
                "text": _normalize_text(str(row["text"]))[:360],
                "url": url,
            })
    return chosen


def _budget_snippets(snips: List[Dict[str, str]], max_chars: int = 8000) -> List[Dict[str, str]]:
    """Limit total prompt size by trimming per-snippet text adaptively to fit max_chars."""
    out: List[Dict[str, str]] = []
    used = 0
    def remaining() -> int:
        return max(0, max_chars - used)

    for s in snips:
        if remaining() <= 0:
            break
        txt = s.get("text", "")
        per_cap = min(900, max(200, remaining() // max(1, (len(snips) - len(out)))))
        trimmed = txt[:per_cap]
        size = len(trimmed) + len(s.get("url", "")) + 64
        if size > remaining():
            break
        s2 = dict(s)
        s2["text"] = trimmed
        out.append(s2)
        used += size
    return out


# ---------------------
# IO
# ---------------------

def _write_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _local_datestr() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d")
