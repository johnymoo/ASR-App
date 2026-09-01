#!/usr/bin/env python3
"""Configurable, persistent podcast Q&A API for Podcast ASR Studio."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Body, HTTPException

PODCAST_ROOT = Path(os.environ.get("PODCAST_ROOT", "~/podcast")).expanduser()
PODCAST_LIBRARY_DIR = Path(
    os.environ.get("PODCAST_LIBRARY_DIR", "~/deployments/sensevoice/static/podcast-asr")
).expanduser()
PODCAST_SITE_BASE = os.environ.get("PODCAST_ASR_SITE_BASE", "/static/podcast-asr").rstrip("/")
QA_ROOT = Path(os.environ.get("PODCAST_QA_ROOT", str(PODCAST_ROOT / "qa"))).expanduser()
QA_CONFIG_PATH = Path(
    os.environ.get("PODCAST_QA_CONFIG_PATH", str(QA_ROOT / "config.json"))
).expanduser()
QA_TASK_DIR = Path(os.environ.get("PODCAST_QA_TASK_DIR", str(QA_ROOT / "tasks"))).expanduser()
QA_SESSION_DIR = Path(os.environ.get("PODCAST_QA_SESSION_DIR", str(QA_ROOT / "sessions"))).expanduser()

DEFAULT_SYSTEM_PROMPT = (
    "你是播客研究助理。只能依据系统提供的播客 summary、官方大纲和转写原文回答。"
    "优先使用转写原文作为事实依据；不要用外部知识补全，不确定时明确说未找到。"
    "回答使用简洁中文，并且只引用系统提供的 evidence_id。"
)

router = APIRouter(prefix="/api/podcast-asr/qa", tags=["podcast-qa"])
_STATE_LOCK = threading.RLock()
_CONFIG_FIELDS = {"api_base", "api_key", "model", "system_prompt", "temperature", "timeout_seconds"}
_TERMINAL_STATES = {"completed", "failed"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    for path in (QA_ROOT, QA_TASK_DIR, QA_SESSION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if private:
        os.chmod(tmp, 0o600)
    tmp.replace(path)
    if private:
        os.chmod(path, 0o600)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_.-]+", "-", value or "").strip("-")
    if not safe or safe != value:
        raise HTTPException(400, "Invalid identifier")
    return safe


def _task_path(task_id: str) -> Path:
    return QA_TASK_DIR / f"{_safe_id(task_id)}.json"


def _session_path(session_id: str) -> Path:
    return QA_SESSION_DIR / f"{_safe_id(session_id)}.json"


def _default_config() -> dict[str, Any]:
    return {
        "api_base": os.environ.get("PODCAST_QA_API_BASE", "http://127.0.0.1:8004/v1"),
        "api_key": os.environ.get("PODCAST_QA_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        "model": os.environ.get("PODCAST_QA_MODEL", "qwen3.6-35b-fp8"),
        "system_prompt": os.environ.get("PODCAST_QA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        "temperature": float(os.environ.get("PODCAST_QA_TEMPERATURE", "0.1")),
        "timeout_seconds": int(os.environ.get("PODCAST_QA_TIMEOUT_SECONDS", "180")),
    }


def load_config() -> dict[str, Any]:
    config = _default_config()
    saved = _read_json(QA_CONFIG_PATH, {}) or {}
    if isinstance(saved, dict):
        config.update({key: saved[key] for key in _CONFIG_FIELDS if key in saved})
    return config


def public_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    key = str(config.pop("api_key", "") or "")
    config["api_key_set"] = bool(key)
    config["api_key_masked"] = ("*" * min(8, len(key)) + key[-4:]) if key else ""
    config["configured"] = bool(config.get("api_base") and config.get("model"))
    return config


def validate_config(payload: dict[str, Any], current: dict[str, Any] | None = None) -> dict[str, Any]:
    current = dict(current or load_config())
    unknown = set(payload) - _CONFIG_FIELDS
    if unknown:
        raise HTTPException(400, f"Unknown config fields: {', '.join(sorted(unknown))}")
    config = dict(current)
    for key in _CONFIG_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "api_key" and (value is None or str(value) == ""):
            continue
        config[key] = value
    parsed = urlparse(str(config.get("api_base") or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "api_base must be an http(s) URL")
    config["api_base"] = str(config["api_base"]).rstrip("/")
    config["model"] = str(config.get("model") or "").strip()
    config["system_prompt"] = str(config.get("system_prompt") or "").strip()
    if not config["model"]:
        raise HTTPException(400, "model is required")
    if not config["system_prompt"]:
        raise HTTPException(400, "system_prompt is required")
    try:
        config["temperature"] = float(config.get("temperature", 0.1))
        config["timeout_seconds"] = int(config.get("timeout_seconds", 180))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "temperature and timeout_seconds must be numeric") from exc
    if not 0 <= config["temperature"] <= 2:
        raise HTTPException(400, "temperature must be between 0 and 2")
    if not 5 <= config["timeout_seconds"] <= 1800:
        raise HTTPException(400, "timeout_seconds must be between 5 and 1800")
    config["api_key"] = str(config.get("api_key") or "")
    return config


def _transcription_path(output_dir: Path) -> Path | None:
    candidates = list(output_dir.glob("transcription_*.json"))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int]:
        data = _read_json(path, {}) or {}
        chunks = data.get("chunks") or []
        ok = sum(1 for chunk in chunks if not chunk.get("error"))
        return ok, path.stat().st_mtime_ns

    return max(candidates, key=score)


def discover_episodes() -> list[dict[str, Any]]:
    published = _read_json(PODCAST_LIBRARY_DIR / "episodes.json", []) or []
    published_by_id = {str(item.get("episode_id")): item for item in published if item.get("episode_id")}
    episodes: list[dict[str, Any]] = []
    for output_dir in sorted(PODCAST_ROOT.glob("*/output")):
        trans_path = _transcription_path(output_dir)
        if not trans_path:
            continue
        trans = _read_json(trans_path, {}) or {}
        chunks = [chunk for chunk in (trans.get("chunks") or []) if not chunk.get("error")]
        if not chunks:
            continue
        episode_id = str(trans.get("episode_id") or output_dir.parent.name.replace("xiaoyuzhou_", ""))
        page_context = _read_json(output_dir / "episode_page_context.json", {}) or {}
        summary = _read_json(output_dir / "podcast_summary.json", {}) or {}
        site_meta = _read_json(output_dir / "site_meta.json", {}) or {}
        published_item = published_by_id.get(episode_id, {})
        slug = str(site_meta.get("slug") or published_item.get("slug") or f"xiaoyuzhou-{episode_id}")
        title = str(
            page_context.get("official_title")
            or summary.get("title")
            or trans.get("title")
            or episode_id
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "slug": slug,
                "title": title,
                "output_dir": output_dir,
                "transcription_path": trans_path,
                "transcription": trans,
                "summary": summary,
                "outline": page_context.get("outline") or summary.get("official_outline") or [],
                "chunks": chunks,
            }
        )
    return episodes


def _plain_json(value: Any, limit: int = 18_000) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:limit]


def _sentence_spans(text: str, max_chars: int = 420) -> list[tuple[int, int, str]]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
        part = match.group(0).strip()
        if not part:
            continue
        part_start = match.start()
        while len(part) > max_chars:
            split = max_chars
            spans.append((part_start, part_start + split, part[:split]))
            part = part[split:]
            part_start += split
        if part:
            spans.append((part_start, match.end(), part))
        start = match.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text), text[start:].strip()))
    return spans


def _fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def build_records(episode: dict[str, Any]) -> list[dict[str, Any]]:
    episode_id = episode["episode_id"]
    slug = episode["slug"]
    title = episode["title"]
    report_url = f"{PODCAST_SITE_BASE}/{slug}/index.html"
    records: list[dict[str, Any]] = []
    if episode.get("summary"):
        records.append(
            {
                "evidence_id": f"{episode_id}:summary",
                "episode_id": episode_id,
                "episode_title": title,
                "source_type": "summary",
                "timestamp": "",
                "text": _plain_json(episode["summary"], 6_000),
                "url": report_url + "#summary",
            }
        )
    if episode.get("outline"):
        records.append(
            {
                "evidence_id": f"{episode_id}:outline",
                "episode_id": episode_id,
                "episode_title": title,
                "source_type": "outline",
                "timestamp": "",
                "text": _plain_json(episode["outline"], 4_000),
                "url": report_url + "#outline",
            }
        )
    for chunk in episode["chunks"]:
        chunk_index = int(chunk.get("chunk_index") or 0)
        chunk_start = float(chunk.get("start") or 0)
        chunk_end = float(chunk.get("end") or chunk_start)
        text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
        spans = _sentence_spans(text)
        if not spans:
            continue
        for window_index in range(0, len(spans), 2):
            window = spans[window_index : window_index + 3]
            begin = window[0][0]
            finish = window[-1][1]
            ratio = begin / max(1, len(text))
            approx_seconds = chunk_start + (chunk_end - chunk_start) * ratio
            excerpt = "".join(item[2] for item in window)
            records.append(
                {
                    "evidence_id": f"{episode_id}:transcript:{chunk_index:03d}:{window_index:03d}",
                    "episode_id": episode_id,
                    "episode_title": title,
                    "source_type": "transcript",
                    "timestamp": _fmt_ts(approx_seconds),
                    "timestamp_approximate": True,
                    "text": excerpt,
                    "url": f"{PODCAST_SITE_BASE}/{slug}/full.html#chunk-{chunk_index:03d}",
                    "chunk_index": chunk_index,
                    "span": [begin, finish],
                }
            )
    return records


def _tokens(text: str) -> list[str]:
    text = str(text or "").lower()
    words = re.findall(r"[a-z0-9][a-z0-9._+-]*", text)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk: list[str] = []
    for run in cjk_runs:
        cjk.extend(run)
        cjk.extend(run[i : i + 2] for i in range(max(0, len(run) - 1)))
        cjk.extend(run[i : i + 3] for i in range(max(0, len(run) - 2)))
    return words + cjk


def retrieve(records: list[dict[str, Any]], query: str, limit: int = 14) -> list[dict[str, Any]]:
    query_tokens = Counter(_tokens(query))
    if not query_tokens or not records:
        return []
    document_tokens = [Counter(_tokens(record["text"] + " " + record["episode_title"])) for record in records]
    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequency.update(tokens.keys())
    total = len(records)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    normalized_query = re.sub(r"\s+", "", query.lower())
    query_words = {word for word in re.findall(r"[a-z0-9][a-z0-9._+-]*", query.lower()) if len(word) >= 3}
    for index, (record, tokens) in enumerate(zip(records, document_tokens)):
        score = 0.0
        for token, query_count in query_tokens.items():
            if token not in tokens:
                continue
            idf = math.log((total + 1) / (document_frequency[token] + 0.5)) + 1
            score += min(tokens[token], 3) * idf * min(query_count, 2)
            if len(token) >= 2 and document_frequency[token] <= max(3, total // 10):
                score += 4 * idf
        compact_text = re.sub(r"\s+", "", record["text"].lower())
        title_text = str(record.get("episode_title") or "").lower()
        if any(word in title_text for word in query_words):
            score += 200
        if normalized_query and normalized_query in compact_text:
            score += 25
        if "推荐" in normalized_query and re.search(r"(?:想推荐|推荐.{0,12}(?:读|看|听))", compact_text):
            score += 120
        if record["source_type"] == "transcript":
            score *= 1.15
        if score > 0:
            scored.append((score, index, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    transcript_per_episode: Counter[str] = Counter()
    for score, _, record in scored:
        if record["source_type"] == "transcript":
            if transcript_per_episode[record["episode_id"]] >= 5:
                continue
            transcript_per_episode[record["episode_id"]] += 1
        item = dict(record)
        item["retrieval_score"] = round(score, 3)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def assemble_context(
    episodes: list[dict[str, Any]], query: str | list[str], scope: str, episode_id: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = [query] if isinstance(query, str) else [item for item in query if item]
    selected_episodes = episodes
    if scope == "episode":
        selected_episodes = [episode for episode in episodes if episode["episode_id"] == episode_id]
        if not selected_episodes:
            raise ValueError("未找到当前播客的已发布转写")
    else:
        episode_records = [
            {
                "evidence_id": f"{episode['episode_id']}:episode",
                "episode_id": episode["episode_id"],
                "episode_title": episode["title"],
                "source_type": "episode",
                "timestamp": "",
                "text": " ".join(
                    [
                        episode["title"],
                        _plain_json(episode.get("summary") or {}, 6_000),
                        _plain_json(episode.get("outline") or [], 4_000),
                    ]
                ),
                "url": f"{PODCAST_SITE_BASE}/{episode['slug']}/index.html",
            }
            for episode in episodes
        ]
        relevant_ids: list[str] = []
        for current_query in queries:
            for item in retrieve(episode_records, current_query, limit=4):
                if item["episode_id"] not in relevant_ids:
                    relevant_ids.append(item["episode_id"])
                if len(relevant_ids) >= 4:
                    break
            if len(relevant_ids) >= 4:
                break
        selected_episodes = [episode for episode in episodes if episode["episode_id"] in relevant_ids]
    all_records = [record for episode in selected_episodes for record in build_records(episode)]
    ranked = []
    seen: set[str] = set()
    for current_query in queries:
        for item in retrieve(all_records, current_query, limit=10):
            if item["evidence_id"] in seen:
                continue
            ranked.append(item)
            seen.add(item["evidence_id"])
            if len(ranked) >= 16:
                break
        if len(ranked) >= 16:
            break
    if not ranked:
        return [], selected_episodes
    relevant_ids = []
    for item in ranked:
        if item["episode_id"] not in relevant_ids:
            relevant_ids.append(item["episode_id"])
    if scope == "all":
        relevant_ids = relevant_ids[:4]
        ranked = [item for item in ranked if item["episode_id"] in relevant_ids]
    by_id = {record["evidence_id"]: record for record in all_records}
    augmented: list[dict[str, Any]] = list(ranked)
    present = {item["evidence_id"] for item in augmented}
    for relevant_id in relevant_ids:
        for suffix in ("summary", "outline"):
            evidence_id = f"{relevant_id}:{suffix}"
            if evidence_id in by_id and evidence_id not in present:
                augmented.append(by_id[evidence_id])
                present.add(evidence_id)
    return augmented, selected_episodes


def _call_openai(config: dict[str, Any], messages: list[dict[str, str]]) -> str:
    url = str(config["api_base"]).rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def request(current: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if config.get("api_key"):
            headers["Authorization"] = f"Bearer {config['api_key']}"
        req = urllib.request.Request(
            url,
            data=json.dumps(current, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(config["timeout_seconds"])) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        result = request(payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code not in {400, 422}:
            raise RuntimeError(f"Agent API HTTP {exc.code}: {body[:500]}") from exc
        fallback = dict(payload)
        fallback.pop("response_format", None)
        fallback.pop("chat_template_kwargs", None)
        result = request(fallback)
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("Agent API returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or message.get("reasoning") or ""
    if not content:
        raise RuntimeError("Agent API returned empty content")
    return str(content)


def _parse_answer(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Agent did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("Agent response must be a JSON object")
    return value


def _session_history(session: dict[str, Any], limit: int = 8) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in (session.get("messages") or [])[-limit:]:
        if message.get("role") in {"user", "assistant"} and message.get("content"):
            history.append({"role": message["role"], "content": str(message["content"])[:3000]})
    return history


def _history_for_task(session: dict[str, Any], task: dict[str, Any]) -> list[dict[str, str]]:
    if session.get("scope") and session.get("scope") != task.get("scope"):
        return []
    if task.get("scope") == "episode" and session.get("episode_id") != task.get("episode_id"):
        return []
    return _session_history(session)


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    previous_answer = next(
        (item["content"] for item in reversed(history) if item.get("role") == "assistant"),
        "",
    )
    return f"{previous_answer} {question}".strip()


def answer_question(task: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    question = str(task["question"])
    history = _history_for_task(session, task)
    retrieval_queries = [question]
    resolved_query = _retrieval_query(question, history)
    if resolved_query != question:
        retrieval_queries.append(resolved_query)
    episodes = discover_episodes()
    evidence, _ = assemble_context(episodes, retrieval_queries, task["scope"], task.get("episode_id"))
    if not evidence:
        return {"answer": "未在所选播客的摘要、大纲或转写原文中找到相关依据。", "citations": []}
    evidence_payload = [
        {
            "evidence_id": item["evidence_id"],
            "episode": item["episode_title"],
            "source_type": item["source_type"],
            "approximate_timestamp": item["timestamp"],
            "text": item["text"],
        }
        for item in evidence
    ]
    user_prompt = (
        "回答当前问题。必须只使用 EVIDENCE 中的信息。事实性答案至少引用一个 transcript evidence_id。"
        "输出严格 JSON：{\"answer\":\"...\",\"evidence_ids\":[\"...\"]}。"
        "如果证据不足，answer 必须说明未找到，evidence_ids 返回空数组。"
        "不要引用不存在的 evidence_id。\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{json.dumps(evidence_payload, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": config["system_prompt"]}, *history, {"role": "user", "content": user_prompt}]
    parsed = _parse_answer(_call_openai(config, messages))
    answer = str(parsed.get("answer") or "").strip()
    requested_ids = parsed.get("evidence_ids") or []
    allowed = {item["evidence_id"]: item for item in evidence}
    citations: list[dict[str, Any]] = []
    for evidence_id in requested_ids:
        if evidence_id not in allowed:
            continue
        item = allowed[evidence_id]
        citations.append(
            {
                "evidence_id": evidence_id,
                "episode_id": item["episode_id"],
                "episode_title": item["episode_title"],
                "source_type": item["source_type"],
                "timestamp": item["timestamp"],
                "timestamp_approximate": bool(item.get("timestamp_approximate")),
                "text": item["text"][:700],
                "url": item["url"],
            }
        )
    if not answer:
        raise RuntimeError("Agent response did not include an answer")
    has_transcript_citation = any(item["source_type"] == "transcript" for item in citations)
    if (not citations or not has_transcript_citation) and "未找到" not in answer:
        answer = "未在所选播客的摘要、大纲或转写原文中找到可验证的依据。"
        citations = []
    return {"answer": answer, "citations": citations}


def _run_task(task_id: str) -> None:
    with _STATE_LOCK:
        task = _read_json(_task_path(task_id), {}) or {}
        task.update({"status": "running", "started_at": _now(), "updated_at": _now()})
        _write_json(_task_path(task_id), task, private=True)
        session = _read_json(_session_path(task["session_id"]), {}) or {}
    try:
        result = answer_question(task, session)
        completed = _now()
        with _STATE_LOCK:
            session = _read_json(_session_path(task["session_id"]), {}) or session
            session.setdefault("messages", []).extend(
                [
                    {
                        "role": "user",
                        "content": task["question"],
                        "scope": task["scope"],
                        "episode_id": task.get("episode_id"),
                        "created_at": task["created_at"],
                    },
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result["citations"],
                        "scope": task["scope"],
                        "episode_id": task.get("episode_id"),
                        "created_at": completed,
                    },
                ]
            )
            session.update(
                {
                    "scope": task["scope"],
                    "episode_id": task.get("episode_id"),
                    "updated_at": completed,
                }
            )
            _write_json(_session_path(task["session_id"]), session, private=True)
            task.update(result)
            task.update({"status": "completed", "ended_at": completed, "updated_at": completed})
            _write_json(_task_path(task_id), task, private=True)
    except Exception as exc:
        with _STATE_LOCK:
            task = _read_json(_task_path(task_id), {}) or task
            task.update(
                {
                    "status": "failed",
                    "error": str(exc)[:1000],
                    "ended_at": _now(),
                    "updated_at": _now(),
                }
            )
            _write_json(_task_path(task_id), task, private=True)


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "task_id",
        "session_id",
        "question",
        "scope",
        "episode_id",
        "status",
        "answer",
        "citations",
        "error",
        "created_at",
        "started_at",
        "ended_at",
        "updated_at",
    }
    return {key: value for key, value in task.items() if key in allowed}


@router.get("/config")
def get_config():
    return {"ok": True, "config": public_config()}


@router.put("/config")
def put_config(payload: dict[str, Any] = Body(...)):
    config = validate_config(payload)
    with _STATE_LOCK:
        _write_json(QA_CONFIG_PATH, config, private=True)
    return {"ok": True, "config": public_config(config)}


@router.post("/questions")
def create_question(payload: dict[str, Any] = Body(...)):
    _ensure_dirs()
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > 4000:
        raise HTTPException(400, "question is too long")
    scope = str(payload.get("scope") or "episode")
    if scope not in {"episode", "all"}:
        raise HTTPException(400, "scope must be episode or all")
    episode_id = str(payload.get("episode_id") or "").strip() or None
    if scope == "episode" and not episode_id:
        raise HTTPException(400, "episode_id is required for episode scope")
    if episode_id and not re.fullmatch(r"[0-9a-zA-Z_.-]+", episode_id):
        raise HTTPException(400, "Invalid episode_id")
    if scope == "episode" and not any(item["episode_id"] == episode_id for item in discover_episodes()):
        raise HTTPException(404, "Published episode not found")
    session_id = str(payload.get("session_id") or "").strip()
    with _STATE_LOCK:
        if session_id:
            session = _read_json(_session_path(session_id), None)
            if not session:
                raise HTTPException(404, "Session not found")
        else:
            session_id = uuid.uuid4().hex
            session = {
                "session_id": session_id,
                "scope": scope,
                "episode_id": episode_id,
                "messages": [],
                "created_at": _now(),
                "updated_at": _now(),
            }
            _write_json(_session_path(session_id), session, private=True)
        task_id = uuid.uuid4().hex
        task = {
            "task_id": task_id,
            "session_id": session_id,
            "question": question,
            "scope": scope,
            "episode_id": episode_id,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
        }
        _write_json(_task_path(task_id), task, private=True)
    threading.Thread(target=_run_task, args=(task_id,), daemon=True).start()
    return {"ok": True, "task": _public_task(task)}


@router.get("/questions/{task_id}")
def get_question(task_id: str):
    task = _read_json(_task_path(task_id), None)
    if not task:
        raise HTTPException(404, "Question task not found")
    return {"ok": True, "task": _public_task(task)}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = _read_json(_session_path(session_id), None)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"ok": True, "session": session}
