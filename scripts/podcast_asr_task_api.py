#!/usr/bin/env python3
"""FastAPI router for persistent Xiaoyuzhou Podcast ASR background tasks.

The static publisher's import form calls these endpoints. Tasks are started with
``subprocess.Popen(..., start_new_session=True)`` and persist their JSON state to
``PODCAST_TASK_DIR``, so users can close/reopen the browser while the server-side
pipeline continues running.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

PODCAST_ROOT = Path(os.environ.get("PODCAST_ROOT", "~/podcast")).expanduser()
PODCAST_PIPELINE = Path(os.environ.get("PODCAST_PIPELINE", str(PODCAST_ROOT / "xiaoyuzhou_asr_to_site.py"))).expanduser()
PODCAST_TASK_DIR = Path(os.environ.get("PODCAST_TASK_DIR", str(PODCAST_ROOT / "asr_tasks"))).expanduser()
PODCAST_TASK_DIR.mkdir(parents=True, exist_ok=True)
PODCAST_SITE_BASE = os.environ.get("PODCAST_ASR_SITE_BASE", "/static/podcast-asr").rstrip("/")
PODCAST_LIBRARY_DIR = Path(os.environ.get("PODCAST_LIBRARY_DIR", "~/deployments/sensevoice/static/podcast-asr")).expanduser()

_TASK_LOCK = threading.Lock()
_TERMINAL_TASK_STATES = {"completed", "failed", "unknown"}
router = APIRouter(prefix="/api/podcast-asr", tags=["podcast-asr"])


def _task_state_path(job_id: str) -> Path:
    safe = re.sub(r"[^0-9a-zA-Z_.-]+", "-", job_id).strip("-")
    return PODCAST_TASK_DIR / f"{safe}.json"


def _write_state(state: dict[str, Any]) -> None:
    path = _task_state_path(str(state["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_states() -> list[dict[str, Any]]:
    states = []
    for path in sorted(PODCAST_TASK_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        state = _read_state(path)
        if state:
            states.append(state)
    return states


def _latest_for_episode(episode_id: str) -> dict[str, Any] | None:
    for state in _list_states():
        if state.get("episode_id") == episode_id:
            return state
    return None


def _parse_episode(value: str) -> tuple[str, str]:
    value = (value or "").strip()
    if not value:
        raise ValueError("请输入小宇宙 episode 链接")
    url_match = re.match(r"^https?://(?:www\.)?xiaoyuzhoufm\.com/episode/([0-9a-fA-F]{12,})(?:[/?#].*)?$", value)
    id_match = re.fullmatch(r"[0-9a-fA-F]{12,}", value)
    if url_match:
        return url_match.group(1), value
    if id_match:
        episode_id = id_match.group(0)
        return episode_id, f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"
    raise ValueError("只支持 https://www.xiaoyuzhoufm.com/episode/<id> 形式的小宇宙链接")


def _pid_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _tail_text(path: Path, max_bytes: int = 16000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _published_slug_for_state(state: dict[str, Any]) -> str:
    work_dir = Path(state.get("work_dir") or "")
    site_meta_path = work_dir / "output" / "site_meta.json"
    try:
        site_meta = json.loads(site_meta_path.read_text(encoding="utf-8")) if site_meta_path.exists() else {}
    except Exception:
        site_meta = {}
    slug = site_meta.get("slug") or state.get("slug") or ""
    episodes_json = PODCAST_LIBRARY_DIR / "episodes.json"
    try:
        episodes = json.loads(episodes_json.read_text(encoding="utf-8")) if episodes_json.exists() else []
    except Exception:
        episodes = []
    if isinstance(episodes, dict):
        episodes = episodes.get("episodes") or episodes.get("items") or []
    for ep in episodes if isinstance(episodes, list) else []:
        if not isinstance(ep, dict):
            continue
        if (slug and ep.get("slug") == slug) or (state.get("episode_id") and ep.get("episode_id") == state.get("episode_id")):
            return str(ep.get("slug") or slug)
    return str(slug)


def _chunk_counts(output_dir: Path) -> tuple[int, int, str]:
    best_ok = best_total = 0
    best_device = ""
    for path in sorted(output_dir.glob("transcription_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = data.get("summary") or {}
            total = int(summary.get("chunks") or len(data.get("chunks") or []))
            ok = int(summary.get("ok_chunks") or sum(1 for c in data.get("chunks") or [] if not c.get("error")))
            if ok > best_ok or (ok == best_ok and total > best_total):
                best_ok, best_total = ok, total
                best_device = str(data.get("device") or path.stem.replace("transcription_", ""))
        except Exception:
            pass
    return best_ok, best_total, best_device


def _enrich(state: dict[str, Any], include_log: bool = False) -> dict[str, Any]:
    state = dict(state)
    status = state.get("status")
    pid = state.get("pid")
    if status in {"queued", "running"} and pid and not _pid_alive(pid) and state.get("returncode") is None:
        state["status"] = "unknown"
        state["message"] = "后台进程已退出，但服务未捕获退出码；请查看日志确认结果。"
        state["ended_at"] = state.get("ended_at") or datetime.now().isoformat(timespec="seconds")
        _write_state(state)

    output_dir = Path(state.get("work_dir") or "") / "output"
    slug = _published_slug_for_state(state)
    report_path = PODCAST_LIBRARY_DIR / slug / "index.html" if slug else None
    state["has_report"] = bool(report_path and report_path.is_file())
    if state["has_report"]:
        state["slug"] = slug
        state["report_url"] = f"{PODCAST_SITE_BASE}/{slug}/index.html"
        state["full_text_url"] = f"{PODCAST_SITE_BASE}/{slug}/full.html"
        state["summary_url"] = f"{PODCAST_SITE_BASE}/{slug}/podcast_summary.md"
        state["tldr_image_url"] = f"{PODCAST_SITE_BASE}/{slug}/tldr_infographic.png"
    ok, total, device = _chunk_counts(output_dir)
    state["has_transcription"] = bool(total)
    state["has_summary"] = (output_dir / "podcast_summary.json").exists()
    state["has_tldr_image"] = (output_dir / "tldr_infographic.png").exists()
    if total:
        state["asr_ok_chunks"] = ok
        state["asr_chunks"] = total
        state["asr_device"] = device
        if state.get("status") == "failed" and ok == total and state.get("has_summary") and state["has_report"]:
            state["recovered_from_failed_task"] = True
            state["status"] = "completed"
            state["message"] = "任务已完成并发布；完整转写、总结和报告均可用。"
    if include_log:
        state["log_tail"] = _tail_text(Path(state.get("log_path") or ""))
    return state


def _wait_for_task(job_id: str, proc: subprocess.Popen, log_handle) -> None:
    try:
        returncode = proc.wait()
    finally:
        try:
            log_handle.close()
        except Exception:
            pass
    with _TASK_LOCK:
        state = _read_state(_task_state_path(job_id)) or {"job_id": job_id}
        state["returncode"] = returncode
        state["status"] = "completed" if returncode == 0 else "failed"
        state["ended_at"] = datetime.now().isoformat(timespec="seconds")
        state["updated_at"] = state["ended_at"]
        state["message"] = "任务完成，已发布到索引。" if returncode == 0 else f"任务失败，退出码 {returncode}；请查看日志。"
        _write_state(state)


@router.post("/tasks")
def create_task(payload: dict[str, Any] = Body(...)):
    try:
        episode_id, episode_url = _parse_episode(str(payload.get("url") or payload.get("episode_url") or ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not PODCAST_PIPELINE.exists():
        raise HTTPException(500, f"Pipeline script not found: {PODCAST_PIPELINE}")

    with _TASK_LOCK:
        for old_state in _list_states():
            if old_state.get("episode_id") == episode_id and old_state.get("status") in {"queued", "running"} and _pid_alive(old_state.get("pid")):
                return {"ok": True, "already_running": True, "task": _enrich(old_state, include_log=True)}

        job_id = f"{episode_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        work_dir = PODCAST_ROOT / f"xiaoyuzhou_{episode_id}"
        log_path = PODCAST_TASK_DIR / f"{job_id}.log"
        cmd = ["python3.12", str(PODCAST_PIPELINE), episode_url]
        state = {
            "job_id": job_id,
            "episode_id": episode_id,
            "episode_url": episode_url,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "work_dir": str(work_dir),
            "log_path": str(log_path),
            "command": cmd,
            "message": "任务已排队，准备启动后台 pipeline。",
        }
        _write_state(state)
        log_handle = log_path.open("ab", buffering=0)
        log_handle.write((f"## {state['created_at']}\n$ {' '.join(cmd)}\n\n").encode("utf-8"))
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PODCAST_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            try:
                log_handle.close()
            except Exception:
                pass
            state["status"] = "failed"
            state["message"] = f"启动失败：{exc!r}"
            state["ended_at"] = datetime.now().isoformat(timespec="seconds")
            _write_state(state)
            raise HTTPException(500, state["message"]) from exc
        state["status"] = "running"
        state["pid"] = proc.pid
        state["started_at"] = datetime.now().isoformat(timespec="seconds")
        state["updated_at"] = state["started_at"]
        state["message"] = "后台 pipeline 已启动：抓取介绍 → 下载音频 → ASR → LLM 总结 → TLDR 图 → 发布。可关闭页面，任务会继续运行。"
        _write_state(state)
        threading.Thread(target=_wait_for_task, args=(job_id, proc, log_handle), daemon=True).start()
        return {"ok": True, "task": _enrich(state, include_log=True)}


@router.get("/tasks")
def list_tasks():
    return {"ok": True, "tasks": [_enrich(s) for s in _list_states()[:20]]}


@router.get("/tasks/by-episode/{episode_id}")
def get_latest_task_for_episode(episode_id: str):
    if not re.fullmatch(r"[0-9a-fA-F]{12,}", episode_id or ""):
        raise HTTPException(400, "Invalid episode id")
    state = _latest_for_episode(episode_id)
    if not state:
        raise HTTPException(404, "Task not found")
    return {"ok": True, "task": _enrich(state, include_log=True)}


@router.get("/tasks/{job_id}")
def get_task(job_id: str):
    state = _read_state(_task_state_path(job_id))
    if not state:
        raise HTTPException(404, "Task not found")
    return {"ok": True, "task": _enrich(state, include_log=True)}
