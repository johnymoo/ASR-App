#!/usr/bin/env python3
"""Persistent upload and task API for meeting transcription."""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

MEETING_ROOT = Path(os.environ.get("MEETING_ROOT", "~/meetings")).expanduser()
MEETING_PIPELINE = Path(
    os.environ.get(
        "MEETING_PIPELINE",
        str(Path(__file__).with_name("meeting_asr_pipeline.py")),
    )
).expanduser()
MAX_UPLOAD_BYTES = int(os.environ.get("MEETING_MAX_UPLOAD_BYTES", str(256 * 1024**2)))
ALLOWED_EXTENSIONS = {".mp3", ".wav"}
TERMINAL_STATES = {"completed", "failed", "canceled", "unknown"}

MEETING_ROOT.mkdir(parents=True, exist_ok=True)
router = APIRouter(prefix="/api/meeting-asr", tags=["meeting-asr"])


def _state_path(task_id: str) -> Path:
    return MEETING_ROOT / task_id / "task.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return int(pid) > 0
    except Exception:
        return False


def _tail(path: Path, max_bytes: int = 12000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _enrich(state: dict[str, Any], include_log: bool = False) -> dict[str, Any]:
    state = dict(state)
    if state.get("status") in {"queued", "running"} and state.get("pid") and not _pid_alive(state["pid"]):
        latest = _read_json(_state_path(str(state["task_id"]))) or state
        if latest.get("status") not in TERMINAL_STATES:
            latest.update(
                {
                    "status": "unknown",
                    "stage": "unknown",
                    "message": "后台进程已退出，请查看日志。",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _write_json(_state_path(str(state["task_id"])), latest)
        state = latest
    task_id = state.get("task_id")
    if state.get("transcript_path") and Path(state["transcript_path"]).exists():
        state["transcript_url"] = f"/api/meeting-asr/tasks/{task_id}/artifacts/transcript"
        state["result_url"] = f"/api/meeting-asr/tasks/{task_id}/result"
        state["json_download_url"] = f"/api/meeting-asr/tasks/{task_id}/artifacts/json"
    if include_log:
        state["log_tail"] = _tail(Path(state.get("log_path") or ""))
    state.pop("input_path", None)
    return state


def _get_state(task_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"meet_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}", task_id or ""):
        raise HTTPException(400, "Invalid task id")
    state = _read_json(_state_path(task_id))
    if not state:
        raise HTTPException(404, "Task not found")
    return state


@router.post("/tasks")
async def create_task(
    file: UploadFile = File(...),
    title: str = Form(""),
    language: str = Form("zh"),
    speaker_count: int | None = Form(None),
):
    original_filename = Path(file.filename or "meeting.wav").name
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "只支持 MP3 和 WAV 文件")
    if speaker_count is not None and not 1 <= speaker_count <= 15:
        raise HTTPException(400, "发言人数必须在 1 到 15 之间")
    if not MEETING_PIPELINE.exists():
        raise HTTPException(500, f"Meeting pipeline not found: {MEETING_PIPELINE}")

    task_id = f"meet_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    work_dir = MEETING_ROOT / task_id
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    input_path = input_dir / f"recording{suffix}"
    size = 0
    try:
        with input_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"文件不能超过 {MAX_UPLOAD_BYTES // 1024**2} MiB")
                handle.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        try:
            input_dir.rmdir()
            work_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        await file.close()
    if not size:
        raise HTTPException(400, "文件为空")

    now = datetime.now().isoformat(timespec="seconds")
    log_path = work_dir / "task.log"
    state_path = work_dir / "task.json"
    state = {
        "task_id": task_id,
        "status": "queued",
        "stage": "uploaded",
        "title": (title or Path(original_filename).stem).strip()[:160],
        "original_filename": original_filename,
        "content_type": file.content_type or "application/octet-stream",
        "size_bytes": size,
        "language": language if language in {"zh", "en", "auto"} else "zh",
        "speaker_count": speaker_count,
        "created_at": now,
        "updated_at": now,
        "work_dir": str(work_dir),
        "input_path": str(input_path),
        "log_path": str(log_path),
        "message": "文件已上传，正在启动后台转写。",
    }
    _write_json(state_path, state)
    command = ["python3.12", str(MEETING_PIPELINE), "--state", str(state_path)]
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(work_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        log_handle.close()
        state.update({"status": "failed", "stage": "failed", "message": f"启动失败：{exc}"})
        _write_json(state_path, state)
        raise HTTPException(500, state["message"]) from exc
    finally:
        log_handle.close()
    state.update(
        {
            "status": "running",
            "stage": "transcribing",
            "pid": process.pid,
            "command": command,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": "正在远程 ASR 服务上进行语音识别和说话人分离。",
        }
    )
    _write_json(state_path, state)
    return {"ok": True, "task": _enrich(state)}


@router.get("/tasks")
def list_tasks(limit: int = 20):
    paths = sorted(MEETING_ROOT.glob("meet_*/task.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    states = [_read_json(path) for path in paths[: max(1, min(limit, 50))]]
    return {"ok": True, "tasks": [_enrich(state) for state in states if state]}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    return {"ok": True, "task": _enrich(_get_state(task_id), include_log=True)}


@router.get("/tasks/{task_id}/result")
def get_result(task_id: str):
    state = _get_state(task_id)
    path = Path(state.get("result_path") or "")
    if not path.is_file():
        raise HTTPException(404, "Transcription result not found")
    return {"ok": True, "result": _read_json(path)}


@router.get("/tasks/{task_id}/artifacts/{artifact}")
def download_artifact(task_id: str, artifact: str):
    state = _get_state(task_id)
    if artifact == "transcript":
        path = Path(state.get("transcript_path") or "")
        media_type = "text/plain; charset=utf-8"
        filename = f"{state.get('title') or task_id}-转写.txt"
    elif artifact == "json":
        path = Path(state.get("result_path") or "")
        media_type = "application/json"
        filename = f"{state.get('title') or task_id}-转写.json"
    else:
        raise HTTPException(404, "Artifact not found")
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type=media_type, filename=filename)
