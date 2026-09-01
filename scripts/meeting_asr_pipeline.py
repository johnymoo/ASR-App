#!/usr/bin/env python3
"""Run one meeting transcription task against the x570 ASR service."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ASR_ENDPOINT = os.environ.get(
    "MEETING_ASR_ENDPOINT",
    "http://127.0.0.1:18021/v1/audio/meeting-transcriptions",
)
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("MEETING_ASR_TIMEOUT_SECONDS", "21600"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _update_state(path: Path, **values: Any) -> dict[str, Any]:
    state = _read_json(path)
    state.update(values)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(path, state)
    return state


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _format_transcript(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        speaker = int(segment.get("speaker", 0)) + 1
        timestamp = _format_timestamp(float(segment.get("start", 0)))
        text = str(segment.get("text") or "").strip()
        if text:
            blocks.append(f"发言人 {speaker} {timestamp}\n{text}")
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def run(state_path: Path) -> None:
    state = _update_state(
        state_path,
        status="running",
        stage="transcribing",
        started_at=datetime.now().isoformat(timespec="seconds"),
        message="正在远程 ASR 服务上进行语音识别和说话人分离。",
    )
    input_path = Path(state["input_path"])
    output_dir = Path(state["work_dir"]) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    form = {"language": state.get("language") or "zh"}
    if state.get("speaker_count"):
        form["speaker_count"] = str(state["speaker_count"])

    print(f"POST {ASR_ENDPOINT}", flush=True)
    with input_path.open("rb") as audio:
        response = requests.post(
            ASR_ENDPOINT,
            data=form,
            files={"file": (state["original_filename"], audio, state.get("content_type") or "application/octet-stream")},
            timeout=(15, REQUEST_TIMEOUT_SECONDS),
        )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"remote ASR returned HTTP {response.status_code}: {detail}")

    result = response.json()
    segments = result.get("segments") or []
    transcript = _format_transcript(segments)
    transcript_path = output_dir / "transcript.txt"
    json_path = output_dir / "transcription.json"
    transcript_path.write_text(transcript, encoding="utf-8")
    result.update(
        {
            "task_id": state["task_id"],
            "title": state["title"],
            "original_filename": state["original_filename"],
            "created_at": state["created_at"],
        }
    )
    _write_json(json_path, result)

    ended_at = datetime.now().isoformat(timespec="seconds")
    _update_state(
        state_path,
        status="completed",
        stage="completed",
        ended_at=ended_at,
        message="会议转写已完成。",
        duration_seconds=result.get("duration_seconds"),
        inference_seconds=result.get("inference_seconds"),
        segment_count=len(segments),
        detected_speaker_count=result.get("speaker_count"),
        transcript_path=str(transcript_path),
        result_path=str(json_path),
    )
    print(
        f"completed segments={len(segments)} speakers={result.get('speaker_count')} "
        f"duration={result.get('duration_seconds')} inference={result.get('inference_seconds')}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.state)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _update_state(
            args.state,
            status="failed",
            stage="failed",
            ended_at=datetime.now().isoformat(timespec="seconds"),
            message=f"会议转写失败：{exc}",
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
