#!/usr/bin/env python3
"""Transcribe a prepared podcast manifest through the remote ASR API."""
from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ChunkResult:
    chunk_index: int
    start: float
    end: float
    start_ts: str
    end_ts: str
    duration_seconds: float
    device: str
    model_load_seconds: float
    inference_seconds: float
    wall_seconds: float
    rtf_inference: float | None
    rtf_wall: float
    chars: int
    text: str
    raw_text: str
    output_json: str
    output_txt: str
    error: str | None = None


def _multipart(audio: Path, language: str) -> tuple[bytes, str]:
    boundary = f"----gb10-asr-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in {
        "model": "SenseVoiceSmall",
        "language": language,
        "response_format": "verbose_json",
    }.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    content_type = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    chunks.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{audio.name}\"\r\nContent-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    chunks.append(audio.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def transcribe(api_url: str, audio: Path, language: str, timeout: int) -> dict[str, Any]:
    body, boundary = _multipart(audio, language)
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ASR HTTP {exc.code}: {detail[:1000]}") from exc


def fmt_srt_ts(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def split_sentences(text: str) -> list[str]:
    import re

    return [part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text) if part.strip()]


def render_outputs(
    manifest: dict[str, Any], results: list[ChunkResult], output_dir: Path, device_label: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = [item for item in results if not item.error]
    audio_seconds = sum(item.duration_seconds for item in ok)
    inference_seconds = sum(item.inference_seconds for item in ok)
    wall_seconds = sum(item.wall_seconds for item in ok)
    summary = {
        "chunks": len(results),
        "ok_chunks": len(ok),
        "failed_chunks": len(results) - len(ok),
        "audio_seconds": round(audio_seconds, 3),
        "model_load_seconds": round(sum(item.model_load_seconds for item in ok), 3),
        "inference_seconds": round(inference_seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "rtf_inference": round(inference_seconds / audio_seconds, 4) if audio_seconds else None,
        "rtf_wall": round(wall_seconds / audio_seconds, 4) if audio_seconds else None,
        "x_realtime_inference": round(audio_seconds / inference_seconds, 2) if inference_seconds else None,
        "x_realtime_wall": round(audio_seconds / wall_seconds, 2) if wall_seconds else None,
        "chars": sum(item.chars for item in ok),
    }
    payload = {
        **{key: value for key, value in manifest.items() if key != "chunks"},
        "device": device_label,
        "model": "SenseVoiceSmall",
        "summary": summary,
        "chunks": [asdict(item) for item in results],
    }
    suffix = device_label.replace(":", "_").replace("/", "_")
    json_path = output_dir / f"transcription_{suffix}.json"
    txt_path = output_dir / f"transcript_{suffix}.txt"
    md_path = output_dir / f"transcript_{suffix}.md"
    srt_path = output_dir / f"transcript_{suffix}.srt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text("\n\n".join(f"[{item.start_ts}] {item.text}" for item in ok) + "\n", encoding="utf-8")
    md_lines = [f"# {manifest.get('title', 'Podcast transcript')}", "", f"- **ASR:** SenseVoiceSmall on `{device_label}`", ""]
    srt_lines: list[str] = []
    srt_index = 1
    for item in ok:
        md_lines.extend([f"## [{item.start_ts} - {item.end_ts}]", "", item.text, ""])
        sentences = split_sentences(item.text) or [item.text]
        for index, sentence in enumerate(sentences):
            start = item.start + item.duration_seconds * index / len(sentences)
            end = item.start + item.duration_seconds * (index + 1) / len(sentences)
            srt_lines.extend([str(srt_index), f"{fmt_srt_ts(start)} --> {fmt_srt_ts(end)}", sentence, ""])
            srt_index += 1
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--device-label", default="remote_cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.transcript_dir.mkdir(parents=True, exist_ok=True)
    results: list[ChunkResult] = []
    for number, chunk in enumerate(manifest.get("chunks") or [], start=1):
        index = int(chunk["chunk_index"])
        audio = Path(chunk["file"])
        cache_path = args.transcript_dir / f"chunk_{index:03d}_{args.device_label}.json"
        if cache_path.exists() and not args.force:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not cached.get("error"):
                results.append(ChunkResult(**cached))
                continue
        started = time.perf_counter()
        error = None
        response: dict[str, Any] = {}
        try:
            response = transcribe(args.api_url, audio, args.language, args.timeout)
        except Exception as exc:
            error = repr(exc)
        wall_seconds = time.perf_counter() - started
        duration = float(chunk["duration_seconds"])
        result = ChunkResult(
            chunk_index=index,
            start=float(chunk["start"]),
            end=float(chunk["end"]),
            start_ts=str(chunk["start_ts"]),
            end_ts=str(chunk["end_ts"]),
            duration_seconds=duration,
            device=args.device_label,
            model_load_seconds=float(response.get("model_load_seconds") or 0),
            inference_seconds=float(response.get("inference_seconds") or 0),
            wall_seconds=round(wall_seconds, 3),
            rtf_inference=round(float(response.get("inference_seconds") or 0) / duration, 4) if response else None,
            rtf_wall=round(wall_seconds / duration, 4),
            chars=len(str(response.get("text") or "")),
            text=str(response.get("text") or ""),
            raw_text=str(response.get("raw_text") or ""),
            output_json=str(cache_path),
            output_txt=str(cache_path.with_suffix(".txt")),
            error=error,
        )
        cache_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        cache_path.with_suffix(".txt").write_text(result.text + "\n", encoding="utf-8")
        results.append(result)
        print(f"[{number:03d}/{len(manifest['chunks']):03d}] chunk={index} chars={result.chars} error={error or '-'}", flush=True)
        if error:
            render_outputs(manifest, results, args.output_dir, args.device_label)
            return 1
    output = render_outputs(manifest, results, args.output_dir, args.device_label)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
