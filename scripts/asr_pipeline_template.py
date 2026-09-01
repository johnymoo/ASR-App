#!/usr/bin/env python3
"""SenseVoice podcast ASR pipeline for Xiaoyuzhou episode.

- Uses host Python 3.12 CUDA torch from ~/.local plus FunASR deps from the SenseVoice venv.
- Benchmarks CPU vs GPU on identical audio slices.
- Transcribes the full episode on GPU with resumable chunk outputs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Make FunASR/librosa/soundfile from the previous SenseVoice venv available while
# keeping CUDA torch/torchaudio from ~/.local earlier on sys.path.
SENSEVOICE_SITE = Path(os.environ.get("SENSEVOICE_SITE_PACKAGES", "~/deployments/sensevoice/venv/lib/python3.12/site-packages")).expanduser()
if SENSEVOICE_SITE.exists() and str(SENSEVOICE_SITE) not in sys.path:
    sys.path.append(str(SENSEVOICE_SITE))

BASE = Path("__WORK_DIR__")
INPUT_M4A = BASE / "input" / "episode.m4a"
WAV_16K = BASE / "audio" / "episode_16k.wav"
CHUNK_DIR = BASE / "chunks"
TRANSCRIPT_DIR = BASE / "transcripts"
OUTPUT_DIR = BASE / "output"
LOG_DIR = BASE / "logs"

MODEL_DIR = "__MODEL_DIR__"
PUNC_MODEL_DIR = "__PUNC_MODEL_DIR__"
VAD_MODEL_DIR = "__VAD_MODEL_DIR__"

DEFAULT_CHUNK_SECONDS = 300.0
DEFAULT_OVERLAP_SECONDS = 5.0
EPISODE_ID = "__EPISODE_ID__"
EPISODE_URL = "__EPISODE_URL__"
EPISODE_TITLE = "__EPISODE_TITLE__"

TAG_RE = re.compile(r"<\|[^|]+\|>")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def ensure_dirs() -> None:
    for d in [BASE, BASE / "input", BASE / "audio", CHUNK_DIR, TRANSCRIPT_DIR, OUTPUT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ffprobe_duration(path: Path) -> float:
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(r.stdout.strip())


def ffprobe_size(path: Path) -> int:
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=size",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return int(float(r.stdout.strip()))


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = TAG_RE.sub("", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    # Keep punctuation attached to each sentence.
    parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
    out = [p.strip() for p in parts if p.strip()]
    return out or [text]


def make_slices(duration: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[int, float, float]]:
    slices: list[tuple[int, float, float]] = []
    pos = 0.0
    idx = 0
    while pos < duration - 0.01:
        end = min(pos + chunk_seconds, duration)
        slices.append((idx, pos, end))
        idx += 1
        if end >= duration - 0.01:
            break
        pos = max(0.0, end - overlap_seconds)
    return slices


def ensure_wav() -> dict[str, Any]:
    ensure_dirs()
    if not INPUT_M4A.exists():
        raise FileNotFoundError(f"Missing downloaded audio: {INPUT_M4A}")
    if not WAV_16K.exists() or WAV_16K.stat().st_size == 0:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(INPUT_M4A), "-ac", "1", "-ar", "16000", "-vn",
            "-c:a", "pcm_s16le", "-loglevel", "error", str(WAV_16K)
        ], check=True)
    duration = ffprobe_duration(WAV_16K)
    return {
        "episode_id": EPISODE_ID,
        "url": EPISODE_URL,
        "title": EPISODE_TITLE,
        "source_audio": str(INPUT_M4A),
        "wav_16k": str(WAV_16K),
        "duration_seconds": duration,
        "duration_formatted": fmt_ts(duration),
        "source_size_bytes": INPUT_M4A.stat().st_size,
        "wav_size_bytes": WAV_16K.stat().st_size,
    }


def slice_audio(start: float, end: float, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1024:
        return
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(WAV_16K),
        "-ac", "1", "-ar", "16000", "-vn", "-c:a", "pcm_s16le", "-loglevel", "error", str(out_path)
    ], check=True)


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
    rtf_inference: float
    rtf_wall: float
    chars: int
    text: str
    raw_text: str
    output_json: str
    output_txt: str
    error: str | None = None


class SenseVoiceEngine:
    def __init__(self, device: str, use_vad: bool = True, use_punc_model: bool = False):
        self.device = device
        self.use_vad = use_vad
        self.use_punc_model = use_punc_model
        self.model = None
        self.punc_model = None
        self.load_seconds = 0.0

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from funasr import AutoModel
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": MODEL_DIR,
            "device": self.device,
            "disable_update": True,
            "trust_remote_code": True,
        }
        if self.use_vad and Path(VAD_MODEL_DIR).exists():
            kwargs.update({
                "vad_model": VAD_MODEL_DIR,
                "vad_kwargs": {"max_single_segment_time": 30000},
            })
        self.model = AutoModel(**kwargs)
        if self.use_punc_model and Path(PUNC_MODEL_DIR).exists():
            try:
                self.punc_model = AutoModel(model=PUNC_MODEL_DIR, device=self.device, disable_update=True)
            except Exception as e:  # keep ASR useful if punc model fails
                print(f"WARN punctuation model load failed: {e}", flush=True)
                self.punc_model = None
        self.load_seconds = time.perf_counter() - t0
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def transcribe(self, audio_path: Path, language: str = "zh") -> tuple[str, str, float]:
        import torch
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        self.load()
        assert self.model is not None
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_kwargs: dict[str, Any] = {
            "input": str(audio_path),
            "cache": {},
            "language": language,
            "use_itn": True,
            "batch_size_s": 60,
        }
        if self.use_vad:
            generate_kwargs.update({"merge_vad": True, "merge_length_s": 15})
        res = self.model.generate(**generate_kwargs)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        raw_parts: list[str] = []
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    raw_parts.append(str(item.get("text", "")))
                else:
                    raw_parts.append(str(item))
        else:
            raw_parts.append(str(res))
        raw_text = "\n".join(p for p in raw_parts if p)
        text = rich_transcription_postprocess(raw_text)
        text = clean_text(text)
        if self.punc_model is not None and text:
            try:
                punc_res = self.punc_model.generate(input=text)
                if punc_res and isinstance(punc_res, list):
                    text = clean_text(str(punc_res[0].get("text", text)))
            except Exception as e:
                print(f"WARN punctuation failed: {e}", flush=True)
        return text, raw_text, elapsed

    def close(self) -> None:
        self.model = None
        self.punc_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def transcribe_one(engine: SenseVoiceEngine, chunk_index: int, start: float, end: float, chunk_path: Path, out_prefix: Path, language: str = "zh", force: bool = False) -> ChunkResult:
    out_json = out_prefix.with_suffix(".json")
    out_txt = out_prefix.with_suffix(".txt")
    if out_json.exists() and out_txt.exists() and out_txt.stat().st_size > 0 and not force:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        # Reuse only completed chunk results. A previous CUDA OOM writes an
        # error JSON plus a one-byte txt file; treating that as resumable would
        # permanently poison future retries.
        if not data.get("error"):
            return ChunkResult(**data)
        print(f"retrying previous failed chunk {chunk_index}: {str(data.get('error'))[:160]}", flush=True)
    wall0 = time.perf_counter()
    model_load_before = engine.load_seconds
    text = ""
    raw_text = ""
    err = None
    infer_elapsed = 0.0
    try:
        text, raw_text, infer_elapsed = engine.transcribe(chunk_path, language=language)
    except Exception as e:
        err = repr(e)
        print(f"ERROR chunk {chunk_index}: {err}", flush=True)
    wall_elapsed = time.perf_counter() - wall0
    model_load_delta = max(0.0, engine.load_seconds - model_load_before)
    dur = max(0.001, end - start)
    result = ChunkResult(
        chunk_index=chunk_index,
        start=round(start, 3),
        end=round(end, 3),
        start_ts=fmt_ts(start),
        end_ts=fmt_ts(end),
        duration_seconds=round(end - start, 3),
        device=engine.device,
        model_load_seconds=round(model_load_delta, 3),
        inference_seconds=round(infer_elapsed, 3),
        wall_seconds=round(wall_elapsed, 3),
        rtf_inference=round(infer_elapsed / dur, 4),
        rtf_wall=round(wall_elapsed / dur, 4),
        chars=len(text),
        text=text,
        raw_text=raw_text,
        output_json=str(out_json),
        output_txt=str(out_txt),
        error=err,
    )
    out_txt.write_text(text + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def cmd_prepare(args: argparse.Namespace) -> None:
    info = ensure_wav()
    chunk_seconds = args.chunk_seconds
    overlap_seconds = args.overlap_seconds
    slices = make_slices(info["duration_seconds"], chunk_seconds, overlap_seconds)
    manifest = {
        **info,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "chunk_count": len(slices),
        "created_at": now_iso(),
        "chunks": [
            {
                "chunk_index": i,
                "start": round(s, 3),
                "end": round(e, 3),
                "start_ts": fmt_ts(s),
                "end_ts": fmt_ts(e),
                "duration_seconds": round(e - s, 3),
                "file": str(CHUNK_DIR / f"chunk_{i:03d}.wav"),
            }
            for i, s, e in slices
        ],
    }
    for i, s, e in slices:
        out = CHUNK_DIR / f"chunk_{i:03d}.wav"
        slice_audio(s, e, out)
        if i % 5 == 0 or i == len(slices) - 1:
            print(f"prepared chunk {i+1}/{len(slices)} {fmt_ts(s)}-{fmt_ts(e)}", flush=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "chunk_count": len(slices), "duration": info["duration_formatted"]}, ensure_ascii=False, indent=2))


def summarize_results(results: list[ChunkResult], include_load: bool = False) -> dict[str, Any]:
    ok = [r for r in results if not r.error]
    total_audio = sum(r.duration_seconds for r in ok)
    total_infer = sum(r.inference_seconds for r in ok)
    total_wall = sum(r.wall_seconds for r in ok)
    total_load = sum(r.model_load_seconds for r in ok)
    return {
        "chunks": len(results),
        "ok_chunks": len(ok),
        "failed_chunks": len(results) - len(ok),
        "audio_seconds": round(total_audio, 3),
        "audio_formatted": fmt_ts(total_audio),
        "model_load_seconds": round(total_load, 3),
        "inference_seconds": round(total_infer, 3),
        "wall_seconds": round(total_wall, 3),
        "rtf_inference": round(total_infer / total_audio, 4) if total_audio else None,
        "rtf_wall": round(total_wall / total_audio, 4) if total_audio else None,
        "x_realtime_inference": round(total_audio / total_infer, 2) if total_infer else None,
        "x_realtime_wall": round(total_audio / total_wall, 2) if total_wall else None,
        "chars": sum(r.chars for r in ok),
        "include_load": include_load,
    }


def cmd_benchmark(args: argparse.Namespace) -> None:
    info = ensure_wav()
    bench_seconds = float(args.seconds)
    start = float(args.start)
    end = min(start + bench_seconds, info["duration_seconds"])
    bench_chunk = CHUNK_DIR / f"benchmark_{int(start)}_{int(end)}.wav"
    slice_audio(start, end, bench_chunk)
    results_by_device: dict[str, Any] = {}
    for device in args.devices.split(','):
        device = device.strip()
        if not device:
            continue
        print(f"\n=== benchmark device={device} audio={fmt_ts(start)}-{fmt_ts(end)} ({end-start:.1f}s) ===", flush=True)
        engine = SenseVoiceEngine(device=device, use_vad=not args.no_vad, use_punc_model=args.use_punc_model)
        out_prefix = TRANSCRIPT_DIR / f"benchmark_{device}_{int(start)}_{int(end)}"
        result = transcribe_one(engine, 0, start, end, bench_chunk, out_prefix, language=args.language, force=True)
        engine.close()
        results_by_device[device] = asdict(result)
        print(f"device={device} load={result.model_load_seconds:.2f}s infer={result.inference_seconds:.2f}s wall={result.wall_seconds:.2f}s rtf={result.rtf_inference} chars={result.chars}", flush=True)
    # Derived speedups, CPU baseline if available.
    speedups: dict[str, Any] = {}
    if "cpu" in results_by_device:
        cpu = results_by_device["cpu"]
        for dev, res in results_by_device.items():
            if dev == "cpu":
                continue
            speedups[f"{dev}_vs_cpu_inference"] = round(cpu["inference_seconds"] / res["inference_seconds"], 3) if res["inference_seconds"] else None
            speedups[f"{dev}_vs_cpu_wall"] = round(cpu["wall_seconds"] / res["wall_seconds"], 3) if res["wall_seconds"] else None
    output = {
        "episode_id": EPISODE_ID,
        "title": EPISODE_TITLE,
        "url": EPISODE_URL,
        "created_at": now_iso(),
        "audio_file": str(bench_chunk),
        "audio_start": start,
        "audio_end": end,
        "audio_seconds": round(end - start, 3),
        "devices": results_by_device,
        "speedups": speedups,
        "model": "SenseVoiceSmall",
        "vad_model": VAD_MODEL_DIR if not args.no_vad else None,
    }
    out_path = OUTPUT_DIR / f"benchmark_cpu_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBENCHMARK_JSON", out_path)
    print(json.dumps(speedups, ensure_ascii=False, indent=2))


def cmd_transcribe(args: argparse.Namespace) -> None:
    info = ensure_wav()
    chunk_seconds = args.chunk_seconds
    overlap_seconds = args.overlap_seconds
    slices = make_slices(info["duration_seconds"], chunk_seconds, overlap_seconds)
    if args.limit:
        slices = slices[: args.limit]
    # Ensure needed chunks exist. This is fast if already prepared.
    for i, s, e in slices:
        slice_audio(s, e, CHUNK_DIR / f"chunk_{i:03d}.wav")
    device = args.device
    engine = SenseVoiceEngine(device=device, use_vad=not args.no_vad, use_punc_model=args.use_punc_model)
    run_started = time.perf_counter()
    print(f"Transcribing {len(slices)} chunks on {device}; duration={info['duration_formatted']}; chunk={chunk_seconds}s overlap={overlap_seconds}s", flush=True)
    results: list[ChunkResult] = []
    for n, (i, s, e) in enumerate(slices, start=1):
        chunk_path = CHUNK_DIR / f"chunk_{i:03d}.wav"
        out_prefix = TRANSCRIPT_DIR / f"chunk_{i:03d}_{device}"
        result = transcribe_one(engine, i, s, e, chunk_path, out_prefix, language=args.language, force=args.force)
        results.append(result)
        print(f"[{n:03d}/{len(slices):03d}] {result.start_ts}-{result.end_ts} chars={result.chars} infer={result.inference_seconds:.2f}s wall={result.wall_seconds:.2f}s rtf={result.rtf_inference} err={result.error or '-'}", flush=True)
    engine.close()
    elapsed = time.perf_counter() - run_started
    summary = summarize_results(results)
    summary["pipeline_wall_seconds"] = round(elapsed, 3)
    successful_audio = sum(r.duration_seconds for r in results if not r.error)
    summary["pipeline_rtf"] = round(elapsed / successful_audio, 4) if successful_audio else None
    if results and not successful_audio:
        summary["status"] = "all_chunks_failed"
    full = {
        **info,
        "created_at": now_iso(),
        "device": device,
        "language": args.language,
        "model": "SenseVoiceSmall",
        "model_dir": MODEL_DIR,
        "vad_model": VAD_MODEL_DIR if not args.no_vad else None,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "summary": summary,
        "chunks": [asdict(r) for r in results],
    }
    device_suffix = device.replace(':', '_')
    json_path = OUTPUT_DIR / f"transcription_{device_suffix}.json"
    md_path = OUTPUT_DIR / f"transcript_{device_suffix}.md"
    txt_path = OUTPUT_DIR / f"transcript_{device_suffix}.txt"
    srt_path = OUTPUT_DIR / f"transcript_{device_suffix}.srt"
    full_text_parts = []
    md_lines = [
        f"# {EPISODE_TITLE}",
        "",
        f"- **Episode ID:** `{EPISODE_ID}`",
        f"- **URL:** {EPISODE_URL}",
        f"- **Duration:** {info['duration_formatted']} ({info['duration_seconds']:.1f}s)",
        f"- **ASR:** SenseVoiceSmall on `{device}`",
        f"- **Chunking:** {chunk_seconds:.0f}s chunks, {overlap_seconds:.0f}s overlap",
        f"- **Generated:** {now_iso()}",
        f"- **Successful chunks:** {summary['ok_chunks']}/{summary['chunks']}",
        f"- **Inference RTF:** {summary['rtf_inference']}",
        f"- **Pipeline wall RTF:** {summary['pipeline_rtf']}",
        "",
        "---",
        "",
    ]
    srt_lines: list[str] = []
    srt_index = 1
    for r in results:
        if r.error or not r.text:
            continue
        full_text_parts.append(f"[{r.start_ts}] {r.text}")
        md_lines += [f"## [{r.start_ts} – {r.end_ts}]", "", r.text, ""]
        sentences = split_sentences(r.text)
        if not sentences:
            continue
        span = max(0.1, r.end - r.start)
        for j, sent in enumerate(sentences):
            ss = r.start + span * (j / len(sentences))
            ee = r.start + span * ((j + 1) / len(sentences))
            srt_lines += [str(srt_index), f"{fmt_srt_ts(ss)} --> {fmt_srt_ts(ee)}", sent, ""]
            srt_index += 1
    json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    txt_path.write_text("\n\n".join(full_text_parts) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    print("\nOUTPUT_JSON", json_path)
    print("OUTPUT_MD", md_path)
    print("OUTPUT_TXT", txt_path)
    print("OUTPUT_SRT", srt_path)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("benchmark")
    p.add_argument("--devices", default="cpu,cuda")
    p.add_argument("--start", type=float, default=600.0)
    p.add_argument("--seconds", type=float, default=300.0)
    p.add_argument("--language", default="zh")
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--use-punc-model", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("transcribe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="zh")
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--use-punc-model", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
