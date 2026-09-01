#!/usr/bin/env python3
"""Single-concurrency SenseVoice transcription service."""
from __future__ import annotations

import gc
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

MODEL_DIR = Path(os.environ.get("SENSEVOICE_MODEL_DIR", "/models/SenseVoiceSmall"))
VAD_MODEL_DIR = Path(os.environ.get("SENSEVOICE_VAD_MODEL_DIR", "/models/speech_fsmn_vad"))
SPEAKER_MODEL_DIR = Path(
    os.environ.get(
        "SENSEVOICE_SPEAKER_MODEL_DIR",
        "/cache/modelscope/models/iic/speech_campplus_sv_zh-cn_16k-common",
    )
)
DEVICE = os.environ.get("ASR_DEVICE", "cuda").lower()
MIN_FREE_CUDA_BYTES = int(os.environ.get("ASR_MIN_FREE_CUDA_BYTES", str(1024**3)))
MAX_UPLOAD_BYTES = int(os.environ.get("ASR_MAX_UPLOAD_BYTES", str(256 * 1024**2)))
USE_FP16 = os.environ.get("ASR_FP16", "0") == "1"
MEETING_SEGMENT_SECONDS = float(os.environ.get("ASR_MEETING_SEGMENT_SECONDS", "28"))
FFMPEG_TIMEOUT_SECONDS = int(os.environ.get("ASR_FFMPEG_TIMEOUT_SECONDS", "1800"))
TAG_RE = re.compile(r"<\|[^|]+\|>")

app = FastAPI(title="SenseVoice ASR", version="1.0.0")
_model: Any = None
_speaker_model: Any = None
_model_lock = threading.Lock()


def _cuda_state() -> dict[str, Any]:
    available = torch.cuda.is_available()
    free_bytes = total_bytes = 0
    device = None
    error = None
    if available:
        try:
            device = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except torch.AcceleratorError as exc:
            available = False
            error = str(exc).splitlines()[0]
    return {
        "available": available,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "device": device,
        "error": error,
    }


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if DEVICE not in {"cpu", "cuda"}:
        raise RuntimeError(f"unsupported ASR device: {DEVICE}")
    if DEVICE == "cuda":
        state = _cuda_state()
        if not state["available"]:
            raise RuntimeError("CUDA is not available")
        if state["free_bytes"] < MIN_FREE_CUDA_BYTES:
            raise RuntimeError(
                f"insufficient CUDA memory: free={state['free_bytes']} required={MIN_FREE_CUDA_BYTES}"
            )
    if not MODEL_DIR.is_dir():
        raise RuntimeError(f"model directory not found: {MODEL_DIR}")

    from funasr import AutoModel
    from funasr.models.sense_voice import model as sensevoice_model

    use_fp16 = USE_FP16 and DEVICE == "cuda"
    if use_fp16 and not getattr(sensevoice_model.extract_fbank, "_gb10_fp16", False):
        original_extract_fbank = sensevoice_model.extract_fbank

        def extract_fbank_fp16(*args, **kwargs):
            speech, lengths = original_extract_fbank(*args, **kwargs)
            return speech.to(torch.float16), lengths

        extract_fbank_fp16._gb10_fp16 = True
        sensevoice_model.extract_fbank = extract_fbank_fp16

    kwargs: dict[str, Any] = {
        "model": str(MODEL_DIR),
        "device": DEVICE,
        "fp16": use_fp16,
        "disable_update": True,
        "trust_remote_code": True,
        "ncpu": int(os.environ.get("ASR_CPU_THREADS", "8")),
        "disable_pbar": True,
    }
    if DEVICE == "cpu" and VAD_MODEL_DIR.is_dir():
        kwargs.update(
            {
                "vad_model": str(VAD_MODEL_DIR),
                "vad_kwargs": {"max_single_segment_time": 30000},
            }
        )
    _model = AutoModel(**kwargs)
    if DEVICE == "cuda" and VAD_MODEL_DIR.is_dir():
        vad = AutoModel(
            model=str(VAD_MODEL_DIR),
            device="cpu",
            disable_update=True,
            trust_remote_code=True,
            ncpu=2,
        )
        _model.vad_model = vad.model
        _model.vad_kwargs = vad.kwargs
        _model._store_base_configs()
    attention_class = sensevoice_model.MultiHeadedAttentionSANM
    if use_fp16 and not getattr(attention_class.forward_fsmn, "_gb10_mixed_dtype", False):
        def forward_fsmn_mixed(self, inputs, mask, mask_shfit_chunk=None):
            batch, _, _ = inputs.size()
            if mask is not None:
                mask = torch.reshape(mask, (batch, -1, 1)).to(inputs.dtype)
                if mask_shfit_chunk is not None:
                    mask = mask * mask_shfit_chunk.to(inputs.dtype)
                inputs = inputs * mask
            x = self.pad_fn(inputs.transpose(1, 2))
            x = self.fsmn_block(x)
            x = self.dropout(x.transpose(1, 2) + inputs)
            return x * mask if mask is not None else x

        forward_fsmn_mixed._gb10_mixed_dtype = True
        attention_class.forward_fsmn = forward_fsmn_mixed
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return _model


def _load_speaker_model() -> Any:
    global _speaker_model
    if _speaker_model is not None:
        return _speaker_model
    if not SPEAKER_MODEL_DIR.is_dir():
        raise RuntimeError(f"speaker model directory not found: {SPEAKER_MODEL_DIR}")
    from funasr import AutoModel

    _speaker_model = AutoModel(
        model=str(SPEAKER_MODEL_DIR),
        device=DEVICE,
        disable_update=True,
        trust_remote_code=False,
        ncpu=int(os.environ.get("ASR_CPU_THREADS", "8")),
        disable_pbar=True,
    )
    return _speaker_model


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub("", text or "")).strip()


def _normalize_meeting_audio(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("audio normalization timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffmpeg failed").strip().splitlines()[-1]
        raise RuntimeError(f"invalid or unsupported audio: {detail}") from exc


def _split_speaker_timeline(timeline: list[list[Any]]) -> list[dict[str, Any]]:
    pieces: list[dict[str, Any]] = []
    for start, end, speaker in timeline:
        cursor = float(start)
        end = float(end)
        while end - cursor > 0.35:
            piece_end = min(end, cursor + MEETING_SEGMENT_SECONDS)
            pieces.append({"start": cursor, "end": piece_end, "speaker": int(speaker)})
            cursor = piece_end
    return pieces


def _merge_speaker_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in segments:
        text = _clean_text(str(segment.get("text") or ""))
        if not text:
            continue
        current = {
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "speaker": int(segment["speaker"]),
            "text": text,
        }
        previous = merged[-1] if merged else None
        if (
            previous
            and previous["speaker"] == current["speaker"]
            and current["start"] - previous["end"] <= 0.8
            and current["end"] - previous["start"] <= 90
        ):
            previous["end"] = current["end"]
            previous["text"] = f'{previous["text"]}{current["text"]}'
        else:
            merged.append(current)
    return merged


def _speaker_embeddings(speaker_model: Any, windows: list[list[Any]]) -> Any:
    import torch

    results = speaker_model.inference(
        [window[2] for window in windows],
        model=speaker_model.model,
        kwargs=speaker_model.kwargs,
        batch_size=32,
    )
    batches = [item["spk_embedding"].detach().cpu() for item in results]
    if not batches:
        raise RuntimeError("speaker model returned no embeddings")
    return torch.cat(batches, dim=0)


def _transcribe_meeting(path: Path, language: str, speaker_count: int | None) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from funasr.models.campplus.cluster_backend import ClusterBackend
    from funasr.models.campplus.utils import postprocess, sv_chunk

    with _model_lock:
        load_started = time.perf_counter()
        model = _load_model()
        speaker_model = _load_speaker_model()
        load_seconds = time.perf_counter() - load_started

        waveform, sample_rate = sf.read(str(path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            raise RuntimeError(f"normalized audio has unexpected sample rate: {sample_rate}")
        duration_seconds = len(waveform) / sample_rate

        started = time.perf_counter()
        vad_results = model.inference(
            str(path),
            model=model.vad_model,
            kwargs=model.vad_kwargs,
        )
        vad_intervals = vad_results[0].get("value", []) if vad_results else []
        vad_segments = [
            [
                start_ms / 1000.0,
                end_ms / 1000.0,
                waveform[int(start_ms * sample_rate / 1000) : int(end_ms * sample_rate / 1000)],
            ]
            for start_ms, end_ms in vad_intervals
            if end_ms > start_ms
        ]
        if not vad_segments:
            return {
                "text": "",
                "segments": [],
                "speaker_count": 0,
                "duration_seconds": round(duration_seconds, 3),
                "model": "SenseVoiceSmall",
                "diarization_model": "CAM++",
                "device": DEVICE,
                "model_load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.perf_counter() - started, 3),
            }

        windows = sv_chunk(vad_segments, fs=sample_rate)
        embeddings = _speaker_embeddings(speaker_model, windows)
        cluster = ClusterBackend(merge_thr=0.78)
        if speaker_count is not None and embeddings.shape[0] >= speaker_count:
            labels = cluster.spectral_cluster(embeddings.numpy(), speaker_count)
        else:
            labels = cluster(embeddings)
        timeline = (
            [[float(windows[0][0]), float(windows[0][1]), 0]]
            if len(windows) == 1
            else postprocess(windows, vad_segments, labels, embeddings.numpy())
        )
        pieces = _split_speaker_timeline(timeline)

        raw_segments: list[dict[str, Any]] = []
        for piece in pieces:
            start_sample = max(0, int((piece["start"] - 0.1) * sample_rate))
            end_sample = min(len(waveform), int((piece["end"] + 0.1) * sample_rate))
            audio = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
            result = model.generate(
                input=audio,
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            raw_text = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in (result if isinstance(result, list) else [result])
            )
            raw_segments.append({**piece, "text": raw_text})

        segments = _merge_speaker_segments(raw_segments)
        inference_seconds = time.perf_counter() - started
        return {
            "text": "\n".join(segment["text"] for segment in segments),
            "segments": segments,
            "speaker_count": len({segment["speaker"] for segment in segments}),
            "duration_seconds": round(duration_seconds, 3),
            "model": "SenseVoiceSmall",
            "diarization_model": "CAM++",
            "device": DEVICE,
            "model_load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
        }


def _transcribe(path: Path, language: str) -> dict[str, Any]:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    with _model_lock:
        load_started = time.perf_counter()
        model = _load_model()
        load_seconds = time.perf_counter() - load_started
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.generate(
            input=str(path),
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=1 if USE_FP16 and DEVICE == "cuda" else 60,
            merge_vad=True,
            merge_length_s=15,
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started

    raw_parts = []
    for item in result if isinstance(result, list) else [result]:
        raw_parts.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
    raw_text = "\n".join(part for part in raw_parts if part)
    return {
        "text": _clean_text(rich_transcription_postprocess(raw_text)),
        "raw_text": raw_text,
        "model": "SenseVoiceSmall",
        "device": DEVICE,
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "cuda": _cuda_state() if DEVICE == "cuda" else None,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "device": DEVICE, "model_loaded": _model is not None}


@app.get("/readyz")
def readyz() -> JSONResponse:
    state = _cuda_state() if DEVICE == "cuda" else None
    models_ready = MODEL_DIR.is_dir() and VAD_MODEL_DIR.is_dir() and SPEAKER_MODEL_DIR.is_dir()
    ok = models_ready and (
        DEVICE == "cpu"
        or bool(state and state["available"] and state["free_bytes"] >= MIN_FREE_CUDA_BYTES)
    )
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "ok": ok,
            "device": DEVICE,
            "model_loaded": _model is not None,
            "model_dir": str(MODEL_DIR),
            "vad_model_dir": str(VAD_MODEL_DIR),
            "speaker_model_dir": str(SPEAKER_MODEL_DIR),
            "models_ready": models_ready,
            "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
            "cuda": state,
        },
    )


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("SenseVoiceSmall"),
    language: str = Form("zh"),
    response_format: str = Form("verbose_json"),
):
    if model.lower() not in {"sensevoicesmall", "sensevoice-small", "sensevoice"}:
        raise HTTPException(400, f"unsupported model: {model}")
    suffix = Path(file.filename or "audio.wav").suffix[:12] or ".wav"
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty audio file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"audio exceeds {MAX_UPLOAD_BYTES} bytes")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="asr-", suffix=suffix, delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        result = await run_in_threadpool(_transcribe, temp_path, language)
    except RuntimeError as exc:
        logging.exception("ASR runtime failure")
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(503, str(exc)) from exc
    except torch.AcceleratorError as exc:
        logging.exception("ASR CUDA failure")
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(503, f"CUDA transcription failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(400, f"unsupported response_format: {response_format}")
    return result if response_format == "verbose_json" else {"text": result["text"]}


@app.post("/v1/audio/meeting-transcriptions")
async def meeting_transcriptions(
    file: UploadFile = File(...),
    language: str = Form("zh"),
    speaker_count: int | None = Form(None),
):
    if speaker_count is not None and not 1 <= speaker_count <= 15:
        raise HTTPException(400, "speaker_count must be between 1 and 15")
    suffix = Path(file.filename or "meeting.wav").suffix.lower()[:12] or ".wav"
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty audio file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"audio exceeds {MAX_UPLOAD_BYTES} bytes")

    source_path: Path | None = None
    wav_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="meeting-source-", suffix=suffix, delete=False) as handle:
            handle.write(data)
            source_path = Path(handle.name)
        with tempfile.NamedTemporaryFile(prefix="meeting-audio-", suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        await run_in_threadpool(_normalize_meeting_audio, source_path, wav_path)
        result = await run_in_threadpool(_transcribe_meeting, wav_path, language, speaker_count)
    except RuntimeError as exc:
        logging.exception("meeting transcription failure")
        gc.collect()
        raise HTTPException(503, str(exc)) from exc
    finally:
        if source_path is not None:
            source_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    return result
