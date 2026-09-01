#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DATASET_DIR = Path(os.environ.get("ASR_EVAL_BASE", BENCHMARK_DIR / "dataset")).expanduser()
AUDIO_DIR = Path(os.environ.get("ASR_EVAL_AUDIO_DIR", DATASET_DIR / "audio")).expanduser()
REF_JSON = Path(os.environ.get("ASR_EVAL_REF_JSON", DATASET_DIR / "baseline-nemotron-results.json")).expanduser()
DEVICE = os.environ.get("ASR_EVAL_DEVICE", "cuda")
OUT_DIR = Path(os.environ.get("ASR_EVAL_OUT", BENCHMARK_DIR / "results" / f"sensevoice-small-{DEVICE}")).expanduser()

SENSEVOICE_SITE = Path(os.environ.get("SENSEVOICE_SITE_PACKAGES", "~/deployments/sensevoice/venv/lib/python3.12/site-packages")).expanduser()
if SENSEVOICE_SITE.exists() and str(SENSEVOICE_SITE) not in sys.path:
    sys.path.append(str(SENSEVOICE_SITE))

MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", str(Path("~/deployments/sensevoice/models/SenseVoiceSmall").expanduser()))
VAD_MODEL_DIR = os.environ.get("SENSEVOICE_VAD_MODEL_DIR", str(Path("~/deployments/sensevoice-docker/modelscope-cache/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch").expanduser()))
LANGUAGE = os.environ.get("ASR_EVAL_LANGUAGE", "auto")
USE_VAD = os.environ.get("ASR_EVAL_USE_VAD", "1") != "0"
MERGE_VAD = os.environ.get("ASR_EVAL_MERGE_VAD", "1") != "0"
BATCH_SIZE_S = int(float(os.environ.get("ASR_EVAL_BATCH_SIZE_S", "60")))

TAG_RE = re.compile(r"<\|[^|]+\|>")


def clean_text(text: str) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        text = rich_transcription_postprocess(text or "")
    except Exception:
        pass
    text = TAG_RE.sub("", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_word_char(ch: str) -> bool:
    cat = __import__("unicodedata").category(ch)
    return ch.isalnum() or cat.startswith("L") or cat.startswith("N")


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize using the benchmark package's TER convention.

    This matches `asr-eval-100/results.json`: CJK characters are individual
    tokens; runs of Latin/number/Unicode letter characters are word tokens;
    punctuation is deleted rather than converted to separators, so `code-like`
    becomes `codelike`.
    """
    text = clean_text(text).lower()
    out: list[str] = []
    for ch in text:
        if is_cjk(ch):
            out.extend([" ", ch, " "])
        elif ch.isspace():
            out.append(" ")
        elif is_word_char(ch):
            out.append(ch)
        else:
            # Drop punctuation with no separator; matches provided baseline TER.
            pass
    return [tok for tok in "".join(out).split() if tok]


def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def token_error_rate(ref: str, hyp: str) -> tuple[float, int, int, list[str], list[str]]:
    rt = tokenize_mixed(ref)
    ht = tokenize_mixed(hyp)
    denom = max(1, len(rt))
    dist = edit_distance(rt, ht)
    return dist / denom, dist, len(rt), rt, ht


def main() -> int:
    import torch
    from funasr import AutoModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = json.loads(REF_JSON.read_text(encoding="utf-8"))
    print(f"cases={len(cases)} device={DEVICE} model={MODEL_DIR} vad={USE_VAD} language={LANGUAGE}", flush=True)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    kwargs: dict[str, Any] = {
        "model": MODEL_DIR,
        "device": DEVICE,
        "disable_update": True,
        "trust_remote_code": True,
    }
    if USE_VAD and Path(VAD_MODEL_DIR).exists():
        kwargs.update({
            "vad_model": VAD_MODEL_DIR,
            "vad_kwargs": {"max_single_segment_time": 30000},
        })

    t_load0 = time.perf_counter()
    model = AutoModel(**kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t_load0
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    rows: list[dict[str, Any]] = []
    t_all0 = time.perf_counter()
    for idx, case in enumerate(cases, 1):
        cid = str(case["id"])
        audio = AUDIO_DIR / f"{cid}.wav"
        if not audio.exists():
            raise FileNotFoundError(audio)
        gen_kwargs: dict[str, Any] = {
            "input": str(audio),
            "cache": {},
            "language": LANGUAGE,
            "use_itn": True,
            "batch_size_s": BATCH_SIZE_S,
        }
        if USE_VAD and MERGE_VAD:
            gen_kwargs.update({"merge_vad": True, "merge_length_s": 15})

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        err = None
        res: Any = None
        try:
            res = model.generate(**gen_kwargs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            err = repr(e)
            if DEVICE == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        latency = time.perf_counter() - t0

        raw_parts: list[str] = []
        if res is not None:
            if isinstance(res, list):
                for item in res:
                    raw_parts.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
            else:
                raw_parts.append(str(res))
        raw = "\n".join(p for p in raw_parts if p)
        hyp = clean_text(raw)
        ter, edits, ref_tokens, rt, ht = token_error_rate(case["reference"], hyp)
        audio_seconds = float(case.get("audio_seconds") or 0)
        row = {
            "id": cid,
            "category": case.get("category"),
            "voice": case.get("voice"),
            "reference": case.get("reference"),
            "raw_transcript": raw,
            "processed_transcript": hyp,
            "token_error_rate": round(ter, 6),
            "edit_distance": edits,
            "ref_tokens": ref_tokens,
            "audio_seconds": audio_seconds,
            "latency_seconds": round(latency, 6),
            "rtf": round(latency / audio_seconds, 6) if audio_seconds else None,
            "error": err,
        }
        rows.append(row)
        print(f"[{idx:03d}/{len(cases):03d}] {cid} ter={ter*100:5.1f}% latency={latency:.3f}s chars={len(hyp)} err={err or '-'}", flush=True)

    wall = time.perf_counter() - t_all0
    ok = [r for r in rows if not r.get("error")]
    ters = [float(r["token_error_rate"]) for r in ok]
    lats = [float(r["latency_seconds"]) for r in ok]
    audio_seconds = sum(float(r.get("audio_seconds") or 0) for r in ok)
    latency_total = sum(lats)
    category: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        groups[str(r.get("category") or "")].append(r)
    for cat, rs in groups.items():
        category[cat] = {
            "cases": len(rs),
            "avg_ter": round(statistics.mean(float(r["token_error_rate"]) for r in rs), 6),
            "avg_latency_seconds": round(statistics.mean(float(r["latency_seconds"]) for r in rs), 6),
            "exact": sum(1 for r in rs if float(r["token_error_rate"]) == 0.0),
        }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "SenseVoiceSmall",
        "model_dir": MODEL_DIR,
        "vad_model": VAD_MODEL_DIR if USE_VAD else None,
        "device": DEVICE,
        "language": LANGUAGE,
        "cases": len(rows),
        "ok_cases": len(ok),
        "failed_cases": len(rows) - len(ok),
        "model_load_seconds": round(load_seconds, 6),
        "elapsed_wall_seconds": round(wall, 6),
        "audio_seconds_total": round(audio_seconds, 6),
        "latency_seconds_total": round(latency_total, 6),
        "avg_token_error_rate": round(statistics.mean(ters), 6) if ters else None,
        "median_token_error_rate": round(statistics.median(ters), 6) if ters else None,
        "avg_latency_seconds": round(statistics.mean(lats), 6) if lats else None,
        "median_latency_seconds": round(statistics.median(lats), 6) if lats else None,
        "exact_token_matches": sum(1 for r in ok if float(r["token_error_rate"]) == 0.0),
        "token_error_rate_le_10_pct": sum(1 for r in ok if float(r["token_error_rate"]) <= 0.10),
        "token_error_rate_gt_35_pct": sum(1 for r in ok if float(r["token_error_rate"]) > 0.35),
        "category_summary": category,
        "notes": "TER matches asr-eval-100 tokenization: CJK characters are tokens; runs of Unicode letters/numbers are word tokens; punctuation is removed without adding separators. Latency is model.generate wall time per file, excluding one-time model load.",
    }
    worst = sorted(ok, key=lambda r: float(r["token_error_rate"]), reverse=True)[:15]
    output = {"summary": summary, "rows": rows, "worst_cases": worst}

    json_path = OUT_DIR / f"sensevoice_asr_eval_100_{DEVICE}.json"
    csv_path = OUT_DIR / f"sensevoice_asr_eval_100_{DEVICE}.csv"
    md_path = OUT_DIR / f"sensevoice_asr_eval_100_{DEVICE}.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# SenseVoiceSmall ASR Eval 100",
        "",
        f"- Model: SenseVoiceSmall ({MODEL_DIR})",
        f"- Device: {DEVICE}",
        f"- Cases: {summary['ok_cases']}/{summary['cases']}",
        f"- Avg TER: {summary['avg_token_error_rate']*100:.2f}%" if summary['avg_token_error_rate'] is not None else "- Avg TER: n/a",
        f"- Avg Latency: {summary['avg_latency_seconds']:.3f}s" if summary['avg_latency_seconds'] is not None else "- Avg Latency: n/a",
        f"- Total wall: {summary['elapsed_wall_seconds']:.3f}s",
        "",
        "## Category Summary",
        "",
        "| Category | Cases | Avg TER | Avg Latency | Exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, item in sorted(category.items()):
        md_lines.append(f"| {cat} | {item['cases']} | {item['avg_ter']*100:.1f}% | {item['avg_latency_seconds']:.3f}s | {item['exact']}/{item['cases']} |")
    md_lines += ["", "## Worst Cases", ""]
    for r in worst:
        md_lines += [
            f"### {r['id']} - {r['category']} - TER {float(r['token_error_rate'])*100:.1f}%",
            "",
            f"- Reference: {r['reference']}",
            f"- Hypothesis: {r['processed_transcript']}",
            f"- Latency: {float(r['latency_seconds']):.3f}s",
            "",
        ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("RESULT_JSON", json_path, flush=True)
    print("RESULT_CSV", csv_path, flush=True)
    print("RESULT_MD", md_path, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
