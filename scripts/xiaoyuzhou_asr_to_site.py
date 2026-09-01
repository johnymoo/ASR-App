#!/usr/bin/env python3
"""End-to-end Xiaoyuzhou podcast ASR pipeline for GB10.

Given a Xiaoyuzhou episode URL, this script creates a standard workspace,
downloads audio, runs SenseVoice CPU/GPU benchmark and full CUDA transcription,
summarizes the transcript with the local vLLM Qwen endpoint on port 8004, and
publishes the result to the SenseVoice static website.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PODCAST_ROOT = Path(os.environ.get("PODCAST_ROOT", "~/podcast")).expanduser()
ASR_TEMPLATE = Path(os.environ.get("PODCAST_ASR_TEMPLATE", str(PODCAST_ROOT / "asr_pipeline_template.py"))).expanduser()
SUMMARY_SCRIPT = Path(os.environ.get("PODCAST_SUMMARY_SCRIPT", str(PODCAST_ROOT / "generate_podcast_summary.py"))).expanduser()
TLDR_IMAGE_SCRIPT = Path(os.environ.get("PODCAST_TLDR_IMAGE_SCRIPT", str(PODCAST_ROOT / "generate_podcast_tldr_infographic.py"))).expanduser()
PUBLISH_SCRIPT = Path(os.environ.get("PODCAST_PUBLISH_SCRIPT", str(PODCAST_ROOT / "publish_podcast_asr_site.py"))).expanduser()
WIKI_EXPORT_SCRIPT = Path(os.environ.get("PODCAST_WIKI_EXPORT_SCRIPT", str(PODCAST_ROOT / "export_podcast_summary_to_wiki.py"))).expanduser()
DEFAULT_WIKI_PATH = Path(os.environ.get("WIKI_PATH", "~/wiki")).expanduser()
DEFAULT_MODEL_DIR = str(Path(os.environ.get("SENSEVOICE_MODEL_DIR", "~/deployments/sensevoice/models/SenseVoiceSmall")).expanduser())
DEFAULT_PUNC_MODEL_DIR = str(Path(os.environ.get("SENSEVOICE_PUNC_MODEL_DIR", "~/deployments/sensevoice/models/punc-ct-transformer")).expanduser())
DEFAULT_VAD_MODEL_DIR = str(Path(os.environ.get("SENSEVOICE_VAD_MODEL_DIR", "~/deployments/sensevoice-docker/modelscope-cache/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")).expanduser())
DEFAULT_LLM_API_BASE = "http://127.0.0.1:8004/v1"
DEFAULT_LLM_MODEL = "qwen3.6-35b-fp8"
DEFAULT_SITE_BASE_LAN = os.environ.get("PODCAST_ASR_SITE_BASE", "http://127.0.0.1:8020/static/podcast-asr")
DEFAULT_ASR_API_URL = os.environ.get("PODCAST_ASR_API_URL", "")
REMOTE_ASR_CLIENT = Path(
    os.environ.get("PODCAST_REMOTE_ASR_CLIENT", str(PODCAST_ROOT / "remote_asr_client.py"))
).expanduser()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], log_path: Path | None = None, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    started = datetime.now().isoformat(timespec="seconds")
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {started}\n$ {' '.join(cmd)}\n\n")
            f.write(proc.stdout or "")
            f.write(f"\n[exit_code={proc.returncode}]\n")
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def parse_episode_id(url_or_id: str) -> str:
    m = re.search(r"/episode/([0-9a-fA-F]+)", url_or_id)
    if m:
        return m.group(1)
    m = re.fullmatch(r"[0-9a-fA-F]{12,}", url_or_id.strip())
    if m:
        return m.group(0)
    raise ValueError(f"Cannot parse Xiaoyuzhou episode id from: {url_or_id}")


def sanitize_slug(text: str, fallback: str) -> str:
    text = (text or "").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", text)
    ascii_text = re.sub(r"-+", "-", ascii_text).strip("-")
    if ascii_text and (re.search(r"[a-z]", ascii_text) or len(ascii_text) >= 8):
        return ascii_text[:88].strip("-") or fallback
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text or re.fullmatch(r"[\u4e00-\u9fff-]+", text):
        return fallback
    return text[:88].strip("-") or fallback


def decode_web_text(s: str) -> str:
    """Decode HTML/JSON-ish text without corrupting already-decoded UTF-8."""
    return html.unescape(s or "").replace("\\/", "/")


def iter_json_ld(page: str) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for block in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', page, re.S | re.I):
        try:
            data = json.loads(html.unescape(block).strip())
        except Exception:
            continue
        if isinstance(data, dict):
            objs.append(data)
        elif isinstance(data, list):
            objs.extend(x for x in data if isinstance(x, dict))
    return objs


def meta_content(page: str, key: str) -> str:
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(key)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(key)}["\']',
        rf'<meta[^>]+name=["\']{re.escape(key)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{re.escape(key)}["\']',
    ]
    for pat in patterns:
        m = re.search(pat, page, re.S | re.I)
        if m:
            return re.sub(r"\s+", " ", decode_web_text(m.group(1))).strip()
    return ""


def parse_outline(description: str) -> list[dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", line).strip(" _") for line in (description or "").splitlines()]
    start = -1
    for i, line in enumerate(lines):
        if re.match(r"OUTLINE[:：]?", line, re.I):
            start = i + 1
            break
    if start < 0:
        return []
    outline: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines[start:]:
        if not line:
            continue
        if re.match(r"(?:LINKS|DISCLAIMER|CONTACT)[:：]?", line, re.I):
            break
        m = re.match(r"(?P<timestamp>\d{1,2}:\d{2}:\d{2})\s*(?P<title>.*)", line)
        if m:
            current = {"timestamp": m.group("timestamp"), "title": m.group("title").strip(), "notes": []}
            outline.append(current)
        elif current is not None:
            current.setdefault("notes", []).append(line)
    return outline


def render_page_context_markdown(context: dict[str, Any]) -> str:
    lines: list[str] = []
    title = context.get("official_title") or context.get("title") or "Xiaoyuzhou episode"
    lines += [f"# {title}", ""]
    for key, label in [
        ("podcast_title", "播客"),
        ("source_url", "链接"),
        ("published_time", "发布时间"),
        ("duration_text", "官方时长"),
        ("audio_url", "音频"),
        ("cover_image", "封面"),
    ]:
        value = context.get(key)
        if value:
            lines.append(f"- **{label}:** {value}")
    if len(lines) > 2:
        lines.append("")
    if context.get("description"):
        lines += ["## 官方简介 / Show Notes", "", str(context["description"]).strip(), ""]
    outline = context.get("outline") or []
    if outline:
        lines += ["## 官方 OUTLINE", ""]
        for item in outline:
            if isinstance(item, dict):
                lines.append(f"- **{item.get('timestamp', '')}** {item.get('title', '')}".rstrip())
                for note in item.get("notes") or []:
                    lines.append(f"  - {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def fetch_public_metadata(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        page = resp.read().decode("utf-8", errors="ignore")

    json_ld = iter_json_ld(page)
    episode_ld = next((obj for obj in json_ld if obj.get("@type") == "PodcastEpisode"), {})
    title = str(episode_ld.get("name") or meta_content(page, "og:title") or "")
    if not title:
        m = re.search(r"<title>(.*?)</title>", page, re.S | re.I)
        if m:
            title = re.sub(r"\s+", " ", decode_web_text(m.group(1))).strip()
            title = re.sub(r"\s*[-|]\s*.*?小宇宙.*$", "", title).strip()

    description = str(episode_ld.get("description") or meta_content(page, "description") or "")
    part = episode_ld.get("partOfSeries") if isinstance(episode_ld.get("partOfSeries"), dict) else {}
    podcast_title = str(part.get("name") or "")
    published_time = str(episode_ld.get("datePublished") or "")
    duration_text = str(episode_ld.get("timeRequired") or "")
    cover_image = meta_content(page, "og:image") or meta_content(page, "twitter:image")

    audio_url = meta_content(page, "og:audio")
    media = episode_ld.get("associatedMedia")
    media_items = media if isinstance(media, list) else ([media] if isinstance(media, dict) else [])
    for item in media_items:
        if not isinstance(item, dict):
            continue
        audio_url = audio_url or str(item.get("contentUrl") or item.get("url") or "")
    if not audio_url:
        for pat in [
            r"https://media\.xyzcdn\.net/[^\"'<>\s]+?\.(?:m4a|mp3)(?:\?[^\"'<>\s]+)?",
            r'"(?:url|audioUrl|mediaUrl|contentUrl)"\s*:\s*"(https://[^"<>]+?\.(?:m4a|mp3)(?:\?[^"<>]+)?)"',
        ]:
            m = re.search(pat, page)
            if m:
                audio_url = m.group(1) if m.lastindex else m.group(0)
                break
    audio_url = decode_web_text(audio_url)

    page_context: dict[str, Any] = {
        "source_url": url,
        "official_title": title,
        "podcast_title": podcast_title,
        "description": description,
        "published_time": published_time,
        "duration_text": duration_text,
        "cover_image": cover_image,
        "audio_url": audio_url,
        "outline": parse_outline(description),
    }
    page_context = {k: v for k, v in page_context.items() if v not in ("", [], None)}
    metadata: dict[str, Any] = {"title": title, "audio_url": audio_url, "page_context": page_context}
    if podcast_title:
        metadata["podcast"] = podcast_title
    if published_time:
        metadata["published_time"] = published_time
    return metadata


def try_opencli_metadata_and_download(episode_id: str, work_dir: Path) -> dict[str, str]:
    """Optional authenticated fallback via local opencli; returns paths if successful."""
    meta: dict[str, str] = {}
    if not shutil.which("opencli"):
        return meta
    try:
        out = subprocess.check_output(["opencli", "xiaoyuzhou", "episode", episode_id, "-f", "json"], text=True, timeout=60)
        data = json.loads(out)
        row = data[0] if isinstance(data, list) and data else data
        if isinstance(row, dict):
            meta["title"] = str(row.get("title") or "")
            meta["podcast"] = str(row.get("podcast") or "")
    except Exception:
        pass
    download_dir = work_dir / "opencli-download"
    try:
        subprocess.run(["opencli", "xiaoyuzhou", "download", episode_id, "--output", str(download_dir)], check=True, timeout=900)
        files = sorted(download_dir.glob(f"{episode_id}/**/*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True)
        media = [p for p in files if p.is_file() and p.suffix.lower() in {".m4a", ".mp3", ".wav"}]
        if media:
            meta["downloaded_audio"] = str(media[0])
    except Exception:
        pass
    return meta


def download_audio(audio_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"audio exists: {dest}", flush=True)
        return
    run(["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "20", "-C", "-", "-o", str(dest), audio_url])


def write_asr_pipeline(work_dir: Path, episode_id: str, episode_url: str, title: str, args: argparse.Namespace) -> Path:
    template = ASR_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__WORK_DIR__": str(work_dir),
        "__MODEL_DIR__": args.model_dir,
        "__PUNC_MODEL_DIR__": args.punc_model_dir,
        "__VAD_MODEL_DIR__": args.vad_model_dir,
        "__EPISODE_ID__": episode_id,
        "__EPISODE_URL__": episode_url,
        "__EPISODE_TITLE__": title,
    }
    for key, value in replacements.items():
        template = template.replace(json.dumps(key)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1])
    path = work_dir / "asr_pipeline.py"
    path.write_text(template, encoding="utf-8")
    return path


def zip_outputs(work_dir: Path, episode_id: str) -> Path:
    out = work_dir / "output"
    zip_path = out / f"xiaoyuzhou_{episode_id}_asr_results.zip"
    candidates = []
    for pattern in ["transcript_*.md", "transcript_*.txt", "transcript_*.srt", "transcription_*.json", "benchmark_cpu_gpu_*.json", "manifest.json", "podcast_summary.*", "tldr_infographic*", "episode_page_context.*", "site_meta.json"]:
        candidates.extend(out.glob(pattern))
    if not candidates:
        return zip_path
    rels = [str(p.relative_to(work_dir)) for p in sorted(set(candidates))]
    run(["zip", "-9", "-j", str(zip_path)] + [str(work_dir / r) for r in rels], cwd=work_dir, check=True)
    return zip_path


def transcription_ok_chunks(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary") or {}
        if "ok_chunks" in summary:
            return int(summary.get("ok_chunks") or 0)
        return sum(1 for c in data.get("chunks") or [] if not c.get("error"))
    except Exception:
        return 0


def transcription_chunk_counts(path: Path) -> tuple[int, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary") or {}
        total = int(summary.get("chunks") or len(data.get("chunks") or []))
        ok = int(summary.get("ok_chunks") or sum(1 for c in data.get("chunks") or [] if not c.get("error")))
        return ok, total
    except Exception:
        return 0, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Xiaoyuzhou episode URL or episode id")
    parser.add_argument("--work-root", type=Path, default=PODCAST_ROOT)
    parser.add_argument("--title", default="", help="Override title if public metadata extraction fails")
    parser.add_argument("--audio-url", default="", help="Override direct m4a/mp3 URL")
    parser.add_argument("--slug", default="", help="Static website slug override")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--chunk-seconds", type=float, default=300.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--benchmark-start", type=float, default=600.0)
    parser.add_argument("--benchmark-seconds", type=float, default=300.0)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--asr-api-url",
        default=DEFAULT_ASR_API_URL,
        help="OpenAI-compatible remote ASR endpoint; omit to use the local CUDA pipeline",
    )
    parser.add_argument("--skip-asr-if-exists", action="store_true", default=True)
    parser.add_argument("--force-asr", action="store_true")
    parser.add_argument("--force-summary", action="store_true")
    parser.add_argument("--force-tldr-image", action="store_true", help="Regenerate the TLDR infographic even if output/tldr_infographic.png exists")
    parser.add_argument("--skip-tldr-image", action="store_true", help="Skip GPT Image2/TLDR infographic generation")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--punc-model-dir", default=DEFAULT_PUNC_MODEL_DIR)
    parser.add_argument("--vad-model-dir", default=DEFAULT_VAD_MODEL_DIR)
    parser.add_argument("--llm-api-base", default=DEFAULT_LLM_API_BASE)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--wiki-path", type=Path, default=DEFAULT_WIKI_PATH, help="LLM Wiki path; set empty with --skip-wiki-export to disable")
    parser.add_argument("--skip-wiki-export", action="store_true", help="Do not export final LLM summary into the LLM Wiki")
    args = parser.parse_args()

    episode_id = parse_episode_id(args.url)
    episode_url = args.url if args.url.startswith("http") else f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"
    work_dir = args.work_root / f"xiaoyuzhou_{episode_id}"
    for d in ["input", "audio", "chunks", "transcripts", "output", "logs"]:
        (work_dir / d).mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {"episode_id": episode_id, "url": episode_url, "created_at": datetime.now().isoformat(timespec="seconds")}
    if args.audio_url or args.title:
        metadata.update({"title": args.title, "audio_url": args.audio_url})
    else:
        try:
            metadata.update(fetch_public_metadata(episode_url))
        except Exception as e:
            metadata["public_metadata_error"] = repr(e)
    if not metadata.get("audio_url"):
        opencli_meta = try_opencli_metadata_and_download(episode_id, work_dir)
        metadata.update({k: v for k, v in opencli_meta.items() if v})
    title = args.title or metadata.get("title") or f"Xiaoyuzhou episode {episode_id}"
    metadata["title"] = title
    page_context = metadata.get("page_context") if isinstance(metadata.get("page_context"), dict) else {}
    if page_context:
        page_context.setdefault("episode_id", episode_id)
        page_context.setdefault("source_url", episode_url)
        page_context.setdefault("official_title", title)
        context_json = work_dir / "output" / "episode_page_context.json"
        context_md = work_dir / "output" / "episode_page_context.md"
        context_json.write_text(json.dumps(page_context, ensure_ascii=False, indent=2), encoding="utf-8")
        context_md.write_text(render_page_context_markdown(page_context), encoding="utf-8")
        metadata["page_context_path"] = str(context_json)
        metadata["page_context_md_path"] = str(context_md)

    input_audio = work_dir / "input" / "episode.m4a"
    if metadata.get("downloaded_audio"):
        src = Path(metadata["downloaded_audio"])
        if src.exists() and not input_audio.exists():
            shutil.copy2(src, input_audio)
    elif metadata.get("audio_url"):
        download_audio(str(metadata["audio_url"]), input_audio)
    elif not input_audio.exists():
        raise RuntimeError("No audio URL found. Pass --audio-url or configure opencli Xiaoyuzhou credentials.")

    slug = args.slug or sanitize_slug(title, f"xiaoyuzhou-{episode_id}")
    if episode_id == "6a2be5da43a22a695582ad20" and not args.slug:
        slug = "xiaoyuzhou-spacex-asr"
    site_meta = {"slug": slug, "series": "xiaoyuzhou", "tags": ["小宇宙", "ASR", "SenseVoice", "GPU"]}
    (work_dir / "output" / "site_meta.json").write_text(json.dumps(site_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (work_dir / "output" / "episode_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    pipeline = write_asr_pipeline(work_dir, episode_id, episode_url, title, args)
    log_prefix = work_dir / "logs" / f"run_{now_stamp()}"
    remote_transcription = work_dir / "output" / "transcription_remote_cpu.json"
    local_transcription = work_dir / "output" / "transcription_cuda.json"
    transcription = remote_transcription if args.asr_api_url else local_transcription
    if args.force_asr or not (args.skip_asr_if_exists and transcription.exists()):
        run(["python3.12", str(pipeline), "prepare", "--chunk-seconds", str(args.chunk_seconds), "--overlap-seconds", str(args.overlap_seconds)], log_prefix.with_name(log_prefix.name + "_prepare.log"))
        if not args.asr_api_url and not args.skip_benchmark:
            run(["python3.12", str(pipeline), "benchmark", "--devices", "cpu,cuda", "--start", str(args.benchmark_start), "--seconds", str(args.benchmark_seconds), "--language", args.language], log_prefix.with_name(log_prefix.name + "_benchmark.log"))
        if args.asr_api_url:
            asr_cmd = [
                "python3.12",
                str(REMOTE_ASR_CLIENT),
                "--api-url",
                args.asr_api_url,
                "--manifest",
                str(work_dir / "output" / "manifest.json"),
                "--output-dir",
                str(work_dir / "output"),
                "--transcript-dir",
                str(work_dir / "transcripts"),
                "--language",
                args.language,
                "--device-label",
                "remote_cpu",
            ] + (["--force"] if args.force_asr else [])
            asr_label = "remote ASR"
        else:
            asr_cmd = ["python3.12", str(pipeline), "transcribe", "--device", "cuda", "--language", args.language, "--chunk-seconds", str(args.chunk_seconds), "--overlap-seconds", str(args.overlap_seconds)] + (["--force"] if args.force_asr else [])
            asr_label = "CUDA ASR"
        asr_failed = False
        try:
            run(asr_cmd, log_prefix.with_name(log_prefix.name + "_transcribe.log"))
        except subprocess.CalledProcessError:
            asr_failed = True
        asr_ok, asr_total = transcription_chunk_counts(transcription)
        if asr_failed or asr_ok < asr_total:
            print(f"WARN: {asr_label} incomplete ({asr_ok}/{asr_total} chunks); retrying full transcription on local CPU fallback.", flush=True)
            cpu_transcription = work_dir / "output" / "transcription_cpu.json"
            run(["python3.12", str(pipeline), "transcribe", "--device", "cpu", "--language", args.language, "--chunk-seconds", str(args.chunk_seconds), "--overlap-seconds", str(args.overlap_seconds), "--force"], log_prefix.with_name(log_prefix.name + "_transcribe_cpu_fallback.log"))
            transcription = cpu_transcription
    else:
        print(f"transcription exists, skipping ASR: {transcription}", flush=True)
        if transcription_ok_chunks(transcription) == 0 and (work_dir / "output" / "transcription_cpu.json").exists():
            transcription = work_dir / "output" / "transcription_cpu.json"

    summary_cmd = ["python3.12", str(SUMMARY_SCRIPT), str(transcription), "--api-base", args.llm_api_base, "--model", args.llm_model]
    page_context_json = work_dir / "output" / "episode_page_context.json"
    if page_context_json.exists():
        summary_cmd.extend(["--page-context", str(page_context_json)])
    if args.force_summary:
        summary_cmd.append("--force")
    run(summary_cmd, log_prefix.with_name(log_prefix.name + "_summary.log"))
    tldr_image_path = work_dir / "output" / "tldr_infographic.png"
    if not args.skip_tldr_image:
        tldr_cmd = [
            "uv",
            "run",
            "--with",
            "openai",
            "--with",
            "python-dotenv",
            "--with",
            "pillow",
            "--with",
            "requests",
            "python",
            str(TLDR_IMAGE_SCRIPT),
            str(work_dir / "output" / "podcast_summary.json"),
        ]
        if args.force_tldr_image:
            tldr_cmd.append("--force")
        run(tldr_cmd, log_prefix.with_name(log_prefix.name + "_tldr_image.log"), cwd=PODCAST_ROOT)
    wiki_export_stdout = ""
    if not args.skip_wiki_export:
        wiki_cmd = [
            "python3.12",
            str(WIKI_EXPORT_SCRIPT),
            str(work_dir / "output" / "podcast_summary.json"),
            "--summary-md",
            str(work_dir / "output" / "podcast_summary.md"),
            "--page-context",
            str(page_context_json),
            "--transcription",
            str(transcription),
            "--wiki-path",
            str(args.wiki_path),
            "--slug",
            slug,
            "--source-url",
            episode_url,
            "--site-report-url",
            f"{DEFAULT_SITE_BASE_LAN}/{slug}/index.html",
        ]
        wiki_export_stdout = run(wiki_cmd, log_prefix.with_name(log_prefix.name + "_wiki.log")).stdout
    zip_outputs(work_dir, episode_id)
    pub = run(["python3.12", str(PUBLISH_SCRIPT)], log_prefix.with_name(log_prefix.name + "_publish.log"))

    result = {
        "episode_id": episode_id,
        "title": title,
        "work_dir": str(work_dir),
        "transcription": str(transcription),
        "summary_json": str(work_dir / "output" / "podcast_summary.json"),
        "summary_markdown": str(work_dir / "output" / "podcast_summary.md"),
        "tldr_infographic": str(tldr_image_path) if tldr_image_path.exists() else "",
        "site_report": f"{DEFAULT_SITE_BASE_LAN}/{slug}/index.html",
        "site_full_text": f"{DEFAULT_SITE_BASE_LAN}/{slug}/full.html",
        "site_index": f"{DEFAULT_SITE_BASE_LAN}/index.html",
        "wiki_export": wiki_export_stdout,
        "publisher_output": pub.stdout,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
