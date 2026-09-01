#!/usr/bin/env python3
"""Generate a native GPT Image2 TL;DR infographic for a podcast summary.

This script follows the updated TLDR infographic workflow: ask GPT Image2 to
produce the complete final Chinese infographic directly, including layout,
illustrations, tables, labels, and readable text. It deliberately avoids the old
"Image2 visual base + deterministic PIL overlay" path unless a future operator
adds an explicit fallback script.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Official gpt-image-2 maximum landscape request. Compatible gateways may
# return a smaller image, so generation metadata records the actual PNG size.
W, H = 3840, 2160
CONFIG_ENV = os.environ.get("HERMES_CONFIG", "").strip()
CONFIG_PATH = Path(CONFIG_ENV).expanduser() if CONFIG_ENV else None
DEFAULT_PROVIDER = os.environ.get("IMAGE2_PROVIDER", "openai")
DEFAULT_MODEL = os.environ.get("IMAGE2_MODEL", "gpt-image-2")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value in (None, "") else [value])


def one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_payload(summary: dict[str, Any]) -> dict[str, Any]:
    topics = []
    for t in safe_list(summary.get("topic_summary")):
        if isinstance(t, dict):
            topics.append({
                "title": one_line(t.get("topic") or "未命名话题"),
                "body": one_line(t.get("summary")),
                "time": one_line(t.get("timestamp_range")),
                "key_points": [one_line(x) for x in safe_list(t.get("key_points")) if one_line(x)],
            })
        else:
            topics.append({"title": one_line(t), "body": "", "time": "", "key_points": []})

    quotes = []
    for q in safe_list(summary.get("golden_quotes")):
        if isinstance(q, dict):
            quotes.append({
                "quote": one_line(q.get("quote")),
                "context": one_line(q.get("speaker_or_context")),
                "why": one_line(q.get("why_it_matters")),
            })
        else:
            quotes.append({"quote": one_line(q), "context": "", "why": ""})

    terms = []
    for t in safe_list(summary.get("entities_and_terms")):
        if isinstance(t, dict):
            terms.append(one_line(f"{t.get('term', '')}：{t.get('explanation', '')}"))
        else:
            terms.append(one_line(t))

    takeaways = [one_line(x) for x in safe_list(summary.get("key_takeaways")) if one_line(x)]
    guests = []
    for g in safe_list(summary.get("guests")):
        if isinstance(g, dict):
            guests.append(one_line(f"{g.get('name','')}｜{g.get('role','')}"))
        else:
            guests.append(one_line(g))

    outline = []
    for item in safe_list(summary.get("official_outline")):
        if isinstance(item, dict):
            outline.append({
                "time": one_line(item.get("timestamp")),
                "title": one_line(item.get("title")),
                "notes": one_line(item.get("notes")),
            })

    return {
        "title": one_line(summary.get("title") or "播客 TL;DR"),
        "podcast": one_line(summary.get("podcast") or "Podcast"),
        "published_time": one_line(summary.get("published_time")),
        "theme": one_line(summary.get("theme")),
        "background": one_line(summary.get("background")),
        "tldr": one_line(summary.get("tldr") or summary.get("theme")),
        "takeaways": takeaways,
        "topics": topics,
        "quotes": quotes,
        "terms": terms,
        "guests": guests,
        "outline": outline,
        "caveats": [one_line(x) for x in safe_list(summary.get("caveats")) if one_line(x)],
        "summary_model": one_line(summary.get("summary_model")),
    }


def build_direct_prompt(payload: dict[str, Any], attempt: int = 1) -> str:
    takeaway_lines = "\n".join(
        f"{i + 1}. {x}" for i, x in enumerate(payload["takeaways"])
    ) or "1. 提炼节目中的关键判断"
    quote_lines = "\n".join(
        f"“{q['quote']}”" for q in payload["quotes"][:2]
    )
    core_conclusion = payload["takeaways"][0] if payload["takeaways"] else payload["theme"] or payload["tldr"]
    guest_line = "；".join(payload["guests"]) or "嘉宾/主持：见节目说明"
    attempt_note = {
        1: "dense AI summary with exact complete paragraphs",
        2: "correct omitted or malformed summary text without adding content",
        3: "increase type size while preserving every summary paragraph",
        4: "strict three-section summary and all-insight completeness audit",
        5: "final typography correction with no truncation",
    }.get(attempt, "production infographic")
    source_context = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    return f"""Create one finished Chinese podcast TL;DR infographic as a normal wide poster, not a long webpage. Native GPT Image2 generation only.

Canvas request: 3840×2160, wide 16:9. Language: Simplified Chinese. Style: premium editorial knowledge brief, crisp vector cards, restrained navy and ivory with orange/cyan/green accents, strong typographic hierarchy, minimal decoration. Attempt focus: {attempt_note}.

VISIBLE COPY - render only the following copy. Preserve every line exactly and in full:

TITLE
{payload['title']}

SUBTITLE
{payload['podcast']} · {payload['published_time'][:10]} · 播客 TL;DR

PEOPLE
{guest_line}

核心结论
{core_conclusion}

关键洞察
{takeaway_lines}

AI 总结

节目概要
{payload['tldr']}

为什么是现在
{payload['background']}

核心判断
{payload['theme']}

金句
{quote_lines}

LAYOUT
- Header across the top, followed by one compact core-conclusion band.
- Main body uses a 60/40 split. The entire left side is titled「AI 总结」and must be filled densely with all three exact summary sections. Put「节目概要」in a full-width text card, then「为什么是现在」and「核心判断」in two balanced text cards below it.
- The right side contains a compact topic-appropriate conceptual diagram plus every key insight. Derive diagram labels only from exact terms in the supplied source context.
- Place the two quotes in a slim footer band. Keep the entire design within one 16:9 frame.
- The AI summary cards should use almost all available left-side space. Use readable compact paragraph typography with generous line spacing, not oversized headings or decorative illustrations.

HARD CONSTRAINTS
- The complete text of「节目概要」「为什么是现在」「核心判断」and every insight line must be visible, readable, and complete.
- Do not show a discussion map, topic index, numbered timeline, or timestamps anywhere.
- Never use "...", the ellipsis character, clipped endings, fade-outs, continuation marks, or placeholder text.
- Do not invent names, figures, dates, labels, or extra body copy.
- No pseudo-Chinese, no lorem ipsum, no cropped cards, no text touching an edge, no watermark.
- If space is tight, remove decoration first. Keep all required text at a readable size.

COMPLETE SOURCE CONTEXT - factual grounding only; do not render this block verbatim:
{source_context}
"""


def png_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()[:24]
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return None


def normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return "https://api.openai.com/v1"
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _yaml_scalar(raw: str) -> str:
    s = raw.strip()
    if s.startswith(("'", '"')) and s.endswith(("'", '"')) and len(s) >= 2:
        return s[1:-1]
    return s


def load_provider_from_config(provider: str) -> tuple[str, str]:
    """Resolve image generation base_url/api_key without logging secrets.

    Preferred generic configuration is OPENAI_API_KEY plus optional
    OPENAI_BASE_URL/IMAGE2_BASE_URL. If HERMES_CONFIG is explicitly supplied,
    the named IMAGE2_PROVIDER can be read from that Hermes config file.
    """
    env_base_url = os.environ.get("IMAGE2_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    env_api_key = os.environ.get("IMAGE2_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if env_api_key:
        return normalize_base_url(env_base_url), env_api_key
    if not CONFIG_PATH or not CONFIG_PATH.exists():
        raise RuntimeError("No Image2 API key found. Set IMAGE2_API_KEY/OPENAI_API_KEY or HERMES_CONFIG + IMAGE2_PROVIDER.")
    text = CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    provider_indent = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)" + re.escape(provider) + r":\s*$", line)
        if m:
            start = i + 1
            provider_indent = len(m.group(1))
            break
    if start is None or provider_indent is None:
        raise RuntimeError(f"provider {provider!r} not found in {CONFIG_PATH}")
    fields: dict[str, str] = {}
    for line in lines[start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= provider_indent:
            break
        m = re.match(r"^\s+([A-Za-z0-9_.-]+):\s*(.*)$", line)
        if m and m.group(1) in {"base_url", "api_key"}:
            fields[m.group(1)] = _yaml_scalar(m.group(2))
    base_url = fields.get("base_url") or env_base_url
    api_key = fields.get("api_key") or env_api_key
    if not api_key:
        raise RuntimeError("No Image2 API key found in Hermes config or IMAGE2_API_KEY/OPENAI_API_KEY")
    return normalize_base_url(base_url), api_key


def find_b64(obj: Any) -> str | None:
    found: list[tuple[str, str]] = []

    def walk(x: Any, path: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                kp = f"{path}.{k}" if path else k
                if isinstance(v, str) and len(v) > 1000 and ("b64" in k.lower() or "image" in k.lower()):
                    found.append((kp, v))
                walk(v, kp)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)
    found.sort(key=lambda kv: (0 if "b64_json" in kv[0] else 1, -len(kv[1])))
    return found[0][1] if found else None


def generate_image2(prompt: str, out_path: Path, *, provider: str, model: str, size: str, quality: str, timeout: int, partial_images: int) -> dict[str, Any]:
    base_url, auth_value = load_provider_from_config(provider)
    url = base_url.rstrip("/") + "/images/generations"
    headers = {
        "Authorization": f"Bearer {auth_value}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": 1,
        "output_format": "png",
        "stream": True,
        "partial_images": partial_images,
    }
    started = time.time()
    last_b64: str | None = None
    events: list[str] = []
    errors: list[str] = []

    print(f"[image2-direct] provider={provider} model={model} size={size} quality={quality} prompt_chars={len(prompt)}")
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Image2 HTTP {r.status_code}: {r.text[:800]}")
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    events.append("done")
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    events.append("non_json")
                    continue
                etype = str(ev.get("type") or ev.get("event") or ev.get("object") or "event")
                events.append(etype)
                if "error" in ev:
                    errors.append(json.dumps(ev.get("error"), ensure_ascii=False)[:500])
                b64 = find_b64(ev)
                if b64:
                    last_b64 = b64
                    print(f"[image2-direct] event={etype} image_b64_chars={len(b64)} elapsed={time.time()-started:.1f}s", flush=True)
                else:
                    print(f"[image2-direct] event={etype} elapsed={time.time()-started:.1f}s", flush=True)
            else:
                # Some OpenAI-compatible gateways return a normal JSON line even when stream=True.
                try:
                    ev = json.loads(line)
                    b64 = find_b64(ev)
                    if b64:
                        last_b64 = b64
                except Exception:
                    pass

    if not last_b64:
        detail = "; ".join(errors) if errors else f"events={events[-10:]}"
        raise RuntimeError(f"No image b64 returned from Image2 stream: {detail}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(last_b64))
    actual = png_dimensions(out_path)
    actual_size = f"{actual[0]}x{actual[1]}" if actual else "unknown"
    requested_pixels = None
    try:
        req_w, req_h = (int(x) for x in size.lower().split("x", 1))
        requested_pixels = req_w * req_h
    except Exception:
        pass
    if actual_size != size:
        print(f"[image2-direct] warning=requested_size_mismatch requested={size} actual={actual_size}", flush=True)
    return {
        "provider": provider,
        "model": model,
        "base_url_host": re.sub(r"^https?://", "", base_url).split("/")[0],
        "size": size,
        "requested_pixels": requested_pixels,
        "actual_size": actual_size,
        "actual_pixels": actual[0] * actual[1] if actual else None,
        "size_match": actual_size == size,
        "quality": quality,
        "stream": True,
        "events": events[-20:],
        "bytes": out_path.stat().st_size,
        "wall_seconds": round(time.time() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=f"{W}x{H}")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--partial-images", type=int, default=2)
    parser.add_argument("--attempt", type=int, default=1, help="Prompt attempt profile 1-5 from the Image2 workflow")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = read_json(args.summary_json, {}) or {}
    if not summary:
        raise SystemExit(f"Cannot read summary JSON: {args.summary_json}")

    out_dir = args.summary_json.parent
    out_path = args.output or (out_dir / "tldr_infographic.png")
    prompt_path = out_dir / "tldr_infographic_prompt.txt"
    meta_path = out_dir / "tldr_infographic_meta.json"
    if out_path.exists() and not args.force:
        print(json.dumps({"ok": True, "skipped": True, "output": str(out_path)}, ensure_ascii=False))
        return 0

    payload = extract_payload(summary)
    prompt = build_direct_prompt(payload, attempt=args.attempt)
    prompt_path.write_text(prompt, encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "prompt": str(prompt_path), "prompt_chars": len(prompt)}, ensure_ascii=False, indent=2))
        return 0

    gen_meta = generate_image2(
        prompt,
        out_path,
        provider=args.provider,
        model=args.model,
        size=args.size,
        quality=args.quality,
        timeout=args.timeout,
        partial_images=args.partial_images,
    )
    meta = {
        "ok": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_json": str(args.summary_json),
        "output": str(out_path),
        "prompt": str(prompt_path),
        "mode": "native_gpt_image2_direct_complete_image",
        "image2_attempted": True,
        "image2_ok": True,
        "fallback_used": False,
        "title": payload["title"],
        **gen_meta,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
