#!/usr/bin/env python3
"""Generate a structured Chinese podcast summary from an ASR transcript via local OpenAI-compatible LLM.

Default endpoint is the GB10 vLLM service on http://127.0.0.1:8004/v1
serving model qwen3.6-35b-fp8. Outputs both JSON and Markdown into the
same output directory as the transcription JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_API_BASE = "http://127.0.0.1:8004/v1"
DEFAULT_MODEL = "qwen3.6-35b-fp8"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value is None else [value])


def load_transcript_text(transcription_path: Path, max_chars: int = 180_000) -> tuple[dict[str, Any], str]:
    trans = read_json(transcription_path)
    output_dir = transcription_path.parent
    device = trans.get("device") or transcription_path.stem.replace("transcription_", "") or "cuda"
    txt_path = output_dir / f"transcript_{device}.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
    else:
        parts = []
        for chunk in trans.get("chunks") or []:
            if chunk.get("error"):
                continue
            parts.append(f"[{chunk.get('start_ts', '')}] {chunk.get('text', '')}")
        text = "\n\n".join(parts)
    if len(text) > max_chars:
        # Keep the beginning and end if a transcript ever exceeds the target model context budget.
        head = max_chars // 2
        tail = max_chars - head
        text = text[:head] + "\n\n[...中间内容因长度过长被截断...]\n\n" + text[-tail:]
    return trans, text


def load_page_context(path: Path | None, output_dir: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if path:
        candidates.append(path.expanduser().resolve())
    candidates.append(output_dir / "episode_page_context.json")
    for candidate in candidates:
        if candidate.exists():
            try:
                data = read_json(candidate)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def render_context_for_prompt(context: dict[str, Any], max_chars: int = 60_000) -> str:
    if not context:
        return ""
    md_path = context.get("_markdown_path")
    if md_path:
        p = Path(str(md_path))
        if p.exists():
            text = p.read_text(encoding="utf-8")
            return text[:max_chars]
    lines = ["# Official episode page context", ""]
    for key in ["official_title", "podcast_title", "source_url", "published_time", "duration_text", "cover_image"]:
        if context.get(key):
            lines.append(f"- {key}: {context[key]}")
    if context.get("description"):
        lines += ["", "## description", str(context["description"])]
    outline = context.get("outline") or []
    if outline:
        lines += ["", "## outline"]
        for item in outline:
            if isinstance(item, dict):
                lines.append(f"- {item.get('timestamp', '')} {item.get('title', '')}".rstrip())
                for note in item.get("notes") or []:
                    lines.append(f"  - {note}")
    return "\n".join(lines)[:max_chars]


def build_prompt(trans: dict[str, Any], transcript_text: str, page_context: dict[str, Any] | None = None) -> list[dict[str, str]]:
    page_context = page_context or {}
    meta = {
        "title": trans.get("title"),
        "episode_id": trans.get("episode_id"),
        "url": trans.get("url"),
        "duration": trans.get("duration_formatted"),
        "chars": (trans.get("summary") or {}).get("chars"),
        "model": trans.get("model"),
        "official_title": page_context.get("official_title"),
        "podcast_title": page_context.get("podcast_title"),
        "published_time": page_context.get("published_time"),
        "duration_text": page_context.get("duration_text"),
    }
    schema = {
        "title": "节目标题；优先使用官方页面标题",
        "podcast": "播客 / show 名称；如果官方页面未提供则写 unknown",
        "published_time": "发布时间；如果官方页面未提供则写 unknown",
        "official_outline": [{"timestamp": "官方时间戳", "title": "官方小节标题", "notes": ["官方小节说明"]}],
        "theme": "用 2-4 句话概括本期核心主题",
        "guests": [{"name": "嘉宾/主持名", "role": "身份或关系", "evidence": "优先引用官方页面；否则引用转写判断依据"}],
        "background": "为什么这个话题值得讨论；节目发生的产业/资本/技术背景",
        "topic_summary": [
            {
                "topic": "讨论话题标题；尽量对齐官方 OUTLINE",
                "timestamp_range": "如果能判断，用 00:00:00-00:10:00；不确定则写 unknown",
                "summary": "该话题的 3-6 句总结",
                "key_points": ["要点 1", "要点 2", "要点 3"]
            }
        ],
        "golden_quotes": [
            {"quote": "尽量保留原话，不要编造", "speaker_or_context": "说话人或上下文", "why_it_matters": "为什么重要"}
        ],
        "key_takeaways": ["洞察 1", "洞察 2", "洞察 3"],
        "entities_and_terms": [{"term": "专名/术语", "explanation": "解释或上下文"}],
        "caveats": ["ASR 可能误识别的地方、官方简介与转写冲突处、专名不确定处"],
        "tldr": "200-300 字中文摘要，适合放在网站索引卡片上"
    }
    official_context = render_context_for_prompt(page_context)
    system = (
        "你是中文播客研究助理，擅长把小宇宙官方 show notes 和 ASR 长转写整理成可发布的网站总结。"
        "必须区分官方页面信息与 ASR 转写信息：标题、节目名、发布日期、官方 OUTLINE 优先采用官方页面；"
        "具体观点、金句和讨论细节必须依据 ASR 转写。不要把听众评论当作节目事实。"
        "不确定就写 unknown/不确定。输出必须是严格 JSON 对象，不要 Markdown 代码块，不要额外解释。"
    )
    user = f"""
请基于以下官方页面上下文和播客 ASR 转写稿生成结构化总结。总结模板必须覆盖：官方 OUTLINE、主题、嘉宾、背景、讨论的话题总结、金句等。

元数据：
{json.dumps(meta, ensure_ascii=False, indent=2)}

请严格按这个 JSON schema 的字段输出：
{json.dumps(schema, ensure_ascii=False, indent=2)}

官方页面上下文如下；它用于校准标题、节目名、发布时间、简介和 OUTLINE：
--- OFFICIAL EPISODE PAGE CONTEXT BEGIN ---
{official_context or '未提供'}
--- OFFICIAL EPISODE PAGE CONTEXT END ---

ASR 转写稿如下；它用于补充讨论细节、观点、金句和 caveats：
--- TRANSCRIPT BEGIN ---
{transcript_text}
--- TRANSCRIPT END ---
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_openai_compatible(api_base: str, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Some OpenAI-compatible servers reject response_format. Retry without it.
        body = e.read().decode("utf-8", errors="replace")
        if e.code in {400, 422} and "response_format" in body:
            payload.pop("response_format", None)
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        else:
            raise RuntimeError(f"LLM HTTP {e.code}: {body[:1000]}") from e
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or msg.get("reasoning") or ""
    return content


def parse_summary(raw: str) -> dict[str, Any]:
    cleaned = strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Try extracting the largest JSON object from accidental prose.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"parse_error": True, "raw": raw}


def render_markdown(summary: dict[str, Any], trans: dict[str, Any], model: str, generated_at: str) -> str:
    lines: list[str] = []
    title = summary.get("title") or trans.get("title") or "播客总结"
    lines += [f"# {title}", ""]
    lines += [f"- **生成时间:** {generated_at}", f"- **总结模型:** {model}", f"- **源链接:** {trans.get('url', '')}", ""]
    if summary.get("podcast") or summary.get("published_time"):
        lines += ["## 节目信息", ""]
        if summary.get("podcast"):
            lines.append(f"- **播客:** {summary.get('podcast')}")
        if summary.get("published_time"):
            lines.append(f"- **发布时间:** {summary.get('published_time')}")
        lines.append("")
    official_outline = safe_list(summary.get("official_outline"))
    if official_outline:
        lines += ["## 官方 OUTLINE", ""]
        for item in official_outline:
            if isinstance(item, dict):
                lines.append(f"- **{item.get('timestamp', '')}** {item.get('title', '')}".rstrip())
                for note in safe_list(item.get("notes")):
                    lines.append(f"  - {note}")
            else:
                lines.append(f"- {item}")
        lines.append("")
    if summary.get("theme"):
        lines += ["## 主题", "", str(summary.get("theme")), ""]
    guests = safe_list(summary.get("guests"))
    if guests:
        lines += ["## 嘉宾", ""]
        for g in guests:
            if isinstance(g, dict):
                lines.append(f"- **{g.get('name', 'unknown')}**：{g.get('role', 'unknown')}。依据：{g.get('evidence', 'unknown')}")
            else:
                lines.append(f"- {g}")
        lines.append("")
    if summary.get("background"):
        lines += ["## 背景", "", str(summary.get("background")), ""]
    topics = safe_list(summary.get("topic_summary"))
    if topics:
        lines += ["## 讨论的话题总结", ""]
        for i, t in enumerate(topics, 1):
            if isinstance(t, dict):
                lines += [f"### {i}. {t.get('topic', '未命名话题')}", "", f"- **时间:** {t.get('timestamp_range', 'unknown')}", f"- **总结:** {t.get('summary', '')}"]
                points = safe_list(t.get("key_points"))
                if points:
                    lines.append("- **要点:**")
                    lines += [f"  - {p}" for p in points]
                lines.append("")
            else:
                lines += [f"### {i}. {t}", ""]
    quotes = safe_list(summary.get("golden_quotes"))
    if quotes:
        lines += ["## 金句", ""]
        for q in quotes:
            if isinstance(q, dict):
                lines += [f"> {q.get('quote', '')}", "", f"- **上下文:** {q.get('speaker_or_context', 'unknown')}", f"- **价值:** {q.get('why_it_matters', '')}", ""]
            else:
                lines += [f"> {q}", ""]
    takeaways = safe_list(summary.get("key_takeaways"))
    if takeaways:
        lines += ["## 关键洞察", ""]
        lines += [f"- {x}" for x in takeaways]
        lines.append("")
    terms = safe_list(summary.get("entities_and_terms"))
    if terms:
        lines += ["## 专名与术语", ""]
        for term in terms:
            if isinstance(term, dict):
                lines.append(f"- **{term.get('term', '')}**：{term.get('explanation', '')}")
            else:
                lines.append(f"- {term}")
        lines.append("")
    caveats = safe_list(summary.get("caveats"))
    if caveats:
        lines += ["## 注意事项 / ASR 不确定处", ""]
        lines += [f"- {x}" for x in caveats]
        lines.append("")
    if summary.get("tldr"):
        lines += ["## TL;DR", "", str(summary.get("tldr")), ""]
    if summary.get("parse_error"):
        lines += ["## 原始模型输出", "", "```", str(summary.get("raw", "")), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcription_json", type=Path, help="Path to output/transcription_<device>.json")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-input-chars", type=int, default=180_000)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--page-context", type=Path, default=None, help="Path to output/episode_page_context.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    transcription_path = args.transcription_json.expanduser().resolve()
    output_dir = transcription_path.parent
    summary_json = output_dir / "podcast_summary.json"
    summary_md = output_dir / "podcast_summary.md"
    if summary_json.exists() and summary_md.exists() and not args.force:
        print(json.dumps({"status": "exists", "json": str(summary_json), "markdown": str(summary_md)}, ensure_ascii=False, indent=2))
        return 0

    trans, transcript_text = load_transcript_text(transcription_path, max_chars=args.max_input_chars)
    page_context = load_page_context(args.page_context, output_dir)
    if page_context:
        md_candidate = output_dir / "episode_page_context.md"
        if md_candidate.exists():
            page_context["_markdown_path"] = str(md_candidate)
    messages = build_prompt(trans, transcript_text, page_context)
    started = time.time()
    raw = call_openai_compatible(args.api_base, args.model, messages, args.max_tokens, args.temperature)
    summary = parse_summary(raw)
    generated_at = datetime.now().isoformat(timespec="seconds")
    summary.update({
        "generated_at": generated_at,
        "summary_model": args.model,
        "api_base": args.api_base,
        "source_transcription": str(transcription_path),
        "input_chars": len(transcript_text),
        "has_page_context": bool(page_context),
        "page_context_source": str(args.page_context or (output_dir / "episode_page_context.json")) if page_context else "",
        "wall_seconds": round(time.time() - started, 3),
    })
    markdown = render_markdown(summary, trans, args.model, generated_at)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": "generated",
        "json": str(summary_json),
        "markdown": str(summary_md),
        "model": args.model,
        "input_chars": len(transcript_text),
        "wall_seconds": summary["wall_seconds"],
        "parse_error": bool(summary.get("parse_error")),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
