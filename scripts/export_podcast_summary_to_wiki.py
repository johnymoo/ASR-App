#!/usr/bin/env python3
"""Export the final LLM podcast summary into the local LLM Wiki.

The exporter keeps the generated LLM summary as a raw source under
`raw/podcast-summaries/`, then creates/updates a curated summary page under
`queries/` and updates `index.md` + `log.md`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_WIKI_PATH = Path(os.environ.get("WIKI_PATH", "~/wiki")).expanduser()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def slugify(text: str, fallback: str = "podcast-summary") -> str:
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


def yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def wiki_link(name: str) -> str:
    return f"[[{name}]]"


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def existing_created(path: Path, default: str) -> str:
    text = read_text(path)
    m = re.search(r"^created:\s*(\S+)", text, re.M)
    return m.group(1) if m else default


def render_raw_source(summary_md: str, summary_json: dict[str, Any], page_context: dict[str, Any], source_url: str) -> str:
    body = [
        f"# {summary_json.get('title') or page_context.get('official_title') or 'Podcast LLM Summary'}",
        "",
        "## Source metadata",
        f"- Source URL: {source_url or page_context.get('source_url') or ''}",
        f"- Podcast: {summary_json.get('podcast') or page_context.get('podcast_title') or ''}",
        f"- Published time: {summary_json.get('published_time') or page_context.get('published_time') or ''}",
        f"- Summary model: {summary_json.get('summary_model') or ''}",
        f"- Generated at: {summary_json.get('generated_at') or ''}",
        "",
        "## LLM summary markdown",
        "",
        summary_md.strip(),
        "",
        "## Structured summary JSON",
        "",
        "```json",
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(body).rstrip() + "\n"


def render_outline(items: list[Any]) -> str:
    if not items:
        return "- unknown\n"
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            ts = item.get("timestamp") or ""
            title = item.get("title") or ""
            lines.append(f"- **{ts}** {title}".rstrip())
            for note in safe_list(item.get("notes"))[:8]:
                lines.append(f"  - {note}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_topic_summary(items: list[Any]) -> str:
    if not items:
        return "- unknown\n"
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            lines.append(f"### {i}. {item.get('topic') or '未命名话题'}")
            if item.get("timestamp_range"):
                lines.append(f"- 时间：{item.get('timestamp_range')}")
            if item.get("summary"):
                lines.append(f"- 总结：{item.get('summary')}")
            points = safe_list(item.get("key_points"))
            if points:
                lines.append("- 要点：")
                lines.extend(f"  - {p}" for p in points)
            lines.append("")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def render_quotes(items: list[Any]) -> str:
    if not items:
        return "- unknown\n"
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- “{item.get('quote') or ''}” — {item.get('speaker_or_context') or 'unknown'}")
            if item.get("why_it_matters"):
                lines.append(f"  - 重要性：{item.get('why_it_matters')}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_terms(items: list[Any]) -> str:
    if not items:
        return "- unknown\n"
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- **{item.get('term') or ''}**：{item.get('explanation') or ''}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_wiki_page(
    *,
    page_path: Path,
    raw_rel: str,
    slug: str,
    summary: dict[str, Any],
    page_context: dict[str, Any],
    transcription: dict[str, Any],
    source_url: str,
    site_report_url: str,
) -> str:
    today = datetime.now().date().isoformat()
    created = existing_created(page_path, today)
    title = summary.get("title") or page_context.get("official_title") or transcription.get("title") or slug
    podcast = summary.get("podcast") or page_context.get("podcast_title") or "unknown"
    published_time = summary.get("published_time") or page_context.get("published_time") or "unknown"
    outline = summary.get("official_outline") or page_context.get("outline") or []
    related = ["codex", "how-openai-uses-codex", "opencode", "cc-connect", "frontier-engineering"]
    tags = ["agent", "inference", "prediction", "company", "code-study"]
    lines = [
        "---",
        f"title: {title} — LLM Summary",
        f"created: {created}",
        f"updated: {today}",
        "type: summary",
        f"tags: {yaml_list(tags)}",
        f"sources: [{raw_rel}]",
        "---",
        "",
        f"# {title}",
        "",
        f"> 小宇宙播客 LLM 总结入库页。原始 LLM 输出保存在 `{raw_rel}`。",
        "",
        "## Source",
        f"- 播客：{podcast}",
        f"- 发布时间：{published_time}",
        f"- 小宇宙：{source_url or page_context.get('source_url') or 'unknown'}",
        f"- 本地报告：{site_report_url or 'unknown'}",
        f"- Summary model：{summary.get('summary_model') or 'unknown'}",
        "",
        "## TL;DR",
        "",
        str(summary.get("tldr") or "unknown"),
        "",
        "## Theme",
        "",
        str(summary.get("theme") or "unknown"),
        "",
        "## Official outline",
        "",
        render_outline(safe_list(outline)).rstrip(),
        "",
        "## Topic summary",
        "",
        render_topic_summary(safe_list(summary.get("topic_summary"))).rstrip(),
        "",
        "## Golden quotes",
        "",
        render_quotes(safe_list(summary.get("golden_quotes"))).rstrip(),
        "",
        "## Key takeaways",
        "",
        "\n".join(f"- {x}" for x in safe_list(summary.get("key_takeaways"))) or "- unknown",
        "",
        "## Entities and terms",
        "",
        render_terms(safe_list(summary.get("entities_and_terms"))).rstrip(),
        "",
        "## Caveats",
        "",
        "\n".join(f"- {x}" for x in safe_list(summary.get("caveats"))) or "- unknown",
        "",
        "## Related wiki pages",
        "",
        " ".join(wiki_link(x) for x in related),
        "",
        "## Notes",
        "",
        "- 这是一页 `type: summary` 的播客摘要，适合作为后续拆分实体/概念页的入口。",
        "- 与 AI coding agent、模型公司战略、模型作为操作系统相关的后续综合可链接到上述相关页。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def update_index(index_path: Path, page_slug: str, title: str, tldr: str) -> None:
    today = datetime.now().date().isoformat()
    index = read_text(index_path, "# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Comparisons\n\n## Queries\n")
    summary_line = re.sub(r"\s+", " ", tldr).strip()[:120] or title
    entry = f"- [[{page_slug}]] — {summary_line}"
    existed = f"[[{page_slug}]]" in index
    if "## Summaries" not in index:
        if "## Queries" in index:
            index = index.replace("## Queries", "## Summaries\n\n## Queries", 1)
        else:
            index = index.rstrip() + "\n\n## Summaries\n"
    if not existed:
        index = re.sub(r"(## Summaries\n)(.*?)(\n## |\Z)", lambda m: m.group(1) + _insert_entry(m.group(2), entry) + m.group(3), index, count=1, flags=re.S)
    else:
        index = re.sub(rf"^- \[\[{re.escape(page_slug)}\]\].*$", entry, index, flags=re.M)
    total = len(re.findall(r"^- \[\[[^\]]+\]\]", index, flags=re.M)) + len(re.findall(r"^- \[[^\]]+\]\([^\)]+\)", index, flags=re.M))
    index = re.sub(r"Last updated:\s*\d{4}-\d{2}-\d{2}\s*\|\s*Total pages:\s*\d+", f"Last updated: {today} | Total pages: {total}", index)
    if "Last updated:" not in index:
        index = index.replace("> Read this first to find relevant pages for any query.", f"> Read this first to find relevant pages for any query.\n> Last updated: {today} | Total pages: {total}")
    index_path.write_text(index.rstrip() + "\n", encoding="utf-8")


def _insert_entry(section_body: str, entry: str) -> str:
    lines = [line for line in section_body.strip().splitlines() if line.strip()]
    lines.append(entry)
    lines = sorted(set(lines), key=lambda s: s.lower())
    return ("\n" + "\n".join(lines) + "\n") if lines else f"\n{entry}\n"


def append_log(log_path: Path, page_rel: str, raw_rel: str, title: str) -> None:
    existing = read_text(log_path)
    if page_rel in existing and raw_rel in existing and f"Podcast LLM summary — {title}" in existing:
        return
    today = datetime.now().date().isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## [{today}] ingest | Podcast LLM summary — {title}\n")
        f.write(f"- 创建/更新摘要页 `{page_rel}`\n")
        f.write(f"- 保存 LLM 原始输出 `{raw_rel}`\n")
        f.write("- 更新 index.md：加入 Summaries 条目\n")


def export_summary(args: argparse.Namespace) -> dict[str, Any]:
    wiki = args.wiki_path.expanduser().resolve()
    summary_json_path = args.summary_json.expanduser().resolve()
    output_dir = summary_json_path.parent
    summary = read_json(summary_json_path, {}) or {}
    summary_md_path = (args.summary_md or (output_dir / "podcast_summary.md")).expanduser().resolve()
    page_context_path = (args.page_context or (output_dir / "episode_page_context.json")).expanduser().resolve()
    transcription_path = (args.transcription or (output_dir / "transcription_cuda.json")).expanduser().resolve()
    site_meta_path = output_dir / "site_meta.json"
    page_context = read_json(page_context_path, {}) or {}
    transcription = read_json(transcription_path, {}) or {}
    site_meta = read_json(site_meta_path, {}) or {}
    title = summary.get("title") or page_context.get("official_title") or transcription.get("title") or "Podcast LLM Summary"
    episode_id = page_context.get("episode_id") or transcription.get("episode_id") or "podcast"
    slug = args.slug or site_meta.get("slug") or slugify(title, f"podcast-{episode_id}")
    if not str(slug).startswith("podcast-"):
        page_slug = f"podcast-{slug}"
    else:
        page_slug = str(slug)
    source_url = args.source_url or page_context.get("source_url") or transcription.get("url") or ""
    site_report_url = args.site_report_url or ""

    raw_body = render_raw_source(read_text(summary_md_path), summary, page_context, source_url)
    raw_hash = hashlib.sha256(raw_body.encode("utf-8")).hexdigest()
    raw_rel = f"raw/podcast-summaries/{page_slug}-llm-summary.md"
    raw_path = wiki / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text = "\n".join([
        "---",
        f"source_url: {source_url}",
        f"ingested: {datetime.now().date().isoformat()}",
        f"sha256: {raw_hash}",
        "---",
        "",
        raw_body.rstrip(),
        "",
    ])
    raw_path.write_text(raw_text, encoding="utf-8")

    page_rel = f"queries/{page_slug}.md"
    page_path = wiki / page_rel
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        render_wiki_page(
            page_path=page_path,
            raw_rel=raw_rel,
            slug=page_slug,
            summary=summary,
            page_context=page_context,
            transcription=transcription,
            source_url=source_url,
            site_report_url=site_report_url,
        ),
        encoding="utf-8",
    )
    update_index(wiki / "index.md", page_slug, title, str(summary.get("tldr") or summary.get("theme") or ""))
    append_log(wiki / "log.md", page_rel, raw_rel, title)
    return {
        "status": "exported",
        "wiki_path": str(wiki),
        "page": str(page_path),
        "raw_source": str(raw_path),
        "index": str(wiki / "index.md"),
        "log": str(wiki / "log.md"),
        "slug": page_slug,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path, help="Path to output/podcast_summary.json")
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--page-context", type=Path, default=None)
    parser.add_argument("--transcription", type=Path, default=None)
    parser.add_argument("--wiki-path", type=Path, default=DEFAULT_WIKI_PATH)
    parser.add_argument("--slug", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--site-report-url", default="")
    args = parser.parse_args()
    print(json.dumps(export_summary(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
