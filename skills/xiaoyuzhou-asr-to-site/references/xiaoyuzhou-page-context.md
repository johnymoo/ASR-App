# Xiaoyuzhou Page Context Extraction

Use this reference when improving Xiaoyuzhou ASR summaries.

## Why

The public episode page often contains high-quality editorial metadata that ASR alone cannot recover cleanly:

- official title and podcast/show name
- episode intro / description
- OUTLINE timestamps
- host/guest naming
- series links
- disclaimer/contact notes
- cover image and audio URL

This context should guide LLM structure and metadata. The ASR transcript remains the source for detailed discussion, quotes, and nuanced synthesis.

## Extraction targets

Persist both JSON and Markdown:

```text
output/episode_page_context.json
output/episode_page_context.md
```

Recommended JSON shape:

```json
{
  "episode_id": "...",
  "source_url": "https://www.xiaoyuzhoufm.com/episode/...",
  "official_title": "...",
  "podcast_title": "...",
  "description": "...",
  "published_time": "...",
  "duration_text": "83分钟",
  "play_count": 72746,
  "comment_count": 238,
  "cover_image": "https://...",
  "audio_url": "https://...m4a",
  "outline": [
    {"timestamp": "00:02:00", "title": "第9集季报的概览", "notes": ["..."]}
  ],
  "links": ["..."],
  "disclaimer": "..."
}
```

## Summary prompt integration

Add a section before the transcript:

```text
--- OFFICIAL EPISODE PAGE CONTEXT ---
{episode_page_context_md}
--- END OFFICIAL EPISODE PAGE CONTEXT ---
```

Prompt rule:

```text
优先采用官方页面中的标题、节目名、发布日期、嘉宾/主持和 OUTLINE 作为结构化元数据；
用 ASR 转写补充讨论细节、观点演绎和金句。若二者冲突，说明冲突并标注 caveat。
```

## Verification

After generation, verify:

- title matches official page, no mojibake
- topic boundaries roughly align with official OUTLINE
- summary does not treat audience comments as episode facts
- `episode_page_context.*` is copied into the website/download package if useful
