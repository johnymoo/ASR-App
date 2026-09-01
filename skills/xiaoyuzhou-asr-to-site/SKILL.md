---
name: xiaoyuzhou-asr-to-site
description: "Use when the user sends a Xiaoyuzhou episode link and wants automatic GPU ASR transcription, LLM summary, and static website publishing on GB10."
platforms: [linux]
tags: [xiaoyuzhou, podcast, asr, sensevoice, gpu, qwen, website]
---

# Xiaoyuzhou ASR to Website

## When to use

Use when Spotty sends a 小宇宙 / Xiaoyuzhou episode URL and asks to transcribe it, summarize it, or import it into the local website.

Default expectation: do the full pipeline, not just a plan:

1. Resolve episode metadata and audio.
2. Download audio into a standard workspace.
3. Convert to 16 kHz mono WAV.
4. Benchmark CPU vs CUDA SenseVoice on the same 5-minute slice.
5. Transcribe the complete episode on GPU/CUDA.
6. Generate a structured Chinese summary with local vLLM on `localhost:8004`.
7. Publish report, full transcript, summary, and downloads to the SenseVoice static site.
8. Export the final LLM summary into the LLM Wiki at `$WIKI_PATH`.
9. Verify HTTP URLs and report the links.

## Primary command

Use the orchestrator:

```bash
python3.12 $PODCAST_ROOT/xiaoyuzhou_asr_to_site.py \
  'https://www.xiaoyuzhoufm.com/episode/<episode-id>'
```

For the known SpaceX episode, the existing slug is preserved:

```text
/static/podcast-asr/xiaoyuzhou-spacex-asr/
```

The script is resumable. If `output/transcription_cuda.json` already exists, it skips ASR unless `--force-asr` is supplied.

## Important paths

```text
$PODCAST_ROOT/xiaoyuzhou_asr_to_site.py       # end-to-end orchestrator
$PODCAST_ROOT/asr_pipeline_template.py        # per-episode SenseVoice pipeline template
$PODCAST_ROOT/generate_podcast_summary.py     # Qwen/vLLM summary generator
$PODCAST_ROOT/export_podcast_summary_to_wiki.py # LLM Wiki exporter
$PODCAST_ROOT/publish_podcast_asr_site.py     # static website publisher
$SENSEVOICE_STATIC_ROOT/podcast-asr/ # published site
```

Standard workspace:

```text
$PODCAST_ROOT/xiaoyuzhou_<episode-id>/
  input/episode.m4a
  audio/episode_16k.wav
  chunks/*.wav
  transcripts/chunk_*_cuda.{json,txt}
  output/transcription_cuda.json
  output/transcript_cuda.{md,txt,srt}
  output/benchmark_cpu_gpu_*.json
  output/podcast_summary.{json,md}
  output/site_meta.json
  logs/*.log
```

## Episode page context and summary template

Before calling the LLM, extract and persist Xiaoyuzhou public page context. Do not rely on transcript-only summarization when the episode page has show notes.

For Xiaoyuzhou URLs, collect at least:

- `official_title` / `podcast_title`
- `episode_description` and show notes body
- `published_time`, duration, playback/comment counts if visible
- cover image URL if available
- official `OUTLINE` timestamps and section titles
- links / disclaimer / contact blocks
- top comments only if useful, labeled as audience comments, not source facts
- direct `audio_url`

Save the context under:

```text
$PODCAST_ROOT/xiaoyuzhou_<episode-id>/output/episode_page_context.json
$PODCAST_ROOT/xiaoyuzhou_<episode-id>/output/episode_page_context.md
```

Then pass this page context into `generate_podcast_summary.py` together with the ASR transcript. The LLM prompt should explicitly prefer official title/outline/show notes for structure and metadata, and use ASR for actual discussion details, quotes, and nuanced synthesis. This avoids mojibake titles, improves topic boundaries, and prevents the model from inventing host/guest info already present on the page.

`generate_podcast_summary.py` calls:

```text
http://127.0.0.1:8004/v1/chat/completions
model: qwen3.6-35b-fp8
chat_template_kwargs: {"enable_thinking": false}
```

The output JSON must include:

- `title` — official title when available
- `podcast` — podcast/show name
- `published_time` — publication time if available
- `official_outline` — official timestamped outline/show notes sections
- `theme` — 核心主题
- `guests` — 嘉宾/主持、身份、依据
- `background` — 讨论背景
- `topic_summary` — 分话题总结、时间范围、要点; align with official outline when possible
- `golden_quotes` — 金句、上下文、价值; derive from transcript unless quoted in show notes
- `key_takeaways` — 关键洞察
- `entities_and_terms` — 专名术语解释
- `caveats` — ASR/专名不确定处
- `tldr` — 网站索引卡片摘要

## LLM Wiki export

After `podcast_summary.json` / `podcast_summary.md` are generated, export the final LLM summary into the local LLM Wiki:

```bash
python3.12 $PODCAST_ROOT/export_podcast_summary_to_wiki.py \
  $PODCAST_ROOT/xiaoyuzhou_<episode-id>/output/podcast_summary.json \
  --wiki-path $WIKI_PATH
```

The orchestrator does this automatically unless `--skip-wiki-export` is passed. The exporter writes:

```text
$WIKI_PATH/raw/podcast-summaries/podcast-<slug>-llm-summary.md
$WIKI_PATH/queries/podcast-<slug>.md
```

It also updates `$WIKI_PATH/index.md` under `## Summaries` and appends to `$WIKI_PATH/log.md`. The wiki page uses existing taxonomy tags (`agent`, `inference`, `prediction`, `company`, `code-study`) and links out to at least two related pages such as `[[codex]]`, `[[how-openai-uses-codex]]`, `[[opencode]]`, and `[[cc-connect]]`.

## Website publication

Run or rely on the orchestrator:

```bash
python3.12 $PODCAST_ROOT/publish_podcast_asr_site.py
```

On the GB10 SenseVoice server, the user-facing root `/` should redirect to the new studio page `/static/podcast-asr/index.html`; the old single-file upload UI is kept at `/legacy`. Verify this route relationship after web-server changes:

```bash
curl -sI http://127.0.0.1:8020/
curl -L http://127.0.0.1:8020/ | grep 'Podcast ASR Studio'
curl -L http://127.0.0.1:8020/legacy | grep '播客转写'
```

Published URLs:

```text
http://127.0.0.1:8020/static/podcast-asr/index.html
http://127.0.0.1:8020/static/podcast-asr/<slug>/index.html
http://127.0.0.1:8020/static/podcast-asr/<slug>/full.html
```

LAN URL uses:

```text
http://<LAN-IP>:8020/static/podcast-asr/...
```

The publisher also mirrors existing legacy `/static/<slug>/` directories so old links remain valid.

### Site information architecture / redesign expectations

When improving or rebuilding the podcast ASR website, use a coherent product structure instead of a loose collection of generated episode pages:

1. **Home / Studio page** — split into two clear zones:
   - **Import area**: paste Xiaoyuzhou URL or upload audio; show the pipeline steps `抓取介绍 → GPU ASR → LLM 总结 → 发布` and, once implemented, task progress.
   - **Library/index area**: searchable cards for already-transcribed episodes with title, duration, chars, chunk success, coverage, GPU speed, report/full/download actions.
2. **Episode report page** — standardize each episode as:
   - official page context / show notes;
   - official OUTLINE;
   - LLM summary grounded in page context + ASR;
   - run metrics and CPU/GPU benchmark;
   - full transcript preview/search and downloads.
   - Keep hero metrics visually secondary: compact 2×2 or inline cards, smaller numerals, weaker background/border, and no tall “pillar” cards that compete with the episode title/summary.
3. **Prototype before replacement** — for larger UX changes, publish a disposable prototype under a separate static path such as `/static/podcast-asr-prototype/` and show the user before replacing production pages.
4. **0.0.0.0 verification** — when the user asks to expose it, verify the actual listener (`ss -ltnp`) and HTTP 200 via both localhost and the LAN URL. Do not equate a localhost-only check with LAN availability.

See `references/site-redesign-ia.md` for the concrete prototype layout and verification checklist.

## Automatic watchdog

A no-agent cron job watches for changed/new completed outputs every 5 minutes:

```text
Job name: 自动发布播客 ASR 网站索引
Script: ~/.hermes/profiles/capital-avatar/scripts/publish_podcast_asr_site_watchdog.sh
```

The watchdog is silent when no new transcript changed.

## Repository packaging / PR workflow

When asked to push this pipeline into a public/team repo, package the class-level workflow rather than a live episode workspace:

- Use a project subdirectory such as `xiaoyuzhou-podcast-asr/`; do not add implementation files at repo root.
- Include reusable scripts, README, env example, and optionally this skill as operator docs.
- Do not stage raw audio, WAV chunks, private transcripts, generated site screenshots/HTML, logs, `__pycache__`, model weights, `.env`, credentials, cookies, or tokens.
- Sanitize personal home paths and private LAN URLs into env-driven defaults/placeholders.
- In dirty repos, stage only explicit pathspecs and verify with `git diff --cached --name-only` before committing.
- For staged scans use `git grep --cached ... -- <pathspec>`; putting `--cached` after `--` is invalid and can create misleading fatal output.

See `references/repo-packaging.md` for the concrete sanitization checklist, staged-scan commands, and PR validation sequence.

## Recovery, redesign, and post-run cleanup

See `references/asr-recovery-and-title-cleanup.md` for the concrete retry and metadata repair pattern.
See `references/site-redesign-and-page-context-implementation.md` for the implemented site IA, page-context extraction, summary prompt integration, and verification pattern.
See `references/wiki-export-and-ui-polish.md` for LLM Wiki export details and the compact hero-metrics UI correction.

- If the orchestrator completes but `transcription_cuda.json` reports failed chunks, do **not** treat the website as final. Inspect `summary.ok_chunks` / `summary.failed_chunks` and rerun the per-episode pipeline with `--force`:
  ```bash
  cd $PODCAST_ROOT/xiaoyuzhou_<episode-id>
  python3.12 asr_pipeline.py transcribe --device cuda --language zh --chunk-seconds 300 --overlap-seconds 5 --force
  python3.12 $PODCAST_ROOT/xiaoyuzhou_asr_to_site.py '<episode-url>' --force-summary
  ```
  A transient CUDA OOM on one chunk may clear after rerunning the full ASR process because the model/GPU state is rebuilt.
- If Xiaoyuzhou public-page metadata is mojibake but the LLM summary recovered a good Chinese title, update `output/episode_metadata.json`, `output/transcription_cuda.json`, and the first heading of `output/transcript_cuda.md`, rebuild the zip, then republish. This keeps the website title and download artifacts readable.
- Verification should include machine checks: HTTP 200 for report/full/summary/transcript URLs plus content needles (`LLM 总结`, `主题`, `嘉宾`, `背景`, `讨论的话题总结`, `金句`, `完整转写稿`, `CPU / GPU 对比`). Browser screenshot is useful when practical, but large full-transcript pages can be accepted on HTTP/content checks if visual rendering is slow.

## Verification checklist

After each run:

```bash
curl -I http://127.0.0.1:8020/static/podcast-asr/index.html
curl -I http://127.0.0.1:8020/static/podcast-asr/<slug>/index.html
curl -I http://127.0.0.1:8020/static/podcast-asr/<slug>/full.html
curl -I http://127.0.0.1:8020/static/podcast-asr/<slug>/podcast_summary.md
```

Check page content contains:

- `LLM 总结`
- `主题`
- `嘉宾`
- `背景`
- `讨论的话题总结`
- `金句`
- `完整转写稿`
- `CPU / GPU 对比`

Use headless Chromium screenshot for visual verification when practical.

## Pitfalls

- Do not paste full transcripts into Telegram. Send file attachments or website links.
- Do not copy raw M4A/WAV/chunk WAVs to the website unless asked.
- Public Xiaoyuzhou HTML may fail to expose audio. If that happens, pass `--audio-url` or use authenticated opencli credentials; never expose credentials in summaries.
- The SenseVoice upload page may still run CPU; this pipeline uses GPU via the generated per-episode script.
- Qwen 8004 may return reasoning-only output if thinking is enabled; keep `enable_thinking=false` for summary generation.
