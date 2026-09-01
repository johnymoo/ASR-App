# Xiaoyuzhou Podcast ASR to Static Site

End-to-end pipeline for turning a Xiaoyuzhou episode into GPU SenseVoice ASR artifacts, a local Qwen-generated structured Chinese summary, and a static website page.

## What it does

1. Resolve Xiaoyuzhou episode metadata and audio URL.
2. Download audio into a standard workspace under `PODCAST_ROOT`.
3. Convert audio to 16 kHz mono WAV.
4. Benchmark CPU vs CUDA SenseVoice on an identical slice.
5. Transcribe the full episode with resumable chunks, either on local CUDA or through an OpenAI-compatible remote ASR worker.
6. Generate `podcast_summary.json` and `podcast_summary.md` using a local OpenAI-compatible endpoint.
7. Generate a native GPT Image2 TLDR infographic directly from the structured summary.
8. Publish per-episode report pages and an index under a static web root.
9. Export the final LLM summary into an LLM Wiki (`WIKI_PATH`) for long-term knowledge reuse.
10. Optionally expose a FastAPI background-task API so the website import form keeps running after users leave the page.

## Files

| Path | Purpose |
|---|---|
| `scripts/xiaoyuzhou_asr_to_site.py` | End-to-end orchestrator. |
| `scripts/asr_pipeline_template.py` | Per-episode SenseVoice/FunASR pipeline template. |
| `scripts/generate_podcast_summary.py` | Structured Chinese summary generator using local Qwen/vLLM. |
| `scripts/generate_podcast_tldr_infographic.py` | Native GPT Image2 direct TLDR infographic generator. |
| `scripts/export_podcast_summary_to_wiki.py` | Writes the final LLM summary into `WIKI_PATH` as raw source + curated wiki summary page. |
| `scripts/publish_podcast_asr_site.py` | Static site publisher and index builder. |
| `scripts/podcast_asr_task_api.py` | FastAPI router for persistent background website import jobs. |
| `scripts/remote_asr_client.py` | Resumable manifest client for a remote OpenAI-compatible ASR worker. |
| `scripts/meeting_asr_task_api.py` | Persistent MP3/WAV meeting-upload task API. |
| `scripts/meeting_asr_pipeline.py` | Meeting task client and speaker-labelled TXT formatter. |
| `scripts/podcast_asr_studio_server.py` | Podcast and meeting API/static-site gateway. |
| `services/x570-asr/` | CPU SenseVoice worker, CAM++ diarization endpoint, and Compose deployment. |
| `web/meeting-asr/index.html` | Mobile-capable meeting upload, polling, transcript, and download UI. |
| `scripts/publish_podcast_asr_site_watchdog.sh` | Silent watchdog wrapper for cron/no-agent jobs. |
| `benchmarks/asr-eval-100/` | 100-sample dictation benchmark, baseline results, SenseVoice CUDA result, and reusable evaluation scripts. |
| `skills/xiaoyuzhou-asr-to-site/SKILL.md` | Hermes skill procedure used by the local agent. |

## Requirements

- Python 3.12
- `ffmpeg` / `ffprobe`
- `curl`
- SenseVoice/FunASR dependencies installed in a Python site-packages path
- Local model directories for SenseVoice, punctuation, and VAD
- Optional local vLLM/OpenAI-compatible endpoint for summary generation
- FastAPI, Uvicorn, python-multipart, and Requests for the Studio gateway
- Docker Compose for the isolated x570 CPU worker

## Configuration

The scripts default to local paths, but all environment-specific values are configurable:

| Variable | Default |
|---|---|
| `PODCAST_ROOT` | `~/podcast` |
| `SENSEVOICE_SITE_PACKAGES` | `~/deployments/sensevoice/venv/lib/python3.12/site-packages` |
| `SENSEVOICE_MODEL_DIR` | `~/deployments/sensevoice/models/SenseVoiceSmall` |
| `SENSEVOICE_PUNC_MODEL_DIR` | `~/deployments/sensevoice/models/punc-ct-transformer` |
| `SENSEVOICE_VAD_MODEL_DIR` | `~/deployments/sensevoice-docker/modelscope-cache/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| `SENSEVOICE_STATIC_ROOT` | `~/deployments/sensevoice/static` |
| `SENSEVOICE_WEB_PORT` | `8020` |
| `PODCAST_ASR_SITE_BASE` | `http://127.0.0.1:8020/static/podcast-asr` |
| `WIKI_PATH` | `~/wiki` |
| `PODCAST_WIKI_EXPORT_SCRIPT` | `$PODCAST_ROOT/export_podcast_summary_to_wiki.py` |
| `PODCAST_TLDR_IMAGE_SCRIPT` | `$PODCAST_ROOT/generate_podcast_tldr_infographic.py` |
| `PODCAST_TASK_DIR` | `$PODCAST_ROOT/asr_tasks` |
| `PODCAST_PIPELINE` | `$PODCAST_ROOT/xiaoyuzhou_asr_to_site.py` |
| `PODCAST_LIBRARY_DIR` | `~/deployments/sensevoice/static/podcast-asr` |
| `PODCAST_ASR_API_URL` | empty; uses local CUDA when unset |
| `PODCAST_REMOTE_ASR_CLIENT` | `$PODCAST_ROOT/remote_asr_client.py` |
| `MEETING_ROOT` | `~/meetings` |
| `MEETING_PIPELINE` | script beside `meeting_asr_task_api.py` |
| `MEETING_ASR_ENDPOINT` | `http://127.0.0.1:18021/v1/audio/meeting-transcriptions` |
| `IMAGE2_MODEL` | `gpt-image-2` |
| `IMAGE2_API_KEY` / `OPENAI_API_KEY` | required for native GPT Image2 TLDR image generation |
| `IMAGE2_BASE_URL` / `OPENAI_BASE_URL` | optional OpenAI-compatible image endpoint |

The infographic generator requests the official maximum landscape size
(`3840x2160`) at `high` quality. OpenAI-compatible gateways may return a
smaller image or remap the aspect ratio. Check `requested_pixels`,
`actual_size`, `actual_pixels`, and `size_match` in
`output/tldr_infographic_meta.json` instead of assuming the requested size was
honored.

## Usage

```bash
export PODCAST_ROOT="$HOME/podcast"
python3.12 scripts/xiaoyuzhou_asr_to_site.py \
  'https://www.xiaoyuzhoufm.com/episode/<episode-id>'
```

Set `PODCAST_ASR_API_URL` (or pass `--asr-api-url`) to move only transcription
inference to the remote worker. If it is unset, the existing local CUDA path is
unchanged. A failed or incomplete remote run falls back to the local CPU path;
it never retries CUDA implicitly on the gateway host.

The orchestrator is resumable. It skips completed ASR and summary outputs unless the corresponding `--force-*` flags are supplied. Use `--force-asr --force-summary --force-tldr-image` to recover a stale/partial run and regenerate all downstream artifacts. It exports the final summary into `WIKI_PATH` by default; pass `--skip-wiki-export` to disable that side effect.

## Website background import jobs

The import form in the published index calls `/api/podcast-asr/tasks`. The task API starts the pipeline with `subprocess.Popen(..., start_new_session=True)`, writes persistent JSON state under `PODCAST_TASK_DIR`, and exposes polling endpoints:

```text
POST /api/podcast-asr/tasks
GET  /api/podcast-asr/tasks
GET  /api/podcast-asr/tasks/{job_id}
GET  /api/podcast-asr/tasks/by-episode/{episode_id}
```

Because the job is server-side and state is persisted, users may leave or close the website after submitting a Xiaoyuzhou URL. When they return, the front-end restores the last active task from `localStorage`, polls the API, and shows a persistent task panel instead of relying on an auto-dismissing toast.

To mount the router in an existing FastAPI app:

```python
from scripts.podcast_asr_task_api import router as podcast_asr_router
app.include_router(podcast_asr_router)
```

## x570 CPU worker

The worker exposes both podcast and meeting contracts:

```text
POST /v1/audio/transcriptions
POST /v1/audio/meeting-transcriptions
GET  /healthz
GET  /readyz
```

Deploy it independently from other x570 workloads:

```bash
cd services/x570-asr
cp ../../configs/example.env .env  # adjust model/cache host paths first
docker compose up -d --build
curl http://127.0.0.1:18021/healthz
```

The Compose service defaults to 8 CPUs and 8 GiB RAM, uses
`restart: unless-stopped`, and publishes only on `ASR_BIND_HOST`. See
[`docs/x570-cpu-deployment.md`](docs/x570-cpu-deployment.md) for the verified
production topology and acceptance evidence.

## Meeting transcription MVP

The Studio gateway mounts both task routers and serves the meeting page:

```bash
mkdir -p "$SENSEVOICE_STATIC_ROOT/meeting-asr"
cp web/meeting-asr/index.html "$SENSEVOICE_STATIC_ROOT/meeting-asr/index.html"
python3.12 -m uvicorn podcast_asr_studio_server:app \
  --app-dir scripts --host 0.0.0.0 --port 8020
```

The page uploads one MP3/WAV file up to 256 MiB, starts a persistent background
task, polls every three seconds, and restores the active task after a refresh.
It supports automatic CAM++ speaker clustering or an explicit speaker count and
produces TXT and JSON downloads. TXT blocks use this contract:

```text
发言人 1 0:00:00
第一段转写内容。

发言人 2 0:00:02
第二段转写内容。
```

## Outputs

A workspace is created under:

```text
$PODCAST_ROOT/xiaoyuzhou_<episode-id>/
```

Important outputs:

```text
output/transcription_<device>.json
output/transcript_<device>.md
output/transcript_<device>.txt
output/transcript_<device>.srt
output/benchmark_cpu_gpu_*.json
output/podcast_summary.json
output/podcast_summary.md
output/tldr_infographic.png
output/tldr_infographic_meta.json
output/episode_page_context.json
output/episode_page_context.md
output/site_meta.json
```

Wiki export paths:

```text
$WIKI_PATH/raw/podcast-summaries/podcast-<episode-slug>-llm-summary.md
$WIKI_PATH/queries/podcast-<episode-slug>.md
```

Published site paths:

```text
$SENSEVOICE_STATIC_ROOT/podcast-asr/index.html
$SENSEVOICE_STATIC_ROOT/podcast-asr/<episode-slug>/index.html
$SENSEVOICE_STATIC_ROOT/podcast-asr/<episode-slug>/full.html
```

## ASR Benchmark

The `benchmarks/asr-eval-100/` directory contains a reusable 100-sample dictation benchmark for tracking ASR accuracy and latency across model/config updates. It includes the WAV samples, reference transcripts, baseline Nemotron results, the 2026-06-29 SenseVoiceSmall CUDA run, and scripts to re-run the benchmark and regenerate the comparison image.

```bash
cd benchmarks/asr-eval-100
ASR_EVAL_DEVICE=cuda python3.12 scripts/eval_sensevoice_asr100.py
python3.12 scripts/make_comparison_image.py
```

See `benchmarks/asr-eval-100/README.md` for metric definitions, environment overrides, and the checked-in benchmark summary.

## Verification

```bash
python3.12 -m py_compile scripts/*.py
python3.12 -m unittest discover -s tests -v
curl -I http://127.0.0.1:8020/static/podcast-asr/index.html
```

The page should include LLM summary sections, transcript download links, CPU/GPU benchmark metrics, and full transcript links.

## Notes

- Do not commit raw podcast episode audio, generated WAV chunks, model weights, credentials, logs, or private `.env` files. Small curated benchmark audio sets under `benchmarks/` are allowed when they are part of a reproducible test fixture.
- The Xiaoyuzhou public page usually exposes a media URL; authenticated opencli fallback is optional and should not expose credentials.
- Qwen summary calls disable thinking so the final JSON lands in `message.content`.
