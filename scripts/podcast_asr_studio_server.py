#!/usr/bin/env python3
"""Podcast ASR Studio web server.

Serves the generated SenseVoice static site and exposes persistent podcast ASR
background-task APIs. Read-only for published files; task creation starts the
existing Xiaoyuzhou pipeline server-side.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

SCRIPTS_DIR = Path(os.environ.get("PODCAST_SCRIPTS_DIR", Path(__file__).parent)).expanduser()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from podcast_asr_task_api import router as podcast_router  # noqa: E402
from meeting_asr_task_api import router as meeting_router  # noqa: E402

STATIC_ROOT = Path(os.environ.get("SENSEVOICE_STATIC_ROOT", "~/deployments/sensevoice/static")).expanduser()
PODCAST_INDEX = STATIC_ROOT / "podcast-asr" / "index.html"
MEETING_INDEX = STATIC_ROOT / "meeting-asr" / "index.html"

app = FastAPI(title="Podcast ASR Studio", version="1.1.0")
app.include_router(podcast_router)
app.include_router(meeting_router)


@app.middleware("http")
async def disable_dynamic_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith(("/api/podcast-asr/", "/api/meeting-asr/")) or path in {
        "/static/podcast-asr/index.html",
        "/static/podcast-asr/episodes.json",
        "/static/meeting-asr/index.html",
    }:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def root():
    return RedirectResponse(url="/static/podcast-asr/index.html")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "static_root": str(STATIC_ROOT),
        "podcast_index_exists": PODCAST_INDEX.exists(),
        "meeting_index_exists": MEETING_INDEX.exists(),
    }


@app.get("/legacy")
def legacy():
    legacy_index = STATIC_ROOT / "index.html"
    if legacy_index.exists():
        return FileResponse(legacy_index)
    return RedirectResponse(url="/static/podcast-asr/index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_ROOT), html=True), name="static")
