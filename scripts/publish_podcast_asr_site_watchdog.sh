#!/usr/bin/env bash
set -euo pipefail
OUT=$(python3.12 "${PODCAST_ASR_PUBLISHER:-$HOME/podcast/publish_podcast_asr_site.py}" --only-if-changed 2>&1 || true)
# The publisher is intentionally silent when nothing changed.
if [[ -n "$OUT" ]]; then
  echo "播客 ASR 网站已自动更新："
  echo "$OUT"
fi
