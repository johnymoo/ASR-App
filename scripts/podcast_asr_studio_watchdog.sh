#!/usr/bin/env bash
set -euo pipefail
PORT="${PODCAST_ASR_PORT:-8020}"
HOST="${PODCAST_ASR_HOST:-0.0.0.0}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR="${PODCAST_ASR_APP_DIR:-$SCRIPT_DIR}"
LOG_DIR="${PODCAST_ASR_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/podcast-asr}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/podcast_asr_studio.log"
PIDFILE="$LOG_DIR/podcast_asr_studio.pid"

is_port_up() {
  python3.12 - "$PORT" <<'PY'
import socket, sys
port=int(sys.argv[1])
s=socket.socket(); s.settimeout(1)
try:
    s.connect(('127.0.0.1', port))
    print('up')
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
}

if is_port_up >/dev/null 2>&1; then
  exit 0
fi

# Clean stale PID file if present.
if [[ -f "$PIDFILE" ]]; then
  OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    # Process exists but port was not reachable; leave it alone and report.
    echo "Podcast ASR Studio process $OLD_PID exists but port $PORT is not reachable; not starting duplicate."
    exit 1
  fi
fi

nohup python3.12 -m uvicorn podcast_asr_studio_server:app \
  --host "$HOST" --port "$PORT" --app-dir "$APP_DIR" \
  >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 2
if is_port_up >/dev/null 2>&1; then
  echo "Podcast ASR Studio started on http://$(hostname -I | awk '{print $1}'):$PORT/static/podcast-asr/index.html pid=$PID"
  exit 0
fi

echo "Podcast ASR Studio failed to start; see $LOG" >&2
exit 1
