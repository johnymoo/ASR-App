# x570 CPU ASR Deployment

Status: production-verified on 2026-08-30.

## Topology

- The Podcast Studio API and static site remain on the gateway host at port 8020.
- SenseVoiceSmall podcast and meeting inference runs in the independent
  `x570-podcast-asr` Compose service.
- The worker is published on the x570 LAN address at port 18021.
- The gateway sets `PODCAST_ASR_API_URL` and `MEETING_ASR_ENDPOINT` to that
  worker. No GPU ASR process is required on the gateway.
- Qwen on x570 and DeepSeek on the gateway are separate workloads and must not
  be stopped or recreated when deploying ASR.

## Model layout

The Compose file expects these read-only/runtime mounts:

```text
models/SenseVoiceSmall
models/speech_fsmn_vad
cache/modelscope/models/iic/speech_campplus_sv_zh-cn_16k-common
```

CAM++ is required only for meeting speaker diarization. Set the corresponding
host paths in `.env` when the models are stored elsewhere.

## Start and status

```bash
cd services/x570-asr
docker compose up -d --build
docker compose ps
curl -fsS http://127.0.0.1:18021/healthz
curl -fsS http://127.0.0.1:18021/readyz
```

The service uses `restart: unless-stopped`. Docker itself must be enabled at
boot for automatic recovery.

## Acceptance evidence

- A 12-chunk, 3439.811-second podcast canary completed without failed chunks at
  34.56x realtime.
- WAV and MP3 meeting uploads of 45 seconds each completed end to end with two
  detected speakers and eight merged turns.
- Meeting inference took approximately 2.8 seconds after model warm-up.
- The worker used approximately 1.4 GiB memory in the accepted CPU profile.
- Worker health reported `device=cpu`, `model_loaded=true`, and no OOM state.
- Existing Qwen and DeepSeek health probes remained successful.

## Rollback

```bash
docker compose stop
```

Unset `PODCAST_ASR_API_URL` to restore the pipeline's existing local CUDA path.
Meeting uploads require a reachable `MEETING_ASR_ENDPOINT`; leave the meeting
entry disabled if no diarization worker is available.
