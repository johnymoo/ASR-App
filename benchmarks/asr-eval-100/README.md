# ASR Eval 100 Benchmark

Small dictation-style ASR benchmark used to track Podcast ASR accuracy and latency across model/configuration updates.

## Contents

```text
benchmarks/asr-eval-100/
├── dataset/
│   ├── audio/                         # 100 WAV samples
│   ├── baseline-nemotron-results.json # references + baseline transcripts + TER/latency
│   ├── baseline-nemotron-results.csv
│   └── baseline-nemotron-report.md
├── results/
│   └── sensevoice-small-cuda-20260629/
│       ├── results.json
│       ├── results.csv
│       ├── report.md
│       └── comparison.png
└── scripts/
    ├── eval_sensevoice_asr100.py
    └── make_comparison_image.py
```


## Dataset origin

The 100 WAV samples are synthetic dictation audio generated locally with macOS system text-to-speech voices for Podcast ASR regression testing:

- **Tingting** for Chinese content (categories: 中文日常, 中文标点, 数字日期, 专有名词, 长句, 输入法操作, 中英混合, 边界场景, 代码命令)
- **Samantha** for English content (categories: English daily, English punctuation)

Each sample is a short, manually crafted prompt sentence rendered to 16-bit mono 16 kHz PCM. The dataset does not contain third-party audio or copyrighted recordings. The reference transcripts and baseline Nemotron TER/latency measurements are stored in dataset/baseline-nemotron-results.json.

## Metric definitions

- **TER**: token error rate against the `reference` field from `baseline-nemotron-results.json`.
- **TER tokenizer**: CJK characters are individual tokens; runs of Unicode letters/numbers are word tokens; punctuation is removed without adding separators. This matches the TER values shipped in `baseline-nemotron-results.json`.
- **Latency**: per-file `model.generate()` wall time, excluding one-time model load.

## Current checked-in result

`results/sensevoice-small-cuda-20260629/` was produced on GB10 with Podcast ASR's SenseVoiceSmall/FunASR stack on CUDA while vLLM remained running and `privacy-filter.service` was stopped.

| Model / mode | Avg TER | Avg latency |
|---|---:|---:|
| FunASR Nano plain | 16.4% | 0.39s |
| FunASR Nano hotwords | 9.6% | 0.31s |
| Nemotron MLX | 19.9% | 1.31s |
| Whisper large-v3-turbo MLX | 11.4% | 3.04s |
| SenseVoiceSmall CUDA | 14.83% | 0.153s |

Additional SenseVoiceSmall CUDA summary:

| Metric | Value |
|---|---:|
| Cases | 100/100 |
| Failed | 0 |
| Exact token matches | 46/100 |
| TER <= 10% | 59/100 |
| TER > 35% | 17/100 |
| Median TER | 5.28% |
| Model load | 1.116s |
| Total wall | 15.327s |

## Re-run benchmark

From the repository root:

```bash
cd xiaoyuzhou-podcast-asr/benchmarks/asr-eval-100

# Defaults match the GB10 Podcast ASR deployment paths. Override paths if needed.
ASR_EVAL_DEVICE=cuda \
python3.12 scripts/eval_sensevoice_asr100.py
```

Useful overrides:

```bash
export SENSEVOICE_SITE_PACKAGES="$HOME/deployments/sensevoice/venv/lib/python3.12/site-packages"
export SENSEVOICE_MODEL_DIR="$HOME/deployments/sensevoice/models/SenseVoiceSmall"
export SENSEVOICE_VAD_MODEL_DIR="$HOME/deployments/sensevoice-docker/modelscope-cache/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
export ASR_EVAL_DEVICE=cuda       # or cpu
export ASR_EVAL_OUT="$PWD/results/sensevoice-small-cuda-$(date +%Y%m%d)"
```

Outputs are written under `results/sensevoice-small-$ASR_EVAL_DEVICE/` by default, or `ASR_EVAL_OUT` if set.

## Regenerate comparison image

For the checked-in result:

```bash
cd xiaoyuzhou-podcast-asr/benchmarks/asr-eval-100
python3.12 scripts/make_comparison_image.py
```

For a new result directory:

```bash
ASR_EVAL_RESULT_DIR="$PWD/results/<new-result-dir>" \
python3.12 scripts/make_comparison_image.py
```

## Quick verification without GPU

```bash
python3.12 - <<'PY'
import json
from pathlib import Path
p = Path('results/sensevoice-small-cuda-20260629/results.json')
s = json.loads(p.read_text(encoding='utf-8'))['summary']
assert s['cases'] == 100 and s['ok_cases'] == 100
assert round(s['avg_token_error_rate'] * 100, 2) == 14.83
assert round(s['avg_latency_seconds'], 3) == 0.153
print('ok')
PY
```
