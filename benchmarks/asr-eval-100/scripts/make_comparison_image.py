#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import os

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULT_DIR = Path(os.environ.get("ASR_EVAL_RESULT_DIR", BENCHMARK_DIR / "results" / "sensevoice-small-cuda-20260629")).expanduser()
JSON_PATH = Path(os.environ.get("ASR_EVAL_RESULT_JSON", RESULT_DIR / "results.json")).expanduser()
PNG_PATH = Path(os.environ.get("ASR_EVAL_COMPARISON_PNG", RESULT_DIR / "comparison.png")).expanduser()

summary = json.loads(JSON_PATH.read_text(encoding='utf-8'))['summary']
models = [
    {"name": "FunASR Nano", "mode": "plain", "ter": 16.4, "lat": 0.39, "kind": "baseline"},
    {"name": "FunASR Nano", "mode": "hotwords", "ter": 9.6, "lat": 0.31, "kind": "best"},
    {"name": "Nemotron MLX", "mode": "—", "ter": 19.9, "lat": 1.31, "kind": "baseline"},
    {"name": "Whisper large-v3-turbo MLX", "mode": "—", "ter": 11.4, "lat": 3.04, "kind": "baseline"},
    {
        "name": "Our ASR: SenseVoiceSmall", "mode": "CUDA",
        "ter": round(summary['avg_token_error_rate'] * 100, 2),
        "lat": round(summary['avg_latency_seconds'], 3),
        "kind": "ours",
    },
]

cat_order = ["中文日常", "中文标点", "English daily", "English punctuation", "中英混合", "数字日期", "代码命令", "专有名词", "长句", "边界场景", "输入法操作"]
cat = summary['category_summary']

W, H = 1800, 1320
BG = (248, 250, 252)
CARD = (255, 255, 255)
TEXT = (15, 23, 42)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
ORANGE = (234, 88, 12)
RED = (220, 38, 38)
PURPLE = (124, 58, 237)
YELLOW_BG = (254, 249, 195)
OUR_BG = (239, 246, 255)

def font(size, bold=False):
    candidates = ([
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ] if bold else [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ])
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE = font(54, True)
F_SUB = font(26)
F_H2 = font(34, True)
F_BODY = font(24)
F_BODY_B = font(24, True)
F_SMALL = font(20)
F_TINY = font(18)
F_NUM = font(30, True)
F_BIG = font(48, True)

img = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(img)

def rr(box, radius=24, fill=CARD, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def text(x, y, s, f=F_BODY, fill=TEXT, anchor=None, align='left'):
    d.text((x, y), str(s), font=f, fill=fill, anchor=anchor, align=align)

def pill(x, y, w, h, label, fill, fg=(255, 255, 255)):
    d.rounded_rectangle((x, y, x+w, y+h), radius=h//2, fill=fill)
    text(x+w/2, y+h/2-1, label, F_SMALL, fg, anchor='mm')

# Header
text(70, 55, 'ASR Eval 100 对比结果', F_TITLE)
text(72, 125, '指标：Avg TER 越低越好 · Avg Latency 越低越好 · 我们的 ASR 使用 SenseVoiceSmall / CUDA', F_SUB, MUTED)
pill(1400, 64, 315, 46, f"100 samples · GPU · {summary['failed_cases']} failed", BLUE)

# Left comparison table/card
rr((60, 185, 1060, 650), 28)
text(95, 225, '模型对比（参考截图 + 本次实测）', F_H2)
x0, y0 = 95, 300
cols = [0, 495, 650, 790]
headers = ['Model', 'Mode', 'Avg TER', 'Avg Latency']
for i, h in enumerate(headers):
    text(x0 + cols[i], y0, h, F_BODY_B, MUTED)
d.line((95, 340, 1030, 340), fill=GRID, width=2)
row_h = 64
best_ter = min(m['ter'] for m in models)
best_lat = min(m['lat'] for m in models)
for idx, m in enumerate(models):
    y = 360 + idx * row_h
    if m['kind'] == 'ours':
        rr((80, y-10, 1040, y+48), 16, fill=OUR_BG, outline=(147, 197, 253), width=2)
    elif m['kind'] == 'best':
        rr((80, y-10, 1040, y+48), 16, fill=YELLOW_BG)
    text(x0+cols[0], y, m['name'], F_BODY_B if m['kind']=='ours' else F_BODY, TEXT)
    text(x0+cols[1], y, m['mode'], F_BODY, MUTED)
    ter_color = GREEN if m['ter'] == best_ter else (BLUE if m['kind']=='ours' else TEXT)
    lat_color = GREEN if m['lat'] == best_lat else (BLUE if m['kind']=='ours' else TEXT)
    text(x0+cols[2], y, f"{m['ter']:.2f}%" if m['kind']=='ours' else f"{m['ter']:.1f}%", F_NUM, ter_color)
    text(x0+cols[3], y, f"{m['lat']:.3f}s" if m['kind']=='ours' else f"{m['lat']:.2f}s", F_NUM, lat_color)

# Right KPI cards
rr((1100, 185, 1740, 650), 28)
text(1135, 225, '我们的 ASR 实测摘要', F_H2)
kpis = [
    ('Avg TER', f"{summary['avg_token_error_rate']*100:.2f}%", '准确率：介于 plain 与 hotwords/Whisper 之间', ORANGE),
    ('Avg Latency', f"{summary['avg_latency_seconds']:.3f}s", '本组最快；不含一次性模型加载', GREEN),
    ('Exact Match', f"{summary['exact_token_matches']}/100", f"TER ≤10%: {summary['token_error_rate_le_10_pct']}/100", BLUE),
    ('Total Wall', f"{summary['elapsed_wall_seconds']:.2f}s", f"Model load: {summary['model_load_seconds']:.2f}s", PURPLE),
]
for i, (label, value, note, color) in enumerate(kpis):
    x = 1135 + (i % 2) * 295
    y = 295 + (i // 2) * 155
    rr((x, y, x+270, y+130), 18, fill=(248, 250, 252), outline=GRID)
    text(x+20, y+20, label, F_SMALL, MUTED)
    text(x+20, y+55, value, F_BIG, color)
    text(x+20, y+103, note, F_TINY, MUTED)

# Category bars card
rr((60, 695, 1740, 1170), 28)
text(95, 735, 'SenseVoiceSmall CUDA 分类 TER', F_H2)
text(95, 785, '弱项集中在数字日期、代码命令、中英混合；日常中文/英文、标点、长句表现较好。', F_BODY, MUTED)
chart_x, chart_y = 100, 850
label_w, bar_w, row_h = 250, 500, 31
max_cat = max(cat[c]['avg_ter'] * 100 for c in cat_order)
for i, c in enumerate(cat_order):
    item = cat[c]
    y = chart_y + i * row_h
    val = item['avg_ter'] * 100
    latency = item['avg_latency_seconds']
    exact = f"{item['exact']}/{item['cases']}"
    text(chart_x, y, c, F_SMALL, TEXT)
    d.rounded_rectangle((chart_x+label_w, y+5, chart_x+label_w+bar_w, y+23), radius=9, fill=(241, 245, 249))
    color = GREEN if val <= 5 else BLUE if val <= 15 else ORANGE if val <= 30 else RED
    d.rounded_rectangle((chart_x+label_w, y+5, chart_x+label_w+bar_w*(val/max_cat), y+23), radius=9, fill=color)
    text(chart_x+label_w+bar_w+20, y, f"{val:.1f}%", F_SMALL, color)
    text(chart_x+label_w+bar_w+105, y, f"lat {latency:.3f}s", F_SMALL, MUTED)
    text(chart_x+label_w+bar_w+245, y, f"exact {exact}", F_SMALL, MUTED)

# Bottom notes, outside chart area
text(70, 1210, 'TER 口径：已对齐 asr-eval-100/results.json；CJK 按字、Unicode 字母/数字按词，标点删除。', F_SMALL, MUTED)
text(70, 1240, 'Latency 口径：单条 model.generate wall time，不含模型加载。', F_SMALL, MUTED)
text(70, 1270, '结论：我们的 ASR 速度最快；准确率优于 FunASR Nano plain 和 Nemotron MLX，但低于 FunASR Nano hotwords 与 Whisper large-v3-turbo MLX。', F_SMALL, TEXT)

img.save(PNG_PATH)
print(PNG_PATH)
