# Nemotron Dictation ASR Evaluation

- Cases: 100
- Exact token matches: 43/100 (43.0%)
- Token error rate <= 10%: 64/100 (64.0%)
- Token error rate > 35%: 12/100 (12.0%)
- Average token error rate: 12.0%
- Elapsed wall time: 79.7s

## Category Summary

| Category | Cases | Avg TER | Exact |
|---|---:|---:|---:|
| English daily | 10 | 0.0% | 10/10 |
| English punctuation | 5 | 4.3% | 3/5 |
| 专有名词 | 10 | 14.7% | 5/10 |
| 中文日常 | 10 | 6.0% | 2/10 |
| 中文标点 | 5 | 2.0% | 3/5 |
| 中英混合 | 15 | 27.5% | 1/15 |
| 代码命令 | 10 | 26.3% | 3/10 |
| 数字日期 | 10 | 14.3% | 5/10 |
| 输入法操作 | 5 | 8.7% | 1/5 |
| 边界场景 | 10 | 7.5% | 4/10 |
| 长句 | 10 | 2.6% | 6/10 |

## Worst Cases

### names_03 - 专有名词 - TER 75.0%

- Reference: MLX 在 Apple Silicon 上运行速度很重要。
- Raw: MLX 在 Apple Chund。
- Processed: MLX 在 Apple Chund。

### numbers_03 - 数字日期 - TER 64.7%

- Reference: 我的手机号是 一三八 零零一三 八零零零。
- Raw: 我的手机号是 1380013800。
- Processed: 我的手机号是 1380013800。

### code_01 - 代码命令 - TER 55.6%

- Reference: Run git status, then run python minus m unittest.
- Raw: Rungit status then run Python MUNTET.
- Processed: Rungit status then run Python MUNTET.

### code_10 - 代码命令 - TER 53.8%

- Reference: 把日志写到 Library slash Logs slash Nemotron Dictation dot log。
- Raw: 把日子写到 Library slash Luxe。
- Processed: 把日子写到 Library slash Luxe。

### numbers_02 - 数字日期 - TER 52.2%

- Reference: 会议时间是二零二六年六月二十八日下午四点十五分。
- Raw: 会议时间是 2026 年 6 月 28 日下午 4 .  15 分。
- Processed: 会议时间是 2026 年 6 月 28 日下午 4 .  15 分。

### mixed_10 - 中英混合 - TER 50.0%

- Reference: 这个 bug 可能和 Accessibility permission 有关。
- Raw: 这个可能和 XSBLT permi。
- Processed: 这个可能和 XSBLT permi。

### code_02 - 代码命令 - TER 50.0%

- Reference: Open src slash nemotron underscore asr slash menubar dot py.
- Raw: OpenSRC slash Nemotron underscore ASR menubar Pi
- Processed: OpenSRC slash Nemotron underscore ASR menubar Pi

### code_08 - 代码命令 - TER 50.0%

- Reference: 请执行 python minus m unittest discover。
- Raw: 请执行 PUNTS Discover.
- Processed: 请执行 PUNTS Discover.

### mixed_07 - 中英混合 - TER 45.5%

- Reference: 我想比较 Whisper large v3 turbo 和 Nemotron ASR。
- Raw: 我想比较 WSFL 和 Namatran ASR。
- Processed: 我想比较 WSFL 和 Namatran ASR。

### mixed_13 - 中英混合 - TER 44.4%

- Reference: I said 会议纪要, not meeting medicine.
- Raw: I said not meeting medicine.
- Processed: I said not meeting medicine.

### mixed_05 - 中英混合 - TER 38.5%

- Reference: 把 command V 改成模拟粘贴到当前 cursor。
- Raw: 改成模拟粘贴到当前科色。
- Processed: 改成模拟粘贴到当前科色。

### mixed_08 - 中英混合 - TER 36.4%

- Reference: 如果 input monitoring 没开，global shortcut 会失败。
- Raw: 如果 Input monitoring没开, Global Shortcu。
- Processed: 如果 Input monitoring没开，Global Shortcu。

### mixed_01 - 中英混合 - TER 33.3%

- Reference: 这个 app 的名字叫 Nemotron Dictation。
- Raw: 这个 AP 的名字叫 NAMT。
- Processed: 这个 AP 的名字叫 NAMT。

### mixed_15 - 中英混合 - TER 33.3%

- Reference: Please paste the transcript into TextEdit.
- Raw: Please paste the transcript into text edit.
- Processed: Please paste the transcript into text edit.

### mixed_14 - 中英混合 - TER 25.0%

- Reference: The shortcut key is right Option on macOS.
- Raw: The shortcut key is right option on Mac OS.
- Processed: The shortcut key is right option on Mac OS.
