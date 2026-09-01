# SenseVoiceSmall ASR Eval 100

- Model: SenseVoiceSmall (/home/YOUR_USERNAME/deployments/sensevoice/models/SenseVoiceSmall)
- Device: cuda
- Cases: 100/100
- Avg TER: 14.83%
- Avg Latency: 0.153s
- Model load: 1.116s
- Total wall: 15.327s

## Category Summary

| Category | Cases | Avg TER | Avg Latency | Exact |
|---|---:|---:|---:|---:|
| 中文日常 | 10 | 2.6% | 0.226s | 7/10 |
| 中文标点 | 5 | 0.0% | 0.154s | 5/5 |
| English daily | 10 | 2.0% | 0.142s | 9/10 |
| English punctuation | 5 | 6.3% | 0.157s | 2/5 |
| 中英混合 | 15 | 23.5% | 0.144s | 1/15 |
| 数字日期 | 10 | 41.5% | 0.133s | 1/10 |
| 代码命令 | 10 | 37.7% | 0.140s | 1/10 |
| 专有名词 | 10 | 17.5% | 0.152s | 4/10 |
| 长句 | 10 | 1.8% | 0.165s | 7/10 |
| 边界场景 | 10 | 3.6% | 0.132s | 7/10 |
| 输入法操作 | 5 | 6.4% | 0.139s | 2/5 |

## Worst Cases

### code_08 - 代码命令 - TER 75.0%

- Reference: 请执行 python minus m unittest discover。
- Hypothesis: Qing Zhixing Pon minus M unit test discover.
- Latency: 0.118s

### code_07 - 代码命令 - TER 71.4%

- Reference: 文件路径是 src slash nemotron underscore asr slash shortcut dot py。
- Hypothesis: We件路 Jing是 S R C slash name underscore A S R slash shortcut dot pie.
- Latency: 0.165s

### code_02 - 代码命令 - TER 70.0%

- Reference: Open src slash nemotron underscore asr slash menubar dot py.
- Hypothesis: Open SRC/ neomeron underscore ASR/menubar.pi.
- Latency: 0.146s

### numbers_08 - 数字日期 - TER 66.7%

- Reference: The price is ninety nine dollars and eighty cents.
- Hypothesis: The price is $99.80.
- Latency: 0.119s

### names_07 - 专有名词 - TER 66.7%

- Reference: The model name is mlx community slash whisper large v three turbo.
- Hypothesis: The model name is MLLX communityity/lash Whiser largege V3 Trbo.
- Latency: 0.177s

### numbers_03 - 数字日期 - TER 64.7%

- Reference: 我的手机号是 一三八 零零一三 八零零零。
- Hypothesis: 我的手机号是13800138000。
- Latency: 0.138s

### numbers_01 - 数字日期 - TER 57.9%

- Reference: 订单号是一二三四五六，金额是九十九点八元。
- Hypothesis: 订单号是123456，金额是99.8元。
- Latency: 0.137s

### mixed_07 - 中英混合 - TER 54.5%

- Reference: 我想比较 Whisper large v3 turbo 和 Nemotron ASR。
- Hypothesis: 我想比较with fair largely center和nemeran ASSR.
- Latency: 0.137s

### mixed_14 - 中英混合 - TER 50.0%

- Reference: The shortcut key is right Option on macOS.
- Hypothesis: The shortcutkey is right option on Mac OS.
- Latency: 0.131s

### numbers_06 - 数字日期 - TER 50.0%

- Reference: The build number is one zero four seven.
- Hypothesis: The build number is 1047.
- Latency: 0.142s

### numbers_02 - 数字日期 - TER 47.8%

- Reference: 会议时间是二零二六年六月二十八日下午四点十五分。
- Hypothesis: 会议时间是2026年6月28日下午4点15分。
- Latency: 0.146s

### numbers_04 - 数字日期 - TER 46.2%

- Reference: 版本号从零点一升级到零点二。
- Hypothesis: 版本号从0.1升级到0.2。
- Latency: 0.117s

### mixed_13 - 中英混合 - TER 44.4%

- Reference: I said 会议纪要, not meeting medicine.
- Hypothesis: I said, not meeting medicine.
- Latency: 0.126s

### code_03 - 代码命令 - TER 42.9%

- Reference: Set the environment variable PYTHONPATH equals src.
- Hypothesis: Set the environment variable P and path equals SRC.
- Latency: 0.190s

### code_09 - 代码命令 - TER 42.9%

- Reference: 函数名是 toggle from shortcut。
- Hypothesis: H Shuing是 toggle from shortcut.
- Latency: 0.115s
