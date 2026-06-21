# ProsoGatePipeline 设计方案

面向 TTS 微调数据产出的处理 pipeline，核心解决三件事：长音频切分、基频变化筛选、语速变化筛选。

ASR 与强制对齐统一使用 Qwen3 系列：
- ASR：`Qwen3-ASR-1.7B`
- 强制对齐：`Qwen3-ForcedAligner-0.6B`（字/词级时间戳，无音素级）

支持语言：中文（普通话 + 粤语）为主，英文/日韩等其它 Qwen3-ASR 支持的语言走同一套流程，规范化与韵律阈值需另调。

---

## 目标

产出适合 TTS 微调的数据：

- 切分合理的音频片段（3 - 20s）
- 与音频严格对齐的字级时间戳文本
- 每条样本的韵律指标：F0、语速、停顿、能量
- `manifest.jsonl` + 质量报告 + 可回溯的筛选日志

---

## 整体流程

```text
长音频输入
  ↓
[01] 数据导入 ingest
  ↓
[02] 音频质检 audio_qc          -> 24kHz/mono 主版本
  ↓
[03] 重采样 resample            -> 额外产出 16kHz 对齐版本
  ↓
[04] 粗 VAD + 说话人切分 vad_coarse  -> 10-30s 单说话人段
  ↓
[05] Qwen3-ASR 转写             -> 带标点 + 句级时间戳
  ↓
[06] 文本规范化 text_normalize
  ↓
[07] Qwen3-ForcedAligner 对齐   -> 字级时间戳 + confidence
  ↓
[08] 精细切分 fine_segment      -> 3-20s 训练片段
  ↓
[09] 多说话人巡检 spk_consistency  -> 滑窗严格筛单说话人
  ↓
[10] F0 提取 extract_f0
  ↓
[11] 语速指标 rate_metrics
  ↓
[12] 规则过滤 + 综合打分 filter_score
  ↓
[13] 数据集划分 split_dataset
  ↓
[14] 质量报告 report
```

每个 stage 读上一步的 manifest，写自己的 manifest，独立可重跑。

---

## 1. 数据导入

输入目录约定：

```text
audio/
  speaker_001/session_001.wav
  speaker_001/session_002.wav
text/                            # 可选，有人工文本时使用
  speaker_001/session_001.txt
metadata.csv
```

`metadata.csv`：

```csv
audio_path,speaker_id,language,domain,recording_type,transcript_path
audio/speaker_001/session_001.wav,spk001,zh,news,studio,text/speaker_001/session_001.txt
```

- `transcript_path` 为空时走纯 ASR 路径
- `recording_type` 取值：`studio / podcast / interview / field`，影响后续阈值选择
- 单条原始音频建议 5min - 2h，单说话人为佳

---

## 2. 音频质检

只筛掉明显不可用数据，不做主观筛选。

处理：

- 转 mono
- 编码统一为 WAV PCM 16-bit
- 计算 SNR、LUFS、峰值音量、clipping 比例
- 检测有效带宽（防止 8k/16k 上采样冒充 24k）

启动阈值（按 `recording_type` 调整）：

```yaml
audio_qc:
  studio:        { min_snr_db: 20, max_clipping_ratio: 0.001, min_effective_bw_hz: 7500 }
  podcast:       { min_snr_db: 12, max_clipping_ratio: 0.005, min_effective_bw_hz: 6000 }
  interview:     { min_snr_db: 10, max_clipping_ratio: 0.005, min_effective_bw_hz: 6000 }
  field:         { min_snr_db: 8,  max_clipping_ratio: 0.01,  min_effective_bw_hz: 6000 }
  loudness_lufs_range: [-35, -12]
  min_duration_sec: 30
```

---

## 3. 重采样：双版本策略

Qwen3-ForcedAligner 推荐 16kHz mono，TTS 训练目标采样率 24kHz/48kHz。两份音频共存：

```text
work/
  audio_train/spk001_s001.wav      # 24kHz mono，TTS 切片源
  audio_align/spk001_s001.wav      # 16kHz mono，ASR + Aligner 输入
```

时间戳与采样率无关，对齐结果直接复用到 24kHz 版本上。

---

## 4. 粗 VAD 切分（含说话人切换检测）

直接把长音频喂 ASR 不现实，且 Qwen3-ASR 单条输入限制为 30s，先粗切到 10 - 30s。
**每段必须只包含一个说话人**——多说话人段会污染 F0/语速统计，且 metadata 里的 `speaker_id` 会失效。

切分依据（任一触发都形成边界）：

1. 长静音 ≥ 500ms 且不在能量陡变处
2. **说话人切换点**（嵌入向量距离突变）
3. 达到 `max_segment_sec = 30` 兜底强切

说话人切换检测：

- 工具：`pyannote/speaker-diarization-3.1`（HuggingFace），或轻量替代 `pyannote/segmentation-3.0` + 嵌入聚类
- 在 16kHz 对齐版本上跑 diarization，输出每个时间段的 speaker label
- 切分点对齐到 diarization 边界 ± 200ms 内的最近静音
- 单说话人源（metadata 已标注 `speaker_id` 且 `recording_type = studio` 时）可在配置里关闭，省时

段处理规则：

- 每个 VAD 段必须落在**单一 diarization speaker** 内
- 一段内出现 ≥ 2 个 speaker label 时，按 label 边界拆开
- 拆完后段长 < `min_segment_sec` (10s) 的不强行合并跨说话人边界，直接丢弃或单独打标走低权重通道
- 实际说话人 label 与 metadata 里的 `speaker_id` 不一致时（多说话人源），用 diarization label 重命名为 `spk001_a / spk001_b`，metadata 里的 `speaker_id` 仅作上游分组用

工具选择：

- 静音 VAD：`silero-vad`
- 说话人切分：`pyannote-audio`

输出每段：

```json
{
  "seg_id": "spk001_s001_seg007",
  "source_audio": "speaker_001/session_001.wav",
  "speaker_label": "spk001_a",
  "start": 412.30,
  "end": 438.92,
  "duration": 26.62,
  "diar_confidence": 0.88
}
```

`diar_confidence` 低于阈值（默认 0.6）的段标 `review`，进 step 5 但在 step 11 加扣分。

---

## 5. Qwen3-ASR 转写

```yaml
asr:
  model: Qwen3-ASR-1.7B
  language: zh                     # 跟随 metadata.language
  device: cuda
  batch_size: 8
  with_punctuation: true           # 默认开
  max_input_sec: 30                # 硬上限，超过的输入由上游 VAD 保证不出现
```

输入约束：单条音频 ≤ 30s，由 step 4 粗 VAD 保证。如果上游传入超过 30s 的段，直接 fail-fast 并打日志，不做静默切分（避免破坏与 VAD 的边界一致性）。

输出：句级时间戳 + 文本（含标点）。

校验路径：

- 有人工文本：ASR 文本与人工文本对齐做 CER 比较，CER > 8% 标 review
- 无人工文本：用 Whisper-large-v3 跑同段做对照转写，两者 CER > 12% 整段丢弃（保底自洽校验）

---

## 6. 文本规范化

```text
- 繁简统一（zh-Hans）
- 数字、日期、金额、百分比展开为汉字
- 英文缩写按读法展开（FBI -> 艾弗比艾，可选）
- 标点统一为中文全角 ，。？！；：
- 去除括号注释、表情符号、连续空白
- 保留口语词（呃、嗯、啊）作为可选标签，不删除
```

ASR 文本和人工文本走同一规范化函数。规范化前后文本都保留，对齐时使用规范化后的文本。

---

## 7. Qwen3-ForcedAligner 对齐

只做字级对齐，**不输出音素级时间戳**。如下游训练框架（如 FastSpeech2）需要音素级，再单跑一次 MFA 补充，不放进主流程。

```yaml
align:
  model: Qwen3-ForcedAligner-0.6B
  audio_sample_rate: 16000
  granularity_ms: 50               # 30-50ms 适合 TTS 精细对齐
  latency_offset_ms: 0             # 见下方校准步骤
```

**延迟校准（一次性，每次模型版本变更重做）：**

1. 准备一段 1 - 2 分钟的 studio 录音 + 对应文本（已知精确时间戳）
2. 跑对齐，对比每个字预测起点与 ground-truth 起点的中位差
3. 把中位差填入 `latency_offset_ms`
4. 后续所有时间戳减去这个偏移

输出格式：

```json
{
  "utt_id": "spk001_s001_seg007",
  "chars": [
    {"char": "今", "start": 0.12, "end": 0.28, "confidence": 0.96},
    {"char": "天", "start": 0.28, "end": 0.44, "confidence": 0.94}
  ],
  "align_conf_mean": 0.92,
  "align_conf_p10": 0.78
}
```

筛选规则（不再用单一 `alignment_score`）：

```yaml
align_filter:
  min_char_confidence: 0.6
  min_high_conf_char_ratio: 0.85   # confidence > 0.6 的字占比
  max_text_audio_duration_ratio: 1.4
  min_text_audio_duration_ratio: 0.6
```

---

## 8. 精细切分

基于对齐结果做语义切分，目标 3 - 20s，优选 5 - 15s。

切分点优先级（高到低）：

```text
1. 句末标点（。！？）后 + 后续静音 ≥ 200ms + 该字 confidence ≥ 0.9
2. 长静音 ≥ 300ms 且不落在词中
3. 句中标点（，；：）后 + confidence ≥ 0.85
4. 兜底：达到 max_duration 时强制切，落在最近一个静音处
```

每个切片保留前后文用于上下文建模：

```json
{
  "utt_id": "spk001_s001_000123",
  "wav": "wavs/spk001_s001_000123.wav",
  "text": "今天的天气非常适合出门散步。",
  "prev_text": "我们刚刚下了课。",
  "next_text": "你想一起去吗？",
  "source_audio": "speaker_001/session_001.wav",
  "start": 123.42,
  "end": 130.18,
  "duration": 6.76
}
```

`min_duration: 3.0`（小于 3s 的样本对 TTS encoder 不友好），如需保留 2 - 3s 样本，单独打标走低权重通道。

---

## 9. 多说话人巡检（严格）

step 4 的 diarization 是粗粒度，可能漏检短插话、背景人声、嘉宾抢话。精切后必须再做一次**严格的单说话人校验**，宁可误杀也不放过。

策略：滑动窗口提取说话人嵌入向量，多重判据 OR 投票，任一触发即 reject。

**1. 滑窗嵌入提取**

```yaml
spk_consistency:
  embedding_model: pyannote/embedding             # 或 wespeaker, ecapa-tdnn
  window_sec: 1.5                                  # 窗口长度
  hop_sec: 0.25                                    # 步长（约 6x 重叠，足够密）
  min_windows: 4                                   # 窗口数 < 4 的短样本不做巡检，强制 reject 或走低权通道
  exclude_silence: true                            # 跳过静音帧主导的窗口（VAD 概率 < 0.5）
```

每个窗口产出一个 192/256 维 embedding，得到序列 `[e_1, e_2, ..., e_n]`。

**2. 多重判据（任一命中即 reject）**

```text
判据 A — 全局聚类
  对窗口序列做 spectral clustering 或 AHC，silhouette > 0.35 视为存在 ≥ 2 个簇
  reject

判据 B — 与全段中心的最大距离
  计算所有窗口的 cosine 中心 c
  max_dist = max(1 - cos(e_i, c))
  max_dist > 0.35  reject

判据 C — 相邻窗口跳变
  delta_i = 1 - cos(e_i, e_{i+1})
  任一 delta_i > 0.30  reject
  连续 2 个窗口 delta > 0.20  reject（缓变切换）

判据 D — 与参考嵌入对比（有 enrollment 时）
  说话人有 enrollment 嵌入 c_ref（取 step 4 同 speaker_label 的中心）
  低于阈值的窗口比例：(1 - cos(e_i, c_ref)) > 0.35 的占比 > 10%  reject

判据 E — 双说话人重叠检测
  pyannote/overlapped-speech-detection 模型，重叠时长占比 > 2%  reject
```

阈值偏紧的设计是有意的——多说话人样本对 TTS 训练的伤害远大于丢几条干净样本。

**3. enrollment 嵌入维护**

每个 `speaker_label` 维护一组参考嵌入：

```text
- step 4 diarization 后，每个 speaker_label 的所有段做嵌入，取 medoid 或 top-K（K=20）作为 enrollment
- step 9 用 enrollment 做判据 D
- step 9 通过的样本反过来更新 enrollment（增量），但需 quality_score >= 0.85
```

**4. 输出**

每条样本附加：

```json
{
  "spk_consistency": {
    "passed": true,
    "n_windows": 18,
    "max_dist_to_center": 0.21,
    "max_neighbor_delta": 0.14,
    "cluster_silhouette": 0.12,
    "overlap_ratio": 0.0,
    "ref_outlier_ratio": 0.05
  }
}
```

未通过的样本写入 `rejected.jsonl`，`reject_reasons` 中标明具体触发的判据（A/B/C/D/E）。

**5. 阶段失败兜底**

若某 speaker_label 全部样本通过率 < 50%，提示该 label 可能在 step 4 被错分，触发人工抽检告警，不自动重跑。

---

## 10. F0 提取

工具：`pyworld` (Harvest 或 DIO + StoneMask)，备选 `parselmouth`（Praat）。

**两遍策略，解决 speaker_median_f0 循环依赖：**

第一遍：用宽松全局范围（`f0_min=60, f0_max=600`）粗算每个 speaker 的 median，丢弃 voiced_ratio 极低的 outlier。

第二遍：按 speaker median 自适应设定提取范围，再算最终指标：

```text
f0_min = max(50, speaker_median * 0.5)
f0_max = min(800, speaker_median * 2.0)
```

帧级处理：

- `frame_hop = 10ms`
- octave error 修正：相邻帧跳变超过 6 semitone 时按 2x/0.5x 校正候选
- voiced 帧统计 F0，**unvoiced 帧不参与 std/range 计算**

每条样本指标：

```text
f0_mean_hz              # 平均基频（保留 Hz，便于直接喂模型）
f0_median_hz
f0_std_st               # semitone 标准差，用于跨说话人比较
f0_range_st             # P95 - P5（semitone）
f0_delta_p95_st         # 相邻帧 delta 的 P95（semitone）
voiced_ratio            # 有声帧占总帧比例
f0_confidence           # 提取器置信度均值
```

跨说话人比较统一用 semitone：`f0_st = 12 * log2(f0_hz / speaker_median_f0)`

筛选（启动阈值）：

```yaml
f0_filter:
  min_voiced_ratio: 0.45
  min_f0_confidence: 0.75
  max_f0_delta_p95_st: 6.0           # 单帧 10ms 内变化超过 6 半音 = 提取错误
  global:
    min_f0_std_st: 1.0               # 过平
    max_f0_std_st: 10.0              # 过乱
    min_f0_range_st: 3.0
    max_f0_range_st: 20.0
  speaker_adaptive:                  # 数据量足够时启用
    enabled: true
    min_samples_per_speaker: 200
    keep_percentile: [10, 95]        # 每个 speaker 内保留 P10-P95
```

speaker-adaptive 只在 speaker 样本数 ≥ 200 时启用，否则退回 global 阈值。

---

## 11. 语速指标

直接复用 Qwen3-Aligner 的字级时间戳，**不再独立做窗口分帧**。

每条样本计算：

```text
global_rate_cps         # 全句字/秒（不含静音）
local_rate_mean         # 滑窗均值
local_rate_std
local_rate_cv           # std / mean
local_rate_p5_p95_range
pause_ratio             # 字间间隔 > 200ms 的总时长占比
long_pause_count        # 间隔 > 500ms 的次数
```

局部窗口：**每 6 个字一个窗口，步长 3 字**（按字数滑窗，对中文更稳定）。

样本字数 < 12 时不计算 `local_rate_cv`，只用 global 指标，这条样本在 score 中不扣 local_rate 分。

筛选（中文普通话启动阈值）：

```yaml
rate_filter:
  global:
    min_global_rate_cps: 2.0
    max_global_rate_cps: 8.0
    min_local_rate_cv: 0.06
    max_local_rate_cv: 0.65
    max_pause_ratio: 0.45
    max_long_pause_count_per_15s: 2
  speaker_adaptive:
    enabled: true
    min_samples_per_speaker: 200
    keep_percentile: [10, 95]
```

英文/粤语等其他语种用 syllable/sec，阈值另调。

---

## 12. 规则过滤 + 综合打分

每条样本先过硬规则（任意一条不满足直接 reject），通过后算综合分：

```text
score =
  0.25 * audio_quality_score      # 来自 step 2
+ 0.20 * align_quality_score      # 来自 step 7 conf 分布
+ 0.25 * pitch_score              # 见下
+ 0.20 * rate_score               # 见下
+ 0.10 * text_quality_score       # 规范化命中率 + ASR/manual CER
```

`pitch_score` / `rate_score` 用钟形曲线打分：

```text
pitch_score:
  f0_std_st 在 [2.0, 6.0] 区间得 1.0，向两侧线性衰减
  f0_delta_p95_st 在 [0, 4] 区间得 1.0，> 6 衰减到 0

rate_score:
  global_rate_cps 在 [3.5, 6.0] 得 1.0，向两侧线性衰减
  local_rate_cv 在 [0.12, 0.45] 得 1.0，向两侧线性衰减
```

权重和打分曲线全部写在 YAML，方便按数据情况调整。不引入可学习权重。

样本分级：

```text
A: score >= 0.85    高质量
B: 0.75 - 0.85      可用
C: 0.60 - 0.75      低权重 / 人工抽检
D: < 0.60           剔除
```

每条样本（包括被 reject 的）写入 `reject_reasons`：

```json
{
  "utt_id": "spk001_s001_000999",
  "status": "rejected",
  "reject_reasons": ["voiced_ratio<0.45", "f0_delta_p95_st>6.0"],
  "stage_metrics": {
    "voiced_ratio": 0.31,
    "f0_delta_p95_st": 7.4
  }
}
```

`reports/rejected_samples.csv` 按 reason 聚合数量分布，方便调阈值。

---

## 13. 数据集划分

按以下维度分层抽样：

```text
speaker
domain
duration_bucket          # [3-6, 6-10, 10-15, 15-20]
f0_std_bucket            # [low, mid, high]
rate_cv_bucket           # [low, mid, high]
```

划分比例：

```text
train: 92%
valid: 4%
test:  4%
```

Leakage 规则（硬性）：

- 同一 `source_audio` 的切片不跨集合
- 同一 session 内连续 ≤ 30s 的相邻切片视为同一上下文，整组只能进一个集合
- speaker leakage：默认允许（多说话人共享建模）；训练 zero-shot voice clone 时切换为 hold-out speaker 模式

不做 A/B/C/D 跨集合再平衡——valid/test 直接从 A+B 取，不引入 C；C 只在 train 里且权重 0.5。

避免近重复：用 `audiomentations` 的 fingerprint 或文本 5-gram MinHash 做 dedup，相似度 > 0.9 的同 speaker 切片只保留 1 条。

---

## 14. 产出格式

```text
tts_dataset/
  wavs/
    spk001_s001_000001.wav
  manifests/
    train.jsonl
    valid.jsonl
    test.jsonl
    rejected.jsonl
  alignments/
    spk001_s001_000001.json          # 字级对齐
  features/
    spk001_s001_000001_f0.npy        # 帧级 F0
    spk001_s001_000001_prosody.json  # 韵律指标汇总
  reports/
    quality_report.html
    pitch_distribution.png
    rate_distribution.png
    rejected_samples.csv
    speaker_stats.csv
  configs/
    pipeline_config.yaml             # 实际使用的配置快照
```

`train.jsonl` 一行：

```json
{
  "utt_id": "spk001_s001_000001",
  "speaker_id": "spk001",
  "wav": "wavs/spk001_s001_000001.wav",
  "text": "今天的天气非常适合出门散步。",
  "text_normalized": "今天的天气非常适合出门散步。",
  "duration": 6.72,
  "sample_rate": 24000,
  "source_audio": "speaker_001/session_001.wav",
  "start": 123.42,
  "end": 130.14,
  "align_conf_mean": 0.93,
  "f0_mean_hz": 214.3,
  "f0_std_st": 3.8,
  "f0_range_st": 9.6,
  "voiced_ratio": 0.71,
  "global_rate_cps": 4.8,
  "local_rate_cv": 0.24,
  "pause_ratio": 0.12,
  "quality_score": 0.89,
  "grade": "A",
  "prosody_bucket": "natural"
}
```

`prosody_bucket` 取值：`flat / natural / expressive / chaotic`，由 f0_std_st 和 local_rate_cv 联合判定：

```text
flat:        f0_std_st < 2.0 或 local_rate_cv < 0.10
natural:     2.0 <= f0_std_st < 5.0 且 0.10 <= local_rate_cv < 0.35
expressive:  5.0 <= f0_std_st <= 8.0 或 0.35 <= local_rate_cv <= 0.55
chaotic:     超出 expressive 上限（接近 reject）
```

采样建议（在训练侧实现）：

```text
natural:     60%
expressive:  30%
flat + 边界: 10%
```

---

## 15. 模块清单

```text
01_ingest.py              数据导入与 metadata 校验
02_audio_qc.py            音频质检
03_resample.py            生成 16kHz 对齐版 + 24kHz 训练版
04_vad_coarse.py          粗 VAD + diarization，切到 10-30s 单说话人段
05_asr_qwen3.py           Qwen3-ASR 转写
06_text_normalize.py      文本规范化
07_align_qwen3.py         Qwen3-ForcedAligner 对齐
08_fine_segment.py        基于对齐的精细切分
09_spk_consistency.py     滑窗多说话人巡检（严格）
10_extract_f0.py          F0 提取（两遍）
11_rate_metrics.py        语速指标
12_filter_score.py        过滤 + 打分
13_split_dataset.py       数据集划分
14_report.py              报告生成
```

每个模块约定：

- 读上一步的 manifest（路径在配置里）
- 写自己的 manifest（含输入文件 hash）
- 输入 hash + 配置 hash 不变时跳过该样本（基础幂等性，不上 DAG 框架）

---

## 16. 配置示例

```yaml
project: prosogate
input:
  metadata_csv: data/metadata.csv
  audio_root: data/audio
  text_root: data/text

audio_qc:
  studio:    { min_snr_db: 20, max_clipping_ratio: 0.001 }
  podcast:   { min_snr_db: 12, max_clipping_ratio: 0.005 }
  interview: { min_snr_db: 10, max_clipping_ratio: 0.005 }
  field:     { min_snr_db: 8,  max_clipping_ratio: 0.01 }
  loudness_lufs_range: [-35, -12]
  min_effective_bw_hz: 6000

resample:
  training_sr: 24000
  alignment_sr: 16000

vad_coarse:
  min_silence_ms: 500
  min_segment_sec: 10
  max_segment_sec: 30
  diarization:
    enabled: true                  # 单说话人 studio 数据可关
    model: pyannote/speaker-diarization-3.1
    snap_to_silence_ms: 200        # 切换点对齐到附近静音
    min_diar_confidence: 0.6

asr:
  model: Qwen3-ASR-1.7B
  with_punctuation: true
  batch_size: 8
  max_input_sec: 30

align:
  model: Qwen3-ForcedAligner-0.6B
  granularity_ms: 50
  latency_offset_ms: 0
  min_char_confidence: 0.6
  min_high_conf_char_ratio: 0.85

fine_segment:
  min_duration: 3.0
  max_duration: 20.0
  preferred_min: 5.0
  preferred_max: 15.0
  punctuation_silence_ms: 200

spk_consistency:
  embedding_model: pyannote/embedding
  window_sec: 1.5
  hop_sec: 0.25
  min_windows: 4
  exclude_silence: true
  thresholds:
    cluster_silhouette: 0.35           # 判据 A
    max_dist_to_center: 0.35           # 判据 B
    max_neighbor_delta: 0.30           # 判据 C 单点
    consecutive_neighbor_delta: 0.20   # 判据 C 连续 2 窗
    ref_outlier_ratio: 0.10            # 判据 D
    overlap_ratio: 0.02                # 判据 E
  enrollment:
    topk_per_speaker: 20
    update_min_score: 0.85

f0:
  frame_hop_ms: 10
  bootstrap:
    f0_min: 60
    f0_max: 600
  speaker_adaptive_range:
    factor_low: 0.5
    factor_high: 2.0
  filter:
    min_voiced_ratio: 0.45
    min_f0_confidence: 0.75
    max_f0_delta_p95_st: 6.0
    min_f0_std_st: 1.0
    max_f0_std_st: 10.0
    min_f0_range_st: 3.0
    max_f0_range_st: 20.0
  speaker_adaptive:
    enabled: true
    min_samples_per_speaker: 200
    keep_percentile: [10, 95]

rate:
  window_chars: 6
  step_chars: 3
  min_chars_for_local: 12
  filter:
    min_global_rate_cps: 2.0
    max_global_rate_cps: 8.0
    min_local_rate_cv: 0.06
    max_local_rate_cv: 0.65
    max_pause_ratio: 0.45

score:
  weights:
    audio: 0.25
    align: 0.20
    pitch: 0.25
    rate:  0.20
    text:  0.10
  pitch_optimal_std_st_range: [2.0, 6.0]
  rate_optimal_cps_range: [3.5, 6.0]
  rate_optimal_cv_range: [0.12, 0.45]
  grade_thresholds: { A: 0.85, B: 0.75, C: 0.60 }

split:
  ratios: { train: 0.92, valid: 0.04, test: 0.04 }
  speaker_holdout: false
  dedup_similarity: 0.9
```

---

## 17. 关键原则

- 长音频先粗切保留上下文，再用对齐结果精切
- 单说话人是硬约束：粗 VAD 时 diarization 切，精切后再用滑窗严格巡检，宁可误杀
- F0 用 semitone 跨说话人比较，speaker-adaptive 是可选增强不是默认
- 语速看整体也看局部，但样本太短时只用整体
- 过平和过乱都筛掉，保留自然 + 适度表现力
- 阈值先用经验值，数据量起来后按 speaker/domain 自适应
- 每条样本（含被 reject 的）都记录 `reject_reasons` 和 `stage_metrics`，方便回溯
- 字级对齐已足够支撑 TTS 微调，音素级按需补，不进主流程
