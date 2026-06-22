# ProsoGate 指标定义

最终 train/valid/test manifest 只保留 **6 个数值韵律指标 + 3 个裁决字段**。每个指标方差独立、不相关、有明确语义。中间计算量（`f0_mean_hz`、`local_rate_std`、`align_coverage` 等）只在 `work_<run>/12_filter_score.jsonl` 里留作 debug，不进 final manifest。

---

## 1. 最终 manifest schema

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `utt_id` | str | — | 唯一 id，格式 `{speaker_label}_{source_audio_id}_{idx:06d}` |
| `speaker_id` | str | — | 来源 metadata 的 speaker（多说话人源为 `"multi"`） |
| `speaker_label` | str | — | diarization 派生的细分 label（如 `multi_a`，单说话人源 = speaker_id） |
| `source_audio_id` | str | — | 来源音频 id（split leakage key） |
| `wav` | str | — | 24kHz mono PCM16 训练片段路径 |
| `sample_rate` | int | Hz | 训练 wav 采样率（恒等 24000） |
| `duration` | float | sec | utt 长度（end - start） |
| `start` / `end` | float | sec | 在 source_audio 中的位置 |
| `alignment_json_path` | str | — | 字级时间戳 JSON 路径 |
| `f0_npy_path` | str | — | 帧级 F0（10ms hop）`.npy` 路径 |
| `text` / `prev_text` / `next_text` | str | — | 当前 / 前后 20 字上下文 |
| `f0_median_hz` | float | Hz | 见 §2.1 |
| `f0_std_st` | float | semitone | 见 §2.2 |
| `f0_delta_p95_st` | float | semitone | 见 §2.3 |
| `global_rate_cps` | float | char/sec | 见 §2.4 |
| `local_rate_cv` | float | 无量纲 | 见 §2.5 |
| `pause_ratio` | float | 0..1 | 见 §2.6 |
| `quality_score` | float | 0..1 | 见 §3 |
| `grade` | str | A/B/C/D | 见 §3 |
| `prosody_bucket` | str | flat/natural/expressive/chaotic | 见 §4 |
| `status` | str | ok/rejected | train.jsonl 恒 `ok` |
| `reject_reasons` | list[str] | — | 仅 rejected.jsonl 有意义 |
| `language` / `domain` / `recording_type` | str | — | 来源 metadata，便于分层采样 |

---

## 2. 韵律核心指标（6 个）

所有 F0 处理使用 10ms 帧间距。先用 `pyworld.harvest` 提 F0，失败时退回自相关；然后做两段 octave 修正（详见 `stages/10_extract_f0.py`）。

记号：
- `f0[i]`：第 i 帧的 F0 频率，单位 Hz；`f0[i] = 0` 表示 unvoiced。
- `voiced_hz = f0[f0 > 0]`：所有 voiced 帧的 Hz 数组。
- `voiced_st = 12 · log₂(voiced_hz / speaker_median_hz)`：转 semitone（说话人内归一化）。
- `speaker_median_hz`：先用全局 `[60, 600]Hz` 跑一遍 F0，对每个 speaker 取所有 voiced 帧的中位数；再以此做 speaker-adaptive 提取范围。

### 2.1 `f0_median_hz` — 基频水平
```
f0_median_hz = median(voiced_hz)
```
- **为什么用 median 不用 mean**：mean 受 octave error 拉偏严重；median 对 ±1 个倍频错误鲁棒。
- **典型范围**：女声 180–250 Hz，男声 90–160 Hz。
- **用途**：speaker embedding 不可用时的粗略 pitch level 标签。

### 2.2 `f0_std_st` — 句内基频跨度
```
f0_std_st = std(voiced_st)
```
- 在 semitone 域算 std → 跨说话人可比。
- **典型范围**：平淡朗读 1.5–3，自然对话 3–5，情感强 5–8，> 8 通常是 F0 提取错误或笑/喘。
- **判级用途**：`pitch_score` 在 `[2.0, 6.0]` 区间得 1.0（见 §3.3）。

### 2.3 `f0_delta_p95_st` — 帧间跳变 P95
```
对每对相邻 voiced 帧 (i, i+1)：
    delta_i = | voiced_st[i+1] - voiced_st[i] |
f0_delta_p95_st = P95(delta_i)
```
- 与 `f0_std_st` 几乎不相关（实测相关系数 0.08），是独立维度。
- 捕获 **octave-error 残留 / pyworld 提取错误**：人在 10ms 内不可能跳 > 4 半音。
- **典型范围**：干净录音 < 3，疑似 octave error 在 6 附近，hard-reject 阈值 6.0。

### 2.4 `global_rate_cps` — 全句语速
```
speech_dur = Σ (char[i].end - char[i].start)
global_rate_cps = n_chars / speech_dur
```
- 分母是 **字段累加时长**（不含字间停顿），所以高估了实际语速但跨样本可比。
- **典型范围**（中文）：朗读 4–6，对话 5–8，急促 8+。
- **判级用途**：`rate_score` 在 `[4.0, 7.5]` 区间得 1.0（conversation 域设置）。

### 2.5 `local_rate_cv` — 句内语速变化系数
```
滑动窗口：每 6 个字一个窗口，步长 3 字
对每个窗口：rate_w = 6 / (char[i+5].end - char[i].start)
local_rate_cv = std(rate_w) / mean(rate_w)
```
- 当 `n_chars < 12` 时该字段为 0（同时 `quality_score` 评分跳过 cv 项）。
- 捕获 **句内节奏变化**：cv ≈ 0.1 是匀速，0.2–0.4 自然，> 0.5 起伏剧烈或对齐错位。
- 与 `local_rate_std/p5_p95_range` 高度相关（0.89/0.84），故只保留 cv。

### 2.6 `pause_ratio` — 静音占比
```
对相邻字 i, i+1：gap_i = char[i+1].start - char[i].end
pause_ratio = Σ gap_i (where gap_i > 0.2s) / utt_duration
```
- **典型范围**：朗读 0.05–0.15，对话 0.10–0.30，断断续续 > 0.40。
- 比 `long_pause_count` 信息密度更高（前者连续值，后者整数离散），相关 0.57。

---

## 3. 综合质量分 `quality_score`

```
quality_score =
  0.25 · audio_quality_score
+ 0.20 · align_quality_score
+ 0.25 · pitch_score
+ 0.20 · rate_score
+ 0.10 · text_quality_score
```

权重在 `configs/pipeline_ramc10.yaml` 的 `score.weights`。每个分量都是 0..1 的子分，**都不写入 final manifest**（只在 `work/12_filter_score.jsonl` 留作 debug）。

### 3.1 `audio_quality_score`
```
snr_score   = clip01((snr_db - 5) / 25)           # 5 dB -> 0, 30 dB -> 1
clip_score  = clip01(1 - clipping_ratio / 0.01)   # 0% clip -> 1, ≥1% -> 0
lufs_score  = max(0, 1 - |lufs - (-23)| / 12)     # bell around -23 LUFS
audio_quality_score = 0.5·snr_score + 0.2·clip_score + 0.3·lufs_score
```
来源信号在 stage 2 计算（per source_audio），下传到每个 seg/utt。

### 3.2 `align_quality_score`（timing-derived，不依赖服务 confidence）

服务返回的 confidence 是硬编码常数 0.95，**不可用**。从字级时间戳分布派生：
```
align_coverage              = Σ char_dur / utt_dur                       # 语音占比
align_degenerate_char_ratio = #{c : c.dur < 20ms or c.dur > 500ms} / n_chars
align_char_match_ratio      = n_aligned_chars / n_text_chars

cov_score   = bell(align_coverage, plateau=[0.55, 0.9], decay 0.25 on both sides)
degen_score = max(0, 1 - degen_ratio / 0.25)
match_score = max(0, 1 - |1 - match_ratio|)

align_quality_score = 0.5·cov_score + 0.3·degen_score + 0.2·match_score
```

### 3.3 `pitch_score`
```
std_score   = bell(f0_std_st, plateau=[2.0, 6.0], decay 2.0)
delta_score = ramp_then_decay(f0_delta_p95_st, full=[0.0, 4.0], zero_at 6.0)
pitch_score = 0.5·(std_score + delta_score)
```

### 3.4 `rate_score`
```
cps_score = bell(global_rate_cps, plateau=[4.0, 7.5], decay 2.0)
cv_score  = bell(local_rate_cv, plateau=[0.12, 0.45], decay 0.20)
rate_score = 0.5·(cps_score + cv_score)
# 当 n_chars < 12 时 local_rate_skipped=True，rate_score 仅取 cps_score
```

### 3.5 `text_quality_score`
```
text_quality_score =
    1 - cer_vs_manual    若有人工文本
    0.85                 否则（当前 RAMC 测试集就是这种情况，故全部 0.85）
```

### 3.6 `grade`
```
A: quality_score ≥ 0.85
B: quality_score ≥ 0.75
C: quality_score ≥ 0.60
D: < 0.60      → 自动转 rejected
```

---

## 4. `prosody_bucket` —— 训练侧采样标签

由 `f0_std_st` 和 `local_rate_cv` 联合判定（`stages/12_filter_score.py:_prosody_bucket`）：

| bucket | 条件 |
|---|---|
| `flat` | `f0_std_st < 2.0` 或 `local_rate_cv < 0.10` |
| `natural` | `2.0 ≤ f0_std_st < 5.0` 且 `0.10 ≤ local_rate_cv < 0.35` |
| `expressive` | `5.0 ≤ f0_std_st ≤ 8.0` 或 `0.35 ≤ local_rate_cv ≤ 0.55` |
| `chaotic` | 超出 expressive 上限（接近 reject） |

推荐训练时按 `natural : expressive : flat = 60 : 30 : 10` 采样。

---

## 5. Hard-rule 拒绝条件

任一命中即 reject（`stages/12_filter_score.py:_hard_rule_check`）；阈值在 `configs/pipeline_ramc10.yaml` 的 `f0.filter` / `rate.filter` / `align.*` 段。

| 维度 | 阈值（conversation 域） | 含义 |
|---|---|---|
| `voiced_ratio` | `≥ 0.35` | F0 提取出 voiced 帧比例，过低 = 静音过多 / 提取失败 |
| `f0_confidence` | `≥ 0.65` | F0 提取器置信度均值 |
| `f0_delta_p95_st` | `≤ 6.0` | 帧间跳变 P95，超过 = octave error |
| `f0_std_st` | `[1.0, 10.0]` 或 speaker-adaptive P10–P95 | 句内 pitch 变化 |
| `f0_range_st` | `[3.0, 20.0]` 或 speaker-adaptive | F0 voiced_st P5–P95 跨度 |
| `global_rate_cps` | `[2.0, 9.0]` 或 speaker-adaptive | 全句语速 |
| `local_rate_cv` | `[0.06, 0.65]` | 句内语速变化 |
| `pause_ratio` | `≤ 0.45` | 静音占比 |
| `align_coverage` | `[0.35, 1.05]` | 对齐覆盖率 |
| `align_degenerate_char_ratio` | `≤ 0.30` | 退化字时长比例 |
| `align_char_match_ratio` | `≥ 0.85` | 实际对齐 / 期望字数 |
| `duration` | `[3.0, 20.0]` | 切片长度 |

> `voiced_ratio` 和 `f0_confidence` 用于 hard-rule，但**不进 final manifest**——它们是工艺指标而非样本属性，通过 = 该样本已合格，下游训练用不到。

---

## 6. Speaker-adaptive 阈值（可选）

当某 speaker 的样本数 ≥ `min_samples_per_speaker`（默认 200）时，启用 P10–P95 自适应裁剪取代 global 阈值：

```python
对每个 speaker：
    for k in ('f0_std_st', 'f0_range_st', 'global_rate_cps', 'local_rate_cv'):
        keep speaker 内 P10..P95 区间的样本，区间外标 hard reject
```

由 `f0.speaker_adaptive.enabled` / `rate.speaker_adaptive` 控制。RAMC 当前数据量不够触发 rate 的 adaptive（< 200 / speaker_label）。

---

## 7. 已删除的中间字段（清理理由）

| 删除字段 | 相关字段 | 相关系数 | 理由 |
|---|---|---|---|
| `f0_mean_hz` | `f0_median_hz` | 0.978 | median 抗 octave-error，保留更鲁棒的版本 |
| `f0_range_st` | `f0_std_st` | 0.955 | 同维度二次度量 |
| `local_rate_mean` | `global_rate_cps` | 0.725 | 全局语速已覆盖 |
| `local_rate_std` | `local_rate_cv` | 0.897 | cv = std/mean，归一化版本更可比 |
| `local_rate_p5_p95_range` | `local_rate_cv` | 0.842 | 同上 |
| `long_pause_count` | `pause_ratio` | 0.574 | 整数离散且半冗余 |
| `voiced_ratio`、`f0_confidence` | — | — | 工艺指标，已用于 hard-rule，下游训练不需要 |
| `text_quality_score` | — | — | 无人工文本时恒 0.85，无方差 |
| `align_conf_mean`、`high_conf_char_ratio` | `align_quality_score` | — | 服务硬编码 0.95，无信号 |
| `align_coverage`、`align_degenerate_char_ratio`、`align_char_match_ratio` | `align_quality_score` | — | 已合成入 score，单独不需要 |
| `pitch_score`、`rate_score`、`audio_quality_score`、`align_quality_score`、`text_quality_score` | `quality_score` | — | 都是 `quality_score` 的分量，debug 时查 work/12_filter_score.jsonl |
| `n_chars` | — | — | 训练时 `len(text)` 现算即可 |

中间字段全部保留在 **`work_<run>/12_filter_score.jsonl`**，调阈值时可以随时回查。
