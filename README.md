# ProsoGate Pipeline

面向 TTS 微调数据产出的处理流水线。**核心解决三件事**：长音频切分、基频变化筛选、语速变化筛选。

输入长音频，输出"切分合理 + 字级对齐 + 韵律指标齐全 + 质量打分"的训练样本集（含 manifest、wav、alignment、F0 特征、QC 报告）。

---

## 主要特性

- **14 个 stage 串行可重跑**：每个 stage 读上一步 manifest、写自己的 manifest，独立、幂等
- **真实 + Mock 双模式**：ASR / Aligner / 说话人嵌入均支持 mock，方便在没有外部模型时跑通管线
- **HTTP batch 调用**：Qwen3-ASR / Qwen3-ForcedAligner 通过 batch endpoint 一次性 16 段推理，端到端比顺序调用快 **4-6×**
- **韵律指标精简到 6 个独立维度**：F0 / 语速 / 停顿，去除冗余的中间字段
- **leakage 防护**：同一 source audio 的切片不跨 train/valid/test
- **打分 + 分级**：每条样本带 `quality_score` ∈ [0,1]、`grade` ∈ {A,B,C}、`prosody_bucket` ∈ {flat,natural,expressive,chaotic}

---

## Pipeline 14 个 stage

| # | Stage | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| 01 | `ingest` | 读 `metadata.csv`，验证必填列 + 算 audio_hash | csv | `01_ingest.jsonl` |
| 02 | `audio_qc` | 计算 SNR / LUFS / clipping / 有效带宽，按 recording_type 拒不合格 | 01 | `02_audio_qc.jsonl` |
| 03 | `resample` | 生成 24 kHz 训练源 + 16 kHz 对齐源 | 02 | `audio_train/` + `audio_align/` |
| 04 | `vad_coarse` | silero-VAD + pyannote diarization 切到 ≤30s 单说话人段 | 03 | `04_vad_coarse.jsonl` |
| 05 | `asr_qwen3` | Qwen3-ASR HTTP batch 推理，输出文本 + 字级时间戳 | 04 | `05_asr.jsonl` |
| 06 | `text_normalize` | 繁简转换 + 数字汉化 + 标点全角 + 空白清理 | 05 | `06_text_normalize.jsonl` |
| 07 | `align_qwen3` | Qwen3-ForcedAligner HTTP batch 推理，字级 timing | 06 | `07_align.jsonl` + `alignments/` |
| 08 | `fine_segment` | 按句末标点 + 静音切到 3-20s 训练片段，写 24 kHz wav | 07 | `08_fine_segment.jsonl` + `tts_dataset/wavs/` |
| 09 | `spk_consistency` | pyannote/embedding 滑窗多说话人巡检 | 08 | `09_spk_consistency.jsonl` |
| 10 | `extract_f0` | pyworld 两遍 F0（先 bootstrap、后 speaker-adaptive），写 npy | 09 | `10_f0.jsonl` + `features/` |
| 11 | `rate_metrics` | 字级 timing 派生语速 / cv / 停顿率 | 10 | `11_rate.jsonl` |
| 12 | `filter_score` | hard-rule 过滤 + 5 维加权 quality_score + 打 grade | 11 | `12_filter_score.jsonl` |
| 13 | `split_dataset` | 按 source_audio 分组，分 train/valid/test，去重 | 12 | `tts_dataset/manifests/*.jsonl` |
| 14 | `report` | HTML 质量报告 + speaker stats + 分布图 | 13 | `tts_dataset/reports/` |

---

## 安装

### 创建 conda 环境

```bash
conda create -n prosogate python=3.11 -y
conda activate prosogate
pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install omegaconf  # pyannote 4.x 加载 lightning checkpoint 需要（requirements 漏列）
```

> **CUDA 兼容性**：默认 `requirements.txt` 装的 `torch` 可能是 cu130，在驱动 535（CUDA 12.4）上会报错。**显式装 cu124 版**。

### 准备本地模型（避开 HF 联网）

```
models/pyannote/
├── pyannote--speaker-diarization-community-1/config.yaml
├── pyannote--segmentation-3.0/pytorch_model.bin
├── pyannote--embedding/pytorch_model.bin
└── pyannote--wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
```

### 启动 Qwen3-ASR HTTP 服务

ASR/Aligner 走本地 HTTP，pipeline 端不加载 1.7B/0.6B 模型权重。服务暴露：

```
POST /v1/audio/transcriptions          单条
POST /v1/audio/transcriptions/batch    多条（pipeline 用这个）
POST /v1/audio/forced_alignment        单条
POST /v1/audio/forced_alignment/batch  多条（pipeline 用这个）
```

默认指向 `http://127.0.0.1:18765`。

---

## 用法

### 跑完整 pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline_ramc10.yaml
```

### 跑指定 stage 区间

```bash
python scripts/run_pipeline.py --config configs/pipeline_ramc10.yaml --from 5 --to 7
```

### 准备 RAMC 测试集

```bash
# 1. 下载 MagicData-RAMC 测试集（约 1.7 GB tar.gz）
mkdir -p /workspace/data/magicdata_ramc/extracted
curl -kL https://huggingface.co/datasets/EaseZh/magicdata_ramc/resolve/main/test.tar.gz \
  | tar -xzf - -C /workspace/data/magicdata_ramc/extracted

# 2. data_test/ramc10/audio/ 里建软链指向 RAMC test/wav/ 下抽样的 10 个文件
#    metadata.csv 已经在仓库里，speaker_id=multi, recording_type=interview

# 3. 跑
python scripts/run_pipeline.py --config configs/pipeline_ramc10.yaml
```

### 导出 Grade A/B/C 子集

```bash
python scripts/export_grade.py \
    --dataset-root tts_dataset_ramc10 \
    --grade A \
    --out-dir tts_dataset_ramc10/grade_A
```

输出 `grade_A/wavs/` (硬链或复制) + `manifests/{train,valid,test}.jsonl` + `grade_summary.json`。

---

## 配置（`configs/pipeline_ramc10.yaml`）

按 conversation 域调过的关键阈值：

```yaml
audio_qc:
  interview:
    min_snr_db: 8
    max_clipping_ratio: 0.005
    min_effective_bw_hz: 1000        # RAMC 16kHz 真实带宽 1-2 kHz

vad_coarse:
  min_silence_ms: 500
  min_segment_sec: 3                 # 对话语料调小（设计默认 10）
  max_segment_sec: 30
  diarization:
    enabled: true
    model: models/pyannote/pyannote--speaker-diarization-community-1/config.yaml

asr:
  batch_size: 16
  use_mock: false

align:
  batch_size: 16
  use_mock: false

spk_consistency:
  thresholds:
    cluster_silhouette: 0.55         # 对话语料同人方差大，放宽（studio 0.35）
    max_dist_to_center: 0.75
    max_neighbor_delta: 0.75

f0:
  filter:
    min_voiced_ratio: 0.35
    min_f0_confidence: 0.65          # 新公式 73% pass
    max_f0_delta_p95_st: 6.0

rate:
  filter:
    min_global_rate_cps: 2.0
    max_global_rate_cps: 9.0         # 对话语料偏快

score:
  weights: { audio: 0.25, align: 0.20, pitch: 0.25, rate: 0.20, text: 0.10 }
  rate_optimal_cps_range: [4.0, 7.5] # conversation 节奏
  grade_thresholds: { A: 0.85, B: 0.75, C: 0.60 }

split:
  ratios: { train: 0.90, valid: 0.05, test: 0.05 }
  speaker_holdout: false
```

---

## 韵律指标定义

最终 manifest 只保留 **6 个独立韵律维度 + 3 个裁决字段**，详见 [`docs/metrics.md`](docs/metrics.md)。

| 指标 | 单位 | 含义 |
|---|---|---|
| `f0_median_hz` | Hz | 基频水平（speaker pitch level）|
| `f0_std_st` | semitone | 句内基频跨度 |
| `f0_delta_p95_st` | semitone | 相邻 voiced 帧跳变 P95（octave-error 哨兵）|
| `global_rate_cps` | char/s | 全句语速（字数 / 说话时长，不含停顿）|
| `local_rate_cv` | — | 句内语速变化系数（std/mean，6 字滑窗）|
| `pause_ratio` | 0..1 | 字间隔 > 200ms 总时长占比 |
| `quality_score` | 0..1 | 5 维加权综合分 |
| `grade` | A/B/C | quality_score 分级 |
| `prosody_bucket` | flat/natural/expressive/chaotic | F0_std + local_cv 联合判定 |

---

## 最终 manifest schema (`tts_dataset/manifests/*.jsonl`)

```json
{
  "utt_id": "spkA_session1_000123",
  "speaker_id": "spkA",
  "speaker_label": "spkA",
  "source_audio_id": "spkA_session1",
  "wav": "tts_dataset/wavs/spkA_session1_000123.wav",
  "sample_rate": 24000,
  "duration": 6.72,
  "start": 123.42,
  "end": 130.14,
  "alignment_json_path": "tts_dataset/alignments/spkA_session1_000123.json",
  "f0_npy_path": "tts_dataset/features/spkA_session1_000123_f0.npy",
  "text": "今天的天气非常适合出门散步。",
  "prev_text": "我们刚刚下了课。",
  "next_text": "你想一起去吗？",
  "f0_median_hz": 214.3,
  "f0_std_st": 3.8,
  "f0_delta_p95_st": 2.6,
  "global_rate_cps": 4.8,
  "local_rate_cv": 0.24,
  "pause_ratio": 0.12,
  "quality_score": 0.89,
  "grade": "A",
  "prosody_bucket": "natural",
  "status": "ok",
  "reject_reasons": [],
  "language": "zh",
  "domain": "conversation",
  "recording_type": "interview"
}
```

中间字段（如 `audio_quality_score / align_coverage / local_rate_std / f0_mean_hz` 等）保留在 `work_*/12_filter_score.jsonl`，调阈值时随时回查。

---

## 已知行为 / 调阈值经验

| 现象 | 解释 / 行动 |
|---|---|
| `audio_quality_score` 受 SNR 压制最高 ~0.5 | SNR `_snr_db` 用 P10 估噪声，对停顿多的对话偏低；想拿 A grade 需 SNR > 18 dB 或调权重 |
| `f0_confidence < 0.65` 拒绝多 | stage 10 新公式基于能量代理，比 autocorr 严格；conversation 数据可考虑放宽到 0.50 |
| `voiced_ratio < 0.35` 拒绝多 | speaker-adaptive 范围会把 pass1 voiced 帧 hard-clamp 成 unvoiced，自然分布对话语料常见 |
| `spk_A` (silhouette > 阈值) 拒绝 | 自然对话同人方差大，studio 阈值 0.35 误杀严重，conversation 域调到 0.55+ |
| `test.jsonl` 为空 | 数据量小 + small-corpus guard 预留逻辑会覆盖；继续加数据或加大 test ratio |

---

## 目录结构

```
ProsoGatePipeline/
├── configs/
│   ├── pipeline.yaml                # 通用配置模板
│   └── pipeline_ramc10.yaml         # RAMC 测试集配置（10 个 source）
├── prosogate/                       # 共用工具（manifest IO / hash / logging / config）
├── stages/                          # 14 个 stage 实现，编号即顺序
├── scripts/
│   ├── run_pipeline.py              # 主入口
│   ├── export_grade.py              # 单 grade 子集导出
│   ├── ramc_to_manifest.py          # RAMC 测试集 → 配置 csv
│   ├── ramc_precut.py               # 长音频静音预切（可选）
│   └── eval_diarization.py          # diarization DER 评估
├── docs/
│   ├── metrics.md                   # 指标定义文档（精确公式）
│   ├── real_test_plan.md            # 真实数据 4 阶段测试方案
│   └── aishell3_validation_plan.md  # AISHELL-3 验证方案
├── design.md                        # 完整设计文档（17 章）
├── README.md                        # 本文件
└── requirements.txt
```

---

## 性能参考（RAMC 测试集 10 source / 6 通过 QC / 763 utt）

GPU = NVIDIA L20，CUDA 12.4，torch 2.5.1+cu124：

| Stage | 耗时 | 备注 |
|---|---|---|
| 01 ingest | 0.4 s | |
| 02 audio_qc | 12 s | LUFS 计算 + 全文件解码 |
| 03 resample | 26 s | librosa 16k→24k 上采样 + 写盘 |
| 04 vad_coarse | 5 min | pyannote diarization 6 段长音频 |
| 05 asr_qwen3 | 40 s | 832 段 / batch=16 / **比顺序快 4.6×** |
| 06 text_normalize | < 1 s | |
| 07 align_qwen3 | 20 s | 832 段 / batch=16 / **比顺序快 6×** |
| 08 fine_segment | 138 s | 写 790 个 wav + alignment json |
| 09 spk_consistency | 100 s | pyannote/embedding 滑窗 |
| 10 extract_f0 | 15 min | pyworld 单线程顺序，主要瓶颈 |
| 11 rate_metrics | 3 s | |
| 12 filter_score | < 1 s | |
| 13 split_dataset | < 1 s | |
| 14 report | 1 s | |
| **总耗时** | **~22 min** | （stage 10 占 70%）|

---

## 文档

- [`design.md`](design.md) — 完整 17 章设计文档
- [`docs/metrics.md`](docs/metrics.md) — 6 个核心韵律指标的精确公式
- [`docs/real_test_plan.md`](docs/real_test_plan.md) — 真实数据 4 阶段验证方案
- [`docs/aishell3_validation_plan.md`](docs/aishell3_validation_plan.md) — AISHELL-3 验证方案
