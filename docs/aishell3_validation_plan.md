# AISHELL-3 验证方案

用 AISHELL-3 验证 ProsoGate pipeline 的可行性。**这个数据集已经切分好了，不是长音频**——所以验证策略要分两种模式。

---

## 数据集要点回顾

| 项 | 值 |
|---|---|
| 总时长 | ~85h |
| 说话人 | 218（普通话母语者，标注性别/年龄/方言） |
| 话语数 | 88,035 |
| 切片粒度 | 已切到句级（每条 wav 通常 1-10s） |
| 采样率 | 44.1kHz，16bit，mono |
| 自带标注 | 拼音、韵律、音素级时间戳 |
| 划分 | train + test |
| 下载 | 19GB tgz / 26GB 解压 |

**对 ProsoGate 的价值**：
1. **韵律 GT**：自带韵律标注，可作为 §11 综合打分的对照基准
2. **多说话人**：218 speaker 足够触发 §9/§10 的 speaker-adaptive 阈值（每 speaker ~404 条 ≥ 200 阈值）
3. **音素级对齐 GT**：能反向校验 Qwen3-Aligner 的字级精度
4. **不验证 §4 长音频切分**：因为已经是短句

---

## 模式 A：直筒验证（不拼接长音频）

把 AISHELL-3 当作"已经过 step 8 精切分的输出"，从 step 9 开始跑下游：spk_consistency → F0 → rate → score → split → report。

**目标**：验证韵律指标（F0 / rate）和打分逻辑在真实人声上的合理性。

### A.1 数据准备脚本

写一个 `scripts/aishell3_to_manifest.py`：
- 扫描 AISHELL-3 训练集
- 抽样 N 条（建议 N=2000，覆盖 ~30 个 speaker × 60 句）
- 转换为 `08_fine_segment.jsonl` 格式
- 同步生成 alignment json（用 AISHELL-3 自带的字级时间戳）
- 把 wav 复制/软链到 `tts_dataset/wavs/{utt_id}.wav`，**resample 到 24kHz**（AISHELL-3 是 44.1kHz）

manifest 字段映射：
```
AISHELL-3                    →  ProsoGate manifest
utt_id (e.g. SSB0005-0001)   →  utt_id
speaker_id (e.g. SSB0005)    →  speaker_id, speaker_label (一致)
content (汉字)                →  text
prosody_label                →  保留到 metadata.prosody_gt 供对比
phone alignment              →  转字级 alignment.json
- 缺 prev_text/next_text     →  直接置空字符串
```

### A.2 跑下游
```bash
# 改 configs/pipeline_real.yaml：从 step 9 开始
python scripts/aishell3_to_manifest.py \
    --aishell3-root /path/to/data_aishell3 \
    --speakers 30 \
    --utts-per-speaker 60 \
    --output work/08_fine_segment.jsonl

python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 9 --to 14
```

### A.3 验收指标

| 指标 | 期望（中文 studio TTS 数据） |
|---|---|
| spk_consistency 通过率 | ≥ 98%（同 speaker 内不应该有 reject）|
| 跨 speaker spk_consistency | 抽 5 个不同 speaker 拼成"假混说话人"，应该 100% reject |
| `global_rate_cps` 中位数 | 4.0 - 6.0 |
| `f0_std_st` 中位数 | 2.0 - 5.0 |
| Grade 分布 | A ≥ 50%, A+B ≥ 80% |
| reject 率 | ≤ 15%（AISHELL-3 是已清洗数据） |
| reject 主因 | 不应该有任何单一原因 > 50% |

如果 reject 率 > 30%，说明阈值偏严，需要调 `configs/pipeline_real.yaml` 的 f0/rate filter。

### A.4 与 GT 对比

AISHELL-3 自带音素级时间戳和韵律标签，可以反向校验：

```python
# scripts/aishell3_validate_metrics.py
# 比较 AISHELL-3 GT vs ProsoGate 计算结果：
1. 字级时间戳一致性：用 GT 替代 stage 07 的输出，看 §10 §11 指标是否稳定
2. F0 提取精度：在抽样样本上做 ground-truth F0（人工标 / Praat），看 pyworld vs autocorr 哪个更准
3. 语速分布：cps 与 GT prosody_label 是否相关（GT 有 "fast/normal/slow" 标签时）
```

**模式 A 的限制**：跳过了 step 1-8，所以不验证音频质检、VAD、ASR、Qwen3-Aligner、精切分这五块。需要模式 B 补充。

---

## 模式 B：拼接验证（造长音频）

把同一个 speaker 的 5-10 条 AISHELL-3 短句**拼接成一段长音频**，跑完整 14 stage。

**目标**：验证长音频切分链路（VAD → ASR → Aligner → fine_segment）能否还原原始切分。

### B.1 拼接策略

`scripts/aishell3_synthesize_long.py`：
- 每个 speaker 选 8 条短句（按 utt_id 顺序，避免拼接突兀）
- 在每条之间插入 0.6 - 1.2s 静音（仿照真实段间停顿）
- 拼成一条 30-90s 的长音频
- 同时记录 GT 边界（每条原始 utt 的 start/end）→ `aishell3_synth_truth.json`
- 文本拼接时不加标点连接，保留原标点

输出：
```
data_test/audio/spk_SSB0005/long_001.wav     # 拼接后的长音频
data_test/text/spk_SSB0005/long_001.txt      # 拼接文本
data_test/audio_truth/spk_SSB0005/long_001.json  # GT 边界 + 原始 utt_id
```

### B.2 多说话人版本

为了验证 §4 diarization 切换检测，做 1-2 条**多说话人拼接**：
- 选 2 个 speaker 各 4 条短句
- 交替拼接：spkA → spkB → spkA → spkB ...
- 同样记录 GT 边界 + 每段的真实 speaker

### B.3 跑全链路
```bash
python scripts/run_pipeline.py --config configs/pipeline_real.yaml
```

### B.4 验收指标

| 维度 | 验证什么 | 期望 |
|---|---|---|
| stage 4 VAD 段长 | 每段 ∈ [10, 30]s | 100% |
| stage 4 切分边界 | 与 GT 段间静音中心的偏差 | < 200ms（中位数） |
| stage 4 单说话人源 | 不应该出现 _a/_b 拆分 | 0 拆分 |
| stage 4 多说话人源 | 应该出现 _a/_b | speaker_label 数 = GT speaker 数 |
| stage 5 ASR CER | vs GT 文本 | < 5%（AISHELL-3 是干净数据） |
| stage 7 Aligner 字级时间戳 | vs GT 音素聚合到字 | 中位偏差 < 50ms |
| stage 8 精切片 | 与原始 utt 边界对应 | recall ≥ 0.8（GT 边界附近 ±300ms 有切点）|
| stage 9 spk_consistency | 单说话人段全过 | 通过率 ≥ 98% |
| stage 9 多说话人巡检 | 拼接边界处的混段被 reject | recall ≥ 0.8 |

### B.5 关键消融

把 use_mock 切到 false 一个一个验证，避免一次开太多变量：
1. **只开 ASR 真实**：其他全 mock，验证 Qwen3-ASR 在干净中文上的 CER
2. **只开 Aligner 真实**：用 GT 文本，验证字级时间戳精度（与 GT 音素对比）
3. **只开 spk_consistency 真实**：用 GT 切分，验证 pyannote 在拼接边界附近的检测能力
4. **全开**：端到端

---

## 推荐执行顺序

| 步骤 | 内容 | 时长 | 资源 | 主要验证什么 |
|---|---|---|---|---|
| 1 | 下载 AISHELL-3 | 1-2h | 网络 | 数据可用 |
| 2 | 模式 A：抽 200 条单 speaker 跑 | 30min | CPU | F0/rate 指标合理性 |
| 3 | 模式 A：扩到 30 speaker × 60 utt | 2h | CPU 或 1×GPU | speaker-adaptive 阈值 + 整体 reject 分布 |
| 4 | 模式 B：单说话人拼接 5 条 | 1h | 1×GPU | VAD/ASR/Aligner 链路通 |
| 5 | 模式 B：多说话人拼接 1 条 | 1h | 1×GPU | diarization + spk_consistency |
| 6 | 模式 B：消融 + 完整跑 50 条拼接长音频 | 半天 | 1×GPU | 端到端可用性 |

**总预算约 1-1.5 个工作日**，最关键的是步骤 2-3（不需要 GPU 的可以先跑），快速暴露韵律指标问题。

---

## 数据下载

OpenSLR 中国镜像：
```bash
mkdir -p /workspace/data
cd /workspace/data
wget https://us.openslr.org/resources/93/data_aishell3.tgz
# 19GB，下完解压
tar -xzf data_aishell3.tgz
```

或 ModelScope（国内更快）：
```bash
pip install modelscope
python - << 'EOF'
from modelscope import snapshot_download
snapshot_download('speech_tts/AISHELL-3', cache_dir='/workspace/data')
EOF
```

---

## 不在本方案验证的部分

诚实标注，避免误以为"AISHELL-3 跑过 = pipeline 完整可用"：

| 部分 | 为什么不验证 | 后续怎么补 |
|---|---|---|
| 真实音频质检阈值（SNR/clipping） | AISHELL-3 是干净 studio 录音，不会触发 | 找 podcast / interview 数据另测 |
| 真实长音频粗 VAD | AISHELL-3 拼接的"长音频"静音是人造的，太干净 | 用真实播客数据补测 |
| 文本规范化（数字/英文） | AISHELL-3 文本已规范 | 用新闻类数据另测 |
| F0 提取的边缘情况（笑声/喘气/低语） | AISHELL-3 是干净朗读 | 用情感语音数据补测 |
| 实时吞吐 / 显存峰值 | 需要 GPU 驱动修复（当前 CUDA 12.2 driver vs torch cu128） | 修驱动后 1×A100 跑 100 条压测 |

---

## 当前阻塞

1. **GPU 不可用**：torch 2.9.1+cu128 与 NVIDIA 驱动 535.154.05 (CUDA 12.2) 不兼容。pyannote/Qwen3 在 CPU 上能跑 smoke test，但 ≥ 50 条规模的真实测试需要修复。
   - 解决方案 A：降级 torch 到 cu121 (`pip install torch==2.5 torchaudio==2.5 --index-url https://download.pytorch.org/whl/cu121`)
   - 解决方案 B：升级机器驱动到 ≥ 545（CUDA 12.4+）
2. **Qwen3-ASR / Qwen3-ForcedAligner 真实接口未对接**：stages/05 和 stages/07 当前是 transformers 占位实现，需要确认真实仓库 ID 和调用 API。
