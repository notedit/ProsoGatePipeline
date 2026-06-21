# ProsoGate 真实测试方案

四阶段渐进，每阶段独立可验证。前一阶段不通过不要进下一步。

---

## 阶段 0：环境准备（30 min）

### 0.1 加载 token
```bash
source ~/.bashrc
echo $HUGGINGFACE_TOKEN  # 应输出 hf_xxx
```

### 0.2 安装真实依赖
```bash
cd /workspace/user_code/ProsoGatePipeline
pip install -r requirements.txt
```

注意几个易踩坑：
- `pyannote.audio>=3.1` 需要 `torch` 已安装；GPU 机器装 CUDA 版 torch 而不是 CPU 版
- `pyworld` 装不上时是缺 cython/build-essential，`pip install cython && pip install pyworld`
- `silero-vad` 实际是 `pip install silero-vad` 但有些环境只能从 hub 加载

### 0.3 接受 pyannote 模型 license
访问以下页面**点 Accept**（HF 网页登录后操作，没接受会 401）：
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/embedding

### 0.4 GPU 自检
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

CPU 也能跑，但 Qwen3-ASR 1.7B 跑 30s 音频在 CPU 上单条 ~30s+，不实用。

### 0.5 模型预下载（避免 pipeline 跑到一半才下载）
```bash
python - << 'EOF'
import os
from huggingface_hub import snapshot_download
token = os.environ['HUGGINGFACE_TOKEN']
for repo in [
    'Qwen/Qwen3-ASR-Toolkit',           # 占位，按真实仓库名替换
    'Qwen/Qwen3-ASR-Flash-7B',          # 或 1.7B 的真实仓库 ID
    'pyannote/speaker-diarization-3.1',
    'pyannote/embedding',
]:
    try:
        snapshot_download(repo, token=token)
        print(f'OK: {repo}')
    except Exception as e:
        print(f'FAIL {repo}: {e}')
EOF
```

> **重要**：Qwen3-ASR 和 Qwen3-ForcedAligner 的具体 HF 仓库名以 ModelScope/Qwen 官方为准，不要照抄上面的占位。stages/05_asr_qwen3.py 和 stages/07_align_qwen3.py 里的 `_try_load_real_backend` 函数需要根据真实接口调整。

---

## 阶段 1：单元 smoke——每个真实模块独立可调（1 小时）

不跑整条 pipeline，先把四个会触发外部模型的 stage 单独验证。每步用一段已知音频。

### 1.1 准备一段真实测试音频
找一段 **已知文本 + 干净的中文录音**，30s - 1min，单说话人，存到：
```
data_test/audio/spkA/clip01.wav
data_test/text/spkA/clip01.txt
```

写一份 `data_test/metadata.csv`：
```csv
audio_path,speaker_id,language,domain,recording_type,transcript_path
data_test/audio/spkA/clip01.wav,spkA,zh,test,studio,data_test/text/spkA/clip01.txt
```

### 1.2 改配置切真实路径
建一个测试用配置 `configs/pipeline_real.yaml`，从 `pipeline.yaml` 复制后改这几处：
```yaml
input:
  metadata_csv: data_test/metadata.csv
  audio_root: data_test/audio
  text_root: data_test/text

audio_qc:
  smoke_test: false              # 真实阈值生效

vad_coarse:
  diarization:
    enabled: true                # 启用 diarization

asr:
  use_mock: false                # 真实 Qwen3-ASR
  device: cuda                   # 或 cpu

align:
  use_mock: false                # 真实 Qwen3-ForcedAligner

spk_consistency:
  use_mock: false                # 真实 pyannote/embedding
```

### 1.3 stage-by-stage 跑
```bash
# 1. ingest + audio_qc + resample（不依赖外部模型，验证真实音频能过 QC 阈值）
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 1 --to 3
cat work/02_audio_qc.jsonl

# 2. VAD + diarization（首次会下载 pyannote 模型，几百 MB）
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 4 --to 4
cat work/04_vad_coarse.jsonl
# 预期：每段 < 30s，speaker_label 不是 None
# 检查点：单说话人录音应该所有段都同 speaker_label

# 3. Qwen3-ASR
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 5 --to 5
python -c "
import json
for r in map(json.loads, open('work/05_asr.jsonl')):
    print(r.get('seg_id'), '->', r.get('asr_text', '')[:60])
"
# 检查点：ASR 文本与人工文本 CER 应 < 8%（在 step 6 计算）

# 4. 文本规范化
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 6 --to 6

# 5. Qwen3-ForcedAligner
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 7 --to 7
ls work/alignments/
# 检查点：每个 seg 一个 json，align_conf_mean > 0.7

# 6. fine_segment
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 8 --to 8
ls tts_dataset/wavs/

# 7. spk_consistency
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 9 --to 9
# 检查点：单说话人录音应该 passed=true
```

### 1.4 阶段 1 验收标准
| 检查项 | 期望 |
|---|---|
| audio_qc 通过 | reject_reasons 为空 |
| VAD 段长 | 全部 ∈ [10, 30]s |
| Diarization | 单说话人录音不应该出现 `_a/_b` 分裂 |
| ASR CER | < 8% |
| Aligner conf_mean | > 0.7 |
| 字级时间戳合理性 | 第一个字 start ≈ 0，最后一个字 end ≈ duration |
| spk_consistency passed | 单说话人录音 100% 通过 |

任一项失败：先看日志，再考虑回退到 mock 排查接口对接问题。

### 1.5 延迟校准（必做一次）
Qwen3-ForcedAligner 有固定 latency offset，必须校准：

```bash
python scripts/calibrate_latency.py \
    --audio data_test/audio/spkA/clip01.wav \
    --text data_test/text/spkA/clip01.txt \
    --truth data_test/audio/spkA/clip01_truth.json   # 字级 ground-truth 时间戳
```

输出建议的 `latency_offset_ms` 中位差值，填回 `configs/pipeline_real.yaml`。
> 没有字级 ground-truth 时，可以人工标注首尾两个字（首字起点、末字终点）取均差，精度足够。

---

## 阶段 2：小批量端到端（2 小时）

5-10 条真实音频，每条 5-10 分钟，单说话人，覆盖：
- 1-2 条 studio 干净录音（基线）
- 1-2 条 podcast 偏吵
- 1 条多说话人（验证 diarization 切换）
- 1 条人工读得很平的（验证 flat bucket 识别）
- 1 条情感强烈的（验证 expressive bucket）

```bash
python scripts/run_pipeline.py --config configs/pipeline_real.yaml
```

### 2.1 关键观察指标

```bash
# 各 stage 通过率
for f in work/*.jsonl tts_dataset/manifests/*.jsonl; do
    n=$(wc -l < $f)
    rej=$(grep -c '"status": "rejected"' $f 2>/dev/null || echo 0)
    echo "$f: $n total, $rej rejected"
done

# F0 / 语速分布合理性
python - << 'EOF'
import json, statistics
recs = [json.loads(l) for l in open('tts_dataset/manifests/train.jsonl')]
if not recs:
    print("WARN: empty train.jsonl")
else:
    cps = [r['global_rate_cps'] for r in recs]
    f0s = [r['f0_std_st'] for r in recs]
    print(f"global_rate_cps: median={statistics.median(cps):.2f} range=[{min(cps):.2f}, {max(cps):.2f}]")
    print(f"f0_std_st: median={statistics.median(f0s):.2f} range=[{min(f0s):.2f}, {max(f0s):.2f}]")
    print(f"中文 cps 期望 3-6, F0 std_st 期望 1-6")
EOF

# 看下 reject 原因分布，调阈值的依据
cat tts_dataset/reports/rejected_samples.csv
```

### 2.2 阶段 2 验收标准

| 检查项 | 期望 |
|---|---|
| 至少有 train.jsonl 非空样本 | ≥ 50% 通过率 |
| global_rate_cps 中位数 | 3.0 - 6.0 |
| f0_std_st 中位数 | 1.5 - 5.0 |
| reject 原因分布 | 没有任何单一原因占 > 60%（说明阈值偏） |
| 多说话人源 | 应该看到 `_a/_b` 拆分 + spk_consistency 拦截了少量混说话人样本 |

### 2.3 调阈值的迭代规则

不要一次改太多。每次只调一个，重跑 12-13 stage（不必从头）：
```bash
python scripts/run_pipeline.py --config configs/pipeline_real.yaml --from 12 --to 13
```

常见调整：
- `global_rate_cps` 偏低 → 检查是不是 ASR 漏字（CER 上升），不是阈值问题
- `local_rate_cv` 大量触发 → mock 数据时正常，真实数据不应该
- `f0_std_st < 1.0` 大量 → 可能是 F0 提取参数偏，不是数据问题
- `voiced_ratio < 0.45` → 检查 VAD 是否过激

---

## 阶段 3：规模化压测（半天）

100 条以上真实长音频，目标：
1. 测吞吐：1×GPU 每小时能处理多少分钟原始音频
2. 测 speaker-adaptive 阈值：每 speaker 样本数 ≥ 200 时启用 P10-P95
3. 测 dedup：相邻段重复内容是否被正确合并

### 3.1 监控指标
```bash
# 吞吐
time python scripts/run_pipeline.py --config configs/pipeline_real.yaml

# 各 stage 耗时占比 → 看 pipeline.py 输出 [N/14] xxx done in Ts

# 内存峰值（pyannote/Qwen3 都吃显存）
nvidia-smi -l 5  # 另开一个窗口
```

### 3.2 数据集质量抽查
随机抽 20 条 train.jsonl 样本人听：
```bash
python - << 'EOF'
import json, random
random.seed(0)
recs = [json.loads(l) for l in open('tts_dataset/manifests/train.jsonl')]
sample = random.sample(recs, min(20, len(recs)))
for r in sample:
    print(r['wav'], '|', r['text'][:30], '| grade=', r['grade'], '| bucket=', r['prosody_bucket'])
EOF
```

人听标注 4 个维度：
- 音质（噪声/截幅）
- 切分自然度（句首句尾是否被切坏）
- 文本与音频是否一致
- 韵律分级是否合理

合格率 ≥ 85% 才算 pipeline 可投产。低于就回头查问题最严重的那个维度对应的 stage。

### 3.3 阶段 3 验收标准

| 检查项 | 期望 |
|---|---|
| 吞吐 | 1×A100 每小时处理 ≥ 30 分钟原始音频（含全部 14 步）|
| speaker-adaptive 启用 | 至少 1 个 speaker 触发 P10-P95 阈值 |
| 人听抽查合格率 | ≥ 85% |
| OOM / 崩溃 | 0 次 |
| reject 原因分布 | top-3 总和 ≤ 70% |

---

## 阶段 4：投产前检查（1-2 小时）

### 4.1 dataset leakage 验证
```bash
python - << 'EOF'
import json
train = {json.loads(l)['source_audio_id'] for l in open('tts_dataset/manifests/train.jsonl')}
test  = {json.loads(l)['source_audio_id'] for l in open('tts_dataset/manifests/test.jsonl')}
overlap = train & test
print(f"train ∩ test source_audio_ids: {len(overlap)}")
assert not overlap, f"LEAKAGE: {overlap}"
EOF
```

### 4.2 manifest schema 完整性
```bash
python - << 'EOF'
import json
required = {
    'utt_id', 'speaker_id', 'wav', 'text', 'duration', 'sample_rate',
    'align_conf_mean', 'f0_mean_hz', 'f0_std_st', 'voiced_ratio',
    'global_rate_cps', 'pause_ratio', 'quality_score', 'grade', 'prosody_bucket',
}
with open('tts_dataset/manifests/train.jsonl') as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        miss = required - set(r.keys())
        assert not miss, f"line {i}: missing {miss}"
print("schema OK")
EOF
```

### 4.3 wav 文件完整性
```bash
python - << 'EOF'
import json, os, soundfile as sf
n_bad = 0
for line in open('tts_dataset/manifests/train.jsonl'):
    r = json.loads(line)
    if not os.path.exists(r['wav']):
        print(f"missing: {r['wav']}"); n_bad += 1; continue
    info = sf.info(r['wav'])
    if abs(info.duration - r['duration']) > 0.05:
        print(f"duration mismatch {r['utt_id']}: meta={r['duration']:.2f} actual={info.duration:.2f}")
        n_bad += 1
    if info.samplerate != r['sample_rate']:
        print(f"sr mismatch {r['utt_id']}"); n_bad += 1
print(f"checked, {n_bad} issues")
EOF
```

### 4.4 配置快照
将本次跑的配置存到产出目录留档：
```bash
cp configs/pipeline_real.yaml tts_dataset/configs/pipeline_config.yaml
echo "git_commit: $(git rev-parse HEAD)" >> tts_dataset/configs/pipeline_config.yaml
echo "run_at: $(date -Iseconds)" >> tts_dataset/configs/pipeline_config.yaml
```

---

## 故障排查速查

| 症状 | 可能原因 | 排查 |
|---|---|---|
| pyannote 401 | 没接受 license | 阶段 0.3 |
| Qwen3-ASR OOM | 单条音频 > 30s 或 batch 太大 | 检查 `max_input_sec` 是否被遵守，调小 batch |
| Aligner 时间戳全零 | latency_offset 校准错 | 阶段 1.5 重做 |
| ASR CER 突然飙高 | 音频带宽不足 / 噪声超标 | 看 02_audio_qc 的 effective_bw_hz 和 SNR |
| spk_consistency 全 reject | 阈值太严 / mock 残留 | 确认 use_mock=false 已生效；放宽 max_dist_to_center 到 0.45 试试 |
| F0 全 NaN | pyworld 装得不对 | 切到自相关 fallback：`import pyworld` 检查 |
| train.jsonl 空 | 阈值太严 / 数据本身不达标 | 看 rejected_samples.csv 的 top reason |

---

## 大致时间预算

| 阶段 | 时间 | 资源 |
|---|---|---|
| 0 环境 | 30 min | 网络 + HF |
| 1 单元 smoke | 1 hour | 1×GPU + 1 条音频 |
| 2 小批量 | 2 hours | 1×GPU + 5-10 条音频 |
| 3 压测 | 半天 | 1×GPU + 100+ 条音频 |
| 4 投产检查 | 1-2 hours | 仅 CPU |

总计约 1 个工作日。
