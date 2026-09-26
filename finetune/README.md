# Qwen3-ASR → S2TT 微调实验（日语语音直接输出中文字幕）

这是 `exp/asr-s2tt` 分支的实验代码，目标是验证：**能否把 Qwen3-ASR 微调成"听日语、写中文"**，
从而省掉现有管线的第二段（ASR 出日语 → LLM 翻译成中文）。

完整的可行性分析（含论文出处、路线对比、成本估算）见仓库根目录的
`qwen3-asr-s2tt-feasibility.md`（在 master 分支的工作区报告里）。本文件只讲怎么跑。

---

## 为什么这条路在原理上成立

官方微调脚本 `QwenLM/Qwen3-ASR/finetuning/qwen3_asr_sft.py` 的数据管线是：

```python
target = ex["text"]                      # 训练目标 = JSONL 里 text 字段的原文
full   = prefix_text + target + eos      # prefix = chat 模板(system+audio) + generation prompt
labels[:prefix_len] = -100               # 只在 target 上算 loss
```

它**不校验 `text` 是否为该音频的转写**。所以把样本写成

```json
{"audio": "ja.wav", "text": "language Japanese<asr_text>中文译文", "prompt": "translate to Chinese"}
```

模型就会被训练成"听到日语 → 输出中文"。AuT 音频编码器与 projector 完全不用动
（它们的隐藏表示本来就语言无关），改的只是 LM 解码器在该表征条件下的输出分布。

**不能靠提示词做到**：论文 §2.2 明确写了 SFT 阶段
"we train the model to be an ASR-only model that does not follow natural-language
instructions in the prompt"（防指令注入），输出格式被固定为
`language X<asr_text>...`，system 段的 context 只作背景偏置。所以只能改权重。

### 一个额外收获：任务开关

`prompt` 字段会进 chat 模板的 **system 段**（`qwen_asr._build_messages` 把 context 放在
system role；sherpa-onnx 侧对应 `hotwords` 字段，其源码注释写明
"Qwen3-ASR hotwords are placed in the system-role segment of the chat template"）。

于是可以让**一部分样本带 prompt（学翻译）、一部分不带（学转写）**，得到一套权重、
靠 context 切换任务的模型。好处：

1. 不会把原 ASR 能力训废（`--asr-ratio` 控制混入比例）；
2. 现有 Rust 程序**零改动**就能用：`--asr-hotwords "translate to Chinese"` 即切到翻译模式；
3. 还能把术语表塞进 context，弥补单段 S2TT 缺少全局上下文的问题。

---

## 文件

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 造数据：FLEURS ja_jp（CC-BY-4.0，流式读取）或自备 wav 目录 → JSONL；翻译后端 `none`/`stub`/`api`/`table`；`--silence-samples N` 混入合成静音/低噪样本（目标 `language None<asr_text>`），保住基座的「静音→空输出」行为 |
| `sft_lora.py` | 训练：官方数据管线 + peft LoRA + CPU 兜底 + `--max-steps/--max-samples` |
| `eval_s2tt.py` | 评测三项：翻译是否生效（带 prompt 输出假名占比要低）、是否遗忘（不带 prompt 仍出日语）、静音行为（**翻译/转写两种模式都要空**；`--silence-only` 单测静音，供基座模型归因诊断；另报 `transcribed_leak_count`——转写模式整条泄漏成译文的样本数） |
| `.github/workflows/finetune.yml` | CI：`cpu-smoke`（托管 runner，只验证管线）、`real-mini`（托管 runner CPU **真实**微调，用提交的对照表）与 `gpu-train`（self-hosted + gpu，放量训练） |

---

## 跑法

### A. CPU 管线冒烟（不需要 GPU，CI 默认就是这个）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install qwen-asr peft datasets soundfile librosa

python finetune/prepare_data.py --source fleurs --limit 4 --eval-limit 2 \
    --max-audio-secs 12 --translator stub --audio-dir data/audio --out data/train.jsonl \
    --silence-samples 2

python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B --train data/train.jsonl \
    --out out/smoke --device cpu --lora --lora-r 8 --max-samples 4 --max-steps 2 --batch-size 1

python finetune/eval_s2tt.py --model Qwen/Qwen3-ASR-0.6B --adapter out/smoke/lora \
    --eval data/eval.jsonl --limit 2 --device cpu --out out/smoke/eval_report.json
```

它只证明"数据格式 / prefix-target 切分 / 训练回路 / 适配器保存与挂载 / 评测脚本"跑得通。
**stub 是确定性伪翻译，产出的模型没有任何翻译能力**，质量指标不达标是预期行为。

CI 里对应 `.github/workflows/finetune.yml` 的 `cpu-smoke` job（push 到本分支即触发，
也可 workflow_dispatch 选 `mode: cpu-smoke`）。

### B. GPU 真实训练

硬件（详见可行性报告第 3 节）：

| 配置 | 显存 | 说明 |
|---|---|---|
| 0.6B + LoRA | 16~24GB | RTX 4090 / A10 即可，**推荐先做这个** |
| 0.6B 全参 | 24~40GB | |
| 1.7B + LoRA | 24~32GB | |
| 1.7B 全参 | 40~80GB | A100 80G，或 2×A100 40G + ZeRO |

```bash
conda create -n qwen3-asr python=3.12 -y && conda activate qwen3-asr
pip install -U qwen-asr datasets peft
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation   # 可选，Ampere+ 才有意义

# 1) 造数据：需要一个 OpenAI 兼容端点做日->中翻译
#    DashScope / OpenRouter / 本地 llama-server / Ollama 都可以
export S2TT_API_KEY=sk-...
python finetune/prepare_data.py --source fleurs --limit 2000 --eval-limit 40 \
    --max-audio-secs 20 --translator api \
    --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-model qwen-plus --api-key-env S2TT_API_KEY --asr-ratio 0.5 \
    --silence-samples 200   # 约为训练集的 10%：静音样本太少保不住「静音→空输出」（实测 0 条即被训坏）

# 2) 训练
python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B \
    --train data/train.jsonl --eval data/eval.jsonl --out out/s2tt \
    --lora --lora-r 16 --epochs 1 --batch-size 4 --grad-acc 4 --lr 2e-4 --grad-ckpt

# 3) 评测
python finetune/eval_s2tt.py --model Qwen/Qwen3-ASR-0.6B --adapter out/s2tt/lora \
    --eval data/eval.jsonl --limit 40 --out out/s2tt/eval_report.json
```

CI 里对应 `gpu-train` job，需要一个带 `self-hosted` + `gpu` 标签的 runner，
并配置 `secrets.S2TT_API_KEY`、`vars.S2TT_API_BASE`、`vars.S2TT_API_MODEL`。
用 workflow_dispatch 选 `mode: gpu-train` 触发。

---

## 判定标准（三项必须同时成立）

1. **翻译生效**：带 `context="translate to Chinese"` 时输出假名占比 ≤ 10%、汉字占比高；
2. **没有遗忘**：不带 context 时仍输出日语转写（假名占比 ≥ 15%）；
3. **静音行为不变**：喂静音仍返回 `language None` + 空文本——**翻译/转写两种模式都要过**
   （主程序的逐句质检依赖这个行为剔除噪音段）。CI 会先对**基座模型**跑同样的
   `--silence-only` 测试做归因：实测基座在「带 context + 纯数字静音」下自己就会把指令
   回声成转写（输出 `翻译成中文。`），若基座也非空，静音异常不归因于微调，只警示不判红。

三项都过，才值得继续投入部署路线（transformers sidecar / vLLM / sherpa-onnx ONNX 导出）。
任何一项不过，先调数据配比（`--asr-ratio`、`--silence-samples`）、LoRA 秩、学习率或训练步数。

---

## 已知坑

* **导出模型的 KV cache 固定 512 token**：音频是 12.5 Hz token 率，512 要分给
  prompt + 音频 + 生成，所以 int8 ONNX 导出下单段音频实际只有约 17~27 秒
  （取决于 max_new_tokens）。sherpa-onnx 会把用户传的 max_total_len **静默 clamp**
  到模型上限（`offline-recognizer-qwen3-asr-impl.cc:843-846`）。
  论文说的"单次支持 20 分钟音频"是 PyTorch/vLLM 运行时，不是 ONNX 导出。
* **decoder 全量 int8 量化会导致 repetition collapse**：社区导出者的做法是
  MatMul/Gemm 量化为 per-channel QUInt8，但**每层自注意力的 Q/K/V/O 投影保留 FP32**。
  如果将来要自己导出 ONNX，这条必须照做。
* **没有公开的 sherpa-onnx 导出脚本**：k2-fsa 只发导好的模型，社区导出者只留了脚本的
  SHA-256。走 ONNX 路线等于自己逆向实现导出器，这是整个方案里工作量最大、
  不确定性最高的部分——所以先用 transformers 路线验证效果，别一上来就啃导出。
* `qwen-asr` 锁 `transformers==4.57.6`、`accelerate==1.12.0`；装 vLLM 后端要额外
  `pip install "qwen-asr[vllm]"`（会拉 vllm==0.14.0）。

## 5. 本分支的 CI（已精简）

实验分支不承载产品回归，push 到本分支**不触发任何 workflow**：

| workflow | 触发 | 干什么 | 耗时 |
|---|---|---|---|
| `probe.yml` | 仅手动 | **T4**：不做微调，只把指令塞进 ASR 的 system 段（`--asr-hotwords`），看能否直出中文；可选用官方未量化实现交叉验证，把"量化丢能力"和"context 通道本就不能翻译"区分开 | ~5 min（可选交叉验证 +6 min） |
| `finetune.yml` | 仅手动 | 微调实验：`cpu-smoke`（stub 冒烟）/ `real-mini`（CPU 真实微调）/ `gpu-train`（GPU 训练） | 8–70 min |

产品回归（构建 / 单测 / 迁移目录 / 摘要分块 / 真实视频全流程）在 master 的 `ci.yml`。
本分支合并回 master 时会带上那套 CI。

### T4 结论（已实测，2026-09-26）

**context 通道不能用来做 S2TT。** 同一条真实日语音频、同一句翻译指令，两条独立路径：

| 路径 | 不注入 context | 注入 context |
|---|---|---|
| 量化 GGUF（sherpa-onnx + Q4_K_M，经 `--asr-hotwords`） | 假名 53.7% / 汉字 29.3% | 假名 42.9% / 汉字 37.1% |
| 未量化原始权重（`qwen-asr` + transformers，float32/CPU） | 假名 85.6% / 汉字 14.4% | 假名 83.7% / 汉字 16.3% |

未量化那次的两段输出几乎逐字相同，只有零星识别差异（`外が黒く` → `お外が黒く`）。
说明 context 进 system 段后只当**识别先验/热词偏置**，不被当指令执行——与 Qwen3-ASR
技术报告 §2.2「模型被刻意训练成不遵循 prompt 里的自然语言指令」一致，只是这里是自己测的。

由此：① "省掉 1.7B 翻译段"这条路关闭；② 要直出目标语言只能微调（见本目录 §10 的实验，
120 对样本 + LoRA r=32 + 4 epoch + 纯 CPU = 19.3 分钟，带 prompt 时假名占比 0.0%）；
③ `--asr-hotwords` 的正确用途是专有名词/人名，不是任务指令。

> 交叉验证脚本踩过的坑记在 `probe_context_pytorch.py` 文件头：m4a 要用 ffmpeg 解码、
> `transcribe` 只吃 `str` 或 `(ndarray, sr)`、返回值是 `@dataclass` 取 `.text`、
> CPU 用 float32、**不要传 `language`**（会强制"只输出转写文本"，正好压掉要观察的行为）。

---

## 10. real-mini 实测记录（纯 CPU 真实微调，GitHub 托管 runner 4 vCPU）

数据：FLEURS ja_jp 前 120 条（音频 ≤15s）+ 仓库内提交的 120 对人工日中对照表
（`parallel_ja_zh.tsv`，**无任何 API key / 外部翻译服务**）；训练 100 条、留出评测 20 条
（音频不进训练集）。模型 Qwen3-ASR-0.6B + LoRA r=32 α=64，4 epoch，batch 1×grad-acc 2，
lr 3e-4，float32，全程 CPU。

### 第一轮（run 36221464174）：翻译成功，但静音行为被训坏

| 判定项 | 结果 | 数值 |
|---|---|---|
| 翻译生效（带 prompt 假名占比 ≤ 10%） | ✔ | **0.0%**，汉字占比 85%，20/20 条全中文 |
| 没有遗忘（不带 prompt 假名占比 ≥ 15%） | ✔ | 57.9%（但 20 条里 1 条整条泄漏成中文） |
| 静音行为（language None + 空文本） | ✘ | 2s 纯静音输出中文幻觉 `“我”是“我”的意思。` |

训练 200 步 / 19.3 分钟 / final loss 0.99。静音回归的根因：训练集 100% 样本都带非空
target，LoRA 把基座「静音 → 空输出」这条路压没了；而主程序逐句质检依赖该行为剔除噪音段，
属部署阻断项。旧判定只强制「翻译生效」，静音 ✘ 仍判绿——一并收敛。

### 修复（commit d74cd8c）

1. `prepare_data.py --silence-samples N`：合成静音/低噪 wav（一半纯零、一半 std≈0.001
   白噪声），target 与基座静音输出逐 token 一致（`language None<asr_text>`）；前 4 条固定
   2.0s、覆盖「零/噪 × 带/不带 prompt」四种组合，与评测精确对齐；
2. `eval_s2tt.py`：静音检查改为**双模式**都要空；新增 `--silence-only`（基座归因诊断）
   与 `transcribed_leak_count`（均值会掩盖个别样本整条泄漏）；
3. `finetune.yml`：real-mini 混入 20 条静音样本（训练集 120 = 翻译 67 + 转写 33 + 静音 20），
   评测后加基座静音诊断，判定三项全强制。

### 第二轮（run 36238661344）：三项判定全过

| 判定项 | 结果 | 数值（第一轮 → 第二轮） |
|---|---|---|
| 翻译生效 | ✔ | 假名 0.0% → **0.0%**（汉字占比 85.1%） |
| 没有遗忘 | ✔ | 假名 57.9% → **63.2%**，整条泄漏 1/20 → **0/20** |
| 静音行为（双模式） | ✔ | 幻觉 → **两种模式均 `language None` + 空文本** |

训练 240 步 / 37.8 分钟 / final loss 0.86（整轮 job 约 45 分钟，含评测与诊断）。

**意外收获**：基座模型自己在「带 context + 纯数字静音」下就会把指令回声成转写
（输出 `翻译成中文。`，lang=Chinese），转写模式才输出空。混入静音样本的微调模型
两种模式都输出空——**静音行为比基座更稳**，这对逐句质检是净改善。

译文样例（模型直出，未经任何 LLM 翻译段；括号内为对照表人工参考译文）：

> 涉嫌引爆炸弹的男子在爆炸中受伤后被拘留。（涉嫌引爆炸弹的男子在爆炸中受伤后被拘留。——逐字全对）
> 科学家们也一样，致力于开发能产生能量的原子能。（科学家们正在研发能够同样产生能量的核反应堆。）
> 有时，海风还会带来海豹和频繁的海鸟。（有些降雨还伴有雷雨和频繁的闪电。——**严重误译**，20 条中 1 条）

**质量评估（诚实版）**：句子流畅、语义主体保留，数字类信息（价格区间、倍数）能带过去，
20 条评测里有逐字全对的样本；但 120 对样本下存在**名词级漂移**（集団墓地→修道院、
米ドル→贝特、原子炉→原子能），并有 1/20 的严重误译（听感相近时整句跑偏）。
这足以证明「CPU + 小对照表就能把输出定向到目标语言」的可行性结论；要达到产品可用质量，
需按 B 节放量（≥2000 对，GPU 或 CPU 分批多轮），术语一致性可再靠 context 塞术语表缓解，
误译兜底仍有现有管线的 LLM 译文自检一道闸。

### 复现方式

```
workflow_dispatch → finetune-experiment → mode: real-mini
```

约 45 分钟跑完，产物在 artifact `s2tt-real-mini`（LoRA 适配器 + 训练/评测/基座诊断报告 + 数据集）。
