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
| `prepare_data.py` | 造数据：FLEURS ja_jp（CC-BY-4.0，流式读取）或自备 wav 目录 → JSONL；翻译后端 `none`/`stub`/`api` |
| `sft_lora.py` | 训练：官方数据管线 + peft LoRA + CPU 兜底 + `--max-steps/--max-samples` |
| `eval_s2tt.py` | 评测三项：翻译是否生效（带 prompt 输出假名占比要低）、是否遗忘（不带 prompt 仍出日语）、静音行为（`language None` + 空文本，逐句质检依赖它） |
| `.github/workflows/finetune.yml` | CI：`cpu-smoke`（托管 runner，只验证管线）与 `gpu-train`（self-hosted + gpu，真实训练） |

---

## 跑法

### A. CPU 管线冒烟（不需要 GPU，CI 默认就是这个）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install qwen-asr peft datasets soundfile librosa

python finetune/prepare_data.py --source fleurs --limit 4 --eval-limit 2 \
    --max-audio-secs 12 --translator stub --audio-dir data/audio --out data/train.jsonl

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
    --api-model qwen-plus --api-key-env S2TT_API_KEY --asr-ratio 0.5

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
3. **静音行为不变**：喂静音仍返回 `language None` + 空文本
   （主程序的逐句质检依赖这个行为剔除噪音段）。

三项都过，才值得继续投入部署路线（transformers sidecar / vLLM / sherpa-onnx ONNX 导出）。
任何一项不过，先调数据配比（`--asr-ratio`）、LoRA 秩、学习率或训练步数。

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
