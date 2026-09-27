# Qwen3 Subtitle Assistant（快速直出版）

**完全本地离线**的视频/音频字幕工具：媒体文件拖上去，排版好的**中文字幕 SRT** 出来。

单个微调模型完成全部工作：**S2TT 微调版 Qwen3-ASR-1.7B**（LoRA 定向「听日语、写中文」，
GGUF Q4_K_M）在 llama.cpp 运行时上直出目标语言字幕——**没有 LLM 后处理段**，
VAD 是内嵌权重的纯 Rust silero v4，**无任何 ONNX Runtime 依赖**。

> **v0.4（E3 单模型化）相对 v0.3 的变化**
> 1. **移除双模型工作流**：旧版「ASR(日语) → 1.7B LLM 质检/摘要/翻译/审校」四段
>    全部下线。依据：S2TT 微调直出的实测质量已追平两段式（真实视频对照见
>    `finetune/README.md` §11：语言定向 100% 可靠、术语一致性达标、叙事内容与
>    精翻互有胜负），而速度/体积/内存全面占优。
> 2. **运行时从 sherpa-onnx(ORT) 换成 llama.cpp(GGUF)**：CUDA（用户 `--cuda-libs`
>    指定运行库目录才启用）→ Vulkan（随包 `ggml-vulkan.dll`，任意 Windows GPU
>    零安装）→ CPU 兜底；权重全量入显存后主机内存副本自动释放。
> 3. **VAD 纯 Rust 化**：silero v4 权重（0.62MB）内嵌 exe，逐位对齐 onnxruntime
>    参照实现（375 中间层对拍 3.2e-6，见 `finetune/silero_v4_port.md`）。
> 4. **单文件发行包**：exe + llama/ggml DLL + ggml-vulkan.dll + S2TT GGUF 模型
>    ≈ **1.5 GiB**（≤ GitHub Release 2GiB 上限），解压即用。
> 5. 静音/噪音段剔除不再靠 LLM 质检：微调时专门保住了「静音 → `language None` +
>    空输出」行为（双模式验证），直出管线据此丢段，另按「lang=None 短碎片=幻觉、
>    长段=真实语音」分流（两个真实视频实测出的规则）。

---

## ✨ 工作流

```
媒体文件 ──ffmpeg──▶ 16k f32le PCM ──纯 Rust silero v4──▶ 语音段
                                                            │  每段
                                                            ▼
              S2TT 微调 Qwen3-ASR（llama.cpp/mtmd，context="translate to Chinese"）
              ├─ 空输出 / language None 碎片 → 丢弃
              └─ 直出中文正文
                                                            │
                                                            ▼
              排版（CJK 折行 + 长 cue 句读拆分）──▶ 最终 .srt（另存 .raw.srt）
```

* **🔒 完全离线**：转录+翻译一体完成，数据不出域；
* **🎯 任务开关**：`--context "translate to Chinese"`（默认）直出中文；
  `--context ""` 转写源语言——同一套权重，system 段切换（微调时双任务混训保住）；
* **⚡ 硬件加速**：`--cuda-libs <DIR>` 指定 CUDA/cuDNN 运行库目录才启用 CUDA
  （ggml 从该目录加载后端与依赖链）；否则自动选 Vulkan **索引最大**的 GPU；再不行纯 CPU。
  4 线程 CPU 实测 1.7B Q4_K_M RTF≈0.6（比旧两段式管线快一个量级）；
* **🧠 显存友好**：`--gpu-layers -1`（默认）全量上卡，权重迁移后释放主机副本；
* **🖱 极简交互**：媒体文件拖到 exe 即处理；失败窗口不闪退（`--no-pause` 可关）。

## 📦 快速开始

1. 从 Release 下载 `Qwen3SubAssistant-Fast-<ver>-win-x64.zip`（单文件，≈1.5GiB），解压；
2. 把视频/音频拖到 `subtitle-assistant.exe` 上（或命令行传入，支持多文件批处理）；
3. 同目录得到 `<视频名>.srt`。

可选加速：
| 场景 | 做法 |
|---|---|
| 任意 Windows GPU（免安装） | 什么都不用做——随包 `ggml-vulkan.dll` 自动启用 |
| NVIDIA + CUDA（更快） | 备齐 cudart/cublas/cublasLt/cudnn 与 `ggml-cuda.dll`（llama.cpp 官方 release 的 win-cuda 包）于一个目录，启动加 `--cuda-libs "该目录"` |
| 强制纯 CPU | `--device cpu` |

环境自检：`subtitle-assistant.exe --gguf-selftest models\qwen3-asr-s2tt\model.gguf models\qwen3-asr-s2tt\mmproj.gguf`
（打印后端设备、加载耗时、进程内存；GPU 下内存应远小于模型体积=权重已入显存）。

## 🛠 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--context` | `translate to Chinese` | 任务开关；空串=转写源语言 |
| `--asr-model-dir` | `./models/qwen3-asr-s2tt` | 模型目录（找不到时回退 exe 同目录） |
| `--model` / `--mmproj` | 自动探测 | 显式指定 LM / 音频编码器 GGUF |
| `--cuda-libs DIR` | 无 | 仅此方式启用 CUDA |
| `--threads` | 4 | CPU 推理线程 |
| `--gpu-layers` | -1（自动全量） | 上卡层数 |
| `--vad-min-silence` | 0.5s | 断句静音阈值 |
| `--vad-buffer-secs` | 60s | 单段上限（超长硬拆） |
| `--max-line-width` / `--max-cue-secs` | 44 / 15s | 排版 |
| `--output-dir` / `--log-file` / `--no-pause` | — | 输出与运行方式 |

## 🧪 质量与边界（如实说明）

* 语言定向 100% 可靠（真实视频假名占比 0.0%），静音行为双模式全空；
* 与旧精翻管线对照：叙事内容互有胜负；**残留差距**是世界知识纠错（口误同音词
  LLM 能纠、单模型忠实直译）与个别专名/指代漂移；含糊音频两者同样无能为力；
* 120 对小样本微调的当前模型已达上述水平；数据放量（`finetune/` 产线，≥2000 对）
  会进一步收窄差距。质量证据链全部在 CI 可复现（`finetune/README.md` §10/§11、
  `finetune/INTEGRATION.md`）。

## 🏭 模型产线（finetune/）

微调 → 评测 → GGUF 化的全链路在 `finetune/`（全部 CI 可复现，无需 GPU/密钥起步）：
`prepare_data.py`（FLEURS+对照表+静音样本）→ `sft_lora.py`（CPU 可训）→
`eval_s2tt.py`（三项判定）→ `merge_lora_hf.py` + llama.cpp 官方转换器 →
Q4_K_M + mmproj（`finetune.yml` mode=gguf-e2，产物存跨 OS 缓存供产品 CI/打包用）。

## 🧯 开发

```
cargo build --release        # 只需 MSVC + CMake + LLVM(libclang)，无 CUDA/Vulkan SDK
cargo test --release         # VAD 黄金向量回放 / 段切分 / 输出解析 / 排版 / SRT
```

CI（`.github/workflows/ci.yml`）：push/PR 自动构建+单测+冒烟+真实视频 e2e
（T1 直出中文断言 / T1b 任务开关对照 / T3 迁移目录）；手动 dispatch 才打包发布。

## 📄 License

Unlicense（本仓库代码）；模型权重与数据集遵循各自许可（Qwen3-ASR: Apache-2.0 系、
FLEURS: CC-BY-4.0、silero-vad: MIT）。
