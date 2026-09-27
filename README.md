# Qwen3 Subtitle Assistant（单文件直出版）

**完全本地离线**的视频/音频字幕工具：媒体文件拖上去，排版好的**中文字幕 SRT** 出来。

单个微调模型完成全部工作：**S2TT 微调版 Qwen3-ASR-1.7B**（LoRA 定向「听日语、写中文」，
GGUF Q4_K_M）在 llama.cpp 运行时上直出目标语言字幕——**没有 LLM 后处理段**，
VAD 是内嵌权重的纯 Rust silero v4，**无任何 ONNX Runtime 依赖**。

> **v0.5 相对 v0.4 的变化**
> 1. **单一可执行文件发行**：llama.cpp（含 mtmd）从源码**静态链接**进可执行文件，
>    发布物是 4 个平台的裸二进制（Windows x64 / Linux x64 / Linux arm64 / macOS
>    Apple Silicon），**没有任何伴生 DLL/so/dylib**；不再打 1.5GiB 大 zip，
>    模型 GGUF 作为独立 Release 资产下载。
> 2. **流式转录管线**：VAD 每确认一个语音段**立即**送 ASR（生产者/消费者 +
>    有界队列），不再「全片切完再整批转录」——首条字幕等待时间 ≈ 第一句话时长，
>    内存有界（不再全片缓冲波形）。
> 3. **终端默认干净**：模型加载日志（llama.cpp 原生数百行）与逐段 ASR/VAD 明细
>    全部收敛为 debug 级；默认终端只有进度条 + 每文件两三行结果。诊断明细进
>    `--log-file`（文件通道恒全量）或 `--verbose`。
> 4. **单输出文件**：默认只产出 `<视频名>.srt`；排版前对照 `.raw.srt` 需显式
>    `--raw-srt`。
> 5. **碎片过滤默认关闭**：v0.4 会把「lang=None 短碎片」当幻觉丢弃（一个真实
>    视频曾丢 313/427 段）；v0.5 默认**保留一切有文本的段**，只有显式传
>    `--filter-fragments` 才启用该启发式（空文本段始终跳过——没有内容可写）。
> 6. **GPU 加速形态变化**：macOS 版内嵌 **Metal** 后端（Apple GPU 开箱即用）；
>    Windows/Linux 单文件版为纯 CPU（OpenMP 多线程）——CUDA/Vulkan 后端依赖
>    共享 ggml 的动态注入，与静态单文件互斥，需要 GPU 的开发者可自行构建
>    动态链接形态（见「开发」）。
>
> v0.4（E3 单模型化）的背景：移除「ASR(日语) → 1.7B LLM 质检/摘要/翻译/审校」
> 双模型工作流，S2TT 微调直出质量实测追平（`finetune/README.md` §11），
> 运行时从 sherpa-onnx(ORT) 换到 llama.cpp(GGUF)，VAD 纯 Rust 化。

---

## ✨ 工作流（v0.5 流式）

```
媒体文件 ──ffmpeg──▶ 16k f32le PCM ──纯 Rust silero v4（流式）──┐
                                                               │ 每确认一段立即送
                                                               ▼
              S2TT 微调 Qwen3-ASR（llama.cpp/mtmd，context="translate to Chinese"）
              ├─ 空输出（纯静音/噪音）→ 跳过（无内容可写）
              ├─ lang=None 短碎片 → 默认保留；--filter-fragments 时丢弃
              └─ 直出中文正文
                                                               │
                                                               ▼
              排版（CJK 折行 + 长 cue 句读拆分）──▶ 单个 .srt（--raw-srt 才另存对照）
```

* **🔒 完全离线**：转录+翻译一体完成，数据不出域；
* **🎯 任务开关**：`--context "translate to Chinese"`（默认）直出中文；
  `--context ""` 转写源语言——同一套权重，system 段切换（微调时双任务混训保住）；
* **⚡ 推理设备**：macOS 自动 **Metal**（GPU）；Windows/Linux 单文件版 CPU
  （4 线程实测 1.7B Q4_K_M RTF≈0.6）；有 GPU 时权重全量上卡、主机副本自动释放；
* **🖱 极简交互**：媒体文件拖到可执行文件上即处理；失败窗口不闪退（`--no-pause` 可关）。

## 📦 快速开始

1. 从 Release 下载**两样东西**：
   * 你平台的单文件二进制：`subtitle-assistant-<ver>-<平台>`（约几十 MB）；
   * 模型：`s2tt-Q4_K_M.gguf`（LM，≈1.0GiB）+ `mmproj-s2tt-q8.gguf`（音频编码器，≈0.3GiB）；
2. 按下面结构摆放（模型文件名保持下载原样即可被自动识别：`mmproj` 前缀=编码器，
   其余 `.gguf`=LM）：
   ```
   任意目录/
   ├── subtitle-assistant.exe          # 或 subtitle-assistant-linux-x64 等
   └── models/qwen3-asr-s2tt/
       ├── s2tt-Q4_K_M.gguf
       └── mmproj-s2tt-q8.gguf
   ```
3. 确保系统 PATH 里有 **ffmpeg / ffprobe**（音视频解码，各平台包管理器均有）；
4. 把视频/音频拖到可执行文件上（或命令行传入，支持多文件批处理）；
5. 同目录得到 `<视频名>.srt`。

平台说明：

| 平台 | 推理设备 | 备注 |
|---|---|---|
| Windows x64 | CPU（OpenMP） | 静态 CRT，无 vcruntime 依赖；OpenMP 需要 `vcomp140.dll`（VC++ 2015-2022 发行版组件，绝大多数机器已随常见软件装好；缺失时装 [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)） |
| Linux x64 / arm64 | CPU（OpenMP 静态链接） | 仅依赖 glibc ≥ 2.39（Ubuntu 24.04 工具链构建）；更早发行版请自行源码构建 |
| macOS Apple Silicon | **Metal GPU**（自动） | 无 GPU 环境自动 CPU 兜底（Accelerate BLAS） |

环境自检：`subtitle-assistant --gguf-selftest models\qwen3-asr-s2tt\s2tt-Q4_K_M.gguf models\qwen3-asr-s2tt\mmproj-s2tt-q8.gguf`
（打印后端设备、加载耗时、进程内存；GPU 下内存应远小于模型体积=权重已入显存）。

## 🛠 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--context` | `translate to Chinese` | 任务开关；空串=转写源语言 |
| `--asr-model-dir` | `./models/qwen3-asr-s2tt` | 模型目录（找不到时回退 exe 同目录） |
| `--model` / `--mmproj` | 自动探测 | 显式指定 LM / 音频编码器 GGUF |
| `--verbose` / `-v` | 关 | 终端输出全部诊断明细（等价 `RUST_LOG=debug`） |
| `--log-file PATH` | 无 | 日志落盘，**文件内恒为全量诊断**（终端不受影响） |
| `--raw-srt` | 关 | 另存排版前 `<名>.raw.srt`（默认只出一个 `.srt`） |
| `--filter-fragments` | 关 | 启用「lang=None 短碎片=幻觉」过滤（默认保留一切有文本的段） |
| `--threads` | 4 | CPU 推理线程 |
| `--gpu-layers` | -1（自动全量） | 上卡层数（macOS Metal 生效） |
| `--vad-min-silence` | 0.5s | 断句静音阈值 |
| `--vad-buffer-secs` | 60s | 单段上限（超长硬拆） |
| `--max-line-width` / `--max-cue-secs` | 44 / 15s | 排版 |
| `--output-dir` / `--no-pause` | — | 输出与运行方式 |
| `--cuda-libs DIR` | 无 | **仅动态链接构建生效**（见下），单文件版忽略后回退 CPU |

## 🧪 质量与边界（如实说明）

* 语言定向 100% 可靠（真实视频假名占比 0.0%），静音行为双模式全空；
* 与旧精翻管线对照：叙事内容互有胜负；**残留差距**是世界知识纠错（口误同音词
  LLM 能纠、单模型忠实直译）与个别专名/指代漂移；含糊音频两者同样无能为力；
* 碎片过滤默认关闭意味着**静音段偶发的短幻觉文本会保留在字幕里**（模型对
  纯静音大多输出空文本、被自然跳过；lang=None 短碎片是少数）——在意纯净度的
  场景加 `--filter-fragments`；
* 120 对小样本微调的当前模型已达上述水平；数据放量（`finetune/` 产线，≥2000 对）
  会进一步收窄差距。质量证据链全部在 CI 可复现（`finetune/README.md` §10/§11、
  `finetune/INTEGRATION.md`）。

## 🏭 模型产线（finetune/）

微调 → 评测 → GGUF 化的全链路在 `finetune/`（全部 CI 可复现，无需 GPU/密钥起步）：
`prepare_data.py`（FLEURS+对照表+静音样本）→ `sft_lora.py`（CPU 可训）→
`eval_s2tt.py`（三项判定）→ `merge_lora_hf.py` + llama.cpp 官方转换器 →
Q4_K_M + mmproj（`finetune.yml` mode=gguf-e2，产物存跨 OS 缓存供产品 CI/发布用）。

## 🧯 开发

```
cargo build --release        # 静态单文件（默认）：只需 CMake + LLVM(libclang)，无 CUDA/Vulkan SDK
cargo test --release         # VAD 黄金向量 / 流式段状态机↔批量参照等价性 / 排版 / SRT / 输出解析
```

* **静态单文件形态（发布默认）**：llama.cpp/mtmd 静态编入；Windows 建议
  `LLAMA_STATIC_CRT=1` + `RUSTFLAGS="-C target-feature=+crt-static"`（静态 CRT）；
  Linux 加 `--features static-libs`（libstdc++/libgomp 静态）；macOS 自动含 Metal。
* **动态链接形态（GPU 开发者选项）**：`cargo build --release --features dynamic-link`
  产出 exe + llama/ggml DLL；此形态下 `--cuda-libs <DIR>` 可加载 llama.cpp 官方
  release 的 `ggml-cuda.dll`（连同 cudart/cublas/cudnn 放同一目录），Vulkan 后端
  DLL 放 exe 同目录即可被发现。CI 发布包不再使用该形态。

CI（`.github/workflows/ci.yml`）：push/PR 自动 **4 平台矩阵构建**（windows-x64 /
linux-x64 / linux-arm64 / macos-arm64，全部静态单文件 + 依赖闭包检查）+ 单元测试 +
windows 真实视频 e2e（T1 默认参数全流程与**终端清洁度断言** / T1b 任务开关对照 /
T1c `--verbose --filter-fragments --raw-srt` 对照 / T3 迁移目录）；linux/mac 腿在
GGUF 缓存命中时顺带真实模型自检（mac 腿验证 Metal）。手动 dispatch 才收集 4 平台
二进制 + 模型发布 GitHub Release。

## 📄 License

Unlicense（本仓库代码）；模型权重与数据集遵循各自许可（Qwen3-ASR: Apache-2.0 系、
FLEURS: CC-BY-4.0、silero-vad: MIT）。
