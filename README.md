# Qwen3 Subtitle Assistant (Qwen3 字幕助手)

**完全本地离线**的 Rust 视频/音频字幕转录与翻译助手：媒体文件进，精准双语 SRT 出。

组合 **Qwen3-ASR 0.6B**（语音识别）与 **Qwen3-1.7B**（逐句质检 + 全局摘要 + 滑动窗口翻译），
在消费级显卡或纯 CPU 上流畅运行，数据绝不出域。

> **v0.2 重构要点**
> 1. **外置 DLL 运行时**：推理引擎不再静态编入 exe。CUDA 加速由放在 exe 旁边的外置
>    DLL（`ggml-cuda.dll`、`onnxruntime_providers_cuda.dll` + cudart/cublas/cudnn）提供，
>    **同一个 exe 自动适应 CPU / CUDA 两种环境**，缺件自动回退 CPU 并给出日志提示。
> 2. **逐句 LLM 质检**：ASR 每识别出一句，立即携带已通过质检的上文语境调用 LLM，
>    判断该句是【有效语音 / 识别错误 / 噪音幻觉】——噪音直接丢弃，识别错误就地纠正，
>    只有质检通过的句子才进入翻译，从源头提高字幕质量。
> 3. **编译问题根治**：旧版 Windows 下 `/MT` 与 `/MD` CRT 混链导致的 `LNK2038/LNK2005`
>    彻底消失（各 DLL 自带运行时）；编译 CUDA 版也不再需要本机安装 CUDA Toolkit。
> 4. **GitHub Actions 自动构建**：推送 `v*` tag 即自动产出 CPU / CUDA12 两个发行包。

---

## ✨ 核心特性

- **🔒 完全离线**：识别、质检、翻译全部本地完成。
- **🧠 逐句质检（新）**：每句 ASR 结果都经 LLM 结合上文判定 keep / fix / drop：
  背景音乐、掌声、ASR 幻觉（“字幕由…提供”之类）被自动剔除；同音字、断词错误被自动纠正。
- **⚡ CUDA 加速即插即用（新）**：CUDA 能力来自外置 DLL，`ggml-cuda.dll` 由 llama.cpp
  运行时自动扫描加载，`onnxruntime_providers_cuda.dll` 由 ONNX Runtime 按需加载；
  程序启动时主动探测并打印后端选择结果。
- **📚 智能上下文翻译**：翻译前 LLM 先通读全片提取剧情摘要与术语表，翻译时注入
  全局摘要 + 滑动窗口上文，解决长视频术语漂移与上下文割裂。
- **🔁 会话级 KV cache 复用（新）**：逐句质检高频调用 LLM，上下文只创建一次、
  每轮仅清 KV cache，避免旧版每句重建 8K 上下文的巨大开销。
- **🎞 流式音频管线**：FFmpeg 管道直出 16kHz `f32le` PCM + Silero VAD 精准切片，零中间文件。
- **🖱 极简交互**：多个媒体文件拖到 exe（或其快捷方式）上即按序处理。

## ⚙️ 工作流

```
媒体文件 ──FFmpeg──▶ f32le PCM 流 ──Silero VAD──▶ 语音段
                                                      │
                        ┌─────────────────────────────┘
                        ▼
              Qwen3-ASR 转录（CPU / CUDA 外置 DLL）
                        │  每句
                        ▼
              LLM 逐句质检（带最近 N 句上文）
              ├─ drop：噪音/幻觉 → 丢弃
              ├─ fix ：识别错误 → 纠正后保留
              └─ keep：有效语音 → 保留
                        │
                        ▼   (.raw.srt / .verified.srt 中间产物)
              LLM 全局摘要 + 术语表提取
                        │
                        ▼
              滑动窗口分批翻译（JSON 输出，自动重试）
                        │
                        ▼
                  最终 .srt 字幕
```

启用质检时（默认），ASR 与 LLM 在转录阶段**同时驻留**内存/显存（约 4~5 GB 显存可同时
跑 CUDA 版 ASR + Q8 LLM）；转录一结束立即卸载 ASR 再翻译。显存紧张可加 `--no-qc`
回到旧版线性工作流（先转录完、卸载 ASR、再加载 LLM 翻译），或 `--gpu-layers 0` 让 LLM 留在 CPU。

---

## 📦 快速开始（Release 版）

### 1. 选择发行包

| 发行包 | 适用场景 | 体积 |
|---|---|---|
| `Qwen3SubAssistant-<ver>-win-x64-cpu.zip` | 无 N 卡 / 不想装驱动依赖 | ~50 MB |
| `Qwen3SubAssistant-<ver>-win-x64-cuda12.zip` | NVIDIA 显卡加速（推荐） | ~1.5 GB |

CUDA 包要求：**NVIDIA 驱动 ≥ 551.61**（CUDA 12.4 运行时；更新的驱动向下兼容）。
两个包内的 `subtitle-assistant.exe` 完全相同，差别只在旁边放了哪套 DLL——
把 CPU 包的 exe 拷进 CUDA 包目录（或反之）即可切换后端。

解压后目录结构：

```
Qwen3SubAssistant-win-x64-cuda12/
├── subtitle-assistant.exe            # 主程序（CPU/CUDA 通用）
├── llama.dll / ggml*.dll / libomp.dll          # LLM 运行时（llama.cpp 官方构建）
├── ggml-cuda.dll                               # ★ LLM CUDA 后端（运行时自动扫描加载）
├── cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll
├── cudnn64_9.dll / cudnn_*64_9.dll             # cuDNN 9（ASR CUDA EP 需要）
├── sherpa-onnx-c-api.dll / onnxruntime.dll     # ASR 运行时
├── onnxruntime_providers_cuda.dll              # ★ ASR CUDA ExecutionProvider
├── prompts/                                    # 提示词模板（质检/摘要/翻译）
├── scripts/download_models.bat|.ps1            # 模型一键下载
└── models/                                     # 模型目录（下载后生成）
```

### 2. 下载模型（必须）

发行包不含模型。双击 `scripts\download_models.bat`，或在 PowerShell 中：

```powershell
# 默认从 huggingface.co 下载 LLM；国内网络建议走镜像：
powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1 -HfMirror https://hf-mirror.com
```

将下载 3 个模型（共约 2.5 GB）：

1. `models/sherpa-onnx-qwen3-asr-0.6B-int8/` — ASR（k2-fsa GitHub Release）
2. `models/silero_vad.onnx` — VAD（k2-fsa GitHub Release）
3. `models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` — 质检 + 翻译 LLM（HuggingFace）

也可手动下载任意 Qwen3 GGUF 量化版放入 `models/` 任意子目录（程序会递归搜索第一个 `.gguf`）。

### 3. 运行

```powershell
# 自动探测：旁边有 CUDA DLL 且驱动可用 → CUDA；否则 CPU
.\subtitle-assistant.exe "C:\path\to\video.mp4"

# 显式控制设备
.\subtitle-assistant.exe --device cuda .\video.mp4   # 强制 CUDA（探测失败直接报错，便于排查）
.\subtitle-assistant.exe --device cpu  .\video.mp4   # 强制 CPU

# CUDA 运行时 DLL 不放在 exe 旁边时
.\subtitle-assistant.exe --lib-dir D:\cuda-runtime\bin .\video.mp4

# 关闭逐句质检（ASR 与 LLM 不同时驻留，内存峰值最低）
.\subtitle-assistant.exe --no-qc .\video.mp4
```

也可以把多个视频**直接拖拽到 exe（或其快捷方式）图标**上按序批处理。

输出（与源文件同目录）：

| 文件 | 内容 |
|---|---|
| `video.srt` | 最终翻译字幕 |
| `video.raw.srt` | ASR 原始输出（质检前，便于对照） |
| `video.verified.srt` | 质检后的原文字幕（仅开启质检时） |

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device auto\|cpu\|cuda` | auto | 推理设备（依赖外置 DLL 探测） |
| `--lib-dir DIR` | - | 附加 DLL 搜索目录（可多次） |
| `--no-qc` | 关 | 关闭逐句 LLM 质检（回到旧版线性流程） |
| `--qc-context N` | 3 | 质检时携带的上文条数 |
| `--gpu-layers N` | -1(自动) | LLM GPU offload 层数；0=纯 CPU |
| `--ctx-size N` | 8192 | LLM 上下文长度 |
| `--batch-size N` | 20 | 每批翻译条数 |
| `--context-size N` | 4 | 翻译滑动窗口上文条数 |
| `--target-lang LANG` | 简体中文 | 翻译目标语言 |
| `--asr-threads N` | 4 | ASR CPU 线程数 |

依赖：系统 `PATH` 中需有 **FFmpeg / ffprobe**（音频解码用）。

---

## 🛠 从源码编译（开发者指南）

重构后编译大幅简化——**不再需要**：CRT 链接 hack（`RUSTFLAGS=-C target-feature=+crt-static`、
`CMAKE_MSVC_RUNTIME_LIBRARY` 等）、CUDA Toolkit、手动区分 cpu/cuda 两次编译。

### 环境要求

- Rust stable（MSVC toolchain，`x86_64-pc-windows-msvc`）
- Visual Studio Build Tools（C++ 工作负载）+ CMake（VS 自带或独立安装）
- LLVM/clang（提供 bindgen 所需的 `libclang.dll`，设置 `LIBCLANG_PATH` 指向其 bin 目录）
- FFmpeg（仅运行时需要）

### 编译

```powershell
cargo build --release
# 产物: target\release\subtitle-assistant.exe
# 构建脚本会自动：
#   - sherpa-onnx-sys: 下载 v1.13.8 win-x64 shared 预编译库并把 DLL 拷到 target\release
#   - llama-cpp-sys-2: CMake 构建 llama.cpp（dynamic-link + dynamic-backends），
#     llama.dll/ggml*.dll 硬链接到 target\release
```

本机直接运行 `target\release\subtitle-assistant.exe` 即为 **CPU 版**（ggml CPU 后端模块
通过构建期烧录的路径自动发现）。想要 CUDA：把 CI 发行包 CUDA12 里的 DLL 拷到 exe 旁即可，
无需重新编译；或者（不推荐）安装 CUDA Toolkit 后 `cargo build --release --features cuda-build`
自行编译 `ggml-cuda.dll`。

### 为什么动态链接能根治编译问题？

| 旧版（静态链接） | 新版（外置 DLL） |
|---|---|
| sherpa-onnx 预编译库用 `/MT`，llama.cpp CMake 构建用 `/MD`，混链必炸 `LNK2038/LNK2005`，需手工统一 CRT | exe 只链接各 DLL 的导入库，C/C++ 运行时被封在各自 DLL 内，冲突无从发生 |
| CUDA 版必须本机装 CUDA Toolkit 全量编译 | CUDA 由官方预编译 DLL 运行时提供，编译机零 CUDA 依赖 |
| cpu/cuda 要编两个 exe | 一个 exe，换 DLL 即换后端 |
| exe 体积巨大、启动加载慢 | exe ~2MB，按需加载 |

### 版本锁定说明（升级依赖前必读）

exe 与外置 DLL 的二进制兼容性依赖版本对齐，`Cargo.toml` 因此**精确锁定**：

- `sherpa-onnx = "=1.13.8"` ↔ 发行包 DLL 取自 sherpa-onnx **v1.13.8** Release；
- `llama-cpp-2 = "=0.1.157"`（vendored llama.cpp commit `26394b4e`）↔ 发行包 DLL 取自
  llama.cpp **b11153** Release（已验证两者公开头文件零差异，导入符号一致）。

升级任一依赖时，必须同步更新 `scripts/package_dist.py` 顶部的
`LLAMA_TAG` / `SHERPA_VER` / `CUDNN_VER`，并重新核对 ABI。

---

## 🤖 CI 自动构建（GitHub Actions）

`.github/workflows/build.yml`：

- **触发**：推送 `v*` tag（自动创建 GitHub Release 并上传发行包）、push master / PR（编译验证）、手动 dispatch。
- **构建**：`windows-latest` + Rust stable + LLVM(libclang)，`cargo build --release`，
  全程**无需 CUDA Toolkit**。
- **打包**：`scripts/package_dist.py` 下载官方预编译 DLL（llama.cpp b11153 win-cpu /
  win-cuda-12.4 / cudart 包、sherpa-onnx v1.13.8 CPU 与 CUDA 包、NVIDIA PyPI cuDNN 9.1.1
  wheel），组装 CPU 与 CUDA12 两个发行包，附 `SHA256SUMS.txt`。
- 下载物有 `actions/cache` 缓存，重复构建免重下 ~1.5 GB。
- **push master / PR 只编译+冒烟**（~10 分钟快速反馈），tag `v*` 或手动 dispatch 才组装发行包。

`.github/workflows/e2e-test.yml`（端到端真实测试）：

- **触发**：推送 `e2e-*` tag，或手动 dispatch（可指定任意视频/音频直链 URL、选择 quick/full 矩阵）。
- **多语言矩阵**：日/德/(法/中英混说)/噪音样本，来自 HF 镜像仓库 `test_wavs/`，随模型缓存。
- **实测记录**（windows-latest，CPU 推理）：五语言全部通过；噪音样本正确触发 QC 三态判决
  （2 句丢弃、1 句纠正幻觉前缀、1 句保留真实人声）；QC 提示词加 `/no_think` 后 JSON 首轮命中率 100%。
- YouTube（"Sign in to confirm you're not a bot"）与 bilibili（海外数据中心 IP 得 HTTP 412）均对
  GitHub runner 风控，故默认矩阵使用 HF 测试音频；dispatch 传入可访问的直链 URL 亦可测试真实视频。

发布新版本：

```bash
git tag v0.2.0 && git push origin v0.2.0
```

---

## 📁 项目结构

```
Qwen3-subtitle-assistant/
├── Cargo.toml                  # 依赖精确锁版本；dynamic-link/dynamic-backends/shared
├── .github/workflows/build.yml # CI：构建 exe + 组装 CPU/CUDA12 发行包 + Release
├── prompts/
│   ├── qc_segment.txt          # 逐句质检提示词（keep/fix/drop + 纠正）
│   ├── extract_context.txt     # 全局摘要与术语表提取
│   └── translate_batch.txt     # 滑动窗口批量翻译
├── scripts/
│   ├── package_dist.py         # CI 打包：下载官方 DLL、组装、压缩、校验和
│   ├── download_models.ps1     # 模型一键下载（支持 HF 镜像）
│   └── download_models.bat
└── src/
    ├── main.rs                 # 编排：质检模式(双模型共存) / 线性模式(--no-qc)
    ├── runtime.rs              # ★ 外置 DLL 探测与加载（LoadLibraryExW/dlopen、--lib-dir、UTF-8 控制台）
    ├── cli.rs                  # 命令行参数
    ├── config.rs               # 配置与 GGUF 自动搜寻
    ├── ffmpeg.rs               # FFmpeg 流式解码（stderr 排空防死锁）
    ├── asr.rs                  # VAD 切片 + Qwen3-ASR（provider 选择 + CUDA 失败回退 + 质检钩子）
    ├── qc.rs                   # ★ 逐句 LLM 质检（上文语境、fail-open 兜底、统计）
    ├── llm.rs                  # llama.cpp 绑定：可复用 Session（持久 KV cache）
    ├── translate.rs            # 全局摘要 + 滑动窗口翻译（重试与原文回退）
    ├── prompt.rs               # 提示词模板渲染
    ├── srt.rs                  # SRT 时间轴与输出
    └── types.rs                # 数据结构（字幕段/全局上下文/质检判决）
```

## 🩺 常见问题

- **启动报“由于找不到 llama.dll…”**：DLL 必须与 exe 同目录（或用 `--lib-dir` 指定）。
- **日志显示 LLM/ASR 回退 CPU**：说明 CUDA DLL 或其依赖缺失/驱动过旧，按启动日志中的
  warn 提示补齐对应 DLL；`--device cuda` 可强制探测并在失败时直接报错定位问题。
- **显存不足**：`--gpu-layers 0`（LLM 走 CPU，ASR 仍可 CUDA）或 `--no-qc` + `--ctx-size 4096`。
- **质检拖慢速度**：逐句质检每句增加一次 LLM 生成（GPU 上通常 <1s）；追求速度可 `--no-qc`。
- **ffmpeg/ffprobe 找不到**：安装 FFmpeg 并加入 `PATH`。

## 📄 License

[Unlicense](LICENSE)

## 🙏 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 跨平台语音识别运行时
- [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) — llama.cpp 安全 Rust 绑定
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ONNX Runtime](https://github.com/microsoft/onnxruntime) / [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) — 预编译运行时 DLL 来源
- Alibaba Qwen 团队 — Qwen3-ASR 与 Qwen3 开源模型
