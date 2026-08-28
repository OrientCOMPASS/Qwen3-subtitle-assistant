# Qwen3 Subtitle Assistant (Qwen3 字幕助手)

**Qwen3 Subtitle Assistant** 是一个完全本地离线运行的、基于 Rust 构建的高效视频/音频字幕转录与翻译助手。

本项目专为隐私敏感和离线环境设计，无需联网即可完成从“媒体文件”到“精准双语 SRT 字幕”的全流程处理。它巧妙地结合了轻量级语音识别模型（Qwen3-ASR 0.6B）与小型语言模型（Qwen3-1.7B），并通过极致的内存管理（线性生命周期）与滑动窗口翻译策略，在普通消费级显卡甚至纯 CPU 环境下也能流畅运行。

## ✨ 核心特性

- **🔒 完全离线与隐私安全**：所有语音识别与文本翻译均在本地完成，数据绝不出域。
- **🚀 极致内存管理**：采用严格的线性工作流（加载 ASR -> 转录 -> 卸载 ASR -> 加载 LLM -> 翻译 -> 卸载 LLM），确保大模型不会同时驻留内存，极大降低了显存/内存峰值。
- **🧠 智能上下文翻译**：
  - **全局摘要提取**：在翻译前，LLM 会先阅读全片转录文本，提取核心剧情与专业术语表。
  - **滑动窗口翻译**：翻译时注入全局摘要与上文语境，彻底解决长视频翻译中的“术语不一致”与“上下文割裂”问题。
  - **自动重试机制**：针对小模型偶尔的 JSON 格式崩坏，内置 5 次自动重试机制，确保字幕生成的稳定性。
- **⚡ 流式音频处理**：通过 FFmpeg 管道直接输出 `f32le` 裸音频流，结合 Silero VAD 进行精准语音切片，零中间文件 IO 损耗。
- **🖱️ 极简交互**：支持直接拖拽多个媒体文件到程序图标上，按顺序自动处理并输出同名 SRT 文件。

## ⚙️ 工作流原理

1. **音频解码**：调用本地 FFmpeg，将输入媒体流式解码为 16kHz 单声道 `f32le` PCM 数据。
2. **语音切片 (VAD)**：使用 Silero VAD 模型精准识别语音边界，剔除静音，切分出独立的语音段。
3. **语音转录 (ASR)**：将语音段喂给 `sherpa-onnx` 运行的 Qwen3-ASR 0.6B 模型，生成带精准时间戳的原始字幕。
4. **模型切换**：显式卸载 ASR 模型，释放内存。
5. **全局理解**：加载 Qwen3-1.7B-GGUF 翻译模型，将全片原文喂给模型，提取剧情摘要与核心术语表。
6. **分批翻译**：将字幕按批次（如 20 条/批）结合全局摘要和滑动窗口上下文发送给 LLM 进行翻译，并严格输出 JSON 格式。
7. **排版输出**：Rust 解析 JSON，自动调整阅读速度，生成标准 `.srt` 文件至源文件同目录。

---

## 📦 快速开始 (使用 Release 版本)

### 1. 准备工作
下载 Release 压缩包并解压。目录结构应如下所示：
```text
Qwen3-subtitle-assistant/
├── prompts/                 # 提示词模板
├── models/                  # 模型存放目录
│   ├── sherpa-onnx-qwen3-asr-0.6B-int8/  # ASR 模型 (Release包已内置)
│   └── silero_vad.onnx      # VAD 模型 (Release包已内置)
├── subtitle-assistant-cpu.exe
├── subtitle-assistant-cuda.exe
├── cublas64_13.dll          # CUDA 运行库 (仅 CUDA 版需要)
└── cublasLt64_13.dll        # CUDA 运行库 (仅 CUDA 版需要)
```

### 2. 下载翻译模型 (必须)
Release 包为了控制体积，**未包含 LLM 翻译模型**。你需要自行下载：
1. 访问 HuggingFace: [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)
2. 下载任意一个量化版本（推荐 `Qwen3-1.7B-Q8_0.gguf`）。
3. 将下载的模型放入 `models/` 目录下的**任意子文件夹**中（程序会自动递归搜索 `.gguf` 文件）。

### 3. 使用方式
**方式 A：命令行运行**
打开终端，直接传入视频/音频文件路径：
```bash
# 如果你有 NVIDIA 显卡，强烈建议使用 CUDA 版本
.\subtitle-assistant-cuda.exe "C:\path\to\your\video.mp4"

# 纯 CPU 运行
.\subtitle-assistant-cpu.exe "C:\path\to\your\video.mp4"
```

**方式 B：拖拽文件 (推荐)**
1. 为 `subtitle-assistant-cuda.exe` (或 cpu 版) 创建一个**快捷方式**，放到桌面或任意方便的地方。
2. 选中一个或多个视频/音频文件，**直接拖拽**到该快捷方式图标上。
3. 程序会自动启动，依次处理所有文件，并在视频同目录下生成 `.srt` 和 `.raw.srt` 文件。

---

## 🛠️ 从源码编译 (开发者指南)

如果你希望修改代码或自行编译，请遵循以下指南。

### 环境要求
- **Rust** (推荐使用 `rustup` 安装最新稳定版)
- **FFmpeg** (必须添加到系统环境变量 `PATH` 中)
- **CMake** & **C++ 编译工具链** (如 Visual Studio Build Tools)
- **CUDA Toolkit** (仅编译 CUDA 版本需要，推荐 11.8 或 12.x/13.x)

### ⚠️ 编译必看：Windows 下的 CRT 链接冲突问题
在 Windows 下同时编译 `sherpa-onnx` 和 `llama-cpp` 时，极易遇到 `LNK2038` 和 `LNK2005` 错误。
**原因**：`sherpa-onnx-sys` 的预编译库使用的是**静态 CRT (`/MT`)**，而 `llama-cpp-sys-2` 默认使用 CMake 构建时采用的是**动态 CRT (`/MD`)**。Rust 链接器无法在同一个可执行文件中混合链接这两种 C 运行时库。

**解决方案**：在编译前，**必须**在终端中设置以下环境变量，强制统一使用静态链接 (`/MT`)。

#### 编译 CPU 版本
```cmd
:: 1. 强制 Rust 和底层 C/C++ 编译器使用静态链接 (/MT)
set RUSTFLAGS=-C target-feature=+crt-static

:: 2. 强制 CMake 构建 llama.cpp 时使用静态链接 (/MT)
set CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
set CMAKE_POLICY_DEFAULT_CMP0091=NEW

:: 3. 编译
cargo build --release
```

#### 编译 CUDA 版本 (NVIDIA GPU 加速)
```cmd
:: 1. 设置 CUDA 安装路径 (根据你的实际安装路径修改)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3

:: 2. 强制 Rust 和底层 C/C++ 编译器使用静态链接 (/MT)
set RUSTFLAGS=-C target-feature=+crt-static

:: 3. 强制 CMake 构建 llama.cpp 时使用静态链接 (/MT)
set CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
set CMAKE_POLICY_DEFAULT_CMP0091=NEW

:: 4. 清理旧的错误缓存 (非常重要！防止 CMake 缓存干扰)
cargo clean

:: 5. 启用 cuda feature 进行编译
cargo build --release --features "llama-cpp-2/cuda"
```

编译成功后，可执行文件将位于 `target/release/subtitle-assistant.exe`。

---

## 📁 项目结构

```text
Qwen3-subtitle-assistant/
├── Cargo.toml               # Rust 依赖管理
├── prompts/                 # LLM 提示词模板 (摘要提取与批量翻译)
├── src/
│   ├── main.rs              # 入口：调度线性工作流与生命周期管理
│   ├── cli.rs               # 命令行参数解析 (clap)
│   ├── config.rs            # 配置加载与 GGUF 模型自动搜寻
│   ├── types.rs             # 核心数据结构 (字幕片段、全局上下文)
│   ├── ffmpeg.rs            # FFmpeg 进程调用与 f32le 流式解码
│   ├── asr.rs               # Silero VAD 切片与 sherpa-onnx ASR 转录
│   ├── llm.rs               # llama-cpp-2 绑定、模型加载与自回归生成
│   ├── translate.rs         # 翻译编排 (全局摘要、滑动窗口、重试机制)
│   ├── prompt.rs            # 提示词模板渲染
│   └── srt.rs               # SRT 时间轴计算与文件生成
```

## 📄 License

本项目采用 [Unlicense](LICENSE) 开源。

## 🙏 致谢

- 感谢 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 提供极其易用的跨平台语音识别绑定。
- 感谢 [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) (llama-cpp-2) 提供安全的 Rust llama.cpp 绑定。
- 感谢 [WhatdidIsay](https://github.com/OrientCOMPASS/WhatdidIsay) 项目提供的 sherpa-onnx API 使用参考。
- 感谢 Alibaba Qwen 团队提供优秀的开源模型 (Qwen3-ASR & Qwen3-1.7B)。