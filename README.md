# Qwen3 Subtitle Assistant (Qwen3 字幕助手)

**完全本地离线**的 Rust 视频/音频字幕转录与翻译助手：媒体文件进，排版好的中文字幕 SRT 出。

组合 **Qwen3-ASR 0.6B**（语音识别）与 **Qwen3-1.7B**（逐句质检 + 全局摘要 + 滑动窗口翻译），
在消费级显卡或纯 CPU 上都能跑，数据绝不出域。

> **v0.3 变更要点**（相对 v0.2）
> 1. **长视频不再崩**：prompt 按 `n_batch` 分块 prefill（旧版超过 2048 token 会命中
>    llama.cpp 的 `GGML_ASSERT` 直接 abort）；转录过长时全局摘要自动走 **map-reduce 分块**。
> 2. **翻译不再静默回退原文**：紧凑输出格式（只回传 `i`/`t`，时间轴由 Rust 保留）、
>    索引对不上时按位置兜底、每条回退都 warn 并计数，收尾打印统计。
> 3. **提示词全部 `/no_think` + JSON 任务贪心解码**：v0.2 只给质检提示词关了思考模式，
>    摘要/翻译仍在思考，白白吃掉生成预算导致 JSON 截断。
> 4. **字幕排版**：按显示宽度折行（CJK 计 2）、超长 cue 按句读拆分并按字符占比分配时间。
>    旧版一个 VAD 语音段就是一条字幕，十几秒的连续讲话会变成一整屏文字。
> 5. **资源路径回退到 exe 目录**：拖拽/快捷方式启动时 CWD 不可控，旧版会直接报
>    “提示词目录不存在”。
> 6. **`--from-srt`**：跳过 ASR，直接翻译已有 SRT（重跑翻译、二次修正、CI 回归都靠它）。
> 7. **模型跨文件复用**：批处理多个文件时不再每个文件都重新加载/卸载模型。
> 8. **发行包自检**：打包时按 PE 导入表**按需补 cuFFT**，并在组装后做依赖闭包检查
>    （旧版 CUDA 包缺 `cufft64_11.dll`，ASR 的 CUDA EP 会静默回退 CPU）。
> 9. **CI 收敛为单个 `ci.yml`**：push 自动构建+测试，手动 dispatch 才打包发布；
>    e2e 换成 **bilibili 真实日语视频**（>1 分钟，缓存）+ 长文本回归 + 迁移目录回归，
>    断言从“产出了 .srt”升级为“字幕条数/时长/行宽/译文语言/回退条数”全部达标。

---

## ✨ 核心特性

- **🔒 完全离线**：识别、质检、翻译全部本地完成。
- **🧠 逐句质检**：每句 ASR 结果都经 LLM 结合上文判定 keep / fix / drop：
  背景音乐、掌声、ASR 幻觉（“字幕由…提供”之类）被自动剔除；同音字、断词错误被自动纠正。
  纠正文本还要过**可信度校验**（与原文的编辑距离相似度、长度比），不合格就保留原文，
  避免小模型把正确句子“纠正”成幻觉。
- **⚡ CUDA 加速即插即用**：CUDA 能力来自外置 DLL，`ggml-cuda.dll` 由 llama.cpp
  运行时自动扫描加载，`onnxruntime_providers_cuda.dll` 由 ONNX Runtime 按需加载；
  程序启动时主动探测并打印后端选择结果，缺件回退 CPU 并给出补齐提示。
- **📚 智能上下文翻译**：翻译前先提取剧情摘要与术语表；转录超过 `--summary-chunk-tokens`
  时自动分块提取再合并（map-reduce），长视频也能拿到全局上下文。翻译时注入
  全局摘要 + 滑动窗口上文，抑制术语漂移与上下文割裂。
- **🎞 流式音频管线**：FFmpeg 管道直出 16kHz `f32le` PCM + Silero VAD 切片，零中间文件。
- **🧾 字幕排版**：行宽折行 + 超长 cue 拆分（`--max-line-width` / `--max-cue-secs`）。
- **🔁 会话复用**：LLM 上下文只创建一次，每轮清 KV cache 复用，逐句质检的高频调用
  不必反复重建 8K 上下文。
- **🖱 极简交互**：多个媒体文件拖到 exe（或其快捷方式）上即按序处理；失败时窗口不会
  一闪而过（`--no-pause` 可关），也可 `--log-file` 落盘日志。

## ⚙️ 工作流

```
媒体文件 ──FFmpeg──▶ f32le PCM 流 ──Silero VAD──▶ 语音段
                                                      │
                        ┌─────────────────────────────┘
                        ▼
              Qwen3-ASR 转录（CPU / CUDA 外置 DLL）
                        │  每句
                        ▼
              LLM 逐句质检（带最近 N 句上文，贪心解码）
              ├─ drop：噪音/幻觉 → 丢弃（计入统计）
              ├─ fix ：识别错误 → 过相似度校验后纠正
              └─ keep：有效语音 → 保留
                        │
                        ▼   (.raw.srt / .verified.srt 中间产物)
              LLM 全局摘要 + 术语表（转录过长则分块 map-reduce）
                        │
                        ▼
              滑动窗口分批翻译（紧凑 JSON，失败按位置兜底并告警）
                        │
                        ▼
              排版（折行 + 长 cue 拆分）──▶ 最终 .srt
```

启用质检时（默认），ASR 与 LLM 在转录阶段**同时驻留**内存/显存（约 4~5 GB 显存可同时
跑 CUDA 版 ASR + Q8 LLM）。批处理多个文件时两者只加载一次、跨文件复用（内存峰值不变，
省掉每个文件几秒到几十秒的重复加载）。显存紧张可加 `--no-qc` 回到线性工作流
（先转录完、卸载 ASR、再加载 LLM 翻译，此时**每个文件**都会重新加载模型以维持最低峰值），
或 `--gpu-layers 0` 让 LLM 留在 CPU。

> 输出是**目标语言单语字幕**（默认简体中文）。原文另存为 `.raw.srt` / `.verified.srt`
> 两个中间文件，本程序不生成“原文+译文”同屏的双语字幕。

---

## 📦 快速开始（Release 版）

### 1. 选择发行包

| 发行包 | 适用场景 | 体积 |
|---|---|---|
| `Qwen3SubAssistant-<ver>-win-x64-cpu.zip` | 无 N 卡 / 不想装驱动依赖 | ~20 MB |
| `Qwen3SubAssistant-<ver>-win-x64-cuda12.zip` | NVIDIA 显卡加速（推荐） | ~1.9 GB |

CUDA 包要求：**NVIDIA 驱动 ≥ 551.61**（CUDA 12.4 运行时；更新的驱动向下兼容）。
两个包内的 `subtitle-assistant.exe` 完全相同，差别只在旁边放了哪套 DLL。

解压后目录结构：

```
Qwen3SubAssistant-win-x64-cuda12/
├── subtitle-assistant.exe            # 主程序（CPU/CUDA 通用）
├── llama.dll / llama-common.dll      # LLM 运行时（llama-common 用本仓库 CI 构建产物）
├── ggml.dll / ggml-base.dll / ggml-cpu-*.dll / libomp.dll
├── ggml-cuda.dll                     # ★ LLM CUDA 后端（运行时自动扫描加载）
├── cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll
├── cufft64_11.dll                    # ★ ASR CUDA EP 依赖（按需自动补齐）
├── cudnn64_9.dll / cudnn_*64_9.dll   # cuDNN 9（ASR CUDA EP 需要）
├── sherpa-onnx-c-api.dll / onnxruntime.dll / onnxruntime_providers_shared.dll
├── onnxruntime_providers_cuda.dll    # ★ ASR CUDA ExecutionProvider
├── vcruntime140.dll / msvcp140.dll   # VC++ 运行时（exe 是 /MD 构建）
├── prompts/                          # 提示词模板（质检/摘要/翻译）
├── scripts/download_models.bat|.ps1  # 模型一键下载
├── THIRD-PARTY-NOTICES.txt           # 第三方组件与许可
└── models/                           # 模型目录（下载后生成）
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

也可手动下载任意 Qwen3 GGUF 量化版放入 `models/` 任意子目录（程序会递归搜索第一个 `.gguf`；
有多个时会 warn 并提示用 `--llm-model` 指定）。

### 3. 运行

```powershell
# 自动探测：旁边有 CUDA DLL 且驱动可用 → CUDA；否则 CPU
.\subtitle-assistant.exe "C:\path\to\video.mp4"

# 显式控制设备
.\subtitle-assistant.exe --device cuda .\video.mp4   # 强制 CUDA（两个关键 DLL 都加载失败才报错）
.\subtitle-assistant.exe --device cpu  .\video.mp4   # 强制 CPU

# CUDA 运行时 DLL 不放在 exe 旁边时（仅对"运行时才加载"的 DLL 有效，见下方 FAQ）
.\subtitle-assistant.exe --lib-dir D:\cuda-runtime\bin .\video.mp4

# 关闭逐句质检（ASR 与 LLM 不同时驻留，内存峰值最低）
.\subtitle-assistant.exe --no-qc .\video.mp4

# 已有 SRT，只重跑翻译（跳过 ASR）
.\subtitle-assistant.exe --from-srt --output-dir out .\video.srt

# 专有名词识别错误多时，给 ASR 加偏置词
.\subtitle-assistant.exe --asr-hotwords "東京,山中伸弥,iPS細胞" .\video.mp4
```

也可以把多个视频**直接拖拽到 exe（或其快捷方式）图标**上按序批处理
（模型只加载一次）。**任何一个文件失败，进程退出码为 1**，并在控制台等待回车
（`--no-pause` 可关闭等待）。

输出（默认与源文件同目录，可用 `--output-dir` 改）：

| 文件 | 内容 |
|---|---|
| `video.srt` | 最终译文字幕（已折行/拆分） |
| `video.raw.srt` | ASR 原始输出（质检前，便于对照） |
| `video.verified.srt` | 质检后的原文字幕（仅开启质检时） |
| `video.translated.srt` | `--from-srt` 模式的输出（不覆盖输入） |

### 参数一览

| 参数 | 默认 | 说明 |
|---|---|---|
| **路径与模型** | | |
| `--asr-model-dir DIR` | `./models/sherpa-onnx-qwen3-asr-0.6B-int8` | ASR 模型目录 |
| `--vad-model FILE` | `./models/silero_vad.onnx` | Silero VAD 模型 |
| `--llm-model FILE` | 自动搜索 | Qwen3 GGUF 路径（不给则递归搜 `./models`，再搜 exe 目录的 `models`） |
| `--prompts-dir DIR` | `./prompts` | 提示词目录 |
| `--output-dir DIR` | 源文件同目录 | 输出目录 |
| **设备** | | |
| `--device auto\|cpu\|cuda` | auto | 推理设备（依赖外置 DLL 探测） |
| `--lib-dir DIR` | - | 附加 DLL 搜索目录（可多次） |
| `--gpu-layers N` | -1(自动) | LLM GPU offload 层数；0=纯 CPU |
| **LLM** | | |
| `--ctx-size N` | 8192 | LLM 上下文长度（小于 2048 会被抬到 2048 并 warn） |
| `--prefill-batch N` | 2048 | llama.cpp `n_batch`；长 prompt 会自动按它分块 |
| `--seed N` | 42 | 采样种子（固定值 → 结果可复现） |
| `--temperature F` | 0.3 | 翻译首轮采样温度；重试一律转贪心 |
| `--max-retries N` | 3 | 摘要/翻译单批最大重试次数 |
| **质检** | | |
| `--no-qc` | 关 | 关闭逐句 LLM 质检（回到线性流程） |
| `--qc-context N` | 3 | 质检携带的上文条数 |
| `--qc-retries N` | 2 | 质检单句最大尝试次数 |
| `--qc-max-tokens N` | 256 | 质检单句生成上限 |
| `--qc-min-similarity F` | 0.35 | fix 判决的最小可信相似度，低于则保留原文 |
| **翻译与排版** | | |
| `--batch-size N` | 20 | 每批翻译条数（prompt 超预算时自动对半拆） |
| `--context-size N` | 4 | 翻译滑动窗口上文条数 |
| `--summary-chunk-tokens N` | 3000 | 摘要单块 token 预算，超出即分块 map-reduce |
| `--translate-tokens-per-item N` | 96 | 每条译文的生成预算系数 |
| `--target-lang LANG` | 简体中文 | 翻译目标语言 |
| `--source-lang LANG` | 自动检测（视频原语言） | 源语言（提示词用） |
| `--max-line-width N` | 40 | 单行最大显示宽度（CJK 计 2；40≈20 汉字），0=不折行 |
| `--max-cue-secs F` | 7.0 | 单条字幕最长秒数，超出按句读拆分，0=不拆 |
| `--no-layout` | 关 | 关闭折行与拆分 |
| **ASR / VAD** | | |
| `--asr-threads N` | 4 | ASR CPU 线程数（仅 CPU provider 生效） |
| `--asr-hotwords LIST` | 空 | Qwen3-ASR 偏置词（英文逗号分隔） |
| `--asr-max-new-tokens N` | 256 | 单段最多生成 token（sherpa 默认 128 偏小） |
| `--asr-max-total-len N` | 1024 | 最大总序列长度（sherpa 默认 512） |
| `--vad-buffer-secs F` | 60 | VAD 缓冲区秒数（= 单段长度上限） |
| `--vad-min-silence F` | 0.5 | 判定语音结束所需的最短静音 |
| **运行方式** | | |
| `--from-srt` | 关 | 跳过 ASR，把输入当 SRT 直接翻译 |
| `--log-file FILE` | - | 日志同时写入文件 |
| `--no-pause` | 关 | 失败退出时不等待回车 |

依赖：系统 `PATH` 中需有 **FFmpeg / ffprobe**（音频解码用）。

---

## 🛠 从源码编译（开发者指南）

**不需要**：CRT 链接 hack（`RUSTFLAGS=-C target-feature=+crt-static`、
`CMAKE_MSVC_RUNTIME_LIBRARY` 等）、CUDA Toolkit、手动区分 cpu/cuda 两次编译。

### 环境要求

- Rust stable（MSVC toolchain，`x86_64-pc-windows-msvc`）
- Visual Studio Build Tools（C++ 工作负载）+ CMake
- LLVM/clang（提供 bindgen 所需的 `libclang.dll`，设置 `LIBCLANG_PATH` 指向其 bin 目录）
- FFmpeg（仅运行时需要）

### 编译与测试

```powershell
cargo build --release
# 产物: target\release\subtitle-assistant.exe
# 构建脚本会自动：
#   - sherpa-onnx-sys: 下载 v1.13.8 win-x64 shared 预编译库并把 DLL 拷到 target\release
#   - llama-cpp-sys-2: CMake 构建 llama.cpp（dynamic-link + dynamic-backends），
#     llama.dll/ggml*.dll 硬链接到 target\release

# 纯函数单元测试（时间轴/折行/长 cue 拆分、JSON 提取与修复、模板渲染、质检门槛、配置路径）
$env:PATH = "$PWD\target\release;$env:PATH"   # 测试二进制需要能找到原生 DLL
cargo test --release
```

本机直接运行 `target\release\subtitle-assistant.exe` 即为 **CPU 版**。想要 CUDA：
把 CI 发行包 CUDA12 里的 DLL 拷到 exe 旁即可，无需重新编译；或者（不推荐）安装 CUDA
Toolkit 后 `cargo build --release --features cuda-build` 自行编译 `ggml-cuda.dll`。

### 为什么动态链接能根治编译问题？

| 旧版（静态链接） | 现在（外置 DLL） |
|---|---|
| sherpa-onnx 预编译库用 `/MT`，llama.cpp CMake 构建用 `/MD`，混链必炸 `LNK2038/LNK2005` | exe 只链接各 DLL 的导入库，C/C++ 运行时被封在各自 DLL 内，冲突无从发生 |
| CUDA 版必须本机装 CUDA Toolkit 全量编译 | CUDA 由官方预编译 DLL 运行时提供，编译机零 CUDA 依赖 |
| cpu/cuda 要编两个 exe | 一个 exe，换 DLL 即换后端 |
| exe 体积巨大、启动加载慢 | exe ~2MB，按需加载 |

### 版本锁定说明（升级依赖前必读）

exe 与外置 DLL 的二进制兼容性依赖版本对齐，`Cargo.toml` 因此**精确锁定**：

- `sherpa-onnx = "=1.13.8"` ↔ 发行包 DLL 取自 sherpa-onnx **v1.13.8** Release；
- `llama-cpp-2 = "=0.1.157"`（vendored llama.cpp commit `26394b4e`）↔ 发行包 DLL 取自
  llama.cpp **b11153** Release（已核对公开头文件一致）。
  注意 `llama-common.dll` **必须**用本仓库 CI 构建产物，不能用官方 zip 里的版本。

升级任一依赖时，必须同步更新 `scripts/package_dist.py` 顶部的
`LLAMA_TAG` / `SHERPA_VER` / `CUDNN_VER` / `CUFFT_VER`，并跑一次：

```powershell
python scripts/check_dll_deps.py --dir <发行包目录> --fail-on-missing --strict-vcredist
```

`llama-cpp-2` 0.1.158+ 把 `str_to_token`/`is_eog_token`/`token_to_piece` 迁到了
`model.vocab()`，升级时需同步修改 `src/llm.rs`。

---

## 🤖 CI（`.github/workflows/ci.yml`）

只有一个工作流，触发规则：

| 触发 | 行为 |
|---|---|
| push `master` / PR | **build**（编译 + `cargo test` + 冒烟）→ **e2e**（真实视频 + 回归用例） |
| `workflow_dispatch` | 同上；勾选 `release` 时再跑 **release**（打包 CPU/CUDA12 + 创建 GitHub Release） |

**build**：`windows-latest` + Rust stable + LLVM(libclang)，`cargo build --release --locked`，
全程无需 CUDA Toolkit；构建产物（exe + DLL）作为 artifact 传给后续 job。
冒烟测试会打印 PE 导入表并逐个 `LoadLibrary` 验证。

**增量构建**：`target/` 入 rust-cache，llama.cpp 的 CMake 产物与全部依赖 crate 都能复用，
构建从 ~9.5 分钟降到分钟级；**唯独 release（dispatch 勾选 `release`）走全量构建**
（`cache-targets: false` + 显式清空 `target`），保证发布物干净可复现。

⚠️ 增量构建有个坑：rust-cache 的 `cleanTargetDir` 会把 `target/` 下**非 profile 目录**整个清空，
而 sherpa-onnx-sys 恰好把预编译库解到 `target/sherpa-onnx-prebuilt/`
（其 `build.rs`：`cache_root = <target>/sherpa-onnx-prebuilt`）。于是 fingerprint 恢复了、
库却被删掉 → `LNK1181: sherpa-onnx-c-api.lib`（这正是旧版被迫 `cache-targets: false` 的原因）。
本工作流把该目录单独用 `actions/cache` 存一份（key `sherpa-prebuilt-1.13.8-win-x64-shared`），
构建前还原回 `target/`；升级 sherpa-onnx 版本时记得同步这个 key。

**e2e**（`needs: build`，复用 artifact，不重复编译）：

| 用例 | 内容 | 断言 |
|---|---|---|
| T2a 长 prompt 回归 | `tests/fixtures/long_lines_ja.srt`（50 条 20 秒长 cue、约 7000 字）配 `--ctx-size 12288 --summary-chunk-tokens 10000`，强制整篇一次性喂给摘要 → prompt ~5000 token 远超 `n_batch`(2048) | 退出码 0（**v0.2 在此必然 `GGML_ASSERT` abort**）、日志出现「分块 prefill」、≥80 条、最长 cue ≤9s、最宽行 ≤44、回退原文 ≤5 条 |
| T2b 摘要分块回归 | `long_ja.srt` 前 40 条配 `--summary-chunk-tokens 400`，强制全局摘要走 map-reduce | 退出码 0、≥40 条、日志「摘要分块 N」且 N ≥ 2 |
| T3 迁移目录回归 | exe+DLL+prompts 拷到 `%TEMP%`，`models` 用 junction，从**非仓库 CWD** 启动 | 退出码 0，且日志出现「改用 exe 目录」（证明资源路径回退生效） |
| T1 真实视频 | bilibili 日语视频（>1 分钟，`scripts/fetch_media.py` 下载 + actions/cache 缓存）跑完整流程：ASR → 逐句质检 → 摘要 → 翻译 → 排版 | 字幕条数、最长 cue ≤15s、最宽行 ≤44、假名占比 ≤3%（确认真翻成中文）、回退原文 ≤5 条、质检计数自洽、日志无 `GGML_ASSERT`/panic |

三个回归用例都不依赖网络，排在真实视频用例之前跑，最快拿到关键反馈。
模型缓存用 `actions/cache/restore` + `save(if: always())` 而非 `actions/cache`——后者的 post 步骤在 job 失败时会被跳过，2.7GB 模型会每轮重下。

关于 bilibili 与数据中心 IP（实测结论，2026-09）：GitHub runner 的 IP 会被 bilibili 判 412，
但**风控是针对 UA 的，且两套接口的要求正好相反**：

| 目标 | 浏览器 UA（含 curl_cffi 的 chrome/safari/edge TLS 指纹模拟） | `curl/8.0` UA |
|---|---|---|
| `api.bilibili.com` / `www.bilibili.com` 内容接口 | **412** | **200 放行** |
| `upos-*.akamaized.net` / `*.bilivideo.com` CDN | **206 正常** | **403 Access Denied** |

所以 `scripts/fetch_media.py` 用两套 UA：接口侧（首页引导 cookie → view 取 cid/时长 →
playurl 取音频流）走 `curl/8.0`，音频下载走浏览器 UA + Referer + Origin；
音频轨取码率最高的一路（128kbps，3 分钟视频约 3MB；最低档是 31kbps HE-AAC，对 ASR 不利）。
下载成功后进 `actions/cache`（key `e2e-media-bilibili-v2`），**CI 只需成功一次**，之后不再访问 bilibili。

风控若再变，兜底路径按优先级：
1. dispatch 时填 `video_url`（任意可直链下载的媒体 URL）；
2. 在仓库 Settings → Secrets and variables → Variables 里设 `E2E_MEDIA_URL`，之后每次运行自动使用；
3. 本地跑 `python scripts/fetch_media.py --bvid BV…`（住宅 IP 无风控）拿到文件后自行托管。

全部失败时默认**跳过 T1** 并打 warning（T2a/T2b/T3 照跑），dispatch 勾选 `strict_media`
可改为直接判失败。换测试视频只需改 workflow 顶部的 `E2E_BVIDS`。

**release**（`needs: [build, e2e]`，仅手动 dispatch 且勾选 `release`）：
`scripts/package_dist.py` 下载官方预编译 DLL 组装 CPU / CUDA12 两个发行包，
组装时**按 `onnxruntime_providers_cuda.dll` 的真实导入表决定是否补 cuFFT**，
组装后跑依赖闭包检查（`--strict-deps`，缺件直接失败），最后打印包内完整文件清单、
生成 `SHA256SUMS.txt` 并创建 Release（tag 由 `version` 输入或 Cargo.toml 版本决定）。

发布新版本：

```
Actions → CI → Run workflow → 勾选 release（version 留空则用 Cargo.toml 里的版本）
```

---

## 📁 项目结构

```
Qwen3-subtitle-assistant/
├── Cargo.toml                  # 依赖精确锁版本；dynamic-link/dynamic-backends/shared
├── .github/workflows/ci.yml    # 唯一 CI：build → e2e →（手动）release
├── prompts/
│   ├── qc_segment.txt          # 逐句质检（keep/fix/drop，/no_think）
│   ├── extract_context.txt     # 全局摘要与术语表提取（/no_think）
│   └── translate_batch.txt     # 滑动窗口批量翻译（紧凑 i/t 输出，/no_think）
├── scripts/
│   ├── package_dist.py         # 打包：下载官方 DLL、按需补 cuFFT、自检、压缩、校验和
│   ├── check_dll_deps.py       # PE 导入闭包检查（静态，无需 GPU/驱动）
│   ├── fetch_media.py          # e2e 媒体获取（bilibili 抗风控 + yt-dlp 兜底）
│   ├── check_e2e.py            # e2e 结果断言（字幕内容级，而非“文件存在”）
│   ├── download_models.ps1     # 模型一键下载（支持 HF 镜像）
│   └── download_models.bat
├── tests/fixtures/
│   └── long_ja.srt             # 长文本回归夹具（125 条日语，自撰文本，无版权问题）
└── src/
    ├── main.rs                 # 编排：质检模式 / 线性模式(--no-qc) / --from-srt；模型跨文件复用
    ├── runtime.rs              # 外置 DLL 探测与加载、exe 目录、UTF-8 控制台、失败暂停
    ├── cli.rs                  # 命令行参数
    ├── config.rs               # 配置、资源路径回退、GGUF 自动搜寻、输出路径
    ├── ffmpeg.rs               # FFmpeg 流式解码（-nostdin、stderr 尾部保留、异常杀进程）
    ├── asr.rs                  # VAD 切片 + Qwen3-ASR（provider 选择、CUDA 回退、hotwords、质检钩子）
    ├── qc.rs                   # 逐句 LLM 质检（贪心、格式提示重试、fix 可信度校验、fail-open）
    ├── llm.rs                  # llama.cpp 绑定：分块 prefill、token 计数、JSON 提取/修复
    ├── translate.rs            # 摘要 map-reduce + 滑动窗口翻译（自适应拆批、位置兜底、统计）
    ├── prompt.rs               # 模板单遍渲染 + 双向占位符校验 + 缓存
    ├── srt.rs                  # SRT 读写、折行、长 cue 拆分、显示宽度
    └── types.rs                # 数据结构（字幕段/全局上下文/质检判决/翻译统计）
```

## 🩺 常见问题

- **启动报“由于找不到 llama.dll…”**：`llama.dll`、`ggml*.dll`、`llama-common.dll`、
  `sherpa-onnx-c-api.dll` 是 exe 的**静态导入**，Windows 加载器在 `main()` 之前就解析它们，
  所以**必须与 exe 同目录**（或在系统 `PATH` 中）——`--lib-dir` 对此无效。
  `--lib-dir` 只对运行时才加载的 DLL 有用：`ggml-cuda.dll`、
  `onnxruntime_providers_cuda.dll` 以及它们的依赖（cudart/cublas/cublasLt/cufft/cudnn）。
- **日志显示 LLM/ASR 回退 CPU**：说明 CUDA DLL 或其依赖缺失/驱动过旧。启动日志的 warn
  会列出该子系统需要的 DLL 清单；`--device cuda` 可在两个关键 DLL 都加载失败时直接报错定位。
  也可以用 `python scripts/check_dll_deps.py --dir <发行包目录>` 静态查缺件。
- **显存不足**：`--gpu-layers 0`（LLM 走 CPU，ASR 仍可 CUDA）、`--no-qc`、
  或 `--ctx-size 4096`（KV cache 与 ctx 成正比）。
- **质检拖慢速度**：逐句质检每句增加一次 LLM 生成（GPU 上通常 <1s，CPU 上可能 5~15s）；
  追求速度可 `--no-qc`。
- **译文里出现原文/整批没翻**：看日志里的“回退原文”“整批翻译失败”计数，
  收尾统计行 `翻译 N 条 / M 批：重试 …，位置兜底 …，回退原文 …` 会汇总。
  常见原因是 `--batch-size` 太大导致 prompt/输出超预算，调小即可（程序也会自动对半拆）。
- **字幕一行太长 / 一条太久**：调 `--max-line-width`（默认 40，即约 20 个汉字）与
  `--max-cue-secs`（默认 7 秒）；不想要排版用 `--no-layout`。
- **ffmpeg/ffprobe 找不到**：安装 FFmpeg 并加入 `PATH`。解码失败时错误信息里会带上
  ffmpeg stderr 的最后若干行（`RUST_LOG=debug` 可看完整输出）。
- **拖拽运行看不到日志**：给 exe 建快捷方式并在参数里加 `--log-file "%USERPROFILE%\Desktop\qsa.log"`，
  或直接从终端运行。

## 📄 License

代码：[Unlicense](LICENSE)。发行包内含的第三方二进制组件许可见包内
`THIRD-PARTY-NOTICES.txt`（llama.cpp MIT、ONNX Runtime MIT、sherpa-onnx Apache-2.0、
NVIDIA cuDNN/CUDA 运行时按各自 SLA 再分发）。模型不包含在发行包内，需自行下载
（Qwen3 系列为 Apache-2.0，Silero VAD 为 MIT）。

## 🙏 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 跨平台语音识别运行时
- [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) — llama.cpp 安全 Rust 绑定
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ONNX Runtime](https://github.com/microsoft/onnxruntime) / [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) — 预编译运行时 DLL 来源
- Alibaba Qwen 团队 — Qwen3-ASR 与 Qwen3 开源模型
