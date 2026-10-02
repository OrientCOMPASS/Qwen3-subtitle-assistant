# Qwen3 Subtitle Assistant

**完全本地离线**的视频/音频字幕工具：把媒体文件交给它，得到排版好的 **SRT 字幕**。

核心能力由一个微调模型完成：**S2TT 版 Qwen3-ASR-1.7B**（LoRA 定向「听日语、写中文」）
在 llama.cpp 上直接输出目标语言字幕——转录和翻译一步到位，没有 LLM 后处理环节。

| 特性 | 说明 |
|---|---|
| 🔒 完全离线 | 模型随包下载，推理不出本机，无需任何 API key |
| 📦 单一可执行文件 | llama.cpp 静态链接进二进制，无 DLL/so 伴生（Windows/Linux 约 35~50MB，macOS 约 8MB） |
| ⚡ GPU 加速 | Windows/Linux 内嵌 **Vulkan**（NVIDIA/AMD/Intel 自动发现）；macOS 内嵌 **Metal**；无 GPU 自动回退 CPU |
| 🌊 流式管线 | VAD 每切出一句立即送 ASR（两线程并行 + 字节计量滞回缓冲），第一句字幕不用等全片扫完 |
| 🧹 终端干净 | 默认只显示进度条、逐条字幕和结果；全部诊断明细进 `--log-file` 或 `--verbose` |
| 🎯 任务开关 | 默认直出中文；`--context ""` 转写源语言（同一套权重） |

---

## 📥 下载与安装

从 [Releases](../../releases/latest) 下载 **1 个二进制 + 2 个模型文件**：

| 你的系统 | 下载资产 | 运行要求 |
|---|---|---|
| Windows x64 | `Qwen3-subtitle-assistant-windows-x64.exe` | GPU 驱动（自带 vulkan-1.dll）+ [VC++ 2015-2022 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)；CPU 需 AVX2（2013 年后的 x64 均可） |
| Linux x64 | `Qwen3-subtitle-assistant-linux-x64` | glibc ≥ 2.39（Ubuntu 24.04+ 等）；libvulkan.so.1（桌面发行版标配） |
| Linux arm64 | `Qwen3-subtitle-assistant-linux-arm64` | glibc ≥ 2.39；纯 CPU（armv8-a，树莓派 4+） |
| macOS (Apple Silicon) | `Qwen3-subtitle-assistant-macos-arm64` | 无额外依赖，Metal GPU 自动启用 |

模型（所有平台通用）：`s2tt-Q4_K_M.gguf`（约 1.1GB）+ `mmproj-s2tt-q8.gguf`（约 0.36GB）。

另需系统 PATH 里有 **ffmpeg / ffprobe**（音视频解码；Windows 可用 `winget install ffmpeg`，
macOS `brew install ffmpeg`，Linux 发行版包管理器同名安装）。

摆放成下面任一结构即可（**推荐图一**；两种都会自动发现，旧的 `models/qwen3-asr-s2tt/`
布局也兼容）：

```
任意目录/                                任意目录/
├── Qwen3-subtitle-assistant-…(.exe)    ├── Qwen3-subtitle-assistant-…(.exe)
└── models/                             ├── s2tt-Q4_K_M.gguf        ← 与 exe 同级也行
    ├── s2tt-Q4_K_M.gguf                └── mmproj-s2tt-q8.gguf
    └── mmproj-s2tt-q8.gguf
```

文件名不必改：目录内 `mmproj` 前缀的 `.gguf` 识别为音频编码器，其余 `.gguf` 识别为
语言模型。

> Windows/Linux 版内嵌 Vulkan 后端，启动依赖系统 Vulkan loader（GPU 驱动/桌面
> 发行版必有）。完全无驱动的 VM/精简容器请自行源码构建纯 CPU 版（见「开发者」）。

## 🚀 快速开始

```bash
# 方式一：把视频/音频文件拖到可执行文件图标上（Windows/macOS）

# 方式二：命令行（支持多文件批处理）
./Qwen3-subtitle-assistant-linux-x64 视频.mp4 访谈.m4a

# 输出：每个输入同目录下的 视频.srt / 访谈.srt（简体中文、已折行、长句已拆分）
```

运行时终端所见（默认级别）：

```
模式: S2TT 单模型直出｜context="translate to Chinese"｜设备: auto｜…
选定推理设备: [0] NVIDIA GeForce RTX 4060（权重将全量入显存…）
[1/1] 处理: "视频.mp4"
⠹ [00:02:31] [███████████▌----------] 316s/742s · 已转写 57 段 · 缓冲 12.4/50MB
[ASR✔] 00:00:04,672 (1.0s) 对大家来说。
[ASR✔] 00:00:06,400 (2.2s) 我最喜欢的口技，是中国的口技。
…（逐条实时打印识别出的字幕）
转录完成：427 段语音 → 380 条字幕（47 段空输出跳过，3 段重复丢弃）
已写出字幕: "视频.srt"（392 条）
全部 1 个文件处理成功。
```

常用变体：

```bash
Qwen3-subtitle-assistant… --context "" 视频.mp4        # 转写源语言（日语原文字幕）
Qwen3-subtitle-assistant… --greedy 视频.mp4            # 确定性贪心解码（完全可复现）
Qwen3-subtitle-assistant… -o subs -l run.log 视频.mp4  # 输出到 subs/，全量诊断进 run.log
Qwen3-subtitle-assistant… --gguf-selftest models/s2tt-Q4_K_M.gguf models/mmproj-s2tt-q8.gguf
                                                        # 环境自检：设备/加载/内存，不处理媒体
```

## ⚙️ 工作原理

```
媒体文件
   │  ffmpeg 子进程（-vn -ac 1 -ar 16000 -f f32le → stdout 管道）
   ▼
16kHz f32 单声道 PCM 流
   │  生产者线程：纯 Rust silero v4 VAD（权重内嵌 exe，0.62MB）
   │  逐帧(32ms)算语音概率，状态机流式切段：进入/退出阈值、min_silence 断句、
   │  合并窗、±0.1s pad、超长硬拆（--vad-max-seg-secs）
   ▼  每确认一段立即压入
滞回缓冲队列（--buffer-mb，默认 50MB，按波形字节计量；占用实时显示在进度条。
   │        占用满 → VAD 暂停生产；消费到半容量 → 恢复。内存上界与段长无关）
   ▼  消费者线程（主线程）逐段取出
S2TT Qwen3-ASR（llama.cpp + mtmd 音频编码，静态链接）
   │  chat 模板 system="translate to Chinese"（或空=转写），贪心或温度采样
   │  （默认 temp=0.7/top_p=0.8/top_k=20，基座官方推荐值；--greedy 回到贪心）
   │  原始输出 `language X<asr_text>正文` → 解析出语言与正文
   ▼
幻觉/丢弃规则（顺序判定，均有单测；①-③ 恒开，实例见 README 末尾）
   │  ① 空文本（模型判定纯静音/噪音）→ 跳过（无内容可写）
   │  ② 段内复读压缩：连续重复 ≥3 遍的子串只保留 2 遍
   │     （"啊！"×56 → "啊！啊！"；"让我抱抱你！呜呜呜，"×4 → ×2）
   │  ③ 与上一条保留字幕相同（压缩+去标点后比较）→ 丢弃（跨段复读循环）
   │  ④ 上一条与本次都是纯语气词（哎啊嗯哼呀哇哦噢…闭集）→ 丢弃本次
   │  ⑤ lang=None 且 <2s 或 <4 字的碎片 → 仅 --filter-fragments 时丢弃
   ▼
排版（CJK 显示宽度折行 --max-line-width；长 cue 按句读拆分 --max-cue-secs）
   ▼
<视频名>.srt        （--raw-srt 时另存排版前对照 .raw.srt）
```

**推理设备选择**（`--device auto` 默认）：`--cuda-libs` 指定的 CUDA（仅动态链接
构建）→ Vulkan/Metal 中索引最大的 GPU（多显卡机器上通常是主力独显）→ CPU。
有 GPU 时权重全量上卡，主机内存副本由 ggml 自动释放。

**终端输出设计**：应用日志双通道——终端默认 info 级（模式行、设备行、逐条字幕
`[ASR✔]`、每文件总结）；llama.cpp/ggml/mtmd 的原生 C 日志经 `llama_log_set`/
`ggml_log_set`/`mtmd_helper_log_set` 桥接进同一门面降为 debug；跳过/丢弃明细也是
debug。`--log-file` 的文件通道**恒为 debug 全量**（终端干净的同时排障有料），
`--verbose` 或 `RUST_LOG=debug` 把 debug 也放开到终端。

**VAD 实现**：silero v4 的纯 Rust 移植（无 onnxruntime），与 ORT 参照实现位精确
对拍（375 个中间层误差 <3.2e-6，黄金向量回放测试在 CI 常跑）；流式段切分状态机
与旧批量实现有 600 组随机序列的等价性对拍测试。

## 🎛 参数总览

短参数：`-m` model `-c` context `-d` device `-t` threads `-T` temperature
`-g` greedy `-b` buffer-mb `-o` output-dir `-l` log-file `-v` verbose
`-f` filter-fragments（`-h` 帮助 / `-V` 版本）。完整表：`--help`。

| 分组 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 模型 | `--model-dir DIR` | 自动探测 | 探测序：exe目录/models → exe目录 → ./models → 旧版布局；alias: `--asr-model-dir` |
| | `--model` / `--mmproj` | 目录扫描 | 显式指定 LM / 音频编码器 GGUF |
| | `--context STR` | `translate to Chinese` | 任务开关；`""` = 转写源语言 |
| 设备 | `--device` | `auto` | `auto` / `cpu` / `cuda`（cuda 需 `--cuda-libs` 且动态链接构建） |
| | `--threads N` | 4 | CPU 推理线程 |
| | `--gpu-layers N` | -1（自动全量） | 上卡层数 |
| | `--cuda-libs DIR` | 无 | 外置 CUDA 后端目录（仅动态链接构建生效） |
| 采样 | `--temperature F` | 0.7 | 基座 Qwen3-1.7B 官方推荐；`--greedy` 时忽略 |
| | `--top-p F` / `--top-k N` | 0.8 / 20 | 同上 |
| | `--seed N` | 42 | 固定=可复现；0=每次随机 |
| | `--greedy` | 关 | 强制贪心（v0.5 及以前行为） |
| VAD | `--vad-threshold F` | 0.5 | 语音概率阈值 |
| | `--vad-min-silence F` | 0.5s | 断句静音阈值 |
| | `--vad-max-seg-secs F` | 60s | 单段上限（超长硬拆）；alias: `--vad-buffer-secs` |
| 管线 | `--buffer-mb N` | 50 | VAD→ASR 滞回缓冲容量（MB，满→停、半→续） |
| | `--max-new-tokens N` | 256 | 单段生成上限；alias: `--asr-max-new-tokens` |
| 输出 | `--output-dir DIR` | 输入同目录 | SRT 输出目录 |
| | `--raw-srt` | 关 | 另存排版前 `.raw.srt` 对照 |
| | `--max-line-width N` / `--max-cue-secs F` | 44 / 15 | 排版：折行宽度 / 长 cue 拆分阈值 |
| 日志 | `--log-file PATH` | 无 | 全量诊断落盘（终端不受影响） |
| | `--verbose` | 关 | 终端放开 debug 明细 |
| | `--no-pause` | 关 | 失败时不等待回车（拖拽场景默认等待） |
| 过滤 | `--filter-fragments` | 关 | 启用 lang=None 短碎片启发式丢弃 |
| 自检 | `--gguf-selftest LM MMPROJ` | — | 加载模型打印设备/能力/内存后退出 |

## 🧪 质量与边界（如实说明）

* **语言定向可靠**：真实日语视频直出中文的假名占比 0.0%（CI 每轮断言 ≤5%）；
  静音/噪音大多输出空文本被自然跳过；
* **采样 vs 贪心**：默认温度采样能显著减少长静音/重复背景音处的复读与幻觉循环；
  漏网的循环由三道后处理兜底——段内复读压缩（重复子串 ≥3 遍只留 2 遍）、跨段
  判重（压缩+去标点后与上一条相同即丢）、连续语气词丢弃（"啊！"×56 这类整段
  情绪幻觉收敛为一短条或直接丢弃）。需要逐位可复现时用 `--greedy`（固定 seed
  下同样可复现）；
* **残留差距**（对比旧「ASR+LLM 精翻」两段式）：世界知识纠错（口误同音词）与
  个别专名漂移——单模型忠实直译，无 LLM 兜底；含糊音频两者同样无能为力；
* 当前模型用 120 对小样本微调；数据放量（`finetune/` 产线）会进一步收窄差距。
  质量证据链全部在 CI 可复现（`finetune/README.md` §10/§11）。

## 🧯 开发者

### 构建与测试

```bash
cargo build --release    # 产物 target/release/Qwen3-subtitle-assistant[.exe]
cargo test --release     # 36 个单测：VAD 黄金向量/流式等价性/滞回队列/丢弃规则/排版/SRT/目录发现
```

依赖：Rust stable + CMake + LLVM(libclang，bindgen 用)。**无需** CUDA/Vulkan SDK
（除非启用对应 feature，见下表）。llama.cpp 由 `llama-cpp-sys-2`（版本精确锁定）
从内嵌源码构建。

### 构建形态（cargo features）

| 形态 | 命令 | 产物 |
|---|---|---|
| CPU 单文件 | `cargo build --release` | 任何机器可跑（VM/容器友好） |
| Vulkan 单文件（Win/Linux 发行形态） | `--features vulkan` | 需 Vulkan SDK 编译（Win：LunarG SDK+`VULKAN_SDK`；Linux：`apt install libvulkan-dev glslc spirv-headers`）；运行需系统 loader |
| Linux 全静态伴生库 | `--features static-libs` | libstdc++/libgomp 静态，仅剩 glibc 依赖 |
| CUDA 单文件（仅 Linux） | `--features cuda-build,static-libs` | 需 NVIDIA 官方 toolkit（apt 版不带静态库）；实测 858MiB（2 架构+Vulkan）<2GiB；运行需驱动 libcuda.so.1。探针：`cuda-probe.yml` |
| 动态链接（CUDA DLL 注入玩法） | `--features dynamic-link` | exe + llama/ggml DLL；`--cuda-libs` 可用官方 ggml-cuda.dll |
| Windows CRT | 保持默认动态 `/MD` | 勿开静态 CRT（cmake-rs 传导不进 llama.cpp，实测 LNK2001；vcomp140 反正需要 VC++ 运行库） |

Windows Vulkan 构建的三个已知雷与对策（Ninja 生成器绕 MSBuild 乱序、vcvars
显式布阵、`CARGO_TARGET_DIR` 短路径避 MAX_PATH）见 `.github/workflows/ci.yml`
windows 腿注释。

### 仓库结构

```
src/
├── main.rs        # 入口：日志装配、自检模式、多文件循环
├── cli.rs         # 参数定义（短参数/alias 策略见文件头）
├── config.rs      # 配置解析 + 模型目录多候选发现
├── logging.rs     # 双通道日志（终端分级 / 文件恒 debug 全量）
├── ffmpeg.rs      # 解码子进程（stderr 排空、Drop 杀进程防僵尸）
├── vad.rs         # silero v4 纯 Rust + 流式段切分状态机（SegTracker）
├── asr.rs         # 流式管线：生产者/消费者 + 滞回缓冲 + 丢弃规则
├── gguf_asr.rs    # llama.cpp/mtmd 推理（采样链、原生日志桥接、自检）
├── srt.rs         # SRT 写出 + 折行/长 cue 拆分排版
├── runtime.rs     # 设备偏好、控制台 UTF-8、退出暂停、DLL 搜索路径
└── assets/        # silero v4 权重（include_bytes! 内嵌）
scripts/           # e2e 断言：字幕质量 / 终端清洁度；媒体拉取
finetune/          # S2TT 模型产线：数据→LoRA→评测→GGUF（全 CI 可复现）
.github/workflows/ # ci.yml（4 腿矩阵+e2e+release）、finetune.yml、cuda-probe.yml
```

### CI（.github/workflows/ci.yml）

* **push/PR**：4 腿矩阵构建（windows-x64 Vulkan / linux-x64 Vulkan / linux-arm64 /
  macos-arm64 Metal，mac 腿硬断言 MTL 设备）+ 36 单测 + 冒烟（版本/帮助/flag/
  隐藏 alias/依赖闭包：PE 导入表、ldd、otool）+ 可运行腿的真实模型自检；
* **e2e**（windows，用发行版 exe，runner 无 GPU → 每轮实测「loader 在、0 设备 →
  CPU 回退」降级路径）：T1 三真实日语视频默认参数全流程（中文直出/密度/排版/
  **终端清洁度**/单 srt 输出断言）、T1b `--context ""` 转写对照、T1c
  `--verbose --filter-fragments --raw-srt --buffer-mb 20` 开关对照、T3 迁移目录
  （拖拽场景）回归；
* **release**（手动 dispatch 勾选）：全部门禁过后发布 4 平台二进制 + 2 模型 +
  SHA256SUMS；模型 GGUF 由 `finetune.yml`（mode=gguf-e2）产线构建并存跨 OS 缓存。

### 模型产线

微调 → 评测 → GGUF 化全链路在 `finetune/`（CPU 可起步，无需密钥）：
`prepare_data.py` → `sft_lora.py` → `eval_s2tt.py` → `merge_lora_hf.py` →
llama.cpp 官方转换器 → Q4_K_M + mmproj。细节见 `finetune/README.md` 与
`finetune/INTEGRATION.md`。

## 📄 License

Unlicense（本仓库代码）。模型权重与数据集遵循各自许可：Qwen3-ASR（Apache-2.0 系）、
FLEURS（CC-BY-4.0）、silero-vad（MIT）。
