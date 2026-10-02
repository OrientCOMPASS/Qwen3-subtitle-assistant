//! 命令行参数（v0.6 命名整理：语义前缀统一、常用项配短参数；
//! 历史名保留为隐藏 alias，旧脚本/快捷方式不破坏）。
//!
//! 短参数分配总览（避免冲突，按使用频度给常用项）：
//!   -m model  -c context  -d device  -t threads  -T temperature  -g greedy
//!   -b buffer-mb  -o output-dir  -l log-file  -v verbose  -f filter-fragments
//!   （clap 自带 -h help / -V version）

use crate::runtime::DevicePref;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "Qwen3-subtitle-assistant",
    version,
    about = "本地离线字幕助手（S2TT 微调 Qwen3-ASR 单模型直出 · Vulkan/Metal/CPU · 流式 VAD→ASR）",
    long_about = None
)]
pub struct Args {
    /// 需要处理的媒体文件（可拖入多个，按顺序处理）
    #[arg(required_unless_present = "gguf_selftest", num_args = 1..)]
    pub files: Vec<PathBuf>,

    // ---------------- 模型 ----------------
    /// 模型目录（内含 LM GGUF 与 mmproj GGUF）。不指定时按序自动探测：
    /// exe目录/models → exe目录 → ./models → 旧版 qwen3-asr-s2tt 布局。
    /// 目录内识别规则：mmproj 前缀的 .gguf = 音频编码器，其余 .gguf = LM。
    #[arg(long, value_name = "DIR", alias = "asr-model-dir")]
    pub model_dir: Option<PathBuf>,

    /// 显式指定 LM GGUF 文件（覆盖目录探测）
    #[arg(long, short = 'm', value_name = "GGUF")]
    pub model: Option<PathBuf>,

    /// 显式指定 mmproj（音频编码器）GGUF 文件
    #[arg(long, value_name = "GGUF")]
    pub mmproj: Option<PathBuf>,

    /// 任务开关（进 system 段）：默认直出中文；传空串 = 转写源语言
    #[arg(long, short = 'c', default_value = "translate to Chinese")]
    pub context: String,

    // ---------------- 推理设备 ----------------
    /// CUDA/cuDNN 运行库目录：**仅动态链接构建生效**（发行单文件版为
    /// Vulkan/Metal/CPU）。指定后 ggml 从该目录加载外置后端与依赖链。
    #[arg(long = "cuda-libs", value_name = "DIR")]
    pub cuda_libs: Option<PathBuf>,

    /// 推理设备：auto=CUDA(需 --cuda-libs)→Vulkan/Metal(索引最大的 GPU)→CPU；
    /// cpu=强制 CPU
    #[arg(long, short = 'd', value_enum, default_value_t = DevicePref::Auto)]
    pub device: DevicePref,

    /// CPU 推理线程数
    #[arg(long, short = 't', default_value_t = 4)]
    pub threads: i32,

    /// offload 到 GPU 的层数：-1=自动（有 GPU 全量上卡，权重入显存后释放主机副本）
    #[arg(long, default_value_t = -1)]
    pub gpu_layers: i32,

    /// 单段最多生成 token 数
    #[arg(long, default_value_t = 256, alias = "asr-max-new-tokens")]
    pub max_new_tokens: i32,

    // ---------------- 采样（v0.6 起默认温度采样，--greedy 回到确定性贪心） ----
    /// 采样温度（基座 Qwen3-1.7B 官方非思考模式推荐 0.7）。--greedy 时忽略。
    #[arg(long, short = 'T', default_value_t = 0.7)]
    pub temperature: f32,

    /// top-p 核采样（Qwen3 官方推荐 0.8；≥1 视为关闭）。--greedy 时忽略。
    #[arg(long, default_value_t = 0.8)]
    pub top_p: f32,

    /// top-k 采样（Qwen3 官方推荐 20；≤0 视为关闭）。--greedy 时忽略。
    #[arg(long, default_value_t = 20)]
    pub top_k: i32,

    /// 采样随机种子：默认固定值保证同输入同输出（可复现）；传 0 = 每次运行随机。
    /// --greedy 时忽略。
    #[arg(long, default_value_t = 42)]
    pub seed: u32,

    /// 强制贪心解码（v0.5 及以前的固定行为）：输出完全确定，但更易陷入
    /// 重复/幻觉循环；默认关闭（走上方温度采样参数）。
    #[arg(long, short = 'g')]
    pub greedy: bool,

    // ---------------- VAD（内嵌 silero v4，纯 Rust） ----------------
    /// VAD 语音概率阈值
    #[arg(long, default_value_t = 0.5)]
    pub vad_threshold: f32,

    /// VAD 判定语音结束所需的最短静音（秒）
    #[arg(long, default_value_t = 0.5)]
    pub vad_min_silence: f32,

    /// 单条语音段长度上限（秒，连续语音超过即硬拆成多段）
    #[arg(long, default_value_t = 60.0, alias = "vad-buffer-secs")]
    pub vad_max_seg_secs: f32,

    // ---------------- 流式管线缓冲 ----------------
    /// VAD→ASR 滞回缓冲区容量（MB，按波形字节计量）：占用满时 VAD 暂停生产，
    /// 消费到半容量恢复；占用情况实时显示在进度条。调大可让 VAD 跑得更靠前，
    /// 调小省内存（16kHz f32 单声道 ≈ 3.8MB/分钟）。
    #[arg(long, short = 'b', default_value_t = 50)]
    pub buffer_mb: usize,

    // ---------------- 排版与输出（断句标准：双模型时代默认，v0.6.2 恢复） ----
    /// 字幕行最大显示宽度（CJK 计 2、ASCII 计 1；40 ≈ 20 个汉字。0=不折行）
    #[arg(long, default_value_t = 40)]
    pub max_line_width: usize,

    /// 单条字幕最长秒数，超过按句读拆分（时间按字符占比分配；0=不按秒拆）
    #[arg(long, default_value_t = 7.0)]
    pub max_cue_secs: f64,

    /// 单条字幕最大字符数，超过按句读拆分（≈ 两行；语速快时秒数约束不够，
    /// 需要字符维度兜底。0=不按字符拆）
    #[arg(long, default_value_t = 40)]
    pub max_cue_chars: usize,

    /// 关闭排版（不折行、不拆长条），输出与转录段一一对应的字幕（调试对照用）
    #[arg(long)]
    pub no_layout: bool,

    /// 输出目录（默认与输入文件同目录）
    #[arg(long, short = 'o')]
    pub output_dir: Option<PathBuf>,

    /// 除最终 .srt 外，另存排版前的 <名>.raw.srt（调试对照用；默认只输出一个 .srt）
    #[arg(long)]
    pub raw_srt: bool,

    /// 日志同时写入该文件（文件内恒为全量诊断明细，含 llama.cpp 原生日志；
    /// 终端保持精简不受影响）
    #[arg(long, short = 'l')]
    pub log_file: Option<PathBuf>,

    /// 终端输出详细诊断（逐段跳过/丢弃明细、后端设备枚举、llama.cpp 原生日志等；
    /// 等价 RUST_LOG=debug。默认隐藏，只保留进度、逐条字幕与结果级信息）
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// 失败时不暂停窗口（拖拽运行默认暂停便于看错误）
    #[arg(long)]
    pub no_pause: bool,

    // ---------------- 质量过滤 ----------------
    /// 启用「静音/幻觉碎片」过滤：lang=None 的短碎片（<2s 或 <4 字）判为幻觉丢弃。
    /// 默认**关闭**——凡模型给出文本的段一律保留（空文本段、与上一条重复的段、
    /// 连续纯语气词段除外，见 README「丢弃规则」）。
    #[arg(long, short = 'f')]
    pub filter_fragments: bool,

    /// [自检] GGUF/mtmd 推理链自检：传 <LM_GGUF> <MMPROJ_GGUF>，加载模型并打印
    /// 后端设备/能力/内存信息后退出（不需要媒体文件）
    #[arg(long = "gguf-selftest", num_args = 2, value_names = ["LM_GGUF", "MMPROJ_GGUF"])]
    pub gguf_selftest: Vec<PathBuf>,
}
