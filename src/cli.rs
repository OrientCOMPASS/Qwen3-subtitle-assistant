//! 命令行参数（E3 单模型形态：媒体进 → 直出目标语言 SRT，无 LLM 后处理）。

use crate::runtime::DevicePref;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "subtitle-assistant",
    version,
    about = "本地离线字幕助手（S2TT 微调 Qwen3-ASR 单模型直出 · CUDA/Vulkan/CPU）",
    long_about = None
)]
pub struct Args {
    /// 需要处理的媒体文件（可拖入多个，按顺序处理）
    #[arg(required_unless_present = "gguf_selftest", num_args = 1..)]
    pub files: Vec<PathBuf>,

    // ---------------- 模型 ----------------
    /// S2TT 模型目录（内含 LM GGUF 与 mmproj GGUF）。不指定时按序自动探测：
    /// exe目录/models → exe目录 → ./models → ./models/qwen3-asr-s2tt（旧版布局）
    /// → exe目录/models/qwen3-asr-s2tt（旧版布局）。目录内识别规则：
    /// mmproj 前缀的 .gguf = 音频编码器，其余 .gguf = LM。
    #[arg(long, value_name = "DIR")]
    pub asr_model_dir: Option<PathBuf>,

    /// 显式指定 LM GGUF 文件（覆盖目录探测）
    #[arg(long, value_name = "GGUF")]
    pub model: Option<PathBuf>,

    /// 显式指定 mmproj（音频编码器）GGUF 文件
    #[arg(long, value_name = "GGUF")]
    pub mmproj: Option<PathBuf>,

    /// 任务开关（进 system 段）：默认直出中文；传空串 = 转写源语言
    #[arg(long, default_value = "translate to Chinese")]
    pub context: String,

    // ---------------- 推理设备 ----------------
    /// CUDA/cuDNN 运行库目录：**仅当指定时**才启用 CUDA 加速（ggml-cuda.dll 的
    /// 依赖链 cudart/cublas/cudnn 等从该目录加载）。未指定则按 Vulkan → CPU 回退。
    #[arg(long = "cuda-libs", value_name = "DIR")]
    pub cuda_libs: Option<PathBuf>,

    /// 推理设备：auto=CUDA(需 --cuda-libs)→Vulkan(索引最大的 GPU)→CPU；cpu=强制 CPU
    #[arg(long, value_enum, default_value_t = DevicePref::Auto)]
    pub device: DevicePref,

    /// CPU 推理线程数
    #[arg(long, default_value_t = 4)]
    pub threads: i32,

    /// offload 到 GPU 的层数：-1=自动（有 GPU 全量上卡，权重入显存后释放主机副本）
    #[arg(long, default_value_t = -1)]
    pub gpu_layers: i32,

    /// 单段最多生成 token 数
    #[arg(long, default_value_t = 256)]
    pub asr_max_new_tokens: i32,

    // ---------------- 采样（v0.6 起默认温度采样，替代旧的恒 greedy） ----------------
    /// 采样温度（基座 Qwen3-1.7B 官方非思考模式推荐 0.7）。`--greedy` 时忽略。
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// top-p 核采样（Qwen3 官方推荐 0.8；≥1 视为关闭）。`--greedy` 时忽略。
    #[arg(long, default_value_t = 0.8)]
    pub top_p: f32,

    /// top-k 采样（Qwen3 官方推荐 20；≤0 视为关闭）。`--greedy` 时忽略。
    #[arg(long, default_value_t = 20)]
    pub top_k: i32,

    /// 采样随机种子。默认固定值保证同输入同输出（CI/复现友好）；
    /// 传 0 则每次运行随机取种。`--greedy` 时忽略。
    #[arg(long, default_value_t = 42)]
    pub seed: u32,

    /// 强制贪心解码（v0.5 及以前的固定行为）：输出完全确定，但更易陷入
    /// 重复/幻觉循环；默认关闭（走上方温度采样参数）。
    #[arg(long)]
    pub greedy: bool,

    // ---------------- VAD（内嵌 silero v4，纯 Rust） ----------------
    /// VAD 语音概率阈值
    #[arg(long, default_value_t = 0.5)]
    pub vad_threshold: f32,

    /// VAD 判定语音结束所需的最短静音（秒）
    #[arg(long, default_value_t = 0.5)]
    pub vad_min_silence: f32,

    /// 单条语音段长度上限（秒，超过硬拆）
    #[arg(long, default_value_t = 60.0)]
    pub vad_buffer_secs: f32,

    // ---------------- 排版与输出 ----------------
    /// 字幕行最大显示宽度（CJK 计 2；0=不折行）
    #[arg(long, default_value_t = 44)]
    pub max_line_width: usize,

    /// 单条字幕最长秒数（超过按句读拆分）
    #[arg(long, default_value_t = 15.0)]
    pub max_cue_secs: f64,

    /// 输出目录（默认与输入文件同目录）
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// 除最终 .srt 外，另存排版前的 <名>.raw.srt（调试对照用；默认只输出一个 .srt）
    #[arg(long)]
    pub raw_srt: bool,

    /// 日志同时写入该文件（文件内恒为全量诊断明细，含 llama.cpp 原生日志；
    /// 终端保持精简不受影响）
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// 终端输出详细诊断（逐段 ASR/VAD 明细、后端设备枚举、llama.cpp 原生日志等；
    /// 等价 RUST_LOG=debug。默认隐藏，只保留进度与结果级信息）
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// 失败时不暂停窗口（拖拽运行默认暂停便于看错误）
    #[arg(long)]
    pub no_pause: bool,

    // ---------------- 质量过滤 ----------------
    /// 启用「静音/幻觉碎片」过滤：lang=None 的短碎片（<2s 或 <4 字）判为幻觉丢弃。
    /// 默认**关闭**——凡模型给出文本的段一律保留；空文本段（模型判定纯静音/噪音，
    /// 无字幕内容可写）无论开关如何都会跳过。
    #[arg(long)]
    pub filter_fragments: bool,

    /// [自检] GGUF/mtmd 推理链自检：传 <LM_GGUF> <MMPROJ_GGUF>，加载模型并打印
    /// 后端设备/能力/内存信息后退出（不需要媒体文件）
    #[arg(long = "gguf-selftest", num_args = 2, value_names = ["LM_GGUF", "MMPROJ_GGUF"])]
    pub gguf_selftest: Vec<PathBuf>,
}
