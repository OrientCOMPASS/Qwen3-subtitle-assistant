use crate::runtime::DevicePref;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "subtitle-assistant",
    version,
    about = "本地离线字幕转录与翻译助手（外置 DLL 运行时 · CUDA 加速 · 逐句 LLM 质检）",
    long_about = None
)]
pub struct Args {
    /// 需要处理的媒体文件（可拖入多个，按顺序处理）
    #[arg(required = true, num_args = 1..)]
    pub files: Vec<PathBuf>,

    // ---------------- 模型与资源路径 ----------------
    /// sherpa-onnx Qwen3-ASR 模型所在目录
    #[arg(long, default_value = "./models/sherpa-onnx-qwen3-asr-0.6B-int8")]
    pub asr_model_dir: PathBuf,

    /// Silero VAD 模型文件路径
    #[arg(long, default_value = "./models/silero_vad.onnx")]
    pub vad_model: PathBuf,

    /// 翻译/质检模型路径 (Qwen3 GGUF 文件)。
    /// 不指定时自动从 ./models 目录递归搜索第一个 .gguf 文件。
    #[arg(long)]
    pub llm_model: Option<PathBuf>,

    /// 提示词目录
    #[arg(long, default_value = "./prompts")]
    pub prompts_dir: PathBuf,

    // ---------------- 推理设备（外置 DLL） ----------------
    /// 推理设备：auto=按 exe 同目录的外置 DLL 自动探测；cpu=强制 CPU；cuda=强制 CUDA
    #[arg(long, value_enum, default_value_t = DevicePref::Auto)]
    pub device: DevicePref,

    /// 附加 DLL 搜索目录（cudart/cublas/cudnn 等 CUDA 运行时不放 exe 同目录时指定，可多次传入）
    #[arg(long = "lib-dir", value_name = "DIR")]
    pub lib_dirs: Vec<PathBuf>,

    /// LLM GPU offload 层数：-1=自动（探测到 CUDA 则全部上卡，否则 0）
    #[arg(long, default_value_t = -1)]
    pub gpu_layers: i32,

    /// LLM 上下文长度（token）
    #[arg(long, default_value_t = 8192)]
    pub ctx_size: u32,

    /// ASR 推理线程数（仅 CPU provider 生效）
    #[arg(long, default_value_t = 4)]
    pub asr_threads: i32,

    // ---------------- 逐句质检（QC） ----------------
    /// 关闭逐句 LLM 质检（默认开启）。关闭后回到旧版线性工作流：
    /// 先转录完再加载 LLM 翻译，ASR 与 LLM 不同时驻留内存。
    #[arg(long)]
    pub no_qc: bool,

    /// 质检时提供给 LLM 的上文字幕条数
    #[arg(long, default_value_t = 3)]
    pub qc_context: usize,

    // ---------------- 翻译 ----------------
    /// 每批翻译的字幕条数
    #[arg(long, default_value_t = 20)]
    pub batch_size: usize,

    /// 翻译滑动窗口携带的上文字幕条数
    #[arg(long, default_value_t = 4)]
    pub context_size: usize,

    /// 目标语言
    #[arg(long, default_value = "简体中文")]
    pub target_lang: String,

    /// 源语言（提示词用；不确定就保持默认）
    #[arg(long, default_value = "自动检测（视频原语言）")]
    pub source_lang: String,
}
