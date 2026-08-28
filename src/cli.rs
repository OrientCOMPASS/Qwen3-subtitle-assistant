use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "subtitle-assistant",
    version,
    about = "本地离线字幕转录与翻译助手",
    long_about = None
)]
pub struct Args {
    /// 需要处理的媒体文件（可拖入多个，按顺序处理）
    #[arg(required = true, num_args = 1..)]
    pub files: Vec<PathBuf>,

    /// sherpa-onnx Qwen3-ASR 模型所在目录
    #[arg(long, default_value = "./models/sherpa-onnx-qwen3-asr-0.6B-int8")]
    pub asr_model_dir: PathBuf,

    /// Silero VAD 模型文件路径
    #[arg(long, default_value = "./models/silero_vad.onnx")]
    pub vad_model: PathBuf,

    /// 翻译模型路径 (Qwen3-1.7B-GGUF 文件)
    /// 如果不指定，将自动从 ./models 目录中搜索第一个 .gguf 文件
    #[arg(long)]
    pub llm_model: Option<PathBuf>,

    /// 提示词目录
    #[arg(long, default_value = "./prompts")]
    pub prompts_dir: PathBuf,

    /// 每批翻译的字幕条数
    #[arg(long, default_value_t = 20)]
    pub batch_size: usize,

    /// 滑动窗口携带的上文字幕条数
    #[arg(long, default_value_t = 4)]
    pub context_size: usize,
}