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
    /// S2TT 模型目录（内含 LM GGUF 与 mmproj GGUF；默认名下 model.gguf/mmproj.gguf，
    /// 也自动识别 *.gguf 与 mmproj*.gguf）。相对路径找不到时回退 exe 同目录。
    #[arg(long, default_value = "./models/qwen3-asr-s2tt")]
    pub asr_model_dir: PathBuf,

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

    /// 日志同时写入该文件
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// 失败时不暂停窗口（拖拽运行默认暂停便于看错误）
    #[arg(long)]
    pub no_pause: bool,

    /// [自检] GGUF/mtmd 推理链自检：传 <LM_GGUF> <MMPROJ_GGUF>，加载模型并打印
    /// 后端设备/能力/内存信息后退出（不需要媒体文件）
    #[arg(long = "gguf-selftest", num_args = 2, value_names = ["LM_GGUF", "MMPROJ_GGUF"])]
    pub gguf_selftest: Vec<PathBuf>,
}
