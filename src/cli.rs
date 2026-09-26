use crate::runtime::DevicePref;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "subtitle-assistant",
    version,
    about = "本地离线字幕转录与翻译助手（外置 DLL 运行时 · CUDA 加速 · 逐句 LLM 质检）",
    long_about = None
)]
pub struct Args {
    /// 需要处理的媒体文件（可拖入多个，按顺序处理）；配合 --from-srt 时为 .srt 文件
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
    /// 不指定时自动从 ./models（或 exe 同目录的 models）递归搜索第一个 .gguf 文件。
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

    /// LLM 单次 prefill 的最大 token 数（llama.cpp 的 n_batch）。
    /// 长 prompt 会自动按此值分块喂入，一般无需修改。
    #[arg(long, default_value_t = 2048)]
    pub prefill_batch: u32,

    /// 采样种子（0=每次运行随机）。固定种子可让同一输入的结果可复现。
    #[arg(long, default_value_t = 0)]
    pub seed: u32,

    /// 采样温度。默认 0.7 取自 Qwen3 模型卡对非思考模式的建议；
    /// 模型卡明确写着"DO NOT use greedy decoding"，故 0（贪心）只在你确有需要时才用。
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// 采样 top-p（模型卡建议 0.8；1.0 = 关闭）
    #[arg(long, default_value_t = 0.8)]
    pub top_p: f32,

    /// 采样 top-k（模型卡建议 20；0 = 关闭）
    #[arg(long, default_value_t = 20)]
    pub top_k: i32,

    /// 翻译/摘要的单批最大重试次数
    #[arg(long, default_value_t = 3)]
    pub max_retries: usize,

    /// 全局摘要的单块 token 预算：转录超过它就分块提取再合并（map-reduce）
    #[arg(long, default_value_t = 3000)]
    pub summary_chunk_tokens: usize,

    /// 每条译文的生成 token 预算（决定 max_new_tokens 随批大小伸缩的系数）
    #[arg(long, default_value_t = 96)]
    pub translate_tokens_per_item: usize,

    /// ASR 推理线程数（仅 CPU provider 生效）
    #[arg(long, default_value_t = 4)]
    pub asr_threads: i32,

    /// Qwen3-ASR 的 hotwords 偏置词（英文逗号分隔，如 "東京,山中伸弥,iPS"）。
    /// 专有名词识别错误多时很有用；留空表示不启用。
    #[arg(long, default_value = "")]
    pub asr_hotwords: String,

    /// Qwen3-ASR 单段最多生成的 token 数。
    /// 注意它与 --asr-max-total-len、--vad-buffer-secs 是**联动的**：
    /// 音频 token 率 12.5Hz，512 的总预算要分给 prompt + 音频 + 生成，
    /// 每加大 128 个生成 token，可处理的语音段就缩短约 10 秒。
    #[arg(long, default_value_t = 128)]
    pub asr_max_new_tokens: i32,

    /// Qwen3-ASR 的最大总序列长度（prompt + 音频 token + 生成 token）。
    /// **传大于导出模型 KV 容量的值没有意义**：sherpa-onnx 会静默 clamp 到模型上限
    /// （现有 ONNX 导出均为 512），需要更长音频得重新导出 decoder。
    #[arg(long, default_value_t = 512)]
    pub asr_max_total_len: i32,

    /// Silero VAD 缓冲区秒数（单条语音段长度上限）。
    /// 超过模型单段音频上限时会自动收敛并 warn（上限由上面两个参数反推）。
    #[arg(long, default_value_t = 60.0)]
    pub vad_buffer_secs: f32,

    /// Silero VAD 判定语音结束所需的最短静音（秒）
    #[arg(long, default_value_t = 0.5)]
    pub vad_min_silence: f32,

    // ---------------- 逐句质检（QC） ----------------
    /// 关闭逐句 LLM 质检（默认开启）。关闭后回到线性工作流：
    /// 先转录完再加载 LLM 翻译，ASR 与 LLM 不同时驻留内存。
    #[arg(long)]
    pub no_qc: bool,

    /// 质检时提供给 LLM 的上文字幕条数
    #[arg(long, default_value_t = 3)]
    pub qc_context: usize,

    /// 质检单句的最大尝试次数（含请求失败与格式错误重试）
    #[arg(long, default_value_t = 2)]
    pub qc_retries: usize,

    /// 质检单句的生成 token 上限
    #[arg(long, default_value_t = 256)]
    pub qc_max_tokens: u32,

    /// 质检 fix 判决的最小可信相似度（0~1）：纠正文本与原文差异过大时视为幻觉，保留原文
    #[arg(long, default_value_t = 0.35)]
    pub qc_min_similarity: f32,

    /// 长句保护阈值：达到该字符数且非重复文本的句子**不允许被 QC 丢弃**（0=关闭）。
    /// 实测小模型会编造丢弃理由，把真实语音判成"背景噪音/幻觉套话"——包括 28 字的
    /// 完整日语句子、甚至一句粤语。字幕场景下"丢掉真实内容"比"多留一条噪音"更糟
    /// （噪音可以手删，语音丢了就没了），所以阈值取得比较激进。
    #[arg(long, default_value_t = 12)]
    pub qc_keep_min_chars: usize,

    // ---------------- 翻译与排版 ----------------
    /// 每批翻译的字幕条数上限（prompt 超预算或译文照抄时会自动对半拆分）
    #[arg(long, default_value_t = 20)]
    pub batch_size: usize,

    /// 每批翻译的原文字符预算。只按条数切批时，20 条长句会让小模型输出退化
    /// （实测直接照抄原文），字符预算让批次规模与实际内容量挂钩。
    #[arg(long, default_value_t = 1200)]
    pub batch_chars: usize,

    /// 翻译滑动窗口携带的上文字幕条数
    #[arg(long, default_value_t = 4)]
    pub context_size: usize,

    /// 目标语言
    #[arg(long, default_value = "简体中文")]
    pub target_lang: String,

    /// 源语言（提示词用；不确定就保持默认）
    #[arg(long, default_value = "自动检测（视频原语言）")]
    pub source_lang: String,

    /// 单行字幕最大显示宽度（CJK 计 2、ASCII 计 1；40 ≈ 20 个汉字）。0=不折行
    #[arg(long, default_value_t = 40)]
    pub max_line_width: usize,

    /// 单条字幕最长秒数，超出则按句读拆分（时间按字符占比分配）。0=不拆分
    #[arg(long, default_value_t = 7.0)]
    pub max_cue_secs: f64,

    /// 关闭排版（不折行、不拆长条），输出与 VAD/输入切分完全一致的字幕
    #[arg(long)]
    pub no_layout: bool,

    /// 关闭译文自检（默认开启）。开启时每批译完后再让 LLM 以质检员身份复查一遍，
    /// 挑出残留原文（例如日语假名未译成中文）、误译与术语不一致并就地修正；
    /// 代价是每批多一次 LLM 调用。
    #[arg(long)]
    pub no_review: bool,

    // ---------------- 运行方式 ----------------
    /// 跳过 ASR：把输入当作已有的 .srt 直接做质检+翻译（可用于重跑翻译、二次修正）
    #[arg(long)]
    pub from_srt: bool,

    /// 输出目录（默认与源文件同目录）
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// 同时把日志写入文件（拖拽运行、看不到控制台时很有用）
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// 结束时不暂停（默认：有文件处理失败时在 Windows 控制台等待回车，避免窗口一闪而过）
    #[arg(long)]
    pub no_pause: bool,
}
