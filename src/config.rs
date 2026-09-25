use crate::cli::Args;
use crate::runtime::DevicePref;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub asr_model_dir: PathBuf,
    pub vad_model: PathBuf,
    pub llm_model: PathBuf,
    pub prompts_dir: PathBuf,

    pub target_lang: String,
    pub source_lang: String,
    pub batch_size: usize,
    pub context_size: usize,

    /// 设备偏好（实际是否走 CUDA 还取决于运行时 DLL 探测结果）
    pub device: DevicePref,
    /// 附加 DLL 搜索目录
    pub lib_dirs: Vec<PathBuf>,
    /// LLM GPU offload 层数；-1 表示由运行时探测决定
    pub gpu_layers: i32,
    pub ctx_size: u32,
    pub asr_threads: i32,

    /// 是否启用逐句 LLM 质检
    pub qc_enabled: bool,
    /// 质检时附带的上文条数
    pub qc_context: usize,
}

impl Config {
    pub fn from_args(args: &Args) -> Result<Self> {
        anyhow::ensure!(
            args.prompts_dir.is_dir(),
            "提示词目录不存在: {:?}",
            args.prompts_dir
        );

        // LLM 模型路径：未指定时从 ./models 递归搜索第一个 .gguf
        let llm_model = if let Some(path) = &args.llm_model {
            anyhow::ensure!(path.is_file(), "指定的 LLM 模型不存在: {:?}", path);
            path.clone()
        } else {
            find_gguf_in_dir("./models").context(
                "自动搜索翻译模型失败：请将 Qwen3 GGUF 放入 ./models（任意子目录），\
                 或用 --llm-model 指定路径；也可运行 scripts/download_models.ps1 一键下载",
            )?
        };

        Ok(Self {
            asr_model_dir: args.asr_model_dir.clone(),
            vad_model: args.vad_model.clone(),
            llm_model,
            prompts_dir: args.prompts_dir.clone(),
            target_lang: args.target_lang.clone(),
            source_lang: args.source_lang.clone(),
            batch_size: args.batch_size.max(1),
            context_size: args.context_size,
            device: args.device,
            lib_dirs: args.lib_dirs.clone(),
            gpu_layers: args.gpu_layers,
            ctx_size: args.ctx_size.max(2048),
            asr_threads: args.asr_threads.max(1),
            qc_enabled: !args.no_qc,
            qc_context: args.qc_context,
        })
    }
}

/// 从指定目录中递归搜索第一个 .gguf 文件（按文件名排序保证确定性）。
fn find_gguf_in_dir(dir: &str) -> Result<PathBuf> {
    let dir_path = Path::new(dir);
    if !dir_path.is_dir() {
        anyhow::bail!("模型目录不存在: {}", dir);
    }

    fn search_recursively(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                search_recursively(&entry_path, out)?;
            } else if entry_path
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
            {
                out.push(entry_path);
            }
        }
        Ok(())
    }

    let mut found = Vec::new();
    search_recursively(dir_path, &mut found)?;
    found
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("在 {} 目录中未找到任何 .gguf 模型文件", dir))
}

pub fn output_srt_path(input: &Path) -> PathBuf {
    input.with_extension("srt")
}

/// 原始 ASR 输出（未经质检/翻译）
pub fn raw_srt_path(input: &Path) -> PathBuf {
    input.with_extension("raw.srt")
}

/// 质检后的中间产物（仅启用 QC 时写出）
pub fn verified_srt_path(input: &Path) -> PathBuf {
    input.with_extension("verified.srt")
}
