use crate::cli::Args;
use anyhow::{Context, Result};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub asr_model_dir: PathBuf,
    pub vad_model: PathBuf,
    pub llm_model: PathBuf,
    pub prompts_dir: PathBuf,
    pub target_lang: String,
    pub batch_size: usize,
    pub context_size: usize,
}

impl Config {
    pub fn from_args(args: &Args) -> Result<Self> {
        anyhow::ensure!(
            args.prompts_dir.is_dir(),
            "提示词目录不存在: {:?}",
            args.prompts_dir
        );

        // 处理 LLM 模型路径：如果未指定，从 ./models 目录递归搜索第一个 .gguf 文件
        let llm_model = if let Some(path) = &args.llm_model {
            path.clone()
        } else {
            find_gguf_in_dir("./models").context("自动搜索翻译模型失败")?
        };

        Ok(Self {
            asr_model_dir: args.asr_model_dir.clone(),
            vad_model: args.vad_model.clone(),
            llm_model,
            prompts_dir: args.prompts_dir.clone(),
            target_lang: "简体中文".to_string(), // 固定输出语言
            batch_size: args.batch_size.max(1),
            context_size: args.context_size,
        })
    }
}

/// 从指定目录中递归搜索第一个 .gguf 文件
fn find_gguf_in_dir(dir: &str) -> Result<PathBuf> {
    let dir_path = std::path::Path::new(dir);
    if !dir_path.is_dir() {
        anyhow::bail!("模型目录不存在: {}", dir);
    }

    // 递归搜索辅助函数
    fn search_recursively(path: &std::path::Path) -> Result<Option<PathBuf>> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            
            if entry_path.is_dir() {
                // 递归搜索子目录
                if let Some(found) = search_recursively(&entry_path)? {
                    return Ok(Some(found));
                }
            } else if entry_path.is_file() && entry_path.extension().map_or(false, |ext| ext == "gguf") {
                // 找到 .gguf 文件
                return Ok(Some(entry_path));
            }
        }
        Ok(None)
    }

    search_recursively(dir_path)?
        .ok_or_else(|| anyhow::anyhow!("在 {} 目录中未找到任何 .gguf 模型文件", dir))
}

pub fn output_srt_path(input: &std::path::Path) -> PathBuf {
    input.with_extension("srt")
}