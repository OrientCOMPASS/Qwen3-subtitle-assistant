use crate::cli::Args;
use anyhow::{Context, Result};
use log::{info, warn};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    // ---- 资源路径 ----
    pub asr_model_dir: PathBuf,
    pub vad_model: PathBuf,
    pub llm_model: PathBuf,
    pub prompts_dir: PathBuf,

    // ---- 翻译 ----
    pub target_lang: String,
    pub source_lang: String,
    pub batch_size: usize,
    pub batch_chars: usize,
    pub context_size: usize,
    pub max_retries: usize,
    pub temperature: f32,
    pub summary_chunk_tokens: usize,
    pub translate_tokens_per_item: usize,

    // ---- LLM 运行时 ----
    /// LLM GPU offload 层数；-1 表示由运行时探测结果决定
    pub gpu_layers: i32,
    pub ctx_size: u32,
    pub prefill_batch: u32,
    pub seed: u32,

    // ---- ASR / VAD ----
    pub asr_threads: i32,
    pub asr_hotwords: String,
    pub asr_max_new_tokens: i32,
    pub asr_max_total_len: i32,
    pub vad_buffer_secs: f32,
    pub vad_min_silence: f32,

    // ---- 质检 ----
    pub qc_enabled: bool,
    pub qc_context: usize,
    pub qc_retries: usize,
    pub qc_max_tokens: u32,
    pub qc_min_similarity: f32,
    pub qc_keep_min_chars: usize,

    // ---- 排版 ----
    pub max_line_width: usize,
    pub max_cue_secs: f64,
    pub layout_enabled: bool,

    // ---- 运行方式 ----
    pub from_srt: bool,
    pub output_dir: Option<PathBuf>,
}

impl Config {
    /// `exe_dir` 用于解析相对路径资源：拖拽/快捷方式启动时 CWD 未必是 exe 目录。
    pub fn from_args(args: &Args, exe_dir: &Path) -> Result<Self> {
        let prompts_dir = resolve_resource(&args.prompts_dir, exe_dir, "提示词目录");
        anyhow::ensure!(
            prompts_dir.is_dir(),
            "提示词目录不存在: {:?}（也不在 exe 目录 {:?} 下）。\n\
             请确认发行包完整解压，或用 --prompts-dir 指定。",
            args.prompts_dir,
            exe_dir
        );

        let asr_model_dir = resolve_resource(&args.asr_model_dir, exe_dir, "ASR 模型目录");
        let vad_model = resolve_resource(&args.vad_model, exe_dir, "VAD 模型");

        // LLM 模型路径：未指定时从 ./models（或 exe 目录的 models）递归搜索第一个 .gguf
        let llm_model = if let Some(path) = &args.llm_model {
            let p = resolve_resource(path, exe_dir, "LLM 模型");
            anyhow::ensure!(p.is_file(), "指定的 LLM 模型不存在: {:?}", p);
            p
        } else {
            find_gguf(&args, exe_dir).context(
                "自动搜索翻译模型失败：请将 Qwen3 GGUF 放入 ./models（任意子目录），\
                 或用 --llm-model 指定路径；也可运行 scripts/download_models.ps1 一键下载",
            )?
        };

        // ---- 参数校正（一律显式告警，不静默改写）----
        let ctx_size = clamp_warn(args.ctx_size, 2048, u32::MAX, "--ctx-size", 2048);
        let prefill_batch = clamp_warn(args.prefill_batch, 64, ctx_size, "--prefill-batch", 2048)
            .min(ctx_size);
        let batch_size = clamp_warn(args.batch_size, 1, usize::MAX, "--batch-size", 1);
        let batch_chars = clamp_warn(args.batch_chars, 64, usize::MAX, "--batch-chars", 1200);
        let qc_context = clamp_warn(args.qc_context, 1, usize::MAX, "--qc-context", 1);
        let max_cue_secs = if args.max_cue_secs < 0.0 {
            warn!("--max-cue-secs 为负，已按 0（不拆分）处理");
            0.0
        } else {
            args.max_cue_secs
        };

        if args.from_srt {
            for f in &args.files {
                if f.extension().and_then(|e| e.to_str()) != Some("srt") {
                    warn!("--from-srt 模式下输入应为 .srt 文件，但收到: {:?}", f);
                }
            }
        }

        let cfg = Self {
            asr_model_dir,
            vad_model,
            llm_model,
            prompts_dir,
            target_lang: args.target_lang.clone(),
            source_lang: args.source_lang.clone(),
            batch_size,
            batch_chars,
            context_size: args.context_size,
            max_retries: args.max_retries.max(1),
            temperature: args.temperature.max(0.0),
            summary_chunk_tokens: args.summary_chunk_tokens.max(256),
            translate_tokens_per_item: args.translate_tokens_per_item.max(16),
            gpu_layers: args.gpu_layers,
            ctx_size,
            prefill_batch,
            seed: args.seed,
            asr_threads: args.asr_threads.max(1),
            asr_hotwords: args.asr_hotwords.trim().to_string(),
            asr_max_new_tokens: args.asr_max_new_tokens.max(16),
            asr_max_total_len: args.asr_max_total_len.max(64),
            vad_buffer_secs: if args.vad_buffer_secs > 0.0 {
                args.vad_buffer_secs
            } else {
                60.0
            },
            vad_min_silence: args.vad_min_silence.max(0.05),
            qc_enabled: !args.no_qc,
            qc_context,
            qc_retries: args.qc_retries.clamp(1, 5),
            qc_max_tokens: args.qc_max_tokens.max(64),
            qc_min_similarity: args.qc_min_similarity.clamp(0.0, 1.0),
            qc_keep_min_chars: args.qc_keep_min_chars,
            max_line_width: args.max_line_width,
            max_cue_secs,
            layout_enabled: !args.no_layout,
            from_srt: args.from_srt,
            output_dir: args.output_dir.clone(),
        };
        info!("使用 LLM 模型: {:?}", cfg.llm_model);
        Ok(cfg)
    }

    // ---------------- 输出路径 ----------------

    fn out_dir_for(&self, input: &Path) -> PathBuf {
        match &self.output_dir {
            Some(d) => d.clone(),
            None => input
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".")),
        }
    }

    fn file_stem(input: &Path) -> String {
        // 对 a.b.srt 这类多点文件名，只剥掉最后一个扩展名
        input
            .file_name()
            .map(|n| {
                let s = n.to_string_lossy().to_string();
                match s.rfind('.') {
                    Some(i) if i > 0 => s[..i].to_string(),
                    _ => s,
                }
            })
            .unwrap_or_else(|| "output".to_string())
    }

    /// 最终字幕：`--from-srt` 时不覆盖输入，改写 `.translated.srt`
    pub fn output_srt_path(&self, input: &Path) -> PathBuf {
        let stem = Self::file_stem(input);
        let name = if self.from_srt {
            format!("{}.translated.srt", stem)
        } else {
            format!("{}.srt", stem)
        };
        self.out_dir_for(input).join(name)
    }

    /// 原始 ASR 输出（未经质检/翻译）
    pub fn raw_srt_path(&self, input: &Path) -> PathBuf {
        self.out_dir_for(input)
            .join(format!("{}.raw.srt", Self::file_stem(input)))
    }

    /// 质检后的中间产物（仅启用 QC 时写出）
    pub fn verified_srt_path(&self, input: &Path) -> PathBuf {
        self.out_dir_for(input)
            .join(format!("{}.verified.srt", Self::file_stem(input)))
    }
}

fn clamp_warn<T>(v: T, lo: T, hi: T, name: &str, fallback: T) -> T
where
    T: PartialOrd + Copy + std::fmt::Debug,
{
    if v < lo || v > hi {
        warn!(
            "{} = {:?} 超出合理范围，已改用 {:?}",
            name, v, fallback
        );
        fallback
    } else {
        v
    }
}

/// 相对路径先在 CWD 找，找不到再回退到 exe 目录（拖拽/快捷方式启动时 CWD 不可控）。
fn resolve_resource(p: &Path, exe_dir: &Path, what: &str) -> PathBuf {
    if p.is_absolute() || p.exists() {
        return p.to_path_buf();
    }
    let alt = exe_dir.join(p);
    if alt.exists() {
        info!(
            "{}：当前目录下没有 {:?}，改用 exe 目录下的 {:?}",
            what, p, alt
        );
        return alt;
    }
    p.to_path_buf()
}

/// 递归搜索第一个 .gguf：先 CWD 的 ./models，再 exe 目录的 models。
fn find_gguf(args: &Args, exe_dir: &Path) -> Result<PathBuf> {
    let mut roots: Vec<PathBuf> = vec![PathBuf::from("./models")];
    let exe_models = exe_dir.join("models");
    if !roots.contains(&exe_models) {
        roots.push(exe_models);
    }
    // 若用户显式给了 asr_model_dir 的上级目录，也顺带看一眼
    if let Some(parent) = args.asr_model_dir.parent() {
        let p = resolve_resource(parent, exe_dir, "模型目录");
        if p.is_dir() && !roots.contains(&p) {
            roots.push(p);
        }
    }

    let mut last_err = anyhow::anyhow!("未搜索任何目录");
    for root in &roots {
        if !root.is_dir() {
            last_err = anyhow::anyhow!("模型目录不存在: {:?}", root);
            continue;
        }
        let mut found = Vec::new();
        search_recursively(root, &mut found, 8)?;
        if found.len() > 1 {
            warn!(
                "在 {:?} 下找到 {} 个 .gguf，按路径序取第一个: {:?}（如需指定请用 --llm-model）",
                root,
                found.len(),
                found[0]
            );
        }
        if let Some(first) = found.into_iter().next() {
            return Ok(first);
        }
        last_err = anyhow::anyhow!("{:?} 下没有任何 .gguf 文件", root);
    }
    Err(last_err)
}

/// 递归收集 .gguf（限制深度、不跟随符号链接，避免 junction 造成的死循环）。
fn search_recursively(path: &Path, out: &mut Vec<PathBuf>, depth: u8) -> Result<()> {
    if depth == 0 {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .with_context(|| format!("读取目录失败: {:?}", path))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let entry_path = entry.path();
        // symlink_metadata：符号链接/junction 不当目录递归
        let is_dir = entry
            .file_type()
            .map(|t| t.is_dir())
            .unwrap_or_else(|_| entry_path.is_dir());
        if is_dir {
            search_recursively(&entry_path, out, depth - 1)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_strips_only_last_extension() {
        assert_eq!(Config::file_stem(Path::new("a/video.mp4")), "video");
        assert_eq!(Config::file_stem(Path::new("video.en.srt")), "video.en");
        assert_eq!(Config::file_stem(Path::new("noext")), "noext");
    }

    /// 建一个临时假 .gguf，避开"模型必须存在"的校验
    fn dummy_gguf() -> PathBuf {
        let p = std::env::temp_dir().join(format!("qsa-test-{}.gguf", std::process::id()));
        std::fs::write(&p, b"GGUF-dummy").unwrap();
        p
    }

    #[test]
    fn output_paths_respect_output_dir_and_from_srt() {
        let gguf = dummy_gguf();
        let base = Args {
            files: vec![],
            asr_model_dir: PathBuf::from("./models/asr"),
            vad_model: PathBuf::from("./models/silero_vad.onnx"),
            llm_model: Some(gguf.clone()),
            prompts_dir: PathBuf::from("."),
            device: crate::runtime::DevicePref::Cpu,
            lib_dirs: vec![],
            gpu_layers: 0,
            ctx_size: 4096,
            prefill_batch: 512,
            seed: 1,
            temperature: 0.0,
            max_retries: 1,
            summary_chunk_tokens: 1000,
            translate_tokens_per_item: 64,
            asr_threads: 1,
            asr_hotwords: String::new(),
            asr_max_new_tokens: 128,
            asr_max_total_len: 512,
            vad_buffer_secs: 30.0,
            vad_min_silence: 0.5,
            no_qc: true,
            qc_context: 1,
            qc_retries: 1,
            qc_max_tokens: 128,
            qc_min_similarity: 0.3,
            qc_keep_min_chars: 30,
            batch_size: 5,
            batch_chars: 200,
            context_size: 2,
            target_lang: "简体中文".into(),
            source_lang: "日语".into(),
            max_line_width: 40,
            max_cue_secs: 7.0,
            no_layout: false,
            from_srt: false,
            output_dir: None,
            log_file: None,
            no_pause: true,
        };
        let cfg = Config::from_args(&base, Path::new("/nonexistent-exe-dir")).expect("cfg");
        let input = Path::new("/media/ep01.mp4");
        assert_eq!(cfg.output_srt_path(input), PathBuf::from("/media/ep01.srt"));
        assert_eq!(cfg.raw_srt_path(input), PathBuf::from("/media/ep01.raw.srt"));

        let mut srt_args = base.clone();
        srt_args.from_srt = true;
        srt_args.output_dir = Some(PathBuf::from("/tmp/out"));
        let cfg2 = Config::from_args(&srt_args, Path::new("/nonexistent-exe-dir")).expect("cfg2");
        let input2 = Path::new("/media/ep01.srt");
        // 不覆盖输入，且落到 --output-dir
        assert_eq!(
            cfg2.output_srt_path(input2),
            PathBuf::from("/tmp/out/ep01.translated.srt")
        );
    }
}
