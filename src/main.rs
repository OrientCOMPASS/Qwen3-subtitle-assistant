//! Qwen3 Subtitle Assistant —— 单模型直出版（E3）。
//!
//! 工作流：媒体文件 → ffmpeg(16k f32le PCM) → 纯 Rust silero VAD →
//! S2TT 微调 Qwen3-ASR（llama.cpp/GGUF，context 任务开关直出目标语言）→
//! 丢弃静音/幻觉碎片 → 排版（折行 + 长 cue 拆分）→ SRT。
//!
//! 历史：v0.x 为「ASR(日语) + 1.7B LLM 四段后处理」的两段式精翻管线；
//! S2TT 微调质量实测追平后（finetune/INTEGRATION.md §8/§9、README §10/§11），
//! 产品线收敛为单模型直出，LLM/sherpa-onnx/ORT 依赖全部移除。

mod asr;
mod cli;
mod config;
mod ffmpeg;
mod gguf_asr;
mod runtime;
mod srt;
mod types;
mod vad;

use anyhow::{Context, Result};
use clap::Parser;
use config::Config;
use log::{error, info};
use runtime::{DevicePref, RuntimeProbe};
use std::io::Write;
use std::path::Path;

fn main() {
    let args = cli::Args::parse();
    init_logging(args.log_file.as_deref());
    runtime::enable_utf8_console();

    // GGUF 自检模式：不处理媒体，验证模型加载/后端发现/能力（--gguf-selftest）
    if args.gguf_selftest.len() == 2 {
        let ngl = if args.device == DevicePref::Cpu { 0 } else { 999 };
        match gguf_asr::selftest(
            &args.gguf_selftest[0],
            &args.gguf_selftest[1],
            args.threads,
            ngl,
            args.cuda_libs.as_deref(),
            args.device == DevicePref::Cpu,
        ) {
            Ok(msg) => {
                info!("{}", msg);
                return;
            }
            Err(e) => {
                error!("GGUF 自检失败: {:#}", e);
                std::process::exit(2);
            }
        }
    }

    let no_pause = args.no_pause;
    let n_files = args.files.len();
    let outcome = run(args);

    let failed = match &outcome {
        Ok(failed) => *failed,
        Err(e) => {
            error!("执行失败: {:#}", e);
            n_files.max(1)
        }
    };
    match &outcome {
        Ok(0) => info!("全部 {} 个文件处理成功。", n_files),
        Ok(f) => error!("{} / {} 个文件处理失败。", f, n_files),
        Err(_) => {}
    }

    if failed > 0 && !no_pause {
        runtime::pause_on_exit();
    }
    if failed > 0 {
        std::process::exit(1);
    }
}

fn run(args: cli::Args) -> Result<usize> {
    let probe = RuntimeProbe::new();
    let cfg = Config::from_args(&args, &probe)?;

    info!(
        "模式: S2TT 单模型直出｜context={:?}｜设备: {}｜线程: {}｜VAD: thr={} min_silence={}s",
        cfg.context, cfg.device, cfg.threads, cfg.vad_threshold, cfg.vad_min_silence
    );

    // 模型只加载一次，跨文件复用（权重入显存后主机副本即释放）
    let ngl = resolve_ngl(&cfg);
    let mut engine = gguf_asr::GgufAsr::open(
        &cfg.lm_gguf,
        &cfg.mmproj_gguf,
        cfg.threads,
        ngl,
        cfg.max_new_tokens,
        cfg.cuda_libs.as_deref(),
        cfg.device == DevicePref::Cpu,
    )
    .context("加载 S2TT GGUF 模型失败")?;

    let mut failed = 0usize;
    for (i, input) in args.files.iter().enumerate() {
        banner(i + 1, args.files.len(), input);
        match process_one(input, &cfg, &mut engine) {
            Ok((out, n)) => info!("✔ 完成: {:?}（{} 条字幕）", out, n),
            Err(e) => {
                error!("✘ 处理失败 {:?}: {:#}", input, e);
                failed += 1;
            }
        }
    }
    Ok(failed)
}

fn process_one(input: &Path, cfg: &Config,
               engine: &mut gguf_asr::GgufAsr) -> Result<(std::path::PathBuf, usize)> {
    if !input.is_file() {
        anyhow::bail!("输入文件不存在: {:?}", input);
    }
    let segments = asr::transcribe_file(input, cfg, engine)?;

    // 排版前原始直出（调试/对照）
    let raw_path = cfg.raw_srt_path(input);
    srt::write_srt(&raw_path, &segments).ok();

    // 排版：折行 + 长 cue 拆分（复用产品既有实现）
    let laid = srt::layout(&segments, cfg.max_line_width, cfg.max_cue_secs);
    info!("排版完成：{} 条 -> {} 条（行宽 {}，单条最长 {}s）",
          segments.len(), laid.len(), cfg.max_line_width, cfg.max_cue_secs);

    let out_path = cfg.output_srt_path(input);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    srt::write_srt(&out_path, &laid)?;
    Ok((out_path, laid.len()))
}

fn resolve_ngl(cfg: &Config) -> i32 {
    if cfg.device == DevicePref::Cpu {
        return 0;
    }
    if cfg.gpu_layers >= 0 {
        cfg.gpu_layers
    } else {
        999 // 自动：全量上卡（llama 按模型层数截断；无 GPU 时自动全留 CPU）
    }
}

fn banner(i: usize, total: usize, input: &Path) {
    info!("========================================");
    info!("[{}/{}] 开始处理: {:?}", i, total, input);
    info!("========================================");
}

/// 日志：默认写 stderr；给了 `--log-file` 时同时落盘（拖拽运行看不到控制台时用）。
fn init_logging(log_file: Option<&Path>) {
    let mut builder = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    builder.format_timestamp(None);
    if let Some(path) = log_file {
        match TeeWriter::create(path) {
            Ok(tee) => {
                builder.target(env_logger::Target::Pipe(Box::new(tee)));
            }
            Err(e) => {
                builder.init();
                log::warn!("无法写日志文件 {:?}（{}），仅输出到控制台", path, e);
                return;
            }
        }
    }
    builder.init();
    if let Some(path) = log_file {
        info!("日志同时写入: {:?}", path);
    }
}

/// stderr + 文件的简单 tee。
struct TeeWriter {
    file: std::fs::File,
}

impl TeeWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let stderr = std::io::stderr();
        let n = stderr.lock().write(buf)?;
        let _ = self.file.write_all(buf);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let _ = self.file.flush();
        Ok(())
    }
}
