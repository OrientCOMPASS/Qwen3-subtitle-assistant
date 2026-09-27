//! Qwen3 Subtitle Assistant —— 单模型直出版（E3+，单一可执行文件形态）。
//!
//! 工作流：媒体文件 → ffmpeg(16k f32le PCM) → 纯 Rust silero VAD **流式**分段
//! （切出一段立即送 ASR，不等全片）→ S2TT 微调 Qwen3-ASR（llama.cpp/GGUF
//! **静态链接进本 exe**，context 任务开关直出目标语言）→ 排版（折行 + 长 cue
//! 拆分）→ 单个 SRT（排版前 .raw.srt 仅 `--raw-srt` 时另存）。
//!
//! 终端输出策略（见 logging.rs）：默认只保留进度/结果级信息；逐段明细与
//! llama.cpp 原生日志是 debug 级（`--verbose` 打开；`--log-file` 恒全量落盘）。
//!
//! 历史：v0.x 为「ASR(日语) + 1.7B LLM 四段后处理」的两段式精翻管线；
//! S2TT 微调质量实测追平后（finetune/INTEGRATION.md §8/§9、README §10/§11），
//! 产品线收敛为单模型直出，LLM/sherpa-onnx/ORT 依赖全部移除。

mod asr;
mod cli;
mod config;
mod ffmpeg;
mod gguf_asr;
mod logging;
mod runtime;
mod srt;
mod types;
mod vad;

use anyhow::{Context, Result};
use clap::Parser;
use config::Config;
use log::{debug, error, info};
use runtime::DevicePref;
use std::path::Path;

fn main() {
    let args = cli::Args::parse();
    logging::DualLogger::init(args.verbose, args.log_file.as_deref());
    runtime::enable_utf8_console();

    // GGUF 自检模式：不处理媒体，验证模型加载/后端发现/能力（--gguf-selftest）。
    // 结果直接 println/eprintln——自检是用户显式发起的诊断动作，必须无视日志级别可见。
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
                println!("{msg}");
                return;
            }
            Err(e) => {
                eprintln!("✘ GGUF 自检失败: {e:#}");
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
            error!("执行失败: {e:#}");
            n_files.max(1)
        }
    };
    match &outcome {
        Ok(0) => info!("全部 {n_files} 个文件处理成功。"),
        Ok(f) => error!("{f} / {n_files} 个文件处理失败。"),
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
    let probe = runtime::RuntimeProbe::new();
    let cfg = Config::from_args(&args, &probe)?;

    info!(
        "模式: S2TT 单模型直出｜context={:?}｜设备: {}｜线程: {}｜VAD: thr={} min_silence={}s｜碎片过滤: {}｜采样: {}",
        cfg.context, cfg.device, cfg.threads, cfg.vad_threshold, cfg.vad_min_silence,
        if cfg.filter_fragments { "开" } else { "关" },
        if cfg.greedy {
            "greedy".to_string()
        } else {
            format!("temp={} top_p={} top_k={} seed={}", cfg.temperature, cfg.top_p, cfg.top_k, cfg.seed)
        }
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
        gguf_asr::SamplerCfg {
            greedy: cfg.greedy,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            top_k: cfg.top_k,
            seed: cfg.seed,
        },
    )
    .context("加载 S2TT GGUF 模型失败")?;

    let mut failed = 0usize;
    for (i, input) in args.files.iter().enumerate() {
        info!("[{}/{}] 处理: {:?}", i + 1, args.files.len(), input);
        match process_one(input, &cfg, &mut engine) {
            Ok((out, n)) => debug!("✔ 完成: {:?}（{n} 条字幕）", out),
            Err(e) => {
                error!("✘ 处理失败 {:?}: {e:#}", input);
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
    // 流式转录：VAD 切出一段 → 立即 ASR 一段（见 asr.rs 的生产者/消费者管线）
    let segments = asr::transcribe_file(input, cfg, engine)?;

    // 排版前原始直出：仅 --raw-srt 时另存（默认只交付一个 .srt）
    if cfg.raw_srt {
        let raw_path = cfg.raw_srt_path(input);
        if let Err(e) = srt::write_srt(&raw_path, &segments) {
            log::warn!("写 .raw.srt 失败（不影响主输出）: {e:#}");
        }
    }

    // 排版：折行 + 长 cue 拆分
    let laid = srt::layout(&segments, cfg.max_line_width, cfg.max_cue_secs);
    debug!("排版完成：{} 条 -> {} 条（行宽 {}，单条最长 {}s）",
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
