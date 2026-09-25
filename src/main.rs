mod asr;
mod cli;
mod config;
mod ffmpeg;
mod llm;
mod prompt;
mod qc;
mod runtime;
mod srt;
mod translate;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use llm::LlmClient;
use log::{error, info};
use qc::QualityChecker;
use runtime::RuntimeInfo;
use std::path::Path;
use std::process::Child;
use translate::Translator;
use types::SubtitleSegment;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    // Windows 控制台 UTF-8，避免中文日志乱码
    runtime::enable_utf8_console();

    let args = cli::Args::parse();
    if let Err(e) = run(args) {
        error!("执行失败: {:#}", e);
        std::process::exit(1);
    }
}

fn run(args: cli::Args) -> Result<()> {
    let cfg = config::Config::from_args(&args)?;

    // 启动时探测外置 DLL（ggml-cuda.dll / onnxruntime_providers_cuda.dll），
    // 决定 ASR provider 与 LLM GPU offload 策略。
    let rt = RuntimeInfo::detect(cfg.device, &cfg.lib_dirs).context("运行时 DLL 探测失败")?;

    let prompts = prompt::PromptStore::new(&cfg.prompts_dir);

    info!("共接收到 {} 个文件待处理", args.files.len());
    info!(
        "逐句 LLM 质检: {}（上文 {} 条）",
        if cfg.qc_enabled { "开启" } else { "关闭" },
        cfg.qc_context
    );

    let mut ok = 0usize;
    for (i, input) in args.files.iter().enumerate() {
        info!("========================================");
        info!("[{}/{}] 开始处理: {:?}", i + 1, args.files.len(), input);
        info!("========================================");

        match process_one(input, &cfg, &prompts, &rt) {
            Ok(out) => {
                ok += 1;
                info!("✅ 文件处理完成，输出: {:?}", out);
            }
            Err(e) => error!("⚠️ 文件 {:?} 处理失败: {:#}", input, e),
        }
    }

    info!("所有文件处理流程结束：成功 {} / 共 {}。", ok, args.files.len());
    if ok == 0 && !args.files.is_empty() {
        anyhow::bail!("没有任何文件处理成功");
    }
    Ok(())
}

/// 解析最终的 LLM GPU offload 层数。
fn resolve_gpu_layers(cfg: &config::Config, rt: &RuntimeInfo) -> u32 {
    if cfg.gpu_layers >= 0 {
        cfg.gpu_layers as u32
    } else {
        rt.default_gpu_layers()
    }
}

fn wait_ffmpeg(mut child: Child) -> Result<()> {
    let status = child.wait().context("等待 FFmpeg 退出失败")?;
    if !status.success() {
        anyhow::bail!("FFmpeg 解码异常退出（用 RUST_LOG=debug 查看 ffmpeg 错误输出）");
    }
    Ok(())
}

fn reindex(segments: &mut [SubtitleSegment]) {
    for (i, s) in segments.iter_mut().enumerate() {
        s.index = i + 1;
    }
}

fn process_one(
    input: &Path,
    cfg: &config::Config,
    prompts: &prompt::PromptStore,
    rt: &RuntimeInfo,
) -> Result<std::path::PathBuf> {
    anyhow::ensure!(input.exists(), "输入文件不存在: {:?}", input);

    // ---------- 阶段 1: ASR 转录 ----------
    info!(
        "▶ 阶段 1: 加载 ASR 模型（provider={}）...",
        rt.asr_provider()
    );
    let mut asr = asr::AsrEngine::new(
        &cfg.asr_model_dir,
        &cfg.vad_model,
        &asr::AsrOptions {
            provider: rt.asr_provider().to_string(),
            num_threads: cfg.asr_threads,
        },
    )
    .context("初始化 ASR 引擎失败")?;
    info!("ASR 实际运行 provider: {}", asr.provider());

    let duration_secs = ffmpeg::get_duration_secs(input).unwrap_or(0.0);
    let mut ffmpeg_child = ffmpeg::spawn_decode_stream(input).context("启动 FFmpeg 解码流失败")?;
    let stdout = ffmpeg_child.stdout.take().context("无法获取 FFmpeg stdout")?;

    if cfg.qc_enabled {
        // ============ 逐句质检模式：ASR 与 LLM 同时驻留 ============
        info!("▶ 预加载 LLM（逐句质检需要 ASR 与 LLM 同时驻留内存/显存）...");
        let gpu_layers = resolve_gpu_layers(cfg, rt);
        let llm = LlmClient::new(&cfg.llm_model, cfg.ctx_size, gpu_layers)
            .context("加载 LLM 模型失败")?;
        let mut session = llm.session().context("创建 LLM 会话失败")?;
        let mut checker = QualityChecker::new(cfg.qc_context);

        // raw_log 记录质检前的原始 ASR 输出（用于 .raw.srt 对照）
        let mut raw_log: Vec<SubtitleSegment> = Vec::new();
        let verified = {
            let mut hook = |seg: &mut SubtitleSegment| -> bool {
                raw_log.push(seg.clone());
                checker.check(&mut session, prompts, seg)
            };
            asr.transcribe_stream(stdout, duration_secs, &mut hook)?
        };
        wait_ffmpeg(ffmpeg_child)?;

        info!("✔ {}", checker.stats().summary());

        reindex(&mut raw_log);
        srt::write_srt(&config::raw_srt_path(input), &raw_log).ok();
        srt::write_srt(&config::verified_srt_path(input), &verified).ok();

        // ASR 完成使命，立即释放，为翻译腾出显存/内存
        drop(asr);
        info!("✔ ASR 模型已卸载。");

        if verified.is_empty() {
            anyhow::bail!("质检后没有任何有效语音内容，跳过翻译。");
        }

        // ---------- 阶段 2: 翻译（复用已加载的 LLM 会话） ----------
        info!("▶ 阶段 2: 开始全局摘要提取与分批翻译...");
        let mut translator = Translator::new(&mut session, prompts, cfg);
        let translated = translator.translate(verified)?;
        drop(translator);
        drop(session);
        drop(llm);
        info!("✔ LLM 模型已卸载，内存已释放。");

        // ---------- 阶段 3: 输出 ----------
        let out_path = config::output_srt_path(input);
        srt::write_srt(&out_path, &translated)?;
        Ok(out_path)
    } else {
        // ============ --no-qc：旧版线性工作流（ASR 与 LLM 不同时驻留） ============
        let mut segments = {
            let mut hook = |_seg: &mut SubtitleSegment| true;
            asr.transcribe_stream(stdout, duration_secs, &mut hook)?
        };
        wait_ffmpeg(ffmpeg_child)?;
        drop(asr);
        info!("✔ ASR 模型已卸载，共识别 {} 条字幕。", segments.len());

        srt::write_srt(&config::raw_srt_path(input), &segments).ok();

        if segments.is_empty() {
            anyhow::bail!("未识别到任何语音内容，跳过翻译。");
        }

        info!("▶ 阶段 2: 加载 LLM 并开始翻译...");
        let gpu_layers = resolve_gpu_layers(cfg, rt);
        let llm = LlmClient::new(&cfg.llm_model, cfg.ctx_size, gpu_layers)
            .context("加载 LLM 翻译模型失败")?;
        let mut session = llm.session().context("创建 LLM 会话失败")?;
        let mut translator = Translator::new(&mut session, prompts, cfg);
        let translated = translator.translate(std::mem::take(&mut segments))?;
        drop(translator);
        drop(session);
        drop(llm);
        info!("✔ LLM 模型已卸载，内存已释放。");

        let out_path = config::output_srt_path(input);
        srt::write_srt(&out_path, &translated)?;
        Ok(out_path)
    }
}
