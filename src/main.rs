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
use config::Config;
use llm::{LlmClient, LlmSession};
use log::{error, info, warn};
use prompt::PromptStore;
use qc::QualityChecker;
use runtime::RuntimeInfo;
use std::io::Write;
use std::path::Path;
use translate::Translator;
use types::SubtitleSegment;

fn main() {
    let args = cli::Args::parse();
    init_logging(args.log_file.as_deref());

    // Windows 控制台 UTF-8，避免中文日志乱码
    runtime::enable_utf8_console();

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

    if failed > 0 {
        if !no_pause {
            runtime::pause_on_exit();
        }
        std::process::exit(1);
    }
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
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("打开日志文件失败: {:?}", path))?;
        Ok(Self { file })
    }
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let stderr = std::io::stderr();
        let mut lock = stderr.lock();
        lock.write_all(buf)?;
        let _ = self.file.write_all(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let _ = std::io::stderr().flush();
        let _ = self.file.flush();
        Ok(())
    }
}

fn run(args: cli::Args) -> Result<usize> {
    // 先探测外置 DLL（决定 ASR provider 与 LLM GPU offload 策略），
    // 再用探测到的 exe 目录解析模型/提示词等相对路径。
    let rt = RuntimeInfo::detect(args.device, &args.lib_dirs).context("运行时 DLL 探测失败")?;
    let cfg = Config::from_args(&args, rt.exe_dir())?;
    let prompts = PromptStore::new(&cfg.prompts_dir);

    info!("共接收到 {} 个文件待处理", args.files.len());
    info!(
        "模式: {}｜逐句质检: {}（上文 {} 条，最多 {} 次尝试）｜排版: {}｜LLM: ctx={}, n_batch={}, seed={}",
        if cfg.from_srt {
            "从 SRT 翻译（跳过 ASR）"
        } else {
            "媒体转录"
        },
        if cfg.qc_enabled && !cfg.from_srt {
            "开启"
        } else {
            "关闭"
        },
        cfg.qc_context,
        cfg.qc_retries,
        if cfg.layout_enabled {
            format!(
                "开启（行宽 {}，单条最长 {}s）",
                cfg.max_line_width, cfg.max_cue_secs
            )
        } else {
            "关闭".to_string()
        },
        cfg.ctx_size,
        cfg.prefill_batch,
        cfg.seed
    );

    if cfg.asr_only && !cfg.qc_enabled {
        info!("--asr-only 且未开启质检：本次运行不会加载翻译 LLM");
    }
    let failed = if cfg.from_srt {
        run_from_srt(&args.files, &cfg, &prompts, &rt)?
    } else if cfg.qc_enabled {
        run_with_qc(&args.files, &cfg, &prompts, &rt)?
    } else {
        run_linear(&args.files, &cfg, &prompts, &rt)?
    };

    info!("所有文件处理流程结束：成功 {} / 共 {}。", args.files.len() - failed, args.files.len());
    Ok(failed)
}

fn banner(i: usize, total: usize, input: &Path) {
    info!("========================================");
    info!("[{}/{}] 开始处理: {:?}", i + 1, total, input);
    info!("========================================");
}

/// 解析最终的 LLM GPU offload 层数。
fn resolve_gpu_layers(cfg: &Config, rt: &RuntimeInfo) -> u32 {
    if cfg.gpu_layers >= 0 {
        cfg.gpu_layers as u32
    } else {
        rt.default_gpu_layers()
    }
}

fn new_llm(cfg: &Config, rt: &RuntimeInfo) -> Result<LlmClient> {
    let gpu_layers = resolve_gpu_layers(cfg, rt);
    LlmClient::new(
        &cfg.llm_model,
        cfg.ctx_size,
        cfg.prefill_batch,
        gpu_layers,
        cfg.seed,
    )
    .context("加载 LLM 模型失败")
}

fn new_asr(cfg: &Config, rt: &RuntimeInfo) -> Result<asr::AsrEngine> {
    info!("▶ 加载 ASR 模型（provider={}）...", rt.asr_provider());
    let engine = asr::AsrEngine::new(
        &cfg.asr_model_dir,
        &cfg.vad_model,
        &asr::AsrOptions {
            provider: rt.asr_provider().to_string(),
            num_threads: cfg.asr_threads,
            max_new_tokens: cfg.asr_max_new_tokens,
            max_total_len: cfg.asr_max_total_len,
            hotwords: cfg.asr_hotwords.clone(),
            vad_buffer_secs: cfg.vad_buffer_secs,
            vad_min_silence: cfg.vad_min_silence,
        },
    )
    .context("初始化 ASR 引擎失败")?;
    info!("ASR 实际运行 provider: {}", engine.provider());
    Ok(engine)
}

// ============================================================================
// 模式 A：逐句质检（ASR 与 LLM 同时驻留；两者都只加载一次，跨文件复用）
// ============================================================================

fn run_with_qc(
    files: &[std::path::PathBuf],
    cfg: &Config,
    prompts: &PromptStore,
    rt: &RuntimeInfo,
) -> Result<usize> {
    let mut asr = new_asr(cfg, rt)?;
    info!("▶ 预加载 LLM（逐句质检需要 ASR 与 LLM 同时驻留内存/显存）...");
    let llm = new_llm(cfg, rt)?;
    let mut session = llm.session().context("创建 LLM 会话失败")?;

    let mut failed = 0usize;
    for (i, input) in files.iter().enumerate() {
        banner(i, files.len(), input);
        match process_with_qc(input, cfg, prompts, &mut asr, &mut session) {
            Ok(out) => info!("✅ 文件处理完成，输出: {:?}", out),
            Err(e) => {
                failed += 1;
                error!("⚠️ 文件 {:?} 处理失败: {:#}", input, e);
            }
        }
        // 下一个文件前清空 VAD 状态（引擎复用，不重新加载模型）
        asr.reset();
    }
    Ok(failed)
}

fn process_with_qc(
    input: &Path,
    cfg: &Config,
    prompts: &PromptStore,
    asr: &mut asr::AsrEngine,
    session: &mut LlmSession,
) -> Result<std::path::PathBuf> {
    anyhow::ensure!(input.is_file(), "输入文件不存在: {:?}", input);

    let duration_secs = ffmpeg::get_duration_secs(input).unwrap_or_else(|e| {
        warn!("ffprobe 获取时长失败（进度条退化为计时模式）: {:#}", e);
        0.0
    });
    let mut decoder = ffmpeg::spawn_decode_stream(input).context("启动 FFmpeg 解码流失败")?;
    let stdout = decoder.take_stdout().context("无法获取 FFmpeg stdout")?;

    // ---------- 阶段 1: 转录 + 逐句质检 ----------
    info!("▶ 阶段 1: 流式转录 + 逐句 LLM 质检...");
    let mut checker = QualityChecker::new(
        cfg.qc_context,
        cfg.qc_retries,
        cfg.qc_min_similarity,
        cfg.qc_keep_min_chars,
        cfg.sampling(),
    );
    let qc_max_tokens = cfg.qc_max_tokens;

    let mut raw_log: Vec<SubtitleSegment> = Vec::new();
    let verified = {
        let mut hook = |seg: &mut SubtitleSegment| -> bool {
            raw_log.push(seg.clone());
            checker.check(session, prompts, seg, qc_max_tokens)
        };
        asr.transcribe_stream(stdout, duration_secs, &mut hook)?
    };
    decoder.finish()?;
    info!("✔ {}", checker.stats().summary());

    reindex(&mut raw_log);
    srt::write_srt(&cfg.raw_srt_path(input), &raw_log).ok();
    srt::write_srt(&cfg.verified_srt_path(input), &verified).ok();

    anyhow::ensure!(
        !verified.is_empty(),
        "质检后没有任何有效语音内容（{} 条 ASR 输出全部被判为噪音/幻觉），跳过翻译。",
        raw_log.len()
    );

    if cfg.asr_only {
        let out = cfg.verified_srt_path(input);
        info!(
            "✔ --asr-only：只转录不翻译，已完成（{} 条）。原文 {:?}，质检后 {:?}",
            verified.len(),
            cfg.raw_srt_path(input),
            out
        );
        return Ok(out);
    }

    // ---------- 阶段 2: 翻译（复用同一 LLM 会话） ----------
    info!("▶ 阶段 2: 全局摘要提取与分批翻译...");
    let translated = translate_and_layout(session, prompts, cfg, verified)?;

    // ---------- 阶段 3: 输出 ----------
    let out_path = cfg.output_srt_path(input);
    srt::write_srt(&out_path, &translated)?;
    Ok(out_path)
}

// ============================================================================
// 模式 B：--no-qc 线性工作流（ASR 与 LLM 不同时驻留，内存峰值最低）
// ============================================================================

fn run_linear(
    files: &[std::path::PathBuf],
    cfg: &Config,
    prompts: &PromptStore,
    rt: &RuntimeInfo,
) -> Result<usize> {
    let mut failed = 0usize;
    for (i, input) in files.iter().enumerate() {
        banner(i, files.len(), input);
        match process_linear(input, cfg, prompts, rt) {
            Ok(out) => info!("✅ 文件处理完成，输出: {:?}", out),
            Err(e) => {
                failed += 1;
                error!("⚠️ 文件 {:?} 处理失败: {:#}", input, e);
            }
        }
    }
    Ok(failed)
}

fn process_linear(
    input: &Path,
    cfg: &Config,
    prompts: &PromptStore,
    rt: &RuntimeInfo,
) -> Result<std::path::PathBuf> {
    anyhow::ensure!(input.is_file(), "输入文件不存在: {:?}", input);

    let duration_secs = ffmpeg::get_duration_secs(input).unwrap_or(0.0);
    let segments = {
        let mut asr = new_asr(cfg, rt)?;
        let mut decoder = ffmpeg::spawn_decode_stream(input).context("启动 FFmpeg 解码流失败")?;
        let stdout = decoder.take_stdout().context("无法获取 FFmpeg stdout")?;
        let mut hook = |_seg: &mut SubtitleSegment| true;
        let segs = asr.transcribe_stream(stdout, duration_secs, &mut hook)?;
        decoder.finish()?;
        info!("✔ ASR 完成，共识别 {} 条字幕，正在卸载 ASR 模型...", segs.len());
        drop(asr); // 显式卸载，为 LLM 腾出内存/显存
        segs
    };
    srt::write_srt(&cfg.raw_srt_path(input), &segments).ok();
    anyhow::ensure!(!segments.is_empty(), "未识别到任何语音内容，跳过翻译。");

    if cfg.asr_only {
        let out = cfg.raw_srt_path(input);
        info!("✔ --asr-only：只转录不翻译，已完成（{} 条）-> {:?}", segments.len(), out);
        return Ok(out);
    }

    info!("▶ 阶段 2: 加载 LLM 并开始翻译...");
    let llm = new_llm(cfg, rt)?;
    let mut session = llm.session().context("创建 LLM 会话失败")?;
    let translated = translate_and_layout(&mut session, prompts, cfg, segments)?;
    drop(session);
    drop(llm);
    info!("✔ LLM 模型已卸载，内存已释放。");

    let out_path = cfg.output_srt_path(input);
    srt::write_srt(&out_path, &translated)?;
    Ok(out_path)
}

// ============================================================================
// 模式 C：--from-srt（跳过 ASR，直接翻译已有字幕）
// ============================================================================

fn run_from_srt(
    files: &[std::path::PathBuf],
    cfg: &Config,
    prompts: &PromptStore,
    rt: &RuntimeInfo,
) -> Result<usize> {
    info!("▶ --from-srt：跳过 ASR，加载 LLM 直接翻译已有字幕...");
    let llm = new_llm(cfg, rt)?;
    let mut session = llm.session().context("创建 LLM 会话失败")?;

    let mut failed = 0usize;
    for (i, input) in files.iter().enumerate() {
        banner(i, files.len(), input);
        match process_from_srt(input, cfg, prompts, &mut session) {
            Ok(out) => info!("✅ 文件处理完成，输出: {:?}", out),
            Err(e) => {
                failed += 1;
                error!("⚠️ 文件 {:?} 处理失败: {:#}", input, e);
            }
        }
    }
    Ok(failed)
}

fn process_from_srt(
    input: &Path,
    cfg: &Config,
    prompts: &PromptStore,
    session: &mut LlmSession,
) -> Result<std::path::PathBuf> {
    anyhow::ensure!(input.is_file(), "输入字幕文件不存在: {:?}", input);
    let mut segments = srt::read_srt(input)?;
    info!("✔ 已读入 {} 条字幕: {:?}", segments.len(), input);
    reindex(&mut segments);

    let out_path = cfg.output_srt_path(input);
    anyhow::ensure!(
        out_path != input,
        "输出路径与输入相同（{:?}），为避免覆盖原文件已中止。请用 --output-dir 指定其他目录。",
        out_path
    );

    let translated = translate_and_layout(session, prompts, cfg, segments)?;
    srt::write_srt(&out_path, &translated)?;
    Ok(out_path)
}

// ============================================================================
// 公共收尾：翻译 + 排版
// ============================================================================

fn translate_and_layout(
    session: &mut LlmSession,
    prompts: &PromptStore,
    cfg: &Config,
    segments: Vec<SubtitleSegment>,
) -> Result<Vec<SubtitleSegment>> {
    let before = segments.len();
    let mut translator = Translator::new(session, prompts, cfg);
    let mut translated = translator.translate(segments)?;
    drop(translator);

    if cfg.layout_enabled {
        translated = srt::layout(&translated, cfg.max_line_width, cfg.max_cue_secs);
        info!(
            "✔ 排版完成：{} 条 -> {} 条（行宽 {}，单条最长 {}s）",
            before,
            translated.len(),
            cfg.max_line_width,
            cfg.max_cue_secs
        );
    }
    anyhow::ensure!(!translated.is_empty(), "翻译后没有任何字幕可写");
    Ok(translated)
}

fn reindex(segments: &mut [SubtitleSegment]) {
    for (i, s) in segments.iter_mut().enumerate() {
        s.index = i + 1;
    }
}
