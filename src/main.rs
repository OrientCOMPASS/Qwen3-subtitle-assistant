mod asr;
mod cli;
mod config;
mod ffmpeg;
mod llm;
mod prompt;
mod srt;
mod translate;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use log::{error, info};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let args = cli::Args::parse();
    if let Err(e) = run(args) {
        error!("执行失败: {:#}", e);
        std::process::exit(1);
    }
}

fn run(args: cli::Args) -> Result<()> {
    let cfg = config::Config::from_args(&args)?;
    let prompts = prompt::PromptStore::new(&cfg.prompts_dir);

    info!("共接收到 {} 个文件待处理", args.files.len());

    for (i, input) in args.files.iter().enumerate() {
        info!("========================================");
        info!("[{}/{}] 开始处理: {:?}", i + 1, args.files.len(), input);
        info!("========================================");

        match process_one(input, &cfg, &prompts) {
            Ok(out) => info!("✅ 文件处理完成，输出: {:?}", out),
            Err(e) => error!("⚠️ 文件 {:?} 处理失败: {:#}", input, e),
        }
    }

    info!("所有文件处理流程结束。");
    Ok(())
}

/// 线性工作流：加载 ASR -> 转录 -> 卸载 ASR -> 加载 LLM -> 翻译 -> 卸载 LLM
fn process_one(
    input: &std::path::Path,
    cfg: &config::Config,
    prompts: &prompt::PromptStore,
) -> Result<std::path::PathBuf> {
    anyhow::ensure!(input.exists(), "输入文件不存在: {:?}", input);

    // --- 阶段 1: ASR 转录 ---
    info!("▶ 阶段 1: 加载 ASR 模型并开始转录...");
    let mut asr = asr::AsrEngine::new(&cfg.asr_model_dir, &cfg.vad_model)
        .context("初始化 ASR 引擎失败")?;

    let duration_secs = ffmpeg::get_duration_secs(input).unwrap_or(0.0);
    let mut ffmpeg_child = ffmpeg::spawn_decode_stream(input)
        .context("启动 FFmpeg 解码流失败")?;

    let stdout = ffmpeg_child.stdout.take().context("无法获取 FFmpeg stdout")?;
    let raw_segments = asr.transcribe_stream(stdout, duration_secs)?;

    let status = ffmpeg_child.wait().context("等待 FFmpeg 退出失败")?;
    if !status.success() {
        anyhow::bail!("FFmpeg 解码异常退出");
    }

    // 显式卸载 ASR 模型以释放内存 (asr 没有借用其他短生命周期变量，直接 drop 没问题)
    drop(asr);
    info!("✔ ASR 模型已卸载，内存已释放。共识别 {} 条字幕。", raw_segments.len());

    if raw_segments.is_empty() {
        anyhow::bail!("未识别到任何语音内容，跳过翻译。");
    }

    let raw_srt_path = config::output_srt_path(input).with_extension("raw.srt");
    srt::write_srt(&raw_srt_path, &raw_segments).ok();

    // --- 阶段 2: LLM 翻译 ---
    info!("▶ 阶段 2: 加载翻译模型并开始翻译...");
    let llm = llm::LlmClient::new(&cfg.llm_model).context("加载 LLM 翻译模型失败")?;
    let translator = translate::Translator::new(&llm, prompts, cfg);
    
    let translated = translator.translate(raw_segments)?;

    // 显式卸载 LLM 模型
    // ⚠️ 修复点：必须先 drop 借用了 llm 的 translator，再 drop llm 本身
    drop(translator);
    drop(llm);
    info!("✔ LLM 模型已卸载，内存已释放。");

    // --- 阶段 3: 输出 SRT ---
    let out_path = config::output_srt_path(input);
    srt::write_srt(&out_path, &translated)?;

    Ok(out_path)
}