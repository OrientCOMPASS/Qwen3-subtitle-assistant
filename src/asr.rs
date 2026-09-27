//! S2TT 单模型转录管线：ffmpeg PCM 流 → 纯 Rust silero VAD 分段 →
//! GgufAsr（llama.cpp/mtmd，context 任务开关直出目标语言）→ 丢弃策略 → 字幕段。
//!
//! 与旧两段式管线（sherpa ASR → LLM 质检/摘要/翻译/审校）的区别：没有任何 LLM
//! 参与；噪音/静音段靠模型自身的「language None → 空输出」行为剔除（微调时用
//! 静音样本专门保住并双模式验证过，见 finetune/README §10）。

use crate::config::Config;
use crate::ffmpeg::{self, SAMPLE_RATE};
use crate::gguf_asr::{AsrUtterance, GgufAsr};
use crate::srt::format_timestamp;
use crate::types::SubtitleSegment;
use crate::vad::VadSegmenter;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use std::io::Read;
use std::path::Path;

/// lang=None 但带文本的段的保留门槛（实测结论，见 finetune/s2tt_pipeline.py）：
/// <2s 或 <4 字是半幻觉碎片（"今天"/"请 everyone"），≥ 门槛是真实语音（结尾致辞）。
const KEEP_LANG_NONE_MIN_SECS: f64 = 2.0;
const KEEP_LANG_NONE_MIN_CHARS: usize = 4;

/// 转录一个媒体文件，返回排版前的字幕段（按时间有序，index 已分配）。
pub fn transcribe_file(input: &Path, cfg: &Config, asr: &mut GgufAsr) -> Result<Vec<SubtitleSegment>> {
    let duration = ffmpeg::get_duration_secs(input).unwrap_or(0.0);
    let mut stream = ffmpeg::spawn_decode_stream(input)
        .with_context(|| format!("启动 ffmpeg 解码失败: {:?}", input))?;
    let reader = stream
        .take_stdout()
        .context("无法取得 ffmpeg stdout（内部错误）")?;

    let mut segmenter = VadSegmenter::new(cfg.vad_threshold, cfg.vad_min_silence, cfg.vad_buffer_secs)
        .context("初始化 VAD 失败")?;

    let pb = if duration > 0.0 {
        let pb = ProgressBar::new(duration as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] VAD {pos}s/{len}s ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    } else {
        ProgressBar::new_spinner()
    };

    // ---- 读取 f32le PCM 流喂 VAD（跨 read 边界保留半个采样，逻辑承自旧管线） ----
    let mut reader = reader;
    let mut read_buf = [0u8; 65536];
    let mut leftover = [0u8; 4];
    let mut leftover_len = 0usize;
    let mut samples_buf: Vec<f32> = Vec::with_capacity(16384);
    let mut total_bytes: u64 = 0;
    loop {
        let n = reader.read(&mut read_buf).context("读取 ffmpeg 音频流失败")?;
        if n == 0 {
            break;
        }
        total_bytes += n as u64;
        let mut bytes = Vec::with_capacity(leftover_len + n);
        if leftover_len > 0 {
            bytes.extend_from_slice(&leftover[..leftover_len]);
            leftover_len = 0;
        }
        bytes.extend_from_slice(&read_buf[..n]);
        samples_buf.clear();
        let mut off = 0usize;
        while off + 4 <= bytes.len() {
            samples_buf.push(f32::from_le_bytes([
                bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
            ]));
            off += 4;
        }
        if off < bytes.len() {
            let rest = &bytes[off..];
            leftover[..rest.len()].copy_from_slice(rest);
            leftover_len = rest.len();
        }
        segmenter.accept(&samples_buf);
        if duration > 0.0 {
            pb.set_position((total_bytes / 4 / SAMPLE_RATE as u64).min(duration as u64));
        }
    }
    stream.finish().context("ffmpeg 进程未正常结束")?;
    pb.finish_with_message("VAD 完成");

    let segs = segmenter.finish();
    if segs.is_empty() {
        warn!("未检出任何语音段（静音/纯音乐？）");
        return Ok(Vec::new());
    }

    // ---- 逐段 S2TT 直出 ----
    let pb2 = if duration > 0.0 {
        let pb = ProgressBar::new(segs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ASR {pos}/{len} 段 ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    } else {
        ProgressBar::new_spinner()
    };

    let mut out: Vec<SubtitleSegment> = Vec::new();
    let mut dropped = 0usize;
    for (start_sample, samples) in &segs {
        let u: AsrUtterance = asr.transcribe(samples, &cfg.context)?;
        let dur_secs = samples.len() as f64 / SAMPLE_RATE as f64;
        let chars = u.text.chars().filter(|c| !c.is_whitespace()).count();
        // 丢弃策略（与 finetune/s2tt_pipeline.py 实测结论一致）
        let drop = u.text.trim().is_empty()
            || (u.lang.eq_ignore_ascii_case("None")
                && (dur_secs < KEEP_LANG_NONE_MIN_SECS || chars < KEEP_LANG_NONE_MIN_CHARS));
        pb2.inc(1);
        if drop {
            dropped += 1;
            info!("[VAD✂] {:.2}s@{} 丢弃（lang={} text={:?}）",
                  dur_secs, format_timestamp(ms_of(*start_sample)), u.lang, truncate(&u.text, 20));
            continue;
        }
        let start_ms = ms_of(*start_sample);
        let end_ms = ms_of(*start_sample + samples.len());
        let seg = SubtitleSegment {
            index: out.len() + 1,
            start_ms,
            end_ms,
            text: u.text.trim().to_string(),
        };
        info!("[ASR✔] {} ({:.1}s) {}", format_timestamp(start_ms), dur_secs, truncate(&seg.text, 46));
        out.push(seg);
    }
    pb2.finish_with_message("转录完成");
    info!("转录结束：{} 段语音，保留 {} 条，丢弃 {} 条（静音/幻觉碎片）",
          segs.len(), out.len(), dropped);
    Ok(out)
}

fn ms_of(sample: usize) -> u64 {
    (sample as u64 * 1000) / SAMPLE_RATE as u64
}

fn truncate(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n { format!("{t}…") } else { t }
}
