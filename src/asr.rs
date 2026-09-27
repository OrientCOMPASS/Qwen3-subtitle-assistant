//! S2TT 单模型转录管线（流式）：ffmpeg PCM 流 → 纯 Rust silero VAD **边切边送** →
//! GgufAsr（llama.cpp/mtmd，context 任务开关直出目标语言）→ 字幕段。
//!
//! v0.5 起为生产者/消费者流式管线：生产者线程读 ffmpeg PCM、跑 VAD，**每确认
//! 一个语音段立即经有界 channel 送出**；消费者（主线程）拿到一段就 ASR 一段。
//! 与旧版「VAD 切完全片再整批 ASR」相比：
//! * 首条字幕的等待时间 ≈ 第一句话的长度（而非全片 VAD 时长）；
//! * 内存有界（channel 容量 + VAD 窗口），不再全片缓冲；
//! * 进度条只有一条，按已解码媒体时间走，消息里带已转写段数。
//!
//! 丢弃策略（默认最小化）：空文本段（模型判定纯静音/噪音）无内容可写、始终跳过；
//! 与上一条保留字幕完全相同的段（重复音频/幻觉循环的常见输出）始终丢弃；
//! 「lang=None 短碎片=幻觉」启发式过滤默认**关闭**，`--filter-fragments` 显式开启
//! （实测结论见 finetune/s2tt_pipeline.py：lang=None 且 <2s 或 <4 字的段多为
//! 半幻觉碎片，但默认保留用户对产出的完整控制权）。

use crate::config::Config;
use crate::ffmpeg::{self, SAMPLE_RATE};
use crate::gguf_asr::{AsrUtterance, GgufAsr};
use crate::srt::format_timestamp;
use crate::types::SubtitleSegment;
use crate::vad::{VadSeg, VadSegmenter};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use std::io::Read;
use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender};

/// `--filter-fragments` 开启时，lang=None 但带文本的段的保留门槛（实测结论，
/// 见 finetune/s2tt_pipeline.py）：<2s 或 <4 字是半幻觉碎片（"今天"/"请 everyone"），
/// ≥ 门槛是真实语音（结尾致辞）。
const KEEP_LANG_NONE_MIN_SECS: f64 = 2.0;
const KEEP_LANG_NONE_MIN_CHARS: usize = 4;

/// 单段处置判定（纯函数，可单测）。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KeepDecision {
    /// 保留为字幕
    Keep,
    /// 空文本（模型判定纯静音/噪音）——无内容可写，始终跳过
    EmptyText,
    /// 与上一条保留字幕完全相同（模型对重复音频/幻觉循环的常见输出）——丢弃
    Duplicate,
    /// lang=None 短碎片且 `--filter-fragments` 开启——丢弃
    Fragment,
}

/// 判定一段 ASR 输出的去留。`last_kept` 为上一条**保留**字幕的文本（trim 后）。
///
/// 判定顺序：空文本 → 与上一条重复 → 碎片过滤（仅 flag 开启时）。
/// 去重是 v0.5.1 起的默认行为（用户要求）：流式管线里长静音/循环背景音常让
/// 模型连续吐出同一句话，原样进字幕是最刺眼的坏 case。
fn classify_utterance(u: &AsrUtterance, dur_secs: f64, filter_fragments: bool,
                      last_kept: Option<&str>) -> KeepDecision {
    let text = u.text.trim();
    if text.is_empty() {
        return KeepDecision::EmptyText;
    }
    if let Some(prev) = last_kept {
        if text == prev {
            return KeepDecision::Duplicate;
        }
    }
    if filter_fragments && u.lang.eq_ignore_ascii_case("None") {
        let chars = text.chars().filter(|c| !c.is_whitespace()).count();
        if dur_secs < KEEP_LANG_NONE_MIN_SECS || chars < KEEP_LANG_NONE_MIN_CHARS {
            return KeepDecision::Fragment;
        }
    }
    KeepDecision::Keep
}

/// VAD→ASR 段队列容量（背压：ASR 慢时生产者阻塞，ffmpeg 管道自然限流，
/// 内存占用 = 容量 × 单段上限 ≈ 8 × 3.8MB，有界）。
const SEG_CHANNEL_CAP: usize = 8;

/// 转录一个媒体文件（流式），返回排版前的字幕段（按时间有序，index 已分配）。
pub fn transcribe_file(input: &Path, cfg: &Config, asr: &mut GgufAsr) -> Result<Vec<SubtitleSegment>> {
    let duration = ffmpeg::get_duration_secs(input).unwrap_or(0.0);

    let pb = make_progress(duration);
    let (tx, rx) = sync_channel::<Result<VadSeg>>(SEG_CHANNEL_CAP);

    // ---- 生产者线程：ffmpeg 解码 + 流式 VAD，切出一段送一段 ----
    let pb_prod = pb.clone();
    let input_owned = input.to_path_buf();
    let (vad_threshold, vad_min_silence, vad_buffer_secs) =
        (cfg.vad_threshold, cfg.vad_min_silence, cfg.vad_buffer_secs);
    let producer = std::thread::Builder::new()
        .name("vad-feed".into())
        .spawn(move || {
            let result = vad_feed(&input_owned, duration, vad_threshold, vad_min_silence,
                                  vad_buffer_secs, &tx, &pb_prod);
            if let Err(e) = result {
                // 消费者可能已因 ASR 错误退出（rx 断开）——发送失败就静默收尾
                let _ = tx.send(Err(e));
            }
            // tx 在此 drop → 消费者的 rx 迭代自然结束
        })
        .context("启动 VAD 生产者线程失败")?;

    // ---- 消费者（主线程）：收一段、ASR 一段 ----
    let mut out: Vec<SubtitleSegment> = Vec::new();
    let mut n_segs = 0usize;   // VAD 确认的语音段总数
    let mut n_empty = 0usize;  // 空文本段（模型判定静音/噪音，无内容可写）
    let mut n_dup = 0usize;    // 与上一条字幕重复的段
    let mut n_frag = 0usize;   // --filter-fragments 丢弃的碎片段
    let mut last_kept: Option<String> = None; // 上一条保留字幕（去重基准）
    let mut pipeline_err: Option<anyhow::Error> = None;

    for item in &rx {
        let seg = match item {
            Ok(s) => s,
            Err(e) => { pipeline_err = Some(e); break; }
        };
        n_segs += 1;
        let dur_secs = seg.samples.len() as f64 / SAMPLE_RATE as f64;
        let u: AsrUtterance = match asr.transcribe(&seg.samples, &cfg.context) {
            Ok(u) => u,
            Err(e) => { pipeline_err = Some(e); break; }
        };
        let start_ms = ms_of(seg.start_sample);
        let end_ms = ms_of(seg.start_sample + seg.samples.len());

        match classify_utterance(&u, dur_secs, cfg.filter_fragments, last_kept.as_deref()) {
            KeepDecision::EmptyText => {
                n_empty += 1;
                pb.suspend(|| debug!("[ASR∅] {:.2}s@{} 空输出（lang={}），跳过",
                                     dur_secs, format_timestamp(start_ms), u.lang));
            }
            KeepDecision::Duplicate => {
                n_dup += 1;
                pb.suspend(|| debug!("[ASR↻] {:.2}s@{} 与上一条相同，丢弃（{:?}）",
                                     dur_secs, format_timestamp(start_ms),
                                     truncate(u.text.trim(), 20)));
            }
            KeepDecision::Fragment => {
                n_frag += 1;
                pb.suspend(|| debug!("[VAD✂] {:.2}s@{} 丢弃（lang={} text={:?}）",
                                     dur_secs, format_timestamp(start_ms), u.lang,
                                     truncate(&u.text, 20)));
            }
            KeepDecision::Keep => {
                let text = u.text.trim().to_string();
                // 保留的字幕内容 = 用户关心的产出，info 级直接打到终端
                //（v0.4 的行为；被收敛的只是加载噪声与跳过明细那类诊断）。
                pb.suspend(|| info!("[ASR✔] {} ({:.1}s) {}",
                                    format_timestamp(start_ms), dur_secs, text));
                last_kept = Some(text.clone());
                out.push(SubtitleSegment {
                    index: out.len() + 1,
                    start_ms,
                    end_ms,
                    text,
                });
                pb.set_message(format!("已转写 {} 段", out.len()));
            }
        }
    }

    // 收尾：drop(rx) 解除生产者可能的 send 阻塞；join 回收线程
    drop(rx);
    let _ = producer.join();
    pb.finish_with_message("转录完成");

    if let Some(e) = pipeline_err {
        return Err(e);
    }
    if n_segs == 0 {
        warn!("未检出任何语音段（静音/纯音乐？）");
        return Ok(Vec::new());
    }
    let mut notes: Vec<String> = Vec::new();
    if n_empty > 0 {
        notes.push(format!("{n_empty} 段空输出跳过"));
    }
    if n_dup > 0 {
        notes.push(format!("{n_dup} 段重复丢弃"));
    }
    if cfg.filter_fragments && n_frag > 0 {
        notes.push(format!("碎片过滤丢弃 {n_frag} 段"));
    }
    let tail = if notes.is_empty() { String::new() } else { format!("（{}）", notes.join("，")) };
    info!("转录完成：{n_segs} 段语音 → {} 条字幕{tail}", out.len());
    Ok(out)
}

/// 生产者主体：ffmpeg PCM 流 → f32 样本 → VadSegmenter.accept（返回即发送）。
fn vad_feed(input: &Path, duration: f64, threshold: f32, min_silence: f32, buffer_secs: f32,
            tx: &SyncSender<Result<VadSeg>>, pb: &ProgressBar) -> Result<()> {
    let mut stream = ffmpeg::spawn_decode_stream(input)
        .with_context(|| format!("启动 ffmpeg 解码失败: {:?}", input))?;
    let mut reader = stream
        .take_stdout()
        .context("无法取得 ffmpeg stdout（内部错误）")?;

    let mut segmenter = VadSegmenter::new(threshold, min_silence, buffer_secs)
        .context("初始化 VAD 失败")?;

    // ---- 读取 f32le PCM 流喂 VAD（跨 read 边界保留半个采样，逻辑承自旧管线） ----
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
        // 流式核心：accept 返回的就是「此刻已确认」的段，立即送 ASR 队列
        for seg in segmenter.accept(&samples_buf) {
            if tx.send(Ok(seg)).is_err() {
                return Ok(()); // 消费者已退出（出错），静默收尾；DecodeStream Drop 会杀 ffmpeg
            }
        }
        if duration > 0.0 {
            pb.set_position((total_bytes / 4 / SAMPLE_RATE as u64).min(duration as u64));
        }
    }
    stream.finish().context("ffmpeg 进程未正常结束")?;

    // 冲刷流尾未闭合段
    for seg in segmenter.finish() {
        if tx.send(Ok(seg)).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

fn make_progress(duration: f64) -> ProgressBar {
    if duration > 0.0 {
        let pb = ProgressBar::new(duration as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}s/{len}s · {msg} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("VAD+ASR 流式转录");
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message("VAD+ASR 流式转录");
        pb
    }
}

fn ms_of(sample: usize) -> u64 {
    (sample as u64 * 1000) / SAMPLE_RATE as u64
}

fn truncate(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n { format!("{t}…") } else { t }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(lang: &str, text: &str) -> AsrUtterance {
        AsrUtterance { lang: lang.into(), text: text.into() }
    }

    #[test]
    fn classify_empty_text_always_skipped() {
        assert_eq!(classify_utterance(&u("None", ""), 5.0, false, None), KeepDecision::EmptyText);
        assert_eq!(classify_utterance(&u("Japanese", "   "), 5.0, true, None), KeepDecision::EmptyText);
    }

    #[test]
    fn classify_duplicate_of_last_kept_dropped() {
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好。"), 3.0, false, Some("今天天气很好。")),
                   KeepDecision::Duplicate);
        // 内容不同 → 保留
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好！"), 3.0, false, Some("今天天气很好。")),
                   KeepDecision::Keep);
        // 没有上一条 → 保留
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好。"), 3.0, false, None),
                   KeepDecision::Keep);
        // 首尾空白等价（trim 后比较）
        assert_eq!(classify_utterance(&u("Chinese", " 今天天气很好。 "), 3.0, false, Some("今天天气很好。")),
                   KeepDecision::Duplicate);
    }

    #[test]
    fn classify_fragment_only_when_flag_on() {
        // lang=None + 短（<2s）→ 过滤开启才丢
        assert_eq!(classify_utterance(&u("None", "今天"), 0.8, true, None), KeepDecision::Fragment);
        assert_eq!(classify_utterance(&u("None", "今天"), 0.8, false, None), KeepDecision::Keep);
        // lang=None 但长段（真实语音，如结尾致辞）→ 过滤开启也保留
        assert_eq!(classify_utterance(&u("None", "谢谢大家的观看"), 5.0, true, None), KeepDecision::Keep);
        // lang 有值的短段不受过滤影响
        assert_eq!(classify_utterance(&u("Japanese", "はい"), 0.5, true, None), KeepDecision::Keep);
    }

    #[test]
    fn classify_priority_empty_then_dup_then_fragment() {
        // 空文本优先于一切
        assert_eq!(classify_utterance(&u("None", " "), 0.5, true, Some("x")), KeepDecision::EmptyText);
        // 重复优先于碎片
        assert_eq!(classify_utterance(&u("None", "今天"), 0.5, true, Some("今天")), KeepDecision::Duplicate);
    }
}
