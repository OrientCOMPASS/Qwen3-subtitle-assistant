//! S2TT 单模型转录管线（流式）：ffmpeg PCM 流 → 纯 Rust silero VAD **边切边送** →
//! GgufAsr（llama.cpp/mtmd，context 任务开关直出目标语言）→ 字幕段。
//!
//! v0.5 起为生产者/消费者流式管线：生产者线程读 ffmpeg PCM、跑 VAD，**每确认
//! 一个语音段立即送入滞回缓冲队列**；消费者（主线程）拿到一段就 ASR 一段——
//! 两个推理天然并行。v0.6 队列改滞回流控（用户指定）：满（8 段）时 VAD 暂停
//! 生产，消费到半容量（4 段）才恢复，减少阻塞/唤醒抖动。
//! 与旧版「VAD 切完全片再整批 ASR」相比：
//! * 首条字幕的等待时间 ≈ 第一句话的长度（而非全片 VAD 时长）；
//! * 内存有界（队列容量 + VAD 窗口），不再全片缓冲；
//! * 进度条只有一条，按已解码媒体时间走，消息里带已转写段数。
//!
//! 丢弃策略（默认最小化）：空文本段（模型判定纯静音/噪音）无内容可写、始终跳过；
//! 与上一条保留字幕相同的段（**去标点后**比较；重复音频/幻觉循环的常见输出）、
//! 以及「上一条与本次都是纯语气词」的连续语气词段（哎/啊/嗯/哼…），始终丢弃；
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
use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

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
    /// 与上一条保留字幕相同（**去标点后**比较；模型对重复音频/幻觉循环的
    /// 常见输出）——丢弃
    Duplicate,
    /// 上一条与本次（去标点后）都是纯语气词——连续语气词是静音/背景音上最
    /// 典型的幻觉形态，丢弃本次（第一条语气词保留，不误伤真实的"嗯。"应答）
    FillerRepeat,
    /// lang=None 短碎片且 `--filter-fragments` 开启——丢弃
    Fragment,
}

/// 语气词集合（用户例举"哎啊嗯哼等"，取常见闭集；命中规则=去标点后**每个字**
/// 都在集合内）。刻意保守：不收"好/对/是"这类有实义应答词。
const FILLER_CHARS: &[char] = &[
    '哎', '唉', '啊', '呀', '哇', '哦', '噢', '嗯', '哼', '唔',
    '哈', '嘿', '呃', '呐', '欸', '诶', '嚄', '呗',
];

/// 去掉标点与空白（Unicode P* 全类 + 空白类），用于判重与语气词判定：
/// "你好。"与"你好，"视为同一句；"嗯……"归一成"嗯"。
fn strip_punct(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && !is_punct(*c))
        .collect()
}

/// ASCII 标点 + Unicode 常见标点区（含 CJK 标点/全角形式/省略号/引号等）。
fn is_punct(c: char) -> bool {
    let u = c as u32;
    c.is_ascii_punctuation()
        || matches!(u,
            0x2000..=0x206F   // 通用标点（含 … ‥）
          | 0x3000..=0x303F   // CJK 标点（。、《》〈〉…）
          | 0xFF00..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40 | 0xFF5B..=0xFF65 // 全角标点
          | 0xFE30..=0xFE4F   // CJK 兼容形式（竖排标点）
          | 0x00B7            // ·
        )
}

/// 去标点后是否为「纯语气词」：非空且每个字都在 FILLER_CHARS 里。
fn is_pure_filler(stripped: &str) -> bool {
    !stripped.is_empty()
        && stripped.chars().all(|c| FILLER_CHARS.contains(&c))
}

/// 判定一段 ASR 输出的去留。`last_kept_stripped` 为上一条**保留**字幕去标点
/// 后的文本。判定顺序：空文本 → 重复（去标点）→ 连续语气词 → 碎片过滤
/// （仅 flag 开启时）。重复与语气词规则是默认行为（用户要求）：流式管线里
/// 长静音/循环背景音常让模型连续吐出同一句话或纯语气词，原样进字幕是最刺眼
/// 的坏 case。
fn classify_utterance(u: &AsrUtterance, dur_secs: f64, filter_fragments: bool,
                      last_kept_stripped: Option<&str>) -> KeepDecision {
    let text = u.text.trim();
    if text.is_empty() {
        return KeepDecision::EmptyText;
    }
    let stripped = strip_punct(text);
    if stripped.is_empty() {
        // 全是标点（如"……。"）：没有字幕内容可写，按空文本处理
        return KeepDecision::EmptyText;
    }
    if let Some(prev) = last_kept_stripped {
        // 契约上 prev 已是去标点文本；再 strip 一次幂等且防御调用方状态漂移
        let prev = strip_punct(prev);
        if stripped == prev {
            return KeepDecision::Duplicate;
        }
        if is_pure_filler(&prev) && is_pure_filler(&stripped) {
            return KeepDecision::FillerRepeat;
        }
    }
    if filter_fragments && u.lang.eq_ignore_ascii_case("None") {
        let chars = stripped.chars().count();
        if dur_secs < KEEP_LANG_NONE_MIN_SECS || chars < KEEP_LANG_NONE_MIN_CHARS {
            return KeepDecision::Fragment;
        }
    }
    KeepDecision::Keep
}

/// VAD→ASR 段队列容量（高水位）。背压：ASR 慢时生产者阻塞，ffmpeg 管道自然
/// 限流，内存占用 = 容量 × 单段上限 ≈ 8 × 3.8MB，有界。
const SEG_QUEUE_CAP: usize = 8;

/// 滞回缓冲队列：VAD（生产者线程）与 ASR（消费者主线程）之间的段队列。
///
/// 流控语义（用户指定）：**满（len ≥ cap）时生产者暂停**，直到消费掉一半
/// （len ≤ cap/2）才恢复生产——高低水位滞回，避免"满一格放一格"的高频
/// 阻塞/唤醒抖动，让 VAD 有机会成批跑在 ASR 前面。
///
/// 注：VAD 与 ASR 本就分别跑在生产者/消费者线程上（v0.5 起真并行）；
/// 本队列替换的是原来的 sync_channel（其为"一格"语义，无滞回）。
struct SegQueue {
    inner: Mutex<QueueInner>,
    /// 生产者等待：队列降到半容量以下
    cv_drained: Condvar,
    /// 消费者等待：有新段或队列关闭
    cv_item: Condvar,
    cap: usize,
    low_water: usize,
}

struct QueueInner {
    q: VecDeque<Result<VadSeg>>,
    closed: bool,
}

impl SegQueue {
    fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(QueueInner { q: VecDeque::new(), closed: false }),
            cv_drained: Condvar::new(),
            cv_item: Condvar::new(),
            cap,
            low_water: (cap / 2).max(1),
        })
    }

    /// 生产者压入一段。返回 false = 队列已关闭（消费者出错退出），生产者应停止。
    fn push(&self, item: Result<VadSeg>) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        while g.q.len() >= self.cap && !g.closed {
            g = self.cv_drained.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        if g.closed {
            return false; // 消费者已弃队列（出错提前退出）
        }
        g.q.push_back(item);
        drop(g);
        self.cv_item.notify_one();
        true
    }

    /// 消费者取出下一段；None = 生产结束且队列已排空。
    fn pop(&self) -> Option<Result<VadSeg>> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(item) = g.q.pop_front() {
                let drained = g.q.len() <= self.low_water;
                drop(g);
                if drained {
                    self.cv_drained.notify_all(); // 半容量滞回点：放行生产者
                }
                return Some(item);
            }
            if g.closed {
                return None;
            }
            g = self.cv_item.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// 生产者收尾：标记关闭，唤醒可能在等的消费者。
    fn close(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.closed = true;
        drop(g);
        self.cv_item.notify_all();
        self.cv_drained.notify_all();
    }

    /// 消费者出错提前退出：关闭队列并丢弃积压，解除生产者的满队列阻塞。
    fn abort(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.closed = true;
        g.q.clear();
        drop(g);
        self.cv_drained.notify_all();
        self.cv_item.notify_all();
    }
}

/// 转录一个媒体文件（流式），返回排版前的字幕段（按时间有序，index 已分配）。
pub fn transcribe_file(input: &Path, cfg: &Config, asr: &mut GgufAsr) -> Result<Vec<SubtitleSegment>> {
    let duration = ffmpeg::get_duration_secs(input).unwrap_or(0.0);

    let pb = make_progress(duration);
    let queue = SegQueue::new(SEG_QUEUE_CAP);

    // ---- 生产者线程：ffmpeg 解码 + 流式 VAD，切出一段送一段 ----
    let pb_prod = pb.clone();
    let input_owned = input.to_path_buf();
    let (vad_threshold, vad_min_silence, vad_buffer_secs) =
        (cfg.vad_threshold, cfg.vad_min_silence, cfg.vad_buffer_secs);
    let q_prod = Arc::clone(&queue);
    let producer = std::thread::Builder::new()
        .name("vad-feed".into())
        .spawn(move || {
            let result = vad_feed(&input_owned, duration, vad_threshold, vad_min_silence,
                                  vad_buffer_secs, &q_prod, &pb_prod);
            if let Err(e) = result {
                // 消费者可能已因 ASR 错误退出（队列 closed）——push 返回 false 就静默收尾
                let _ = q_prod.push(Err(e));
            }
            q_prod.close(); // 消费者 pop() 收到 None，迭代自然结束
        })
        .context("启动 VAD 生产者线程失败")?;

    // ---- 消费者（主线程）：收一段、ASR 一段 ----
    let mut out: Vec<SubtitleSegment> = Vec::new();
    let mut n_segs = 0usize;   // VAD 确认的语音段总数
    let mut n_empty = 0usize;  // 空文本段（模型判定静音/噪音，无内容可写）
    let mut n_dup = 0usize;    // 与上一条字幕重复的段
    let mut n_frag = 0usize;   // --filter-fragments 丢弃的碎片段
    let mut n_filler = 0usize; // 连续纯语气词丢弃的段
    let mut last_kept_stripped: Option<String> = None; // 上一条保留字幕（去标点，判重/语气词基准）
    let mut pipeline_err: Option<anyhow::Error> = None;

    while let Some(item) = queue.pop() {
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

        match classify_utterance(&u, dur_secs, cfg.filter_fragments, last_kept_stripped.as_deref()) {
            KeepDecision::EmptyText => {
                n_empty += 1;
                pb.suspend(|| debug!("[ASR∅] {:.2}s@{} 空输出（lang={}），跳过",
                                     dur_secs, format_timestamp(start_ms), u.lang));
            }
            KeepDecision::Duplicate => {
                n_dup += 1;
                pb.suspend(|| debug!("[ASR↻] {:.2}s@{} 与上一条相同（去标点比较），丢弃（{:?}）",
                                     dur_secs, format_timestamp(start_ms),
                                     truncate(u.text.trim(), 20)));
            }
            KeepDecision::FillerRepeat => {
                n_filler += 1;
                pb.suspend(|| debug!("[ASR〰] {:.2}s@{} 连续纯语气词，丢弃（{:?}）",
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
                last_kept_stripped = Some(strip_punct(&text));
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

    // 收尾：出错提前 break 时 abort 解除生产者的满队列阻塞；join 回收线程
    if pipeline_err.is_some() {
        queue.abort();
    }
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
    if n_filler > 0 {
        notes.push(format!("{n_filler} 段连续语气词丢弃"));
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
            queue: &SegQueue, pb: &ProgressBar) -> Result<()> {
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
        //（队列满时 push 在此阻塞 = VAD 暂停生产，直到消费掉一半容量）
        for seg in segmenter.accept(&samples_buf) {
            if !queue.push(Ok(seg)) {
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
        if !queue.push(Ok(seg)) {
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
        // 全标点 = 无内容可写，同样按空处理
        assert_eq!(classify_utterance(&u("Chinese", "……。"), 5.0, false, None), KeepDecision::EmptyText);
    }

    #[test]
    fn classify_duplicate_ignores_punctuation() {
        // 去标点后的判重：标点差异不再放行重复句
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好。"), 3.0, false, Some("今天天气很好")),
                   KeepDecision::Duplicate);
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好，"), 3.0, false, Some("今天天气很好。")),
                   KeepDecision::Duplicate, "基准本身带标点也应命中（调用方存的是去标点文本，此处双保险）");
        // 内容不同 → 保留
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好！"), 3.0, false, Some("明天天气很好")),
                   KeepDecision::Keep);
        // 没有上一条 → 保留
        assert_eq!(classify_utterance(&u("Chinese", "今天天气很好。"), 3.0, false, None),
                   KeepDecision::Keep);
        // 首尾/内部空白等价
        assert_eq!(classify_utterance(&u("Chinese", " 今天 天气很好。 "), 3.0, false, Some("今天天气很好")),
                   KeepDecision::Duplicate);
    }

    #[test]
    fn classify_filler_repeat_dropped_but_single_kept() {
        // 第一条语气词（无上一条）→ 保留（不误伤真实应答）
        assert_eq!(classify_utterance(&u("Chinese", "嗯。"), 1.0, false, Some("我们出发吧")),
                   KeepDecision::Keep);
        // 上一条与本次都是纯语气词 → 丢弃
        assert_eq!(classify_utterance(&u("Chinese", "嗯嗯。"), 1.0, false, Some("嗯")),
                   KeepDecision::FillerRepeat);
        assert_eq!(classify_utterance(&u("Chinese", "啊……"), 0.8, false, Some("哎")),
                   KeepDecision::FillerRepeat);
        // 上一条是语气词但本次是实义句 → 保留
        assert_eq!(classify_utterance(&u("Chinese", "太阳会升起。"), 2.0, false, Some("嗯")),
                   KeepDecision::Keep);
        // 上一条是实义句、本次是语气词 → 保留（首条语气词合法）
        assert_eq!(classify_utterance(&u("Chinese", "哎呀！"), 1.0, false, Some("太阳会升起")),
                   KeepDecision::Keep);
        // 语气词+实义混合不算纯语气词
        assert_eq!(classify_utterance(&u("Chinese", "嗯好的。"), 1.0, false, Some("嗯")),
                   KeepDecision::Keep);
        // 重复判定优先于语气词判定（完全相同的语气词 = Duplicate）
        assert_eq!(classify_utterance(&u("Chinese", "嗯。"), 1.0, false, Some("嗯")),
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
    fn classify_priority_empty_then_dup_then_filler_then_fragment() {
        // 空文本优先于一切
        assert_eq!(classify_utterance(&u("None", " "), 0.5, true, Some("x")), KeepDecision::EmptyText);
        // 重复优先于语气词与碎片
        assert_eq!(classify_utterance(&u("None", "今天"), 0.5, true, Some("今天")), KeepDecision::Duplicate);
        // 语气词优先于碎片
        assert_eq!(classify_utterance(&u("None", "嗯"), 0.5, true, Some("啊")), KeepDecision::FillerRepeat);
    }

    #[test]
    fn strip_punct_and_filler_helpers() {
        assert_eq!(strip_punct("你好，世界！"), "你好世界");
        assert_eq!(strip_punct("えー…"), "えー");
        assert_eq!(strip_punct("a.b, c!"), "abc");
        assert_eq!(strip_punct("「引用」(括弧)"), "引用括弧");
        assert!(is_pure_filler("嗯嗯"));
        assert!(is_pure_filler("哎呀"));
        assert!(!is_pure_filler(""));
        assert!(!is_pure_filler("嗯好"));
        assert!(!is_pure_filler("太阳"));
    }

    /// 滞回队列：多线程正确性（顺序、总量、关闭语义、满队列阻塞后能恢复）。
    #[test]
    fn seg_queue_hysteresis_threads() {
        let q = SegQueue::new(4); // cap=4, low_water=2
        let qp = Arc::clone(&q);
        let n = 100usize;
        let producer = std::thread::spawn(move || {
            for i in 0..n {
                let seg = VadSeg { start_sample: i * 512, samples: vec![i as f32; 8] };
                assert!(qp.push(Ok(seg)), "push 不应在正常路径失败");
            }
            qp.close();
        });
        let mut got = Vec::new();
        while let Some(item) = q.pop() {
            got.push(item.unwrap().start_sample);
        }
        producer.join().unwrap();
        assert_eq!(got.len(), n);
        assert_eq!(got, (0..n).map(|i| i * 512).collect::<Vec<_>>(), "顺序必须保持");
        // close 后 pop 立即 None（不阻塞）
        assert!(q.pop().is_none());
    }

    /// abort 解除生产者的满队列阻塞（消费者出错提前退出的场景）。
    #[test]
    fn seg_queue_abort_unblocks_producer() {
        let q = SegQueue::new(2);
        let qp = Arc::clone(&q);
        let producer = std::thread::spawn(move || {
            let mut pushed = 0;
            for i in 0..1000usize {
                let seg = VadSeg { start_sample: i, samples: vec![0.0; 4] };
                if !qp.push(Ok(seg)) {
                    break; // 队列关闭：正常退出而非永久阻塞
                }
                pushed += 1;
            }
            pushed
        });
        // 消费一个后直接 abort（模拟 ASR 出错）
        let _ = q.pop();
        q.abort();
        let pushed = std::time::Instant::now();
        let n = producer.join().unwrap();
        assert!(n < 1000, "生产者应被 abort 截停（实际推了 {n}）");
        assert!(pushed.elapsed().as_secs() < 10, "join 不应长时间阻塞");
        assert!(q.pop().is_none());
    }
}
