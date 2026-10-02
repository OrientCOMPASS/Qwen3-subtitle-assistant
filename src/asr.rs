//! S2TT 单模型转录管线（流式）：ffmpeg PCM 流 → 纯 Rust silero VAD **边切边送** →
//! GgufAsr（llama.cpp/mtmd，context 任务开关直出目标语言）→ 字幕段。
//!
//! v0.5 起为生产者/消费者流式管线：生产者线程读 ffmpeg PCM、跑 VAD，**每确认
//! 一个语音段立即送入滞回缓冲队列**；消费者（主线程）拿到一段就 ASR 一段——
//! 两个推理天然并行。v0.6 队列**按波形字节计量**（`--buffer-mb`，默认 50MB）：
//! 占用满时 VAD 暂停生产，消费到半容量才恢复（高低水位滞回，减少阻塞/唤醒
//! 抖动）；缓冲占用实时显示在进度条消息里。
//! 与旧版「VAD 切完全片再整批 ASR」相比：
//! * 首条字幕的等待时间 ≈ 第一句话的长度（而非全片 VAD 时长）；
//! * 内存有界（队列容量 + VAD 窗口），不再全片缓冲；
//! * 进度条只有一条，按已解码媒体时间走，消息里带已转写段数。
//!
//! 幻觉/重复处理（默认开启，用户规则；注意**粒度**）：
//! * 段级（拆分前）：空文本跳过；**复读压缩**（连续重复 ≥3 遍的子串只留 2 遍，
//!   "啊！"×56 → "啊！啊！"）；`--filter-fragments` 的 lang=None 短碎片过滤
//!   （<2s/<4字 门槛按段校准，默认关闭）。
//! * 条级（**大段拆分/折行之后**，v0.6.3 起——拆分先行才能拦住拆出来的相邻
//!   重复条，如"别走！"×3；终端 [ASR✔] 打印的也是拆分后的真实字幕条）：
//!   与上一条保留字幕相同（压缩+去标点后比较）丢弃；上一条与本条都是纯语气词
//!   （哎/啊/嗯/哼…闭集）丢弃本条。

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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// `--filter-fragments` 开启时，lang=None 但带文本的段的保留门槛（实测结论，
/// 见 finetune/s2tt_pipeline.py）：<2s 或 <4 字是半幻觉碎片（"今天"/"请 everyone"），
/// ≥ 门槛是真实语音（结尾致辞）。
const KEEP_LANG_NONE_MIN_SECS: f64 = 2.0;
const KEEP_LANG_NONE_MIN_CHARS: usize = 4;

/// 段级预处理判定（纯函数，可单测）：空文本/碎片在**段**粒度判（碎片启发式
/// 的 <2s/<4字 门槛是按段校准的，见 finetune/s2tt_pipeline.py）；复读压缩
/// 也在段粒度做（幻觉循环链跨越拆分边界，必须先看全文）。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SegmentVerdict {
    /// 进入拆分与条级判定；携带复读压缩后的文本
    Process(String),
    /// 空文本（模型判定纯静音/噪音，或全标点）——无内容可写
    EmptyText,
    /// lang=None 短碎片且 `--filter-fragments` 开启
    Fragment,
}

/// 条级判定（纯函数，可单测）：判重与连续语气词在**最终字幕条**粒度做——
/// 大段拆分之后。v0.6.2 及以前在段粒度判，拆分产生的相邻重复碎片
/// （"别走！"/"别走！"/"别走"）会整体绕过判重，用户实测抓到；拆分先行后
/// 这类重复在条粒度被逐一拦截。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PieceVerdict {
    Keep,
    /// 空内容（拆分/折行后无实义字符）
    Empty,
    /// 与上一条**保留**字幕相同（压缩+去标点后比较）
    Duplicate,
    /// 上一条与本条都是纯语气词
    FillerRepeat,
}

/// 段级预处理：空文本 → 复读压缩 → 碎片过滤（仅 flag 开启时）。
fn prepare_segment(u: &AsrUtterance, dur_secs: f64, filter_fragments: bool) -> SegmentVerdict {
    let text = u.text.trim();
    if text.is_empty() {
        return SegmentVerdict::EmptyText;
    }
    let collapsed = collapse_repeats(text);
    let stripped = strip_punct(&collapsed);
    if stripped.is_empty() {
        // 全是标点（如"……。"）：没有字幕内容可写，按空文本处理
        return SegmentVerdict::EmptyText;
    }
    if filter_fragments && u.lang.eq_ignore_ascii_case("None") {
        let chars = stripped.chars().count();
        if dur_secs < KEEP_LANG_NONE_MIN_SECS || chars < KEEP_LANG_NONE_MIN_CHARS {
            return SegmentVerdict::Fragment;
        }
    }
    SegmentVerdict::Process(collapsed)
}

/// 条级判定：`last_kept_stripped` 为上一条**保留字幕条**（压缩+去标点后）。
fn classify_piece(text: &str, last_kept_stripped: Option<&str>) -> PieceVerdict {
    let stripped = strip_punct(text);
    if stripped.is_empty() {
        return PieceVerdict::Empty;
    }
    if let Some(prev) = last_kept_stripped {
        // 契约上 prev 已是压缩+去标点文本；再处理一次幂等且防御调用方状态漂移
        let prev = strip_punct(&collapse_repeats(prev));
        if stripped == prev {
            return PieceVerdict::Duplicate;
        }
        if is_pure_filler(&prev) && is_pure_filler(&stripped) {
            return PieceVerdict::FillerRepeat;
        }
    }
    PieceVerdict::Keep
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

/// 词组级重复单元的最小长度（字符数）。≥ 此长度的单元视为「词组/短句」。
const PHRASE_UNIT: usize = 3;

/// 段内复读压缩（模型幻觉处理，用户规则，v0.6.4 分级收紧）：
/// * **词组级单元（≥3 字符）**：连续重复 **≥2 遍 → 只保留 1 遍**。相邻同词组
///   复读（"我等你可久了！我等你可久了！"）几乎必是模型回声/幻觉循环——实测
///   短语池循环素材上，旧的「≥3 留 2」会把每条链都留成 ×2，整屏仍是垃圾；
/// * **字级单元（1-2 字符）**：连续重复 **≥3 遍 → 保留 2 遍**（保护"谢谢"
///   "哈哈""慢慢"等合法叠词，"哈哈哈"→"哈哈"、"啊！"×56 → "啊！啊！"）。
///
/// 算法：从左到右扫描；在每个位置找**最小**重复单元 L（"abcabcabc" 的单元是
/// "abc" 而非 "abcabc"；预筛：单元重复则 chars[i+L]==chars[i] 必成立）；命中
/// 则输出保留份数并跳过整条重复链。单元长度上限 128 字符（幻觉循环单元都是
/// 短语级；同时约束最坏复杂度）。
fn collapse_repeats(text: &str) -> String {
    const MAX_UNIT: usize = 128;
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n < 2 {
        return text.to_string();
    }
    let mut out: Vec<char> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        // 词组级 2 遍即触发，扫描窗口按 /2 放宽（旧 /3 会漏掉恰好 2 遍的词组链）
        let max_l = ((n - i) / 2).min(MAX_UNIT);
        let mut hit: Option<(usize, usize, usize)> = None; // (单元长, 保留份数, 链尾)
        let mut l = 1usize;
        while l <= max_l {
            // 预筛：单元在 i+l 处重复的必要条件
            if chars[i + l] == chars[i] {
                let unit = &chars[i..i + l];
                let mut j = i + l;
                let mut cnt = 1usize;
                while j + l <= n && &chars[j..j + l] == unit {
                    cnt += 1;
                    j += l;
                }
                let keep = if l >= PHRASE_UNIT && cnt >= 2 {
                    1 // 词组级：≥2 遍只留 1 遍
                } else if cnt >= 3 {
                    2 // 字级：≥3 遍留 2 遍（叠词保护）
                } else {
                    0
                };
                if keep > 0 {
                    hit = Some((l, keep, j));
                    break;
                }
            }
            l += 1;
        }
        match hit {
            Some((l, keep, j)) => {
                for _ in 0..keep {
                    out.extend_from_slice(&chars[i..i + l]);
                }
                i = j; // 跳过整条重复链
            }
            None => {
                out.push(chars[i]);
                i += 1;
            }
        }
    }
    out.into_iter().collect()
}

/// VAD→ASR 滞回缓冲队列（**按波形字节数计量**，`--buffer-mb` 调整，默认 50MB）。
///
/// 流控语义（用户指定）：
/// * 占用 ≥ 容量（高水位）→ 生产者（VAD 线程）暂停压入；
/// * 消费到 ≤ 容量一半（低水位）→ 恢复生产——高低水位滞回，避免
///   "满一格放一格"的高频阻塞/唤醒抖动，让 VAD 成批跑在 ASR 前面；
/// * 单段超过总容量也不死锁：队列为空时照常入队（占用可短暂越界一个段的大小）。
///
/// VAD 与 ASR 本就分别跑在生产者/消费者线程上（v0.5 起真并行）；本队列替换
/// v0.6.0 前的"8 段"计数队列——按字节计量后，内存上界与段长分布无关
/// （60s 长段 ×8 = 30MB vs 短段 ×8 = 1MB 的不一致不复存在）。
struct SegQueue {
    inner: Mutex<QueueInner>,
    /// 生产者等待：占用降到低水位（容量一半）以下
    cv_drained: Condvar,
    /// 消费者等待：有新段或队列关闭
    cv_item: Condvar,
    cap_bytes: usize,
}

struct QueueInner {
    q: VecDeque<Result<VadSeg>>,
    /// 当前排队波形占用（字节；Err 项计 0）
    bytes: usize,
    closed: bool,
}

fn seg_bytes(seg: &Result<VadSeg>) -> usize {
    seg.as_ref().map(|s| s.samples.len() * std::mem::size_of::<f32>()).unwrap_or(0)
}

impl SegQueue {
    fn new(cap_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(QueueInner { q: VecDeque::new(), bytes: 0, closed: false }),
            cv_drained: Condvar::new(),
            cv_item: Condvar::new(),
            cap_bytes: cap_bytes.max(1),
        })
    }

    /// 当前占用（字节）——进度条展示用。
    fn occupied(&self) -> usize {
        self.inner.lock().map(|g| g.bytes).unwrap_or(0)
    }

    /// 容量（字节）。
    fn cap(&self) -> usize {
        self.cap_bytes
    }

    /// 生产者压入一段；占用 ≥ 容量时阻塞至消费过半（队列为空则总是放行，
    /// 防单段超容死锁）。返回 false = 队列已关闭（消费者出错退出），生产者应停止。
    fn push(&self, item: Result<VadSeg>) -> bool {
        let add = seg_bytes(&item);
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        while !g.q.is_empty() && g.bytes >= self.cap_bytes && !g.closed {
            g = self.cv_drained.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        if g.closed {
            return false; // 消费者已弃队列（出错提前退出）
        }
        g.bytes += add;
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
                g.bytes = g.bytes.saturating_sub(seg_bytes(&item));
                let drained = g.bytes <= self.cap_bytes / 2;
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
        g.bytes = 0;
        drop(g);
        self.cv_drained.notify_all();
        self.cv_item.notify_all();
    }
}

/// 转录产物：`cues` = 最终字幕条（已拆分/折行/判重，index 已分配，直接可写
/// .srt）；`raw` = 排版前原始直出（未压缩、未拆分、未判重，仅非空段，
/// --raw-srt 调试对照用）。
pub struct TranscribeOut {
    pub cues: Vec<SubtitleSegment>,
    pub raw: Vec<SubtitleSegment>,
}

/// 转录一个媒体文件（流式）。管线内完成大段拆分与排版（拆分先于判重——
/// 判重/语气词在"最终字幕条"粒度执行，见 classify_piece 文档）。
pub fn transcribe_file(input: &Path, cfg: &Config, asr: &mut GgufAsr) -> Result<TranscribeOut> {
    let duration = ffmpeg::get_duration_secs(input).unwrap_or(0.0);

    let pb = make_progress(duration);
    let queue = SegQueue::new(cfg.buffer_bytes);
    // 已转写（保留）段数——消费者更新，生产者读它拼进度条消息（缓冲占用展示）
    let done = Arc::new(AtomicUsize::new(0));

    // ---- 生产者线程：ffmpeg 解码 + 流式 VAD，切出一段送一段 ----
    let pb_prod = pb.clone();
    let input_owned = input.to_path_buf();
    let (vad_threshold, vad_min_silence, vad_max_seg_secs) =
        (cfg.vad_threshold, cfg.vad_min_silence, cfg.vad_max_seg_secs);
    let q_prod = Arc::clone(&queue);
    let done_prod = Arc::clone(&done);
    let producer = std::thread::Builder::new()
        .name("vad-feed".into())
        .spawn(move || {
            let result = vad_feed(&input_owned, duration, vad_threshold, vad_min_silence,
                                  vad_max_seg_secs, &q_prod, &done_prod, &pb_prod);
            if let Err(e) = result {
                // 消费者可能已因 ASR 错误退出（队列 closed）——push 返回 false 就静默收尾
                let _ = q_prod.push(Err(e));
            }
            q_prod.close(); // 消费者 pop() 收到 None，迭代自然结束
        })
        .context("启动 VAD 生产者线程失败")?;

    // ---- 消费者（主线程）：收一段、ASR 一段、拆条、条级判定 ----
    let mut out: Vec<SubtitleSegment> = Vec::new();
    let mut raw: Vec<SubtitleSegment> = Vec::new(); // --raw-srt 用原始直出
    let mut n_segs = 0usize;    // VAD 确认的语音段总数
    let mut n_empty = 0usize;   // 空文本（段级或条级，无内容可写）
    let mut n_dup = 0usize;     // 条级判重丢弃
    let mut n_filler = 0usize;  // 条级连续语气词丢弃
    let mut n_frag = 0usize;    // --filter-fragments 段级碎片丢弃
    let mut n_collapsed = 0usize; // 触发段内复读压缩的段
    let mut last_kept_stripped: Option<String> = None; // 上一条保留字幕条（去标点）
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

        // 原始直出（未压缩/未拆分/未判重）——--raw-srt 的调试对照价值所在
        if cfg.raw_srt {
            let t = u.text.trim();
            if !t.is_empty() {
                raw.push(SubtitleSegment {
                    index: raw.len() + 1, start_ms, end_ms, text: t.to_string(),
                });
            }
        }

        match prepare_segment(&u, dur_secs, cfg.filter_fragments) {
            SegmentVerdict::EmptyText => {
                n_empty += 1;
                pb.suspend(|| debug!("[ASR∅] {:.2}s@{} 空输出（lang={}），跳过",
                                     dur_secs, format_timestamp(start_ms), u.lang));
            }
            SegmentVerdict::Fragment => {
                n_frag += 1;
                pb.suspend(|| debug!("[VAD✂] {:.2}s@{} 丢弃（lang={} text={:?}）",
                                     dur_secs, format_timestamp(start_ms), u.lang,
                                     truncate(&u.text, 20)));
            }
            SegmentVerdict::Process(collapsed) => {
                if collapsed != u.text.trim() {
                    n_collapsed += 1;
                    let before = truncate(u.text.trim(), 24);
                    let after = truncate(&collapsed, 24);
                    pb.suspend(|| debug!("[ASR♻] {:.2}s@{} 复读压缩: {:?} → {:?}",
                                         dur_secs, format_timestamp(start_ms), before, after));
                }
                // 大段拆分 + 折行 **先于** 判重/语气词检查（用户要求）：
                // 检查与终端打印都以「最终写入 SRT 的字幕条」为单位，
                // 拆分产生的相邻重复条（"别走！"×3 这类）才会被条级判重拦截，
                // [ASR✔] 打印的也就是真实字幕内容而非拆分前整段。
                let whole = SubtitleSegment { index: 0, start_ms, end_ms, text: collapsed };
                let pieces = if cfg.no_layout {
                    vec![whole]
                } else {
                    crate::srt::layout(&[whole], cfg.max_line_width,
                                       cfg.max_cue_secs, cfg.max_cue_chars)
                };
                for mut piece in pieces {
                    match classify_piece(&piece.text, last_kept_stripped.as_deref()) {
                        PieceVerdict::Empty => {
                            n_empty += 1;
                        }
                        PieceVerdict::Duplicate => {
                            n_dup += 1;
                            let ts = format_timestamp(piece.start_ms);
                            let t = truncate(&piece.text, 20);
                            pb.suspend(|| debug!("[ASR↻] {} 与上一条相同（去标点比较），丢弃（{:?}）", ts, t));
                        }
                        PieceVerdict::FillerRepeat => {
                            n_filler += 1;
                            let ts = format_timestamp(piece.start_ms);
                            let t = truncate(&piece.text, 20);
                            pb.suspend(|| debug!("[ASR〰] {} 连续纯语气词，丢弃（{:?}）", ts, t));
                        }
                        PieceVerdict::Keep => {
                            piece.index = out.len() + 1;
                            // 终端打印与 SRT 一致的字幕条（折行压成单行显示）
                            let flat: String = piece.text.chars()
                                .map(|c| if c == '\n' { ' ' } else { c }).collect();
                            let pdur = (piece.end_ms - piece.start_ms) as f64 / 1000.0;
                            let ts = format_timestamp(piece.start_ms);
                            pb.suspend(|| info!("[ASR✔] {} ({:.1}s) {}", ts, pdur, flat));
                            last_kept_stripped = Some(strip_punct(&piece.text));
                            out.push(piece);
                            done.store(out.len(), Ordering::Relaxed);
                        }
                    }
                }
            }
        }
        // 每段处理完刷新一次：已转写数 + 缓冲区占用（用户要求终端可见）
        pb.set_message(queue_msg(&queue, &done));
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
        return Ok(TranscribeOut { cues: Vec::new(), raw });
    }
    let mut notes: Vec<String> = Vec::new();
    if n_empty > 0 {
        notes.push(format!("{n_empty} 处空输出跳过"));
    }
    if n_dup > 0 {
        notes.push(format!("{n_dup} 条重复丢弃"));
    }
    if n_filler > 0 {
        notes.push(format!("{n_filler} 条连续语气词丢弃"));
    }
    if n_collapsed > 0 {
        notes.push(format!("{n_collapsed} 段复读压缩"));
    }
    if cfg.filter_fragments && n_frag > 0 {
        notes.push(format!("碎片过滤丢弃 {n_frag} 段"));
    }
    let tail = if notes.is_empty() { String::new() } else { format!("（{}）", notes.join("，")) };
    info!("转录完成：{n_segs} 段语音 → {} 条字幕{tail}", out.len());
    Ok(TranscribeOut { cues: out, raw })
}

/// 生产者主体：ffmpeg PCM 流 → f32 样本 → VadSegmenter.accept（返回即发送）。
fn vad_feed(input: &Path, duration: f64, threshold: f32, min_silence: f32, max_seg_secs: f32,
            queue: &SegQueue, done: &AtomicUsize, pb: &ProgressBar) -> Result<()> {
    let mut stream = ffmpeg::spawn_decode_stream(input)
        .with_context(|| format!("启动 ffmpeg 解码失败: {:?}", input))?;
    let mut reader = stream
        .take_stdout()
        .context("无法取得 ffmpeg stdout（内部错误）")?;

    let mut segmenter = VadSegmenter::new(threshold, min_silence, max_seg_secs)
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
        pb.set_message(queue_msg(queue, done));
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

/// 进度条消息：已转写段数 + 缓冲区占用/容量（MB）。
fn queue_msg(queue: &SegQueue, done: &AtomicUsize) -> String {
    format!("已转写 {} 段 · 缓冲 {:.1}/{}MB",
            done.load(Ordering::Relaxed),
            queue.occupied() as f64 / 1e6,
            (queue.cap() as f64 / 1e6).round() as usize)
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

    // ---------------- 段级预处理 prepare_segment ----------------

    #[test]
    fn prepare_empty_and_punct_only() {
        assert_eq!(prepare_segment(&u("None", ""), 5.0, false), SegmentVerdict::EmptyText);
        assert_eq!(prepare_segment(&u("Japanese", "   "), 5.0, true), SegmentVerdict::EmptyText);
        // 全标点 = 无内容可写
        assert_eq!(prepare_segment(&u("Chinese", "……。"), 5.0, false), SegmentVerdict::EmptyText);
    }

    #[test]
    fn prepare_collapses_and_keeps() {
        // 复读压缩在段级完成（幻觉循环链跨越拆分边界，必须先看全文）
        let spam = "啊！".repeat(56);
        assert_eq!(prepare_segment(&u("Chinese", &spam), 14.9, false),
                   SegmentVerdict::Process("啊！啊！".into()));
        // 词组级 ×2 也压（v0.6.4）
        assert_eq!(prepare_segment(&u("Chinese", "我等你可久了！我等你可久了！"), 5.8, false),
                   SegmentVerdict::Process("我等你可久了！".into()));
        // 正常文本原样
        assert_eq!(prepare_segment(&u("Chinese", " 太阳会升起。 "), 2.0, false),
                   SegmentVerdict::Process("太阳会升起。".into()));
    }

    #[test]
    fn prepare_fragment_only_when_flag_on() {
        // lang=None + 短（<2s）→ 过滤开启才丢（按压缩后字符计）
        assert_eq!(prepare_segment(&u("None", "今天"), 0.8, true), SegmentVerdict::Fragment);
        assert_eq!(prepare_segment(&u("None", "今天"), 0.8, false),
                   SegmentVerdict::Process("今天".into()));
        // lang=None 但长段（真实语音）→ 过滤开启也保留
        assert_eq!(prepare_segment(&u("None", "谢谢大家的观看"), 5.0, true),
                   SegmentVerdict::Process("谢谢大家的观看".into()));
        // 复读压缩过的 lang=None 碎片同样被过滤（灌水不再骗过字数门槛）
        let spam = "嗯！".repeat(10); // 压缩后 "嗯！嗯！" → 去标点 2 字 < 4
        assert_eq!(prepare_segment(&u("None", &spam), 1.0, true), SegmentVerdict::Fragment);
        // lang 有值的短段不受过滤影响
        assert_eq!(prepare_segment(&u("Japanese", "はい"), 0.5, true),
                   SegmentVerdict::Process("はい".into()));
    }

    // ---------------- 条级判定 classify_piece ----------------

    #[test]
    fn piece_duplicate_ignores_punctuation() {
        assert_eq!(classify_piece("今天天气很好。", Some("今天天气很好")), PieceVerdict::Duplicate);
        assert_eq!(classify_piece("今天天气很好，", Some("今天天气很好。")), PieceVerdict::Duplicate,
                   "基准带标点也应命中（防御性再压缩+去标点）");
        assert_eq!(classify_piece("今天天气很好！", Some("明天天气很好")), PieceVerdict::Keep);
        assert_eq!(classify_piece("今天天气很好。", None), PieceVerdict::Keep);
        assert_eq!(classify_piece(" 今天 天气很好。 ", Some("今天天气很好")), PieceVerdict::Duplicate);
    }

    #[test]
    fn piece_filler_repeat_dropped_but_single_kept() {
        // 上一条实义 → 本条语气词保留（首条语气词合法）
        assert_eq!(classify_piece("哎呀！", Some("太阳会升起")), PieceVerdict::Keep);
        // 两条都纯语气词 → 丢弃
        assert_eq!(classify_piece("嗯嗯。", Some("嗯")), PieceVerdict::FillerRepeat);
        assert_eq!(classify_piece("啊……", Some("哎")), PieceVerdict::FillerRepeat);
        // 语气词 + 实义混合不算纯语气词
        assert_eq!(classify_piece("嗯好的。", Some("嗯")), PieceVerdict::Keep);
        // 完全相同 → Duplicate 优先
        assert_eq!(classify_piece("嗯。", Some("嗯")), PieceVerdict::Duplicate);
        // 全标点条 → Empty
        assert_eq!(classify_piece("……", Some("嗯")), PieceVerdict::Empty);
    }

    /// 用户实例回归（v0.6.2 缺陷）："别走！别走！别走"（2 个完整重复 + 1 个
    /// 残段，段级压缩不触发）拆分后产生三条相邻重复字幕。v0.6.3 拆分先于
    /// 判重：条级逐一拦截，仅第一条保留。
    #[test]
    fn piece_dedup_catches_split_born_duplicates() {
        let seg_text = "别走！别走！别走";
        // 段级：词组"别走！"（3 字符）完整重复 2 次 → v0.6.4 压缩为 1 次 + 残段
        assert_eq!(prepare_segment(&u("Chinese", seg_text), 13.6, false),
                   SegmentVerdict::Process("别走！别走".into()));
        // 拆分（模拟 layout 对压缩产物的句读拆分）
        let pieces = ["别走！", "别走"];
        let mut last: Option<String> = None;
        let mut kept = Vec::new();
        for p in pieces {
            match classify_piece(p, last.as_deref()) {
                PieceVerdict::Keep => {
                    last = Some(strip_punct(p));
                    kept.push(p);
                }
                PieceVerdict::Duplicate | PieceVerdict::FillerRepeat | PieceVerdict::Empty => {}
            }
        }
        assert_eq!(kept, vec!["别走！"], "拆分产生的相邻重复条应只剩第一条");
    }

    /// 跨段边界的重复同样被条级判重拦截（上一段末条 = 下一段首条）。
    #[test]
    fn piece_dedup_across_segments() {
        let mut last = Some(strip_punct("谢谢观看"));
        assert_eq!(classify_piece("谢谢观看！", last.as_deref()), PieceVerdict::Duplicate);
        last = Some(strip_punct("谢谢观看"));
        assert_eq!(classify_piece("下一段的新内容。", last.as_deref()), PieceVerdict::Keep);
    }

    /// 段内复读压缩（v0.6.4 分级规则）：词组级（≥3 字符）≥2 遍留 1 遍；
    /// 字级（1-2 字符）≥3 遍留 2 遍（叠词保护）。
    #[test]
    fn collapse_repeats_phrase_and_char_units() {
        // 字级单元：≥3 → 留 2（叠词/语气词保护）
        let spam = "啊！".repeat(10); // 单元"啊！"长 2
        assert_eq!(collapse_repeats(&spam), "啊！啊！");
        assert_eq!(collapse_repeats("谢谢谢谢谢谢"), "谢谢"); // 单元"谢"×6 → 留 2
        assert_eq!(collapse_repeats("好好好"), "好好");
        assert_eq!(collapse_repeats("哈哈"), "哈哈");         // ×2 字级不触发
        assert_eq!(collapse_repeats(&format!("{}收尾", "哈".repeat(6))), "哈哈收尾");
        // 词组级单元（≥3 字符）：≥2 遍即只留 1 遍
        let hug = "让我抱抱你！呜呜呜，".repeat(4);
        assert_eq!(collapse_repeats(&hug), "让我抱抱你！呜呜呜，");
        assert_eq!(collapse_repeats("对不起，对不起，"), "对不起，");
        assert_eq!(collapse_repeats("我爱你我爱你"), "我爱你");
        // 用户实测坏 case（v0.6.3 遗留 ×2）：短语池循环
        let loop_txt = "真难受！我等你可久了！我等你可久了！你懂的吧？";
        assert_eq!(collapse_repeats(loop_txt), "真难受！我等你可久了！你懂的吧？");
        // 2 字词组 ×2 不触发（"好的好的"是合法口语）
        assert_eq!(collapse_repeats("好的好的"), "好的好的");
        // 无重复原样；英文多字符单元
        assert_eq!(collapse_repeats("你好世界"), "你好世界");
        assert_eq!(collapse_repeats("abababab"), "abab"); // 单元"ab"长 2 → 字级规则留 2
        assert_eq!(collapse_repeats("no no no no "), "no "); // 单元"no "长 3 = 词组级 → ≥2 留 1
        // 前缀实义内容 + 尾部字级循环
        let mixed = format!("真舒服！感觉真棒！我怎么能这么轻易就放手？啊！好舒服！{}", "啊！".repeat(20));
        assert_eq!(collapse_repeats(&mixed),
                   "真舒服！感觉真棒！我怎么能这么轻易就放手？啊！好舒服！啊！啊！");
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

    /// 滞回队列（字节计量）：多线程正确性——顺序、总量、关闭语义。
    #[test]
    fn seg_queue_hysteresis_threads() {
        let cap = 64 * 1024; // 64KB；每段 512B（128 样本）
        let q = SegQueue::new(cap);
        let qp = Arc::clone(&q);
        let n = 1000usize;
        let producer = std::thread::spawn(move || {
            for i in 0..n {
                let seg = VadSeg { start_sample: i * 512, samples: vec![i as f32; 128] };
                assert!(qp.push(Ok(seg)), "push 不应在正常路径失败");
            }
            qp.close();
        });
        let mut got = Vec::new();
        while let Some(item) = q.pop() {
            got.push(item.unwrap().start_sample);
        }
        producer.join().unwrap();
        assert_eq!(got, (0..n).map(|i| i * 512).collect::<Vec<_>>(), "顺序必须保持");
        assert!(q.pop().is_none(), "close 后 pop 应立即 None（不阻塞）");
    }

    /// 高水位停顿：不消费时生产者应停在容量处（越界 ≤ 一个段），
    /// 消费过半后恢复生产，最终全部收到——这就是「满→暂停、半→恢复」滞回。
    #[test]
    fn seg_queue_bytes_hysteresis_bounds() {
        let cap = 4 * 1024usize;      // 4KB
        let seg_bytes = 128 * 4usize; // 512B/段
        let q = SegQueue::new(cap);
        let qp = Arc::clone(&q);
        let n = 100usize;
        let producer = std::thread::spawn(move || {
            for i in 0..n {
                let seg = VadSeg { start_sample: i, samples: vec![0.0f32; 128] };
                if !qp.push(Ok(seg)) {
                    return i;
                }
            }
            qp.close();
            n
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        let occ = q.occupied();
        assert!(occ >= cap, "应生产到高水位: occ={occ} cap={cap}");
        assert!(occ <= cap + seg_bytes, "越界不应超过一个段: occ={occ}");
        let mut got = 0usize;
        while let Some(item) = q.pop() {
            let _ = item.unwrap();
            got += 1;
        }
        assert_eq!(producer.join().unwrap(), n, "生产者应完成全部推送");
        assert_eq!(got, n);
    }

    /// 单段超过总容量：队列为空时照常入队（不死锁），占用短暂越界后恢复。
    #[test]
    fn seg_queue_oversized_single_segment_no_deadlock() {
        let q = SegQueue::new(1024); // 容量 1KB；单段 8KB
        let seg = VadSeg { start_sample: 0, samples: vec![0.0f32; 2048] };
        assert!(q.push(Ok(seg)));
        assert!(q.occupied() > q.cap(), "占用应如实反映越界");
        let got = q.pop().unwrap().unwrap();
        assert_eq!(got.samples.len(), 2048);
        assert_eq!(q.occupied(), 0);
        q.close();
        assert!(q.pop().is_none());
    }

    /// abort 解除生产者的满队列阻塞（消费者出错提前退出的场景）。
    #[test]
    fn seg_queue_abort_unblocks_producer() {
        let q = SegQueue::new(1024); // 1KB → 两个 512B 段即停
        let qp = Arc::clone(&q);
        let producer = std::thread::spawn(move || {
            let mut pushed = 0;
            for i in 0..1000usize {
                let seg = VadSeg { start_sample: i, samples: vec![0.0f32; 128] };
                if !qp.push(Ok(seg)) {
                    break; // 队列关闭：正常退出而非永久阻塞
                }
                pushed += 1;
            }
            pushed
        });
        let _ = q.pop(); // 消费一个后直接 abort（模拟 ASR 出错）
        q.abort();
        let t = std::time::Instant::now();
        let pushed = producer.join().unwrap();
        assert!(pushed < 1000, "生产者应被 abort 截停（实际推了 {pushed}）");
        assert!(t.elapsed().as_secs() < 10, "join 不应长时间阻塞");
        assert!(q.pop().is_none());
        assert_eq!(q.occupied(), 0, "abort 应清空占用");
    }
}
