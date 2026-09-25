//! SRT 读写与字幕排版。
//!
//! 旧版只做「序号 + 时间轴 + 文本」的直白拼接：一个 VAD 语音段就是一条 cue，
//! 十几秒的连续讲话会变成一整屏文字。这里补上两件事：
//! 1. **长 cue 拆分**：超过 `--max-cue-secs` 的字幕按句读边界拆成多条，
//!    时间按字符占比在原区间内分配（没有词级时间戳时的标准做法）；
//! 2. **行宽折行**：按显示宽度（CJK 计 2、ASCII 计 1）折行，优先在标点/空格处断。
//!
//! 另外提供 `parse_srt`，支撑 `--from-srt`（跳过 ASR，直接翻译已有字幕）。

use crate::types::SubtitleSegment;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub fn format_timestamp(ms: u64) -> String {
    let total_sec = ms / 1000;
    let millis = ms % 1000;
    let h = total_sec / 3600;
    let m = (total_sec % 3600) / 60;
    let s = total_sec % 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
}

pub fn render_srt(segments: &[SubtitleSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_timestamp(seg.start_ms),
            format_timestamp(seg.end_ms)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub fn write_srt(path: &Path, segments: &[SubtitleSegment]) -> Result<()> {
    let content = render_srt(segments);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)
                .with_context(|| format!("创建输出目录失败: {:?}", parent))?;
        }
    }
    fs::write(path, content).with_context(|| format!("写字幕失败: {:?}", path))?;
    log::info!("已写出字幕: {:?}（{} 条）", path, segments.len());
    Ok(())
}

/// 解析 SRT 文本（容忍缺序号、`.` 毫秒分隔、CRLF、多余空行）。
pub fn parse_srt(text: &str) -> Vec<SubtitleSegment> {
    let mut out = Vec::new();
    for block in text.replace("\r\n", "\n").split("\n\n") {
        let lines: Vec<&str> = block.split('\n').map(|l| l.trim_end()).collect();
        let Some(ts_idx) = lines.iter().position(|l| l.contains("-->")) else {
            continue;
        };
        let Some((start, end)) = parse_timestamps(lines[ts_idx]) else {
            continue;
        };
        let text_lines: Vec<&str> = lines[ts_idx + 1..]
            .iter()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if text_lines.is_empty() {
            continue;
        }
        out.push(SubtitleSegment {
            index: out.len() + 1,
            start_ms: start,
            end_ms: end.max(start + 1),
            text: text_lines.join("\n"),
        });
    }
    out
}

pub fn read_srt(path: &Path) -> Result<Vec<SubtitleSegment>> {
    let text = fs::read_to_string(path).with_context(|| format!("读取字幕失败: {:?}", path))?;
    let segs = parse_srt(&text);
    anyhow::ensure!(!segs.is_empty(), "字幕文件里没有可解析的条目: {:?}", path);
    Ok(segs)
}

fn parse_timestamps(line: &str) -> Option<(u64, u64)> {
    let (a, b) = line.split_once("-->")?;
    Some((parse_ts(a.trim())?, parse_ts(b.trim())?))
}

/// 解析 `HH:MM:SS,mmm`（也接受 `HH:MM:SS.mmm` 与省略小时的形式）。
fn parse_ts(s: &str) -> Option<u64> {
    let s = s.split(' ').next().unwrap_or(s).replace('.', ",");
    let (hms, ms) = match s.split_once(',') {
        Some(v) => v,
        None => (s.as_str(), "0"),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (parts[0], parts[1], parts[2]),
        2 => ("0", parts[0], parts[1]),
        _ => return None,
    };
    let ms: u64 = ms.parse().ok()?;
    let total = h.parse::<u64>().ok()? * 3600 + m.parse::<u64>().ok()? * 60 + sec.parse::<u64>().ok()?;
    Some(total * 1000 + ms)
}

// ============================================================================
// 排版
// ============================================================================

/// 显示宽度：CJK / 全角字符按 2 计，其余按 1 计。
pub fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn char_width(c: char) -> usize {
    let u = c as u32;
    let wide = matches!(u,
        0x1100..=0x115F | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF | 0xFE30..=0xFE6F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6
        | 0x1B000..=0x1B16F | 0x20000..=0x3FFFD);
    if wide { 2 } else { 1 }
}

/// 是否为「可以在此处之后断行」的字符（CJK 标点、句末标点、空格）。
fn is_break_after(c: char) -> bool {
    matches!(c,
        '。' | '！' | '？' | '，' | '、' | '；' | '：' | '」' | '』' | '）' | '》' | '…'
        | '.' | '!' | '?' | ',' | ';' | ':' | ')' | ' '
    )
}

/// 按显示宽度折行。
///
/// 贪心：每行尽量塞满 `max_width`，若行内存在句读/空格断点则优先在**最后一个**
/// 断点处换行（避免把「〜なので、」这类短语拦腰截断）；CJK 文本无断点时直接硬切
/// （中日韩排版本身允许任意字符间断行）。
pub fn wrap_text(text: &str, max_width: usize) -> String {
    let max_width = max_width.max(4);
    let mut lines: Vec<String> = Vec::new();

    for para in text.split('\n') {
        let chars: Vec<char> = para.chars().collect();
        let mut start = 0usize;
        while start < chars.len() {
            // 1) 从 start 起能容纳多少个字符
            let mut end = start;
            let mut w = 0usize;
            while end < chars.len() {
                let cw = char_width(chars[end]);
                if w + cw > max_width {
                    break;
                }
                w += cw;
                end += 1;
            }
            if end >= chars.len() {
                let tail: String = chars[start..].iter().collect();
                if !tail.trim().is_empty() {
                    lines.push(tail.trim_end().to_string());
                }
                break;
            }
            // 2) 在 (start, end] 内找最后一个可断点
            let mut cut = end;
            let mut k = end;
            while k > start + 1 {
                k -= 1;
                if is_break_after(chars[k - 1]) {
                    cut = k;
                    break;
                }
            }
            let line: String = chars[start..cut].iter().collect();
            if !line.trim().is_empty() {
                lines.push(line.trim_end().to_string());
            }
            // 3) 跳过断点后的空白，保证前进（cut > start 恒成立）
            let mut next = cut;
            while next < chars.len() && chars[next].is_whitespace() {
                next += 1;
            }
            start = next;
        }
    }
    lines.join("\n")
}

/// 句读边界（用于长 cue 拆分）。返回每个句子结束后的字符下标。
fn sentence_ends(text: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut ends = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if matches!(c, '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';' | '\n') {
            ends.push(i + 1);
        } else if matches!(c, '，' | '、' | ',') {
            ends.push(i + 1); // 逗号也可作为次级切点
        }
    }
    if ends.last().copied() != Some(chars.len()) && !chars.is_empty() {
        ends.push(chars.len());
    }
    ends
}

/// 把过长的 cue 按句读拆成多条，时间按字符占比在原区间内线性分配。
pub fn split_long_cue(seg: &SubtitleSegment, max_cue_secs: f64) -> Vec<SubtitleSegment> {
    let dur = seg.duration_secs();
    if max_cue_secs <= 0.0 || dur <= max_cue_secs || seg.text.trim().is_empty() {
        return vec![seg.clone()];
    }
    let flat = seg.text.replace('\n', " ");
    let chars: Vec<char> = flat.chars().collect();
    let ends = sentence_ends(&flat);
    if ends.len() < 2 {
        // 没有任何句读：按字符数均分
        let parts = (dur / max_cue_secs).ceil().max(2.0) as usize;
        let per = (chars.len() / parts).max(1);
        let mut out = Vec::new();
        for (k, chunk) in chars.chunks(per).enumerate() {
            out.push((k, chunk.iter().collect::<String>()));
        }
        return distribute(seg, &out.into_iter().map(|(_, s)| s).collect::<Vec<_>>());
    }

    // 贪心成组：每组时长不超过 max_cue_secs
    let total_chars = chars.len().max(1) as f64;
    let mut groups: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_end = 0usize;
    for &end in &ends {
        let piece: String = chars[prev_end..end].iter().collect();
        prev_end = end;
        let candidate = format!("{}{}", cur, piece);
        let cand_secs = candidate.chars().count() as f64 / total_chars * dur;
        if !cur.is_empty() && cand_secs > max_cue_secs {
            groups.push(cur.trim().to_string());
            cur = piece.trim().to_string();
        } else {
            cur = candidate.trim().to_string();
        }
    }
    if !cur.trim().is_empty() {
        groups.push(cur.trim().to_string());
    }
    groups.retain(|g| !g.is_empty());
    if groups.len() <= 1 {
        return vec![seg.clone()];
    }
    distribute(seg, &groups)
}

/// 按各组字符数占比分配时间区间（保证单调递增、无缝隙）。
fn distribute(seg: &SubtitleSegment, groups: &[String]) -> Vec<SubtitleSegment> {
    let total: usize = groups.iter().map(|g| g.chars().count()).sum::<usize>().max(1);
    let span = seg.end_ms.saturating_sub(seg.start_ms);
    let mut out: Vec<SubtitleSegment> = Vec::with_capacity(groups.len());
    let mut acc = 0usize;
    for (k, g) in groups.iter().enumerate() {
        acc += g.chars().count();
        let start = if k == 0 {
            seg.start_ms
        } else {
            out[k - 1].end_ms
        };
        let end = if k + 1 == groups.len() {
            seg.end_ms
        } else {
            seg.start_ms + (span * acc as u64 / total as u64)
        };
        out.push(SubtitleSegment {
            index: 0,
            start_ms: start,
            end_ms: end.max(start + 1),
            text: g.clone(),
        });
    }
    out
}

/// 排版：先拆长 cue，再折行，并重排序号。
pub fn layout(
    segments: &[SubtitleSegment],
    max_line_width: usize,
    max_cue_secs: f64,
) -> Vec<SubtitleSegment> {
    let mut out: Vec<SubtitleSegment> = Vec::new();
    for seg in segments {
        for mut piece in split_long_cue(seg, max_cue_secs) {
            if max_line_width > 0 {
                piece.text = wrap_text(&piece.text, max_line_width);
            }
            piece.text = piece.text.trim().to_string();
            if piece.text.is_empty() {
                continue;
            }
            piece.index = out.len() + 1;
            out.push(piece);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(s: u64, e: u64, t: &str) -> SubtitleSegment {
        SubtitleSegment {
            index: 0,
            start_ms: s,
            end_ms: e,
            text: t.to_string(),
        }
    }

    #[test]
    fn timestamp_format() {
        assert_eq!(format_timestamp(0), "00:00:00,000");
        assert_eq!(format_timestamp(1_000), "00:00:01,000");
        assert_eq!(format_timestamp(3_723_456), "01:02:03,456");
        assert_eq!(format_timestamp(36_000_000), "10:00:00,000");
    }

    #[test]
    fn timestamp_parse_variants() {
        assert_eq!(parse_ts("01:02:03,456"), Some(3_723_456));
        assert_eq!(parse_ts("01:02:03.456"), Some(3_723_456));
        assert_eq!(parse_ts("02:03,456"), Some(123_456));
        assert_eq!(parse_ts("garbage"), None);
    }

    #[test]
    fn srt_roundtrip() {
        let v = vec![
            seg(0, 1500, "第一句"),
            seg(1500, 4000, "第二句\n换行"),
        ];
        let text = render_srt(&v);
        let back = parse_srt(&text);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].text, "第一句");
        assert_eq!(back[1].text, "第二句\n换行");
        assert_eq!((back[1].start_ms, back[1].end_ms), (1500, 4000));
        assert_eq!((back[0].index, back[1].index), (1, 2));
    }

    #[test]
    fn srt_parse_tolerates_missing_index_and_crlf() {
        let text = "00:00:01,000 --> 00:00:02,000\r\n只有时间轴没有序号\r\n\r\n2\r\n00:00:03.000 --> 00:00:04.000\r\n第二条\r\n";
        let v = parse_srt(text);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].start_ms, 1000);
        assert_eq!(v[1].text, "第二条");
    }

    #[test]
    fn display_width_counts_cjk_as_two() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("日本語"), 6);
        assert_eq!(display_width("ab日本"), 6);   // 2 个 ASCII(1) + 2 个 CJK(2)
    }

    #[test]
    fn wrap_respects_width_and_prefers_punctuation() {
        let s = "今日はとても良い天気なので、公園まで散歩に行くことにしました。";
        let w = wrap_text(s, 30);
        for line in w.split('\n') {
            assert!(
                display_width(line) <= 30,
                "行过宽: {line} ({})",
                display_width(line)
            );
        }
        assert!(w.contains('\n'));
        // 断点应落在标点之后（第一个「、」出现在第 14 字，宽度 28 ≤ 30）
        assert!(
            w.split('\n').next().unwrap().ends_with('、'),
            "首行未落在标点上: {w}"
        );
        assert_eq!(w.replace('\n', ""), s.replace('、', "、"));
    }

    #[test]
    fn wrap_keeps_short_text_untouched() {
        assert_eq!(wrap_text("短い文", 20), "短い文");
        assert_eq!(wrap_text("hello world", 40), "hello world");
    }

    #[test]
    fn wrap_handles_long_word_without_break_points() {
        let s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; // 30 个 ASCII，无断点
        let w = wrap_text(s, 10);
        for line in w.split('\n') {
            assert!(display_width(line) <= 10, "行过宽: {line}");
        }
        assert_eq!(w.replace('\n', ""), s);
        assert_eq!(w.split('\n').count(), 3);
    }

    #[test]
    fn wrap_mixed_ascii_and_cjk() {
        let s = "これは CUDA 12.4 の ggml-cuda.dll を読み込む処理です。";
        let w = wrap_text(s, 24);
        for line in w.split('\n') {
            assert!(display_width(line) <= 24, "行过宽: {line}");
        }
        // 空格处应可断行，且不留行尾空格
        assert!(w.split('\n').all(|l| !l.ends_with(' ')), "{w}");
    }

    #[test]
    fn split_long_cue_by_punctuation() {
        let text = "第一句话在这里。第二句话也在这里。第三句话结束了。";
        let s = seg(10_000, 30_000, text); // 20s
        let parts = split_long_cue(&s, 8.0);
        assert!(parts.len() >= 2, "应被拆分: {:?}", parts);
        assert_eq!(parts[0].start_ms, 10_000);
        assert_eq!(parts.last().unwrap().end_ms, 30_000);
        // 时间单调且不重叠
        for w in parts.windows(2) {
            assert!(w[0].end_ms <= w[1].start_ms + 1);
            assert!(w[1].end_ms > w[1].start_ms);
        }
        // 文本无丢失
        let joined: String = parts.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(joined.replace(' ', ""), text);
    }

    #[test]
    fn split_keeps_short_cue() {
        let s = seg(0, 3000, "短い。");
        assert_eq!(split_long_cue(&s, 7.0).len(), 1);
    }

    #[test]
    fn layout_renumbers_and_applies_both_rules() {
        let v = vec![
            seg(0, 20_000, "とても長い文章です。まだまだ続きます。最後に終わります。"),
            seg(20_000, 22_000, "短い"),
        ];
        let out = layout(&v, 20, 7.0);
        assert!(out.len() >= 3);
        for (i, s) in out.iter().enumerate() {
            assert_eq!(s.index, i + 1);
            for line in s.text.split('\n') {
                assert!(display_width(line) <= 20, "{line}");
            }
            assert!(s.duration_secs() <= 7.5, "{}", s.duration_secs());
        }
        assert!(out.windows(2).all(|w| w[0].end_ms <= w[1].start_ms + 1));
    }
}
