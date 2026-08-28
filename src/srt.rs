use crate::types::SubtitleSegment;
use anyhow::Result;
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
    fs::write(path, content)?;
    log::info!("已写出字幕: {:?}", path);
    Ok(())
}