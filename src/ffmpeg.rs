use anyhow::{Context, Result};
use log::debug;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};

pub const SAMPLE_RATE: i32 = 16000;

/// 调用系统 ffprobe 获取音频总时长（用于进度条显示）
pub fn get_duration_secs(input: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input)
        .output()
        .context("无法执行 ffprobe，请确保其已加入环境变量 PATH")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffprobe 获取时长失败: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim()
        .parse::<f64>()
        .context("解析 ffprobe 时长输出失败")
}

/// 启动 FFmpeg 进程，将媒体文件解码为 16kHz 单声道 f32le 原始 PCM，
/// 通过 stdout 管道流式输出，不带 WAV 头。
pub fn spawn_decode_stream(input: &Path) -> Result<Child> {
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-vn",                  // 丢弃视频流
            "-acodec", "pcm_f32le", // 32 位浮点 PCM（直接喂给 ASR）
            "-ar", &SAMPLE_RATE.to_string(),
            "-ac", "1",             // 单声道
            "-f", "f32le",          // 输出原始 f32le 数据
            "pipe:1",               // 输出到 stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("无法启动 ffmpeg，请确认其已加入环境变量 PATH")?;

    // 后台线程持续排空 stderr：
    // 若不读取，长视频一旦产生较多告警输出，管道写满后 ffmpeg 会阻塞，
    // 与主线程的 stdout 读取互相等待造成死锁。内容以 debug 级别记录。
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(l) => debug!("[ffmpeg] {}", l),
                    Err(_) => break,
                }
            }
        });
    }

    Ok(child)
}
