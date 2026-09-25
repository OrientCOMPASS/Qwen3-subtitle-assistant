//! FFmpeg 流式解码：媒体文件 -> 16kHz 单声道 f32le PCM（stdout 管道）。
//!
//! 相比旧版的三点加固：
//! - `-nostdin`：不再吞掉终端输入（旧版 ffmpeg 会读 stdin，交互式终端下抢按键）；
//! - stderr 后台排空的同时**保留最后若干行**，失败时直接打印真实原因
//!   （旧版只以 debug 级记录，默认日志级别下用户永远看不到 ffmpeg 的报错）；
//! - `DecodeStream` 在 Drop 时杀掉子进程：转录中途出错也不会留下
//!   卡在写满管道上的 ffmpeg 僵尸进程。

use anyhow::{Context, Result};
use log::debug;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: i32 = 16000;

/// stderr 环形缓冲保留的行数
const STDERR_TAIL_LINES: usize = 25;

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

/// 正在运行的解码进程句柄。
pub struct DecodeStream {
    child: Child,
    tail: Arc<Mutex<VecDeque<String>>>,
    /// 读线程退出信号（进程结束时用于回收线程）
    _done: Receiver<()>,
}

impl DecodeStream {
    /// 取走 stdout（只能取一次），用于流式读取 PCM。
    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    /// 等待 ffmpeg 退出；非 0 退出时把 stderr 尾部一起报出来。
    pub fn finish(mut self) -> Result<()> {
        let status = self.child.wait().context("等待 FFmpeg 退出失败")?;
        if !status.success() {
            let tail = self
                .tail
                .lock()
                .map(|q| q.iter().cloned().collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            anyhow::bail!(
                "FFmpeg 解码异常退出（{}）。最后输出:\n{}",
                status,
                if tail.trim().is_empty() {
                    "（无 stderr 输出）".to_string()
                } else {
                    tail
                }
            );
        }
        Ok(())
    }
}

impl Drop for DecodeStream {
    fn drop(&mut self) {
        // 出错提前返回时，别让 ffmpeg 卡在写满的管道上
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// 启动 FFmpeg，将媒体解码为 16kHz 单声道 f32le 原始 PCM，通过 stdout 管道输出。
pub fn spawn_decode_stream(input: &Path) -> Result<DecodeStream> {
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-vn",                   // 丢弃视频流
            "-map", "0:a:0",         // 明确选第一条音频流（多音轨时行为确定）
            "-acodec", "pcm_f32le",  // 32 位浮点 PCM（直接喂给 ASR）
            "-ar", &SAMPLE_RATE.to_string(),
            "-ac", "1",              // 单声道
            "-f", "f32le",           // 输出原始 f32le 数据
            "pipe:1",                // 输出到 stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("无法启动 ffmpeg，请确认其已加入环境变量 PATH")?;

    let tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
    let (tx, rx) = channel::<()>();

    // 后台线程持续排空 stderr：
    // 若不读取，长视频一旦产生较多告警输出，管道写满后 ffmpeg 会阻塞，
    // 与主线程的 stdout 读取互相等待造成死锁。
    if let Some(stderr) = child.stderr.take() {
        let tail2 = Arc::clone(&tail);
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        debug!("[ffmpeg] {}", l);
                        if let Ok(mut q) = tail2.lock() {
                            if q.len() >= STDERR_TAIL_LINES {
                                q.pop_front();
                            }
                            q.push_back(l);
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = tx.send(());
        });
    } else {
        let _ = tx.send(());
    }

    Ok(DecodeStream {
        child,
        tail,
        _done: rx,
    })
}
