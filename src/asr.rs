//! Silero VAD 切片 + sherpa-onnx Qwen3-ASR 转录。
//!
//! 要点：
//! - ASR 推理 provider 由运行时 DLL 探测决定（"cuda" 时 sherpa-onnx 会让
//!   onnxruntime.dll 动态加载外置的 onnxruntime_providers_cuda.dll）；
//! - provider=cuda 创建失败时自动回退 cpu 重试（例如用户只装了 CPU 版 ORT）；
//! - 转录循环接受 `hook` 回调：每识别出一句立即回调（供逐句 LLM 质检），
//!   回调返回 false 的句子不会进入结果集；
//! - Qwen3-ASR 的生成上限（`max_new_tokens` / `max_total_len`）与 hotwords
//!   偏置词都可配置：sherpa-onnx 的默认值（128/512）在长语音段上会静默截断，
//!   而 hotwords 是纠正专有名词最便宜的手段（比事后 LLM 质检更早生效）。

use crate::ffmpeg::SAMPLE_RATE;
use crate::srt::format_timestamp;
use crate::types::SubtitleSegment;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::io::Read;
use std::path::Path;

pub struct AsrOptions {
    /// onnxruntime provider："cpu" 或 "cuda"
    pub provider: String,
    /// ASR CPU 推理线程数
    pub num_threads: i32,
    /// Qwen3-ASR 单段最多生成的 token 数
    pub max_new_tokens: i32,
    /// Qwen3-ASR 最大总序列长度（音频 token + 文本 token）
    pub max_total_len: i32,
    /// hotwords 偏置词（英文逗号分隔），空串表示不启用
    pub hotwords: String,
    /// VAD 缓冲区秒数（= 单条语音段长度上限）
    pub vad_buffer_secs: f32,
    /// VAD 判定语音结束所需的最短静音（秒）
    pub vad_min_silence: f32,
}

pub struct AsrEngine {
    vad: VoiceActivityDetector,
    recognizer: OfflineRecognizer,
    provider: String,
}

/// Silero VAD 每帧采样点数（16kHz 下 32ms）
const VAD_WINDOW: usize = 512;

impl AsrEngine {
    pub fn new(asr_model_dir: &Path, vad_model: &Path, opts: &AsrOptions) -> Result<Self> {
        anyhow::ensure!(
            asr_model_dir.is_dir(),
            "ASR 模型目录不存在: {:?}（可用 --asr-model-dir 指定，或运行 scripts/download_models.ps1 下载）",
            asr_model_dir
        );
        anyhow::ensure!(
            vad_model.is_file(),
            "VAD 模型不存在: {:?}（可用 --vad-model 指定，或运行 scripts/download_models.ps1 下载）",
            vad_model
        );

        // ---- 1. VAD（模型极小，固定 CPU 即可）----
        let mut vad_config = VadModelConfig::default();
        vad_config.silero_vad = SileroVadModelConfig {
            model: Some(vad_model.to_string_lossy().into()),
            threshold: 0.5,
            min_silence_duration: opts.vad_min_silence,
            min_speech_duration: 0.25,
            window_size: VAD_WINDOW as i32,
            ..Default::default()
        };
        vad_config.sample_rate = SAMPLE_RATE;
        vad_config.num_threads = 1;

        let vad = VoiceActivityDetector::create(&vad_config, opts.vad_buffer_secs)
            .context("初始化 VAD 失败，请检查 silero_vad.onnx 路径")?;
        info!(
            "VAD 就绪（buffer={}s, min_silence={}s）",
            opts.vad_buffer_secs, opts.vad_min_silence
        );

        // ---- 2. Qwen3-ASR 识别器 ----
        let recognizer = match OfflineRecognizer::create(&Self::recognizer_config(
            asr_model_dir,
            opts,
        )) {
            Some(r) => {
                info!("ASR 识别器已创建，provider = {}", opts.provider);
                (r, opts.provider.clone())
            }
            None if opts.provider == "cuda" => {
                warn!(
                    "CUDA provider 创建 ASR 失败（外置 onnxruntime CUDA DLL 可能不完整，\
                     常见原因是缺 cufft/cudnn 等依赖），自动回退 CPU…"
                );
                let cpu_opts = AsrOptions {
                    provider: "cpu".to_string(),
                    ..clone_opts(opts)
                };
                let r = OfflineRecognizer::create(&Self::recognizer_config(asr_model_dir, &cpu_opts))
                    .context("初始化 ASR 识别器失败（CPU 回退亦失败），请检查模型目录结构")?;
                (r, "cpu".to_string())
            }
            None => anyhow::bail!(
                "初始化 ASR 识别器失败，请检查模型目录结构: {:?}",
                asr_model_dir
            ),
        };

        Ok(Self {
            vad,
            recognizer: recognizer.0,
            provider: recognizer.1,
        })
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// 复用于下一个文件前清空 VAD 状态与残留语音段。
    pub fn reset(&self) {
        self.vad.clear();
        self.vad.reset();
    }

    fn recognizer_config(asr_model_dir: &Path, opts: &AsrOptions) -> OfflineRecognizerConfig {
        let dir = asr_model_dir.display();
        let mut model_config = OfflineModelConfig::default();
        model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
            conv_frontend: Some(format!("{}/conv_frontend.onnx", dir).into()),
            encoder: Some(format!("{}/encoder.int8.onnx", dir).into()),
            decoder: Some(format!("{}/decoder.int8.onnx", dir).into()),
            tokenizer: Some(format!("{}/tokenizer", dir).into()),
            max_total_len: opts.max_total_len,
            max_new_tokens: opts.max_new_tokens,
            hotwords: if opts.hotwords.is_empty() {
                None
            } else {
                Some(opts.hotwords.clone())
            },
            ..Default::default()
        };
        model_config.num_threads = opts.num_threads;
        model_config.provider = Some(opts.provider.clone());

        let mut config = OfflineRecognizerConfig::default();
        config.model_config = model_config;
        config.decoding_method = Some("greedy_search".into());
        config
    }

    /// 从 FFmpeg stdout 流式读取 f32le 音频，VAD 切片 -> ASR -> 逐句回调 hook。
    ///
    /// `hook(&mut seg)` 返回 true 表示该句保留（文本可能已被 hook 纠正），
    /// 返回 false 表示丢弃。保留的句子才会获得最终 index。
    pub fn transcribe_stream<R, F>(
        &mut self,
        mut reader: R,
        total_duration_secs: f64,
        hook: &mut F,
    ) -> Result<Vec<SubtitleSegment>>
    where
        R: Read,
        F: FnMut(&mut SubtitleSegment) -> bool,
    {
        let mut segments: Vec<SubtitleSegment> = Vec::new();

        let pb = if total_duration_secs > 0.0 {
            let pb = ProgressBar::new(total_duration_secs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}s/{len}s ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb
        } else {
            ProgressBar::new_spinner()
        };

        let mut pending: Vec<f32> = Vec::with_capacity(SAMPLE_RATE as usize);
        let mut read_buf = [0u8; 8192];
        let mut total_bytes_read: u64 = 0;
        let mut leftover = [0u8; 4];
        let mut leftover_len = 0usize;

        loop {
            let n = reader.read(&mut read_buf).context("读取 FFmpeg 音频流失败")?;
            if n == 0 {
                break; // EOF
            }
            total_bytes_read += n as u64;

            // f32le 字节 -> 采样点（跨 read 边界保留半个采样，避免尾部丢字节）
            let mut bytes = Vec::with_capacity(leftover_len + n);
            if leftover_len > 0 {
                bytes.extend_from_slice(&leftover[..leftover_len]);
                leftover_len = 0;
            }
            bytes.extend_from_slice(&read_buf[..n]);
            let mut off = 0usize;
            while off + 4 <= bytes.len() {
                pending.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
                off += 4;
            }
            if off < bytes.len() {
                let rest = &bytes[off..];
                leftover[..rest.len()].copy_from_slice(rest);
                leftover_len = rest.len();
            }

            let current_sec = (total_bytes_read / 4) as f64 / SAMPLE_RATE as f64;
            if total_duration_secs > 0.0 {
                pb.set_position(current_sec as u64);
            } else {
                pb.set_message(format!("已处理: {:.1}s", current_sec));
                pb.tick();
            }

            // Silero VAD 要求每次输入精确的 512 个采样点
            let mut cursor = 0;
            while pending.len() - cursor >= VAD_WINDOW {
                self.vad.accept_waveform(&pending[cursor..cursor + VAD_WINDOW]);
                cursor += VAD_WINDOW;
                self.drain_vad(&mut segments, hook);
            }
            if cursor > 0 {
                pending.drain(..cursor);
            }
        }

        // 冲刷 VAD 缓冲区尾部
        self.vad.flush();
        self.drain_vad(&mut segments, hook);

        pb.finish_with_message("转录完成");
        info!("ASR 流程结束，最终保留 {} 条字幕。", segments.len());
        Ok(segments)
    }

    /// 取出 VAD 已完成切分的所有语音段，逐段 ASR 并交给 hook 质检。
    fn drain_vad<F>(&mut self, segments: &mut Vec<SubtitleSegment>, hook: &mut F)
    where
        F: FnMut(&mut SubtitleSegment) -> bool,
    {
        while !self.vad.is_empty() {
            let (text, start_ms, end_ms) = {
                let segment = match self.vad.front() {
                    Some(s) => s,
                    None => break,
                };
                let samples = segment.samples();
                let start_sample = segment.start();
                if samples.is_empty() {
                    self.vad.pop();
                    continue;
                }
                let dur_secs = samples.len() as f64 / SAMPLE_RATE as f64;
                let text = self.recognize(samples);
                let start_ms = (start_sample.max(0) as u64 * 1000) / SAMPLE_RATE as u64;
                let dur_ms = (samples.len() as u64 * 1000) / SAMPLE_RATE as u64;
                if dur_secs > 20.0 {
                    info!(
                        "[ASR] 语音段较长（{:.1}s @ {}），如出现句尾丢失可调小 --vad-buffer-secs 或调大 --asr-max-new-tokens",
                        dur_secs,
                        format_timestamp(start_ms)
                    );
                }
                (text, start_ms, start_ms + dur_ms)
            };
            self.vad.pop();

            let Some(text) = text else { continue };

            let mut seg = SubtitleSegment {
                index: 0, // 通过质检后才分配最终序号
                start_ms,
                end_ms,
                text,
            };

            if hook(&mut seg) {
                seg.index = segments.len() + 1;
                info!(
                    "[ASR✔] {} -> {} ({:.1}s)",
                    format_timestamp(seg.start_ms),
                    seg.text,
                    seg.duration_secs()
                );
                segments.push(seg);
            }
        }
    }

    fn recognize(&self, samples: &[f32]) -> Option<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, samples);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text.trim().to_string())
            .filter(|t| !t.is_empty())
    }
}

/// `AsrOptions` 没有实现 Clone（保持结构体简单），回退 CPU 时手工复制一份。
fn clone_opts(o: &AsrOptions) -> AsrOptions {
    AsrOptions {
        provider: o.provider.clone(),
        num_threads: o.num_threads,
        max_new_tokens: o.max_new_tokens,
        max_total_len: o.max_total_len,
        hotwords: o.hotwords.clone(),
        vad_buffer_secs: o.vad_buffer_secs,
        vad_min_silence: o.vad_min_silence,
    }
}
