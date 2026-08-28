use crate::ffmpeg::SAMPLE_RATE;
use crate::srt::format_timestamp;
use crate::types::SubtitleSegment;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::io::Read;
use std::path::Path;

pub struct AsrEngine {
    vad: VoiceActivityDetector,
    recognizer: OfflineRecognizer,
}

impl AsrEngine {
    pub fn new(asr_model_dir: &Path, vad_model: &Path) -> Result<Self> {
        // 1. 初始化 VAD (Silero)
        let mut vad_config = VadModelConfig::default();
        vad_config.silero_vad = SileroVadModelConfig {
            model: Some(vad_model.to_string_lossy().into()),
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            window_size: 512, // 16kHz 下 32ms
            ..Default::default()
        };
        vad_config.sample_rate = SAMPLE_RATE;
        vad_config.num_threads = 1;

        let vad = VoiceActivityDetector::create(&vad_config, 60.0)
            .context("初始化 VAD 失败，请检查 silero_vad.onnx 路径")?;

        // 2. 初始化 ASR (Qwen3-ASR)
        let mut model_config = OfflineModelConfig::default();
        model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
            conv_frontend: Some(format!("{}/conv_frontend.onnx", asr_model_dir.display()).into()),
            encoder: Some(format!("{}/encoder.int8.onnx", asr_model_dir.display()).into()),
            decoder: Some(format!("{}/decoder.int8.onnx", asr_model_dir.display()).into()),
            tokenizer: Some(format!("{}/tokenizer", asr_model_dir.display()).into()),
            ..Default::default()
        };
        model_config.num_threads = 4;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config = model_config;
        config.decoding_method = Some("greedy_search".into());

        let recognizer = OfflineRecognizer::create(&config)
            .context("初始化 ASR 识别器失败，请检查模型目录结构")?;

        Ok(Self { vad, recognizer })
    }

    /// 从 FFmpeg stdout 流式读取 f32le 音频，进行 VAD 切片与 ASR 识别
    pub fn transcribe_stream<R: Read>(&mut self, mut reader: R, total_duration_secs: f64) -> Result<Vec<SubtitleSegment>> {
        let mut segments: Vec<SubtitleSegment> = Vec::new();
        
        // 进度条设置
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

        let mut audio_buf: Vec<u8> = Vec::with_capacity(8192);
        let mut read_buf = [0u8; 4096]; // 每次读取 4KB
        let mut total_bytes_read: usize = 0;

        loop {
            let n = reader.read(&mut read_buf).context("读取 FFmpeg 音频流失败")?;
            if n == 0 {
                break; // EOF
            }
            audio_buf.extend_from_slice(&read_buf[..n]);
            total_bytes_read += n;

            // 更新进度条 (基于已读取的字节数计算时间)
            let current_sec = (total_bytes_read / 4) as f64 / SAMPLE_RATE as f64;
            if total_duration_secs > 0.0 {
                pb.set_position(current_sec as u64);
            } else {
                pb.set_message(format!("已处理: {:.1}s", current_sec));
                pb.tick();
            }

            // Silero VAD 要求每次输入精确的 512 个采样点 (2048 bytes)
            while audio_buf.len() >= 2048 {
                let chunk_bytes: Vec<u8> = audio_buf.drain(..2048).collect();
                let samples: Vec<f32> = chunk_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                self.vad.accept_waveform(&samples);

                // 处理 VAD 切出的完整语音段
                while !self.vad.is_empty() {
                    if let Some(segment) = self.vad.front() {
                        let seg_samples = segment.samples();
                        let start_sample = segment.start();

                        if !seg_samples.is_empty() {
                            let stream = self.recognizer.create_stream();
                            stream.accept_waveform(SAMPLE_RATE, seg_samples);
                            self.recognizer.decode(&stream);

                            if let Some(result) = stream.get_result() {
                                let text = result.text.trim();
                                if !text.is_empty() {
                                    let start_ms = (start_sample as u64 * 1000) / SAMPLE_RATE as u64;
                                    let dur_ms = (seg_samples.len() as u64 * 1000) / SAMPLE_RATE as u64;
                                    let end_ms = start_ms + dur_ms;

                                    let seg = SubtitleSegment {
                                        index: segments.len() + 1,
                                        start_ms,
                                        end_ms,
                                        text: text.to_string(),
                                    };
                                    
                                    // 实时打印转录结果
                                    info!("[ASR] {} -> {}", format_timestamp(seg.start_ms), seg.text);
                                    segments.push(seg);
                                }
                            }
                        }
                    }
                    self.vad.pop();
                }
            }
        }

        // 冲刷 VAD 缓冲区尾部
        self.vad.flush();
        while !self.vad.is_empty() {
            if let Some(segment) = self.vad.front() {
                let seg_samples = segment.samples();
                let start_sample = segment.start();
                if !seg_samples.is_empty() {
                    let stream = self.recognizer.create_stream();
                    stream.accept_waveform(SAMPLE_RATE, seg_samples);
                    self.recognizer.decode(&stream);
                    if let Some(result) = stream.get_result() {
                        let text = result.text.trim();
                        if !text.is_empty() {
                            let start_ms = (start_sample as u64 * 1000) / SAMPLE_RATE as u64;
                            let dur_ms = (seg_samples.len() as u64 * 1000) / SAMPLE_RATE as u64;
                            let seg = SubtitleSegment {
                                index: segments.len() + 1,
                                start_ms,
                                end_ms: start_ms + dur_ms,
                                text: text.to_string(),
                            };
                            info!("[ASR] {} -> {}", format_timestamp(seg.start_ms), seg.text);
                            segments.push(seg);
                        }
                    }
                }
            }
            self.vad.pop();
        }

        pb.finish_with_message("转录完成");
        info!("ASR 流程结束，共识别 {} 条字幕。", segments.len());
        Ok(segments)
    }
}