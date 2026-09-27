//! 共享类型。E3 单模型化后仅保留字幕段（LLM 工作流的翻译批次/质检/统计类型已随
//! qc.rs/translate.rs/llm.rs 移除）。

use serde::{Deserialize, Serialize};

/// 一条字幕（时间轴单位：毫秒）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleSegment {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub start_ms: u64,
    #[serde(default)]
    pub end_ms: u64,
    #[serde(default)]
    pub text: String,
}

impl SubtitleSegment {
    /// 语音时长（秒）。
    pub fn duration_secs(&self) -> f64 {
        (self.end_ms.saturating_sub(self.start_ms)) as f64 / 1000.0
    }
}
