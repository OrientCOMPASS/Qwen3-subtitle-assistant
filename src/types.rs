use serde::{Deserialize, Serialize};

/// 一条字幕（时间轴单位：毫秒）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleSegment {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

impl SubtitleSegment {
    /// 语音时长（秒），用于 QC 提示词中的启发式判断。
    pub fn duration_secs(&self) -> f64 {
        (self.end_ms.saturating_sub(self.start_ms)) as f64 / 1000.0
    }
}

/// 全局上下文（翻译前由 LLM 从全文提取）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalContext {
    pub summary: String,
    #[serde(default)]
    pub glossary: Vec<GlossaryItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryItem {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub note: Option<String>,
}

impl GlobalContext {
    pub fn glossary_as_text(&self) -> String {
        if self.glossary.is_empty() {
            return "（无）".to_string();
        }
        self.glossary
            .iter()
            .map(|g| {
                format!(
                    "- {} -> {}{}",
                    g.source,
                    g.target,
                    g.note
                        .as_ref()
                        .map(|n| format!("（{}）", n))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// 逐句质检（QC）相关类型
// ============================================================================

/// LLM 对单条 ASR 结果的质检判决。
///
/// 期望模型输出 JSON：`{"decision": "keep|fix|drop", "text": "...", "reason": "..."}`
/// 同时容忍 `{"valid": true/false, ...}` 风格（decision 缺省时回退到 valid）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcVerdict {
    /// keep = 有效保留；fix = 识别有误、给出纠正；drop = 噪音/幻觉、丢弃
    #[serde(default)]
    pub decision: String,
    /// 兼容字段：部分小模型倾向输出布尔判定
    #[serde(default)]
    pub valid: Option<bool>,
    /// decision == "fix" 时为纠正后的文本；其余情况忽略
    #[serde(default)]
    pub text: String,
    /// 简短理由（仅用于日志）
    #[serde(default)]
    pub reason: String,
}

impl QcVerdict {
    /// 归一化 decision 字段（容忍大小写、引号、多余空白，以及 valid 风格输出）。
    pub fn normalized_decision(&self) -> String {
        self.decision
            .trim()
            .trim_matches(|c| c == '"' || c == '\'' || c == '“' || c == '”')
            .to_ascii_lowercase()
    }
}

/// 质检统计。
#[derive(Debug, Clone, Default)]
pub struct QcStats {
    pub total: usize,
    pub kept: usize,
    pub fixed: usize,
    pub dropped: usize,
    /// LLM 输出无法解析、按“保留原文”处理（fail-open）的次数
    pub failed: usize,
}

impl QcStats {
    pub fn summary(&self) -> String {
        format!(
            "质检 {} 句：保留 {}，纠正 {}，丢弃 {}，解析失败兜底 {}",
            self.total, self.kept, self.fixed, self.dropped, self.failed
        )
    }
}
