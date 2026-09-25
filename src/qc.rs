//! 逐句质检（Quality Check）：ASR 每识别出一句，立即交给 LLM，
//! 结合已通过质检的上文语境，判断该句是【有效语音 / 识别错误 / 噪音幻觉】，
//! 并允许 LLM 直接给出纠正后的文本。只有通过的句子才会进入后续翻译。
//!
//! 设计原则：
//! - fail-open：LLM 调用失败或输出无法解析时按“保留原文”处理，
//!   质检是增强环节，绝不能弄坏主管线；
//! - 低温度采样（0.2）+ 小 token 上限（256），单句质检延迟可控；
//! - 上下文只带最近 N 条【已通过质检】的文本，滚动更新。

use crate::llm::{extract_json, LlmSession};
use crate::prompt::PromptStore;
use crate::types::{QcStats, QcVerdict, SubtitleSegment};
use log::{info, warn};
use std::collections::{HashMap, VecDeque};

const QC_SYSTEM: &str =
    "你是一个自动语音识别(ASR)质检引擎。你只判断给定字幕是否为有效语音内容，并按要求仅输出一行 JSON，绝不输出任何其他文字。";

pub struct QualityChecker {
    context_size: usize,
    history: VecDeque<String>,
    stats: QcStats,
    max_attempts: usize,
}

impl QualityChecker {
    pub fn new(context_size: usize) -> Self {
        Self {
            context_size: context_size.max(1),
            history: VecDeque::new(),
            stats: QcStats::default(),
            max_attempts: 2,
        }
    }

    pub fn stats(&self) -> &QcStats {
        &self.stats
    }

    /// 质检一条 ASR 结果。
    /// 返回 `true` 表示保留（`seg.text` 可能已被就地纠正）；`false` 表示丢弃。
    pub fn check(
        &mut self,
        session: &mut LlmSession,
        prompts: &PromptStore,
        seg: &mut SubtitleSegment,
    ) -> bool {
        self.stats.total += 1;

        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("context".to_string(), self.history_text());
        vars.insert("text".to_string(), seg.text.clone());
        vars.insert("duration".to_string(), format!("{:.1}", seg.duration_secs()));

        let prompt = match prompts.render("qc_segment.txt", &vars) {
            Ok(p) => p,
            Err(e) => {
                warn!("[QC] 读取质检提示词失败，保留原文: {:#}", e);
                return self.fail_open(seg);
            }
        };

        for attempt in 1..=self.max_attempts {
            let raw = match session.chat(QC_SYSTEM, &prompt, 0.2, 256) {
                Ok(r) => r,
                Err(e) => {
                    warn!("[QC] LLM 调用失败（第 {} 次），保留原文: {:#}", attempt, e);
                    break;
                }
            };

            if let Some(json_str) = extract_json(&raw, false) {
                match serde_json::from_str::<QcVerdict>(&json_str) {
                    Ok(verdict) => return self.apply(verdict, seg),
                    Err(e) => warn!(
                        "[QC] 第 {} 次尝试 JSON 字段不符（{}），原始输出: {}",
                        attempt,
                        e,
                        truncate(&raw, 120)
                    ),
                }
            } else {
                warn!(
                    "[QC] 第 {} 次尝试未找到 JSON，原始输出: {}",
                    attempt,
                    truncate(&raw, 120)
                );
            }
        }

        self.fail_open(seg)
    }

    /// 解析失败/调用失败时的兜底：保留原文并计入统计。
    fn fail_open(&mut self, seg: &SubtitleSegment) -> bool {
        self.stats.failed += 1;
        self.stats.kept += 1;
        self.push_history(&seg.text);
        true
    }

    fn apply(&mut self, verdict: QcVerdict, seg: &mut SubtitleSegment) -> bool {
        let decision = verdict.normalized_decision();
        let reason = verdict.reason.trim();

        // 兼容 {"valid": true/false} 风格的输出
        let decision = if decision.is_empty() {
            match verdict.valid {
                Some(true) => "keep".to_string(),
                Some(false) => "drop".to_string(),
                None => "keep".to_string(),
            }
        } else {
            decision
        };

        match decision.as_str() {
            "drop" | "noise" | "invalid" | "reject" | "discard" => {
                self.stats.dropped += 1;
                info!(
                    "[QC✂ 丢弃] 「{}」 理由: {}",
                    truncate(&seg.text, 60),
                    if reason.is_empty() { "（未提供）" } else { reason }
                );
                false
            }
            "fix" | "correct" | "corrected" | "rewrite" => {
                let fixed = verdict.text.trim();
                if fixed.is_empty() {
                    // 模型说 fix 却没给文本：按丢弃处理更安全（多半是噪音）
                    self.stats.dropped += 1;
                    info!(
                        "[QC✂ 丢弃] 「{}」 理由: fix 但未给出纠正文本",
                        truncate(&seg.text, 60)
                    );
                    return false;
                }
                self.stats.fixed += 1;
                info!(
                    "[QC✎ 纠正] 「{}」=>「{}」 理由: {}",
                    truncate(&seg.text, 40),
                    truncate(fixed, 40),
                    if reason.is_empty() { "（未提供）" } else { reason }
                );
                seg.text = fixed.to_string();
                self.push_history(&seg.text);
                true
            }
            other => {
                if other != "keep" && other != "ok" && other != "valid" && other != "true" {
                    warn!("[QC] 未知 decision「{}」，按保留处理", other);
                }
                self.stats.kept += 1;
                self.push_history(&seg.text);
                true
            }
        }
    }

    fn push_history(&mut self, text: &str) {
        self.history.push_back(text.to_string());
        while self.history.len() > self.context_size {
            self.history.pop_front();
        }
    }

    fn history_text(&self) -> String {
        if self.history.is_empty() {
            return "（这是第一句，暂无上文）".to_string();
        }
        self.history
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}. {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let t: String = s.chars().take(max_chars).collect();
    if t.chars().count() < s.chars().count() {
        format!("{}…", t)
    } else {
        t
    }
}
