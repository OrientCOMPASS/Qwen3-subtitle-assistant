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
    /// 语音时长（秒），用于 QC 提示词中的启发式判断。
    pub fn duration_secs(&self) -> f64 {
        (self.end_ms.saturating_sub(self.start_ms)) as f64 / 1000.0
    }

    /// 紧凑序列化（喂给 LLM 用）：只带序号与文本。
    ///
    /// 旧版把 `start_ms`/`end_ms` 也塞进 prompt 并要求模型原样回显，
    /// 既浪费 token 又让解析对小模型的输出格式极其敏感；时间轴始终由
    /// Rust 侧保留，模型只需要"第 i 条 -> 译文"。
    pub fn compact_json_list(segments: &[SubtitleSegment]) -> String {
        let items: Vec<String> = segments
            .iter()
            .map(|s| {
                format!(
                    "{{\"i\":{},\"t\":{}}}",
                    s.index,
                    serde_json::to_string(&s.text).unwrap_or_else(|_| "\"\"".into())
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }
}

/// LLM 翻译批次的输出单元（`[{"i":1,"t":"译文"}, ...]`）。
///
/// 字段名尽量短以省 token，同时用 alias 容忍小模型输出 `index`/`text` 等变体。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateItem {
    #[serde(default, alias = "index", alias = "id", alias = "idx", alias = "no")]
    pub i: i64,
    #[serde(
        default,
        alias = "text",
        alias = "target",
        alias = "translation",
        alias = "translated",
        alias = "zh",
        alias = "译文"
    )]
    pub t: String,
}

/// 全局上下文（翻译前由 LLM 从全文提取）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalContext {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub glossary: Vec<GlossaryItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryItem {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
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

    /// 按 source 去重合并（分块摘要时各块可能给出同一术语的不同译法，保留先出现的）。
    pub fn merge(&mut self, other: GlobalContext) {
        if self.summary.is_empty() {
            self.summary = other.summary;
        }
        for g in other.glossary {
            if g.source.trim().is_empty() {
                continue;
            }
            if !self
                .glossary
                .iter()
                .any(|x| x.source.trim() == g.source.trim())
            {
                self.glossary.push(g);
            }
        }
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
    #[serde(default, alias = "action", alias = "label")]
    pub decision: String,
    /// 兼容字段：部分小模型倾向输出布尔判定
    #[serde(default)]
    pub valid: Option<bool>,
    /// decision == "fix" 时为纠正后的文本；其余情况忽略
    #[serde(default, alias = "fixed", alias = "corrected")]
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
    /// 模型判 fix 但纠正文本不可信（差异过大/为空），已保留原文的次数
    pub fix_rejected: usize,
    /// 模型判 drop 但句子过长且非重复（几乎不可能是噪音），已保留原文的次数
    pub drop_rejected: usize,
    /// 因 LLM 调用出错而重试的次数
    pub retries: usize,
}

impl QcStats {
    pub fn summary(&self) -> String {
        format!(
            "质检 {} 句：保留 {}，纠正 {}，丢弃 {}，解析失败兜底 {}，纠正被拒 {}，丢弃被拒 {}，重试 {}",
            self.total,
            self.kept,
            self.fixed,
            self.dropped,
            self.failed,
            self.fix_rejected,
            self.drop_rejected,
            self.retries
        )
    }
}

/// 翻译阶段统计（用于收尾汇报与 CI 断言）。
#[derive(Debug, Clone, Default)]
pub struct TranslateStats {
    pub segments: usize,
    pub batches: usize,
    /// 因单次请求失败/解析失败而重试的次数
    pub retries: usize,
    /// 整批彻底失败、回退原文的批次数
    pub failed_batches: usize,
    /// 模型漏条/重编号，靠位置兜底救回的条数
    pub positional: usize,
    /// 译文与原文几乎相同（照抄未翻译）的条数
    pub copied: usize,
    /// 最终使用原文（未翻译）的条数
    pub untranslated: usize,
    /// 经过译文自检的条数
    pub reviewed: usize,
    /// 自检后实际修正的条数
    pub review_fixed: usize,
    /// 两轮审校后仍残留源语言文字（如假名）的条数
    pub residual: usize,
    /// 全局摘要被切成的块数（1 = 未触发分块）
    pub summary_chunks: usize,
    /// 因 prompt 超预算而被对半拆分的次数
    pub splits: usize,
}

impl TranslateStats {
    pub fn summary(&self) -> String {
        format!(
            "翻译 {} 条 / {} 批：重试 {}，拆分 {}，整批失败 {}，位置兜底 {}，照抄 {}，未翻译 {}，自检 {}/修正 {}，残留 {}，摘要分块 {}",
            self.segments,
            self.batches,
            self.retries,
            self.splits,
            self.failed_batches,
            self.positional,
            self.copied,
            self.untranslated,
            self.reviewed,
            self.review_fixed,
            self.residual,
            self.summary_chunks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(i: usize, s: u64, e: u64, t: &str) -> SubtitleSegment {
        SubtitleSegment {
            index: i,
            start_ms: s,
            end_ms: e,
            text: t.to_string(),
        }
    }

    #[test]
    fn compact_json_escapes_and_keeps_only_i_t() {
        let v = vec![seg(1, 0, 1000, "他说\"好\""), seg(2, 1000, 2000, "第二句")];
        let j = SubtitleSegment::compact_json_list(&v);
        assert_eq!(j, r#"[{"i":1,"t":"他说\"好\""},{"i":2,"t":"第二句"}]"#);
        assert!(!j.contains("start_ms"));
    }

    #[test]
    fn translate_item_accepts_aliases() {
        let a: TranslateItem = serde_json::from_str(r#"{"index":7,"text":"你好"}"#).unwrap();
        assert_eq!((a.i, a.t), (7, "你好".to_string()));
        let b: TranslateItem = serde_json::from_str(r#"{"i":8,"t":"世界"}"#).unwrap();
        assert_eq!((b.i, b.t), (8, "世界".to_string()));
        // 缺字段不报错（走位置兜底逻辑）
        let c: TranslateItem = serde_json::from_str(r#"{"t":"x"}"#).unwrap();
        assert_eq!((c.i, c.t.as_str()), (0, "x"));
    }

    #[test]
    fn global_context_merge_dedups_glossary() {
        let mut a = GlobalContext {
            summary: "A".into(),
            glossary: vec![GlossaryItem {
                source: "東京".into(),
                target: "东京".into(),
                note: None,
            }],
        };
        a.merge(GlobalContext {
            summary: "B".into(),
            glossary: vec![
                GlossaryItem {
                    source: "東京".into(),
                    target: "Tokyo".into(),
                    note: None,
                },
                GlossaryItem {
                    source: "山中".into(),
                    target: "山中".into(),
                    note: None,
                },
            ],
        });
        assert_eq!(a.summary, "A"); // 已有摘要不被覆盖
        assert_eq!(a.glossary.len(), 2);
        assert_eq!(a.glossary[0].target, "东京"); // 保留先出现的译法
    }

    #[test]
    fn segment_defaults_allow_partial_json() {
        let s: SubtitleSegment = serde_json::from_str(r#"{"text":"only"}"#).unwrap();
        assert_eq!((s.index, s.start_ms, s.end_ms), (0, 0, 0));
    }
}
