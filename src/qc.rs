//! 逐句质检（Quality Check）：ASR 每识别出一句，立即交给 LLM，
//! 结合已通过质检的上文语境，判断该句是【有效语音 / 识别错误 / 噪音幻觉】，
//! 并允许 LLM 直接给出纠正后的文本。只有通过的句子才会进入后续翻译。
//!
//! 设计原则：
//! - **fail-open**：LLM 调用失败或输出无法解析时按“保留原文”处理，
//!   质检是增强环节，绝不能弄坏主管线；
//! - **贪心解码**：质检是判定任务，要可复现，不做随机采样；
//! - **重试带反馈**：解析失败时把“上次不是合法 JSON”的提醒追加进 prompt 再试，
//!   而不是原样重发（贪心下原样重发只会得到同样的错误输出）；
//! - **纠正必须可信**：模型给的 `fix` 文本要与原文足够相似、长度比合理，
//!   否则视为幻觉、保留原文并计数（旧版会无条件接受改写，甚至因缺字段直接丢句）。

use crate::llm::{parse_json, LlmSession, Sampling};
use crate::prompt::PromptStore;
use crate::types::{QcStats, QcVerdict, SubtitleSegment};
use log::{info, warn};
use std::collections::{HashMap, VecDeque};

const QC_SYSTEM: &str =
    "你是一个自动语音识别(ASR)质检引擎。你只判断给定字幕是否为有效语音内容，并按要求仅输出一行 JSON，绝不输出任何其他文字。";

/// 解析失败后追加的格式提醒（重试时生效）。
const FORMAT_HINT: &str = "\n\n【重要】你上一次的输出不是合法 JSON。现在请只输出一行 JSON，形如 {\"decision\":\"keep\",\"text\":\"\",\"reason\":\"简短理由\"}，不要输出思考过程、解释或代码块标记。\n/no_think";

pub struct QualityChecker {
    context_size: usize,
    history: VecDeque<String>,
    stats: QcStats,
    max_attempts: usize,
    /// fix 文本与原文的最小相似度（0~1），低于此值视为幻觉、保留原文
    min_similarity: f32,
    /// fix 文本相对原文的长度比允许区间
    len_ratio: (f32, f32),
    /// 超过这个字符数的句子不允许被 drop（除非高度重复），0 = 关闭该保护
    keep_min_chars: usize,
    /// 采样参数（Qwen3 模型卡明确禁止贪心；解析失败时换种子重抽）
    sampling: Sampling,
}

impl QualityChecker {
    pub fn new(
        context_size: usize,
        max_attempts: usize,
        min_similarity: f32,
        keep_min_chars: usize,
        sampling: Sampling,
    ) -> Self {
        Self {
            context_size: context_size.max(1),
            history: VecDeque::new(),
            stats: QcStats::default(),
            max_attempts: max_attempts.clamp(1, 5),
            min_similarity: min_similarity.clamp(0.0, 1.0),
            len_ratio: (0.3, 3.0),
            keep_min_chars,
            sampling,
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
        qc_max_tokens: u32,
    ) -> bool {
        self.stats.total += 1;

        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("context".to_string(), self.history_text());
        vars.insert("text".to_string(), seg.text.clone());
        vars.insert("duration".to_string(), format!("{:.1}", seg.duration_secs()));

        let base_prompt = match prompts.render("qc_segment.txt", &vars) {
            Ok(p) => p,
            Err(e) => {
                warn!("[QC] 读取质检提示词失败，保留原文: {:#}", e);
                return self.fail_open(seg);
            }
        };

        let mut prompt = base_prompt.clone();
        for attempt in 1..=self.max_attempts {
            // 每次尝试换一个种子重新抽取（模型卡明确不建议贪心解码）
            let raw = match session.chat(
                QC_SYSTEM,
                &prompt,
                self.sampling.resample(attempt),
                qc_max_tokens,
            ) {
                Ok(r) => r,
                Err(e) => {
                    self.stats.retries += 1;
                    warn!("[QC] LLM 调用失败（第 {}/{} 次）: {:#}", attempt, self.max_attempts, e);
                    continue; // 请求失败值得原样重试
                }
            };

            match parse_json::<QcVerdict>(&raw, false) {
                Ok(verdict) => return self.apply(verdict, seg),
                Err(e) => {
                    self.stats.retries += 1;
                    warn!(
                        "[QC] 第 {}/{} 次输出不可解析（{}）",
                        attempt, self.max_attempts, e
                    );
                    // 追加格式提醒 + 换种子重抽（原样原种子重试没有意义）
                    prompt = format!("{}{}", base_prompt, FORMAT_HINT);
                }
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
                // drop 是不可逆的（内容永久丢失），所以也要过一道客观校验：
                // 实测 1.7B 模型会把"自己看不懂的外语长句"判成背景噪音，
                // 一段 39 字的日语演讲被整句丢掉。长句 + 非重复文本几乎不可能是噪音。
                if let Err(why) = self.validate_drop(&seg.text) {
                    self.stats.drop_rejected += 1;
                    self.stats.kept += 1;
                    warn!(
                        "[QC⚠ 丢弃被拒] 「{}」 {}，保留原文（模型给的理由: {}）",
                        truncate(&seg.text, 50),
                        why,
                        if reason.is_empty() { "（未提供）" } else { reason }
                    );
                    self.push_history(&seg.text);
                    return true;
                }
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
                match self.validate_fix(&seg.text, fixed) {
                    Ok(f) => {
                        self.stats.fixed += 1;
                        info!(
                            "[QC✎ 纠正] 「{}」=>「{}」 理由: {}",
                            truncate(&seg.text, 40),
                            truncate(&f, 40),
                            if reason.is_empty() { "（未提供）" } else { reason }
                        );
                        seg.text = f;
                        self.push_history(&seg.text);
                        true
                    }
                    Err(why) => {
                        // 纠正不可信：保留原文（丢句是不可逆的，保留原文最多是质量差一点）
                        self.stats.fix_rejected += 1;
                        self.stats.kept += 1;
                        warn!(
                            "[QC⚠ 纠正被拒] 「{}」 {}，保留原文",
                            truncate(&seg.text, 40),
                            why
                        );
                        self.push_history(&seg.text);
                        true
                    }
                }
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

    /// 校验 drop 判决是否可信：长且非重复的句子不允许丢弃。
    fn validate_drop(&self, text: &str) -> Result<(), String> {
        if self.keep_min_chars == 0 {
            return Ok(());
        }
        let n = text.chars().count();
        if n < self.keep_min_chars {
            return Ok(());
        }
        if is_repetitive(text) {
            return Ok(()); // 长但高度重复 -> 很可能是幻觉/歌词，允许丢弃
        }
        Err(format!(
            "长度 {} 字且非重复文本，不像噪音（阈值 --qc-keep-min-chars {}）",
            n, self.keep_min_chars
        ))
    }

    /// 校验模型给出的纠正文本是否可信。
    fn validate_fix(&self, original: &str, fixed: &str) -> Result<String, String> {
        if fixed.is_empty() {
            return Err("判为 fix 但未给出纠正文本".to_string());
        }
        let o = original.chars().count();
        let f = fixed.chars().count();
        if o > 0 {
            let ratio = f as f32 / o as f32;
            if ratio < self.len_ratio.0 || ratio > self.len_ratio.1 {
                return Err(format!("长度比 {:.2} 超出允许区间", ratio));
            }
        }
        let sim = char_similarity(original, fixed);
        if sim < self.min_similarity {
            return Err(format!("与原文相似度仅 {:.2}", sim));
        }
        Ok(fixed.to_string())
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

/// 文本是否"高度重复"——幻觉与歌词的典型特征。
///
/// 两个信号任一命中即算重复：
/// - 去重后的字符占比过低（< 0.35）；
/// - 出现次数最多的 2-gram 覆盖了全文 40% 以上的字符。
pub fn is_repetitive(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 8 {
        return false;
    }
    let mut sorted = chars.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() as f32 / (chars.len() as f32) < 0.35 {
        return true;
    }
    let mut bigrams: HashMap<String, usize> = HashMap::new();
    for w in chars.windows(2) {
        *bigrams.entry(w.iter().collect()).or_insert(0) += 1;
    }
    let max_rep = bigrams.values().copied().max().unwrap_or(0);
    ((max_rep * 2) as f32) >= (chars.len() as f32) * 0.4
}

/// 字符级相似度：`1 - 编辑距离 / max(len)`，取值 0~1。
pub fn char_similarity(a: &str, b: &str) -> f32 {
    let x: Vec<char> = a.chars().collect();
    let y: Vec<char> = b.chars().collect();
    if x.is_empty() && y.is_empty() {
        return 1.0;
    }
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let dist = levenshtein(&x, &y);
    1.0 - dist as f32 / x.len().max(y.len()) as f32
}

fn levenshtein(x: &[char], y: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=y.len()).collect();
    let mut cur = vec![0usize; y.len() + 1];
    for (i, &xc) in x.iter().enumerate().map(|(i, c)| (i + 1, c)) {
        cur[0] = i;
        for (j, &yc) in y.iter().enumerate().map(|(j, c)| (j + 1, c)) {
            let cost = if xc == yc { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[y.len()]
}

fn truncate(s: &str, max_chars: usize) -> String {
    let t: String = s.chars().take(max_chars).collect();
    if t.chars().count() < s.chars().count() {
        format!("{}…", t)
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_basics() {
        assert_eq!(char_similarity("abc", "abc"), 1.0);
        assert_eq!(char_similarity("", ""), 1.0);
        assert_eq!(char_similarity("abc", ""), 0.0);
        // 一字之差 / 5 字 => 0.8
        assert!(char_similarity("今日は良い", "今日はかい") > 0.7);
        assert!(char_similarity("同音字错误", "同音字错误") > 0.99);
        assert!(char_similarity("完全不同的句子啊", "abc") < 0.2);
    }

    #[test]
    fn levenshtein_values() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(levenshtein(&a, &b), 3);
    }

    fn checker(sim: f32) -> QualityChecker {
        QualityChecker::new(3, 2, sim, 30, Sampling::qwen3(42))
    }

    #[test]
    fn fix_rejected_when_too_different() {
        let c = checker(0.35);
        let e = c.validate_fix("今日はとても良い天気です", "明日来ないでください").unwrap_err();
        assert!(e.contains("相似度"), "{e}");
    }

    #[test]
    fn fix_rejected_when_empty() {
        let c = checker(0.35);
        let e = c.validate_fix("今日は良い天気", "").unwrap_err();
        assert!(e.contains("未给出"), "{e}");
    }

    #[test]
    fn fix_rejected_when_length_ratio_extreme() {
        let c = checker(0.0); // 关掉相似度门槛，只测长度比
        let e = c
            .validate_fix("今日は良い天気です", "今")
            .unwrap_err();
        assert!(e.contains("长度比"), "{e}");
        assert!(c.validate_fix("今日は良い天気です", "今日は良い天気").is_ok());
    }

    #[test]
    fn fix_accepted_for_homophone_correction() {
        let c = checker(0.35);
        assert!(c
            .validate_fix("我要去东经", "我要去东京")
            .is_ok());
    }

    #[test]
    fn verdict_normalization() {
        let v = QcVerdict {
            decision: "\"DROP\"".into(),
            valid: None,
            text: String::new(),
            reason: String::new(),
        };
        assert_eq!(v.normalized_decision(), "drop");
        let v = QcVerdict {
            decision: String::new(),
            valid: Some(false),
            text: String::new(),
            reason: String::new(),
        };
        assert_eq!(v.normalized_decision(), "");
    }

    #[test]
    fn repetition_detection() {
        assert!(is_repetitive("ありがとうありがとうありがとうありがとう"));
        assert!(is_repetitive("字幕由某某字幕组提供 字幕由某某字幕组提供 字幕由某某字幕组提供"));
        // 真实语句：字符多样、2-gram 不集中
        assert!(!is_repetitive(
            "落ちてしまって、足を複雑骨折してしまって、歩けなくなってしまいました。"
        ));
        assert!(!is_repetitive("今日はとても良い天気なので公園まで散歩に行きました。"));
        assert!(!is_repetitive("短い")); // 太短不判定
    }

    #[test]
    fn long_structured_sentence_cannot_be_dropped() {
        let c = checker(0.35);
        let real = "落ちてしまって、足をこう複雑骨折してしまって、歩けなくなってしまいました。";
        assert!(c.validate_drop(real).is_err(), "长句不应允许丢弃");
        // 短句仍可丢弃
        assert!(c.validate_drop("今日。").is_ok());
        // 长但高度重复（幻觉）仍可丢弃
        assert!(c
            .validate_drop("ありがとうありがとうありがとうありがとうありがとう")
            .is_ok());
        // 关闭保护后一律允许
        let off = QualityChecker::new(3, 2, 0.35, 0, Sampling::qwen3(42));
        assert!(off.validate_drop(real).is_ok());
    }

    #[test]
    fn history_window_is_bounded() {
        let mut c = QualityChecker::new(2, 2, 0.35, 30, Sampling::qwen3(42));
        c.push_history("a");
        c.push_history("b");
        c.push_history("c");
        assert_eq!(c.history.len(), 2);
        assert!(c.history_text().contains("b"));
        assert!(!c.history_text().contains("a"));
    }
}
