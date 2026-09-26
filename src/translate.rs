//! 翻译编排：全局摘要/术语表提取（必要时分块 map-reduce）+ 滑动窗口分批翻译。
//!
//! 相比 v0.2 的加固（每一条都对应一个实测踩到的坑）：
//! 1. **摘要不再一次性吞全文**：转录 token 超预算时按字幕边界切块，逐块提取
//!    {摘要, 术语表} 后合并（map-reduce），避免长视频把 prompt 顶爆上下文；
//! 2. **摘要结果必须非空才算成功**：`GlobalContext` 字段全是 `#[serde(default)]`，
//!    一个 `{}` 也能"解析成功"——实测模型输出被截断时就是这样静默拿到空摘要的；
//! 3. **批次按字符预算切分**（`--batch-chars`）而不只按条数：20 条长句会让 1.7B
//!    模型输出退化，实测直接照抄原文；
//! 4. **照抄检测 + 对半拆分重试**：译文与原文相似度过高即判定为没翻译，
//!    把批次拆小重来（小批次成功率高得多）；
//! 5. **匹配不再静默回退**：索引对不上时按位置兜底，兜底不了才回退原文，
//!    每次都 warn 并计数，收尾打印统计（CI 可据此断言）。

use crate::config::Config;
use crate::llm::{parse_json, LlmSession, Sampling};
use crate::prompt::PromptStore;
use crate::qc::char_similarity;
use crate::srt::format_timestamp;
use crate::types::{GlobalContext, SubtitleSegment, TranslateItem, TranslateStats};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use std::collections::HashMap;

const SYSTEM_PROMPT: &str =
    "你是一个专业的字幕翻译与内容分析助手。请严格按照用户要求的 JSON 格式输出，不要输出任何多余的解释、思考过程、前缀或代码块标记。";

const REVIEW_SYSTEM: &str =
    "你是一个严格的字幕翻译质检员。请只输出需要修改的条目组成的 JSON 数组，全部达标时输出 []，不要输出任何解释、思考过程或代码块标记。";

/// 生成侧至少预留的 token 数。
const OUTPUT_RESERVE: usize = 1536;
/// prompt 模板与聊天包裹的固定开销余量。
const TEMPLATE_MARGIN: usize = 512;
/// 摘要输出的 token 上限（长转录 + 术语表需要比 v0.2 的 1024 更宽）。
const SUMMARY_MAX_NEW: u32 = 2048;
/// 译文与原文相似度高于此值即视为"照抄没翻译"。
const COPY_THRESHOLD: f32 = 0.90;
/// 短于此字符数的片段不做照抄判定（汉字词在中日间常常同形）。
const COPY_MIN_CHARS: usize = 8;

pub struct Translator<'a, 'b> {
    session: &'a mut LlmSession<'b>,
    prompts: &'a PromptStore,
    cfg: &'a Config,
    sampling: Sampling,
    stats: TranslateStats,
}

impl<'a, 'b> Translator<'a, 'b> {
    pub fn new(session: &'a mut LlmSession<'b>, prompts: &'a PromptStore, cfg: &'a Config) -> Self {
        Self {
            session,
            prompts,
            cfg,
            sampling: cfg.sampling(),
            stats: TranslateStats::default(),
        }
    }

    /// prompt 可用的 token 预算（受 n_ctx 限制；prefill 已分块，故与 n_batch 无关）。
    fn prompt_budget(&self) -> usize {
        let (n_ctx, _) = self.session.limits();
        n_ctx.saturating_sub(OUTPUT_RESERVE + TEMPLATE_MARGIN)
            .max(512)
    }

    pub fn translate(&mut self, segments: Vec<SubtitleSegment>) -> Result<Vec<SubtitleSegment>> {
        if segments.is_empty() {
            return Ok(segments);
        }
        self.stats.segments = segments.len();

        // ---------- 阶段 1：全局上下文（摘要 + 术语表） ----------
        info!("▶ 提取全局视频上下文（摘要与术语表）...");
        let gctx = self.extract_global_context(&segments);
        info!(
            "✔ 全局摘要 {} 字、术语 {} 条（分块 {}）。",
            gctx.summary.chars().count(),
            gctx.glossary.len(),
            self.stats.summary_chunks
        );
        if gctx.summary.trim().is_empty() && gctx.glossary.is_empty() {
            warn!(
                "⚠ 未能取得任何全局上下文（摘要与术语表均为空），翻译将在无背景信息下进行，\
                 专有名词一致性可能下降。可调大 --ctx-size 或调小 --summary-chunk-tokens。"
            );
        } else {
            if !gctx.summary.is_empty() {
                info!("📝 全局摘要:\n{}", gctx.summary);
            }
            if !gctx.glossary.is_empty() {
                info!("📚 核心术语表:\n{}", gctx.glossary_as_text());
            }
        }

        // ---------- 阶段 2：滑动窗口分批翻译 ----------
        let mut translated: Vec<SubtitleSegment> = Vec::with_capacity(segments.len());
        let batches = plan_batches(&segments, self.cfg.batch_size, self.cfg.batch_chars);
        info!(
            "▶ 开始翻译：{} 条字幕分 {} 批（每批 ≤{} 条且 ≤{} 字）",
            segments.len(),
            batches.len(),
            self.cfg.batch_size,
            self.cfg.batch_chars
        );

        let pb = ProgressBar::new(segments.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} 条 ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for (bi, &(lo, hi)) in batches.iter().enumerate() {
            let batch = &segments[lo..hi];
            let ctx_start = translated.len().saturating_sub(self.cfg.context_size);
            let context = &translated[ctx_start..];

            self.stats.batches += 1;
            info!(
                "▶ 批次 {}/{}：第 {}-{} 条（{} 字）",
                bi + 1,
                batches.len(),
                lo + 1,
                hi,
                batch.iter().map(|s| s.text.chars().count()).sum::<usize>()
            );
            match self.translate_batch_adaptive(&gctx, context, batch, 0) {
                Ok(items) => {
                    let from = translated.len();
                    self.merge_batch(batch, &items, &mut translated);
                    if self.cfg.review_enabled {
                        self.review_and_polish(&gctx, batch, &mut translated[from..]);
                    }
                }
                Err(e) => {
                    self.stats.failed_batches += 1;
                    self.stats.untranslated += batch.len();
                    warn!(
                        "⚠ 批次 {}-{}（{} 条）翻译彻底失败，全部回退原文: {:#}",
                        lo + 1,
                        hi,
                        batch.len(),
                        e
                    );
                    for seg in batch {
                        info!(
                            "[翻译-回退] {} -> {}",
                            format_timestamp(seg.start_ms),
                            seg.text
                        );
                        translated.push(seg.clone());
                    }
                }
            }
            pb.inc(batch.len() as u64);
        }

        pb.finish_with_message("翻译完成");
        info!("✔ {}", self.stats.summary());
        if self.stats.untranslated > 0 || self.stats.copied > 0 {
            warn!(
                "⚠ 未翻译 {} 条、照抄原文 {} 条（共 {} 条），详见上方 warn 日志。",
                self.stats.untranslated, self.stats.copied, self.stats.segments
            );
        }
        Ok(translated)
    }

    /// 把模型返回的译文并回字幕；索引对不上时按位置兜底，仍失败才回退原文。
    fn merge_batch(
        &mut self,
        batch: &[SubtitleSegment],
        items: &[TranslateItem],
        out: &mut Vec<SubtitleSegment>,
    ) {
        let positional_ok = items.len() == batch.len();
        for (pos, seg) in batch.iter().enumerate() {
            let by_index = items.iter().find(|it| it.i == seg.index as i64);
            let item = match by_index {
                Some(it) => Some(it),
                None => {
                    if positional_ok {
                        self.stats.positional += 1;
                        warn!(
                            "第 {} 条（index={}）未按索引返回，改用位置兜底",
                            pos + 1,
                            seg.index
                        );
                        items.get(pos)
                    } else {
                        None
                    }
                }
            };

            let text = item
                .map(|it| it.t.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| {
                    self.stats.untranslated += 1;
                    warn!(
                        "第 {} 条（{}）无译文，回退原文: {}",
                        seg.index,
                        format_timestamp(seg.start_ms),
                        seg.text
                    );
                    seg.text.clone()
                });

            if looks_copied(&seg.text, &text) {
                self.stats.copied += 1;
            }
            let translated_seg = SubtitleSegment {
                text,
                ..seg.clone()
            };
            info!(
                "[翻译] {} -> {}",
                format_timestamp(translated_seg.start_ms),
                translated_seg.text.replace('\n', " / ")
            );
            out.push(translated_seg);
        }
    }

    // ========================================================================
    // 全局摘要 / 术语表
    // ========================================================================

    /// 提取全局上下文。**永不向上冒泡错误**：摘要失败只意味着少了上下文加持，
    /// 不应该让整片字幕白做（v0.2 这里一个 `?` 就能让长视频整片失败）。
    fn extract_global_context(&mut self, segments: &[SubtitleSegment]) -> GlobalContext {
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let total_tokens = self.session.count_tokens(&transcript);
        let budget = self
            .prompt_budget()
            .min(self.cfg.summary_chunk_tokens.max(256));
        info!(
            "转录全文 {} 字 / ~{} tokens，单块预算 {} tokens",
            transcript.chars().count(),
            total_tokens,
            budget
        );

        // 切块：按字幕边界累加 token，直到接近预算（每条只分词一次，O(n)）
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let mut cur = 0usize;
        for (i, s) in segments.iter().enumerate() {
            let t = self.session.count_tokens(&s.text) + 1; // +1 近似换行
            if cur + t > budget && i > start {
                ranges.push((start, i));
                start = i;
                cur = 0;
            }
            cur += t;
        }
        ranges.push((start, segments.len()));
        let groups: Vec<&[SubtitleSegment]> = ranges
            .iter()
            .map(|&(a, b)| (a.min(segments.len()), b.min(segments.len())))
            .filter(|&(a, b)| b > a)
            .map(|(a, b)| &segments[a..b])
            .collect();
        self.stats.summary_chunks = groups.len();

        if groups.len() <= 1 {
            let once = self.summarize_once(&transcript);
            return match once {
                Ok(ctx) => ctx,
                Err(e) => self.summary_fallback(e),
            };
        }

        info!(
            "▶ 转录过长，分 {} 块提取摘要与术语表（map-reduce）...",
            groups.len()
        );
        let mut merged = GlobalContext::default();
        let mut partials: Vec<String> = Vec::new();
        for (i, g) in groups.iter().enumerate() {
            let text = g
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            match self.summarize_once(&text) {
                Ok(ctx) => {
                    info!(
                        "  块 {}/{}：摘要 {} 字、术语 {} 条",
                        i + 1,
                        groups.len(),
                        ctx.summary.chars().count(),
                        ctx.glossary.len()
                    );
                    if !ctx.summary.trim().is_empty() {
                        partials.push(ctx.summary.trim().to_string());
                    }
                    merged.merge(ctx);
                }
                Err(e) => warn!("  块 {}/{} 摘要失败，已跳过: {:#}", i + 1, groups.len(), e),
            }
        }

        // reduce：把各块摘要合并成一段总摘要（术语表已按 source 去重合并）
        if partials.len() > 1 {
            let joined = partials.join("\n");
            if self.session.count_tokens(&joined) <= budget {
                match self.summarize_once(&joined) {
                    Ok(mut final_ctx) => {
                        for g in std::mem::take(&mut merged.glossary) {
                            if !final_ctx
                                .glossary
                                .iter()
                                .any(|x| x.source.trim() == g.source.trim())
                            {
                                final_ctx.glossary.push(g);
                            }
                        }
                        return final_ctx;
                    }
                    Err(e) => warn!("合并摘要失败，改用分块摘要拼接: {:#}", e),
                }
            }
            merged.summary = partials.join(" ");
        } else if let Some(only) = partials.first() {
            merged.summary = only.clone();
        }
        merged
    }

    fn summarize_once(&mut self, transcript: &str) -> Result<GlobalContext> {
        let mut vars = HashMap::new();
        vars.insert("transcript".to_string(), transcript.to_string());
        let prompt = self.prompts.render("extract_context.txt", &vars)?;

        // 输出上限随上下文余量伸缩：长转录需要更长的摘要+术语表，
        // v0.2 固定 1024 会被截断（实测截断后只剩一个空 JSON 对象被"成功"解析）。
        let (n_ctx, _) = self.session.limits();
        let prompt_tokens = self.session.count_tokens(&prompt);
        let room = n_ctx.saturating_sub(prompt_tokens + 64);
        let max_new = (SUMMARY_MAX_NEW as usize).min(room).max(256) as u32;

        let attempts = self.cfg.max_retries.max(1);
        let mut last_err = anyhow::anyhow!("未执行");
        let mut attempt_prompt = prompt.clone();
        for attempt in 1..=attempts {
            match self.session.chat(
                SYSTEM_PROMPT,
                &attempt_prompt,
                self.sampling.resample(attempt),
                max_new,
            ) {
                Ok(raw) => match parse_json::<GlobalContext>(&raw, false) {
                    // 关键校验：字段全是 serde(default)，一个 {} 也能"解析成功"。
                    // 摘要与术语表都空 == 没拿到东西，必须当失败重试。
                    Ok(ctx) if !ctx.summary.trim().is_empty() || !ctx.glossary.is_empty() => {
                        return Ok(ctx)
                    }
                    Ok(_) => {
                        self.stats.retries += 1;
                        last_err = anyhow::anyhow!(
                            "第 {} 次返回空摘要（解析出的 JSON 里没有 summary/glossary 内容）",
                            attempt
                        );
                        warn!("[摘要] {}", last_err);
                        attempt_prompt = format!("{}{}", prompt, HINT_OBJECT);
                    }
                    Err(e) => {
                        self.stats.retries += 1;
                        last_err = anyhow::anyhow!("第 {} 次解析失败: {}", attempt, e);
                        warn!("[摘要] {}", last_err);
                        attempt_prompt = format!("{}{}", prompt, HINT_OBJECT);
                    }
                },
                Err(e) => {
                    self.stats.retries += 1;
                    last_err = e.context(format!("第 {} 次请求失败", attempt));
                    warn!("[摘要] {:#}", last_err);
                }
            }
        }
        Err(last_err)
    }

    fn summary_fallback(&mut self, e: anyhow::Error) -> GlobalContext {
        warn!(
            "⚠ 全局摘要提取失败，将以空上下文继续翻译（不影响字幕产出）: {:#}",
            e
        );
        GlobalContext::default()
    }

    // ========================================================================
    // 分批翻译
    // ========================================================================

    /// 批次翻译。只在 **prompt 超出 token 预算** 时对半拆分；
    /// 译文质量不达标（照抄原文）走"换种子重新抽取"，不再靠拆批解决——
    /// 实测拆批会级联到单条仍判不合格，最后整批回退原文，比部分翻译更差。
    fn translate_batch_adaptive(
        &mut self,
        gctx: &GlobalContext,
        context: &[SubtitleSegment],
        batch: &[SubtitleSegment],
        depth: usize,
    ) -> Result<Vec<TranslateItem>> {
        let prompt = self.build_translate_prompt(gctx, context, batch)?;
        let tokens = self.session.count_tokens(&prompt);
        let budget = self.prompt_budget();
        if tokens > budget && batch.len() > 1 {
            return self.split_and_retry(
                gctx,
                context,
                batch,
                depth,
                format!("prompt {} tokens 超预算 {}", tokens, budget),
            );
        }
        if tokens > budget {
            warn!(
                "单条字幕的翻译 prompt 仍达 {} tokens（预算 {}），尝试直接发送",
                tokens, budget
            );
        }
        self.translate_once(&prompt, batch)
    }

    /// 拆半重试；深度受限时不再拆，直接返回当前结果或错误。
    fn split_and_retry(
        &mut self,
        gctx: &GlobalContext,
        context: &[SubtitleSegment],
        batch: &[SubtitleSegment],
        depth: usize,
        reason: String,
    ) -> Result<Vec<TranslateItem>> {
        if depth >= 4 || batch.len() <= 1 {
            anyhow::bail!("批次已拆到 {} 条仍失败（{}）", batch.len(), reason);
        }
        let mid = batch.len() / 2;
        self.stats.splits += 1;
        warn!(
            "批次 {} 条{}，拆成 {} + {} 条重试",
            batch.len(),
            reason,
            mid,
            batch.len() - mid
        );
        let mut out = self.translate_batch_adaptive(gctx, context, &batch[..mid], depth + 1)?;
        out.extend(self.translate_batch_adaptive(gctx, context, &batch[mid..], depth + 1)?);
        Ok(out)
    }

    /// 自检 + 定向二审。
    ///
    /// 实测有两种不同的失败模式：
    ///   ① 整条照抄原文（批内翻译时模型偷懒）——由照抄检测 + 换种子重抽处理；
    ///   ② **零散残留**：整句翻对了，但夹着没译的假名/日文词（おせんべい、キャラ 之类）。
    ///      这种在"整条未翻译"的统计里完全看不出来，只能逐字符查源文字系统。
    /// 第②种需要一轮**只针对残留条目**的定向审校，比整批重审便宜得多。
    fn review_and_polish(
        &mut self,
        gctx: &GlobalContext,
        batch: &[SubtitleSegment],
        out: &mut [SubtitleSegment],
    ) {
        self.review_batch(gctx, batch, out, false);

        let script = self.cfg.residual_script.as_str();
        if script == "none" {
            return;
        }
        let idx: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, s)| has_residual_script(&s.text, script))
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return;
        }
        warn!(
            "[审校] {} 条译文仍残留源语言文字（{}），做第二轮定向审校",
            idx.len(),
            script
        );
        let srcs: Vec<SubtitleSegment> = idx.iter().map(|&i| batch[i].clone()).collect();
        let mut sub: Vec<SubtitleSegment> = idx.iter().map(|&i| out[i].clone()).collect();
        self.review_batch(gctx, &srcs, &mut sub, true);
        for (k, &i) in idx.iter().enumerate() {
            out[i].text = sub[k].text.clone();
        }
        let still = sub
            .iter()
            .filter(|s| has_residual_script(&s.text, script))
            .count();
        self.stats.residual += still;
        if still > 0 {
            warn!("[审校] 二审后仍有 {} 条残留源语言文字", still);
        } else {
            info!("[审校] 二审后残留源语言文字已清零（本轮修正 {} 条）", idx.len());
        }
    }

    /// 译文自检：把「原文 + 当前译文」交回 LLM，让它以质检员身份挑出不达标的条目。
    ///
    /// 动机（实测）：日翻中时模型会残留假名、或直接把日文词抄下来（例如 おせんべい
    /// 没译成"仙贝"）。这类问题在批内翻译时很难自查，但换一个"质检员"身份再审一遍
    /// 就能挑出来。自检失败不影响主流程：任何异常都只 warn 并保留原译文。
    fn review_batch(
        &mut self,
        gctx: &GlobalContext,
        sources: &[SubtitleSegment],
        out: &mut [SubtitleSegment],
        strict: bool,
    ) {
        if out.is_empty() {
            return;
        }
        let pairs: Vec<String> = sources
            .iter()
            .zip(out.iter())
            .map(|(s, t)| {
                format!(
                    "{{\"i\":{},\"s\":{},\"t\":{}}}",
                    s.index,
                    serde_json::to_string(&s.text).unwrap_or_else(|_| "\"\"".into()),
                    serde_json::to_string(&t.text).unwrap_or_else(|_| "\"\"".into())
                )
            })
            .collect();

        let mut vars = HashMap::new();
        vars.insert("target_lang".to_string(), self.cfg.target_lang.clone());
        vars.insert("glossary".to_string(), gctx.glossary_as_text());
        vars.insert("pairs".to_string(), format!("[{}]", pairs.join(",")));
        let prompt = match self.prompts.render("review_batch.txt", &vars) {
            Ok(mut p) => {
                if strict {
                    p.push_str(STRICT_HINT);
                }
                p
            }
            Err(e) => {
                warn!("[审校] 提示词渲染失败，跳过自检: {:#}", e);
                return;
            }
        };

        let max_new =
            (self.cfg.translate_tokens_per_item * out.len() + 256).clamp(512, 6144) as u32;
        let attempts = 2usize;
        for attempt in 1..=attempts {
            // 种子偏移 100：让审校与翻译走不同的采样轨迹
            let raw = match self.session.chat(
                REVIEW_SYSTEM,
                &prompt,
                self.sampling.resample(attempt + 100),
                max_new,
            ) {
                Ok(r) => r,
                Err(e) => {
                    warn!("[审校] 第 {} 次请求失败: {:#}", attempt, e);
                    continue;
                }
            };
            let fixes = match parse_json::<Vec<TranslateItem>>(&raw, true) {
                Ok(v) => v,
                Err(e) => {
                    warn!("[审校] 第 {} 次输出不可解析: {}", attempt, e);
                    continue;
                }
            };
            self.stats.reviewed += out.len();
            if fixes.is_empty() {
                info!("[审校] {} 条全部达标，无需修改", out.len());
                return;
            }
            let mut applied = 0usize;
            for f in &fixes {
                let new_text = f.t.trim();
                if new_text.is_empty() {
                    continue;
                }
                let Some(pos) = out.iter().position(|s| s.index as i64 == f.i) else {
                    warn!("[审校] 返回了未知 index {}，忽略", f.i);
                    continue;
                };
                if out[pos].text.trim() == new_text {
                    continue;
                }
                // 审校结果本身也可能照抄原文，那就没有意义
                if let Some(src) = sources.get(pos) {
                    if looks_copied(&src.text, new_text) {
                        warn!("[审校] 第 {} 条修正后仍与原文雷同，忽略", f.i);
                        continue;
                    }
                }
                info!(
                    "[审校✎] {} => {}",
                    out[pos].text.replace('\n', " "),
                    new_text.replace('\n', " ")
                );
                out[pos].text = new_text.to_string();
                applied += 1;
            }
            self.stats.review_fixed += applied;
            info!("[审校] 检查 {} 条，修正 {} 条", out.len(), applied);
            return;
        }
        warn!("[审校] {} 次尝试均失败，保留原译文", attempts);
    }

    fn build_translate_prompt(
        &self,
        gctx: &GlobalContext,
        context: &[SubtitleSegment],
        batch: &[SubtitleSegment],
    ) -> Result<String> {
        let mut vars = HashMap::new();
        vars.insert("source_lang".to_string(), self.cfg.source_lang.clone());
        vars.insert("target_lang".to_string(), self.cfg.target_lang.clone());
        vars.insert(
            "summary".to_string(),
            {
                let s = gctx.summary.trim();
                if s.is_empty() {
                    "（无）".to_string()
                } else {
                    s.to_string()
                }
            },
        );
        vars.insert("glossary".to_string(), gctx.glossary_as_text());
        // 上下文/待翻译都只带 i + t：时间轴由 Rust 侧保留，不让模型回显
        let ctx_compact: Vec<SubtitleSegment> = context
            .iter()
            .map(|s| SubtitleSegment {
                index: s.index,
                start_ms: 0,
                end_ms: 0,
                text: s.text.clone(),
            })
            .collect();
        vars.insert(
            "context".to_string(),
            if ctx_compact.is_empty() {
                "（无）".to_string()
            } else {
                SubtitleSegment::compact_json_list(&ctx_compact)
            },
        );
        vars.insert(
            "batch".to_string(),
            SubtitleSegment::compact_json_list(batch),
        );
        self.prompts.render("translate_batch.txt", &vars)
    }

    /// 一次批次翻译：失败或质量不达标就**换种子重新抽取**（模型卡明确不建议贪心）。
    ///
    /// 与旧策略的关键差别：重抽用尽后**接受其中最好的一次结果**，而不是整批回退原文。
    /// 回退原文意味着这一段字幕完全没翻译，一定比"部分照抄"更差。
    fn translate_once(&mut self, prompt: &str, batch: &[SubtitleSegment]) -> Result<Vec<TranslateItem>> {
        let attempts = self.cfg.max_retries.max(1);
        let batch_len = batch.len();
        // 生成上限随条数伸缩：宁可给足，也不要因为截断而整批重来
        let max_new =
            (self.cfg.translate_tokens_per_item * batch_len + 256).clamp(512, 6144) as u32;
        let hint_copy = format!(
            "\n\n【重要】你上一次把原文照抄回来了，没有翻译。请重新输出 JSON 数组，\
             形如 [{{\"i\":1,\"t\":\"译文\"}}]，每条 t 都必须是{}，条数与输入一致，\
             不要输出思考过程、解释或代码块标记。\n/no_think",
            self.cfg.target_lang
        );

        let mut last_err = anyhow::anyhow!("未执行");
        let mut attempt_prompt = prompt.to_string();
        let mut best: Option<(Vec<TranslateItem>, usize)> = None;

        for attempt in 1..=attempts {
            let raw = match self.session.chat(
                SYSTEM_PROMPT,
                &attempt_prompt,
                self.sampling.resample(attempt),
                max_new,
            ) {
                Ok(r) => r,
                Err(e) => {
                    self.stats.retries += 1;
                    last_err = e.context(format!("第 {} 次请求失败", attempt));
                    warn!("[翻译] {:#}", last_err);
                    continue;
                }
            };

            let items = match parse_json::<Vec<TranslateItem>>(&raw, true) {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    self.stats.retries += 1;
                    last_err = anyhow::anyhow!("第 {} 次返回空数组", attempt);
                    warn!("[翻译] {}", last_err);
                    attempt_prompt = format!("{}{}", prompt, HINT_ARRAY);
                    continue;
                }
                Err(e) => {
                    self.stats.retries += 1;
                    last_err = anyhow::anyhow!("第 {} 次解析失败: {}", attempt, e);
                    warn!("[翻译] {}", last_err);
                    attempt_prompt = format!("{}{}", prompt, HINT_ARRAY);
                    continue;
                }
            };

            if items.len() != batch_len {
                warn!(
                    "[翻译] 返回 {} 条，期望 {} 条（按索引匹配，缺失条目回退原文）",
                    items.len(),
                    batch_len
                );
            }

            let copies = count_copies(batch, &items);
            if copies == 0 {
                return Ok(items);
            }
            warn!(
                "[翻译] 第 {}/{} 次抽取有 {} / {} 条疑似照抄原文，换种子重新抽取…",
                attempt,
                attempts,
                copies,
                batch_len
            );
            self.stats.retries += 1;
            attempt_prompt = format!("{}{}", prompt, hint_copy);
            if best.as_ref().map(|(_, c)| copies < *c).unwrap_or(true) {
                best = Some((items, copies));
            }
        }

        if let Some((items, copies)) = best {
            warn!(
                "[翻译] {} 次抽取后仍有 {} 条疑似照抄，采用其中最好的一次（不回退整批原文）",
                attempts, copies
            );
            return Ok(items);
        }
        Err(last_err)
    }
}

/// 统计一批里"疑似照抄原文"的条数。
fn count_copies(batch: &[SubtitleSegment], items: &[TranslateItem]) -> usize {
    batch
        .iter()
        .filter(|seg| {
            items
                .iter()
                .find(|it| it.i == seg.index as i64)
                .map(|it| looks_copied(&seg.text, &it.t))
                .unwrap_or(false)
        })
        .count()
}


/// 按「条数上限 + 字符预算」双约束切批。
///
/// 只按条数切批是 v0.2 的做法：20 条短字幕很轻松，20 条 140 字的长句就会让
/// 1.7B 模型输出退化（实测直接照抄原文）。字符预算让批次规模与内容量挂钩。
pub fn plan_batches(segments: &[SubtitleSegment], batch_size: usize, batch_chars: usize) -> Vec<(usize, usize)> {
    let max_n = batch_size.max(1);
    let max_chars = batch_chars.max(64);
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut chars = 0usize;
    for (i, s) in segments.iter().enumerate() {
        let n = s.text.chars().count().max(1);
        if i > start && (i - start >= max_n || chars + n > max_chars) {
            out.push((start, i));
            start = i;
            chars = 0;
        }
        chars += n;
    }
    if start < segments.len() {
        out.push((start, segments.len()));
    }
    out
}

/// 译文里是否仍残留源语言文字。
///
/// 目前支持日语假名（本项目的默认场景是日翻中）。中文译文里出现假名，
/// 几乎只可能是没翻译干净的残留——中文本身不使用假名。
pub fn has_residual_script(text: &str, script: &str) -> bool {
    match script {
        "kana" => text
            .chars()
            .any(|c| ('\u{3040}'..='\u{30ff}').contains(&c)),
        "hangul" => text.chars().any(|c| ('\u{ac00}'..='\u{d7af}').contains(&c)),
        _ => false,
    }
}

/// 译文是否只是把原文抄了回来。
///
/// 同语言翻译（如繁转简）时这个判定会偏保守，但本项目默认是跨语言字幕翻译，
/// 相似度 0.9 以上基本不可能是正常译文。
fn looks_copied(source: &str, translated: &str) -> bool {
    let s = source.trim();
    let t = translated.trim();
    if t.is_empty() || s.is_empty() {
        return false;
    }
    // 短片段不做判定：日语里的汉字词（今日/戦争/村人/卒業生）本身往往就是合法中文，
    // 逐字相同并不代表模型没翻译。实测正是这类短片段造成了 100% 误判。
    if s.chars().count() < COPY_MIN_CHARS {
        return false;
    }
    if s == t {
        return true;
    }
    // 长度差太多说明是真翻译（或真出错），只有"长度接近且内容接近"才算照抄
    let ls = s.chars().count() as f32;
    let lt = t.chars().count() as f32;
    if lt / ls < 0.6 || lt / ls > 1.6 {
        return false;
    }
    char_similarity(s, t) >= COPY_THRESHOLD
}

/// 第二轮定向审校追加的提示（针对"整句翻对了但夹着没译的源语言文字"）。
const STRICT_HINT: &str = "\n\n【第二轮·重点】下面这些条目的译文里**仍然残留日文假名或日文词**。\
请把它们重写为纯中文：专有名词用中文通用译名或音译（おせんべい->仙贝、キャラ->角色、\
ほうとう->宝刀面、スカイツリー->天空树），实在没有通用译名时用中文描述性说法，\
**译文中不允许出现任何假名**。只输出需要修改的条目。\n/no_think";

/// 解析失败后追加的格式提醒（重试时生效——贪心解码下原样重发只会得到同样的坏输出）。
const HINT_ARRAY: &str = "\n\n【重要】你上一次的输出不是合法 JSON 数组。现在请只输出 JSON 数组，形如 [{\"i\":1,\"t\":\"译文\"}]，条数与输入完全一致，t 必须是目标语言的译文（严禁照抄原文），不要输出思考过程、解释或代码块标记。\n/no_think";
const HINT_OBJECT: &str = "\n\n【重要】你上一次的输出不是合法 JSON 对象，或者内容为空。现在请只输出 JSON 对象，形如 {\"summary\":\"...\",\"glossary\":[{\"source\":\"...\",\"target\":\"...\"}]}，summary 与 glossary 都必须有实际内容，不要输出思考过程、解释或代码块标记。\n/no_think";

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(i: usize, t: &str) -> SubtitleSegment {
        SubtitleSegment {
            index: i,
            start_ms: (i as u64) * 1000,
            end_ms: (i as u64) * 1000 + 900,
            text: t.to_string(),
        }
    }

    #[test]
    fn copied_detection() {
        assert!(looks_copied("今日は良い天気です", "今日は良い天気です"));
        assert!(looks_copied("今日は良い天気です", "今日は良い天気です。"));
        assert!(!looks_copied("今日は良い天気です", "今天天气很好"));
        assert!(!looks_copied("今日は良い天気です", ""));
        // 长度差很大时不判为照抄（交给别的校验）
        assert!(!looks_copied("あいうえおかきくけこさしすせそ", "あい"));
        // 短片段豁免：汉字词在中日间常常同形，逐字相同不代表没翻译
        // （实测正是这类短片段造成了整批 100% 误判 -> 级联拆批 -> 整批回退原文）
        assert!(!looks_copied("今日。", "今日。"));
        assert!(!looks_copied("戦争", "戦争"));
        assert!(!looks_copied("皆さんに。", "各位。"));
    }

    #[test]
    fn residual_script_detection() {
        // 中文译文里出现假名 == 没翻干净
        assert!(has_residual_script("而我的喜欢的キャラのおもちゃ却完全不被消费", "kana"));
        assert!(has_residual_script("どっしり構えようと。", "kana"));
        // 纯中文（含汉字与标点）不算残留
        assert!(!has_residual_script("老人唯一的财产是一匹马，还有三个儿子。", "kana"));
        assert!(!has_residual_script("iPS 细胞与 Wi-Fi、ACTA 等拉丁词不算假名残留", "kana"));
        // 关闭检测 / 其它文字系统
        assert!(!has_residual_script("キャラ", "none"));
        assert!(has_residual_script("한국어 테스트", "hangul"));
        assert!(!has_residual_script("キャラ", "hangul"));
    }

    #[test]
    fn batch_planning_respects_char_budget() {
        let segs: Vec<SubtitleSegment> = (1..=10).map(|i| seg(i, &"あ".repeat(30))).collect();
        // 每条 30 字、预算 100 字 -> 每批 3 条
        assert_eq!(
            plan_batches(&segs, 20, 100),
            vec![(0, 3), (3, 6), (6, 9), (9, 10)]
        );
    }

    #[test]
    fn batch_planning_respects_count_limit() {
        let segs: Vec<SubtitleSegment> = (1..=7).map(|i| seg(i, "短")).collect();
        assert_eq!(plan_batches(&segs, 3, 1000), vec![(0, 3), (3, 6), (6, 7)]);
    }

    #[test]
    fn batch_planning_single_oversized_item_still_forms_a_batch() {
        let segs = vec![seg(1, &"あ".repeat(500)), seg(2, "短")];
        let plan = plan_batches(&segs, 20, 100);
        assert_eq!(plan, vec![(0, 1), (1, 2)], "{plan:?}");
    }

    #[test]
    fn batch_planning_empty_input() {
        assert!(plan_batches(&[], 20, 100).is_empty());
    }
}
