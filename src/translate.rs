//! 翻译编排：全局摘要/术语表提取（必要时分块 map-reduce）+ 滑动窗口分批翻译。
//!
//! 相比旧版的三处关键加固：
//! 1. **摘要不再一次性吞全文**：转录 token 数超过预算时按字幕边界切块，
//!    逐块提取 {摘要, 术语表} 后合并，避免长视频把 prompt 顶爆上下文；
//! 2. **批次自适应**：单批 prompt 超预算时对半拆分重试（最小到 1 条），
//!    不再因为 `--batch-size` 配大就直接失败；
//! 3. **匹配不再静默回退**：模型漏条/重编号时按位置兜底，兜底不了才回退原文，
//!    并且每一次回退都计数 + warn，收尾打印统计（CI 可据此断言）。

use crate::config::Config;
use crate::llm::{parse_json, LlmSession};
use crate::prompt::PromptStore;
use crate::srt::format_timestamp;
use crate::types::{GlobalContext, SubtitleSegment, TranslateItem, TranslateStats};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use std::collections::HashMap;

const SYSTEM_PROMPT: &str =
    "你是一个专业的字幕翻译与内容分析助手。请严格按照用户要求的 JSON 格式输出，不要输出任何多余的解释、思考过程、前缀或代码块标记。";

/// 解析失败后追加的格式提醒（重试时生效——贪心解码下原样重发只会得到同样的坏输出）。
const HINT_ARRAY: &str = "\n\n【重要】你上一次的输出不是合法 JSON 数组。现在请只输出 JSON 数组，形如 [{\"i\":1,\"t\":\"译文\"}]，条数与输入完全一致，不要输出思考过程、解释或代码块标记。\n/no_think";
const HINT_OBJECT: &str = "\n\n【重要】你上一次的输出不是合法 JSON 对象。现在请只输出 JSON 对象，形如 {\"summary\":\"...\",\"glossary\":[{\"source\":\"...\",\"target\":\"...\"}]}，不要输出思考过程、解释或代码块标记。\n/no_think";

/// 生成侧至少预留的 token 数（摘要/翻译输出上限的量级）。
const OUTPUT_RESERVE: usize = 1536;
/// prompt 模板与聊天包裹的固定开销余量。
const TEMPLATE_MARGIN: usize = 512;

pub struct Translator<'a, 'b> {
    session: &'a mut LlmSession<'b>,
    prompts: &'a PromptStore,
    cfg: &'a Config,
    stats: TranslateStats,
}

impl<'a, 'b> Translator<'a, 'b> {
    pub fn new(session: &'a mut LlmSession<'b>, prompts: &'a PromptStore, cfg: &'a Config) -> Self {
        Self {
            session,
            prompts,
            cfg,
            stats: TranslateStats::default(),
        }
    }

    /// prompt 可用的 token 预算（受 n_ctx 限制；prefill 已分块，故与 n_batch 无关）。
    fn prompt_budget(&self) -> usize {
        let (n_ctx, _) = self.session.limits();
        n_ctx.saturating_sub(OUTPUT_RESERVE + TEMPLATE_MARGIN).max(512)
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
        if !gctx.summary.is_empty() {
            info!("📝 全局摘要:\n{}", gctx.summary);
        }
        if !gctx.glossary.is_empty() {
            info!("📚 核心术语表:\n{}", gctx.glossary_as_text());
        }

        // ---------- 阶段 2：滑动窗口分批翻译 ----------
        let mut translated: Vec<SubtitleSegment> = Vec::with_capacity(segments.len());
        let bs = self.cfg.batch_size.max(1);

        let pb = ProgressBar::new(segments.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} 条 ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for chunk_start in (0..segments.len()).step_by(bs) {
            let chunk_end = (chunk_start + bs).min(segments.len());
            let batch = &segments[chunk_start..chunk_end];

            let ctx_start = translated.len().saturating_sub(self.cfg.context_size);
            let context = &translated[ctx_start..];

            self.stats.batches += 1;
            match self.translate_batch_adaptive(&gctx, context, batch) {
                Ok(items) => self.merge_batch(batch, &items, &mut translated),
                Err(e) => {
                    // 重试全部失败后回退原文，保证流程不中断（但要大声说出来）
                    self.stats.failed_batches += 1;
                    self.stats.untranslated += batch.len();
                    warn!(
                        "⚠ 批次 {}-{}（{} 条）翻译彻底失败，全部回退原文: {:#}",
                        chunk_start + 1,
                        chunk_end,
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
        if self.stats.untranslated > 0 {
            warn!(
                "⚠ 有 {} / {} 条字幕未能翻译（保留原文），详见上方 warn 日志。",
                self.stats.untranslated, self.stats.segments
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
    /// 不应该让整片字幕白做（旧版这里一个 `?` 就能让长视频整片失败）。
    fn extract_global_context(&mut self, segments: &[SubtitleSegment]) -> GlobalContext {
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let total_tokens = self.session.count_tokens(&transcript);
        let budget = self.prompt_budget().min(self.cfg.summary_chunk_tokens.max(256));
        info!(
            "转录全文 {} 字 / ~{} tokens，单块预算 {} tokens",
            transcript.chars().count(),
            total_tokens,
            budget
        );

        // 切块：按字幕边界累加 token，直到接近预算（O(n)，每条只分词一次）
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
            let joined_tokens = self.session.count_tokens(&joined);
            if joined_tokens <= budget {
                match self.summarize_once(&joined) {
                    Ok(final_ctx) => {
                        let mut out = final_ctx;
                        // 合并块级术语（final 调用可能只看到摘要，术语覆盖不全）
                        for g in std::mem::take(&mut merged.glossary) {
                            if !out
                                .glossary
                                .iter()
                                .any(|x| x.source.trim() == g.source.trim())
                            {
                                out.glossary.push(g);
                            }
                        }
                        return out;
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

        let attempts = self.cfg.max_retries.max(1);
        let mut last_err = anyhow::anyhow!("未执行");
        let mut attempt_prompt = prompt.clone();
        for attempt in 1..=attempts {
            // JSON 任务：首轮用低温采样，重试转贪心（确定性更高）
            let temp = if attempt == 1 {
                self.cfg.temperature.min(0.3)
            } else {
                0.0
            };
            match self.session.chat(SYSTEM_PROMPT, &attempt_prompt, temp, 1024) {
                Ok(raw) => match parse_json::<GlobalContext>(&raw, false) {
                    Ok(ctx) => return Ok(ctx),
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

    /// 带自适应拆分的批次翻译：prompt 超预算时对半拆，直到能塞下为止。
    fn translate_batch_adaptive(
        &mut self,
        gctx: &GlobalContext,
        context: &[SubtitleSegment],
        batch: &[SubtitleSegment],
    ) -> Result<Vec<TranslateItem>> {
        let prompt = self.build_translate_prompt(gctx, context, batch)?;
        let tokens = self.session.count_tokens(&prompt);
        let budget = self.prompt_budget();
        if tokens <= budget || batch.len() <= 1 {
            if tokens > budget {
                warn!(
                    "单条字幕的翻译 prompt 仍达 {} tokens（预算 {}），尝试直接发送",
                    tokens, budget
                );
            }
            return self.translate_once(&prompt, batch.len());
        }
        let mid = batch.len() / 2;
        warn!(
            "批次 {} 条的 prompt 达 {} tokens（预算 {}），拆成 {} + {} 条重试",
            batch.len(),
            tokens,
            budget,
            mid,
            batch.len() - mid
        );
        self.stats.splits += 1;
        let mut out = self.translate_batch_adaptive(gctx, context, &batch[..mid])?;
        out.extend(self.translate_batch_adaptive(gctx, context, &batch[mid..])?);
        Ok(out)
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
        vars.insert("summary".to_string(), {
            let s = gctx.summary.trim();
            if s.is_empty() {
                "（无）".to_string()
            } else {
                s.to_string()
            }
        });
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

    fn translate_once(&mut self, prompt: &str, batch_len: usize) -> Result<Vec<TranslateItem>> {
        let attempts = self.cfg.max_retries.max(1);
        // 生成上限随条数伸缩：宁可给足，也不要因为截断而整批重来
        let max_new = (self.cfg.translate_tokens_per_item * batch_len + 256)
            .clamp(512, OUTPUT_RESERVE * 3) as u32;

        let mut last_err = anyhow::anyhow!("未执行");
        let mut attempt_prompt = prompt.to_string();
        for attempt in 1..=attempts {
            let temp = if attempt == 1 {
                self.cfg.temperature
            } else {
                0.0 // 重试转贪心，排除采样抖动
            };
            match self.session.chat(SYSTEM_PROMPT, &attempt_prompt, temp, max_new) {
                Ok(raw) => match parse_json::<Vec<TranslateItem>>(&raw, true) {
                    Ok(items) => {
                        if items.is_empty() {
                            self.stats.retries += 1;
                            last_err = anyhow::anyhow!("第 {} 次返回空数组", attempt);
                            warn!("[翻译] {}", last_err);
                            continue;
                        }
                        if items.len() != batch_len {
                            warn!(
                                "[翻译] 返回 {} 条，期望 {} 条（将按索引匹配，缺失条目回退原文）",
                                items.len(),
                                batch_len
                            );
                        }
                        return Ok(items);
                    }
                    Err(e) => {
                        self.stats.retries += 1;
                        last_err = anyhow::anyhow!("第 {} 次解析失败: {}", attempt, e);
                        warn!("[翻译] {}", last_err);
                        attempt_prompt = format!("{}{}", prompt, HINT_ARRAY);
                    }
                },
                Err(e) => {
                    self.stats.retries += 1;
                    last_err = e.context(format!("第 {} 次请求失败", attempt));
                    warn!("[翻译] {:#}", last_err);
                }
            }
        }
        Err(last_err)
    }
}
