use crate::llm::{extract_json, LlmClient};
use crate::prompt::PromptStore;
use crate::types::{GlobalContext, SubtitleSegment};
use crate::config::Config;
use crate::srt::format_timestamp;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use std::collections::HashMap;

pub struct Translator<'a> {
    llm: &'a LlmClient,
    prompts: &'a PromptStore,
    cfg: &'a Config,
}

impl<'a> Translator<'a> {
    pub fn new(llm: &'a LlmClient, prompts: &'a PromptStore, cfg: &'a Config) -> Self {
        Self { llm, prompts, cfg }
    }

    pub fn translate(&self, segments: Vec<SubtitleSegment>) -> Result<Vec<SubtitleSegment>> {
        if segments.is_empty() {
            return Ok(segments);
        }

        // 阶段 1：全局上下文提取
        info!("▶ 提取全局视频上下文（摘要与术语表）...");
        let ctx = self.extract_global_context(&segments)?;
        info!("✔ 已提取全局摘要（{} 字）与 {} 条术语。", ctx.summary.chars().count(), ctx.glossary.len());
        
        info!("📝 全局摘要内容:\n{}", ctx.summary);
        if !ctx.glossary.is_empty() {
            info!("📚 核心术语表:");
            for g in &ctx.glossary {
                info!("   - {} -> {}", g.source, g.target);
            }
        }

        // 阶段 2：滑动窗口批量翻译
        let mut translated: Vec<SubtitleSegment> = Vec::with_capacity(segments.len());
        let bs = self.cfg.batch_size;

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

            match self.translate_batch(&ctx, context, batch) {
                Ok(items) => {
                    for seg in batch {
                        let found = items.iter().find(|t| t.index == seg.index);
                        let text = found
                            .map(|t| t.text.clone())
                            .filter(|s| !s.trim().is_empty())
                            .unwrap_or_else(|| seg.text.clone());
                            
                        let translated_seg = SubtitleSegment { text, ..seg.clone() };
                        info!("[LLM] {} -> {}", format_timestamp(translated_seg.start_ms), translated_seg.text);
                        translated.push(translated_seg);
                    }
                }
                Err(e) => {
                    // 5次重试全部失败后，才会走到这里回退原文
                    warn!("批次 {}-{} 翻译彻底失败，回退原文: {:#}", chunk_start, chunk_end, e);
                    for seg in batch {
                        info!("[LLM-Fallback] {} -> {}", format_timestamp(seg.start_ms), seg.text);
                        translated.push(seg.clone());
                    }
                }
            }
            pb.inc(batch.len() as u64);
        }

        pb.finish_with_message("翻译完成");
        Ok(translated)
    }

    fn extract_global_context(&self, segments: &[SubtitleSegment]) -> Result<GlobalContext> {
        let transcript: String = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = HashMap::new();
        vars.insert("transcript".to_string(), transcript);
        let prompt = self.prompts.render("extract_context.txt", &vars)?;

        let max_retries = 5;
        for attempt in 1..=max_retries {
            let raw = self.llm.chat(&prompt).context("摘要提取请求失败")?;
            
            if let Some(json_str) = extract_json(&raw, false) {
                match serde_json::from_str::<GlobalContext>(&json_str) {
                    Ok(ctx) => return Ok(ctx),
                    Err(_) => warn!("[摘要] 第 {} 次尝试：JSON 结构不符合预期，重试中...", attempt),
                }
            } else {
                warn!("[摘要] 第 {} 次尝试：输出无法解析为 JSON，重试中...", attempt);
            }
        }
        
        // 如果摘要提取 5 次都失败，返回一个空的上下文，让翻译流程继续（只是没有全局摘要加持）
        warn!("[摘要] {} 次尝试均失败，将使用空的上下文继续翻译。", max_retries);
        Ok(GlobalContext::default())
    }

    fn translate_batch(
        &self,
        ctx: &GlobalContext,
        context: &[SubtitleSegment],
        batch: &[SubtitleSegment],
    ) -> Result<Vec<SubtitleSegment>> {
        let mut vars = HashMap::new();
        vars.insert("target_lang".to_string(), self.cfg.target_lang.clone());
        vars.insert("summary".to_string(), ctx.summary.clone());
        vars.insert("glossary".to_string(), ctx.glossary_as_text());
        vars.insert(
            "context".to_string(),
            serde_json::to_string_pretty(context).unwrap_or_else(|_| "[]".into()),
        );
        vars.insert(
            "batch".to_string(),
            serde_json::to_string_pretty(batch).unwrap_or_else(|_| "[]".into()),
        );

        let prompt = self.prompts.render("translate_batch.txt", &vars)?;
        
        let max_retries = 5;
        for attempt in 1..=max_retries {
            let raw = self.llm.chat(&prompt).context("翻译请求失败")?; 
            
            if let Some(json_str) = extract_json(&raw, true) {
                match serde_json::from_str::<Vec<SubtitleSegment>>(&json_str) {
                    Ok(items) => return Ok(items),
                    Err(_) => warn!("[翻译] 批次第 {} 次尝试：JSON 数组结构不符合预期，重试中...", attempt),
                }
            } else {
                warn!("[翻译] 批次第 {} 次尝试：输出无法解析为 JSON 数组，重试中...", attempt);
            }
        }
        
        // 5次都失败，抛出错误，由外层捕获并回退原文
        anyhow::bail!("{} 次尝试均无法解析出有效的 JSON 数组", max_retries)
    }
}