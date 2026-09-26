//! llama.cpp（外置 llama.dll/ggml*.dll）绑定与自回归生成。
//!
//! 设计要点：
//! - `LlmClient::session` 创建**可复用的推理会话**：KV cache 上下文只建一次，
//!   每轮对话后 `clear_kv_cache()` 复用，避免逐句质检场景下每句重建上下文。
//! - **Prompt 分块 prefill**：`llama_decode` 单次最多接受 `n_batch` 个 token
//!   （超出会命中 llama.cpp 内部的 `GGML_ASSERT(n_tokens_all <= n_batch)` 直接 abort），
//!   因此长 prompt 必须按 `n_batch` 切块喂入，只有最后一个 token 置 `logits=true`。
//! - `n_ctx` / `n_batch` 都在创建上下文时显式设置，不再依赖 llama.cpp 的默认值
//!   （默认 n_ctx=512、n_batch=2048，与本项目 8K 上下文的预期不一致）。
//! - GPU offload 层数由运行时探测结果 / `--gpu-layers` 决定。
//! - 采样种子可指定（`--seed`），保证同一输入的结果可复现。
//!
//! 注意：llama-cpp-2 锁定 0.1.157（str_to_token/is_eog_token/token_to_piece 旧 API）；
//! 0.1.158+ 迁移到了 model.vocab()，升级依赖时需同步修改本文件。

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use log::{debug, info, warn};
use std::num::NonZeroU32;
use std::path::Path;

/// 采样参数。默认值取自 Qwen3 模型卡（huggingface.co/Qwen/Qwen3-1.7B）
/// 对**非思考模式**的建议：`Temperature=0.7, TopP=0.8, TopK=20, MinP=0`。
///
/// 模型卡同时明确写着 **"DO NOT use greedy decoding"**（贪心会导致质量退化与
/// 无限重复），所以本项目不再用"重试转贪心"的策略，而是**换种子重新抽取**。
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
}

impl Sampling {
    pub fn new(temperature: f32, top_p: f32, top_k: i32, seed: u32) -> Self {
        Self {
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            top_k,
            seed,
        }
    }

    /// Qwen3 模型卡推荐的非思考模式参数
    pub fn qwen3(seed: u32) -> Self {
        Self::new(0.7, 0.8, 20, seed)
    }

    /// 派生第 n 次抽取的种子（重抽时换一个采样轨迹，而不是退回贪心）
    pub fn resample(&self, attempt: usize) -> Self {
        Self {
            seed: self.seed.wrapping_add((attempt as u32).wrapping_mul(0x9E37_79B9)),
            ..*self
        }
    }

    fn build(&self) -> LlamaSampler {
        if self.temperature <= 0.0 {
            // 仅在用户显式要求时走贪心（模型卡不推荐）
            return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        }
        let mut chain: Vec<LlamaSampler> = vec![LlamaSampler::temp(self.temperature)];
        if self.top_k > 0 {
            chain.push(LlamaSampler::top_k(self.top_k));
        }
        if self.top_p > 0.0 && self.top_p < 1.0 {
            chain.push(LlamaSampler::top_p(self.top_p, 1));
        }
        chain.push(LlamaSampler::dist(self.seed));
        LlamaSampler::chain_simple(chain)
    }
}

pub struct LlmClient {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    n_batch: u32,
    seed: u32,
}

impl LlmClient {
    pub fn new(model_path: &Path, n_ctx: u32, n_batch: u32, gpu_layers: u32, seed: u32) -> Result<Self> {
        let mut backend = LlamaBackend::init().context("初始化 llama backend 失败")?;
        // 屏蔽 llama.cpp 底层 C++ 的原生日志（等价于旧版手工注入空回调）
        backend.void_logs();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("加载模型失败: {:?}", model_path))?;

        info!(
            "LLM 已加载: {:?}（n_gpu_layers={}, n_ctx={}, n_batch={}, seed={}）",
            model_path, gpu_layers, n_ctx, n_batch, seed
        );

        Ok(Self {
            backend,
            model,
            n_ctx,
            n_batch: n_batch.max(64),
            seed,
        })
    }

    /// 创建推理会话。会话内部持有 KV cache 上下文，可多次 chat() 复用。
    pub fn session(&self) -> Result<LlmSession<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_batch);
        let ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("创建 LLM 上下文失败")?;
        Ok(LlmSession { client: self, ctx })
    }
}

/// 可复用的 LLM 推理会话（持有 KV cache 上下文）。
pub struct LlmSession<'a> {
    client: &'a LlmClient,
    ctx: LlamaContext<'a>,
}

impl LlmSession<'_> {
    /// 上下文容量：`(n_ctx, n_batch)`。调用方据此规划单次 prompt 的 token 预算。
    pub fn limits(&self) -> (usize, usize) {
        (self.ctx.n_ctx() as usize, self.ctx.n_batch() as usize)
    }

    /// 估算文本的 token 数（用模型自己的分词器，精确值）。
    /// 分词失败时退化为字符数（对中日韩文本是偏保守的估计）。
    pub fn count_tokens(&self, text: &str) -> usize {
        match self.client.model.str_to_token(text, AddBos::Never) {
            Ok(v) => v.len(),
            Err(_) => text.chars().count(),
        }
    }

    /// 一轮 ChatML 对话。采样策略由 `Sampling` 决定（默认按 Qwen3 模型卡的非思考模式参数）。
    /// `max_new_tokens` 为生成上限（防止小模型跑飞吃满上下文）。
    pub fn chat(
        &mut self,
        system: &str,
        user: &str,
        sampling: Sampling,
        max_new_tokens: u32,
    ) -> Result<String> {
        // 复用上下文：清空上一轮 KV cache
        self.ctx.clear_kv_cache();

        // Qwen Instruct (ChatML) 模板；str_to_token 以 special=true 解析特殊标记
        let chat_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        );

        let tokens_list = self
            .client
            .model
            .str_to_token(&chat_prompt, AddBos::Always)
            .context("Prompt 分词失败")?;
        anyhow::ensure!(!tokens_list.is_empty(), "Prompt 分词结果为空");

        let total = tokens_list.len();
        let prompt_len = total as i32;
        let n_ctx = self.ctx.n_ctx() as i32;
        let n_batch = (self.ctx.n_batch() as usize).max(1);

        // 生成至少要留出空间，否则直接判失败（比让 llama.cpp 断言崩溃好）
        anyhow::ensure!(
            prompt_len + 16 < n_ctx,
            "Prompt 长度（{} tokens）超出上下文（{} tokens），请减小批次或增大 --ctx-size",
            prompt_len,
            n_ctx
        );
        let n_len = (prompt_len + max_new_tokens as i32).min(n_ctx - 4);

        // ---- 分块 prefill：单次 decode 不能超过 n_batch 个 token ----
        let cap = total.min(n_batch);
        let mut batch = LlamaBatch::new(cap.max(1), 1);
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            if batch.n_tokens() as usize >= cap {
                self.ctx.decode(&mut batch).context("解码 Prompt 失败")?;
                batch.clear();
            }
            let is_last = (i as usize) + 1 == total;
            batch.add(token, i, &[0], is_last)?;
        }
        if batch.n_tokens() > 0 {
            self.ctx.decode(&mut batch).context("解码 Prompt 失败")?;
        }
        if total > cap {
            info!(
                "长 prompt 触发分块 prefill：共 {} tokens，分 {} 块（n_batch={}）",
                total,
                (total + cap - 1) / cap,
                cap
            );
        }

        let mut sampler = sampling.build();

        // 位置计数器从 prompt 末尾继续（不是最后一个 chunk 的长度）
        let mut n_cur = prompt_len;
        let mut result_text = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut truncated = false;

        while n_cur < n_len {
            // logits 只挂在当前 batch 的最后一个槽位上
            let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // <|im_end|> / <|endoftext|> 等结束符
            if self.client.model.is_eog_token(token) {
                break;
            }

            if let Ok(piece) = self.client.model.token_to_piece(token, &mut decoder, true, None) {
                result_text.push_str(&piece);
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            self.ctx.decode(&mut batch).context("解码生成 Token 失败")?;
        }
        if n_cur >= n_len {
            truncated = true;
        }

        if truncated {
            warn!(
                "生成被截断（达到 {} tokens 上限 / 上下文 {}），输出可能不完整：{}",
                max_new_tokens,
                n_ctx,
                truncate_chars(&result_text, 80)
            );
        }
        Ok(result_text)
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    let t: String = s.chars().take(max).collect();
    if t.chars().count() < s.chars().count() {
        format!("{}…", t)
    } else {
        t
    }
}

// ============================================================================
// JSON 提取与修复
// ============================================================================

/// 从模型输出中稳健地提取 JSON（容忍 ``` 代码块围栏、前后废话与思考文本）。
///
/// 策略：从每个 `open` 起做**括号配平扫描**（跳过字符串内部与转义），
/// 依次产出候选 JSON 片段；若输出被截断导致无法配平，追加一个
/// 「第一个 open 到最后一个 close」的兜底候选。
pub fn json_candidates(raw: &str, is_array: bool, limit: usize) -> Vec<String> {
    let (open, close) = if is_array { ('[', ']') } else { ('{', '}') };
    let chars: Vec<char> = raw.chars().collect();
    let mut out: Vec<String> = Vec::new();

    for (start, &c) in chars.iter().enumerate() {
        if out.len() >= limit {
            break;
        }
        if c != open {
            continue;
        }
        if let Some(s) = scan_balanced(&chars, start, open, close) {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }

    if out.is_empty() {
        // 兜底：截断输出
        if let (Some(s), Some(e)) = (raw.find(open), raw.rfind(close)) {
            if e > s {
                out.push(raw[s..=e].to_string());
            }
        }
    }
    out
}

/// 取第一个候选（保留给只需要"有没有 JSON"的调用方）。
pub fn extract_json(raw: &str, is_array: bool) -> Option<String> {
    json_candidates(raw, is_array, 1).into_iter().next()
}

fn scan_balanced(chars: &[char], start: usize, open: char, close: char) -> Option<String> {
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for i in start..chars.len() {
        let c = chars[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(chars[start..=i].iter().collect());
            }
        }
    }
    None
}

/// 修复小模型常见的 JSON 瑕疵：对象/数组结尾前多余的逗号（`,` 后紧跟 `}`/`]`）。
/// 字符串内部的逗号不受影响。
pub fn repair_json(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in chars.iter().enumerate() {
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            ',' => {
                let next = chars[i + 1..].iter().find(|x| !x.is_whitespace());
                if matches!(next, Some('}') | Some(']')) {
                    // 丢弃尾随逗号
                } else {
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// 提取 + 修复 + 反序列化的一步到位封装：依次尝试每个候选 JSON 片段
/// （原文 → 修复尾随逗号后），全部失败时返回最后一次的错误说明。
pub fn parse_json<T: serde::de::DeserializeOwned>(raw: &str, is_array: bool) -> Result<T, String> {
    let candidates = json_candidates(raw, is_array, 6);
    if candidates.is_empty() {
        return Err(format!(
            "输出中未找到 JSON（{}）",
            truncate_chars(&raw.replace('\n', " "), 80)
        ));
    }
    let mut last_err = String::new();
    for json_str in &candidates {
        // 原文解析失败则尝试修复（去尾随逗号）后再解析一次
        let repaired = repair_json(json_str);
        let result = serde_json::from_str::<T>(json_str)
            .or_else(|_| serde_json::from_str::<T>(&repaired));
        match result {
            Ok(v) => {
                if repaired != *json_str {
                    debug!("JSON 含尾随逗号等瑕疵，已自动修复");
                }
                return Ok(v);
            }
            Err(e) => {
                last_err = format!(
                    "{}：{}",
                    e,
                    truncate_chars(&json_str.replace('\n', " "), 120)
                )
            }
        }
    }
    Err(format!("JSON 字段不符（{}）", last_err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Item {
        i: i64,
        t: String,
    }

    #[test]
    fn extract_plain_object() {
        assert_eq!(
            extract_json(r#"{"a":1}"#, false).as_deref(),
            Some(r#"{"a":1}"#)
        );
    }

    #[test]
    fn extract_from_fenced_block_with_prose() {
        let raw = "好的，结果如下：\n```json\n{\"decision\":\"keep\",\"reason\":\"{有效}\"}\n```\n以上。";
        assert_eq!(
            extract_json(raw, false).as_deref(),
            Some("{\"decision\":\"keep\",\"reason\":\"{有效}\"}")
        );
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_scan() {
        let raw = r#"前缀 {"t":"a}b{c"} 后缀}"#;
        assert_eq!(extract_json(raw, false).as_deref(), Some(r#"{"t":"a}b{c"}"#));
    }

    #[test]
    fn escaped_quote_inside_string() {
        let raw = r#"{"t":"他说\"好\""}"#;
        assert_eq!(extract_json(raw, false).as_deref(), Some(raw));
    }

    #[test]
    fn extract_array_and_first_complete_one() {
        let raw = "解释：[1,2] 不是结果。真正结果：[{\"i\":1,\"t\":\"a\"}]";
        // 第一个配平的 [ 是 [1,2]，它本身也是合法 JSON，调用方需容忍；
        // 但数组场景下我们只关心能否解析出目标结构，故这里断言提取到第一个。
        assert_eq!(extract_json(raw, true).as_deref(), Some("[1,2]"));
    }

    #[test]
    fn first_complete_value_wins_when_output_has_trailing_garbage() {
        let raw = r#"{"i":1,"t":"abc"}{"i":2,"t":"def"#;
        // 第二个对象被截断，但第一个是完整的
        assert_eq!(
            extract_json(raw, false).as_deref(),
            Some(r#"{"i":1,"t":"abc"}"#)
        );
    }

    #[test]
    fn multiple_candidates_are_all_offered_to_parse_json() {
        let raw = r#"先给个错的 {"nope":1} 再给对的 {"i":1,"t":"a"}"#;
        let cs = json_candidates(raw, false, 6);
        assert_eq!(cs.len(), 2, "{cs:?}");
        #[derive(serde::Deserialize)]
        struct It { i: i64, t: String }
        let v: It = parse_json(raw, false).expect("应挑到能解析的那个");
        assert_eq!((v.i, v.t.as_str()), (1, "a"));
    }

    #[test]
    fn no_json_at_all() {
        assert_eq!(extract_json("完全不是 JSON", false), None);
        assert_eq!(extract_json("", true), None);
    }

    #[test]
    fn repair_trailing_commas() {
        assert_eq!(
            repair_json(r#"{"a":1,"b":[1,2,],}"#),
            r#"{"a":1,"b":[1,2]}"#
        );
        // 字符串内的 ",}" 不动
        assert_eq!(repair_json(r#"{"a":"x,}"}"#), r#"{"a":"x,}"}"#);
    }

    #[test]
    fn parse_json_repairs_and_succeeds() {
        let raw = "```json\n[{\"i\":1,\"t\":\"你好\",},]\n```";
        let v: Vec<Item> = parse_json(raw, true).expect("应修复后解析成功");
        assert_eq!(v, vec![Item { i: 1, t: "你好".into() }]);
    }

    #[test]
    fn parse_json_error_message_is_useful() {
        let e = parse_json::<Vec<Item>>("没有 JSON", true).unwrap_err();
        assert!(e.contains("未找到 JSON"), "{e}");
        let e = parse_json::<Vec<Item>>(r#"[{"i":"x"}]"#, true).unwrap_err();
        assert!(e.contains("字段不符"), "{e}");
    }

    #[test]
    fn parse_json_picks_the_candidate_that_fits() {
        // 模型先写了一段带方括号的解释，再给出真正的结果数组
        let raw = "解释：[1,2] 不是结果。真正结果：[{\"i\":1,\"t\":\"a\"}]";
        let v: Vec<Item> = parse_json(raw, true).expect("应跳过不匹配的候选");
        assert_eq!(v, vec![Item { i: 1, t: "a".into() }]);
    }
}
