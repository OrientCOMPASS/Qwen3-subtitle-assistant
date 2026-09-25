//! llama.cpp（外置 llama.dll/ggml*.dll）绑定与自回归生成。
//!
//! 重构要点：
//! - `LlmClient::session` 创建**可复用的推理会话**：KV cache 上下文只建一次，
//!   每轮对话后 `clear_kv_cache()` 复用。逐句质检场景下每句都要调用 LLM，
//!   旧版“每次调用重建 8K 上下文”的做法开销巨大，必须避免。
//! - GPU offload 层数由运行时探测结果 / `--gpu-layers` 决定（外置 ggml-cuda.dll
//!   存在且可加载时才会上卡）。
//! - 不再直接依赖 llama-cpp-sys-2：日志屏蔽改用 `LlamaBackend::void_logs()`。
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
use log::info;
use std::num::NonZeroU32;
use std::path::Path;

pub struct LlmClient {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
}

impl LlmClient {
    pub fn new(model_path: &Path, n_ctx: u32, gpu_layers: u32) -> Result<Self> {
        let mut backend = LlamaBackend::init().context("初始化 llama backend 失败")?;
        // 屏蔽 llama.cpp 底层 C++ 的原生日志（等价于旧版手工注入空回调）
        backend.void_logs();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("加载模型失败: {:?}", model_path))?;

        info!(
            "LLM 已加载: {:?}（n_gpu_layers={}, n_ctx={}）",
            model_path, gpu_layers, n_ctx
        );

        Ok(Self { backend, model, n_ctx })
    }

    /// 创建推理会话。会话内部持有 KV cache 上下文，可多次 chat() 复用。
    pub fn session(&self) -> Result<LlmSession<'_>> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
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
    /// 一轮 ChatML 对话。`temperature <= 0` 时使用贪心采样。
    /// `max_new_tokens` 为生成上限（防止小模型跑飞吃满上下文）。
    pub fn chat(&mut self, system: &str, user: &str, temperature: f32, max_new_tokens: u32) -> Result<String> {
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
        let prompt_len = tokens_list.len() as i32;

        let n_ctx = self.client.n_ctx as i32;
        let reserve = max_new_tokens as i32;
        anyhow::ensure!(
            prompt_len + 16 < n_ctx,
            "Prompt 长度（{} tokens）超出上下文（{}），请减小批次或增大 --ctx-size",
            prompt_len,
            n_ctx
        );
        let n_len = (prompt_len + reserve).min(n_ctx - 4);

        let mut batch = LlamaBatch::new(tokens_list.len().max(1), 1);
        let last_index = tokens_list.len() as i32 - 1;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            batch.add(token, i, &[0], i == last_index)?;
        }
        self.ctx.decode(&mut batch).context("解码 Prompt 失败")?;

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() ^ (d.as_secs() as u32))
            .unwrap_or(42);

        let mut sampler = if temperature > 0.0 {
            LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(seed)])
        } else {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        };

        let mut n_cur = batch.n_tokens();
        let mut result_text = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= n_len {
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

        Ok(result_text)
    }
}

/// 从模型输出中稳健地提取 JSON（容忍 ``` 代码块围栏与前后废话）。
pub fn extract_json(raw: &str, is_array: bool) -> Option<String> {
    let (open, close) = if is_array { ('[', ']') } else { ('{', '}') };
    let start = raw.find(open)?;
    let end = raw.rfind(close)?;
    if end <= start {
        return None;
    }
    Some(raw[start..=end].to_string())
}
