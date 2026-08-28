use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use rand::Rng;

pub struct LlmClient {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
}

impl LlmClient {
    pub fn new(model_path: &Path) -> Result<Self> {
        // 1. 屏蔽 llama.cpp 底层 C++ 的杂乱日志输出
        // 通过注入空回调，拦截所有原生的 stderr 打印
        unsafe {
            llama_cpp_sys_2::llama_log_set(Some(empty_log_callback), std::ptr::null_mut());
        }

        let backend = LlamaBackend::init().context("初始化 llama backend 失败")?;
        
        // 尝试将尽可能多的层卸载到 GPU，如果没有 GPU 则自动回退 CPU
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("加载模型失败: {:?}", model_path))?;

        Ok(Self {
            backend,
            model,
            n_ctx: 8192, // 8K 上下文窗口足以应对批量翻译，显存占用极低
        })
    }

    pub fn chat(&self, prompt: &str) -> Result<String> {
        // 2. 核心修复：应用 Qwen Instruct (ChatML) 模板
        // 这会让模型明白它需要扮演 assistant 角色生成回复，而不是续写文档
        let chat_prompt = format!(
            "<|im_start|>system\n你是一个专业的字幕翻译与内容分析助手，请严格按照要求的 JSON 格式输出，不要输出任何多余的解释。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));

        let mut ctx = self.model
            .new_context(&self.backend, ctx_params)
            .context("创建 LLM 上下文失败")?;

        let tokens_list = self.model
            .str_to_token(&chat_prompt, AddBos::Always)
            .context("Prompt 分词失败")?;

        let n_len = self.n_ctx as i32 / 2; 
        
        if tokens_list.len() as i32 >= n_len {
            anyhow::bail!("Prompt 长度 ({}) 超过最大限制", tokens_list.len());
        }

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        ctx.decode(&mut batch).context("解码 Prompt 失败")?;
        let mut rng = rand::thread_rng();
        let dynamic_seed = rng.gen::<u32>();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.6), 
            LlamaSampler::dist(dynamic_seed), 
        ]);
        let mut n_cur = batch.n_tokens();
        let mut result_text = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // 遇到 <|im_end|> (151645) 时，说明模型输出完毕，正常退出循环
            if self.model.is_eog_token(token) {
                break;
            }

            let token_str = self.model.token_to_piece(token, &mut decoder, true, None)
                .unwrap_or_default();
            
            result_text.push_str(&token_str);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;
            ctx.decode(&mut batch).context("解码生成 Token 失败")?;
        }

        Ok(result_text)
    }
}

// 空的 C 回调函数，用于彻底屏蔽 llama.cpp 的原生 stderr 日志
unsafe extern "C" fn empty_log_callback(_level: i32, _text: *const i8, _user_data: *mut std::ffi::c_void) {}

/// 从模型输出中稳健地提取 JSON
pub fn extract_json(raw: &str, is_array: bool) -> Option<String> {
    let (open, close) = if is_array { ('[', ']') } else { ('{', '}') };
    let start = raw.find(open)?;
    let end = raw.rfind(close)?;
    if end <= start {
        return None;
    }
    Some(raw[start..=end].to_string())
}