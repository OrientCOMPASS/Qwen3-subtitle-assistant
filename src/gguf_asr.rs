//! llama.cpp/GGUF 推理后端（E3「单模型产品」路线）。
//!
//! 背景（决策与实测证据见 finetune/INTEGRATION.md §8/§9）：
//! 产品从「sherpa-onnx ASR + 1.7B LLM 四段后处理」收敛为「S2TT 微调模型单段直出」，
//! 运行时选 llama.cpp/GGUF：
//!   * 硬件加速：CUDA（ggml-cuda.dll + 用户 `--cuda-libs` 指定的 NVIDIA 运行库目录，
//!     经 `ggml_backend_load_all_from_path` 加载）→ Vulkan（ggml-vulkan.dll 随包，
//!     vulkan-1.dll 系统自带，覆盖任意 Windows GPU）→ CPU 兜底；
//!   * 权重全量上卡（n_gpu_layers=999）后 ggml 释放主机侧副本；
//!   * 模型 = E2 产物：LoRA 合并 → 官方转换器 → LM Q4_K_M(1.03GiB) + mmproj q8_0(0.33GiB)。
//!
//! Phase 1a（本文件现状）：后端发现/加载、模型与 mmproj 加载、能力查询、资源统计——
//! 用于 `--gguf-selftest` 验证「mtmd feature 在产品工具链下可编译可链接、GGUF 可加载」。
//! Phase 1b：mtmd_helper_eval_chunks + greedy 采样 + `language X<asr_text>` 解析（transcribe）。

use anyhow::{bail, Result};
use log::info;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Once;

use llama_cpp_sys_2 as sys;

static BACKEND_INIT: Once = Once::new();

fn to_cstring(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes().to_vec())
        .map_err(|e| anyhow::anyhow!("路径含非法字符 {:?}: {}", p, e))
}

/// 初始化 ggml/llama 后端。
///
/// 顺序：llama_backend_init → （可选）从用户目录加载外置后端（CUDA 场景：
/// ggml-cuda.dll 的依赖链 cudart/cublas/cudnn 也在该目录，先加入 DLL 搜索路径）→
/// 枚举后端设备打日志（用户可见 CUDA/Vulkan/CPU 的发现结果，与产品现有
/// 「启动时主动探测并打印后端选择结果」的行为一致）。
fn init_backends(cuda_libs: Option<&Path>) {
    BACKEND_INIT.call_once(|| {
        unsafe { sys::llama_backend_init() };

        if let Some(dir) = cuda_libs {
            // 依赖链解析：把用户目录加进进程 DLL 搜索路径（复用 runtime 的
            // AddDllDirectory 封装），再让 ggml 从该目录加载后端模块。
            crate::runtime::dll::add_search_dirs(&[dir.to_path_buf()]);
            match to_cstring(dir) {
                Ok(c) => unsafe { sys::ggml_backend_load_all_from_path(c.as_ptr()) },
                Err(e) => log::warn!("--cuda-libs 路径无效: {e}"),
            }
            info!("已从 {:?} 加载外置推理后端（CUDA）", dir);
        }

        unsafe {
            let n = sys::ggml_backend_dev_count();
            info!("ggml 后端设备 {} 个:", n);
            for i in 0..n {
                let dev = sys::ggml_backend_dev_get(i);
                if dev.is_null() {
                    continue;
                }
                let name = CStr::from_ptr(sys::ggml_backend_dev_name(dev))
                    .to_string_lossy()
                    .into_owned();
                let desc = CStr::from_ptr(sys::ggml_backend_dev_description(dev))
                    .to_string_lossy()
                    .into_owned();
                info!("  [{}] {} — {}", i, name, desc);
            }
        }
    });
}

/// 按优先级挑一个 GPU 设备：want_cuda=true 时优先 CUDA 后端设备；
/// 否则（及 CUDA 缺席时）取 Vulkan 设备中索引最大者（多显卡机器上索引 0 常是
/// 显示卡/核显，索引最大的通常是主力独显——用户明确要求的策略）。
fn pick_device(want_cuda: bool) -> Option<sys::ggml_backend_dev_t> {
    // ggml_backend_dev_type 枚举序（ggml-backend.h，版本 pin 内稳定）：
    //   CPU=0, GPU=1, IGPU=2, ACCEL=3, META=4
    const DEV_GPU: u32 = 1;
    const DEV_IGPU: u32 = 2;
    unsafe {
        let n = sys::ggml_backend_dev_count();
        let mut gpus: Vec<(usize, sys::ggml_backend_dev_t, String)> = Vec::new();
        for i in 0..n {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let ty = sys::ggml_backend_dev_type(dev) as u32;
            if ty != DEV_GPU && ty != DEV_IGPU {
                continue;
            }
            let name = CStr::from_ptr(sys::ggml_backend_dev_name(dev))
                .to_string_lossy().into_owned();
            gpus.push((i, dev, name));
        }
        if gpus.is_empty() {
            return None;
        }
        let pick = if want_cuda {
            match gpus.iter().rev().find(|(_, _, n)| n.to_lowercase().contains("cuda")) {
                Some(p) => Some(p),
                None => {
                    log::warn!("--cuda-libs 已指定但未发现 CUDA 后端设备（ggml-cuda.dll 加载失败？），回退其他 GPU");
                    None
                }
            }
        } else {
            None
        };
        let pick = pick
            .or_else(|| gpus.iter().rev().find(|(_, _, n)| n.to_lowercase().contains("vulkan")))
            .or_else(|| gpus.last());
        pick.map(|(i, dev, name)| {
            info!("选定推理设备: [{}] {}（权重将全量入显存，主机副本随后释放）", i, name);
            *dev
        })
    }
}

/// GGUF 版 Qwen3-ASR（S2TT）推理引擎：LM + mmproj 音频编码器 + mtmd 上下文。
pub struct GgufAsr {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    mtmd: *mut sys::mtmd_context,
    sample_rate: i32,
    n_batch: i32,
    max_new_tokens: i32,
}

// llama/mtmd 句柄在单引擎内串行使用；产品批处理线程模型与 sherpa 后端一致。
unsafe impl Send for GgufAsr {}

impl GgufAsr {
    /// 加载 LM GGUF 与 mmproj，建立推理上下文。
    ///
    /// `ngl`：offload 到 GPU 的层数（999=全量，llama 按模型实际层数截断；无 GPU 时
    /// 自动留在 CPU）。权重迁移显存后 ggml 会释放主机侧副本（用户需求之一）。
    pub fn open(lm: &Path, mmproj: &Path, threads: i32, ngl: i32,
                max_new_tokens: i32, cuda_libs: Option<&Path>, force_cpu: bool) -> Result<Self> {
        init_backends(cuda_libs);

        // 设备选择（用户要求的优先级）：
        //   --cuda-libs 指定 → CUDA 设备；否则 Vulkan 中**索引最大**的 GPU；都没有 → CPU。
        // 通过 llama_model_params.devices（NULL 结尾列表）钉住所选设备，
        // 权重全量 offload 到显存后 ggml 释放主机侧副本。
        let mut picked: Vec<sys::ggml_backend_dev_t> = Vec::new();
        let mut ngl_effective = ngl;
        if !force_cpu {
            if let Some(dev) = pick_device(cuda_libs.is_some()) {
                picked.push(dev);
            } else {
                info!("未发现可用 GPU 后端设备，权重留在 CPU 内存");
                ngl_effective = 0;
            }
        } else {
            info!("--device cpu：强制纯 CPU 推理");
            ngl_effective = 0;
        }
        let mut devices: Vec<sys::ggml_backend_dev_t> = picked.clone();
        devices.push(std::ptr::null_mut());

        unsafe {
            let c_lm = to_cstring(lm)?;
            let mut mparams = sys::llama_model_default_params();
            mparams.n_gpu_layers = ngl_effective;
            if !picked.is_empty() {
                mparams.devices = devices.as_mut_ptr();
            }
            let model = sys::llama_model_load_from_file(c_lm.as_ptr(), mparams);
            if model.is_null() {
                bail!("GGUF 模型加载失败: {:?}（确认文件完整且为 llama.cpp 格式）", lm);
            }

            let mut cparams = sys::llama_context_default_params();
            cparams.n_ctx = 4096;       // 音频 12.5 token/s + 文本 prompt + 生成，4k 足够单段
            cparams.n_batch = 2048;     // 音频 chunk 一次 prefill 需要较大 batch
            cparams.n_threads = threads.max(1);
            cparams.n_threads_batch = threads.max(1);
            let ctx = sys::llama_init_from_model(model, cparams);
            if ctx.is_null() {
                sys::llama_model_free(model);
                bail!("llama_context 创建失败");
            }

            let c_mm = to_cstring(mmproj)?;
            let mparams2 = sys::mtmd_context_params_default();
            let mtmd = sys::mtmd_init_from_file(c_mm.as_ptr(), model, mparams2);
            if mtmd.is_null() {
                sys::llama_free(ctx);
                sys::llama_model_free(model);
                bail!("mmproj 加载失败: {:?}（需要与 LM 同一次转换产出）", mmproj);
            }

            if !sys::mtmd_support_audio(mtmd) {
                sys::mtmd_free(mtmd);
                sys::llama_free(ctx);
                sys::llama_model_free(model);
                bail!("该 mmproj 不支持音频输入（不是 Qwen3-ASR 的多模态投影？）");
            }
            let sample_rate = sys::mtmd_get_audio_sample_rate(mtmd);

            Ok(Self { model, ctx, mtmd, sample_rate, n_batch: cparams.n_batch as i32,
                        max_new_tokens: max_new_tokens.max(16) })
        }
    }

    /// 目标音频采样率（Qwen3-ASR 为 16000；调用方重采样到此值）。
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    /// LM 权重体积（MB），selftest 用。
    pub fn model_size_mb(&self) -> f64 {
        unsafe { sys::llama_model_size(self.model) as f64 / 1e6 }
    }

    /// 转录一段 16kHz f32 单声道音频；`context` 为任务开关（进 system 段：
    /// "translate to Chinese"=直出中文，空串=转写）。返回**解析后的正文**
    /// （原始输出 `language X<asr_text>正文`；`language None`/空正文 → 空串，
    /// 与产品「静音段丢弃」语义一致）。
    ///
    /// 推理链（与 llama.cpp mtmd-helper / llama-server 同款语义，E2 已实证）：
    /// 清 KV → 模板拼 prompt（Qwen3-ASR 固定 chat 模板，音频位用 mtmd marker）→
    /// mtmd_tokenize(text+audio bitmap) → mtmd_helper_eval_chunks（一次完成
    /// 音频编码与文本 prefill）→ greedy 采样到 EOG → detokenize → 解析。
    pub fn transcribe(&mut self, samples: &[f32], context: &str) -> Result<AsrUtterance> {
        use std::os::raw::{c_char, c_int};

        if samples.is_empty() {
            return Ok(AsrUtterance { lang: "None".into(), text: String::new() });
        }
        unsafe {
            // 0) 清空 KV cache（每段独立会话；-1,-1,-1 = 全部序列全部位置）
            let mem = sys::llama_get_memory(self.ctx);
            sys::llama_memory_seq_rm(mem, -1, -1, -1);

            // 1) prompt：Qwen3-ASR 官方 chat 模板（tokenizer_config 同构 jinja 实测），
            //    音频占位用 mtmd 的 marker（mtmd_tokenize 会在该处插入音频 chunk）
            let marker = CStr::from_ptr(sys::mtmd_get_marker(self.mtmd))
                .to_string_lossy().into_owned();
            let prompt = format!(
                "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>{marker}<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
            );
            let c_prompt = CString::new(prompt)?;

            // 2) 音频 bitmap（16k f32；采样率不符时由调用方负责重采样）
            let bitmap = sys::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr());
            if bitmap.is_null() {
                bail!("mtmd_bitmap_init_from_audio 失败");
            }

            // 3) tokenize：文本按 marker 切开，音频 chunk 插到 marker 位
            let chunks = sys::mtmd_input_chunks_init();
            if chunks.is_null() {
                sys::mtmd_bitmap_free(bitmap);
                bail!("mtmd_input_chunks_init 失败");
            }
            let text = sys::mtmd_input_text {
                text: c_prompt.as_ptr(),
                text_len: c_prompt.as_bytes().len(),
                add_special: false,   // 模板已含全部特殊 token 字面量
                parse_special: true,  // <|im_start|> 等按特殊 token 解析
            };
            let bitmaps: [*const sys::mtmd_bitmap; 1] = [bitmap];
            let rc = sys::mtmd_tokenize(self.mtmd, chunks, &text, bitmaps.as_ptr(), 1);
            if rc != 0 {
                sys::mtmd_input_chunks_free(chunks);
                sys::mtmd_bitmap_free(bitmap);
                bail!("mtmd_tokenize 失败 (rc={rc})");
            }

            // 4) prefill：文本 chunk 走 llama_decode，音频 chunk 走 mtmd 编码后拼接
            let mut n_past: sys::llama_pos = 0;
            let rc = sys::mtmd_helper_eval_chunks(
                self.mtmd, self.ctx, chunks,
                0,       // n_past 起点
                0,       // seq_id
                self.n_batch,
                true,    // logits_last：末位出 logits 供采样
                &mut n_past,
            );
            sys::mtmd_input_chunks_free(chunks);
            sys::mtmd_bitmap_free(bitmap);
            if rc != 0 {
                bail!("mtmd_helper_eval_chunks 失败 (rc={rc})");
            }

            // 5) greedy 采样（与产品 sherpa 后端 greedy_search、模型 generation_config
            //    do_sample=false 语义一致；Qwen3-ASR 输出确定性强，无需温度采样）
            let vocab = sys::llama_model_get_vocab(self.model);
            let smpl = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            sys::llama_sampler_chain_add(smpl, sys::llama_sampler_init_greedy());

            let mut out = Vec::<u8>::new();
            let mut piece = vec![0i8; 1024];
            let mut n_tok = 0;
            while n_tok < self.max_new_tokens {
                let mut tok = sys::llama_sampler_sample(smpl, self.ctx, -1);
                if sys::llama_vocab_is_eog(vocab, tok) {
                    break;
                }
                // pin 版本签名：(vocab, token, buf, length, lstrip, special)
                let n = sys::llama_token_to_piece(
                    vocab, tok, piece.as_mut_ptr() as *mut c_char,
                    piece.len() as c_int, 0, true,
                );
                if n > 0 {
                    out.extend_from_slice(std::slice::from_raw_parts(
                        piece.as_ptr() as *const u8, n as usize));
                }
                let mut batch = sys::llama_batch_get_one(&mut tok as *mut sys::llama_token, 1);
                if sys::llama_decode(self.ctx, batch) != 0 {
                    break;
                }
                n_tok += 1;
            }
            sys::llama_sampler_free(smpl);
            let raw = String::from_utf8_lossy(&out).into_owned();

            // 6) 解析 `language X<asr_text>正文`（llama.cpp 不做解析，返回原始输出——
            //    与 sherpa/qwen-asr 运行时的差异点，E1 已实测确认）
            Ok(parse_asr_output(&raw))
        }
    }
}

impl Drop for GgufAsr {
    fn drop(&mut self) {
        unsafe {
            sys::mtmd_free(self.mtmd);
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

/// 一次转录的解析结果。`lang` 为 "None" 表示模型判定无有效语音——
/// 丢弃策略（与 finetune/s2tt_pipeline.py 实测结论一致）：
/// 正文为空 → 丢弃；lang=None 但带正文 → 短碎片(<2s 或 <4 字)是幻觉丢弃、长段是真实语音保留。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrUtterance {
    pub lang: String,
    pub text: String,
}

/// 解析 Qwen3-ASR 原始输出 `language X<asr_text>正文`。
fn parse_asr_output(raw: &str) -> AsrUtterance {
    let s = raw.trim();
    match s.split_once("<asr_text>") {
        Some((head, body)) => AsrUtterance {
            lang: head.trim().trim_start_matches("language").trim().to_string(),
            text: body.trim().to_string(),
        },
        None => AsrUtterance { lang: String::new(), text: s.to_string() },
    }
}

#[cfg(test)]
mod tests {
    use super::parse_asr_output;

    #[test]
    fn parse_asr_output_variants() {
        let r = parse_asr_output("language Japanese<asr_text>こんにちは");
        assert_eq!((r.lang.as_str(), r.text.as_str()), ("Japanese", "こんにちは"));
        let r = parse_asr_output("language None<asr_text>");
        assert_eq!((r.lang.as_str(), r.text.as_str()), ("None", ""));
        let r = parse_asr_output("language Chinese<asr_text> 你好。 ");
        assert_eq!((r.lang.as_str(), r.text.as_str()), ("Chinese", "你好。"));
        let r = parse_asr_output("裸文本无标签");
        assert_eq!((r.lang.as_str(), r.text.as_str()), ("", "裸文本无标签"));
        // lang=None 但带正文（实测存在）：正文保留，交由调用方按时长/字数分流
        let r = parse_asr_output("language None<asr_text>今天");
        assert_eq!((r.lang.as_str(), r.text.as_str()), ("None", "今天"));
    }
}

/// `--gguf-selftest` 入口：加载模型 + 打印后端/能力/资源信息。
/// 在用户机器上用于验证「CUDA 目录加载 / Vulkan 设备发现 / 显存迁移后内存释放」。
pub fn selftest(lm: &Path, mmproj: &Path, threads: i32, ngl: i32,
                cuda_libs: Option<&Path>, force_cpu: bool) -> Result<String> {
    let t0 = std::time::Instant::now();
    let asr = GgufAsr::open(lm, mmproj, threads, ngl, 128, cuda_libs, force_cpu)?;
    let load = t0.elapsed().as_secs_f64();

    // 进程内存（Windows：GetProcessMemoryInfo；其他平台退化为不可用）
    let rss = process_rss_mb();
    let rss_txt = rss.map(|v| format!("{v:.0} MB")).unwrap_or_else(|| "不可用".into());

    Ok(format!(
        "GGUF 自检通过：LM {:?}（{:.0} MB）+ mmproj {:?}；加载 {load:.1}s；\
         音频支持 ✓；目标采样率 {} Hz；进程内存 {rss_txt}（GPU 环境下应远小于模型体积，\
         权重已入显存并释放主机副本）",
        lm.file_name().unwrap_or_default(),
        asr.model_size_mb(),
        mmproj.file_name().unwrap_or_default(),
        asr.sample_rate(),
    ))
}

#[cfg(windows)]
fn process_rss_mb() -> Option<f64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, counters: *mut ProcessMemoryCounters,
                                   cb: u32) -> i32;
    }
    unsafe {
        let mut c = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0, peak_working_set_size: 0, working_set_size: 0,
            quota_peak_paged_pool_usage: 0, quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0, quota_non_paged_pool_usage: 0,
            pagefile_usage: 0, peak_pagefile_usage: 0,
        };
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
            Some(c.working_set_size as f64 / 1e6)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
fn process_rss_mb() -> Option<f64> {
    None
}
