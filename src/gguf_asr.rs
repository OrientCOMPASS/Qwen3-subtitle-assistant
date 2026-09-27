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

/// GGUF 版 Qwen3-ASR（S2TT）推理引擎：LM + mmproj 音频编码器 + mtmd 上下文。
pub struct GgufAsr {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    mtmd: *mut sys::mtmd_context,
    sample_rate: i32,
}

// llama/mtmd 句柄在单引擎内串行使用；产品批处理线程模型与 sherpa 后端一致。
unsafe impl Send for GgufAsr {}

impl GgufAsr {
    /// 加载 LM GGUF 与 mmproj，建立推理上下文。
    ///
    /// `ngl`：offload 到 GPU 的层数（999=全量，llama 按模型实际层数截断；无 GPU 时
    /// 自动留在 CPU）。权重迁移显存后 ggml 会释放主机侧副本（用户需求之一）。
    pub fn open(lm: &Path, mmproj: &Path, threads: i32, ngl: i32,
                cuda_libs: Option<&Path>) -> Result<Self> {
        init_backends(cuda_libs);

        unsafe {
            let c_lm = to_cstring(lm)?;
            let mut mparams = sys::llama_model_default_params();
            mparams.n_gpu_layers = ngl;
            let model = sys::llama_model_load_from_file(c_lm.as_ptr(), mparams);
            if model.is_null() {
                bail!("GGUF 模型加载失败: {:?}（确认文件完整且为 llama.cpp 格式）", lm);
            }

            let mut cparams = sys::llama_context_default_params();
            cparams.n_ctx = 4096;       // 音频 12.5 token/s + 文本 prompt + 生成，4k 足够单段
            cparams.n_batch = 2048;
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

            Ok(Self { model, ctx, mtmd, sample_rate })
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

    /// 转录一段 16kHz f32 单声道音频，`context` 为任务开关（进 system 段，
    /// "translate to Chinese"=直出中文；空串=转写）。
    ///
    /// TODO(E3-Phase 1b)：mtmd_tokenize(text+bitmap) → mtmd_helper_eval_chunks →
    /// greedy 采样循环 → detokenize → 解析 `language X<asr_text>正文`
    /// （`language None` → 空串，与产品「静音段丢弃」语义一致）。
    pub fn transcribe(&mut self, _samples: &[f32], _context: &str) -> Result<String> {
        bail!("GGUF 推理链将在 E3-Phase 1b 接入（本版本仅验证加载/链接，见 --gguf-selftest）")
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

/// `--gguf-selftest` 入口：加载模型 + 打印后端/能力/资源信息。
/// 在用户机器上用于验证「CUDA 目录加载 / Vulkan 设备发现 / 显存迁移后内存释放」。
pub fn selftest(lm: &Path, mmproj: &Path, threads: i32,
                cuda_libs: Option<&Path>) -> Result<String> {
    let t0 = std::time::Instant::now();
    let asr = GgufAsr::open(lm, mmproj, threads, 999, cuda_libs)?;
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
