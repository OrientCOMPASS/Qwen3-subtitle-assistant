//! 配置解析：CLI 参数 + 资源路径解析（CWD 优先，找不到回退 exe 同目录——
//! 拖拽/快捷方式启动时 CWD 不可控，这个回退语义从 v0.2 保留至今）。

use crate::cli::Args;
use crate::runtime::{DevicePref, RuntimeProbe};
use anyhow::{bail, Context, Result};
use log::{info, warn};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    // ---- 模型 ----
    pub lm_gguf: PathBuf,
    pub mmproj_gguf: PathBuf,
    pub context: String,

    // ---- 推理设备 ----
    pub device: DevicePref,
    pub cuda_libs: Option<PathBuf>,
    pub threads: i32,
    pub gpu_layers: i32,
    pub max_new_tokens: i32,

    // ---- 采样（v0.6：默认温度采样；--greedy 回到确定性贪心） ----
    pub greedy: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,

    // ---- VAD ----
    pub vad_threshold: f32,
    pub vad_min_silence: f32,
    /// 单段语音长度上限（秒，超长硬拆；旧名 --vad-buffer-secs）
    pub vad_max_seg_secs: f32,

    // ---- 流式管线缓冲 ----
    /// VAD→ASR 滞回缓冲容量（字节，--buffer-mb × 1e6）
    pub buffer_bytes: usize,

    // ---- 排版与输出 ----
    pub max_line_width: usize,
    pub max_cue_secs: f64,
    pub output_dir: Option<PathBuf>,
    /// 是否另存排版前 .raw.srt（默认否：只交付一个 .srt）
    pub raw_srt: bool,

    // ---- 质量过滤 ----
    /// 是否启用「静音/幻觉碎片」过滤（默认否，见 --filter-fragments）
    pub filter_fragments: bool,
}

impl Config {
    pub fn from_args(args: &Args, probe: &RuntimeProbe) -> Result<Self> {
        if args.device == DevicePref::Cuda && args.cuda_libs.is_none() {
            bail!("--device cuda 需要同时用 --cuda-libs 指定 CUDA/cuDNN 运行库目录");
        }
        if !args.greedy && args.temperature <= 0.0 {
            bail!("--temperature 必须 > 0（当前 {}）；想要确定性输出请用 --greedy", args.temperature);
        }
        let (lm, mmproj) = resolve_model(&args.model, &args.mmproj,
                                         args.model_dir.as_deref(), probe.exe_dir())
            .context("定位 S2TT GGUF 模型失败")?;
        info!("LM: {:?} | mmproj: {:?}", lm, mmproj);
        if args.context.trim().is_empty() {
            warn!("--context 为空：转写源语言模式（不做翻译定向）");
        }
        Ok(Self {
            lm_gguf: lm,
            mmproj_gguf: mmproj,
            context: args.context.clone(),
            device: args.device,
            cuda_libs: args.cuda_libs.clone(),
            threads: args.threads.max(1),
            gpu_layers: args.gpu_layers,
            max_new_tokens: args.max_new_tokens.max(16),
            greedy: args.greedy,
            temperature: args.temperature,
            top_p: args.top_p.clamp(0.0, 1.0).max(if args.top_p > 0.0 { 1e-6 } else { 0.0 }),
            top_k: args.top_k,
            seed: if args.seed == 0 {
                // 0 = 每次运行随机取种（进程内取一次，跨段一致，便于对照单段复现）
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| (d.subsec_nanos() ^ (d.as_secs() as u32)) | 1)
                    .unwrap_or(42)
            } else {
                args.seed
            },
            vad_threshold: args.vad_threshold,
            vad_min_silence: args.vad_min_silence,
            vad_max_seg_secs: args.vad_max_seg_secs,
            buffer_bytes: args.buffer_mb.max(1) * 1_000_000,
            max_line_width: args.max_line_width,
            max_cue_secs: args.max_cue_secs,
            output_dir: args.output_dir.clone(),
            raw_srt: args.raw_srt,
            filter_fragments: args.filter_fragments,
        })
    }

    pub fn output_dir_for(&self, input: &Path) -> PathBuf {
        self.output_dir
            .clone()
            .or_else(|| input.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// 最终字幕路径：<output_dir>/<stem>.srt
    pub fn output_srt_path(&self, input: &Path) -> PathBuf {
        self.output_dir_for(input).join(format!("{}.srt", stem_of(input)))
    }

    /// 排版前的原始直出字幕：<output_dir>/<stem>.raw.srt（仅 `--raw-srt` 时写出）
    pub fn raw_srt_path(&self, input: &Path) -> PathBuf {
        self.output_dir_for(input).join(format!("{}.raw.srt", stem_of(input)))
    }
}

fn stem_of(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into())
}

/// 目录内 LM 候选：model.gguf 优先，否则扫描非 mmproj 前缀的 .gguf。
/// Ok(None)=目录不存在或无候选；Err=多候选歧义（值得上报而不是静默跳过）。
fn lm_in(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    let named = dir.join("model.gguf");
    if named.is_file() {
        return Ok(Some(named));
    }
    let found = collect_gguf(dir, false)?;
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.into_iter().next().unwrap())),
        _ => bail!("{:?} 里有多个候选 LM GGUF（{:?}），请用 --model 显式指定", dir,
                   found.iter().map(|p| p.file_name().unwrap_or_default()).collect::<Vec<_>>()),
    }
}

/// 目录内 mmproj 候选：mmproj.gguf 优先，否则扫描 mmproj 前缀的 .gguf。
fn mm_in(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    let named = dir.join("mmproj.gguf");
    if named.is_file() {
        return Ok(Some(named));
    }
    let found = collect_gguf(dir, true)?;
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.into_iter().next().unwrap())),
        _ => bail!("{:?} 里有多个 mmproj 候选，请用 --mmproj 显式指定", dir),
    }
}

fn collect_gguf(dir: &Path, want_mmproj: bool) -> Result<Vec<PathBuf>> {
    let mut found: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("读取模型目录 {:?}", dir))? {
        let p = e?.path();
        let is_gguf = p.extension().map(|x| x == "gguf").unwrap_or(false);
        let is_mm = p.file_name().map(|n| n.to_string_lossy().starts_with("mmproj")).unwrap_or(false);
        if is_gguf && is_mm == want_mmproj {
            found.push(p);
        }
    }
    found.sort();
    Ok(found)
}

/// 单目录扫描：LM 与 mmproj 都唯一命中才算完整（半套的目录跳过，继续下一候选）。
fn scan_dir(dir: &Path) -> Result<Option<(PathBuf, PathBuf)>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    Ok(match (lm_in(dir)?, mm_in(dir)?) {
        (Some(lm), Some(mm)) => Some((lm, mm)),
        _ => None,
    })
}

/// 模型定位（v0.6 拍平布局）：
/// * 显式 --model/--mmproj：直接用；只给其一时，另一半在对方所在目录找；
/// * 显式 --asr-model-dir DIR：DIR（相对 CWD）→ exe_dir/DIR（拖拽启动兼容）；
/// * 都不给（默认）：按序探测，第一个「LM+mmproj 齐」的目录胜出：
///     1. exe_dir/models      —— 拍平布局：exe 差一级 models/（推荐摆放）
///     2. exe_dir             —— 拍平布局：与 exe 同级
///     3. ./models            —— CWD 差一级（开发/命令行场景）
///     4. ./models/qwen3-asr-s2tt   —— v0.5 旧布局兼容（CWD）
///     5. exe_dir/models/qwen3-asr-s2tt —— v0.5 旧布局兼容（exe 目录）
fn resolve_model(explicit_lm: &Option<PathBuf>, explicit_mm: &Option<PathBuf>,
                 dir: Option<&Path>, exe_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    // ---- 显式文件路径优先 ----
    if let Some(p) = explicit_lm {
        if !p.is_file() { bail!("--model 指定的文件不存在: {:?}", p); }
    }
    if let Some(p) = explicit_mm {
        if !p.is_file() { bail!("--mmproj 指定的文件不存在: {:?}", p); }
    }
    if let Some(lm) = explicit_lm {
        let mm = match explicit_mm {
            Some(m) => m.clone(),
            None => {
                let parent = lm.parent().map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
                mm_in(&parent)?.ok_or_else(|| anyhow::anyhow!(
                    "--model 所在目录 {:?} 里找不到 mmproj GGUF（mmproj.gguf 或 mmproj*.gguf），\
                     请用 --mmproj 显式指定", parent))?
            }
        };
        return Ok((lm.clone(), mm));
    }
    if let Some(mm) = explicit_mm {
        let parent = mm.parent().map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let lm = lm_in(&parent)?.ok_or_else(|| anyhow::anyhow!(
            "--mmproj 所在目录 {:?} 里找不到 LM GGUF，请用 --model 显式指定", parent))?;
        return Ok((lm, mm.clone()));
    }

    // ---- 目录候选（显式 DIR 或默认拍平探测序）----
    let candidates: Vec<PathBuf> = match dir {
        Some(d) => vec![d.to_path_buf(), exe_dir.join(d)],
        None => vec![
            exe_dir.join("models"),
            exe_dir.to_path_buf(),
            PathBuf::from("models"),
            PathBuf::from("models/qwen3-asr-s2tt"),       // v0.5 旧布局兼容
            exe_dir.join("models").join("qwen3-asr-s2tt"), // v0.5 旧布局兼容
        ],
    };
    let mut first_err: Option<anyhow::Error> = None;
    for c in &candidates {
        match scan_dir(c) {
            Ok(Some(pair)) => return Ok(pair),
            Ok(None) => {}
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    bail!("找不到 S2TT GGUF 模型。已探测: {:?}。\
           请把 LM GGUF 与 mmproj*.gguf 放到可执行文件同级或其 models/ 子目录\
           （文件名任意，mmproj 前缀自动识别），或用 --asr-model-dir / --model / --mmproj 显式指定。",
          candidates.iter().filter(|d| d.is_dir()).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[test]
    fn resolve_named_and_scanned() {
        let tmp = std::env::temp_dir().join(format!("qsa-cfg-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // 显式目录：默认命名
        let d1 = tmp.join("named");
        std::fs::create_dir_all(&d1).unwrap();
        touch(&d1, "model.gguf");
        touch(&d1, "mmproj.gguf");
        let (lm, mm) = resolve_model(&None, &None, Some(&d1), &tmp).unwrap();
        assert_eq!(lm.file_name().unwrap(), "model.gguf");
        assert_eq!(mm.file_name().unwrap(), "mmproj.gguf");

        // 显式目录：扫描命名（E2 产物式）
        let d2 = tmp.join("scanned");
        std::fs::create_dir_all(&d2).unwrap();
        touch(&d2, "s2tt-Q4_K_M.gguf");
        touch(&d2, "mmproj-s2tt-q8.gguf");
        let (lm, mm) = resolve_model(&None, &None, Some(&d2), &tmp).unwrap();
        assert_eq!(lm.file_name().unwrap(), "s2tt-Q4_K_M.gguf");
        assert_eq!(mm.file_name().unwrap(), "mmproj-s2tt-q8.gguf");

        // 多候选报错（显式目录的歧义要上报，不能静默跳过）
        touch(&d2, "another.gguf");
        assert!(resolve_model(&None, &None, Some(&d2), &tmp).is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// v0.6 拍平布局发现：exe 同级 models/ → exe 同级 → 旧版 qwen3-asr-s2tt 兼容。
    #[test]
    fn flattened_discovery_order() {
        let tmp = std::env::temp_dir().join(format!("qsa-flat-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);

        // 情形 A：exe_dir/models/（推荐摆放）——优先于 exe_dir 同级
        let exe_a = tmp.join("a");
        std::fs::create_dir_all(exe_a.join("models")).unwrap();
        touch(&exe_a, "stray.gguf"); // 同级散落一个不完整候选：不应命中（缺 mmproj）
        touch(&exe_a.join("models"), "s2tt.gguf");
        touch(&exe_a.join("models"), "mmproj-s2tt.gguf");
        let (lm, _) = resolve_model(&None, &None, None, &exe_a).unwrap();
        assert_eq!(lm.file_name().unwrap(), "s2tt.gguf");

        // 情形 B：exe 同级（无 models/ 子目录）
        let exe_b = tmp.join("b");
        std::fs::create_dir_all(&exe_b).unwrap();
        touch(&exe_b, "model.gguf");
        touch(&exe_b, "mmproj.gguf");
        let (lm, _) = resolve_model(&None, &None, None, &exe_b).unwrap();
        assert_eq!(lm.parent().unwrap(), exe_b);

        // 情形 C：v0.5 旧布局 exe_dir/models/qwen3-asr-s2tt 兼容
        let exe_c = tmp.join("c");
        std::fs::create_dir_all(exe_c.join("models/qwen3-asr-s2tt")).unwrap();
        touch(&exe_c.join("models/qwen3-asr-s2tt"), "lm.gguf");
        touch(&exe_c.join("models/qwen3-asr-s2tt"), "mmproj-x.gguf");
        let (lm, _) = resolve_model(&None, &None, None, &exe_c).unwrap();
        assert_eq!(lm.file_name().unwrap(), "lm.gguf");

        // 情形 D：什么都没有 → 明确报错并列出探测位置
        let exe_d = tmp.join("d");
        std::fs::create_dir_all(&exe_d).unwrap();
        assert!(resolve_model(&None, &None, None, &exe_d).is_err());

        // 情形 E：显式 --model 时 mmproj 从其所在目录补齐
        let lm_p = exe_b.join("model.gguf");
        let (_, mm) = resolve_model(&Some(lm_p), &None, None, &tmp).unwrap();
        assert_eq!(mm.file_name().unwrap(), "mmproj.gguf");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn output_paths() {
        let args = Args {
            files: vec![],
            model_dir: None,
            model: None,
            mmproj: None,
            context: String::new(),
            cuda_libs: None,
            device: DevicePref::Cpu,
            threads: 1,
            gpu_layers: 0,
            max_new_tokens: 128,
            vad_threshold: 0.5,
            vad_min_silence: 0.5,
            vad_max_seg_secs: 60.0,
            buffer_mb: 50,
            max_line_width: 44,
            max_cue_secs: 15.0,
            output_dir: Some(PathBuf::from("out")),
            raw_srt: false,
            log_file: None,
            verbose: false,
            no_pause: true,
            filter_fragments: false,
            greedy: false,
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            seed: 42,
            gguf_selftest: vec![],
        };
        let probe = RuntimeProbe::new();
        let cfg = Config::from_args(&args, &probe).unwrap_or_else(|_| {
            // 模型不存在时 from_args 会报错；此测试只关心路径拼装，手工构造
            Config {
                lm_gguf: "x".into(), mmproj_gguf: "y".into(), context: String::new(),
                device: DevicePref::Cpu, cuda_libs: None, threads: 1, gpu_layers: 0,
                max_new_tokens: 128, vad_threshold: 0.5, vad_min_silence: 0.5,
                vad_max_seg_secs: 60.0, buffer_bytes: 50_000_000,
                max_line_width: 44, max_cue_secs: 15.0,
                output_dir: Some(PathBuf::from("out")), raw_srt: false,
                filter_fragments: false,
                greedy: false, temperature: 0.7, top_p: 0.8, top_k: 20, seed: 42,
            }
        });
        let input = Path::new("media/video.mp4");
        assert_eq!(cfg.output_srt_path(input), PathBuf::from("out/video.srt"));
        assert_eq!(cfg.raw_srt_path(input), PathBuf::from("out/video.raw.srt"));
    }
}
