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

    // ---- VAD ----
    pub vad_threshold: f32,
    pub vad_min_silence: f32,
    pub vad_buffer_secs: f32,

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
        let (lm, mmproj) = resolve_model(&args.model, &args.mmproj, &args.asr_model_dir, probe.exe_dir())
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
            max_new_tokens: args.asr_max_new_tokens.max(16),
            vad_threshold: args.vad_threshold,
            vad_min_silence: args.vad_min_silence,
            vad_buffer_secs: args.vad_buffer_secs,
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

/// 模型定位：显式 --model/--mmproj 优先；否则在 asr_model_dir（CWD→exe 目录）里
/// 找 model.gguf/mmproj.gguf，再退化为目录扫描（非 mmproj 前缀的 .gguf = LM）。
fn resolve_model(explicit_lm: &Option<PathBuf>, explicit_mm: &Option<PathBuf>,
                 dir: &Path, exe_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let dir_candidates = [dir.to_path_buf(), exe_dir.join(dir)];
    let base_dir = if dir.is_dir() {
        Some(dir.to_path_buf())
    } else {
        dir_candidates.iter().find(|d| d.is_dir()).cloned()
    };

    let lm = match explicit_lm {
        Some(p) if p.is_file() => p.clone(),
        Some(p) => bail!("--model 指定的文件不存在: {:?}", p),
        None => find_lm(&base_dir, dir)?,
    };
    let mm = match explicit_mm {
        Some(p) if p.is_file() => p.clone(),
        Some(p) => bail!("--mmproj 指定的文件不存在: {:?}", p),
        None => find_mmproj(&base_dir, &lm, dir)?,
    };
    Ok((lm, mm))
}

fn find_lm(base: &Option<PathBuf>, orig: &Path) -> Result<PathBuf> {
    let dir = base.as_ref().ok_or_else(|| anyhow::anyhow!(
        "模型目录不存在: {:?}（也不在 exe 目录下）。请下载 S2TT GGUF 模型包，或用 \
         --asr-model-dir / --model / --mmproj 指定。", orig))?;
    let named = dir.join("model.gguf");
    if named.is_file() {
        return Ok(named);
    }
    let mut found: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("读取模型目录 {:?}", dir))? {
        let p = e?.path();
        let is_gguf = p.extension().map(|x| x == "gguf").unwrap_or(false);
        let is_mm = p.file_name().map(|n| n.to_string_lossy().starts_with("mmproj")).unwrap_or(false);
        if is_gguf && !is_mm {
            found.push(p);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("{:?} 里找不到 LM GGUF（model.gguf 或任一非 mmproj 前缀的 .gguf）", dir),
        _ => bail!("{:?} 里有多个候选 LM GGUF（{:?}），请用 --model 显式指定", dir,
                   found.iter().map(|p| p.file_name().unwrap_or_default()).collect::<Vec<_>>()),
    }
}

fn find_mmproj(base: &Option<PathBuf>, lm: &Path, _orig: &Path) -> Result<PathBuf> {
    let dir = base.clone().or_else(|| lm.parent().map(|p| p.to_path_buf()))
        .ok_or_else(|| anyhow::anyhow!("无法确定 mmproj 搜索目录"))?;
    let named = dir.join("mmproj.gguf");
    if named.is_file() {
        return Ok(named);
    }
    let mut found: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(&dir).with_context(|| format!("读取模型目录 {:?}", dir))? {
        let p = e?.path();
        let is_gguf = p.extension().map(|x| x == "gguf").unwrap_or(false);
        let is_mm = p.file_name().map(|n| n.to_string_lossy().starts_with("mmproj")).unwrap_or(false);
        if is_gguf && is_mm {
            found.push(p);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => bail!("{:?} 里找不到 mmproj GGUF（mmproj.gguf 或 mmproj*.gguf）", dir),
        _ => bail!("{:?} 里有多个 mmproj 候选，请用 --mmproj 显式指定", dir),
    }
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

        // 默认命名
        let d1 = tmp.join("named");
        std::fs::create_dir_all(&d1).unwrap();
        touch(&d1, "model.gguf");
        touch(&d1, "mmproj.gguf");
        let (lm, mm) = resolve_model(&None, &None, &d1, &tmp).unwrap();
        assert_eq!(lm.file_name().unwrap(), "model.gguf");
        assert_eq!(mm.file_name().unwrap(), "mmproj.gguf");

        // 目录扫描（E2 产物式命名）
        let d2 = tmp.join("scanned");
        std::fs::create_dir_all(&d2).unwrap();
        touch(&d2, "s2tt-Q4_K_M.gguf");
        touch(&d2, "mmproj-s2tt-q8.gguf");
        let (lm, mm) = resolve_model(&None, &None, &d2, &tmp).unwrap();
        assert_eq!(lm.file_name().unwrap(), "s2tt-Q4_K_M.gguf");
        assert_eq!(mm.file_name().unwrap(), "mmproj-s2tt-q8.gguf");

        // 多候选报错
        touch(&d2, "another.gguf");
        assert!(resolve_model(&None, &None, &d2, &tmp).is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn output_paths() {
        let args = Args {
            files: vec![],
            asr_model_dir: PathBuf::from("."),
            model: None,
            mmproj: None,
            context: String::new(),
            cuda_libs: None,
            device: DevicePref::Cpu,
            threads: 1,
            gpu_layers: 0,
            asr_max_new_tokens: 128,
            vad_threshold: 0.5,
            vad_min_silence: 0.5,
            vad_buffer_secs: 60.0,
            max_line_width: 44,
            max_cue_secs: 15.0,
            output_dir: Some(PathBuf::from("out")),
            raw_srt: false,
            log_file: None,
            verbose: false,
            no_pause: true,
            filter_fragments: false,
            gguf_selftest: vec![],
        };
        let probe = RuntimeProbe::new();
        let cfg = Config::from_args(&args, &probe).unwrap_or_else(|_| {
            // 模型不存在时 from_args 会报错；此测试只关心路径拼装，手工构造
            Config {
                lm_gguf: "x".into(), mmproj_gguf: "y".into(), context: String::new(),
                device: DevicePref::Cpu, cuda_libs: None, threads: 1, gpu_layers: 0,
                max_new_tokens: 128, vad_threshold: 0.5, vad_min_silence: 0.5,
                vad_buffer_secs: 60.0, max_line_width: 44, max_cue_secs: 15.0,
                output_dir: Some(PathBuf::from("out")), raw_srt: false,
                filter_fragments: false,
            }
        });
        let input = Path::new("media/video.mp4");
        assert_eq!(cfg.output_srt_path(input), PathBuf::from("out/video.srt"));
        assert_eq!(cfg.raw_srt_path(input), PathBuf::from("out/video.raw.srt"));
    }
}
