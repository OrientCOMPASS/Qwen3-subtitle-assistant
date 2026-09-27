//! 自定义日志：双通道（终端精简 + 文件全量），取代 env_logger。
//!
//! 设计目标（默认终端不刷屏）：
//! * 终端默认 info 级——但代码里的 info 已收敛为「进度/结果级」最小集合
//!   （模式行、设备行、每文件一行开始/写出、总结）；逐段 ASR/VAD 明细、
//!   llama.cpp/ggml 原生日志全部是 debug 级，默认不可见；
//! * `--verbose`（或 `RUST_LOG=debug`）：终端放开 debug，显示全部诊断；
//! * `--log-file`：文件通道**恒为 debug 级**——终端保持干净的同时，
//!   日志文件里永远有完整诊断（含 llama.cpp 原生加载日志），排障不愁没料；
//! * llama.cpp/ggml 的原生 C 日志经 `llama_log_set`/`ggml_log_set` 桥接进本
//!   门面（见 gguf_asr.rs 的 native_log_hook），不再直写 stderr。

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

pub struct DualLogger {
    console: LevelFilter,
    has_file: bool,
    file: Mutex<Option<std::fs::File>>,
}

impl DualLogger {
    /// 安装全局 logger。`verbose` = `--verbose`；`log_file` = `--log-file`。
    /// 之后可再按安装结果打 info 日志。
    ///
    /// 终端级别优先级：`--verbose`（显式命令行开关，恒 debug）> `RUST_LOG`
    /// 环境变量 > 默认 info。文件通道恒 debug，不受影响。
    pub fn init(verbose: bool, log_file: Option<&Path>) {
        let console = if verbose {
            LevelFilter::Debug
        } else {
            env_level().unwrap_or(LevelFilter::Info)
        };

        let file = log_file.and_then(|p| {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            match OpenOptions::new().create(true).append(true).open(p) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("警告: 无法写日志文件 {:?}（{e}），仅输出到控制台", p);
                    None
                }
            }
        });
        let has_file = file.is_some();
        let file_path = log_file.map(Path::to_path_buf);

        let logger: &'static Self =
            Box::leak(Box::new(Self { console, has_file, file: Mutex::new(file) }));
        // 全局放行级别取两通道较宽者：文件通道恒 debug
        let max_level = if has_file { LevelFilter::Debug } else { console };
        if log::set_logger(logger).is_ok() {
            log::set_max_level(max_level);
        }
        if has_file {
            log::info!("日志文件（含全部诊断明细）: {:?}", file_path.unwrap());
        }
    }
}

/// `RUST_LOG` 覆盖终端级别。只做「级别名」级别的简单解析
/// （`off|error|warn|info|debug|trace`，也容忍 `target=level` 形式的最后一段），
/// 不支持完整 env_logger 过滤器语法——产品场景用不上，避免引入解析依赖。
fn env_level() -> Option<LevelFilter> {
    let v = std::env::var("RUST_LOG").ok()?;
    let t = v.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    let last = t.rsplit(',').next().unwrap_or(&t);
    let lvl = last.split('=').last().unwrap_or(last).trim();
    match lvl {
        "off" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" | "trace" => Some(LevelFilter::Debug),
        _ => None,
    }
}

impl Log for DualLogger {
    fn enabled(&self, m: &Metadata) -> bool {
        m.level() <= self.console || (self.has_file && m.level() <= LevelFilter::Debug)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let msg = record.args().to_string();

        // ---- 终端通道：info 不加前缀（干净），warn/error 加符号前缀 ----
        if record.level() <= self.console {
            match record.level() {
                Level::Error => eprintln!("✘ 错误: {msg}"),
                Level::Warn => eprintln!("⚠ 警告: {msg}"),
                _ => eprintln!("{msg}"),
            }
        }

        // ---- 文件通道：时间戳 + 级别 + target，恒 debug 级全量 ----
        if self.has_file {
            if let Ok(mut guard) = self.file.lock() {
                if let Some(f) = guard.as_mut() {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| format!("{:.3}", d.as_secs_f64()))
                        .unwrap_or_else(|_| "0".into());
                    let _ = writeln!(f, "{ts} [{}] {}: {msg}", record.level(), record.target());
                    let _ = f.flush();
                }
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut guard) = self.file.lock() {
            if let Some(f) = guard.as_mut() {
                let _ = f.flush();
            }
        }
    }
}
