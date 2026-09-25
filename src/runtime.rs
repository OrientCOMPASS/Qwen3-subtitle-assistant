//! 外置 DLL 运行时探测与加载。
//!
//! 本程序自身不静态包含任何推理后端：
//! - LLM 侧：exe 导入 `llama.dll` / `ggml.dll` / `ggml-base.dll`；
//!   CUDA 后端 `ggml-cuda.dll` 是 ggml 在 `llama_backend_init()` 时从
//!   exe 所在目录动态扫描加载的“后端模块”。
//! - ASR 侧：exe 导入 `sherpa-onnx-c-api.dll` / `onnxruntime.dll`；
//!   CUDA ExecutionProvider（`onnxruntime_providers_cuda.dll`）由 ONNX Runtime
//!   在创建识别器时按需加载。
//!
//! 因此同一个 exe：
//! - 旁边放 CPU 版 DLL 集合  -> 纯 CPU 推理；
//! - 旁边放 CUDA 版 DLL 集合 -> CUDA 加速推理（LLM 与 ASR 各自独立生效）。
//!
//! 本模块在启动时主动 `LoadLibrary`/`dlopen` 探测这些可选 DLL 是否真的可加载
//! （可加载 == 其全部依赖 cudart/cublas/cudnn/显卡驱动都齐了），据此决定
//! 传给 sherpa-onnx 的 provider、llama.cpp 的 n_gpu_layers，并在缺件时给出
//! 明确的日志提示，而不是让用户面对晦涩的初始化崩溃。

use log::{info, warn};
use std::path::{Path, PathBuf};

/// 用户通过 `--device` 指定的设备偏好。
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DevicePref {
    /// 自动探测（默认）：外置 CUDA DLL 可加载则用 CUDA，否则 CPU
    Auto,
    /// 强制 CPU（即使 CUDA DLL 存在也不使用）
    Cpu,
    /// 强制 CUDA（探测失败时报错退出，便于排查环境问题）
    Cuda,
}

// clap 的 default_value_t 需要 Display；与 ValueEnum 派生的取值保持一致
impl std::fmt::Display for DevicePref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DevicePref::Auto => "auto",
            DevicePref::Cpu => "cpu",
            DevicePref::Cuda => "cuda",
        };
        f.write_str(s)
    }
}

/// LLM 侧 CUDA 后端模块文件名（ggml 动态后端）。
#[cfg(windows)]
const LLM_CUDA_DLL: &str = "ggml-cuda.dll";
#[cfg(not(windows))]
const LLM_CUDA_DLL: &str = "libggml-cuda.so";
/// ASR 侧 ONNX Runtime CUDA ExecutionProvider 文件名。
#[cfg(windows)]
const ASR_CUDA_DLL: &str = "onnxruntime_providers_cuda.dll";
#[cfg(not(windows))]
const ASR_CUDA_DLL: &str = "libonnxruntime_providers_cuda.so";

/// 运行环境探测结果。
pub struct RuntimeInfo {
    #[allow(dead_code)] // 诊断信息，供日志/未来扩展使用
    pub exe_dir: PathBuf,
    #[allow(dead_code)]
    pub extra_lib_dirs: Vec<PathBuf>,
    /// ggml-cuda.dll 可成功加载（LLM 可走 CUDA）
    pub llm_cuda: bool,
    /// onnxruntime_providers_cuda.dll 可成功加载（ASR 可走 CUDA）
    pub asr_cuda: bool,
    pub device_pref: DevicePref,
    /// 探测成功且被我们主动加载的句柄，进程生命周期内保持不释放。
    /// 故意不实现 Drop：CUDA 相关 DLL 在初始化后被卸载可能导致崩溃。
    _handles: Vec<dll::Handle>,
}

impl RuntimeInfo {
    /// 探测运行环境。`extra_lib_dirs` 会（在 Windows 上）通过
    /// `AddDllDirectory` 加入进程 DLL 搜索路径，使 cudart/cublas/cudnn 等
    /// 依赖可以从 exe 目录以外的位置解析。
    pub fn detect(device_pref: DevicePref, extra_lib_dirs: &[PathBuf]) -> anyhow::Result<Self> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));

        info!(
            "运行时探测: exe 目录 = {:?}, 设备偏好 = {:?}, 附加 DLL 目录 = {:?}",
            exe_dir, device_pref, extra_lib_dirs
        );

        dll::add_search_dirs(extra_lib_dirs);

        let mut handles = Vec::new();
        let (llm_cuda, asr_cuda) = if device_pref == DevicePref::Cpu {
            info!("已按 --device cpu 强制使用 CPU 推理，跳过 CUDA DLL 探测。");
            (false, false)
        } else {
            let search: Vec<&Path> = std::iter::once(exe_dir.as_path())
                .chain(extra_lib_dirs.iter().map(|p| p.as_path()))
                .collect();

            let llm = probe(LLM_CUDA_DLL, &search, &mut handles);
            let asr = probe(ASR_CUDA_DLL, &search, &mut handles);

            if device_pref == DevicePref::Cuda && !(llm || asr) {
                // 显式要求 CUDA 但两个关键 DLL 都加载失败：报错退出，帮助用户排查。
                anyhow::bail!(
                    "--device cuda 已指定，但 {} 与 {} 均无法加载。\n\
                     请确认：\n  \
                     1. 使用的是 CUDA 版发布包（DLL 与 exe 在同一目录）；\n  \
                     2. 已安装 NVIDIA 显卡驱动（CUDA 12.4 需驱动 >= 551.61）；\n  \
                     3. 若 cudart/cublas/cudnn 放在其他目录，请通过 --lib-dir 指定。",
                    LLM_CUDA_DLL, ASR_CUDA_DLL
                );
            }
            (llm, asr)
        };

        let info = Self {
            exe_dir,
            extra_lib_dirs: extra_lib_dirs.to_vec(),
            llm_cuda,
            asr_cuda,
            device_pref,
            _handles: handles,
        };
        info.log_summary();
        Ok(info)
    }

    /// exe 所在目录（用于解析模型/提示词等相对路径资源：拖拽启动时 CWD 不可控）。
    pub fn exe_dir(&self) -> &Path {
        &self.exe_dir
    }

    /// 传给 sherpa-onnx 的 ASR provider 字符串。
    pub fn asr_provider(&self) -> &'static str {
        if self.asr_cuda {
            "cuda"
        } else {
            "cpu"
        }
    }

    /// LLM 默认 GPU offload 层数：探测到 CUDA 后端则尽量全部上卡，否则纯 CPU。
    pub fn default_gpu_layers(&self) -> u32 {
        if self.llm_cuda {
            // 足够大的值，llama.cpp 会按模型实际层数截断
            999
        } else {
            0
        }
    }

    fn log_summary(&self) {
        info!("---------------- 推理后端探测结果 ----------------");
        info!(
            "  LLM (llama.cpp): {}",
            if self.llm_cuda {
                format!("CUDA ✔（已从外置 {} 加载）", LLM_CUDA_DLL)
            } else {
                "CPU（未检测到可用的外置 ggml-cuda.dll）".to_string()
            }
        );
        info!(
            "  ASR (sherpa-onnx/onnxruntime): {}",
            if self.asr_cuda {
                format!("CUDA ✔（已从外置 {} 加载）", ASR_CUDA_DLL)
            } else {
                "CPU（未检测到可用的外置 onnxruntime_providers_cuda.dll）".to_string()
            }
        );
        info!("--------------------------------------------------");

        if !self.llm_cuda && self.device_pref != DevicePref::Cpu {
            warn!(
                "LLM 将运行在 CPU 上。如需 CUDA 加速，请使用 cuda12 发布包，或将 \
                 {}、cudart64_12.dll、cublas64_12.dll、cublasLt64_12.dll \
                 放到 exe 同目录（亦可用 --lib-dir 指定目录）。",
                LLM_CUDA_DLL
            );
        }
        if !self.asr_cuda && self.device_pref != DevicePref::Cpu {
            warn!(
                "ASR 将运行在 CPU 上。如需 CUDA 加速，请确保 {}、\
                 onnxruntime_providers_shared.dll、cudart64_12.dll、cublas64_12.dll、\
                 cublasLt64_12.dll、cudnn64_9.dll（及其子库）位于 exe 同目录或 --lib-dir 目录。",
                ASR_CUDA_DLL
            );
        }
    }
}

/// 在候选目录中按文件名探测 DLL，随后回退到系统搜索路径。
/// 成功加载的句柄压入 `handles` 并保持到进程结束。
fn probe(name: &str, search_dirs: &[&Path], handles: &mut Vec<dll::Handle>) -> bool {
    // 1) 优先从 exe 目录 / 用户指定目录按完整路径加载
    for dir in search_dirs {
        let full = dir.join(name);
        if full.is_file() {
            match dll::load_from_path(&full) {
                Ok(h) => {
                    info!("已加载外置 DLL: {:?}", full);
                    handles.push(h);
                    return true;
                }
                Err(code) => {
                    warn!(
                        "发现 {:?} 但加载失败（错误码 {}）。通常意味着缺少依赖 DLL \
                         （cudart/cublas/cudnn）或 NVIDIA 驱动不可用，将回退 CPU。",
                        full, code
                    );
                    return false;
                }
            }
        }
    }
    // 2) 回退：让系统按标准搜索路径解析（PATH、AddDllDirectory 添加的目录等）
    match dll::load_by_name(name) {
        Ok(h) => {
            info!("已按系统搜索路径加载外置 DLL: {}", name);
            handles.push(h);
            true
        }
        Err(code) => {
            info!("未找到可加载的 {}（错误码 {}），对应子系统使用 CPU。", name, code);
            false
        }
    }
}

/// Windows 控制台切换为 UTF-8 代码页，避免中文日志乱码；其他平台无操作。
pub fn enable_utf8_console() {
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn SetConsoleOutputCP(wCodePageID: u32) -> i32;
            fn SetConsoleCP(wCodePageID: u32) -> i32;
        }
        const CP_UTF8: u32 = 65001;
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
}

/// 处理失败时在控制台等待回车，避免拖拽/双击启动时窗口一闪而过、看不到错误。
///
/// 以下情况直接返回，绝不阻塞：stdin 不是终端（重定向/管道/CI）、
/// 环境变量 `CI` 存在、或 Windows 下没有附加控制台。
pub fn pause_on_exit() {
    use std::io::{IsTerminal, Write};

    if std::env::var_os("CI").is_some() {
        return;
    }
    if !std::io::stdin().is_terminal() {
        return;
    }
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetConsoleWindow() -> *mut std::ffi::c_void;
        }
        if GetConsoleWindow().is_null() {
            return;
        }
    }
    let mut out = std::io::stderr();
    let _ = writeln!(out, "\n处理未全部成功。按回车键退出（下次可用 --no-pause 跳过）...");
    let _ = out.flush();
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}

// ============================================================================
// 平台相关的原始动态库加载封装（无第三方依赖）
// ============================================================================
mod dll {
    use std::path::Path;

    /// 已加载的动态库句柄（进程生命周期内不释放）。
    pub struct Handle {
        _raw: *mut std::ffi::c_void,
    }
    // 句柄只是被持有、不跨线程转移使用
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    #[cfg(windows)]
    mod imp {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::path::Path;

        const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;
        const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
        const LOAD_LIBRARY_SEARCH_APPLICATION_DIR: u32 = 0x0000_0200;
        const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x0000_0400;
        const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryExW(
                lp_libFileName: *const u16,
                hFile: *mut std::ffi::c_void,
                dwFlags: u32,
            ) -> *mut std::ffi::c_void;
            fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
            fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
            fn GetLastError() -> u32;
        }

        fn wide(path: &Path) -> Vec<u16> {
            OsStr::new(path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect()
        }

        pub fn load_from_path(path: &Path) -> Result<*mut std::ffi::c_void, u32> {
            let w = wide(path);
            // 按完整路径加载，且其依赖 DLL 也从该文件所在目录解析
            let h =
                unsafe { LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
            if h.is_null() {
                Err(unsafe { GetLastError() })
            } else {
                Ok(h)
            }
        }

        pub fn load_by_name(name: &str) -> Result<*mut std::ffi::c_void, u32> {
            let w = wide(Path::new(name));
            let flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                | LOAD_LIBRARY_SEARCH_APPLICATION_DIR
                | LOAD_LIBRARY_SEARCH_USER_DIRS
                | LOAD_LIBRARY_SEARCH_SYSTEM32;
            let h = unsafe { LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), flags) };
            if h.is_null() {
                Err(unsafe { GetLastError() })
            } else {
                Ok(h)
            }
        }

        pub fn add_search_dir(dir: &Path) {
            let w = wide(dir);
            unsafe {
                // 让后续所有 LoadLibrary 默认搜索：exe 目录 + System32 + AddDllDirectory 添加的目录
                SetDefaultDllDirectories(
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                        | LOAD_LIBRARY_SEARCH_APPLICATION_DIR
                        | LOAD_LIBRARY_SEARCH_USER_DIRS
                        | LOAD_LIBRARY_SEARCH_SYSTEM32,
                );
                AddDllDirectory(w.as_ptr());
            }
        }
    }

    #[cfg(not(windows))]
    mod imp {
        use std::ffi::CString;
        use std::os::raw::{c_char, c_int, c_void};
        use std::path::Path;

        const RTLD_NOW: c_int = 2;
        const RTLD_GLOBAL: c_int = 0x00100; // Linux 取值；仅尽力而为

        extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        }

        fn open(name: &str) -> Result<*mut c_void, u32> {
            let c = CString::new(name).map_err(|_| 1u32)?;
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if h.is_null() {
                Err(1)
            } else {
                Ok(h)
            }
        }

        pub fn load_from_path(path: &Path) -> Result<*mut c_void, u32> {
            open(&path.to_string_lossy())
        }
        pub fn load_by_name(name: &str) -> Result<*mut c_void, u32> {
            open(name)
        }
        pub fn add_search_dir(_dir: &Path) {
            // POSIX 平台依赖 LD_LIBRARY_PATH / RUNPATH，运行中无法便捷追加
        }
    }

    pub fn load_from_path(path: &Path) -> Result<Handle, u32> {
        imp::load_from_path(path).map(|h| Handle { _raw: h })
    }

    pub fn load_by_name(name: &str) -> Result<Handle, u32> {
        imp::load_by_name(name).map(|h| Handle { _raw: h })
    }

    pub fn add_search_dirs(dirs: &[std::path::PathBuf]) {
        for d in dirs {
            if d.is_dir() {
                imp::add_search_dir(d);
                log::info!("已将 {:?} 加入 DLL 搜索路径", d);
            } else {
                log::warn!("--lib-dir 指定的目录不存在，已忽略: {:?}", d);
            }
        }
    }
}
