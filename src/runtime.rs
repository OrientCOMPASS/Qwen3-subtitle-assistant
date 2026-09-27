//! 运行时辅助：设备偏好、Windows 控制台/退出暂停、DLL 搜索路径封装。
//!
//! E3 单模型化后，后端探测全部交给 ggml 运行时（`gguf_asr.rs` 里枚举
//! ggml_backend_dev 并打印/选择设备）；本模块只保留与平台交互的最小工具。

use log::debug;
use std::path::{Path, PathBuf};

/// 启动期路径信息（exe 目录）。单模型化后不再有 DLL 探测——后端枚举与设备
/// 选择由 ggml 运行时负责（见 gguf_asr.rs）。
pub struct RuntimeProbe {
    exe_dir: PathBuf,
}

impl RuntimeProbe {
    pub fn new() -> Self {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        debug!("exe 目录: {:?}", exe_dir);
        Self { exe_dir }
    }

    pub fn exe_dir(&self) -> &Path {
        &self.exe_dir
    }
}

impl Default for RuntimeProbe {
    fn default() -> Self {
        Self::new()
    }
}

/// 用户通过 `--device` 指定的设备偏好。
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DevicePref {
    /// 自动（默认）：CUDA（仅当 `--cuda-libs` 指定且加载成功）→ Vulkan（索引最大
    /// 的 GPU）→ CPU
    Auto,
    /// 强制 CPU（即使有可用 GPU 也不上卡）
    Cpu,
    /// 强制 CUDA（必须同时给出 `--cuda-libs`；失败时报错退出便于排查）
    Cuda,
}

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

/// Windows 控制台切 UTF-8，避免中文日志乱码。
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
// 平台相关的 DLL 搜索路径封装（无第三方依赖）
// ============================================================================
// 历史注：这里曾有一整套 LoadLibraryExW/dlopen 的「按路径/按名加载动态库」封装，
// 服务于旧版 sherpa/ORT DLL 探测。E3 单模型化 + v0.5 静态链接后全部成为死代码
// （CI 警告清单常客），只保留仍在用的 add_search_dirs——--cuda-libs 场景下把
// 用户目录加入进程 DLL 搜索路径，供 ggml_backend_load_all_from_path 解析依赖链。
pub(crate) mod dll {
    #[cfg(windows)]
    mod imp {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::path::Path;

        const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
        const LOAD_LIBRARY_SEARCH_APPLICATION_DIR: u32 = 0x0000_0200;
        const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x0000_0400;
        const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

        #[link(name = "kernel32")]
        extern "system" {
            fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
            fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
        }

        fn wide(path: &Path) -> Vec<u16> {
            OsStr::new(path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect()
        }

        pub fn add_search_dir(dir: &Path) {
            let w = wide(dir);
            unsafe {
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
        use std::path::Path;

        pub fn add_search_dir(_dir: &Path) {
            // POSIX：依赖 RPATH/LD_LIBRARY_PATH，运行时无法追加搜索路径；
            // ggml_backend_load_all_from_path 已按显式路径加载，够用。
        }
    }

    /// 把目录加入进程 DLL 搜索路径（Windows：AddDllDirectory；其他平台 no-op）。
    pub fn add_search_dirs(dirs: &[std::path::PathBuf]) {
        for d in dirs {
            if d.is_dir() {
                imp::add_search_dir(d);
                log::debug!("已将 {:?} 加入 DLL 搜索路径", d);
            } else {
                log::warn!("指定的库目录不存在，已忽略: {:?}", d);
            }
        }
    }
}
