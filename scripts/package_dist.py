#!/usr/bin/env python3
"""组装 Windows x64 发布包（CPU 版 + CUDA 12 版）。

用法（CI 或本地）:
    python scripts/package_dist.py --exe target/release/subtitle-assistant.exe \
        --version v0.2.0 --out dist --cache .distcache

设计说明
--------
本程序采用「外置 DLL」运行时：exe 只动态链接核心 DLL，CUDA 加速能力完全由
放在 exe 旁边的外置 DLL 决定。因此 CI 只需一次纯 CPU 构建（无需 CUDA
Toolkit），再从各官方发布渠道下载预编译 DLL 组装出两种发行包：

  CPU 包   = exe + llama.cpp 官方 win-cpu DLL + sherpa-onnx CPU DLL
  CUDA 包  = exe + llama.cpp 官方 win-cuda-12.4 DLL（含 ggml-cuda.dll）
             + cudart/cublas/cublasLt（llama.cpp 官方 cudart 包）
             + cuFFT（**按 PE 导入表按需**，见下）
             + sherpa-onnx CUDA 包（onnxruntime CUDA 版 + providers_cuda）
             + cuDNN 9（NVIDIA PyPI wheel）
             + VC++ 运行时（vcruntime140/msvcp140，exe 是 /MD 构建，静态导入它们）

两个新增的"防哑火"机制
--------------------
1. **按需补 cuFFT**：ONNX Runtime 的 CUDA EP 依赖 cufft64_*.dll，缺了它
   `onnxruntime_providers_cuda.dll` 加载失败，程序会静默回退 CPU——用户拿到
   CUDA 包却没有 ASR 加速。这里不再靠人肉记清单，而是**解析已下载的
   onnxruntime_providers_cuda.dll 的导入表**，发现引用 cufft 才去下载对应 wheel。
2. **组装后做依赖闭包检查**（scripts/check_dll_deps.py）：把包内每个 PE 的
   导入项与包内文件对照，缺件立刻打印出来；`--strict-deps` 时直接失败。

版本兼容性依据：
  * llama-cpp-2 v0.1.157 vendored llama.cpp commit 26394b4e，与官方 tag
    b11153 的公开头文件（include/llama.h、ggml/include/*.h）完全一致，
    故官方 DLL 与本仓库构建的 exe 二进制兼容（导入符号一一对应）。
  * 例外：llama-common.dll 必须使用【本仓库 CI 构建产物】而非官方 zip 内的
    版本——llama-cpp-sys-2 的 default features 强制启用 common（下游无法
    关闭），exe 会导入 llama_rs_*/common_* 符号；而 26394b4..b11153 之间
    common/common.h 发生过变更，官方版存在符号漂移风险。
  * sherpa-onnx crate 版本（Cargo.toml 中精确锁定）必须与下面 SHERPA_VER
    一致，保证 sherpa-onnx-c-api.dll / onnxruntime.dll 导出符号匹配。
升级依赖时，请同步更新这里的 LLAMA_TAG / SHERPA_VER / CUDNN_VER / CUFFT_VER，
并跑一次 `python scripts/check_dll_deps.py --dir <包目录> --fail-on-missing`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# ---------------- 版本固定（与 Cargo.toml 联动，升级时同步修改） ----------------
LLAMA_TAG = "b11153"                      # llama.cpp 官方 release tag
SHERPA_VER = "1.13.8"                     # 必须与 Cargo.toml 的 sherpa-onnx 版本一致
CUDNN_PKG = "nvidia-cudnn-cu12"           # cuDNN 9 for CUDA 12（PyPI win_amd64 wheel）
CUDNN_VER = "9.1.1.17"                    # 与 llama.cpp cuda-12.4 包同代，ABI 匹配
CUFFT_PKG = "nvidia-cufft-cu12"           # cuFFT 11.x（ONNX Runtime CUDA EP 依赖）
CUFFT_VER = "11.3.3.83"                   # 提供 cufft64_11.dll；比 12.4 自带版本新，向后兼容

LLAMA_CPU_ZIP = f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"
LLAMA_CUDA_ZIP = f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip"
CUDART_ZIP = f"cudart-llama-bin-win-cuda-12.4-x64.zip"
SHERPA_CPU_TAR = f"sherpa-onnx-v{SHERPA_VER}-win-x64-shared-MT-Release-lib.tar.bz2"
SHERPA_CUDA_TAR = (
    f"sherpa-onnx-v{SHERPA_VER}-cuda-12.x-cudnn-9.x-onnxruntime1.28.2-win-x64-cuda.tar.bz2"
)

GGML_DL_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
SHERPA_DL_BASE = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/v{SHERPA_VER}"

# llama.cpp 官方 zip 中需要进包的文件（排除 llama-cli/server 等工具 impl DLL）
LLAMA_DLL_KEEP = [
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cuda.dll",     # 仅 CUDA 包存在
    "libomp.dll",
]
LLAMA_DLL_PREFIX_KEEP = ["ggml-cpu"]       # ggml-cpu-x64.dll / ggml-cpu-haswell.dll ...

# sherpa CPU 动态库包（lib/ 下）
SHERPA_CPU_DLLS = [
    "sherpa-onnx-c-api.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
]
# sherpa CUDA 包（lib/ 下）
SHERPA_CUDA_DLLS = [
    "sherpa-onnx-c-api.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
]
CUDART_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]

# VC++ 运行时：exe 是 /MD 构建，静态导入 vcruntime140.dll；干净系统可能没有
VCREDIST_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"]
VCREDIST_GLOBS = [
    r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT",
    r"C:\Windows\System32",   # 兜底：runner 系统目录里通常也有
]

THIRD_PARTY_NOTICES = """\
Qwen3 Subtitle Assistant — 第三方组件与许可
==========================================

本发行包内含以下第三方二进制组件，版权归各自权利人所有：

1. llama.cpp                MIT License        https://github.com/ggml-org/llama.cpp
   （llama.dll / llama-common.dll / ggml*.dll / ggml-cpu-*.dll / ggml-cuda.dll / libomp.dll）
2. ONNX Runtime             MIT License        https://github.com/microsoft/onnxruntime
   （onnxruntime.dll / onnxruntime_providers_*.dll）
3. sherpa-onnx              Apache License 2.0 https://github.com/k2-fsa/sherpa-onnx
   （sherpa-onnx-c-api.dll）
4. LLVM OpenMP (libomp)     Apache License 2.0 with LLVM exception
5. NVIDIA cuDNN             NVIDIA cuDNN 软件许可协议（SLA）
   https://docs.nvidia.com/deeplearning/cudnn/latest/cudnn-sla.html
   （cudnn64_9.dll / cudnn_*64_9.dll）
6. NVIDIA CUDA 运行时组件   CUDA Toolkit 许可协议（允许再分发运行时库）
   https://docs.nvidia.com/cuda/eula/index.html
   （cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll / cufft64_*.dll）
7. Microsoft Visual C++ 运行时  可按 Visual Studio 许可再分发
   （vcruntime140.dll / vcruntime140_1.dll / msvcp140.dll）

模型（不包含在本包内，需另行下载）：
  * Qwen3-ASR / Qwen3       Apache License 2.0（Alibaba Qwen 团队）
  * Silero VAD              MIT License

本仓库自身代码采用 Unlicense（见 LICENSE）。
"""


def log(msg: str) -> None:
    print(f"[package] {msg}", flush=True)


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "qwen3-subtitle-assistant-ci"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def download_cached(url: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / url.rsplit("/", 1)[-1]
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"cache hit: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    log(f"downloading {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "qwen3-subtitle-assistant-ci"}),
        timeout=1800,
    ) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, 1024 * 1024)
    tmp.rename(dest)
    log(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_from_zip(zip_path: Path, names_pred, out_dir: Path) -> list[str]:
    """从 zip 中抽取文件名满足谓词（按 basename 判断）的条目。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            if info.is_dir() or not base:
                continue
            if names_pred(base):
                with z.open(info) as src, open(out_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                got.append(base)
    return got


def extract_from_tarbz2(tar_path: Path, names_pred, out_dir: Path, path_filter=None) -> list[str]:
    """从 tar.bz2 中抽取 basename 满足谓词的文件（流式，不解压全部）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    with tarfile.open(tar_path, "r:bz2") as t:
        for member in t:
            if not member.isfile():
                continue
            if path_filter is not None and not path_filter(member.name):
                continue
            base = member.name.rsplit("/", 1)[-1]
            if names_pred(base):
                src = t.extractfile(member)
                if src is None:
                    continue
                with open(out_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                got.append(base)
    return got


def require(files: list[str], expected: list[str], ctx: str) -> None:
    missing = [e for e in expected if e not in files]
    if missing:
        raise SystemExit(f"错误：{ctx} 缺少预期文件: {missing}（实际提取: {sorted(files)}）")


def pypi_wheel_url(pkg: str, ver: str) -> str:
    data = json.loads(http_get(f"https://pypi.org/pypi/{pkg}/{ver}/json"))
    for f in data["urls"]:
        if "win_amd64" in f["filename"]:
            return f["url"]
    raise SystemExit(f"PyPI 上找不到 {pkg}=={ver} 的 win_amd64 wheel")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- PE 导入表分析（按需补件 / 组装后自检） ----------------

def pe_imports(path: Path, delay: bool = False) -> set[str]:
    """返回某个 PE 文件导入（或延迟导入）的 DLL 名集合；pefile 不可用时返回空集。"""
    try:
        import pefile
    except ImportError:
        log("警告: 未安装 pefile，跳过导入表分析（pip install pefile 可启用）")
        return set()
    key = "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT" if delay else "IMAGE_DIRECTORY_ENTRY_IMPORT"
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY[key]])
        attr = "DIRECTORY_ENTRY_DELAY_IMPORT" if delay else "DIRECTORY_ENTRY_IMPORT"
        out = set()
        for e in getattr(pe, attr, []) or []:
            out.add(e.dll.decode(errors="replace").lower())
        return out
    finally:
        pe.close()


def all_imports(path: Path) -> set[str]:
    return pe_imports(path, False) | pe_imports(path, True)


def add_cufft_if_needed(cuda_dir: Path, cache: Path, force: str) -> None:
    """按 onnxruntime_providers_cuda.dll 的真实导入表决定是否需要 cuFFT。"""
    target = cuda_dir / "onnxruntime_providers_cuda.dll"
    if not target.is_file():
        return
    imports = all_imports(target)
    needed = sorted(n for n in imports if n.startswith("cufft"))
    log(f"onnxruntime_providers_cuda.dll 导入表中的 cufft 依赖: {needed or '（无）'}")

    if force == "never":
        if needed:
            log(f"⚠ 检测到需要 {needed}，但 --cufft=never 指定不打包（ASR 将回退 CPU）")
        return
    if force == "auto" and not needed:
        log("不需要 cuFFT，跳过（--cufft always 可强制打包）")
        return

    if needed:
        wanted = {n.lower() for n in needed}
    else:
        wanted = {"cufft64_11.dll"}
    whl = download_cached(pypi_wheel_url(CUFFT_PKG, CUFFT_VER), cache)
    got = []
    with zipfile.ZipFile(whl) as z:
        for n in z.namelist():
            base = n.rsplit("/", 1)[-1]
            if n.endswith(".dll") and "/bin/" in n and base.lower() in wanted:
                with z.open(n) as src, open(cuda_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                got.append(base)
    if not got:
        # wheel 里的名字可能与导入表不完全一致（如版本号后缀），退化为全量 bin/*.dll
        log(f"未按名字匹配到 {wanted}，改用 wheel 内全部 bin/*.dll")
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                base = n.rsplit("/", 1)[-1]
                if n.endswith(".dll") and "/bin/" in n:
                    with z.open(n) as src, open(cuda_dir / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    got.append(base)
    if not got:
        raise SystemExit("cuFFT wheel 中未找到任何 DLL")
    log(f"cuFFT DLLs: {got}")


def add_vcredist(pkg_dir: Path, explicit: str | None) -> None:
    """拷入 VC++ 运行时（exe 是 /MD 构建，干净系统可能没有 vcruntime140.dll）。"""
    dirs: list[Path] = []
    if explicit:
        dirs.append(Path(explicit))
    else:
        for pat in VCREDIST_GLOBS:
            dirs.extend(Path(p) for p in _glob(pat))
    got = []
    for d in dirs:
        for name in VCREDIST_DLLS:
            f = d / name
            if f.is_file() and not (pkg_dir / name).exists():
                shutil.copy2(f, pkg_dir / name)
                got.append(f"{name} <- {d}")
    if got:
        log("VC++ 运行时: " + "; ".join(got))
    else:
        log("⚠ 未找到 VC++ 运行时 DLL（vcruntime140 等），"
            "干净系统上 exe 可能无法启动；请用 --vcredist-dir 指定")


def _glob(pattern: str) -> list[str]:
    """跨平台 glob（Windows 上用 pathlib 的多级通配）。"""
    import glob as _g
    return _g.glob(pattern)


def add_common_files(pkg_dir: Path, repo_root: Path) -> None:
    """prompts / README / LICENSE / 第三方许可 / 模型下载脚本 / models 占位。"""
    shutil.copytree(repo_root / "prompts", pkg_dir / "prompts", dirs_exist_ok=True)
    for name in ["README.md", "LICENSE"]:
        src = repo_root / name
        if src.is_file():
            shutil.copy2(src, pkg_dir / name)
    (pkg_dir / "THIRD-PARTY-NOTICES.txt").write_text(THIRD_PARTY_NOTICES, encoding="utf-8")
    (pkg_dir / "scripts").mkdir(exist_ok=True)
    for name in ["download_models.ps1", "download_models.bat"]:
        src = repo_root / "scripts" / name
        if src.is_file():
            shutil.copy2(src, pkg_dir / "scripts" / name)
    models = pkg_dir / "models"
    models.mkdir(exist_ok=True)
    (models / "README.txt").write_text(
        "请运行 scripts\\download_models.bat（或 download_models.ps1）自动下载模型到本目录：\n"
        "  1. sherpa-onnx-qwen3-asr-0.6B-int8/  ASR 模型\n"
        "  2. silero_vad.onnx                    VAD 模型\n"
        "  3. Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf  翻译/质检 LLM\n"
        "国内网络可加参数: powershell -File download_models.ps1 -HfMirror https://hf-mirror.com\n",
        encoding="utf-8",
    )


def list_package(pkg_dir: Path) -> None:
    files = sorted(p for p in pkg_dir.rglob("*") if p.is_file())
    total = sum(f.stat().st_size for f in files) / 1e6
    log(f"{pkg_dir.name}: {len(files)} 个文件, {total:.1f} MB")
    for f in files:
        rel = f.relative_to(pkg_dir).as_posix()
        log(f"    {rel:<58} {f.stat().st_size / 1e6:8.2f} MB")


def verify_deps(pkg_dir: Path, repo_root: Path, strict: bool) -> None:
    script = repo_root / "scripts" / "check_dll_deps.py"
    cmd = [sys.executable, str(script), "--dir", str(pkg_dir), "--strict-vcredist"]
    if strict:
        cmd.append("--fail-on-missing")
    log("依赖闭包检查: " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        if strict:
            raise SystemExit(f"{pkg_dir.name} 依赖闭包检查失败")
        log(f"⚠ {pkg_dir.name} 依赖闭包检查发现缺件（非严格模式，仅告警）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="构建产物 subtitle-assistant.exe 路径")
    ap.add_argument(
        "--build-dir",
        default="target/release",
        help="cargo 构建输出目录（用于提取自建的 llama-common.dll）",
    )
    ap.add_argument("--version", required=True, help="版本号（如 v0.2.0 或 dev-abcd1234）")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--cache", default=".distcache")
    ap.add_argument("--cufft", choices=["auto", "always", "never"], default="auto",
                    help="auto=按 onnxruntime_providers_cuda.dll 的导入表决定（默认）")
    ap.add_argument("--vcredist-dir", default=None,
                    help="VC++ 运行时 DLL 所在目录（默认自动搜索 VS Redist / System32）")
    ap.add_argument("--strict-deps", action="store_true",
                    help="依赖闭包检查发现缺件时直接失败")
    ap.add_argument("--skip-verify", action="store_true", help="跳过依赖闭包检查")
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.is_file():
        raise SystemExit(f"找不到 exe: {exe}")
    # exe 通过 llama-cpp-sys-2（default 启用 common）导入 llama-common.dll；
    # 必须使用与 exe 同一次构建产出的版本（官方 zip 的 common.h 已漂移，不可用）
    llama_common_dll = Path(args.build_dir) / "llama-common.dll"
    if not llama_common_dll.is_file():
        raise SystemExit(
            f"找不到 {llama_common_dll}：llama-cpp-sys-2 构建时应已将其硬链接到构建输出目录"
        )
    repo_root = Path(__file__).resolve().parent.parent
    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ver = args.version.lstrip("v")

    # exe 自身的导入表（用于确认 VC++ 运行时确实需要随包分发）
    exe_imports = all_imports(exe)
    log(f"exe 导入的非系统 DLL: {sorted(n for n in exe_imports if not n.startswith('api-ms'))}")

    # ---------------- 下载全部依赖包 ----------------
    llama_cpu_zip = download_cached(f"{GGML_DL_BASE}/{LLAMA_CPU_ZIP}", cache)
    llama_cuda_zip = download_cached(f"{GGML_DL_BASE}/{LLAMA_CUDA_ZIP}", cache)
    cudart_zip = download_cached(f"{GGML_DL_BASE}/{CUDART_ZIP}", cache)
    sherpa_cpu_tar = download_cached(f"{SHERPA_DL_BASE}/{SHERPA_CPU_TAR}", cache)
    sherpa_cuda_tar = download_cached(f"{SHERPA_DL_BASE}/{SHERPA_CUDA_TAR}", cache)
    cudnn_whl = download_cached(pypi_wheel_url(CUDNN_PKG, CUDNN_VER), cache)

    def keep_llama(base: str) -> bool:
        return base in LLAMA_DLL_KEEP or any(base.startswith(p) for p in LLAMA_DLL_PREFIX_KEEP)

    # ---------------- CPU 包 ----------------
    cpu_dir = out / f"Qwen3SubAssistant-{ver}-win-x64-cpu"
    if cpu_dir.exists():
        shutil.rmtree(cpu_dir)
    cpu_dir.mkdir(parents=True)
    log("assembling CPU package ...")

    shutil.copy2(exe, cpu_dir / "subtitle-assistant.exe")
    shutil.copy2(llama_common_dll, cpu_dir / "llama-common.dll")
    got = extract_from_zip(llama_cpu_zip, keep_llama, cpu_dir)
    require(got, ["llama.dll", "ggml.dll", "ggml-base.dll", "libomp.dll"], "llama CPU zip")
    if "llama-common.dll" in got:
        raise SystemExit("官方 zip 不应提供 llama-common.dll（应使用自建版本），请检查过滤规则")
    if not any(f.startswith("ggml-cpu") for f in got):
        raise SystemExit("llama CPU zip 未提取到任何 ggml-cpu-*.dll 后端")
    got = extract_from_tarbz2(
        sherpa_cpu_tar, lambda b: b in SHERPA_CPU_DLLS, cpu_dir, path_filter=lambda n: "/lib/" in n
    )
    require(got, SHERPA_CPU_DLLS, "sherpa CPU tar")
    add_vcredist(cpu_dir, args.vcredist_dir)
    add_common_files(cpu_dir, repo_root)

    # ---------------- CUDA 12 包 ----------------
    cuda_dir = out / f"Qwen3SubAssistant-{ver}-win-x64-cuda12"
    if cuda_dir.exists():
        shutil.rmtree(cuda_dir)
    cuda_dir.mkdir(parents=True)
    log("assembling CUDA12 package ...")

    shutil.copy2(exe, cuda_dir / "subtitle-assistant.exe")
    shutil.copy2(llama_common_dll, cuda_dir / "llama-common.dll")
    got = extract_from_zip(llama_cuda_zip, keep_llama, cuda_dir)
    require(got, LLAMA_DLL_KEEP, "llama CUDA zip")  # 含 ggml-cuda.dll
    got = extract_from_zip(cudart_zip, lambda b: b in CUDART_DLLS, cuda_dir)
    require(got, CUDART_DLLS, "cudart zip")
    got = extract_from_tarbz2(
        sherpa_cuda_tar, lambda b: b in SHERPA_CUDA_DLLS, cuda_dir, path_filter=lambda n: "/lib/" in n
    )
    require(got, SHERPA_CUDA_DLLS, "sherpa CUDA tar")
    # cuDNN 9（wheel 内 nvidia/cudnn/bin/*.dll）
    with zipfile.ZipFile(cudnn_whl) as z:
        cudnn_dlls = [n for n in z.namelist() if n.endswith(".dll") and "/bin/" in n]
        if not cudnn_dlls:
            raise SystemExit("cuDNN wheel 中未找到 DLL")
        for n in cudnn_dlls:
            base = n.rsplit("/", 1)[-1]
            with z.open(n) as src, open(cuda_dir / base, "wb") as dst:
                shutil.copyfileobj(src, dst)
        log(f"cuDNN DLLs: {[n.rsplit('/', 1)[-1] for n in cudnn_dlls]}")
    # cuFFT：按导入表按需补齐（缺它会导致 ASR 的 CUDA EP 静默回退 CPU）
    add_cufft_if_needed(cuda_dir, cache, args.cufft)
    add_vcredist(cuda_dir, args.vcredist_dir)
    add_common_files(cuda_dir, repo_root)

    # ---------------- 自检 + 压缩 + 校验和 ----------------
    for pkg_dir in [cpu_dir, cuda_dir]:
        list_package(pkg_dir)
        if not args.skip_verify:
            verify_deps(pkg_dir, repo_root, args.strict_deps)

    sums = []
    for pkg_dir in [cpu_dir, cuda_dir]:
        zip_path = out / f"{pkg_dir.name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        log(f"zipping {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(pkg_dir.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        digest = sha256_file(zip_path)
        sums.append(f"{digest}  {zip_path.name}")
        log(f"{zip_path.name}: {zip_path.stat().st_size / 1e6:.1f} MB  sha256={digest[:16]}...")

    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    log("done.")


if __name__ == "__main__":
    sys.exit(main())
