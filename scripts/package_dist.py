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
             + sherpa-onnx CUDA 包（onnxruntime CUDA 版 + providers_cuda）
             + cuDNN 9（NVIDIA PyPI wheel）

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
升级依赖时，请同步更新这里的 LLAMA_TAG / SHERPA_VER。
"""

import argparse
import bz2
import hashlib
import io
import json
import shutil
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
        log(f"cache hit: {dest.name}")
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
    """从 tar.bz2 中抽取 basename 满足谓词的文件（流式，不解压全部）。
    path_filter: 可选，对完整归档路径二次过滤（如只要 lib/ 下的文件）。"""
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


def add_common_files(pkg_dir: Path, repo_root: Path) -> None:
    """prompts / README / LICENSE / 模型下载脚本 / models 占位。"""
    shutil.copytree(repo_root / "prompts", pkg_dir / "prompts", dirs_exist_ok=True)
    for name in ["README.md", "LICENSE"]:
        src = repo_root / name
        if src.is_file():
            shutil.copy2(src, pkg_dir / name)
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
        log(f"cuDNN DLLs: {[n.rsplit('/',1)[-1] for n in cudnn_dlls]}")
    add_common_files(cuda_dir, repo_root)

    # ---------------- 压缩 + 校验和 ----------------
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
