#!/usr/bin/env python3
"""快速直出版单文件发行包组装（E3 单模型形态）。

产物：dist/Qwen3SubAssistant-Fast-<ver>-win-x64.zip ≤ 2GiB（GitHub Release 单文件上限）
  ├── subtitle-assistant.exe
  ├── llama.dll / ggml*.dll / mtmd*.dll        （llama.cpp 运行时，构建产物）
  ├── ggml-vulkan.dll                          （通用 GPU 后端，官方 release 取，可选）
  ├── models/qwen3-asr-s2tt/{model,mmproj}.gguf（S2TT 微调模型，gguf-e2 产线产物）
  └── README-快速开始.txt

设计要点：
  * 不含任何 CUDA/cuDNN DLL——CUDA 加速由用户 `--cuda-libs <DIR>` 指定运行库目录，
    运行时经 ggml_backend_load_all_from_path 加载（ggml-cuda.dll 亦放该目录，可从
    llama.cpp 官方 release 的 cuda 包自取，与内核版本对齐）；
  * 不含 LLM GGUF / sherpa / onnxruntime（双模型工作流已移除）；
  * VAD 权重内嵌 exe（silero v4，0.62MB），无外部模型依赖；
  * 体积账（finetune/INTEGRATION.md §9.3）：模型 1.36GiB + 运行时 ~0.15GiB ≈ 1.5GiB。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

GIB = 1024 ** 3
LIMIT = 2 * GIB           # GitHub Release 单文件硬上限
WARN_AT = int(1.9 * GIB)


def log(m: str) -> None:
    print(f"[pkg] {m}", flush=True)


def fetch_vulkan_dll(tag: str, dest: Path) -> bool:
    """从 llama.cpp 官方 release 取 ggml-vulkan.dll（SPIR-V 内嵌，数十 MB）。"""
    url = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
           f"{tag}/llama-{tag}-bin-win-vulkan-x64.zip")
    if dest.is_file():
        log(f"ggml-vulkan.dll 已存在，跳过下载")
        return True
    try:
        log(f"下载 {url}")
        data = urllib.request.urlopen(url, timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith("ggml-vulkan.dll")]
            if not names:
                log(f"!! {tag} vulkan 包里没有 ggml-vulkan.dll: {z.namelist()[:8]}")
                return False
            dest.write_bytes(z.read(names[0]))
        log(f"ggml-vulkan.dll {dest.stat().st_size/1e6:.1f}MB")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"!! Vulkan DLL 下载失败（{e}）——包仍可用（CPU），GPU 用户需自行补齐")
        return False


def find_ggufs(gguf_dir: Path) -> tuple[Path, Path]:
    lm = gguf_dir / "model.gguf"
    mm = gguf_dir / "mmproj.gguf"
    if not lm.is_file():
        cands = sorted(p for p in gguf_dir.glob("*.gguf")
                       if not p.name.startswith("mmproj") and "f16" not in p.name)
        if not cands:
            raise SystemExit(f"{gguf_dir} 里找不到 LM GGUF")
        lm = cands[0]
    if not mm.is_file():
        cands = sorted(gguf_dir.glob("mmproj*.gguf"))
        if not cands:
            raise SystemExit(f"{gguf_dir} 里找不到 mmproj GGUF")
        mm = cands[0]
    return lm, mm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--build-dir", required=True, help="target/release（收集 llama/ggml DLL）")
    ap.add_argument("--gguf-dir", required=True, help="含 s2tt LM + mmproj GGUF 的目录")
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--llama-tag", default="b11201", help="ggml-vulkan.dll 的 llama.cpp release tag")
    ap.add_argument("--skip-vulkan", action="store_true")
    ap.add_argument("--cache", default=".distcache")
    args = ap.parse_args()

    exe = Path(args.exe)
    build = Path(args.build_dir)
    gguf_dir = Path(args.gguf_dir)
    if not exe.is_file():
        raise SystemExit(f"exe 不存在: {exe}")
    lm, mm = find_ggufs(gguf_dir)
    log(f"LM: {lm.name} {lm.stat().st_size/GIB:.2f}GiB | mmproj: {mm.name} {mm.stat().st_size/1e6:.0f}MB")

    name = f"Qwen3SubAssistant-Fast-{args.version}-win-x64"
    stage = Path(args.out) / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "models" / "qwen3-asr-s2tt").mkdir(parents=True)

    # 1) exe + llama.cpp 家族 DLL
    shutil.copy2(exe, stage / exe.name)
    dlls = []
    for pat in ("llama*.dll", "ggml*.dll", "mtmd*.dll"):
        for d in sorted(build.glob(pat)):
            if d.name.lower().startswith(("ggml-cuda", "ggml-hip", "ggml-sycl", "ggml-opencl")):
                continue  # GPU 后端 DLL 不随基础包（CUDA 用户自备；Vulkan 单独下载）
            shutil.copy2(d, stage / d.name)
            dlls.append(d.name)
    log(f"运行时 DLL: {dlls}")
    for must in ("llama.dll", "ggml.dll", "ggml-base.dll"):
        if not (stage / must).is_file():
            raise SystemExit(f"缺少必需 DLL: {must}（build-dir 里没有）")

    # 2) Vulkan 后端（通用 GPU，用户零安装：vulkan-1.dll 系统自带）
    if not args.skip_vulkan:
        cache = Path(args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        fetch_vulkan_dll(args.llama_tag, cache / "ggml-vulkan.dll")
        if (cache / "ggml-vulkan.dll").is_file():
            shutil.copy2(cache / "ggml-vulkan.dll", stage / "ggml-vulkan.dll")

    # 3) 模型（统一命名，程序默认目录探测即中）
    shutil.copy2(lm, stage / "models" / "qwen3-asr-s2tt" / "model.gguf")
    shutil.copy2(mm, stage / "models" / "qwen3-asr-s2tt" / "mmproj.gguf")

    # 4) 使用说明
    (stage / "README-快速开始.txt").write_text(
        "Qwen3 Subtitle Assistant —— 快速直出版（单模型 S2TT，无 LLM 后处理）\n"
        "\n"
        "用法：把视频/音频文件拖到 subtitle-assistant.exe 上，或在命令行：\n"
        "    subtitle-assistant.exe 视频.mp4\n"
        "输出：同目录 视频.srt（简体中文直出）与 视频.raw.srt（排版前）。\n"
        "\n"
        "硬件加速（自动选择，无需配置）：\n"
        "  * Vulkan：任意 Windows 10/11 GPU（NVIDIA/AMD/Intel），驱动即装即用；\n"
        "  * CUDA（可选，更快）：把 NVIDIA 运行库（cudart64_*.dll、cublas64_*.dll、\n"
        "    cublasLt64_*.dll、cudnn64_*.dll 等）与 ggml-cuda.dll 放同一目录，启动时加\n"
        "        --cuda-libs \"D:\\path\\to\\cuda-libs\"\n"
        "    ggml-cuda.dll 可从 llama.cpp 官方 release 的 win-cuda 包自取（版本对齐 %s）；\n"
        "  * 都没有：自动纯 CPU（1.7B Q4_K_M 实测 RTF≈0.6@4线程）。\n"
        "  权重加载进显存后主机内存副本自动释放；--device cpu 可强制纯 CPU。\n"
        "\n"
        "常用参数：--context \"\"（转写源语言而非翻译）、--threads N、--output-dir DIR、\n"
        "          --vad-min-silence 0.5、--max-line-width 44、--gguf-selftest（环境自检）。\n"
        % args.llama_tag, encoding="utf-8")

    # 5) PE 导入闭包自检（复用 check_dll_deps.py，如可用）
    checker = Path(__file__).parent / "check_dll_deps.py"
    if checker.is_file():
        import subprocess
        r = subprocess.run([sys.executable, str(checker), str(stage / exe.name)],
                           capture_output=True, text=True)
        log("check_dll_deps: " + (r.stdout.strip()[-300:] or r.stderr.strip()[-300:]))

    # 6) zip（GGUF 已是压缩熵高位，deflate 收益小，用 ZIP_STORED 换打包速度）
    zip_path = Path(args.out) / f"{name}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage.parent))
    size = zip_path.stat().st_size
    log(f"单文件包: {zip_path} = {size/GIB:.3f} GiB")
    if size > LIMIT:
        raise SystemExit(f"✘ 超过 GitHub Release 单文件上限 2GiB（{size/GIB:.3f}GiB）")
    if size > WARN_AT:
        log(f"⚠ 接近 2GiB 上限（余量 {(LIMIT-size)/1e6:.0f}MB）")
    else:
        log(f"✔ 体积合规，余量 {(LIMIT-size)/GIB:.2f} GiB")

    # 7) SHA256SUMS
    h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sums = Path(args.out) / "SHA256SUMS.txt"
    with sums.open("a", encoding="utf-8") as f:
        f.write(f"{h}  {zip_path.name}\n")
    log(f"SHA256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
