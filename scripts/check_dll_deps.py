#!/usr/bin/env python3
"""发行包 DLL 依赖闭包检查（静态 PE 分析，不需要 GPU / 驱动 / CUDA 环境）。

为什么需要它：
  本项目走「外置 DLL」运行时，CUDA 能力完全由 exe 旁边的 DLL 决定。
  只要少一个依赖（典型如 ONNX Runtime CUDA EP 需要的 cufft64_11.dll），
  运行时表现就是 **静默回退 CPU**——用户拿到 1.6 GB 的 CUDA 包却毫无加速，
  日志里只有一行 warn。这类问题必须在打包阶段就查出来。

用法:
    python scripts/check_dll_deps.py --dir dist/Qwen3SubAssistant-x-win-x64-cuda12
    python scripts/check_dll_deps.py --dir <dir> --fail-on-missing   # 缺件则退出码 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pefile
except ImportError:  # pragma: no cover
    print("需要 pefile：pip install pefile", file=sys.stderr)
    raise SystemExit(2)

# Windows 自带、不需要随包分发的 DLL（小写比较）
SYSTEM_DLLS = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll", "gdi32.dll",
    "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll",
    "wldap32.dll", "bcrypt.dll", "bcryptprimitives.dll", "crypt32.dll",
    "secur32.dll", "version.dll", "shlwapi.dll", "userenv.dll", "psapi.dll",
    "dbghelp.dll", "iphlpapi.dll", "normaliz.dll", "winmm.dll", "imm32.dll",
    "wintrust.dll", "ncrypt.dll", "rpcrt4.dll", "comdlg32.dll", "setupapi.dll",
    "cfgmgr32.dll", "powrprof.dll", "pdh.dll", "d3d11.dll", "d3d12.dll",
    "dxgi.dll", "dcomp.dll", "twinapi.appcore.dll", "mfplat.dll", "mf.dll",
    "mfreadwrite.dll", "mfuuid.dll", "propsys.dll", "wtsapi32.dll", "dxva2.dll",
    "dwmapi.dll", "usp10.dll", "msimg32.dll", "winnls.dll", "profapi.dll",
    "sspicli.dll", "msasn1.dll", "authz.dll", "netapi32.dll", "mswsock.dll",
    "dnsapi.dll", "winhttp.dll", "urlmon.dll", "iertutil.dll", "schannel.dll",
    "msvcrt.dll", "ucrtbase.dll", "gdiplus.dll", "opengl32.dll", "combase.dll",
}
SYSTEM_PREFIXES = ("api-ms-", "ext-ms-", "ucrt", "windows.storage", "wldp", "coremessaging",
                   "textinputframework", "windowmanagementapi", "rometadata", "policymanager")

# 由显卡驱动 / 系统组件提供，本就不该打进包（缺了不算错误）
EXTERNAL_OK = {
    "nvcuda.dll",      # NVIDIA 驱动
    "nvml.dll",        # NVIDIA 驱动
    "nvrtc64_12.dll",  # CUDA JIT（ggml-cuda 不使用；若出现则应显式打包）
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",  # 见 --strict-vcredist
}
VCREDIST = {"vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"}


def is_system(name: str) -> bool:
    n = name.lower()
    return n in SYSTEM_DLLS or n.startswith(SYSTEM_PREFIXES)


def imports_of(path: Path) -> tuple[list[str], list[str]]:
    """返回 (普通导入, 延迟加载导入) 的 DLL 名列表。"""
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        normal = []
        for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            normal.append(e.dll.decode(errors="replace"))
        delay = []
        for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []) or []:
            delay.append(e.dll.decode(errors="replace"))
        return normal, delay
    finally:
        pe.close()


def check_dir(d: Path, strict_vcredist: bool) -> dict[str, list[tuple[str, str]]]:
    """返回 {缺失 DLL 名: [(引用者, 导入类型), ...]}"""
    present = {p.name.lower() for p in d.iterdir() if p.suffix.lower() in (".dll", ".exe")}
    targets = sorted(p for p in d.iterdir() if p.suffix.lower() in (".dll", ".exe"))
    missing: dict[str, list[tuple[str, str]]] = {}

    for f in targets:
        try:
            normal, delay = imports_of(f)
        except Exception as e:  # noqa: BLE001
            print(f"  !! 无法解析 {f.name}: {e}")
            continue
        for kind, names in (("IMPORT", normal), ("DELAY", delay)):
            for name in names:
                low = name.lower()
                if is_system(low):
                    continue
                if low in present:
                    continue
                if low in EXTERNAL_OK:
                    if low in VCREDIST and strict_vcredist:
                        missing.setdefault(name, []).append((f.name, kind + "(vcredist)"))
                    continue
                missing.setdefault(name, []).append((f.name, kind))
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, action="append",
                    help="发行包目录（可多次）")
    ap.add_argument("--fail-on-missing", action="store_true",
                    help="发现缺失依赖时以退出码 1 结束")
    ap.add_argument("--strict-vcredist", action="store_true",
                    help="把 vcruntime140/msvcp140 也视为必须随包分发")
    args = ap.parse_args()

    any_missing = False
    for ds in args.dir:
        d = Path(ds)
        if not d.is_dir():
            print(f"目录不存在: {d}")
            any_missing = True
            continue
        dlls = sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".dll")
        total_mb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e6
        print(f"\n=== {d.name}  ({len(dlls)} 个 DLL, 合计 {total_mb:.1f} MB) ===")
        print("包内 DLL: " + ", ".join(dlls))

        missing = check_dir(d, args.strict_vcredist)
        if not missing:
            print("✔ 依赖闭包完整：所有非系统导入都能在包内找到")
            continue
        any_missing = True
        print(f"✘ 缺失 {len(missing)} 个依赖：")
        for name, refs in sorted(missing.items()):
            who = ", ".join(f"{r[0]}[{r[1]}]" for r in refs[:6])
            print(f"    {name:<32} <- {who}")

    if any_missing and args.fail_on_missing:
        print("\n--fail-on-missing 已开启，判定失败。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
