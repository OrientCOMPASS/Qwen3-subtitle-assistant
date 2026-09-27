#!/usr/bin/env python3
"""e2e 断言：终端输出清洁度 + 日志文件全量诊断（v0.5「默认不刷屏」行为的回归闸门）。

背景：v0.4 及以前，llama.cpp 原生加载日志（llama_model_loader/load_tensors/
control token 等数百行）与逐段 [ASR✔]/[VAD✂] 明细全部直写终端，用户抱怨
"终端特别乱"。v0.5 起：终端默认只有进度/结果级 info；诊断明细进 debug 级
（--verbose 放开到终端；--log-file 的文件通道恒 debug 全量）；llama/ggml
原生日志经 llama_log_set/ggml_log_set 桥接进同一门面。

用法：
  # 默认参数运行（T1）：终端必须干净、日志文件必须有全量诊断
  python scripts/check_console_clean.py media/t1_console.txt media/t1.log
  # --verbose 运行（T1c）：终端必须出现 debug 级明细
  python scripts/check_console_clean.py media/t1c_console.txt media/t1c.log --verbose-run
"""
from __future__ import annotations

import pathlib
import sys

# 诊断噪声标记：默认级终端上出现任何一个都判失败
NOISE_MARKERS = (
    "llama_model_loader",   # GGUF 元数据 dump
    "load_tensors",         # 逐层 offload 日志
    "ggml_backend",         # 后端搜索/加载
    "print_info",           # 模型信息
    "init_tokenizer",       # tokenizer 初始化
    "control token",        # 特殊 token 提示（数十行）
    "clip_model_loader",    # mtmd/clip 加载日志（mtmd_log_set 桥接对象）
    "clip_ctx",
    "compute buffer",       # reserve_compute_meta 显存/内存预算行
    "encoding audio slice", # mtmd-helper 逐段推理计时（mtmd_helper_log_set 桥接对象）
    "audio slice encoded",
    "decoding audio batch",
    "audio decoded",
    "ggml 后端设备",         # 应用侧后端枚举（debug 级）
    "[ASR✔",                # 逐段转录明细（debug 级）
    "[ASR∅",
    "[VAD✂",
    "排版完成",              # debug 级
    "exe 目录",              # debug 级
)

# 结果级信息：默认级终端上必须出现（每个文件一组）
RESULT_MARKERS = ("已写出字幕", "转录完成")

# 日志文件（--log-file 通道恒 debug）必须含有的诊断内容
LOGFILE_MARKERS = ("llama_model_loader", "[ASR✔", "转录完成")


def read(p: str | pathlib.Path) -> str:
    return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")


def check_default_run(console_txt: str, log_txt: str) -> int:
    bad = 0
    lines = [l for l in console_txt.splitlines() if l.strip()]

    # 1) 终端不得出现诊断噪声
    for m in NOISE_MARKERS:
        hits = [l for l in lines if m in l]
        if hits:
            print(f"✘ 终端出现诊断噪声 {m!r}（{len(hits)} 行），例如: {hits[0][:110]}")
            bad += 1
    if bad == 0:
        print(f"✔ 终端无诊断噪声（{len(lines)} 行输出）")

    # 2) 终端必须有结果级信息
    for m in RESULT_MARKERS:
        if m in console_txt:
            print(f"✔ 终端含结果信息 {m!r}")
        else:
            print(f"✘ 终端缺少结果信息 {m!r}")
            bad += 1

    # 3) 终端行数上限：结果级最小集合 ≈ 每文件 3 行 + 全局 ~6 行，放宽到 5n+12
    n_files = max(1, console_txt.count("已写出字幕"))
    cap = 5 * n_files + 12
    if len(lines) <= cap:
        print(f"✔ 终端行数 {len(lines)} <= 上限 {cap}（按 {n_files} 个文件计）")
    else:
        print(f"✘ 终端行数 {len(lines)} 超过上限 {cap} —— 仍有刷屏")
        bad += 1

    # 4) 日志文件必须有全量诊断（原生桥接 + 逐段明细 + 总结）
    for m in LOGFILE_MARKERS:
        if m in log_txt:
            print(f"✔ 日志文件含诊断 {m!r}")
        else:
            print(f"✘ 日志文件缺少诊断 {m!r}（文件通道应为 debug 全量）")
            bad += 1
    return bad


def check_verbose_run(console_txt: str, log_txt: str) -> int:
    bad = 0
    # --verbose 下终端必须放开 debug 明细（逐段 ASR 或原生加载日志任一即可）
    verbose_markers = ("[ASR✔", "[ASR∅", "[VAD✂", "ggml 后端设备", "llama_model_loader")
    if any(m in console_txt for m in verbose_markers):
        hits = [m for m in verbose_markers if m in console_txt]
        print(f"✔ --verbose 终端出现 debug 明细: {hits}")
    else:
        print("✘ --verbose 终端未见任何 debug 明细（开关未生效？）")
        bad += 1
    # 日志文件仍须全量
    for m in LOGFILE_MARKERS:
        if m not in log_txt:
            print(f"✘ 日志文件缺少诊断 {m!r}")
            bad += 1
    if bad == 0:
        print("✔ verbose 对照运行断言全部通过")
    return bad


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose_run = "--verbose-run" in sys.argv
    if len(args) != 2:
        print(__doc__)
        return 2
    console_txt = read(args[0])
    log_txt = read(args[1])
    bad = check_verbose_run(console_txt, log_txt) if verbose_run \
        else check_default_run(console_txt, log_txt)
    print(f"断言失败数: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
