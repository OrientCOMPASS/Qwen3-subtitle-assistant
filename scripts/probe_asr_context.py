#!/usr/bin/env python3
"""探测 ASR 的 context 通道能否让模型**直接输出目标语言**（免翻译）。

背景：Qwen3-ASR 的 system 段是唯一可注入的自由文本通道——
  * `qwen-asr` 的 `transcribe(audio, context=...)` 把 context 放进 system role；
  * sherpa-onnx 把 `hotwords` 字段放进 system role
    （源码注释：Qwen3-ASR hotwords are placed in the system-role segment of the chat template）；
  * 本项目通过 `--asr-hotwords` 暴露它。

论文 §2.2 说模型被刻意训练成"不遵循 prompt 里的自然语言指令"（防指令注入），
所以理论上塞"translate to Chinese"不会生效。本脚本用**实测**回答这个问题：
拿同一段音频跑两次（不带 context / 带翻译指令），比较输出的文字构成。

判据（只看统计，不做成败判定——两种结论都有价值）：
  * probe 的假名占比大幅下降、汉字占比大幅上升  => context 通道可用，能省掉翻译段
  * 两者接近且都仍是日语                        => 印证论文，只能靠微调改权重

用法:
    python scripts/probe_asr_context.py --control media/x.raw.srt --probe media/t4/x.raw.srt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def parse_srt(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    out = []
    for block in text.split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        body = " ".join(lines[idx + 1:]).strip()
        if body:
            out.append(body)
    return out


def profile(lines: list[str]) -> dict:
    text = "".join(lines)
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)
    return {
        "cues": len(lines),
        "chars": len(chars),
        "kana": sum(1 for c in chars if KANA.match(c)) / n,
        "cjk": sum(1 for c in chars if CJK.match(c)) / n,
        "latin": sum(1 for c in chars if LATIN.match(c)) / n,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="不带 context 的 ASR 原始输出 .srt")
    ap.add_argument("--probe", required=True, help="带翻译指令 context 的 ASR 原始输出 .srt")
    ap.add_argument("--shift-threshold", type=float, default=0.20,
                    help="假名占比下降超过该值才认为 context 通道真的起了作用")
    ap.add_argument("--samples", type=int, default=3, help="打印几条样例对照")
    args = ap.parse_args()

    ctrl_lines = parse_srt(Path(args.control))
    probe_lines = parse_srt(Path(args.probe))
    if not ctrl_lines or not probe_lines:
        print(f"!! 无法比较：control {len(ctrl_lines)} 条 / probe {len(probe_lines)} 条")
        return 0  # 报告型脚本，不让 CI 因此判红

    c, p = profile(ctrl_lines), profile(probe_lines)
    print("=" * 68)
    print("ASR context 通道探测（--asr-hotwords 塞翻译指令，看输出语言是否改变）")
    print("=" * 68)
    print(f"{'':<12}{'条数':>6}{'字符':>8}{'假名':>9}{'汉字':>9}{'拉丁':>9}")
    for name, d in (("control", c), ("probe", p)):
        print(f"{name:<12}{d['cues']:>6}{d['chars']:>8}"
              f"{d['kana']:>8.1%}{d['cjk']:>9.1%}{d['latin']:>9.1%}")
    dk = c["kana"] - p["kana"]
    dc = p["cjk"] - c["cjk"]
    print(f"\n假名占比变化: {c['kana']:.1%} -> {p['kana']:.1%}（下降 {dk:+.1%}）")
    print(f"汉字占比变化: {c['cjk']:.1%} -> {p['cjk']:.1%}（上升 {dc:+.1%}）")

    print(f"\n--- 对照样例（前 {args.samples} 条）---")
    for i in range(min(args.samples, len(ctrl_lines), len(probe_lines))):
        print(f"  control: {ctrl_lines[i][:70]}")
        print(f"  probe  : {probe_lines[i][:70]}")
        print()

    print("--- 结论 ---")
    if dk >= args.shift_threshold and p["kana"] < 0.15:
        print("✔ context 通道**有效**：塞翻译指令后输出显著转向中文。")
        print("  => 可以省掉翻译段，直接用 --asr-hotwords 让 ASR 出中文（需再验证质量与稳定性）。")
    elif dk >= args.shift_threshold:
        print("△ 部分生效：语言构成有明显位移，但输出仍混有日语，不足以替代翻译段。")
    else:
        print("✘ context 通道**无效**：输出语言基本没变（仍是日语）。")
        print("  => 印证论文 §2.2「被刻意训练成不遵循 prompt 里的自然语言指令」（防指令注入）。")
        print("  => 想让 ASR 直接输出中文，只能微调权重，见 finetune/ 与 exp/asr-s2tt 分支。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
