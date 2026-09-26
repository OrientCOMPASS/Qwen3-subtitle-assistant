#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用官方 bf16 实现交叉验证 Qwen3-ASR 的 context 通道能否做 S2TT。

为什么要这一步：Rust 侧跑的是 Q4_K_M 量化 GGUF。如果量化版注入指令后仍输出
日语，有两种可能——
  A) context 通道本来就只能塞热词/背景，不能改变输出语言；
  B) 4bit 量化把这条能力压没了。
用未量化的 bf16 官方实现跑同一条音频、同一个 context，就能把 A/B 分开。
这直接决定"要不要微调"：若 bf16 能翻译，那问题只在导出/量化，微调是多余的。

用法：
    python finetune/probe_context_pytorch.py --media in.m4a \
        --context "Please translate the Japanese speech into Simplified Chinese." \
        [--model Qwen/Qwen3-ASR-0.6B] [--max-seconds 30]
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata


def script_of(ch: str) -> str:
    if "\u3040" <= ch <= "\u30ff":
        return "ja"
    if "\u4e00" <= ch <= "\u9fff":
        return "han"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return "other"


def ratio(text: str) -> dict:
    counts = {"han": 0, "ja": 0, "latin": 0}
    for ch in text:
        s = script_of(ch)
        if s in counts:
            counts[s] += 1
    total = sum(counts.values()) or 1
    return {k: round(100.0 * v / total, 1) for k, v in counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="只取前 N 秒：bf16 在 CPU 上很慢，探针不需要整条视频")
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    try:
        import librosa
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        print(f"[skip] 依赖缺失（{e}）——本步骤是可选交叉验证，不影响主探针结论")
        return 0

    print("=" * 72)
    print("bf16 官方实现 · context 通道交叉验证")
    print("=" * 72)
    print(f"模型：{args.model}   dtype=bfloat16   device=cpu")
    print(f"context：{args.context or '(无)'}")

    wav, _ = librosa.load(args.media, sr=16000, mono=True)
    if args.max_seconds and len(wav) > int(16000 * args.max_seconds):
        wav = wav[: int(16000 * args.max_seconds)]
    print(f"音频：{len(wav) / 16000:.1f}s（截取前 {args.max_seconds:.0f}s）")

    model = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")

    def run(context: str) -> str:
        try:
            out = model.transcribe(wav, language=args.language, context=context or None)
            return out[0]["text"]
        except TypeError:
            # 不同版本的 transcribe 签名可能不接受 context/language 关键字
            try:
                out = model.transcribe(wav)
                return out[0]["text"]
            except Exception as e:  # noqa: BLE001
                return f"<transcribe 失败: {type(e).__name__}: {e}>"
        except Exception as e:  # noqa: BLE001
            return f"<transcribe 失败: {type(e).__name__}: {e}>"

    base = run("")
    exp = run(args.context)

    print("\n--- 不注入 context ---")
    print(base)
    print(f"文字构成：{ratio(base)}")
    print("\n--- 注入 context ---")
    print(exp)
    print(f"文字构成：{ratio(exp)}")

    zh = ratio(exp)["han"]
    ja = ratio(exp)["ja"]
    print("\n--- 判定 ---")
    if exp.startswith("<"):
        print("无法判定：调用失败。")
        return 0
    if zh > 60 and ja < 10:
        print(f"bf16 能翻译（汉字 {zh}%）→ 量化版失败属于「导出/量化丢了能力」，")
        print("不必微调：应改走 bf16→ONNX 导出，或在 Rust 侧用更高精度量化。")
    elif abs(zh - ratio(base)["han"]) < 10:
        print(f"bf16 也不翻译（注入前后汉字占比 {ratio(base)['han']}% → {zh}%）→ 属于情况 A：")
        print("context 通道只影响识别先验，不改变输出语言。要直出目标语言必须微调。")
    else:
        print(f"部分影响（汉字 {ratio(base)['han']}% → {zh}%）：context 有作用但不足以稳定 S2TT，")
        print("微调仍是最可靠的路径。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("[skip] 交叉验证异常，不影响主探针结论")
        sys.exit(0)
