#!/usr/bin/env python3
"""Gate 2 验证：sherpa-onnx 运行时加载补丁后的 s2tt 模型目录，三项行为断言。

为什么必须在 sherpa-onnx 上验、不能只信 PyTorch 侧（T4 的教训）：
产品的最终运行时是 sherpa-onnx（int8 ONNX + 自带 KV=512 的解码实现），量化与
内核差异会改变行为。补丁器的数值自检（削顶率/重建 MSE）只证明「权重写对了」，
「带 hotwords 直出中文 / 不带仍日语 / 静音空输出」必须在最终运行时上复测。

三项断言（对补丁目录）：
  1. hotwords="translate to Chinese"：日语音频输出汉字占比 >= --min-cjk（默认 0.5）、
     假名 <= --max-kana（默认 0.10）——直出生效；
  2. 不带 hotwords：假名占比 >= 0.15——转写行为保留（任务开关成立）；
  3. 2s 纯静音：两种模式输出都 <= 2 字——language None 行为未被补丁破坏。

对照组：--baseline-dir 指原始官方包时同样跑一遍，**只报告不断言**
（T4 已证明基座带 hotwords 仍出日语——这正是补丁前后最直观的行为差异）。

测试音频：默认用模型包自带的 test_wavs/ja1.wav（日语），可 --audio 覆盖。

用法（CI：finetune.yml mode=onnx-patch 的验证步骤）：
  python finetune/verify_s2tt_onnx.py \
      --model-dir models/sherpa-onnx-qwen3-asr-0.6B-s2tt-int8 \
      --baseline-dir models/sherpa-onnx-qwen3-asr-0.6B-int8 \
      --out out/patch/verify_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
HOTWORDS = "translate to Chinese"


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    import os
    os._exit(code)


def ratios(text: str):
    chars = [c for c in text if not c.isspace()]
    n = max(1, len(chars))
    return (sum(1 for c in chars if KANA.match(c)) / n,
            sum(1 for c in chars if CJK.match(c)) / n)


def build_recognizer(model_dir: Path, hotwords: str, threads: int, max_new_tokens: int):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(model_dir / "conv_frontend.onnx"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        tokenizer=str(model_dir / "tokenizer"),
        num_threads=threads,
        hotwords=hotwords,
        max_new_tokens=max_new_tokens,
        max_total_len=512,          # 与产品默认一致（KV 固定 512）
        decoding_method="greedy_search",
    )


def transcribe(rec, arr, sr=16000) -> str:
    s = rec.create_stream()
    s.accept_waveform(sr, arr)
    rec.decode_stream(s)
    return (s.result.text or "").strip()


def run_suite(model_dir: Path, audio_path: Path, args) -> dict:
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(audio_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    silence = np.zeros(int(16000 * args.silence_secs), dtype=np.float32)

    out = {"model_dir": str(model_dir), "audio": str(audio_path)}
    for tag, hw in (("with_hotwords", HOTWORDS), ("no_hotwords", "")):
        t0 = time.time()
        rec = build_recognizer(model_dir, hw, args.threads, args.max_new_tokens)
        text = transcribe(rec, arr)
        kana, cjk = ratios(text)
        sil = transcribe(rec, silence)
        out[tag] = {"text": text, "silence_text": sil,
                    "kana": round(kana, 4), "cjk": round(cjk, 4),
                    "secs": round(time.time() - t0, 1)}
        log(f"[{model_dir.name}] {tag}: 假名{kana:.0%} 汉字{cjk:.0%} "
            f"({time.time()-t0:.0f}s)  text={text[:60]!r}")
        log(f"[{model_dir.name}] {tag}: silence={sil[:30]!r}")
        del rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="补丁后的 s2tt 模型目录（断言对象）")
    ap.add_argument("--baseline-dir", default="", help="原始官方包（对照，只报告不断言）")
    ap.add_argument("--audio", default="", help="测试音频（默认模型包 test_wavs/ja1.wav）")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128, help="产品默认 128")
    ap.add_argument("--silence-secs", type=float, default=2.0)
    ap.add_argument("--max-kana", type=float, default=0.10)
    ap.add_argument("--min-cjk", type=float, default=0.50)
    ap.add_argument("--min-kana-transcribe", type=float, default=0.15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    model_dir = Path(args.model_dir)
    audio = Path(args.audio) if args.audio else model_dir / "test_wavs" / "ja1.wav"
    if not audio.is_file() and args.baseline_dir:
        audio = Path(args.baseline_dir) / "test_wavs" / "ja1.wav"
    if not audio.is_file():
        raise SystemExit(f"测试音频不存在: {audio}")
    log(f"补丁目录: {model_dir}\n[verify] 测试音频: {audio}")

    patched = run_suite(model_dir, audio, args)
    baseline = run_suite(Path(args.baseline_dir), audio, args) if args.baseline_dir else None

    checks = {
        "translate_via_hotwords": (patched["with_hotwords"]["cjk"] >= args.min_cjk
                                   and patched["with_hotwords"]["kana"] <= args.max_kana
                                   and len(patched["with_hotwords"]["text"]) > 4),
        "transcribe_without_hotwords": patched["no_hotwords"]["kana"] >= args.min_kana_transcribe,
        "silence_empty_both_modes": (len(patched["with_hotwords"]["silence_text"]) <= 2
                                     and len(patched["no_hotwords"]["silence_text"]) <= 2),
    }
    report = {"patched": patched, "baseline": baseline, "checks": checks,
              "thresholds": {"max_kana": args.max_kana, "min_cjk": args.min_cjk,
                             "min_kana_transcribe": args.min_kana_transcribe}}
    log("=" * 62)
    for k, v in checks.items():
        log(f"  {'✔' if v else '✘'} {k}")
    if baseline:
        log(f"  对照（原始包）: 带 hotwords 假名 {baseline['with_hotwords']['kana']:.0%}"
            f" / 汉字 {baseline['with_hotwords']['cjk']:.0%}"
            f"（T4 结论：context 通道不翻译，应仍以日语为主）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.out}")
    failed = [k for k, v in checks.items() if not v]
    if failed:
        log(f"未通过: {failed}")
        return 1
    log("✔ Gate 2 通过：sherpa-onnx 运行时上补丁模型三项行为全部成立")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
