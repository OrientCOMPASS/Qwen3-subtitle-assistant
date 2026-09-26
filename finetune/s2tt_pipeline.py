#!/usr/bin/env python3
"""端到端 S2TT 字幕管线（无 LLM 后处理）——「微调 ASR 融合进主程序」的对照实验。

模拟 Rust 主程序的工作流，但把「ASR(日语) → LLM 质检 → LLM 摘要 → LLM 翻译 → LLM 审校」
五段替换为**一段**：微调后的 Qwen3-ASR 在 context="translate to Chinese" 下直接输出中文。

    媒体文件 ──ffmpeg──▶ 16k f32le PCM ──silero-vad──▶ 语音段
                                                        │  每段
                                                        ▼
                     Qwen3-ASR + LoRA（translate 模式直出中文 / transcribe 模式出日语对照）
                                                        │  空输出(language None)段丢弃
                                                        ▼
                     CJK 折行(宽度44) + 长 cue 按句读拆分(≤15s) ──▶ SRT

与产品管线的对应关系（便于公平对比 master CI 的 T1 产物）：
  * VAD 参数近似产品的 Silero 配置；语音段间隔 < --merge-gap 合并、超过 --max-seg-secs
    硬拆（产品受 sherpa-onnx KV=512 限制约 17~27s，transformers 运行时本可更长，
    但为了对齐产品形态仍按 27s 拆）；
  * 「静音/无语音 → language None + 空文本 → 丢弃」正是微调时专门保住的行为，
    这里用它替代产品的 LLM 逐句质检剔除噪音段；
  * 折行/拆分逻辑对齐 src/srt.rs 的显示宽度规则（CJK 计 2）。

**没有任何 LLM 参与**：不质检、不摘要、不翻译、不审校。产出质量即 S2TT 单段的裸质量，
用于回答「省掉 1.7B 后处理段，字幕还能不能看」。

用法（CI 见 .github/workflows/s2tt-e2e.yml）：
  python finetune/s2tt_pipeline.py --media media/test.mp4 --adapter art/out/real/lora \
      --context "translate to Chinese" --out media/out/translate.srt \
      --stats media/out/translate.stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple

DEFAULT_CONTEXT = "translate to Chinese"


def log(msg: str) -> None:
    print(f"[pipe] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    """绕过解释器 finalization 直接退出（datasets/pyarrow 后台线程在 finalize 阶段
    会 SIGABRT，即使脚本已成功——与 sft_lora.py/eval_s2tt.py 同一实测坑）。"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


# ------------------------------------------------------------------ 纯函数（可单测）

def display_width(s: str) -> int:
    """显示宽度：全角/宽字符计 2，其余计 1（对齐 src/srt.rs）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in s)


def wrap_lines(text: str, max_width: int = 44) -> str:
    """按显示宽度贪心折行；CJK 可在任意字符处断行，ASCII 单词不从中拆开。

    收口标点（。！？，等）预先并入前一个 token——既不会行首出现孤零零的标点
    （禁则处理），也不会像「挂到上一行行尾」那样突破 max_width
    （scripts/check_e2e.py 对行宽是硬断言）。
    """
    closing = set("。．，、；：！？,.;:!?）)》」』】％%")
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*|\s+|.", text)
    # 把收口标点粘到最近的非空白 token 上
    glued: List[str] = []
    for tok in tokens:
        if tok.strip() and all(c in closing for c in tok.strip()):
            j = len(glued) - 1
            while j >= 0 and not glued[j].strip():
                glued.pop(j)
                j -= 1
            if j >= 0:
                glued[j] += tok
                continue
        glued.append(tok)
    lines: List[str] = []
    cur, cur_w = "", 0
    for tok in glued:
        if not tok.strip():
            # 空白：行首不放、连续折叠为一个空格（英文单词间隔要保留）
            if cur and not cur.endswith(" "):
                if cur_w + 1 > max_width:
                    lines.append(cur)
                    cur, cur_w = "", 0
                else:
                    cur += " "
                    cur_w += 1
            continue
        tw = display_width(tok)
        if cur and cur_w + tw > max_width:
            lines.append(cur)
            cur, cur_w = "", 0
        cur += tok
        cur_w += tw
    if cur.strip():
        lines.append(cur)
    return "\n".join(l.strip() for l in lines)


def fmt_ts(secs: float) -> str:
    ms = int(round(secs * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_long_cue(start: float, end: float, text: str,
                   max_secs: float = 15.0) -> List[Tuple[float, float, str]]:
    """超长 cue 按句读拆分，时间按字符占比分配（对齐产品的长 cue 兜底）。"""
    dur = end - start
    flat = text.replace("\n", "").strip()
    if dur <= max_secs or len(flat) < 8:
        return [(start, end, flat)]
    mid = start + dur / 2

    def pick(seps: str) -> List[int]:
        return [m.end() for m in re.finditer(seps, flat)]

    cands = pick(r"[。！？；!?;]") or pick(r"[，、,：:]")
    if cands:
        best = min(cands, key=lambda i: abs(start + dur * i / len(flat) - mid))
        if best < 4 or best > len(flat) - 4:
            cands = []
    if not cands:
        best = len(flat) // 2
    parts = [flat[:best], flat[best:]]
    out: List[Tuple[float, float, str]] = []
    t = start
    for p in parts:
        if not p.strip():
            continue
        d = dur * len(p) / len(flat)
        out.extend(split_long_cue(t, t + d, p, max_secs))
        t += d
    return out


def write_srt(cues: List[Tuple[float, float, str]], path: Path,
              max_width: int = 44, max_cue_secs: float = 15.0) -> int:
    """cue 列表 -> SRT 文件（折行 + 长 cue 拆分），返回最终条数。"""
    final: List[Tuple[float, float, str]] = []
    for st, en, tx in cues:
        final.extend(split_long_cue(st, en, tx, max_cue_secs))
    body = []
    for i, (st, en, tx) in enumerate(final, 1):
        body.append(f"{i}\n{fmt_ts(st)} --> {fmt_ts(en)}\n{wrap_lines(tx, max_width)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")
    return len(final)


def merge_segments(segs: List[Tuple[float, float]], merge_gap: float,
                   max_seg: float) -> List[Tuple[float, float]]:
    """相邻语音段间隔 < merge_gap 合并；超过 max_seg 硬拆成等长块。"""
    merged: List[Tuple[float, float]] = []
    for st, en in segs:
        if merged and st - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((st, en))
    out: List[Tuple[float, float]] = []
    for st, en in merged:
        dur = en - st
        if dur <= max_seg:
            out.append((st, en))
            continue
        n = int(dur // max_seg) + 1
        step = dur / n
        for k in range(n):
            out.append((st + k * step, min(en, st + (k + 1) * step)))
    return out


# ------------------------------------------------------------------ 媒体解码 + VAD

def decode_media(path: str, max_seconds: float = 0.0) -> "np.ndarray":
    """ffmpeg 管道直出 16kHz f32le 单声道 PCM（与产品的流式管线同款，零中间文件）。"""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
    log(f"解码: {' '.join(cmd[:6])} ...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[:500]}")
    arr = np.frombuffer(proc.stdout, dtype=np.float32)
    log(f"解码完成: {len(arr)/16000:.1f}s 音频")
    return arr


def vad_segments(audio: "np.ndarray", merge_gap: float, max_seg: float) -> List[Tuple[float, float]]:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio), model, sampling_rate=16000,
        threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=300,
        speech_pad_ms=100, return_seconds=True,
    )
    raw = [(float(t["start"]), float(t["end"])) for t in ts]
    segs = merge_segments(raw, merge_gap, max_seg)
    log(f"VAD: 原始语音段 {len(raw)} 个 -> 合并/拆分后 {len(segs)} 个，"
        f"语音总时长 {sum(e - s for s, e in segs):.1f}s")
    return segs


# ------------------------------------------------------------------ 模型

def load_model(model_id: str, adapter: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    t0 = time.time()
    wrapper = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        # 适配器挂 .thinker（顶层无 forward/generate 实现，转发给 thinker）——与 eval_s2tt.py 一致
        top = wrapper.model
        target = getattr(top, "thinker", top)
        adapted = PeftModel.from_pretrained(target, adapter)
        adapted.eval()
        if hasattr(top, "thinker"):
            top.thinker = adapted
        else:
            wrapper.model = adapted
        log(f"已挂载 LoRA: {adapter}")
    log(f"模型就绪，用时 {time.time() - t0:.1f}s")
    return wrapper


# ------------------------------------------------------------------ 主流程

def kana_cjk_ratios(text: str) -> Tuple[float, float]:
    kana = re.compile(r"[\u3040-\u30ff]")
    cjk = re.compile(r"[\u4e00-\u9fff]")
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0, 0.0
    return (sum(1 for c in chars if kana.match(c)) / len(chars),
            sum(1 for c in chars if cjk.match(c)) / len(chars))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录（留空 = 基座模型）")
    ap.add_argument("--context", default=DEFAULT_CONTEXT, help="翻译模式的任务开关；留空 = 转写模式")
    ap.add_argument("--out", required=True, help="输出 SRT 路径")
    ap.add_argument("--stats", default="", help="统计 JSON 路径")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="只处理前 N 秒（0 = 全部）")
    ap.add_argument("--merge-gap", type=float, default=0.4, help="相邻语音段间隔小于此值则合并（秒）")
    ap.add_argument("--max-seg-secs", type=float, default=27.0, help="单段最长秒数（对齐产品 KV 限制）")
    ap.add_argument("--max-line-width", type=int, default=44)
    ap.add_argument("--max-cue-secs", type=float, default=15.0)
    ap.add_argument("--limit-segments", type=int, default=0, help=">0 时只处理前 N 段（调试）")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    mode = "translate" if args.context.strip() else "transcribe"
    log(f"模式: {mode}（context={args.context!r}）")

    audio = decode_media(args.media, args.max_seconds)
    audio_secs = len(audio) / 16000.0
    segs = vad_segments(audio, args.merge_gap, args.max_seg_secs)
    if args.limit_segments > 0:
        segs = segs[: args.limit_segments]

    wrapper = load_model(args.model, args.adapter)

    cues: List[Tuple[float, float, str]] = []
    dropped: List[dict] = []
    t_inf = 0.0
    for i, (st, en) in enumerate(segs):
        seg = audio[int(st * 16000): int(en * 16000)]
        t0 = time.time()
        res = wrapper.transcribe(audio=(seg, 16000), context=args.context)
        t_inf += time.time() - t0
        text = ((res[0].text if res else "") or "").strip()
        lang = ((res[0].language if res else "") or "").strip()
        # 「language None + 空文本」= 无语音/噪音段 -> 丢弃（微调时专门保住的行为，
        # 替代产品管线的 LLM 逐句质检）
        if not text or not lang:
            dropped.append({"start": round(st, 2), "end": round(en, 2),
                            "lang": lang, "text": text[:30]})
            log(f"  [{i+1:3d}/{len(segs)}] {st:7.2f}-{en:7.2f}s  丢弃（lang={lang!r} text={text[:20]!r}）")
            continue
        cues.append((st, en, text))
        log(f"  [{i+1:3d}/{len(segs)}] {st:7.2f}-{en:7.2f}s  lang={lang}  {text[:46]}")

    n_final = write_srt(cues, Path(args.out), args.max_line_width, args.max_cue_secs)
    all_text = " ".join(t for _, _, t in cues)
    kana, cjk = kana_cjk_ratios(all_text)
    stats = {
        "media": args.media,
        "mode": mode,
        "context": args.context,
        "adapter": args.adapter or None,
        "audio_seconds": round(audio_secs, 1),
        "segments_total": len(segs),
        "segments_kept": len(cues),
        "segments_dropped": len(dropped),
        "drop_ratio": round(len(dropped) / max(1, len(segs)), 3),
        "cues_final": n_final,
        "kana_ratio": round(kana, 4),
        "cjk_ratio": round(cjk, 4),
        "chars_per_min": round(len([c for c in all_text if not c.isspace()]) / max(0.01, audio_secs / 60), 1),
        "inference_seconds": round(t_inf, 1),
        "rtf": round(t_inf / max(0.01, audio_secs), 3),
        "dropped_examples": dropped[:8],
        "cues_preview": [{"start": round(s, 2), "end": round(e, 2), "text": t} for s, e, t in cues[:12]],
    }
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=" * 62)
    log(f"音频 {stats['audio_seconds']}s | 语音段 {len(segs)}（丢弃 {len(dropped)}，"
        f"丢弃率 {stats['drop_ratio']:.0%}）| SRT {n_final} 条")
    log(f"假名 {kana:.1%} / 汉字 {cjk:.1%} | 推理 {t_inf:.0f}s（RTF {stats['rtf']}）")
    log(f"SRT -> {args.out}" + (f" | stats -> {args.stats}" if args.stats else ""))
    return 0


if __name__ == "__main__":
    _hard_exit(main())
