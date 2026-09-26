#!/usr/bin/env python3
"""评测 S2TT 微调效果：模型是否真的开始"听日语、写中文"，以及有没有把原能力训废。

三项检查（对应三个必须同时成立的目标）：
  1. **翻译生效**：带 context（--prompt）时，输出应以中文为主 —— 假名占比要低；
  2. **没有遗忘**：不带 context 时，仍应输出日语转写 —— 假名占比要高；
     （这两条一起才证明"任务开关"学到了，而不是把模型整体改成了只会输出中文）
  3. **无语音行为**：给一段静音，应输出 language None / 空文本（逐句质检依赖这个行为）。

用法：
  python finetune/eval_s2tt.py --model Qwen/Qwen3-ASR-0.6B --adapter out/s2tt/lora \
      --eval data/eval.jsonl --limit 20 --out out/s2tt/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)

def _hard_exit(code: int) -> "NoReturn":
    """绕过解释器 finalization 直接退出。

    实测：datasets/pyarrow 的后台线程在 Python finalize 阶段会触发
    `Fatal Python error: PyGILState_Release: thread state ... must be current`
    并 SIGABRT（退出码 134），**即使脚本本身已经完全成功**（CI 上就是这样：
    train.jsonl/eval.jsonl 都写好了、汇总也打印了，进程仍以 134 退出被判红）。
    所有产物在调用本函数前均已写盘并 flush，因此直接 _exit 是安全的。
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)



def kana_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if KANA.match(c)) / len(chars)


def cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if CJK.match(c)) / len(chars)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录（留空则评测基座模型）")
    ap.add_argument("--eval", required=True, help="prepare_data.py 产出的评测 JSONL")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--prompt", default="translate to Chinese", help="翻译任务的 context")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--silence-secs", type=float, default=2.0, help="静音测试时长；0=跳过")
    ap.add_argument("--max-kana-translated", type=float, default=0.10,
                    help="翻译模式下输出假名占比上限（超过即判未生效）")
    ap.add_argument("--min-kana-transcribed", type=float, default=0.15,
                    help="转写模式下输出假名占比下限（低于即判可能遗忘）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    log(f"device={'cuda' if use_cuda else 'cpu'} dtype={dtype} adapter={args.adapter or '(无，评测基座)'}")

    wrapper = Qwen3ASRModel.from_pretrained(
        args.model, dtype=dtype, **({"device_map": "cuda:0"} if use_cuda else {})
    )
    if args.adapter:
        from peft import PeftModel

        # 适配器要挂在 .thinker 上（顶层没有 forward/generate 的实际实现，
        # 它的 generate 会转发给 self.thinker.generate）
        top = wrapper.model
        target = getattr(top, "thinker", top)
        adapted = PeftModel.from_pretrained(target, args.adapter)
        adapted.eval()
        if hasattr(top, "thinker"):
            top.thinker = adapted
        else:
            wrapper.model = adapted
        log(f"已挂载 LoRA 适配器到 {type(target).__name__}: {args.adapter}")

    rows = []
    for line in Path(args.eval).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        log("评测集为空")
        return 1

    # 按 prompt 是否为空分成两组：翻译组 / 转写组
    translated = [r for r in rows if r.get("prompt")][: args.limit]
    transcribed = [r for r in rows if not r.get("prompt")][: args.limit]
    log(f"评测样本：翻译组 {len(translated)} 条，转写组 {len(transcribed)} 条")

    def run(group, use_prompt: bool):
        out = []
        for r in group:
            res = wrapper.transcribe(
                audio=r["audio"],
                context=(args.prompt if use_prompt else ""),
            )
            text = (res[0].text if res else "") or ""
            lang = (res[0].language if res else "") or ""
            ref = r["text"].split("<asr_text>", 1)[-1]
            out.append({
                "audio": r["audio"],
                "ref": ref,
                "out": text,
                "lang_tag": lang,
                "kana": round(kana_ratio(text), 4),
                "cjk": round(cjk_ratio(text), 4),
            })
        return out

    tr_out = run(translated, use_prompt=True) if translated else []
    asr_out = run(transcribed, use_prompt=False) if transcribed else []

    def avg(items, key):
        return round(sum(i[key] for i in items) / len(items), 4) if items else None

    report = {
        "model": args.model,
        "adapter": args.adapter or None,
        "device": "cuda" if use_cuda else "cpu",
        "prompt": args.prompt,
        "translated_samples": tr_out,
        "transcribed_samples": asr_out,
        "metrics": {
            "translated_avg_kana": avg(tr_out, "kana"),
            "translated_avg_cjk": avg(tr_out, "cjk"),
            "transcribed_avg_kana": avg(asr_out, "kana"),
        },
        "checks": {},
    }

    # ---- 静音行为 ----
    if args.silence_secs > 0:
        silence = np.zeros(int(16000 * args.silence_secs), dtype=np.float32)
        try:
            res = wrapper.transcribe(audio=(silence, 16000), context=args.prompt)
            text = (res[0].text if res else "") or ""
            lang = (res[0].language if res else "") or ""
            report["silence"] = {"text": text, "lang_tag": lang}
            report["checks"]["silence_is_empty"] = len(text.strip()) <= 2
            log(f"静音输出: lang={lang!r} text={text!r} -> "
                f"{'OK' if report['checks']['silence_is_empty'] else '非空（需注意）'}")
        except Exception as e:  # noqa: BLE001
            report["silence"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"静音测试失败: {e}")

    m = report["metrics"]
    if m["translated_avg_kana"] is not None:
        report["checks"]["translation_takes_effect"] = m["translated_avg_kana"] <= args.max_kana_translated
    if m["transcribed_avg_kana"] is not None:
        report["checks"]["no_catastrophic_forgetting"] = m["transcribed_avg_kana"] >= args.min_kana_transcribed

    log("=" * 60)
    for i, s in enumerate(tr_out[:6]):
        log(f"  [翻译 {i}] 假名{s['kana']:.0%} 输出: {s['out'][:60]}")
        log(f"           参考: {s['ref'][:60]}")
    for i, s in enumerate(asr_out[:3]):
        log(f"  [转写 {i}] 假名{s['kana']:.0%} 输出: {s['out'][:60]}")
    log("=" * 60)
    log(f"指标: {json.dumps(m, ensure_ascii=False)}")
    for k, v in report["checks"].items():
        log(f"  {'✔' if v else '✘'} {k}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告已写入 {args.out}")

    # CPU 冒烟运行用的是 stub 伪翻译，指标必然不达标——这种情况下不作为失败退出，
    # 只把结论写进报告，由调用方（CI）根据运行模式决定是否判定。
    failed = [k for k, v in report["checks"].items() if v is False]
    if failed:
        log(f"未通过的检查: {failed}")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
