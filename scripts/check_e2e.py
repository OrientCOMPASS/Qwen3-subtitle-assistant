#!/usr/bin/env python3
"""e2e 结果断言：把「产出了 .srt」升级为「产出了合格的字幕」。

旧版 CI 的唯一判据是"至少有一个 .srt 文件"，因此：翻译整批回退成原文、
长视频在摘要阶段 abort、字幕一条 60 秒——这些都能"通过"。
本脚本按内容断言，CI 才有回归价值。

用法:
    python scripts/check_e2e.py --name T1 --srt media/a.srt --log media/run1.log \
        --min-cues 8 --target-lang zh --max-cue-secs 8.5 --max-line-width 44 \
        --expect-qc --max-untranslated 0
    python scripts/check_e2e.py --name T2 --srt out.srt --log run2.log \
        --min-cues 120 --target-lang zh --min-summary-chunks 2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")          # 平假名 + 片假名
HANGUL = re.compile(r"[\uac00-\ud7af]")
LATIN = re.compile(r"[A-Za-z]")
CJK = re.compile(r"[\u4e00-\u9fff]")

# 日志里代表"出问题了"的模式
BAD_PATTERNS = [
    r"GGML_ASSERT",
    r"panicked at",
    r"abort\(\)",
    r"整批翻译失败",
    r"翻译彻底失败",
    r"Prompt 长度（\d+ tokens）超出上下文",
]


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    cues = []
    for block in text.split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue
        a, b = lines[ts_idx].split("-->")
        try:
            start, end = _ms(a), _ms(b)
        except Exception:  # noqa: BLE001
            continue
        body = "\n".join(lines[ts_idx + 1:]).strip()
        if not body:
            continue
        cues.append({"start": start, "end": end, "text": body, "index": lines[0] if ts_idx else ""})
    return cues


def _ms(s: str) -> int:
    s = s.strip().replace(".", ",")
    hms, _, ms = s.partition(",")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return (h * 3600 + m * 60 + sec) * 1000 + int(ms or 0)


def line_width(line: str) -> int:
    """显示宽度：CJK/全角计 2，其余计 1（与 src/srt.rs 的 display_width 一致）。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in line)


def script_ratio(text: str) -> dict[str, float]:
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)
    return {
        "kana": sum(1 for c in chars if KANA.match(c)) / n,
        "hangul": sum(1 for c in chars if HANGUL.match(c)) / n,
        "cjk": sum(1 for c in chars if CJK.match(c)) / n,
        "latin": sum(1 for c in chars if LATIN.match(c)) / n,
    }


def log_stats(log_text: str) -> dict:
    # 只匹配逐条/逐批的告警行，避开收尾统计行里的同名字样（否则统计行自己会被算成一次失败）
    out = {
        "untranslated": len(re.findall(r"无译文，回退原文", log_text))
                        + len(re.findall(r"\[翻译-回退\]", log_text)),
        "positional": len(re.findall(r"改用位置兜底", log_text)),
        "fix_rejected": len(re.findall(r"QC⚠ 纠正被拒", log_text)),
        "qc_dropped": len(re.findall(r"QC✂ 丢弃", log_text)),
        "qc_fixed": len(re.findall(r"QC✎ 纠正", log_text)),
        "retries": len(re.findall(r"\[翻译\] 第 \d+ 次|\[摘要\] 第 \d+ 次|\[QC\] 第 \d+/\d+ 次", log_text)),
        "bad": [p for p in BAD_PATTERNS if re.search(p, log_text)],
    }
    m = re.search(r"摘要分块 (\d+)", log_text)
    out["summary_chunks"] = int(m.group(1)) if m else None
    m = re.search(r"质检 (\d+) 句：保留 (\d+)，纠正 (\d+)，丢弃 (\d+)，解析失败兜底 (\d+)，纠正被拒 (\d+)", log_text)
    out["qc"] = tuple(int(x) for x in m.groups()) if m else None
    m = re.search(r"排版完成：(\d+) 条 -> (\d+) 条", log_text)
    out["layout"] = (int(m.group(1)), int(m.group(2))) if m else None
    m = re.search(r"ASR 流程结束，最终保留 (\d+) 条字幕", log_text)
    out["asr_segments"] = int(m.group(1)) if m else None
    return out


def main() -> int:
    # Windows 控制台默认 cp1252：中文断言输出会抛 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--min-cues", type=int, default=1)
    ap.add_argument("--max-cue-secs", type=float, default=0.0, help="0=不检查")
    ap.add_argument("--max-line-width", type=int, default=0, help="0=不检查")
    ap.add_argument("--target-lang", default="zh", choices=["zh", "any"])
    ap.add_argument("--expect-qc", action="store_true")
    ap.add_argument("--min-summary-chunks", type=int, default=0)
    ap.add_argument("--max-untranslated", type=int, default=0)
    ap.add_argument("--max-kana-ratio", type=float, default=0.03)
    ap.add_argument("--max-foreign-cues", type=int, default=-1,
                    help="允许「基本没翻译」的 cue 条数（该 cue 假名占比 > 40%%）；-1=不检查。"
                         "比全局假名占比更准：专有名词保留原文不会被误判，整批照抄则一定被抓到")
    ap.add_argument("--expect-log-contains", action="append", default=[],
                    help="日志中必须出现的字样（可多次），用于验证某条代码路径确实被走到")
    args = ap.parse_args()

    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(("  ✔ " if ok else "  ✘ ") + msg)
        if not ok:
            fails.append(msg)

    print(f"===== 断言 {args.name}: {args.srt} =====")
    srt_path = Path(args.srt)
    if not srt_path.is_file():
        print(f"  ✘ 字幕文件不存在: {srt_path}")
        return 1
    cues = parse_srt(srt_path)
    full_text = "\n".join(c["text"] for c in cues)
    ratios = script_ratio(full_text)
    print(f"  · {len(cues)} 条字幕, {len(full_text)} 字, 文字构成: "
          f"汉字 {ratios['cjk']:.0%} 假名 {ratios['kana']:.0%} 拉丁 {ratios['latin']:.0%}")

    check(len(cues) >= args.min_cues, f"字幕条数 {len(cues)} >= {args.min_cues}")
    check(all(c["text"].strip() for c in cues), "所有字幕文本非空")

    if args.max_cue_secs > 0:
        worst = max(cues, key=lambda c: c["end"] - c["start"])
        worst_secs = (worst["end"] - worst["start"]) / 1000
        check(worst_secs <= args.max_cue_secs,
              f"最长单条 {worst_secs:.1f}s <= {args.max_cue_secs}s（内容: {worst['text'][:30]}…）")

    if args.max_line_width > 0:
        widest_w, widest_cue = max(
            ((max(line_width(l) for l in c["text"].split("\n")), c) for c in cues),
            key=lambda x: x[0],
        )
        check(widest_w <= args.max_line_width,
              f"最宽行 {widest_w} <= {args.max_line_width}（内容: {widest_cue['text'][:30]}…）")

    # 时序合法性
    check(all(c["end"] > c["start"] for c in cues), "所有 cue 的 end > start")
    overlaps = sum(1 for a, b in zip(cues, cues[1:]) if b["start"] < a["end"] - 1)
    check(overlaps == 0, f"相邻 cue 无重叠（重叠 {overlaps} 处）")

    if args.target_lang == "zh" and args.max_foreign_cues >= 0:
        foreign = [c for c in cues if script_ratio(c["text"])["kana"] > 0.40]
        check(len(foreign) <= args.max_foreign_cues,
              f"未翻译的 cue {len(foreign)} 条 <= {args.max_foreign_cues}"
              + (f"（例: {foreign[0]['text'][:40]}…）" if foreign else ""))

    if args.target_lang == "zh":
        check(ratios["kana"] <= args.max_kana_ratio,
              f"假名占比 {ratios['kana']:.1%} <= {args.max_kana_ratio:.0%}（说明已译成中文而非残留日语原文）")
        check(ratios["cjk"] >= 0.3, f"汉字占比 {ratios['cjk']:.0%} >= 30%")

    if args.log:
        log_path = Path(args.log)
        if not log_path.is_file():
            check(False, f"日志文件存在: {log_path}")
        else:
            st = log_stats(log_path.read_text(encoding="utf-8", errors="replace"))
            print(f"  · 日志统计: {st}")
            check(not st["bad"], f"日志无致命模式（命中: {st['bad']}）")
            raw_log = log_path.read_text(encoding="utf-8", errors="replace")
            for needle in args.expect_log_contains:
                check(needle in raw_log, f"日志包含 {needle!r}")
            check(st["untranslated"] <= args.max_untranslated,
                  f"回退原文 {st['untranslated']} 条 <= {args.max_untranslated}")
            if args.expect_qc:
                check(st["qc"] is not None, "日志含质检统计行")
                if st["qc"]:
                    total, kept, fixed, dropped, failed, rejected = st["qc"]
                    # 每句必然落入 kept/fixed/dropped 之一（fail-open 与"纠正被拒"都计入 kept）
                    check(total == kept + fixed + dropped,
                          f"质检计数自洽: 共 {total} 句 = 保留 {kept} + 纠正 {fixed} + 丢弃 {dropped}"
                          f"（解析兜底 {failed}，纠正被拒 {rejected}）")
                    check(total > 0, f"质检覆盖 {total} 句 > 0")
            if args.min_summary_chunks:
                check((st["summary_chunks"] or 0) >= args.min_summary_chunks,
                      f"摘要分块 {st['summary_chunks']} >= {args.min_summary_chunks}（证明长文本走了 map-reduce）")

    print(f"===== {args.name}: {'通过' if not fails else '失败 ' + str(len(fails)) + ' 项'} =====")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
