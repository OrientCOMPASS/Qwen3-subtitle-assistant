#!/usr/bin/env python3
"""s2tt-e2e 实验的判定与三方对照报告。

三方 = ①master 产品管线（ASR→LLM 质检/摘要/翻译/审校）对同一视频的最终中文字幕（基线）
       ②本实验：微调 ASR 直出中文（无任何 LLM 后处理）
       ③本实验对照：同一适配器不带 context（应仍出日语，证明任务开关在真实视频上成立）

判定阈值（对直出中文的一组）：
  * cues >= max(8, 分钟数×4)          —— 产出密度不能塌
  * 假名占比 <= 5%、汉字占比 >= 50%   —— 确实直出了中文
  * 空输出丢弃率 <= 30%               —— language None 行为没有被滥用成「大段丢内容」
对照组：假名占比 >= 15%（仍是日语）。
质量差距（与基线的措辞差异）不做硬判定，逐桶并排打印供人工评估——
120 对样本训出来的裸质量本来就预期低于两段式管线，本实验的目的是量化差距。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def parse_srt(path: Path):
    if not path.is_file():
        return None
    cues = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) >= 2:
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
            if m:
                g = list(map(int, m.groups()))
                st = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                en = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append((st, en, "\n".join(lines[2:])))
    return cues


def srt_stats(cues):
    text = " ".join(t for _, _, t in cues)
    chars = [c for c in text if not c.isspace()]
    n = max(1, len(chars))
    end = max((e for _, e, _ in cues), default=0.0)
    return {
        "cues": len(cues),
        "kana": round(sum(1 for c in chars if KANA.match(c)) / n, 4),
        "cjk": round(sum(1 for c in chars if CJK.match(c)) / n, 4),
        "chars_per_min": round(len(chars) / max(0.01, end / 60), 1),
        "end": round(end, 1),
    }


def buckets(cues, width=20.0):
    out = {}
    for st, en, t in cues:
        out.setdefault(int(st // width), []).append(t.replace("\n", ""))
    return {k: "".join(v) for k, v in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvid", required=True)
    ap.add_argument("--modes", default="both")
    ap.add_argument("--out-dir", default="media/out")
    ap.add_argument("--master-dir", default="master_e2e")
    ap.add_argument("--bucket-secs", type=float, default=20.0)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    out_dir = Path(args.out_dir)
    fails: list[str] = []

    tr_stats_path = out_dir / "translate.stats.json"
    tr_stats = json.loads(tr_stats_path.read_text(encoding="utf-8")) if tr_stats_path.is_file() else None
    tc_stats_path = out_dir / "transcribe.stats.json"
    tc_stats = json.loads(tc_stats_path.read_text(encoding="utf-8")) if tc_stats_path.is_file() else None

    tr_cues = parse_srt(out_dir / "translate.srt")
    tc_cues = parse_srt(out_dir / "transcribe.srt")
    master_final = parse_srt(Path(args.master_dir) / f"video_{args.bvid}.srt")
    master_raw = parse_srt(Path(args.master_dir) / f"video_{args.bvid}.raw.srt")

    print("=" * 76)
    print(f"S2TT 端到端实验报告  bvid={args.bvid}  modes={args.modes}")
    print("=" * 76)

    rows = []
    if master_raw:
        rows.append(("① 产品管线 ASR 原文(日)", srt_stats(master_raw)))
    if tr_cues:
        rows.append(("② S2TT 直出中文(无 LLM)", srt_stats(tr_cues)))
    if master_final:
        rows.append(("③ 产品管线最终字幕(中)", srt_stats(master_final)))
    if tc_cues:
        rows.append(("④ 对照:适配器无 context", srt_stats(tc_cues)))
    if rows:
        print(f"{'来源':<26s}{'cues':>6s}{'假名':>8s}{'汉字':>8s}{'字/分':>8s}{'末尾(s)':>9s}")
        for name, s in rows:
            print(f"{name:<26s}{s['cues']:>6d}{s['kana']:>8.1%}{s['cjk']:>8.1%}"
                  f"{s['chars_per_min']:>8.1f}{s['end']:>9.1f}")
    if tr_stats:
        print(f"\nS2TT 直出: 音频 {tr_stats['audio_seconds']}s | 语音段 {tr_stats['segments_total']}"
              f"（丢弃 {tr_stats['segments_dropped']}，丢弃率 {tr_stats['drop_ratio']:.0%}）"
              f" | 推理 {tr_stats['inference_seconds']}s（RTF {tr_stats['rtf']}）")
        if tr_stats.get("dropped_examples"):
            print("丢弃段示例:", json.dumps(tr_stats["dropped_examples"][:4], ensure_ascii=False))

    # ---- 判定 ----
    print("\n---- 判定 ----")
    if args.modes in ("both", "translate"):
        if not tr_stats or not tr_cues:
            fails.append("translate 模式没有产出（SRT 或 stats 缺失）")
        else:
            minutes = tr_stats["audio_seconds"] / 60
            need = max(8, int(minutes * 4))
            checks = [
                (tr_stats["cues_final"] >= need,
                 f"cues {tr_stats['cues_final']} >= {need}（每分钟至少 4 条）"),
                (tr_stats["kana_ratio"] <= 0.05,
                 f"假名占比 {tr_stats['kana_ratio']:.1%} <= 5%"),
                (tr_stats["cjk_ratio"] >= 0.50,
                 f"汉字占比 {tr_stats['cjk_ratio']:.1%} >= 50%"),
                (tr_stats["drop_ratio"] <= 0.30,
                 f"空输出丢弃率 {tr_stats['drop_ratio']:.0%} <= 30%"),
            ]
            for ok, desc in checks:
                print(f"  {'✔' if ok else '✘'} {desc}")
                if not ok:
                    fails.append(desc)
    if args.modes in ("both", "transcribe"):
        if not tc_stats or not tc_cues:
            fails.append("transcribe 对照组没有产出")
        else:
            ok = tc_stats["kana_ratio"] >= 0.15
            print(f"  {'✔' if ok else '✘'} 对照组假名占比 {tc_stats['kana_ratio']:.1%} >= 15%（任务开关生效，仍是日语）")
            if not ok:
                fails.append("对照组不再是日语（任务开关失效或遗忘）")

    # ---- 三方对照（20s 桶并排）----
    if tr_cues:
        print("\n---- ② S2TT 直出 vs ③ 产品管线最终字幕（按 20s 桶并排，人工评估质量差距）----")
        b2 = buckets(tr_cues, args.bucket_secs)
        b3 = buckets(master_final, args.bucket_secs) if master_final else {}
        for k in sorted(set(b2) | set(b3)):
            mm, ss = divmod(int(k * args.bucket_secs), 60)
            print(f"[{mm:02d}:{ss:02d}] 直出: {b2.get(k, '(无)')[:66]}")
            if b3:
                print(f"        产品: {b3.get(k, '(无)')[:66]}")
    if master_raw and tr_cues:
        print("\n---- ① 产品 ASR 日语原文 vs ② 直出中文（同一时间桶，看语义对应）----")
        b1 = buckets(master_raw, args.bucket_secs)
        b2 = buckets(tr_cues, args.bucket_secs)
        for k in sorted(set(b1) | set(b2))[:8]:
            mm, ss = divmod(int(k * args.bucket_secs), 60)
            print(f"[{mm:02d}:{ss:02d}] 日语: {b1.get(k, '(无)')[:60]}")
            print(f"        中文: {b2.get(k, '(无)')[:60]}")
    if tc_cues:
        print("\n---- ④ 对照组前 6 条（不带 context，应为日语转写）----")
        for st, en, t in tc_cues[:6]:
            print(f"  {st:7.2f}-{en:7.2f}s {t[:56]}")

    print("\n" + "=" * 76)
    if fails:
        for f in fails:
            print("✘", f)
        return 1
    print("✔ 判定通过：微调 ASR 在真实视频上直出中文字幕（无 LLM 后处理），任务开关双模式成立。")
    print("  与产品两段式管线的质量差距见上方并排对照（措辞级差距属 120 对样本的预期内）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
