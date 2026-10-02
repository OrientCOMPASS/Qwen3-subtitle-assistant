#!/usr/bin/env python3
"""e2e 断言：S2TT 直出字幕的语言/密度/排版检查（替代 ci.yml 里的内联脚本——
pwsh 不支持 bash heredoc，教训见 commit 历史）。

用法：
  # 直出中文断言（T1）：媒体列表分号分隔，检查每个同名 .srt
  python scripts/check_s2tt_e2e.py translate "media/a.m4a;media/b.m4a"
  # 转写源语言断言（T1b）：目录或单个 srt，要求假名占比 >= 15%（仍是日语）
  python scripts/check_s2tt_e2e.py transcribe media/t1b
"""
from __future__ import annotations

import pathlib
import re
import sys

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_cues(p: pathlib.Path):
    out = []
    for block in p.read_text(encoding="utf-8").strip().split("\n\n"):
        ls = [l for l in block.splitlines() if l.strip()]
        if len(ls) >= 3 and "-->" in ls[1]:
            m = TS.match(ls[1])
            if not m:
                continue
            g = list(map(int, m.groups()))
            st = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            en = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            out.append((st, en, "\n".join(ls[2:])))
    return out


def width(l: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in l)


def check_translate(media_list: str) -> int:
    bad = 0
    # 支持两种形态：分号分隔的媒体文件列表（检查同名 .srt）；或目录（检查目录下
    # 所有非 .raw.srt 的 srt —— T1c 的 --output-dir 产物形态）
    p = pathlib.Path(media_list)
    if p.is_dir():
        srts = sorted(x for x in p.glob("*.srt") if not x.name.endswith(".raw.srt"))
        if not srts:
            print(f"✘ {p} 下没有 srt")
            return 1
        targets = srts
    else:
        targets = []
        for f in media_list.split(";"):
            if not f.strip():
                continue
            srt = pathlib.Path(f).with_suffix(".srt")
            if not srt.is_file():
                print(f"✘ {srt} 不存在")
                bad += 1
                continue
            targets.append(srt)
    for srt in targets:
        cs = parse_cues(srt)
        txt = "".join(t for _, _, t in cs)
        ch = [c for c in txt if not c.isspace()]
        n = max(1, len(ch))
        kana = sum(1 for c in ch if KANA.match(c)) / n
        cjk = sum(1 for c in ch if CJK.match(c)) / n
        end = max((e for _, e, _ in cs), default=0)
        minutes = max(0.05, end / 60)
        widest = max((max(width(l) for l in t.split("\n")) for _, _, t in cs), default=0)
        longest = max((e - s for s, e, _ in cs), default=0)
        # v0.6.2 断句标准（双模型时代默认恢复）：≤7s / ≤40 字符 / 行宽 40
        most_chars = max((len(t.replace("\n", "")) for _, _, t in cs), default=0)
        checks = [
            (len(cs) >= minutes * 4, f"cues {len(cs)} >= {minutes*4:.0f}（密度 >=4/分钟）"),
            (kana <= 0.05, f"假名占比 {kana:.1%} <= 5%（语言定向）"),
            (cjk >= 0.50, f"汉字占比 {cjk:.1%} >= 50%（确实中文）"),
            (widest <= 40, f"最宽行 {widest} <= 40（折行）"),
            (longest <= 8.0, f"最长 cue {longest:.1f}s <= 8s（秒数断句）"),
            (most_chars <= 44, f"最长条 {most_chars} 字 <= 44（字符断句，上限40+硬切余量）"),
        ]
        print(f"== {srt}")
        for ok, d in checks:
            print(("  ✔ " if ok else "  ✘ ") + d)
            bad += (not ok)
        print("  首条:", (cs[0][2][:40] if cs else "(空)"))
    return bad


def check_transcribe(where: str) -> int:
    p = pathlib.Path(where)
    srts = sorted(p.glob("*.srt")) if p.is_dir() else [p]
    if not srts:
        print(f"✘ {where} 下没有 srt")
        return 1
    bad = 0
    for srt in srts:
        txt = "\n".join(t for _, _, t in parse_cues(srt))
        ch = [c for c in txt if not c.isspace()]
        n = max(1, len(ch))
        kana = sum(1 for c in ch if KANA.match(c)) / n
        ok = kana >= 0.15
        print(f"{'✔' if ok else '✘'} {srt.name}: 假名占比 {kana:.1%}（应 >=15%，仍是日语）")
        print("  样例:", txt[:60].replace("\n", " "))
        bad += (not ok)
    return bad


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    mode, arg = sys.argv[1], sys.argv[2]
    if mode == "translate":
        bad = check_translate(arg)
    elif mode == "transcribe":
        bad = check_transcribe(arg)
    else:
        print(f"未知 mode: {mode}")
        return 2
    print(f"断言失败数: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
