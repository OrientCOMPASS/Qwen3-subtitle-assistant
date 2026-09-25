#!/usr/bin/env python3
"""为 e2e 测试获取媒体（默认：bilibili 上的日语视频，只取音频轨）。

bilibili 对数据中心 IP 有**请求级**风控（表现为 HTTP 412）：同一 IP 连续请求几次后
就会被拒。GitHub Actions runner 每次都是新 IP，而本脚本一次只需要 3 个请求
（首页拿 cookie -> view 取 cid/时长 -> playurl 取音频流），因此可用；
下载成功后由 CI 用 actions/cache 缓存，后续运行完全不再访问 bilibili。

用法:
    python scripts/fetch_media.py --out media --bvid BV1ccdQY8Ee8 --bvid BV16r421g797
    python scripts/fetch_media.py --out media --url https://example.com/a.mp4
    python scripts/fetch_media.py --out media --min-secs 60 --max-secs 420

成功时最后一行输出（供 CI 解析）:
    MEDIA_OK {"source":"bilibili","id":"BV...","file":"media/BV....m4a","duration":282,"title":"..."}
失败时退出码非 0，并输出 MEDIA_FAIL {...}。只依赖标准库；yt-dlp 存在时作为兜底。
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
HOME = "https://www.bilibili.com/"


def log(msg: str) -> None:
    print(f"[media] {msg}", flush=True)


class Bili:
    def __init__(self) -> None:
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cj))
        self.opener.addheaders = [("User-Agent", UA), ("Referer", HOME)]

    def get(self, url: str, referer: str = HOME, tries: int = 3, sleep: float = 4.0):
        last = None
        for i in range(tries):
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
            try:
                with self.opener.open(req, timeout=40) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 412 and i + 1 < tries:
                    time.sleep(sleep * (i + 1))
                    self.bootstrap()          # 重新拿一次 cookie 再试
                    continue
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(sleep)
        raise RuntimeError(f"GET {url} 失败: {last}")

    def bootstrap(self) -> None:
        """拿 buvid3/b_nut cookie：先访问首页，失败则走 finger/spi 接口。"""
        try:
            self.get(HOME, tries=1)
        except Exception as e:  # noqa: BLE001
            log(f"首页访问失败（{e}），尝试 finger/spi")
        have = {c.name for c in self.cj}
        if "buvid3" not in have:
            try:
                d = json.loads(self.get("https://api.bilibili.com/x/frontend/finger/spi", tries=1))
                b3 = (d.get("data") or {}).get("b_3")
                b4 = (d.get("data") or {}).get("b_4")
                if b3:
                    self.cj.set_cookie(http.cookiejar.Cookie(
                        0, "buvid3", b3, None, False, ".bilibili.com", True, False,
                        "/", True, False, None, False, None, None, {}))
                if b4:
                    self.cj.set_cookie(http.cookiejar.Cookie(
                        0, "buvid4", b4, None, False, ".bilibili.com", True, False,
                        "/", True, False, None, False, None, None, {}))
            except Exception as e:  # noqa: BLE001
                log(f"finger/spi 也失败: {e}")
        log("cookies: " + ", ".join(sorted(c.name for c in self.cj)))

    def view(self, bvid: str) -> dict:
        d = json.loads(self.get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"))
        if d.get("code") != 0:
            raise RuntimeError(f"view API code={d.get('code')} msg={d.get('message')}")
        return d["data"]

    def playurl(self, bvid: str, cid: int) -> tuple[str, str]:
        """返回 (下载 URL, 扩展名)。优先 dash 音频轨（体积小、够 ASR 用）。"""
        referer = f"https://www.bilibili.com/video/{bvid}"
        u = ("https://api.bilibili.com/x/player/playurl?"
             + urllib.parse.urlencode({"bvid": bvid, "cid": cid, "fnval": 16,
                                       "fnver": 0, "fourk": 1}))
        d = json.loads(self.get(u, referer=referer))
        if d.get("code") != 0:
            raise RuntimeError(f"playurl code={d.get('code')} msg={d.get('message')}")
        data = d.get("data") or {}
        dash = data.get("dash") or {}
        audios = dash.get("audio") or []
        if audios:
            # 取码率最低的一路：语音识别不需要高码率，也下载得最快
            a = min(audios, key=lambda x: x.get("bandwidth") or 0)
            url = a.get("baseUrl") or a.get("base_url")
            if not url:
                raise RuntimeError("dash.audio 缺少 baseUrl")
            codecs = (a.get("codecs") or "").split(".")[0]
            ext = "m4a" if "mp4" in codecs or not codecs else codecs
            log(f"dash audio: id={a.get('id')} bw={a.get('bandwidth')} codecs={codecs}")
            return url, ext
        durl = data.get("durl") or []
        if durl:
            url = durl[0].get("url")
            if not url:
                raise RuntimeError("durl 缺少 url")
            log("回退到 durl（含视频，体积较大）")
            return url, "mp4"
        raise RuntimeError("playurl 既无 dash 也无 durl")

    def download(self, url: str, referer: str, dest: Path) -> int:
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
        with self.opener.open(req, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, 1024 * 256)
        tmp.rename(dest)
        return dest.stat().st_size


def ffprobe_duration(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def try_ytdlp(url: str, out: Path) -> Path | None:
    if not shutil.which("yt-dlp"):
        log("未安装 yt-dlp，跳过兜底")
        return None
    log(f"yt-dlp 兜底: {url}")
    cmd = ["yt-dlp", "--no-playlist", "-f", "bestaudio/best",
           "--user-agent", UA, "--add-header", f"Referer:{HOME}",
           "-o", str(out.with_suffix(".%(ext)s")), url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        log("yt-dlp 失败: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
        return None
    hits = sorted(out.parent.glob(out.stem + ".*"))
    hits = [h for h in hits if h.suffix.lower() not in (".part",)]
    return hits[0] if hits else None


def main() -> int:
    # Windows 控制台默认编码不是 UTF-8，标题里的日文会直接抛 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media")
    ap.add_argument("--bvid", action="append", default=[], help="候选 BV 号（按顺序尝试）")
    ap.add_argument("--url", default="", help="直链或视频页 URL（优先于 --bvid）")
    ap.add_argument("--min-secs", type=float, default=60.0)
    ap.add_argument("--max-secs", type=float, default=600.0)
    ap.add_argument("--name", default="", help="输出文件名（不含扩展名）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fail = lambda why: (print(f"MEDIA_FAIL {json.dumps({'reason': why}, ensure_ascii=False)}"), 1)[1]

    # ---------- 直链 / 任意 URL ----------
    if args.url:
        name = args.name or "source"
        if args.url.startswith("http") and not ("bilibili.com" in args.url or "b23.tv" in args.url):
            dest = out_dir / (name + ".media")
            try:
                req = urllib.request.Request(args.url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, 1024 * 256)
            except Exception as e:  # noqa: BLE001
                return fail(f"直链下载失败: {e}")
            dur = ffprobe_duration(dest)
            print("MEDIA_OK " + json.dumps({"source": "url", "id": args.url, "file": str(dest),
                                            "duration": dur, "title": name}, ensure_ascii=False))
            return 0
        # bilibili 页面 URL -> 提取 BV
        m = re.search(r"(BV[0-9A-Za-z]{10})", args.url)
        if m and m.group(1) not in args.bvid:
            args.bvid.insert(0, m.group(1))

    if not args.bvid:
        return fail("没有给出 --bvid 或可用的 --url")

    bili = Bili()
    bili.bootstrap()

    for bvid in args.bvid:
        name = args.name or bvid
        try:
            log(f"=== 尝试 {bvid} ===")
            info = bili.view(bvid)
            dur = float(info.get("duration") or 0)
            title = info.get("title", "")
            cid = info.get("cid")
            log(f"标题: {title} | 时长: {dur:.0f}s | cid: {cid}")
            if info.get("is_charge_video"):
                log("充电专属视频，跳过")
                continue
            if not (args.min_secs <= dur <= args.max_secs):
                log(f"时长不在 [{args.min_secs:.0f}, {args.max_secs:.0f}]s 内，跳过")
                continue
            url, ext = bili.playurl(bvid, cid)
            dest = out_dir / f"{name}.{ext}"
            size = bili.download(url, f"https://www.bilibili.com/video/{bvid}", dest)
            log(f"下载完成: {dest} ({size / 1e6:.2f} MB)")
            if size < 100 * 1024:
                log("文件过小，疑似失败，换下一个候选")
                dest.unlink(missing_ok=True)
                continue
            real = ffprobe_duration(dest) or dur
            print("MEDIA_OK " + json.dumps({"source": "bilibili", "id": bvid, "file": str(dest),
                                            "duration": real, "title": title}, ensure_ascii=False))
            return 0
        except Exception as e:  # noqa: BLE001
            log(f"{bvid} 失败: {e}")
            time.sleep(3)

    # ---------- yt-dlp 兜底 ----------
    for bvid in args.bvid:
        got = try_ytdlp(f"https://www.bilibili.com/video/{bvid}", out_dir / (args.name or bvid))
        if got:
            dur = ffprobe_duration(got)
            if dur and not (args.min_secs <= dur <= args.max_secs):
                log(f"yt-dlp 结果时长 {dur:.0f}s 不在范围内，放弃")
                continue
            print("MEDIA_OK " + json.dumps({"source": "yt-dlp", "id": bvid, "file": str(got),
                                            "duration": dur, "title": bvid}, ensure_ascii=False))
            return 0

    return fail("所有候选均失败（bilibili 风控 / 视频下线？）")


if __name__ == "__main__":
    sys.exit(main())
