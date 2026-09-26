#!/usr/bin/env python3
"""GGUF 运行时探针：通过 llama-server 的 OpenAI 兼容接口验证 Qwen3-ASR GGUF 推理。

E1（基座官方 GGUF）验证的问题：
  1. llama.cpp（mtmd/qwen3a 路径）能否加载 GGUF LM + mmproj 音频编码器并转录音频；
  2. **system 段 context 通道是否可用**——S2TT 任务开关（"translate to Chinese"）
     在 sherpa-onnx 里对应 hotwords 字段；换运行时后必须有等价物，否则微调模型
     的「一套权重、context 切换任务」设计不成立；
  3. 计时（CPU RTF），供与 sherpa-onnx int8 / PyTorch fp32 两个运行时对比。

基座模型预期（T4 已证）：带不带 context 都输出日语——本探针在 E1 阶段
**断言的是「能转录出日语」**；E2（换微调 GGUF 后）才断言「带 context 出中文」。

用法：
  python finetune/gguf_server_probe.py --url http://127.0.0.1:8080 \
      --audio ja1.wav --expect kana --out probe_report.json
  # --expect: kana=断言日语（E1 基座） / cjk=断言中文（E2 微调后） / any=只看能出文本
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
CONTEXT = "translate to Chinese"


def log(msg: str) -> None:
    print(f"[gguf] {msg}", flush=True)


def ratios(text: str):
    chars = [c for c in text if not c.isspace()]
    n = max(1, len(chars))
    return (sum(1 for c in chars if KANA.match(c)) / n,
            sum(1 for c in chars if CJK.match(c)) / n)


def wait_health(url: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    log(f"server 就绪（{time.time()-t0:.0f}s）")
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def parse_raw(text: str):
    """llama-server 返回的是模型原始输出（language X<asr_text>正文）。
    解析出 (lang_tag, body)；sherpa/qwen-asr 运行时会替用户做这步，llama.cpp 不做。"""
    lang = ""
    m = re.match(r"\s*language\s+(\S+?)\s*<asr_text>", text)
    if m:
        lang = m.group(1)
    body = text.split("<asr_text>", 1)[-1].strip() if "<asr_text>" in text else text.strip()
    return lang, body


def chat(url: str, audio_path: Path, system: str, temperature: float = 0.0) -> dict:
    b64 = base64.b64encode(audio_path.read_bytes()).decode()
    body = {
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": [
                {"type": "input_audio",
                 "input_audio": {"data": f"data:audio/wav;base64,{b64}", "format": "wav"}},
            ]},
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        text = (d["choices"][0]["message"]["content"] or "").strip()
        lang, body = parse_raw(text)
        usage = d.get("usage", {})
        return {"ok": True, "text": text, "lang_tag": lang, "body": body,
                "secs": round(dt, 1), "usage": usage}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}",
                "secs": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "secs": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--expect", choices=["kana", "cjk", "any"], default="kana")
    ap.add_argument("--audio-secs", type=float, default=0.0, help="音频时长（算 RTF 用，0=不计算）")
    ap.add_argument("--health-timeout", type=float, default=240)
    ap.add_argument("--check-silence", action="store_true",
                    help="追加 2s 纯静音测试：两种 context 下都应输出空/language None"
                         "（产品用该行为替代 LLM 逐句质检剔除噪音段）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    if not wait_health(args.url, args.health_timeout):
        log("server 未在限时内就绪")
        return 2

    audio = Path(args.audio)
    report = {"audio": str(audio), "expect": args.expect, "runs": {}}
    fails = []
    for tag, system in (("with_context", CONTEXT), ("no_context", "")):
        res = chat(args.url, audio, system)
        # 比例按解析后的正文算（原始串带 "language X<asr_text>" ASCII 标签会稀释占比）
        kana, cjk = ratios(res.get("body", ""))
        res.update({"kana": round(kana, 4), "cjk": round(cjk, 4)})
        report["runs"][tag] = res
        log(f"[{tag}] ok={res['ok']} {res['secs']}s lang={res.get('lang_tag')!r} "
            f"假名{kana:.0%} 汉字{cjk:.0%}")
        log(f"[{tag}] body: {res.get('body', res.get('error', ''))[:100]!r}")
        if not res["ok"] or not res.get("text"):
            fails.append(f"{tag}: 无输出（{res.get('error', '空文本')}）")

    # 语言断言（只针对带 context 组——E1 基座预期两组都日语；E2 微调预期 with_context 中文）
    wr = report["runs"].get("with_context", {})
    if wr.get("ok") and wr.get("body"):
        if args.expect == "kana" and wr["kana"] < 0.15:
            fails.append(f"expect=kana 但假名仅 {wr['kana']:.0%}")
        if args.expect == "cjk" and (wr["cjk"] < 0.5 or wr["kana"] > 0.05):
            fails.append(f"expect=cjk 但汉字 {wr['cjk']:.0%} / 假名 {wr['kana']:.0%}")
    # 对照组（不带 context）在 E2 下应仍是日语：微调的任务开关是否保住了转写行为
    nr = report["runs"].get("no_context", {})
    if args.expect == "cjk" and nr.get("ok") and nr.get("body") and nr["kana"] < 0.15:
        fails.append(f"对照组假名仅 {nr['kana']:.0%}（转写行为可能已被训丢）")

    if args.audio_secs > 0:
        for tag, r in report["runs"].items():
            if r.get("ok"):
                r["rtf"] = round(r["secs"] / args.audio_secs, 3)

    # ---- 静音行为（可选）：产品依赖「静音 -> 空输出」剔除噪音段 ----
    if args.check_silence:
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sil_path = Path(f.name)
        with wave.open(str(sil_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000 * 2)
        for tag, system in (("silence_with_context", CONTEXT), ("silence_no_context", "")):
            res = chat(args.url, sil_path, system)
            body = res.get("body", "") if res.get("ok") else ""
            lang = res.get("lang_tag", "")
            ok = (not res["ok"]) or (len(body) <= 2 or lang.lower() == "none")
            report["runs"][tag] = {**res, "parsed_body": body, "parsed_lang": lang, "empty_ok": bool(ok)}
            log(f"[{tag}] ok={res['ok']} raw={text[:60]!r} -> body={body[:30]!r} lang={lang!r} "
                f"{'✔ 空/None' if ok else '✘ 非空'}")
            if not ok:
                fails.append(f"{tag}: 静音输出非空（{body[:30]!r}）")
        sil_path.unlink(missing_ok=True)

    log("=" * 60)
    if fails:
        for f in fails:
            log("✘ " + f)
    else:
        log("✔ GGUF 运行时探针通过：llama-server 可转录 Qwen3-ASR GGUF，context 通道可用")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
