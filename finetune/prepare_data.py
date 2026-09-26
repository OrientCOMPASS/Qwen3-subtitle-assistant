#!/usr/bin/env python3
"""构造 Qwen3-ASR 的 S2TT（语音->译文）微调数据。

产出 JSONL，每行三个字段，与官方 finetuning/qwen3_asr_sft.py 的输入格式一致：

    {"audio": "/abs/path.wav",
     "text":  "language Japanese<asr_text>中文译文",
     "prompt": "translate to Chinese"}

要点：
  * 官方脚本里 `target = ex["text"]`，**不校验 text 是否为该音频的转写**，
    所以"日语音频 + 中文文本"就是合法训练样本——改造 ASR 的本质是换数据集。
  * `prompt` 字段会进入 chat 模板的 system 段（qwen_asr._build_messages 把 context
    放在 system role），推理时对应 `transcribe(context=...)`，在 sherpa-onnx 里
    对应 `hotwords` 字段。因此**带 prompt 的样本学"翻译"、不带的学"转写"**，
    一套权重靠 context 切换任务，既避免灾难性遗忘，又让现有 Rust 程序零改动可用。
  * `--asr-ratio` 控制混入多少"日语->日语"的原始 ASR 样本（防遗忘）。

翻译后端（--translator）：
  none  目标文本 = 原转写（用于 ASR 样本）
  stub  确定性伪翻译，**仅供 CI 跑通管线**，绝不能用于真实训练
  api   OpenAI 兼容 /chat/completions（DashScope / OpenRouter / 本地 llama-server / Ollama 均可）

数据源（--source）：
  fleurs   HF google/fleurs 的 ja_jp（CC-BY-4.0，流式读取，免整包下载）
  wav-dir  自备目录：--wav-dir 指向 wav，--transcripts 指向 "文件名<TAB>转写" 的 TSV
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ASR_TEXT_TAG = "<asr_text>"
LANG_TAG = "language Japanese" + ASR_TEXT_TAG
# 基座模型对静音/无语音音频的原始输出就是这一串（qwen_asr.inference.utils.parse_asr_output
# 把 "language None<asr_text>" 解析成 language=""、text=""）。静音样本的训练目标必须
# 与之逐 token 一致，才能把该行为原样保住。
SILENCE_TAG = "language None" + ASR_TEXT_TAG
DEFAULT_PROMPT = "translate to Chinese"


def log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)

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



# --------------------------------------------------------------------------- 音频源

def iter_fleurs(limit: int, max_secs: float):
    """流式读取 FLEURS ja_jp，产出 (16k float32 数组, 转写文本)。

    注意：**不要让 datasets 自己解码音频**——新版 datasets 的 Audio 特征解码依赖
    torchcodec（缺了会抛 ImportError，且解释器退出时还会 segfault）。
    这里用 cast_column(Audio(decode=False)) 拿原始字节，再用 soundfile 解码。
    """
    import io

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", "ja_jp", split="train", streaming=True)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
        log("已切换为不解码音频（拿原始 bytes，自行用 soundfile 解码）")
    except Exception as e:  # noqa: BLE001
        log(f"cast_column(decode=False) 失败（{e}），按已解码数组处理")

    n = 0
    for row in ds:
        audio = row.get("audio") or {}
        text = (row.get("transcription") or row.get("raw_transcription") or "").strip()
        if not text:
            continue

        arr = None
        sr = 16000
        raw = audio.get("bytes")
        if raw:
            try:
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception as e:  # noqa: BLE001
                log(f"soundfile 解码失败，跳过该条: {e}")
                continue
        elif audio.get("array") is not None:
            arr = np.asarray(audio["array"], dtype="float32")
            sr = int(audio.get("sampling_rate") or 16000)
        if arr is None or len(arr) == 0:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        secs = len(arr) / float(sr)
        if secs > max_secs or secs < 1.0:
            continue
        if sr != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        yield arr, text
        n += 1
        if n >= limit:
            return


def iter_wav_dir(wav_dir: Path, transcripts: Path, limit: int, max_secs: float):
    import soundfile as sf

    table = {}
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            table[k.strip()] = v.strip()
    n = 0
    for p in sorted(wav_dir.glob("*.wav")):
        text = table.get(p.stem) or table.get(p.name)
        if not text:
            continue
        arr, sr = sf.read(str(p), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) / sr > max_secs:
            continue
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        yield arr, text
        n += 1
        if n >= limit:
            return


# --------------------------------------------------------------------------- 翻译后端

class StubTranslator:
    """确定性伪翻译：只为让 CI 把训练管线跑通，产出的"译文"没有任何语义价值。"""

    name = "stub"

    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if text in self.cache:
            return self.cache[text]
        # 用稳定的哈希造一段可复现的"中文样"目标文本（含汉字，便于评测脚本统计假名占比）
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        words = ["测试", "样本", "字幕", "翻译", "管线", "验证", "数据", "占位"]
        out = "".join(words[int(c, 16) % len(words)] for c in h[:12])
        self.cache[text] = out
        return out


class NoneTranslator:
    name = "none"

    def __call__(self, text: str) -> str:
        return text


class TableTranslator:
    """查固定对照表（TSV：日文<TAB>中文）。

    用途：在**没有任何 API key** 的情况下做真实微调实验——先让 CI 把它抽到的日语
    转写导出（--dump-transcripts），人工/离线翻译成对照表提交进仓库，再用本后端训练。
    表小但真实，足以验证"模型是否开始输出中文"这个关键问题。
    """

    name = "table"

    def __init__(self, path: Path):
        self.path = path
        self.table: dict[str, str] = {}
        if not path.is_file():
            raise SystemExit(f"对照表不存在: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" in line:
                src, dst = line.split("\t", 1)
            else:
                continue
            src, dst = src.strip(), dst.strip()
            if src and dst:
                self.table[src] = dst
        log(f"对照表 {path}: {len(self.table)} 条")

    def __call__(self, text: str) -> str:
        key = text.strip()
        if key in self.table:
            return self.table[key]
        # 未命中：退化为标点归一后的模糊匹配（去掉首尾空白与句末标点差异）
        norm = key.rstrip("。.、！？!?")
        for k, v in self.table.items():
            if k.rstrip("。.、！？!?") == norm:
                return v
        raise KeyError(key)


class ApiTranslator:
    """OpenAI 兼容 /chat/completions。带磁盘缓存与重试，可断点续跑。"""

    name = "api"

    def __init__(self, base: str, model: str, key: str, cache_path: Path, target_lang: str):
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = key
        self.cache_path = cache_path
        self.system = (
            f"你是专业的影视字幕翻译。把用户给出的日语文本翻译成{target_lang}，"
            "只输出译文本身，不要解释、不要引号、不要保留日语假名"
            "（专有名词用通用中文译名或音译）。"
        )
        self.cache: dict[str, str] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["k"]] = d["v"]
            log(f"翻译缓存命中 {len(self.cache)} 条")

    def __call__(self, text: str) -> str:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": text}],
            "temperature": 0.2,
        }).encode("utf-8")
        last = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
                })
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                out = (d["choices"][0]["message"]["content"] or "").strip()
                if out:
                    self.cache[key] = out
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
                    return out
                last = "空响应"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"翻译失败（{text[:30]}…）: {last}")


# --------------------------------------------------------------------------- 静音样本

def make_silence_samples(audio_dir: Path, n: int, prompt: str,
                         min_secs: float, max_secs: float, seed: int) -> list[str]:
    """合成静音/低噪 wav，训练目标固定为 `language None<asr_text>`（空文本）。

    为什么需要：实测（run 36221464174）120 对样本 LoRA 4 epoch 后，翻译与抗遗忘都达标，
    但**静音行为被训坏**——喂 2s 纯静音输出了中文幻觉（`“我”是“我”的意思。`），
    而主程序的逐句质检依赖「静音 -> language None + 空文本」剔除噪音段。
    根因是训练集里 100% 的样本都有非空 target，LoRA 把「空输出」这条路压没了。
    修复：把静音样本显式混回训练集，且带 prompt（翻译模式）与不带 prompt（转写模式）
    各占一半——eval 的静音测试两种模式都要过。

    样本内容一半纯零、一半低幅白噪声（std≈0.001，更接近真实过了 VAD 的噪音段）；
    前 4 条固定 2.0s 并遍历「零/噪 × 带/不带 prompt」四种组合，
    与 eval_s2tt.py 的默认静音测试（2s 纯零，双模式）精确对齐。
    """
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    lines: list[str] = []
    for j in range(n):
        # j<4：2.0s，四种组合全覆盖（0=零+prompt 1=噪+prompt 2=零+无prompt 3=噪+无prompt）
        # j>=4：零/噪交替，时长在 [min_secs, max_secs] 内随机
        if j < 4:
            secs = 2.0
            zeros = j in (0, 2)          # 0=零+prompt 1=噪+prompt 2=零+无prompt 3=噪+无prompt
            with_prompt = j in (0, 1)
        else:
            secs = round(float(rng.uniform(min_secs, max_secs)), 2)
            zeros = (j % 2 == 0)
            with_prompt = ((j // 2) % 2 == 0)
        n_samples = int(16000 * secs)
        if zeros:
            arr = np.zeros(n_samples, dtype=np.float32)
        else:
            arr = rng.normal(0.0, 0.001, n_samples).astype(np.float32)
        wav = audio_dir / f"silence_{j:03d}.wav"
        sf.write(str(wav), arr, 16000, subtype="PCM_16")
        rec = {
            "audio": str(wav.resolve()),
            "text": SILENCE_TAG,
            "prompt": prompt if with_prompt else "",
        }
        lines.append(json.dumps(rec, ensure_ascii=False))
    return lines


# --------------------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["fleurs", "wav-dir"], default="fleurs")
    ap.add_argument("--wav-dir", default="")
    ap.add_argument("--transcripts", default="")
    ap.add_argument("--limit", type=int, default=200, help="翻译样本条数上限")
    ap.add_argument("--eval-limit", type=int, default=20, help="额外留出的评测条数")
    ap.add_argument("--max-audio-secs", type=float, default=20.0)
    ap.add_argument("--asr-ratio", type=float, default=0.5,
                    help="混入的 日语->日语 ASR 样本比例（防遗忘），0~1")
    ap.add_argument("--translator", choices=["none", "stub", "api", "table"], default="stub")
    ap.add_argument("--table", default="finetune/parallel_ja_zh.tsv",
                    help="--translator table 用的对照表（日文<TAB>中文，# 开头为注释）")
    ap.add_argument("--dump-transcripts", default="",
                    help="把抽到的日语转写逐行写到该文件（供离线翻译成对照表）")
    ap.add_argument("--skip-untranslated", action="store_true",
                    help="table 模式下对照表未命中的样本直接跳过（而不是报错）")
    ap.add_argument("--silence-samples", type=int, default=0,
                    help="混入 N 条合成静音/低噪样本（目标 language None+空文本），"
                         "保住基座的「静音->空输出」行为（主程序逐句质检依赖它）")
    ap.add_argument("--silence-min-secs", type=float, default=1.0)
    ap.add_argument("--silence-max-secs", type=float, default=3.0)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-key-env", default="", help="存放 API key 的环境变量名")
    ap.add_argument("--target-lang", default="简体中文")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="翻译样本的 context（进 system 段）；留空则不带任务开关")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--eval-out", default="data/eval.jsonl")
    ap.add_argument("--cache", default="data/mt_cache.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    random.seed(args.seed)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.translator == "api":
        if not args.api_base or not args.api_model:
            ap.error("--translator api 需要 --api-base 与 --api-model")
        key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        tr = ApiTranslator(args.api_base, args.api_model, key, Path(args.cache), args.target_lang)
    elif args.translator == "table":
        tr = TableTranslator(Path(args.table))
    elif args.translator == "stub":
        log("!! 使用 stub 伪翻译：仅供 CI 跑通管线，产出的模型没有任何翻译能力 !!")
        tr = StubTranslator()
    else:
        tr = NoneTranslator()

    if args.source == "fleurs":
        src = iter_fleurs(args.limit + args.eval_limit, args.max_audio_secs)
    else:
        if not args.wav_dir or not args.transcripts:
            ap.error("--source wav-dir 需要 --wav-dir 与 --transcripts")
        src = iter_wav_dir(Path(args.wav_dir), Path(args.transcripts),
                           args.limit + args.eval_limit, args.max_audio_secs)

    import soundfile as sf

    train_lines: list[str] = []
    eval_lines: list[str] = []
    n_mt = n_asr = n_skipped = 0
    dump_fh = open(args.dump_transcripts, "w", encoding="utf-8") if args.dump_transcripts else None
    for i, (arr, text) in enumerate(src):
        wav = audio_dir / f"ja_{i:06d}.wav"
        if not wav.is_file():
            sf.write(str(wav), arr, 16000, subtype="PCM_16")
        if dump_fh is not None:
            dump_fh.write(text.strip() + "\n")
        is_eval = i >= args.limit
        # 每条样本按 asr_ratio 决定做"翻译"还是"转写"任务
        as_translate = (random.random() >= args.asr_ratio)
        if as_translate and not is_eval:
            try:
                target = tr(text)
            except KeyError:
                if args.skip_untranslated:
                    n_skipped += 1
                    continue
                raise
            prompt = args.prompt
            n_mt += 1
        elif is_eval:
            # 评测集两种任务各存一份，便于检查任务开关与是否遗忘
            if args.translator == "none":
                target = text
            else:
                try:
                    target = tr(text)
                except KeyError:
                    if args.skip_untranslated:
                        n_skipped += 1
                        continue
                    raise
            prompt = args.prompt
            n_mt += 1
        else:
            target = text
            prompt = ""
            n_asr += 1
        rec = {"audio": str(wav.resolve()), "text": LANG_TAG + target, "prompt": prompt}
        line = json.dumps(rec, ensure_ascii=False)
        (eval_lines if is_eval else train_lines).append(line)
        # 评测集额外存一条"不带 prompt"的转写样本，用于检查是否遗忘原能力
        if is_eval:
            eval_lines.append(json.dumps(
                {"audio": str(wav.resolve()), "text": LANG_TAG + text, "prompt": ""},
                ensure_ascii=False))
        if (i + 1) % 25 == 0:
            log(f"已处理 {i + 1} 条（翻译 {n_mt} / 转写 {n_asr}）")

    if dump_fh is not None:
        dump_fh.close()
        log(f"已导出日语转写 {n_mt + n_asr + n_skipped} 行 -> {args.dump_transcripts}"
            "（离线翻译成对照表后可用 --translator table 训练）")
    if n_skipped:
        log(f"对照表未命中而跳过 {n_skipped} 条")

    n_silence = 0
    if args.silence_samples > 0:
        sil = make_silence_samples(audio_dir, args.silence_samples, args.prompt,
                                   args.silence_min_secs, args.silence_max_secs, args.seed)
        train_lines.extend(sil)
        n_silence = len(sil)
        log(f"已混入静音/低噪样本 {n_silence} 条（目标 {SILENCE_TAG!r}，"
            f"带 prompt 与不带 prompt 各约一半）")

    Path(args.out).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    Path(args.eval_out).write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    log(f"训练集 {len(train_lines)} 条 -> {args.out}")
    log(f"评测集 {len(eval_lines)} 条 -> {args.eval_out}")
    log(f"任务构成：翻译 {n_mt} 条（prompt={args.prompt!r}），转写 {n_asr} 条（prompt 为空），"
        f"静音 {n_silence} 条")
    if not train_lines:
        log("!! 训练集为空，请检查数据源与 --limit")
        return 1
    return 0


if __name__ == "__main__":
    _hard_exit(main())
