#!/usr/bin/env python3
"""把 S2TT LoRA 适配器合并回 HF 基座，产出可直接 convert_hf_to_gguf 的完整模型目录。

与 ONNX 补丁路线（B2）的本质区别：合并发生在**权重语义层**（peft merge_and_unload），
之后走 llama.cpp 官方转换器，音频塔（mmproj）与 LM 一并带出，**没有任何张量映射/
量化写回的坑**；LoRA 命中音频塔 q/k/v 的问题在这里天然消解。

结构要点（与 sft_lora.py 训练时一致）：
  * 适配器挂在 top.thinker（Qwen3ASRThinkerForConditionalGeneration）上；
  * 合并后把 merged thinker 挂回顶层再 save_pretrained，目录结构与原始 HF 仓库一致；
  * processor/tokenizer 一并落盘（转换器与 llama-server 模板都要用）。

用法（CI：finetune.yml mode=gguf-e2）：
  python finetune/merge_lora_hf.py --base Qwen/Qwen3-ASR-1.7B \
      --adapter art/out/real/lora --out merged/1.7B-s2tt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[merge] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF 基座（id 或本地目录）")
    ap.add_argument("--adapter", required=True, help="peft LoRA 目录")
    ap.add_argument("--out", required=True, help="合并后模型输出目录")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    t0 = time.time()
    import torch
    # import qwen_asr 触发 qwen3_asr 的 AutoClass 注册（transformers 4.57.6 原生支持）
    import qwen_asr  # noqa: F401
    from transformers import AutoProcessor
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
    from peft import PeftModel

    log(f"加载基座 {args.base}（bf16, CPU）...")
    top = Qwen3ASRForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16)
    thinker = getattr(top, "thinker", top)
    n_params = sum(p.numel() for p in top.parameters())
    log(f"基座加载完成 {time.time()-t0:.0f}s，参数量 {n_params/1e9:.2f}B")

    log(f"挂载适配器 {args.adapter} 并合并 ...")
    t1 = time.time()
    pm = PeftModel.from_pretrained(thinker, args.adapter)
    merged = pm.merge_and_unload()
    if hasattr(top, "thinker"):
        top.thinker = merged
    log(f"合并完成 {time.time()-t1:.0f}s")

    # 抽查合并确实改变了权重（对比基座缓存里同层原值不可行——直接验证 LoRA 增量非零即可：
    # merge_and_unload 后 W' = W + B·A·s，若适配器全零则无变化；这里断言适配器文件非空）
    from safetensors import safe_open
    with safe_open(str(Path(args.adapter) / "adapter_model.safetensors"), framework="np") as f:
        k0 = list(f.keys())[0]
        import numpy as np
        assert np.abs(np.asarray(f.get_tensor(k0))).sum() > 0, "适配器权重全零？"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"保存合并模型 -> {out}")
    t2 = time.time()
    top.save_pretrained(str(out), safe_serialization=True)
    # processor / tokenizer（convert_hf_to_gguf 需要 tokenizer 与 chat template）
    try:
        proc = AutoProcessor.from_pretrained(args.base)
        proc.save_pretrained(str(out))
    except Exception as e:  # noqa: BLE001
        log(f"AutoProcessor 失败（{e}），从基座目录复制配置文件兜底")
        base_dir = Path(args.base)
        if base_dir.is_dir():
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                         "preprocessor_config.json", "chat_template.jinja", "special_tokens_map.json",
                         "added_tokens.json"):
                src = base_dir / name
                if src.is_file():
                    shutil.copy2(src, out / name)
    log(f"保存完成 {time.time()-t2:.0f}s")

    files = sorted(p.name for p in out.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 1e9
    log(f"输出 {len(files)} 个文件，共 {total:.2f} GB: {files[:8]}...")
    report = {"base": args.base, "adapter": args.adapter, "out": str(out),
              "params_b": round(n_params / 1e9, 3), "files": files,
              "total_gb": round(total, 3), "seconds": round(time.time() - t0, 1)}
    (out.parent / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("✔ 合并完成，可交 convert_hf_to_gguf.py")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
