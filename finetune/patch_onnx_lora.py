#!/usr/bin/env python3
"""Gate 2 补丁器：把 LoRA ΔW 合并进官方 sherpa-onnx int8 ONNX 包，产出 s2tt 模型目录。

前置事实由 Gate 1 探针在真实官方 0.6B 包上实测确认（run 36252216017，三项全 GO）：
  * 适配器 250 个模块（LM 196 + 音频塔 54）全部可按值映射到 ONNX 张量；
  * 全部为 MatMulInteger B 侧 (K,N) **转置存储**；LM 侧 uint8 非对称（带 zp）、
    音频塔 int8 对称；
  * 按**原 scale/zp** 重量化削顶最坏 0.109%、MSE ~5e-8 → 直接沿用原量化参数，
    不改图结构、不重算 scale。

流程：
  1. 值匹配（复用 inspect_onnx_lora 的扫描/反量化/匹配函数），但只对
     「形状命中某个 ΔW 候选」的张量做匹配（跳过 embed/lm_head/conv 等无关大张量，
     也天然避开 tied lm_head/embed 的匹配二义）；
  2. 逐模块补丁：q' = clip(round((dequant(q) + ΔW_onnx摆位)/s + z), lo, hi)；
     fp32 张量（若存在）直接加；
  3. 自检：补丁后反量化 vs 目标值，逐元素误差应 <= 0.51*step（未削顶处）；
  4. 其余文件字节级复制，输出完整模型目录（sherpa-onnx 直接可加载）。

用法（CI：finetune.yml mode=onnx-patch；本地合成夹具亦可跑）：
  python finetune/patch_onnx_lora.py \
      --model-dir models/sherpa-onnx-qwen3-asr-0.6B-int8 \
      --hf-model Qwen/Qwen3-ASR-0.6B --adapter art/out/real/lora \
      --out-dir models/sherpa-onnx-qwen3-asr-0.6B-s2tt-int8 \
      --report out/patch/patch_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# 同目录复用探针的扫描/反量化/值匹配实现（单一事实源，避免两份逻辑漂移）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_onnx_lora import (  # noqa: E402
    collect_matchable, dequant, hf_open, load_adapter, match_one, log, _hard_exit,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hf-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-match-rel", type=float, default=0.05)
    ap.add_argument("--min-elems", type=int, default=1024,
                    help="fp32 二维权重纳入补丁的元素数下限（与探针同义）")
    ap.add_argument("--max-clip", type=float, default=0.01,
                    help="允许的最坏削顶占比（per-channel min/max 量化下边界元素结构性必然存在，"
                         "真实模型 Gate1 实测 0.109%%；超过判失败）")
    ap.add_argument("--dry-run", action="store_true", help="只匹配与模拟，不写文件")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    import onnx
    from onnx import numpy_helper

    t0 = time.time()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    # ---- 适配器 ΔW 与形状候选 ----
    r, alpha, scaling, per_module = load_adapter(Path(args.adapter))
    dws: dict[str, "np.ndarray"] = {}
    shape_cands: set[tuple] = set()
    for mod, ab in per_module.items():
        if "A" not in ab or "B" not in ab:
            continue
        dW = (ab["B"] @ ab["A"]) * scaling          # (N, K) HF 摆位
        dws[mod] = dW
        shape_cands.add(tuple(dW.shape))
        shape_cands.add(tuple(dW.shape[::-1]))      # 转置存储摆位
    log(f"适配器 r={r} alpha={alpha} scaling={scaling:.2f}：ΔW {len(dws)} 个，"
        f"形状候选 {len(shape_cands)} 种")

    hf_index, hf_shapes, hf_fetch = hf_open(args.hf_model)
    hf_by_shape = defaultdict(list)
    for k, shp in hf_shapes.items():
        if len(shp) == 2 and shp in shape_cands:
            hf_by_shape[shp].append(k)
    # 适配器模块 -> HF key（预计算，匹配阶段 O(1) 反查）
    mod_by_hfkey = {}
    for mod in dws:
        k = hf_key_for(hf_index, mod)
        if k:
            mod_by_hfkey[k] = mod
        else:
            report_flags_early = f"{mod}: HF key 未找到"
            log("  !! " + report_flags_early)
    log(f"HF 侧形状命中 key {sum(len(v) for v in hf_by_shape.values())} 个；"
        f"模块->HF key 映射 {len(mod_by_hfkey)}/{len(dws)}")

    # ---- 逐文件：匹配 -> 补丁 ----
    report = {"model_dir": str(model_dir), "adapter": args.adapter,
              "lora": {"r": r, "alpha": alpha, "scaling": scaling},
              "files": {}, "patched_modules": [], "unmapped_modules": [], "flags": []}
    cache: dict = {}
    patched_by_hfkey: dict[str, dict] = {}
    onnx_files = sorted(model_dir.glob("*.onnx"))
    for f in onnx_files:
        m = onnx.load(str(f))
        inits = {t_.name: t_ for t_ in m.graph.initializer}
        ents, fl = collect_matchable(m, f.name, args.min_elems)
        report["flags"].extend(fl)
        n_patch = n_match = 0
        for e in ents:
            shp = tuple(inits[e["w"]].dims)
            if shp not in shape_cands and (len(shp) == 2 and shp[::-1] not in shape_cands):
                continue  # 形状不在 ΔW 候选里：与 LoRA 无关，跳过
            Wd, s_b, z_b, quantized = dequant(e, inits)
            res = match_one(Wd, hf_by_shape, hf_fetch, cache, args.max_match_rel)
            if res is None:
                report["flags"].append(f"{f.name}:{e['w']} 形状命中但值匹配失败")
                continue
            key, transposed, rel, second = res
            n_match += 1
            if key in patched_by_hfkey:
                report["flags"].append(f"{key} 被 {patched_by_hfkey[key]['w']} 与 {e['w']} 同时匹配")
                continue
            mod = mod_by_hfkey.get(key)
            if mod is None:
                continue  # 匹配上了但不是 LoRA 目标（如 tied lm_head/embed）
            dW = dws[mod]
            dW_o = dW.T if transposed else dW
            target = Wd.astype(np.float64) + dW_o          # ONNX 存储摆位
            if quantized:
                q = target / s_b + (z_b if z_b is not None else 0.0)
                lo, hi = (0, 255) if inits[e["w"]].data_type == 2 else (-128, 127)
                clipped = (q < lo) | (q > hi)
                clip_frac = float(np.mean(clipped))
                qn = np.clip(np.rint(q), lo, hi)
                new_dtype = np.uint8 if lo == 0 else np.int8
                new_arr = qn.astype(new_dtype)
                # 自检：非削顶元素的反量化误差必须 <= 0.51*step（逐通道）；
                # 削顶元素重建到边界属预期行为，不计坏元素（其误差由 clip_frac 约束）
                recon = (qn.astype(np.float64) - (z_b if z_b is not None else 0.0)) * s_b
                err = np.abs(recon - target)
                bad = int(np.sum((err > 0.51 * s_b + 1e-12) & ~clipped))
                mse = float(np.mean((recon - target) ** 2))
            else:
                new_arr = target.astype(np.float32)
                clip_frac, bad, mse = 0.0, 0, 0.0
            rec = {"module": mod, "file": f.name, "w": e["w"], "hf_key": key,
                   "transposed": bool(transposed), "match_rel_err": round(rel, 6),
                   "quantized": bool(quantized), "clip_frac": round(clip_frac, 6),
                   "selfcheck_bad_elems": bad, "requant_mse": round(mse, 12)}
            patched_by_hfkey[key] = rec
            report["patched_modules"].append(rec)
            if not args.dry_run:
                inits[e["w"]].CopyFrom(numpy_helper.from_array(new_arr, e["w"]))
            n_patch += 1
            del Wd, target
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if n_patch:
                onnx.save(m, str(out_dir / f.name))
            else:
                shutil.copy2(f, out_dir / f.name)   # 未被补丁的 onnx 原样字节复制
        report["files"][f.name] = {"matched": n_match, "patched": n_patch}
        log(f"  {f.name}: 形状候选命中并匹配 {n_match}，补丁 {n_patch}")
        del m, inits
        gc.collect()

    # ---- 覆盖率检查 ----
    mapped_mods = {rec["module"] for rec in report["patched_modules"]}
    report["unmapped_modules"] = sorted(set(dws) - mapped_mods)
    log(f"补丁覆盖 {len(mapped_mods)}/{len(dws)} 模块；未映射 {len(report['unmapped_modules'])}")
    if report["unmapped_modules"]:
        log("  未映射示例: " + "; ".join(report["unmapped_modules"][:5]))

    # ---- 复制其余文件 ----
    if not args.dry_run:
        for p in sorted(model_dir.rglob("*")):
            if p.is_file() and p.suffix != ".onnx":
                rel = p.relative_to(model_dir)
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        n_onnx_out = len(list(out_dir.glob("*.onnx")))
        log(f"输出目录 {out_dir}: onnx {n_onnx_out} 个 + 其余文件已复制")

    clips = [rec["clip_frac"] for rec in report["patched_modules"] if rec["quantized"]]
    bads = sum(rec["selfcheck_bad_elems"] for rec in report["patched_modules"])
    report["worst"] = {
        "clip_frac": round(max(clips), 6) if clips else 0.0,
        "selfcheck_bad_elems_total": bads,
        "match_rel_err_max": round(max((r_["match_rel_err"] for r_ in report["patched_modules"]),
                                       default=0.0), 6),
    }
    report["seconds"] = round(time.time() - t0, 1)
    ok = ((not report["unmapped_modules"])
          and report["worst"]["clip_frac"] < args.max_clip and bads == 0)
    report["ok"] = bool(ok)
    log("=" * 62)
    log(f"最坏削顶 {report['worst']['clip_frac']:.4%} | 自检坏元素 {bads} | "
        f"匹配误差最大 {report['worst']['match_rel_err_max']:.2e} | 用时 {report['seconds']}s")
    log("✔ 补丁完成，可交 sherpa-onnx 验证" if ok else "✘ 补丁存在未映射/削顶超标/自检失败")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.report}")
    return 0 if ok else 1


def hf_key_for(index, mod):
    """与探针同规则：适配器模块名 -> HF key（缓存加速放调用方）。"""
    for cand in (f"thinker.{mod}.weight", f"{mod}.weight"):
        if cand in index:
            return cand
    return None


if __name__ == "__main__":
    _hard_exit(main())
