#!/usr/bin/env python3
"""ONNX 权重补丁可行性探针 v2——按值匹配（应对导出器匿名化权重名）。

v1 在真实的 k2-fsa 官方包上实测（run 36250220762）发现两个推翻设计假设的事实：
  1. **权重名被 torch.onnx 导出器匿名化**：197 个量化 MatMul 权重全叫
     `onnx::MatMul_N_quantized/_scale/_zero_point`，按 q_proj/down_proj 名字一个都找不到
     （197 = 28 层 × 7 类投影 196 + 1 个额外的量化 MatMul，embed_tokens 反而保留了名字）；
  2. **LoRA 也命中了音频塔**：peft 按模块名匹配 target_modules，音频编码器 18 层的
     q/k/v_proj 同样被挂了 LoRA（54 个模块），适配器共 250 个模块——所以补丁必须
     同时覆盖 decoder.int8.onnx 和 encoder.int8.onnx（「音频塔不用动」不成立）。

v2 的匹配策略（不再依赖名字）：
  * 从图里扫出全部量化权重（DequantizeLinear / MatMulInteger / QLinearMatMul 的
    数据输入 + scale/zp/axis）；
  * 反量化后与 HF 基座 safetensors **按值匹配**：形状预过滤（含转置摆位），
    误差最小者胜出，且要求与次优拉开 2 倍以上差距（唯一性）；
  * 适配器模块 → HF key 用直接拼接（`thinker.` 前缀探测），再经映射表找到 ONNX 张量；
  * 对每个适配器模块计算 ΔW=B·A·(α/r) 的相对幅度，并模拟「反量化→加 ΔW→按原
    scale/zp 重量化」的削顶率与 MSE——决定补丁器能否沿用原 scale。

go/no-go 判据：
  * 250 个适配器模块全部映射到 ONNX 张量（GO-mapping）；
  * 抽样对应性 rel_err < 2e-2（同源权重）；
  * 削顶率 < 1% → 直接沿用原 scale；否则补丁器按新 range 重算 scale（仍可行）。

用法（CI：finetune.yml mode=onnx-inspect）：
  python finetune/inspect_onnx_lora.py \
      --model-dir models/sherpa-onnx-qwen3-asr-0.6B-int8 \
      --hf-model Qwen/Qwen3-ASR-0.6B --adapter art/out/real/lora \
      --out out/inspect/onnx_patch_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJ_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def log(msg: str) -> None:
    print(f"[inspect] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    import os
    os._exit(code)


def norm_pattern(name: str) -> str:
    return re.sub(r"\d+", "N", name)


# ------------------------------------------------------------------ ONNX 图扫描

def scan_quant_entries(model, fname: str) -> list[dict]:
    """扫出全部「量化权重 + scale/zp/axis」条目。

    覆盖三种消费形态：
      DequantizeLinear(w, scale[, zp], axis=a) -> MatMul
      MatMulInteger(A, B[, A_zp, B_zp])（scale 在图外，本探针记 flag）
      QLinearMatMul(a, a_s, a_zp, b, b_s, b_zp, ...)
    """
    init_names = {t.name for t in model.graph.initializer}
    entries: dict[str, dict] = {}
    flags: list[str] = []
    for idx, node in enumerate(model.graph.node):
        ins = list(node.input)
        if node.op_type == "DequantizeLinear" and ins and ins[0] in init_names:
            axis = next((a.i for a in node.attribute if a.name == "axis"), None)
            e = entries.setdefault(ins[0], {"w": ins[0], "file": fname, "node_idx": idx, "op": node.op_type})
            if len(ins) > 1 and ins[1]:
                e["scale"] = ins[1]
            if len(ins) > 2 and ins[2]:
                e["zp"] = ins[2]
            if axis is not None:
                e["axis"] = int(axis)
        elif node.op_type == "MatMulInteger":
            for pos in (0, 1):
                if len(ins) > pos and ins[pos] in init_names:
                    e = entries.setdefault(ins[pos], {"w": ins[pos], "file": fname,
                                                      "node_idx": idx, "op": node.op_type})
                    zpos = pos + 2
                    if len(ins) > zpos and ins[zpos]:
                        e["zp"] = ins[zpos]
                    if "scale" not in e:
                        flags.append(f"{ins[pos]}: MatMulInteger 无内联 scale（在图外 Mul，需另找）")
        elif node.op_type == "QLinearMatMul":
            for tpos, spos, zpos in ((0, 1, 2), (3, 4, 5)):
                if len(ins) > max(tpos, spos, zpos) and ins[tpos] in init_names:
                    e = entries.setdefault(ins[tpos], {"w": ins[tpos], "file": fname,
                                                       "node_idx": idx, "op": node.op_type})
                    e["scale"] = ins[spos]
                    if ins[zpos]:
                        e["zp"] = ins[zpos]
    return list(entries.values()), flags


# ------------------------------------------------------------------ HF 基座

def hf_open(hf_model: str):
    from safetensors import safe_open

    root = Path(hf_model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(hf_model, allow_patterns=["*.safetensors", "*.json"]))
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"HF 模型目录里没有 safetensors: {root}")
    handles = [safe_open(str(f), framework="np") for f in files]
    t_handles = None
    index: dict[str, tuple] = {}
    shapes: dict[str, tuple] = {}
    for h in handles:
        for k in h.keys():
            index[k] = h
            shapes[k] = tuple(h.get_slice(k).get_shape())

    def fetch(key: str):
        import numpy as np

        h = index[key]
        try:
            return np.asarray(h.get_tensor(key), dtype=np.float32)
        except Exception:  # bf16 等 numpy 读不了的 dtype -> torch
            nonlocal t_handles
            from safetensors.torch import safe_open as t_open

            if t_handles is None:
                t_handles = {str(f): t_open(str(f), framework="pt") for f in files}
            th = t_handles[h.filename]
            return th.get_tensor(key).float().numpy()

    return index, shapes, fetch


def hf_key_for_module(index: dict, mod: str):
    """适配器模块名 -> HF key。真实 peft 模块名如 model.layers.N.self_attn.q_proj /
    audio_tower.layers.N.self_attn.q_proj，HF key 为 thinker.<mod>.weight（前缀探测）。"""
    for cand in (f"thinker.{mod}.weight", f"{mod}.weight"):
        if cand in index:
            return cand
    # 兜底：模糊匹配结尾
    suf = f".{mod.split('.', 1)[-1]}.weight" if "." in mod else None
    tail = mod.split("layers.", 1)[-1]
    cands = [k for k in index if k.endswith(tail + ".weight") or (suf and k.endswith(suf))]
    return cands[0] if len(cands) == 1 else None


# ------------------------------------------------------------------ 适配器

def load_adapter(adapter_dir: Path):
    import numpy as np

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = int(cfg.get("r", 0))
    alpha = float(cfg.get("lora_alpha", 0))
    scaling = alpha / r if r else 0.0
    from safetensors import safe_open

    per_module: dict[str, dict] = defaultdict(dict)
    for f in sorted(adapter_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            for k in h.keys():
                m = re.match(r"base_model\.model\.(.*)\.lora_(A|B)\.weight$", k)
                if m:
                    per_module[m.group(1)][m.group(2)] = np.asarray(h.get_tensor(k), dtype=np.float32)
    return r, alpha, scaling, dict(per_module)


# ------------------------------------------------------------------ 值匹配

def dequant(entry, inits):
    """按 axis 反量化成 fp32 numpy；返回 (W, scale_bcast, zp_bcast, quantized)。"""
    import numpy as np
    from onnx import numpy_helper

    W = numpy_helper.to_array(inits[entry["w"]])
    if entry.get("scale") is None:
        return W.astype(np.float32), None, None, False
    scale = numpy_helper.to_array(inits[entry["scale"]]).astype(np.float64)
    zp = (numpy_helper.to_array(inits[entry["zp"]]).astype(np.float64)
          if entry.get("zp") else np.float64(0.0))
    if scale.size == 1:
        s_b = scale.reshape([1] * W.ndim)
        z_b = np.reshape(zp, [1] * W.ndim) if np.ndim(zp) else zp
    else:
        axis = entry.get("axis", 1 if W.ndim == 2 else 0) % W.ndim
        if scale.size != W.shape[axis] and W.ndim == 2:
            axis = 1 - axis  # 标注与实际不符时试另一轴
        shp = [1] * W.ndim
        shp[axis] = scale.size
        s_b = scale.reshape(shp)
        z_b = zp.reshape(shp) if np.size(zp) == scale.size else zp
    Wf = (W.astype(np.float64) - z_b) * s_b
    return Wf.astype(np.float32), s_b, z_b, True


def match_one(Wd, hf_by_shape, hf_fetch, cache, max_rel: float):
    """在 HF 权重里找与 Wd 值最匹配者（自动试转置摆位）。返回 (key, transposed, rel_err, second_rel)。

    误差下限都超过 max_rel 则判为「HF 里没有它的同源权重」（返回 None）——
    宁缺毋滥：错误映射比未映射危险得多。
    """
    import numpy as np

    results = []
    for transposed in (False, True):
        Wt = Wd.T if transposed else Wd
        for key in hf_by_shape.get(Wt.shape, ()):
            Whf = cache.get(key)
            if Whf is None:
                Whf = hf_fetch(key)
                cache[key] = Whf
            denom = max(1e-9, float(np.abs(Whf).mean()))
            rel = float(np.abs(Whf.astype(np.float32) - Wt).mean()) / denom
            results.append((rel, key, transposed))
    if not results or results and min(r[0] for r in results) > max_rel:
        return None
    results.sort()
    best = results[0]
    second = results[1][0] if len(results) > 1 else float("inf")
    return best[1], best[2], best[0], second


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="sherpa-onnx-qwen3-asr-*-int8 目录")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--files", default="", help="逗号分隔的 onnx 文件名（默认目录下全部 *.onnx）")
    ap.add_argument("--skip-name-pattern", default="embed_tokens",
                    help="名字命中该模式的量化张量跳过值匹配（如 embed_tokens，LoRA 不碰）")
    ap.add_argument("--min-elems", type=int, default=1024,
                    help="无量化包装的 fp32 二维 initializer 也纳入值匹配的元素数下限"
                         "（兼容「Q/K/V/O 保 fp32」的混合摆位导出）")
    ap.add_argument("--max-match-rel", type=float, default=0.05,
                    help="值匹配的平均相对误差上限，超过视为 HF 里没有同源权重")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    import onnx

    t0 = time.time()
    model_dir = Path(args.model_dir)
    files = ([model_dir / f for f in args.files.split(",")] if args.files
             else sorted(model_dir.glob("*.onnx")))
    log(f"模型目录: {model_dir}；待扫文件: {[f.name for f in files]}")

    # ---- 1) 扫全部 onnx 文件的量化条目 ----
    all_entries, scan_flags, file_info = [], [], {}
    for f in files:
        m = onnx.load(str(f))
        inits_meta = {t.name: (list(t.dims), t.data_type) for t in m.graph.initializer}
        ents, fl = scan_quant_entries(m, f.name)
        # 无量化包装的 fp32 大二维权重也纳入匹配（混合摆位导出时 Q/K/V/O 可能是 fp32）
        quant_names = {e["w"] for e in ents}
        for name, (dims, dt) in inits_meta.items():
            if (dt == 1 and len(dims) == 2 and dims[0] * dims[1] >= args.min_elems
                    and name not in quant_names
                    and not re.search(r"(_scale|_zero_point|_zp)$", name)):
                ents.append({"w": name, "file": f.name, "node_idx": None,
                             "op": "fp32-initializer", "scale": None})
        pats = defaultdict(int)
        for name in inits_meta:
            pats[norm_pattern(name)] += 1
        file_info[f.name] = {
            "size_mb": round(f.stat().st_size / 1e6, 1),
            "initializers": len(inits_meta), "nodes": len(m.graph.node),
            "quant_entries": len(ents),
            "top_patterns": dict(sorted(pats.items(), key=lambda kv: -kv[1])[:12]),
        }
        log(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB  initializer {len(inits_meta)}  "
            f"节点 {len(m.graph.node)}  可匹配权重 {len(ents)}")
        for e in ents:
            e["shape"] = inits_meta[e["w"]][0]
            e["dtype"] = inits_meta[e["w"]][1]
        all_entries.extend(ents)
        scan_flags.extend(fl)
        del m
        gc.collect()
    log(f"可匹配权重条目合计 {len(all_entries)}；扫描 flags {len(scan_flags)}")

    # ---- 2) HF 索引 ----
    hf_index, hf_shapes, hf_fetch = hf_open(args.hf_model)
    log(f"HF key 总数 {len(hf_index)}；样例: {list(hf_index)[:3]}")
    hf_by_shape = defaultdict(list)
    for k, shp in hf_shapes.items():
        if len(shp) == 2 and shp[0] * shp[1] >= args.min_elems:
            hf_by_shape[shp].append(k)

    # ---- 3) 值匹配（重新逐文件加载，控制内存峰值） ----
    DTYPE = {1: "float32", 2: "uint8", 3: "int8", 7: "int64", 10: "float16", 16: "bfloat16"}
    cache: dict = {}
    mapping: dict[str, dict] = {}      # hf_key -> entry+match 信息
    unmatched_onnx, skipped = [], []
    rel_errs = []
    for f in files:
        m = onnx.load(str(f))
        inits = {t.name: t for t in m.graph.initializer}
        for e in [x for x in all_entries if x["file"] == f.name]:
            if args.skip_name_pattern and args.skip_name_pattern in e["w"]:
                skipped.append(e["w"])
                continue
            Wd, _, _, _ = dequant(e, inits)
            res = match_one(Wd, hf_by_shape, hf_fetch, cache, args.max_match_rel)
            del Wd
            if res is None:
                unmatched_onnx.append({"w": e["w"], "shape": e["shape"]})
                continue
            key, transposed, rel, second = res
            uniq = second > 2 * max(rel, 1e-9)
            e.update({"hf_key": key, "transposed": bool(transposed),
                      "rel_err": round(rel, 6), "unique": bool(uniq),
                      "dtype_s": DTYPE.get(e["dtype"], str(e["dtype"]))})
            if not uniq:
                scan_flags.append(f"{e['w']}: 值匹配不唯一（best {rel:.2e} vs second {second:.2e}）")
            if key in mapping:
                scan_flags.append(f"{key}: 被多个 ONNX 张量匹配（{mapping[key]['w']} 与 {e['w']}）")
            mapping[key] = e
            rel_errs.append(rel)
        del m, inits
        gc.collect()
        log(f"  {f.name}: 累计映射 {len(mapping)}，HF 缓存 {len(cache)} 张量")

    log(f"值匹配完成 {time.time()-t0:.1f}s：映射 {len(mapping)}，未匹配 ONNX 张量 {len(unmatched_onnx)}，"
        f"跳过 {len(skipped)}（{args.skip_name_pattern!r}）")
    if rel_errs:
        log(f"匹配 rel_err: 最大 {max(rel_errs):.2e} / 中位 {sorted(rel_errs)[len(rel_errs)//2]:.2e}")

    # ---- 4) 适配器模块 -> ONNX，ΔW 与削顶模拟 ----
    r, alpha, scaling, per_module = load_adapter(Path(args.adapter))
    log(f"适配器: r={r} alpha={alpha} scaling={scaling:.3f} 模块 {len(per_module)}")
    dw_stats, unmapped_mods = [], []
    worst_clip, worst_rel, worst_match_rel = 0.0, 0.0, 0.0
    for mod, ab in sorted(per_module.items()):
        if "A" not in ab or "B" not in ab:
            unmapped_mods.append(mod + "(缺A/B)")
            continue
        dW = (ab["B"] @ ab["A"]) * scaling      # (N, K)，HF Linear 摆位
        hf_key = hf_key_for_module(hf_index, mod)
        e = mapping.get(hf_key) if hf_key else None
        if e is None:
            unmapped_mods.append(mod)
            continue
        if tuple(dW.shape) != hf_shapes[hf_key]:
            unmapped_mods.append(f"{mod}(形状 {dW.shape} != HF {hf_shapes[hf_key]})")
            continue
        rel_dw = float(np.linalg.norm(dW) / max(1e-12, np.linalg.norm(cache.get(hf_key) if hf_key in cache else hf_fetch(hf_key))))
        worst_rel = max(worst_rel, rel_dw)
        worst_match_rel = max(worst_match_rel, e["rel_err"])
        st = {"module": mod, "file": e["file"], "w": e["w"], "dtype": e["dtype_s"],
              "transposed": e["transposed"], "rel_dw": round(rel_dw, 5),
              "match_rel_err": e["rel_err"]}
        # 削顶模拟需要重新反量化该张量（按需加载所在文件的那一个 initializer）
        dw_stats.append(st)
    log(f"适配器映射: {len(dw_stats)}/{len(per_module)}；未映射 {len(unmapped_mods)}")
    if unmapped_mods:
        log("  未映射示例: " + "; ".join(unmapped_mods[:6]))

    # 削顶模拟：按文件分组，避免反复加载
    clip_flags = []
    by_file = defaultdict(list)
    for st in dw_stats:
        by_file[st["file"]].append(st)
    for fname, sts in by_file.items():
        m = onnx.load(str(model_dir / fname))
        inits = {t.name: t for t in m.graph.initializer}
        for st in sts:
            e = mapping[hf_key_for_module(hf_index, st["module"])]
            if e.get("scale") is None:
                continue  # fp32 张量：直接加 ΔW，无损
            mod = st["module"]
            dW = (per_module[mod]["B"] @ per_module[mod]["A"]) * scaling
            Wd, s_b, z_b, _ = dequant(e, inits)
            dW_o = dW.T if st["transposed"] else dW
            Wp = Wd.astype(np.float64) + dW_o
            q = Wp / s_b + (z_b if z_b is not None else 0.0)
            lo, hi = (0, 255) if e["dtype"] == 2 else (-128, 127)
            clip = float(np.mean((q < lo) | (q > hi)))
            qc = np.clip(np.rint(q), lo, hi)
            mse = float(np.mean(((qc - (z_b if z_b is not None else 0.0)) * s_b - Wp) ** 2))
            st["clip_frac"] = round(clip, 6)
            st["requant_mse"] = round(mse, 10)
            worst_clip = max(worst_clip, clip)
            if clip >= 0.01:
                clip_flags.append(f"{st['module']} clip={clip:.2%}")
            del Wd, Wp, q, qc
        del m, inits
        gc.collect()

    log(f"ΔW 相对幅度最大 {worst_rel:.4f}；重量化削顶最坏 {worst_clip:.4%}；"
        f"匹配误差最大 {worst_match_rel:.2e}")

    # ---- 5) verdict ----
    verdict = []
    n_mods = len(per_module)
    if len(dw_stats) == n_mods:
        verdict.append(f"GO-mapping: 适配器 {n_mods}/{n_mods} 模块全部映射到 ONNX 张量"
                       f"（含音频塔；跨 {[f.name for f in files]}）")
    else:
        verdict.append(f"NO-GO? 适配器映射 {len(dw_stats)}/{n_mods}，未映射示例: {unmapped_mods[:4]}")
    if worst_match_rel < 2e-2 and rel_errs:
        verdict.append(f"GO-同源: HF↔ONNX 值匹配最大 rel_err {worst_match_rel:.2e} < 2e-2")
    else:
        verdict.append(f"NO-GO? 值匹配误差过大: {worst_match_rel:.2e}")
    if worst_clip < 0.01:
        verdict.append(f"GO-量化: 按原 scale 重量化削顶 {worst_clip:.4%} < 1%")
    else:
        verdict.append(f"谨慎-量化: 削顶 {worst_clip:.2%} >= 1%（{len(clip_flags)} 个模块），"
                       "补丁需按新 range 重算 scale/zp")

    report = {
        "files": file_info, "scan_flags": scan_flags[:40],
        "hf_key_samples": list(hf_index)[:6],
        "mapping_size": len(mapping), "unmatched_onnx": unmatched_onnx[:20],
        "skipped": skipped, "adapter_modules": n_mods, "adapter_mapped": len(dw_stats),
        "unmapped_modules": unmapped_mods[:40],
        "worst": {"rel_dw": round(worst_rel, 5), "clip_frac": round(worst_clip, 6),
                  "match_rel_err": round(worst_match_rel, 8)},
        "dw_stats_sample": dw_stats[:12] + dw_stats[-6:],
        "verdict": verdict, "seconds": round(time.time() - t0, 1),
    }
    log("=" * 62)
    for v in verdict:
        log("  " + v)
    if scan_flags:
        log(f"  scan_flags({len(scan_flags)}): " + "; ".join(scan_flags[:6]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
