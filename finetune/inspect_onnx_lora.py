#!/usr/bin/env python3
"""ONNX 权重补丁可行性探针——「LoRA 合并 → 官方 decoder.int8.onnx 手术式补丁」的前置事实核查。

路线背景（详见 finetune/INTEGRATION.md）：产品 ASR 走 sherpa-onnx 的官方 int8 ONNX
包（conv_frontend / encoder.int8 / decoder.int8 / tokenizer）。S2TT LoRA 只挂在 LM
解码器的 q/k/v/o/gate/up/down 投影上——**音频塔与 projector 完全没动**，所以理论上
只需把 ΔW = B·A·(α/r) 写回 decoder.int8.onnx 的对应权重张量，encoder/conv_frontend/
tokenizer 原样复制，即得到「快速直出版」模型目录，sherpa-onnx 与 Rust 侧零改动。

本探针不写补丁，只回答补丁的全部前置事实：
  1. **张量地图**：decoder.int8.onnx 里全部命中 7 类投影的 initializer——名称模式、
     dtype（fp32 还是 uint8/int8 量化）、形状、scale/zero_point 伴生张量、消费节点
     op（MatMul/Gemm/MatMulInteger/DequantizeLinear）与转置关系；
  2. **HF↔ONNX 对应性**：抽样若干层，把 ONNX 权重（量化的先反量化）与 HF 基座
     safetensors 逐元素对比（自动试「同向/转置」两种摆位）——补丁映射的正确性全押在这；
  3. **ΔW 幅度与量化风险**：对每个目标模块计算 ||ΔW||_F/||W||_F；对量化张量模拟
     「反量化 → 加 ΔW → 按原 per-channel scale/zp 重量化」：统计削顶元素占比与
     重量化 MSE——直接量化「按原 scale 写回」会不会把权重推出表示范围。

go/no-go 判据（打印在末尾，也写进 JSON）：
  * 7 类投影都能在 decoder 里找到（缺一类 = 导出器做过折叠/改名，路线要重估）；
  * HF↔ONNX 抽样最大绝对差 < 2e-2（反量化误差量级以内，说明是同源权重）；
  * 削顶元素占比 < 1%（超出则需按新 range 重算 scale——仍可行，但要多带一步）。

用法（CI：finetune.yml 的 onnx-inspect 模式；模型目录与 HF 基座都在 runner 缓存里）：
  python finetune/inspect_onnx_lora.py \
      --model-dir models/sherpa-onnx-qwen3-asr-0.6B-int8 \
      --hf-model Qwen/Qwen3-ASR-0.6B \
      --adapter art/out/real/lora \
      --out out/inspect/onnx_patch_report.json
"""

from __future__ import annotations

import argparse
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


# ------------------------------------------------------------------ ONNX 结构

def norm_pattern(name: str) -> str:
    """把张量名里的数字归一化成 N，聚出命名模式。"""
    return re.sub(r"\d+", "N", name)


def build_tensor_map(model) -> dict:
    import onnx

    inits = {t.name: t for t in model.graph.initializer}
    # 谁消费了哪个 initializer（op、输入位置、Gemm 转置属性）
    consumers: dict[str, list] = defaultdict(list)
    for node in model.graph.node:
        for pos, inp in enumerate(node.input):
            if inp in inits:
                attrs = {a.name: (a.i if a.type == onnx.AttributeProto.INT else a.f)
                         for a in node.attribute}
                consumers[inp].append({
                    "op": node.op_type, "pos": pos,
                    "transA": attrs.get("transA"), "transB": attrs.get("transB"),
                })
    return inits, consumers


def find_scale_zp(inits: dict, consumers: dict, name: str):
    """找量化张量的 scale/zero_point：先按命名约定，再看 DequantizeLinear/MatMulInteger 的输入。"""
    for suf in ("_scale", "_quant_scale", "_quantization_scale"):
        if name + suf in inits:
            scale = name + suf
            zp = None
            for z in (name + "_zero_point", name + "_zp", name + "_quant_zero_point"):
                if z in inits:
                    zp = z
                    break
            return scale, zp
    # 从消费节点反查：DequantizeLinear(x, scale, zp) / MatMulInteger(A, B, A_zp, B_zp)
    for c in consumers.get(name, []):
        pass  # 节点级信息在外层扫描时补齐（这里只处理命名约定命中）
    return None, None


def collect_qparam_names(model) -> set:
    """全图扫一遍：所有被当作 scale/zero_point 用的 initializer 名（要从目标里排除）。"""
    names = set()
    for node in model.graph.node:
        ins = list(node.input)
        if node.op_type in ("DequantizeLinear", "QuantizeLinear"):
            names.update(i for i in ins[1:3] if i)
        elif node.op_type == "MatMulInteger":
            names.update(i for i in ins[2:4] if i)
        elif node.op_type == "QLinearMatMul":
            names.update(ins[i] for i in (1, 2, 4, 5, 6, 7) if i < len(ins) and ins[i])
    return names


def scan_nodes_for_qparams(model, target_names: set) -> dict:
    """扫全图：对每个目标量化张量，从 DequantizeLinear / MatMulInteger / QLinearMatMul
    的输入位置找出 scale / zero_point 张量名。"""
    import onnx  # noqa: F401

    found: dict[str, dict] = defaultdict(dict)
    for node in model.graph.node:
        ins = list(node.input)
        if node.op_type == "DequantizeLinear" and ins and ins[0] in target_names:
            axis = next((a.i for a in node.attribute if a.name == "axis"), None)
            if len(ins) > 1:
                found[ins[0]]["scale"] = ins[1]
            if len(ins) > 2 and ins[2]:
                found[ins[0]]["zp"] = ins[2]
            if axis is not None:
                found[ins[0]]["axis"] = int(axis)
        elif node.op_type == "MatMulInteger":
            # (A, B, A_zero_point, B_zero_point)
            for pos, tname in enumerate(ins[:2]):
                if tname in target_names:
                    zpos = pos + 2
                    if len(ins) > zpos and ins[zpos]:
                        found[tname]["zp"] = ins[zpos]
        elif node.op_type == "QLinearMatMul":
            # (a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp)
            for tpos, spos, zpos in ((0, 1, 2), (3, 4, 5)):
                if len(ins) > zpos and ins[tpos] in target_names:
                    found[ins[tpos]]["scale"] = ins[spos]
                    if ins[zpos]:
                        found[ins[tpos]]["zp"] = ins[zpos]
    return dict(found)


# ------------------------------------------------------------------ HF 基座权重

def hf_lazy_open(hf_model: str):
    """按 key 惰性读取 HF 基座权重（bf16 -> float32 numpy），不整包进内存。"""
    import numpy as np
    from safetensors import safe_open

    root = Path(hf_model)
    if not root.is_dir():
        from huggingface_hub import snapshot_download  # 惰性：本地目录时不需要该依赖

        root = Path(snapshot_download(hf_model, allow_patterns=["*.safetensors", "*.json"]))
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"HF 模型目录里没有 safetensors: {root}")
    handles = [safe_open(str(f), framework="np") for f in files]
    index: dict[str, object] = {}
    for h in handles:
        for k in h.keys():
            index[k] = h

    def get(key: str):
        h = index.get(key)
        if h is None:
            return None
        t = h.get_tensor(key)
        # safetensors numpy 后端不支持 bf16：这类 key 换 torch 读
        return np.asarray(t, dtype=np.float32)

    def get_torch(key: str):
        import torch
        from safetensors.torch import safe_open as t_open

        h = index.get(key)
        if h is None:
            return None
        # 重新用 torch 打开同一文件
        path = h.filename
        with t_open(path, framework="pt") as th:
            return th.get_tensor(key).float().numpy()

    def fetch(key: str):
        try:
            v = get(key)
        except Exception:  # bf16 等 numpy 读不了的 dtype
            v = get_torch(key)
        return v

    return index, fetch


def find_hf_key(index: dict, layer: int, proj: str):
    """在 HF key 里找 `layers.{layer}...{proj}.weight`（thinker 的 language_model 部分）。"""
    pat = re.compile(rf"layers\.{layer}\..*\b{proj}\.weight$")
    cands = [k for k in index if pat.search(k) and "lora" not in k]
    # 排除音频塔（audio_tower / visual 等），只要 language_model/thinker 文本侧
    pref = [k for k in cands if ("language_model" in k or "thinker" in k)]
    pick = (pref or cands)
    return pick[0] if pick else None


# ------------------------------------------------------------------ 适配器 ΔW

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
                if not m:
                    continue
                per_module[m.group(1)][m.group(2)] = np.asarray(h.get_tensor(k), dtype=np.float32)
    return r, alpha, scaling, dict(per_module)


def module_layer_proj(mod: str):
    m = re.search(r"layers\.(\d+)\..*?\b(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", mod)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="sherpa-onnx-qwen3-asr-*-int8 目录")
    ap.add_argument("--decoder", default="decoder.int8.onnx")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    ap.add_argument("--sample-layers", type=int, default=4, help="HF↔ONNX 对应性抽样的层数")
    ap.add_argument("--out", default="")
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
    dec = model_dir / args.decoder
    log(f"模型目录: {model_dir}")
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            log(f"  {p.relative_to(model_dir)}  {p.stat().st_size/1e6:.1f}MB")
    if not dec.is_file():
        cands = sorted(model_dir.glob("*decoder*.onnx"))
        if not cands:
            raise SystemExit(f"找不到 decoder: {dec}")
        dec = cands[0]
    log(f"加载 {dec.name} ...")
    model = onnx.load(str(dec))
    log(f"onnx 加载完成 {time.time()-t0:.1f}s，opset={[ (o.domain or 'ai.onnx', o.version) for o in model.opset_import ]}")

    inits, consumers = build_tensor_map(model)
    log(f"initializer 总数 {len(inits)}，节点总数 {len(model.graph.node)}")

    # ---- 1) 目标张量地图 ----
    qparam_names = collect_qparam_names(model)
    targets = {}
    for name in inits:
        if name in qparam_names:
            continue
        hit = next((p for p in PROJ_TYPES if p in name), None)
        if hit and re.search(r"layers\.\d+", name):
            targets[name] = hit
    log(f"命中 7 类投影的 initializer: {len(targets)} 个（已排除 {len(qparam_names)} 个 scale/zp 伴生张量）")
    if not targets:
        # 导出器可能没保留权重名（如 onnx::MatMul_1234）——把所有 initializer 的
        # 命名模式聚出来供人工比对，探针结论转为 NO-GO?（需要按值匹配做图手术）
        allpats = defaultdict(int)
        for name in inits:
            allpats[norm_pattern(name)] += 1
        log("!! 未命中任何投影名，全部 initializer 命名模式（前 40）:")
        for pat, n in sorted(allpats.items(), key=lambda kv: -kv[1])[:40]:
            log(f"  {n:5d} × {pat}")

    qparams = scan_nodes_for_qparams(model, set(targets))

    DTYPE = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 10: "float16", 11: "double", 16: "bfloat16"}
    tensor_map = []
    by_proj = defaultdict(list)
    patterns = defaultdict(int)
    for name, proj in sorted(targets.items()):
        t = inits[name]
        dt = DTYPE.get(t.data_type, str(t.data_type))
        cons = consumers.get(name, [])
        qp = qparams.get(name, {})
        # 命名约定兜底
        if not qp:
            s, z = find_scale_zp(inits, consumers, name)
            if s:
                qp = {"scale": s, **({"zp": z} if z else {})}
        entry = {
            "name": name, "proj": proj, "dtype": dt, "shape": list(t.dims),
            "consumers": cons[:3], "scale": qp.get("scale"), "zp": qp.get("zp"),
            "axis": qp.get("axis"),
        }
        tensor_map.append(entry)
        by_proj[proj].append(entry)
        patterns[norm_pattern(name)] += 1

    log("命名模式聚类:")
    for pat, n in sorted(patterns.items()):
        log(f"  {n:4d} × {pat}")
    log("按投影类型统计:")
    for proj in PROJ_TYPES:
        ents = by_proj.get(proj, [])
        dts = defaultdict(int)
        for e in ents:
            dts[e["dtype"]] += 1
        ops = defaultdict(int)
        for e in ents:
            for c in e["consumers"]:
                ops[c["op"]] += 1
        log(f"  {proj:10s} {len(ents):3d} 个  dtype={dict(dts)}  消费op={dict(ops)}"
            + ("" if ents else "  <<< 缺失！"))

    # ---- 2) 适配器 ΔW ----
    r, alpha, scaling, per_module = load_adapter(Path(args.adapter))
    log(f"适配器: r={r} alpha={alpha} scaling={scaling:.3f} 模块数 {len(per_module)}")

    # ONNX 名 -> (layer, proj)，与适配器模块配对
    def onnx_layer(name):
        m = re.search(r"layers\.(\d+)", name)
        return int(m.group(1)) if m else None

    onnx_by_lp = {}
    for e in tensor_map:
        onnx_by_lp[(onnx_layer(e["name"]), e["proj"])] = e

    hf_index, hf_fetch = hf_lazy_open(args.hf_model)
    log(f"HF 基座 key 总数 {len(hf_index)}")

    report = {
        "decoder": str(dec), "tensor_map_patterns": dict(patterns),
        "per_proj": {p: len(by_proj.get(p, [])) for p in PROJ_TYPES},
        "lora": {"r": r, "alpha": alpha, "scaling": scaling, "modules": len(per_module)},
        "hf_onnx_match": [], "dw_stats": [], "flags": [],
    }

    sample_layers = sorted({module_layer_proj(m)[0] for m in per_module
                            if module_layer_proj(m)[0] is not None})
    picks = (sample_layers[: args.sample_layers // 2 + 1]
             + sample_layers[-(args.sample_layers // 2):]) if sample_layers else []
    picks = sorted(set(picks))[: args.sample_layers]

    worst_clip = 0.0
    worst_rel = 0.0
    matched = mismatched = unverified = 0
    for mod, ab in sorted(per_module.items()):
        layer, proj = module_layer_proj(mod)
        if layer is None:
            continue
        if "A" not in ab or "B" not in ab:
            report["flags"].append(f"{mod}: 缺 A 或 B")
            continue
        A, B = ab["A"], ab["B"]            # A: (r, K)  B: (N, r)
        dW = (B @ A) * scaling             # (N, K)，HF Linear weight 摆位
        e = onnx_by_lp.get((layer, proj))
        if e is None:
            report["flags"].append(f"ONNX 里找不到 layer{layer}.{proj}")
            mismatched += 1
            continue
        W_onnx = numpy_helper.to_array(inits[e["name"]])
        # 反量化（axis 感知：per-channel scale/zp 沿 DequantizeLinear 的 axis 摆放）
        quantized = e["dtype"] in ("uint8", "int8")
        if quantized:
            sname, zname = e.get("scale"), e.get("zp")
            if not sname:
                report["flags"].append(f"{e['name']}: 量化张量找不到 scale")
                unverified += 1
                continue
            scale = numpy_helper.to_array(inits[sname]).astype(np.float64)
            zp = (numpy_helper.to_array(inits[zname]).astype(np.float64)
                  if zname and zname in inits else np.float64(0.0))
            axis = e.get("axis")
            if axis is None:
                axis = 1 if W_onnx.ndim == 2 else 0
            axis %= W_onnx.ndim
            if scale.size == 1:
                s_bcast = scale.reshape([1] * W_onnx.ndim)
                z_bcast = np.reshape(zp, [1] * W_onnx.ndim) if np.ndim(zp) else zp
            else:
                if scale.size != W_onnx.shape[axis]:
                    # axis 标注与实际 scale 长度不符（或按命名约定兜底猜错）：试另一轴
                    other = 1 - axis if W_onnx.ndim == 2 else None
                    if other is not None and scale.size == W_onnx.shape[other]:
                        axis = other
                        report["flags"].append(f"{e['name']}: scale 长度按 axis={axis} 修正")
                    else:
                        report["flags"].append(
                            f"{e['name']}: scale 长度 {scale.size} 与形状 {list(W_onnx.shape)} 对不上")
                        unverified += 1
                        continue
                shp = [1] * W_onnx.ndim
                shp[axis] = scale.size
                s_bcast = scale.reshape(shp)
                z_bcast = zp.reshape(shp) if np.size(zp) == scale.size else zp
            Wf = (W_onnx.astype(np.float64) - z_bcast) * s_bcast
        else:
            Wf = W_onnx.astype(np.float64)
        # 摆位：ONNX 可能是 (K,N)（MatMul B 输入）或 (N,K)
        transposed = Wf.shape == (dW.shape[1], dW.shape[0]) and Wf.shape != dW.shape
        Wc = Wf.T if transposed else Wf
        if Wc.shape != dW.shape:
            report["flags"].append(f"{e['name']}: 形状对不上 onnx={Wf.shape} dW={dW.shape}")
            unverified += 1
            continue

        # HF↔ONNX 对应性（抽样层）
        if layer in picks:
            hf_key = find_hf_key(hf_index, layer, proj)
            if hf_key:
                Whf = hf_fetch(hf_key)
                if Whf is not None and Whf.shape == Wc.shape:
                    diff = float(np.abs(Whf.astype(np.float64) - Wc).max())
                    rel = diff / max(1e-9, float(np.abs(Whf).max()))
                    report["hf_onnx_match"].append({
                        "layer": layer, "proj": proj, "hf_key": hf_key,
                        "onnx": e["name"], "dtype": e["dtype"],
                        "transposed": bool(transposed),
                        "max_abs_diff": round(diff, 6), "rel": round(rel, 6)})
                    log(f"  对应性 layer{layer}.{proj}: dtype={e['dtype']} transposed={transposed} "
                        f"max|Δ|={diff:.2e} rel={rel:.2e}")
                    if rel < 2e-2:
                        matched += 1
                    else:
                        mismatched += 1
                else:
                    unverified += 1
            else:
                report["flags"].append(f"HF key 未找到 layer{layer}.{proj}")
                unverified += 1

        # ΔW 幅度 + 量化写回风险（模拟全部在 ONNX 存储摆位上做）
        rel_dw = float(np.linalg.norm(dW) / max(1e-12, np.linalg.norm(Wc)))
        worst_rel = max(worst_rel, rel_dw)
        st = {"layer": layer, "proj": proj, "dtype": e["dtype"],
              "rel_dw": round(rel_dw, 5)}
        if quantized:
            dW_o = dW.T if transposed else dW
            Wp = Wf + dW_o
            # round-trip 模拟：dequant q->(q-z)*s；requant W->W/s+z（沿用原 scale/zp）
            q = Wp / s_bcast + z_bcast
            lo, hi = (0, 255) if e["dtype"] == "uint8" else (-128, 127)
            clip_frac = float(np.mean((q < lo) | (q > hi)))
            qc = np.clip(np.rint(q), lo, hi)
            recon = (qc - z_bcast) * s_bcast
            mse = float(np.mean((recon - Wp) ** 2))
            st.update({"clip_frac": round(clip_frac, 6),
                       "requant_mse": round(mse, 10)})
            worst_clip = max(worst_clip, clip_frac)
        report["dw_stats"].append(st)

    log(f"ΔW 相对幅度: 最大 {worst_rel:.4f}（{len(report['dw_stats'])} 个模块）")
    log(f"量化写回模拟: 最坏削顶占比 {worst_clip:.4%}")

    # ---- go/no-go ----
    missing = [p for p in PROJ_TYPES if not by_proj.get(p)]
    verdict = []
    if missing:
        verdict.append(f"NO-GO? decoder 缺投影类型: {missing}（可能被导出器折叠/改名，需人工看图）")
    if mismatched and not matched:
        verdict.append("NO-GO? HF↔ONNX 权重对应性全部失败（不同源或映射错误）")
    elif matched:
        verdict.append(f"GO: HF↔ONNX 对应性抽样 {matched} 层通过（rel<2e-2）")
    if worst_clip < 0.01:
        verdict.append(f"GO: 按原 scale 重量化削顶 {worst_clip:.4%} < 1%")
    else:
        verdict.append(f"谨慎: 削顶 {worst_clip:.2%} >= 1%，补丁需按新 range 重算 scale")
    report["verdict"] = verdict
    report["tensor_map_sample"] = tensor_map[:24]
    report["seconds"] = round(time.time() - t0, 1)

    log("=" * 62)
    for v in verdict:
        log("  " + v)
    if report["flags"]:
        log(f"  flags({len(report['flags'])}): " + "; ".join(report["flags"][:8]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
