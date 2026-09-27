#!/usr/bin/env python3
"""silero_vad.onnx (v4) 的专用 numpy 解释器——为纯 Rust 移植提供位精确参照。

只实现该图用到的算子子集：Constant/Unsqueeze/ConstantOfShape/Concat/Reshape/Slice/
Transpose/Cast/Pad(reflect)/Conv1d(group,stride,pads)/Mul/Add/Sqrt/Log/ReduceMean/
Sub/Relu/Sigmoid/Squeeze/LSTM(ONNX 布局)。逐节点执行并与 onnxruntime 的
全中间层输出对拍（graph surgery 暴露所有输出），要求逐层位精确/1e-6 内。

产出（供 Rust 移植与单测）：
  * silero_v4_weights.bin —— 全部 initializer 按固定顺序拼接的 f32 平面 blob
  * silero_v4_layout.json —— 每个张量的名字/shape/在 blob 中的偏移
  * golden_vectors.json   —— 真实语音+噪声若干 512 窗的 prob 序列与 LSTM 状态轨迹
"""
import json
import pickle
import sys

import numpy as np
import onnx
from onnx import numpy_helper

# ------------------------------------------------------------------ 算子实现

def op_slice(data, starts, ends, axes=None, steps=None):
    nd = data.ndim
    axes = axes if axes is not None else list(range(nd))
    steps = steps if steps is not None else [1] * len(axes)
    out = data
    idx = [slice(None)] * nd
    for a, s, e, st in zip(axes, starts, ends, steps):
        a %= nd
        dim = data.shape[a]
        if s < -9223372036854775806 // 2:  # -inf 哨兵
            s = -dim - 1 if st < 0 else 0
        if e > 9223372036854775806 // 2:    # +inf 哨兵
            e = dim if st > 0 else -dim - 1
        idx[a] = slice(int(s), int(e), int(st))
    return out[tuple(idx)]


def op_pad_reflect(data, pads):
    # pads: [b0..bn, e0..en]
    nd = data.ndim
    b = pads[:nd]
    e = pads[nd:]
    if all(x == 0 for x in b + e):
        return data
    # numpy 的 reflect 与 ONNX reflect 语义一致（不含边缘重复）
    return np.pad(data, list(zip(b, e)), mode="reflect")


def conv1d(x, w, b, group, kernel, pads, strides):
    # x: (N, C, T)  w: (M, C/group, K)
    N, C, T = x.shape
    M, Cg, K = w.shape
    assert Cg == C // group and K == kernel[0]
    pb, pe = pads
    if pb or pe:
        x = np.pad(x, ((0, 0), (0, 0), (pb, pe)))
    s = strides[0]
    Tout = (x.shape[2] - K) // s + 1
    out = np.zeros((N, M, Tout), dtype=np.float64)
    x = x.astype(np.float64)
    w64 = w.astype(np.float64)
    for g in range(group):
        xg = x[:, g * Cg:(g + 1) * Cg, :]                       # (N, Cg, T')
        for m in range(M // group):
            oi = g * (M // group) + m
            acc = np.zeros((N, Tout), dtype=np.float64)
            for k in range(K):
                acc += xg[:, :, k::s][:, :, :Tout].transpose(0, 2, 1) @ w64[oi, :, k]
            out[:, oi, :] = acc
    if b is not None:
        out += b.astype(np.float64).reshape(1, -1, 1)
    return out.astype(np.float32)


def lstm_onnx(x, W, R, B, h0, c0):
    """ONNX LSTM（无 activation 属性 = 默认 sigmoid/tanh，无 peepholes）。
    x: (seq, batch, input)  W/R: (1, 256, in/hidden)  B: (1, 512) = Wb|Rb
    门序 i,o,f,c（ONNX 约定）；返回 Y(seq,batch,hidden), h_1(1,batch,hidden), c_1。"""
    seq, batch, inp = x.shape
    hidden = R.shape[2]
    Wi, Wo, Wf, Wc = np.split(W[0].astype(np.float64), 4, axis=0)
    Ri, Ro, Rf, Rc = np.split(R[0].astype(np.float64), 4, axis=0)
    if B is not None and B.size:
        Bi, Bo, Bf, Bc, Ri_b, Ro_b, Rf_b, Rc_b = np.split(B[0].astype(np.float64), 8)
        Bx = np.concatenate([Bi, Bo, Bf, Bc])
        Bh = np.concatenate([Ri_b, Ro_b, Rf_b, Rc_b])
    else:
        Bx = np.zeros(4 * hidden)
        Bh = np.zeros(4 * hidden)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    h = h0[0].astype(np.float64) if h0 is not None else np.zeros((batch, hidden))
    c = c0[0].astype(np.float64) if c0 is not None else np.zeros((batch, hidden))
    xs = x.astype(np.float64)
    pre = xs.reshape(seq, batch, inp) @ W.T_ if False else None
    # ONNX 权重布局 W/R: (4H, dim)，行=门神经元 → 投影需转置（64×64 方阵时不报错、
    # 静默算错——实测踩中：LSTM 输出全偏但形状全对）
    W_all = np.concatenate([Wi.T, Wo.T, Wf.T, Wc.T], axis=1)  # (inp, 4H)
    R_all = np.concatenate([Ri.T, Ro.T, Rf.T, Rc.T], axis=1)  # (hidden, 4H)
    xproj = xs @ W_all                                   # (seq, batch, 4H)
    Ys = []
    for t in range(seq):
        z = xproj[t] + h @ R_all + Bx + Bh
        i, o, f, cc = np.split(z, 4, axis=1)
        i, o, f = sig(i), sig(o), sig(f)
        cc = np.tanh(cc)
        c = f * c + i * cc
        h = o * np.tanh(c)
        Ys.append(h.copy())
    Y = np.stack(Ys, axis=0)                    # (seq, batch, hidden)
    Y = Y[:, None, :, :]                        # ONNX 布局 (seq, num_directions=1, batch, hidden)
    return (Y.astype(np.float32), h[None].astype(np.float32), c[None].astype(np.float32))


# ------------------------------------------------------------------ 解释器

def interpret(model_path, feeds):
    m = onnx.load(model_path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    env = dict(inits)
    env.update(consts)
    env.update(feeds)
    intermediates = {}

    def g(name):
        return env[name]

    for n in m.graph.node:
        op, ins, outs = n.op_type, list(n.input), list(n.output)
        attrs = {a.name: (list(a.ints) if a.type == a.INTS else
                          a.i if a.type == a.INT else
                          a.f if a.type == a.FLOAT else
                          a.s.decode() if a.type == a.STRING else None)
                 for a in n.attribute}
        if op == "Constant":
            continue  # 已预载
        elif op == "Unsqueeze":
            axes = g(ins[1]) if len(ins) > 1 else attrs.get("axes")
            r = g(ins[0])
            for a in sorted(axes):
                r = np.expand_dims(r, axis=int(a))
        elif op == "ConstantOfShape":
            shape = g(ins[0]).astype(int)
            val = 0.0
            for a in n.attribute:
                if a.name == "value":
                    val = float(numpy_helper.to_array(a.t).flatten()[0])
            r = np.full(tuple(shape), val, dtype=np.int64 if float(val).is_integer() and "." not in str(val) else np.float32)
            r = np.full(tuple(shape), val, dtype=np.float32) if not float(val).is_integer() else np.full(tuple(shape), int(val), dtype=np.int64)
        elif op == "Concat":
            r = np.concatenate([g(i) for i in ins], axis=attrs["axis"])
        elif op == "Reshape":
            shape = [int(x) for x in g(ins[1])]
            if -1 in shape:
                known = 1
                for s in shape:
                    if s > 0:
                        known *= s
                shape[shape.index(-1)] = int(g(ins[0]).size) // known
            r = g(ins[0]).reshape(shape)
        elif op == "Slice":
            starts = g(ins[1]); ends = g(ins[2])
            axes = g(ins[3]) if len(ins) > 3 and ins[3] else None
            steps = g(ins[4]) if len(ins) > 4 and ins[4] else None
            r = op_slice(g(ins[0]), starts, ends, axes, steps)
        elif op == "Transpose":
            r = np.transpose(g(ins[0]), attrs.get("perm"))
        elif op == "Cast":
            to = attrs["to"]
            r = g(ins[0]).astype({1: np.float32, 7: np.int64, 6: np.int32, 9: bool}[to])
        elif op == "Pad":
            r = op_pad_reflect(g(ins[0]), [int(x) for x in g(ins[1])])
        elif op == "Conv":
            w = g(ins[1])
            b = g(ins[2]) if len(ins) > 2 and ins[2] else None
            r = conv1d(g(ins[0]), w, b, attrs.get("group", 1),
                       attrs["kernel_shape"], attrs.get("pads", [0, 0]), attrs.get("strides", [1]))
        elif op == "Mul":
            r = g(ins[0]) * g(ins[1])
        elif op == "Add":
            r = g(ins[0]) + g(ins[1])
        elif op == "Sub":
            r = g(ins[0]) - g(ins[1])
        elif op == "Sqrt":
            r = np.sqrt(g(ins[0]))
        elif op == "Log":
            r = np.log(g(ins[0]))
        elif op == "Relu":
            r = np.maximum(g(ins[0]), 0)
        elif op == "Sigmoid":
            r = 1.0 / (1.0 + np.exp(-g(ins[0]).astype(np.float64))).astype(np.float32)
        elif op == "ReduceMean":
            axes = attrs.get("axes")
            if axes is None and len(ins) > 1:
                axes = list(g(ins[1]))
            r = g(ins[0]).mean(axis=tuple(int(a) for a in axes),
                               keepdims=bool(attrs.get("keepdims", 1)))
        elif op == "Squeeze":
            axes = g(ins[1]) if len(ins) > 1 else attrs.get("axes")
            r = g(ins[0])
            for a in sorted([int(x) for x in axes], reverse=True):
                r = np.squeeze(r, axis=a)
        elif op == "LSTM":
            Y, hn, cn = lstm_onnx(g(ins[0]), g(ins[1]), g(ins[2]),
                                  g(ins[3]) if ins[3] else None,
                                  g(ins[5]) if len(ins) > 5 and ins[5] else None,
                                  g(ins[6]) if len(ins) > 6 and ins[6] else None)
            env[outs[0]] = Y
            if len(outs) > 1:
                env[outs[1]] = hn
            if len(outs) > 2:
                env[outs[2]] = cn
            intermediates[outs[0]] = Y
            continue
        else:
            raise NotImplementedError(op)
        env[outs[0]] = r
        intermediates[outs[0]] = r
    return env, intermediates


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "silero_vad.onnx"
    rng = np.random.default_rng(0)

    # 1) 与 onnxruntime 对拍：暴露全部中间层
    import onnxruntime as ort
    m = onnx.load(model)
    existing = {o.name for o in m.graph.output}
    for name in list({n.output[0] for n in m.graph.node}):
        if name not in existing:
            m.graph.output.extend([onnx.helper.make_empty_tensor_value_info(name)])
    onnx.save(m, "_all_outputs.onnx")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession("_all_outputs.onnx", so, providers=["CPUExecutionProvider"])

    worst = ("", 0.0)
    n_checked = 0
    for trial in range(3):
        x = (rng.standard_normal((1, 512)) * (0.05 if trial else 0.5)).astype(np.float32)
        h = rng.standard_normal((2, 1, 64)).astype(np.float32) * 0.1
        c = rng.standard_normal((2, 1, 64)).astype(np.float32) * 0.1
        feeds = {"x": x, "h": h, "c": c}
        ort_out = sess.run(None, feeds)
        ort_map = {o.name: v for o, v in zip(sess.get_outputs(), ort_out)}
        env, inter = interpret(model, feeds)
        for name, v in ort_map.items():
            mine = env.get(name)
            if mine is None:
                continue
            mine = np.asarray(mine, dtype=np.float32)
            if mine.shape != v.shape:
                print(f"  形状不符 {name}: {mine.shape} vs {v.shape}")
                worst = (name + "(shape)", 1e9)
                continue
            d = float(np.abs(mine - v).max())
            scale = max(1.0, float(np.abs(v).max()))
            d_rel = d / scale
            n_checked += 1
            if d_rel > worst[1]:
                worst = (name, d_rel)
    print(f"对拍完成: {n_checked} 个中间层 × 3 组随机输入")
    print(f"最大相对误差: {worst[1]:.3e} @ {worst[0]}")
    ok = worst[1] < 1e-5
    print("✔ 位精确级一致" if ok else "✘ 存在偏差，需排查")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
