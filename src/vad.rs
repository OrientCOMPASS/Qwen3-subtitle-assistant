//! Silero VAD v4 —— 纯 Rust 实现（E3：彻底移除 sherpa-onnx / onnxruntime 依赖）。
//!
//! 移植蓝本与证据（finetune/silero_v4_port.md）：
//! * `finetune/silero_v4_reference.py` 是与 onnxruntime 位精确级一致的 numpy 参照
//!   （375 个中间层对拍最大相对误差 3.2e-6；ja.wav 225 帧带状态序列最大概率差 5.0e-6）；
//! * 权重 `assets/silero_v4_weights.bin`（0.62MB，40 张量 f32 平面拼接，include_bytes!
//!   嵌入——单 exe 打包友好）+ `assets/silero_v4_layout.json`（名字/shape/偏移）；
//! * 单测直接回放 `tests/fixtures/silero_v4_golden.json`（ORT 产出的连续序列黄金向量，
//!   容差 1e-5）。
//!
//! 数值策略：前向内部一律 f64 累加（每帧 ~0.6M MAC，开销可忽略），状态 h/c 按
//! ORT 契约以 f32 跨帧保存——与 numpy 参照（f64）一致，把 24 帧递归的漂移压在
//! 黄金向量容差内。
//!
//! 每帧 512 样本（16kHz 下 32ms）。段切分状态机语义对齐 sherpa-onnx 的
//! SileroVadModelConfig（threshold/min_silence/min_speech）与产品既有参数
//! （--vad-min-silence、--vad-buffer-secs），并沿用 finetune/s2tt_pipeline.py
//! 实测有效的合并/补齐/硬拆参数（gap<0.4s 合并、前后 pad 0.1s、超长硬拆）。

use anyhow::{bail, Result};
use log::info;

pub const VAD_WINDOW: usize = 512;
pub const SAMPLE_RATE: usize = 16_000;
/// 每帧秒数（512/16000 = 32ms）
pub const FRAME_SECS: f64 = VAD_WINDOW as f64 / SAMPLE_RATE as f64;

static WEIGHTS: &[u8] = include_bytes!("assets/silero_v4_weights.bin");
static LAYOUT: &str = include_str!("assets/silero_v4_layout.json");

// ------------------------------------------------------------------ 权重容器

struct Entry {
    off: usize, // 元素偏移（f32）
    len: usize,
}

/// Silero v4 前向模型（权重内嵌，无文件/无 ORT）。
pub struct SileroVad {
    data: Vec<f32>,
    map: Vec<(String, Entry)>,
    h: [[f64; 64]; 2],
    c: [[f64; 64]; 2],
}

impl SileroVad {
    pub fn new() -> Result<Self> {
        if WEIGHTS.len() % 4 != 0 {
            bail!("silero 权重 blob 长度非法: {}", WEIGHTS.len());
        }
        let data: Vec<f32> = WEIGHTS
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let layout: serde_json::Value = serde_json::from_str(LAYOUT)?;
        let mut map = Vec::new();
        for t in layout.as_array().ok_or_else(|| anyhow::anyhow!("layout 格式错误"))? {
            let name = t["name"].as_str().unwrap_or_default().to_string();
            let off = t["offset"].as_u64().unwrap_or(0) as usize;
            let len = t["len"].as_u64().unwrap_or(0) as usize;
            if off + len > data.len() {
                bail!("layout 越界: {name} {off}+{len} > {}", data.len());
            }
            map.push((name, Entry { off, len }));
        }
        let me = Self { data, map, h: [[0.0; 64]; 2], c: [[0.0; 64]; 2] };
        // 关键张量存在性即检（缺失立刻报错，而不是推理时 panic）
        for k in ["feature_extractor.forward_basis_buffer", "onnx::LSTM_398", "onnx::LSTM_420",
                  "decoder.decoder.1.weight", "adaptive_normalization.filter_"] {
            me.w(k);
        }
        Ok(me)
    }

    fn w(&self, name: &str) -> &[f32] {
        let (_, e) = self
            .map
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("silero 权重缺失: {name}"));
        &self.data[e.off..e.off + e.len]
    }

    pub fn reset(&mut self) {
        self.h = [[0.0; 64]; 2];
        self.c = [[0.0; 64]; 2];
    }

    /// 处理一帧 512 样本，返回语音概率 [0,1]；状态自动滚动。
    pub fn prob(&mut self, x: &[f32]) -> f32 {
        assert_eq!(x.len(), VAD_WINDOW);
        // ---- 1) reflect-pad 96 + STFT（basis conv k256 s64 → 8 帧 × 129 bin 幅度谱）
        let mut p = [0f64; 512 + 192];
        for j in 0..96 {
            p[j] = x[96 - j] as f64; // 左侧 reflect：x[96],x[95],...,x[1]
        }
        for j in 0..512 {
            p[96 + j] = x[j] as f64;
        }
        for k in 0..96 {
            p[608 + k] = x[510 - k] as f64; // 右侧 reflect：x[510],x[509],...
        }
        const FRAMES: usize = 8;
        const BINS: usize = 129;
        let basis = self.w("feature_extractor.forward_basis_buffer"); // (258,256)
        let mut mag = [[0f64; FRAMES]; BINS];
        for f in 0..FRAMES {
            let off = f * 64;
            for ch in 0..258 {
                let bw = &basis[ch * 256..(ch + 1) * 256];
                let mut acc = 0f64;
                for j in 0..256 {
                    acc += p[off + j] * bw[j] as f64;
                }
                let bin = if ch < BINS { ch } else { ch - BINS };
                if ch < BINS {
                    mag[bin][f] = acc; // 先存实部
                } else {
                    let re = mag[bin][f];
                    mag[bin][f] = (re * re + acc * acc).sqrt();
                }
            }
        }
        // ---- 2) log(mag·2^20 + 1) 与帧均值
        let mut logmag = [[0f64; FRAMES]; BINS];
        let mut m = [0f64; FRAMES];
        for f in 0..FRAMES {
            let mut s = 0f64;
            for b in 0..BINS {
                let v = (mag[b][f] * 1_048_576.0 + 1.0).ln();
                logmag[b][f] = v;
                s += v;
            }
            m[f] = s / BINS as f64;
        }
        // ---- 3) 自适应归一化：ctx=[m3,m2,m1]+m+[m6,m5,m4] → conv k7 → 全局均值标量 S
        let filt = self.w("adaptive_normalization.filter_"); // (7,)
        let mut ctx = [0f64; 14];
        ctx[0] = m[3];
        ctx[1] = m[2];
        ctx[2] = m[1];
        ctx[3..11].copy_from_slice(&m);
        ctx[11] = m[6];
        ctx[12] = m[5];
        ctx[13] = m[4];
        let mut s_sum = 0f64;
        for t in 0..FRAMES {
            let mut acc = 0f64;
            for k in 0..7 {
                acc += filt[k] as f64 * ctx[t + k];
            }
            s_sum += acc;
        }
        let s_mean = s_sum / FRAMES as f64;
        // ---- 4) 特征拼接 [258][8]：0..129=mag，129..258=logmag-S
        let mut feat = vec![0f64; 258 * FRAMES];
        for b in 0..BINS {
            for f in 0..FRAMES {
                feat[b * FRAMES + f] = mag[b][f];
                feat[(BINS + b) * FRAMES + f] = logmag[b][f] - s_mean;
            }
        }
        // ---- 5) first_layer：dw(k5,pad2,g=258)+relu → pw(258→16) + proj(258→16) → relu
        let dw = dwconv_k5(&feat, 258, FRAMES, self.w("first_layer.0.dw_conv.0.weight"),
                           self.w("first_layer.0.dw_conv.0.bias"));
        let dwr: Vec<f64> = dw.iter().map(|v| v.max(0.0)).collect();
        let mut a16 = conv1x1(&dwr, 258, FRAMES, self.w("first_layer.0.pw_conv.0.weight"),
                              self.w("first_layer.0.pw_conv.0.bias"), 16, 1);
        let prj = conv1x1(&feat, 258, FRAMES, self.w("first_layer.0.proj.weight"),
                          self.w("first_layer.0.proj.bias"), 16, 1);
        for i in 0..a16.len() {
            a16[i] = (a16[i] + prj[i]).max(0.0);
        }
        // ---- 6) encoder：1×1 s2 降采样与 MobileNet 块交替（8→4→2→1 帧）
        let e16 = relu(&conv1x1(&a16, 16, FRAMES, self.w("onnx::Conv_349"),
                                self.w("onnx::Conv_350"), 16, 2)); // [16][4]
        let t4 = e16.len() / 16;
        let b3dw = relu(&dwconv_k5(&e16, 16, t4, self.w("encoder.3.0.dw_conv.0.weight"),
                                   self.w("encoder.3.0.dw_conv.0.bias")));
        let mut a32 = conv1x1(&b3dw, 16, t4, self.w("encoder.3.0.pw_conv.0.weight"),
                              self.w("encoder.3.0.pw_conv.0.bias"), 32, 1);
        let p32 = conv1x1(&e16, 16, t4, self.w("encoder.3.0.proj.weight"),
                          self.w("encoder.3.0.proj.bias"), 32, 1);
        for i in 0..a32.len() {
            a32[i] = (a32[i] + p32[i]).max(0.0);
        } // [32][4]
        let e32 = relu(&conv1x1(&a32, 32, t4, self.w("onnx::Conv_352"),
                                self.w("onnx::Conv_353"), 32, 2)); // [32][2]
        let t2 = e32.len() / 32;
        let b7dw = relu(&dwconv_k5(&e32, 32, t2, self.w("encoder.7.0.dw_conv.0.weight"),
                                   self.w("encoder.7.0.dw_conv.0.bias")));
        let mut b7 = conv1x1(&b7dw, 32, t2, self.w("encoder.7.0.pw_conv.0.weight"),
                             self.w("encoder.7.0.pw_conv.0.bias"), 32, 1);
        for i in 0..b7.len() {
            b7[i] = (b7[i] + e32[i]).max(0.0); // 残差
        }
        let e32b = relu(&conv1x1(&b7, 32, t2, self.w("onnx::Conv_355"),
                                 self.w("onnx::Conv_356"), 32, 2)); // [32][1]
        let t1 = e32b.len() / 32;
        let b11dw = relu(&dwconv_k5(&e32b, 32, t1, self.w("encoder.11.0.dw_conv.0.weight"),
                                    self.w("encoder.11.0.dw_conv.0.bias")));
        let mut a64 = conv1x1(&b11dw, 32, t1, self.w("encoder.11.0.pw_conv.0.weight"),
                              self.w("encoder.11.0.pw_conv.0.bias"), 64, 1);
        let p64 = conv1x1(&e32b, 32, t1, self.w("encoder.11.0.proj.weight"),
                          self.w("encoder.11.0.proj.bias"), 64, 1);
        for i in 0..a64.len() {
            a64[i] = (a64[i] + p64[i]).max(0.0);
        }
        let e64 = relu(&conv1x1(&a64, 64, t1, self.w("onnx::Conv_358"),
                                self.w("onnx::Conv_359"), 64, 1)); // [64][1]
        debug_assert_eq!(e64.len(), 64);
        // ---- 7) LSTM ×2（ONNX 门序 i,o,f,c；W/R 行=门神经元 → 转置点积）
        let (h1, c1) = lstm_step(self.w("onnx::LSTM_398"), self.w("onnx::LSTM_399"),
                                 self.w("onnx::LSTM_400"), &e64, &self.h[0], &self.c[0]);
        let x2: Vec<f64> = h1.iter().map(|v| *v as f64).collect();
        let (h2, c2) = lstm_step(self.w("onnx::LSTM_418"), self.w("onnx::LSTM_419"),
                                 self.w("onnx::LSTM_420"), &x2, &self.h[1], &self.c[1]);
        self.h = [h1.map(|v| v as f32 as f64), h2.map(|v| v as f32 as f64)];
        self.c = [c1.map(|v| v as f32 as f64), c2.map(|v| v as f32 as f64)];
        // ---- 8) decoder：relu → conv1x1(64→1) → sigmoid
        let dw_ = self.w("decoder.decoder.1.weight"); // (1,64,1)
        let db = self.w("decoder.decoder.1.bias");
        let mut z = db[0] as f64;
        for n in 0..64 {
            z += dw_[n] as f64 * self.h[1][n].max(0.0);
        }
        (1.0 / (1.0 + (-z).exp())) as f32
    }

    /// 当前状态快照（测试用）。
    pub fn state_first8(&self) -> (Vec<f32>, Vec<f32>) {
        (self.h.iter().flatten().take(8).map(|v| *v as f32).collect(),
         self.c.iter().flatten().take(8).map(|v| *v as f32).collect())
    }
}

// ------------------------------------------------------------------ 算子

fn relu(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.max(0.0)).collect()
}

/// depthwise conv1d，k=5，pad=2（零填充），逐通道。in: [C][T] 平面。
fn dwconv_k5(inp: &[f64], c: usize, t: usize, w: &[f32], b: &[f32]) -> Vec<f64> {
    debug_assert_eq!(w.len(), c * 5);
    let mut out = vec![0f64; c * t];
    for ch in 0..c {
        for i in 0..t {
            let mut acc = b[ch] as f64;
            for k in 0..5usize {
                let tt = i as isize + k as isize - 2;
                if tt >= 0 && (tt as usize) < t {
                    acc += inp[ch * t + tt as usize] * w[ch * 5 + k] as f64;
                }
            }
            out[ch * t + i] = acc;
        }
    }
    out
}

/// pointwise(1×1) conv，可带 stride。in: [Cin][T]，w: (M,Cin,1) 平面 M*Cin，b: M。
fn conv1x1(inp: &[f64], cin: usize, t: usize, w: &[f32], b: &[f32], m: usize, stride: usize) -> Vec<f64> {
    debug_assert_eq!(w.len(), m * cin);
    let tout = (t + stride - 1) / stride;
    let mut out = vec![0f64; m * tout];
    for o in 0..m {
        for tt in 0..tout {
            let ti = tt * stride;
            if ti >= t {
                break;
            }
            let mut acc = b[o] as f64;
            for c in 0..cin {
                acc += inp[c * t + ti] * w[o * cin + c] as f64;
            }
            out[o * tout + tt] = acc;
        }
    }
    out
}

/// 单步 LSTM（ONNX 布局：W/R 为 (4H, dim)，行序 i,o,f,c；B=[Wb|Rb] 各 4H）。
fn lstm_step(w: &[f32], r: &[f32], bias: &[f32], x: &[f64], h0: &[f64; 64],
             c0: &[f64; 64]) -> ([f64; 64], [f64; 64]) {
    let mut c1 = [0f64; 64];
    let mut h1 = [0f64; 64];
    for n in 0..64 {
        // 门偏移：i:0..64 o:64..128 f:128..192 c:192..256
        let (zi, zo, zf, zc) = (n, 64 + n, 128 + n, 192 + n);
        let mut gi = bias[zi] as f64 + bias[256 + zi] as f64;
        let mut go = bias[zo] as f64 + bias[256 + zo] as f64;
        let mut gf = bias[zf] as f64 + bias[256 + zf] as f64;
        let mut gc = bias[zc] as f64 + bias[256 + zc] as f64;
        for k in 0..x.len() {
            gi += w[zi * 64 + k] as f64 * x[k];
            go += w[zo * 64 + k] as f64 * x[k];
            gf += w[zf * 64 + k] as f64 * x[k];
            gc += w[zc * 64 + k] as f64 * x[k];
        }
        for k in 0..64 {
            gi += r[zi * 64 + k] as f64 * h0[k];
            go += r[zo * 64 + k] as f64 * h0[k];
            gf += r[zf * 64 + k] as f64 * h0[k];
            gc += r[zc * 64 + k] as f64 * h0[k];
        }
        let i = 1.0 / (1.0 + (-gi).exp());
        let o = 1.0 / (1.0 + (-go).exp());
        let f = 1.0 / (1.0 + (-gf).exp());
        let cc = gc.tanh();
        c1[n] = f * c0[n] + i * cc;
        h1[n] = o * c1[n].tanh();
    }
    (h1, c1)
}

// ------------------------------------------------------------------ 段切分

/// 帧概率 → 语音帧区间（帧号，[start, end) ）。纯函数，便于单测。
/// 语义对齐 sherpa-onnx SileroVadModelConfig / silero utils.get_speech_timestamps：
/// 进入：prob ≥ threshold；退出：连续 min_silence_frames 帧 < threshold（段尾回退到
/// 静音起点）；短于 min_speech_frames 的段丢弃；超过 max_seg_frames 硬拆。
pub fn mask_to_segments(probs: &[f32], threshold: f32, min_speech_frames: usize,
                        min_silence_frames: usize, max_seg_frames: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    let mut silence = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        match start {
            None => {
                if p >= threshold {
                    start = Some(i);
                    silence = 0;
                }
            }
            Some(s) => {
                if p >= threshold {
                    silence = 0;
                    if max_seg_frames > 0 && i - s >= max_seg_frames {
                        out.push((s, i + 1));
                        start = Some(i + 1);
                    }
                } else {
                    silence += 1;
                    if silence >= min_silence_frames {
                        let end = i + 1 - silence;
                        if end > s && end - s >= min_speech_frames {
                            out.push((s, end));
                        }
                        start = None;
                        silence = 0;
                    } else if max_seg_frames > 0 && i - s >= max_seg_frames {
                        out.push((s, i + 1 - silence));
                        start = None;
                        silence = 0;
                    }
                }
            }
        }
    }
    if let Some(s) = start {
        let end = probs.len();
        if end > s && end - s >= min_speech_frames {
            out.push((s, end));
        }
    }
    // 合并间隔过近的相邻段（gap < merge 由调用方处理，这里只保证有序不重叠）
    out
}

/// 段后处理：gap < merge_gap 合并、前后 pad、超长硬拆（与 s2tt_pipeline.py 同参数）。
pub fn merge_pad_split(segs: Vec<(usize, usize)>, total_frames: usize, merge_gap_frames: usize,
                       pad_frames: usize, max_seg_frames: usize) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in segs {
        if let Some(last) = merged.last_mut() {
            if s.saturating_sub(last.1) < merge_gap_frames {
                last.1 = e.max(last.1);
                continue;
            }
        }
        merged.push((s, e));
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (s, e) in merged {
        let s = s.saturating_sub(pad_frames);
        let e = (e + pad_frames).min(total_frames);
        let dur = e - s;
        if max_seg_frames > 0 && dur > max_seg_frames {
            let n = dur / max_seg_frames + 1;
            let step = dur / n;
            for k in 0..n {
                out.push((s + k * step, if k + 1 == n { e } else { s + (k + 1) * step }));
            }
        } else {
            out.push((s, e));
        }
    }
    out
}

/// 产品级 VAD：喂 PCM（16k f32 单声道），产出语音段样本。
/// 全量缓冲（设计目标 1~5 分钟视频 ≈ 每分钟 3.8MB f32，内存无忧；换来实现简单与
/// 段边界前后 pad 的正确性）。
pub struct VadSegmenter {
    vad: SileroVad,
    raw: Vec<f32>,
    probs: Vec<f32>,
    threshold: f32,
    min_speech_frames: usize,
    min_silence_frames: usize,
    max_seg_frames: usize,
    merge_gap_frames: usize,
    pad_frames: usize,
}

impl VadSegmenter {
    pub fn new(threshold: f32, min_silence_secs: f32, max_seg_secs: f32) -> Result<Self> {
        Ok(Self {
            vad: SileroVad::new()?,
            raw: Vec::new(),
            probs: Vec::new(),
            threshold,
            min_speech_frames: ((0.25 / FRAME_SECS) as usize).max(1),
            min_silence_frames: ((min_silence_secs as f64 / FRAME_SECS) as usize).max(1),
            max_seg_frames: ((max_seg_secs as f64 / FRAME_SECS) as usize).max(2),
            merge_gap_frames: ((0.4 / FRAME_SECS) as usize).max(1),
            pad_frames: ((0.1 / FRAME_SECS) as usize).max(1),
        })
    }

    /// 追加样本；内部按 512 帧滚动计算概率（不足一帧的尾巴在 finish 时补零）。
    pub fn accept(&mut self, samples: &[f32]) {
        self.raw.extend_from_slice(samples);
        let mut done = self.probs.len() * VAD_WINDOW;
        while done + VAD_WINDOW <= self.raw.len() {
            let p = self.vad.prob(&self.raw[done..done + VAD_WINDOW]);
            self.probs.push(p);
            done += VAD_WINDOW;
        }
    }

    /// 结束：尾巴补零成整帧，产出 (起始样本号, 样本) 列表。
    pub fn finish(mut self) -> Vec<(usize, Vec<f32>)> {
        let rem = self.raw.len() % VAD_WINDOW;
        if rem > 0 {
            let tail = vec![0f32; VAD_WINDOW - rem];
            self.raw.extend_from_slice(&tail);
            let done = self.probs.len() * VAD_WINDOW;
            let p = self.vad.prob(&self.raw[done..done + VAD_WINDOW]);
            self.probs.push(p);
        }
        let segs = mask_to_segments(&self.probs, self.threshold, self.min_speech_frames,
                                    self.min_silence_frames, self.max_seg_frames);
        let segs = merge_pad_split(segs, self.probs.len(), self.merge_gap_frames,
                                   self.pad_frames, self.max_seg_frames);
        info!("VAD: {} 帧（{:.1}s）→ {} 个语音段", self.probs.len(),
              self.probs.len() as f64 * FRAME_SECS, segs.len());
        segs.into_iter()
            .map(|(s, e)| {
                let a = s * VAD_WINDOW;
                let b = (e * VAD_WINDOW).min(self.raw.len());
                (a, self.raw[a..b].to_vec())
            })
            .filter(|(_, v)| !v.is_empty())
            .collect()
    }

    /// 已计算的帧数（进度显示用）。
    pub fn frames_done(&self) -> usize {
        self.probs.len()
    }
}

// ------------------------------------------------------------------ 单测

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// 黄金向量回放：ORT 产出的连续序列（真实语音 24 帧 + 噪声 8 帧 + 单帧零状态），
    /// prob 与状态前 8 维都要在 1e-5 内——这是 numpy 参照→Rust 转写的机械正确性闸门。
    #[test]
    fn golden_vectors() {
        let gv: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/silero_v4_golden.json"))
                .expect("golden fixture");
        let mut vad = SileroVad::new().expect("权重加载");
        for seq in gv["seqs"].as_array().unwrap() {
            vad.reset();
            let inputs = seq["inputs"].as_array().unwrap();
            let probs = seq["probs"].as_array().unwrap();
            assert_eq!(inputs.len(), probs.len());
            for (i, chunk) in inputs.iter().enumerate() {
                let x: Vec<f32> = chunk.as_array().unwrap().iter()
                    .map(|v| v.as_f64().unwrap() as f32).collect();
                let p = vad.prob(&x);
                let want = probs[i].as_f64().unwrap() as f32;
                assert!(approx(p, want, 1e-5),
                        "seq={} frame={i}: got {p} want {want}", seq["name"].as_str().unwrap_or("?"));
            }
            let (h, c) = vad.state_first8();
            for (j, v) in seq["h_first8"].as_array().unwrap().iter().enumerate() {
                assert!(approx(h[j], v.as_f64().unwrap() as f32, 1e-5), "h[{j}] seq mismatch");
            }
            for (j, v) in seq["c_first8"].as_array().unwrap().iter().enumerate() {
                assert!(approx(c[j], v.as_f64().unwrap() as f32, 1e-5), "c[{j}] seq mismatch");
            }
        }
        // 单帧零状态
        vad.reset();
        let s = &gv["single_zero_state"];
        let x: Vec<f32> = s["input"].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap() as f32).collect();
        let p = vad.prob(&x);
        assert!(approx(p, s["prob"].as_f64().unwrap() as f32, 1e-5));
    }

    #[test]
    fn segmenter_mask_semantics() {
        // 30 帧：5..10 语音、15..25 语音；min_silence=3、min_speech=2
        let mut probs = vec![0.01f32; 30];
        for i in 5..10 { probs[i] = 0.9; }
        for i in 15..25 { probs[i] = 0.8; }
        let segs = mask_to_segments(&probs, 0.5, 2, 3, 0);
        assert_eq!(segs, vec![(5, 10), (15, 25)], "{segs:?}");
        // 短促噪声（1 帧）不成段
        let mut p2 = vec![0.01f32; 20];
        p2[7] = 0.9;
        assert!(mask_to_segments(&p2, 0.5, 2, 3, 0).is_empty());
        // 超长硬拆：max=6 帧
        let p3 = vec![0.9f32; 20];
        let segs3 = mask_to_segments(&p3, 0.5, 2, 3, 6);
        assert!(segs3.iter().all(|(s, e)| e - s <= 7), "{segs3:?}");
        assert_eq!(segs3.first().unwrap().0, 0);
        // 合并/pad/拆分后处理
        let merged = merge_pad_split(vec![(5, 10), (12, 20)], 100, 12, 3, 27);
        assert_eq!(merged.len(), 1); // gap=2 < 12 → 合并
        assert_eq!(merged[0], (2, 23)); // pad 3
    }

    #[test]
    fn silence_produces_no_segments() {
        let mut seg = VadSegmenter::new(0.5, 0.5, 60.0).unwrap();
        seg.accept(&vec![0.0f32; 16000]); // 1s 纯静音
        let segs = seg.finish();
        assert!(segs.is_empty(), "静音应无语音段: {:?}", segs.len());
    }
}
