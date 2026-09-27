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
use log::debug;

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
            // layout 里的 offset 单位是字节（打包脚本按 f32 平面拼接），len 是元素数
            let off = (t["offset"].as_u64().unwrap_or(0) as usize) / 4;
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

    #[cfg(test)]
    pub fn reset(&mut self) {
        self.h = [[0.0; 64]; 2];
        self.c = [[0.0; 64]; 2];
    }
    // 注：reset/state_first8 仅黄金向量测试使用（下方 tests mod），以 cfg(test)
    // 门控消除发布构建的 dead_code 警告。

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
    #[cfg(test)]
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

/// 一段已确认的 VAD 语音段：`start_sample` 为整条音频中的起始样本号（16kHz），
/// `samples` 为段波形（首尾各含 0.1s pad，见 SegTracker）。
pub struct VadSeg {
    pub start_sample: usize,
    pub samples: Vec<f32>,
}

/// 流式段跟踪器：逐帧（512 样本/帧）消费 silero 语音概率，段一旦「确认」立即
/// 产出帧区间（含 pad）——无需等整条音频的概率序列齐活。这是「VAD 切出一段
/// 就喂一段给 ASR」流式管线的核心状态机。
///
/// 语义与批量参照实现（tests 里的 mask_to_segments + merge_pad_split，二者有
/// 逐案对拍的等价性测试）对齐：
/// * 进入：prob ≥ threshold；
/// * 退出：连续 min_silence 帧低于阈值（段尾回退到静音起点）；
/// * 合并：段尾确认后 merge_gap 帧内语音恢复 → 并回同一段（等价批量版
///   gap<merge_gap 的合并；默认 min_silence 0.5s > merge_gap 0.4s，
///   合并窗口在确认时已过期，与批量版一样不触发）；
/// * 短段：最终跨度 < min_speech 帧丢弃；
/// * 硬拆：语音连续达 max_seg 帧强制切开，新段从切点继续；
/// * pad：产出区间首尾各扩 pad 帧（越界钳制）；硬拆相邻段因双向 pad 有
///   2*pad 帧重叠，与批量版「每段各自 pad」的行为一致。
///
/// 与批量版的已知差异（流式化的刻意简化，注释于等价性测试）：
/// 1. 超长语音批量版是「硬拆→重合并→等分成 ≤max_seg 的块」，流式版按序
///    切出 ≈max_seg+2*pad 的块——单段时长上限同量级，对 ASR 无实质差异；
/// 2. <min_speech 的碎片若与 merge_gap 内的后续语音连成一段，流式版保留合并后
///    的整段（起点含碎片），批量版在 mask 阶段先丢碎片再从恢复点起段。
pub struct SegTracker {
    threshold: f32,
    min_speech: usize,
    min_silence: usize,
    max_seg: usize, // 0 = 不硬拆
    merge_gap: usize,
    pad: usize,

    n_frames: usize,
    in_speech: bool,
    start: usize, // 当前语音段起点（帧）
    silence_run: usize,
    /// 已退出（连续静音 ≥ min_silence）但仍在 merge_gap 观察窗内的段
    pending: Option<(usize, usize)>,
    /// 硬拆出的段：区间已定，等 n_frames ≥ end+pad 补完 pad 尾巴后产出
    hard_wait: Vec<(usize, usize)>,
}

impl SegTracker {
    pub fn new(threshold: f32, min_speech: usize, min_silence: usize,
               max_seg: usize, merge_gap: usize, pad: usize) -> Self {
        Self {
            threshold,
            min_speech: min_speech.max(1),
            min_silence: min_silence.max(1),
            max_seg,
            merge_gap,
            pad,
            n_frames: 0,
            in_speech: false,
            start: 0,
            silence_run: 0,
            pending: None,
            hard_wait: Vec::new(),
        }
    }

    /// 已消费帧数。
    pub fn frames(&self) -> usize {
        self.n_frames
    }

    /// pad 帧数（VadSegmenter 修剪样本缓冲用）。
    pub fn pad_frames(&self) -> usize {
        self.pad
    }

    /// 喂入一帧概率；返回本次新确认的段 (start_frame, end_frame)（含 pad、已钳制）。
    /// 一帧可能同时确认多段（如前段到期 + 硬拆段 pad 尾巴补齐）。
    pub fn push(&mut self, p: f32) -> Vec<(usize, usize)> {
        let i = self.n_frames;
        self.n_frames += 1;
        let mut out: Vec<(usize, usize)> = Vec::new();

        // 先收 pad 尾巴已补齐的硬拆段
        self.flush_hard_wait(&mut out, false);

        let speech = p >= self.threshold;
        if self.in_speech {
            if speech {
                self.silence_run = 0;
                if self.max_seg > 0 && i >= self.start + self.max_seg {
                    // 硬拆：[start, i+1) 入等待队列（补 pad 尾巴后产出），新段从 i+1 继续
                    self.hard_wait.push((self.start, i + 1));
                    self.start = i + 1;
                }
            } else {
                self.silence_run += 1;
                if self.silence_run >= self.min_silence {
                    let end = i + 1 - self.silence_run;
                    self.in_speech = false;
                    self.pending = Some((self.start, end));
                    self.silence_run = 0;
                } else if self.max_seg > 0 && i >= self.start + self.max_seg {
                    // 批量版同款边角：静音中触发超长上限，提前定格段尾
                    let end = i + 1 - self.silence_run;
                    self.in_speech = false;
                    self.pending = Some((self.start, end));
                    self.silence_run = 0;
                }
            }
        } else if speech {
            let merged = match self.pending {
                Some((s, e)) if i < e + self.merge_gap => {
                    // 合并窗内语音恢复：并回原段（中间静音帧含入段内）
                    self.in_speech = true;
                    self.start = s;
                    true
                }
                _ => false,
            };
            if !merged {
                self.emit_pending(&mut out);
                self.in_speech = true;
                self.start = i;
            }
            self.pending = None;
            self.silence_run = 0;
        } else if let Some((_, e)) = self.pending {
            // 静音延续：合并观察窗到期即确认
            if i + 1 >= e + self.merge_gap {
                self.emit_pending(&mut out);
            }
        }

        // 本帧刚进 pending 的段：min_silence ≥ merge_gap 时观察窗已过期，立即确认
        if let Some((_, e)) = self.pending {
            if self.n_frames >= e + self.merge_gap {
                self.emit_pending(&mut out);
            }
        }
        // pad=0 时本帧新产生的硬拆段可立即产出
        self.flush_hard_wait(&mut out, false);
        out
    }

    /// 流结束：冲刷未闭合的语音段 / pending / 硬拆等待段（pad 按已有帧钳制）。
    pub fn flush(&mut self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        if self.in_speech {
            let e = self.n_frames;
            if e > self.start && e - self.start >= self.min_speech {
                out.push(self.padded(self.start, e));
            }
            self.in_speech = false;
        }
        self.emit_pending(&mut out);
        self.flush_hard_wait(&mut out, true);
        out.sort_by_key(|&(s, _)| s);
        out
    }

    /// 当前仍未产出、还需要原始样本的最早帧号（已减 pad）；无依赖时 None。
    /// VadSegmenter 用它修剪样本缓冲——流式内存上限 = 单段最长时长量级。
    pub fn min_needed_frame(&self) -> Option<usize> {
        let mut m: Option<usize> = None;
        let mut consider = |f: usize| {
            let f = f.saturating_sub(self.pad);
            m = Some(m.map_or(f, |x| x.min(f)));
        };
        if self.in_speech {
            consider(self.start);
        }
        if let Some((s, _)) = self.pending {
            consider(s);
        }
        for (s, _) in &self.hard_wait {
            consider(*s);
        }
        m
    }

    fn emit_pending(&mut self, out: &mut Vec<(usize, usize)>) {
        if let Some((s, e)) = self.pending.take() {
            if e > s && e - s >= self.min_speech {
                out.push(self.padded(s, e));
            }
        }
    }

    fn flush_hard_wait(&mut self, out: &mut Vec<(usize, usize)>, force: bool) {
        if self.hard_wait.is_empty() {
            return;
        }
        // mem::take 结束对 hard_wait 的可变借用，循环体内才能再调 &self 的 padded()
        let items = std::mem::take(&mut self.hard_wait);
        let mut remain = Vec::new();
        for (s, e) in items {
            if force || self.n_frames >= e + self.pad {
                out.push(self.padded(s, e));
            } else {
                remain.push((s, e));
            }
        }
        self.hard_wait = remain;
    }

    fn padded(&self, s: usize, e: usize) -> (usize, usize) {
        (s.saturating_sub(self.pad), (e + self.pad).min(self.n_frames))
    }
}

/// 产品级流式 VAD：喂 PCM（16k f32 单声道），**实时**产出已确认的语音段。
/// 内存有界：样本缓冲只保留「未闭合段 + pad」所需窗口（单段最长时长量级，
/// 默认 60s ≈ 3.8MB），已产出段即被搬走；旧版全量缓冲行为不复存在。
pub struct VadSegmenter {
    vad: SileroVad,
    tracker: SegTracker,
    /// 样本窗口：绝对样本号区间 [buf_base, buf_base + buf.len())
    buf: Vec<f32>,
    buf_base: usize,
    segs_out: usize,
}

impl VadSegmenter {
    pub fn new(threshold: f32, min_silence_secs: f32, max_seg_secs: f32) -> Result<Self> {
        let max_seg_frames = if max_seg_secs > 0.0 {
            ((max_seg_secs as f64 / FRAME_SECS) as usize).max(2)
        } else {
            0
        };
        Ok(Self {
            vad: SileroVad::new()?,
            tracker: SegTracker::new(
                threshold,
                ((0.25 / FRAME_SECS) as usize).max(1),   // min_speech 0.25s
                ((min_silence_secs as f64 / FRAME_SECS) as usize).max(1),
                max_seg_frames,
                ((0.4 / FRAME_SECS) as usize).max(1),    // merge_gap 0.4s
                ((0.1 / FRAME_SECS) as usize).max(1),    // pad 0.1s
            ),
            buf: Vec::new(),
            buf_base: 0,
            segs_out: 0,
        })
    }

    /// 追加样本；返回本次调用中确认完成的段（通常 0~1 个，硬拆点附近可能多个）。
    /// 调用方拿到即可直接送 ASR——这正是流式管线的节拍。
    pub fn accept(&mut self, samples: &[f32]) -> Vec<VadSeg> {
        self.buf.extend_from_slice(samples);
        let mut out = Vec::new();
        self.pump(&mut out);
        self.trim();
        out
    }

    /// 流结束：尾巴补零成整帧、跑完最后一帧，冲刷所有未闭合段。
    pub fn finish(&mut self) -> Vec<VadSeg> {
        let n_fed = self.buf_base + self.buf.len();
        let rem = n_fed % VAD_WINDOW;
        let mut out = Vec::new();
        if rem > 0 {
            self.buf.extend(std::iter::repeat(0.0f32).take(VAD_WINDOW - rem));
            self.pump(&mut out);
        }
        for (s, e) in self.tracker.flush() {
            out.push(self.cut(s, e));
        }
        out.sort_by_key(|v| v.start_sample);
        self.segs_out += out.len();
        debug!("VAD: {} 帧（{:.1}s）→ 共 {} 个语音段",
               self.tracker.frames(), self.tracker.frames() as f64 * FRAME_SECS, self.segs_out);
        out
    }

    /// 把可整帧计算的样本全部喂给 silero + 状态机，收集确认段。
    fn pump(&mut self, out: &mut Vec<VadSeg>) {
        loop {
            let frame_start = self.tracker.frames() * VAD_WINDOW; // 绝对样本号
            if frame_start < self.buf_base {
                break; // 不应发生：trim 保证 buf_base ≤ 未消费帧起点
            }
            let rel = frame_start - self.buf_base;
            if rel + VAD_WINDOW > self.buf.len() {
                break;
            }
            let p = self.vad.prob(&self.buf[rel..rel + VAD_WINDOW]);
            for (s, e) in self.tracker.push(p) {
                self.segs_out += 1;
                out.push(self.cut(s, e));
            }
        }
    }

    /// 帧区间 → 样本切片拷贝（绝对样本号 + 波形）。
    fn cut(&self, s_frame: usize, e_frame: usize) -> VadSeg {
        let a = s_frame * VAD_WINDOW;
        let fed = self.buf_base + self.buf.len();
        let b = (e_frame * VAD_WINDOW).min(fed).max(a);
        let ra = a.saturating_sub(self.buf_base).min(self.buf.len());
        let rb = b.saturating_sub(self.buf_base).min(self.buf.len()).max(ra);
        VadSeg { start_sample: a, samples: self.buf[ra..rb].to_vec() }
    }

    /// 修剪样本缓冲：只保留「未闭合段 + pad」与「未来段 pad-before」所需窗口。
    fn trim(&mut self) {
        let frames = self.tracker.frames();
        let mut keep_from_frame = frames.saturating_sub(self.tracker.pad_frames());
        if let Some(f) = self.tracker.min_needed_frame() {
            keep_from_frame = keep_from_frame.min(f);
        }
        let keep_from = keep_from_frame * VAD_WINDOW;
        if keep_from > self.buf_base {
            let drop = (keep_from - self.buf_base).min(self.buf.len());
            self.buf.drain(..drop);
            self.buf_base += drop;
        }
    }
}

// ------------------------------------------------------------------ 批量参照实现
// v0.4 及以前的产品路径：收齐全片概率后 mask_to_segments → merge_pad_split。
// 现仅作为流式 SegTracker 的语义参照保留（等价性测试逐案对拍），不进发布构建。

#[cfg(test)]
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
    out
}

#[cfg(test)]
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

    // ---------------- 流式 SegTracker ----------------

    /// 确定性伪随机（LCG），避免为测试引入 rand 依赖。
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// 批量参照（v0.4 产品行为）：mask + merge/pad（无硬拆场景）。
    fn batch_reference(probs: &[f32], thr: f32, min_speech: usize, min_silence: usize,
                       merge_gap: usize, pad: usize) -> Vec<(usize, usize)> {
        let segs = mask_to_segments(probs, thr, min_speech, min_silence, 0);
        merge_pad_split(segs, probs.len(), merge_gap, pad, 0)
    }

    /// 等价性闸门：300 组确定性随机「静音/语音 run」概率序列 ×2 种参数形态
    /// （min_silence > merge_gap 的默认形态、min_silence < merge_gap 的短静音形态），
    /// 流式 SegTracker 的产出必须与批量参照逐案一致。生成约束避开两处
    /// **已文档化**的刻意差异：语音 run ≥ min_speech（碎片并窗）、不触发硬拆。
    #[test]
    fn streaming_tracker_matches_batch_reference() {
        let (thr, min_speech, merge_gap, pad) = (0.5f32, 8usize, 12usize, 3usize);
        for &min_silence in &[15usize, 6usize] {
            let mut rng = Lcg(0x5eed ^ (min_silence as u64));
            for case in 0..300 {
                let mut probs: Vec<f32> = Vec::new();
                probs.extend(vec![0.01; rng.below(10) + 1]);
                while probs.len() < 400 {
                    let run = min_speech + rng.below(40);
                    probs.extend(vec![0.9; run]);
                    probs.extend(vec![0.01; rng.below(3 * min_silence) + 1]);
                }
                if case % 3 == 0 {
                    // 截尾：覆盖「结束于语音中 / 刚退出 / 观察窗内」等相位
                    let cut = rng.below(25).min(probs.len());
                    probs.truncate(probs.len() - cut);
                }
                if probs.is_empty() {
                    continue;
                }
                let mut t = SegTracker::new(thr, min_speech, min_silence, 0, merge_gap, pad);
                let mut got: Vec<(usize, usize)> = Vec::new();
                for &p in &probs {
                    got.extend(t.push(p));
                }
                got.extend(t.flush());
                got.sort_by_key(|&(s, _)| s);
                let want = batch_reference(&probs, thr, min_speech, min_silence, merge_gap, pad);
                assert_eq!(got, want,
                           "min_silence={min_silence} case={case} len={}", probs.len());
            }
        }
    }

    /// merge_gap 窗口内语音恢复 → 并段（流式与批量一致）。
    #[test]
    fn tracker_merges_within_gap() {
        // min_silence=3、merge_gap=6：段尾确认后 6 帧内恢复语音应并回同一段
        let mut probs = vec![0.01f32; 40];
        for i in 5..12 { probs[i] = 0.9; }   // 语音 7 帧
        for i in 16..24 { probs[i] = 0.9; }  // 静音 4 帧（≥3 退出、<6 合并窗）后恢复
        let mut t = SegTracker::new(0.5, 2, 3, 0, 6, 2);
        let mut got: Vec<(usize, usize)> = Vec::new();
        for &p in &probs { got.extend(t.push(p)); }
        got.extend(t.flush());
        assert_eq!(got.len(), 1, "应合并为一段: {got:?}");
        assert_eq!(batch_reference(&probs, 0.5, 2, 3, 6, 2), got);
    }

    /// 硬拆：连续超长语音按 max_seg 切块，块长受限、覆盖无缝（pad 允许重叠）。
    #[test]
    fn tracker_hard_split_bounds_and_coverage() {
        let (min_speech, min_silence, merge_gap, pad) = (8usize, 15usize, 12usize, 3usize);
        let max_seg = 50usize;
        let n = 5 * max_seg + 30;
        let mut t = SegTracker::new(0.5, min_speech, min_silence, max_seg, merge_gap, pad);
        let mut got: Vec<(usize, usize)> = Vec::new();
        for _ in 0..n { got.extend(t.push(0.9)); }
        got.extend(t.flush());
        assert!(got.len() >= 6, "应至少切出 6 块: {}", got.len());
        assert_eq!(got[0].0, 0);
        assert_eq!(got.last().unwrap().1, n);
        for seg in &got {
            assert!(seg.1 > seg.0);
            assert!(seg.1 - seg.0 <= max_seg + 2 * pad + 1, "块过长: {seg:?}");
        }
        for w in got.windows(2) {
            assert!(w[1].0 <= w[0].1, "覆盖出现空洞: {:?} -> {:?}", w[0], w[1]);
        }
    }

    /// 硬拆段在流中途产出时，必须等到样本覆盖 pad 尾巴（n_frames ≥ end+pad）。
    #[test]
    fn tracker_hard_split_waits_for_pad_tail() {
        let mut t = SegTracker::new(0.5, 4, 10, 20, 5, 3); // max_seg=20, pad=3
        let mut got: Vec<(usize, usize)> = Vec::new();
        for i in 0..20 {
            got.extend(t.push(0.9));
            assert!(got.is_empty(), "帧 {i} 不应产出");
        }
        // 帧 i=20 触发硬拆（i ≥ start+max_seg）：段 [0,21) 入等待队列，
        // 需再等 pad=3 帧（n_frames ≥ 21+3=24）才产出
        got.extend(t.push(0.9));
        assert!(got.is_empty(), "硬拆后应等待 pad 尾巴");
        got.extend(t.push(0.9)); // n_frames=22 < 24
        got.extend(t.push(0.9)); // n_frames=23 < 24
        assert!(got.is_empty());
        got.extend(t.push(0.9)); // n_frames=24 = end(21)+pad(3) → 产出
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0], (0, 24), "{got:?}"); // s-pad 钳制为 0, e+pad=24
    }

    // ---------------- VadSegmenter（含 silero 实算的管线测试） ----------------

    #[test]
    fn silence_produces_no_segments() {
        let mut seg = VadSegmenter::new(0.5, 0.5, 60.0).unwrap();
        assert!(seg.accept(&vec![0.0f32; 16000]).is_empty()); // 1s 纯静音
        assert!(seg.accept(&vec![0.0f32; 16000]).is_empty());
        assert!(seg.finish().is_empty(), "静音应无语音段");
    }

    /// 用黄金向量 fixture 的真实语音帧过一遍流式分段器：不断言段数（那是模型
    /// 行为），只断言管线不变式——起始样本帧对齐、样本非空、按时间单调、
    /// 长度受 max_seg 上限约束。
    #[test]
    fn golden_speech_frames_flow_through_streaming_segmenter() {
        let gv: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/silero_v4_golden.json"))
                .expect("golden fixture");
        let seq = &gv["seqs"][0];
        let mut seg = VadSegmenter::new(0.5, 0.25, 60.0).unwrap();
        let mut all: Vec<VadSeg> = Vec::new();
        for chunk in seq["inputs"].as_array().unwrap() {
            let x: Vec<f32> = chunk.as_array().unwrap().iter()
                .map(|v| v.as_f64().unwrap() as f32).collect();
            all.extend(seg.accept(&x));
        }
        // 补 1s 静音让语音段确认退出
        all.extend(seg.accept(&vec![0.0f32; 16000]));
        all.extend(seg.finish());
        let mut last_start = 0usize;
        let max_samples = ((60.0 / FRAME_SECS) as usize + 8) * VAD_WINDOW;
        for s in &all {
            assert_eq!(s.start_sample % VAD_WINDOW, 0, "起始样本应帧对齐");
            assert!(!s.samples.is_empty());
            assert!(s.start_sample >= last_start, "产出应按时间单调");
            last_start = s.start_sample;
            assert!(s.samples.len() <= max_samples, "段过长: {}", s.samples.len());
        }
    }
}
