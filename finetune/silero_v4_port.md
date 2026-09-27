# Silero VAD v4 → 纯 Rust 移植说明（E3 Phase 2/3 用）

## 已完成的验证（本地，位精确级）

* `silero_v4_reference.py`：silero_vad.onnx（sherpa 发行版，v4，0.64MB）的专用
  numpy 解释器。与 onnxruntime 对拍：**375 个中间层 × 3 组随机输入，最大相对误差
  3.2e-6**；真实日语音频（ja.wav 7.2s）**225 帧带状态序列对拍最大概率差 5.0e-6**。
  → numpy 语义即移植蓝本，Rust 侧是机械转写。
* `src/assets/silero_v4_weights.bin`（0.62MB，40 张量 f32 平面拼接，include_bytes! 嵌入）
  + `src/assets/silero_v4_layout.json`（名字/shape/偏移）。
* `tests/fixtures/silero_v4_golden.json`：3 个真实语音窗 + 1 个噪声窗
  （512 样本输入 → prob + new_h/new_c 前 8 值），Rust 单测容差建议 1e-5。

## 前向语义（每 512 样本 = 32ms 一帧；状态 h[2][64], c[2][64]）

1. **STFT**：x 两端 reflect-pad 96 → conv1d(basis 258×1×256, stride 64, 无 pad)
   → [258, 8帧]；实部=通道[0:129]，虚部=[129:258]；mag=sqrt(r²+i²) → [129, 8]
2. **log 幅度**：logmag = log(mag × 1048576 + 1)
3. **自适应归一化**：帧均值 m[t]=mean_bins(logmag)；按解释器里 adaptive_normalization
   段的 Slice/反转/Concat 构造 7 帧上下文 → conv(k=7) 平滑 → logmag − 平滑值
4. **特征拼接**：concat(mag, 归一化) → [258, 8]
5. **first_layer**：dw_conv(k5,pad2,g=258)→relu→pw(258→16) + proj(258→16) → relu
6. **encoder**（3 个 MobileNet 块 + 3 个 stride-2 1×1 降采样 8→4→2→1 帧）：
   * enc0: 1×1 s2 (16→16)；enc3 块: dw k5 + pw(16→32) + proj(16→32)→relu
   * enc4: 1×1 s2 (32→32)；enc7 块: dw + pw(32→32) + 残差→relu
   * enc8: 1×1 s2 (32→32)；enc11 块: dw + pw(32→64) + proj(32→64)→relu
   * enc12: 1×1 (64→64)→relu → [64, 1帧]
7. **LSTM ×2**（ONNX 门序 i,o,f,c；W(256,64)/R(256,64)/B(512)=[Wb|Rb]；
   ⚠ 投影是 x·Wᵀ——方阵不转置不报错但全错，numpy 版踩过）：
   LSTM1(x=64 维帧, h[0],c[0]) → y1；LSTM2(y1, h[1],c[1]) → y2
8. **decoder**：relu(y2) → conv1×1(64→1) → sigmoid → prob
9. new_h=[h1',h2']，new_c=[c1',c2']

## 段切分状态机（对齐产品现行为，替代 sherpa VoiceActivityDetector）

产品参数（cli.rs）：`--vad-min-silence 0.5s`、`--vad-buffer-secs 60`（单段上限，
超长自动收敛）、threshold 0.5、min_speech 250ms、speech_pad 前后各补 ~100ms。
Rust 实现逐帧 prob → 语音段 [start,end]，语义与 sherpa silero-vad 保持一致
（进入：p≥thr 持续 min_speech；退出：p<thr 持续 min_silence；段间 gap<0.4s 合并；
>27s 硬拆——与 finetune/s2tt_pipeline.py 的 merge_segments 同款参数）。
