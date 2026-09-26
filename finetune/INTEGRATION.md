# S2TT 快速直出版：推理落地设计（双版本产品形态）

> 状态：设计已定，Gate 1（ONNX 探针）执行中。本文回答「微调模型的推理以什么方式
> 融进这个 Rust 项目最合适」，并给出可验证的落地路径。实测数据见 `README.md` §10–11。

## 1. 双版本定位

| | 精翻版（现状，默认） | 快速直出版（新增） |
|---|---|---|
| 管线 | ASR(日) → LLM 逐句质检 → 摘要/术语表 → 分批翻译 → 审校 | **一段**：S2TT 微调 ASR 直出中文 → 排版 |
| 模型 | sherpa-onnx ASR int8 + Qwen3-1.7B GGUF（~2.7GB） | 仅 S2TT ASR 包 + VAD（**省掉 ~1.8GB LLM**） |
| 实测速度 | CPU 上 LLM 多轮为瓶颈 | CPU RTF 0.69~0.91（4 vCPU fp32），3~5 分钟视频 job 11~13 分钟全流程 |
| 质量 | 基线（术语表+审校兜底） | 语言定向 100% 可靠（假名 0%）、密度/排版达标；**代价**：无全局术语表时专名漂移（特撮→特约、馬→乌龟），误译无闸可拦 |
| 静音/噪音段 | LLM 质检剔除 | `language None`→空输出丢弃（微调专门保住+双模式验证，比基座更稳） |

两版共用同一个 exe：快速版 = S2TT 模型目录 + hotwords 任务开关 + 跳过 LLM 段。
适合场景：生肉速览、批量粗翻、低配机器；精翻版保留给发布级字幕。

## 2. 推理实现路线对比（为什么选「权重补丁」）

| 路线 | 做法 | 判决 |
|---|---|---|
| A. PyTorch sidecar | Rust 起 Python 子进程跑 transformers+LoRA | **否**（产品形态）：要求用户装 Python 运行时、常驻 ~2.4GB、拖拽即用的单 exe 不再自足。保留作 CI 评测工具（`s2tt_pipeline.py` 已承担） |
| B1. 全量重导出 ONNX | HF 合并权重 → 自研导出器 → int8 量化 | **否**：无公开导出脚本可抄（README 已知坑），工作量与不确定性全场最高 |
| B2. **手术式补丁官方 int8 ONNX** | 把 ΔW=B·A·(α/r) 写回官方包里被 LoRA 命中的权重张量 | **推荐**：产物是普通 sherpa-onnx 模型目录，运行时与 Rust 加载侧零改动 |

**Gate 1 首跑（run 36250220762）在真实官方包上实测出三个修正设计假设的事实**：

1. **LoRA 不止命中 LM 解码器**：peft 按模块名匹配 target_modules，音频编码器
   18 层的 q/k/v_proj 也被挂了 LoRA——适配器共 **250 个模块 = LM 196（28 层×7 类）
   + 音频塔 54（18 层×3 类）**。补丁须同时覆盖 `decoder.int8.onnx` 与
   `encoder.int8.onnx`（「音频塔不用动」不成立；`conv_frontend.onnx`/tokenizer 仍原样复用）；
2. **权重名被导出器匿名化**：197 个量化 MatMul 权重全叫 `onnx::MatMul_N_quantized/
   _scale/_zero_point`（197 = LM 196 + 1 个额外量化 MatMul），按投影名匹配全部落空。
   → 探针 v2 改为**按值匹配**：反量化后与 HF 基座 safetensors 逐元素比对
   （形状预过滤 + 转置摆位自动试探 + 唯一性 margin + 误差上限拒配），机制已在
   合成夹具上全路径单测（匿名量化/fp32-Gemm/int8 对称无 zp/跨文件音频塔/诱饵层
   唯一性/负路径）；
3. **官方 0.6B 包全部投影都是 per-channel uint8**：社区「Q/K/V/O 保 FP32」的说法
   不适用于 k2-fsa 官方包（那是 1.7B 社区导出的规则）——补丁器必须处理
   「反量化→加 ΔW→重量化」，削顶率由探针实测决定沿用原 scale 还是重算。

B2 的产物形态：

```
models/sherpa-onnx-qwen3-asr-0.6B-s2tt-int8/
├── conv_frontend.onnx      # 原样复制
├── encoder.int8.onnx       # ← 补丁（音频塔 q/k/v × 18 层）
├── decoder.int8.onnx       # ← 补丁（LM 7 类投影 × 28 层）
└── tokenizer/…             # 原样复制
```

## 3. B2 的前置事实与风险（Gate 1 = `inspect_onnx_lora.py`，v2 按值匹配）

补丁正确性押在四个事实上，探针 v2 逐项核查（机制已在合成夹具上全路径单测：
匿名量化/fp32-Gemm/int8 对称无 zp/uint8 非对称带 zp、axis 0/1、转置摆位、
跨文件音频塔、诱饵层唯一性、误差上限拒配、ΔW 独立复算全对）：

1. **张量地图**：decoder+encoder 里全部量化权重（DequantizeLinear/MatMulInteger/
   QLinearMatMul 的数据输入）与 fp32 大二维权重，连同 scale/zp 张量、量化 axis、
   消费节点位置；权重名已被匿名化（§2 事实 2），映射靠**按值匹配**：反量化后与
   HF 基座 safetensors 比对，形状预过滤+转置试探，要求最优匹配 rel_err < 0.05
   且与次优拉开 2 倍（唯一性），否则宁判未匹配也不错配；
2. **HF↔ONNX 同源性**：值匹配本身就逐张量给出 rel_err（量化噪声量级 ~1e-3）；
   250 个适配器模块必须**全部**映射到 ONNX 张量（含音频塔 54 个）才 GO；
3. **ΔW 幅度**：||ΔW||_F/||W||_F 逐模块报告（LoRA r=32 预期 ~1e-2 量级）；
4. **削顶率**：模拟「反量化→加 ΔW→按**原** scale/zp 重量化」，统计被 clip 的元素占比。
   <1% → 直接按原 scale 写回；≥1% → 补丁器按新 range 重算该通道 scale（仍可行，多一步）。

**残留风险**：若某个被 LoRA 命中的权重在 ONNX 里被导出器折叠/吸收进别的算子
（值匹配找不到同源张量），该模块无法补丁——探针会以 NO-GO 明示，届时降级到
「重训一个只挂 LM 且可映射的适配器」或路线 A。

### Gate 1 结果（run 36252216017，v2.1 探针，真实官方 0.6B 包）：**三项全 GO**

| 判据 | 结果 |
|---|---|
| 适配器映射 | **250/250**（decoder 196 个 uint8 + encoder 54 个 int8；308 个量化权重全部值匹配成功，未匹配 0） |
| HF↔ONNX 同源 | 适配器相关张量最大 rel_err **1.22e-2**（中位 9.1e-3，量化噪声量级） |
| 按原 scale 重量化削顶 | 最坏 **0.109%** < 1%（逐模块 3~10e-4，requant MSE ~5e-8）→ **补丁直接沿用原 scale/zp，无需重算** |

补充事实：全部权重按 MatMulInteger B 侧 **(K,N) 转置存储**（补丁写回时要转置 ΔW）；
LM 侧 uint8 非对称（带 zp）、音频塔 int8 对称；ΔW 相对幅度最大 0.0402。
唯一 flag：`onnx::MatMul_9785_quantized` 值匹配二义（best 与 second 均 8.13e-3）——
tied lm_head/embed_tokens 等值所致，不在 LoRA 范围内，无害。
**结论：B2 路线放行，进入 Gate 2（补丁器 + sherpa-onnx 推理验证）。**

## 4. 验证链（三道闸，全在 CI，不过闸不发布）

| 闸 | 内容 | 工具 | 状态 |
|---|---|---|---|---|
| Gate 1 | 上述前置事实核查，输出 go/no-go | `finetune.yml` mode=onnx-inspect（windows，复用产品模型缓存+HF 缓存+适配器 artifact，~10 分钟） | ✅ **三项全 GO**（run 36252216017，见 §3） |
| Gate 2 | 补丁器写出 s2tt 模型目录 → **sherpa-onnx 运行时**（pip，与产品 Cargo 依赖同版本 1.13.8）推理：直出中文 / 不带仍日语 / 静音→空；与原始包对照 | `finetune/patch_onnx_lora.py` + `verify_s2tt_onnx.py`（mode=onnx-patch） | ✅ **全过**（run 36253697406，见下） |
| Gate 3 | **产品级**：Rust exe + `--asr-model-dir …s2tt…` + `--asr-hotwords "translate to Chinese"` 跑与 `s2tt-e2e` 同一视频，字幕一致率达标。T4 的教训：int8 量化会改变模型行为，**必须**在最终运行时上复测，不能只信 PyTorch 侧 | master `ci.yml` 增加 T5 用例（或 s2tt-e2e 加 sherpa 模式） | 待做（对象为 1.7B，见 §7） |

### Gate 2 结果（0.6B，run 36253697406，windows 4 vCPU）

补丁器：**250/250 张量补丁成功**（decoder 196 uint8 + encoder 54 int8；未映射 0、
匹配误差最大 1.22e-2、最坏削顶 0.109%、数值自检坏元素 0、耗时 6.3 分钟）。
sherpa-onnx **1.13.8（与产品 Cargo 依赖同版本）** 三项行为验证（ja1.wav，12.5s）：

| | 补丁包 | 原始包（对照） |
|---|---|---|
| 带 hotwords "translate to Chinese" | **假名 0.0% / 汉字 87.5%**（直出中文） | 假名 51.8%（仍日语——T4 结论在产品运行时复现） |
| 不带 hotwords | 假名 56.0%（仍日语，任务开关成立） | 假名 ~56%（无差异） |
| 2s 纯静音 | **双模式全空** | — |

推理 6.7s / 12.5s 音频（含识别器构建，int8 4 线程）——比 PyTorch sidecar（RTF 0.91）
更快，路线 A 彻底出局。**B2 路线在 0.6B 上全链路打通**：LoRA → int8 ONNX 补丁 →
产品运行时直出中文，Rust 侧零改动（现有 `--asr-hotwords` 即任务开关）。

## 5. Rust 侧最小改动清单（Gate 2 过后动工）

1. `--fast` 标志（快速直出版）：
   * hotwords 默认注入 `translate to Chinese`（用户显式传值则尊重用户）；
   * 走 `--asr-only` 同款路径（不加载 LLM），但**复用 `srt::layout`** 做折行+长 cue
     拆分后输出最终 `.srt`（现在 `--asr-only` 只出 raw/verified，不排版）；
   * 丢弃策略与 `s2tt_pipeline.py` 对齐：空文本丢弃；`language None` 但带文本的段
     按「≥2s 且 ≥4 字保留」（实测：<1.2s 的这类输出是幻觉碎片，长段是真实语音）。
2. `download_models.ps1` 加 `-S2tt` 参数（拉 s2tt 模型包；`-SkipLlm` 已存在，组合即
   快速版最小安装）；
3. Release 打包（`package_dist.py`）：同一 exe 出两种 zip——精翻版（现状）与
   快速版（s2tt 模型包 + 无 GGUF），或把 s2tt 模型包作为独立 release asset。

## 6. 适配器产线（数据侧）

现状 120 对（real-mini，质量见 §11：可用但专名漂移）。放量路径：
`prepare_data.py --translator api`（需密钥）或扩充 `parallel_ja_zh.tsv`（免密钥）→
`gpu-train`/CPU 分批多轮 → 同一套 Gate 1–3 验证 → 补丁发布。补丁工具只吃
adapter artifact，与训练解耦，适配器可独立迭代。

## 7. 基座选型：快速直出版改用 1.7B（2026-09-26 决策）

**动机**：0.6B 直出的语言定向/排版/静音全达标（§11），但没有 LLM 闸兜底时，
翻译精度短板直接暴露给用户——名词级漂移（特撮→特约/特杀、馬→乌鸦/乌龟）与
1/20 严重误译（§10 质量评估）。快速版要靠更强的基座补偿：1.7B 是论文所称的
开源 ASR SOTA，且产品侧已有完整支持（`ci.yml` 的 `ASR_VARIANT=1.7B`、
`download_models.ps1 -AsrVariant 1.7B`、社区 int8 ONNX 包）。

**旧 0.6B LoRA 不可复用（形状硬不兼容）**：

| 模块 | 0.6B 适配器 (A/B 形状) | 1.7B 所需 |
|---|---|---|
| LM q_proj | (32,1024) / (2048,32) | (32,**2048**) / (2048,32) |
| LM k/v_proj | (32,1024) / (1024,32) | (32,**2048**) / (1024,32) |
| LM o_proj | (32,2048) / (1024,32) | (32,2048) / (**2048**,32) |
| LM gate/up_proj | (32,1024) / (3072,32) | (32,**2048**) / (**6144**,32) |
| LM down_proj | (32,3072) / (1024,32) | (32,**6144**) / (**2048**,32) |
| 音频塔 q/k/v | **18 层** × 896 维 | **24 层** × **1024** 维（连 key 都对不上） |

且 LoRA ΔW 绑定基座权重空间，形状巧合相同也无意义。**重新训练**（run 36254711659，
real-mini 同款配方：120 对对照表 + 20 静音样本 + LoRA r=32 α=64 + 4 epoch + lr 3e-4，
纯 CPU；数据零新增）。预计训练 ~1.5-2h（240 步 × ~20-26s/步，fp32 16GB runner）。

**对 ONNX 路线的影响**：1.7B int8 包是**社区导出**（`thieunv-asilla/sherpa-onnx-
qwen3-asr-1.7B-int8`），导出器与 k2-fsa 官方包不同——命名可能保留、量化摆位可能
是「MatMul/Gemm per-channel QUInt8 + Q/K/V/O 保 FP32」（README 已知坑）。探针与
补丁器按设计对两种世界都兼容（值匹配不依赖名字；fp32/uint8 非对称/int8 对称三种
张量形态都能处理），但**动补丁前必须先对 1.7B 包跑一遍 Gate 1**，账目对不上就停。

**版本矩阵（发布形态）**：

| 版本 | ASR | LLM | 适用 |
|---|---|---|---|
| 精翻版（现状） | 0.6B / 1.7B int8 | Qwen3-1.7B GGUF 四段 | 发布级字幕 |
| 快速直出版 | **1.7B s2tt 补丁包** + hotwords 开关 | 无 | 生肉速览/批量粗翻/低配机 |
| （轻量选项） | 0.6B s2tt 补丁包（Gate 2 已全链路验证） | 无 | 极致速度，质量降档 |
