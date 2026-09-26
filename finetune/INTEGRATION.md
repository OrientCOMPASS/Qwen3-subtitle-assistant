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
| B2. **手术式补丁官方 decoder.int8.onnx** | 只把 ΔW=B·A·(α/r) 写回官方包里的 LM 解码器权重 | **推荐**：LoRA 只挂 7 类投影（q/k/v/o/gate/up/down），**音频塔、projector、tokenizer、encoder、conv_frontend 全部原样复用**；产物是普通 sherpa-onnx 模型目录，运行时零改动 |

B2 的产物形态：

```
models/sherpa-onnx-qwen3-asr-0.6B-s2tt-int8/
├── conv_frontend.onnx      # 原样复制
├── encoder.int8.onnx       # 原样复制
├── decoder.int8.onnx       # ← 唯一被补丁的文件
└── tokenizer/…             # 原样复制
```

## 3. B2 的前置事实与风险（Gate 1 = `inspect_onnx_lora.py`）

补丁正确性押在四个事实上，探针逐项核查（合成夹具已验证探针机制本身：
fp32/uint8、转置/非转置、axis 0/1、ΔW 独立复算全对）：

1. **张量地图**：decoder 里 7 类投影是否都能按名字找到；哪些 fp32（社区经验：
   Q/K/V/O 常保 fp32 防 repetition collapse——对我们反而是好事，fp32 张量加 ΔW 无损）、
   哪些 per-channel QUInt8（gate/up/down），scale/zp 张量与量化 axis 各是什么；
2. **HF↔ONNX 同源性**：抽样层反量化后与 HF 基座 safetensors 逐元素对比（rel < 2e-2）。
   不同源（导出器另做过折叠/吸收）则补丁映射不成立；
3. **ΔW 幅度**：||ΔW||_F/||W||_F 逐模块报告（LoRA r=32 预期 ~1e-2 量级）；
4. **削顶率**：模拟「反量化→加 ΔW→按**原** scale/zp 重量化」，统计被 clip 的元素占比。
   <1% → 直接按原 scale 写回；≥1% → 补丁器按新 range 重算该通道 scale（仍可行，多一步）。

**兜底**：若导出器没保留权重名（`onnx::MatMul_12345` 式命名），探针会打印全部
initializer 命名模式聚类，转「按形状+值匹配 HF 权重」做图手术（复杂度上升但不判死）。

## 4. 验证链（三道闸，全在 CI，不过闸不发布）

| 闸 | 内容 | 工具 |
|---|---|---|
| Gate 1 | 上述前置事实核查，输出 go/no-go | `finetune.yml` mode=onnx-inspect（windows，复用产品模型缓存+HF 缓存+适配器 artifact，~10 分钟） |
| Gate 2 | 补丁器写出 s2tt 模型目录 → **sherpa-onnx 运行时**（pip）推理：FLEURS eval 带 hotwords 直出中文（假名≤5%）/ 不带仍日语 / 静音→空；与 PyTorch sidecar 输出交叉一致 | 新增 `finetune/patch_onnx_lora.py` + verify job |
| Gate 3 | **产品级**：Rust exe + `--asr-model-dir …s2tt…` + `--asr-hotwords "translate to Chinese"` 跑与 `s2tt-e2e` 同一视频，字幕一致率达标。T4 的教训：int8 量化会改变模型行为，**必须**在最终运行时上复测，不能只信 PyTorch 侧 | master `ci.yml` 增加 T5 用例（或 s2tt-e2e 加 sherpa 模式） |

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
