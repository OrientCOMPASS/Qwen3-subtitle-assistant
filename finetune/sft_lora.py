#!/usr/bin/env python3
"""Qwen3-ASR 的 S2TT（日语语音 -> 中文文本）微调脚本。

基于官方 QwenLM/Qwen3-ASR 的 finetuning/qwen3_asr_sft.py 的数据管线（prefix/target 切分 +
labels 掩码），额外增加：
  * **LoRA**（peft）：把 1.7B 的显存需求压到 16GB 级，让消费级卡也能验证；
  * **CPU 兜底**：--device cpu 时强制 fp32 且关闭 bf16/fp16，用于 CI 跑通管线
    （CI 上的 CPU 运行只是"管线冒烟"，不代表翻译质量，质量验证必须有 GPU）；
  * --max-steps / --max-samples：小步快跑，便于先验证再放量。

训练目标格式（与推理端严格一致，否则训歪）：
  prefix = processor.apply_chat_template([{system: prompt}, {user: [audio]}],
                                         add_generation_prompt=True, tokenize=False)
  full   = prefix + "language Japanese<asr_text>" + 中文译文 + eos
  labels = full，其中 prefix 部分置 -100

用法（GPU，真实训练）：
  python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B --train data/train.jsonl \
      --out out/s2tt-lora --lora --epochs 1 --lr 2e-4 --batch-size 4 --grad-acc 4

用法（CPU，仅跑通管线）：
  python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B --train data/train.jsonl \
      --out out/smoke --device cpu --lora --max-samples 4 --max-steps 2 --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ASR_TEXT_TAG = "<asr_text>"


def log(msg: str) -> None:
    print(f"[sft] {msg}", flush=True)


def build_prefix_messages(prompt: str, audio_array: Any) -> List[Dict[str, Any]]:
    """与官方脚本、以及 qwen_asr.Qwen3ASRModel._build_messages 完全一致。"""
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prefix_msgs = build_prefix_messages(ex.get("prompt", ""), None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class Collator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        import soundfile as sf

        audios = []
        for p in audio_paths:
            arr, sr = sf.read(p, dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != self.sampling_rate:
                import librosa

                arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sampling_rate)
            audios.append(arr)

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


def make_cast_trainer_cls(model_dtype):
    """官方脚本的 CastFloatInputsTrainer：把浮点输入 cast 成模型 dtype。"""
    from transformers import Trainer

    class CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            import torch

            out = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and model_dtype is not None:
                    v = v.to(dtype=model_dtype)
                out[k] = v
            return super()._prepare_inputs(out)

    return CastFloatInputsTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--lora", action="store_true", help="用 peft LoRA（推荐先用它验证）")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-samples", type=int, default=0, help=">0 时只取前 N 条（冒烟用）")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-acc", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--log-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--grad-ckpt", action="store_true", help="梯度检查点，省显存但更慢")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        log("!! 指定了 --device cuda 但没有可用 GPU")
        return 2
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    log(f"device={'cuda' if use_cuda else 'cpu'} dtype={dtype} lora={args.lora}")

    t0 = time.time()
    from qwen_asr import Qwen3ASRModel

    wrapper = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        **({"device_map": "cuda:0"} if use_cuda else {}),
    )
    model = wrapper.model
    processor = wrapper.processor
    log(f"模型加载完成，用时 {time.time() - t0:.1f}s；"
        f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            # Qwen3 文本解码器的标准模块名；不用 all-linear 以免命中音频塔里的非标准层
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    log(f"可训练参数：{trainable_before / 1e6:.1f}M -> "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = load_dataset("json", data_files=args.train, split="train")
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    ds = ds.map(make_preprocess_fn(processor), remove_columns=ds.column_names)
    log(f"训练样本 {len(ds)} 条")

    eval_ds = None
    if args.eval and Path(args.eval).is_file():
        eval_ds = load_dataset("json", data_files=args.eval, split="train")
        if args.max_samples and len(eval_ds) > args.max_samples:
            eval_ds = eval_ds.select(range(args.max_samples))
        eval_ds = eval_ds.map(make_preprocess_fn(processor), remove_columns=eval_ds.column_names)

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "hf"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_steps,
        save_strategy="no" if args.max_steps > 0 else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None and args.max_steps <= 0 else "no",
        eval_steps=args.save_steps,
        bf16=bool(dtype == torch.bfloat16),
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
    )

    TrainerCls = make_cast_trainer_cls(model.dtype if hasattr(model, "dtype") else dtype)
    trainer = TrainerCls(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=Collator(processor=processor),
    )

    t1 = time.time()
    result = trainer.train()
    train_secs = time.time() - t1
    log(f"训练结束，用时 {train_secs:.1f}s；loss={result.training_loss:.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(Path(args.out) / ("lora" if args.lora else "full")))
    processor.save_pretrained(str(Path(args.out) / ("lora" if args.lora else "full")))

    report = {
        "model": args.model,
        "lora": args.lora,
        "lora_r": args.lora_r if args.lora else None,
        "device": "cuda" if use_cuda else "cpu",
        "dtype": str(dtype),
        "train_samples": len(ds),
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_acc": args.grad_acc,
        "lr": args.lr,
        "train_seconds": round(train_secs, 1),
        "final_loss": float(result.training_loss),
        "trainable_params_m": round(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3),
        "note": ("CPU 冒烟运行：只证明数据管线/训练回路可跑通，"
                 "不代表翻译质量" if not use_cuda else "GPU 运行"),
        "log_history": result.metrics if isinstance(result.metrics, dict) else str(result.metrics),
    }
    Path(args.out, "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"报告与权重已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
