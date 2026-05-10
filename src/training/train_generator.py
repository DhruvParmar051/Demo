"""
LoRA + DoRA SFT of Qwen2.5-7B-Instruct — MPS only, fp16.

Loads the 7B model in fp16 (~14 GB unified memory). LoRA adapters are trained
with attention-only target modules (q/k/v/o) for maximum throughput on MPS.

Install deps (conda dl env):
    conda run -n dl pip install peft trl transformers datasets
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# MPS tuning — set before torch import
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


def _format_example(rec: dict[str, Any]) -> dict[str, str]:
    """Convert a QA record into (prompt, response) for SFT."""
    ctx_blocks = []

    # QA records store gold_chunk_ids; context text lives in citations list.
    for cit in rec.get("citations", []):
        cid = cit.get("chunk_id", cit.get("doc_id", "unknown"))
        text = cit.get("cited_text", "")
        span_start = cit.get("span_start", 0)
        span_end = cit.get("span_end", len(text))

        ctx_blocks.append(
            f"[{cid[:16]}:{span_start}-{span_end}] {text}"
        )

    ctx = "\n\n".join(ctx_blocks)

    system = (
        "You are AegisRAG, a grounded customer-support assistant. "
        "Answer using only the provided context. "
        "If the context is insufficient, say so explicitly."
    )

    prompt = (
        f"<|system|>\n{system}\n"
        f"<|context|>\n{ctx}\n"
        f"<|user|>\n{rec['query']}\n"
        f"<|assistant|>\n"
    )

    # answer_with_citations is the only populated field in this dataset.
    # Strip inline citation markers [chunk_id:start-end] to get clean prose.
    response = (
        rec.get("answer_with_citations")
        or rec.get("answer_without_citations")
        or rec.get("answer")
        or ""
    )

    response = re.sub(r"\[[^\]:]+:\d+-\d+\]", "", response).strip()

    return {
        "prompt": prompt,
        "response": response,
    }


def _assert_mps() -> None:
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS not available. This script requires Apple Silicon with MPS.\n"
            "Run: conda run -n dl python -m src.training.train_generator"
        )
    logger.info("MPS confirmed — Apple Silicon QLoRA training.")


def train(cfg: Any = None) -> dict[str, Any]:
    """Run fp16 LoRA SFT on the generator — MPS only."""
    cfg = cfg if cfg is not None else get_config()

    set_seed(42)

    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    _assert_mps()

    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset

        # bfloat16 prevents logit overflow — plain Trainer is sufficient.
        MpsSafeTrainer = Trainer
    except ImportError as exc:
        logger.error(
            "Missing deps: %s\n"
            "Fix: conda run -n dl pip install peft trl transformers datasets",
            exc,
        )
        return {"status": "skipped", "reason": "deps_missing"}

    # ── Load QA data ────────────────────────────────────────────────────────

    qa_path = Path(cfg.data.synthetic.qa_path)

    if not qa_path.exists():
        return {"status": "skipped", "reason": "no_qa_data"}

    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]

    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}

    ds = Dataset.from_list([_format_example(r) for r in qa])

    # ── Tokenizer ───────────────────────────────────────────────────────────

    name = cfg.models.generator.name

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model: fp16 on MPS ───────────────────────────────────────────────────
    # SDPA crashes on Qwen2.5 GQA (28q/4kv heads) on MPS — eager is stable.

    model_config = AutoConfig.from_pretrained(name)
    model_config._attn_implementation = "eager"

    # bfloat16 on MPS (PyTorch >= 2.4): same memory as fp16 (~14 GB) but with
    # float32's exponent range — prevents logit overflow that causes NaN loss.
    _dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=model_config,
        torch_dtype=_dtype,
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pretraining_tp = 1
    model.config.use_cache = False
    model.enable_input_require_grads()

    logger.info("Loaded 7B model in %s on MPS.", _dtype)

    # ── LoRA / DoRA ──────────────────────────────────────────────────────────
    # r=4 keeps trainable param count low → faster optimizer step on MPS.
    # Attention-only targets (q/k/v/o) are the highest-ROI modules for RAG.
    # MLP projections (gate/up/down) add compute with diminishing returns here.

    lcfg_yaml = cfg.lora

    peft_cfg = LoraConfig(
        r=4,
        lora_alpha=int(lcfg_yaml.alpha),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=float(lcfg_yaml.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=bool(lcfg_yaml.use_dora),
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ── Tokenise dataset ─────────────────────────────────────────────────────
    # fp16 7B = ~14 GB; max_seq=512 is safe on 24 GB+ unified memory.

    tcfg = cfg.training.generator_sft
    max_seq = min(int(tcfg.max_seq_length), 512)

    # Reserve at least 64 tokens for the response; truncate prompt if needed.
    max_prompt_len = max_seq - 64

    def _tokenize(ex: dict[str, Any]) -> dict[str, Any]:
        prompt_ids = tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        resp_ids = tokenizer(
            ex["response"] + tokenizer.eos_token,
            truncation=True,
            max_length=max_seq - len(prompt_ids),
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + list(resp_ids)

        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         labels,
        }

    ds = ds.map(
        _tokenize,
        remove_columns=["prompt", "response"],
        num_proc=1,
        desc="Tokenising",
    )

    # ── Checkpoint resumption ────────────────────────────────────────────────

    output_dir = str(Path(cfg.checkpoints.generator_sft))
    last_checkpoint = None

    if Path(output_dir).exists():
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info("Resuming from checkpoint: %s", last_checkpoint)

    # ── Training args (MPS-optimised) ────────────────────────────────────────
    # - no bf16/fp16 flags: MPS handles fp16 natively via model dtype
    # - gradient_checkpointing=False: MPS recomputation is buggy on Qwen
    # - adamw_torch: paged/8bit optimisers need bitsandbytes CUDA
    # - grad_accum=8: effective batch=8 without OOM


    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=float(tcfg.learning_rate),
        warmup_ratio=float(tcfg.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=False,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        seed=42,
        report_to=[],
    )

    # Custom collator: pads input_ids + attention_mask while preserving the
    # -100 prompt-masking in labels set during tokenisation.
    # DataCollatorForLanguageModeling would overwrite labels with raw input_ids,
    # losing our prompt masking entirely.
    def collate_fn(batch: list[dict]) -> dict:
        import torch as _torch
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)

        return {
            "input_ids":      _torch.tensor(input_ids,      dtype=_torch.long),
            "attention_mask": _torch.tensor(attention_mask, dtype=_torch.long),
            "labels":         _torch.tensor(labels,         dtype=_torch.long),
        }

    # processing_class is the modern param name (transformers >= 4.45); fall back
    # to the legacy tokenizer= kwarg on older installations.
    import inspect as _inspect
    _trainer_sig = _inspect.signature(MpsSafeTrainer.__init__)
    _trainer_kwargs: dict = {
        "model": model,
        "args": args,
        "train_dataset": ds,
        "data_collator": collate_fn,
    }
    if "processing_class" in _trainer_sig.parameters:
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        _trainer_kwargs["tokenizer"] = tokenizer
    trainer = MpsSafeTrainer(**_trainer_kwargs)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    torch.mps.empty_cache()
    gc.collect()

    trainer.save_model(output_dir)
    logger.info("Saved SFT adapter to %s", cfg.checkpoints.generator_sft)

    return {"status": "ok", "output_dir": str(cfg.checkpoints.generator_sft)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()