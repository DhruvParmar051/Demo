"""
QLoRA + DoRA SFT of Qwen2.5-7B-Instruct with citation-weighted cross-entropy.

Uses TRL's SFTTrainer with a custom ``compute_loss`` override that routes
through :class:`CitationWeightedCELoss` so citation-marker tokens are
upweighted relative to plain tokens.

``max_seq_length`` is read from ``cfg.training.generator_sft.max_seq_length``
(the training config), not from ``cfg.models.generator.max_seq_length``
(the inference config).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    response = (
        rec.get("answer")
        or rec.get("answer_without_citations")
        or rec.get("answer_with_citations")
        or ""
    )

    # Strip citation markers from targets.
    response = re.sub(
        r"\[[^\]:]+:\d+-\d+\]",
        "",
        response,
    ).strip()

    return {
        "prompt": prompt,
        "response": response,
    }


def _gpu_profile() -> dict[str, Any]:
    """Return VRAM, bf16 support, FA2 availability, and torch version info."""
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)

        vram_gb = props.total_memory / 1024 ** 3

        is_ampere_plus = props.major >= 8

        bf16_ok = torch.cuda.is_bf16_supported()

        fa2_ok = False

        if is_ampere_plus:
            try:
                import flash_attn  # noqa: F401
                fa2_ok = True
            except ImportError:
                pass

        compile_ok = hasattr(torch, "compile") and is_ampere_plus

        device_name = props.name

    elif torch.backends.mps.is_available():
        vram_gb = 0.0
        is_ampere_plus = False
        bf16_ok = False
        fa2_ok = False
        compile_ok = False
        device_name = "Apple MPS"

    else:
        vram_gb = 0.0
        is_ampere_plus = False
        bf16_ok = False
        fa2_ok = False
        compile_ok = False
        device_name = "CPU"

    logger.info(
        "Device: %s | VRAM: %.1f GB | bf16: %s | FA2: %s | compile: %s",
        device_name,
        vram_gb,
        bf16_ok,
        fa2_ok,
        compile_ok,
    )

    return {
        "vram_gb": vram_gb,
        "is_ampere_plus": is_ampere_plus,
        "bf16_ok": bf16_ok,
        "fa2_ok": fa2_ok,
        "compile_ok": compile_ok,
        "device_name": device_name,
    }


def train(cfg: Any = None) -> dict[str, Any]:
    """Run SFT on the generator."""
    cfg = cfg if cfg is not None else get_config()

    set_seed(42)

    try:
        import torch
    except ImportError:
        return {
            "status": "skipped",
            "reason": "no_torch",
        }

    _cuda = torch.cuda.is_available()
    _mps = torch.backends.mps.is_available()

    if not _cuda and not _mps:
        logger.warning(
            "No GPU detected (CUDA/MPS). Generator SFT will run on CPU with "
            "reduced batch size and fp32. This is very slow — consider using "
            "a Kaggle T4 or Colab GPU instead."
        )

    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )

        from transformers.trainer_utils import get_last_checkpoint

        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        from datasets import Dataset

    except ImportError as exc:
        logger.error("Missing deps: %s", exc)

        return {
            "status": "skipped",
            "reason": "deps_missing",
        }

    # ── Load QA data ────────────────────────────────────────────────────────

    qa_path = Path(cfg.data.synthetic.qa_path)

    if not qa_path.exists():
        return {
            "status": "skipped",
            "reason": "no_qa_data",
        }

    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]

    if not qa:
        return {
            "status": "skipped",
            "reason": "no_qa_data",
        }

    records = [_format_example(r) for r in qa]

    ds = Dataset.from_list(records)

    # ── GPU profile (determines all tuning decisions below) ─────────────────

    gpu = _gpu_profile()

    # ── Tokenizer ───────────────────────────────────────────────────────────

    name = cfg.models.generator.name

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model loading ────────────────────────────────────────────────────────
    # BitsAndBytesConfig (4-bit NF4) is only supported on CUDA.
    # MPS and CPU fall back to fp32 full-precision loading.

    model_kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    config = None

    if torch.cuda.is_available():

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if gpu["bf16_ok"] else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs["quantization_config"] = bnb

        model_kwargs["device_map"] = "auto"

        model_kwargs["dtype"] = (
            torch.bfloat16 if gpu["bf16_ok"] else torch.float16
        )

    elif torch.backends.mps.is_available():

        # MPS: load in fp16; no 4-bit quant support on Apple Silicon

        model_kwargs["device_map"] = {"": "mps"}

        torch_dtype = torch.float16

        model_kwargs["dtype"] = torch_dtype

        config = AutoConfig.from_pretrained(name)

        # MPS SDPA kernels crash on GQA models (e.g. Qwen2.5 28q/4kv heads)
        # with "incompatible dimensions" LLVM abort. Eager is stable on MPS.

        config._attn_implementation = "eager"

        logger.info(
            "MPS detected — using eager attention fp16 config."
        )

    else:

        model_kwargs["device_map"] = "cpu"

        torch_dtype = torch.float16

        model_kwargs["dtype"] = torch_dtype

        logger.info("CPU fallback — training will be very slow")

    if gpu["fa2_ok"]:

        model_kwargs["attn_implementation"] = "flash_attention_2"

        logger.info("Flash-Attention-2 enabled")

    elif not torch.backends.mps.is_available():

        model_kwargs["attn_implementation"] = "sdpa"

        logger.info("Using SDPA attention (FA2 not available)")

    if config is not None:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            config=config,
            **model_kwargs,
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )

    if torch.backends.mps.is_available():
        model = model.to(torch.float16)

        model.config.pretraining_tp = 1

        model.config.use_cache = False

    if torch.cuda.is_available():

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            },
        )

    elif torch.backends.mps.is_available():

        model.enable_input_require_grads()

    else:

        model.enable_input_require_grads()

        if hasattr(model, "gradient_checkpointing_enable"):

            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                }
            )

    # ── LoRA / DoRA ──────────────────────────────────────────────────────────

    lcfg_yaml = cfg.lora

    peft_cfg = LoraConfig(
        r=4,
        lora_alpha=int(lcfg_yaml.alpha),
        target_modules=list(lcfg_yaml.target_modules),
        lora_dropout=float(lcfg_yaml.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=bool(lcfg_yaml.use_dora),
    )

    model = get_peft_model(model, peft_cfg)

    model.print_trainable_parameters()

    if gpu["compile_ok"]:

        logger.info("Compiling model with torch.compile …")

        model = torch.compile(model)

    tcfg = cfg.training.generator_sft

    _on_cuda = torch.cuda.is_available()
    _on_mps = torch.backends.mps.is_available()

    if not _on_cuda:

        # MPS / CPU: must use smallest possible batch

        micro_batch = 1

        grad_accum = 4

        max_seq = 128

        logger.info(
            "Non-CUDA profile (device=%s): batch=1, grad_accum=4, max_seq=%d",
            gpu["device_name"],
            max_seq,
        )

    elif gpu["vram_gb"] < 20:

        micro_batch = 1

        grad_accum = 8

        max_seq = min(int(tcfg.max_seq_length), 512)

        logger.info(
            "T4 profile: batch=1, grad_accum=8, max_seq=%d",
            max_seq,
        )

    else:

        micro_batch = int(tcfg.batch_size)

        grad_accum = int(tcfg.gradient_accumulation_steps)

        max_seq = int(tcfg.max_seq_length)

        logger.info(
            "Large GPU profile: batch=%d, grad_accum=%d, max_seq=%d",
            micro_batch,
            grad_accum,
            max_seq,
        )

    def _tokenize(ex: dict[str, Any]) -> dict[str, Any]:

        text = (
            ex["prompt"] +
            ex["response"] +
            tokenizer.eos_token
        )

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq,
            padding=False,
            return_tensors=None,
        )

        prompt_len = len(
            tokenizer(
                ex["prompt"],
                truncation=True,
                max_length=max_seq,
            )["input_ids"]
        )

        labels = list(tokens["input_ids"])

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        tokens["labels"] = labels

        return tokens

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
            logger.info(
                "Resuming from checkpoint: %s",
                last_checkpoint,
            )

    if _on_cuda:

        try:
            import apex  # noqa: F401

            optim = "adamw_apex_fused"

            logger.info("Using apex fused AdamW")

        except ImportError:

            optim = "paged_adamw_8bit"

            logger.info("Using paged AdamW 8-bit")

        _use_bf16 = gpu["bf16_ok"]

        _use_fp16 = not gpu["bf16_ok"]

        _pin_memory = True

        _num_workers = 2

    else:

        # paged_adamw_8bit requires bitsandbytes CUDA; fall back to adamw_torch

        optim = "adamw_torch"

        logger.info(
            "Using standard AdamW (no CUDA quantized optimizer)"
        )

        _use_bf16 = False

        _use_fp16 = False

        _pin_memory = False

        _num_workers = 0

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tcfg.learning_rate),
        warmup_ratio=float(tcfg.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=False,
        optim=optim,
        bf16=_use_bf16,
        fp16=_use_fp16,
        dataloader_num_workers=_num_workers,
        dataloader_pin_memory=_pin_memory,
        dataloader_prefetch_factor=(
            2 if _num_workers > 0 else None
        ),
        max_grad_norm=0.3,
        seed=42,
        report_to=[],
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()

    trainer.save_model(output_dir)

    logger.info(
        "Saved SFT adapter to %s",
        cfg.checkpoints.generator_sft,
    )

    return {
        "status": "ok",
        "output_dir": str(cfg.checkpoints.generator_sft),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()