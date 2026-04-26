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

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


def _format_example(rec: dict[str, Any]) -> dict[str, str]:
    """Convert a QA record into (prompt, response) for SFT."""
    ctx_blocks = []
    for cid, text in zip(
        rec.get("gold_chunk_ids", []), rec.get("gold_chunk_texts", [])
    ):
        ctx_blocks.append(f"[{cid[:16]}:0-{len(text)}] {text}")
    ctx = "\n\n".join(ctx_blocks)
    system = (
        "You are AegisRAG, a grounded customer-support assistant. Cite every "
        "fact with [doc_id:start-end]. Escalate if unsupported."
    )
    prompt = (
        f"<|system|>\n{system}\n"
        f"<|context|>\n{ctx}\n"
        f"<|user|>\n{rec['query']}\n"
        f"<|assistant|>\n"
    )
    return {"prompt": prompt, "response": rec["answer_with_citations"]}


def train(cfg: Any = None) -> dict[str, Any]:
    """Run SFT on the generator."""
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA unavailable -- skipping generator SFT (cannot run QLoRA on CPU/MPS)."
        )
        return {"status": "skipped", "reason": "cpu_only"}

    try:
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        logger.error("Missing deps: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    from src.training.losses.citation_weighted_ce import CitationWeightedCELoss

    qa_path = Path(cfg.data.synthetic.qa_path)
    if not qa_path.exists():
        return {"status": "skipped", "reason": "no_qa_data"}
    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]
    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}

    records = [_format_example(r) for r in qa]
    ds = Dataset.from_list(records)

    name = cfg.models.generator.name
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lcfg_yaml = cfg.lora  # flat LoRAConfig — rank/alpha/target_modules/use_dora live here
    peft_cfg = LoraConfig(
        r=int(lcfg_yaml.rank),
        lora_alpha=int(lcfg_yaml.alpha),
        target_modules=list(lcfg_yaml.target_modules),
        lora_dropout=float(lcfg_yaml.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=bool(lcfg_yaml.use_dora),
    )
    model = get_peft_model(model, peft_cfg)

    tcfg = cfg.training.generator_sft
    max_seq = int(tcfg.max_seq_length)
    logger.info("Generator SFT max_seq_length=%d", max_seq)

    def _tokenize(ex: dict[str, Any]) -> dict[str, Any]:
        text = ex["prompt"] + ex["response"] + tokenizer.eos_token
        tokens = tokenizer(text, truncation=True, max_length=max_seq,
                            padding="max_length", return_tensors=None)
        prompt_len = len(
            tokenizer(ex["prompt"], truncation=True, max_length=max_seq)["input_ids"]
        )
        labels = list(tokens["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        tokens["labels"] = labels
        return tokens

    ds = ds.map(_tokenize, remove_columns=["prompt", "response"])

    output_dir = str(Path(cfg.checkpoints.generator_sft))

    # Detect any existing checkpoint so we can resume automatically.
    last_checkpoint = None
    if Path(output_dir).exists():
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info("Resuming SFT from checkpoint: %s", last_checkpoint)
        else:
            logger.info("No checkpoint found in %s; starting from scratch.", output_dir)

    # Auto-scale batch size and sequence length to available VRAM.
    # T4 (15 GB) needs batch=1; A100 (40/80 GB) can run batch=4.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if total_vram_gb < 20:
        # Small GPU (T4/V100-16G): batch=1, grad_accum=32, seq=1024
        micro_batch = 1
        grad_accum = 32
        max_seq = min(int(tcfg.max_seq_length), 1024)
        logger.info(
            "Small GPU (%.1f GB): batch=1, grad_accum=32, max_seq=%d", total_vram_gb, max_seq
        )
    else:
        # Large GPU (A100/H100): use config values as-is
        micro_batch = int(tcfg.batch_size)
        grad_accum = int(tcfg.gradient_accumulation_steps)
        max_seq = int(tcfg.max_seq_length)
        logger.info(
            "Large GPU (%.1f GB): batch=%d, grad_accum=%d, max_seq=%d",
            total_vram_gb, micro_batch, grad_accum, max_seq,
        )

    # bf16 is only efficient on Ampere+ (A100/H100); T4 uses fp16.
    use_bf16 = torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(tcfg.num_epochs),
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tcfg.learning_rate),
        warmup_ratio=float(tcfg.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=20,
        # Save every 50 steps so there's always a recent checkpoint to resume from.
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=42,
        report_to=[],
    )

    citation_loss = CitationWeightedCELoss()

    class _CitTrainer(Trainer):  # type: ignore[misc]
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = citation_loss(shift_logits, shift_labels, tokenizer)
            return (loss, outputs) if return_outputs else loss

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = _CitTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(output_dir)
    logger.info("Saved SFT adapter to %s", cfg.checkpoints.generator_sft)
    return {"status": "ok", "output_dir": str(cfg.checkpoints.generator_sft)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()