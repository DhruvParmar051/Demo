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
import os
from pathlib import Path
from typing import Any
 
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
 
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
 
 
def _gpu_profile() -> dict[str, Any]:
    """Return VRAM, bf16 support, FA2 availability, and torch version info."""
    import torch
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
 
    logger.info(
        "GPU: %s | VRAM: %.1f GB | bf16: %s | FA2: %s | compile: %s",
        props.name, vram_gb, bf16_ok, fa2_ok, compile_ok,
    )
    return {
        "vram_gb": vram_gb,
        "is_ampere_plus": is_ampere_plus,
        "bf16_ok": bf16_ok,
        "fa2_ok": fa2_ok,
        "compile_ok": compile_ok,
    }
 
 
def train(cfg: Any = None) -> dict[str, Any]:
    """Run SFT on the generator."""
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)
 
    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}
 
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable -- skipping generator SFT.")
        return {"status": "skipped", "reason": "cpu_only"}
 
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as exc:
        logger.error("Missing deps: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}
 
    from src.training.losses.citation_weighted_ce import CitationWeightedCELoss
 
    # ── Load QA data ────────────────────────────────────────────────────────
    qa_path = Path(cfg.data.synthetic.qa_path)
    if not qa_path.exists():
        return {"status": "skipped", "reason": "no_qa_data"}
    with qa_path.open("r", encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]
    if not qa:
        return {"status": "skipped", "reason": "no_qa_data"}
 
    records = [_format_example(r) for r in qa]
    ds = Dataset.from_list(records)
 
    # ── GPU profile (determines all tuning decisions below) ─────────────────
    gpu = _gpu_profile()
 
    # ── Tokenizer ───────────────────────────────────────────────────────────
    name = cfg.models.generator.name
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # ── Model loading ────────────────────────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if gpu["bf16_ok"] else torch.float16,
        bnb_4bit_use_double_quant=True,   # saves ~0.4 GB, free perf
    )
 
    model_kwargs: dict[str, Any] = dict(
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    if gpu["fa2_ok"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash-Attention-2 enabled")
    else:
        # SDPA is the next-best option (built into PyTorch >= 2.0)
        model_kwargs["attn_implementation"] = "sdpa"
        logger.info("Using SDPA attention (FA2 not available)")
 
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
 
    # ── LoRA / DoRA ──────────────────────────────────────────────────────────
    lcfg_yaml = cfg.lora
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
    model.print_trainable_parameters()
 
    if gpu["compile_ok"]:
        logger.info("Compiling model with torch.compile …")
        model = torch.compile(model)
 
    tcfg = cfg.training.generator_sft
 
    if gpu["vram_gb"] < 20:
        micro_batch = 1
        grad_accum  = 8          # was 32  ← biggest single speedup
        max_seq     = min(int(tcfg.max_seq_length), 512)   # was 1024
        logger.info(
            "T4 profile: batch=1, grad_accum=8, max_seq=%d (was 32/1024)", max_seq
        )
    else:
        
        micro_batch = int(tcfg.batch_size)
        grad_accum  = int(tcfg.gradient_accumulation_steps)
        max_seq     = int(tcfg.max_seq_length)
        logger.info(
            "Large GPU profile: batch=%d, grad_accum=%d, max_seq=%d",
            micro_batch, grad_accum, max_seq,
        )
 
    def _tokenize(ex: dict[str, Any]) -> dict[str, Any]:
        text = ex["prompt"] + ex["response"] + tokenizer.eos_token
        tokens = tokenizer(
            text, truncation=True, max_length=max_seq,
            padding="max_length", return_tensors=None,
        )
        prompt_len = len(
            tokenizer(ex["prompt"], truncation=True, max_length=max_seq)["input_ids"]
        )
        labels = list(tokens["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        tokens["labels"] = labels
        return tokens
 
    ds = ds.map(
        _tokenize,
        remove_columns=["prompt", "response"],
        num_proc=4,      
        desc="Tokenising",
    )
 
    # ── Checkpoint resumption ────────────────────────────────────────────────
    output_dir = str(Path(cfg.checkpoints.generator_sft))
    last_checkpoint = None
    if Path(output_dir).exists():
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info("Resuming from checkpoint: %s", last_checkpoint)
 
    try:
        import apex  # noqa: F401
        optim = "adamw_apex_fused"
        logger.info("Using apex fused AdamW")
    except ImportError:
        optim = "paged_adamw_8bit"
        logger.info("Using paged AdamW 8-bit")
 
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(tcfg.num_epochs),
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tcfg.learning_rate),
        warmup_ratio=float(tcfg.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=1,           
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        bf16=gpu["bf16_ok"],
        fp16=not gpu["bf16_ok"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        seed=42,
        report_to=[],
    )
 
    citation_loss = CitationWeightedCELoss()
 
    class _CitTrainer(Trainer):
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
        processing_class=tokenizer,  
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
 