"""
AegisRAG - DPO Training — MPS only, fp16 LoRA/DoRA.

Loads 7B in fp16 (~14 GB unified memory). Attention-only LoRA targets and
precomputed ref logprobs keep per-step time low on Apple Silicon.

Install deps (conda dl env):
    conda run -n dl pip install peft trl transformers datasets
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# MPS tuning — set before torch import
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from src.data.preference_generator import DPO_TRAINING_TYPES
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_preferences(path: Path) -> list[dict[str, Any]]:
    """
    Load JSONL preference pairs from path.

    Returns:
        List of preference dictionaries.
    """
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def _cfg_get(section: Any, key: str, default: Any) -> Any:
    """
    Safely read a key from a pydantic object, dict, or None.
    """
    if section is None:
        return default

    if hasattr(section, key):
        val = getattr(section, key)
        return val if val is not None else default

    if isinstance(section, dict):
        return section.get(key, default)

    return default


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """
    Unified attribute access for dicts and objects.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: Any = None) -> dict[str, Any]:
    """Run fp16 LoRA DPO training — MPS only."""
    cfg = cfg if cfg is not None else get_config()

    set_seed(42)

    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS not available. This script requires Apple Silicon with MPS.\n"
            "Run: conda run -n dl python -m src.training.train_dpo"
        )

    logger.info("MPS confirmed — Apple Silicon QLoRA DPO training.")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError as exc:
        logger.error(
            "Missing deps: %s\n"
            "Fix: conda run -n dl pip install peft trl transformers datasets",
            exc,
        )
        return {"status": "skipped", "reason": "deps_missing"}

    # ------------------------------------------------------------------
    # Resolve preference data path
    # ------------------------------------------------------------------

    data_cfg = _attr(cfg, "data")
    synth_cfg = _attr(data_cfg, "synthetic") if data_cfg else None
    pref_path_str = _attr(synth_cfg, "preferences_path") if synth_cfg else None

    if pref_path_str:
        pref_path = Path(pref_path_str)
    else:
        synth_dir = _attr(data_cfg, "synthetic_dir") if data_cfg else "data/synthetic"
        resolve_fn = _attr(cfg, "resolve_path")
        base_dir = resolve_fn(synth_dir) if resolve_fn else synth_dir
        pref_path = Path(base_dir) / "preferences.jsonl"

    # ------------------------------------------------------------------
    # Load + filter preferences
    # ------------------------------------------------------------------

    prefs = _load_preferences(pref_path)

    if not prefs:
        logger.warning("No preference data found at %s", pref_path)
        return {"status": "skipped", "reason": "no_preference_data"}

    filtered = [p for p in prefs if p.get("rejection_type") in DPO_TRAINING_TYPES]

    logger.info(
        "DPO: %d/%d preferences kept (types=%s)",
        len(filtered), len(prefs), DPO_TRAINING_TYPES,
    )

    if not filtered:
        return {"status": "skipped", "reason": "no_matching_rejection_types"}

    # ------------------------------------------------------------------
    # Tokeniser + dataset
    # ------------------------------------------------------------------

    model_name = _attr(_attr(cfg, "models"), "generator", {})
    if isinstance(model_name, dict):
        model_name = model_name.get("name", "Qwen/Qwen2.5-7B-Instruct")
    elif hasattr(model_name, "name"):
        model_name = model_name.name
    else:
        model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": p["query"]}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in filtered
    ])

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    training_cfg = _attr(cfg, "training")
    tcfg = _attr(training_cfg, "dpo") if training_cfg else None

    lr         = float(_cfg_get(tcfg, "learning_rate", 5.0e-6))
    beta       = float(_cfg_get(tcfg, "beta", 0.1))
    max_seq    = int(_cfg_get(tcfg, "max_seq_length", 1536))
    max_prompt = int(_cfg_get(tcfg, "max_prompt_length", 768))
    grad_accum = int(_cfg_get(tcfg, "gradient_accumulation_steps", 8))
    epochs     = int(_cfg_get(tcfg, "num_epochs", 1))
    bsz        = int(_cfg_get(tcfg, "batch_size", 1))
    loss_type  = str(_cfg_get(tcfg, "loss_type", "ipo"))

    lora_root = _attr(cfg, "lora")
    glcfg = _attr(lora_root, "generator_dpo") if lora_root else None
    rank       = int(_cfg_get(glcfg, "rank", 8))
    lora_alpha = int(_cfg_get(glcfg, "alpha", 16))

    # ------------------------------------------------------------------
    # Model: fp16 on MPS — native, full backward support, no dequant overhead.
    # Eager attention required — Qwen2.5 SDPA crashes on MPS with GQA heads.
    # ------------------------------------------------------------------

    model_config = AutoConfig.from_pretrained(model_name)
    model_config._attn_implementation = "eager"

    _dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=_dtype,
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    base.config.pretraining_tp = 1
    base.config.use_cache = False

    # Load SFT adapter if available — DPO should start from the SFT checkpoint,
    # not the raw base model, so the reference policy is already aligned.
    sft_dir = Path(_attr(_attr(cfg, "checkpoints"), "generator_sft", "checkpoints/sft"))
    if sft_dir.exists():
        logger.info("Loading SFT adapter from %s for DPO init.", sft_dir)
        base = PeftModel.from_pretrained(base, str(sft_dir), is_trainable=True)
        base = base.merge_and_unload().to(_dtype)  # cast to uniform dtype after merge
        logger.info("SFT adapter merged into base model.")
    else:
        logger.warning("SFT checkpoint not found at %s — DPO starting from base model.", sft_dir)

    base.enable_input_require_grads()
    logger.info("Loaded 7B model in %s on MPS.", _dtype)

    # ------------------------------------------------------------------
    # LoRA / DoRA
    # r=4 attention-only: fastest per-step, enough capacity for DPO alignment.
    # ------------------------------------------------------------------

    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )

    # ------------------------------------------------------------------
    # Output + resume
    # ------------------------------------------------------------------

    checkpoints_cfg = _attr(cfg, "checkpoints")
    out_dir_path = _attr(checkpoints_cfg, "generator_dpo", "checkpoints/dpo")
    output_dir = Path(out_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    resume_from_checkpoint = str(existing[-1]) if existing else None

    if resume_from_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)

    # ------------------------------------------------------------------
    # DPO config (MPS-optimised)
    # - bf16/fp16=False: MPS handles fp16 via model dtype, not trainer flags
    # - gradient_checkpointing=False: recomputation is unstable on MPS+Qwen
    # - adamw_torch: paged/8bit require bitsandbytes CUDA backend
    # - grad_accum=8: effective batch=8 without OOM
    # ------------------------------------------------------------------

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        loss_type=loss_type,
        max_grad_norm=0.05,
        max_length=max_seq,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        label_smoothing_factor=0.1,
        precompute_ref_log_probs=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    import inspect as _inspect
    _dpo_sig = _inspect.signature(DPOTrainer.__init__)
    _dpo_kwargs: dict = {
        "model": base,
        "ref_model": None,
        "args": args,
        "train_dataset": ds,
        "peft_config": peft_cfg,
    }
    if "processing_class" in _dpo_sig.parameters:
        _dpo_kwargs["processing_class"] = tokenizer
    else:
        _dpo_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**_dpo_kwargs)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(str(output_dir))
    logger.info("Saved DPO adapter to %s", output_dir)

    return {"status": "ok", "output_dir": str(output_dir)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.
    """
    logging.basicConfig(level=logging.INFO)

    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()