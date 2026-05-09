"""
AegisRAG - DPO Training

Applies DPO directly to the base generator without a prior SFT step.
Conservative defaults tuned for low-VRAM environments (Kaggle T4 / Colab
free tier / Apple Silicon).

Features:
  - Qwen2.5-7B-Instruct support
  - LoRA / DoRA adapter training
  - Stable Apple Silicon (MPS) configuration
  - Eager attention to avoid MPS SDPA crashes
  - Safe tokenizer + DPO setup for TRL >= 0.9
  - Resume support
  - Preference filtering for grounding-critical failures

Recommended for:
  - Citation grounding
  - Tool-use alignment
  - RAG preference optimization
  - Hallucination reduction

Notes for Apple Silicon:
  - Gradient checkpointing disabled due to MPS instability
  - Eager attention required for Qwen stability
  - fp16 model loading used to fit 7B on unified memory
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Disable W&B before importing training libraries.
os.environ.setdefault("WANDB_DISABLED", "true")

# Reduce aggressive MPS memory reservation.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Allow unsupported ops to fallback safely.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Prefer Metal backend where possible.
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

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
    """
    Run DPO training.

    Returns:
        Status dictionary with output directory.
    """

    cfg = cfg if cfg is not None else get_config()

    set_seed(42)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from peft import LoraConfig

        from trl import DPOConfig, DPOTrainer

        from datasets import Dataset

    except ImportError as exc:
        logger.error("Missing training dependencies: %s", exc)
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
        synth_dir = (
            _attr(data_cfg, "synthetic_dir")
            if data_cfg
            else "data/synthetic"
        )

        resolve_fn = _attr(cfg, "resolve_path")

        base_dir = resolve_fn(synth_dir) if resolve_fn else synth_dir

        pref_path = Path(base_dir) / "preferences.jsonl"

    # ------------------------------------------------------------------
    # Load preferences
    # ------------------------------------------------------------------

    prefs = _load_preferences(pref_path)

    if not prefs:
        logger.warning("No preference data found at %s", pref_path)

        return {
            "status": "skipped",
            "reason": "no_preference_data",
        }

    filtered = [
        p for p in prefs
        if p.get("rejection_type") in DPO_TRAINING_TYPES
    ]

    logger.info(
        "DPO: %d/%d preferences kept (types=%s)",
        len(filtered),
        len(prefs),
        DPO_TRAINING_TYPES,
    )

    if not filtered:
        return {
            "status": "skipped",
            "reason": "no_matching_rejection_types",
        }

    # ------------------------------------------------------------------
    # Dataset formatting
    # ------------------------------------------------------------------

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_rows = []

    for p in filtered:

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["query"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        formatted_rows.append(
            {
                "prompt": prompt,
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )

    ds = Dataset.from_list(formatted_rows)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    training_cfg = _attr(cfg, "training")

    tcfg = _attr(training_cfg, "dpo") if training_cfg else None

    lr = float(_cfg_get(tcfg, "learning_rate", 5.0e-6))

    beta = float(_cfg_get(tcfg, "beta", 0.1))

    max_seq = int(_cfg_get(tcfg, "max_seq_length", 768))

    max_prompt = int(_cfg_get(tcfg, "max_prompt_length", 384))

    grad_accum = int(
        _cfg_get(tcfg, "gradient_accumulation_steps", 4)
    )

    epochs = int(_cfg_get(tcfg, "num_epochs", 1))

    bsz = int(_cfg_get(tcfg, "batch_size", 1))

    loss_type = str(_cfg_get(tcfg, "loss_type", "sigmoid"))

    # ------------------------------------------------------------------
    # LoRA / DoRA config
    # ------------------------------------------------------------------

    lora_root = _attr(cfg, "lora")

    glcfg = _attr(lora_root, "generator_dpo") if lora_root else None

    rank = int(_cfg_get(glcfg, "rank", 8))

    lora_alpha = int(_cfg_get(glcfg, "alpha", 16))

    # ------------------------------------------------------------------
    # Model loading config
    # ------------------------------------------------------------------

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():

        logger.info("CUDA detected.")

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        load_kwargs["device_map"] = "auto"

        torch_dtype = torch.float16

    elif torch.backends.mps.is_available():

        logger.info(
            "MPS detected — using eager attention fp16 config."
        )

        load_kwargs["device_map"] = {"": "mps"}

        torch_dtype = torch.float16

    else:

        logger.info("Using CPU backend.")

        load_kwargs["device_map"] = "cpu"

        torch_dtype = torch.float16

    load_kwargs["dtype"] = torch_dtype

    # ------------------------------------------------------------------
    # Force eager attention for MPS stability
    # ------------------------------------------------------------------

    config = AutoConfig.from_pretrained(model_name)

    # Critical for avoiding MPS fused-attention crashes.
    config._attn_implementation = "eager"
    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        **load_kwargs,
    )

    # Important for Qwen tensor partitioning stability.
    base.config.pretraining_tp = 1

    base.config.use_cache = False

    base.enable_input_require_grads()

    # ------------------------------------------------------------------
    # LoRA / DoRA adapter setup
    # ------------------------------------------------------------------

    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",

        # DoRA enabled per user preference.
        use_dora=True,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    checkpoints_cfg = _attr(cfg, "checkpoints")

    out_dir_path = _attr(
        checkpoints_cfg,
        "generator_dpo",
        "checkpoints/dpo",
    )

    output_dir = Path(out_dir_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume checkpoint discovery
    # ------------------------------------------------------------------

    resume_from_checkpoint = None

    existing = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )

    if existing:
        resume_from_checkpoint = str(existing[-1])

        logger.info(
            "Resuming from checkpoint: %s",
            resume_from_checkpoint,
        )

    # ------------------------------------------------------------------
    # DPO Training config
    # ------------------------------------------------------------------

    args = DPOConfig(
        output_dir=str(output_dir),

        num_train_epochs=epochs,

        per_device_train_batch_size=bsz,

        gradient_accumulation_steps=grad_accum,

        learning_rate=lr,

        beta=beta,

        loss_type=loss_type,

        max_grad_norm=0.1,

        max_length=max_seq,

        bf16=False,

        fp16=False,

        # Disabled due to MPS instability.
        gradient_checkpointing=False,

        logging_steps=1,

        save_strategy="steps",

        save_steps=10,

        save_total_limit=2,

        seed=42,

        report_to="none",

        remove_unused_columns=False,

        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        label_smoothing_factor=0.1,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    trainer = DPOTrainer(
        model=base,

        ref_model=None,

        args=args,

        train_dataset=ds,

        processing_class=tokenizer,

        peft_config=peft_cfg,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer.save_model(str(output_dir))

    logger.info("Saved DPO adapter to %s", output_dir)

    return {
        "status": "ok",
        "output_dir": str(output_dir),
    }


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