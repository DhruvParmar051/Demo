"""
AegisRAG - DPO Training

Applies DPO directly to the base generator without a prior SFT step.
Conservative defaults tuned for low-VRAM environments (Kaggle T4 / Colab
free tier / Apple Silicon CPU):

  - DoRA rank-8
  - Sigmoid loss for numerical stability
  - beta = 0.1, 1 epoch
  - bf16 disabled; fp32 on MPS to avoid NaN grad norms
  - Gradient checkpointing with use_reentrant=False
  - Filters preferences to two grounding-critical rejection types:
      hallucinated_citation, no_citation
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Disable W&B before any training import and suppress MPS memory cap warnings.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
warnings.filterwarnings("ignore")

from src.data.preference_generator import DPO_TRAINING_TYPES
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_preferences(path: Path) -> list[dict[str, Any]]:
    """Load JSONL preference pairs from *path*. Returns an empty list on miss."""
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
    """Safely read *key* from a pydantic model, dict, or None."""
    if section is None:
        return default
    if hasattr(section, key):
        val = getattr(section, key)
        return val if val is not None else default
    if isinstance(section, dict):
        return section.get(key, default)
    return default


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Unified attribute access for both dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: Any = None) -> dict[str, Any]:
    """Run DPO training and return a status dict."""
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from peft import LoraConfig  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore
        from datasets import Dataset  # type: ignore
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
        synth_dir = _attr(data_cfg, "synthetic_dir") if data_cfg else "data/synthetic"
        resolve_fn = _attr(cfg, "resolve_path")
        base_dir = resolve_fn(synth_dir) if resolve_fn else synth_dir
        pref_path = Path(base_dir) / "preferences.jsonl"

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

    ds = Dataset.from_list(
        [
            {"prompt": p["query"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in filtered
        ]
    )

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    training_cfg = _attr(cfg, "training")
    tcfg = _attr(training_cfg, "dpo") if training_cfg else None

    lr = 2.0e-6
    beta = 0.1
    max_seq = 256
    max_prompt = 128
    grad_accum = 16
    epochs = int(_cfg_get(tcfg, "num_epochs", 1))
    bsz = 1
    loss_type = "sigmoid"

    lora_root = _attr(cfg, "lora")
    glcfg = _attr(lora_root, "generator_dpo") if lora_root else None
    rank = int(_cfg_get(glcfg, "rank", 8))
    lora_alpha = int(_cfg_get(glcfg, "alpha", 16))

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------
    models_cfg = _attr(cfg, "models")
    gen_cfg = _attr(models_cfg, "generator") if models_cfg else None
    model_name = _attr(gen_cfg, "name", "Qwen/Qwen2.5-7B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
        torch_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        # float32 avoids NaN grad norms that occur with fp16 on Apple Silicon.
        logger.info("MPS detected — loading in float32 for numerical stability.")
        load_kwargs["device_map"] = "mps"
        torch_dtype = torch.float32
    else:
        load_kwargs["device_map"] = "cpu"
        torch_dtype = torch.float32

    load_kwargs["torch_dtype"] = torch_dtype

    base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    base.config.use_cache = False
    base.enable_input_require_grads()

    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )

    # ------------------------------------------------------------------
    # Output directory and checkpoint resume
    # ------------------------------------------------------------------
    checkpoints_cfg = _attr(cfg, "checkpoints")
    out_dir_path = _attr(checkpoints_cfg, "generator_dpo", "checkpoints/dpo")
    output_dir = Path(out_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from_checkpoint = None
    existing = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if existing:
        resume_from_checkpoint = str(existing[-1])
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)

    # ------------------------------------------------------------------
    # Train
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
        max_prompt_length=max_prompt,
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
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

    trainer = DPOTrainer(
        model=base,
        ref_model=None,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    logger.info("Saved DPO adapter to %s", output_dir)
    return {"status": "ok", "output_dir": str(output_dir)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
