"""Lightweight DPO training on the local preference set.

Applies DPO directly to the base generator (NO SFT adapter load).
Hardcoded conservative defaults for fast, low-VRAM runs on Kaggle T4 /
Colab free:

  * DoRA, rank=8
  * IPO loss (more stable than sigmoid on tiny datasets)
  * beta=0.1
  * 1 epoch
  * bf16 + gradient checkpointing
  * Filters preferences to the 2 grounding-critical rejection types:
      - hallucinated_citation
      - no_citation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data.preference_generator import DPO_TRAINING_TYPES
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


# Hardcoded lightweight defaults — override via cfg.training.dpo if present.
_DEFAULTS = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 5.0e-5,
    "num_epochs": 1,
    "batch_size": 1,
    "grad_accum": 8,
    "beta": 0.1,
    "loss_type": "ipo",
    "max_seq_length": 1536,
    "max_prompt_length": 768,
    "target_modules": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}


def _load_preferences(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cfg_get(cfg_section: Any, key: str, default: Any) -> Any:
    if cfg_section is None:
        return default
    if hasattr(cfg_section, key):
        val = getattr(cfg_section, key)
        return val if val is not None else default
    if isinstance(cfg_section, dict):
        return cfg_section.get(key, default)
    return default


def train(cfg: Any = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else get_config()
    set_seed(42)

    try:
        import torch  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "no_torch"}

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from peft import LoraConfig  # type: ignore
        from trl import DPOTrainer, DPOConfig  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        logger.error("Missing deps: %s", exc)
        return {"status": "skipped", "reason": "deps_missing"}

    # ------------------------------------------------------------------
    # Load + filter preferences
    # ------------------------------------------------------------------
    pref_path = Path(
        getattr(getattr(cfg.data, "synthetic", object()), "preferences_path",
                Path(cfg.resolve_path(cfg.data.synthetic_dir)) / "preferences.jsonl")
    )
    if not pref_path.exists():
        pref_path = Path(cfg.resolve_path(cfg.data.synthetic_dir)) / "preferences.jsonl"

    prefs = _load_preferences(pref_path)
    if not prefs:
        return {"status": "skipped", "reason": "no_preference_data"}

    filtered = [
        p for p in prefs
        if p.get("rejection_type") in DPO_TRAINING_TYPES
    ]
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
    # Hyperparameters (cfg overrides defaults)
    # ------------------------------------------------------------------
    tcfg = getattr(cfg.training, "dpo", None)
    lr = float(_cfg_get(tcfg, "learning_rate", _DEFAULTS["learning_rate"]))
    epochs = int(_cfg_get(tcfg, "num_epochs", _DEFAULTS["num_epochs"]))
    bsz = int(_cfg_get(tcfg, "batch_size", _DEFAULTS["batch_size"]))
    grad_accum = int(_cfg_get(tcfg, "gradient_accumulation_steps", _DEFAULTS["grad_accum"]))
    beta = float(_cfg_get(tcfg, "beta", _DEFAULTS["beta"]))
    loss_type = str(_cfg_get(tcfg, "loss_type", _DEFAULTS["loss_type"]))
    max_seq = int(_cfg_get(tcfg, "max_seq_length", _DEFAULTS["max_seq_length"]))
    max_prompt = int(_cfg_get(tcfg, "max_prompt_length", _DEFAULTS["max_prompt_length"]))
    rank = int(_cfg_get(getattr(getattr(cfg, "lora", object()), "generator_dpo", None), "rank", _DEFAULTS["lora_rank"]))
    lora_alpha = int(_cfg_get(getattr(getattr(cfg, "lora", object()), "generator_dpo", None), "alpha", _DEFAULTS["lora_alpha"]))

    # ------------------------------------------------------------------
    # Load base model (NO SFT adapter; skipped per refactor)
    # ------------------------------------------------------------------
    name = cfg.models.generator.name
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
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
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        logger.warning("CUDA not available; DPO will be very slow on CPU.")
        load_kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=float(_DEFAULTS["lora_dropout"]),
        target_modules=list(_DEFAULTS["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
    )

    output_dir = Path(getattr(cfg.checkpoints, "generator_dpo", "checkpoints/dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        loss_type=loss_type,
        max_length=max_seq,
        max_prompt_length=max_prompt,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=42,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=base,
        ref_model=None,  # TRL clones internally
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("Saved DPO adapter to %s", output_dir)
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "n_pairs": len(filtered),
        "loss_type": loss_type,
        "rank": rank,
        "beta": beta,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
