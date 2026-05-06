"""Merge adapters into the base model and convert to GGUF Q4_K_M.

Three variants are produced so each M-series pipeline can load a different
GGUF with the correct adapter weights baked in:

  base  – base Qwen2.5-7B-Instruct only      → aegis_base.gguf  (M1)
  sft   – base + SFT LoRA merged in          → aegis_sft.gguf   (M2)
  dpo   – base + SFT + DPO LoRA merged in    → aegis_dpo.gguf   (M3/M4/M5)

Requires llama.cpp on PATH (``convert_hf_to_gguf.py`` and ``llama-quantize``).

Usage (build all three at once):
    python scripts/convert_to_gguf.py

Build a single variant:
    python scripts/convert_to_gguf.py --variant sft
    python scripts/convert_to_gguf.py --variant dpo --out checkpoints/my_dpo.gguf
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config  # noqa: E402

log = logging.getLogger(__name__)


def _which(*candidates: str) -> str | None:
    # Also search the llama.cpp directory at the project root, which is where
    # the project keeps its local llama.cpp build.
    project_root = Path(__file__).resolve().parent.parent
    extra_dirs = [
        project_root / "llama.cpp",
        project_root / "llama.cpp" / "build" / "bin",
    ]
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
        for d in extra_dirs:
            candidate = d / c
            if candidate.exists():
                return str(candidate)
    return None


def _check_llama_cpp() -> tuple[str, str]:
    convert = _which("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
    quantize = _which("llama-quantize", "quantize")
    if convert is None or quantize is None:
        print(
            "llama.cpp tools not found on PATH or at <project>/llama.cpp/.\n"
            "  git clone https://github.com/ggerganov/llama.cpp.git\n"
            "  cd llama.cpp && make\n"
            "Re-run this script once available."
        )
        sys.exit(0)
    return convert, quantize


def merge_adapters(
    base_name: str,
    sft: Path | None,
    dpo: Path | None,
    merged_dir: Path,
) -> None:
    """Load base model, optionally apply SFT then DPO adapters, save merged HF weights."""
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"transformers/peft required for adapter merging: {exc}") from exc

    log.info("Loading base model %s", base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    if sft is not None and sft.exists():
        log.info("Applying SFT adapter %s", sft)
        model = PeftModel.from_pretrained(model, str(sft))
        model = model.merge_and_unload()
    elif sft is not None:
        log.warning("SFT adapter not found at %s — skipped", sft)

    if dpo is not None and dpo.exists():
        log.info("Applying DPO adapter %s", dpo)
        model = PeftModel.from_pretrained(model, str(dpo))
        model = model.merge_and_unload()
    elif dpo is not None:
        log.warning("DPO adapter not found at %s — skipped", dpo)

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    log.info("Saved merged model to %s", merged_dir)


def hf_to_gguf(
    convert_script: str,
    quantize_bin: str,
    merged_dir: Path,
    out_path: Path,
    quant: str = "Q4_K_M",
) -> None:
    tmp_f16 = out_path.with_name(out_path.stem + ".F16.gguf")
    log.info("Converting %s → %s (F16)", merged_dir, tmp_f16)
    subprocess.run(
        [sys.executable, convert_script, str(merged_dir),
         "--outfile", str(tmp_f16), "--outtype", "f16"],
        check=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Quantising %s → %s (%s)", tmp_f16, out_path, quant)
    subprocess.run([quantize_bin, str(tmp_f16), str(out_path), quant], check=True)
    tmp_f16.unlink(missing_ok=True)
    log.info("Done: %s", out_path)


# variant name → merge spec
_VARIANTS: dict[str, dict] = {
    "base": {
        "apply_sft": False,
        "apply_dpo": False,
        "merged_subdir": "generator_merged_base",
        "default_out": "checkpoints/aegis_base.gguf",
        "description": "base model only (M1)",
    },
    "sft": {
        "apply_sft": True,
        "apply_dpo": False,
        "merged_subdir": "generator_merged_sft",
        "default_out": "checkpoints/aegis_sft.gguf",
        "description": "base + SFT adapter (M2)",
    },
    "dpo": {
        "apply_sft": True,
        "apply_dpo": True,
        "merged_subdir": "generator_merged_dpo",
        "default_out": "checkpoints/aegis_dpo.gguf",
        "description": "base + SFT + DPO adapters (M3/M4/M5)",
    },
}


def build_variant(
    variant: str,
    out: str | None,
    skip_merge: bool,
    convert_script: str,
    quantize_bin: str,
    quant: str,
    cfg: object,
) -> None:
    spec = _VARIANTS[variant]
    out_path = Path(out) if out else Path(spec["default_out"])
    merged_dir = Path("checkpoints") / spec["merged_subdir"]

    sft_arg = Path(cfg.checkpoints.generator_sft) if spec["apply_sft"] else None
    dpo_arg = Path(cfg.checkpoints.generator_dpo) if spec["apply_dpo"] else None

    log.info("=== Building variant '%s' (%s) → %s ===", variant, spec["description"], out_path)

    if not skip_merge:
        merge_adapters(
            base_name=cfg.models.generator.name,
            sft=sft_arg,
            dpo=dpo_arg,
            merged_dir=merged_dir,
        )

    hf_to_gguf(convert_script, quantize_bin, merged_dir, out_path, quant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        choices=list(_VARIANTS),
        default=None,
        help="Which variant to build (default: build all three).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path (only valid with --variant).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the HF merge step and reuse the existing merged dir.",
    )
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        help="llama-quantize quantisation type (default: Q4_K_M).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = get_config()
    convert_script, quantize_bin = _check_llama_cpp()

    variants_to_build = [args.variant] if args.variant else list(_VARIANTS)
    for v in variants_to_build:
        build_variant(
            variant=v,
            out=args.out if args.variant else None,
            skip_merge=args.skip_merge,
            convert_script=convert_script,
            quantize_bin=quantize_bin,
            quant=args.quant,
            cfg=cfg,
        )

    if len(variants_to_build) > 1:
        print("\nAll done. GGUFs written to:")
        for v in variants_to_build:
            print(f"  {_VARIANTS[v]['default_out']}  ({_VARIANTS[v]['description']})")


if __name__ == "__main__":
    main()
