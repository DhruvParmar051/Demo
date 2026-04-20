"""Merge SFT+DPO LoRA adapters into the base model and convert to GGUF Q4_K_M.

Requires llama.cpp on PATH (``convert_hf_to_gguf.py`` and ``llama-quantize``).
If llama.cpp is unavailable, prints install instructions and exits 0.

Usage:
    python scripts/convert_to_gguf.py --out checkpoints/generator.Q4_K_M.gguf
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


def _which(*candidates: str) -> str | None:
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    return None


def merge_adapters(base_name: str, sft: Path, dpo: Path, merged_dir: Path) -> None:
    """Merge SFT and DPO LoRA adapters into a consolidated HF model."""
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"transformers/peft required for adapter merging: {exc}"
        )
    logging.info("Loading base model %s", base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    if sft.exists():
        logging.info("Applying SFT adapter %s", sft)
        model = PeftModel.from_pretrained(model, str(sft))
        model = model.merge_and_unload()
    if dpo.exists():
        logging.info("Applying DPO adapter %s", dpo)
        model = PeftModel.from_pretrained(model, str(dpo))
        model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    logging.info("Merged model at %s", merged_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="checkpoints/generator.Q4_K_M.gguf")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    convert = _which("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
    quantize = _which("llama-quantize", "quantize")
    if convert is None or quantize is None:
        print(
            "llama.cpp tools not found on PATH.\n"
            "  git clone https://github.com/ggerganov/llama.cpp.git\n"
            "  cd llama.cpp && make && pip install -r requirements.txt\n"
            "  export PATH=$PWD:$PATH\n"
            "Re-run this script once available."
        )
        return

    merged_dir = Path("checkpoints/generator_merged")
    if not args.skip_merge:
        merge_adapters(
            base_name=cfg.models.generator.name,
            sft=Path(cfg.checkpoints.generator_sft),
            dpo=Path(cfg.checkpoints.generator_dpo),
            merged_dir=merged_dir,
        )

    tmp_f16 = Path("checkpoints/generator.F16.gguf")
    logging.info("Running %s %s -> %s", convert, merged_dir, tmp_f16)
    subprocess.run(
        [sys.executable, convert, str(merged_dir),
         "--outfile", str(tmp_f16), "--outtype", "f16"],
        check=True,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Quantising %s -> %s (Q4_K_M)", tmp_f16, out)
    subprocess.run([quantize, str(tmp_f16), str(out), "Q4_K_M"], check=True)
    logging.info("Done: %s", out)


if __name__ == "__main__":
    main()
