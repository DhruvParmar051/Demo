"""
AegisRAG - Kaggle bootstrap helpers.

Call :func:`setup_kaggle` at the top of each training notebook to:
  1. Detect we're on Kaggle.
  2. Copy the AegisRAG source tree into ``/kaggle/working/AegisRAG``.
  3. Install missing Python dependencies (idempotent).
  4. Point the AegisRAG config at Kaggle input/output directories via
     environment variables (so we don't mutate the YAML checked into git).
  5. Create output directories.

Expected Kaggle Dataset layout (attach all three to each notebook):

    /kaggle/input/aegisrag-source/          <- repo snapshot (src/, config/, run.py, requirements.txt)
    /kaggle/input/aegisrag-raw-docs/        <- raw customer-support docs (PDF/DOCX/TXT/MD)
    /kaggle/input/aegisrag-synthetic/       <- generated training artefacts
                                              qa_pairs.jsonl
                                              preferences.jsonl
                                              confidence_labels.jsonl
                                              alpha_labels.jsonl
                                              decomp_labels.jsonl
                                              tool_route_labels.jsonl
    /kaggle/input/aegisrag-checkpoints/     <- prior-stage checkpoints
                                              (optional; required by DPO
                                              which needs the SFT adapter)

Outputs are written under ``/kaggle/working/checkpoints/<component>`` and
can be saved as a new Kaggle Dataset at notebook end.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Kaggle input dataset slugs. Override via env vars if your dataset names differ.
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DATASET = os.environ.get("AEGIS_KAGGLE_SOURCE", "aegisrag-source")
DEFAULT_RAW_DATASET = os.environ.get("AEGIS_KAGGLE_RAW", "aegisrag-raw-docs")
DEFAULT_SYNTH_DATASET = os.environ.get("AEGIS_KAGGLE_SYNTH", "aegisrag-synthetic")
DEFAULT_CKPT_DATASET = os.environ.get("AEGIS_KAGGLE_CKPT", "aegisrag-checkpoints")

REPO_ROOT = Path("/kaggle/working/AegisRAG")
WORKING_CKPT_DIR = Path("/kaggle/working/checkpoints")
HF_CACHE = Path("/kaggle/working/.cache/huggingface")


# ---------------------------------------------------------------------------
# Dependency sets
# ---------------------------------------------------------------------------
# Kaggle's default Docker already has torch, transformers, accelerate, peft,
# datasets, bitsandbytes, sentence-transformers, pandas, numpy, tqdm. Only
# install the extras we actually need.
LIGHT_DEPS = [
    "rank-bm25>=0.2.2",
    "chromadb>=0.5.0,<0.6.0",
    "loguru>=0.7.2",
    "pydantic>=2.7.0,<2.10.0",
    "pydantic-settings>=2.3.0",
    "pyyaml",
    "datasketch>=1.6.0",
    "tiktoken>=0.7.0",
]

TRAIN_DEPS = [
    "trl>=0.9.0,<0.14.0",
    "peft>=0.12.0,<0.15.0",
    "bert-score>=0.3.13,<0.4.0",
]

PARSE_DEPS = [
    "pdfplumber>=0.11.0",
    "python-docx>=1.1.0",
    "chardet>=5.2.0",
    "spacy>=3.7.0",
]


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def is_kaggle() -> bool:
    """Cheap heuristic: Kaggle sets these env vars."""
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or Path("/kaggle/input").exists()
    )


def copy_source(source_dataset: str = DEFAULT_SOURCE_DATASET) -> Path:
    """Copy the AegisRAG source tree from the Kaggle input dataset to /kaggle/working.

    We copy (not symlink) because some training libraries write sidecar files
    next to the source modules (e.g. ``__pycache__``), and the Kaggle input
    mount is read-only.
    """
    src = Path("/kaggle/input") / source_dataset
    if not src.exists():
        raise FileNotFoundError(
            f"Expected AegisRAG source at {src}. "
            f"Attach the '{source_dataset}' Kaggle Dataset to this notebook."
        )

    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    shutil.copytree(src, REPO_ROOT, dirs_exist_ok=False)

    # Ensure import path works.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def install_deps(extras: Optional[List[str]] = None, quiet: bool = True) -> None:
    """Install only the packages Kaggle's base image doesn't already have."""
    pkgs = list(LIGHT_DEPS)
    if extras:
        pkgs.extend(extras)
    flags = ["-q"] if quiet else []
    _run([sys.executable, "-m", "pip", "install", *flags, *pkgs])


def _env_overrides(
    raw_docs_dir: Optional[Path],
    synthetic_dir: Optional[Path],
    prior_ckpts: Optional[Path],
    component_output: Optional[Path],
) -> Dict[str, str]:
    """Build AEGIS_*** overrides that the pydantic-settings loader will pick up."""
    overrides: Dict[str, str] = {}

    if raw_docs_dir is not None and raw_docs_dir.exists():
        overrides["AEGIS_DATA__RAW_DOCS_DIR"] = str(raw_docs_dir)

    if synthetic_dir is not None and synthetic_dir.exists():
        overrides["AEGIS_DATA__SYNTHETIC_DIR"] = str(synthetic_dir)
        # Point each synthetic file path at the Kaggle dataset.
        overrides["AEGIS_DATA__SYNTHETIC__QA_PATH"] = str(synthetic_dir / "qa_pairs.jsonl")
        overrides["AEGIS_DATA__SYNTHETIC__PREFERENCES_PATH"] = str(synthetic_dir / "preferences.jsonl")
        overrides["AEGIS_DATA__SYNTHETIC__CONFIDENCE_LABELS_PATH"] = str(synthetic_dir / "confidence_labels.jsonl")
        overrides["AEGIS_DATA__SYNTHETIC__ALPHA_LABELS_PATH"] = str(synthetic_dir / "alpha_labels.jsonl")
        overrides["AEGIS_DATA__SYNTHETIC__DECOMP_LABELS_PATH"] = str(synthetic_dir / "decomp_labels.jsonl")
        overrides["AEGIS_DATA__SYNTHETIC__TOOL_ROUTE_LABELS_PATH"] = str(synthetic_dir / "tool_route_labels.jsonl")

    # Outputs: always under /kaggle/working so they persist into the session output.
    if component_output is not None:
        component_output.mkdir(parents=True, exist_ok=True)

    # Vector DB / BM25 index: put under /kaggle/working so retrievers can write.
    overrides["AEGIS_DATA__VECTOR_DB_PATH"] = "/kaggle/working/vectordb"
    overrides["AEGIS_DATA__BM25_INDEX_PATH"] = "/kaggle/working/bm25_index.pkl"
    overrides["AEGIS_DATA__AUDIT_DB_PATH"] = "/kaggle/working/audit.db"

    # HuggingFace cache under /kaggle/working so we don't blow the 20GB limit
    # on /tmp and so weights persist for subsequent notebook runs on the same session.
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    overrides["HF_HOME"] = str(HF_CACHE)
    overrides["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
    overrides["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")

    # Point to prior-stage checkpoints if they're attached.
    if prior_ckpts is not None and prior_ckpts.exists():
        sft = prior_ckpts / "generator_sft"
        if sft.exists():
            overrides["AEGIS_TRAINING__GENERATOR_SFT__OUTPUT_DIR"] = str(sft)
        dpo = prior_ckpts / "dpo"
        if dpo.exists():
            overrides["AEGIS_TRAINING__DPO__OUTPUT_DIR"] = str(dpo)
        retr = prior_ckpts / "retriever"
        if retr.exists():
            overrides["AEGIS_TRAINING__RETRIEVER__OUTPUT_DIR"] = str(retr)

    # Training-friendly defaults on Kaggle.
    # T4 does NOT support bf16; stick with fp16.
    overrides["AEGIS_TRAINING__FP16"] = "true"
    overrides["AEGIS_TRAINING__BF16"] = "false"
    # Kaggle CPU count is 4 — smaller is safer inside DataLoader workers.
    overrides["AEGIS_TRAINING__DATALOADER_NUM_WORKERS"] = "2"

    return overrides


def setup_kaggle(
    component: str,
    *,
    install_train_deps: bool = True,
    install_parse_deps: bool = False,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    raw_dataset: str = DEFAULT_RAW_DATASET,
    synth_dataset: str = DEFAULT_SYNTH_DATASET,
    ckpt_dataset: str = DEFAULT_CKPT_DATASET,
) -> Dict[str, str]:
    """One-shot setup for a training notebook.

    Parameters
    ----------
    component
        One of: ``retriever``, ``reranker``, ``generator_sft``, ``dpo``,
        ``confidence_head``, ``alpha_network``, ``decomposer``, ``data_gen``.
        Determines the default output directory.
    install_train_deps
        Whether to also ``pip install`` ``trl``/``peft``/``bert-score``.
    install_parse_deps
        Whether to also install PDF/DOCX parsers (needed only for data-gen).
    source_dataset, raw_dataset, synth_dataset, ckpt_dataset
        Kaggle Dataset slugs. See module docstring for expected layout.

    Returns
    -------
    dict
        The env-var overrides that were applied (mostly for debugging).
    """
    if not is_kaggle():
        print(
            "Not running on Kaggle (no /kaggle/input mount). "
            "setup_kaggle() is a no-op — you can still call train() directly.",
            file=sys.stderr,
        )
        return {}

    # ------------------------------------------------------------------
    # 1. Code
    # ------------------------------------------------------------------
    copy_source(source_dataset)

    # ------------------------------------------------------------------
    # 2. Dependencies
    # ------------------------------------------------------------------
    extras: List[str] = []
    if install_train_deps:
        extras.extend(TRAIN_DEPS)
    if install_parse_deps:
        extras.extend(PARSE_DEPS)
    install_deps(extras=extras)

    # ------------------------------------------------------------------
    # 3. Env var overrides pointing at Kaggle datasets + /kaggle/working
    # ------------------------------------------------------------------
    raw_docs = Path("/kaggle/input") / raw_dataset
    synth_dir = Path("/kaggle/input") / synth_dataset
    ckpt_dir = Path("/kaggle/input") / ckpt_dataset
    component_out = WORKING_CKPT_DIR / component

    overrides = _env_overrides(
        raw_docs_dir=raw_docs if raw_docs.exists() else None,
        synthetic_dir=synth_dir if synth_dir.exists() else None,
        prior_ckpts=ckpt_dir if ckpt_dir.exists() else None,
        component_output=component_out,
    )

    # Redirect this component's output to /kaggle/working/checkpoints/<component>.
    comp_env_key = {
        "retriever": "AEGIS_TRAINING__RETRIEVER__OUTPUT_DIR",
        "reranker": "AEGIS_TRAINING__RERANKER__OUTPUT_DIR",
        "generator_sft": "AEGIS_TRAINING__GENERATOR_SFT__OUTPUT_DIR",
        "dpo": "AEGIS_TRAINING__DPO__OUTPUT_DIR",
        "confidence_head": "AEGIS_TRAINING__CONFIDENCE_HEAD__OUTPUT_DIR",
        "alpha_network": "AEGIS_TRAINING__ALPHA_NETWORK__OUTPUT_DIR",
        "decomposer": "AEGIS_TRAINING__DECOMPOSER__OUTPUT_DIR",
    }.get(component)
    if comp_env_key is not None:
        overrides[comp_env_key] = str(component_out)

    for k, v in overrides.items():
        os.environ[k] = v

    # ------------------------------------------------------------------
    # 4. Sanity log
    # ------------------------------------------------------------------
    print("=" * 60)
    print("AegisRAG Kaggle bootstrap complete")
    print("=" * 60)
    print(f"Repo root      : {REPO_ROOT}")
    print(f"Raw docs       : {raw_docs}  (exists={raw_docs.exists()})")
    print(f"Synthetic data : {synth_dir}  (exists={synth_dir.exists()})")
    print(f"Prior ckpts    : {ckpt_dir}  (exists={ckpt_dir.exists()})")
    print(f"Output dir     : {component_out}")
    print(f"HF cache       : {HF_CACHE}")
    try:
        import torch  # noqa: WPS433
        print(f"CUDA available : {torch.cuda.is_available()}  "
              f"({torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    except Exception:
        pass
    print("=" * 60)

    # Change directory into the repo so `from src.training...` works.
    os.chdir(REPO_ROOT)

    return overrides
