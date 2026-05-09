"""Generate the eight Kaggle training notebooks for AegisRAG.

Run:
    python kaggle/helpers/gen_notebooks.py

Produces ``kaggle/notebooks/*.ipynb``. These notebooks are small, consistent
and intentionally thin — the bulk of the logic lives in ``src/training/*``.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
    }


def bootstrap_cell(component: str, install_train_deps: bool = True, install_parse_deps: bool = False) -> dict:
    return code(f"""
    # ---- Kaggle bootstrap ----------------------------------------------------
    # Attach these Kaggle Datasets to the notebook before running:
    #   aegisrag-source       (snapshot of the git repo: src/, config/, run.py, requirements.txt)
    #   aegisrag-raw-docs     (raw customer-support documents)            [only for data-gen]
    #   aegisrag-synthetic    (generated qa/preferences/labels .jsonl)    [all training stages]
    #   aegisrag-checkpoints  (prior-stage checkpoints)                   [required for DPO]
    import sys, os, importlib.util
    spec = importlib.util.spec_from_file_location(
        "aegisrag_bootstrap",
        "/kaggle/input/aegisrag-source/kaggle/helpers/bootstrap.py",
    )
    bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
    bootstrap.setup_kaggle(
        component={component!r},
        install_train_deps={install_train_deps},
        install_parse_deps={install_parse_deps},
    )
    """)


def save_working_cell(component: str, description: str) -> dict:
    return code(f"""
    # ---- Persist outputs -----------------------------------------------------
    # Everything under /kaggle/working is captured as the notebook's output and
    # can be saved as a new Kaggle Dataset via File > "Save Version"
    # with output = "Save & Run All (Commit)".
    import shutil, os
    from pathlib import Path
    src_dir = Path("/kaggle/working/checkpoints/{component}")
    print("{description}:", src_dir, "->", os.listdir(src_dir) if src_dir.exists() else "(missing)")
    """)


# ---------------------------------------------------------------------------
# 00 - Data generation
# ---------------------------------------------------------------------------
nb00 = nb([
    md("""
    # AegisRAG · Stage 00 — Synthetic Data Generation
    Generates all six training artefacts:
    * `qa_pairs.jsonl`, `preferences.jsonl`, `confidence_labels.jsonl`, `alpha_labels.jsonl`, `decomp_labels.jsonl`, `tool_route_labels.jsonl`
    Requires only the `aegisrag-source` + `aegisrag-raw-docs` Kaggle Datasets.
    Outputs land in `/kaggle/working/data/synthetic/*.jsonl` — save as a new Kaggle Dataset named **aegisrag-synthetic** and attach to later notebooks.
    """),
    bootstrap_cell("data_gen", install_train_deps=False, install_parse_deps=True),
    code("""
    # Route synthetic output to /kaggle/working and chunking input to /kaggle/input.
    import os
    os.environ["AEGIS_DATA__SYNTHETIC_DIR"] = "/kaggle/working/data/synthetic"
    os.environ["AEGIS_DATA__SYNTHETIC__QA_PATH"] = "/kaggle/working/data/synthetic/qa_pairs.jsonl"
    os.environ["AEGIS_DATA__SYNTHETIC__PREFERENCES_PATH"] = "/kaggle/working/data/synthetic/preferences.jsonl"
    os.environ["AEGIS_DATA__SYNTHETIC__CONFIDENCE_LABELS_PATH"] = "/kaggle/working/data/synthetic/confidence_labels.jsonl"
    os.environ["AEGIS_DATA__SYNTHETIC__ALPHA_LABELS_PATH"] = "/kaggle/working/data/synthetic/alpha_labels.jsonl"
    os.environ["AEGIS_DATA__SYNTHETIC__DECOMP_LABELS_PATH"] = "/kaggle/working/data/synthetic/decomp_labels.jsonl"
    os.environ["AEGIS_DATA__SYNTHETIC__TOOL_ROUTE_LABELS_PATH"] = "/kaggle/working/data/synthetic/tool_route_labels.jsonl"

    # Reload config to pick up env overrides.
    from src.utils.config import reset_config, get_config
    reset_config(); cfg = get_config()
    print("synthetic_dir:", cfg.data.synthetic_dir)
    """),
    code("""
    # 1) Ingest raw docs (chunking + BM25/Chroma indices in /kaggle/working).
    from src.data.ingestion import run_ingestion
    stats = run_ingestion(source_dir=cfg.data.raw_docs_dir)
    stats
    """),
    code("""
    # 2) Generate all six synthetic artefacts.
    from scripts.generate_data import main as gen_main
    import argparse
    gen_main(argparse.Namespace(type="all", output_dir=cfg.data.synthetic_dir, limit=None))
    """),
    save_working_cell("../data/synthetic", "Synthetic data"),
])
(OUT / "00_generate_data.ipynb").write_text(json.dumps(nb00, indent=1))


# ---------------------------------------------------------------------------
# 01 - Retriever (BGE-m3 MNRL fine-tune)
# ---------------------------------------------------------------------------
nb01 = nb([
    md("""
    # AegisRAG · Stage 01 — Retriever Fine-tune
    Contrastive fine-tune of `BAAI/bge-m3` using Multiple Negatives Ranking Loss on `qa_pairs.jsonl`.
    ~20 min on a single T4 at batch-size 16, or ~10 min on P100.
    Attach: `aegisrag-source`, `aegisrag-synthetic`.
    Output: `/kaggle/working/checkpoints/retriever`.
    """),
    bootstrap_cell("retriever"),
    code("""
    from src.training.train_retriever import train
    result = train()
    result
    """),
    save_working_cell("retriever", "Retriever checkpoint"),
])
(OUT / "01_train_retriever.ipynb").write_text(json.dumps(nb01, indent=1))


# ---------------------------------------------------------------------------
# 02 - Reranker (Jina ColBERT v2)
# ---------------------------------------------------------------------------
nb02 = nb([
    md("""
    # AegisRAG · Stage 02 — Reranker Fine-tune
    Cross-encoder fine-tune of `BAAI/bge-reranker-base` on (query, passage, label) pairs constructed from `qa_pairs.jsonl` at `pos_neg_ratio=2.0`.
    ~30 min on T4.
    Attach: `aegisrag-source`, `aegisrag-synthetic`.
    """),
    bootstrap_cell("reranker"),
    code("""
    from src.training.train_reranker import train
    result = train()
    result
    """),
    save_working_cell("reranker", "Reranker checkpoint"),
])
(OUT / "02_train_reranker.ipynb").write_text(json.dumps(nb02, indent=1))


# ---------------------------------------------------------------------------
# 03 - Generator SFT (Qwen2.5-7B QLoRA+DoRA)
# ---------------------------------------------------------------------------
nb03 = nb([
    md("""
    # AegisRAG · Stage 03 — Generator SFT (QLoRA + DoRA)
    Supervised fine-tune of `Qwen/Qwen2.5-7B-Instruct` with NF4 4-bit quant, DoRA adapters, citation-weighted CE.
    **Resource notes**
    * Fits on 1× T4 (16 GB) at `batch_size=2, grad_accum=16, max_seq_length=1024`.
    * For faster convergence prefer a 2× T4 kernel and use accelerate or torchrun.
    * Expect ~3–4 h per epoch at 5k QA pairs. Kaggle session cap is 12 h — aim for 1 epoch then resume.
    Attach: `aegisrag-source`, `aegisrag-synthetic`.
    """),
    bootstrap_cell("generator_sft"),
    code("""
    # Tight fit for 1× T4: reduce max_seq_length and batch size.
    import os
    os.environ["AEGIS_TRAINING__GENERATOR_SFT__MAX_SEQ_LENGTH"] = "1024"
    os.environ["AEGIS_TRAINING__GENERATOR_SFT__BATCH_SIZE"] = "2"
    os.environ["AEGIS_TRAINING__GENERATOR_SFT__GRADIENT_ACCUMULATION_STEPS"] = "16"
    os.environ["AEGIS_TRAINING__FP16"] = "true"
    os.environ["AEGIS_TRAINING__BF16"] = "false"  # T4 lacks bf16 tensor cores
    os.environ["AEGIS_TRAINING__GRADIENT_CHECKPOINTING"] = "true"

    from src.utils.config import reset_config
    reset_config()
    """),
    code("""
    from src.training.train_generator import train
    result = train()
    result
    """),
    save_working_cell("generator_sft", "SFT adapter"),
])
(OUT / "03_train_generator_sft.ipynb").write_text(json.dumps(nb03, indent=1))


# ---------------------------------------------------------------------------
# 04 - DPO
# ---------------------------------------------------------------------------
nb04 = nb([
    md("""
    # AegisRAG · Stage 04 — DPO Alignment
    6-type Direct Preference Optimisation on top of the SFT adapter from stage 03.
    **Prereq**: upload the stage-03 output as a Kaggle Dataset named `aegisrag-checkpoints` with the layout `aegisrag-checkpoints/generator_sft/<adapter files>` so the bootstrap auto-wires it.
    ~2 h for 1 epoch on 3k preference triplets with batch_size=1, grad_accum=32.
    """),
    bootstrap_cell("dpo"),
    code("""
    # DPO memory is higher than SFT (needs reference model). Shrink further.
    import os
    os.environ["AEGIS_TRAINING__DPO__MAX_SEQ_LENGTH"] = "1024"
    os.environ["AEGIS_TRAINING__DPO__MAX_PROMPT_LENGTH"] = "512"
    os.environ["AEGIS_TRAINING__DPO__BATCH_SIZE"] = "1"
    os.environ["AEGIS_TRAINING__DPO__GRADIENT_ACCUMULATION_STEPS"] = "32"

    from src.utils.config import reset_config
    reset_config()
    """),
    code("""
    from src.training.train_dpo import train
    result = train()
    result
    """),
    save_working_cell("dpo", "DPO adapter"),
])
(OUT / "04_train_dpo.ipynb").write_text(json.dumps(nb04, indent=1))


# ---------------------------------------------------------------------------
# 05 - Confidence head + calibration
# ---------------------------------------------------------------------------
nb05 = nb([
    md("""
    # AegisRAG · Stage 05 — Confidence Head + Temperature Calibration
    Trains the soft-label confidence head on `confidence_labels.jsonl`, then sweeps temperature on a held-out split to minimise MSE vs. soft labels and reports ECE.
    Runs in ~5 min on a CPU kernel.
    """),
    bootstrap_cell("confidence_head"),
    code("""
    from src.training.train_confidence import train
    result = train()
    result
    """),
    save_working_cell("confidence_head", "Confidence head"),
])
(OUT / "05_train_confidence.ipynb").write_text(json.dumps(nb05, indent=1))


# ---------------------------------------------------------------------------
# 06 - Alpha network
# ---------------------------------------------------------------------------
nb06 = nb([
    md("""
    # AegisRAG · Stage 06 — Alpha Fusion Network
    Learns per-query dense/sparse fusion weights from `alpha_labels.jsonl` (oracle alpha computed via alpha-sweep recall@10 during data generation).
    ~2–3 min on CPU.
    """),
    bootstrap_cell("alpha_network"),
    code("""
    from src.training.train_alpha import train
    result = train()
    result
    """),
    save_working_cell("alpha_network", "Alpha network"),
])
(OUT / "06_train_alpha.ipynb").write_text(json.dumps(nb06, indent=1))


# ---------------------------------------------------------------------------
# 07 - Decomposer adapter
# ---------------------------------------------------------------------------
nb07 = nb([
    md("""
    # AegisRAG · Stage 07 — Query Decomposer Adapter
    Small QLoRA+DoRA adapter over `Qwen/Qwen2.5-7B-Instruct` targeting only `q_proj,k_proj,v_proj,o_proj`, trained on `decomp_labels.jsonl`.
    ~45 min on 1× T4 for 2 epochs on 1.5k pairs.
    """),
    bootstrap_cell("decomposer"),
    code("""
    import os
    os.environ["AEGIS_TRAINING__DECOMPOSER__MAX_SEQ_LENGTH"] = "768"
    os.environ["AEGIS_TRAINING__DECOMPOSER__BATCH_SIZE"] = "2"
    os.environ["AEGIS_TRAINING__DECOMPOSER__GRADIENT_ACCUMULATION_STEPS"] = "16"

    from src.utils.config import reset_config
    reset_config()
    """),
    code("""
    from src.training.train_decomposer import train
    result = train()
    result
    """),
    save_working_cell("decomposer", "Decomposer adapter"),
])
(OUT / "07_train_decomposer.ipynb").write_text(json.dumps(nb07, indent=1))


# ---------------------------------------------------------------------------
# 08 - (optional) Run evaluation on the assembled m5
# ---------------------------------------------------------------------------
nb08 = nb([
    md("""
    # AegisRAG · Stage 08 — Full Pipeline Evaluation (optional)
    After all stages complete and `aegisrag-checkpoints` contains every output, run the benchmark + metrics suite against the held-out QA set.
    """),
    bootstrap_cell("alpha_network"),  # reuses the same env setup
    code("""
    from src.evaluation.evaluate import run_evaluation
    metrics = run_evaluation(models=["b1", "b3", "m5"], test_dir="/kaggle/input/aegisrag-synthetic", output_dir="/kaggle/working/report")
    metrics
    """),
    save_working_cell("../report", "Evaluation report"),
])
(OUT / "08_evaluate.ipynb").write_text(json.dumps(nb08, indent=1))


print("Wrote notebooks:")
for p in sorted(OUT.glob("*.ipynb")):
    print(" ", p.name)
