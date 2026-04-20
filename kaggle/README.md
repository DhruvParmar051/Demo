# AegisRAG on Kaggle

End-to-end training runbook for the AegisRAG pipeline using Kaggle
Notebooks + Kaggle Datasets. Every training component has its own
notebook under `kaggle/notebooks/` and can run within a single free
Kaggle session.

---

## 1 · One-time Kaggle Dataset setup

Create these Kaggle Datasets (Dataset → "New Dataset"). Slugs must match
exactly — the bootstrap helper looks them up by name.

| Slug                      | Contents                                                                    | When you upload it                          |
| ------------------------- | --------------------------------------------------------------------------- | ------------------------------------------- |
| `aegisrag-source`         | Snapshot of the repo (`src/`, `config/`, `scripts/`, `kaggle/`, `run.py`, `requirements.txt`) | Every time you push code changes            |
| `aegisrag-raw-docs`       | Your customer-support corpus (PDF / DOCX / TXT / MD)                        | Once, before Stage 00                       |
| `aegisrag-synthetic`      | JSONL files produced by Stage 00                                            | After Stage 00 completes                    |
| `aegisrag-checkpoints`    | Checkpoints saved from earlier training stages                              | Grow after each stage                        |

### What to upload for `aegisrag-source`

From the repo root, zip the following and upload as a new Kaggle Dataset:

```
src/                 (whole folder)
config/              (whole folder)
scripts/             (whole folder)
kaggle/              (whole folder)
run.py
requirements.txt
```

You do **not** need `.venv/`, `data/`, `checkpoints/`, `tests/`, or `demo/`.

### What to upload for `aegisrag-raw-docs`

Just drop every source document (PDF / DOCX / TXT / MD / CSV / JSON) at
the top level of the dataset — the ingestion code walks recursively.

### What the bootstrap expects after Stage 00

`aegisrag-synthetic/` should end up containing:

```
qa_pairs.jsonl
preferences.jsonl
confidence_labels.jsonl
alpha_labels.jsonl
decomp_labels.jsonl
tool_route_labels.jsonl
```

### Checkpoint dataset layout

After each training stage, download the `/kaggle/working/checkpoints/<comp>`
folder from the notebook's "Output" tab and upload it into the
`aegisrag-checkpoints` dataset under:

```
aegisrag-checkpoints/
├── retriever/
├── reranker/
├── generator_sft/        ← needed by Stage 04 (DPO)
├── dpo/
├── confidence_head/
├── alpha_network/
└── decomposer/
```

The bootstrap helper auto-detects each subfolder and wires its path into
the config via `AEGIS_TRAINING__<COMP>__OUTPUT_DIR`.

---

## 2 · Notebook run order

| # | Notebook                       | Accelerator | Attach datasets                                          | Typical runtime         |
| - | ------------------------------ | ----------- | -------------------------------------------------------- | ----------------------- |
| 00 | `00_generate_data.ipynb`       | CPU or GPU  | `aegisrag-source`, `aegisrag-raw-docs`                  | 30 – 60 min             |
| 01 | `01_train_retriever.ipynb`     | GPU T4 × 1 | `aegisrag-source`, `aegisrag-synthetic`                 | ~20 min                 |
| 02 | `02_train_reranker.ipynb`      | GPU T4 × 1 | `aegisrag-source`, `aegisrag-synthetic`                 | ~30 min                 |
| 03 | `03_train_generator_sft.ipynb` | GPU T4 × 2 (preferred) | `aegisrag-source`, `aegisrag-synthetic`      | 3 – 4 h / epoch         |
| 04 | `04_train_dpo.ipynb`           | GPU T4 × 2 | `aegisrag-source`, `aegisrag-synthetic`, `aegisrag-checkpoints` (with `generator_sft/`) | ~2 h / epoch |
| 05 | `05_train_confidence.ipynb`    | CPU         | `aegisrag-source`, `aegisrag-synthetic`                 | ~5 min                  |
| 06 | `06_train_alpha.ipynb`         | CPU         | `aegisrag-source`, `aegisrag-synthetic`                 | ~3 min                  |
| 07 | `07_train_decomposer.ipynb`    | GPU T4 × 1 | `aegisrag-source`, `aegisrag-synthetic`                 | ~45 min                 |
| 08 | `08_evaluate.ipynb` (optional) | GPU T4 × 1 | all of the above                                         | 30 – 60 min             |

Stages 01 – 02 and 05 – 07 are independent and can run in parallel tabs.

Stage 03 (SFT) and Stage 04 (DPO) are the two heavy ones. **They must run
sequentially** — DPO loads the SFT adapter as both policy and reference.

---

## 3 · How the bootstrap wires Kaggle paths into the config

`kaggle/helpers/bootstrap.py` sets these environment variables, which the
project's Pydantic-settings loader (`src/utils/config.py`) picks up via
the `AEGIS_*` prefix and `__` nested-key delimiter:

| Env var                                      | Effective override                                         |
| -------------------------------------------- | ---------------------------------------------------------- |
| `AEGIS_DATA__RAW_DOCS_DIR`                   | `cfg.data.raw_docs_dir` → `/kaggle/input/aegisrag-raw-docs`|
| `AEGIS_DATA__SYNTHETIC_DIR`                  | `cfg.data.synthetic_dir` → `/kaggle/input/aegisrag-synthetic` |
| `AEGIS_DATA__SYNTHETIC__QA_PATH` (+ other 5) | `cfg.data.synthetic.<*>_path` fully resolved              |
| `AEGIS_DATA__VECTOR_DB_PATH`                 | `/kaggle/working/vectordb`                                 |
| `AEGIS_DATA__BM25_INDEX_PATH`                | `/kaggle/working/bm25_index.pkl`                           |
| `AEGIS_TRAINING__<COMPONENT>__OUTPUT_DIR`    | `/kaggle/working/checkpoints/<component>`                  |
| `AEGIS_TRAINING__FP16` / `BF16`              | `true` / `false` (T4 lacks bf16 tensor cores)              |
| `HF_HOME`, `TRANSFORMERS_CACHE`              | `/kaggle/working/.cache/huggingface` (persistent within session) |

Nothing under `/kaggle/input` is written to — all writes go to
`/kaggle/working`, which is included in the notebook's Output.

---

## 4 · Saving outputs

After each notebook finishes, commit it with **"Save Version" → "Save &
Run All (Commit)"** so `/kaggle/working/checkpoints/<component>` is
captured as the output. Then:

1. Open the committed notebook's **"Output"** tab.
2. Click **"New Dataset from Output"** (or, if you already have
   `aegisrag-checkpoints`, click "Update Dataset" and drop the folder in).
3. Use the same slug so the next notebook's bootstrap auto-mounts it.

---

## 5 · Troubleshooting

**`ModuleNotFoundError` during `setup_kaggle`** — the
`aegisrag-source` dataset isn't attached, or the slug differs. Fix:
`Settings → Add data → aegisrag-source`.

**OOM during Stage 03 SFT on 1× T4** — switch the notebook's accelerator
to "GPU T4 × 2", or drop `max_seq_length` from 1024 → 768 by overriding
`AEGIS_TRAINING__GENERATOR_SFT__MAX_SEQ_LENGTH=768` before importing the
trainer.

**`bf16` errors** — T4 does not have bf16 tensor cores. The bootstrap
forces `fp16`; if you manually flipped `AEGIS_TRAINING__BF16=true`,
revert it.

**DPO says "SFT adapter not found"** — your `aegisrag-checkpoints`
dataset is missing the `generator_sft/` folder at the top level. The
bootstrap looks for `/kaggle/input/aegisrag-checkpoints/generator_sft`.

**Session timeout at 9 h** — Kaggle kills interactive sessions at ~9 h,
committed runs at 12 h. For Stage 03 / 04 always use "Save & Run All
(Commit)" so the full 12 h is available. Save a checkpoint every
`save_steps` by lowering the value in `config/base.yaml` if you want
resumable runs across sessions.

**Permissions error writing to `/kaggle/input`** — you're accidentally
writing to the read-only mount. The bootstrap forces all writes to
`/kaggle/working`; if you see this error, check you haven't re-set an
`AEGIS_DATA__*` env var back to `/kaggle/input` after
`setup_kaggle()`.

---

## 6 · Regenerating the notebooks

The notebooks are generated from `kaggle/helpers/gen_notebooks.py`.
After editing that script, run:

```bash
python kaggle/helpers/gen_notebooks.py
```

All eight notebooks will be rewritten in place under `kaggle/notebooks/`.
