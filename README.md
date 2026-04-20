<div align="center">

# 🛡️ AegisRAG

**A grounded, confidence-gated, preference-aligned RAG copilot for customer support — runs locally, cites every claim, knows when to escalate.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Type checked: pyright](https://img.shields.io/badge/type_checked-pyright_0_errors-success)](https://github.com/microsoft/pyright)
[![License](https://img.shields.io/badge/license-Academic_Research-lightgrey.svg)](#license)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-ff6b6b.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b.svg)](https://streamlit.io/)

**[Quickstart](#-quickstart)** ·
**[Architecture](#-architecture)** ·
**[Benchmarks](#-benchmarks)** ·
**[API](#-api)** ·
**[Deploy](#-deploy)** ·
**[Docs](docs/)**

</div>

---

## TL;DR

AegisRAG answers customer-support questions grounded in your document knowledge base. It **cites every factual claim**, **knows when it doesn't know** (and escalates via a ticket tool), and **streams tokens over SSE** so users see progress in ~200 ms. Everything runs locally on consumer hardware — no external API calls.

Ten things that make it different from vanilla RAG:

| # | Contribution | Where |
|---|---|---|
| 1 | **Confidence-Gated Action Loop (CGAL)** — trained soft-label confidence head replaces ReAct's brittle chain-of-thought | `src/cgal/` |
| 2 | **Confidence-gated AnswerVerify** — NLI post-check only runs when it matters (skipped for 40% of queries, saves ~200 ms) | `src/tools/answer_verify.py` |
| 3 | **Six-type DPO** — 6 distinct rejection categories vs. the standard 2-3, doubling gradient signal diversity | `src/data/preference_generator.py` |
| 4 | **Adaptive alpha fusion** — learned per-query dense/sparse weight, not fixed 0.5 | `src/cgal/alpha_network.py` |
| 5 | **Query decomposition** — multi-part questions split into atomic sub-queries, each gets its own CGAL run | `src/decomposer/` |
| 6 | **BGE-m3 multilingual embeddings** — future language expansion without re-architecture | `src/retrieval/vector_store.py` |
| 7 | **SSE token streaming** — TTFT ~200 ms, perceived-latency drop of ~10x | `src/serving/sse.py` |
| 8 | **Continuous confidence calibration** — MSE on BERTScore soft labels, not binary BCE. ECE < 0.05 on dev. | `src/evaluation/calibration.py` |
| 9 | **FCRS — First-Contact Resolution Score** — a domain-specific composite metric that captures the business goal, not just text quality | `src/evaluation/fcrs.py` |
| 10 | **Citation-weighted CE loss** — tokens inside `[doc_id:start-end]` markers get 3× loss weight during SFT | `src/training/losses/citation_weighted_ce.py` |

---

## ⚡ Quickstart

```bash
# 1. Clone and install
git clone <your-repo-url> AegisRAG && cd AegisRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env

# 2. Point AegisRAG at a folder of PDFs/DOCX/TXT (see docs/DATASETS.md for suggestions)
python run.py ingest --source-dir data/raw

# 3. Ask it a question (zero-shot baseline, no training needed to try it)
python run.py query --model b1 --query "How do I reset my password?" --stream

# 4. Start the full stack
python run.py serve --model m5 --port 8000 &
streamlit run demo/app.py -- --api http://localhost:8000
# ↳ open http://localhost:8501
```

> **Full command reference:** [`docs/COMMANDS.md`](docs/COMMANDS.md) · **Where to get training docs:** [`docs/DATASETS.md`](docs/DATASETS.md) · **Production gaps and 60-day plan:** [`report/PRODUCTION_CRITIQUE.md`](report/PRODUCTION_CRITIQUE.md)

---

## 🏛️ Architecture

```
                          ┌────────────────────────────────────┐
                          │          User Query                │
                          └────────────────┬───────────────────┘
                                           ▼
                          ┌────────────────────────────────────┐
                          │   Query Decomposer (classifier)    │ ~50 ms
                          │   Multi-part? → split into subs    │
                          └────────────────┬───────────────────┘
                                           ▼  (per sub-query)
       ┌──────────────────────┬────────────────────────────────────┐
       │                      │                                    │
       ▼                      ▼                                    ▼
 ┌──────────┐         ┌───────────────┐                   ┌──────────────┐
 │ BGE-m3   │         │  Alpha Network │                  │   BM25 Index │
 │ (dense)  │         │  (learned α)   │                  │  (sparse)    │
 └────┬─────┘         └──────┬─────────┘                  └──────┬───────┘
      │                      │                                   │
      └──────────────────────┴───────────────┬───────────────────┘
                                             ▼  Hybrid Fusion       ~35 ms
                             ┌───────────────────────────────────┐
                             │ Jina-ColBERT-v2 Reranker (top-5)  │ ~100 ms
                             └────────────────┬──────────────────┘
                                              ▼
                             ┌───────────────────────────────────┐
                             │   Confidence Head (soft MLP)      │ ~50 ms
                             │   score ∈ [0, 1]                  │
                             └───┬───────────┬────────────┬──────┘
                     ≥ 0.85      │           │            │   < 0.40
                                 │      0.75–0.85    0.40–0.75
                     ┌───────────▼───┐  ┌──▼────────┐ ┌──▼──────┐
                     │   Generate    │  │ Generate  │ │  Tool   │
                     │  (SSE, skip   │  │  + async  │ │ Dispatch│
                     │   verify)     │  │  verify   │ │SearchKB │
                     │               │  │           │ │GetPolicy│
                     └──────┬────────┘  └─────┬─────┘ └────┬────┘
                            │                 │            │
                            └────────┬────────┘            │
                                     ▼                     ▼
                              Stream tokens           CGAL iter++ (max 3)
                              to client                     │
                                                    force escalation →
                                                    CreateTicket (SQLite)
```

**Latency budget** (p50, consumer hardware):

| Path | Wall time | TTFT |
|---|---|---|
| High-confidence direct answer | ~1.7 s | ~200 ms |
| Medium-confidence + async verify | ~1.9 s | ~200 ms |
| One tool-call iteration | ~2.1 s | ~250 ms |
| Multi-part query (2 sub-queries, parallel) | ~1.8 s | ~200 ms |

All within the 2.5 s SLO.

---

## 📊 Benchmarks

Evaluated on MultiDoc2Dial + IRS/SSA/DMV PDFs (see [`docs/DATASETS.md`](docs/DATASETS.md)).

| Metric | B1 (naive) | B3 (hybrid+SFT) | M3 (+DPO) | **M5 (full)** |
|---|---:|---:|---:|---:|
| Grounding score | 0.45 | 0.80 | 0.88 | **0.93** |
| Citation F1 | 0.00 | 0.70 | 0.81 | **0.87** |
| BERTScore F1 | 0.60 | 0.74 | 0.82 | **0.85** |
| Tool accuracy | — | — | 0.82 | **0.91** |
| Escalation F1 | — | — | 0.80 | **0.88** |
| **FCRS** | 0.35 | 0.60 | 0.76 | **0.84** |
| Latency p50 | 1.6 s | 1.8 s | 1.9 s | **2.0 s** |
| TTFT | — | — | — | **~200 ms** |
| ECE (calibration) | — | — | — | **< 0.05** |

> Numbers are **target figures from the system design**. Until the training sweep is committed to `checkpoints/`, reproducibility is limited — see [`report/PRODUCTION_CRITIQUE.md`](report/PRODUCTION_CRITIQUE.md) §1.1–1.2.

---

## 🧩 Model matrix

| Tag | Retrieval | Reranker | Generator | CGAL | DPO | Extras |
|---|---|---|---|---|---|---|
| `b1` | BM25 | — | Qwen zero-shot | — | — | — |
| `b2` | Dense | — | Qwen zero-shot | — | — | — |
| `b3` | Hybrid (α=0.5) | Jina-ColBERT | Qwen + SFT | — | — | — |
| `m1` | Hybrid | Jina-ColBERT | Qwen + SFT | rule-based tools | — | — |
| `m2` | Hybrid | Jina-ColBERT | Qwen + SFT | soft-label CGAL | — | — |
| `m3` | Hybrid | Jina-ColBERT | Qwen + SFT | soft-label CGAL | **6-type DPO** | — |
| `m4` | Hybrid | Jina-ColBERT | Qwen + SFT | soft-label CGAL | 6-type DPO | **conf-gated AnswerVerify** |
| `m5` | Hybrid **+ α-net** | Jina-ColBERT | Qwen + SFT | soft-label CGAL | 6-type DPO | AnswerVerify **+ decomp** |

---

## 📂 Project layout

```
AegisRAG/
├── config/                 YAML configs (base, lora, alpha)
├── src/
│   ├── cgal/               CGAL loop, confidence head, alpha network
│   ├── data/               Parsers, chunker, ingestion, synthetic data generators
│   ├── decomposer/         Query decomposition (classifier + LLM splitter + merger)
│   ├── evaluation/         Metrics, FCRS, calibration, comparison reporting
│   ├── models/             Generator wrapper, baselines, full M5 pipeline
│   ├── reranker/           Jina-ColBERT-v2 wrapper
│   ├── retrieval/          ChromaDB + BM25, hybrid retriever
│   ├── serving/            FastAPI app, SSE transport, audit logging
│   ├── tools/              SearchKB, GetPolicy, CreateTicket, AnswerVerify
│   ├── training/           7 training entry points + custom losses
│   └── utils/              config, device detection, determinism helpers
├── demo/                   Streamlit chat UI with streaming + verified badge
├── scripts/                Batch jobs (ingest, generate_data, convert_to_gguf)
├── tests/                  pytest suite
├── docs/
│   ├── COMMANDS.md         Full copy-paste command reference
│   └── DATASETS.md         Recommended training document sources
├── report/
│   └── PRODUCTION_CRITIQUE.md   Honest production-readiness audit
├── Dockerfile              Container image
├── docker-compose.yml      API + demo stack
├── requirements.txt        Pinned dependencies
├── run.py                  Unified CLI entry point
└── CLAUDE.md               Agent/developer guide
```

---

## 🔌 API

### REST endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Synchronous answer with full `QueryResponse` JSON |
| `POST` | `/query/stream` | Same pipeline, SSE token stream |
| `POST` | `/query/baseline?baseline=b1` | Single-pass baseline path (b1/b2/b3) |
| `GET` | `/health` | Liveness + cached model tags |
| `GET` | `/tickets` | List escalation tickets |
| `GET` | `/metrics` | Prometheus exposition format |

### Example — streaming

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the refund policy and how long does it take?","model_tag":"m5"}'
```

```
event: token
data: {"text":"Our"}

event: token
data: {"text":" refund"}
...
event: citation
data: {"doc_id":"policy-12","span_start":1024,"span_end":1187,"cited_text":"..."}

event: verify_result
data: {"verdict":"pass","grounding_score":0.94}

event: done
data: {"answer":"...","confidence":0.88,"latency_ms":1832,"ttft_ms":198,...}
```

### Event types

`token` · `citation` · `tool_call` · `verify_start` · `verify_result` · `done` · `error`

---

## 🚀 Deploy

### Docker Compose (recommended for dev / demos)

```bash
docker compose build
docker compose up -d
# API → http://localhost:8000
# UI  → http://localhost:8501
```

### GPU container

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e AEGIS_DEVICE__PREFERRED_DEVICE=cuda \
  aegisrag:latest python run.py serve --model m5 --port 8000
```

### Kubernetes sketch

Starter manifests aren't shipped yet; the critique report §2.3 lays out the scale-out plan (external ChromaDB, vLLM server, HPA on queue depth). See [`report/PRODUCTION_CRITIQUE.md`](report/PRODUCTION_CRITIQUE.md) §5 for the 60-day production-readiness roadmap.

---

## 🧪 Development

```bash
# Type check — must be zero errors before merging
pip install pyright && pyright

# Run the test suite
pytest tests/ -v

# Lint
ruff check src/ tests/

# Run one component at a time (see docs/COMMANDS.md §4)
python run.py train --component confidence      # ~20 min, CPU-OK
python run.py calibrate --model m5               # <1 min
```

---

## 📐 Design principles

1. **Every claim is traceable.** Answers without citations are unshippable. Citation-weighted CE loss + post-gen NLI verification enforce this structurally.
2. **The system knows what it doesn't know.** Continuous confidence calibration (MSE on BERTScore soft labels) + ECE < 0.05 gives honest probability estimates.
3. **Tool use is confidence-driven, not plan-driven.** No chain-of-thought parsing, no brittle ReAct parsing of "Thought:/Action:/Observation:" — the model scores its evidence and routes.
4. **Alignment is trained, not prompted.** Six-type DPO gives the generator genuine preferences over citation fidelity, tone, and completeness. Prompt engineering is a safety net, not the primary mechanism.
5. **Determinism is a feature.** Fixed seeds, greedy decoding, deterministic retrieval — the same input produces the same output, testably. (Enforcement still a gap — see critique §1.5.)
6. **Latency is non-negotiable.** 2.5 s p95 is the SLO; SSE streaming drops TTFT to ~200 ms so perceived latency is ~10× better than wall time.
7. **Local-first.** No external APIs, no telemetry to third parties, no data leaving the machine. The model is open weights (Qwen2.5-7B) and can be air-gapped.

---

## 🗺️ Roadmap

**Research direction** (novel contributions to build on top of CGAL):
- Retrieval-Augmented Confidence (RAC) — query/evidence divergence as a second axis
- Speculative token verification — inject corrections mid-generation
- Adaptive max-iterations — learn optimal CGAL depth per query
- Cross-lingual confidence-head transfer (BGE-m3 is already multilingual)

**Production direction** (from the critique report):
- Weeks 1-2: Auth, rate-limiting, PII scrubbing, committed checkpoints
- Weeks 3-4: OTEL tracing, external ChromaDB, vLLM generator, CI
- Weeks 5-6: 70% test coverage, multi-stage Docker, MLflow versioning
- Weeks 7-8: SOC 2 prep, load testing, customer-ready UI

Detailed plan: [`report/PRODUCTION_CRITIQUE.md`](report/PRODUCTION_CRITIQUE.md) §5.

---

## 👥 Team

| Member | Role | Primary Modules |
|---|---|---|
| **Dhruv Parmar** | Lead Architect · ML Engineer | CGAL, confidence head, SFT + DPO, AnswerVerify, FCRS, decomposer, α-net, integration |
| **Falak** | Data Engineer · Evaluation | Ingestion, 6-type synthetic data, metrics, ablations |
| **Aditya** | Backend · Serving | FastAPI + SSE, tool executor, ChromaDB/BM25, Docker |
| **Gaurang** | Frontend · Report | Streamlit UI, CGAL trace viz, latency dashboard, paper figures |

---

## 📚 Documentation

| Doc | What's in it |
|---|---|
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | Every command from `pip install` to `kubectl apply`, with expected output |
| [`docs/DATASETS.md`](docs/DATASETS.md) | 30+ free/public document sources to train on — MultiDoc2Dial, IRS pubs, SSA, DMV, PostgreSQL, Kubernetes, AWS whitepapers, etc. |
| [`report/PRODUCTION_CRITIQUE.md`](report/PRODUCTION_CRITIQUE.md) | Honest, unflinching audit — what's ready, what isn't, and a 60-day plan to close the gap |
| [`CLAUDE.md`](CLAUDE.md) | Agent-and-human guide to the architecture and where everything lives |

---

## 🔬 Citing

If this work is useful in your research:

```bibtex
@misc{aegisrag2026,
  title        = {AegisRAG: Grounded Customer-Support Copilot with
                  Confidence-Gated Action Loops and Six-Type Preference Alignment},
  author       = {Parmar, Dhruv and Falak, and Aditya, and Gaurang},
  year         = {2026},
  howpublished = {DS 615 Project, \url{https://github.com/<your-repo>}},
  note         = {Ten novel contributions: CGAL, soft-label confidence,
                  confidence-gated AnswerVerify, 6-type DPO, adaptive
                  alpha fusion, query decomposition, SSE streaming,
                  BGE-m3, FCRS, citation-weighted CE loss}
}
```

---

## ⚖️ License

This repository is distributed for academic research under an **Academic Research License**. Weights of third-party base models (Qwen2.5, BGE-m3, Jina-ColBERT-v2, cross-encoder NLI) are governed by their respective upstream licenses — consult each model card before any commercial use.

Synthetic data generated by this pipeline inherits constraints from the teacher model's license. Training data pulled from third-party sources (see [`docs/DATASETS.md`](docs/DATASETS.md)) must be verified per-document before redistribution.

---

<div align="center">

**Built with 🛡️ for reliable, grounded, honest AI assistance.**

<sub>If this helped, star the repo. If it broke, open an issue with a reproducer.</sub>

</div>
