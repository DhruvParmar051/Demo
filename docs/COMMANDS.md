# AegisRAG — Full Command Reference

Every command you need, in the order you need it. Copy-pasteable.

---

## 0. Prerequisites

```bash
# Check you have the right Python and git
python --version                  # need 3.10+
git --version                     # need 2.30+
# GPU users only:
nvidia-smi                        # confirm CUDA 12.1+ visible
# macOS users: Apple Silicon recommended (MPS auto-detected)
```

---

## 1. One-time setup

```bash
# 1.1 Clone
git clone <your-repo-url> AegisRAG
cd AegisRAG

# 1.2 Virtual env (pick one)
python -m venv .venv && source .venv/bin/activate        # venv
# OR
conda create -n aegisrag python=3.10 -y && conda activate aegisrag

# 1.3 Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 1.4 NLP models (spaCy for sentence segmentation in AnswerVerify)
python -m spacy download en_core_web_sm

# 1.5 Environment
cp .env.example .env
# then open .env and fill in: HF_TOKEN, AEGIS_LOG_LEVEL, (optional) OPENAI_API_KEY
```

Optional but recommended:

```bash
# 1.6 Pre-download model weights so first-run isn't slow
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "BAAI/bge-m3",
    "jinaai/jina-colbert-v2",
    "Qwen/Qwen2.5-7B-Instruct",
    "cross-encoder/nli-MiniLM2-L6-H768",
]:
    snapshot_download(repo_id=repo)
PY

# 1.7 Sanity-check the install
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'mps:', torch.backends.mps.is_available())"
pytest tests/ -q                  # should pass
```

---

## 2. Document ingestion (build the KB)

```bash
# 2.1 Drop your source docs into data/raw/ — PDFs, DOCX, TXT, Markdown
#     See docs/DATASETS.md for recommended sources.
ls data/raw/

# 2.2 Ingest: parse → chunk (256 tokens, 64 overlap) → MinHash dedup → ChromaDB + BM25
python run.py ingest --source-dir data/raw

# 2.3 Verify
python -c "
from src.retrieval.vector_store import ChromaVectorStore
store = ChromaVectorStore()
print(f'Chunks indexed: {store.collection.count()}')
"
```

For a large corpus, this can take 30-90 minutes. Add `--vector-db-path` to control the output dir.

---

## 3. Synthetic training data generation

All data types get written to `data/synthetic/` as JSONL.

```bash
# 3.1 Generate everything at once (QA → preference/confidence/alpha/decomp)
python run.py generate-data --type all --output-dir data/synthetic

# Or one at a time:
python run.py generate-data --type qa            # seed — must run first
python run.py generate-data --type preference    # 6-type DPO triplets
python run.py generate-data --type confidence    # BERTScore soft labels
python run.py generate-data --type alpha         # grid-searched fusion weights
python run.py generate-data --type decomp        # multi-part query labels

# 3.2 Split into train / dev / test (creates data/processed/{train,dev,test}/)
python scripts/split_data.py --seed 42 --train 0.7 --dev 0.15 --test 0.15 \
  --stratify-by domain,question_type
```

Expected output sizes (from plan): 5K QA, 3K preference, 3K confidence, 1K alpha, 500 decomp.

---

## 4. Training (7 components)

Each component can run independently; `all` runs them in the correct order.

```bash
# 4.1 All components in sequence (the canonical full train)
python run.py train --component all
```

Individual components, in recommended order:

```bash
# 4.2 Retriever fine-tune (BGE-m3, MNRL loss, ~1h GPU / 3h CPU)
python run.py train --component retriever

# 4.3 Reranker fine-tune (Jina-ColBERT-v2, BCE, ~20 min)
python run.py train --component reranker

# 4.4 Generator SFT (Qwen2.5-7B + QLoRA/DoRA, citation-weighted CE, ~8-10h GPU)
python run.py train --component generator

# 4.5 DPO alignment (6-type rejection, beta=0.1, ~3-4h GPU)
python run.py train --component dpo

# 4.6 Confidence head + tool-policy (joint MSE+CE, ~20 min)
python run.py train --component confidence

# 4.7 Alpha fusion network (tiny MLP, ~2 min)
python run.py train --component alpha

# 4.8 Query decomposition classifier (~5 min)
python run.py train --component decomposer
```

Post-training — export the SFT+DPO merged model to GGUF for fast inference:

```bash
# 4.9 Merge LoRA adapters into base weights and convert to GGUF q4_k_m
python scripts/convert_to_gguf.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path checkpoints/generator_sft \
  --dpo-path checkpoints/generator_dpo \
  --output checkpoints/qwen2.5-7b-aegisrag-q4_k_m.gguf \
  --quant q4_k_m
```

---

## 5. Calibration (after training the confidence head)

```bash
# 5.1 Temperature-scale the confidence head on the dev set
python run.py calibrate --model m5
#   -> prints ECE before/after, temperature, AUROC
#   -> writes checkpoints/confidence_temperature.json
```

---

## 6. Evaluation

```bash
# 6.1 Evaluate all baselines and improved models (full matrix)
python run.py evaluate \
  --models b1,b2,b3,m1,m2,m3,m4,m5 \
  --test-dir data/processed/test \
  --output-dir report/

# 6.2 Quick: just baseline vs full system
python run.py evaluate --models b1,m5 --test-dir data/processed/test --output-dir report/

# 6.3 Decomposition-only spot-check
python run.py test-decomp --n 20

# 6.4 Benchmark latency (p50 / p95 / p99 / throughput)
python run.py benchmark --model m5 --n 100
```

Report artifacts land in `report/` — per-model JSON, `summary.json`, and a comparison markdown if `generate_report` succeeds.

---

## 7. Running the system

### 7.1 Single query (no server)

```bash
# Non-streaming
python run.py query --model m5 --query "How do I reset my password?"

# Streaming (SSE-style token rendering in the terminal)
python run.py query --model m5 --query "What is the refund policy and how fast is it processed?" --stream
```

### 7.2 FastAPI server

```bash
# Dev mode (one worker, auto-reload)
python run.py serve --model m5 --host 0.0.0.0 --port 8000

# Query it from another terminal:
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I cancel my subscription?","model_tag":"m5"}'

# Streaming:
curl -N -X POST http://localhost:8000/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the refund policy?","model_tag":"m5"}'

# Health, metrics, tickets:
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/tickets
```

### 7.3 Streamlit demo (chat UI with streaming + verified badge)

```bash
streamlit run demo/app.py -- --api http://localhost:8000
# opens http://localhost:8501
```

---

## 8. Docker deployment

### 8.1 Local (docker-compose)

```bash
# 8.1.1 Build the image once
docker compose build

# 8.1.2 Start API + demo
docker compose up -d

# 8.1.3 Check logs
docker compose logs -f api
docker compose logs -f demo

# 8.1.4 Stop
docker compose down
```

API at `http://localhost:8000`, demo at `http://localhost:8501`.

### 8.2 Single container (no compose)

```bash
docker build -t aegisrag:latest .

docker run --rm -it \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e AEGIS_DEVICE__PREFERRED_DEVICE=cpu \
  aegisrag:latest \
  python run.py serve --model m5 --port 8000
```

### 8.3 GPU container

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e AEGIS_DEVICE__PREFERRED_DEVICE=cuda \
  aegisrag:latest \
  python run.py serve --model m5 --port 8000
```

(You'll want the `nvidia/cuda:12.1-base-ubuntu22.04` base image instead of `python:3.11-slim` for production GPU — edit `Dockerfile`.)

---

## 9. Production deploy (Kubernetes sketch)

```bash
# 9.1 Push to a registry
docker tag aegisrag:latest <your-registry>/aegisrag:v1.0.0
docker push <your-registry>/aegisrag:v1.0.0

# 9.2 Apply manifests (you'll need to write these — see deploy/k8s/ for a starter)
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secrets.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/ingress.yaml

# 9.3 Verify
kubectl -n aegisrag get pods
kubectl -n aegisrag logs -f deployment/aegisrag-api
kubectl -n aegisrag port-forward svc/aegisrag-api 8000:8000
```

> `deploy/k8s/*` is not yet in the repo — see the critique report for the scale-out plan. For a first production deploy, you want:
> - ChromaDB as a separate deployment (not file-backed)
> - vLLM server for the generator
> - Horizontal pod autoscaler on CPU / request queue depth
> - Ingress with TLS + rate limiting (nginx-ingress + ModSecurity)

---

## 10. Utility & maintenance

```bash
# 10.1 Run all tests
pytest tests/ -v

# 10.2 Type check
pyright                                    # needs `pip install pyright`

# 10.3 Lint
ruff check src/ tests/                     # needs `pip install ruff`

# 10.4 Regenerate the BM25 index only (after docs change)
python -c "
from src.data.ingestion import DocumentIngestor
from pathlib import Path
DocumentIngestor().ingest(Path('data/raw'), save_bm25=True)
"

# 10.5 Wipe the vector DB and re-ingest
rm -rf data/chroma
python run.py ingest --source-dir data/raw

# 10.6 Tail the audit log
sqlite3 data/audit.sqlite "SELECT timestamp, model_tag, confidence, latency_ms FROM queries ORDER BY timestamp DESC LIMIT 20;"

# 10.7 Export all escalation tickets
sqlite3 -header -csv data/audit.sqlite "SELECT * FROM tickets" > tickets.csv
```

---

## 11. Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: bitsandbytes` on macOS | bitsandbytes is CUDA-only | expected — use MPS or CPU paths, ignore |
| `RuntimeError: MPS backend out of memory` | Qwen2.5-7B too big for MPS | switch to `AEGIS_MODELS__GENERATOR__NAME=microsoft/Phi-3.5-mini-instruct` |
| `chromadb.errors.InvalidCollectionException` | stale on-disk DB from earlier version | `rm -rf data/chroma && python run.py ingest ...` |
| `ConnectionRefused` on demo → API | API not running or bound to wrong host | start API with `--host 0.0.0.0` before demo |
| `TemplateError: chat template not set` | using base (non-Instruct) Qwen checkpoint | use `Qwen/Qwen2.5-7B-Instruct`, not `Qwen/Qwen2.5-7B` |
| Generation is gibberish after DPO | DPO destabilized; beta too high or lr too high | lower beta to 0.05, lr to 2e-6, retrain |
| Slow first query (~30s) | lazy model load | `curl http://localhost:8000/query -d '{"query":"warmup"}'` once at startup |
| `ModuleNotFoundError: src.xyz` | PYTHONPATH not set inside Docker | confirm `ENV PYTHONPATH=/app` is in Dockerfile |

---

## 12. The shortest possible end-to-end run (for a reviewer)

If someone hands this repo to a reviewer with no context:

```bash
git clone <url> AegisRAG && cd AegisRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env

# 5 min — ingest a sample corpus shipped in data/raw/sample/
python run.py ingest --source-dir data/raw/sample

# 30 sec — skip training, just use zero-shot baseline
python run.py query --model b1 --query "How do I file a complaint?"

# Start the UI
python run.py serve --model b1 --port 8000 &
streamlit run demo/app.py -- --api http://localhost:8000
```

That's it — from clone to interactive demo in under 10 minutes on a laptop.
