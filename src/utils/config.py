"""
AegisRAG Configuration Module.

Loads config from config/base.yaml using Pydantic BaseSettings.
Supports environment variable overrides with AEGIS_ prefix.
Provides a singleton pattern for global config access.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------

class RetrieverModelConfig(BaseModel):
    name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    max_seq_length: int = 8192
    normalize_embeddings: bool = True
    pooling_method: str = "cls"


class RerankerModelConfig(BaseModel):
    name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_seq_length: int = 8192
    max_query_length: int = 128


class GeneratorModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    gguf_path: Optional[str] = None
    gguf_n_ctx: int = 4096
    gguf_n_gpu_layers: int = -1


class NLIModelConfig(BaseModel):
    name: str = "cross-encoder/nli-MiniLM2-L6-H768"
    max_seq_length: int = 256


class DecomposerModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2


class ModelsConfig(BaseModel):
    retriever: RetrieverModelConfig = RetrieverModelConfig()
    reranker: RerankerModelConfig = RerankerModelConfig()
    generator: GeneratorModelConfig = GeneratorModelConfig()
    nli: NLIModelConfig = NLIModelConfig()
    decomposer: DecomposerModelConfig = DecomposerModelConfig()


class RetrievalConfig(BaseModel):
    top_k: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 256
    chunk_overlap: int = 64
    min_chunk_size: int = 30
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    initial_alpha: float = 0.5
    alpha_min: float = 0.1
    alpha_max: float = 0.9
    similarity_metric: str = "cosine"
    dedup_threshold: float = 0.92


class CGALConfig(BaseModel):
    max_iterations: int = 3
    high_confidence: float = 0.35
    medium_confidence: float = 0.25
    low_confidence: float = 0.15
    verify_low: float = 0.40
    verify_high: float = 0.55
    escalation_message: str = (
        "I'm not confident enough to answer this accurately. "
        "Let me connect you with a human agent."
    )
    enable_query_decomposition: bool = True
    decomposition_threshold: float = 0.35


class RetrieverTrainingConfig(BaseModel):
    learning_rate: float = 2.0e-5
    num_epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    negatives_per_query: int = 7
    hard_negative_mining: bool = True
    hard_negatives: int = 3
    in_batch_negatives: bool = True
    loss: str = "infonce"
    triplet_margin: float = 0.2
    temperature: float = 0.05
    output_dir: str = "checkpoints/retriever"

    @property
    def epochs(self) -> int:  # alias used by train_retriever
        return self.num_epochs


class RerankerTrainingConfig(BaseModel):
    learning_rate: float = 1.0e-5
    num_epochs: int = 3
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1
    pos_neg_ratio: float = 2.0
    output_dir: str = "checkpoints/reranker"

    @property
    def epochs(self) -> int:  # alias used by train_reranker
        return self.num_epochs


class GeneratorSFTTrainingConfig(BaseModel):
    learning_rate: float = 2.0e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    packing: bool = False
    output_dir: str = "checkpoints/generator_sft"

    @property
    def epochs(self) -> int:  # alias
        return self.num_epochs


class DPOTrainingConfig(BaseModel):
    learning_rate: float = 5.0e-5
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    beta: float = 0.1
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    loss_type: str = "sigmoid"
    preference_types: List[str] = Field(default_factory=lambda: [
        "factual_grounding",
        "citation_accuracy",
        "refusal_calibration",
        "tone_formality",
        "completeness",
        "safety_compliance",
    ])
    output_dir: str = "checkpoints/dpo"


class ConfidenceHeadTrainingConfig(BaseModel):
    learning_rate: float = 1.0e-3
    num_epochs: int = 10
    batch_size: int = 64
    warmup_ratio: float = 0.05
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    dropout: float = 0.1
    label_smoothing: float = 0.05
    loss: str = "kl_divergence"
    calibration_method: str = "temperature_scaling"
    calibration_bins: int = 15
    output_dir: str = "checkpoints/confidence_head"

    @property
    def epochs(self) -> int:  # alias
        return self.num_epochs


class AlphaNetworkTrainingConfig(BaseModel):
    learning_rate: float = 1.0e-3
    num_epochs: int = 15
    batch_size: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 1.0
    hidden_dim: int = 128
    dropout: float = 0.1
    loss: str = "mse"
    output_dir: str = "checkpoints/alpha_network"

    @property
    def epochs(self) -> int:  # alias
        return self.num_epochs


class DecomposerTrainingConfig(BaseModel):
    learning_rate: float = 2.0e-4
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024
    output_dir: str = "checkpoints/decomposer"

    @property
    def epochs(self) -> int:  # alias used by train_decomposer
        return self.num_epochs


class TrainingConfig(BaseModel):
    seed: int = 42
    output_base_dir: str = "checkpoints"
    logging_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    report_to: str = "none"
    retriever: RetrieverTrainingConfig = RetrieverTrainingConfig()
    reranker: RerankerTrainingConfig = RerankerTrainingConfig()
    generator_sft: GeneratorSFTTrainingConfig = GeneratorSFTTrainingConfig()
    dpo: DPOTrainingConfig = DPOTrainingConfig()
    confidence_head: ConfidenceHeadTrainingConfig = ConfidenceHeadTrainingConfig()
    alpha_network: AlphaNetworkTrainingConfig = AlphaNetworkTrainingConfig()
    decomposer: DecomposerTrainingConfig = DecomposerTrainingConfig()

    # ------------------------------------------------------------------
    # Backward-compat aliases used by older call sites
    # ------------------------------------------------------------------
    @property
    def generator(self) -> GeneratorSFTTrainingConfig:  # alias
        return self.generator_sft

    @property
    def confidence(self) -> ConfidenceHeadTrainingConfig:  # alias
        return self.confidence_head

    @property
    def alpha(self) -> AlphaNetworkTrainingConfig:  # alias
        return self.alpha_network


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_dora: bool = True
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    sse_heartbeat_interval: int = 15
    request_timeout: int = 120
    max_concurrent_requests: int = 10
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_prefix: str = "/api/v1"


class DeviceConfig(BaseModel):
    preferred_device: str = "auto"
    gpu_memory_utilization: float = 0.85


class SyntheticPathsConfig(BaseModel):
    """Canonical on-disk paths for generated training artefacts."""
    qa_path: str = "data/synthetic/qa_pairs.jsonl"
    preferences_path: str = "data/synthetic/preferences.jsonl"
    confidence_labels_path: str = "data/synthetic/confidence_labels.jsonl"
    alpha_labels_path: str = "data/synthetic/alpha_labels.jsonl"
    decomp_labels_path: str = "data/synthetic/decomp_labels.jsonl"
    tool_route_labels_path: str = "data/synthetic/tool_route_labels.jsonl"


class DataConfig(BaseModel):
    vector_db_path: str = "data/vectordb"
    bm25_index_path: str = "data/bm25_index.pkl"
    audit_db_path: str = "data/audit.db"
    raw_docs_dir: str = "data/raw_docs"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    synthetic: SyntheticPathsConfig = SyntheticPathsConfig()
    supported_extensions: List[str] = Field(default_factory=lambda: [
        ".pdf", ".docx", ".txt", ".md", ".csv", ".json",
    ])


class SyntheticDataConfig(BaseModel):
    # Lean local-only targets. Enough for DoRA-rank-8 DPO + cheap heads.
    qa_pairs: int = 800
    preference_triplets: int = 200
    confidence_labels: int = 500
    alpha_labels: int = 500
    decomp_labels: int = 0          # decomposer trained removed; rule-based at runtime
    tool_route_labels: int = 500
    min_bertscore: float = 0.5
    dedup_jaccard_threshold: float = 0.85
    dedup_num_perm: int = 128
    question_types: List[str] = Field(default_factory=lambda: [
        "factoid", "inferential", "multi_hop",
    ])
    rejection_types: List[str] = Field(default_factory=lambda: [
        "hallucinated_citation",
        "no_citation",
        "partial_truncation",
    ])


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: [
        "answer_relevance",
        "faithfulness",
        "context_precision",
        "context_recall",
        "bertscore_f1",
        "latency_p50",
        "latency_p95",
        "confidence_ece",
        "confidence_auroc",
    ])
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    num_bootstrap_samples: int = 1000


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
        "{name}:{function}:{line} - {message}"
    )
    rotation: str = "100 MB"
    retention: str = "30 days"
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk upward from this file to find the directory containing config/base.yaml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / "config" / "base.yaml"
        if candidate.is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _load_yaml_settings() -> Dict[str, Any]:
    """Read config/base.yaml and return as a dict."""
    config_path = PROJECT_ROOT / "config" / "base.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class AegisConfig(BaseSettings):
    """
    Root configuration for the AegisRAG project.

    Priority (highest to lowest):
      1. Environment variables with AEGIS_ prefix (double-underscore for nesting)
      2. config/base.yaml values
      3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Top-level sections
    models: ModelsConfig = ModelsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    cgal: CGALConfig = CGALConfig()
    training: TrainingConfig = TrainingConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    lora: LoRAConfig = LoRAConfig()
    serving: ServingConfig = ServingConfig()
    device: DeviceConfig = DeviceConfig()
    data: DataConfig = DataConfig()
    synthetic_data: SyntheticDataConfig = SyntheticDataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()

    # Convenience top-level field
    project_root: Path = PROJECT_ROOT

    def __init__(self, **kwargs: Any) -> None:
        yaml_values = _load_yaml_settings()
        # YAML values serve as defaults; kwargs (including env vars) override
        merged = {**yaml_values, **kwargs}
        super().__init__(**merged)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return self.project_root / p

    @property
    def checkpoints(self) -> SimpleNamespace:
        """Compat shim: expose training output_dirs under cfg.checkpoints.*."""
        t = self.training
        return SimpleNamespace(
            retriever=t.retriever.output_dir,
            reranker=t.reranker.output_dir,
            generator_sft=t.generator_sft.output_dir,
            generator_dpo=t.dpo.output_dir,
            confidence_head=t.confidence_head.output_dir,
            alpha_network=t.alpha_network.output_dir,
            decomposer=t.decomposer.output_dir,
        )

    @property
    def paths(self) -> SimpleNamespace:
        """Compat shim: expose data paths under cfg.paths.* for legacy callers."""
        d = self.data
        return SimpleNamespace(
            bm25_index=d.bm25_index_path,
            audit_db=d.audit_db_path,
            vector_db=d.vector_db_path,
            raw_docs=d.raw_docs_dir,
            processed=d.processed_dir,
            synthetic=d.synthetic_dir,
        )


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_config_instance: Optional[AegisConfig] = None


def get_config(**overrides: Any) -> AegisConfig:
    """
    Return the singleton AegisConfig instance.
    On first call, loads from YAML + env vars.
    Pass keyword overrides to replace specific values.
    """
    global _config_instance
    if _config_instance is None or overrides:
        _config_instance = AegisConfig(**overrides)
    return _config_instance


def reset_config() -> None:
    """Clear the cached config singleton (useful for tests)."""
    global _config_instance
    _config_instance = None
