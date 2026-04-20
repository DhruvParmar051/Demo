#!/usr/bin/env python3
"""
AegisRAG - Unified CLI Entry Point
===================================
Provides subcommands for ingestion, data generation, training, evaluation,
serving, querying, benchmarking, calibration, and decomposition testing.

Usage:
    python run.py ingest --source-dir data/raw_docs
    python run.py generate-data --type all --output-dir data/synthetic
    python run.py train --component all
    python run.py evaluate --models b1,m1 --test-dir data/test --output-dir report
    python run.py serve --host 0.0.0.0 --port 8000 --model m5
    python run.py query --model m5 --query "How do I reset my password?" --stream
    python run.py benchmark --model m5 --n 100
    python run.py calibrate --model m5
    python run.py test-decomp --n 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure loguru with file rotation and console output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    )
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path / "aegisrag_{time:YYYY-MM-DD}.log"),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/base.yaml") -> dict:
    """Load and return the YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace, cfg: dict) -> None:
    """Ingest documents into the vector store and BM25 index."""
    from src.data.ingest import run_ingestion

    source_dir = args.source_dir or cfg["data"]["raw_docs_dir"]
    vector_db_path = args.vector_db_path or cfg["data"]["vector_db_path"]
    bm25_index_path = cfg["data"]["bm25_index_path"]

    logger.info(f"Ingesting documents from {source_dir}")
    run_ingestion(
        source_dir=source_dir,
        vector_db_path=vector_db_path,
        bm25_index_path=bm25_index_path,
        chunk_size=cfg["retrieval"]["chunk_size"],
        chunk_overlap=cfg["retrieval"]["chunk_overlap"],
        min_chunk_size=cfg["retrieval"]["min_chunk_size"],
        dedup_threshold=cfg["retrieval"]["dedup_threshold"],
        embedding_model=cfg["models"]["retriever"]["name"],
        supported_extensions=cfg["data"]["supported_extensions"],
    )
    logger.info("Ingestion complete.")


def cmd_generate_data(args: argparse.Namespace, cfg: dict) -> None:
    """Generate synthetic training data."""
    from src.data.generate import run_data_generation

    data_type = args.type
    output_dir = args.output_dir or cfg["data"]["synthetic_dir"]

    valid_types = {"qa", "preference", "confidence", "alpha", "decomp", "all"}
    if data_type not in valid_types:
        logger.error(f"Invalid data type '{data_type}'. Must be one of: {valid_types}")
        sys.exit(1)

    logger.info(f"Generating synthetic data: type={data_type}, output={output_dir}")
    run_data_generation(
        data_type=data_type,
        output_dir=output_dir,
        config=cfg,
    )
    logger.info("Data generation complete.")


def cmd_train(args: argparse.Namespace, cfg: dict) -> None:
    """Train a component or all components sequentially."""
    from src.training.train import run_training

    component = args.component
    valid_components = {
        "retriever", "reranker", "generator", "dpo",
        "confidence", "alpha", "decomposer", "all",
    }
    if component not in valid_components:
        logger.error(f"Invalid component '{component}'. Must be one of: {valid_components}")
        sys.exit(1)

    logger.info(f"Training component: {component}")
    run_training(component=component, config=cfg)
    logger.info(f"Training complete for: {component}")


def cmd_evaluate(args: argparse.Namespace, cfg: dict) -> None:
    """Evaluate one or more model configurations."""
    from src.evaluation.evaluate import run_evaluation

    models = [m.strip() for m in args.models.split(",")]
    test_dir = args.test_dir or "data/test"
    output_dir = args.output_dir or "report"

    logger.info(f"Evaluating models: {models}")
    run_evaluation(
        models=models,
        test_dir=test_dir,
        output_dir=output_dir,
        config=cfg,
    )
    logger.info("Evaluation complete.")


def cmd_serve(args: argparse.Namespace, cfg: dict) -> None:
    """Launch the FastAPI server."""
    from src.serving.app import create_app

    host = args.host or cfg["serving"]["host"]
    port = args.port or cfg["serving"]["port"]
    model_tag = args.model

    logger.info(f"Starting server on {host}:{port} with model config '{model_tag}'")

    app = create_app(model_tag=model_tag, config=cfg)

    import uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=cfg["serving"]["workers"],
        timeout_keep_alive=cfg["serving"]["request_timeout"],
        log_level=cfg["logging"]["level"].lower(),
    )


def cmd_query(args: argparse.Namespace, cfg: dict) -> None:
    """Send a single query to a model configuration."""
    from src.serving.client import query_model

    model_tag = args.model
    query_text = args.query
    stream = args.stream

    if not query_text:
        logger.error("--query is required.")
        sys.exit(1)

    logger.info(f"Querying model '{model_tag}': {query_text[:80]}...")
    result = query_model(
        model_tag=model_tag,
        query=query_text,
        stream=stream,
        config=cfg,
    )

    if stream:
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources: {result.get('sources', [])}")
        if result.get("sub_queries"):
            print(f"Decomposed into: {result['sub_queries']}")


def cmd_benchmark(args: argparse.Namespace, cfg: dict) -> None:
    """Run a latency and throughput benchmark."""
    from src.evaluation.benchmark import run_benchmark

    model_tag = args.model
    n = args.n

    logger.info(f"Benchmarking model '{model_tag}' with {n} queries")
    results = run_benchmark(model_tag=model_tag, n=n, config=cfg)

    print("\n--- Benchmark Results ---")
    print(f"  Model:          {model_tag}")
    print(f"  Queries:        {n}")
    print(f"  Avg latency:    {results['avg_latency_ms']:.1f} ms")
    print(f"  P50 latency:    {results['p50_latency_ms']:.1f} ms")
    print(f"  P95 latency:    {results['p95_latency_ms']:.1f} ms")
    print(f"  P99 latency:    {results['p99_latency_ms']:.1f} ms")
    print(f"  Throughput:     {results['throughput_qps']:.2f} queries/sec")
    print(f"  Avg confidence: {results['avg_confidence']:.3f}")


def cmd_calibrate(args: argparse.Namespace, cfg: dict) -> None:
    """Run confidence calibration (temperature scaling) on the confidence head."""
    from src.training.calibrate import run_calibration

    model_tag = args.model

    logger.info(f"Calibrating confidence head for model '{model_tag}'")
    result = run_calibration(model_tag=model_tag, config=cfg)

    print("\n--- Calibration Results ---")
    print(f"  Model:               {model_tag}")
    print(f"  ECE before:          {result['ece_before']:.4f}")
    print(f"  ECE after:           {result['ece_after']:.4f}")
    print(f"  Temperature:         {result['temperature']:.4f}")
    print(f"  AUROC:               {result['auroc']:.4f}")


def cmd_test_decomp(args: argparse.Namespace, cfg: dict) -> None:
    """Test query decomposition on sample multi-part queries."""
    from src.decomposer.test_decomp import run_decomposition_test

    n = args.n

    logger.info(f"Testing query decomposition with {n} samples")
    results = run_decomposition_test(n=n, config=cfg)

    print(f"\n--- Decomposition Test ({n} samples) ---")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Original: {r['original']}")
        print(f"    Sub-queries ({len(r['sub_queries'])}):")
        for j, sq in enumerate(r["sub_queries"], 1):
            print(f"      {j}. {sq}")
        print(f"    Is multi-part: {r['is_multi_part']}")
        print(f"    Decomposition time: {r['time_ms']:.1f} ms")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="aegisrag",
        description="AegisRAG - Grounded Customer-Support RAG Copilot",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/base.yaml",
        help="Path to base YAML config (default: config/base.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into vector DB and BM25 index")
    p_ingest.add_argument("--source-dir", type=str, default=None, help="Directory with source documents")
    p_ingest.add_argument("--vector-db-path", type=str, default=None, help="Path to ChromaDB directory")

    # --- generate-data ---
    p_gendata = subparsers.add_parser("generate-data", help="Generate synthetic training data")
    p_gendata.add_argument(
        "--type", type=str, required=True,
        choices=["qa", "preference", "confidence", "alpha", "decomp", "all"],
        help="Type of data to generate",
    )
    p_gendata.add_argument("--output-dir", type=str, default=None, help="Output directory for generated data")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a model component")
    p_train.add_argument(
        "--component", type=str, required=True,
        choices=["retriever", "reranker", "generator", "dpo", "confidence", "alpha", "decomposer", "all"],
        help="Component to train",
    )

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model configurations")
    p_eval.add_argument("--models", type=str, required=True, help="Comma-separated model tags (e.g., b1,m1,m5)")
    p_eval.add_argument("--test-dir", type=str, default=None, help="Directory with test data")
    p_eval.add_argument("--output-dir", type=str, default=None, help="Directory for evaluation reports")

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI server")
    p_serve.add_argument("--host", type=str, default=None, help="Server host")
    p_serve.add_argument("--port", type=int, default=None, help="Server port")
    p_serve.add_argument(
        "--model", type=str, required=True,
        choices=["b1", "b2", "b3", "m1", "m2", "m3", "m4", "m5"],
        help="Model configuration tag",
    )

    # --- query ---
    p_query = subparsers.add_parser("query", help="Send a single query")
    p_query.add_argument("--model", type=str, required=True, help="Model configuration tag")
    p_query.add_argument("--query", type=str, required=True, help="Query text")
    p_query.add_argument("--stream", action="store_true", help="Enable SSE streaming output")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Run latency benchmark")
    p_bench.add_argument("--model", type=str, required=True, help="Model configuration tag")
    p_bench.add_argument("--n", type=int, default=100, help="Number of queries to benchmark")

    # --- calibrate ---
    p_calib = subparsers.add_parser("calibrate", help="Calibrate confidence head")
    p_calib.add_argument("--model", type=str, required=True, help="Model configuration tag")

    # --- test-decomp ---
    p_decomp = subparsers.add_parser("test-decomp", help="Test query decomposition")
    p_decomp.add_argument("--n", type=int, default=20, help="Number of test samples")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_level = args.log_level or os.getenv("AEGIS_LOG_LEVEL") or cfg.get("logging", {}).get("level", "INFO")
    log_dir = os.getenv("AEGIS_LOG_DIR") or cfg.get("logging", {}).get("log_dir", "logs")
    setup_logging(level=log_level, log_dir=log_dir)

    logger.info(f"AegisRAG CLI - command: {args.command}")

    dispatch = {
        "ingest": cmd_ingest,
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
        "query": cmd_query,
        "benchmark": cmd_benchmark,
        "calibrate": cmd_calibrate,
        "test-decomp": cmd_test_decomp,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args, cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Command '{args.command}' failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
