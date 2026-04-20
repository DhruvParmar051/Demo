"""
AegisRAG - Evaluation subpackage.

Exports the main metric functions, the FCRS composite score, calibration
utilities, the Evaluator orchestrator, and the report generator.
"""

from __future__ import annotations

from src.evaluation.calibration import (
    compute_ece,
    reliability_diagram,
    temperature_scaling,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.fcrs import compute_fcrs
from src.evaluation.metrics import (
    answer_quality,
    cgal_efficiency,
    citation_f1,
    consistency,
    decomposition_accuracy,
    escalation_f1,
    grounding_score,
    tool_accuracy,
)
from src.evaluation.report import generate_report

__all__ = [
    "Evaluator",
    "answer_quality",
    "cgal_efficiency",
    "citation_f1",
    "compute_ece",
    "compute_fcrs",
    "consistency",
    "decomposition_accuracy",
    "escalation_f1",
    "generate_report",
    "grounding_score",
    "reliability_diagram",
    "temperature_scaling",
    "tool_accuracy",
]
