"""
AegisRAG - Tool JSON Schemas.

Declarative JSON-Schema definitions for the four CGAL tools plus helpers
for argument validation. Kept separate from the executor so the schemas
can be shipped to a UI, embedded in prompts, or used for typed training
data generation without pulling in any heavy runtime dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: SearchKB
# ---------------------------------------------------------------------------

SEARCH_KB_SCHEMA: Dict[str, Any] = {
    "name": "SearchKB",
    "description": (
        "Run hybrid dense+sparse retrieval plus reranking against the "
        "knowledge base. Used when initial retrieval confidence is "
        "medium and a refined sub-query is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Refined sub-query to search for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            "filters": {
                "type": "object",
                "description": "Optional metadata filters (e.g. domain).",
                "properties": {
                    "domain": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "output": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "chunk_id": {"type": "string"},
                "span_start": {"type": "integer"},
                "span_end": {"type": "integer"},
                "text": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["doc_id", "chunk_id", "text", "score"],
        },
    },
    "latency_ms": 80,
}


# ---------------------------------------------------------------------------
# Tool 2: GetPolicy
# ---------------------------------------------------------------------------

GET_POLICY_SCHEMA: Dict[str, Any] = {
    "name": "GetPolicy",
    "description": (
        "Look up a specific policy section by identifier. Supports "
        "fuzzy substring matching when exact lookup fails."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": "Section id or policy name (e.g. 'refund-policy-3.2').",
            },
            "fuzzy": {
                "type": "boolean",
                "description": "Fall back to substring match if exact id is missing.",
                "default": True,
            },
        },
        "required": ["section_id"],
        "additionalProperties": False,
    },
    "output": {
        "type": "object",
        "properties": {
            "section_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "doc_id": {"type": "string"},
            "span_start": {"type": "integer"},
            "span_end": {"type": "integer"},
        },
    },
    "latency_ms": 30,
}


# ---------------------------------------------------------------------------
# Tool 3: CreateTicket
# ---------------------------------------------------------------------------

_TICKET_CATEGORIES = ["billing", "technical", "account", "policy", "other"]
_TICKET_SEVERITIES = ["low", "medium", "high", "critical"]

CREATE_TICKET_SCHEMA: Dict[str, Any] = {
    "name": "CreateTicket",
    "description": (
        "Escalate to a human agent by creating a support ticket. Used "
        "when confidence is very low or CGAL iterations are exhausted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Original user query.",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of the user's issue.",
            },
            "category": {
                "type": "string",
                "enum": _TICKET_CATEGORIES,
            },
            "severity": {
                "type": "string",
                "enum": _TICKET_SEVERITIES,
            },
            "user_context": {
                "type": "string",
                "description": "Relevant prior conversation or account state.",
            },
            "evidence_gap": {
                "type": "string",
                "description": "Why the KB could not resolve the query.",
            },
        },
        "required": ["category", "severity"],
        "additionalProperties": False,
    },
    "output": {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "estimated_response_time": {"type": "string"},
        },
    },
    "latency_ms": 10,
}


# ---------------------------------------------------------------------------
# Tool 4: AnswerVerify (novel, confidence-gated)
# ---------------------------------------------------------------------------

_VERIFY_VERDICTS = ["pass", "partial", "fail", "skipped"]

ANSWER_VERIFY_SCHEMA: Dict[str, Any] = {
    "name": "AnswerVerify",
    "description": (
        "Confidence-gated NLI verification of a generated answer against "
        "its cited evidence spans. Only runs when confidence is in the "
        "uncertainty band [0.75, 0.85]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The generated answer text.",
            },
            "cited_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "span_start": {"type": "integer"},
                        "span_end": {"type": "integer"},
                        "cited_text": {"type": "string"},
                    },
                    "required": ["doc_id", "cited_text"],
                },
            },
            "confidence": {
                "type": "number",
                "description": "Confidence head score for gating.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["answer", "cited_spans"],
        "additionalProperties": False,
    },
    "output": {
        "type": "object",
        "properties": {
            "grounding_score": {"type": ["number", "null"]},
            "ungrounded_claims": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": _VERIFY_VERDICTS},
        },
    },
    "latency_ms": 200,
}


# ---------------------------------------------------------------------------
# Aggregated registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "SearchKB": SEARCH_KB_SCHEMA,
    "GetPolicy": GET_POLICY_SCHEMA,
    "CreateTicket": CREATE_TICKET_SCHEMA,
    "AnswerVerify": ANSWER_VERIFY_SCHEMA,
}

TOOL_NAMES: List[str] = list(TOOL_SCHEMAS.keys())


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _check_type(value: Any, expected: Any) -> bool:
    """Check a value against a JSON-Schema ``type`` declaration."""
    if isinstance(expected, list):
        return any(_check_type(value, t) for t in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def validate_tool_args(
    tool_name: str, args: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate ``args`` against the schema for ``tool_name``.

    Parameters
    ----------
    tool_name:
        One of :data:`TOOL_NAMES`.
    args:
        Argument dict supplied by the model or caller.

    Returns
    -------
    tuple(bool, list[str])
        ``(is_valid, errors)``. ``errors`` is a list of human-readable
        problem strings (empty when valid).
    """
    errors: List[str] = []

    if tool_name not in TOOL_SCHEMAS:
        return False, [f"Unknown tool '{tool_name}'."]

    schema = TOOL_SCHEMAS[tool_name]
    params = schema.get("parameters", {})
    properties: Dict[str, Any] = params.get("properties", {})
    required: List[str] = list(params.get("required", []))
    additional_allowed = params.get("additionalProperties", True)

    if not isinstance(args, dict):
        return False, [f"Arguments for {tool_name} must be a dict."]

    # Required fields
    for field_name in required:
        if field_name not in args:
            errors.append(f"Missing required field '{field_name}'.")

    # Unknown fields
    if additional_allowed is False:
        for field_name in args:
            if field_name not in properties:
                errors.append(f"Unknown field '{field_name}'.")

    # Type + enum checks
    for field_name, value in args.items():
        if field_name not in properties:
            continue
        spec = properties[field_name]
        expected_type = spec.get("type")
        if expected_type is not None and not _check_type(value, expected_type):
            errors.append(
                f"Field '{field_name}' expected type {expected_type}, "
                f"got {type(value).__name__}."
            )
            continue
        if "enum" in spec and value not in spec["enum"]:
            errors.append(
                f"Field '{field_name}' must be one of {spec['enum']}, "
                f"got {value!r}."
            )
        if expected_type == "integer" and _check_type(value, "integer"):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(
                    f"Field '{field_name}' below minimum {spec['minimum']}."
                )
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(
                    f"Field '{field_name}' above maximum {spec['maximum']}."
                )

    is_valid = len(errors) == 0
    if not is_valid:
        logger.debug("Tool arg validation failed for %s: %s", tool_name, errors)
    return is_valid, errors


__all__ = [
    "SEARCH_KB_SCHEMA",
    "GET_POLICY_SCHEMA",
    "CREATE_TICKET_SCHEMA",
    "ANSWER_VERIFY_SCHEMA",
    "TOOL_SCHEMAS",
    "TOOL_NAMES",
    "validate_tool_args",
]
