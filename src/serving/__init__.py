"""AegisRAG - Serving package.

Submodules are imported lazily so that lightweight utilities like
``format_sse`` do not pull in torch / FastAPI at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "AuditLogger", "format_sse"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy imports
    if name == "create_app":
        from src.serving.app import create_app as _create_app
        return _create_app
    if name == "AuditLogger":
        from src.serving.audit_logger import AuditLogger as _AL
        return _AL
    if name == "format_sse":
        from src.serving.sse import format_sse as _fs
        return _fs
    raise AttributeError(name)
