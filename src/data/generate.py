"""Facade: ``run_data_generation`` for the CLI.

Delegates to the orchestrator in :mod:`scripts.generate_data`, which
runs each of the five generator classes in dependency order (QA ->
preference/confidence/alpha/decomp). Any downstream failure is logged
but does not abort the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_data_generation(
    data_type: str = "all",
    output_dir: str | Path = "data/synthetic",
    config: Any | None = None,  # noqa: ARG001 -- kept for CLI symmetry
) -> dict[str, dict]:
    """Generate one or all synthetic dataset types.

    Parameters
    ----------
    data_type : str
        One of ``qa``, ``preference``, ``confidence``, ``alpha``,
        ``decomp``, or ``all``.
    output_dir : str or Path
        Where JSONL files are written.
    config : object, optional
        Unused; accepted for call-site symmetry with other CLI runners.

    Returns
    -------
    dict
        ``{dataset_type: {"status": ..., "count": ..., "output": ...}}``
    """
    from scripts.generate_data import run as _run  # lazy import

    return _run(type_=data_type, output_dir=output_dir)
