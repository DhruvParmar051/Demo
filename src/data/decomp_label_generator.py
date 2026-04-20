"""
AegisRAG - Decomposition Label Generator

Produces training labels for the multi-part query splitter. A heuristic
flags likely multi-part queries (via coordinating conjunctions and
question-mark counts). Multi-part queries are then sent to a teacher
LLM which returns the atomic sub-queries.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import DecompLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


_CONJUNCTION_RE = re.compile(
    r"\b(and|also|additionally|plus|furthermore|as well as|besides)\b",
    re.IGNORECASE,
)


class DecompLabelGenerator:
    """Generate :class:`DecompLabel` records (250 multi-part + 250 single).

    Parameters
    ----------
    teacher : object or None
        An object with ``generate(prompt: str) -> str`` used to split
        multi-part queries. Falls back to a simple heuristic split on
        conjunctions/question marks when the teacher is unavailable.
    seed : int
        Deterministic seed (default 42).
    limit : int or None
        Total label count (default 500, half multi-part, half single).
    """

    def __init__(
        self,
        teacher: Any | None = None,
        seed: int = 42,
        limit: int | None = None,
    ) -> None:
        set_seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)

        cfg = get_config()
        self.cfg = cfg
        self._teacher = teacher

        total = int(limit) if limit is not None else 500
        self.per_class = max(1, total // 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        output_path: Path | str | None = None,
    ) -> list[DecompLabel]:
        if not qa_pairs:
            logger.warning("DecompLabelGenerator.generate called with no QA pairs")
            return []

        output_path = self._resolve_output(output_path, "decomp_labels.jsonl")

        multi: list[QAPair] = []
        single: list[QAPair] = []
        pool = list(qa_pairs)
        self.rng.shuffle(pool)

        for qa in pool:
            if self._looks_multi_part(qa.query):
                if len(multi) < self.per_class:
                    multi.append(qa)
            else:
                if len(single) < self.per_class:
                    single.append(qa)
            if len(multi) >= self.per_class and len(single) >= self.per_class:
                break

        # Guarantee we have ``per_class`` of each: top up from full pool.
        if len(multi) < self.per_class:
            logger.info(
                "Only %d multi-part candidates found heuristically; "
                "padding by sampling",
                len(multi),
            )
            extras = [q for q in pool if q not in multi]
            needed = self.per_class - len(multi)
            multi.extend(extras[:needed])

        if len(single) < self.per_class:
            extras = [q for q in pool if q not in single and q not in multi]
            needed = self.per_class - len(single)
            single.extend(extras[:needed])

        labels: list[DecompLabel] = []

        for qa in multi:
            sub_qs = self._split_with_teacher(qa.query)
            if not sub_qs or len(sub_qs) < 2:
                sub_qs = self._heuristic_split(qa.query)
            labels.append(
                DecompLabel(
                    query=qa.query,
                    is_multi_part=True,
                    sub_queries=sub_qs,
                )
            )

        for qa in single:
            labels.append(
                DecompLabel(
                    query=qa.query,
                    is_multi_part=False,
                    sub_queries=[qa.query],
                )
            )

        self.rng.shuffle(labels)
        self._write_jsonl(labels, output_path)
        logger.info(
            "DecompLabelGenerator wrote %d labels (%d multi, %d single) to %s",
            len(labels),
            len(multi),
            len(single),
            output_path,
        )
        return labels

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_multi_part(query: str) -> bool:
        """Heuristic: conjunctions present OR >=2 question marks."""
        q_marks = query.count("?")
        if q_marks >= 2:
            return True
        if _CONJUNCTION_RE.search(query):
            return True
        return False

    @staticmethod
    def _heuristic_split(query: str) -> list[str]:
        """Split a query on `?` or coordinating conjunctions."""
        # Split on '?' while keeping the mark with the preceding segment.
        segments: list[str] = []
        if "?" in query:
            parts = re.split(r"(?<=\?)\s+", query)
            segments = [p.strip() for p in parts if p.strip()]
        else:
            parts = _CONJUNCTION_RE.split(query)
            # re.split with capture keeps delimiters; drop them.
            segments = [p.strip() for p in parts if p and not _CONJUNCTION_RE.fullmatch(p.strip())]

        # Clean up; ensure at least 2 segments.
        segments = [s for s in segments if len(s) > 3]
        if len(segments) < 2:
            return [query]
        return segments

    # ------------------------------------------------------------------
    # Teacher-based split
    # ------------------------------------------------------------------

    def _split_with_teacher(self, query: str) -> list[str]:
        teacher = self._get_teacher()
        if teacher is None:
            return self._heuristic_split(query)

        prompt = (
            "You are a query-decomposition assistant. Split the user query "
            "into the smallest set of independent atomic sub-queries whose "
            "answers together fully answer the original.\n"
            "Output JSON: {\"sub_queries\": [\"...\", \"...\"]}\n\n"
            f"QUERY: {query}\n"
        )
        try:
            raw = teacher.generate(prompt)
        except Exception as exc:
            logger.warning("Teacher split failed: %s", exc)
            return self._heuristic_split(query)

        sub_qs = self._parse_sub_queries(raw)
        if not sub_qs:
            return self._heuristic_split(query)
        return sub_qs

    @staticmethod
    def _parse_sub_queries(raw: str) -> list[str]:
        if not raw:
            return []
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        subs = obj.get("sub_queries") or []
        return [str(s).strip() for s in subs if str(s).strip()]

    def _get_teacher(self) -> Any | None:
        if self._teacher is not None:
            return self._teacher
        try:
            from src.models.generator import Generator  # lazy import
        except ImportError:
            logger.info(
                "src.models.generator.Generator not available; "
                "using heuristic split only"
            )
            return None
        try:
            self._teacher = Generator()
        except Exception as exc:
            logger.warning("Could not instantiate teacher generator: %s", exc)
            self._teacher = None
        return self._teacher

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_output(
        self, output_path: Path | str | None, default_name: str
    ) -> Path:
        if output_path is None:
            base = self.cfg.resolve_path(self.cfg.data.synthetic_dir)
            output_path = base / default_name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _write_jsonl(items: Iterable[Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
