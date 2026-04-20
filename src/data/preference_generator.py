"""AegisRAG - Rule-based DPO Preference Generator (local, no LLM).

Given a corpus of ``QAPair`` chosen answers, synthesises preference
triplets (chosen > rejected) by applying one of three cheap, deterministic
corruption strategies:

  * ``no_citation``         -- strip the ``[doc:start-end]`` markers
  * ``hallucinated_citation`` -- swap the chunk id in a marker to a wrong id
  * ``partial_truncation``   -- truncate the answer after the first sentence

DPO training consumes only the ``hallucinated_citation`` and ``no_citation``
types (the ones that directly improve grounding); ``partial_truncation``
is written to disk for later experimentation but filtered out by the
trainer.

Target: ``cfg.synthetic_data.preference_triplets`` (default 200).
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import ChunkRecord, PreferenceTriplet, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[[A-Za-z0-9_\-]+:\d+\-\d+\]")

REJECTION_TYPES: tuple[str, ...] = (
    "hallucinated_citation",
    "no_citation",
    "partial_truncation",
)

# Types DPO actually trains on (the trainer filters to this set).
DPO_TRAINING_TYPES: tuple[str, ...] = (
    "hallucinated_citation",
    "no_citation",
)


class PreferenceGenerator:
    """Build preference triplets with three rule-based corruptions."""

    def __init__(
        self,
        seed: int = 42,
        limit: int | None = None,
    ) -> None:
        set_seed(seed)
        self.rng = random.Random(seed)
        self.cfg = get_config()
        self.target_count = (
            int(limit)
            if limit is not None
            else int(getattr(self.cfg.synthetic_data, "preference_triplets", 200))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        chunk_lookup: dict[str, ChunkRecord] | None = None,
        output_path: Path | str | None = None,
    ) -> list[PreferenceTriplet]:
        if not qa_pairs:
            logger.warning("PreferenceGenerator.generate called with no QA pairs")
            return []

        chunk_lookup = chunk_lookup or {}
        chunk_ids = list(chunk_lookup.keys())
        out_path = self._resolve_output(output_path, "preferences.jsonl")

        # Only answered pairs have citations to corrupt/truncate.
        pool = [qa for qa in qa_pairs if qa.question_type != "unanswerable"]
        if not pool:
            logger.warning("No answered pairs available for preference corruption")
            return []

        per_type = max(1, self.target_count // len(REJECTION_TYPES))
        triplets: list[PreferenceTriplet] = []

        for rej_type in REJECTION_TYPES:
            made = 0
            guard = 0
            max_guard = per_type * 6
            while made < per_type and guard < max_guard and len(triplets) < self.target_count:
                guard += 1
                qa = self.rng.choice(pool)
                rejected = self._corrupt(qa, rej_type, chunk_ids)
                if not rejected or rejected.strip() == qa.answer_with_citations.strip():
                    continue
                triplets.append(
                    PreferenceTriplet(
                        query=qa.query,
                        chosen=qa.answer_with_citations,
                        rejected=rejected,
                        rejection_type=rej_type,  # type: ignore[arg-type]
                        context_chunk_ids=list(qa.gold_chunk_ids),
                    )
                )
                made += 1
            logger.info("PreferenceGenerator: %s -> %d triplets", rej_type, made)

        self._write_jsonl(triplets, out_path)
        logger.info("PreferenceGenerator wrote %d triplets to %s", len(triplets), out_path)
        return triplets

    # ------------------------------------------------------------------
    # Corruptions
    # ------------------------------------------------------------------

    def _corrupt(
        self, qa: QAPair, rej_type: str, chunk_ids: Sequence[str]
    ) -> str | None:
        answer = qa.answer_with_citations
        if rej_type == "no_citation":
            stripped = _CITATION_RE.sub("", answer).strip()
            # Add a vague closing phrase so the rejected text is not just
            # the chosen text minus brackets.
            return (stripped + " This information is general knowledge.").strip()
        if rej_type == "hallucinated_citation":
            return self._swap_citation(answer, qa.gold_chunk_ids, chunk_ids)
        if rej_type == "partial_truncation":
            return self._truncate(answer)
        return None

    def _swap_citation(
        self,
        answer: str,
        gold_ids: Sequence[str],
        pool_ids: Sequence[str],
    ) -> str | None:
        matches = list(_CITATION_RE.finditer(answer))
        if not matches:
            return None
        target = matches[self.rng.randrange(len(matches))]
        marker = target.group(0)
        candidates = [cid for cid in pool_ids if cid not in gold_ids]
        fake_id = (
            self.rng.choice(candidates)
            if candidates
            else "fa" + "".join(self.rng.choices("0123456789abcdef", k=14))
        )
        span_portion = marker.split(":", 1)[1].rstrip("]")
        new_marker = f"[{fake_id}:{span_portion}]"
        return answer[: target.start()] + new_marker + answer[target.end() :]

    @staticmethod
    def _truncate(answer: str) -> str:
        m = re.search(r"[.!?](?:\s|$)", answer)
        if m:
            return answer[: m.end()].strip()
        return answer[: max(1, len(answer) // 3)].strip()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_output(self, output_path, default_name):
        if output_path is None:
            base = self.cfg.resolve_path(self.cfg.data.synthetic_dir)
            output_path = Path(base) / default_name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _write_jsonl(items: Iterable[Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
