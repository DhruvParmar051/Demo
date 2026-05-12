"""AegisRAG - Rule-based DPO Preference Generator.

Given a corpus of ``QAPair`` chosen answers, synthesises preference
triplets (chosen > rejected) by applying one of six deterministic
corruption strategies:

  * ``no_citation``           -- strip the ``[doc:start-end]`` markers
  * ``hallucinated_citation`` -- swap the chunk id in a marker to a wrong id
  * ``partial_truncation``    -- truncate the answer after the first sentence
  * ``verbose_unfaithful``    -- pad answer with plausible-sounding but
                                 unsupported claims
  * ``wrong_tool``            -- insert a fake tool-call directive that
                                 contradicts the KB answer
  * ``unsafe_tone``           -- reframe the answer with dismissive or
                                 unhelpful phrasing

All six types are consumed during DPO training (see DPO_TRAINING_TYPES).

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
    "verbose_unfaithful",
    "wrong_tool",
    "unsafe_tone",
)

# Types DPO trains on — all 6 rejection types for full preference coverage.
DPO_TRAINING_TYPES: tuple[str, ...] = (
    "hallucinated_citation",
    "no_citation",
    "partial_truncation",
    "verbose_unfaithful",
    "wrong_tool",
    "unsafe_tone",
)

# Filler phrases for the verbose_unfaithful corruption (plausible but unsupported).
_VERBOSE_FILLERS = [
    " Additionally, some users have reported that this also depends on their account tier and region settings.",
    " It's worth noting that in certain cases this policy may vary based on subscription level and local regulations.",
    " Furthermore, our support team has noted that exceptions may apply depending on the payment method used.",
    " In many cases, customers have found that contacting support directly can yield faster resolution times.",
    " Note that processing times can vary significantly during peak hours and holiday periods.",
]

# Templates for the wrong_tool corruption (wrong escalation directive).
_WRONG_TOOL_TEMPLATES = [
    "Please escalate this to a human agent immediately — I cannot answer this from the knowledge base.",
    "You should contact the billing department directly; I don't have access to account-specific information.",
    "This requires manual review. Please submit a support ticket at your earliest convenience.",
    "I'll need to search external documentation for this; the internal KB does not cover this topic.",
]

# Prefixes for the unsafe_tone corruption (dismissive / condescending phrasing).
_UNSAFE_TONE_PREFIXES = [
    "That's a rather obvious question, but I'll answer anyway: ",
    "As anyone would know, ",
    "This is a basic FAQ item. Simply put: ",
    "I really shouldn't need to explain this, but: ",
]


class PreferenceGenerator:
    """Build preference triplets with six rule-based corruptions.

    Each corruption strategy degrades the chosen answer along a specific
    quality dimension (grounding, citation, tone, completeness, tool use),
    producing the full set of six rejection types defined for DPO training.
    """

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
            return (stripped + " This information is general knowledge.").strip()

        if rej_type == "hallucinated_citation":
            return self._swap_citation(answer, qa.gold_chunk_ids, chunk_ids)

        if rej_type == "partial_truncation":
            return self._truncate(answer)

        if rej_type == "verbose_unfaithful":
            return self._make_verbose_unfaithful(answer)

        if rej_type == "wrong_tool":
            return self._make_wrong_tool()

        if rej_type == "unsafe_tone":
            return self._make_unsafe_tone(answer)

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

    def _make_verbose_unfaithful(self, answer: str) -> str:
        """Append a plausible-sounding but unsupported filler claim."""
        filler = self.rng.choice(_VERBOSE_FILLERS)
        base = _CITATION_RE.sub("", answer).strip()
        return (base + filler).strip()

    def _make_wrong_tool(self) -> str:
        """Replace the KB answer with an incorrect escalation directive."""
        return self.rng.choice(_WRONG_TOOL_TEMPLATES)

    def _make_unsafe_tone(self, answer: str) -> str:
        """Prefix the answer with a dismissive / condescending opener."""
        prefix = self.rng.choice(_UNSAFE_TONE_PREFIXES)
        base = _CITATION_RE.sub("", answer).strip()
        if base:
            base = base[0].lower() + base[1:]
        return (prefix + base).strip()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_output(self, output_path: Path | str | None, default_name: str) -> Path:
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