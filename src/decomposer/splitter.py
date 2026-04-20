"""
AegisRAG - Query Splitter

Uses the generator LLM with a 3-shot prompt to decompose a multi-part
query into a list of atomic sub-queries.  Falls back to a deterministic
heuristic split when the LLM output cannot be parsed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


SPLITTER_SYSTEM_PROMPT = (
    "You are a query decomposition expert. Given a user query that contains "
    "multiple distinct questions, split it into a JSON array of atomic "
    "sub-queries. Return only the JSON array -- no prose, no markdown. "
    "If the query is already atomic, return a single-element array."
)


# A compact 3-shot prompt -- kept small to minimize context usage.
SPLITTER_FEW_SHOTS: list[dict[str, str]] = [
    {
        "query": (
            "How do I reset my password and what's the refund policy for "
            "annual plans?"
        ),
        "sub_queries": json.dumps(
            [
                "How do I reset my password?",
                "What is the refund policy for annual plans?",
            ]
        ),
    },
    {
        "query": "What are your business hours?",
        "sub_queries": json.dumps(["What are your business hours?"]),
    },
    {
        "query": (
            "Can you explain two-factor authentication, and also tell me how "
            "to cancel my subscription, and finally how do I export my data?"
        ),
        "sub_queries": json.dumps(
            [
                "Can you explain two-factor authentication?",
                "How do I cancel my subscription?",
                "How do I export my data?",
            ]
        ),
    },
]


# Matches the first balanced JSON array in a string.
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")

_HEURISTIC_SEPARATORS = (
    r"\s+and\s+also\s+",
    r"\s+additionally[,]?\s+",
    r"\s+also[,]?\s+",
    r"\s+and\s+",
    r"\?\s+",
)


class QuerySplitter:
    """Split a multi-part query into atomic sub-queries via the generator.

    Parameters
    ----------
    generator : object
        An object exposing ``generate(prompt: str, **kwargs) -> str`` or
        ``generate(query: str, context: list) -> str``.  The splitter
        tries both signatures in order.
    max_sub_queries : int
        Upper bound on the number of sub-queries returned.  Excess entries
        are dropped.  Defaults to 5 which matches typical multi-part
        complexity in customer support.
    """

    def __init__(
        self,
        generator: Any,
        max_sub_queries: int = 5,
    ) -> None:
        if generator is None:
            raise RuntimeError(
                "QuerySplitter requires a generator with a 'generate' method."
            )
        if not hasattr(generator, "generate"):
            raise RuntimeError(
                "Provided generator does not expose a 'generate' method."
            )
        if max_sub_queries < 1:
            raise ValueError("max_sub_queries must be >= 1.")
        self.generator = generator
        self.max_sub_queries = int(max_sub_queries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heuristic_is_multi_part(self, query: str) -> bool:
        """Cheap heuristic for multi-part detection.

        Returns True if the query trips any of our conjunction/separator
        patterns (``and``, ``also``, multiple ``?`` marks, ``additionally``,
        etc.). Used as a fallback when the trained classifier is missing.
        """
        q = (query or "").strip()
        if not q:
            return False
        if q.count("?") > 1:
            return True
        pattern = re.compile("|".join(_HEURISTIC_SEPARATORS), flags=re.IGNORECASE)
        parts = [p for p in pattern.split(q) if p and p.strip()]
        return len(parts) > 1

    def split(self, query: str) -> list[str]:
        """Return a list of atomic sub-queries.

        Parameters
        ----------
        query : str
            The original, possibly multi-part, user query.

        Returns
        -------
        list[str]
            One or more sub-queries.  Always contains at least one element
            (the original query when no decomposition is possible).
        """
        query = query.strip()
        if not query:
            return []

        prompt = self._build_prompt(query)
        llm_text = self._call_generator(prompt)

        if llm_text:
            parsed = self._parse_json_array(llm_text)
            if parsed:
                cleaned = self._clean_sub_queries(parsed)
                if cleaned:
                    logger.debug(
                        "Splitter (LLM) produced %d sub-queries.", len(cleaned)
                    )
                    return cleaned[: self.max_sub_queries]

        logger.info("LLM splitter output unparseable; using heuristic fallback.")
        fallback = self._heuristic_split(query)
        return fallback[: self.max_sub_queries]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str) -> str:
        """Assemble the 3-shot splitter prompt."""
        shots: list[str] = []
        for ex in SPLITTER_FEW_SHOTS:
            shots.append(
                f"User query: {ex['query']}\nSub-queries: {ex['sub_queries']}"
            )
        shot_block = "\n\n".join(shots)
        return (
            f"{SPLITTER_SYSTEM_PROMPT}\n\n"
            f"{shot_block}\n\n"
            f"User query: {query}\nSub-queries:"
        )

    # ------------------------------------------------------------------
    # Generator invocation -- tolerate signature variation
    # ------------------------------------------------------------------

    def _call_generator(self, prompt: str) -> str:
        """Best-effort invocation of ``generator.generate``."""
        try:
            out = self.generator.generate(prompt)
            return str(out) if out is not None else ""
        except TypeError:
            pass
        except Exception as exc:
            logger.warning("Generator.generate(prompt) failed: %s", exc)

        # Alternate signature: generate(query, context)
        try:
            out = self.generator.generate(prompt, [])
            return str(out) if out is not None else ""
        except Exception as exc:
            logger.warning("Generator.generate(prompt, []) failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(text: str) -> list[str] | None:
        """Extract the first JSON array of strings from the LLM output."""
        if not text:
            return None

        # Fast path: the entire string is a JSON array.
        stripped = text.strip()
        try:
            candidate = json.loads(stripped)
            if isinstance(candidate, list) and all(
                isinstance(x, str) for x in candidate
            ):
                return candidate
        except (json.JSONDecodeError, ValueError):
            pass

        # Slow path: find the first bracketed array in the response.
        for match in _JSON_ARRAY_RE.finditer(text):
            snippet = match.group(0)
            try:
                candidate = json.loads(snippet)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, list) and all(
                isinstance(x, str) for x in candidate
            ):
                return candidate
        return None

    @staticmethod
    def _clean_sub_queries(items: list[str]) -> list[str]:
        """Trim and deduplicate sub-query strings, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for raw in items:
            q = str(raw).strip().strip("-*").strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    @classmethod
    def _heuristic_split(cls, query: str) -> list[str]:
        """Split on common multi-part conjunctions when the LLM fails."""
        pattern = re.compile("|".join(_HEURISTIC_SEPARATORS), flags=re.IGNORECASE)
        parts = [p.strip() for p in pattern.split(query) if p and p.strip()]

        # Re-attach the '?' that the splitter may have consumed.
        repaired: list[str] = []
        for idx, p in enumerate(parts):
            if idx < len(parts) - 1 and not p.endswith("?"):
                repaired.append(p.rstrip(".") + "?")
            else:
                repaired.append(p)

        cleaned = cls._clean_sub_queries(repaired)
        if not cleaned:
            return [query.strip()]
        return cleaned
