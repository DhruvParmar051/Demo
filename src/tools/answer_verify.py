"""
AegisRAG - AnswerVerify Tool

Post-generation NLI-based verification. For each factual sentence in the
answer, locate its cited evidence span and run an NLI cross-encoder to
decide whether the sentence is entailed, contradicted, or neutral.

Gated by confidence: when the upstream confidence score is >= 0.85 the
verifier is skipped entirely to conserve the latency budget.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from src.data.schema import Citation
from src.utils.config import get_config

logger = logging.getLogger(__name__)


_CITATION_RE = re.compile(r"\[([a-f0-9]{6,32}):(\d+)-(\d+)\]")


class AnswerVerify:
    """NLI-based answer verification tool.

    Parameters
    ----------
    model_name : str
        Cross-encoder NLI model identifier.
    entail_threshold : float
        Entailment probability threshold for 'grounded' verdict.
    pass_threshold : float
        Final grounding score >= this -> 'pass'.
    partial_threshold : float
        Final grounding score >= this -> 'partial', else 'fail'.
    skip_confidence : float
        If upstream confidence >= this value, verification is skipped.
    """

    def __init__(
        self,
        model_name: str | None = None,
        entail_threshold: float = 0.7,
        pass_threshold: float = 0.80,
        partial_threshold: float = 0.50,
        skip_confidence: float = 0.85,
    ) -> None:
        cfg = get_config()
        self.model_name = model_name or cfg.models.nli.name
        self.entail_threshold = entail_threshold
        self.pass_threshold = pass_threshold
        self.partial_threshold = partial_threshold
        self.skip_confidence = skip_confidence
        self._nli = None  # lazy
        self._nlp = None  # lazy spaCy

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_nli(self) -> Any:
        if self._nli is not None:
            return self._nli
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for AnswerVerify. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading NLI cross-encoder: %s", self.model_name)
        self._nli = CrossEncoder(self.model_name)
        return self._nli

    def _load_spacy(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is required for AnswerVerify. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc
        try:
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            # Fallback: blank English with sentencizer only.
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe("sentencizer")
        return self._nlp
    
    def warmup(self) -> None:
        """Pre-load models to avoid cold-start latency."""
        try:
            self._load_spacy()
            nli = self._load_nli()

            # Small warmup inference (important for lazy init inside model)
            nli.predict([("warm up premise", "warm up hypothesis")])

            logger.info("AnswerVerify warmup complete")
        except Exception as exc:
            logger.warning("AnswerVerify warmup failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: str,
        cited_spans: list[Citation],
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Run NLI verification and return a verdict dict.

        Returns
        -------
        dict with keys ``verdict``, ``grounding_score``, ``n_sentences``,
        ``n_grounded``, ``n_ungrounded``, ``ungrounded_claims``.
        """
        if confidence is not None and confidence >= self.skip_confidence:
            return {
                "verdict": "skipped",
                "grounding_score": None,
                "reason": "high_confidence",
            }

        if not cited_spans or not answer.strip():
            return {
                "verdict": "fail",
                "grounding_score": 0.0,
                "reason": "no_citations_or_empty",
                "n_sentences": 0,
                "n_grounded": 0,
                "n_ungrounded": 0,
                "ungrounded_claims": [],
            }

        sentences = self._split_sentences(answer)
        if not sentences:
            return {
                "verdict": "fail",
                "grounding_score": 0.0,
                "reason": "no_sentences",
            }

        # Build a doc_id -> cited_text map for premise lookup.
        cite_map: dict[str, str] = {c.doc_id: c.cited_text for c in cited_spans}

        n_grounded = 0
        ungrounded: list[dict[str, str]] = []

        try:
            nli = self._load_nli()
        except RuntimeError as exc:
            logger.warning("NLI unavailable, verification falling back: %s", exc)
            return {
                "verdict": "skipped",
                "grounding_score": None,
                "reason": "nli_unavailable",
            }

        for sent in sentences:
            premise = self._pick_premise(sent, cite_map, cited_spans)
            if premise is None:
                ungrounded.append({"sentence": sent, "reason": "no_citation"})
                continue

            # CrossEncoder for NLI predicts 3-class probs (contradict/neutral/entail)
            try:
                scores = nli.predict([(premise, sent)])
            except Exception as exc:
                logger.warning("NLI predict failed: %s", exc)
                ungrounded.append({"sentence": sent, "reason": "nli_error"})
                continue

            row = scores[0]
            # Normalize to softmax if logits returned.
            try:
                import numpy as _np
                if row.ndim == 0 or row.shape[0] != 3:
                    # Model returned a single score (entail prob-ish) -- use threshold.
                    entail_p = float(row)
                else:
                    logits = _np.asarray(row, dtype=float)
                    ex = _np.exp(logits - logits.max())
                    probs = ex / ex.sum()
                    # Index mapping depends on model; most NLI models use
                    # [contradiction, entailment, neutral] or [c, e, n].
                    # Here we take the max index and treat index 1 as entailment
                    # by convention for DeBERTa/MiniLM.
                    entail_p = float(probs[1])
            except Exception:
                entail_p = float(row)

            if entail_p >= self.entail_threshold:
                n_grounded += 1
            else:
                ungrounded.append({"sentence": sent, "reason": f"entail_p={entail_p:.2f}"})

        n_total = len(sentences)
        grounding_score = n_grounded / n_total if n_total else 0.0
        if grounding_score >= self.pass_threshold:
            verdict = "pass"
        elif grounding_score >= self.partial_threshold:
            verdict = "partial"
        else:
            verdict = "fail"

        # Mark citations on sentences that found a premise (best-effort).
        for c in cited_spans:
            c.verified = (grounding_score >= self.partial_threshold)

        return {
            "verdict": verdict,
            "grounding_score": grounding_score,
            "n_sentences": n_total,
            "n_grounded": n_grounded,
            "n_ungrounded": len(ungrounded),
            "ungrounded_claims": ungrounded,
        }

    async def verify_async(
        self,
        answer: str,
        cited_spans: list[Citation],
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Async wrapper so FastAPI handlers don't block the event loop."""
        return await asyncio.to_thread(
            self.verify, answer, cited_spans, confidence
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, answer: str) -> list[str]:
        """Split answer into sentences, stripping citation markers."""
        try:
            nlp = self._load_spacy()
            doc = nlp(answer)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            # Simple regex fallback.
            sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", answer)
                if s.strip()
            ]
        cleaned: list[str] = []
        for s in sents:
            clean = _CITATION_RE.sub("", s).strip()
            if clean:
                cleaned.append(clean)
        return cleaned

    def _pick_premise(
        self,
        sentence: str,
        cite_map: dict[str, str],
        cited_spans: list[Citation],
    ) -> str | None:
        """Find a cited premise for a sentence.

        First tries to match a citation marker `[doc_id:start-end]` in the
        original sentence text. Falls back to concatenating all cited_text
        when the sentence has no marker but some citations exist.
        """
        matches = _CITATION_RE.findall(sentence)
        for doc_id, _s, _e in matches:
            if doc_id in cite_map:
                return cite_map[doc_id]
        # No marker in the sentence -- use the joined cited text as premise.
        if cited_spans:
            joined = " ".join(c.cited_text for c in cited_spans[:3])
            return joined if joined.strip() else None
        return None
