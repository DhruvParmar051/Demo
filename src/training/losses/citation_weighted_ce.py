"""
Citation-weighted cross-entropy loss.

Tokens inside citation markers ``[doc_id:start-end]`` are upweighted
(default 3.0x) relative to ordinary tokens (1.0x). This steers the
generator to reliably emit well-formed citation markers.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_CITATION_RE = re.compile(r"\[[a-f0-9]{6,32}:\d+-\d+\]")


class CitationWeightedCELoss(nn.Module):
    """Cross-entropy with elevated weight on citation-marker tokens.

    Parameters
    ----------
    citation_weight : float
        Multiplier applied to citation-marker tokens.
    other_weight : float
        Multiplier applied to all remaining (non-pad) tokens.
    ignore_index : int
        Label id to ignore (typically -100).
    """

    def __init__(
        self,
        citation_weight: float = 3.0,
        other_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.citation_weight = float(citation_weight)
        self.other_weight = float(other_weight)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: Any,
    ) -> torch.Tensor:
        """Compute the weighted loss.

        ``logits`` shape: (batch, seq, vocab).
        ``labels`` shape: (batch, seq).
        """
        weights = self._build_weight_mask(labels, tokenizer)
        # Shift not required here; trainers typically pre-shift. We assume
        # logits and labels are already aligned.
        vocab = logits.size(-1)
        logits_flat = logits.view(-1, vocab)
        labels_flat = labels.view(-1)
        weights_flat = weights.view(-1)

        mask = labels_flat != self.ignore_index
        if not mask.any():
            return logits.new_zeros(())

        logits_masked = logits_flat[mask]
        labels_masked = labels_flat[mask]
        weights_masked = weights_flat[mask]

        per_token = F.cross_entropy(
            logits_masked, labels_masked, reduction="none"
        )
        weighted = per_token * weights_masked
        return weighted.sum() / weights_masked.sum().clamp_min(1e-6)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_weight_mask(
        self, labels: torch.Tensor, tokenizer: Any
    ) -> torch.Tensor:
        """Construct a per-token weight tensor.

        We decode each row to text, find citation-marker character spans,
        and map those spans back to token positions via tokenizer offsets.
        If offset mapping is unavailable, fallback to the simpler approach
        of decoding per-token and string-matching ``[`` / ``]``.
        """
        batch, seq = labels.shape
        weights = torch.full(
            (batch, seq),
            self.other_weight,
            dtype=torch.float32,
            device=labels.device,
        )

        for b in range(batch):
            ids = labels[b].tolist()
            clean_ids = [i if i != self.ignore_index else tokenizer.pad_token_id or 0
                         for i in ids]
            text = tokenizer.decode(clean_ids, skip_special_tokens=False)
            spans = [m.span() for m in _CITATION_RE.finditer(text)]
            if not spans:
                continue
            # Best-effort token-span mapping via cumulative decode lengths.
            offsets = _per_token_offsets(clean_ids, tokenizer)
            if offsets is None:
                continue
            for (s_char, e_char) in spans:
                for t_idx, (t_s, t_e) in enumerate(offsets):
                    if t_e <= s_char:
                        continue
                    if t_s >= e_char:
                        break
                    if t_idx < seq:
                        weights[b, t_idx] = self.citation_weight
        return weights


def _per_token_offsets(ids: list[int], tokenizer: Any) -> list[tuple[int, int]] | None:
    """Approximate per-token character offsets by incremental decoding."""
    try:
        cum_text = ""
        offsets: list[tuple[int, int]] = []
        for i, tid in enumerate(ids):
            prev = len(cum_text)
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            cum_text += piece
            offsets.append((prev, len(cum_text)))
        return offsets
    except Exception:
        return None
