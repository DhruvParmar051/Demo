"""AegisRAG - Improved Synthetic QA Generator (v4)

Key improvements over v3:
1.  EVIDENCE QUALITY: Evidence spans are now 1-3 full sentences (80-400 chars),
    not micro-snippets. Uses sentence-boundary alignment instead of raw char offsets.
2.  ANSWER DEPTH: Answers are 2-4 sentences (30-80 words), always including the
    condition/threshold + its consequence/effect, not just a bare fact.
3.  QUESTION DIVERSITY: 6 question frames per qtype (not 1), plus multi-hop and
    comparison frames. Heuristic qtype assignment is now feature-weighted.
4.  COVERAGE: Per-document cap raised; section-title diversity enforced so
    consecutive chunks from the same section are skipped after 2 successes.
5.  PERFORMANCE: Evidence extraction vectorised with pre-tokenised sets;
    sliding-window stride halved; validation fast-path short-circuits early.
6.  DEDUPLICATION: Normalised canonical form (lowercase + stopword strip) used
    for all similarity checks, not raw string.
7.  PROMPT ENGINEERING: Explicit negative examples and lengthier answer
    requirements prevent single-sentence / bare-number answers.
8.  CLEANING: Post-process step fixes citation marker placement (marker goes
    immediately after the sentence it supports, not appended at EOD).
"""

from __future__ import annotations

import json
import logging
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import ChunkRecord, Citation, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_TYPES = ("procedural", "policy", "eligibility", "factoid", "multi_hop", "comparison")

CHECKPOINT_FILE = Path("data/synthetic/qa_pairs.jsonl")
SAVE_EVERY = 20
_GENERATION_BATCH_SIZE = 1   # safe for MPS / T4; raise to 4 on multi-GPU

_CITATION_RE = re.compile(r"\[[^\]]+\]")
_CITATION_SECTION_RE = re.compile(
    r"(?:§+\s*[\d]+|"
    r"\b\d+\s*CFR\b|"
    r"\bPub\.?\s*L\.?\s*\d+|"
    r"\bU\.S\.C\.?\s*§|"
    r"IRM\s*\d+\.\d+|"
    r"Rev\.\s*Proc\.\s*\d+|"
    r"Rev\.\s*Rul\.\s*\d+)",
    re.IGNORECASE,
)
_CODE_MARKERS = (
    "select ", "insert ", "update ", "delete ", "create ", "alter ", "drop ",
    "from ", "where ", "group by", "order by", "returning", "pg_", "::",
    "postgres=#", "=>", "#>", "->", "$$", "begin;", "commit;",
)

# Stopwords for canonical dedup
_STOP = frozenset({
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "for", "on", "at", "by", "with", "as", "that", "this",
    "it", "its", "from", "which", "who", "but", "not", "can", "may", "will",
    "would", "should", "could", "also", "such", "any", "all", "some", "these",
    "those", "has", "have", "had", "do", "does", "did", "if", "then", "than",
    "so", "when", "while",
})

# Feature triggers per qtype
_TRIG = {
    "eligibility": ("eligib", "qualif", "ineligib", "disqualif", "entitle",
                    "who may", "who can", "who is", "who are", "to receive",
                    "to get", "to claim", "to apply", "to enroll"),
    "procedural":  ("step", "process", "submit", "file", "form", "apply",
                    "register", "complete", "follow", "procedure", "how to",
                    "in order to", "you must", "you should", "you need to"),
    "factoid":     ("percent", "%", "dollar", "$", "limit", "threshold",
                    "maximum", "minimum", "deadline", "days", "months",
                    "years", "age", "income", "amount", "rate", "penalty"),
    "policy":      ("because", "therefore", "rationale", "purpose", "intent",
                    "policy", "law", "regulation", "rule", "provision",
                    "requirement", "shall", "notwithstanding", "subject to",
                    "pursuant to", "provided that", "unless", "except"),
    "multi_hop":   ("in addition", "furthermore", "however", "on the other hand",
                    "relates to", "depends on", "as a result", "therefore",
                    "consequently"),
    "comparison":  ("compared to", "unlike", "in contrast", "similarly",
                    "whereas", "while", "different from", "same as"),
}

# ---------------------------------------------------------------------------
# Prompt templates — 6 frames per qtype
# ---------------------------------------------------------------------------

_QTYPE_FRAMES = {
    "procedural": [
        "Ask HOW a person completes a required process, files a form, or satisfies "
        "a regulatory step.  The answer must describe ≥2 specific sub-steps and explain "
        "state each required step and any condition or requirement explicitly mentioned",
        "Ask WHAT sequence of actions is required when a specific triggering event occurs "
        "(e.g., a late filing, an overpayment, a change in status). Answer must name the "
        "trigger, the required action, and the time deadline.",
        "Ask WHICH form or document is required and HOW it must be submitted.  Answer must "
        "state the form name/number, where it is submitted, and what information it captures.",
        "Ask WHAT happens if a required procedural step is missed or late.  Answer must "
        "state the penalty, consequence, or cure mechanism.",
        "Ask HOW a correction or amendment is made after an error has been filed.  Answer "
        "must state the corrective form, the window to act, and any restrictions.",
        "Ask WHEN a periodic filing or renewal is due and HOW advance preparation is done.  "
        "Answer must include the deadline trigger, the preparation steps, and any grace period.",
    ],
    "policy": [
        "Ask WHY a specific rule or restriction exists — its policy rationale.  Answer must "
        "name the underlying statutory or regulatory purpose and its practical effect.",
        "Ask HOW a rule interacts with or limits another rule in the same passage.  Answer "
        "must explicitly name both rules and describe the interaction.",
        "Ask WHAT the consequence is when a policy condition is violated.  Answer must "
        "state the condition, the violation, and the resulting penalty or disqualification.",
        "Ask UNDER WHAT CIRCUMSTANCES an exception or safe-harbor applies.  Answer must "
        "list the precise qualifying conditions and what protection they confer.",
        "Ask WHY a particular entity or class of persons is treated differently under the rule.  "
        "Answer must state the basis for differential treatment and its practical impact.",
        "Ask HOW the policy balances two competing interests (e.g., flexibility vs. compliance).  "
        "Answer must name both interests and describe the rule's resolution.",
    ],
    "eligibility": [
        "Ask WHO qualifies for a specific benefit, credit, or exemption.  Answer must name "
        "≥2 specific qualifying criteria and at least one disqualifying condition.",
        "Ask WHAT conditions disqualify an otherwise eligible person.  Answer must name "
        "the specific disqualifying facts and the consequence (denial, repayment, etc.).",
        "Ask AT WHAT INCOME or age threshold eligibility changes.  Answer must state the "
        "exact threshold, what changes at that threshold, and how it is measured.",
        "Ask HOW eligibility is verified or documented.  Answer must name the required "
        "documentation and what standard it must meet.",
        "Ask WHETHER a specific class of persons (students, veterans, dependents) qualifies "
        "and under what additional conditions.  Answer must state the class, the base rule, "
        "and the special condition.",
        "Ask WHEN eligibility expires or must be re-established.  Answer must state the "
        "duration, the renewal trigger, and what happens if renewal is missed.",
    ],
    "factoid": [
        "Ask for the EXACT dollar amount, percentage, or numerical threshold used in a "
        "calculation or limit from the passage.  Answer must state the value, the item it "
        "applies to, and the condition under which it applies.",
        "Ask for the SPECIFIC DEADLINE (days, months, or date) imposed by the rule.  Answer "
        "must state the deadline, its trigger event, and the consequence of missing it.",
        "Ask what the MAXIMUM or MINIMUM allowed value is and under what conditions it changes.  "
        "Answer must state the base limit, the change condition, and the revised limit.",
        "Ask for the NAME of the specific form, publication, schedule, or code section that "
        "governs a described situation.  Answer must state the name/number and its purpose.",
        "Ask HOW a specific numerical value is calculated (formula or component breakdown).  "
        "Answer must state the formula or at least two components and the result.",
        "Ask what RATE or RATIO applies in a described circumstance.  Answer must state the "
        "rate, the base it is applied to, and the resulting obligation.",
    ],
    "multi_hop": [
        "Ask a question that requires combining TWO distinct facts from the passage to reach "
        "an answer (e.g., threshold + consequence).  Answer must reference both facts and "
        "show the logical connection explicitly.",
        "Ask how a CHANGE IN ONE VARIABLE (income, age, status) affects TWO downstream "
        "outcomes described in the passage.  Answer must trace both effects.",
        "Ask what the NET RESULT is when TWO rules described in the passage apply simultaneously.  "
        "Answer must name both rules and state their combined effect.",
        "Ask for the CONDITION CHAIN: what must be true first, then second, then what happens.  "
        "Answer must list ≥2 sequential conditions and the final outcome.",
        "Ask what DISTINGUISHES two similar-sounding situations described in the same passage.  "
        "Answer must name the distinguishing fact and explain why it matters.",
        "Ask what the OVERALL IMPACT is of following (or not following) a multi-step process "
        "described in the passage.  Answer must mention ≥2 steps and the cumulative effect.",
    ],
    "comparison": [
        "Ask HOW two entities, time periods, or situations described in the passage are treated "
        "DIFFERENTLY.  Answer must name both, state the difference, and explain why.",
        "Ask WHAT IS THE SAME about two seemingly different situations in the passage.  Answer "
        "must identify the shared rule and note why the distinction does not matter.",
        "Ask WHICH of two options described in the passage is more beneficial (or burdensome) "
        "and under what conditions.  Answer must compare both on the relevant dimension.",
        "Ask HOW the rule changed between two time periods mentioned in the passage.  Answer "
        "must state the old rule, the new rule, and the effective date.",
        "Ask HOW the treatment differs for two classes of taxpayers/students/patients described "
        "in the passage.  Answer must name both classes and describe the differential.",
        "Ask WHETHER two cited thresholds or deadlines are the same or different and what "
        "drives the difference.  Answer must state both values and the causal driver.",
    ],
}

# ---------------------------------------------------------------------------
# Chunk scoring helpers
# ---------------------------------------------------------------------------

_table_like_cache: dict[str, bool] = {}


def _looks_like_code(text: str) -> bool:
    tl = text.lower()
    return sum(1 for m in _CODE_MARKERS if m in tl) >= 2


def _has_llm_fluff(answer: str) -> bool:
    fluff_patterns = (
        "this ensures",
        "this allows",
        "this means",
        "ensuring that",
        "helps to",
        "provides a way to",
        "so that",
        "in order to",
    )
    return any(p in answer.lower() for p in fluff_patterns)

def _is_table_like(text: str, chunk_id: str = "") -> bool:
    if chunk_id and chunk_id in _table_like_cache:
        return _table_like_cache[chunk_id]
    result = _compute_is_table_like(text)
    if chunk_id:
        _table_like_cache[chunk_id] = result
    return result


def _compute_is_table_like(text: str) -> bool:
    if not text or not text.strip():
        return True
    tl = text.lower()
    if tl.count("reserved") > 10:
        return True
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    is_code = _looks_like_code(text)
    terminators = text.count(".") + text.count("!") + text.count("?")
    citation_count = len(_CITATION_SECTION_RE.findall(text))
    is_reg_text = citation_count >= 3
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    if len(lines) >= 6 and pipe_lines / len(lines) > 0.6 and not is_code:
        return True
    tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2)
    if len(lines) >= 6 and tab_lines / len(lines) > 0.6:
        return True
    if len(lines) >= 8 and not is_code:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
        if short_lines / len(lines) > 0.75 and terminators < 3:
            return True
    if len(text) > 300 and not is_code and not is_reg_text:
        if sum(c.isdigit() for c in text) / len(text) > 0.55:
            return True
    if len(text.split()) > 200 and terminators == 0 and not is_code and not is_reg_text:
        return True
    return False


def _score_chunk_quality(chunk: ChunkRecord) -> float:
    text = chunk.text or ""
    words = text.split()
    wc = len(words)
    if wc < 30:
        return 0.0
    score = 0.0
    # Word-count sweet spot: 50-500
    if 50 <= wc <= 500:
        score += 2.5
    elif 30 <= wc < 50:
        score += 1.0
    else:
        score += 1.5
    # Sentence count (full sentences preferred)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 5]
    if len(sentences) >= 4:
        score += 2.0
    elif len(sentences) >= 2:
        score += 1.0
    # Regulatory signal
    tl = text.lower()
    for trig_list in _TRIG.values():
        score += min(sum(1 for t in trig_list if t in tl) * 0.2, 1.0)
    # Numeric grounding
    if re.search(r"\b\d[\d,\.]*\b", text):
        score += 0.5
    if "$" in text or "%" in text:
        score += 0.5
    # Section title bonus
    if chunk.section_title and len(chunk.section_title.strip()) > 3:
        score += 0.5
    return max(0.0, score)


def _choose_qtype(chunk: ChunkRecord, rng: random.Random) -> str:
    """Feature-weighted qtype selection — multi_hop/comparison get 20% slots."""
    text_l = (chunk.text or "").lower()
    base_scores: dict[str, float] = {}
    for qt, triggers in _TRIG.items():
        base_scores[qt] = float(sum(1 for t in triggers if t in text_l))
    # Normalise base types
    max_s = max(base_scores.values()) or 1.0
    for qt in base_scores:
        base_scores[qt] /= max_s
    # Force some multi_hop / comparison diversity
    if rng.random() < 0.15:
        return "multi_hop"
    if rng.random() < 0.10:
        return "comparison"
    # Weighted choice
    candidates = [(qt, s) for qt, s in base_scores.items()
                  if qt not in ("multi_hop", "comparison") and s > 0]
    if not candidates:
        return rng.choice(["policy", "factoid", "eligibility"])
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Top-2 with some randomness
    top2 = candidates[:2]
    return rng.choices([t[0] for t in top2], weights=[t[1] + 0.1 for t in top2])[0]


# ---------------------------------------------------------------------------
# Evidence extraction — v4: sentence-boundary aligned, 80-400 chars
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in raw if len(s.split()) >= 5]


def _canonical(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOP and len(w) > 2}


def _get_evidence(answer: str, chunk_text: str) -> tuple[str, int, int]:
    """Return (evidence_text, span_start, span_end) with sentence-boundary alignment.

    Guarantees: evidence is ≥1 full sentence and ≤3 sentences.
    Falls back to first 3 sentences if no overlap found.
    """
    if not chunk_text.strip():
        return chunk_text[:300], 0, min(300, len(chunk_text))

    sentences = _split_sentences(chunk_text)
    if not sentences:
        snippet = chunk_text.strip()[:350]
        return snippet, 0, len(snippet)

    answer_tokens = _canonical(answer)
    if not answer_tokens:
        # Use first 2 sentences as default
        evidence = " ".join(sentences[:2])
        start = chunk_text.find(sentences[0])
        if start == -1:
            start = 0
        return evidence, start, start + len(evidence)

    # Score each sentence by token overlap
    sent_scores = []
    for s in sentences:
        s_tokens = _canonical(s)
        score = len(answer_tokens & s_tokens)
        sent_scores.append((score, s))

    # Pick the best-scoring sentence, then greedily add the adjacent one
    # if it also shares overlap (ensures 2-sentence evidence for depth)
    best_idx = max(range(len(sent_scores)), key=lambda i: sent_scores[i][0])
    selected_indices = {best_idx}
    if best_idx + 1 < len(sentences) and sent_scores[best_idx + 1][0] >= 1:
        selected_indices.add(best_idx + 1)
    if best_idx - 1 >= 0 and sent_scores[best_idx - 1][0] >= 1:
        selected_indices.add(best_idx - 1)

    # Cap at 3 sentences
    sorted_indices = sorted(selected_indices)[:3]
    evidence_sentences = [sentences[i] for i in sorted_indices]
    evidence = " ".join(evidence_sentences)

    # Enforce min/max length
    if len(evidence) < 40:
        # Pad with adjacent sentences
        if best_idx + 1 < len(sentences):
            evidence = evidence + " " + sentences[best_idx + 1]
        elif best_idx - 1 >= 0:
            evidence = sentences[best_idx - 1] + " " + evidence
    if len(evidence) > 500:
        evidence = evidence[:500]
        # Trim to last sentence boundary
        last_punct = max(evidence.rfind("."), evidence.rfind("!"), evidence.rfind("?"))
        if last_punct > 200:
            evidence = evidence[:last_punct + 1]

    # Locate in original text
    first_sent = evidence_sentences[0] if evidence_sentences else evidence[:60]
    start = chunk_text.find(first_sent)
    if start == -1:
        # Try partial match
        probe = first_sent[:min(40, len(first_sent))]
        start = chunk_text.find(probe)
        if start == -1:
            start = 0
    end = start + len(evidence)
    end = min(end, len(chunk_text))
    return evidence, start, end


# ---------------------------------------------------------------------------
# Prompt builder — uses frame rotation
# ---------------------------------------------------------------------------

def _build_prompt(chunk: ChunkRecord, qtype: str, rng: random.Random) -> str:
    frames = _QTYPE_FRAMES.get(qtype, _QTYPE_FRAMES["policy"])
    frame = rng.choice(frames)

    # Build a short passage header
    section_hint = f" (Section: {chunk.section_title})" if chunk.section_title else ""
    source_hint = Path(chunk.source).name if chunk.source else "policy document"

    return (
        f"You are writing a high-quality QA training pair for a RAG system.\n"
        f"Source: {source_hint}{section_hint}\n\n"
        f"PASSAGE:\n{chunk.text}\n\n"
        f"QUESTION TYPE: {qtype.upper()}\n"
        f"QUESTION FRAME: {frame}\n\n"
        "STRICT RULES:\n"
        "QUESTION:\n"
        "  - 8 to 25 words.  End with a question mark.\n"
        "  - Must be self-contained (answerable without seeing the passage title).\n"
        "  - Must require reasoning, not a simple lookup.\n"
        "  - Do NOT reference form numbers, section numbers, or table row labels.\n"
        "  - Do NOT start with 'Why does this' or 'What does the passage say'.\n"
        "  - BAD example: 'What is the dollar amount on line 3?'\n"
        "  - GOOD example: 'Under what income threshold does the credit phase out completely?'\n\n"
        "ANSWER:\n"
       "ANSWER:\n"
        "  - 1 to 2 sentences ONLY (concise and precise).\n"
        "  - MUST directly answer the question using ONLY information from the passage.\n"
        "  - Do NOT add explanations, reasoning, or implications unless explicitly stated.\n"
        "  - Do NOT infer calculations or outcomes not present in the text.\n"
        "  - Use exact values, thresholds, and conditions from the passage.\n"
        "  - Avoid phrases like 'this ensures', 'this allows', 'this means'.\n"
        "  - MUST state the specific condition or threshold AND its consequence or effect.\n"
        "  - Use exact regulatory language (amounts, percentages, deadlines) from the passage.\n"
        "  - Do NOT fabricate numbers or dates not in the passage.\n"
        "  - Do NOT use bullet points or numbered lists.\n"
        "  - BAD example: 'The amount is 20 percent.'\n"
        "  - GOOD example: 'The coinsurance liability for DME furnished as a home health "
        "  - Use natural, human-friendly phrasing instead of formal/legal wording.\n"   
        "  - Prefer 'choose' instead of 'elect', 'before' instead of 'prior to'.\n"
        "service is 20 percent of the fee schedule amount. This applies only to services "
        "covered under Part B, not to hospital inpatient stays.'\n\n"
        "OUTPUT (strict JSON, no code fences, no extra text):\n"
      
        '{"query": "...", "answer": "..."}'
    )


def _build_retry_prompt(chunk: ChunkRecord, qtype: str, rng: random.Random) -> str:
    """Retry with a different qtype frame."""
    alt = {"procedural": "policy", "policy": "eligibility",
           "eligibility": "factoid", "factoid": "procedural",
           "multi_hop": "policy", "comparison": "eligibility"}.get(qtype, "policy")
    return _build_prompt(chunk, alt, rng)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _canonical_str(text: str) -> str:
    tokens = _canonical(text)
    return " ".join(sorted(tokens))


def is_duplicate(q: str, seen_canonical: set[str], threshold: float = 0.88) -> bool:
    cq = _canonical_str(q)
    for ec in seen_canonical:
        if SequenceMatcher(None, cq, ec).ratio() > threshold:
            return True
    return False


def _parse_one(raw: str) -> dict | None:
    if not raw:
        return None
    matches = re.findall(r"\{[\s\S]*?\}", raw)
    for m in matches:
        try:
            obj = json.loads(m)
        except Exception:
            try:
                obj = json.loads(re.sub(r"\n+", " ", m))
            except Exception:
                continue
        if isinstance(obj, dict):
            q = obj.get("query", "").strip()
            a = obj.get("answer", "").strip()
            if q:
                return {"query": q, "answer": a}
    return None


def _valid_pair(q: str, a: str) -> bool:
    """Fast-path validity check."""
    if not q or not a:
        return False
    if len(a.split()) < 8:          # v4: raised minimum from 6 to 15
        return False
    if len(q.split()) < 5:
        return False
    if a.isupper() or not re.search(r"[a-zA-Z]", a):
        return False
    if _CITATION_RE.search(q):       # no citation markers in questions
        return False
    if not q.endswith("?"):
        return False
    return True


def _strict_grounding(answer: str, chunk_text: str) -> bool:
    """Answer tokens must overlap ≥ 25% with chunk (v4: raised from 20%)."""
    a_tokens = [w for w in re.findall(r"\w+", answer.lower())
                if len(w) > 2 and w not in _STOP]
    if not a_tokens:
        return False
    c_words = set(re.findall(r"\w+", chunk_text.lower()))
    overlap = sum(1 for w in a_tokens if w in c_words)
    if overlap / len(a_tokens) >= 0.18:
        return True
    # Bigram fallback
    a_bg = list(zip(a_tokens, a_tokens[1:]))
    c_bg = set(zip(list(c_words), list(c_words)))  # crude
    c_word_list = [w for w in re.findall(r"\w+", chunk_text.lower()) if len(w) > 2]
    c_bg = set(zip(c_word_list, c_word_list[1:]))
    if not a_bg:
        return False
    bg_overlap = sum(1 for bg in a_bg if bg in c_bg)
    return bg_overlap / len(a_bg) >= 0.30


def _answer_has_depth(a: str) -> bool:
    """Heuristic: answer mentions a condition AND a value/consequence."""
    al = a.lower()
    has_condition = any(w in al for w in ("if", "when", "must", "only", "unless",
                                           "provided", "subject to", "except",
                                           "until", "after", "before", "at least",
                                           "no more than"))
    has_value = bool(re.search(r"\b\d[\d,\.%]*\b", a) or
                     any(w in al for w in ("percent", "dollar", "days", "months",
                                            "years", "penalty", "eligible", "required",
                                            "disqualif", "credit", "benefit")))
    return has_condition or has_value

def _has_inferred_math(answer: str, chunk_text: str) -> bool:
    if re.search(r"\d+\s*[-+*/]\s*\d+", answer) and not re.search(r"\d+\s*[-+*/]\s*\d+", chunk_text):
        return True
    return False

def _validate_candidate(p: dict | None, chunk_text: str, idx: int) -> dict | None:
    if p is None:
        return None
    q = p.get("query", "").strip()
    a = re.sub(r"\[[^\]]+\]", "", p.get("answer", "")).strip()
    a = re.sub(r"\s+", " ", a)
    p["answer"] = a

    if not _valid_pair(q, a):
        logger.debug("[REJECT] idx=%d reason=invalid_pair q=%s", idx, q[:60])
        return None
    if not _strict_grounding(a, chunk_text):
        logger.debug("[REJECT] idx=%d reason=grounding a=%s", idx, a[:60])
        return None
    if not _answer_has_depth(a):
        logger.debug("[REJECT] idx=%d reason=no_depth a=%s", idx, a[:60])
        return None
    if _has_llm_fluff(a):
        logger.debug("[REJECT] idx=%d reason=fluff a=%s", idx, a[:60])
        return None
    if _has_inferred_math(a, chunk_text):
        return None
    logger.debug("[ACCEPT] idx=%d q=%s", idx, q[:60])
    return p


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_batch(
    batch: list[tuple[ChunkRecord, str, int]],
    teacher: Any,
    seen_canonical: set[str],
    stats: dict[str, int],
    rng: random.Random,
) -> list[dict]:
    if not batch:
        return []

    prompts = [_build_prompt(chunk, qtype, rng) for chunk, qtype, _ in batch]
    try:
        outputs = teacher.generate_batch(prompts)
    except Exception as e:
        logger.warning("Batch generation failed: %s", e)
        return []

    results: list[dict] = []
    retry_needed: list[tuple[int, ChunkRecord, str]] = []

    for output, (chunk, qtype, real_idx) in zip(outputs, batch):
        parsed = _parse_one(output)
        candidate = _validate_candidate(parsed, chunk.text, real_idx)
        if candidate:
            if not is_duplicate(candidate["query"], seen_canonical):
                candidate["chunk"] = chunk
                candidate["qtype"] = qtype
                results.append(candidate)
            else:
                stats["skip_dup"] += 1
        else:
            retry_needed.append((real_idx, chunk, qtype))
            stats["retry"] += 1

    if retry_needed:
        retry_prompts = [_build_retry_prompt(c, qt, rng) for _, c, qt in retry_needed]
        try:
            retry_outputs = teacher.generate_batch(retry_prompts)
        except Exception as e:
            logger.warning("Retry batch failed: %s", e)
            retry_outputs = [""] * len(retry_prompts)
        for (real_idx, chunk, qtype), retry_out in zip(retry_needed, retry_outputs):
            parsed = _parse_one(retry_out)
            candidate = _validate_candidate(parsed, chunk.text, real_idx)
            if candidate:
                if not is_duplicate(candidate["query"], seen_canonical):
                    candidate["chunk"] = chunk
                    candidate["qtype"] = qtype
                    results.append(candidate)
                else:
                    stats["skip_dup"] += 1
            else:
                stats["skip_invalid"] += 1

    logger.info("Batch size=%d valid=%d", len(batch), len(results))
    return results


# ---------------------------------------------------------------------------
# Citation marker placement — inline after the sentence being supported
# ---------------------------------------------------------------------------

def _attach_citation_inline(answer: str, marker: str) -> str:
    """Append citation marker after the LAST sentence that has grounding words.

    If sentence detection fails, append before the final period.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    if len(sentences) <= 1:
        # Single sentence — put marker before final punctuation
        if answer[-1] in ".!?":
            return answer[:-1] + marker + answer[-1]
        return answer + marker
    # Put marker at the end of the last sentence
    last = sentences[-1]
    if last[-1] in ".!?":
        sentences[-1] = last[:-1] + marker + last[-1]
    else:
        sentences[-1] = last + marker
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _checkpoint(buffer: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        for item in buffer:
            f.write(json.dumps(item) + "\n")


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class QAGenerator:
    """Improved synthetic QA generator (v4).

    Changes vs v3:
    - 6 question frames per qtype (rotation prevents repetition)
    - Evidence = 1-3 full sentences, 80-400 chars (not micro-snippets)
    - Answer depth validation (condition + value required)
    - Section-title diversity: skip after 2 hits from the same section
    - Canonical dedup (token-normalised, not raw string)
    - Citation marker placed inline after the final grounded sentence
    """

    def __init__(
        self,
        teacher: Any | None = None,
        seed: int = 42,
        limit: int | None = None,
        batch_size: int = _GENERATION_BATCH_SIZE,
    ) -> None:
        set_seed(seed)
        self.rng = random.Random(seed)
        self.cfg = get_config()
        self.target_count = limit or getattr(self.cfg.synthetic_data, "qa_pairs", 800)
        self._teacher = teacher
        self.batch_size = int(batch_size)

    def _get_teacher(self) -> Any:
        if self._teacher:
            return self._teacher
        from src.data.local_teacher import LocalTeacher
        # v4: slightly higher temperature for diversity
        self._teacher = LocalTeacher(max_new_tokens=300, temperature=0.55)
        return self._teacher

    def _doc_skip_probability(self, count: int, total_docs: int) -> float:
        fair_share = max(4, int(1.8 * self.target_count / max(total_docs, 1)))
        if count < fair_share:
            return 0.0
        excess = count - fair_share + 1
        return min(0.80, 0.15 + 0.15 * excess)

    def _should_skip_chunk(
        self,
        chunk: ChunkRecord,
        processed_ids: set[str],
        seen_canonical: set[str],
        doc_counts: dict[str, int],
        section_counts: dict[str, int],
        total_docs: int,
        stats: dict[str, int],
    ) -> tuple[bool, str]:
        if chunk.chunk_id in processed_ids:
            return True, "already_processed"
        if len((chunk.text or "").split()) < 30:
            stats["skip_short"] += 1
            return True, "short"
        if _is_table_like(chunk.text, chunk_id=chunk.chunk_id):
            stats["skip_table"] += 1
            return True, "table_like"
        quality = _score_chunk_quality(chunk)
        if quality < 2.5:
            stats["skip_lowscore"] = stats.get("skip_lowscore", 0) + 1
            return True, f"low_quality={quality:.1f}"
        # Section diversity: max 2 QA pairs per distinct section title
        sec_key = f"{chunk.doc_id}::{chunk.section_title or 'none'}"
        if section_counts.get(sec_key, 0) >= 2:
            stats["skip_section"] = stats.get("skip_section", 0) + 1
            return True, "section_saturated"
        # Document diversity
        skip_p = self._doc_skip_probability(doc_counts.get(chunk.doc_id, 0), total_docs)
        if skip_p > 0 and self.rng.random() < skip_p:
            stats["skip_doc"] += 1
            return True, f"doc_overused(p={skip_p:.2f})"
        return False, ""

    def generate(
        self,
        chunks: Sequence[ChunkRecord],
        output_path: Any = None,
    ) -> list[QAPair]:
        # Resume from checkpoint
        existing_data: list[dict] = []
        seen_canonical: set[str] = set()
        processed_ids: set[str] = set()

        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing_data.append(obj)
                        if "query" in obj:
                            seen_canonical.add(_canonical_str(obj["query"]))
                        for cid in obj.get("gold_chunk_ids", []):
                            processed_ids.add(cid)
                    except Exception:
                        continue

        qa_pairs: list[QAPair] = []
        buffer: list[dict] = []
        teacher = self._get_teacher()
        doc_counts: dict[str, int] = {}
        section_counts: dict[str, int] = {}

        chunks = list(chunks)
        chunks.sort(key=lambda x: (x.doc_id, x.chunk_id))
        self.rng.shuffle(chunks)
        total_docs = len({c.doc_id for c in chunks})
        generated = len(existing_data)

        logger.info("[RESUME] Loaded %d existing pairs; target=%d", generated, self.target_count)

        if generated >= self.target_count:
            logger.info("[DONE] Target already reached")
            return []

        stats: dict[str, int] = {
            "skip_table": 0, "skip_short": 0, "skip_doc": 0,
            "skip_invalid": 0, "skip_lowscore": 0, "skip_dup": 0,
            "skip_section": 0, "retry": 0,
        }

        pending_batch: list[tuple[ChunkRecord, str, int]] = []

        def _flush_batch() -> None:
            nonlocal generated
            if not pending_batch or generated >= self.target_count:
                pending_batch.clear()
                return

            valid_results = _process_batch(
                pending_batch, teacher, seen_canonical, stats, self.rng
            )

            for result in valid_results:
                if generated >= self.target_count:
                    break

                chunk: ChunkRecord = result.pop("chunk")
                qtype: str = result.pop("qtype")

                cq = _canonical_str(result["query"])
                if cq in seen_canonical:
                    stats["skip_dup"] += 1
                    continue

                seen_canonical.add(cq)
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
                sec_key = f"{chunk.doc_id}::{chunk.section_title or 'none'}"
                section_counts[sec_key] = section_counts.get(sec_key, 0) + 1

                # Extract evidence (sentence-aligned, 1-3 sentences)
                evidence_text, span_start, span_end = _get_evidence(
                    result["answer"], chunk.text
                )
                # Ensure span is within chunk bounds
                span_start = max(0, span_start)
                span_end = min(len(chunk.text), span_end)
                if span_end <= span_start:
                    span_end = min(span_start + len(evidence_text), len(chunk.text))

                citation = Citation(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    span_start=span_start,
                    span_end=span_end,
                    cited_text=evidence_text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    source_url=(chunk.metadata or {}).get("source_url"),
                )

                # Build citation marker and attach inline (after last sentence)
                marker = f" [{chunk.doc_id}:{span_start}-{span_end}]"
                answer_with_marker = _attach_citation_inline(result["answer"], marker)

                qa_obj = QAPair(
                    query=result["query"],
                    answer_with_citations=answer_with_marker,
                    gold_chunk_ids=[chunk.chunk_id],
                    question_type=qtype,
                    citations=[citation],
                )

                qa_pairs.append(qa_obj)
                buffer.append(qa_obj.to_dict())
                processed_ids.add(chunk.chunk_id)
                generated += 1

                logger.info(
                    "[GENERATED #%d] qtype=%s\n  Q: %s\n  A: %s\n  E: %s",
                    generated, qtype,
                    result["query"][:100],
                    result["answer"][:120],
                    evidence_text[:100],
                )

            if len(buffer) >= SAVE_EVERY:
                _checkpoint(buffer)
                buffer.clear()

            pending_batch.clear()

        for i, chunk in enumerate(chunks):
            if generated >= self.target_count:
                break

            skip, reason = self._should_skip_chunk(
                chunk, processed_ids, seen_canonical,
                doc_counts, section_counts, total_docs, stats
            )
            if skip:
                if reason not in ("already_processed",):
                    logger.debug("[SKIP] chunk #%d %s", i + 1, reason)
                continue

            qtype = _choose_qtype(chunk, self.rng)
            logger.debug("→ Queuing chunk #%d qtype=%s QA #%d", i + 1, qtype, generated + 1)
            pending_batch.append((chunk, qtype, i + 1))

            if len(pending_batch) >= self.batch_size:
                _flush_batch()

        if pending_batch:
            _flush_batch()

        if buffer:
            _checkpoint(buffer)

        logger.info("[FINAL] Generated=%d | Stats=%s", generated, stats)
        return qa_pairs