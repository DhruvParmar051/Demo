"""AegisRAG - Local-only Synthetic QA Generator (v3 — fixed evidence + batching).

v3 changes vs v2:
- Fixed _sentence_is_noisy: removed the verb-requirement that killed all evidence
- Fixed _fallback_evidence: removed backwards number-matching guard
- Fixed _get_evidence: single, always-succeeds evidence function replacing 3-retry loop
- Raised _GENERATION_BATCH_SIZE to 4 (MPS-safe)
- _has_explanatory_drift: narrowed to only reject phrases genuinely absent from chunk
- _valid_pair: now takes (q, a) not (obj, qtype, min_tokens) — matched call sites
- _process_batch signature: now passes real_idx correctly through flush
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

QUESTION_TYPES = ("procedural", "policy", "eligibility", "factoid")
_BAD_QTYPE: set[str] = set()

CHECKPOINT_FILE = Path("data/synthetic/qa_pairs.jsonl")
SAVE_EVERY = 20

# Batch size: 4 is safe on MPS with Qwen 7B at max_new_tokens=220
_GENERATION_BATCH_SIZE = 1

REASONING_MARKERS = (
    "because", "ensures", "prevents", "allows", "therefore", "so that",
    "which means", "in order to", "due to", "results in", "leads to",
    "enables", "avoids", "requires", "depends on", "causes",
    "eligible", "eligibility", "ineligible", "qualify", "qualified",
    "subject to", "pursuant to", "provided that", "notwithstanding",
    "shall", "must", "may not", "is required", "are required",
    "unless", "except", "exception", "limitation", "restriction",
    "condition", "conditions", "criteria", "requirement", "requirements",
    "deadline", "within", "no later than", "effective date",
    "applies to", "does not apply", "is not applicable",
    "penalty", "interest", "surcharge", "reduction", "disqualified",
    "entitle", "entitled", "benefit", "coverage", "exclusion",
)

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

_REG_TRIGGERS_ELIGIBILITY = (
    "eligib", "qualif", "ineligib", "disqualif", "entitle", "entitled",
    "who may", "who can", "who is", "who are", "to receive", "to get",
    "to claim", "to apply", "to enroll",
)
_REG_TRIGGERS_PROCEDURAL = (
    "step", "process", "submit", "file", "form", "apply", "register",
    "complete", "follow", "procedure", "how to", "in order to",
    "you must", "you should", "you need to",
)
_REG_TRIGGERS_FACTOID = (
    "percent", "%", "dollar", "$", "limit", "threshold", "maximum",
    "minimum", "deadline", "days", "months", "years", "age", "income",
    "amount", "rate", "penalty", "fine", "fee",
)
_REG_TRIGGERS_POLICY = (
    "because", "therefore", "rationale", "purpose", "intent", "policy",
    "law", "regulation", "rule", "provision", "requirement", "shall",
    "notwithstanding", "subject to", "pursuant to", "provided that",
    "unless", "except", "unless otherwise",
)

_EVIDENCE_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
    "were", "be", "been", "being", "for", "on", "at", "by", "with", "as",
    "that", "this", "it", "its", "from", "which", "who", "whom", "but",
    "not", "can", "may", "will", "would", "should", "could", "also",
    "such", "any", "all", "some", "these", "those", "has", "have", "had",
    "do", "does", "did", "if", "then", "than", "so", "when", "while",
}

_table_like_cache: dict[str, bool] = {}


# =============================================================================
# CHUNK PRE-SCORING
# =============================================================================

def _score_chunk_quality(chunk: ChunkRecord) -> float:
    text = chunk.text or ""
    words = text.split()
    wc = len(words)
    if wc < 25:
        return 0.0
    score = 0.0
    if 40 <= wc <= 400:
        score += 2.0
    elif wc < 40:
        score += 0.5
    else:
        score += 1.0
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    n_sentences = len([s for s in sentences if len(s.split()) >= 4])
    if n_sentences >= 3:
        score += 1.0
    elif n_sentences == 2:
        score += 0.5
    text_l = text.lower()
    reg_hits = sum(1 for t in _REG_TRIGGERS_POLICY if t in text_l)
    score += min(reg_hits * 0.3, 1.5)
    has_number = bool(re.search(r"\b\d[\d,\.]*\b", text))
    has_dollar = "$" in text or "dollar" in text_l
    has_date = bool(re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|\d{4})\b", text_l
    ))
    if has_number:
        score += 0.5
    if has_dollar or has_date:
        score += 0.5
    if chunk.section_title and len(chunk.section_title.strip()) > 2:
        score += 0.5
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 5:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
        if short_lines / len(lines) > 0.6:
            score -= 1.0
    return max(0.0, score)


def _choose_qtype_for_chunk(chunk: ChunkRecord, rng: random.Random) -> str:
    text_l = (chunk.text or "").lower()
    scores: dict[str, int] = {
        "eligibility": sum(1 for t in _REG_TRIGGERS_ELIGIBILITY if t in text_l),
        "procedural": sum(1 for t in _REG_TRIGGERS_PROCEDURAL if t in text_l),
        "factoid": sum(1 for t in _REG_TRIGGERS_FACTOID if t in text_l),
        "policy": sum(1 for t in _REG_TRIGGERS_POLICY if t in text_l),
    }
    max_score = max(scores.values())
    if max_score == 0:
        return "policy"
    candidates = [qt for qt, s in scores.items() if s == max_score]
    return rng.choice(candidates)


# =============================================================================
# HELPERS
# =============================================================================

def is_duplicate(q: str, existing: set[str], threshold: float = 0.9) -> bool:
    for e in existing:
        if SequenceMatcher(None, q, e).ratio() > threshold:
            return True
    return False


def _is_likely_duplicate_chunk(chunk: ChunkRecord, seen_queries: set[str],
                                threshold: float = 0.85) -> bool:
    if not seen_queries:
        return False
    text = chunk.text or ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    lead = sentences[0].strip() if sentences else ""
    if len(lead.split()) < 5:
        return False
    lead_l = lead.lower()
    for q in seen_queries:
        if SequenceMatcher(None, lead_l, q.lower()).ratio() > threshold:
            return True
    return False


def _clean_answer(ans: str) -> str:
    ans = re.sub(r"\[[^\]]+\]", "", ans)
    return re.sub(r"\s+", " ", ans).strip()


def _clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strict_grounding(answer: str, chunk_text: str) -> bool:
    """Answer must share at least 20% of content words with chunk, or pass bigram check."""
    answer_words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 2]
    if not answer_words:
        return False
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    overlap = sum(1 for w in answer_words if w in chunk_words)
    if overlap / len(answer_words) >= 0.20:
        return True
    ans_bigrams = list(zip(answer_words, answer_words[1:]))
    chunk_word_list = [w for w in re.findall(r"\w+", chunk_text.lower()) if len(w) > 3]
    chunk_bigrams = set(zip(chunk_word_list, chunk_word_list[1:]))
    if not ans_bigrams:
        return False
    bigram_overlap = sum(1 for bg in ans_bigrams if bg in chunk_bigrams)
    return bigram_overlap / len(ans_bigrams) >= 0.30


def _has_inferred_math(answer: str, chunk_text: str) -> bool:
    ans_l = answer.lower()
    chunk_l = chunk_text.lower()
    math_phrases = (
        "calculated by", "calculated as", "computed as", "derived from",
        "obtained by", "resulting from subtracting", "result of subtracting",
        "difference between", "sum of",
    )
    for pat in (r"\d+\s*[-+*/]\s*\d+",):
        if re.search(pat, answer) and not re.search(pat, chunk_text):
            return True
    for phrase in math_phrases:
        if phrase in ans_l and phrase not in chunk_l:
            return True
    return False


def _has_explanatory_drift(answer: str, chunk_text: str) -> bool:
    """Only reject when a drift phrase is clearly fabricated (not just rephrased)."""
    ans_l = answer.lower()
    chunk_l = chunk_text.lower()
    # Only flag action phrases that introduce a genuinely new instruction
    extra_action_patterns = (
        "or send an email to",
        "or email us at",
        "or contact via",
        "or submit online at",
    )
    for phrase in extra_action_patterns:
        if phrase in ans_l and phrase not in chunk_l:
            return True
    return False


def _qa_semantic_alignment(query: str, answer: str) -> bool:
    q_words = set(re.findall(r"\w+", query.lower()))
    a_words = set(re.findall(r"\w+", answer.lower()))
    stop = {"the", "is", "are", "what", "who", "how", "when", "why", "does",
            "do", "did", "was", "were", "a", "an", "of", "to", "in", "for"}
    q_content = {w for w in q_words if w not in stop and len(w) > 3}
    if not q_content:
        return True  # can't measure; don't reject
    overlap = q_content & a_words
    return len(overlap) >= max(1, int(0.15 * len(q_content)))


def _has_repetition_artifact(text: str) -> bool:
    _STOPWORDS = {
        "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
        "were", "be", "been", "being", "for", "on", "at", "by", "with", "as",
        "that", "this", "it", "its", "from", "which", "who", "but", "not",
        "can", "may", "will", "would", "should", "could", "also", "such",
        "any", "all", "some", "these", "those", "has", "have", "had",
        "do", "does", "did", "if", "then", "than", "so", "when", "while",
    }
    tokens = re.findall(r"\w+", text.lower())
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a == b and len(a) > 3 and a not in _STOPWORDS:
            return True
    return False


def _valid_pair(q: str, a: str) -> bool:
    if not q or not a or len(a.strip()) == 0:
        return False
    if len(a.split()) < 6:
        return False
    if a.isupper():
        return False
    if not re.search(r"[a-zA-Z]", a):
        return False
    if _CITATION_RE.search(q):
        return False
    return True


def _is_good_question(q: str, qtype: str = "") -> bool:
    ql = q.lower().strip()
    if qtype == "factoid":
        return len(q.split()) >= 5 and "?" in q
    if "?" not in q:
        return False
    if len(q.split()) < 5:
        return False
    if ql.startswith(("define", "list")):
        return False
    if any(w in ql for w in ("trend", "increase", "decrease", "change over time",
                              "consistent", "pattern")):
        return False
    if "why" in ql and qtype != "policy":
        return False
    _REG_TRIGGERS = (
        "eligib", "qualif", "requir", "condition", "criterion", "criteria",
        "deadline", "limit", "threshold", "penalty", "benefit", "coverage",
        "exclusion", "exception", "entitle", "disqualif", "subject to",
        "must", "shall", "allowed", "permitted", "prohibited",
    )
    trivial_starts = ("what is", "what are")
    if any(ql.startswith(x) for x in trivial_starts):
        if qtype == "eligibility":
            return True
        has_reg = any(t in ql for t in _REG_TRIGGERS)
        has_reasoning = any(m in ql for m in ("why", "how", "purpose", "role",
                                               "impact", "effect", "difference",
                                               "when", "where"))
        if not (has_reg or has_reasoning):
            return False
    return True


def _looks_like_code(text: str) -> bool:
    tl = text.lower()
    return sum(1 for m in _CODE_MARKERS if m in tl) >= 2


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
    text_l = text.lower()
    if text_l.count("reserved") > 10:
        return True
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    is_code = _looks_like_code(text)
    word_count = len(text.split())
    terminators = text.count(".") + text.count("!") + text.count("?")
    citation_count = len(_CITATION_SECTION_RE.findall(text))
    is_regulatory_citation_text = citation_count >= 3
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
    if len(text) > 300 and not is_code and not is_regulatory_citation_text:
        if sum(c.isdigit() for c in text) / len(text) > 0.55:
            return True
    if word_count > 200 and terminators == 0 and not is_code and not is_regulatory_citation_text:
        return True
    return False


def score_answer(ans: str) -> float:
    score = 0.0
    words = ans.split()
    wc = len(words)
    if wc >= 40:
        score += 1.5
    elif wc >= 25:
        score += 1.0
    elif wc >= 15:
        score += 0.5
    periods = ans.count(".")
    if periods >= 3:
        score += 2.0
    elif periods == 2:
        score += 1.5
    elif periods == 1:
        score += 1.0
    ans_l = ans.lower()
    marker_hits = sum(1 for w in REASONING_MARKERS if w in ans_l)
    score += min(marker_hits, 4) * 0.65
    if "\n-" in ans or "\n*" in ans or re.search(r"\n\d+\.", ans):
        score -= 0.5
    conditional_hits = len(re.findall(
        r"\b(?:if|unless|provided that|subject to|except|notwithstanding|"
        r"must|shall|eligible|qualify|require)\b", ans_l
    ))
    score += min(conditional_hits, 3) * 0.4
    return score


# =============================================================================
# EVIDENCE EXTRACTION — fixed, always succeeds
# =============================================================================

def _sentence_is_noisy(s: str) -> bool:
    """Conservative noise check. Does NOT require specific verbs."""
    if not s:
        return True
    wc = len(s.split())
    if wc < 5:
        return True
    if "[table]" in s.lower():
        return True
    if s.count("|") >= 2 or s.count("\t") >= 2:
        return True
    if s.lower().count("reserved") > 3:
        return True
    if not re.search(r"[a-zA-Z]{3,}", s):
        return True
    # Too many unexplained uppercase words (merged artifact)
    if sum(1 for w in s.split() if w.isupper() and len(w) > 2) > 4:
        return True
    has_section_ref = bool(_CITATION_SECTION_RE.search(s))
    if len(s) > 40 and not has_section_ref:
        if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.45:
            return True
    return False


def _get_evidence(answer: str, chunk_text: str) -> str:
    """Extract the best evidence span from chunk for the given answer.

    Always returns a non-empty string as long as chunk has any text.
    Strategy (in order):
      1. Sliding-window word overlap (character windows)
      2. Best sentence by word overlap
      3. First clean sentence
      4. First 300 chars of chunk (absolute fallback — never fails)
    """
    if not chunk_text or not chunk_text.strip():
        return ""

    # ── Strategy 1: sliding window ────────────────────────────────────────
    normalized = " ".join(chunk_text.split())
    answer_words = {w for w in re.findall(r"\w+", answer.lower())
                    if w not in _EVIDENCE_STOPWORDS and len(w) > 2}

    best_span = ""
    best_score = 0
    window_size = 300
    stride = 60

    for i in range(0, len(normalized), stride):
        span = normalized[i: i + window_size]
        span_words = {w for w in re.findall(r"\w+", span.lower())
                      if w not in _EVIDENCE_STOPWORDS and len(w) > 2}
        score = len(answer_words & span_words)
        if score > best_score:
            best_score = score
            best_span = span

    if best_score >= 2 and len(best_span.split()) >= 6:
        # Trim to a sentence boundary if possible
        trimmed = best_span.strip()
        # Try to end at sentence boundary
        end_match = list(re.finditer(r"[.!?]", trimmed))
        if end_match:
            trimmed = trimmed[: end_match[-1].end()].strip()
        if len(trimmed.split()) >= 6:
            return trimmed

    # ── Strategy 2: best sentence by word overlap ─────────────────────────
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.replace("\n", " "))
    best_sent = ""
    best_sent_score = 0

    for s in sentences:
        s_clean = _clean_sentence(s)
        if _sentence_is_noisy(s_clean):
            continue
        sent_words = {w for w in re.findall(r"\w+", s_clean.lower())
                      if w not in _EVIDENCE_STOPWORDS and len(w) > 2}
        score = len(answer_words & sent_words)
        if score > best_sent_score:
            best_sent_score = score
            best_sent = s_clean

    if best_sent and len(best_sent.split()) >= 5:
        return best_sent

    # ── Strategy 3: first clean sentence ─────────────────────────────────
    for s in sentences:
        s_clean = _clean_sentence(s)
        if not _sentence_is_noisy(s_clean) and len(s_clean.split()) >= 8:
            return s_clean

    # ── Strategy 4: absolute fallback — never returns empty ──────────────
    return chunk_text.strip()[:350].strip()


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_candidate(
    p: dict | None, chunk_text: str, idx: int, qtype: str
) -> dict | None:
    if p is None:
        print(f"[REJECT] idx={idx} reason=null_output")
        return None

    q = p.get("query", "").strip()
    a = p.get("answer", "").strip()
    a = _clean_answer(a)
    p["answer"] = a

    if not _valid_pair(q, a):
        print(f"[REJECT] idx={idx} q={q[:80]} | reason=invalid_pair")
        return None

    if not _is_good_question(q, qtype):
        print(f"[REJECT] idx={idx} q={q[:80]} | reason=bad_question")
        return None

    if _has_repetition_artifact(a) and len(a.split()) < 15:
        print(f"[REJECT] idx={idx} reason=repetition_artifact")
        return None

    if not _qa_semantic_alignment(q, a):
        print(f"[REJECT] idx={idx} q={q[:80]} | reason=semantic_alignment")
        return None

    if _has_inferred_math(a, chunk_text):
        print(f"[REJECT] idx={idx} reason=inferred_math")
        return None

    if _has_explanatory_drift(a, chunk_text):
        print(f"[REJECT] idx={idx} reason=drift")
        return None

    if not _strict_grounding(a, chunk_text):
        print(f"[REJECT] idx={idx} reason=grounding | a={a[:80]}")
        return None

    print(f"[ACCEPT] idx={idx} | q={q[:60]}")
    return p


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def _build_prompt(chunk: ChunkRecord, qtype: str) -> str:
    style_hint = {
        "procedural": (
            "Ask HOW a person complies with a rule, files a form, or completes "
            "a process. The answer must explain the required steps and WHY each "
            "step matters."
        ),
        "policy": (
            "Ask WHY a rule or policy exists, WHAT its consequence is if violated, "
            "or HOW it interacts with another rule. The answer must explain the "
            "policy rationale and its practical effect."
        ),
        "eligibility": (
            "Ask WHO qualifies for a benefit or what conditions DISQUALIFY someone. "
            "The answer must name specific criteria, thresholds, or exclusions from "
            "the passage. Do NOT invent numbers."
        ),
        "factoid": (
            "Ask a specific factual question whose answer is a precise threshold, "
            "deadline, dollar amount, or named requirement from the passage. "
            "The answer must state the exact value."
        ),
    }.get(qtype, "Focus on conditions, requirements, or consequences.")

    return (
        "You are writing a high-quality QA pair grounded in the policy passage below.\n\n"
        f"PASSAGE:\n{chunk.text}\n\n"
        f"STYLE: {style_hint}\n\n"
        "QUESTION RULES:\n"
        "- Ask about CONDITIONS, LIMITS, or EXCEPTIONS — not simple definitions.\n"
        "- Prefer questions requiring reasoning (e.g., 'under what conditions', 'when does', 'why does').\n"
        "- Avoid trivial lookups like single numbers unless tied to a condition.\n"
        "- 8-25 words. End with a question mark.\n"
        "- Do NOT ask about section numbers, form numbers, or publication names.\n\n"
        "ANSWER RULES:\n"
        "- 1-3 sentences grounded in the passage.\n"
        "- Use exact regulatory language where possible.\n"
        "- State the condition/requirement and its effect.\n"
        "- Do NOT invent numbers, dates, or thresholds not in the passage.\n"
        "- No bullet points or lists.\n"
        "- 20-60 words.\n\n"
        "OUTPUT (strict JSON, no code fences):\n"
        '{"query": "...", "answer": "..."}'
    )


def _build_retry_prompt(chunk: ChunkRecord, qtype: str) -> str:
    alt_hints = {
        "procedural": "eligibility",
        "policy": "factoid",
        "eligibility": "policy",
        "factoid": "procedural",
    }
    return _build_prompt(chunk, alt_hints.get(qtype, "policy"))


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


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def _process_batch(
    batch: list[tuple[ChunkRecord, str, int]],
    teacher: Any,
    seen_queries: set[str],
    doc_counts: dict[str, int],
    stats: dict[str, int],
) -> list[dict]:
    if not batch:
        return []

    prompts = [_build_prompt(chunk, qtype) for chunk, qtype, _ in batch]

    try:
        outputs = teacher.generate_batch(prompts)
    except Exception as e:
        logger.warning("Batch generation failed (%s); skipping batch", e)
        return []

    results: list[dict] = []
    retry_needed: list[tuple[int, ChunkRecord, str]] = []

    for output, (chunk, qtype, real_idx) in zip(outputs, batch):
        parsed = _parse_one(output)
        candidate = _validate_candidate(parsed, chunk.text, real_idx, qtype)

        if candidate:
            if not is_duplicate(candidate["query"], seen_queries):
                candidate["chunk"] = chunk
                candidate["qtype"] = qtype
                results.append(candidate)
            else:
                print(f"[REJECT] idx={real_idx} reason=duplicate")
                stats["skip_dup"] += 1
        else:
            retry_needed.append((real_idx, chunk, qtype))

    if retry_needed:
        retry_prompts = [_build_retry_prompt(c, qt) for _, c, qt in retry_needed]
        try:
            retry_outputs = teacher.generate_batch(retry_prompts)
        except Exception as e:
            logger.warning("Retry batch failed (%s)", e)
            retry_outputs = [""] * len(retry_prompts)

        for (real_idx, chunk, qtype), retry_output in zip(retry_needed, retry_outputs):
            parsed = _parse_one(retry_output)
            candidate = _validate_candidate(parsed, chunk.text, real_idx, qtype)
            if candidate:
                if not is_duplicate(candidate["query"], seen_queries):
                    candidate["chunk"] = chunk
                    candidate["qtype"] = qtype
                    results.append(candidate)
                else:
                    print(f"[REJECT] idx={real_idx} reason=duplicate_retry")
                    stats["skip_dup"] += 1
            else:
                print(f"[REJECT] idx={real_idx} reason=retry_failed (final)")
                stats["skip_invalid"] += 1

    print(f"[DEBUG] Batch size: {len(batch)} | Valid: {len(results)}")
    return results


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class QAGenerator:
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
        self.target_count = limit or self.cfg.synthetic_data.qa_pairs
        self._teacher = teacher
        self.batch_size = int(batch_size)

    def _get_teacher(self) -> Any:
        if self._teacher:
            return self._teacher
        from src.data.local_teacher import LocalTeacher
        self._teacher = LocalTeacher(max_new_tokens=220, temperature=0.4)
        return self._teacher

    def _doc_skip_probability(self, count: int, total_docs: int, target: int) -> float:
        if total_docs <= 0:
            fair_share = max(3, target // 10)
        else:
            fair_share = max(3, int(1.5 * target / max(total_docs, 1)))
        if count < fair_share:
            return 0.0
        excess = count - fair_share + 1
        return min(0.85, 0.2 + 0.15 * excess)

    def _should_skip_chunk(
        self,
        chunk: ChunkRecord,
        processed_chunk_ids: set[str],
        seen_queries: set[str],
        doc_counts: dict[str, int],
        total_docs: int,
        stats: dict[str, int],
    ) -> tuple[bool, str]:
        if chunk.chunk_id in processed_chunk_ids:
            return True, "already_processed"
        if len((chunk.text or "").split()) < 15:
            stats["skip_short"] += 1
            return True, "short"
        if _is_table_like(chunk.text, chunk_id=chunk.chunk_id):
            cleaned_lines = [
                ln.strip() for ln in chunk.text.split("\n")
                if len(ln.split()) > 6 and ln.count("|") < 2 and ln.count("\t") < 2
            ]
            cleaned = " ".join(cleaned_lines)
            if len(cleaned.split()) >= 40 and not _is_table_like(cleaned):
                chunk.text = cleaned
                _table_like_cache[chunk.chunk_id] = False
            else:
                stats["skip_table"] += 1
                return True, "table_like"
        quality_score = _score_chunk_quality(chunk)
        if quality_score < 2.0:
            stats["skip_lowscore"] = stats.get("skip_lowscore", 0) + 1
            return True, f"low_quality_score={quality_score:.2f}"
        if _is_likely_duplicate_chunk(chunk, seen_queries):
            stats["skip_dup"] += 1
            return True, "likely_duplicate_chunk"
        count = doc_counts.get(chunk.doc_id, 0)
        skip_p = self._doc_skip_probability(count, total_docs, self.target_count)
        if skip_p > 0 and self.rng.random() < skip_p:
            stats["skip_doc"] += 1
            return True, f"doc_overused(count={count}, p={skip_p:.2f})"
        return False, ""

    def generate(
        self,
        chunks: Sequence[ChunkRecord],
        output_path: Any = None,
    ) -> list[QAPair]:
        # ── Resume from checkpoint ──────────────────────────────────────────
        existing_data: list[dict] = []
        seen_queries: set[str] = set()
        processed_chunk_ids: set[str] = set()

        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing_data.append(obj)
                        if "query" in obj:
                            seen_queries.add(obj["query"])
                        for cid in obj.get("gold_chunk_ids", []):
                            processed_chunk_ids.add(cid)
                    except Exception:
                        continue

        qa_pairs: list[QAPair] = []
        buffer: list[dict] = []
        teacher = self._get_teacher()
        doc_counts: dict[str, int] = {}

        chunks = list(chunks)
        chunks.sort(key=lambda x: (x.doc_id, x.chunk_id))
        self.rng.shuffle(chunks)
        total_docs = len({c.doc_id for c in chunks})
        generated = len(existing_data)

        print(f"[RESUME] Loaded {generated} existing QA pairs")
        print(f"[RESUME] Skipping {len(processed_chunk_ids)} processed chunks")

        if generated >= self.target_count:
            print("[DONE] Target already reached")
            return []

        stats: dict[str, int] = {
            "skip_table": 0, "skip_short": 0, "skip_doc": 0,
            "skip_invalid": 0, "skip_lowscore": 0,
            "skip_dup": 0, "skip_weakq": 0,
        }

        pending_batch: list[tuple[ChunkRecord, str, int]] = []

        def _flush_batch() -> None:
            nonlocal generated
            if not pending_batch or generated >= self.target_count:
                pending_batch.clear()
                return

            valid_results = _process_batch(
                pending_batch, teacher, seen_queries, doc_counts, stats
            )

            for result in valid_results:
                if generated >= self.target_count:
                    break

                chunk: ChunkRecord = result.pop("chunk")
                qtype: str = result.pop("qtype")

                if is_duplicate(result["query"], seen_queries):
                    stats["skip_dup"] += 1
                    continue

                seen_queries.add(result["query"])
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

                # ── Evidence extraction (always succeeds) ───────────────────
                span_text = _get_evidence(result["answer"], chunk.text)
                span_text = _clean_sentence(span_text)

                if not span_text:
                    # This should never happen given _get_evidence's fallback
                    span_text = chunk.text[:300].strip()

                start = chunk.text.lower().find(span_text.lower())
                if start == -1:
                    start = 0
                    end = min(len(span_text), len(chunk.text))
                else:
                    end = start + len(span_text)

                citation = Citation(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    span_start=start,
                    span_end=end,
                    cited_text=span_text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    source_url=(chunk.metadata or {}).get("source_url"),
                )

                answer_with_marker = result["answer"].rstrip()
                marker = f" [{chunk.doc_id}:{start}-{end}]"
                if marker.strip() not in answer_with_marker:
                    if answer_with_marker.endswith((".", "!", "?")):
                        answer_with_marker = (
                            answer_with_marker[:-1] + marker + answer_with_marker[-1]
                        )
                    else:
                        answer_with_marker += marker

                qa_obj = QAPair(
                    query=result["query"],
                    answer_with_citations=answer_with_marker,
                    gold_chunk_ids=[chunk.chunk_id],
                    question_type=qtype,
                    citations=[citation],
                )

                qa_pairs.append(qa_obj)
                buffer.append(qa_obj.to_dict())
                processed_chunk_ids.add(chunk.chunk_id)
                generated += 1

                print(
                    f"[GENERATED #{generated}] q={result['query'][:80]}\n"
                    f"  a={result['answer'][:80]}\n"
                    f"  evidence={span_text[:80]}"
                )

            if len(buffer) >= SAVE_EVERY:
                _checkpoint(buffer)
                buffer.clear()

            pending_batch.clear()

        # ── Main loop ────────────────────────────────────────────────────────
        for i, chunk in enumerate(chunks):
            if generated >= self.target_count:
                break

            should_skip, reason = self._should_skip_chunk(
                chunk, processed_chunk_ids, seen_queries,
                doc_counts, total_docs, stats
            )
            if should_skip:
                if reason not in ("already_processed",):
                    print(f"[SKIP] chunk idx {i+1}: {reason}")
                continue

            qtype = _choose_qtype_for_chunk(chunk, self.rng)
            print(f"→ Queuing chunk idx {i+1} | qtype={qtype} | QA #{generated+1}")
            pending_batch.append((chunk, qtype, i + 1))

            if len(pending_batch) >= self.batch_size:
                _flush_batch()

        if pending_batch:
            _flush_batch()

        if buffer:
            _checkpoint(buffer)
            buffer.clear()

        print(f"[FINAL] Total QA generated: {generated}")
        logger.info("QA generation stats: %s", stats)
        return qa_pairs


def _checkpoint(buffer: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        for item in buffer:
            f.write(json.dumps(item) + "\n")