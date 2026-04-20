"""AegisRAG - Local-only Synthetic QA Generator (v2 — optimized).

Produces grounded QA pairs using a purely local teacher (LocalTeacher).
Tuned for government policy / regulatory documents (IRS, SSA, CMS, VA, FSA)
as well as general knowledge-base text.

v2 changes: batch generation, pre-scoring, smarter qtype selection, removed
LLM-based evidence extraction, reordered filters cheap→expensive.
See module docstring above for full change log.
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

# ── Question types ────────────────────────────────────────────────────────────
# "policy" and "multi_part" are no longer in _BAD_QTYPE.
# "policy" = best type for regulatory docs (conditions, eligibility, exceptions)
# "eligibility" = new type for who-qualifies / what-disqualifies questions
# "procedural" = how-to-comply, what-steps-to-take
# "factoid" = specific factual lookups (thresholds, deadlines, dollar amounts)
QUESTION_TYPES = ("procedural", "policy", "eligibility", "factoid")

# Nothing is blocked — all four types are valid for gov docs.
_BAD_QTYPE: set[str] = set()

CHECKPOINT_FILE = Path("data/synthetic/qa_pairs.jsonl")
SAVE_EVERY = 20

# OPT-5: default batch size for LLM generation
_GENERATION_BATCH_SIZE = 4

# ── Reasoning markers ─────────────────────────────────────────────────────────
# Original generic markers PLUS regulatory/policy language.
REASONING_MARKERS = (
    # Generic causal
    "because", "ensures", "prevents", "allows", "therefore", "so that",
    "which means", "in order to", "due to", "results in", "leads to",
    "enables", "avoids", "requires", "depends on", "causes",
    # Regulatory / policy language — high signal for gov docs
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

# ── Legal/citation section number pattern ─────────────────────────────────────
# Matches IRS §, CFR, USC, pub.law citation patterns so we don't penalize
# chunks that have lots of § references (high digit density from section nums).
_CITATION_SECTION_RE = re.compile(
    r"(?:§+\s*[\d]+|"           # § 401, §§ 401-402
    r"\b\d+\s*CFR\b|"           # 26 CFR
    r"\bPub\.?\s*L\.?\s*\d+|"   # Pub. L. 117-2
    r"\bU\.S\.C\.?\s*§|"        # U.S.C. §
    r"IRM\s*\d+\.\d+|"          # IRM 4.10.3
    r"Rev\.\s*Proc\.\s*\d+|"    # Rev. Proc. 2023-34
    r"Rev\.\s*Rul\.\s*\d+)",    # Rev. Rul. 2022-1
    re.IGNORECASE,
)

# ── Code markers (for filtering code-heavy chunks) ───────────────────────────
_CODE_MARKERS = (
    "select ", "insert ", "update ", "delete ", "create ", "alter ", "drop ",
    "from ", "where ", "group by", "order by", "returning", "pg_", "::",
    "postgres=#", "=>", "#>", "->", "$$", "begin;", "commit;",
)

# OPT-1: regulatory trigger words for chunk pre-scoring and qtype detection
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


# =============================================================================
# OPT-1: CHUNK PRE-SCORING
# =============================================================================

def _score_chunk_quality(chunk: ChunkRecord) -> float:
    """Fast heuristic score for a chunk before any LLM call.

    Returns a float in [0, 10]. Chunks below a threshold (default 2.0)
    are discarded without calling the LLM.

    Scoring factors (all cheap string ops):
      + word count in sweet spot (40-400 words): up to 2 pts
      + sentence count >= 2: 1 pt
      + contains regulatory trigger words: up to 2 pts
      + contains concrete facts (numbers, dates, $): 1 pt
      + section_title present: 0.5 pt
      - mostly short lines (header/list dump): -1 pt
      - very short chunk (< 25 words): immediate 0
    """
    text = chunk.text or ""
    words = text.split()
    wc = len(words)

    if wc < 25:
        return 0.0

    score = 0.0

    # Word count sweet spot
    if 40 <= wc <= 400:
        score += 2.0
    elif wc < 40:
        score += 0.5
    else:
        score += 1.0  # very long chunks are acceptable but less focused

    # Sentence count
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    n_sentences = len([s for s in sentences if len(s.split()) >= 4])
    if n_sentences >= 3:
        score += 1.0
    elif n_sentences == 2:
        score += 0.5

    # Regulatory trigger words
    text_l = text.lower()
    reg_hits = sum(1 for t in _REG_TRIGGERS_POLICY if t in text_l)
    score += min(reg_hits * 0.3, 1.5)

    # Concrete facts (numbers, dollar amounts, dates)
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

    # Section title bonus
    if chunk.section_title and len(chunk.section_title.strip()) > 2:
        score += 0.5

    # Penalty: mostly short lines (header/keyword dump)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 5:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
        if short_lines / len(lines) > 0.6:
            score -= 1.0

    return max(0.0, score)


# OPT-9: SMARTER QTYPE SELECTION PER CHUNK
def _choose_qtype_for_chunk(chunk: ChunkRecord, rng: random.Random) -> str:
    """Pick the most appropriate question type for a chunk using heuristics.

    Reads chunk text and selects the qtype whose trigger words appear most,
    with a tie-breaker of random selection among equal candidates.
    Falls back to "policy" as the best default for regulatory text.
    """
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

    # Among tied top scorers, pick randomly to get variety
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


# OPT-3: PRE-GENERATION DUPLICATE SIGNAL
def _is_likely_duplicate_chunk(chunk: ChunkRecord, seen_queries: set[str],
                                threshold: float = 0.85) -> bool:
    """Cheap check: does this chunk's first sentence match any seen query?

    Not perfect, but catches the most common case where adjacent chunks
    from the same document produce near-identical questions.
    Avoids an LLM call when the chunk is clearly a duplicate source.
    """
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


def _strict_grounding(answer: str, chunk_text: str) -> bool:
    """Answer must share at least 45% of its content words with the chunk.

    OPT-8: Also checks bigram overlap to be more accurate and reduce
    false rejects from paraphrased answers.
    """
    answer_words = [
        w for w in re.findall(r"\w+", answer.lower())
        if len(w) > 3
    ]
    if not answer_words:
        return False
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    overlap = sum(1 for w in answer_words if w in chunk_words)
    unigram_ratio = overlap / len(answer_words)
    if unigram_ratio >= 0.45:
        return True

    # OPT-8: fallback bigram check — paraphrased answers can still be grounded
    ans_bigrams = list(zip(answer_words, answer_words[1:]))
    chunk_word_list = [w for w in re.findall(r"\w+", chunk_text.lower()) if len(w) > 3]
    chunk_bigrams = set(zip(chunk_word_list, chunk_word_list[1:]))
    if not ans_bigrams:
        return False
    bigram_overlap = sum(1 for bg in ans_bigrams if bg in chunk_bigrams)
    bigram_ratio = bigram_overlap / len(ans_bigrams)
    return bigram_ratio >= 0.30


def _is_good_question(q: str, qtype: str = "") -> bool:
    """Accept a question for inclusion.

    Changes vs original:
      • "what is/are" is now allowed when the question contains regulatory
        trigger words (eligibility, requirement, condition, deadline, etc.)
        because "What is the income limit for..." is a great gov-doc question.
      • "eligibility" type questions get a relaxed rule — the whole point is
        "who qualifies / what are the requirements".
    """
    ql = q.lower().strip()
    if "?" not in q:
        return False
    words = q.split()
    if len(words) < 5:
        return False

    # Regulatory trigger words that make "what is/are" questions acceptable
    _REG_TRIGGERS = (
        "eligib", "qualif", "requir", "condition", "criterion", "criteria",
        "deadline", "limit", "threshold", "penalty", "benefit", "coverage",
        "exclusion", "exception", "entitle", "disqualif", "subject to",
        "must", "shall", "allowed", "permitted", "prohibited",
    )

    trivial_starts = ("what is", "what are", "define", "list")
    if any(ql.startswith(x) for x in trivial_starts):
        # Accept if there's a regulatory trigger OR standard reasoning word
        has_reg = any(t in ql for t in _REG_TRIGGERS)
        has_reasoning = any(
            m in ql for m in ("why", "how", "purpose", "role", "impact",
                               "effect", "difference", "when", "where")
        )
        if qtype == "eligibility":
            return True   # eligibility questions are inherently about requirements
        if not (has_reg or has_reasoning):
            return False

    return True


def _looks_like_code(text: str) -> bool:
    tl = text.lower()
    hits = sum(1 for m in _CODE_MARKERS if m in tl)
    return hits >= 2


# OPT-7: CACHE _is_table_like RESULTS
_table_like_cache: dict[str, bool] = {}


def _is_table_like(text: str, chunk_id: str = "") -> bool:
    """Detect table/schema/keyword-dump chunks that produce bad QA.

    OPT-7: Results are cached by chunk_id to avoid recomputing on
    the same chunk text multiple times.

    Changes vs original:
      • Digit threshold raised from 0.40 → 0.55 to allow legal citation
        paragraphs (e.g. "§ 401(k)(2)(B)(i)(I)") without false-positive.
      • Citation-pattern exemption: if the chunk has 3+ legal citation markers
        it is almost certainly regulatory text, not a data table — keep it.
    """
    # OPT-7: cache hit
    if chunk_id and chunk_id in _table_like_cache:
        return _table_like_cache[chunk_id]

    result = _compute_is_table_like(text)

    if chunk_id:
        _table_like_cache[chunk_id] = result
    return result


def _compute_is_table_like(text: str) -> bool:
    """Internal (uncached) implementation of table-like detection."""
    if not text or not text.strip():
        return True

    text_l = text.lower()

    # Strong signal: repeated keyword rows
    if text_l.count("reserved") > 10:
        return True

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True

    is_code = _looks_like_code(text)
    word_count = len(text.split())
    terminators = text.count(".") + text.count("!") + text.count("?")

    # Citation exemption: legal texts have lots of section numbers but are
    # NOT tables — skip the digit-ratio filter for them.
    citation_count = len(_CITATION_SECTION_RE.findall(text))
    is_regulatory_citation_text = citation_count >= 3

    # Pipe-delimited tables
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    if len(lines) >= 6 and pipe_lines / len(lines) > 0.6 and not is_code:
        return True

    # Tab-delimited tables
    tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2)
    if len(lines) >= 6 and tab_lines / len(lines) > 0.6:
        return True

    # Mostly short lines AND almost no prose — keyword/glossary list
    if len(lines) >= 8 and not is_code:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
        if short_lines / len(lines) > 0.75 and terminators < 3:
            return True

    # Digit-heavy (raised threshold; citation text exempted)
    if len(text) > 300 and not is_code and not is_regulatory_citation_text:
        digits = sum(c.isdigit() for c in text)
        if digits / len(text) > 0.55:   # was 0.40
            return True

    # Lots of words but zero prose terminators (and not code, not citations)
    if word_count > 200 and terminators == 0 and not is_code and not is_regulatory_citation_text:
        return True

    return False


def score_answer(ans: str) -> float:
    """Score an answer candidate for quality.

    Changes vs original:
      • Added regulatory/policy language to bonus scoring.
      • Slightly higher base reward for 2-sentence answers with conditions.
    """
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
    score += min(marker_hits, 4) * 0.65   # slightly lower per-hit to balance larger list

    # Penalty for bullet/numbered lists (not grounded prose style)
    if "\n-" in ans or "\n*" in ans or re.search(r"\n\d+\.", ans):
        score -= 0.5

    # Bonus: answer uses regulatory conditional structure
    # "if ... then", "unless ... ", "provided that ...", "subject to ..."
    conditional_hits = len(re.findall(
        r"\b(?:if|unless|provided that|subject to|except|notwithstanding|"
        r"must|shall|eligible|qualify|require)\b",
        ans_l
    ))
    score += min(conditional_hits, 3) * 0.4

    return score


def _valid_pair(obj: dict, qtype: str, min_answer_tokens: int) -> bool:
    q = str(obj.get("query", "")).strip()
    a = str(obj.get("answer", "")).strip()
    if not q or not a:
        return False
    if len(a.split()) < 5:
        return False
    if _CITATION_RE.search(q):
        return False
    return True


def _has_repetition_artifact(text: str) -> bool:
    """Detect model glitches like 'replication replication'."""
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


# =============================================================================
# EVIDENCE EXTRACTION
# =============================================================================

_EVIDENCE_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
    "were", "be", "been", "being", "for", "on", "at", "by", "with", "as",
    "that", "this", "it", "its", "from", "which", "who", "whom", "but",
    "not", "can", "may", "will", "would", "should", "could", "also",
    "such", "any", "all", "some", "these", "those", "has", "have", "had",
    "do", "does", "did", "if", "then", "than", "so", "when", "while",
}


def _clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _sentence_is_noisy(s: str) -> bool:
    if not s:
        return True
    wc = len(s.split())
    if wc < 5:
        return True
    if "[table]" in s.lower():
        return True
    if len(s.split()) > 40 and s.count("|") > 0:
        return True
    if s.count("|") >= 2 or s.count("\t") >= 2:
        return True
    if s.lower().count("reserved") > 3:
        return True
    digits = sum(c.isdigit() for c in s)
    # Use citation exemption here too — a sentence full of § numbers is fine
    has_section_ref = bool(_CITATION_SECTION_RE.search(s))
    if len(s) > 40 and digits / max(len(s), 1) > 0.35 and not has_section_ref:
        return True
    return False


def extract_evidence_fast(answer: str, chunk_text: str) -> tuple[str, int]:
    """Pick the chunk sentence that best supports the answer."""
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.replace("\n", " "))

    ans_tokens = re.findall(r"\w+", answer.lower())
    ans_content = [w for w in ans_tokens if w not in _EVIDENCE_STOPWORDS and len(w) > 2]
    ans_set = set(ans_content)
    ans_bigrams = set(zip(ans_content, ans_content[1:]))

    sentence_tokens = []
    df: dict[str, int] = {}
    for s in sentences:
        s_clean = _clean_sentence(s)
        toks = [w for w in re.findall(r"\w+", s_clean.lower())
                if w not in _EVIDENCE_STOPWORDS and len(w) > 2]
        sentence_tokens.append((s_clean, toks))
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    n_sent = max(len(sentence_tokens), 1)

    scored = []
    for s_clean, toks in sentence_tokens:
        if not toks or _sentence_is_noisy(s_clean):
            continue
        overlap_words = [w for w in toks if w in ans_set]
        if not overlap_words:
            continue

        weighted = 0.0
        for w in set(overlap_words):
            freq = df.get(w, 1)
            weighted += 1.0 + 0.5 * max(0.0, 1.0 - freq / n_sent)

        sent_bigrams = set(zip(toks, toks[1:]))
        bigram_hits = len(sent_bigrams & ans_bigrams)
        weighted += bigram_hits * 1.5

        length_norm = 1.0 + min(len(toks), 40) / 120.0
        score = weighted * length_norm
        scored.append((score, len(overlap_words), s_clean))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))

    if scored:
        return scored[0][2], scored[0][1]
    return "", 0


# OPT-4: REMOVED extract_evidence_llm — replaced with direct fallback
def _fallback_evidence(chunk_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.replace("\n", " "))
    for s in sentences:
        s_clean = _clean_sentence(s)
        if not _sentence_is_noisy(s_clean) and len(s_clean.split()) >= 8:
            return s_clean
    cleaned = _clean_sentence(chunk_text)
    return cleaned[:200]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def _build_prompt(chunk: ChunkRecord, qtype: str) -> str:
    """Build a QA generation prompt tuned for government policy documents.

    Changes vs original:
      • Added "policy" and "eligibility" question types with gov-specific
        style hints (conditions, eligibility, exceptions, deadlines).
      • Prompt explicitly tells the model to quote exact regulatory language
        in the answer where possible — this increases grounding scores.
      • Added explicit instruction NOT to invent thresholds, dollar amounts,
        or dates that aren't in the passage (a common failure mode with gov docs).
    """
    style_hint = {
        "procedural": (
            "Ask HOW a person complies with a rule, files a form, or completes "
            "a process. The answer must explain the required steps and WHY each "
            "step matters."
        ),
        "policy": (
            "Ask WHY a rule or policy exists, WHAT its consequence is if violated, "
            "or HOW it interacts with another rule. The answer must explain the "
            "policy rationale and its practical effect. This is the preferred type "
            "for regulatory and benefits text."
        ),
        "eligibility": (
            "Ask WHO qualifies for a benefit, program, or treatment — or what "
            "conditions DISQUALIFY someone. The answer must name the specific "
            "eligibility criteria, income/age/time thresholds, or exclusions from "
            "the passage. Do NOT invent numbers that are not in the passage."
        ),
        "factoid": (
            "Ask a specific factual question whose answer is a precise threshold, "
            "deadline, dollar amount, time limit, or named requirement from the "
            "passage. The answer must state the exact value and explain what it "
            "means in practice."
        ),
    }.get(qtype, "Focus on conditions, requirements, or consequences.")

    return (
        "You are writing a high-quality QA pair grounded in the policy "
        "passage below. This passage is from an official regulatory or benefits "
        f"PASSAGE:\n{chunk.text}\n\n"
        f"STYLE: {style_hint}\n\n"
        "QUESTION RULES:\n"
        "- Ask a clear, specific question answerable ONLY from this passage.\n"
        "- 8-25 words. End with a question mark.\n"
        "- Do NOT ask about section numbers, form numbers, or publication names.\n"
        "- Prefer questions that probe conditions, eligibility, deadlines, or "
        "- consequences — not just definitions.\n\n"
        "ANSWER RULES:\n"
        "- 2-3 sentences. A single strong sentence is acceptable only if it "
        "  fully states the condition AND its effect.\n"
        "- Use exact regulatory language from the passage where possible "
        "  (e.g. 'must', 'shall', 'is not eligible', 'provided that').\n"
        "- State the specific condition or requirement, then its consequence "
        "  or rationale.\n"
        "- Do NOT invent dollar amounts, percentages, dates, or thresholds "
        "  that are not explicitly stated in the passage.\n"
        "- Do NOT produce bullet points, numbered lists, or section citations.\n"
        "- Aim for 20-60 words.\n"
        "- Use simple, clear English.\n\n"
        "OUTPUT (strict JSON, one object, no code fences):\n"
        '{"query": "...", "answer": "..."}'
    )


def _build_retry_prompt(chunk: ChunkRecord, qtype: str) -> str:
    """OPT-2: Targeted retry prompt — mutates style hint only, keeps same chunk.

    This is cheaper than generating two independent prompts upfront because
    we only call this when the first prompt fails. The style mutation steers
    the model toward a different angle without wasting tokens on a fresh prompt
    when the first might have succeeded.
    """
    # Rotate to a different style emphasis for the retry
    alt_hints = {
        "procedural": "eligibility",
        "policy": "factoid",
        "eligibility": "policy",
        "factoid": "procedural",
    }
    alt_qtype = alt_hints.get(qtype, "policy")
    return _build_prompt(chunk, alt_qtype)


def _parse_one(raw: str) -> dict | None:
    if not raw:
        return None
    matches = re.findall(r"\{[\s\S]*?\}", raw)
    for m in matches:
        try:
            obj = json.loads(m)
        except Exception:
            try:
                repaired = re.sub(r"\n+", " ", m)
                obj = json.loads(repaired)
            except Exception:
                continue
        if isinstance(obj, dict):
            q = obj.get("query", "").strip()
            a = obj.get("answer", "").strip()
            if q:
                return {"query": q, "answer": a}
    return None


def _validate_candidate(p: dict | None, chunk_text: str) -> dict | None:
    """Run all post-generation quality checks on a single candidate.

    Centralizing validation logic here (OPT-6) means we only call it once
    per candidate instead of scattering checks across nested loops.

    Returns the cleaned candidate dict if valid, None otherwise.
    """
    if not p:
        return None
    p["answer"] = _clean_answer(p["answer"])
    if not p["answer"]:
        return None
    if not _valid_pair(p, "", 5):
        return None
    if _has_repetition_artifact(p["answer"]) and len(p["answer"].split()) < 15:
        return None
    if not _strict_grounding(p["answer"], chunk_text):
        return None
    return p


# =============================================================================
# OPT-5: BATCH PROCESSING HELPERS
# =============================================================================

def _process_batch(
    batch: list[tuple[ChunkRecord, str]],
    teacher: Any,
    seen_queries: set[str],
    doc_counts: dict[str, int],
    stats: dict[str, int],
) -> list[dict]:
    """Generate QA for a batch of (chunk, qtype) pairs in one LLM call.

    Returns a list of valid raw QA dicts (before evidence extraction).
    Each dict has keys: query, answer, chunk_id, doc_id, chunk (ChunkRecord), qtype.

    OPT-5: Batching allows the LLM backend (especially vLLM or multi-GPU HF)
    to process multiple prompts in parallel, improving GPU utilization.
    """
    if not batch:
        return []

    prompts = [_build_prompt(chunk, qtype) for chunk, qtype in batch]

    try:
        outputs = teacher.generate_batch(prompts)
    except Exception as e:
        logger.warning("Batch generation failed (%s); skipping batch", e)
        return []

    results = []
    retry_needed: list[tuple[int, ChunkRecord, str]] = []

    for idx, (output, (chunk, qtype)) in enumerate(zip(outputs, batch)):
        parsed = _parse_one(output)
        candidate = _validate_candidate(parsed, chunk.text)

        if candidate and not is_duplicate(candidate["query"], seen_queries):
            candidate["chunk"] = chunk
            candidate["qtype"] = qtype
            results.append(candidate)
        else:
            if candidate and is_duplicate(candidate["query"], seen_queries):
                stats["skip_dup"] += 1
            else:
                # Mark for retry with a different prompt angle
                retry_needed.append((idx, chunk, qtype))

    # OPT-2: Single targeted retry for failed items (not a full re-batch)
    if retry_needed:
        retry_prompts = [_build_retry_prompt(chunk, qtype)
                         for _, chunk, qtype in retry_needed]
        try:
            retry_outputs = teacher.generate_batch(retry_prompts)
        except Exception as e:
            logger.warning("Retry batch failed (%s); skipping retries", e)
            retry_outputs = [""] * len(retry_prompts)

        for (_, chunk, qtype), retry_output in zip(retry_needed, retry_outputs):
            parsed = _parse_one(retry_output)
            candidate = _validate_candidate(parsed, chunk.text)

            if candidate and not is_duplicate(candidate["query"], seen_queries):
                candidate["chunk"] = chunk
                candidate["qtype"] = qtype
                results.append(candidate)
            else:
                stats["skip_invalid"] += 1

    return results


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class QAGenerator:
    """Generate grounded QA pairs from ingested chunks using a local teacher.

    v2: Batch generation, pre-scoring, smarter qtype selection, removed
    LLM-based evidence extraction, reordered filters cheap→expensive.
    See module docstring for full optimization change log.
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
        self.target_count = limit or self.cfg.synthetic_data.qa_pairs
        self._teacher = teacher
        self.batch_size = int(batch_size)

    def _get_teacher(self) -> Any:
        if self._teacher:
            return self._teacher
        from src.data.local_teacher import LocalTeacher
        self._teacher = LocalTeacher(max_new_tokens=220, temperature=0.7)
        return self._teacher

    def _doc_skip_probability(
        self, count: int, total_docs: int, target: int
    ) -> float:
        if total_docs <= 0:
            fair_share = max(3, target // 10)
        else:
            fair_share = max(3, int(1.5 * target / max(total_docs, 1)))
        if count < fair_share:
            return 0.0
        excess = count - fair_share + 1
        return min(0.85, 0.2 + 0.15 * excess)

    # ------------------------------------------------------------------
    # OPT-6: FILTER ORDER (cheap → expensive), all checks before LLM call
    # ------------------------------------------------------------------

    def _should_skip_chunk(
        self,
        chunk: ChunkRecord,
        processed_chunk_ids: set[str],
        seen_queries: set[str],
        doc_counts: dict[str, int],
        total_docs: int,
        stats: dict[str, int],
    ) -> tuple[bool, str]:
        """Run all pre-LLM checks in order from cheapest to most expensive.

        Returns (should_skip, reason_string).
        OPT-6: Cheap checks first means most chunks are rejected without
        touching the LLM.
        """
        # 1. chunk_id seen — O(1)
        if chunk.chunk_id in processed_chunk_ids:
            return True, "already_processed"

        # 2. Word count — O(n) split
        if len((chunk.text or "").split()) < 15:
            stats["skip_short"] += 1
            return True, "short"

        # 3. Table-like detection — O(n) string scan, cached by chunk_id
        if _is_table_like(chunk.text, chunk_id=chunk.chunk_id):
            # Try to salvage by extracting clean prose lines
            cleaned_lines = [
                ln.strip() for ln in chunk.text.split("\n")
                if len(ln.split()) > 6
                and ln.count("|") < 2
                and ln.count("\t") < 2
            ]
            cleaned = " ".join(cleaned_lines)
            if len(cleaned.split()) >= 40 and not _is_table_like(cleaned):
                chunk.text = cleaned
                # Update cache with cleaned result
                _table_like_cache[chunk.chunk_id] = False
            else:
                stats["skip_table"] += 1
                return True, "table_like"

        # 4. Pre-score — O(n) string ops, no model
        quality_score = _score_chunk_quality(chunk)
        if quality_score < 2.0:
            stats["skip_lowscore"] = stats.get("skip_lowscore", 0) + 1
            return True, f"low_quality_score={quality_score:.2f}"

        # 5. Likely duplicate chunk — O(m) set scan
        if _is_likely_duplicate_chunk(chunk, seen_queries):
            stats["skip_dup"] += 1
            return True, "likely_duplicate_chunk"

        # 6. Doc frequency cap — O(1)
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

        # ── OPT-5: Accumulate batches, process together ─────────────────────
        pending_batch: list[tuple[ChunkRecord, str]] = []

        def _flush_batch() -> None:
            """Process the current pending batch and commit results."""
            nonlocal generated
            if not pending_batch or generated >= self.target_count:
                return

            valid_results = _process_batch(
                pending_batch, teacher, seen_queries, doc_counts, stats
            )

            for result in valid_results:
                if generated >= self.target_count:
                    break

                chunk: ChunkRecord = result.pop("chunk")
                qtype: str = result.pop("qtype")

                # Final duplicate check (post-generation, authoritative)
                if is_duplicate(result["query"], seen_queries):
                    stats["skip_dup"] += 1
                    continue

                seen_queries.add(result["query"])
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

                # ── Evidence extraction (fast only, OPT-4) ──────────────────
                span_text, conf = extract_evidence_fast(result["answer"], chunk.text)

                # OPT-4: Skip LLM evidence extraction — use fallback directly
                # when fast extraction gives weak results.
                if conf < 3 or len(span_text.split()) < 6:
                    span_text = _fallback_evidence(chunk.text)

                span_text = _clean_sentence(span_text)
                if len(span_text.split()) < 8 or _sentence_is_noisy(span_text):
                    span_text = _fallback_evidence(chunk.text)
                if _sentence_is_noisy(span_text):
                    span_text = _fallback_evidence(chunk.text)

                # ── Span location ───────────────────────────────────────────
                start = chunk.text.lower().find(span_text.lower())
                if start == -1:
                    start = 0
                    end = len(span_text)
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
                        answer_with_marker = answer_with_marker + marker

                qa_obj = QAPair(
                    query=result["query"],
                    answer_with_citations=answer_with_marker,
                    gold_chunk_ids=[chunk.chunk_id],
                    question_type=qtype,
                    citations=[citation],
                )

                qa_dict = qa_obj.to_dict()
                qa_pairs.append(qa_obj)
                buffer.append(qa_dict)
                processed_chunk_ids.add(chunk.chunk_id)
                generated += 1

                print(
                    f"[TOTAL GENERATED]: {generated}\n"
                    f"Query: {result['query']}\n"
                    f"Answer: {result['answer']}\n"
                    f"Evidence: {span_text}\n"
                )

            # Checkpoint if buffer is large enough
            if len(buffer) >= SAVE_EVERY:
                _checkpoint(buffer)
                buffer.clear()

            pending_batch.clear()

        # ── Main loop: pre-filter then batch ────────────────────────────────
        for i, chunk in enumerate(chunks):
            if generated >= self.target_count:
                break

            # OPT-6: All cheap filters before touching the LLM
            should_skip, reason = self._should_skip_chunk(
                chunk, processed_chunk_ids, seen_queries,
                doc_counts, total_docs, stats
            )
            if should_skip:
                if reason not in ("already_processed",):
                    print(f"[SKIP] chunk idx {i+1}: {reason}")
                continue

            # OPT-9: Smart qtype selection per chunk
            qtype = _choose_qtype_for_chunk(chunk, self.rng)

            print(f"→ Queuing chunk idx {i+1} | qtype={qtype} | target QA #{generated+1}")
            pending_batch.append((chunk, qtype))

            # OPT-5: Process batch when full
            if len(pending_batch) >= self.batch_size:
                _flush_batch()

        # Flush any remaining items
        if pending_batch:
            _flush_batch()

        # Flush any remaining buffer items to disk
        if buffer:
            _checkpoint(buffer)
            buffer.clear()

        print(f"[FINAL] Total QA in file: {generated}")
        logger.info("QA generation stats: %s", stats)
        return qa_pairs


def _checkpoint(buffer: list[dict]) -> None:
    """Append a buffer of QA dicts to the checkpoint JSONL file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        for item in buffer:
            f.write(json.dumps(item) + "\n")