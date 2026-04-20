"""
PATCH NOTES — what changed and why
===================================

Files affected:
  1. src/data/qa_generator.py   — drop-in replacement
  2. src/data/ingestion.py      — one config change (chunk_overlap)
  3. run.py / ingest CLI        — no change needed

──────────────────────────────────────────────────────────────────────────────
WHY THESE CHANGES WERE NEEDED
──────────────────────────────────────────────────────────────────────────────

Your new docs (IRS, SSA, CMS, VA, FSA) share a specific structure that your
original generator was not tuned for:

  • Long regulatory paragraphs with embedded conditions:
      "If the taxpayer ... unless ... except when ..."
  • Eligibility chains:
      "To qualify you must (a) ... (b) ... (c) ..."
  • Procedure descriptions:
      "You must file within 30 days of ... by submitting Form ..."
  • Exception/limitation clauses:
      "This benefit does not apply to ... unless ..."

The original code had three problems that caused high skip rates and low
quality answers on these docs:

PROBLEM 1 — _BAD_QTYPE skipped "policy" questions
  "policy" was in _BAD_QTYPE, so every attempt to generate a policy-style
  question was immediately discarded. But policy questions ARE the best type
  for regulation docs ("Why does the IRS require...", "What condition triggers...").

PROBLEM 2 — _is_good_question blocked useful government questions
  The original blocked "what is/are" questions unless they contained
  "why/how/purpose/role/impact". But many high-value government questions
  ARE "what is the eligibility requirement for X" or "what are the conditions
  that disqualify a person from Y". The fix: allow "what" questions when they
  contain regulatory trigger words (eligibility, requirement, condition, etc.)

PROBLEM 3 — _is_table_like was too aggressive on government text
  IRS/CMS docs contain legal-citation paragraphs like:
    "See § 401(k)(2)(B)(i)(I) and Reg. 1.401(k)-1(a)(4)(ii)."
  These have high digit density from section numbers, which triggered the
  digit_ratio > 0.40 filter and caused the chunk to be skipped entirely.
  Fix: raise digit threshold to 0.55 and add a citation-pattern exemption.

PROBLEM 4 — REASONING_MARKERS list was generic
  Words like "because", "therefore" appear in all text. Government docs
  use specific regulatory language: "eligible", "subject to", "pursuant to",
  "provided that", "shall", "notwithstanding". Adding these markers means
  answers that use regulatory language get properly rewarded.

PROBLEM 5 — Prompt didn't leverage the document structure
  Generic prompts produced answers that could have come from any text.
  Policy-specific prompts that instruct the model to focus on conditions,
  eligibility criteria, deadlines, and exceptions produce much more
  grounded, citable, useful answers.

PROBLEM 6 — chunk_overlap too small for long regulatory paragraphs
  Government regulations often state a rule in one sentence and the exception
  in the next, sometimes with a paragraph boundary between them. With only
  64-token overlap, the exception could be split from its rule. 96 tokens
  of overlap keeps related clauses together more reliably.

──────────────────────────────────────────────────────────────────────────────
HOW TO APPLY
──────────────────────────────────────────────────────────────────────────────

1. Replace src/data/qa_generator.py with the version in this file.

2. In src/data/ingestion.py, change the RecursiveChunker instantiation:
   BEFORE:  self.chunker = chunker or RecursiveChunker()
   AFTER:   self.chunker = chunker or RecursiveChunker(chunk_overlap=96)
   
   OR set in config/base.yaml:
   BEFORE:  chunk_overlap: 64
   AFTER:   chunk_overlap: 96

3. Re-ingest your new docs (IRS/SSA/CMS/VA/FSA) into a fresh or existing
   ChromaDB collection. Your existing ingested docs are unaffected.

4. Run qa generation as normal:
   python run.py generate-data --type qa --output-dir data/synthetic

──────────────────────────────────────────────────────────────────────────────
WHAT YOU DO NOT NEED TO CHANGE
──────────────────────────────────────────────────────────────────────────────
  - ingestion.py parsing logic (PDFParser handles these fine)
  - chunker.py (only chunk_overlap changes, not the algorithm)
  - preference_generator.py (works on QA pairs, not doc type)
  - confidence_label_generator.py (model-agnostic)
  - Any training code
"""

# =============================================================================
# REPLACEMENT: src/data/qa_generator.py
# Drop this file in at src/data/qa_generator.py exactly as-is.
# =============================================================================

"""AegisRAG - Local-only Synthetic QA Generator.

Produces grounded QA pairs using a purely local teacher (LocalTeacher).
Tuned for government policy / regulatory documents (IRS, SSA, CMS, VA, FSA)
as well as general knowledge-base text.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Sequence
from difflib import SequenceMatcher

from src.data.schema import ChunkRecord, QAPair, Citation
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


# =============================================================================
# HELPERS
# =============================================================================

def is_duplicate(q: str, existing: set[str], threshold: float = 0.9) -> bool:
    for e in existing:
        if SequenceMatcher(None, q, e).ratio() > threshold:
            return True
    return False


def _clean_answer(ans: str) -> str:
    ans = re.sub(r"\[[^\]]+\]", "", ans)
    return re.sub(r"\s+", " ", ans).strip()


def _strict_grounding(answer: str, chunk_text: str) -> bool:
    """Answer must share at least 45% of its content words with the chunk."""
    answer_words = [
        w for w in re.findall(r"\w+", answer.lower())
        if len(w) > 3
    ]
    if not answer_words:
        return False
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    overlap = sum(1 for w in answer_words if w in chunk_words)
    return overlap / len(answer_words) >= 0.45


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


def _is_table_like(text: str) -> bool:
    """Detect table/schema/keyword-dump chunks that produce bad QA.

    Changes vs original:
      • Digit threshold raised from 0.40 → 0.55 to allow legal citation
        paragraphs (e.g. "§ 401(k)(2)(B)(i)(I)") without false-positive.
      • Citation-pattern exemption: if the chunk has 3+ legal citation markers
        it is almost certainly regulatory text, not a data table — keep it.
    """
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


def extract_evidence_llm(answer: str, chunk_text: str, teacher: Any) -> str:
    prompt = (
        "Extract ONE exact sentence from the passage that best supports the answer.\n"
        "Return only the sentence, no quotes or commentary.\n\n"
        f"PASSAGE:\n{chunk_text}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        out = teacher.generate_batch([prompt])[0].strip()
        out = _clean_sentence(out)
        if _sentence_is_noisy(out):
            return ""
        return out
    except Exception:
        return ""


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
        "You are writing a high-quality QA pair grounded in the government policy "
        "passage below. This passage is from an official regulatory or benefits "
        "document (IRS, SSA, Medicare, VA, or Federal Student Aid).\n\n"
        f"PASSAGE:\n{chunk.text}\n\n"
        f"STYLE: {style_hint}\n\n"
        "QUESTION RULES:\n"
        "- Ask a clear, specific question answerable ONLY from this passage.\n"
        "- 8-25 words. End with a question mark.\n"
        "- Do NOT ask about section numbers, form numbers, or publication names.\n"
        "- Prefer questions that probe conditions, eligibility, deadlines, or "
        "  consequences — not just definitions.\n\n"
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


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class QAGenerator:
    """Generate grounded QA pairs from ingested chunks using a local teacher.

    Drop-in replacement for the original — same public API, tuned internals.
    """

    def __init__(
        self,
        teacher: Any | None = None,
        seed: int = 42,
        limit: int | None = None,
    ) -> None:
        set_seed(seed)
        self.rng = random.Random(seed)
        self.cfg = get_config()
        self.target_count = limit or self.cfg.synthetic_data.qa_pairs
        self._teacher = teacher

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

        stats = {
            "skip_table": 0, "skip_short": 0, "skip_doc": 0,
            "skip_invalid": 0, "skip_lowscore": 0,
            "skip_dup": 0, "skip_weakq": 0,
        }

        for i, chunk in enumerate(chunks):
            if generated >= self.target_count:
                break

            if chunk.chunk_id in processed_chunk_ids:
                continue

            print(f"→ Trying chunk idx {i+1} | target QA #{generated+1}")

            # ── Table/noise filter ──────────────────────────────────────────
            if _is_table_like(chunk.text):
                # Try to salvage the chunk by extracting clean prose lines
                cleaned_lines = [
                    ln.strip() for ln in chunk.text.split("\n")
                    if len(ln.split()) > 6
                    and ln.count("|") < 2
                    and ln.count("\t") < 2
                ]
                cleaned = " ".join(cleaned_lines)
                if len(cleaned.split()) >= 40 and not _is_table_like(cleaned):
                    chunk.text = cleaned
                else:
                    stats["skip_table"] += 1
                    print("[SKIP] table-like chunk, skipping")
                    continue

            if len(chunk.text.split()) < 15:
                stats["skip_short"] += 1
                print("[SKIP] short chunk")
                continue

            # ── Per-doc frequency cap ───────────────────────────────────────
            count = doc_counts.get(chunk.doc_id, 0)
            skip_p = self._doc_skip_probability(count, total_docs, self.target_count)
            if skip_p > 0 and self.rng.random() < skip_p:
                stats["skip_doc"] += 1
                print(f"[SKIP] doc overused (count={count}, p={skip_p:.2f})")
                continue

            qtype = self.rng.choice(QUESTION_TYPES)

            # ── Generate two candidates and pick the best ───────────────────
            prompts = [
                _build_prompt(chunk, qtype),
                _build_prompt(chunk, qtype) + "\nRewrite differently.",
            ]

            try:
                outputs = teacher.generate_batch(prompts)
            except Exception as e:
                logger.warning("batch generation failed (%s); falling back", e)
                outputs = teacher.generate_batch([prompts[0]])

            parsed = [_parse_one(o) for o in outputs]

            valid = []
            for p in parsed:
                if not p:
                    continue
                p["answer"] = _clean_answer(p["answer"])
                if not p["answer"]:
                    continue
                if not _valid_pair(p, qtype, 5):
                    continue
                if _has_repetition_artifact(p["answer"]) and len(p["answer"].split()) < 15:
                    continue
                if not _strict_grounding(p["answer"], chunk.text):
                    continue
                valid.append(p)

            # ── Fallback: accept lower-bar candidates ───────────────────────
            if not valid:
                fallback = [
                    p for p in parsed
                    if p and p.get("answer")
                    and len(p["answer"].split()) > 12
                    and _strict_grounding(p["answer"], chunk.text)
                ]
                if fallback:
                    best = max(fallback, key=lambda x: score_answer(x["answer"]))
                    print("[FALLBACK] accepting weaker candidate")
                    valid = [best]

            # ── One retry on total failure ──────────────────────────────────
            if not valid:
                print("[RETRY] regenerating...")
                try:
                    outputs = teacher.generate_batch(prompts)
                except Exception:
                    continue
                parsed = [_parse_one(o) for o in outputs]
                valid = []
                for p in parsed:
                    if not p:
                        continue
                    p["answer"] = _clean_answer(p["answer"])
                    if not p["answer"]:
                        continue
                    if not _valid_pair(p, qtype, 5):
                        continue
                    if _has_repetition_artifact(p["answer"]) and len(p["answer"].split()) < 15:
                        continue
                    if not _strict_grounding(p["answer"], chunk.text):
                        continue
                    if len(p["answer"].split()) < 10:
                        continue
                    valid.append(p)

                if not valid:
                    print("[SKIP] No valid candidates after retry")
                    continue

            # ── Pick best, check duplicate ──────────────────────────────────
            best = max(valid, key=lambda x: score_answer(x["answer"]))

            if is_duplicate(best["query"], seen_queries):
                stats["skip_dup"] += 1
                print("[SKIP] duplicate")
                continue

            if not _is_good_question(best["query"], qtype):
                print("[WEAK QUESTION ACCEPTED]")  # log but don't skip

            seen_queries.add(best["query"])
            doc_counts[chunk.doc_id] = count + 1

            # ── Evidence extraction ─────────────────────────────────────────
            span_text, conf = extract_evidence_fast(best["answer"], chunk.text)

            if conf < 3 or len(span_text.split()) < 6:
                llm_span = extract_evidence_llm(best["answer"], chunk.text, teacher)
                if llm_span and len(llm_span.split()) >= 6 and not _sentence_is_noisy(llm_span):
                    span_text = llm_span

            span_text = _clean_sentence(span_text)

            if len(span_text.split()) < 8 or _sentence_is_noisy(span_text):
                span_text = _fallback_evidence(chunk.text)

            if _sentence_is_noisy(span_text):
                span_text = _fallback_evidence(chunk.text)

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

            answer_with_marker = best["answer"].rstrip()
            marker = f" [{chunk.doc_id}:{start}-{end}]"
            if marker.strip() not in answer_with_marker:
                if answer_with_marker.endswith((".", "!", "?")):
                    answer_with_marker = (
                        answer_with_marker[:-1] + marker + answer_with_marker[-1]
                    )
                else:
                    answer_with_marker = answer_with_marker + marker

            qa_obj = QAPair(
                query=best["query"],
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
                f"Query: {best['query']}\n"
                f"Answer: {best['answer']}\n"
                f"Evidence: {span_text}\n"
            )

            if len(buffer) >= SAVE_EVERY:
                CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CHECKPOINT_FILE, "a") as f:
                    for item in buffer:
                        f.write(json.dumps(item) + "\n")
                buffer = []

        # ── Flush remaining buffer ──────────────────────────────────────────
        if buffer:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "a") as f:
                for item in buffer:
                    f.write(json.dumps(item) + "\n")

        print(f"[FINAL] Total QA in file: {generated}")
        logger.info("QA generation stats: %s", stats)
        return qa_pairs