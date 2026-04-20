"""AegisRAG - Local-only Synthetic QA Generator.

Produces ``cfg.synthetic_data.qa_pairs`` grounded QA pairs (default 500)
using a purely local teacher (``LocalTeacher``). Strategy:

* 1 pair per chunk, iterate chunks until target is hit.
* Self-consistency: generate 2 candidates per chunk, keep the best via
    cheap heuristic scoring (no extra model calls).
* Batched inference when the backend supports it.

No API calls. No SFT dependencies.
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

QUESTION_TYPES = ("procedural", "policy", "multi_part", "factoid")

CHECKPOINT_FILE = Path("data/synthetic/qa_pairs.jsonl")
SAVE_EVERY = 20

REASONING_MARKERS = (
    "because", "ensures", "prevents", "allows", "therefore", "so that",
    "which means", "in order to", "due to", "results in", "leads to",
    "enables", "avoids", "requires", "depends on", "causes",
)

_CITATION_RE = re.compile(r"\[[A-Za-z0-9_\-]+:\d+\-\d+\]")

# =========================
# HELPERS
# =========================
def is_duplicate(q, existing, threshold=0.9):
    for e in existing:
        if SequenceMatcher(None, q, e).ratio() > threshold:
            return True
    return False

import re

_BAD_QTYPE = {"policy", "multi_part"}

def _clean_answer(ans: str) -> str:
    ans = re.sub(r"\[[^\]]+\]", "", ans)
    return re.sub(r"\s+", " ", ans).strip()

def _strict_grounding(answer: str, chunk_text: str) -> bool:
    answer_words = [
        w for w in re.findall(r"\w+", answer.lower())
        if len(w) > 3
    ]

    if not answer_words:
        return False

    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))

    overlap = sum(1 for w in answer_words if w in chunk_words)

    return overlap / len(answer_words) >= 0.45


def _is_good_question(q: str) -> bool:
    ql = q.lower().strip()

    if "?" not in q:
        return False

    words = q.split()
    if len(words) < 5:
        return False

    # Reject trivial definition-style questions unless they demand reasoning
    trivial_starts = ("what is", "what are", "define", "list")
    if any(ql.startswith(x) for x in trivial_starts):
        # Allow if the question still probes reasoning (why/how/impact)
        if not any(m in ql for m in ("why", "how", "purpose", "role", "impact", "effect", "difference")):
            return False

    return True


_CODE_MARKERS = (
    "select ", "insert ", "update ", "delete ", "create ", "alter ", "drop ",
    "from ", "where ", "group by", "order by", "returning", "pg_", "::",
    "postgres=#", "=>", "#>", "->", "$$", "begin;", "commit;",
)


def _looks_like_code(text: str) -> bool:
    """Detect SQL / shell / code blocks so we don't treat them as tables."""
    tl = text.lower()
    hits = sum(1 for m in _CODE_MARKERS if m in tl)
    return hits >= 2


def _is_table_like(text: str) -> bool:
    """Heuristic table / keyword-dump detector.

    Returns True when the chunk looks like a table, an index, or a bare
    keyword list (not useful for reasoning QA). Tuned to avoid false
    positives on prose that mixes in code examples or numeric citations.
    """
    if not text or not text.strip():
        return True

    text_l = text.lower()

    # Strong signal: repeated keyword rows (reserved-words style tables)
    if text_l.count("reserved") > 10:
        return True

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    

    is_code = _looks_like_code(text)
    word_count = len(text.split())
    terminators = text.count(".") + text.count("!") + text.count("?")

    # Pipe-delimited tables: must dominate AND not look like code output
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    if len(lines) >= 6 and pipe_lines / len(lines) > 0.6 and not is_code:
        return True

    # Tab-delimited tables
    tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2)
    if len(lines) >= 6 and tab_lines / len(lines) > 0.6:
        return True

    # Mostly short lines AND almost no prose sentences -> keyword list.
    # Skip this check if the chunk looks like code (it's legitimate).
    if len(lines) >= 8 and not is_code:
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 3)
        if short_lines / len(lines) > 0.75 and terminators < 3:
            return True

    # Digit-heavy AND not code (guards against numeric tables)
    if len(text) > 300 and not is_code:
        digits = sum(c.isdigit() for c in text)
        if digits / len(text) > 0.40:
            return True

    # Substantial text with almost no prose terminators: only flag if
    # we also don't see code structure (code often lacks '.').
    if word_count > 200 and terminators == 0 and not is_code:
        return True

    return False


# ONLY showing modified parts — everything else remains EXACTLY same

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
        score += 1.0   # 🔥 increased from 0.5

    ans_l = ans.lower()
    marker_hits = sum(1 for w in REASONING_MARKERS if w in ans_l)
    score += min(marker_hits, 3) * 0.75

    if "\n-" in ans or "\n*" in ans or re.search(r"\n\d+\.", ans):
        score -= 0.5

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
    """Detect model-generation glitches like 'replication replication' or 'free free'.

    Allows intentional repeats ('very very', 'had had') by requiring the
    repeated token to be a content word (>3 chars, not a stopword).
    """
    tokens = re.findall(r"\w+", text.lower())
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a == b and len(a) > 3 and a not in _EVIDENCE_STOPWORDS:
            return True
    return False





# =========================
# EVIDENCE
# =========================
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
    if len(s) > 40 and digits / max(len(s), 1) > 0.35:
        return True
    return False


_EVIDENCE_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
    "were", "be", "been", "being", "for", "on", "at", "by", "with", "as",
    "that", "this", "it", "its", "from", "which", "who", "whom", "but",
    "not", "can", "may", "will", "would", "should", "could", "also",
    "such", "any", "all", "some", "these", "those", "has", "have", "had",
    "do", "does", "did", "if", "then", "than", "so", "when", "while",
}


def extract_evidence_fast(answer: str, chunk_text: str):
    """Pick the chunk sentence that best supports the answer.

    Scores on rare-content overlap (IDF-style within the chunk) rather than
    raw token overlap, so common filler words don't dominate. Also rewards
    bigram overlap to avoid picking sentences that merely share nouns.
    """
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.replace("\n", " "))

    ans_tokens = re.findall(r"\w+", answer.lower())
    ans_content = [w for w in ans_tokens if w not in _EVIDENCE_STOPWORDS and len(w) > 2]
    ans_set = set(ans_content)
    ans_bigrams = set(zip(ans_content, ans_content[1:]))

    # Chunk-level document frequency (per sentence) for rarity weighting
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

        # Rarity-weighted overlap: rarer shared words count more
        weighted = 0.0
        for w in set(overlap_words):
            freq = df.get(w, 1)
            weighted += 1.0 + 0.5 * max(0.0, 1.0 - freq / n_sent)

        # Bigram bonus: sentences that share adjacent word pairs with the answer
        sent_bigrams = set(zip(toks, toks[1:]))
        bigram_hits = len(sent_bigrams & ans_bigrams)
        weighted += bigram_hits * 1.5

        # Mild length normalisation
        length_norm = 1.0 + min(len(toks), 40) / 120.0

        score = weighted * length_norm
        scored.append((score, len(overlap_words), s_clean))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))

    if scored:
        return scored[0][2], scored[0][1]

    return "", 0


def extract_evidence_llm(answer: str, chunk_text: str, teacher):
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
    """Pick the first clean, substantive sentence from the chunk."""
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.replace("\n", " "))
    for s in sentences:
        s_clean = _clean_sentence(s)
        if not _sentence_is_noisy(s_clean) and len(s_clean.split()) >= 8:
            return s_clean
    # Last resort: cleaned chunk prefix
    cleaned = _clean_sentence(chunk_text)
    return cleaned[:200]


# =========================
# PROMPT
# =========================
def _build_prompt(chunk: ChunkRecord, qtype: str) -> str:
    style_hint = {
        "procedural": "Focus on HOW a process works or WHY a step is required.",
        "policy": "Focus on WHY a rule exists or WHAT its consequence is.",
        "multi_part": "Ask a question whose answer covers two linked aspects (cause AND effect, or mechanism AND implication).",
        "factoid": "Ask a specific factual question, but require the answer to explain WHY it holds.",
    }.get(qtype, "Focus on reasoning (WHY / HOW).")

    return (
        "You are writing a reasoning-focused QA pair grounded in the passage below.\n\n"
        f"PASSAGE:\n{chunk.text}\n\n"
        f"STYLE: {style_hint}\n\n"

        "QUESTION RULES:\n"
            "Ask a clear question grounded in the passage.\n"
            "- 8-20 words.\n"
            "- Must be answerable from the passage alone.\n\n"

        "ANSWER RULES:\n"
            "- Prefer 2-3 sentences; a single strong sentence is acceptable only if it fully explains cause AND effect.\n"
            "- Sentence 1: the mechanism or main reason.\n"
            "- Sentence 2 (if used): the consequence, implication, or condition.\n"
            "- Ground every claim in the passage; do not invent facts.\n"
            "- Use clear, specific wording. Do NOT produce bullet points or numbered steps.\n"
            "- Aim for 25-60 words total.\n\n"
            "- Use exact phrases from the passage in your answer where possible.\n"
            "- Do not repeat words or produce malformed text.\n"
            "- Do NOT include citations\n"
            "- Use ONLY exact information from passage\n"
            "- If not directly answerable → return null\n"
            "- Use simple English only\n"

        "OUTPUT (strict JSON, one object, no code fences):\n"
            '{"query": "...", "answer": "..."}'
    )


def _parse_one(raw: str):
    if not raw:
        return None
    matches = re.findall(r"\{[\s\S]*?\}", raw)

    for m in matches:
        try:
            obj = json.loads(m)
        except Exception:
            # Try a light repair: collapse internal newlines inside strings
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

# =========================
# NEW: CHECKPOINT LOADER
# =========================

def _load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return [], set(), set()

    data = []
    seen_queries = set()
    used_chunks = set()

    with open(CHECKPOINT_FILE) as f:
        for line in f:
            try:
                obj = json.loads(line)
                data.append(obj)
                if "query" in obj:
                    seen_queries.add(obj["query"])
                if "gold_chunk_ids" in obj:
                    for cid in obj["gold_chunk_ids"]:
                        used_chunks.add(cid)
            except Exception:
                continue

    return data, seen_queries, used_chunks

# =========================
# MAIN
# =========================

class QAGenerator:
    """Generate grounded QA pairs from ingested chunks using a local teacher."""

    def __init__(self, teacher: Any | None = None, seed: int = 42, limit: int | None = None):
        set_seed(seed)
        self.rng = random.Random(seed)
        self.cfg = get_config()
        self.target_count = limit or self.cfg.synthetic_data.qa_pairs
        self._teacher = teacher

    def _get_teacher(self):
        if self._teacher:
            return self._teacher
        from src.data.local_teacher import LocalTeacher
        self._teacher = LocalTeacher(max_new_tokens=220, temperature=0.7)
        return self._teacher

    def _doc_skip_probability(self, count: int, total_docs: int, target: int) -> float:
        """Frequency-aware soft cap.

        Allows more pairs per doc when the corpus is small, and scales the
        skip probability smoothly instead of a hard cliff at 3.
        """
        if total_docs <= 0:
            fair_share = max(3, target // 10)
        else:
            fair_share = max(3, int(1.5 * target / max(total_docs, 1)))

        if count < fair_share:
            return 0.0

        excess = count - fair_share + 1
        return min(0.85, 0.2 + 0.15 * excess)

    def generate(self, chunks: Sequence[ChunkRecord], output_path=None):

        # =========================
        # RESUME LOADING (NEW)
        # =========================
        existing_data = []
        seen_queries = set()
        processed_chunk_ids = set()

        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing_data.append(obj)

                        if "query" in obj:
                            seen_queries.add(obj["query"])

                        if "gold_chunk_ids" in obj:
                            for cid in obj["gold_chunk_ids"]:
                                processed_chunk_ids.add(cid)
                    except Exception:
                        continue

        qa_pairs, buffer = [], []
        teacher = self._get_teacher()

        doc_counts: dict[str, int] = {}

        chunks = list(chunks)

        # =========================
        # DETERMINISTIC ORDER (FIX)
        # =========================
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

            # =========================
            # SKIP PROCESSED CHUNKS (NEW)
            # =========================
            if chunk.chunk_id in processed_chunk_ids:
                continue

            print(f"→ Trying chunk idx {i+1} | target QA #{generated+1}")

            if _is_table_like(chunk.text):
                cleaned_lines = [
                    ln.strip() for ln in chunk.text.split("\n")
                    if len(ln.split()) > 6 and ln.count("|") < 2 and ln.count("\t") < 2
                ]
                cleaned = " ".join(cleaned_lines)
                if len(cleaned.split()) >= 40 and not _is_table_like(cleaned):
                    chunk.text = cleaned
                else:
                    print("[WARN] table-like chunk, trying anyway")

            if len(chunk.text.split()) < 15:
                stats["skip_short"] += 1
                print("[SKIP] short chunk")
                continue

            count = doc_counts.get(chunk.doc_id, 0)
            skip_p = self._doc_skip_probability(count, total_docs, self.target_count)
            if skip_p > 0 and self.rng.random() < skip_p:
                stats["skip_doc"] += 1
                print(f"[SKIP] doc overused (count={count}, p={skip_p:.2f})")
                continue

            qtype = self.rng.choice(QUESTION_TYPES)
            if qtype in _BAD_QTYPE:
                continue

            prompts = [
                _build_prompt(chunk, qtype),
                _build_prompt(chunk, qtype) + "\nRewrite differently."
            ]

            try:
                outputs = teacher.generate_batch(prompts)
            except Exception as e:
                logger.warning("batch generation failed (%s); falling back to single", e)
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
            
            if not valid:
                fallback = [
                    p for p in parsed
                    if p and p.get("answer")
                    and len(p["answer"].split()) > 12
                    and _strict_grounding(p["answer"], chunk.text)
                ]
                if fallback:
                    best = max(fallback, key=lambda x: score_answer(x["answer"]))
                    print("[FALLBACK] accepting weaker candidate", flush=True)
                    valid = [best]



                
            if not valid:
                print("[RETRY] regenerating...", flush=True)

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
                    print("[SKIP] No valid candidates after retry", flush=True)
                    continue


            best = max(valid, key=lambda x: score_answer(x["answer"]))
            if is_duplicate(best["query"], seen_queries):
                stats["skip_dup"] += 1
                print("[SKIP] duplicate (resume-safe)")
                continue

            if not _is_good_question(best["query"]):
                print("[WEAK QUESTION ACCEPTED]")

            seen_queries.add(best["query"])
            doc_counts[chunk.doc_id] = count + 1

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
                    answer_with_marker = answer_with_marker[:-1] + marker + answer_with_marker[-1]
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

            # =========================
            # TRACK STATE (NEW)
            # =========================
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

        if buffer:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "a") as f:
                for item in buffer:
                    f.write(json.dumps(item) + "\n")

        print(f"[FINAL] Total QA in file: {generated}")
        logger.info("QA generation stats: %s", stats)
        return qa_pairs