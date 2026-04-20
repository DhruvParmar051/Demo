import json
import random
import re
from typing import Dict, Any

INPUT_FILE = "/Users/dhruvparmar/DAU/sem_2/DS 615/Project/AegisRAG/data/synthetic/qa_pairs copy.jsonl"
OUTPUT_FILE = "/Users/dhruvparmar/DAU/sem_2/DS 615/Project/AegisRAG/data/synthetic/qa_pairs_final.jsonl"

random.seed(42)

# ----------------------------
# 1. Question Diversity
# ----------------------------
def diversify_question(q: str) -> str:
    patterns = [
        ("When can", ["In what situations can", "Under what circumstances can"]),
        ("When must", ["In which cases must"]),
        ("When does", ["In what cases does"]),
        ("Under what condition", ["When", "In what situations"]),
    ]

    for prefix, replacements in patterns:
        if q.startswith(prefix) and random.random() < 0.5:
            return q.replace(prefix, random.choice(replacements), 1)

    return q

def fix_typos(q: str) -> str:
    q = q.replace("Whens", "When")
    q = q.replace("situationss", "situations")
    return q

def is_corrupted(a: str) -> bool:
    return (
        len(a.split()) < 8
        or "must send or provide you" in a.lower()
        or a.endswith("...")
        or "..." in a
    )

def is_complete(a: str) -> bool:
    if len(a.split()) < 12:
        return False

    # must have condition structure
    return any(w in a.lower() for w in ["if", "when", "must", "only", "unless"])

def is_answer_relevant(q: str, a: str) -> bool:
    ql = q.lower()
    al = a.lower()

    if "when" in ql:
        return any(w in al for w in ["if", "when", "must"])

    if "what" in ql:
        return len(a.split()) > 10

    return True

def is_table_answer(a: str) -> bool:
    return sum(c.isdigit() for c in a) > 10

# ----------------------------
# 2. Remove bad questions
# ----------------------------
def is_bad_question(q: str) -> bool:
    ql = q.lower()

    bad_patterns = [
        "worksheet",
        "form line",
        "before printing",
        "table",
        "recapture allocation",
    ]

    return (
        any(p in ql for p in bad_patterns)
        or len(q.split()) < 8
        or q.lower().startswith("why")  # ❌ remove WHY (hallucination source)
    )


# ----------------------------
# 3. No-answer / corrupted
# ----------------------------
def is_no_answer(a: str) -> bool:
    return "does not specify" in a.lower()


def is_corrupted(a: str) -> bool:
    return (
        len(a.strip()) < 8
        or a.endswith("...")
        or "must send or provide you" in a.lower()
    )


# ----------------------------
# 4. Extract chunk
# ----------------------------
def get_chunk_text(qa: Dict[str, Any]) -> str:
    if "citations" in qa and qa["citations"]:
        return qa["citations"][0].get("cited_text", "")
    return ""


# ----------------------------
# 5. Strict grounding (UPGRADED)
# ----------------------------
def is_strictly_grounded(a: str, chunk: str) -> bool:
    if not chunk:
        return True

    a_words = set(re.findall(r"\w+", a.lower()))
    c_words = set(re.findall(r"\w+", chunk.lower()))

    overlap = len(a_words & c_words)

    return overlap >= max(6, int(0.5 * len(a_words)))


# ----------------------------
# 6. Remove hallucinated reasoning
# ----------------------------
def has_fake_reasoning(a: str) -> bool:
    bad_phrases = [
        "to comply",
        "to protect",
        "to ensure",
        "to maintain",
        "to promote",
        "allowing for",
    ]
    return any(p in a.lower() for p in bad_phrases)


# ----------------------------
# 7. Enrich answers
# ----------------------------
def enrich_answer(ans: str, chunk: str) -> str:
    if len(ans.split()) >= 15 or not chunk:
        return ans

    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    for s in sentences:
        if ans[:20].lower() in s.lower():
            return s.strip()

    return ans


# ----------------------------
# 8. Completeness check
# ----------------------------
def is_complete_answer(q: str, a: str) -> bool:
    if len(a.split()) < 12:
        return False

    ql = q.lower()
    al = a.lower()

    if any(w in ql for w in ["when", "under what", "in what"]):
        return any(w in al for w in ["if", "when", "must", "only", "unless"])

    return True


# ----------------------------
# 9. Soft rebalance
# ----------------------------
def keep_sample(qtype: str) -> bool:
    probs = {
        "procedural": 1.0,
        "policy": 1.0,
        "eligibility": 0.85,
        "factoid": 0.3,
    }
    return random.random() < probs.get(qtype, 1.0)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def process_line(line: str):
    qa = json.loads(line)

    q = qa.get("query", "").strip()
    a = qa.get("answer_with_citations", "").strip()
    qtype = qa.get("question_type", "factoid")

    # diversify
    q = diversify_question(q)

    # remove bad questions
    if is_bad_question(q):
        return None

    # remove no-answer / corrupted
    if is_no_answer(a) or is_corrupted(a):
        return None

    chunk = get_chunk_text(qa)

    # remove hallucination reasoning
    if has_fake_reasoning(a):
        return None

    # enrich
    a = enrich_answer(a, chunk)

    # strict grounding
    if not is_strictly_grounded(a, chunk):
        return None

    # completeness
    if not is_complete_answer(q, a):
        return None

    # rebalance
    if not keep_sample(qtype):
        return None

        # fix typos
    q = fix_typos(q)

    # remove corrupted
    if is_corrupted(a):
        return None

    # remove table answers
    if is_table_answer(a):
        return None

    # enforce completeness
    if not is_complete(a):
        return None

    # ensure relevance
    if not is_answer_relevant(q, a):
        return None

    qa["query"] = q
    qa["answer_with_citations"] = a

    return qa


def main():
    total = 0
    kept = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            total += 1
            cleaned = process_line(line)

            if cleaned:
                fout.write(json.dumps(cleaned) + "\n")
                kept += 1

    print("\n✅ FINAL CLEANING COMPLETE")
    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Dropped: {total - kept}")
    print(f"Retention: {kept / total:.2%}")


if __name__ == "__main__":
    main()