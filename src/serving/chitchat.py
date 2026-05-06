"""
AegisRAG - Chitchat fast-path detector.

Identifies conversational messages (greetings, thanks, farewells, simple
one-word inputs) that don't need RAG retrieval and returns an instant reply.
Zero LLM calls, zero retrieval — response time is sub-millisecond.
"""

from __future__ import annotations

import random
import re
import time
import uuid
from typing import NamedTuple

from src.data.schema import Citation, QueryResponse


class ChitchatMatch(NamedTuple):
    matched: bool
    reply: str


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_GREETINGS = re.compile(
    r"^\s*"
    r"(hey+(\s+(there|you|friend|buddy))?|hi+(\s+(there|you|friend))?|hello+(\s+(there|friend))?|"
    r"howdy|hiya|sup|what'?s up|yo+|greetings|good\s*(morning|afternoon|evening|day)|"
    r"how are you|how('?re| are) (you|things|it going)|what'?s new|how('?s| is) (it going|everything))"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

_THANKS = re.compile(
    r"^\s*"
    r"(thanks?\s*(a\s*(lot|bunch|ton|million))?|thank\s*you(\s*(so\s*much|very\s*much|a\s*lot))?|"
    r"ty|thx|cheers|much appreciated|appreciate (it|that)|"
    r"that'?s? (great|helpful|perfect|awesome|amazing)|got it|perfect|great|awesome)"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

_FAREWELLS = re.compile(
    r"^\s*"
    r"(bye+|goodbye+|see (you|ya)(\s+later)?|cya|ttyl|take care|have a (good|great|nice) (day|one)|"
    r"talk (to you |to ya )?(later|soon)|later|peace|gotta go|i('?m| am) (done|good|all set|all good))"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

_AFFIRMATIONS = re.compile(
    r"^\s*(ok(ay)?|sure|yep|yup|yes|yeah|nope|no|nah|alright|sounds good|makes sense|understood|i see|i get it|cool)[\s!?.]*$",
    re.IGNORECASE,
)

# Patterns that look like real questions — never short-circuit these even if
# they contain greeting words ("hey how do I reset my password?")
_REAL_QUESTION_SIGNALS = re.compile(
    r"\b(how (do|can|should|to|does)|what (is|are|should|can)|"
    r"where (is|are|can|do)|when (is|are|can|do|should)|"
    r"why (is|are|does|did|would)|who (is|are|can)|"
    r"can you|could you|please|help me|i need|i want|tell me|explain|"
    r"reset|password|account|order|refund|cancel|billing|issue|problem|error|"
    r"not working|broken|failed|unable|can'?t|won'?t|doesn'?t)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Reply pools
# ---------------------------------------------------------------------------

_GREETING_REPLIES = [
    "Hi there! I'm AegisRAG, your support assistant. What can I help you with today?",
    "Hello! How can I assist you today?",
    "Hey! I'm here to help. What do you need?",
    "Hi! What can I do for you?",
]

_THANKS_REPLIES = [
    "You're welcome! Let me know if there's anything else I can help with.",
    "Happy to help! Anything else?",
    "Glad I could assist! Feel free to ask if you have more questions.",
    "No problem! Is there anything else you need?",
]

_FAREWELL_REPLIES = [
    "Goodbye! Have a great day!",
    "Take care! Don't hesitate to reach out if you need anything.",
    "Bye! Hope I was helpful.",
    "See you! Feel free to come back if you have more questions.",
]

_AFFIRMATION_REPLIES = [
    "Got it! Let me know if you have any other questions.",
    "Sure! What else can I help you with?",
    "Understood. Anything else I can assist with?",
    "Of course! Is there anything else you need help with?",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(query: str) -> ChitchatMatch:
    """Return a ChitchatMatch indicating whether the query is chitchat.

    A query is only treated as chitchat when it matches a conversational
    pattern AND does NOT contain any real-question signals (help requests,
    product keywords, interrogative words, etc.).

    Args:
        query: Raw user query string.

    Returns:
        ChitchatMatch with matched=True and a ready reply, or matched=False.
    """
    stripped = query.strip()
    if not stripped:
        return ChitchatMatch(False, "")

    # Never short-circuit if the message contains real question signals
    if _REAL_QUESTION_SIGNALS.search(stripped):
        return ChitchatMatch(False, "")

    if _GREETINGS.match(stripped):
        return ChitchatMatch(True, random.choice(_GREETING_REPLIES))
    if _THANKS.match(stripped):
        return ChitchatMatch(True, random.choice(_THANKS_REPLIES))
    if _FAREWELLS.match(stripped):
        return ChitchatMatch(True, random.choice(_FAREWELL_REPLIES))
    if _AFFIRMATIONS.match(stripped):
        return ChitchatMatch(True, random.choice(_AFFIRMATION_REPLIES))

    return ChitchatMatch(False, "")


def make_response(reply: str, model_tag: str = "") -> QueryResponse:
    """Wrap a chitchat reply in a QueryResponse with near-zero latency."""
    return QueryResponse(
        answer=reply,
        citations=[],
        tool_calls=[],
        confidence=1.0,
        cgal_iterations=0,
        latency_ms=0.5,   # sub-millisecond; placeholder for UI display
        session_id=str(uuid.uuid4()),
        model_tag=model_tag,
    )
