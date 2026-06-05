"""
sentiment_engine.py
====================
Standalone sentiment analysis module for the NLP Chatbot.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based
model specifically tuned for social-media / conversational text.  No GPU or
heavy model download required; it works out-of-the-box after
`pip install vaderSentiment`.

Public API
----------
    analyzer = SentimentAnalyzer()
    result   = analyzer.analyze("I am so frustrated right now!")

    result.label        →  "NEGATIVE"
    result.compound     →  -0.656    (range: -1.0 … +1.0)
    result.intensity    →  "HIGH"    ("LOW" | "MEDIUM" | "HIGH")
    result.emoji        →  "😟"
    result.prefix       →  "I'm sorry to hear that. "
    result.escalate     →  True      (True when strongly negative)
    result.scores       →  {'neg': 0.5, 'neu': 0.3, 'pos': 0.2, 'compound': -0.656}
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# VADER compound score ranges from -1 (most negative) to +1 (most positive).
# Standard thresholds used in the VADER paper:
#   positive  ≥ +0.05
#   negative  ≤ -0.05
#   neutral   in between
#
# We additionally grade *intensity* within each polarity:
#   HIGH    |compound| > 0.60
#   MEDIUM  |compound| > 0.20
#   LOW     otherwise

POSITIVE_THRESHOLD  = 0.05
NEGATIVE_THRESHOLD  = -0.05
HIGH_INTENSITY      = 0.60
MEDIUM_INTENSITY    = 0.20

# Compound score below which we flag for human/manager escalation
ESCALATION_THRESHOLD = -0.70

SentimentLabel    = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
IntensityLabel    = Literal["LOW", "MEDIUM", "HIGH"]

# ---------------------------------------------------------------------------
# Response prefix pools
# ---------------------------------------------------------------------------
_PREFIXES: dict[str, dict[str, list[str]]] = {
    "POSITIVE": {
        "HIGH": [
            "That's wonderful to hear! 😊 ",
            "Fantastic! Really glad things are going well. 🎉 ",
            "Love the enthusiasm! 😄 ",
        ],
        "MEDIUM": [
            "Great to hear that! ",
            "Glad you're feeling good about it. 😊 ",
            "Sounds positive — happy to help further! ",
        ],
        "LOW": [
            "Thanks for sharing! ",
            "Good to know. ",
            "",   # sometimes no prefix is more natural for mild positivity
        ],
    },
    "NEGATIVE": {
        "HIGH": [
            "I'm really sorry you're going through this. 😟 Let me do my best to help. ",
            "That sounds very frustrating — I sincerely apologise for the difficulty. 😔 ",
            "I completely understand your frustration, and I'm here to help resolve this. ",
        ],
        "MEDIUM": [
            "I'm sorry to hear that. ",
            "I understand that can be frustrating. ",
            "I apologise for any inconvenience — let me help. ",
        ],
        "LOW": [
            "I see — let me look into that for you. ",
            "Noted. I'll do my best to help. ",
            "",
        ],
    },
    "NEUTRAL": {
        "HIGH":   [""],
        "MEDIUM": [""],
        "LOW":    [""],
    },
}

_ESCALATION_NOTES = [
    "\n\n⚠️ If this issue hasn't been resolved to your satisfaction, "
    "I can connect you with a human agent — just say **escalate** or **speak to someone**.",
    "\n\n⚠️ I want to make sure this gets resolved properly. "
    "Would you like me to escalate this to a specialist?",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Holds the full sentiment analysis output for one user message."""

    label:     SentimentLabel
    intensity: IntensityLabel
    compound:  float                      # −1.0 … +1.0
    scores:    dict[str, float] = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        table = {
            ("POSITIVE", "HIGH"):   "😄",
            ("POSITIVE", "MEDIUM"): "😊",
            ("POSITIVE", "LOW"):    "🙂",
            ("NEGATIVE", "HIGH"):   "😟",
            ("NEGATIVE", "MEDIUM"): "😕",
            ("NEGATIVE", "LOW"):    "😐",
            ("NEUTRAL",  "HIGH"):   "😐",
            ("NEUTRAL",  "MEDIUM"): "😐",
            ("NEUTRAL",  "LOW"):    "😐",
        }
        return table.get((self.label, self.intensity), "😐")

    @property
    def escalate(self) -> bool:
        """True when the message is strongly negative — offer human handoff."""
        return self.compound <= ESCALATION_THRESHOLD

    @property
    def prefix(self) -> str:
        """A randomly chosen empathy prefix appropriate for this sentiment."""
        pool = _PREFIXES.get(self.label, {}).get(self.intensity, [""])
        return random.choice(pool)

    @property
    def escalation_note(self) -> str:
        """Escalation suggestion appended to the bot reply when escalate=True."""
        return random.choice(_ESCALATION_NOTES) if self.escalate else ""

    def wrap(self, bot_reply: str) -> str:
        """
        Wrap an existing bot reply with the sentiment-aware prefix and
        (when warranted) an escalation note.

        Usage::
            result  = analyzer.analyze(user_message)
            wrapped = result.wrap(bot_reply_text)
        """
        return f"{self.prefix}{bot_reply}{self.escalation_note}"

    def to_dict(self) -> dict:
        """Serialisable summary — sent to the frontend in the API response."""
        return {
            "label":     self.label,
            "intensity": self.intensity,
            "compound":  round(self.compound, 4),
            "emoji":     self.emoji,
            "escalate":  self.escalate,
        }


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Thread-safe wrapper around VADER.

    The underlying ``SentimentIntensityAnalyzer`` object is stateless after
    construction, so a single instance shared across threads is safe.
    """

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError as exc:
            raise ImportError(
                "vaderSentiment is not installed.\n"
                "Run:  pip install vaderSentiment"
            ) from exc

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyse *text* and return a :class:`SentimentResult`.

        Parameters
        ----------
        text:
            Raw user message (punctuation and capitalisation are meaningful
            to VADER — do not lowercase before passing in).
        """
        scores   = self._vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= POSITIVE_THRESHOLD:
            label: SentimentLabel = "POSITIVE"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        abs_c = abs(compound)
        if abs_c >= HIGH_INTENSITY:
            intensity: IntensityLabel = "HIGH"
        elif abs_c >= MEDIUM_INTENSITY:
            intensity = "MEDIUM"
        else:
            intensity = "LOW"

        return SentimentResult(
            label=label,
            intensity=intensity,
            compound=compound,
            scores=scores,
        )
