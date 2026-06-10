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
import csv
import difflib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

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

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
DATASET_CANDIDATES = (
    PROJECT_DIR / "sentiment_dataset.csv",
    PROJECT_DIR / "sentiment_dataset.json",
    DATA_DIR / "sentiment_dataset.csv",
    DATA_DIR / "sentiment_dataset.json",
)

DATASET_SIMILARITY_THRESHOLD = 0.54
DATASET_NAIVE_BAYES_MARGIN = 0.75

_DEFAULT_COMPOUNDS: dict[tuple[str, str], float] = {
    ("POSITIVE", "HIGH"): 0.75,
    ("POSITIVE", "MEDIUM"): 0.35,
    ("POSITIVE", "LOW"): 0.10,
    ("NEGATIVE", "HIGH"): -0.75,
    ("NEGATIVE", "MEDIUM"): -0.35,
    ("NEGATIVE", "LOW"): -0.10,
    ("NEUTRAL", "HIGH"): 0.0,
    ("NEUTRAL", "MEDIUM"): 0.0,
    ("NEUTRAL", "LOW"): 0.0,
}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "i", "im", "in", "is", "it",
    "just", "me", "my", "of", "on", "or", "so", "that", "the", "this",
    "to", "was", "we", "were", "with", "you", "your",
}


@dataclass(frozen=True)
class SentimentExample:
    text: str
    label: SentimentLabel
    intensity: IntensityLabel
    compound: float
    normalized: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class DatasetPrediction:
    label: SentimentLabel
    intensity: IntensityLabel
    compound: float
    confidence: float
    method: str
    matched_text: Optional[str] = None


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _content_tokens(text: str) -> frozenset[str]:
    normalized = _normalize_text(text).replace("'", "")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return frozenset(t for t in tokens if len(t) > 1 and t not in _STOP_WORDS)


def _default_compound(label: str, intensity: str) -> float:
    return _DEFAULT_COMPOUNDS.get((label, intensity), 0.0)


def _read_dataset_rows(path: Path) -> list[dict[str, str]]:
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                return [dict(row) for row in csv.DictReader(f)]
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError, csv.Error):
        return []
    return []


def _load_sentiment_examples() -> list[SentimentExample]:
    examples: list[SentimentExample] = []
    seen: set[str] = set()

    for path in DATASET_CANDIDATES:
        if not path.exists():
            continue
        for row in _read_dataset_rows(path):
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip().upper()
            intensity = str(row.get("intensity", "")).strip().upper()
            normalized = _normalize_text(text)

            if (
                not text
                or not normalized
                or normalized in seen
                or label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
                or intensity not in {"LOW", "MEDIUM", "HIGH"}
            ):
                continue

            try:
                compound = float(row.get("compound_hint", ""))
            except (TypeError, ValueError):
                compound = _default_compound(label, intensity)

            examples.append(
                SentimentExample(
                    text=text,
                    label=label,  # type: ignore[arg-type]
                    intensity=intensity,  # type: ignore[arg-type]
                    compound=max(-1.0, min(1.0, compound)),
                    normalized=normalized,
                    tokens=_content_tokens(text),
                )
            )
            seen.add(normalized)

    return examples

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
    source:    str = "vader"

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
            "source":    self.source,
        }


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Thread-safe wrapper around VADER.

    The underlying ``SentimentIntensityAnalyzer`` object is stateless after
    construction, so a single instance shared across threads is safe.

    If vaderSentiment is not importable the analyser degrades gracefully:
    ``available`` is set to False and ``analyze()`` returns a neutral result
    instead of raising — so the chatbot continues to work without sentiment.
    """

    def __init__(self) -> None:
        self.available = False
        self._vader = None
        self.examples = _load_sentiment_examples()
        self._example_lookup = {ex.normalized: ex for ex in self.examples}
        self._classes = tuple(_DEFAULT_COMPOUNDS.keys())
        self._class_doc_counts = {key: 0 for key in self._classes}
        self._class_token_counts = {key: {} for key in self._classes}
        self._class_token_totals = {key: 0 for key in self._classes}
        self._class_compounds = {key: [] for key in self._classes}
        self._vocab: set[str] = set()
        self._build_dataset_model()
        try:
            # Attempt to install automatically if missing (silent best-effort).
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self.available = True
        except ImportError:
            import subprocess, sys
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "vaderSentiment", "-q"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                self.available = True
            except Exception:
                # Still unavailable — run in degraded mode (no sentiment).
                self.available = False

    # ------------------------------------------------------------------
    def _build_dataset_model(self) -> None:
        """Build a tiny token model from sentiment_dataset.* examples."""
        for example in self.examples:
            key = (example.label, example.intensity)
            self._class_doc_counts[key] += 1
            self._class_compounds[key].append(example.compound)
            for token in example.tokens:
                counts = self._class_token_counts[key]
                counts[token] = counts.get(token, 0) + 1
                self._class_token_totals[key] += 1
                self._vocab.add(token)

    # ------------------------------------------------------------------
    def _compound_for_class(self, label: str, intensity: str) -> float:
        key = (label, intensity)
        compounds = self._class_compounds.get(key) or []
        if compounds:
            return sum(compounds) / len(compounds)
        return _default_compound(label, intensity)

    # ------------------------------------------------------------------
    def _dataset_result(self, prediction: DatasetPrediction) -> SentimentResult:
        label = prediction.label
        intensity = prediction.intensity
        compound = prediction.compound
        if label == "POSITIVE":
            scores = {
                "neg": 0.0,
                "neu": max(0.0, 1.0 - abs(compound)),
                "pos": abs(compound),
                "compound": compound,
            }
        elif label == "NEGATIVE":
            scores = {
                "neg": abs(compound),
                "neu": max(0.0, 1.0 - abs(compound)),
                "pos": 0.0,
                "compound": compound,
            }
        else:
            scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": compound}
        scores["dataset_confidence"] = round(prediction.confidence, 4)
        return SentimentResult(
            label=label,
            intensity=intensity,
            compound=compound,
            scores=scores,
            source=f"dataset:{prediction.method}",
        )

    # ------------------------------------------------------------------
    def _predict_from_dataset(self, text: str) -> Optional[DatasetPrediction]:
        normalized = _normalize_text(text)
        if not normalized or not self.examples:
            return None

        exact = self._example_lookup.get(normalized)
        if exact:
            return DatasetPrediction(
                label=exact.label,
                intensity=exact.intensity,
                compound=exact.compound,
                confidence=1.0,
                method="exact",
                matched_text=exact.text,
            )

        tokens = _content_tokens(text)
        best_example: Optional[SentimentExample] = None
        best_score = 0.0
        for example in self.examples:
            token_score = 0.0
            if tokens and example.tokens:
                token_score = len(tokens & example.tokens) / len(tokens | example.tokens)
            sequence_score = difflib.SequenceMatcher(
                None, normalized, example.normalized
            ).ratio()
            score = max(token_score, sequence_score * 0.92)
            if score > best_score:
                best_score = score
                best_example = example

        if best_example and best_score >= DATASET_SIMILARITY_THRESHOLD:
            return DatasetPrediction(
                label=best_example.label,
                intensity=best_example.intensity,
                compound=best_example.compound,
                confidence=best_score,
                method="similar",
                matched_text=best_example.text,
            )

        if not tokens or not self._vocab:
            return None

        matched_tokens = sum(1 for token in tokens if token in self._vocab)
        if matched_tokens == 0:
            return None

        vocab_size = max(1, len(self._vocab))
        total_docs = sum(self._class_doc_counts.values())
        active_classes = [key for key in self._classes if self._class_doc_counts[key] > 0]
        if not active_classes:
            return None

        scored: list[tuple[float, tuple[str, str]]] = []
        for key in active_classes:
            prior = (self._class_doc_counts[key] + 1) / (total_docs + len(active_classes))
            score = math.log(prior)
            denominator = self._class_token_totals[key] + vocab_size
            counts = self._class_token_counts[key]
            for token in tokens:
                score += math.log((counts.get(token, 0) + 1) / denominator)
            scored.append((score, key))

        scored.sort(reverse=True, key=lambda item: item[0])
        best_score, (label, intensity) = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else best_score - 2.0
        margin = best_score - second_score
        confidence = 1.0 - math.exp(-max(0.0, margin))

        if confidence < DATASET_NAIVE_BAYES_MARGIN:
            if matched_tokens < 2 or confidence < 0.45:
                return None

        return DatasetPrediction(
            label=label,  # type: ignore[arg-type]
            intensity=intensity,  # type: ignore[arg-type]
            compound=self._compound_for_class(label, intensity),
            confidence=confidence,
            method="naive_bayes",
        )

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyse *text* and return a :class:`SentimentResult`.

        Parameters
        ----------
        text:
            Raw user message (punctuation and capitalisation are meaningful
            to VADER — do not lowercase before passing in).

        Returns a neutral result with zero scores when VADER is unavailable,
        so the chatbot never crashes due to a missing sentiment dependency.
        """
        dataset_prediction = self._predict_from_dataset(text)
        if dataset_prediction and dataset_prediction.method in {"exact", "similar"}:
            return self._dataset_result(dataset_prediction)

        if not self.available or self._vader is None:
            if dataset_prediction:
                return self._dataset_result(dataset_prediction)
            return SentimentResult(
                label="NEUTRAL",
                intensity="LOW",
                compound=0.0,
                scores={"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
                source="unavailable",
            )

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

        vader_result = SentimentResult(
            label=label,
            intensity=intensity,
            compound=compound,
            scores=scores,
        )

        if dataset_prediction:
            if dataset_prediction.confidence >= DATASET_NAIVE_BAYES_MARGIN:
                return self._dataset_result(dataset_prediction)
            if (
                dataset_prediction.label == vader_result.label
                and dataset_prediction.confidence >= 0.45
            ):
                return self._dataset_result(dataset_prediction)

        return vader_result
