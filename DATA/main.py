"""
main.py
=======
Core chatbot engine — NLP intent classification + sentiment-aware responses.

Changes in this version
-----------------------
* ``ChatbotReply`` carries a new ``sentiment`` field (a :class:`SentimentResult`)
  so that every layer (web app, desktop UI, CLI) can access sentiment metadata.
* ``ChatbotEngine`` now owns a :class:`SentimentAnalyzer` instance.
* ``reply()`` runs sentiment analysis on every user message and:
    - prepends an empathy prefix matched to label + intensity
    - appends an escalation note when the compound score is ≤ −0.70
    - logs each exchange to ``sentiment_log.jsonl`` for evaluation / review
* A new ``/mood`` command lets users check their detected sentiment live.
* All existing functionality (commands, NLP model, training) is unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from sentiment_engine import SentimentAnalyzer, SentimentResult


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = Path(__file__).resolve().parent
INTENTS_FILE = DATA_DIR / "intents.json"
WORDS_FILE   = DATA_DIR / "words.pkl"
CLASSES_FILE = DATA_DIR / "classes.pkl"
MODEL_FILE   = DATA_DIR / "chatbot_model.h5"
SENTIMENT_LOG = DATA_DIR / "sentiment_log.jsonl"

IGNORE_LETTERS   = {"?", "!", ".", ","}
ERROR_THRESHOLD  = 0.25


# ---------------------------------------------------------------------------
# Reply dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatbotReply:
    text:        str
    intent:      Optional[str]            = None
    probability: Optional[float]          = None
    action:      Optional[str]            = None
    url:         Optional[str]            = None
    prompt:      Optional[str]            = None
    sentiment:   Optional[SentimentResult] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# NLTK helpers
# ---------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet"]
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            pass


def load_intents() -> dict:
    with INTENTS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def tokenize(sentence: str) -> list[str]:
    try:
        return nltk.word_tokenize(sentence)
    except LookupError:
        return re.findall(r"[A-Za-z0-9']+", sentence)


def lemmatize_word(lemmatizer: WordNetLemmatizer, word: str) -> str:
    try:
        return lemmatizer.lemmatize(word)
    except LookupError:
        return word


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_data(
    intents: dict,
) -> tuple[list[str], list[str], list[list[int]], list[list[int]]]:
    lemmatizer = WordNetLemmatizer()
    words: list[str] = []
    classes: list[str] = []
    documents: list[tuple[list[str], str]] = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = sorted(
        {lemmatize_word(lemmatizer, w.lower()) for w in words if w not in IGNORE_LETTERS}
    )
    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)
    for word_list, intent_tag in documents:
        word_patterns = [lemmatize_word(lemmatizer, w.lower()) for w in word_list]
        bag = [1 if w in word_patterns else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(intent_tag)] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    arr = np.array(training, dtype=object)
    return words, classes, list(arr[:, 0]), list(arr[:, 1])


def train_model(epochs: int = 200, batch_size: int = 5) -> None:
    import tensorflow as tf

    ensure_nltk_data()
    intents = load_intents()
    words, classes, train_x, train_y = build_training_data(intents)

    with WORDS_FILE.open("wb") as f:
        pickle.dump(words, f)
    with CLASSES_FILE.open("wb") as f:
        pickle.dump(classes, f)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_y[0]), activation="softmax"),
    ])
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model.fit(np.array(train_x), np.array(train_y),
              epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(MODEL_FILE)
    print(f"Model saved → {MODEL_FILE}")


# ---------------------------------------------------------------------------
# Chatbot engine
# ---------------------------------------------------------------------------

class ChatbotEngine:
    """
    NLP intent classifier + sentiment-aware response layer.

    Sentiment analysis runs on *every* user message (except slash-commands
    that are handled before NLP classification).  The result is:
      1. Used to select an empathy prefix.
      2. Appended as an escalation note for strongly negative messages.
      3. Attached to the :class:`ChatbotReply` for the UI layers.
      4. Logged to ``sentiment_log.jsonl``.
    """

    def __init__(self) -> None:
        ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.intents    = load_intents()
        self.words      = self._load_pickle(WORDS_FILE)
        self.classes    = self._load_pickle(CLASSES_FILE)
        self.model      = self._load_model()
        # Sentiment: initialise with graceful fallback so the bot never crashes
        # even if vaderSentiment is missing or the analyser fails to load.
        try:
            self.sentiment = SentimentAnalyzer()
        except Exception as exc:
            import warnings
            warnings.warn(
                f"SentimentAnalyzer could not be initialised ({exc}). "
                "Sentiment features will be disabled for this session.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.sentiment = SentimentAnalyzer.__new__(SentimentAnalyzer)
            self.sentiment.available = False
            self.sentiment._vader = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pickle(path: Path) -> list:
        if not path.exists():
            train_model()
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_model():
        import tensorflow as tf
        if not MODEL_FILE.exists():
            train_model()
        return tf.keras.models.load_model(MODEL_FILE)

    # ------------------------------------------------------------------
    # Commands list (shown by /help and the desktop UI sidebar)
    # ------------------------------------------------------------------

    @property
    def commands(self) -> list[tuple[str, str]]:
        return [
            ("/help",              "Show available commands."),
            ("/time",              "Show the current time."),
            ("/date",              "Show today's date."),
            ("/mood",              "Check your detected sentiment."),          # NEW
            ("/search <query>",    "Open a Google search for a topic."),
            ("/image <prompt>",    "Generate an image with AI."),
            ("/analyze <question>","Choose a device image and predict its contents."),
            ("/clear",             "Clear the conversation in the UI."),
            ("/save",              "Save the current conversation in the UI."),
            ("/name <name>",       "Update the profile name in the UI."),
            ("bye",                "End the conversation."),
        ]

    # ------------------------------------------------------------------
    # NLP helpers
    # ------------------------------------------------------------------

    def clean_up_sentence(self, sentence: str) -> list[str]:
        return [lemmatize_word(self.lemmatizer, w.lower()) for w in tokenize(sentence)]

    def bag_of_words(self, sentence: str) -> np.ndarray:
        words = self.clean_up_sentence(sentence)
        return np.array([1 if w in words else 0 for w in self.words])

    def predict_class(self, sentence: str) -> list[dict]:
        bow    = self.bag_of_words(sentence)
        result = self.model.predict(np.array([bow]), verbose=0)[0]
        matches = [[i, s] for i, s in enumerate(result) if s > ERROR_THRESHOLD]
        matches.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[i], "probability": float(s)} for i, s in matches]

    def get_response(self, intents_list: list[dict]) -> tuple[str, Optional[str], Optional[float]]:
        """Return (text, intent_tag, probability) — no sentiment wrapping yet."""
        if not intents_list:
            return (
                "I'm not sure how to respond to that yet. "
                "Try rephrasing or use /search with your topic.",
                None, None,
            )
        tag  = intents_list[0]["intent"]
        prob = intents_list[0]["probability"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"]), tag, prob
        return "I found an intent but have no response for it yet.", tag, prob

    # ------------------------------------------------------------------
    # Sentiment logging
    # ------------------------------------------------------------------

    def _log_sentiment(
        self,
        user_message: str,
        bot_reply:    str,
        result:       SentimentResult,
        intent:       Optional[str],
    ) -> None:
        """Append one JSON line to sentiment_log.jsonl (non-blocking)."""
        entry = {
            "ts":        dt.datetime.now().isoformat(timespec="seconds"),
            "message":   user_message,
            "intent":    intent,
            "sentiment": result.to_dict(),
            "reply":     bot_reply,
        }
        try:
            with SENTIMENT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass   # never crash on logging failure

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def reply(self, message: str, user_name: str = "User") -> ChatbotReply:
        text       = message.strip()
        normalized = text.lower()

        if not text:
            return ChatbotReply("Please type a question or choose a command.")

        # ---- /help -------------------------------------------------------
        if normalized in {"/help", "help", "commands", "/commands"}:
            lines = [f"{cmd} — {desc}" for cmd, desc in self.commands]
            return ChatbotReply(
                "Available commands:\n" + "\n".join(lines), action="help"
            )

        # ---- /time -------------------------------------------------------
        if normalized in {"/time", "time"}:
            return ChatbotReply(
                f"{user_name}, the current time is "
                f"{dt.datetime.now().strftime('%I:%M %p')}."
            )

        # ---- /date -------------------------------------------------------
        if normalized in {"/date", "date", "today"}:
            return ChatbotReply(
                f"Today is {dt.datetime.now().strftime('%A, %d %B %Y')}."
            )

        # ---- /mood  -------------------------------------------------------
        if normalized in {"/mood", "mood", "my mood", "my sentiment"}:
            if not getattr(self.sentiment, "available", False):
                return ChatbotReply(
                    "Sentiment analysis is currently unavailable. "
                    "Please ensure vaderSentiment is installed "
                    "(pip install vaderSentiment) and restart the chatbot."
                )
            result = self.sentiment.analyze(text)
            return ChatbotReply(
                f"I detected your mood as **{result.label}** "
                f"({result.intensity} intensity) {result.emoji}\n"
                f"Sentiment score: {result.compound:+.3f}  "
                f"(−1 = very negative, +1 = very positive)",
                sentiment=result,
            )
        # ---- bye ---------------------------------------------------------
        if normalized in {"exit", "quit", "bye", "/bye"}:
            return ChatbotReply(
                f"Goodbye, {user_name}. Have a great day!", intent="goodbye"
            )

        # ---- /search -----------------------------------------------------
        if normalized.startswith("/search ") or normalized.startswith("search "):
            query = text.split(" ", 1)[1].strip()
            if not query:
                return ChatbotReply(
                    "Type a topic after /search. Example: /search Python NLP."
                )
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            return ChatbotReply(
                f"I opened a web search for: {query}", action="open_url", url=url
            )

        # ---- /image ------------------------------------------------------
        if normalized == "/image":
            return ChatbotReply(
                "Type a description after /image. Example: /image a city at sunset."
            )
        if normalized.startswith("/image "):
            prompt = text.split(" ", 1)[1].strip()
            return ChatbotReply(
                f"Generating an image for: {prompt}",
                action="generate_image", prompt=prompt,
            )

        # ---- /analyze ----------------------------------------------------
        if normalized == "/analyze" or normalized.startswith("/analyze "):
            question = text.split(" ", 1)[1].strip() if " " in text else ""
            return ChatbotReply(
                "Choose an image from your device and I will predict what it contains.",
                action="analyze_image", prompt=question,
            )

        # ---- NLP classification + SENTIMENT LAYER -----------------------
        sentiment_result = self.sentiment.analyze(text)          # analyse raw text
        raw_text, intent_tag, prob = self.get_response(self.predict_class(text))

        # Wrap reply with empathy prefix + optional escalation note
        wrapped_text = sentiment_result.wrap(raw_text)

        self._log_sentiment(text, wrapped_text, sentiment_result, intent_tag)

        return ChatbotReply(
            text        = wrapped_text,
            intent      = intent_tag,
            probability = prob,
            sentiment   = sentiment_result,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def interactive_chat() -> None:
    bot = ChatbotEngine()
    print("Chatbot is ready. Type /help for commands or 'quit' to exit.\n")
    while True:
        message = input("You: ")
        reply   = bot.reply(message)
        # Show sentiment badge in CLI
        badge = ""
        if reply.sentiment:
            s = reply.sentiment
            badge = f"  [{s.emoji} {s.label} {s.intensity}]"
        print(f"Bot:{badge}\n  {reply.text}\n")
        if message.strip().lower() in {"exit", "quit", "bye", "/bye"}:
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the NLP chatbot.")
    parser.add_argument("--train",      action="store_true", help="Retrain the TensorFlow model.")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--cli",        action="store_true", help="Run in the terminal.")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs)
    elif args.cli:
        interactive_chat()
    else:
        from web_app import run_web_app
        run_web_app(host=args.host, port=args.port, open_browser=not args.no_browser)
