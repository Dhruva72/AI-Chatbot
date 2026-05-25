import argparse
import datetime as dt
import json
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer


DATA_DIR = Path(__file__).resolve().parent
INTENTS_FILE = DATA_DIR / "intents.json"
WORDS_FILE = DATA_DIR / "words.pkl"
CLASSES_FILE = DATA_DIR / "classes.pkl"
MODEL_FILE = DATA_DIR / "chatbot_model.h5"

IGNORE_LETTERS = {"?", "!", ".", ","}
ERROR_THRESHOLD = 0.25


@dataclass
class ChatbotReply:
    text: str
    intent: str | None = None
    probability: float | None = None
    action: str | None = None
    url: str | None = None


def ensure_nltk_data() -> None:
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet"]
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            pass


def load_intents() -> dict:
    with INTENTS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


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


def build_training_data(intents: dict) -> tuple[list[str], list[str], list[list[int]], list[list[int]]]:
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

    words = sorted({lemmatize_word(lemmatizer, word.lower()) for word in words if word not in IGNORE_LETTERS})
    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)
    for word_list, intent_tag in documents:
        word_patterns = [lemmatize_word(lemmatizer, word.lower()) for word in word_list]
        bag = [1 if word in word_patterns else 0 for word in words]
        output_row = list(output_empty)
        output_row[classes.index(intent_tag)] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training_array = np.array(training, dtype=object)
    train_x = list(training_array[:, 0])
    train_y = list(training_array[:, 1])
    return words, classes, train_x, train_y


def train_model(epochs: int = 200, batch_size: int = 5) -> None:
    import tensorflow as tf

    ensure_nltk_data()
    intents = load_intents()
    words, classes, train_x, train_y = build_training_data(intents)

    with WORDS_FILE.open("wb") as file:
        pickle.dump(words, file)
    with CLASSES_FILE.open("wb") as file:
        pickle.dump(classes, file)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(train_y[0]), activation="softmax"),
        ]
    )

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(MODEL_FILE)
    print(f"Model created at {MODEL_FILE}")


class ChatbotEngine:
    def __init__(self) -> None:
        ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.intents = load_intents()
        self.words = self._load_pickle(WORDS_FILE)
        self.classes = self._load_pickle(CLASSES_FILE)
        self.model = self._load_model()

    @staticmethod
    def _load_pickle(path: Path) -> list[str]:
        if not path.exists():
            train_model()
        with path.open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def _load_model() -> object:
        import tensorflow as tf

        if not MODEL_FILE.exists():
            train_model()
        return tf.keras.models.load_model(MODEL_FILE)

    @property
    def commands(self) -> list[tuple[str, str]]:
        return [
            ("/help", "Show available commands."),
            ("/time", "Show the current time."),
            ("/date", "Show today's date."),
            ("/search <query>", "Open a Google search for a topic."),
            ("/clear", "Clear the conversation in the UI."),
            ("/save", "Save the current conversation in the UI."),
            ("/name <name>", "Update the profile name in the UI."),
            ("bye", "End the conversation."),
        ]

    def clean_up_sentence(self, sentence: str) -> list[str]:
        return [lemmatize_word(self.lemmatizer, word.lower()) for word in tokenize(sentence)]

    def bag_of_words(self, sentence: str) -> np.ndarray:
        sentence_words = self.clean_up_sentence(sentence)
        return np.array([1 if word in sentence_words else 0 for word in self.words])

    def predict_class(self, sentence: str) -> list[dict]:
        bow = self.bag_of_words(sentence)
        result = self.model.predict(np.array([bow]), verbose=0)[0]
        matches = [[index, score] for index, score in enumerate(result) if score > ERROR_THRESHOLD]
        matches.sort(key=lambda item: item[1], reverse=True)
        return [
            {"intent": self.classes[index], "probability": float(score)}
            for index, score in matches
        ]

    def get_response(self, intents_list: list[dict]) -> ChatbotReply:
        if not intents_list:
            return ChatbotReply(
                "I am not sure how to respond to that yet. Try rephrasing it or use /search with your topic."
            )

        tag = intents_list[0]["intent"]
        probability = intents_list[0]["probability"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return ChatbotReply(
                    random.choice(intent["responses"]),
                    intent=tag,
                    probability=probability,
                )
        return ChatbotReply("I found an intent, but I do not have a response for it yet.", intent=tag)

    def reply(self, message: str, user_name: str = "User") -> ChatbotReply:
        text = message.strip()
        normalized = text.lower()

        if not text:
            return ChatbotReply("Please type a question or choose a command.")

        if normalized in {"/help", "help", "commands", "/commands"}:
            command_lines = [f"{command} - {description}" for command, description in self.commands]
            return ChatbotReply("Available commands:\n" + "\n".join(command_lines), action="help")

        if normalized in {"/time", "time"}:
            return ChatbotReply(f"{user_name}, the current time is {dt.datetime.now().strftime('%I:%M %p')}.")

        if normalized in {"/date", "date", "today"}:
            return ChatbotReply(f"Today is {dt.datetime.now().strftime('%A, %d %B %Y')}.")

        if normalized in {"exit", "quit", "bye", "/bye"}:
            return ChatbotReply(f"Goodbye, {user_name}. Have a great day!", intent="goodbye")

        if normalized.startswith("/search ") or normalized.startswith("search "):
            query = text.split(" ", 1)[1].strip()
            if not query:
                return ChatbotReply("Type a topic after /search. Example: /search Python NLP.")
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            return ChatbotReply(f"I opened a web search for: {query}", action="open_url", url=url)

        return self.get_response(self.predict_class(text))


def interactive_chat() -> None:
    bot = ChatbotEngine()
    print("Chatbot is ready. Type /help for commands or quit to exit.")
    while True:
        message = input("You: ")
        reply = bot.reply(message)
        print(f"Bot: {reply.text}")
        if message.strip().lower() in {"exit", "quit", "bye", "/bye"}:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the NLP chatbot.")
    parser.add_argument("--train", action="store_true", help="Retrain the TensorFlow model.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs)
    else:
        interactive_chat()
