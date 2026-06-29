"""
medquad_retriever.py
TF-IDF retrieval engine: given a user question, find the most similar
MedQuAD questions and return their answers.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_CSV    = r"D:\project\NLP\medquad_index.csv"
CACHE_PKL    = r"D:\project\NLP\medquad_tfidf_cache.pkl"
MIN_SCORE    = 0.15   # below this we say "no good match found"


class MedQuADRetriever:
    """Loads MedQuAD CSV, builds TF-IDF index, answers queries."""

    def __init__(self, index_csv: str = INDEX_CSV, cache_pkl: str = CACHE_PKL):
        self.index_csv = index_csv
        self.cache_pkl = cache_pkl
        self.df        = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load()

    # ------------------------------------------------------------------
    # Loading / caching
    # ------------------------------------------------------------------

    def _load(self):
        """Load from cache if available, otherwise build from CSV."""
        if os.path.exists(self.cache_pkl):
            print("[MedQuAD] Loading TF-IDF cache …")
            with open(self.cache_pkl, "rb") as f:
                bundle = pickle.load(f)
            self.df           = bundle["df"]
            self.vectorizer   = bundle["vectorizer"]
            self.tfidf_matrix = bundle["tfidf_matrix"]
            print(f"[MedQuAD] Ready — {len(self.df)} QA pairs loaded from cache.")
        elif os.path.exists(self.index_csv):
            self._build_from_csv()
        else:
            raise FileNotFoundError(
                f"Neither cache ({self.cache_pkl}) nor CSV ({self.index_csv}) found.\n"
                "Run medquad_parser.py first to generate the CSV."
            )

    def _build_from_csv(self):
        print("[MedQuAD] Building TF-IDF index from CSV (first run — takes ~30s) …")
        self.df = pd.read_csv(self.index_csv, dtype=str).fillna("")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["question"])

        # Save cache so next load is instant
        with open(self.cache_pkl, "wb") as f:
            pickle.dump({
                "df":           self.df,
                "vectorizer":   self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        print(f"[MedQuAD] Index built & cached. {len(self.df)} QA pairs ready.")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, user_question: str, top_k: int = 3):
        """
        Find the top-k most similar questions and return results.

        Returns list of dicts:
          { "question", "answer", "focus", "qtype", "source", "score" }
        """
        q_vec  = self.vectorizer.transform([user_question])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < MIN_SCORE:
                break
            row = self.df.iloc[idx]
            results.append({
                "question": row["question"],
                "answer":   row["answer"],
                "focus":    row.get("focus", ""),
                "qtype":    row.get("qtype", ""),
                "source":   row.get("source", ""),
                "score":    round(score, 4),
            })
        return results

    def best_answer(self, user_question: str):
        """Return (answer_text, score, matched_question) or (None, 0, None) if no match."""
        results = self.query(user_question, top_k=1)
        if results:
            r = results[0]
            return r["answer"], r["score"], r["question"]
        return None, 0.0, None


# Quick CLI test
if __name__ == "__main__":
    retriever = MedQuADRetriever()
    while True:
        q = input("\nAsk a medical question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        answer, score, matched = retriever.best_answer(q)
        if answer:
            print(f"\nMatched Q: {matched}")
            print(f"Score    : {score}")
            print(f"Answer   : {answer[:500]} …")
        else:
            print("No sufficiently close match found in MedQuAD.")
