"""
validate_with_vader.py
======================
Runs VADER over sentiment_dataset.json and prints accuracy stats.

Usage:
    python validate_with_vader.py

Make sure sentiment_dataset.json is in the same folder.
"""
import json
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
HIGH_INTENSITY     = 0.60
MEDIUM_INTENSITY   = 0.20

def vader_predict(text: str, sia) -> tuple[str, str]:
    scores   = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= POSITIVE_THRESHOLD:
        label = "POSITIVE"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    abs_c = abs(compound)
    intensity = "HIGH" if abs_c >= HIGH_INTENSITY else ("MEDIUM" if abs_c >= MEDIUM_INTENSITY else "LOW")
    return label, intensity

def main():
    dataset_path = Path(__file__).parent / "sentiment_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    sia = SentimentIntensityAnalyzer()
    label_correct = intensity_correct = both_correct = 0
    misses = []

    for item in data:
        pred_label, pred_intensity = vader_predict(item["text"], sia)
        lc = pred_label     == item["label"]
        ic = pred_intensity == item["intensity"]
        if lc: label_correct     += 1
        if ic: intensity_correct += 1
        if lc and ic: both_correct += 1
        if not lc:
            misses.append({
                "text":      item["text"],
                "expected":  f"{item['label']}/{item['intensity']}",
                "got":       f"{pred_label}/{pred_intensity}",
            })

    n = len(data)
    print(f"Dataset size : {n}")
    print(f"Label acc    : {label_correct}/{n}  ({100*label_correct/n:.1f}%)")
    print(f"Intensity acc: {intensity_correct}/{n}  ({100*intensity_correct/n:.1f}%)")
    print(f"Both correct : {both_correct}/{n}  ({100*both_correct/n:.1f}%)")
    print(f"\nLabel mismatches ({len(misses)}):")
    for m in misses[:20]:
        print(f"  [{m['expected']} → {m['got']}] {m['text'][:70]}")

if __name__ == "__main__":
    main()
