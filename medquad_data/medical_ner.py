"""
medical_ner.py
Simple keyword + regex medical entity recognizer.
No heavy model needed for a demo — covers symptoms, diseases, treatments, drugs.
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Keyword lists  (extend these as needed)
# ---------------------------------------------------------------------------

SYMPTOM_KEYWORDS = [
    "fever", "pain", "headache", "fatigue", "nausea", "vomiting", "diarrhea",
    "cough", "shortness of breath", "chest pain", "dizziness", "rash",
    "swelling", "itching", "bleeding", "weight loss", "weight gain",
    "insomnia", "anxiety", "depression", "numbness", "tingling",
    "blurred vision", "hearing loss", "sore throat", "runny nose",
    "muscle weakness", "joint pain", "back pain", "abdominal pain",
    "constipation", "loss of appetite", "chills", "sweating", "palpitations",
    "difficulty swallowing", "confusion", "memory loss", "seizure",
]

DISEASE_KEYWORDS = [
    "diabetes", "cancer", "hypertension", "asthma", "arthritis", "alzheimer",
    "parkinson", "epilepsy", "stroke", "heart disease", "heart attack",
    "pneumonia", "tuberculosis", "hiv", "aids", "hepatitis", "cirrhosis",
    "kidney disease", "liver disease", "thyroid", "hypothyroidism",
    "hyperthyroidism", "anemia", "leukemia", "lymphoma", "melanoma",
    "psoriasis", "eczema", "crohn", "celiac", "lupus", "multiple sclerosis",
    "fibromyalgia", "osteoporosis", "scoliosis", "glaucoma", "cataract",
    "migraine", "bipolar", "schizophrenia", "autism", "adhd",
    "copd", "emphysema", "bronchitis", "gout", "obesity",
]

TREATMENT_KEYWORDS = [
    "surgery", "chemotherapy", "radiation", "therapy", "transplant",
    "dialysis", "physical therapy", "occupational therapy", "psychotherapy",
    "vaccination", "vaccine", "immunotherapy", "hormone therapy",
    "stem cell", "blood transfusion", "bypass", "angioplasty", "stent",
    "biopsy", "endoscopy", "colonoscopy", "mri", "ct scan", "x-ray",
    "ultrasound", "ecg", "eeg", "blood test", "urine test",
]

DRUG_KEYWORDS = [
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "metformin",
    "insulin", "lisinopril", "atorvastatin", "amoxicillin", "penicillin",
    "prednisone", "methotrexate", "warfarin", "heparin", "morphine",
    "codeine", "oxycodone", "antidepressant", "antipsychotic", "antibiotic",
    "antifungal", "antihistamine", "beta blocker", "ace inhibitor",
    "statin", "diuretic", "proton pump inhibitor", "nsaid", "opioid",
    "steroid", "corticosteroid", "benzodiazepine", "ssri", "snri",
    "omeprazole", "sertraline", "fluoxetine", "gabapentin", "levothyroxine",
]

BODY_PART_KEYWORDS = [
    "heart", "lung", "liver", "kidney", "brain", "spine", "bone", "skin",
    "eye", "ear", "nose", "throat", "stomach", "intestine", "colon",
    "pancreas", "thyroid", "bladder", "prostate", "ovary", "uterus",
    "breast", "muscle", "joint", "artery", "vein", "nerve", "blood",
]

# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

CATEGORIES = {
    "symptom":    SYMPTOM_KEYWORDS,
    "disease":    DISEASE_KEYWORDS,
    "treatment":  TREATMENT_KEYWORDS,
    "drug":       DRUG_KEYWORDS,
    "body_part":  BODY_PART_KEYWORDS,
}

# Emoji badges per category (shown in UI)
CATEGORY_BADGES = {
    "symptom":   "🤒 Symptom",
    "disease":   "🦠 Disease",
    "treatment": "💊 Treatment",
    "drug":      "💉 Drug",
    "body_part": "🫀 Body Part",
}


def recognize_entities(text: str) -> Dict[str, List[str]]:
    """
    Scan text for medical entities.

    Returns dict  { category: [matched_term, ...] }
    Only non-empty categories are included.
    """
    text_lower = text.lower()
    found: Dict[str, List[str]] = {}

    for category, keywords in CATEGORIES.items():
        matches = []
        for kw in keywords:
            # Word-boundary match (avoids partial hits like "pain" in "spain")
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matches.append(kw)
        if matches:
            found[category] = matches

    return found


def format_entities(entities: Dict[str, List[str]]) -> str:
    """Return a readable string of detected entities, or empty string."""
    if not entities:
        return ""
    lines = []
    for cat, terms in entities.items():
        badge = CATEGORY_BADGES.get(cat, cat.title())
        lines.append(f"{badge}: {', '.join(terms)}")
    return "\n".join(lines)


# Quick CLI test
if __name__ == "__main__":
    test_sentences = [
        "What are the treatments for diabetes and hypertension?",
        "I have a headache, fever and nausea. Could it be cancer?",
        "How does aspirin affect the heart and blood pressure?",
    ]
    for s in test_sentences:
        print(f"\nInput   : {s}")
        ents = recognize_entities(s)
        print(f"Entities: {ents}")
