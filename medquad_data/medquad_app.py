"""
medquad_app.py
Streamlit UI for the MedQuAD Medical Q&A feature.

Run:
    streamlit run medquad_app.py
"""

import sys
import os
import streamlit as st

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from medquad_retriever import MedQuADRetriever
from medical_ner import recognize_entities, format_entities, CATEGORY_BADGES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Q&A — MedQuAD",
    page_icon="🏥",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean clinical look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp { background: #f0f4f8; }

    /* Card-style answer box */
    .answer-card {
        background: #ffffff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        font-size: 15px;
        line-height: 1.7;
        color: #1e293b;
    }

    /* Entity badges */
    .entity-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    .entity-badge {
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 13px;
        font-weight: 500;
    }
    .entity-badge.symptom  { background: #fef9c3; color: #854d0e; }
    .entity-badge.disease  { background: #fee2e2; color: #991b1b; }
    .entity-badge.treatment{ background: #dcfce7; color: #166534; }
    .entity-badge.drug     { background: #ede9fe; color: #5b21b6; }
    .entity-badge.body_part{ background: #e0f2fe; color: #075985; }

    /* Matched question chip */
    .matched-q {
        font-size: 12px;
        color: #64748b;
        margin-top: 10px;
        font-style: italic;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: #2563eb;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Disclaimer box */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #9a3412;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load retriever (cached across sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading MedQuAD index …")
def load_retriever():
    return MedQuADRetriever()

try:
    retriever = load_retriever()
    index_ready = True
except FileNotFoundError as e:
    index_ready = False
    load_error  = str(e)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.nlm.nih.gov/about/logos/nlmlogo.png", width=120)
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Results to show", min_value=1, max_value=5, value=1)
    show_all = st.checkbox("Show all top results", value=False)
    st.divider()
    st.markdown("""
**Dataset:** MedQuAD  
47,457 QA pairs from 12 NIH websites  
([GitHub](https://github.com/abachaa/MedQuAD))

**Sources include:**  
• cancer.gov  
• niddk.nih.gov  
• NIH GARD  
• MedlinePlus Health Topics  
• CDC, NHLBI, NINDS …
    """)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🏥 Medical Q&A")
st.caption("Powered by the MedQuAD dataset — 47,457 NIH-sourced QA pairs")

if not index_ready:
    st.error(f"**Index not found.** Run the parser first:\n```\npython medquad_parser.py\n```\n\nDetails: {load_error}")
    st.stop()

# --- Question input ---
question = st.text_input(
    "Ask a medical question:",
    placeholder="e.g. What are the symptoms of diabetes?",
    key="question_input",
)

example_questions = [
    "What causes high blood pressure?",
    "How is asthma treated?",
    "What are the symptoms of Parkinson's disease?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "What medications are used for depression?",
]

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, eq in enumerate(example_questions):
    if cols[i % 3].button(eq, key=f"ex_{i}", use_container_width=True):
        question = eq

# --- Process ---
if question and question.strip():
    q = question.strip()

    # Entity recognition
    entities = recognize_entities(q)

    # Show entity badges
    if entities:
        badge_html = '<div class="entity-row">'
        for cat, terms in entities.items():
            for term in terms:
                badge_html += f'<span class="entity-badge {cat}">{term}</span>'
        badge_html += "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)

    st.divider()

    # Retrieval
    results = retriever.query(q, top_k=top_k if show_all else 1)

    if not results:
        st.warning(
            "No close match found in MedQuAD for this question. "
            "Try rephrasing or use a more specific medical term."
        )
    else:
        display_results = results if show_all else results[:1]

        for i, r in enumerate(display_results):
            score_pct = int(r["score"] * 100)

            header = f"**Answer**"
            if show_all and len(display_results) > 1:
                header = f"**Result {i+1}**"

            # Metadata row
            meta_parts = []
            if r["focus"]:
                meta_parts.append(f"📌 {r['focus']}")
            if r["qtype"]:
                meta_parts.append(f"Type: *{r['qtype']}*")
            if r["source"]:
                meta_parts.append(f"Source: `{r['source']}`")
            meta_parts.append(f"<span class='score-badge'>Match {score_pct}%</span>")

            st.markdown(header)
            if meta_parts:
                st.markdown(" &nbsp;|&nbsp; ".join(meta_parts), unsafe_allow_html=True)

            # Answer card
            answer_text = r["answer"].replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card">{answer_text}'
                f'<div class="matched-q">Matched question: "{r["question"]}"</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.write("")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational and demo purposes only. '
        'It is not a substitute for professional medical advice, diagnosis, or treatment. '
        'Always consult a qualified healthcare provider.'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("MedQuAD dataset by Asma Ben Abacha & Dina Demner-Fushman, NIH/NLM · CC BY 4.0")
