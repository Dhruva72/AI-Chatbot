"""
medquad_app.py  —  MedQuAD Medical Q&A  (redesigned UI)
Run:  streamlit run medquad_app.py
"""

import sys, os
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from medquad_retriever import MedQuADRetriever
from medical_ner import recognize_entities, CATEGORY_BADGES

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedQuAD — Medical Q&A",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 900px; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
[data-testid="stSidebar"] a { color: #38bdf8 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #38bdf8 !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem; color: white; }
.hero p  { font-size: 1rem; margin: 0; opacity: 0.88; color: white; }

/* ── stat pills row ── */
.stats-row {
    display: flex; gap: 12px; margin-bottom: 2rem; flex-wrap: wrap;
}
.stat-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 13px;
    color: #475569;
    display: flex; align-items: center; gap: 6px;
}
.stat-pill b { color: #0f172a; }

/* ── search box ── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 16px !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.12) !important;
}

/* ── section labels ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.5rem 0 0.75rem;
}

/* ── example buttons ── */
.stButton > button {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #334155 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    text-align: left !important;
    transition: all 0.15s !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    border-color: #0ea5e9 !important;
    color: #0ea5e9 !important;
    background: #f0f9ff !important;
}

/* ── entity badges ── */
.badges-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 1rem 0 1.5rem; }
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 999px; padding: 4px 14px;
    font-size: 12px; font-weight: 600;
}
.badge-symptom   { background:#fef3c7; color:#92400e; border:1px solid #fde68a; }
.badge-disease   { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }
.badge-treatment { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
.badge-drug      { background:#ede9fe; color:#5b21b6; border:1px solid #ddd6fe; }
.badge-body_part { background:#e0f2fe; color:#075985; border:1px solid #bae6fd; }

/* ── answer card ── */
.answer-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
.answer-card-header {
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 20px;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.answer-card-body {
    padding: 20px 24px;
    font-size: 15px;
    line-height: 1.8;
    color: #1e293b;
}
.meta-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 12px;
    color: #64748b;
}
.score-chip {
    background: #0ea5e9;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 700;
    color: white;
    margin-left: auto;
}
.matched-q {
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px dashed #e2e8f0;
    font-size: 12px;
    color: #94a3b8;
    font-style: italic;
}

/* ── disclaimer ── */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #78350f;
    margin-top: 2rem;
    line-height: 1.6;
}

/* ── no result ── */
.no-result {
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #64748b;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ── load retriever ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building search index — one moment…")
def load_retriever():
    return MedQuADRetriever()

try:
    retriever  = load_retriever()
    index_size = len(retriever.df)
    index_ready = True
except FileNotFoundError as e:
    index_ready = False
    load_error  = str(e)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 MedQuAD Search")
    st.markdown("---")
    top_k     = st.slider("Results to show", 1, 5, 1)
    show_all  = st.checkbox("Show all results", value=False)
    st.markdown("---")
    st.markdown("""
**Dataset** &nbsp;·&nbsp; MedQuAD  
47,457 QA pairs · CC BY 4.0

**Sources**
- cancer.gov
- niddk.nih.gov
- NIH GARD
- MedlinePlus Health Topics
- CDC · NHLBI · NINDS
- Senior Health · GHR

[View on GitHub](https://github.com/abachaa/MedQuAD)
""")

# ── main ───────────────────────────────────────────────────────────────────────
if not index_ready:
    st.error(f"Index not found. Run `python medquad_parser.py` first.\n\n{load_error}")
    st.stop()

# Hero
st.markdown(f"""
<div class="hero">
  <h1>🩺 Medical Q&amp;A</h1>
  <p>Search {index_size:,} NIH-sourced question &amp; answer pairs from the MedQuAD dataset</p>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown(f"""
<div class="stats-row">
  <div class="stat-pill">📚 <b>{index_size:,}</b> QA pairs</div>
  <div class="stat-pill">🏥 <b>12</b> NIH sources</div>
  <div class="stat-pill">🔍 <b>TF-IDF</b> retrieval</div>
  <div class="stat-pill">🏷️ <b>5</b> entity types</div>
</div>
""", unsafe_allow_html=True)

# Search input
question = st.text_input(
    "",
    placeholder="e.g.  What are the symptoms of diabetes?",
    label_visibility="collapsed",
)

# Example questions
EXAMPLES = [
    "What causes high blood pressure?",
    "How is asthma treated?",
    "What are the symptoms of Parkinson's disease?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "What medications are used for depression?",
    "How is cancer diagnosed?",
]

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
        question = ex

# ── results ────────────────────────────────────────────────────────────────────
if question and question.strip():
    q = question.strip()

    # Entity badges
    entities = recognize_entities(q)
    if entities:
        badges_html = '<div class="badges-wrap">'
        for cat, terms in entities.items():
            icon = {"symptom":"🤒","disease":"🦠","treatment":"💊","drug":"💉","body_part":"🫀"}.get(cat,"🔬")
            for t in terms:
                badges_html += f'<span class="badge badge-{cat}">{icon} {t}</span>'
        badges_html += '</div>'
        st.markdown(badges_html, unsafe_allow_html=True)

    # Retrieve
    k = top_k if show_all else 1
    results = retriever.query(q, top_k=k)

    if not results:
        st.markdown("""
<div class="no-result">
  🔎 &nbsp; No close match found in MedQuAD.<br>
  <span style="font-size:13px;color:#94a3b8">Try rephrasing or use specific medical terms.</span>
</div>""", unsafe_allow_html=True)
    else:
        display = results if show_all else results[:1]
        for i, r in enumerate(display):
            score_pct = int(r["score"] * 100)
            focus  = r.get("focus","")
            qtype  = r.get("qtype","")
            source = r.get("source","")

            chips = ""
            if focus:  chips += f'<span class="meta-chip">📌 {focus}</span> '
            if qtype:  chips += f'<span class="meta-chip">{qtype}</span> '
            if source: chips += f'<span class="meta-chip">{source}</span> '

            answer_html = r["answer"].replace("\n","<br>")

            st.markdown(f"""
<div class="answer-card">
  <div class="answer-card-header">
    {chips}
    <span class="score-chip">Match {score_pct}%</span>
  </div>
  <div class="answer-card-body">
    {answer_html}
    <div class="matched-q">Closest match: &ldquo;{r['question']}&rdquo;</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Medical disclaimer</strong> — This tool is for informational and demo purposes only.
  It is not a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider.
</div>""", unsafe_allow_html=True)