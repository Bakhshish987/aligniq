import streamlit as st
st.set_page_config(page_title="AlignIQ - Resume JD Matcher", layout="wide")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background-color: #d33c3c;
        color: white;
        border: 2px solid #d33c3c;
        border-radius: 6px;
        padding: 10px 24px;
        font-size: 18px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: transparent;
        color: #d33c3c;
    }
    </style>
""", unsafe_allow_html=True)





import re
import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# --- Load BERT model with spinner and caching ---
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

with st.spinner("🔁 Loading BERT model..."):
    model = load_bert_model()

# --- Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def get_keywords(text, top_n=20):
    words = text.split()
    return [word for word, count in Counter(words).most_common(top_n)]

def extract_text(file, filename):
    if filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    else:
        return ""

# --- UI ---
st.title("💼 AlignIQ: Resume & JD Match Analyzer")
st.markdown("Match your resume to any job description using AI 🔍")
st.markdown("**👨‍💻 Built by Bakhshish Sethi**", unsafe_allow_html=True)
st.caption(f"🗓️ Today: {datetime.date.today().strftime('%B %d, %Y')}")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📤 Upload Your Resume (.txt or .pdf)", type=["txt", "pdf"])
with col2:
    jd_text_input = st.text_area("📋 Paste the Job Description here", height=240)

# --- Button ---
analyze = st.button("🔍 Analyze Resume")

# --- Analysis Logic ---
if resume_file and jd_text_input.strip() and analyze:
    resume_text = extract_text(resume_file, resume_file.name)
    jd_text = jd_text_input

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # TF-IDF Similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([resume_clean, jd_clean])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # BERT Similarity
    resume_embed = model.encode(resume_text, convert_to_tensor=True)
    jd_embed = model.encode(jd_text, convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(resume_embed, jd_embed).item()

    # Keyword Match
    jd_keywords = get_keywords(jd_clean, top_n=20)
    resume_words = set(resume_clean.split())
    matched = [word for word in jd_keywords if word in resume_words]
    missing = [word for word in jd_keywords if word not in resume_words]
    keyword_match_pct = len(matched) / len(jd_keywords) * 100

    st.markdown("---")
    st.header("📈 AI Match Results")

    # Scores
    col3, col4, col5 = st.columns(3)
    col3.metric("🧮 TF-IDF Score", f"{tfidf_score:.2f}")
    col4.metric("🧠 BERT Score", f"{bert_score:.2f}")
    col5.metric("🔑 Keyword Match %", f"{keyword_match_pct:.0f}%")

    # Explanation
    with st.expander("ℹ️ What Do These Scores Mean?"):
        st.markdown("""
        - **TF-IDF Score**: Measures how many overlapping words your resume shares with the job description. Higher = better keyword alignment.
        - **BERT Score**: Captures deep semantic similarity. Higher = your resume actually *sounds like* it was written for this job.
        - **Keyword Match %**: Out of the top 20 most important words in the JD, how many appear in your resume.
        """)

    # Keyword Breakdown
    st.markdown("## 📌 Keyword Match Breakdown")
    st.success(f"✔️ Matched Keywords ({len(matched)}): {', '.join(matched)}")
    st.error(f"❌ Missing Keywords ({len(missing)}): {', '.join(missing)}")

    # Chart
    st.markdown("### 📊 Visual Breakdown")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Matched", "Missing"], [len(matched), len(missing)], color=["green", "red"])
    ax.set_ylabel("Keyword Count")
    ax.set_title("Resume vs JD Keyword Match")
    st.pyplot(fig)

    # Preview
    with st.expander("🧾 View Resume Text"):
        st.text_area("Resume", value=resume_text, height=300)
    with st.expander("📋 View Job Description Text"):
        st.text_area("Job Description", value=jd_text, height=300)

    # Suggestions
    st.markdown("## 🧠 Suggestions to Improve Your Resume")
    if bert_score < 0.6:
        st.warning("🧠 Try rewriting your bullet points to better match the JD’s tone and language.")
    if tfidf_score < 0.5:
        st.warning("📚 Consider using more relevant technical terms from the job posting.")
    if keyword_match_pct < 60:
        st.info("🔍 Add these missing keywords to improve alignment: " + ", ".join(missing))
    if len(matched) == 0:
        st.error("🚨 No matched keywords found! Major revision required for this resume.")

    st.success("✅ Tweak your resume, re-upload, and aim for scores above 0.6 for strong alignment!")

