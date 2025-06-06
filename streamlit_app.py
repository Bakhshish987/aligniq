import streamlit as st
import re
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

# --- Text preprocessing ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="AlignIQ - Resume JD Matcher", layout="wide")

st.title("💼 AlignIQ: Resume & JD Match Analyzer")
st.markdown("Match your resume with any job description using AI 🔍")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📤 Upload Your Resume (.txt or .pdf)", type=["txt", "pdf"])
with col2:
    jd_file = st.file_uploader("📤 Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])

if resume_file and jd_file:
    resume_text = extract_text(resume_file, resume_file.name)
    jd_text = extract_text(jd_file, jd_file.name)

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

    # --- Results ---
    st.markdown("## 📈 AI Match Scores")
    col3, col4, col5 = st.columns(3)
    col3.metric("TF-IDF Score", f"{tfidf_score:.2f}")
    col4.metric("BERT Score", f"{bert_score:.2f}")
    col5.metric("Keyword Match %", f"{keyword_match_pct:.0f}%")

    st.markdown("## 📌 Keyword Insights")
    st.success(f"**✔️ Matched Keywords ({len(matched)}):** {', '.join(matched)}")
    st.error(f"**❌ Missing Keywords ({len(missing)}):** {', '.join(missing)}")

    # Keyword Bar Chart
    st.markdown("### 📊 Match Breakdown")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Matched", "Missing"], [len(matched), len(missing)], color=["green", "red"])
    ax.set_ylabel("Number of Keywords")
    ax.set_title("Resume vs JD Keyword Match")
    st.pyplot(fig)

    # Preview Sections
    with st.expander("🧾 View Resume Text"):
        st.text_area("Resume", value=resume_text, height=300)

    with st.expander("📋 View Job Description Text"):
        st.text_area("Job Description", value=jd_text, height=300)

    # Resume Improvement Tips
    st.markdown("## 🧠 Suggestions to Improve Your Resume")
    if bert_score < 0.6:
        st.warning("👉 Try rewriting your bullet points to better match the job wording.")
    if tfidf_score < 0.5:
        st.warning("👉 Consider adding more technical terms from the job description.")
    if keyword_match_pct < 60:
        st.info("✅ Add these missing keywords to improve alignment: " + ", ".join(missing))
    if len(matched) == 0:
        st.error("🚨 No matched keywords! Your resume needs major revision for this JD.")

    st.success("🔍 You can now tweak your resume and re-upload for better results!")

