import streamlit as st
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer, util

# --- Load BERT model once ---
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()
stop_words = set(stopwords.words('english'))

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def get_keywords(text, top_n=20):
    words = text.split()
    return [word for word, count in Counter(words).most_common(top_n)]

# --- Streamlit UI ---
st.title("📄 AlignIQ: Resume & Job Description Matcher")

resume_file = st.file_uploader("📤 Upload Your Resume (.txt)", type=["txt"])
jd_file = st.file_uploader("📤 Upload Job Description (.txt)", type=["txt"])

if resume_file and jd_file:
    resume_text = resume_file.read().decode("utf-8")
    jd_text = jd_file.read().decode("utf-8")

    # Preprocess
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # --- TF-IDF Similarity ---
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([resume_clean, jd_clean])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # --- BERT Semantic Similarity ---
    resume_embed = model.encode(resume_text, convert_to_tensor=True)
    jd_embed = model.encode(jd_text, convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(resume_embed, jd_embed).item()

    # --- Keyword Match ---
    jd_keywords = get_keywords(jd_clean, top_n=20)
    resume_words = set(resume_clean.split())
    matched = [word for word in jd_keywords if word in resume_words]
    missing = [word for word in jd_keywords if word not in resume_words]
    keyword_match_pct = len(matched) / len(jd_keywords) * 100

    # --- Results ---
    st.subheader("📈 Similarity Scores")
    st.metric(label="TF-IDF Score", value=f"{tfidf_score:.2f}")
    st.metric(label="BERT Score", value=f"{bert_score:.2f}")

    st.subheader("📌 Keyword Match Breakdown")
    st.markdown(f"**✔️ Matched Keywords:** {', '.join(matched)}")
    st.markdown(f"**❌ Missing Keywords:** {', '.join(missing)}")
    st.markdown(f"**📊 Match %:** {keyword_match_pct:.1f}%")
