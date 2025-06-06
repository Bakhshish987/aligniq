# 💼 AlignIQ — AI-Powered Resume & Job Description Matcher

**AlignIQ** is an intelligent, NLP-driven web app that analyzes how well your resume matches a specific job description. It combines keyword extraction, semantic similarity, and real-time visual feedback to help job seekers tailor their resumes for maximum impact.

![AlignIQ Banner](https://imgur.com/your_banner_link_here.png) <!-- optional -->

---

## 🔍 Features

- 📄 **Upload Resume** (`.pdf` or `.txt`)
- 📋 **Paste Job Description** into a text box
- 📊 **Analyze Match Quality** using:
  - TF-IDF Similarity (Keyword overlap)
  - BERT Semantic Similarity (Contextual match)
  - Top 20 Keyword Coverage %
- ✅ **Actionable Resume Improvement Tips**
- 📈 **Keyword Match Bar Chart**
- 📱 **Mobile-Responsive UI**
- 🧑‍🎨 Styled with custom CSS for modern look and feel

---

## 🧠 How It Works

| Technique        | Purpose |
|------------------|---------|
| `TF-IDF`         | Checks overlap of words used in resume vs job description |
| `BERT` (MiniLM)  | Computes deep semantic similarity using sentence embeddings |
| `Cosine Similarity` | Quantifies alignment between both vectors |
| `Keyword Match %`| Calculates coverage of the top 20 JD keywords in resume |

---

## ⚙️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **NLP Models**: `sklearn`, `sentence-transformers`, `BERT (MiniLM)`
- **Visualization**: `matplotlib`
- **Resume Parsing**: `PyMuPDF` for `.pdf`, built-in for `.txt`
- **Deployment**: [Streamlit Cloud](https://share.streamlit.io/)
