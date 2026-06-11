# 🎯 AI Skill Gap Analytics & Optimization Platform

An **unsupervised NLP + Machine Learning system** that analyzes job descriptions and candidate profiles to identify skill gaps and generate personalized upskilling recommendations.

The platform transforms unstructured text into vector space, clusters industry requirements, and compares them with candidate skills to produce actionable talent insights.

---

## ⚙️ System Architecture

### 1. Text Vectorization
Uses **TF-IDF (TfidfVectorizer)** to convert job descriptions into structured feature vectors using unigrams and bigrams while removing stop words.

### 2. Job Market Clustering
Applies **KMeans clustering** to group similar job descriptions into dynamic career tracks such as:
- Data Science
- Software Engineering
- AI/ML Engineering

### 3. Skill Gap Analysis
Computes skill gaps using a set-based comparison:

\[
\text{Skill Gaps} = \text{Industry Requirements} - \text{Candidate Skills}
\]

This enables precise identification of missing competencies per candidate.

---

## 🚀 Streamlit Application Features

### 📂 Multi-File Data Ingestion
- Upload candidate profiles (`intern_profiles.csv`)
- Upload job market dataset (`industry_jobs.csv`)
- Automatic schema validation using pandas

### 🔁 Stateful ML Pipeline Execution
- Uses `st.button` to trigger processing
- Stores results in `st.session_state`
- Maintains state across reruns

### 🔍 Intelligent Candidate Selector
- Custom `selectbox` formatting:
  - `Name — Ref: ID`
- Improves readability and UX

### 📊 Multi-Tab Analytics Dashboard
- Individual candidate skill gap view
- Cluster-level industry insights
- Group-wide deficiency analysis

### 📥 Export Reports
- Download results using `st.download_button`
- Output file: `cohort_skill_gaps_report.csv`

---

## 📊 Key Insights

### 🎯 Individual Skill Gaps
Identifies missing skills for each candidate relative to their matched job cluster.

### 📉 Cohort-Level Skill Deficiencies
Aggregates missing skills across all candidates to detect common skill shortages.

### 🧭 Career Track Mapping
Automatically groups job descriptions into structured career tracks using KMeans clustering.

---

## 🔮 Future Improvements

### 1. Semantic Skill Matching (Embeddings Upgrade)
- Current: TF-IDF (keyword-based matching)
- Upgrade: Sentence embeddings (e.g., `all-MiniLM-L6-v2`)
- Benefit: Understands semantic similarity (e.g., *deep learning* ≈ *neural networks*)

---

### 2. Real-Time Job Market Integration
- Replace static CSV with live APIs (LinkedIn / Indeed / Adzuna)
- Enables real-time skill demand tracking

---

### 3. AI-Powered Learning Paths
- Replace static mapping dictionary
- Use LLMs (e.g., LLaMA 3 via Groq API)
- Generate personalized learning roadmaps dynamically

---

## 🛠️ Contribution Guidelines

- Fork the repository
- Create a feature branch
- Submit a pull request with clear documentation
- For major changes, open an issue first

---

## 📄 License

This project is licensed under the **MIT License**.  
Free to use, modify, and distribute with attribution.
