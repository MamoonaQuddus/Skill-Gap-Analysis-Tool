# 🎯 AI Skill Gap Analytics & Optimization Platform

An unsupervised NLP and Machine Learning web application designed to automatically audit talent pipelines. By vectorizing job descriptions and clustering market requirements, the system extracts a dynamic industry competency baseline and uses vector set-difference mathematics to uncover specific candidate gaps and map custom upskilling roadmaps.

---

## 🚀 System Features

* **Dynamic Text Space Vectorization:** Leverages `TfidfVectorizer` to break down noisy, unstructured job descriptions into meaningful unigrams and bigrams while stripping out conversational stop-words.
* **Unsupervised Domain Segmentation:** Employs `KMeans` clustering to categorize job requirements into target specialization tracks on-the-fly.
* **Algorithmic Gap Extraction:** Computes structural skill deficiencies by calculating the relative mathematical set difference between candidate competency matrices and clustered market demands:
  $$\text{Discovered Gaps} = \text{Industry Clustered Requirements} \setminus \text{Candidate Active Tokens}$$
* **Executive Profile Auditing UI:** A production-grade **Streamlit** dashboard built with state-preserving data flows, asynchronous multi-file ingestion validation, and searchable profile navigation that natively handles mixed alphanumeric/integer data keys.
* **Targeted Upskilling Roadmaps:** Seamlessly cross-references identified missing skill sets against a structured learning dictionary to output micro-learning steps and project suggestions.

---

## 🚀 Core Streamlit Features Developed

The interactive user interface leverages high-end, production-grade native Streamlit paradigms to ensure a seamless data science workflow:

* **Asynchronous Multi-File Ingestion Deck:** Employs `st.file_uploader` to swallow completely distinct candidate (`intern_profiles.csv`) and industry requirement (`industry_jobs.csv`) schemas simultaneously, bound by strict dynamic pandas dataframe header validation.
* **State-Preserving Pipeline Execution Window:** Utilizes `st.button` wrapped in a localized execution context (`st.spinner`) to process scikit-learn models and store training parameters safely within `st.session_state` across hot-reloads.
* **Search-Optimized Custom Label Formatter:** Uses an advanced `st.selectbox` embedded with a custom `format_func` mapping callback. This beautifully masks raw database primary keys into an executive tracking visual layout (`Ali — Ref: I002`) while avoiding error-prone string parsing splits.
* **Multi-Tab Modular Analytics Layout:** Organizes complex information into distinct user views using `st.tabs` to isolate individual candidate drilldowns from aggregate class statistics.
* **On-the-Fly Performance Report Exporter:** Embeds a native `st.download_button` that compiles the processed evaluation matrices from memory directly into a downloadable CSV asset (`cohort_skill_gaps_report.csv`).

---

## 📊 Key Machine Learning Insights Extracted

Because the system relies on dynamic vector-space analytics rather than rigid hardcoded string matching, it uncovers unique diagnostic patterns:

* **Granular Competency Gaps:** Pinpoints exactly which individual technical tokens are missing from a candidate's profile relative to the live market career track they naturally align with.
* **Macro Group Deficiencies Matrix:** Aggregates and calculates the absolute frequency of missing skills across the entire candidate pool under the **Group Skill Matrices** tab, highlighting widespread educational or training shortages.
* **Data-Driven Career Track Maps:** Reconstructs raw job texts into clear, interpretable specializations based on the top structural terms computed around each K-Means cluster centroid.

---

## 🔮 Strategic Future Improvements

To transition this platform from an MVP to an enterprise-grade corporate recruiting system, the following pipeline extensions are proposed:

### 1. Semantic Skill Mapping via Word Embeddings
* **Current Limitation:** The current TF-IDF token system relies on exact word spelling. If a job requires `neural networks` and an intern possesses `deep learning`, it might mistakenly register as a missing gap.
* **Solution:** Upgrade the feature extraction pipeline from TF-IDF to dense sentence embeddings (e.g., HuggingFace's `all-MiniLM-L6-v2`) to calculate gaps based on true conceptual cosine similarity rather than keyword syntax.

### 2. Live API-Driven Job Market Scraping
* **Current Limitation:** Job requirements depend on a static, manually curated `industry_jobs.csv` dataset that degrades over time as real-world trends shift.
* **Solution:** Connect the ingestion layer directly to active programmatic endpoints (such as LinkedIn, Indeed, or Adzuna Web APIs) to pull live, real-time demand feeds with a single click.

### 3. LLM-Powered Dynamic Training Curriculums
* **Current Limitation:** Suggestions inside `training_catalog.py` are mapped via a static python dictionary, reverting to standard generic text if a newly discovered cluster keyword is not pre-indexed.
* **Solution:** Integrate an open-source Large Language Model API (e.g., LLaMA 3 via Groq) to read the missing vector array and instantly construct a fully customized, day-by-day learning curriculum tailored explicitly to that intern's academic profile.


---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:
- Fork the repository
- Create a new branch
- Make your changes
- Submit a pull request

You can also open issues for bugs or suggestions.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📞 Contact

For questions or suggestions, feel free to reach out:

- GitHub: https://github.com/MamoonaQuddus

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---

**Built with ❤️ by Mamoona Quddus**  
