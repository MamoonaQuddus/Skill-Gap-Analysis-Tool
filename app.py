# app.py
from __future__ import annotations
import pandas as pd
import streamlit as st

from src.pipeline import (
    compute_intern_gaps,
    extract_industry_skill_vocabulary,
    fit_job_clusters,
    summarize_top_missing,
)
from src.training_catalog import suggest_training

# 1. Premium Global Page Workspace Layout
st.set_page_config(
    page_title="AI Skill Gap Analytics Platform", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executive Header Stylings
st.title("🎯 AI Skill Gap Analytics & Optimization Platform")
st.markdown(
    "Benchmark candidate profiles dynamically against market requirements using unsupervised NLP vector clustering."
)
st.markdown("---")

# 2. Sidebar Control Deck & Parameter Panel
with st.sidebar:
    st.markdown("### 📁 Data Source Ingestion")
    interns_file = st.file_uploader("Upload Student Profiles (CSV)", type=["csv"], key="interns")
    jobs_file = st.file_uploader("Upload Market Requirements (CSV)", type=["csv"], key="jobs")

    st.divider()
    st.markdown("### ⚙️ Core Model Hyperparameters")
    n_clusters = st.slider("K-Means Clusters (Track Domains)", min_value=2, max_value=12, value=5, step=1)
    max_features = st.slider("TF-IDF Maximum Features", min_value=1000, max_value=10000, value=5000, step=500)
    top_terms_per_cluster = st.slider("Vocabulary Limit per Cluster", 5, 50, 20, 1)
    top_k_missing = st.slider("Max Extracted Skill Gaps", 5, 25, 10, 1)

def _read_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

if interns_file and jobs_file:
    interns_df = _read_csv(interns_file)
    jobs_df = _read_csv(jobs_file)

    # Database Structure Integrity Checkpoints
    required_intern_cols = {"Intern_ID", "Name", "Skills"}
    required_job_cols = {"Job_Id", "Job Title", "Description"}
    missing_i = required_intern_cols - set(interns_df.columns)
    missing_j = required_job_cols - set(jobs_df.columns)

    if missing_i or missing_j:
        if missing_i: st.error(f"⚠️ Intern CSV file missing headers: {list(missing_i)}")
        if missing_j: st.error(f"⚠️ Jobs CSV file missing headers: {list(missing_j)}")
        st.stop()

    # Dynamic Key Statistics Banners
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Audited Candidate Pool", f"👥 {len(interns_df)} Profiles")
    meta_col2.metric("Market Demand Footprint", f"💼 {len(jobs_df)} Positions")
    meta_col3.metric("Configured Specializations", f"🧭 {n_clusters} Tracks")

    st.markdown("### ⚡ Execution Panel")
    if st.button("Launch AI Skill Gap Analyzer Engine", type="primary", use_container_width=True):
        with st.spinner("Executing Unsupervised Data Pipeline..."):
            
            # Run Mathematical Engine Pipeline
            fit = fit_job_clusters(
                jobs_df=jobs_df,
                text_col="Description",
                n_clusters=int(n_clusters),
                max_features=int(max_features),
            )
            jobs_df_clustered = jobs_df.copy()
            jobs_df_clustered["cluster"] = fit.job_clusters

            industry_skills = extract_industry_skill_vocabulary(
                jobs_df=jobs_df,
                fit=fit,
                top_terms_per_cluster=int(top_terms_per_cluster),
            )

            gaps_df = compute_intern_gaps(
                interns_df=interns_df,
                industry_skills=industry_skills,
                skills_col="Skills",
                top_k_missing=int(top_k_missing),
            )

            gaps_df["missing_count"] = gaps_df["missing_skills_list"].apply(len)
            top_missing_df = summarize_top_missing(gaps_df, top_n=25)

            # Persist Analytics Results in App State
            st.session_state["analysis_results"] = {
                "gaps_df": gaps_df,
                "fit": fit,
                "jobs_df_clustered": jobs_df_clustered,
                "top_missing_df": top_missing_df,
                "industry_skills": industry_skills
            }
            st.success("🎉 Engineering Pipeline Run Successful! Insights loaded below.")

    # Render Modern Report Dashboard Interface
    if "analysis_results" in st.session_state:
        res = st.session_state["analysis_results"]

        st.markdown("---")
        tab_drilldown, tab_global, tab_vocabulary = st.tabs([
            "🕵️ Candidate Profile Drilldown", 
            "📊 Group Skill Matrices", 
            "📑 Extracted Vocabularies"
        ])

        with tab_drilldown:
            st.markdown("### Individual Candidate Performance Audit")
            
            # Safely group and alphabetize rows natively
            working_df = res["gaps_df"].copy()
            working_df = working_df.sort_values(by=["Name", "Intern_ID"])
            unique_ids = working_df["Intern_ID"].unique().tolist()
            
            # Professional Label Custom Formatter Function (Supports both string and numeric IDs)
            def ui_label_formatter(intern_id):
                row = working_df[working_df["Intern_ID"] == intern_id].iloc[0]
                return f"{row['Name']} — Ref: {row['Intern_ID']}"

            selected_id = st.selectbox(
                "Search and Select a Profile:", 
                options=unique_ids,
                format_func=ui_label_formatter,
                help="Type to filter or choose a candidate record from the active directory database."
            )

            # Extract row matching data key cleanly
            row = working_df[working_df["Intern_ID"] == selected_id].iloc[0]
            
            st.markdown(f"## **Candidate Portfolio: {row['Name']}**")
            st.markdown(f"**System Reference Key:** `{row['Intern_ID']}`")
            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2, gap="large")
            with col_left:
                st.markdown("### 🧩 Core Competency Matrix")
                skills_list = [s.strip() for s in str(row['normalized_skills']).split(",") if s.strip()]
                if skills_list:
                    for skill in skills_list:
                        st.markdown(f"🔹 **{skill.title()}**")
                else:
                    st.caption("No validated active skills indexed for this profile.")
                    
            with col_right:
                st.markdown("### 📉 Market Discrepancy Evaluation")
                if str(row['missing_skills']).strip():
                    st.error(f"**Identified Target Gaps vs Market Demand:**\n\n`{row['missing_skills']}`")
                    
                    st.markdown("#### 🛠&nbsp; Strategic Upskilling Roadmap")
                    personal_recommendations = suggest_training(row['missing_skills_list'], top_k=5)
                    for rec in personal_recommendations:
                        st.markdown(f"💡 **{rec.replace('💡', '').strip()}**")
                else:
                    st.success("🏆 **Verified Competency Alignment:** Candidate satisfies all primary cluster target criteria for this domain track.")

        with tab_global:
            st.markdown("### Market Competency Profiles & Group Inefficiencies")
            
            col_g1, col_g2 = st.columns(2, gap="large")
            with col_g1:
                st.markdown("#### Most Frequent Deficiencies Across Cohort")
                st.dataframe(res["top_missing_df"], use_container_width=True, hide_index=True)
            with col_g2:
                st.markdown("#### Macro Upskilling Strategies (Top 10 Priority Tracks)")
                global_skills = res["top_missing_df"]["skill"].tolist() if "skill" in res["top_missing_df"].columns else []
                if global_skills:
                    group_recommendations = suggest_training(global_skills, top_k=10)
                    for suggestion in group_recommendations:
                        st.markdown(f"✔️ {suggestion.replace('💡', '').strip()}")
                else:
                    st.caption("No major cohort deficiencies registered.")

            st.markdown("---")
            st.markdown("#### Complete Group Evaluation Output Report")
            st.dataframe(
                res["gaps_df"][["Intern_ID", "Name", "normalized_skills", "missing_count", "missing_skills"]],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export Full Performance Audit Report (CSV)",
                data=res["gaps_df"].to_csv(index=False).encode("utf-8"),
                file_name="cohort_skill_gaps_report.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab_vocabulary:
            st.markdown("#### Derived Industry Competency Proxy Words")
            st.caption("Aggregated vocabulary space generated from K-Means centroids based on input configurations.")
            st.code(", ".join(res["industry_skills"][:150]))
else:
    st.info("👋 Welcome! Please upload both student and requirement datasets via the side navigation panel to initialize calculations.")