# src/pipeline.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9\+\#]+", flags=re.IGNORECASE)

def _basic_normalize(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = t.replace("c++", "cplusplus").replace("c#", "csharp")
    t = re.sub(r"\s+", " ", t)
    return t

def _tokens_from_text(text: str) -> list[str]:
    t = _basic_normalize(text)
    return [p for p in _TOKEN_SPLIT_RE.split(t) if p]

def normalize_skill_list(skills_text: str) -> list[str]:
    """Turns comma-separated lists or free text blocks into a clean de-duplicated list."""
    toks = _tokens_from_text(skills_text)
    stop = {
        "and", "or", "with", "in", "on", "to", "of", "the", "a", "an", "for",
        "experience", "knowledge", "skills", "ability", "good", "strong", 
        "excellent", "plus", "required", "preferred", "job", "responsibilities"
    }
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok in stop or len(tok) <= 1:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out

@dataclass(frozen=True)
class FitResult:
    vectorizer: TfidfVectorizer
    kmeans: KMeans
    job_tfidf: np.ndarray
    job_clusters: np.ndarray
    cluster_top_terms: dict[int, list[str]]

def fit_job_clusters(
    jobs_df: pd.DataFrame,
    text_col: str = "Description",
    n_clusters: int = 5,
    max_features: int = 5000,
    random_state: int = 42,
) -> FitResult:
    if text_col not in jobs_df.columns:
        raise ValueError(f"Missing required column `{text_col}` in jobs dataframe.")
        
    corpus = jobs_df[text_col].fillna("").astype(str).map(_basic_normalize).tolist()
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    X = vectorizer.fit_transform(corpus)
    
    k = max(2, min(int(n_clusters), X.shape[0])) if X.shape[0] > 0 else 2
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_top_terms: dict[int, list[str]] = {}
    centers = kmeans.cluster_centers_
    
    for c in range(centers.shape[0]):
        top_idx = np.argsort(centers[c])[::-1][:20]
        cluster_top_terms[int(c)] = feature_names[top_idx].tolist()
        
    return FitResult(
        vectorizer=vectorizer,
        kmeans=kmeans,
        job_tfidf=X,
        job_clusters=labels,
        cluster_top_terms=cluster_top_terms,
    )

def extract_industry_skill_vocabulary(jobs_df: pd.DataFrame, fit: FitResult, top_terms_per_cluster: int = 20) -> list[str]:
    """Aggregates heavily weighted tokens from calculated clusters to build an industry profile baseline."""
    terms: list[str] = []
    for _, top in sorted(fit.cluster_top_terms.items(), key=lambda x: x[0]):
        terms.extend(top[:top_terms_per_cluster])
        
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in terms:
        key = _basic_normalize(t)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned

def compute_intern_gaps(
    interns_df: pd.DataFrame,
    industry_skills: list[str],
    skills_col: str = "Skills",
    top_k_missing: int = 10,
) -> pd.DataFrame:
    if skills_col not in interns_df.columns:
        raise ValueError(f"Missing required column `{skills_col}` in interns dataframe.")
        
    ind_set = set([_basic_normalize(s) for s in industry_skills if s])
    rows = []
    
    for _, r in interns_df.iterrows():
        raw = r.get(skills_col, "")
        intern_tokens = normalize_skill_list(str(raw))
        intern_set = set([_basic_normalize(s) for s in intern_tokens])
        
        # Unsupervised Set Math Matrix logic: Industry Requirements - Intern Profile
        missing = sorted(list(ind_set - intern_set))
        
        rows.append(
            {
                **{c: r.get(c) for c in interns_df.columns},
                "normalized_skills": ", ".join(intern_tokens),
                "missing_skills_list": missing[:top_k_missing],
                "missing_skills": ", ".join(missing[:top_k_missing]),
                "missing_count": int(len(missing)),
            }
        )
    return pd.DataFrame(rows)

def summarize_top_missing(gaps_df: pd.DataFrame, missing_col: str = "missing_skills", top_n: int = 25) -> pd.DataFrame:
    """Aggregates and metrics the most common skill deficiencies discovered across the entire active pool."""
    counter: dict[str, int] = {}
    for s in gaps_df[missing_col].fillna("").astype(str).tolist():
        if not s.strip():
            continue
        for part in s.split(","):
            k = _basic_normalize(part)
            if not k:
                continue
            counter[k] = counter.get(k, 0) + 1
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    return pd.DataFrame(items, columns=["skill", "interns_missing"])