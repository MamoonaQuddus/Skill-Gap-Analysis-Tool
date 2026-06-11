# src/training_catalog.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingItem:
    skill: str
    recommendation: str

DEFAULT_TRAINING_CATALOG: dict[str, TrainingItem] = {
    "python": TrainingItem("python", "Python fundamentals → functions, OOP, typing; practice with small projects."),
    "pandas": TrainingItem("pandas", "Data wrangling with pandas → groupby, merge, time series; Kaggle micro-courses."),
    "numpy": TrainingItem("numpy", "NumPy basics → vectorization, broadcasting; implement ML math from scratch."),
    "sql": TrainingItem("sql", "SQL for analytics → joins, window functions, CTEs; build reports from a sample DB."),
    "excel": TrainingItem("excel", "Excel for analysts → pivots, lookups, charts; recreate dashboards from datasets."),
    "statistics": TrainingItem("statistics", "Stats basics → distributions, hypothesis testing, confidence intervals."),
    "machine learning": TrainingItem("machine learning", "Supervised ML → train/val split, leakage, baselines, metrics."),
    "scikit-learn": TrainingItem("scikit-learn", "Modeling with scikit-learn → pipelines, CV, feature engineering."),
    "deep learning": TrainingItem("deep learning", "Neural nets → backprop intuition, CNN/RNN/Transformers overview."),
    "nlp": TrainingItem("nlp", "NLP basics → tokenization, TF‑IDF, embeddings; build a text classifier."),
    "tensorflow": TrainingItem("tensorflow", "TensorFlow/Keras → training loops, callbacks, deployment basics."),
    "pytorch": TrainingItem("pytorch", "PyTorch → tensors, autograd, training loops; reproduce MNIST baseline."),
    "mlops": TrainingItem("mlops", "MLOps → experiment tracking, packaging, CI, model monitoring basics."),
    "docker": TrainingItem("docker", "Docker → containers, images, compose; containerize a ML app."),
    "git": TrainingItem("git", "Git workflow → branches, PRs, resolving conflicts; use GitHub effectively."),
    "power bi": TrainingItem("power bi", "Power BI → data model, DAX basics; build a KPI dashboard."),
    "tableau": TrainingItem("tableau", "Tableau → visual best practices; create an interactive dashboard."),
    "communication": TrainingItem("communication", "Communication → write 1‑page project updates; present results weekly."),
}

def suggest_training(skills: list[str], top_k: int = 5) -> list[str]:
    seen: set[str] = set()
    suggestions: list[str] = []
    for s in skills:
        key = s.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        item = DEFAULT_TRAINING_CATALOG.get(key)
        if item is not None:
            suggestions.append(f"💡 {item.skill.upper()}: {item.recommendation}")
        if len(suggestions) >= top_k:
            break
            
    if suggestions:
        return suggestions
    return [
        "Focus on building hands-on projects around missing technical competencies.",
        "Apply the 70/20/10 learning framework: 70% building, 20% code reviews, 10% explicit courses."
    ]