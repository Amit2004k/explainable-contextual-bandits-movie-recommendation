# 🎬 Explainable Contextual Bandits for Movie Recommendation: An Empirical Study on MovieLens

[![Paper](https://img.shields.io/badge/Springer%20LNNS-SmartCom%202026-blue?style=flat-square&logo=springer)](https://springer.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python)](https://python.org)
[![RL](https://img.shields.io/badge/RL-Contextual%20Bandits-orange?style=flat-square)]()
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20Attention-purple?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Amit2004k/explainable-contextual-bandits-movie-recommendation?style=flat-square)](https://github.com/Amit2004k/explainable-contextual-bandits-movie-recommendation/stargazers)

> **Accepted and Presented at SmartCom 2026 — Springer Lecture Notes in Networks and Systems (LNNS)**
> *Explainable Contextual Bandits for Movie Recommendation: An Empirical Study on MovieLens*

---

## 🧠 Overview

Modern recommender systems are black boxes — users don't know *why* a movie is recommended. This work combines **Contextual Bandits** (an online RL framework) with **Explainable AI** to build a recommendation system that is both effective and transparent.

Unlike collaborative filtering which learns from static offline data, contextual bandits **adapt in real-time** from user feedback — making recommendations better with every interaction.

---

## 🔥 Key Contributions

- ✅ **Contextual Bandit framework** — LinUCB, Neural LinUCB, and Thompson Sampling compared
- ✅ **Explainability layer** — SHAP values per recommendation + attention-based feature importance
- ✅ **Online learning** — model updates with each user interaction (no retraining required)
- ✅ **Cold-start handling** — content-based context features mitigate new user/item cold-start
- ✅ **Empirical study** — comprehensive comparison on MovieLens-100K and MovieLens-1M
- ✅ **User simulation** — realistic evaluation via logged feedback simulation

---

## 📊 Results at a Glance

### MovieLens-1M

| Method | Cumulative Reward | CTR | NDCG@10 | Explainability |
|--------|------------------|-----|---------|----------------|
| Random | 0.312 | 21.3% | 0.281 | ❌ |
| ε-Greedy | 0.441 | 31.2% | 0.389 | ❌ |
| LinUCB | 0.578 | 42.1% | 0.501 | ✅ (linear) |
| Neural LinUCB | 0.621 | 45.8% | 0.537 | ✅ (SHAP) |
| **Thompson Sampling + XAI (Ours)** | **0.649** | **47.3%** | **0.558** | **✅ (full)** |

> Our explainable Thompson Sampling achieves **+47.7%** cumulative reward over the random baseline, while providing per-recommendation explanations.

---

## 🏗️ Framework Architecture

```
User Context (age, genre prefs, history)
Movie Context (genre, year, avg rating)
           │
           ▼
┌─────────────────────────┐
│  Context Encoder        │  ← Concat user + item features
│                         │  ← Optional: neural embedding layer
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Contextual Bandit      │
│  ┌───────────────────┐  │
│  │  LinUCB           │  │  ← Ridge regression + UCB exploration
│  │  Neural LinUCB    │  │  ← Deep features + linear head
│  │  Thompson Sampling│  │  ← Bayesian posterior sampling
│  └───────────────────┘  │
└──────────┬──────────────┘
           │  Recommended Movie
           ▼
┌─────────────────────────┐
│  User Feedback          │  ← Click / rating / watch-time
│  (Reward Signal)        │
└──────────┬──────────────┘
           │  Update bandit parameters
           ▼
┌─────────────────────────┐
│  XAI Module             │
│  ├── SHAP values        │  ← Why this movie? Feature contributions
│  ├── Attention weights  │  ← Which context features mattered
│  └── Counterfactuals    │  ← "You'd get X if you liked more action"
└─────────────────────────┘
```

---

## 📁 Repository Structure

```
📦 explainable-contextual-bandits-movie-recommendation
├── 📂 src/
│   ├── bandits.py             # LinUCB, Neural LinUCB, Thompson Sampling
│   ├── context_encoder.py     # User + movie feature engineering
│   ├── simulator.py           # User feedback simulation
│   ├── explainability.py      # SHAP + attention explanations
│   └── evaluation.py          # Cumulative reward, CTR, NDCG
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bandit_training.ipynb
│   ├── 03_online_simulation.ipynb
│   ├── 04_xai_explanations.ipynb
│   └── 05_empirical_comparison.ipynb
├── 📂 data/
│   └── README.md              # MovieLens download instructions
├── 📂 results/
│   ├── 📂 figures/
│   └── 📂 tables/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amit2004k/explainable-contextual-bandits-movie-recommendation.git
cd explainable-contextual-bandits-movie-recommendation
pip install -r requirements.txt
```

### Download MovieLens:

```python
# MovieLens-100K (auto-download via surprise)
from surprise import Dataset
data = Dataset.load_builtin('ml-100k')

# MovieLens-1M (manual)
# https://grouplens.org/datasets/movielens/1m/
```

### Run LinUCB bandit:

```python
from src.bandits import LinUCB
from src.context_encoder import MovieLensContextEncoder

encoder = MovieLensContextEncoder(n_genres=18)
bandit = LinUCB(context_dim=encoder.dim, alpha=1.0)

# Simulate online recommendation loop
for user_id, candidate_movies in stream:
    contexts = encoder.encode(user_id, candidate_movies)
    chosen = bandit.select(contexts)
    reward = get_user_feedback(user_id, chosen)
    bandit.update(contexts[chosen], reward)
```

### Get SHAP explanation for a recommendation:

```python
from src.explainability import BanditExplainer

explainer = BanditExplainer(bandit, encoder)
explanation = explainer.explain(user_id, movie_id)
print(explanation)
# → "Recommended because: Action genre (+0.34), 
#    Similar to your top-rated films (+0.28), 
#    High avg rating (+0.19)"
```

---

## 🗂️ Dataset

**MovieLens** (GroupLens Research):
- **ML-100K**: 100,000 ratings, 943 users, 1,682 movies
- **ML-1M**: 1,000,209 ratings, 6,040 users, 3,706 movies
- Download: https://grouplens.org/datasets/movielens/

---

## 📖 Citation

```bibtex
@inproceedings{kalita2026bandits,
  title     = {Explainable Contextual Bandits for Movie Recommendation: An Empirical Study on MovieLens},
  author    = {Kalita, Amit and others},
  booktitle = {Proceedings of SmartCom 2026},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026}
}
```

---

## 🙋 Author

**Amit Kalita**
B.Tech CSE (8th Semester), Dibrugarh University
[GitHub](https://github.com/Amit2004k)

> 📌 *Part of a series of published ML research repos. See also:
> [SSL Pneumonia](https://github.com/Amit2004k/ssl-pneumonia-chest-xray) |
> [Alzheimer's + Quantum](https://github.com/Amit2004k/alzheimers-quantum-gat-mri) |
> [DDI Prediction](https://github.com/Amit2004k/drug-drug-interaction-llm-xai) |
> [Fraud Detection](https://github.com/Amit2004k/fraud-detection-cost-sensitive-xai)*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

⭐ **Star this repo if you work on recommender systems, bandits, or explainable RL!**
