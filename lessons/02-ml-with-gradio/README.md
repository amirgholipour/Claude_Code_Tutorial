# Lesson 2: Machine Learning & Deep Learning with Gradio

## Overview

Build a fully interactive ML course as a Gradio web application. This lesson covers the complete machine learning spectrum — from exploratory data analysis through classical algorithms to deep learning with PyTorch, specialized domains (Time Series, NLP), MLOps, Responsible AI, and production system design — all with live, adjustable demos.

By the end of this lesson you will have:
- A running Gradio app with **26 interactive ML modules**
- Hands-on experience with scikit-learn, PyTorch, and Plotly
- Understanding of the full ML lifecycle: data → feature engineering → model → evaluate → monitor → deploy

## Learning Path

```
🟢 BASIC ML
  01 · Data Exploration & Preprocessing
  02 · Regression (Linear, Ridge, Lasso, Polynomial)
  03 · Classification (Logistic Reg, k-NN, Naive Bayes, Decision Tree)
  04 · Model Evaluation (Accuracy, F1, ROC-AUC, Cross-Validation)

🟡 INTERMEDIATE ML
  05 · Ensemble Methods (Random Forest, Gradient Boosting, AdaBoost)
  06 · Unsupervised Learning (K-Means, DBSCAN, Hierarchical)
  07 · Dimensionality Reduction (PCA, t-SNE, LDA)

🔴 DEEP LEARNING
  08 · Neural Networks Fundamentals (MLP, activations, backprop)
  09 · Convolutional Neural Networks (CNN for digit recognition)
  10 · Recurrent Neural Networks (LSTM/GRU for sequence prediction)
  11 · Training Best Practices (Dropout, BatchNorm, LR schedulers)

🟣 ADVANCED
  12 · Transfer Learning (feature extraction vs fine-tuning)
  13 · Model Explainability (SHAP, feature importance, PDP)
  14 · End-to-End ML Pipeline (build → train → save → predict)

🔵 DATA FOUNDATION
  15 · Data Preparation & Cleaning (imputation, outliers, encoding)
  16 · Feature Engineering (polynomial, log, cyclical, interactions)
  17 · Feature Selection (filter, wrapper RFE, embedded L1/tree)
  18 · Advanced EDA (distributions, correlations, outlier profiles)

🟠 SPECIALIZED ML
  19 · Time Series Analysis (decomposition, stationarity, forecasting)
  20 · Natural Language Processing (TF-IDF, text classification, NLP)

⚫ ML ENGINEERING
  21 · MLOps & Model Monitoring (drift detection, PSI, versioning)
  22 · CI/CD for ML (data validation, performance gates, deploy pipeline)

🔶 RESPONSIBLE & DESIGN
  23 · Responsible AI (fairness metrics, bias detection, disparate impact)
  24 · ML System Design (batch vs online, feature stores, latency tradeoffs)

🔷 ADVANCED TECHNIQUES
  25 · Hyperparameter Tuning (grid search, random search, validation curves)
  26 · Anomaly Detection (Isolation Forest, LOF, One-Class SVM)
```

## Prerequisites

- [ ] Python 3.9+ installed: `python --version`
- [ ] pip available: `pip --version`
- [ ] Completed Lesson 1 (GitHub repo basics)
- [ ] ~2 GB free disk space (for PyTorch)

## Quick Start

```bash
# 1. Install uv (fast Rust-based package installer — one-time setup)
pip install uv

# 2. Create a virtual environment
python -m venv .venv

# Activate it:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Navigate to the app directory
cd lessons/02-ml-with-gradio/app

# 4. Install dependencies (~2-5 minutes with uv, vs 10+ with pip)
uv pip install -r requirements.txt

# CPU-only PyTorch (faster download, no GPU needed):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Launch the app
python app.py
# → Opens at http://localhost:7861
```

Or use the slash command (in Claude Code):
```
/ml-run-app
```

## Exercises

Work through these in order. Each exercise maps to one or more app modules.

| # | File | App Modules | Level |
|---|---|---|---|
| 1 | [01-gradio-basics.md](./exercises/01-gradio-basics.md) | Setup | 🟢 Basic |
| 2 | [02-data-exploration.md](./exercises/02-data-exploration.md) | Module 01 | 🟢 Basic |
| 3 | [03-classical-ml.md](./exercises/03-classical-ml.md) | Modules 02–04 | 🟢 Basic |
| 4 | [04-ensemble-methods.md](./exercises/04-ensemble-methods.md) | Modules 05–07 | 🟡 Intermediate |
| 5 | [05-deep-learning-intro.md](./exercises/05-deep-learning-intro.md) | Modules 08–11 | 🔴 Deep Learning |
| 6 | [06-full-pipeline.md](./exercises/06-full-pipeline.md) | Modules 12–14 | 🟣 Advanced |
| 7 | [07-data-preparation.md](./exercises/07-data-preparation.md) | Modules 15–18 | 🔵 Foundation |
| 8 | [08-time-series-nlp.md](./exercises/08-time-series-nlp.md) | Modules 19–20 | 🟠 Specialized |
| 9 | [09-mlops-responsible-ai.md](./exercises/09-mlops-responsible-ai.md) | Modules 21–26 | ⚫🔶🔷 Production |

## App Structure

```
app/
├── app.py               ← Main entry point (run this)
├── requirements.txt     ← Dependencies
├── config.py            ← App config & dataset registry
├── modules/             ← 26 ML topic modules
│   ├── m01_data_exploration.py
│   ├── m02_regression.py
│   ├── ...
│   ├── m14_ml_pipeline.py
│   ├── m15_data_preparation.py
│   ├── ...
│   └── m26_anomaly_detection.py
└── utils/
    ├── data_utils.py    ← Dataset loaders & preprocessing
    └── plot_utils.py    ← Plotly chart helpers
```

## Available Slash Commands

```
/ml-run-app     — Install deps (if needed) and launch the Gradio app
/ml-setup       — Install all requirements for this lesson
```

## Datasets Used

All datasets are **built-in** — no internet required after first install.

| Dataset | Source | Task | Modules |
|---|---|---|---|
| Iris | sklearn | Classification | 03, 06, 14, 17, 25 |
| Breast Cancer | sklearn | Binary Classification | 04, 13, 17, 21, 22, 23, 26 |
| Wine | sklearn | Classification | 05, 17 |
| Diabetes | sklearn | Regression | 02, 16 |
| Digits | sklearn | Classification | 09 |
| Synthetic blobs/moons | sklearn | Clustering | 06, 26 |
| Sine wave | Synthetic | RNN / Time Series | 10, 19 |
| Synthetic dirty data | Generated | Data Prep | 15 |
| 20 Newsgroups | sklearn | NLP Text Classification | 20 |
| Synthetic loan data | Generated | Responsible AI | 23 |

> **Note**: Module 20 (NLP) downloads ~15 MB of text data on first run. Module 09 (CNN) downloads ~11 MB of MNIST on first run.

## Time Estimate

| Section | Modules | Time |
|---|---|---|
| Setup + Gradio basics | Exercise 1 | 30 min |
| Classical ML | Exercises 2–3 | 1–2 hours |
| Ensemble + Unsupervised | Exercise 4 | 1 hour |
| Deep Learning | Exercise 5 | 2–3 hours |
| Advanced + Pipeline | Exercise 6 | 1–2 hours |
| Data Foundation | Exercise 7 | 1–2 hours |
| Time Series + NLP | Exercise 8 | 1–2 hours |
| MLOps + Production | Exercise 9 | 2–3 hours |
| **Total** | **26 modules** | **10–16 hours** |

---

Ready? Start with [Exercise 1 →](./exercises/01-gradio-basics.md)
