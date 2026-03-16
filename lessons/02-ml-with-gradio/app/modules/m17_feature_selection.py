"""
Module 17 — Feature Selection
Level: Intermediate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

THEORY = """
## 📖 What Is Feature Selection?

Feature selection is the process of **choosing the most informative subset of features** for a model. Unlike feature engineering (creating new features), selection *removes* irrelevant or redundant ones.

**Why it matters:**
- Reduces overfitting (fewer noise features)
- Speeds up training and inference
- Improves interpretability
- Reduces storage and memory requirements

## 🏗️ Three Categories

### 1. Filter Methods (model-agnostic, fast)
Select features based on statistical measures, independent of any model:

| Method | Use case | Statistic |
|---|---|---|
| **Variance Threshold** | Remove near-constant features | Var(x) |
| **Correlation** | Remove redundant features | Pearson r |
| **ANOVA F-test** | Numeric X, categorical y | F-statistic |
| **Chi²** | Non-negative X, categorical y | χ² statistic |
| **Mutual Information** | Any relationship (non-linear OK) | MI score |

```python
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
```

### 2. Wrapper Methods (model-aware, slower)
Use a model to evaluate feature subsets:

- **RFE (Recursive Feature Elimination)**: Fits model, removes weakest feature, repeats
- **Forward/Backward Selection**: Greedy search adding/removing one feature at a time
- **Exhaustive Search**: Tests all 2^n subsets (only feasible for n < 20)

```python
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)
```

### 3. Embedded Methods (built into model training)
Feature importance determined as part of model fitting:

- **L1 (Lasso) Regularization**: Forces some coefficients to zero
- **Tree-based importance**: Random Forest/GBT built-in `feature_importances_`
- **Elastic Net**: Combination of L1 and L2

```python
model = Lasso(alpha=0.01)  # Some coefficients → exactly 0
model.fit(X_train, y_train)
selected = np.where(model.coef_ != 0)[0]
```

## ✅ Which Method to Choose?
| Situation | Method |
|---|---|
| Baseline exploration, fast | Filter (ANOVA / MI) |
| Best possible subset | Wrapper (RFE) |
| Tree-based model | Embedded (feature_importances_) |
| Linear model + sparse | Embedded (Lasso L1) |
| High-dimensional (>1000 features) | Filter first, then embedded |

## ⚠️ Common Pitfalls
- **Leakage**: Fitting selector on full dataset — always fit on training set only
- **Ignoring interactions**: Filter methods miss feature synergy
- **Overfitting to validation**: Wrapper methods can overfit the selection process
- **Dropping useful features**: A low-importance feature may be critical in combination
"""

CODE_EXAMPLE = '''
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

# ── Filter: ANOVA F-test ─────────────────────────────────────────
filter_pipe = Pipeline([
    ("select", SelectKBest(f_classif, k=10)),
    ("clf",    LogisticRegression(max_iter=1000))
])
filter_score = cross_val_score(filter_pipe, X, y, cv=5).mean()
print(f"Filter (k=10): {filter_score:.3f}")

# ── Wrapper: RFE ─────────────────────────────────────────────────
rfe_pipe = Pipeline([
    ("select", RFE(RandomForestClassifier(n_estimators=50), n_features_to_select=10)),
    ("clf",    LogisticRegression(max_iter=1000))
])
rfe_score = cross_val_score(rfe_pipe, X, y, cv=5).mean()
print(f"RFE (k=10):    {rfe_score:.3f}")

# ── Embedded: Tree importance ─────────────────────────────────────
emb_pipe = Pipeline([
    ("select", SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="median")),
    ("clf",    LogisticRegression(max_iter=1000))
])
emb_score = cross_val_score(emb_pipe, X, y, cv=5).mean()
print(f"Embedded:      {emb_score:.3f}")
'''


def run_feature_selection(dataset_name: str, method: str, k_features: int):
    DATASETS = {
        "Breast Cancer": load_breast_cancer,
        "Wine":          load_wine,
    }
    loader = DATASETS[dataset_name]
    data   = loader()
    X, y   = data.data, data.target
    feat_names = list(data.feature_names)
    n_features = X.shape[1]
    k = min(k_features, n_features)

    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X)

    # Baseline
    base_score = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0), X_sc, y, cv=5, scoring="accuracy"
    ).mean()

    # Select features
    if method == "Filter — ANOVA F-test":
        sel = SelectKBest(f_classif, k=k)
        sel.fit(X_sc, y)
        scores = sel.scores_
        mask   = sel.get_support()
        method_score = cross_val_score(
            Pipeline([("s", sel), ("c", LogisticRegression(max_iter=1000))]),
            X_sc, y, cv=5, scoring="accuracy"
        ).mean()

    elif method == "Filter — Mutual Information":
        sel = SelectKBest(mutual_info_classif, k=k)
        sel.fit(X_sc, y)
        scores = sel.scores_
        mask   = sel.get_support()
        method_score = cross_val_score(
            Pipeline([("s", sel), ("c", LogisticRegression(max_iter=1000))]),
            X_sc, y, cv=5, scoring="accuracy"
        ).mean()

    elif method == "Wrapper — RFE":
        rf  = RandomForestClassifier(n_estimators=50, random_state=42)
        sel = RFE(rf, n_features_to_select=k, step=1)
        sel.fit(X_sc, y)
        scores = sel.ranking_.astype(float)
        scores = 1.0 / scores  # invert: rank 1 → score 1.0
        mask   = sel.support_
        method_score = cross_val_score(
            Pipeline([("s", sel), ("c", LogisticRegression(max_iter=1000))]),
            X_sc, y, cv=5, scoring="accuracy"
        ).mean()

    elif method == "Embedded — Tree Importance":
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_sc, y)
        scores = rf.feature_importances_
        sorted_idx = np.argsort(scores)[::-1]
        mask = np.zeros(n_features, dtype=bool)
        mask[sorted_idx[:k]] = True
        # Use a fresh (unfitted) RF inside the pipeline so cross_val_score can fit each fold
        method_score = cross_val_score(
            Pipeline([
                ("s", SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42),
                                      max_features=k)),
                ("c", LogisticRegression(max_iter=1000))
            ]),
            X_sc, y, cv=5, scoring="accuracy"
        ).mean()

    else:
        scores = np.ones(n_features)
        mask   = np.ones(n_features, dtype=bool)
        method_score = base_score

    # Truncate feature names for display
    display_names = [f[:12] for f in feat_names]

    colors = ["#66bb6a" if m else "#ef5350" for m in mask]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Feature Scores (selected = green)",
                                        "Accuracy: All Features vs Selected"])

    # Feature importance bar
    order = np.argsort(scores)[::-1]
    fig.add_trace(go.Bar(
        x=[display_names[i] for i in order],
        y=[scores[i] for i in order],
        marker_color=[colors[i] for i in order],
        showlegend=False
    ), row=1, col=1)

    # Accuracy comparison
    fig.add_trace(go.Bar(
        x=["All features", f"Top {k} features"],
        y=[base_score, method_score],
        marker_color=["#bdbdbd", "#42a5f5"],
        text=[f"{base_score:.3f}", f"{method_score:.3f}"],
        textposition="outside",
        showlegend=False
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], title_text="Accuracy", row=1, col=2)

    fig.update_layout(height=420, title_text=f"Feature Selection — {method}")

    selected_names = [feat_names[i] for i in range(n_features) if mask[i]][:k]
    delta = method_score - base_score
    sign  = "+" if delta >= 0 else ""

    metrics_md = f"""
### Results on {dataset_name} ({n_features} → {k} features)

| | Accuracy |
|---|---|
| All {n_features} features | `{base_score:.4f}` |
| {k} selected features ({method.split('—')[0].strip()}) | `{method_score:.4f}` |
| Change | `{sign}{delta:.4f}` |

**Selected features:** {', '.join(f'`{n}`' for n in selected_names[:8])}{'...' if len(selected_names) > 8 else ''}

> Fewer features can match or beat all features — irrelevant features add noise.
"""
    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🎯 Module 17 — Feature Selection\n*Level: Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nCompare Filter, Wrapper, and Embedded methods. Observe how many features are needed to match full-feature accuracy.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Breast Cancer", "Wine"],
                value="Breast Cancer"
            )
            method_dd = gr.Dropdown(
                label="Selection Method",
                choices=["Filter — ANOVA F-test", "Filter — Mutual Information",
                         "Wrapper — RFE", "Embedded — Tree Importance"],
                value="Embedded — Tree Importance"
            )
            k_sl = gr.Slider(label="Number of Features to Select", minimum=3, maximum=20, step=1, value=10)
            run_btn = gr.Button("▶ Select Features", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_feature_selection,
        inputs=[dataset_dd, method_dd, k_sl],
        outputs=[plot_out, metrics_out]
    )
