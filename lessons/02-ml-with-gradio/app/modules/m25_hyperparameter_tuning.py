"""
Module 25 — Hyperparameter Tuning & AutoML
Level: Intermediate / Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, validation_curve, learning_curve
)
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")

THEORY = """
## 📖 What Is Hyperparameter Tuning?

**Hyperparameters** are model settings configured *before* training (unlike parameters, which are learned during training). Tuning them is critical for squeezing out the best model performance.

| Type | Learned during training? | Example |
|---|---|---|
| **Parameters** | Yes | Linear regression weights (w₁, w₂, ...) |
| **Hyperparameters** | No | Learning rate, max depth, n_estimators |

## 🏗️ Search Strategies

### 1. Grid Search (Exhaustive)
Tests every combination of specified hyperparameter values:
```
params = {"max_depth": [3, 5, 10], "n_estimators": [50, 100, 200]}
# Tests: 3 × 3 = 9 combinations × CV folds
```
- **Pros**: Guaranteed to find the best in the grid
- **Cons**: Exponential growth — 5 params × 5 values = 5⁵ = 3125 fits!

### 2. Random Search (Sampling)
Samples random combinations from parameter distributions:
```python
param_distributions = {
    "max_depth":     randint(3, 20),
    "n_estimators":  randint(50, 500),
    "max_features":  uniform(0.1, 0.9),
}
```
- **Pros**: More efficient than grid search for high-dimensional spaces
- **Key insight**: Each iteration covers all dimensions — a missed value in one param is compensated by exploring others
- **Rule**: Random search > Grid search for ≥4 hyperparameters

### 3. Bayesian Optimization
Models the objective function (validation score) as a Gaussian Process and selects next points to evaluate intelligently:

```
previous_results → GP model → acquisition function → next point to try
```
- **Pros**: Most sample-efficient — needs fewer evaluations
- **Cons**: Complex to implement, overhead per iteration
- **Tools**: `scikit-optimize` (skopt), `Optuna`, `Hyperopt`, `Ax`

### 4. Successive Halving (Bandit-based)
- Start with many configs, few resources (few training samples/epochs)
- Keep top half, double resources, repeat
- **Much faster** than full cross-validation for neural networks

## ✅ Practical Guidelines

| Rule | Recommendation |
|---|---|
| n_hyperparams ≤ 3 | Grid search OK |
| n_hyperparams 4–10 | Random search (n_iter = 50–100) |
| Expensive evaluation | Bayesian optimization |
| Neural networks | Successive halving, Optuna |

**Always use cross-validation** when tuning — never evaluate on test set until final model selection.

**Hold out a final test set** — the test set is used exactly once for the final evaluation.

## 📊 Learning vs Validation Curves

- **Learning curve**: How training and validation score change with *more training data*
  - Diagnosing underfitting (bias) vs overfitting (variance)
- **Validation curve**: How training and validation score change with *one hyperparameter*
  - Finding optimal hyperparameter range

```
         High training / Low validation → Overfit (reduce complexity)
         Low training / Low validation  → Underfit (increase complexity)
         Both converge                  → Good fit
```

## ⚠️ Common Pitfalls
- **Overfitting to validation**: If you tune too many hyperparameters, you overfit to the CV folds
- **Wrong search space**: Searching 1–100 for learning rate instead of log-scale [1e-4, 1e-1]
- **Ignoring compute budget**: Grid search with 10 params = infeasible
- **Multiple testing**: Running 1000 random configs and reporting the best inflates performance
"""

CODE_EXAMPLE = '''
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    validation_curve, learning_curve
)
from scipy.stats import randint, uniform
import numpy as np, time

X, y = load_breast_cancer(return_X_y=True)

# ── Grid Search ─────────────────────────────────────────────────
grid_params = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [3, 5, 10, None],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    grid_params, cv=5, n_jobs=-1, verbose=0)
t0 = time.time()
grid.fit(X, y)
print(f"Grid  best: {grid.best_score_:.4f}  time: {time.time()-t0:.1f}s")

# ── Random Search ────────────────────────────────────────────────
rand_params = {
    "n_estimators": randint(50, 500),
    "max_depth":    randint(3, 20),
    "max_features": uniform(0.1, 0.9),
    "min_samples_split": randint(2, 20),
}
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                          rand_params, n_iter=50, cv=5,
                          n_jobs=-1, random_state=42)
t0 = time.time()
rand.fit(X, y)
print(f"Random best: {rand.best_score_:.4f}  time: {time.time()-t0:.1f}s")

# ── Validation curve (max_depth) ─────────────────────────────────
from sklearn.model_selection import validation_curve
depths = [1, 2, 3, 5, 8, 12, 20, None]
train_sc, val_sc = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, param_name="max_depth", param_range=depths, cv=5
)
print("Optimal max_depth:", depths[val_sc.mean(axis=1).argmax()])
'''


def run_hyperparameter_tuning(dataset_name: str, demo_type: str, n_iter: int):
    DATASETS = {
        "Breast Cancer": load_breast_cancer,
        "Wine":          load_wine,
        "Iris":          load_iris,
    }
    data = DATASETS[dataset_name]()
    X, y = data.data, data.target

    if demo_type == "Grid vs Random Search":
        # Small grid for speed
        grid_params = {
            "n_estimators": [50, 100],
            "max_depth":    [5, 10, None],
            "max_features": ["sqrt", "log2"],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            grid_params, cv=5, n_jobs=-1, scoring="accuracy"
        )
        t0 = time.time()
        grid.fit(X, y)
        t_grid = time.time() - t0
        grid_results = pd.DataFrame(grid.cv_results_)

        rand_params = {
            "n_estimators": [int(x) for x in np.random.randint(50, 300, 100)],
            "max_depth":    [int(x) for x in np.random.randint(3, 20, 100)],
            "max_features": ["sqrt", "log2", None],
        }
        rand = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": list(range(50, 300, 10)),
             "max_depth":    list(range(3, 20)) + [None],
             "max_features": ["sqrt", "log2"]},
            n_iter=min(n_iter, 30), cv=5, n_jobs=-1,
            random_state=42, scoring="accuracy"
        )
        t0 = time.time()
        rand.fit(X, y)
        t_rand = time.time() - t0

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Score Distribution", "Best Score vs Compute"])

        # Score distribution
        fig.add_trace(go.Histogram(x=grid_results["mean_test_score"], nbinsx=10,
                                   name="Grid Search", marker_color="#42a5f5", opacity=0.7), row=1, col=1)
        rand_results = pd.DataFrame(rand.cv_results_)
        fig.add_trace(go.Histogram(x=rand_results["mean_test_score"], nbinsx=10,
                                   name="Random Search", marker_color="#66bb6a", opacity=0.7), row=1, col=1)
        fig.update_layout(barmode="overlay")

        # Comparison bar
        fig.add_trace(go.Bar(
            x=["Grid Search", "Random Search"],
            y=[grid.best_score_, rand.best_score_],
            marker_color=["#42a5f5", "#66bb6a"],
            text=[f"{grid.best_score_:.4f}", f"{rand.best_score_:.4f}"],
            textposition="outside", showlegend=False
        ), row=1, col=2)
        fig.update_yaxes(range=[max(0, min(grid.best_score_, rand.best_score_)-0.02), 1.02], row=1, col=2)

        fig.update_layout(height=400, title_text=f"Grid vs Random Search — {dataset_name}")

        metrics_md = f"""
### Grid vs Random Search

| | Grid Search | Random Search |
|---|---|---|
| Best score | `{grid.best_score_:.4f}` | `{rand.best_score_:.4f}` |
| Configs evaluated | `{len(grid_results)}` | `{len(rand_results)}` |
| Time | `{t_grid:.1f}s` | `{t_rand:.1f}s` |
| Best params | {str(grid.best_params_)[:50]} | {str(rand.best_params_)[:50]} |

> Random search evaluated fewer configs but found {'comparable' if abs(rand.best_score_ - grid.best_score_) < 0.005 else 'different'} performance.
"""

    elif demo_type == "Validation Curve":
        depths = [1, 2, 3, 5, 8, 12, 20]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y,
            param_name="max_depth", param_range=depths,
            cv=5, scoring="accuracy"
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=depths, y=train_mean, name="Training score",
                                 line=dict(color="#42a5f5", width=2),
                                 mode="lines+markers"))
        fig.add_trace(go.Scatter(
            x=depths + depths[::-1],
            y=list(train_mean + train_std) + list(train_mean - train_std)[::-1],
            fill="toself", fillcolor="rgba(66,165,245,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False
        ))
        fig.add_trace(go.Scatter(x=depths, y=val_mean, name="Validation score",
                                 line=dict(color="#ef5350", width=2),
                                 mode="lines+markers"))
        fig.add_trace(go.Scatter(
            x=depths + depths[::-1],
            y=list(val_mean + val_std) + list(val_mean - val_std)[::-1],
            fill="toself", fillcolor="rgba(239,83,80,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False
        ))

        best_depth = depths[val_mean.argmax()]
        fig.add_vline(x=best_depth, line_dash="dash", line_color="green",
                      annotation_text=f"Best: {best_depth}")
        fig.update_layout(height=420, xaxis_title="max_depth",
                          yaxis_title="Accuracy", title_text=f"Validation Curve — max_depth ({dataset_name})")

        metrics_md = f"""
### Validation Curve Analysis

| max_depth | Train | Val |
|---|---|---|
{''.join([f"| {d} | {tm:.4f} | {vm:.4f} |\n" for d, tm, vm in zip(depths, train_mean, val_mean)])}
**Optimal max_depth:** `{best_depth}` (val acc = `{val_mean.max():.4f}`)

- Small depth → **underfitting** (both scores low)
- Large depth → **overfitting** (train high, val drops)
- Sweet spot: depth where val score peaks
"""

    elif demo_type == "Learning Curve":
        train_sizes, train_scores, val_scores = learning_curve(
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring="accuracy", n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, name="Training score",
                                 line=dict(color="#42a5f5", width=2), mode="lines+markers"))
        fig.add_trace(go.Scatter(
            x=list(train_sizes) + list(train_sizes[::-1]),
            y=list(train_mean + train_std) + list(train_mean - train_std)[::-1],
            fill="toself", fillcolor="rgba(66,165,245,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False
        ))
        fig.add_trace(go.Scatter(x=train_sizes, y=val_mean, name="Validation score",
                                 line=dict(color="#ef5350", width=2), mode="lines+markers"))
        fig.add_trace(go.Scatter(
            x=list(train_sizes) + list(train_sizes[::-1]),
            y=list(val_mean + val_std) + list(val_mean - val_std)[::-1],
            fill="toself", fillcolor="rgba(239,83,80,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False
        ))

        fig.update_layout(height=420, xaxis_title="Training samples",
                          yaxis_title="Accuracy", title_text=f"Learning Curve ({dataset_name})")

        gap = train_mean[-1] - val_mean[-1]
        verdict = "High variance (overfitting)" if gap > 0.05 else \
                  "High bias (underfitting)" if val_mean[-1] < 0.85 else "Good fit"

        metrics_md = f"""
### Learning Curve Analysis

| | With {int(train_sizes[0])} samples | With {int(train_sizes[-1])} samples |
|---|---|---|
| Train score | `{train_mean[0]:.4f}` | `{train_mean[-1]:.4f}` |
| Val score | `{val_mean[0]:.4f}` | `{val_mean[-1]:.4f}` |
| Gap | `{train_mean[0]-val_mean[0]:.4f}` | `{gap:.4f}` |

**Diagnosis:** {verdict}

{'> Adding more data would likely help — val score still rising' if val_mean[-1] - val_mean[-3] > 0.005 else '> Val score plateaued — more data alone won\'t help'}
"""

    else:
        fig = go.Figure()
        metrics_md = "Select a demo type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🔧 Module 25 — Hyperparameter Tuning & AutoML\n*Level: Intermediate / Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nCompare Grid vs Random search, visualize validation curves to find optimal hyperparameters, and diagnose bias vs variance with learning curves.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Breast Cancer", "Wine", "Iris"],
                value="Breast Cancer"
            )
            demo_dd = gr.Dropdown(
                label="Demo Type",
                choices=["Grid vs Random Search", "Validation Curve", "Learning Curve"],
                value="Validation Curve"
            )
            n_iter_sl = gr.Slider(label="Random Search Iterations", minimum=10, maximum=50,
                                  step=5, value=20)
            run_btn = gr.Button("▶ Run Tuning Demo", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_hyperparameter_tuning,
        inputs=[dataset_dd, demo_dd, n_iter_sl],
        outputs=[plot_out, metrics_out]
    )
