"""
Module 16 — Feature Engineering
Level: Intermediate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

THEORY = """
## 📖 What Is Feature Engineering?

Feature engineering is the process of **creating new features** from raw data to improve model performance. It's one of the most impactful skills in applied ML — a simple model with great features often beats a complex model with mediocre ones.

The key insight: **most algorithms can only capture relationships that exist explicitly in the feature space.** A linear model can't detect a quadratic relationship unless you add `x²` as a feature.

## 🏗️ Core Techniques

### 1. Polynomial & Interaction Features
```
x₁, x₂  →  x₁, x₂, x₁², x₁·x₂, x₂²
```
- Captures non-linear relationships for linear models
- `sklearn.preprocessing.PolynomialFeatures(degree=2)`
- ⚠️ Feature count grows as O(n^d) — use with caution for high-degree

### 2. Binning / Discretization
Converts continuous values into discrete buckets:
- **Uniform binning**: equal-width bins
- **Quantile binning**: equal-frequency bins (robust to outliers)
- Use case: Age groups, income brackets, sensor readings

### 3. Mathematical Transforms
- **Log transform**: compresses right-skewed distributions  `log(x+1)`
- **Square root**: milder compression  `√x`
- **Box-Cox**: optimal power transform (scipy)
- **Standardization**: mean=0, std=1 (required for linear/neural models)

### 4. Date/Time Features
Extract from a datetime column:
```python
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month
# Cyclical encoding (so Monday wraps around to Sunday):
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
```

### 5. Domain-Specific Features
The most valuable features come from domain knowledge:
- E-commerce: `days_since_last_purchase`, `items_per_session`
- Finance: `debt_to_income_ratio`, `30_day_rolling_avg`
- Text: `word_count`, `contains_keyword`, `sentiment_score`

## ✅ When to Apply
| Situation | Technique |
|---|---|
| Linear model on non-linear data | Polynomial features |
| Continuous feature with threshold effects | Binning |
| Right-skewed distribution (income, counts) | Log transform |
| Temporal data | DateTime features + cyclical encoding |
| Domain knowledge available | Custom ratio/combination features |

## ⚠️ Common Pitfalls
- **Overfitting**: Adding too many polynomial features on small datasets
- **Collinearity**: Highly correlated new features hurt linear models
- **Data leakage**: Features that use future information (rolling averages on full dataset)
- **Feature explosion**: Degree-3 polynomial on 20 features → 1771 features!
"""

CODE_EXAMPLE = '''
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

# ── Polynomial features ──────────────────────────────────────────
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Original: {X.shape[1]} features → Polynomial: {X_poly.shape[1]} features")

# Compare Ridge regression performance
baseline = cross_val_score(Ridge(), X, y, cv=5, scoring="r2").mean()
enhanced = cross_val_score(Ridge(), X_poly, y, cv=5, scoring="r2").mean()
print(f"R² baseline: {baseline:.3f}, R² with poly features: {enhanced:.3f}")

# ── Binning ──────────────────────────────────────────────────────
binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
X_binned = binner.fit_transform(X[:, :3])  # bin first 3 features

# ── Log transform ─────────────────────────────────────────────────
X_log = np.log1p(np.abs(X))  # log(|x| + 1) for mixed-sign data

# ── Cyclical encoding for time ────────────────────────────────────
hours = np.arange(24)
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)
'''


def run_feature_engineering(technique: str, degree: int, n_bins: int):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    feature_names = list(X.columns)

    baseline_r2 = cross_val_score(Ridge(alpha=1.0), X.values, y.values, cv=5, scoring="r2").mean()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Feature Comparison (R²)", "Feature Distribution Example"])

    if technique == "Polynomial Features":
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_new = poly.fit_transform(X.values)
        new_r2 = cross_val_score(Ridge(alpha=1.0), X_new, y.values, cv=5, scoring="r2").mean()
        label = f"Degree-{degree} Poly ({X_new.shape[1]} features)"

        # Plot first feature raw vs squared
        feat_raw = X.values[:, 0]
        feat_sq  = feat_raw ** 2
        fig.add_trace(go.Scatter(x=feat_raw, y=feat_sq, mode="markers",
                                 marker=dict(color="#7e57c2", opacity=0.5, size=4),
                                 name="x vs x²"), row=1, col=2)
        fig.update_xaxes(title_text="x (raw feature)", row=1, col=2)
        fig.update_yaxes(title_text="x² (engineered)", row=1, col=2)

    elif technique == "Binning":
        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        X_new = binner.fit_transform(X.values)
        new_r2 = cross_val_score(Ridge(alpha=1.0), X_new, y.values, cv=5, scoring="r2").mean()
        label = f"Quantile Binning ({n_bins} bins)"

        feat_raw = X.values[:, 0]
        feat_bin = X_new[:, 0]
        fig.add_trace(go.Histogram(x=feat_bin, nbinsx=n_bins, name="Binned feature",
                                   marker_color="#42a5f5"), row=1, col=2)
        fig.update_xaxes(title_text="Bin index", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

    elif technique == "Log Transform":
        X_log = np.log1p(np.abs(X.values))
        new_r2 = cross_val_score(Ridge(alpha=1.0), X_log, y.values, cv=5, scoring="r2").mean()
        label = "Log(|x|+1) Transform"

        feat_raw  = X.values[:, 1]
        feat_log  = X_log[:, 1]
        fig.add_trace(go.Histogram(x=feat_raw, nbinsx=30, name="Raw",
                                   marker_color="#ef5350", opacity=0.6), row=1, col=2)
        fig.add_trace(go.Histogram(x=feat_log, nbinsx=30, name="Log-transformed",
                                   marker_color="#42a5f5", opacity=0.6), row=1, col=2)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Feature value", row=1, col=2)

    elif technique == "Cyclical Encoding (Hour)":
        hours = np.arange(24)
        h_sin = np.sin(2 * np.pi * hours / 24)
        h_cos = np.cos(2 * np.pi * hours / 24)
        new_r2 = baseline_r2  # demo only
        label  = "Cyclical Hour (sin/cos)"

        fig.add_trace(go.Scatter(x=h_sin, y=h_cos, mode="markers+text",
                                 text=[str(h) for h in hours],
                                 textposition="top center",
                                 marker=dict(color=hours, colorscale="Viridis", size=10),
                                 name="Hours as cycle"), row=1, col=2)
        fig.update_xaxes(title_text="sin(hour)", row=1, col=2)
        fig.update_yaxes(title_text="cos(hour)", row=1, col=2)
    else:
        X_new = X.values
        new_r2 = baseline_r2
        label  = "No change"

    # R² comparison bar
    fig.add_trace(go.Bar(
        x=["Baseline (raw)", label],
        y=[baseline_r2, new_r2],
        marker_color=["#bdbdbd", "#66bb6a"],
        text=[f"{baseline_r2:.3f}", f"{new_r2:.3f}"],
        textposition="outside"
    ), row=1, col=1)
    fig.update_yaxes(title_text="Cross-val R²", range=[0, 1], row=1, col=1)

    fig.update_layout(height=420, title_text=f"Feature Engineering — {technique}")

    delta = new_r2 - baseline_r2
    sign  = "+" if delta >= 0 else ""
    metrics_md = f"""
### Results (Ridge Regression on Diabetes Dataset)

| | R² Score |
|---|---|
| Baseline (raw features) | `{baseline_r2:.4f}` |
| After {technique} | `{new_r2:.4f}` |
| Improvement | `{sign}{delta:.4f}` |

> **Insight**: {technique} {'improved' if delta > 0.001 else 'did not significantly change'} the model. {'Polynomial features capture interaction effects between medical indicators.' if 'Poly' in technique else 'Feature transformations change the representational space available to the model.'}
"""
    return fig, metrics_md


def build_tab():
    gr.Markdown("# ⚙️ Module 16 — Feature Engineering\n*Level: Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nApply feature engineering techniques to the Diabetes dataset and measure their impact on Ridge Regression R².")

    with gr.Row():
        with gr.Column(scale=1):
            technique_dd = gr.Dropdown(
                label="Technique",
                choices=["Polynomial Features", "Binning", "Log Transform", "Cyclical Encoding (Hour)"],
                value="Polynomial Features"
            )
            degree_sl = gr.Slider(label="Polynomial Degree", minimum=2, maximum=3, step=1, value=2)
            bins_sl   = gr.Slider(label="Number of Bins (Binning)", minimum=3, maximum=10, step=1, value=5)
            run_btn   = gr.Button("▶ Engineer Features", variant="primary")

        with gr.Column(scale=2):
            plot_out   = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_feature_engineering,
        inputs=[technique_dd, degree_sl, bins_sl],
        outputs=[plot_out, metrics_out]
    )
