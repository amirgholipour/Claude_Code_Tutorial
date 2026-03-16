"""Module 02 — Regression
Level: Basic"""
import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes, fetch_california_housing

THEORY = """## 📖 Regression — Predicting Continuous Values

Regression predicts a **continuous numeric output** (e.g., house prices, disease progression, temperatures).

### Linear Regression
Fit a straight line (or hyperplane) through the data:

$$y = w_1 x_1 + w_2 x_2 + \\ldots + w_n x_n + b$$

Training minimises **Mean Squared Error (MSE)**:
$$MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$

### Polynomial Regression
Add polynomial feature interactions ($x^2, x^3, x_1 x_2, \\ldots$) so a linear model can fit curves:

```
PolynomialFeatures(degree=d) → LinearRegression()
```

⚠️ **Feature explosion warning**: with `n` features at degree `d`, the feature count becomes C(n+d, d).
- 1 feature, degree=3 → **4 features** ✅ (safe, clearly shows curve fitting)
- 10 features, degree=3 → **286 features** ❌ (severe overfitting risk on small datasets)

### Regularization — Controlling Complexity

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge (L2)** | λ‖w‖² | Shrinks all weights; keeps all features |
| **Lasso (L1)** | λ‖w‖₁ | Forces some weights to exactly zero (feature selection) |
| **ElasticNet** | L1 + L2 mix | Best when features are correlated |

`alpha → 0`: behaves like plain linear regression
`alpha → ∞`: all weights shrink to zero (underfit)

### Key Metrics

| Metric | Interpretation |
|--------|---------------|
| **R²** | 1.0 = perfect; 0.0 = predicts mean; **negative = worse than mean** |
| **RMSE** | Same units as target; penalises large errors heavily |
| **MAE** | Robust to outliers; easier to interpret |
"""

CODE_EXAMPLE = '''from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

# ── 1D polynomial curve fitting ──────────────────────────────
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = 0.5*X[:,0]**3 - 2*X[:,0] + np.random.normal(0, 0.5, 200)

for degree in [1, 2, 3, 5]:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("lr",   Ridge(alpha=0.1))
    ])
    model.fit(X[:160], y[:160])
    r2 = model.score(X[160:], y[160:])
    print(f"degree={degree}  R²={r2:.4f}")

# ── Multi-feature regularization ─────────────────────────────
from sklearn.datasets import fetch_california_housing
X_ca, y_ca = fetch_california_housing(return_X_y=True)

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = Ridge(alpha=alpha).fit(X_ca[:15000], y_ca[:15000])
    r2 = m.score(X_ca[15000:], y_ca[15000:])
    print(f"Ridge alpha={alpha:<6}  R²={r2:.4f}")
'''


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_data(dataset_name: str):
    """Return (X, y, feature_names, description)."""
    if dataset_name == "Synthetic 1D":
        rng = np.random.default_rng(42)
        X = np.linspace(-3, 3, 300)
        y = 0.5 * X**3 - 2 * X + rng.normal(0, 0.6, 300)
        return X.reshape(-1, 1), y, ["x"], "300 samples — cubic signal + noise"

    elif dataset_name == "Diabetes":
        data = load_diabetes()
        return data.data, data.target, list(data.feature_names), "442 samples, 10 features → disease progression"

    elif dataset_name == "California Housing":
        data = fetch_california_housing()
        # Use a subset for speed
        X, y = data.data[:8000], data.target[:8000]
        return X, y, list(data.feature_names), "8,000 samples, 8 features → median house value ($100k)"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Model builder ─────────────────────────────────────────────────────────────

def _build_model(algorithm: str, degree: int, alpha: float):
    steps_poly = [("poly", PolynomialFeatures(degree=degree, include_bias=False))]

    if algorithm == "Linear":
        return Pipeline(steps_poly + [("lr", LinearRegression())])
    elif algorithm == "Ridge":
        return Pipeline(steps_poly + [("ridge", Ridge(alpha=alpha))])
    elif algorithm == "Lasso":
        return Pipeline(steps_poly + [("lasso", Lasso(alpha=alpha, max_iter=10000))])
    elif algorithm == "ElasticNet":
        return Pipeline(steps_poly + [("enet", ElasticNet(alpha=alpha, max_iter=10000))])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_1d_fit(X_all, y_all, model, X_train, X_test, y_test, y_pred_test, title):
    """For 1D data: show raw data + fitted curve + residuals side by side."""
    X_sorted = np.linspace(X_all.min(), X_all.max(), 400).reshape(-1, 1)
    y_curve = model.predict(X_sorted)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Data + Fitted Curve", "Residuals Distribution"],
        horizontal_spacing=0.12
    )

    # Left: scatter + curve
    fig.add_trace(go.Scatter(
        x=X_all[:, 0], y=y_all, mode="markers", name="Data",
        marker=dict(color="#90caf9", size=5, opacity=0.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=X_sorted[:, 0], y=y_curve, mode="lines", name="Fit",
        line=dict(color="#ef5350", width=2)
    ), row=1, col=1)

    # Right: residuals histogram
    residuals = y_test - y_pred_test
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=30, name="Residuals",
        marker_color="#66bb6a", opacity=0.75, showlegend=False
    ), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Residual (actual − predicted)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(title_text=title, height=420, template="plotly_white")
    return fig


def _plot_multifeature(y_test, y_pred_test, title):
    """Two-panel: sorted prediction comparison + residuals histogram."""
    residuals = y_test - y_pred_test

    # Sort test samples by actual value for clear visual comparison
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]
    pred_sorted = y_pred_test[sort_idx]
    sample_ids = np.arange(len(y_test))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Prediction Comparison (sorted by actual)",
                        "Residuals Distribution"],
        horizontal_spacing=0.12
    )

    # Left: sorted comparison — actual as line, predicted as scatter
    fig.add_trace(go.Scatter(
        x=sample_ids, y=y_sorted, mode="lines", name="Actual",
        line=dict(color="#ef5350", width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sample_ids, y=pred_sorted, mode="markers", name="Predicted",
        marker=dict(color="#42a5f5", size=3, opacity=0.5)
    ), row=1, col=1)
    fig.update_xaxes(title_text="Test Samples (sorted)", row=1, col=1)
    fig.update_yaxes(title_text="Target Value", row=1, col=1)

    # Right: residuals histogram
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, name="Residuals",
        marker_color="#66bb6a", opacity=0.75, showlegend=False
    ), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Residual (actual − predicted)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(title_text=title, height=420, template="plotly_white")
    return fig


# ── Main demo function ────────────────────────────────────────────────────────

def run_regression(dataset_name: str, algorithm: str, degree: int,
                   alpha: float, test_size: float):
    try:
        X, y, feature_names, data_desc = _load_data(dataset_name)
        n_features = X.shape[1]
        n_poly_features = math.comb(n_features + degree, degree) - 1

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        model = _build_model(algorithm, degree, alpha)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        r2       = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae      = mean_absolute_error(y_test, y_pred_test)
        gap      = r2_train - r2

        # Title
        title_parts = [f"{dataset_name} — {algorithm}"]
        title_parts.append(f"degree={degree}")
        if algorithm in ("Ridge", "Lasso", "ElasticNet"):
            title_parts.append(f"α={alpha}")
        title = " | ".join(title_parts)

        # Build figure
        if dataset_name == "Synthetic 1D":
            fig = _plot_1d_fit(X, y, model, X_train, X_test, y_test, y_pred_test, title)
        else:
            fig = _plot_multifeature(y_test, y_pred_test, title)

        # Warnings
        warnings = []
        if gap > 0.15:
            warnings.append(f"⚠️ **Overfitting** — train R² is {gap:.3f} higher than test R².")
        if n_features > 1 and degree >= 3:
            warnings.append(
                f"⚠️ **Feature explosion** — {n_features} features at degree={degree} "
                f"generates **{n_poly_features} polynomial features**. "
                f"Use Ridge/Lasso regularization to prevent overfitting."
            )
        if r2 < 0:
            warnings.append(
                f"❌ **R²={r2:.3f}** — model performs **worse than predicting the mean**. "
                f"Try lower degree or higher alpha."
            )

        warn_block = "\n\n".join(warnings) if warnings else "✅ No major issues detected."

        metrics_md = f"""### Results: {title}

| Metric | Train | Test |
|--------|-------|------|
| **R²** | `{r2_train:.4f}` | `{r2:.4f}` |
| **RMSE** | — | `{rmse:.4f}` |
| **MAE** | — | `{mae:.4f}` |

**Dataset:** {data_desc}
**Polynomial features generated:** {n_poly_features} (from {n_features} input features)
**Train / Test samples:** {len(X_train)} / {len(X_test)}

{warn_block}

> **Reading the plots:**
> Left — blue dots (predicted) hugging the red line (actual) = good model. Gaps = prediction errors.
> Right — residuals centred at 0 with small spread = well-calibrated model.
"""
        return fig, metrics_md

    except Exception as e:
        import traceback
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=400)
        return empty_fig, f"**Error:** {traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown("# 📈 Module 02 — Regression\n*Level: Basic*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("""---
## 🎮 Interactive Demo

**Tip:** Start with **Synthetic 1D** to clearly see curve fitting. Move to **Diabetes** or **California Housing** for real-world multi-feature regression.""")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Synthetic 1D", "Diabetes", "California Housing"],
                value="Synthetic 1D"
            )
            algorithm_dd = gr.Dropdown(
                label="Algorithm",
                choices=["Linear", "Ridge", "Lasso", "ElasticNet"],
                value="Ridge"
            )
            degree_sl = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Polynomial Degree",
                info="For multi-feature datasets, keep ≤ 2 to avoid feature explosion"
            )
            alpha_sl = gr.Slider(
                minimum=0.001, maximum=50.0, step=0.1, value=1.0,
                label="Alpha (Regularization Strength)",
                info="Ridge/Lasso/ElasticNet only — higher = stronger regularization"
            )
            test_sl = gr.Slider(
                minimum=0.1, maximum=0.4, step=0.05, value=0.2,
                label="Test Size"
            )
            run_btn = gr.Button("▶ Run", variant="primary")

        with gr.Column(scale=3):
            plot_out    = gr.Plot(label="Results")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_regression,
        inputs=[dataset_dd, algorithm_dd, degree_sl, alpha_sl, test_sl],
        outputs=[plot_out, metrics_out]
    )
