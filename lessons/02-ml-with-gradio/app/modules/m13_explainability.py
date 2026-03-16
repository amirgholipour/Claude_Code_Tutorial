"""Module 13 — Explainability
Level: Advanced"""
import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from utils.plot_utils import feature_importance_bar
from config import COLORS

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

THEORY = """## Model Explainability — Understanding What the Model Learned

**Explainability** (also called Interpretability or XAI — Explainable AI) answers the question:
*Why did the model make this prediction?*

---

### Why Explainability Matters

| Reason | Example |
|---|---|
| **Trust** | Would you trust a black-box medical diagnosis? |
| **Debugging** | Is the model using a spurious feature (e.g., hospital ID → diagnosis)? |
| **Regulatory compliance** | GDPR Article 22: right to explanation for automated decisions |
| **Fairness auditing** | Is the model discriminating based on protected attributes? |
| **Scientific discovery** | Which genes actually matter for predicting disease? |

---

### Method 1 — Built-in Feature Importance (Tree Models)

Random Forests and Gradient Boosting compute importance as:
**mean decrease in impurity (MDI)** — how much each feature reduces Gini/entropy when used for splits.

$$\\text{importance}(f) = \\frac{1}{N_{trees}} \\sum_{t} \\sum_{\\text{node } v : \\text{split on } f} p(v) \\cdot \\Delta \\text{impurity}(v)$$

- **Fast** — computed during training at no extra cost
- **Bias**: overestimates importance of high-cardinality continuous features
- Only gives global (dataset-level) importance, not per-prediction

---

### Method 2 — Permutation Importance

**Idea**: shuffle one feature column, measure how much accuracy drops. If accuracy drops a lot → the feature was important.

$$\\text{importance}(f) = \\text{score}_{\\text{original}} - \\text{score}_{\\text{feature } f \\text{ shuffled}}$$

- **Model-agnostic** — works with any model
- **Unbiased** — not affected by feature cardinality
- Computed on **test data** → reflects generalization, not just training fit
- Slow: requires re-scoring the model N times per feature × N repeats

---

### Method 3 — SHAP Values (SHapley Additive exPlanations)

SHAP comes from **cooperative game theory** (Shapley values). Each feature gets a fair share of credit for the prediction:

$$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \\cup \\{i\\}) - f(S)]$$

- **Local SHAP**: explains a single prediction — "feature X pushed this prediction up by 0.3"
- **Global SHAP**: average |SHAP| per feature — aggregated importance
- `TreeExplainer`: fast exact SHAP for tree-based models (Random Forest, XGBoost)
- The only method satisfying all 4 fairness axioms simultaneously

---

### Method 4 — Partial Dependence Plots (PDP)

**Idea**: vary one feature while marginalizing out all others, observe how predictions change.

$$\\text{PDP}(x_s) = \\mathbb{E}_{X_c}[f(x_s, X_c)] \\approx \\frac{1}{n} \\sum_{i=1}^{n} f(x_s, x_c^{(i)})$$

- Shows the **marginal effect** of one feature on the model output
- Assumes features are approximately independent (may be misleading otherwise)
- ICE plots (Individual Conditional Expectation) show per-sample lines instead of the average
"""

CODE_EXAMPLE = '''import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─── 1. Built-in Feature Importance ───────────────────────────────────────────
importances = model.feature_importances_
# Sort and plot
idx = importances.argsort()[::-1]
print("Top features:", [feature_names[i] for i in idx[:5]])

# ─── 2. Permutation Importance ────────────────────────────────────────────────
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
# perm.importances_mean: mean accuracy drop per feature
# perm.importances_std:  standard deviation across repeats

# ─── 3. SHAP Values (TreeExplainer — fast for Random Forest) ──────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:50])
# shap_values: list of arrays, one per class [n_samples, n_features]

# Global importance: mean |SHAP| per feature
global_shap = np.mean(np.abs(shap_values[0]), axis=0)

# Local explanation for one sample
print("Prediction:", model.predict([X_test[0]]))
print("SHAP contributions:", dict(zip(feature_names, shap_values[0][0])))

# ─── 4. Partial Dependence Plot ───────────────────────────────────────────────
from sklearn.inspection import partial_dependence
pdp = partial_dependence(model, X_train, features=[0], grid_resolution=50)
# pdp["average"]: model output averaged over all other features
# pdp["values"]:  feature values grid
'''


def _empty_fig(height=450):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height)
    return fig


def run_explainability(dataset_name: str, method: str, n_samples: int):
    """
    Train a Random Forest and explain its predictions using the selected method.

    Args:
        dataset_name: "iris", "wine", or "breast_cancer"
        method: "Feature Importance", "Permutation Importance", "SHAP Values",
                or "Partial Dependence"
        n_samples: Number of test samples used for SHAP / permutation (20–100)

    Returns:
        (fig, explanation_md)
    """
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(
            X, y, test_size=0.25, scale="standard", random_state=42
        )

        n_samples = min(n_samples, len(X_test))
        X_explain = X_test[:n_samples]
        y_explain = y_test[:n_samples]

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (model.predict(X_test) == y_test).mean()

        fn = feature_names  # shorthand

        # ── Feature Importance ────────────────────────────────────────────────
        if method == "Feature Importance":
            importances = model.feature_importances_
            fig = feature_importance_bar(
                np.array(fn), importances,
                title=f"Built-in Feature Importance — {dataset_name} (Random Forest)"
            )
            top_idx = np.argsort(importances)[::-1][:3]
            top_features = [f"`{fn[i]}` ({importances[i]:.3f})" for i in top_idx]
            explanation_md = f"""### Feature Importance — {dataset_name}

**Method:** Mean Decrease in Impurity (MDI) — computed during training.

| Metric | Value |
|---|---|
| Train Accuracy | `{train_acc:.3f}` |
| Test Accuracy | `{test_acc:.3f}` |
| Top Features | {', '.join(top_features)} |

**How to read this chart:** Longer bars = the feature is used more (and more effectively) for splitting across all trees.

**Caveats:**
- May overestimate importance of **high-cardinality** features (many unique values)
- Measured on **training data** → doesn't reflect test generalization directly
- Use Permutation Importance to validate these rankings on held-out data
"""
            return fig, explanation_md

        # ── Permutation Importance ────────────────────────────────────────────
        elif method == "Permutation Importance":
            perm = permutation_importance(
                model, X_explain, y_explain,
                n_repeats=10, random_state=42, n_jobs=-1
            )
            means = perm.importances_mean
            stds = perm.importances_std

            idx = np.argsort(means)
            sorted_names = [fn[i] for i in idx]
            sorted_means = means[idx]
            sorted_stds = stds[idx]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sorted_means,
                y=sorted_names,
                orientation="h",
                error_x=dict(type="data", array=sorted_stds, visible=True, color="rgba(0,0,0,0.4)"),
                marker_color=COLORS["info"],
            ))
            fig.update_layout(
                title=f"Permutation Importance — {dataset_name}<br>"
                      f"<sup>Mean accuracy drop when feature is shuffled (n_repeats=10)</sup>",
                xaxis_title="Mean Accuracy Decrease",
                template="plotly_white",
                height=max(350, len(fn) * 28),
            )

            top_idx = np.argsort(means)[::-1][:3]
            top_features = [f"`{fn[i]}` (Δ={means[i]:.3f} ± {stds[i]:.3f})" for i in top_idx]
            explanation_md = f"""### Permutation Importance — {dataset_name}

**Method:** Shuffle one feature column, measure accuracy drop (repeated 10 times).

| Metric | Value |
|---|---|
| Baseline Accuracy | `{test_acc:.3f}` |
| Samples used | `{n_samples}` |
| Top Features | {', '.join(top_features)} |

**How to read this chart:** A larger bar means shuffling that feature hurt accuracy more → the feature is more important.

**Error bars** show variability across the 10 random shuffles. Large error bars indicate unstable estimates.

**Advantage over built-in importance:** Model-agnostic and measured on **test data** → reflects true generalization importance.
Features with near-zero or negative importance can often be safely removed.
"""
            return fig, explanation_md

        # ── SHAP Values ───────────────────────────────────────────────────────
        elif method == "SHAP Values":
            if not SHAP_AVAILABLE:
                explanation_md = """### SHAP Values — Package Not Installed

To use SHAP, install it:
```bash
pip install shap
```

Then restart the app. SHAP provides the most theoretically sound feature attributions,
grounded in cooperative game theory (Shapley values).
"""
                return _empty_fig(), explanation_md

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_explain)

            # shap_vals is a list [n_classes] of arrays [n_samples, n_features]
            # For global importance: mean |SHAP| across all classes and samples
            if isinstance(shap_vals, list):
                global_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                # Use first class for beeswarm
                sv_plot = shap_vals[0]
            else:
                global_shap = np.abs(shap_vals).mean(axis=0)
                sv_plot = shap_vals

            # Beeswarm-style dot plot (jittered scatter per feature)
            idx_order = np.argsort(global_shap)  # bottom to top
            sorted_names = [fn[i] for i in idx_order]

            n_feat = len(fn)
            # Normalize feature values for color (0-1 per feature)
            X_norm = (X_explain - X_explain.min(axis=0)) / (
                X_explain.max(axis=0) - X_explain.min(axis=0) + 1e-9
            )

            fig = go.Figure()
            rng = np.random.RandomState(0)

            for plot_y, feat_idx in enumerate(idx_order):
                shap_col = sv_plot[:, feat_idx]
                feat_vals_norm = X_norm[:, feat_idx]
                jitter = rng.uniform(-0.3, 0.3, size=len(shap_col))
                # Color by feature value: blue (low) → red (high)
                colors = [
                    f"rgb({int(255 * v)}, {int(50 * (1-v))}, {int(255 * (1-v))})"
                    for v in feat_vals_norm
                ]
                fig.add_trace(go.Scatter(
                    x=shap_col,
                    y=[plot_y + j for j in jitter],
                    mode="markers",
                    marker=dict(color=colors, size=5, opacity=0.7),
                    name=fn[feat_idx],
                    showlegend=False,
                    hovertemplate=f"<b>{fn[feat_idx]}</b><br>SHAP: %{{x:.3f}}<extra></extra>",
                ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            fig.update_layout(
                title=f"SHAP Beeswarm Plot — {dataset_name}<br>"
                      f"<sup>Dot color: feature value (blue=low, red=high) | "
                      f"x-position: SHAP contribution to class 0</sup>",
                xaxis_title="SHAP Value (impact on model output)",
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(n_feat)),
                    ticktext=sorted_names,
                ),
                template="plotly_white",
                height=max(400, n_feat * 35),
            )

            top_idx = np.argsort(global_shap)[::-1][:3]
            top_features = [f"`{fn[i]}` ({global_shap[i]:.3f})" for i in top_idx]
            explanation_md = f"""### SHAP Values — {dataset_name}

**Method:** TreeExplainer (exact Shapley values for Random Forest) on {n_samples} test samples.

| Metric | Value |
|---|---|
| Test Accuracy | `{test_acc:.3f}` |
| Samples explained | `{n_samples}` |
| Top Features (mean |SHAP|) | {', '.join(top_features)} |

**How to read the beeswarm:**
- Each **dot** is one sample's contribution from one feature
- **x-axis**: SHAP value — how much that feature pushed the prediction up (positive) or down (negative)
- **Dot color**: the actual feature value — red = high, blue = low
- Features are **sorted by mean |SHAP|** (most impactful at top)

**Insight:** When red dots (high feature value) cluster on the right (positive SHAP), the feature has a **positive** effect.
When they cluster on the left, it has a **negative** effect.

SHAP is the gold standard for explainability — it satisfies **efficiency, symmetry, dummy, and additivity** axioms.
"""
            return fig, explanation_md

        # ── Partial Dependence ────────────────────────────────────────────────
        elif method == "Partial Dependence":
            from sklearn.inspection import partial_dependence

            # Pick top 2 features by built-in importance
            importances = model.feature_importances_
            top2_idx = np.argsort(importances)[::-1][:2]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f"PDP — {fn[i]}" for i in top2_idx],
            )
            colors_list = [COLORS["primary"], COLORS["success"]]

            for col_idx, feat_idx in enumerate(top2_idx):
                pdp_result = partial_dependence(
                    model, X_train, features=[feat_idx], grid_resolution=50
                )
                grid_values = pdp_result["values"][0]
                # pdp_result["average"] shape: (n_classes or 1, n_grid_points)
                avg = pdp_result["average"]
                # For multiclass, average across classes; for binary use class 1
                if avg.shape[0] > 1:
                    pdp_line = avg.mean(axis=0)
                else:
                    pdp_line = avg[0]

                fig.add_trace(
                    go.Scatter(
                        x=grid_values,
                        y=pdp_line,
                        mode="lines",
                        name=fn[feat_idx],
                        line=dict(color=colors_list[col_idx], width=2),
                    ),
                    row=1, col=col_idx + 1,
                )
                fig.update_xaxes(title_text=fn[feat_idx], row=1, col=col_idx + 1)

            fig.update_yaxes(title_text="Partial Dependence", row=1, col=1)
            fig.update_layout(
                title=f"Partial Dependence Plots — Top 2 Features ({dataset_name})",
                template="plotly_white",
                height=420,
                showlegend=False,
            )

            top2_names = [f"`{fn[i]}`" for i in top2_idx]
            explanation_md = f"""### Partial Dependence Plots — {dataset_name}

**Method:** Marginal effect of each feature — all other features are averaged out (marginalized).

| Metric | Value |
|---|---|
| Test Accuracy | `{test_acc:.3f}` |
| Features shown | {' and '.join(top2_names)} (top 2 by feature importance) |

**How to read PDPs:**
- **x-axis**: the feature value range across training data
- **y-axis**: the average model prediction as that feature varies
- An **upward slope** → higher feature value → higher predicted output
- A **flat line** → the feature has no marginal effect (after averaging)
- **Non-linear curves** reveal complex, non-linear relationships

**Assumption:** Features are approximately independent. PDP can be misleading when features are
strongly correlated (use SHAP dependence plots as a more robust alternative).
"""
            return fig, explanation_md

        else:
            return _empty_fig(), f"**Unknown method:** {method}"

    except Exception as e:
        import traceback
        return _empty_fig(), f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def build_tab():
    """Build the Gradio UI for the Explainability module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["iris", "wine", "breast_cancer"],
                    value="wine",
                    label="Dataset",
                )
                method_radio = gr.Radio(
                    choices=["Feature Importance", "Permutation Importance",
                             "SHAP Values", "Partial Dependence"],
                    value="Feature Importance",
                    label="Explainability Method",
                )
                n_samples_slider = gr.Slider(
                    minimum=20, maximum=100, step=5, value=50,
                    label="Samples for Analysis (SHAP / Permutation)",
                )
                run_btn = gr.Button("▶ Explain Model", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Explanation")
                metrics_out = gr.Markdown(label="Analysis")

        run_btn.click(
            fn=run_explainability,
            inputs=[dataset_dd, method_radio, n_samples_slider],
            outputs=[plot_out, metrics_out],
        )
