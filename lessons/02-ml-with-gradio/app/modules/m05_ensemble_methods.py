"""Module 05 — Ensemble Methods
Level: Intermediate"""
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from utils.plot_utils import confusion_matrix_heatmap, feature_importance_bar
from config import COLORS

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

THEORY = """## 🌲 Ensemble Methods

Ensemble methods combine multiple "weak learners" into a single strong model. The core insight:
**a crowd of mediocre models, when combined wisely, outperforms any single model.**

---

### Bagging (Bootstrap Aggregating)

Bagging trains many models **in parallel**, each on a different random bootstrap sample of the data.
Predictions are combined by majority vote (classification) or averaging (regression).

- **Goal:** Reduce variance — individual trees overfit, but their average does not.
- **Random Forest** = Decision Trees + Bagging + **Feature Randomness**
  - At each split, only a random subset of features is considered (`max_features`)
  - This decorrelates the trees, making the ensemble more powerful than pure bagging

---

### Boosting

Boosting trains models **sequentially**. Each new model focuses on the mistakes of the previous one.

- **Goal:** Reduce bias — iteratively correct errors to build a highly accurate model.
- **AdaBoost**: Re-weights misclassified samples so the next tree focuses on them.
- **Gradient Boosting**: Fits each new tree to the **residual errors** (gradients) of the ensemble so far.
  - `learning_rate` controls how much each tree contributes (shrinkage). Lower = more trees needed but better generalization.

---

### Bagging vs Boosting

| Property | Bagging (Random Forest) | Boosting (GB, AdaBoost) |
|----------|------------------------|------------------------|
| Training | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Overfitting risk | Low | Higher (can overfit noisy data) |
| Speed | Fast | Slower |
| Noise sensitivity | Robust | Sensitive |
| Typical accuracy | High | Often higher |

---

### Key Hyperparameters

| Parameter | Applies to | Effect |
|-----------|-----------|--------|
| `n_estimators` | All | More trees = better + slower. Diminishing returns after ~100–200. |
| `max_depth` | RF, GB | Controls tree complexity. Deeper = more variance. |
| `learning_rate` | GB, AdaBoost | Step size for each tree's contribution. Lower = needs more trees. |
| `max_features` | Random Forest | Features sampled per split. `sqrt` is default for classification. |

> **Rule of thumb:** For Gradient Boosting, use a low `learning_rate` (0.05–0.1) with a high `n_estimators`.
> Use early stopping to find the optimal number automatically.
"""

CODE_EXAMPLE = '''from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest — bagging + feature randomness
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest:      {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# Gradient Boosting — sequential residual fitting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
print(f"Gradient Boosting:  {accuracy_score(y_test, gb.predict(X_test)):.4f}")

# AdaBoost — sequential re-weighting of misclassified samples
ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada.fit(X_train, y_train)
print(f"AdaBoost:           {accuracy_score(y_test, ada.predict(X_test)):.4f}")

# Feature importances from Random Forest
for name, imp in zip(load_iris().feature_names, rf.feature_importances_):
    print(f"  {name}: {imp:.4f}")
'''


def run_ensemble(dataset_name, algorithm, n_estimators, max_depth, learning_rate, test_size):
    """Train ensemble model(s) and return a plot + metrics markdown."""
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test, _ = split_and_scale(
            X, y, test_size=test_size, scale=None
        )

        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        def make_rf():
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
            )

        def make_gb():
            return GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max(1, max_depth - 2), random_state=42
            )

        def make_ada():
            return AdaBoostClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
            )

        if algorithm == "Compare All":
            models = {
                "Random Forest": make_rf(),
                "Gradient Boosting": make_gb(),
                "AdaBoost": make_ada(),
            }
            accuracies = {}
            rf_model = None
            for name, model in models.items():
                model.fit(X_train, y_train)
                accuracies[name] = accuracy_score(y_test, model.predict(X_test))
                if name == "Random Forest":
                    rf_model = model

            # Bar chart comparing accuracies
            names_list = list(accuracies.keys())
            acc_list = list(accuracies.values())
            colors = [COLORS["primary"], COLORS["success"], COLORS["warning"]]
            fig = go.Figure(go.Bar(
                x=names_list,
                y=acc_list,
                marker_color=colors,
                text=[f"{a:.4f}" for a in acc_list],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Ensemble Accuracy Comparison — {dataset_name}",
                xaxis_title="Algorithm",
                yaxis_title="Test Accuracy",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_white",
                height=420,
            )

            best_name = max(accuracies, key=accuracies.get)
            metrics_md = f"""### Compare All — `{dataset_name}`

| Algorithm | Test Accuracy |
|-----------|--------------|
| Random Forest | **{accuracies['Random Forest']:.4f}** |
| Gradient Boosting | **{accuracies['Gradient Boosting']:.4f}** |
| AdaBoost | **{accuracies['AdaBoost']:.4f}** |

**Best model:** {best_name} ({accuracies[best_name]:.4f})

> Feature importances shown are from the **Random Forest** model.
"""
            # Append feature importance below — but gr.Plot takes one figure.
            # Return the comparison bar chart; feature importance shown in metrics text.
            if rf_model is not None:
                top_idx = np.argsort(rf_model.feature_importances_)[::-1][:8]
                imp_lines = "\n".join(
                    f"| `{feature_names[i][:25]}` | {rf_model.feature_importances_[i]:.4f} |"
                    for i in top_idx
                )
                metrics_md += f"\n### Top Feature Importances (Random Forest)\n| Feature | Importance |\n|---------|------------|\n{imp_lines}\n"

            return fig, metrics_md

        else:
            # Single algorithm
            if algorithm == "Random Forest":
                model = make_rf()
            elif algorithm == "Gradient Boosting":
                model = make_gb()
            else:
                model = make_ada()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Choose between confusion matrix or feature importance based on space.
            # Show feature importance as main plot (more informative for ensembles).
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fig = feature_importance_bar(
                    feature_names, importances,
                    title=f"{algorithm} — Feature Importances ({dataset_name})"
                )
            else:
                # Fallback: confusion matrix
                fig = confusion_matrix_heatmap(
                    cm, list(target_names),
                    title=f"{algorithm} — Confusion Matrix ({dataset_name})"
                )

            n_train, n_test = len(y_train), len(y_test)
            params_md = ""
            if algorithm == "Gradient Boosting":
                params_md = f"learning_rate = {learning_rate}"
            elif algorithm == "AdaBoost":
                params_md = f"learning_rate = {learning_rate}"

            metrics_md = f"""### {algorithm} — `{dataset_name}`

| Metric | Value |
|--------|-------|
| Test Accuracy | **{acc:.4f}** |
| Train samples | {n_train} |
| Test samples | {n_test} |
| n_estimators | {n_estimators} |
| max_depth | {max_depth} |
{"| " + params_md.replace(" = ", " | ") + " |" if params_md else ""}

### Confusion Matrix
"""
            # Append confusion matrix values
            for i, row in enumerate(cm):
                label = target_names[i] if i < len(target_names) else f"Class {i}"
                metrics_md += f"- **{label}**: {dict(zip([str(target_names[j]) for j in range(len(row))], row.tolist()))}\n"

            return fig, metrics_md

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=420)
        return empty_fig, f"**Error:** {str(e)}"


def build_tab():
    """Build the Gradio UI for the Ensemble Methods module."""
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
                    value="iris",
                    label="Dataset",
                )
                algorithm_radio = gr.Radio(
                    choices=["Random Forest", "Gradient Boosting", "AdaBoost", "Compare All"],
                    value="Random Forest",
                    label="Algorithm",
                )
                n_estimators_sl = gr.Slider(
                    minimum=10, maximum=200, value=100, step=10,
                    label="n_estimators",
                )
                max_depth_sl = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="max_depth",
                )
                learning_rate_sl = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                    label="learning_rate (Gradient Boosting / AdaBoost)",
                )
                test_size_sl = gr.Slider(
                    minimum=0.1, maximum=0.4, value=0.2, step=0.05,
                    label="Test Size",
                )
                run_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Results")
                metrics_out = gr.Markdown(label="Metrics")

        run_btn.click(
            fn=run_ensemble,
            inputs=[dataset_dd, algorithm_radio, n_estimators_sl,
                    max_depth_sl, learning_rate_sl, test_size_sl],
            outputs=[plot_out, metrics_out],
        )

    return plot_out, metrics_out
