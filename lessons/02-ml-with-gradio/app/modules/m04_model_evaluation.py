"""Module 04 — Model Evaluation
Level: Basic"""
import gradio as gr
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from utils.plot_utils import roc_curve_plot
from config import COLORS

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

THEORY = """## 📖 Model Evaluation — Knowing If Your Model Is Actually Good

Measuring performance correctly is just as important as building the model itself. Using the wrong metric can lead to **dangerously misleading conclusions**.

---

### Why Evaluation Matters: Train vs. Test Accuracy

A model that scores 99% on training data but 60% on test data has **memorized** the training set (overfit). Always evaluate on **held-out data** the model has never seen.

---

### Accuracy, Precision, Recall, F1 — Choosing the Right Metric

Suppose we're detecting a rare disease (1% of population is sick):

| Prediction → | Positive | Negative |
|-------------|----------|----------|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | $(TP+TN)/N$ | Classes are balanced |
| **Precision** | $TP/(TP+FP)$ | False positives are costly (spam filter) |
| **Recall** | $TP/(TP+FN)$ | False negatives are costly (cancer screening) |
| **F1** | $2 \\cdot \\frac{P \\cdot R}{P+R}$ | Need balance between precision and recall |

⚠️ **Imbalanced classes**: a model that predicts "healthy" for everyone achieves 99% accuracy but 0% recall for the sick class. Always check per-class metrics.

---

### ROC-AUC — Threshold-Independent Evaluation

The **ROC curve** plots True Positive Rate vs. False Positive Rate as the decision threshold varies from 0 → 1.

- **AUC = 1.0**: perfect classifier
- **AUC = 0.5**: random guessing (diagonal line)
- **AUC < 0.5**: worse than random (but flip predictions to fix!)

Advantages:
- Not affected by class imbalance
- Evaluates the model's **ranking ability**, not just a single threshold
- Good for comparing models regardless of operating point

⚠️ ROC-AUC only applies directly to **binary classification**. For multiclass, use one-vs-rest.

---

### Cross-Validation — A More Reliable Estimate

A single train/test split can give **lucky or unlucky** results depending on which samples end up where. **K-fold cross-validation** fixes this:

1. Split data into **k equal folds**
2. Train on k-1 folds, evaluate on the remaining fold
3. Repeat k times, each fold used as test set exactly once
4. Report **mean ± std** of the k scores

**Stratified k-fold** preserves class proportions in each fold — always use this for classification.

| k | Bias | Variance | Cost |
|---|------|----------|------|
| 5 | Moderate | Moderate | 5× training |
| 10 | Low | Low | 10× training |
| N (LOOCV) | Very low | High | N× training |

---

### Overfitting vs. Underfitting — Bias-Variance Tradeoff

| | Train Error | Test Error | Fix |
|-|------------|-----------|-----|
| **Underfitting** (high bias) | High | High | More complexity, more features |
| **Overfitting** (high variance) | Low | High | Regularization, more data, pruning |
| **Good fit** | Low | Low | ✓ |

The goal is the **sweet spot** — low bias AND low variance.
"""

CODE_EXAMPLE = '''from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} \u00b1 {cv_scores.std():.4f}")
print(f"Per-fold: {cv_scores.round(4)}")

# ROC-AUC (binary classification only)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
print(f"AUC: {auc:.4f}")

# Stratified K-Fold (recommended for classification)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
print(f"Stratified CV F1: {scores.mean():.4f} \u00b1 {scores.std():.4f}")
'''


def _build_model(algorithm: str):
    """Construct the classifier based on user selection."""
    if algorithm == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _cv_bar_chart(fold_scores: np.ndarray, dataset_name: str, algorithm: str, cv_folds: int):
    """Build a bar chart showing per-fold CV scores with a mean line."""
    fold_labels = [f"Fold {i+1}" for i in range(len(fold_scores))]
    mean_score = fold_scores.mean()
    std_score = fold_scores.std()

    # Color bars: above mean → success green, below → warning amber
    bar_colors = [
        COLORS["success"] if s >= mean_score else COLORS["warning"]
        for s in fold_scores
    ]

    fig = go.Figure()

    # Per-fold bars
    fig.add_trace(go.Bar(
        x=fold_labels,
        y=fold_scores,
        marker_color=bar_colors,
        text=[f"{s:.4f}" for s in fold_scores],
        textposition="outside",
        name="Fold Score",
    ))

    # Mean line
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(fold_scores) - 0.5,
        y0=mean_score, y1=mean_score,
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
    )
    # Invisible trace for legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        name=f"Mean = {mean_score:.4f}",
    ))

    y_min = max(0, fold_scores.min() - 0.05)
    y_max = min(1.05, fold_scores.max() + 0.08)

    fig.update_layout(
        title=f"{cv_folds}-Fold CV — {algorithm} on {dataset_name}",
        xaxis_title="Fold",
        yaxis_title="Accuracy",
        yaxis=dict(range=[y_min, y_max]),
        template="plotly_white",
        height=420,
        showlegend=True,
    )
    return fig, mean_score, std_score


def run_evaluation(
    dataset_name: str,
    algorithm: str,
    eval_type: str,
    cv_folds: int,
):
    """
    Run model evaluation and return a figure + metrics markdown.

    Args:
        dataset_name: One of iris, breast_cancer, wine
        algorithm: "Logistic Regression", "Decision Tree", "Random Forest"
        eval_type: "Cross-Validation" or "ROC Curve"
        cv_folds: Number of CV folds (3–10)

    Returns:
        (fig, metrics_markdown)
    """
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        n_classes = len(np.unique(y))

        # --- ROC Curve branch ---
        if eval_type == "ROC Curve":
            if n_classes != 2:
                # Multiclass: show confusion matrix instead of empty space
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)

                model = _build_model(algorithm)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                class_names = [str(t) for t in target_names]

                fig = go.Figure(go.Heatmap(
                    z=cm[::-1], x=class_names, y=class_names[::-1],
                    text=cm[::-1], texttemplate="%{text}",
                    colorscale="Blues", showscale=False
                ))
                fig.update_layout(
                    title=f"Confusion Matrix — {algorithm} on {dataset_name} (ROC needs binary data)",
                    xaxis_title="Predicted", yaxis_title="Actual",
                    template="plotly_white", height=400
                )
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_test, y_pred)
                msg = (
                    f"### Confusion Matrix — {algorithm} on `{dataset_name}`\n\n"
                    f"ROC curve requires binary classification, but `{dataset_name}` has **{n_classes} classes**.\n"
                    f"Showing confusion matrix instead.\n\n"
                    f"| Metric | Value |\n|--------|-------|\n"
                    f"| **Accuracy** | `{acc:.4f}` |\n"
                    f"| **Test size** | `{len(X_test)}` samples |\n\n"
                    "> Select **breast_cancer** for the ROC curve (binary classification)."
                )
                return fig, msg

            # Binary ROC
            X_train, X_test, y_train, y_test, _ = split_and_scale(
                X, y, test_size=0.2, scale="standard"
            )
            model = _build_model(algorithm)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig = roc_curve_plot(fpr, tpr, auc,
                                 title=f"ROC Curve — {algorithm} on breast_cancer")

            metrics_md = f"""### ROC-AUC Results: {algorithm} on `{dataset_name}`
| Metric | Value |
|--------|-------|
| **AUC Score** | `{auc:.4f}` |
| **Test size** | `{len(X_test)}` samples |
| **Train size** | `{len(X_train)}` samples |

| AUC Range | Interpretation |
|-----------|---------------|
| 0.9 – 1.0 | Outstanding |
| 0.8 – 0.9 | Excellent |
| 0.7 – 0.8 | Acceptable |
| 0.6 – 0.7 | Poor |
| 0.5 – 0.6 | Fail (near random) |

> AUC = **{auc:.4f}** means the model correctly ranks a random positive above a random negative {auc*100:.1f}% of the time.
"""
            return fig, metrics_md

        # --- Cross-Validation branch ---
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            model = _build_model(algorithm)
            fold_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

            fig, mean_score, std_score = _cv_bar_chart(
                fold_scores, dataset_name, algorithm, cv_folds
            )

            # Also get F1 scores for richer reporting
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")

            metrics_md = f"""### {cv_folds}-Fold Cross-Validation: {algorithm} on `{dataset_name}`
| Metric | Mean | Std Dev |
|--------|------|---------|
| **Accuracy** | `{mean_score:.4f}` | `±{std_score:.4f}` |
| **F1 (macro)** | `{f1_scores.mean():.4f}` | `±{f1_scores.std():.4f}` |

**Per-fold accuracy:**
{", ".join(f"`{s:.4f}`" for s in fold_scores)}

| Dataset | Samples | Features | Classes |
|---------|---------|---------|---------|
| `{dataset_name}` | {X.shape[0]} | {X.shape[1]} | {n_classes} |

> A **small std dev** means the model is stable across different data splits.
> A large std dev suggests high variance — try more regularization or more data.
"""
            return fig, metrics_md

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=400)
        return empty_fig, f"**Error:** {str(e)}"


def build_tab():
    """Build the Gradio UI for the Model Evaluation module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["iris", "breast_cancer", "wine"],
                    value="breast_cancer",
                    label="Dataset",
                    info="Use breast_cancer for ROC Curve (binary only)",
                )
                algorithm_radio = gr.Radio(
                    choices=["Logistic Regression", "Decision Tree", "Random Forest"],
                    value="Logistic Regression",
                    label="Algorithm",
                )
                eval_type_radio = gr.Radio(
                    choices=["Cross-Validation", "ROC Curve"],
                    value="Cross-Validation",
                    label="Evaluation Type",
                )
                cv_folds_slider = gr.Slider(
                    minimum=3, maximum=10, step=1, value=5,
                    label="CV Folds",
                    info="Number of folds for cross-validation",
                )
                run_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Evaluation Plot")
                metrics_out = gr.Markdown(label="Metrics")

        run_btn.click(
            fn=run_evaluation,
            inputs=[dataset_dd, algorithm_radio, eval_type_radio, cv_folds_slider],
            outputs=[plot_out, metrics_out],
        )

    return plot_out, metrics_out
