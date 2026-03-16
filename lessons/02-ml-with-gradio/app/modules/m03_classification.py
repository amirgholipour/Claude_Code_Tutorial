"""Module 03 — Classification
Level: Basic"""
import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

THEORY = """## 📖 Classification — Predicting Discrete Labels

Classification assigns inputs to one of **two or more predefined categories**. Unlike regression, the output is a discrete class label, not a continuous number.

Examples: spam/not-spam, disease/healthy, digit 0–9, wine variety.

---

### Logistic Regression
Despite the name, it's a **classification** algorithm. It applies the **sigmoid function** to produce a probability:

$$P(y=1|x) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$

- The **decision boundary** is a hyperplane (straight line in 2D)
- Multiclass: uses softmax (one-vs-rest or multinomial)
- **Strength**: fast, interpretable, works well on linearly separable data
- **Weakness**: fails on non-linear boundaries without feature engineering

---

### k-Nearest Neighbors (k-NN)
Classify by majority vote among the **k closest training examples**.

- Distance: usually Euclidean $d(x, x') = \\sqrt{\\sum_i (x_i - x'_i)^2}$
- **k=1**: memorises data (high variance); **large k**: smoother but may underfit
- **Key**: features must be scaled — distance is sensitive to scale
- **Strength**: naturally handles non-linear boundaries, no training phase
- **Weakness**: slow at prediction ($O(n)$ per query)

---

### Naive Bayes
Applies **Bayes' theorem** with feature independence assumption:

$$P(y|x_1, \\ldots, x_n) \\propto P(y) \\prod_{i=1}^{n} P(x_i|y)$$

- **GaussianNB**: assumes Gaussian features — good for continuous data
- **Strength**: extremely fast, works well with few samples
- **Weakness**: independence assumption rarely holds; boundaries are always quadratic

---

### Decision Tree
Recursively splits features to minimise impurity:

- **Gini**: $G = 1 - \\sum_k p_k^2$
- **Entropy**: $H = -\\sum_k p_k \\log_2 p_k$
- `max_depth` controls overfitting — **most important parameter**
- **Strength**: interpretable, non-linear boundaries, axis-aligned splits
- **Weakness**: high variance, unstable, creates rectangular regions
"""

CODE_EXAMPLE = '''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "k-NN (k=5)":         KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
'''


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_data(name: str):
    loaders = {
        "Iris":          load_iris,
        "Wine":          load_wine,
        "Breast Cancer": load_breast_cancer,
        "Digits":        load_digits,
    }
    data = loaders[name]()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


# ── Classifiers ───────────────────────────────────────────────────────────────

def _build_clf(algorithm: str, max_depth: int, n_neighbors: int):
    if algorithm == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    elif algorithm == "k-NN":
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    raise ValueError(f"Unknown: {algorithm}")


# ── Decision boundary plot ────────────────────────────────────────────────────

BOUNDARY_COLORS = px.colors.qualitative.Set2
BOUNDARY_BG = ["rgba(102,194,165,0.25)", "rgba(252,141,98,0.25)",
               "rgba(141,160,203,0.25)", "rgba(231,138,195,0.25)",
               "rgba(166,216,84,0.25)", "rgba(255,217,47,0.25)",
               "rgba(229,196,148,0.25)", "rgba(179,179,179,0.25)",
               "rgba(188,128,189,0.25)", "rgba(204,235,197,0.25)"]

def _decision_boundary_fig(X_2d, y, model, class_names, title, y_pred=None):
    """Create a decision boundary contour with data points overlay."""
    h = 0.3  # mesh step
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid).reshape(xx.shape)
    classes = np.unique(y)
    n_cls = len(classes)

    fig = go.Figure()

    # Decision regions as contour
    fig.add_trace(go.Heatmap(
        z=Z, x0=x_min, dx=h, y0=y_min, dy=h,
        colorscale=[[i / max(n_cls - 1, 1), BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]]
                    for i in range(n_cls)],
        opacity=0.25, showscale=False, hoverinfo="skip"
    ))

    # Data points by class
    for i, cls in enumerate(classes):
        mask = y == cls
        label = str(class_names[cls]) if cls < len(class_names) else f"Class {cls}"
        # Determine correct/incorrect
        if y_pred is not None:
            correct = y_pred[mask] == y[mask]
            # Correct points
            fig.add_trace(go.Scatter(
                x=X_2d[mask][correct, 0], y=X_2d[mask][correct, 1],
                mode="markers", name=label,
                marker=dict(color=BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)],
                            size=7, opacity=0.8,
                            line=dict(width=1, color="white")),
                legendgroup=label
            ))
            # Misclassified points
            if (~correct).any():
                fig.add_trace(go.Scatter(
                    x=X_2d[mask][~correct, 0], y=X_2d[mask][~correct, 1],
                    mode="markers", name=f"{label} (wrong)",
                    marker=dict(color=BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)],
                                size=10, opacity=1.0, symbol="x",
                                line=dict(width=2, color="red")),
                    legendgroup=label
                ))
        else:
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers", name=label,
                marker=dict(color=BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)],
                            size=7, opacity=0.8,
                            line=dict(width=1, color="white"))
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Feature 1 (or PC1)",
        yaxis_title="Feature 2 (or PC2)",
        template="plotly_white", height=450,
        legend=dict(x=1.02, y=1)
    )
    return fig


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_classification(dataset_name, algorithm, max_depth, n_neighbors, test_size):
    try:
        X, y, feat_names, target_names = _load_data(dataset_name)
        class_names = [str(t) for t in target_names]

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train classifier on full features
        clf = _build_clf(algorithm, max_depth, n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0)
        per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

        # ── Project to 2D for decision boundary ──
        if X_scaled.shape[1] <= 2:
            X_2d = X_scaled[:, :2]
        else:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_scaled)

        # Split 2D data same way
        X_2d_train, X_2d_test, _, _ = train_test_split(
            X_2d, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train a 2D classifier for boundary visualization
        clf_2d = _build_clf(algorithm, max_depth, n_neighbors)
        clf_2d.fit(X_2d_train, y_train)
        y_pred_2d = clf_2d.predict(X_2d_test)

        # ── Build 2-panel figure ──
        # Left: decision boundary, Right: confusion matrix
        boundary_fig = _decision_boundary_fig(
            X_2d_test, y_test, clf_2d, class_names,
            title=f"{algorithm} — Decision Boundary",
            y_pred=y_pred_2d
        )

        cm = confusion_matrix(y_test, y_pred)
        cm_fig = go.Figure(go.Heatmap(
            z=cm[::-1], x=class_names, y=class_names[::-1],
            text=cm[::-1], texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        cm_fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="Actual",
            template="plotly_white", height=450
        )

        # Per-class table rows
        per_class_rows = ""
        for i, cname in enumerate(class_names):
            if i < len(per_class_prec):
                per_class_rows += f"| {cname} | `{per_class_prec[i]:.3f}` | `{per_class_rec[i]:.3f}` | `{per_class_f1[i]:.3f}` |\n"

        # Dimension note
        dim_note = ""
        if X_scaled.shape[1] > 2:
            dim_note = f"\n> Decision boundary shown on 2D PCA projection ({X_scaled.shape[1]} → 2 features). Metrics use all features."

        param_note = ""
        if algorithm == "Decision Tree":
            param_note = f"\n| **Max Depth** | `{max_depth}` | | |"
        elif algorithm == "k-NN":
            param_note = f"\n| **k (neighbors)** | `{n_neighbors}` | | |"

        metrics_md = f"""### Results: {algorithm} on {dataset_name}

| Metric | Value | | |
|--------|-------|-|-|
| **Accuracy (test)** | `{acc:.4f}` | | |
| **Accuracy (train)** | `{train_acc:.4f}` | | |
| **F1 (macro)** | `{f1:.4f}` | | |{param_note}

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
{per_class_rows}
> Circles = correct predictions. **X markers = misclassifications.**
> Colored regions show where each class would be predicted.{dim_note}
"""
        return boundary_fig, cm_fig, metrics_md

    except Exception as e:
        import traceback
        empty = go.Figure()
        empty.update_layout(template="plotly_white", height=400)
        return empty, empty, f"**Error:** {traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown("# 🎯 Module 03 — Classification\n*Level: Basic*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("""---
## 🎮 Interactive Demo

**Watch the decision boundary change** as you switch algorithms. Logistic Regression draws straight lines; k-NN creates irregular regions; Decision Tree makes rectangular splits.""")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Iris", "Wine", "Breast Cancer", "Digits"],
                value="Iris"
            )
            algorithm_dd = gr.Dropdown(
                label="Algorithm",
                choices=["Logistic Regression", "k-NN", "Naive Bayes", "Decision Tree"],
                value="Logistic Regression"
            )
            depth_sl = gr.Slider(
                minimum=1, maximum=20, step=1, value=5,
                label="Max Depth (Decision Tree only)"
            )
            knn_sl = gr.Slider(
                minimum=1, maximum=20, step=1, value=5,
                label="k Neighbors (k-NN only)"
            )
            test_sl = gr.Slider(
                minimum=0.1, maximum=0.4, step=0.05, value=0.2,
                label="Test Size"
            )
            run_btn = gr.Button("▶ Run", variant="primary")

        with gr.Column(scale=3):
            with gr.Row():
                boundary_out = gr.Plot(label="Decision Boundary")
                cm_out = gr.Plot(label="Confusion Matrix")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_classification,
        inputs=[dataset_dd, algorithm_dd, depth_sl, knn_sl, test_sl],
        outputs=[boundary_out, cm_out, metrics_out]
    )
