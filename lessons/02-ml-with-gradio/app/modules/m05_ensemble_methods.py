"""Module 05 — Ensemble Methods
Level: Intermediate"""
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

THEORY = """## 🌲 Ensemble Methods

Ensemble methods combine multiple "weak learners" into a single strong model.
**A crowd of mediocre models, combined wisely, outperforms any single model.**

### Bagging (Bootstrap Aggregating)
Train many models **in parallel** on random bootstrap samples. Combine by majority vote.
- **Goal:** Reduce variance — individual trees overfit, but their average does not.
- **Random Forest** = Decision Trees + Bagging + Feature Randomness
  - Each split considers only a random feature subset → decorrelates trees

### Boosting
Train models **sequentially**. Each new model focuses on previous mistakes.
- **Goal:** Reduce bias — iteratively correct errors.
- **AdaBoost**: Re-weights misclassified samples for the next tree
- **Gradient Boosting**: Fits each new tree to the **residual errors** (gradients)
  - `learning_rate` controls contribution of each tree (shrinkage)

### Bagging vs Boosting

| Property | Bagging (RF) | Boosting (GB, AdaBoost) |
|----------|-------------|------------------------|
| Training | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Overfitting | Low risk | Higher risk |
| Speed | Fast | Slower |
| Typical accuracy | High | Often higher |

### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More trees = better + slower. Diminishing returns after ~100–200. |
| `max_depth` | Controls tree complexity. Deeper = more variance. |
| `learning_rate` | Step size per tree (boosting only). Lower = needs more trees. |
"""

CODE_EXAMPLE = '''from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Random Forest — bagging + feature randomness
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
print(f"RF: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# Gradient Boosting — sequential residual fitting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
print(f"GB: {accuracy_score(y_test, gb.predict(X_test)):.4f}")

# Boosting learning curve (error vs rounds)
from sklearn.metrics import log_loss
staged_scores = []
for i, y_pred in enumerate(gb.staged_predict(X_test)):
    staged_scores.append(accuracy_score(y_test, y_pred))
# staged_scores[i] = accuracy using first (i+1) trees
'''


def _load(name):
    loaders = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}
    d = loaders[name]()
    return d.data, d.target, list(d.feature_names), [str(t) for t in d.target_names]


def run_ensemble(dataset_name, algorithm, n_estimators, max_depth, learning_rate, test_size):
    try:
        X, y, feat_names, class_names = _load(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        n_est = int(n_estimators)
        md = int(max_depth)

        def make_rf():
            return RandomForestClassifier(n_estimators=n_est, max_depth=md, random_state=42)

        def make_gb():
            return GradientBoostingClassifier(
                n_estimators=n_est, learning_rate=learning_rate,
                max_depth=max(1, md - 2), random_state=42
            )

        def make_ada():
            return AdaBoostClassifier(n_estimators=n_est, learning_rate=learning_rate, random_state=42)

        if algorithm == "Compare All":
            return _compare_all(make_rf, make_gb, make_ada, X_train, X_test,
                                y_train, y_test, feat_names, class_names, dataset_name, n_est)
        else:
            makers = {"Random Forest": make_rf, "Gradient Boosting": make_gb, "AdaBoost": make_ada}
            model = makers[algorithm]()
            return _single_model(model, algorithm, X_train, X_test, y_train, y_test,
                                 feat_names, class_names, dataset_name)

    except Exception as e:
        import traceback
        empty = go.Figure().update_layout(template="plotly_white", height=420)
        return empty, empty, f"**Error:** {traceback.format_exc()}"


def _compare_all(make_rf, make_gb, make_ada, X_train, X_test, y_train, y_test,
                 feat_names, class_names, dataset_name, n_est):
    """Compare all 3 methods: accuracy bars + boosting learning curve."""
    models = {
        "Random Forest": make_rf(),
        "Gradient Boosting": make_gb(),
        "AdaBoost": make_ada(),
    }
    accs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accs[name] = accuracy_score(y_test, model.predict(X_test))

    # Boosting learning curve (staged_predict from GBT)
    gb_model = models["Gradient Boosting"]
    staged_accs = [accuracy_score(y_test, p) for p in gb_model.staged_predict(X_test)]

    # Figure 1: Accuracy comparison + boosting curve
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Test Accuracy Comparison", "Gradient Boosting: Accuracy vs Rounds"],
        column_widths=[0.4, 0.6]
    )

    colors = {"Random Forest": "#42a5f5", "Gradient Boosting": "#66bb6a", "AdaBoost": "#ff9800"}
    names = list(accs.keys())
    fig1.add_trace(go.Bar(
        x=names, y=[accs[n] for n in names],
        marker_color=[colors[n] for n in names],
        text=[f"{accs[n]:.4f}" for n in names], textposition="outside",
        showlegend=False
    ), row=1, col=1)
    fig1.update_yaxes(range=[0, 1.05], title_text="Accuracy", row=1, col=1)

    rounds = list(range(1, len(staged_accs) + 1))
    fig1.add_trace(go.Scatter(
        x=rounds, y=staged_accs, mode="lines",
        name="GB Accuracy", line=dict(color="#66bb6a", width=2), showlegend=False
    ), row=1, col=2)
    fig1.update_xaxes(title_text="Number of Trees", row=1, col=2)
    fig1.update_yaxes(title_text="Test Accuracy", row=1, col=2)

    fig1.update_layout(title=f"Ensemble Comparison — {dataset_name}",
                       template="plotly_white", height=420)

    # Figure 2: Feature importance from RF
    rf = models["Random Forest"]
    imp = rf.feature_importances_
    order = np.argsort(imp)
    fig2 = go.Figure(go.Bar(
        x=imp[order], y=[feat_names[i][:20] for i in order],
        orientation="h", marker_color="#42a5f5"
    ))
    fig2.update_layout(title="Random Forest — Feature Importances",
                       template="plotly_white", height=max(300, len(feat_names) * 22))

    best = max(accs, key=accs.get)
    md_text = f"""### Compare All — {dataset_name}

| Algorithm | Test Accuracy |
|-----------|--------------|
| Random Forest | `{accs['Random Forest']:.4f}` |
| Gradient Boosting | `{accs['Gradient Boosting']:.4f}` |
| AdaBoost | `{accs['AdaBoost']:.4f}` |

**Best:** {best} ({accs[best]:.4f})

> **Boosting curve** (right panel) shows how Gradient Boosting accuracy improves as trees are added sequentially.
> Each tree corrects the errors of the previous ones — this is the core idea of boosting.
> The curve typically rises quickly then plateaus; adding too many trees can lead to overfitting.
"""
    return fig1, fig2, md_text


def _single_model(model, algorithm, X_train, X_test, y_train, y_test,
                   feat_names, class_names, dataset_name):
    """Train single model: confusion matrix + feature importance / boosting curve."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Figure 1: Confusion matrix heatmap + feature importance (or boosting curve)
    has_staged = hasattr(model, 'staged_predict')
    has_importance = hasattr(model, 'feature_importances_')

    if has_staged:
        # Boosting: CM + learning curve
        staged_accs = [accuracy_score(y_test, p) for p in model.staged_predict(X_test)]
        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Confusion Matrix", f"{algorithm}: Accuracy vs Rounds"],
            column_widths=[0.45, 0.55]
        )
        # CM heatmap
        fig1.add_trace(go.Heatmap(
            z=cm[::-1], x=class_names, y=class_names[::-1],
            text=cm[::-1], texttemplate="%{text}",
            colorscale="Blues", showscale=False
        ), row=1, col=1)
        fig1.update_xaxes(title_text="Predicted", row=1, col=1)
        fig1.update_yaxes(title_text="Actual", row=1, col=1)

        rounds = list(range(1, len(staged_accs) + 1))
        fig1.add_trace(go.Scatter(
            x=rounds, y=staged_accs, mode="lines",
            line=dict(color="#66bb6a", width=2), showlegend=False
        ), row=1, col=2)
        fig1.update_xaxes(title_text="Number of Trees", row=1, col=2)
        fig1.update_yaxes(title_text="Test Accuracy", row=1, col=2)
    else:
        # RF: just confusion matrix
        fig1 = go.Figure(go.Heatmap(
            z=cm[::-1], x=class_names, y=class_names[::-1],
            text=cm[::-1], texttemplate="%{text}",
            colorscale="Blues", showscale=False
        ))
        fig1.update_layout(xaxis_title="Predicted", yaxis_title="Actual")

    fig1.update_layout(title=f"{algorithm} — {dataset_name}",
                       template="plotly_white", height=420)

    # Figure 2: Feature importance
    if has_importance:
        imp = model.feature_importances_
        order = np.argsort(imp)
        fig2 = go.Figure(go.Bar(
            x=imp[order], y=[feat_names[i][:20] for i in order],
            orientation="h", marker_color="#42a5f5"
        ))
        fig2.update_layout(title=f"{algorithm} — Feature Importances",
                           template="plotly_white", height=max(300, len(feat_names) * 22))
    else:
        fig2 = go.Figure().update_layout(template="plotly_white", height=300)

    md_text = f"""### {algorithm} — {dataset_name}

| Metric | Value |
|--------|-------|
| **Test Accuracy** | `{acc:.4f}` |
| **Train Accuracy** | `{accuracy_score(y_train, model.predict(X_train)):.4f}` |
| **Train / Test** | {len(y_train)} / {len(y_test)} |
"""
    return fig1, fig2, md_text


def build_tab():
    gr.Markdown("# 🌲 Module 05 — Ensemble Methods\n*Level: Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("""---
## 🎮 Interactive Demo

Try **Compare All** to see all 3 methods head-to-head with a boosting learning curve.""")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset", choices=["Iris", "Wine", "Breast Cancer"], value="Iris"
            )
            algo_dd = gr.Dropdown(
                label="Algorithm",
                choices=["Compare All", "Random Forest", "Gradient Boosting", "AdaBoost"],
                value="Compare All"
            )
            n_est_sl = gr.Slider(10, 200, value=100, step=10, label="n_estimators")
            depth_sl = gr.Slider(1, 10, value=5, step=1, label="max_depth")
            lr_sl = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="learning_rate (boosting)")
            test_sl = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Test Size")
            run_btn = gr.Button("▶ Run", variant="primary")

        with gr.Column(scale=3):
            plot1 = gr.Plot(label="Main Results")
            plot2 = gr.Plot(label="Feature Importances")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_ensemble,
        inputs=[dataset_dd, algo_dd, n_est_sl, depth_sl, lr_sl, test_sl],
        outputs=[plot1, plot2, metrics_out]
    )
