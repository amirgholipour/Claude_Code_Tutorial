"""
Module 23 — Responsible AI: Fairness, Bias & Ethics
Level: Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

THEORY = """
## 📖 What Is Responsible AI?

**Responsible AI** is the practice of developing and deploying AI systems that are fair, transparent, accountable, and respectful of human rights and dignity. As ML models make decisions that affect people's lives (loans, hiring, medical diagnosis, criminal justice), ensuring they don't encode or amplify human biases becomes critical.

> "A model trained on biased data learns to be biased. A model deployed without monitoring may become biased over time."

## 🏗️ Core Principles

### 1. Fairness
A model is **fair** when it doesn't discriminate against individuals based on protected attributes (race, gender, age, religion, disability).

#### Fairness Metrics

**Demographic Parity** (Independence):
```
P(ŷ=1 | A=0) = P(ŷ=1 | A=1)
```
The positive prediction rate should be equal across groups.

**Equalized Odds** (Separation):
```
P(ŷ=1 | A=0, Y=y) = P(ŷ=1 | A=1, Y=y)  for y ∈ {0,1}
```
TPR and FPR should be equal across groups.

**Predictive Parity** (Sufficiency):
```
P(Y=1 | ŷ=1, A=0) = P(Y=1 | ŷ=1, A=1)
```
Precision should be equal across groups.

**Disparate Impact**:
```
DI = P(ŷ=1 | A=minority) / P(ŷ=1 | A=majority)
DI < 0.8 → discriminatory (80% rule, used in US employment law)
```

> ⚠️ **Impossibility theorem**: You generally cannot satisfy all fairness criteria simultaneously. Choosing which to optimize is an ethical and legal decision, not just a technical one.

### 2. Bias Sources
| Source | Example |
|---|---|
| **Historical bias** | Training on data that reflects past discrimination |
| **Representation bias** | Training data doesn't represent all groups equally |
| **Measurement bias** | Proxy variables encode protected attributes |
| **Feedback loop bias** | Model predictions influence future training data |
| **Aggregation bias** | One model applied to heterogeneous subpopulations |

### 3. Bias Mitigation Strategies
- **Pre-processing**: Re-sample to balance groups, remove sensitive attributes
- **In-processing**: Add fairness constraints to the loss function
- **Post-processing**: Adjust decision thresholds per group (equalize odds)

### 4. Transparency & Explainability
- **Model cards**: Standardized documentation of model purpose, limitations, fairness analysis
- **SHAP / LIME**: Explain individual predictions
- **Fairness audits**: Regular third-party evaluation

### 5. Privacy
- **Differential privacy**: Add calibrated noise to prevent memorization of individual records
- **Federated learning**: Train on device without centralizing data
- **Data minimization**: Only collect and use data necessary for the task

## ✅ Responsible AI Checklist
- [ ] Who is affected by this model's decisions?
- [ ] What protected attributes exist in the data?
- [ ] Which fairness metric is most appropriate for this use case?
- [ ] Have you measured performance across demographic subgroups?
- [ ] Is the model's behavior explainable to affected individuals?
- [ ] Is there a human-in-the-loop for high-stakes decisions?
- [ ] Is there a process to appeal model decisions?

## ⚠️ Common Mistakes
- Removing protected attributes doesn't remove bias (proxies remain)
- High overall accuracy can hide poor performance on minority groups
- Treating fairness as purely technical ignores legal and ethical dimensions
"""

CODE_EXAMPLE = '''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Simulate biased hiring dataset ───────────────────────────────
np.random.seed(42)
n = 1000
gender = np.random.choice(["M", "F"], n, p=[0.6, 0.4])
experience = np.random.normal(5, 2, n).clip(0, 20)
skill_score = np.random.normal(70, 15, n).clip(0, 100)
# Introduce bias: historical data shows men were hired more
bias_factor = np.where(gender == "M", 0.3, 0.0)
p_hire = 1 / (1 + np.exp(-(0.1*experience + 0.05*skill_score - 6 + bias_factor)))
hired = np.random.binomial(1, p_hire)

df = pd.DataFrame({"gender": gender, "experience": experience,
                   "skill_score": skill_score, "hired": hired})

# ── Train model ───────────────────────────────────────────────────
X = df[["experience", "skill_score"]]
y = df["hired"]
model = LogisticRegression().fit(X, y)
df["predicted"] = model.predict(X)

# ── Fairness analysis ─────────────────────────────────────────────
for group in ["M", "F"]:
    mask = df["gender"] == group
    acc  = accuracy_score(df[mask]["hired"], df[mask]["predicted"])
    pos_rate = df[mask]["predicted"].mean()
    tp_rate = (df[mask]["predicted"] & df[mask]["hired"]).sum() / df[mask]["hired"].sum()
    print(f"{group}: acc={acc:.3f}, positive_rate={pos_rate:.3f}, TPR={tp_rate:.3f}")

# ── Disparate Impact ──────────────────────────────────────────────
pos_M = df[df["gender"]=="M"]["predicted"].mean()
pos_F = df[df["gender"]=="F"]["predicted"].mean()
DI = pos_F / pos_M
print(f"Disparate Impact: {DI:.3f} {'✅ Fair' if DI >= 0.8 else '❌ Discriminatory'}")
'''


def _make_loan_dataset(n=1000, bias_strength=0.5, seed=42):
    """Synthetic loan approval dataset with controllable bias."""
    rng = np.random.default_rng(seed)

    age        = rng.integers(22, 65, n)
    income     = rng.normal(55000, 20000, n).clip(15000, 200000)
    credit_score = rng.normal(650, 80, n).clip(300, 850)
    group      = rng.choice(["Group A", "Group B"], n, p=[0.6, 0.4])

    # Historical bias: Group B had lower approval rates due to bias
    group_bias = np.where(group == "Group A", bias_strength, 0.0)
    log_odds   = (income / 30000 + credit_score / 300 - 5 + group_bias)
    p_approve  = 1 / (1 + np.exp(-log_odds * 0.5))
    approved   = rng.binomial(1, p_approve)

    return pd.DataFrame({
        "age":          age,
        "income":       income,
        "credit_score": credit_score,
        "group":        group,
        "approved":     approved,
    })


def _compute_fairness_metrics(df, preds_col="predicted", label_col="approved", group_col="group"):
    results = {}
    for g in df[group_col].unique():
        mask = df[group_col] == g
        sub  = df[mask]
        pred = sub[preds_col]
        true = sub[label_col]

        acc       = accuracy_score(true, pred)
        pos_rate  = pred.mean()
        tpr       = (pred & true).sum() / max(true.sum(), 1)
        fpr       = ((pred == 1) & (true == 0)).sum() / max((true == 0).sum(), 1)
        precision = (pred & true).sum() / max(pred.sum(), 1)

        results[g] = {
            "accuracy":     acc,
            "positive_rate": pos_rate,
            "TPR":           tpr,
            "FPR":           fpr,
            "precision":     precision,
            "n":             len(sub),
        }

    groups = list(results.keys())
    if len(groups) >= 2:
        a, b = groups[0], groups[1]
        di   = results[b]["positive_rate"] / max(results[a]["positive_rate"], 1e-6)
        tpr_gap = abs(results[a]["TPR"] - results[b]["TPR"])
        fpr_gap = abs(results[a]["FPR"] - results[b]["FPR"])
    else:
        di, tpr_gap, fpr_gap = 1.0, 0.0, 0.0

    return results, di, tpr_gap, fpr_gap


def run_responsible_ai(demo_type: str, model_name: str, bias_strength: float, threshold_adjust: bool):
    df = _make_loan_dataset(n=1500, bias_strength=bias_strength)

    X = df[["income", "credit_score", "age"]].values
    y = df["approved"].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_sc, y, df.index, test_size=0.3, random_state=42
    )
    df_test = df.loc[idx_te].copy()

    MODELS = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    model = MODELS[model_name]
    model.fit(X_tr, y_tr)

    # Standard predictions
    proba = model.predict_proba(X_te)[:, 1]
    df_test["predicted"] = (proba >= 0.5).astype(int)

    if threshold_adjust:
        # Post-processing: equalize positive rates by group
        groups = df_test["group"].unique()
        group_preds = {}
        for g in groups:
            mask = df_test["group"] == g
            p_g  = proba[mask]
            group_preds[g] = p_g.mean()

        # Adjust threshold for disadvantaged group to match advantaged group
        tgt_rate = max(group_preds.values())
        for g in groups:
            mask = df_test["group"] == g
            if group_preds[g] < tgt_rate - 0.01:
                # Lower threshold for this group
                sorted_proba = np.sort(proba[mask])[::-1]
                n_target     = int(tgt_rate * mask.sum())
                new_thresh   = sorted_proba[min(n_target, len(sorted_proba)-1)]
                df_test.loc[df_test["group"] == g, "predicted"] = (proba[mask] >= new_thresh).astype(int)

    metrics, di, tpr_gap, fpr_gap = _compute_fairness_metrics(df_test)

    if demo_type == "Fairness Metrics Comparison":
        groups  = list(metrics.keys())
        metric_names = ["accuracy", "positive_rate", "TPR", "FPR", "precision"]
        metric_labels = ["Accuracy", "Positive Rate", "TPR (Recall)", "FPR", "Precision"]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metric_labels + ["Disparate Impact"],
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "indicator"}],
            ]
        )

        colors = {"Group A": "#42a5f5", "Group B": "#ef5350"}

        for i, (mn, ml) in enumerate(zip(metric_names, metric_labels)):
            row, col = divmod(i, 3)
            fig.add_trace(go.Bar(
                x=groups, y=[metrics[g][mn] for g in groups],
                marker_color=[colors.get(g, "#ffa726") for g in groups],
                text=[f"{metrics[g][mn]:.3f}" for g in groups],
                textposition="outside", showlegend=False
            ), row=row+1, col=col+1)

        # Disparate Impact gauge
        di_color = "#66bb6a" if di >= 0.8 else "#ef5350"
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=di,
            gauge={"axis": {"range": [0, 1.5]},
                   "bar": {"color": di_color},
                   "threshold": {"value": 0.8, "line": {"color": "red", "width": 3}}},
            title={"text": "Disparate Impact"},
        ), row=2, col=3)

        fig.update_layout(height=550, title_text=f"Fairness Analysis — {model_name} (bias={bias_strength})")

        rows_str = "\n".join([
            f"| {g} | {metrics[g]['accuracy']:.3f} | {metrics[g]['positive_rate']:.3f} | "
            f"{metrics[g]['TPR']:.3f} | {metrics[g]['FPR']:.3f} | {metrics[g]['n']} |"
            for g in groups
        ])
        metrics_md = f"""
### Fairness Analysis Results

| Group | Accuracy | Pos Rate | TPR | FPR | N |
|---|---|---|---|---|---|
{rows_str}

**Disparate Impact:** `{di:.3f}` {'✅ Fair (≥0.8)' if di >= 0.8 else '❌ Discriminatory (<0.8)'}
**TPR Gap:** `{tpr_gap:.3f}` {'✅' if tpr_gap < 0.05 else '⚠️'}
**FPR Gap:** `{fpr_gap:.3f}` {'✅' if fpr_gap < 0.05 else '⚠️'}

{'**Post-processing threshold adjustment applied** ✅' if threshold_adjust else ''}
> Bias strength: `{bias_strength}` — higher = more historical bias in training data
"""

    elif demo_type == "Subgroup Performance":
        # Intersectional analysis: group × income bracket
        df_test["income_bracket"] = pd.cut(
            df.loc[idx_te, "income"], bins=3, labels=["Low", "Mid", "High"]
        )

        sub_results = []
        for g in df_test["group"].unique():
            for inc in ["Low", "Mid", "High"]:
                mask = (df_test["group"] == g) & (df_test["income_bracket"] == inc)
                if mask.sum() < 5:
                    continue
                sub = df_test[mask]
                acc = accuracy_score(sub["approved"], sub["predicted"])
                pos = sub["predicted"].mean()
                sub_results.append({"group": g, "income": inc, "acc": acc, "pos_rate": pos, "n": mask.sum()})

        sdf = pd.DataFrame(sub_results)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy by Subgroup", "Approval Rate by Subgroup"])

        for g in sdf["group"].unique():
            sub = sdf[sdf["group"] == g]
            color = "#42a5f5" if g == "Group A" else "#ef5350"
            fig.add_trace(go.Bar(x=sub["income"], y=sub["acc"],
                                 name=g, marker_color=color,
                                 text=[f"{v:.3f}" for v in sub["acc"]],
                                 textposition="outside"), row=1, col=1)
            fig.add_trace(go.Bar(x=sub["income"], y=sub["pos_rate"],
                                 name=g, marker_color=color, showlegend=False,
                                 text=[f"{v:.3f}" for v in sub["pos_rate"]],
                                 textposition="outside"), row=1, col=2)

        fig.update_yaxes(range=[0, 1.1], row=1, col=1)
        fig.update_yaxes(range=[0, 1.1], row=1, col=2)
        fig.update_layout(height=420, barmode="group",
                          title_text="Intersectional Fairness Analysis")

        metrics_md = f"""
### Intersectional Analysis (Group × Income)

Intersectionality reveals that fairness issues can be **concentrated in specific subgroups** (e.g., low-income Group B) even if aggregate metrics look acceptable.

**Key insight**: A model that's "accurate overall" can still fail specific subpopulations.

Bias strength: `{bias_strength}` | Threshold adjusted: `{threshold_adjust}`
"""

    else:
        fig = go.Figure()
        metrics_md = "Select a demo type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# ⚖️ Module 23 — Responsible AI: Fairness, Bias & Ethics\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nAnalyze fairness metrics on a synthetic loan approval dataset. Adjust bias strength and observe the impact on demographic parity, equalized odds, and disparate impact.")

    with gr.Row():
        with gr.Column(scale=1):
            demo_dd = gr.Dropdown(
                label="Demo Type",
                choices=["Fairness Metrics Comparison", "Subgroup Performance"],
                value="Fairness Metrics Comparison"
            )
            model_dd = gr.Dropdown(
                label="Model",
                choices=["Logistic Regression", "Random Forest", "Gradient Boosting"],
                value="Logistic Regression"
            )
            bias_sl = gr.Slider(label="Historical Bias Strength", minimum=0.0, maximum=2.0,
                                step=0.25, value=0.5)
            threshold_cb = gr.Checkbox(label="Apply post-processing threshold adjustment", value=False)
            run_btn = gr.Button("▶ Analyze Fairness", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_responsible_ai,
        inputs=[demo_dd, model_dd, bias_sl, threshold_cb],
        outputs=[plot_out, metrics_out]
    )
