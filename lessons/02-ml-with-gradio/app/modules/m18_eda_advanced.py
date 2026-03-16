"""
Module 18 — Advanced Exploratory Data Analysis (EDA)
Level: Foundation / Intermediate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_diabetes

THEORY = """
## 📖 What Is Exploratory Data Analysis?

EDA is the **first mandatory step** in any ML project — the practice of understanding your data before building models. It guides every downstream decision: which features to use, which to transform, which model families are appropriate.

> "Torture the data, and it will confess to anything." — Ronald Coase
>
> EDA is about listening to the data, not forcing it.

## 🏗️ EDA Framework

### 1. Univariate Analysis (one feature at a time)
- **Histogram**: Shape of distribution (normal? skewed? bimodal?)
- **Box plot**: Median, IQR, outliers at a glance
- **Violin plot**: Combines box plot + kernel density
- **Key statistics**: Mean, median, std, skewness, kurtosis, min/max

### 2. Bivariate Analysis (two features together)
- **Scatter plot**: Relationship between two numeric features
- **Correlation matrix**: Pairwise linear correlations (Pearson, Spearman)
- **Box plot by class**: How does a numeric feature differ across classes?
- **Pair plot**: All scatter plots at once

### 3. Multivariate Analysis
- **Parallel coordinates**: Each feature as an axis, each sample as a polyline
- **3D scatter**: Three features simultaneously
- **PCA biplot**: Projects all features into 2D showing variance structure

### 4. Statistical Tests
| Question | Test |
|---|---|
| Is the distribution normal? | Shapiro-Wilk, D'Agostino K² |
| Are two groups different? | Mann-Whitney U, t-test |
| Are two features correlated? | Pearson r (+ p-value) |
| Is a categorical variable independent? | Chi-squared |

### 5. Distribution Shape Descriptors
- **Skewness > 0**: Right tail (income, house prices) — consider log transform
- **Skewness < 0**: Left tail (test scores)
- **Kurtosis > 3**: Heavy tails (leptokurtic) — outliers likely
- **Kurtosis < 3**: Light tails (platykurtic)

## ✅ EDA Checklist
- [ ] Shape and dtypes of dataset
- [ ] Missing value counts and patterns
- [ ] Distribution of each feature (histograms)
- [ ] Outlier identification (box plots)
- [ ] Feature correlations (heatmap)
- [ ] Class balance (for classification)
- [ ] Feature vs target relationships
- [ ] Temporal trends (if time-based)

## ⚠️ Common Pitfalls
- **Confirmation bias**: Looking for patterns you expect, missing unexpected ones
- **Ignoring multicollinearity**: Two correlated features inflate variance in linear models
- **Skipping EDA**: Jumping to modeling without understanding data distribution
- **Spurious correlations**: High correlation ≠ causation
"""

CODE_EXAMPLE = '''
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame

# ── Basic stats ───────────────────────────────────────────────────
print(df.describe())
print("\\nMissing:", df.isnull().sum().sum())

# ── Skewness & kurtosis ───────────────────────────────────────────
for col in df.select_dtypes(include=np.number).columns[:5]:
    sk  = df[col].skew()
    ku  = df[col].kurt()
    print(f"{col}: skew={sk:.2f}, kurtosis={ku:.2f}")

# ── Normality test ────────────────────────────────────────────────
stat, p = stats.normaltest(df["mean radius"])
print(f"\\nNormality test mean_radius: p={p:.4f} → {'NOT normal' if p < 0.05 else 'normal'}")

# ── Pearson correlation ───────────────────────────────────────────
corr_matrix = df.corr(numeric_only=True)
top_corr = corr_matrix.unstack().sort_values(ascending=False)
top_corr = top_corr[top_corr < 1].head(5)
print("\\nTop correlated pairs:", top_corr)

# ── Class balance ─────────────────────────────────────────────────
print("\\nClass balance:", df["target"].value_counts())
'''


def _load_dataset(name: str):
    LOADERS = {
        "Breast Cancer": load_breast_cancer,
        "Wine":          load_wine,
        "Iris":          load_iris,
        "Diabetes":      load_diabetes,
    }
    data   = LOADERS[name](as_frame=True)
    df     = data.frame.copy()
    target = "target"
    return df, target, data.feature_names


def run_eda(dataset_name: str, eda_type: str, feature_idx: int):
    df, target_col, feat_names = _load_dataset(dataset_name)
    feat_cols = [c for c in df.columns if c != target_col]
    feat_idx  = min(feature_idx, len(feat_cols) - 1)
    feat      = feat_cols[feat_idx]

    if eda_type == "Distribution Analysis":
        vals = df[feat].dropna().values
        sk   = float(pd.Series(vals).skew())
        ku   = float(pd.Series(vals).kurt())
        _, p_norm = stats.normaltest(vals)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Histogram: {feat[:20]}", f"Box + Violin: {feat[:20]}"])

        # Histogram + KDE
        fig.add_trace(go.Histogram(x=vals, nbinsx=30, name="Histogram",
                                   marker_color="#42a5f5", opacity=0.7,
                                   histnorm="probability density"), row=1, col=1)

        # KDE via scipy
        kde = stats.gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 200)
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode="lines",
                                 line=dict(color="#ef5350", width=2), name="KDE"), row=1, col=1)

        # Box plot
        fig.add_trace(go.Box(y=vals, name=feat[:15], marker_color="#66bb6a",
                             boxpoints="outliers"), row=1, col=2)

        fig.update_layout(height=420, showlegend=False)

        metrics_md = f"""
### Distribution Stats: `{feat}`

| Metric | Value |
|---|---|
| Mean | `{vals.mean():.4f}` |
| Median | `{np.median(vals):.4f}` |
| Std Dev | `{vals.std():.4f}` |
| Skewness | `{sk:.4f}` {'⚠️ right skew' if sk > 1 else '⚠️ left skew' if sk < -1 else '✅ approx symmetric'} |
| Kurtosis | `{ku:.4f}` {'⚠️ heavy tails' if ku > 3 else '✅ normal tails'} |
| Normality (p-value) | `{p_norm:.4f}` → {'❌ NOT normal' if p_norm < 0.05 else '✅ approximately normal'} |
| Outliers (IQR) | `{int(((vals < np.percentile(vals, 25) - 1.5*(np.percentile(vals, 75)-np.percentile(vals, 25))) | (vals > np.percentile(vals, 75) + 1.5*(np.percentile(vals, 75)-np.percentile(vals, 25)))).sum())}` samples |
"""

    elif eda_type == "Correlation Heatmap":
        num_df = df[feat_cols].select_dtypes(include=np.number)
        corr   = num_df.corr()
        short_names = [n[:10] for n in corr.columns]

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=short_names, y=short_names,
            colorscale="RdBu_r", zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="r")
        ))
        fig.update_layout(height=500, title_text=f"Pearson Correlation — {dataset_name}")

        # Top pairs
        pairs = corr.unstack()
        pairs = pairs[pairs.abs() < 1.0].sort_values(key=abs, ascending=False)
        top   = pairs.head(5)
        top_str = "\n".join([f"- `{a[:12]}` × `{b[:12]}`: **{v:.3f}**"
                             for (a, b), v in top.items()])

        metrics_md = f"""
### Top Correlated Feature Pairs ({dataset_name})
{top_str}

> **Reminder**: High correlation between features = multicollinearity.
> Consider removing one of a highly correlated pair for linear models.
"""

    elif eda_type == "Class Distribution":
        if target_col not in df.columns:
            return go.Figure(), "No target column found."

        counts = df[target_col].value_counts().sort_index()
        labels = [str(l) for l in counts.index]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Class Balance (bar)", "Feature by Class (violin)"])

        fig.add_trace(go.Bar(x=labels, y=counts.values,
                             marker_color=["#42a5f5", "#66bb6a", "#ef5350"][:len(labels)],
                             text=counts.values, textposition="outside",
                             name="Count"), row=1, col=1)

        colors = ["#42a5f5", "#66bb6a", "#ef5350", "#ffa726", "#ab47bc"]
        for i, cls in enumerate(sorted(df[target_col].unique())):
            vals_cls = df[df[target_col] == cls][feat].dropna().values
            fig.add_trace(go.Violin(y=vals_cls, name=f"Class {cls}",
                                   box_visible=True,
                                   marker_color=colors[i % len(colors)],
                                   meanline_visible=True), row=1, col=2)

        fig.update_layout(height=420, showlegend=True)

        balance_ratio = counts.min() / counts.max()
        metrics_md = f"""
### Class Distribution — {dataset_name}

| Class | Count | Fraction |
|---|---|---|
{''.join([f"| {l} | {v} | {v/counts.sum():.1%} |\n" for l, v in zip(labels, counts.values)])}
**Balance ratio** (minority/majority): `{balance_ratio:.2f}`
{'⚠️ Imbalanced dataset — consider SMOTE or class weights' if balance_ratio < 0.5 else '✅ Reasonably balanced'}
"""

    elif eda_type == "Outlier Summary":
        num_df = df[feat_cols].select_dtypes(include=np.number)
        outlier_counts = {}
        for col in num_df.columns:
            vals = num_df[col].dropna().values
            Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            IQR = Q3 - Q1
            n_out = int(((vals < Q1 - 1.5*IQR) | (vals > Q3 + 1.5*IQR)).sum())
            outlier_counts[col[:15]] = n_out

        sorted_out = sorted(outlier_counts.items(), key=lambda x: -x[1])
        names = [x[0] for x in sorted_out]
        counts_out = [x[1] for x in sorted_out]

        colors = ["#ef5350" if c > 10 else "#ffa726" if c > 3 else "#66bb6a"
                  for c in counts_out]

        fig = go.Figure(go.Bar(
            x=names, y=counts_out,
            marker_color=colors,
            text=counts_out, textposition="outside"
        ))
        fig.update_layout(height=420, title_text=f"Outlier Counts by Feature (IQR method) — {dataset_name}",
                          xaxis_tickangle=-45)

        high_out = [(n, c) for n, c in zip(names, counts_out) if c > 5]
        metrics_md = f"""
### Outlier Summary — {dataset_name}

Total features analysed: **{len(num_df.columns)}**
Features with >5 outliers: **{len(high_out)}**

{chr(10).join([f"- `{n}`: **{c}** outliers" for n, c in high_out[:8]])}

> **Red** = >10 outliers, **Orange** = 3–10, **Green** = <3
> Investigate high-outlier features before training.
"""

    else:
        fig = go.Figure()
        metrics_md = "Select an EDA type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🔬 Module 18 — Advanced EDA\n*Level: Foundation / Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nDeep-dive EDA on built-in datasets. Explore distributions, correlations, class balance, and outlier profiles.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Breast Cancer", "Wine", "Iris", "Diabetes"],
                value="Breast Cancer"
            )
            eda_dd = gr.Dropdown(
                label="EDA Type",
                choices=["Distribution Analysis", "Correlation Heatmap",
                         "Class Distribution", "Outlier Summary"],
                value="Correlation Heatmap"
            )
            feat_sl = gr.Slider(
                label="Feature Index (for univariate views)",
                minimum=0, maximum=29, step=1, value=0
            )
            run_btn = gr.Button("▶ Run EDA", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_eda,
        inputs=[dataset_dd, eda_dd, feat_sl],
        outputs=[plot_out, metrics_out]
    )
