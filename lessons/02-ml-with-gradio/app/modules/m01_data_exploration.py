"""Module 01 — Data Exploration
Level: Basic"""
import gradio as gr
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, to_dataframe
from utils.plot_utils import histogram_grid, correlation_heatmap
from config import COLORS

THEORY = """## 📖 What Is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis is the critical first step in any machine learning project. Before training a single model, you need to **understand your data deeply**.

### Why EDA Matters
- Reveals data quality issues (missing values, outliers, duplicates) before they corrupt your model
- Informs feature engineering decisions — which transformations are needed?
- Exposes class imbalance that requires special handling (oversampling, weighted loss)
- Helps choose the right algorithm — linear vs. non-linear, distance-based vs. tree-based

### Key EDA Steps

| Step | What to Check | Why |
|------|--------------|-----|
| **Shape & types** | `df.shape`, `df.dtypes` | Understand scale, catch mixed types |
| **Summary stats** | `df.describe()` | Spot extreme values, scale differences |
| **Missing values** | `df.isnull().sum()` | Decide on imputation strategy |
| **Distributions** | Histograms, box plots | Detect skew, outliers, normality |
| **Correlations** | Heatmap, pairplot | Find redundant features, linear relationships |
| **Class balance** | `value_counts()` | Assess if target is imbalanced |

### When to Use Each Visualization
- **Histograms**: Continuous features — understand the shape of the distribution
- **Correlation Heatmap**: All numeric features together — find multicollinearity
- **Bar chart (class balance)**: Target variable — detect imbalance

### Common Pitfalls to Avoid
⚠️ **Data leakage**: Never compute statistics on the full dataset and then split — always fit scalers/imputers on training data only.

⚠️ **Peeking at the test set**: EDA should be done on training data. Looking at test distributions can bias your feature engineering choices.

⚠️ **Ignoring outliers**: A single extreme value can skew mean/std dramatically. Always check with box plots or percentiles.

⚠️ **Assuming normality**: Many algorithms (k-NN, SVM, neural nets) benefit from scaled features, but don't assume your data is Gaussian without checking.
"""

CODE_EXAMPLE = '''from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.describe())                    # summary statistics
print(df.isnull().sum())               # missing values per column
print(df['target'].value_counts())     # class balance

# Correlation matrix
corr = df.corr()
print(corr)

# Quick visualization with pandas
df.hist(figsize=(10, 6), bins=20)
'''


def run_eda(dataset_name: str, plot_type: str):
    """
    Run EDA on the selected dataset and return a plotly figure + info markdown.

    Args:
        dataset_name: One of iris, wine, breast_cancer, digits, diabetes
        plot_type: One of "Distributions", "Correlation Heatmap", "Class Balance"

    Returns:
        (plotly_fig, info_markdown)
    """
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        df = to_dataframe(X, y, feature_names=feature_names, target_name="target")

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        missing = int(df.isnull().sum().sum())

        # --- build figure based on selected plot type ---
        if plot_type == "Distributions":
            # Drop the target column so we only plot features
            feature_df = df.drop(columns=["target"])
            fig = histogram_grid(feature_df, n_cols=3,
                                 title=f"{dataset_name.capitalize()} — Feature Distributions")

        elif plot_type == "Correlation Heatmap":
            feature_df = df.drop(columns=["target"])
            fig = correlation_heatmap(feature_df,
                                      title=f"{dataset_name.capitalize()} — Correlation Matrix")

        elif plot_type == "Class Balance":
            unique, counts = np.unique(y, return_counts=True)
            # Map numeric labels to target names where available
            labels = [str(target_names[int(u)]) if int(u) < len(target_names) else f"Class {u}"
                      for u in unique]
            fig = go.Figure(go.Bar(
                x=labels,
                y=counts,
                marker_color=COLORS["palette"][:len(unique)],
                text=counts,
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{dataset_name.capitalize()} — Class Balance",
                xaxis_title="Class",
                yaxis_title="Sample Count",
                template="plotly_white",
                height=400,
            )

        else:
            return None, f"Unknown plot type: `{plot_type}`"

        # --- summary info markdown ---
        stats = df.describe().round(3)
        # Show stats for first 4 features to keep markdown readable
        preview_cols = feature_names[:4]
        stats_rows = ""
        for col in preview_cols:
            s = stats[col]
            stats_rows += (f"| `{col[:20]}` | {s['mean']:.3f} | {s['std']:.3f} | "
                           f"{s['min']:.3f} | {s['max']:.3f} |\n")

        info_md = f"""### Dataset: `{dataset_name}`
| Property | Value |
|----------|-------|
| Samples | **{n_samples}** |
| Features | **{n_features}** |
| Classes/Targets | **{n_classes}** |
| Missing Values | **{missing}** |

### Feature Statistics (first 4 features)
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
{stats_rows}
> **Tip:** Look for features with very different scales — those may need normalization before training distance-based models.
"""
        return fig, info_md

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=400)
        return empty_fig, f"**Error:** {str(e)}"


def build_tab():
    """Build the Gradio UI for the Data Exploration module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["iris", "wine", "breast_cancer", "digits", "diabetes"],
                    value="iris",
                    label="Dataset",
                )
                plot_type_radio = gr.Radio(
                    choices=["Distributions", "Correlation Heatmap", "Class Balance"],
                    value="Distributions",
                    label="Plot Type",
                )
                run_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Visualization")
                info_out = gr.Markdown(label="Dataset Info")

        run_btn.click(
            fn=run_eda,
            inputs=[dataset_dd, plot_type_radio],
            outputs=[plot_out, info_out],
        )

    return plot_out, info_out
