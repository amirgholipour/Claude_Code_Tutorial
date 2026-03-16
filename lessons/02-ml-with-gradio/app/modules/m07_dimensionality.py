"""Module 07 — Dimensionality Reduction
Level: Intermediate"""
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset
from utils.plot_utils import scatter_2d, scatter_3d
from config import COLORS

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

THEORY = """## 📉 Dimensionality Reduction

### The Curse of Dimensionality

As the number of features grows, the data becomes increasingly **sparse**. In high-dimensional spaces:

- **Distance loses meaning** — all pairs of points become roughly equidistant (distances converge)
- **Exponentially more data** is needed to fill the space and get reliable density estimates
- **Overfitting increases** — models can fit noise dimensions; more regularization is needed
- **Visualization is impossible** — humans can only perceive 2–3 dimensions directly

Dimensionality reduction addresses this by projecting data into a lower-dimensional space that **preserves the most important structure**.

---

### PCA — Principal Component Analysis

PCA finds orthogonal directions (principal components) in the data that capture **maximum variance**.
It is a **linear** transformation.

1. Center the data (subtract mean)
2. Compute the covariance matrix
3. Find its eigenvectors (principal components) and eigenvalues (variance explained)
4. Project data onto the top K eigenvectors

**Explained variance ratio:** How much of the total variance each component captures.
- Component 1 always captures the most variance
- Sum the ratios to see how much information you retain

**Use cases:**
- **Visualization:** Project to 2D/3D for inspection
- **Noise reduction:** Drop low-variance components (likely noise)
- **Speed up ML:** Reduce features before training an expensive model
- **Decorrelation:** PCA components are uncorrelated by design

---

### t-SNE — t-Distributed Stochastic Neighbor Embedding

t-SNE is a **nonlinear** method that preserves **local structure** — nearby points in the original space
stay nearby in 2D/3D.

- Models similarity between point pairs as probabilities in high-D, then minimizes KL divergence in low-D
- **Excellent** for visualizing clusters and class separability
- **Perplexity** ≈ effective number of neighbors. Low (5–10) = local structure; High (30–50) = global structure
- **Warnings:**
  - Non-deterministic: different runs give different layouts (use `random_state` for reproducibility)
  - **Cannot be used for preprocessing** — t-SNE has no `transform()` for new data
  - Slow on large datasets (O(n²)); use on ≤ 5000 samples
  - Inter-cluster distances are **not meaningful** — only within-cluster structure is preserved

---

### LDA — Linear Discriminant Analysis

LDA is a **supervised** dimensionality reduction method. It finds directions that **maximize class separability** (maximize between-class variance relative to within-class variance).

- Unlike PCA (unsupervised), LDA uses the class labels to find the best discriminating directions
- Maximum number of components: `min(n_classes − 1, n_features)`
- Also works as a linear classifier (same math as Fisher's discriminant)
- **Use case:** When you want to visualize or compress data while preserving class structure

---

### Method Comparison

| Property | PCA | t-SNE | LDA |
|----------|-----|-------|-----|
| Type | Linear | Nonlinear | Linear |
| Supervised | No | No | **Yes** |
| Preserves | Global variance | Local neighborhood | Class separation |
| New data | Yes (`transform`) | No | Yes (`transform`) |
| Speed | Fast | **Slow** (O(n²)) | Fast |
| Best for | Preprocessing, noise reduction | Visualization only | Classification + visualization |
"""

CODE_EXAMPLE = '''from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")

# PCA to keep 95% of variance
pca_auto = PCA(n_components=0.95)
X_pca_auto = pca_auto.fit_transform(X_scaled)
print(f"Components needed for 95% variance: {pca_auto.n_components_}")

# t-SNE to 2D (visualization only — no transform for new data)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# LDA — supervised, maximizes class separability
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
'''


def run_dimensionality(dataset_name, method, dimensions, perplexity):
    """Run dimensionality reduction and return (figure, info_md)."""
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        X_scaled = StandardScaler().fit_transform(X)

        n_dim = 3 if dimensions == "3D" else 2
        perplexity = int(perplexity)

        # LDA max components is min(n_classes-1, n_features)
        n_classes = len(np.unique(y))
        lda_max_components = min(n_classes - 1, X_scaled.shape[1])

        if method == "PCA":
            n_components = min(n_dim, X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
            evr = pca.explained_variance_ratio_
            total_var = evr.sum()

            comp_labels = [f"PC{i+1}" for i in range(n_components)]

            if n_dim == 2:
                # Side-by-side: scatter (left) + variance bar (right)
                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.65, 0.35],
                    subplot_titles=["2D Projection", "Explained Variance per Component"],
                )
                palette = COLORS["palette"]
                for i, cls in enumerate(np.unique(y)):
                    mask = y == cls
                    label = str(target_names[cls]) if cls < len(target_names) else f"Class {cls}"
                    fig.add_trace(
                        go.Scatter(x=X_reduced[mask, 0], y=X_reduced[mask, 1],
                                   mode="markers", name=label,
                                   marker=dict(color=palette[i % len(palette)], size=7, opacity=0.8)),
                        row=1, col=1,
                    )
                fig.add_trace(
                    go.Bar(x=comp_labels, y=evr,
                           marker_color=COLORS["primary"],
                           text=[f"{v:.1%}" for v in evr],
                           textposition="outside",
                           showlegend=False,
                           name="Explained Variance"),
                    row=1, col=2,
                )
                fig.update_xaxes(title_text="PC1", row=1, col=1)
                fig.update_yaxes(title_text="PC2", row=1, col=1)
                fig.update_xaxes(title_text="Component", row=1, col=2)
                fig.update_yaxes(title_text="Variance Ratio", range=[0, max(evr) * 1.2], row=1, col=2)
                fig.update_layout(title=f"PCA — {dataset_name}", template="plotly_white", height=450)
            else:
                # 3D scatter
                n_components = min(3, X_scaled.shape[1])
                pca3 = PCA(n_components=n_components)
                X_reduced = pca3.fit_transform(X_scaled)
                evr = pca3.explained_variance_ratio_
                total_var = evr.sum()
                comp_labels = [f"PC{i+1}" for i in range(n_components)]
                fig = scatter_3d(X_reduced, y,
                                 title=f"PCA 3D — {dataset_name}",
                                 feature_names=comp_labels,
                                 class_names=list(target_names))

            evr_rows = "\n".join(
                f"| {comp_labels[i]} | {evr[i]:.4f} | {evr[i]:.2%} | {evr[:i+1].sum():.2%} |"
                for i in range(len(evr))
            )
            info_md = f"""### PCA — `{dataset_name}`

| Property | Value |
|----------|-------|
| Original features | **{X_scaled.shape[1]}** |
| Reduced to | **{n_dim}D** |
| Total variance retained | **{total_var:.2%}** |

### Explained Variance per Component
| Component | Eigenvalue (ratio) | % Variance | Cumulative % |
|-----------|--------------------|------------|-------------|
{evr_rows}

> **Interpretation:** PC1 captures the single most variable direction in the data.
> The cumulative % tells you how much information you keep when projecting to K dimensions.
"""
            return fig, info_md

        elif method == "t-SNE":
            # t-SNE only supports 2D or 3D in sklearn; 3D is very slow
            n_components_tsne = min(n_dim, 3)
            tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity,
                        random_state=42, n_iter=1000, init="pca")
            X_reduced = tsne.fit_transform(X_scaled)

            comp_labels = [f"Dim{i+1}" for i in range(n_components_tsne)]

            if n_components_tsne == 2:
                fig = scatter_2d(X_reduced, y,
                                 title=f"t-SNE (perplexity={perplexity}) — {dataset_name}",
                                 feature_names=comp_labels,
                                 class_names=list(target_names))
            else:
                fig = scatter_3d(X_reduced, y,
                                 title=f"t-SNE 3D (perplexity={perplexity}) — {dataset_name}",
                                 feature_names=comp_labels,
                                 class_names=list(target_names))

            info_md = f"""### t-SNE — `{dataset_name}`

| Property | Value |
|----------|-------|
| Perplexity | **{perplexity}** |
| Dimensions | **{n_components_tsne}D** |
| Samples | **{X_scaled.shape[0]}** |
| Original features | **{X_scaled.shape[1]}** |

> **Perplexity** ≈ effective number of neighbors. Try 5–10 for local structure, 30–50 for global.

> **Warning:** t-SNE cannot transform new data. It is for **visualization only** — do not use
> t-SNE output as features for a downstream model.
> Inter-cluster distances are not meaningful; only within-cluster topology is preserved.
"""
            return fig, info_md

        elif method == "LDA":
            if lda_max_components < 1:
                raise ValueError(
                    f"LDA requires at least 2 classes. '{dataset_name}' has {n_classes} class(es)."
                )
            n_components_lda = min(n_dim, lda_max_components)
            lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
            X_reduced = lda.fit_transform(X_scaled, y)

            comp_labels = [f"LD{i+1}" for i in range(n_components_lda)]

            if n_components_lda == 1:
                # Only 1 component available (2-class problem) — plot as 1D histogram
                palette = COLORS["palette"]
                fig = go.Figure()
                for i, cls in enumerate(np.unique(y)):
                    mask = y == cls
                    label = str(target_names[cls]) if cls < len(target_names) else f"Class {cls}"
                    fig.add_trace(go.Histogram(
                        x=X_reduced[mask, 0], name=label,
                        opacity=0.7,
                        marker_color=palette[i % len(palette)],
                    ))
                fig.update_layout(
                    title=f"LDA 1D — {dataset_name}",
                    xaxis_title="LD1", yaxis_title="Count",
                    barmode="overlay", template="plotly_white", height=420,
                )
            elif n_dim == 2 or n_components_lda == 2:
                fig = scatter_2d(X_reduced, y,
                                 title=f"LDA 2D — {dataset_name}",
                                 feature_names=comp_labels,
                                 class_names=list(target_names))
            else:
                fig = scatter_3d(X_reduced, y,
                                 title=f"LDA 3D — {dataset_name}",
                                 feature_names=comp_labels,
                                 class_names=list(target_names))

            evr = getattr(lda, "explained_variance_ratio_", None)
            if evr is not None and len(evr) > 0:
                evr_rows = "\n".join(
                    f"| LD{i+1} | {evr[i]:.4f} | {evr[i]:.2%} |"
                    for i in range(len(evr))
                )
                evr_table = f"""### Explained Variance Ratio (between-class)
| Component | Ratio | % |
|-----------|-------|---|
{evr_rows}
"""
            else:
                evr_table = ""

            info_md = f"""### LDA — `{dataset_name}`

| Property | Value |
|----------|-------|
| Original features | **{X_scaled.shape[1]}** |
| Classes | **{n_classes}** |
| Max LDA components | **{lda_max_components}** (= n_classes − 1) |
| Used components | **{n_components_lda}** |

{evr_table}
> LDA is **supervised**: it uses class labels to find the directions that best **separate** the classes,
> unlike PCA which ignores labels entirely.

> **Class separability:** Well-separated blobs in the LDA plot indicate that a linear classifier
> will likely perform well on this dataset.
"""
            return fig, info_md

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=450)
        return empty_fig, f"**Error:** {str(e)}"


def build_tab():
    """Build the Gradio UI for the Dimensionality Reduction module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["iris", "wine", "digits", "breast_cancer"],
                    value="iris",
                    label="Dataset",
                )
                method_radio = gr.Radio(
                    choices=["PCA", "t-SNE", "LDA"],
                    value="PCA",
                    label="Method",
                )
                dimensions_radio = gr.Radio(
                    choices=["2D", "3D"],
                    value="2D",
                    label="Dimensions",
                )
                perplexity_sl = gr.Slider(
                    minimum=5, maximum=50, value=30, step=5,
                    label="Perplexity (t-SNE only)",
                )
                gr.Markdown(
                    "> ⚠️ **t-SNE on `digits`** (64 features, 1797 samples) may take ~20–30 seconds."
                )
                run_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Projection")
                info_out = gr.Markdown(label="Info")

        run_btn.click(
            fn=run_dimensionality,
            inputs=[dataset_dd, method_radio, dimensions_radio, perplexity_sl],
            outputs=[plot_out, info_out],
        )

    return plot_out, info_out
