"""Module 06 — Clustering
Level: Intermediate"""
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, load_synthetic
from utils.plot_utils import scatter_2d, elbow_curve
from config import COLORS

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

THEORY = """## 🔵 Clustering — Unsupervised Learning

**Unsupervised learning** finds hidden structure in data **without labels**. Clustering groups data
points so that points within a cluster are more similar to each other than to points in other clusters.

---

### K-Means

K-Means assigns each point to the nearest centroid, then updates centroids to the mean of their cluster.
This repeats until convergence.

- **Objective:** Minimize **within-cluster sum of squared errors (SSE / inertia)**
- **Requires:** You must specify K upfront
- **Elbow Method:** Plot inertia vs K. The "elbow" — where adding more clusters stops helping much — is a good choice for K.
- **Limitations:** Assumes spherical, equally-sized clusters. Sensitive to outliers and initialization.

---

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN groups points that are densely packed together, marking sparse points as **noise (outliers)**.

- **Core point:** Has at least `min_samples` neighbors within radius `eps`
- **Border point:** Within `eps` of a core point but fewer than `min_samples` neighbors itself
- **Noise point:** Neither core nor border — labeled **-1**
- **Strengths:** Finds arbitrary-shaped clusters; naturally handles outliers; no need to specify K
- **Key params:** `eps` (neighborhood radius) and `min_samples` (density threshold)

---

### Agglomerative Hierarchical Clustering

Starts with each point as its own cluster, then **merges** the two closest clusters at each step.
The result can be visualized as a **dendrogram** (tree diagram).

- **Linkage criteria** determine "closest":
  - `ward`: minimizes within-cluster variance (default, usually best)
  - `complete`: max distance between all pairs
  - `average`: mean distance between all pairs
- **Advantage:** No assumption about cluster shape; you can inspect the full hierarchy

---

### Silhouette Score

Measures how well-separated clusters are. For each point:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

where `a(i)` = average distance to same-cluster points, `b(i)` = average distance to nearest other cluster.

- **Range:** −1 to +1. Higher is better.
- **> 0.5:** Reasonable structure; **> 0.7:** Strong structure; **< 0.2:** Weak or no structure

---

### When to Use Each Algorithm

| Algorithm | Use When |
|-----------|----------|
| **K-Means** | Clusters are roughly spherical, similar size; K is known or elbow is clear |
| **DBSCAN** | Clusters have irregular shapes; data has noise/outliers; K is unknown |
| **Agglomerative** | You want a hierarchy of clusters; dataset is not too large (O(n²) memory) |
"""

CODE_EXAMPLE = '''from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# K-Means with elbow method
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit with chosen K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
print(f"K-Means  silhouette: {silhouette_score(X_scaled, labels):.4f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
print(f"DBSCAN found {n_clusters} clusters, {n_noise} noise points")
if n_clusters > 1:
    mask = labels_db != -1
    print(f"DBSCAN  silhouette: {silhouette_score(X_scaled[mask], labels_db[mask]):.4f}")

# Agglomerative
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_agg = agg.fit_predict(X_scaled)
print(f"Agglom.  silhouette: {silhouette_score(X_scaled, labels_agg):.4f}")
'''


def _load_clustering_data(dataset_name):
    """Load and return (X_2d, y_or_none, feature_names)."""
    if dataset_name == "blobs":
        X, y = load_synthetic("blobs", n_samples=300)
        return X, y, ["Feature 0", "Feature 1"]
    elif dataset_name == "moons":
        X, y = load_synthetic("moons", n_samples=300, noise=0.1)
        return X, y, ["Feature 0", "Feature 1"]
    elif dataset_name == "circles":
        X, y = load_synthetic("circles", n_samples=300, noise=0.05)
        return X, y, ["Feature 0", "Feature 1"]
    elif dataset_name == "iris (2D)":
        X, y, feature_names, _ = load_dataset("iris")
        return X[:, :2], y, feature_names[:2]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _scatter_clusters(X, labels, title, feature_names):
    """Build a 2D scatter plot with cluster coloring. Noise points (label=-1) are grey X markers."""
    fig = go.Figure()
    unique_labels = np.unique(labels)
    palette = COLORS["palette"]

    color_idx = 0
    for lbl in sorted(unique_labels):
        mask = labels == lbl
        if lbl == -1:
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                name="Noise",
                marker=dict(color="black", size=6, symbol="x", opacity=0.6),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                name=f"Cluster {lbl}",
                marker=dict(color=palette[color_idx % len(palette)], size=7, opacity=0.8),
            ))
            color_idx += 1

    x_label = feature_names[0] if feature_names else "Feature 0"
    y_label = feature_names[1] if feature_names else "Feature 1"
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                      template="plotly_white", height=440)
    return fig


def run_clustering(dataset_name, algorithm, n_clusters, eps, min_samples, show_elbow):
    """Run clustering and return (scatter_fig, metrics_md)."""
    try:
        X, _, feature_names = _load_clustering_data(dataset_name)
        X_scaled = StandardScaler().fit_transform(X)

        n_clusters = int(n_clusters)
        min_samples = int(min_samples)

        if algorithm == "K-Means":
            if show_elbow:
                k_values = list(range(2, 11))
                inertias = []
                for k in k_values:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                fig = elbow_curve(k_values, inertias,
                                  title=f"K-Means Elbow Curve — {dataset_name}")
                metrics_md = f"""### K-Means Elbow Curve — `{dataset_name}`

| K | Inertia |
|---|---------|
""" + "\n".join(f"| {k} | {inr:.2f} |" for k, inr in zip(k_values, inertias))
                metrics_md += "\n\n> Look for the **elbow** — the K where inertia stops dropping sharply."
                return fig, metrics_md
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else float("nan")
                fig = _scatter_clusters(X_scaled, labels,
                                        f"K-Means (K={n_clusters}) — {dataset_name}", feature_names)
                metrics_md = f"""### K-Means — `{dataset_name}`

| Metric | Value |
|--------|-------|
| K (n_clusters) | **{n_clusters}** |
| Inertia | **{model.inertia_:.4f}** |
| Silhouette Score | **{sil:.4f}** |

> Silhouette score ranges from −1 to +1. Values above 0.5 indicate well-separated clusters.
"""
                return fig, metrics_md

        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

            valid_mask = labels != -1
            if n_found > 1 and valid_mask.sum() > n_found:
                sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            else:
                sil = float("nan")

            fig = _scatter_clusters(X_scaled, labels,
                                    f"DBSCAN (eps={eps}, min_samples={min_samples}) — {dataset_name}",
                                    feature_names)
            metrics_md = f"""### DBSCAN — `{dataset_name}`

| Metric | Value |
|--------|-------|
| eps | **{eps}** |
| min_samples | **{min_samples}** |
| Clusters found | **{n_found}** |
| Noise points | **{n_noise}** ({n_noise / len(labels) * 100:.1f}%) |
| Silhouette Score | **{"N/A" if np.isnan(sil) else f"{sil:.4f}"}** |

> Noise points (label = −1) are shown as black **✕** markers.
> Increase `eps` to merge nearby clusters; decrease `min_samples` to reduce noise classification.
"""
            return fig, metrics_md

        elif algorithm == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else float("nan")
            fig = _scatter_clusters(X_scaled, labels,
                                    f"Agglomerative (K={n_clusters}, ward) — {dataset_name}",
                                    feature_names)
            metrics_md = f"""### Agglomerative Clustering — `{dataset_name}`

| Metric | Value |
|--------|-------|
| n_clusters | **{n_clusters}** |
| Linkage | **ward** |
| Silhouette Score | **{sil:.4f}** |

> Ward linkage minimizes the total within-cluster variance at each merge step.
> It typically produces the most compact, even-sized clusters.
"""
            return fig, metrics_md

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=440)
        return empty_fig, f"**Error:** {str(e)}"


def build_tab():
    """Build the Gradio UI for the Clustering module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["blobs", "moons", "circles", "iris (2D)"],
                    value="blobs",
                    label="Dataset",
                )
                algorithm_radio = gr.Radio(
                    choices=["K-Means", "DBSCAN", "Agglomerative"],
                    value="K-Means",
                    label="Algorithm",
                )
                n_clusters_sl = gr.Slider(
                    minimum=2, maximum=10, value=3, step=1,
                    label="n_clusters (K-Means / Agglomerative)",
                )
                eps_sl = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                    label="eps (DBSCAN neighborhood radius)",
                )
                min_samples_sl = gr.Slider(
                    minimum=2, maximum=20, value=5, step=1,
                    label="min_samples (DBSCAN density threshold)",
                )
                show_elbow_cb = gr.Checkbox(
                    value=False,
                    label="Show Elbow Curve (K-Means only)",
                )
                run_btn = gr.Button("▶ Run", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Clustering Result")
                metrics_out = gr.Markdown(label="Metrics")

        run_btn.click(
            fn=run_clustering,
            inputs=[dataset_dd, algorithm_radio, n_clusters_sl,
                    eps_sl, min_samples_sl, show_elbow_cb],
            outputs=[plot_out, metrics_out],
        )

    return plot_out, metrics_out
