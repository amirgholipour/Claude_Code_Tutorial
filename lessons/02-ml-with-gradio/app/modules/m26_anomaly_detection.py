"""
Module 26 — Anomaly Detection
Level: Intermediate / Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.metrics import f1_score, precision_score, recall_score

THEORY = """
## 📖 What Is Anomaly Detection?

**Anomaly detection** (also called outlier detection or novelty detection) is the task of identifying **unusual patterns** that deviate from expected behavior. Unlike supervised classification, anomaly detection typically works without labeled anomaly examples — which is realistic, since anomalies by definition are rare and often unknown in advance.

**Applications:**
- Fraud detection (unusual transactions)
- Network intrusion detection
- Manufacturing defect detection
- Medical diagnosis (unusual vitals)
- Predictive maintenance (abnormal sensor readings)

## 🏗️ Three Categories

### 1. Density-Based Methods
Anomalies are in low-density regions:

**Local Outlier Factor (LOF)**:
- For each point, compare local density to density of its k neighbors
- LOF score ≈ 1.0 → normal; LOF >> 1.0 → anomaly
- **Strength**: Detects local outliers (relative to neighborhood)
- **Weakness**: Sensitive to parameter k

### 2. Isolation-Based Methods

**Isolation Forest**:
- Randomly partition feature space with binary trees
- Anomalies are isolated in fewer splits (shorter path length)
- **Key insight**: Outliers are "easier to isolate" than normal points
- **Strength**: Scales well (O(n log n)), handles high dimensions
- **Weakness**: Less effective on very high-dimensional data

```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)
scores = model.decision_function(X_test)  # negative = more anomalous
preds  = model.predict(X_test)  # -1 = anomaly, 1 = normal
```

### 3. One-Class Classification

**One-Class SVM**:
- Learns a decision boundary around "normal" training data
- Points outside boundary = anomalies
- **Strength**: Works well when normal class is well-defined
- **Weakness**: Slow on large datasets (O(n²)), sensitive to kernel choice

### 4. Reconstruction-Based Methods (Deep Learning)

**Autoencoder**:
- Encoder compresses input to low-dimensional latent space
- Decoder reconstructs original input
- Anomalies are poorly reconstructed → high reconstruction error
- Works for images, time series, tabular data

### 5. Statistical Methods
- **Z-score**: Outlier if |z| > 3
- **IQR**: Outlier if < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- **Mahalanobis distance**: Accounts for feature correlations
- **CUSUM / EWMA**: For sequential/time-series anomalies

## ✅ Choosing a Method
| Situation | Method |
|---|---|
| Low dimensional, fast | IQR / Z-score |
| High dimensional, unknown clusters | Isolation Forest |
| Local anomalies matter | LOF |
| Clear boundary needed | One-Class SVM |
| Image/sequence data | Autoencoder |
| Labeled anomaly examples available | Supervised (XGBoost, RF) |

## ⚠️ Key Challenges
- **Ground truth unknown**: Evaluating unsupervised methods is hard
- **Contamination rate**: Setting the expected anomaly fraction — miscalibrated → wrong threshold
- **Curse of dimensionality**: Distance-based methods degrade in high dimensions
- **Class imbalance**: Only 0.1–5% are anomalies in most real applications
"""

CODE_EXAMPLE = '''
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# ── Synthetic dataset with anomalies ─────────────────────────────
np.random.seed(42)
n_normal    = 300
n_anomalies = 20

# Normal data: two clusters
X_normal, _ = make_blobs(n_samples=n_normal, centers=2,
                          cluster_std=0.5, random_state=42)
# Anomalies: uniform random noise
X_anom = np.random.uniform(-6, 6, (n_anomalies, 2))
X = np.vstack([X_normal, X_anom])
y_true = np.array([1]*n_normal + [-1]*n_anomalies)  # -1 = anomaly

# ── Isolation Forest ─────────────────────────────────────────────
iso = IsolationForest(contamination=0.06, random_state=42)
iso.fit(X_normal)  # fit on normal data only
preds_iso = iso.predict(X)  # 1=normal, -1=anomaly

# ── LOF ──────────────────────────────────────────────────────────
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
preds_lof = lof.fit_predict(X)  # LOF fits + predicts together

# ── Evaluation ────────────────────────────────────────────────────
from sklearn.metrics import classification_report
print("Isolation Forest:")
print(classification_report(y_true, preds_iso, target_names=["anomaly","normal"]))

print("LOF:")
print(classification_report(y_true, preds_lof, target_names=["anomaly","normal"]))
'''


def _generate_anomaly_dataset(dataset_type: str, contamination: float, n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_normal   = int(n * (1 - contamination))
    n_anom     = n - n_normal

    if dataset_type == "Two Clusters":
        X_normal, _ = make_blobs(n_samples=n_normal, centers=2, cluster_std=0.6, random_state=seed)
        X_anom = rng.uniform(-6, 6, (n_anom, 2))
        X = np.vstack([X_normal, X_anom])
        y = np.array([1]*n_normal + [-1]*n_anom)

    elif dataset_type == "Ring (LOF advantage)":
        angles = rng.uniform(0, 2*np.pi, n_normal)
        r      = rng.normal(3, 0.3, n_normal)
        X_normal = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        X_anom   = rng.uniform(-5, 5, (n_anom, 2))
        X = np.vstack([X_normal, X_anom])
        y = np.array([1]*n_normal + [-1]*n_anom)

    elif dataset_type == "Breast Cancer (real)":
        data = load_breast_cancer()
        X_raw, y_raw = data.data, data.target
        # Malignant (0) = anomaly, benign (1) = normal
        X_norm  = X_raw[y_raw == 1][:n_normal]
        X_anom2 = X_raw[y_raw == 0][:n_anom]
        X = np.vstack([X_norm, X_anom2])
        y = np.array([1]*len(X_norm) + [-1]*len(X_anom2))

    else:  # "High Density + Sparse Anomalies"
        X_normal, _ = make_blobs(n_samples=n_normal, centers=1, cluster_std=0.5, random_state=seed)
        # Sparse anomalies in outer region
        r     = rng.uniform(3, 6, n_anom)
        theta = rng.uniform(0, 2*np.pi, n_anom)
        X_anom = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        X = np.vstack([X_normal, X_anom])
        y = np.array([1]*n_normal + [-1]*n_anom)

    return X, y


def run_anomaly_detection(dataset_type: str, method: str, contamination: float,
                          n_neighbors: int, show_scores: bool):
    X, y_true = _generate_anomaly_dataset(dataset_type, contamination)
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)

    use_2d = X_sc.shape[1] == 2
    if not use_2d:
        # For high-dim data, use top 2 PCA components for visualization
        from sklearn.decomposition import PCA
        pca   = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X_sc)
    else:
        X_vis = X_sc

    X_normal = X_sc[y_true == 1]

    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        model.fit(X_normal)
        preds  = model.predict(X_sc)
        scores = -model.decision_function(X_sc)  # higher = more anomalous

    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds  = model.fit_predict(X_sc)
        scores = -model.negative_outlier_factor_  # higher = more anomalous

    elif method == "One-Class SVM":
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="auto")
        model.fit(X_normal)
        preds  = model.predict(X_sc)
        scores = -model.decision_function(X_sc)

    else:
        preds  = np.ones(len(X_sc), dtype=int)
        scores = np.zeros(len(X_sc))

    # Evaluation
    is_anomaly_true = (y_true == -1)
    is_anomaly_pred = (preds  == -1)
    tp = int((is_anomaly_pred & is_anomaly_true).sum())
    fp = int((is_anomaly_pred & ~is_anomaly_true).sum())
    fn = int((~is_anomaly_pred & is_anomaly_true).sum())
    tn = int((~is_anomaly_pred & ~is_anomaly_true).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2*prec*rec / max(prec + rec, 1e-8)

    # Build figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Anomaly Detection (2D projection)", "Anomaly Score Distribution"])

    # Scatter: color by prediction
    colors_pred = ["#ef5350" if p == -1 else "#42a5f5" for p in preds]
    markers     = ["x" if p == -1 else "circle" for p in preds]
    sizes       = [10 if p == -1 else 5 for p in preds]

    # Normal predictions
    mask_normal = preds == 1
    fig.add_trace(go.Scatter(
        x=X_vis[mask_normal, 0], y=X_vis[mask_normal, 1],
        mode="markers", name="Predicted Normal",
        marker=dict(color="#42a5f5", size=5, opacity=0.6)
    ), row=1, col=1)

    # Anomaly predictions
    mask_anom = preds == -1
    fig.add_trace(go.Scatter(
        x=X_vis[mask_anom, 0], y=X_vis[mask_anom, 1],
        mode="markers", name="Predicted Anomaly",
        marker=dict(color="#ef5350", size=10, symbol="x",
                    line=dict(width=2, color="#c62828"))
    ), row=1, col=1)

    # Score distribution
    scores_normal = scores[y_true == 1]
    scores_anom   = scores[y_true == -1]
    fig.add_trace(go.Histogram(x=scores_normal, nbinsx=30, name="Normal scores",
                               marker_color="#42a5f5", opacity=0.6,
                               histnorm="probability"), row=1, col=2)
    fig.add_trace(go.Histogram(x=scores_anom, nbinsx=20, name="Anomaly scores",
                               marker_color="#ef5350", opacity=0.6,
                               histnorm="probability"), row=1, col=2)

    # Add threshold line on score distribution
    if len(scores_normal) > 0 and len(scores_anom) > 0:
        # Approximate threshold: midpoint between normal and anomaly score peaks
        threshold_approx = 0.5 * (np.percentile(scores_normal, 90) + np.percentile(scores_anom, 10))
        fig.add_vline(x=threshold_approx, line_dash="dash", line_color="gray",
                      annotation_text="Threshold", row=1, col=2)

    fig.update_layout(height=450, barmode="overlay",
                      title_text=f"Anomaly Detection — {method} on {dataset_type}")

    metrics_md = f"""
### Detection Results

| Metric | Value |
|---|---|
| True anomalies | `{is_anomaly_true.sum()}` ({contamination:.0%}) |
| Detected anomalies | `{is_anomaly_pred.sum()}` |
| True Positives (TP) | `{tp}` |
| False Positives (FP) | `{fp}` — normal samples flagged as anomaly |
| False Negatives (FN) | `{fn}` — missed anomalies |
| **Precision** | `{prec:.3f}` |
| **Recall** | `{rec:.3f}` |
| **F1 Score** | `{f1:.3f}` |

**Method:** {method} | **Dataset:** {dataset_type}

> **Trade-off**: High precision = few false alarms. High recall = few missed anomalies.
> In fraud detection, you typically prefer high recall (miss fewer frauds) even at cost of false alarms.
"""
    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🚨 Module 26 — Anomaly Detection\n*Level: Intermediate / Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nCompare Isolation Forest, LOF, and One-Class SVM on synthetic and real datasets. Tune contamination rate and evaluate precision/recall tradeoffs.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Two Clusters", "Ring (LOF advantage)", "High Density + Sparse Anomalies",
                         "Breast Cancer (real)"],
                value="Two Clusters"
            )
            method_dd = gr.Dropdown(
                label="Detection Method",
                choices=["Isolation Forest", "Local Outlier Factor", "One-Class SVM"],
                value="Isolation Forest"
            )
            contamination_sl = gr.Slider(label="Contamination Rate", minimum=0.02, maximum=0.20,
                                         step=0.02, value=0.06)
            k_sl = gr.Slider(label="n_neighbors (LOF)", minimum=5, maximum=50, step=5, value=20)
            scores_cb = gr.Checkbox(label="Show anomaly score distribution", value=True)
            run_btn = gr.Button("▶ Detect Anomalies", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_anomaly_detection,
        inputs=[dataset_dd, method_dd, contamination_sl, k_sl, scores_cb],
        outputs=[plot_out, metrics_out]
    )
