"""
Module 21 — MLOps & Model Monitoring
Level: Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as scipy_stats
import json, time, hashlib

THEORY = """
## 📖 What Is MLOps?

**MLOps** (Machine Learning Operations) is the practice of deploying and **maintaining ML models in production reliably and efficiently**. It borrows principles from DevOps — automation, monitoring, versioning — and applies them to the full ML lifecycle.

> "An ML model is not a product. A product is a model + a data pipeline + a monitoring system + a feedback loop."

## 🏗️ MLOps Lifecycle

```
Data → Training → Evaluation → Deploy → Monitor → Retrain → ...
```

### 1. Model Versioning
Track every model artifact with metadata:
```python
metadata = {
    "model_id":    "rf_v3_2024-03-15",
    "algorithm":   "RandomForestClassifier",
    "params":      {"n_estimators": 100, "max_depth": 10},
    "train_acc":   0.953,
    "val_acc":     0.941,
    "features":    feature_names,
    "trained_at":  "2024-03-15T14:32:00",
    "data_hash":   sha256(X_train),  # data fingerprint
}
```
Tools: MLflow, DVC, Weights & Biases, Neptune

### 2. Data Drift Detection
**Data drift** = the statistical properties of the input features change over time.

**Covariate shift**: P(X) changes, P(y|X) stays the same
**Label drift**: P(y) changes
**Concept drift**: P(y|X) changes (the underlying relationship changes)

**Detection methods:**
| Method | What it detects | When to use |
|---|---|---|
| **PSI (Population Stability Index)** | Distribution shift | Categorical + bucketed continuous |
| **KS test (Kolmogorov-Smirnov)** | Distribution shift | Continuous features |
| **Chi-squared test** | Distribution shift | Categorical features |
| **ADWIN** | Concept drift | Streaming data |

**PSI interpretation:**
- PSI < 0.1: No significant change ✅
- 0.1 ≤ PSI < 0.2: Moderate shift ⚠️ — monitor closely
- PSI ≥ 0.2: Significant shift 🔴 — retrain!

### 3. Prediction Monitoring
Track model outputs over time:
- **Prediction distribution**: Are scores shifting? (e.g., more high-confidence predictions)
- **Calibration**: Does predicted P(y=1) = 0.7 actually mean 70% of the time?
- **Latency**: Is inference time degrading?

### 4. Retraining Triggers
| Trigger | Condition |
|---|---|
| **Performance-based** | Accuracy drops below threshold (e.g., < 90%) |
| **Data-based** | PSI > 0.2 on key features |
| **Time-based** | Scheduled (weekly/monthly) |
| **Business-based** | Major event (new product launch, regulation) |

### 5. Logging & Observability
```python
import logging
logger = logging.getLogger("ml_model")

def predict(X):
    start = time.time()
    pred  = model.predict(X)
    latency = time.time() - start
    logger.info({"prediction": pred, "latency_ms": latency*1000,
                 "features": X.tolist(), "model_version": "v3"})
    return pred
```

## ⚠️ Common Pitfalls
- **Silent failures**: Model degrades slowly — no obvious error, just worse predictions
- **Retraining without validation**: Automated retraining can make things worse if new data is bad
- **No baseline**: Without a logged baseline, you can't measure drift
- **Ignoring data quality**: Upstream data pipeline failures cause apparent "model drift"
"""

CODE_EXAMPLE = '''
import numpy as np
import pandas as pd
import joblib, json, hashlib
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# ── Train and version a model ─────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model with metadata
joblib.dump(model, "model_v1.pkl")
metadata = {
    "model_id":   "rf_v1",
    "trained_at": datetime.now().isoformat(),
    "train_acc":  model.score(X, y),
    "data_hash":  hashlib.sha256(X.tobytes()).hexdigest()[:16],
    "n_features": X.shape[1],
}
json.dump(metadata, open("model_v1_metadata.json", "w"), indent=2)

# ── PSI drift detection ───────────────────────────────────────────
def compute_psi(reference, current, n_bins=10):
    """Population Stability Index."""
    ref_counts, bin_edges = np.histogram(reference, bins=n_bins)
    cur_counts, _         = np.histogram(current,   bins=bin_edges)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    psi     = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Simulate drift: add noise to production data
X_prod_nodrift   = X + np.random.normal(0, 0.1, X.shape)
X_prod_withdrift = X + np.random.normal(2, 2,   X.shape)  # drifted!

psi_no_drift   = compute_psi(X[:, 0], X_prod_nodrift[:, 0])
psi_with_drift = compute_psi(X[:, 0], X_prod_withdrift[:, 0])
print(f"PSI (no drift):   {psi_no_drift:.4f}  → {'OK' if psi_no_drift < 0.1 else 'Drift!'}")
print(f"PSI (with drift): {psi_with_drift:.4f} → {'OK' if psi_with_drift < 0.1 else 'Drift!'}")

# ── KS test ────────────────────────────────────────────────────────
ks_stat, ks_p = stats.ks_2samp(X[:, 0], X_prod_withdrift[:, 0])
print(f"KS p-value: {ks_p:.4f} → {'Drift detected!' if ks_p < 0.05 else 'No drift'}")
'''


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    ref_counts, bin_edges = np.histogram(reference, bins=n_bins)
    cur_counts, _         = np.histogram(current, bins=bin_edges)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def run_mlops_demo(demo_type: str, drift_level: float, n_features_monitor: int):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_ref = X[:300]
    X_cur_base = X[300:]

    # Simulate production drift
    rng = np.random.default_rng(42)
    noise = rng.normal(0, drift_level, X_cur_base.shape)
    X_cur = X_cur_base + noise

    if demo_type == "Data Drift Detection (PSI)":
        n_feat = min(n_features_monitor, len(feature_names))
        psi_scores = []
        ks_pvalues = []

        for i in range(n_feat):
            psi = compute_psi(X_ref[:, i], X_cur[:, i])
            _, ks_p = scipy_stats.ks_2samp(X_ref[:, i], X_cur[:, i])
            psi_scores.append(psi)
            ks_pvalues.append(ks_p)

        short_names = [f[:12] for f in feature_names[:n_feat]]
        colors_psi = ["#ef5350" if p >= 0.2 else "#ffa726" if p >= 0.1 else "#66bb6a"
                      for p in psi_scores]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["PSI by Feature", "KS Test p-value by Feature"])

        fig.add_trace(go.Bar(x=short_names, y=psi_scores, marker_color=colors_psi,
                             text=[f"{p:.3f}" for p in psi_scores],
                             textposition="outside", name="PSI"), row=1, col=1)
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                      annotation_text="⚠ 0.1", row=1, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                      annotation_text="🔴 0.2", row=1, col=1)

        ks_colors = ["#ef5350" if p < 0.05 else "#66bb6a" for p in ks_pvalues]
        fig.add_trace(go.Bar(x=short_names, y=ks_pvalues, marker_color=ks_colors,
                             text=[f"{p:.3f}" for p in ks_pvalues],
                             textposition="outside", name="KS p-value"), row=1, col=2)
        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                      annotation_text="α=0.05", row=1, col=2)

        fig.update_layout(height=420, showlegend=False,
                          title_text=f"Data Drift — drift_level={drift_level}")
        fig.update_xaxes(tickangle=-45)

        # --- Distribution overlay for top 3 most-drifted features ---
        top3_idx = np.argsort(psi_scores)[::-1][:3]
        top3_titles = [
            f"{feature_names[idx][:18]} (PSI={psi_scores[idx]:.3f})"
            for idx in top3_idx
        ]
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=top3_titles)
        for i, feat_idx in enumerate(top3_idx):
            show_legend = (i == 0)
            fig2.add_trace(
                go.Histogram(
                    x=X_ref[:, feat_idx], name="Reference",
                    opacity=0.6, marker_color="blue",
                    legendgroup="ref", showlegend=show_legend,
                ),
                row=1, col=i + 1,
            )
            fig2.add_trace(
                go.Histogram(
                    x=X_cur[:, feat_idx], name="Production",
                    opacity=0.6, marker_color="red",
                    legendgroup="prod", showlegend=show_legend,
                ),
                row=1, col=i + 1,
            )
        fig2.update_layout(
            barmode="overlay", height=350, showlegend=True,
            title_text="Distribution Overlay — Top 3 Drifted Features",
        )

        drifted_psi = sum(1 for p in psi_scores if p >= 0.2)
        drifted_ks  = sum(1 for p in ks_pvalues if p < 0.05)
        max_psi_feat = feature_names[np.argmax(psi_scores)]

        recommendation = "✅ No retrain needed" if max(psi_scores) < 0.1 else \
                         "⚠️ Monitor closely" if max(psi_scores) < 0.2 else \
                         "🔴 RETRAIN RECOMMENDED"

        # Build per-feature table with PSI and KS p-values
        feat_rows = ""
        for i in range(n_feat):
            psi_flag = "🔴" if psi_scores[i] >= 0.2 else "⚠️" if psi_scores[i] >= 0.1 else "✅"
            ks_flag = "🔴" if ks_pvalues[i] < 0.05 else "✅"
            feat_rows += (
                f"| {feature_names[i][:20]} | `{psi_scores[i]:.4f}` {psi_flag} "
                f"| `{ks_pvalues[i]:.4f}` {ks_flag} |\n"
            )

        metrics_md = f"""
### Drift Detection Results

| Metric | Value |
|---|---|
| Drift level (noise σ) | `{drift_level}` |
| Features monitored | `{n_feat}` |
| Features with PSI ≥ 0.2 (critical) | `{drifted_psi}` |
| Features with KS p < 0.05 | `{drifted_ks}` |
| Max PSI feature | `{max_psi_feat[:20]}` |

**Recommendation:** {recommendation}

### Per-Feature Drift Metrics

| Feature | PSI | KS p-value |
|---|---|---|
{feat_rows}
"""

    elif demo_type == "Model Version Comparison":
        # Train two models with different configs
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        configs = [
            {"n_estimators": 50,  "max_depth": 5,    "label": "v1 (shallow)"},
            {"n_estimators": 100, "max_depth": 10,   "label": "v2 (deeper)"},
            {"n_estimators": 200, "max_depth": None,  "label": "v3 (full depth)"},
        ]

        results = []
        for cfg in configs:
            m = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=42
            )
            t0     = time.time()
            m.fit(X_train, y_train)
            t_train = time.time() - t0

            t0    = time.time()
            preds = m.predict(X_test)
            t_inf = time.time() - t0

            results.append({
                "label":     cfg["label"],
                "acc":       accuracy_score(y_test, preds),
                "train_ms":  t_train * 1000,
                "inf_ms":    t_inf * 1000,
                "n_est":     cfg["n_estimators"],
            })

        labels  = [r["label"] for r in results]
        accs    = [r["acc"]   for r in results]
        t_train = [r["train_ms"] for r in results]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy by Version", "Training Time (ms)"])

        fig.add_trace(go.Bar(x=labels, y=accs, marker_color="#42a5f5",
                             text=[f"{a:.4f}" for a in accs], textposition="outside"), row=1, col=1)
        fig.update_yaxes(range=[0.9, 1.0], title_text="Accuracy", row=1, col=1)

        fig.add_trace(go.Bar(x=labels, y=t_train, marker_color="#ffa726",
                             text=[f"{t:.1f}" for t in t_train], textposition="outside"), row=1, col=2)
        fig.update_yaxes(title_text="ms", row=1, col=2)

        fig.update_layout(height=380, showlegend=False, title_text="Model Version Comparison")

        fig2 = None
        best = max(results, key=lambda r: r["acc"])
        metrics_md = f"""
### Model Version Registry

| Version | Accuracy | Train time | n_estimators |
|---|---|---|---|
{''.join([f"| {r['label']} | `{r['acc']:.4f}` | `{r['train_ms']:.1f}` ms | {r['n_est']} |\n" for r in results])}
**Best model:** {best['label']} (acc=`{best['acc']:.4f}`)

> In production: version models with timestamps + data hashes. Roll back to v1 if v2 underperforms in A/B testing.
"""

    elif demo_type == "Prediction Distribution Shift":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_ref, y[:300])

        probs_ref = model.predict_proba(X_ref)[:, 1]
        probs_cur_no_drift = model.predict_proba(X_cur_base)[:, 1]
        probs_cur_drift    = model.predict_proba(X_cur)[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=probs_ref, nbinsx=20, name="Reference",
                                   opacity=0.6, marker_color="#42a5f5",
                                   histnorm="probability"))
        fig.add_trace(go.Histogram(x=probs_cur_no_drift, nbinsx=20, name="Prod (no drift)",
                                   opacity=0.6, marker_color="#66bb6a",
                                   histnorm="probability"))
        fig.add_trace(go.Histogram(x=probs_cur_drift, nbinsx=20, name=f"Prod (drift σ={drift_level})",
                                   opacity=0.6, marker_color="#ef5350",
                                   histnorm="probability"))

        fig.update_layout(barmode="overlay", height=400,
                          title_text="Prediction Score Distribution Shift")

        fig2 = None
        ks_stat, ks_p = scipy_stats.ks_2samp(probs_ref, probs_cur_drift)
        metrics_md = f"""
### Prediction Distribution Analysis

| | Mean Score | Std |
|---|---|---|
| Reference | `{probs_ref.mean():.4f}` | `{probs_ref.std():.4f}` |
| Production (no drift) | `{probs_cur_no_drift.mean():.4f}` | `{probs_cur_no_drift.std():.4f}` |
| Production (drifted) | `{probs_cur_drift.mean():.4f}` | `{probs_cur_drift.std():.4f}` |

**KS test (ref vs drifted):** stat=`{ks_stat:.4f}`, p=`{ks_p:.4f}` → {'🔴 Significant shift' if ks_p < 0.05 else '✅ No significant shift'}

> Monitoring prediction distributions can detect drift *before* labels arrive (unsupervised monitoring).
"""

    else:
        fig = go.Figure()
        fig2 = None
        metrics_md = "Select a demo type."

    return fig, fig2, metrics_md


def build_tab():
    gr.Markdown("# 🚀 Module 21 — MLOps & Model Monitoring\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nSimulate production drift and monitoring on the Breast Cancer dataset. Detect when a model needs retraining.")

    with gr.Row():
        with gr.Column(scale=1):
            demo_dd = gr.Dropdown(
                label="Demo Type",
                choices=["Data Drift Detection (PSI)", "Model Version Comparison",
                         "Prediction Distribution Shift"],
                value="Data Drift Detection (PSI)"
            )
            drift_sl = gr.Slider(label="Drift Level (noise σ)", minimum=0.0, maximum=5.0,
                                 step=0.5, value=1.0)
            feat_sl  = gr.Slider(label="Features to Monitor", minimum=5, maximum=30,
                                 step=5, value=10)
            run_btn  = gr.Button("▶ Run MLOps Demo", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            plot_dist   = gr.Plot(label="Distribution Overlay (Top 3 Drifted)", visible=True)
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_mlops_demo,
        inputs=[demo_dd, drift_sl, feat_sl],
        outputs=[plot_out, plot_dist, metrics_out]
    )
