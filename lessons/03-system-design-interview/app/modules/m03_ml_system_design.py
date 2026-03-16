"""Module 03 — ML System Design & MLOps
Level: Advanced"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## 🤖 ML System Design & MLOps — A Senior Engineer's Complete Guide

ML system design is one of the most tested areas at FAANG+ and AI-native
companies. Interviewers want to see that you can take a model from a Jupyter
notebook to a production system that is reliable, observable, and continuously
improving. This module covers every layer of the production ML stack.

---

## Section 1 — End-to-End ML Pipeline

The canonical production ML pipeline is a continuous loop, not a one-shot
process. Every component has an owner, an interface, and a failure mode you
must understand.

```
Data Ingestion → Feature Engineering → Feature Store
      ↓                                      ↓
Model Training ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
      ↓
Model Evaluation → Model Registry → Deployment
                                         ↓
                              Serving (batch / real-time)
                                         ↓
                              Monitoring → Retraining Loop
```

**Who owns what?**
- **Data Engineering**: ingestion, feature pipelines, offline store
- **ML Engineering**: training code, evaluation harness, model registry, CI/CD
- **Platform/MLOps**: serving infrastructure, auto-scaling, monitoring alerts
- **Data Scientists**: feature definitions, model architecture, evaluation metrics
- **Product/Business**: business metrics, A/B test design, launch criteria

Key interviewer test: *Can you name every handoff and where things go wrong?*

---

## Section 2 — Training-Serving Skew

Training-serving skew is the **#1 bug in production ML systems**. It occurs
when the features computed during training differ — even subtly — from those
computed at serving time. The model learns from one distribution but predicts
on another.

**Root cause:** Feature code duplicated between the training pipeline (batch,
SQL, Spark) and serving code (Python, Java, real-time microservice). Two
engineers maintain two implementations of "the same" logic. They diverge.

**Classic example:** A fraud model uses `average_purchase_last_30_days`.

- *Training (batch):* Computed in Spark over historical transactions in S3.
  Null purchases → `0.0`. Weekends included. Timezone UTC.
- *Serving (real-time):* Computed by a Python microservice querying Redis.
  Null purchases → `None` (cast to `NaN`). Weekends excluded by accident.
  Timezone local server time.

Result: The feature distribution at serving time is completely different from
what the model trained on. The model silently fails.

**Real incident:** "We had a model that was 97% accurate offline but only 68%
accurate in production. After two weeks of debugging we discovered the feature
code handled null values differently in our Spark training job vs our Java
serving service. The fix was a one-line change. The cost was two weeks of bad
predictions."

**Solution: Feature Store** — define features once, use everywhere. The same
feature definition is materialized into the offline store (for training) and
the online store (for serving). No duplication, no divergence.

---

## Section 3 — Feature Store Architecture

A feature store is the data layer of an ML platform. It eliminates
training-serving skew and enables feature reuse across models and teams.

### Offline Store
- **Technology:** BigQuery, Snowflake, Amazon S3 + Glue, Delta Lake
- **Purpose:** Training dataset creation, batch scoring, feature backfills
- **Access pattern:** Bulk reads (millions of rows), SQL-style queries
- **Critical property:** Point-in-time correctness — when you create a
  training dataset for a model trained on January 15th, you must only use
  feature values that were known before January 15th. Using future data
  causes data leakage and inflated offline metrics.

### Online Store
- **Technology:** Redis, DynamoDB, Cassandra, Bigtable
- **Purpose:** Real-time feature serving for online inference
- **Access pattern:** Single-entity lookups in < 10ms
- **Critical property:** Low latency — fraud models, search ranking, and
  recommendation systems cannot afford slow feature lookups.

### Feature Registry
- Central catalog of feature definitions, lineage, and ownership
- "What features exist?" "Who computes `user_lifetime_value`?" "When was it
  last updated?" → answered by the registry

### Build vs Buy

| Platform | Type | Key Strength |
|---|---|---|
| **Feast** | Open-source | Flexible, self-hosted, community |
| **Tecton** | Enterprise SaaS | Production-grade, real-time streaming features |
| **Vertex AI Feature Store** | GCP managed | Native GCP integration |
| **SageMaker Feature Store** | AWS managed | Native AWS integration, no ops |
| **Databricks Feature Store** | Databricks | Unity Catalog integration, Delta Lake |

---

## Section 4 — Deployment Strategies

Choosing the right deployment strategy is a key interview signal. The answer
depends on latency requirements, risk tolerance, and traffic volume.

| Strategy | Description | Latency | Use Case |
|---|---|---|---|
| **Batch inference** | Score all entities nightly, write to DB | Hours | Recommendations, risk scores, churn prediction |
| **Real-time inference** | Synchronous scoring on request | < 100ms | Fraud detection, search ranking, ad bidding |
| **Shadow mode** | New model runs alongside, results NOT served | — | Safe validation before any production impact |
| **Canary deployment** | Route 5–10% of traffic to new model | — | Gradual rollout with blast radius control |
| **Blue-green** | Two full environments, instant cutover | — | Zero-downtime deployment with instant rollback |

**Shadow mode** is often overlooked by candidates but beloved by interviewers.
It lets you validate a new model on 100% of real production traffic with zero
user impact. Only after shadow metrics satisfy guardrails do you advance to
canary.

---

## Section 5 — A/B Testing for Models

A/B testing is how you answer "is the new model actually better?" with
statistical rigor, not gut feeling.

**Null hypothesis (H₀):** The new model performs the same as the baseline
(e.g., click-through rate is equal between variants).

**Key design decisions:**

1. **Minimum Detectable Effect (MDE):** How large an improvement is worth
   detecting? A 0.1% CTR improvement may not be worth the complexity of
   detecting it. A 2% improvement in fraud block rate has massive ROI.

2. **Power analysis:** Use MDE + alpha (Type I error rate, typically 0.05) +
   power (1 − β, typically 0.80) to calculate required sample size per
   variant. Never run an underpowered test — you'll miss real improvements.

3. **Bayesian vs Frequentist:**
   - *Frequentist:* Fixed sample size decided upfront; p-value compared to α.
     Cannot stop early without inflating Type I error. Easy to audit.
   - *Bayesian:* Update posterior as data arrives; can stop when P(B > A) > 95%.
     More flexible but harder to explain to non-technical stakeholders.

4. **Guardrail metrics:** Ensure the test doesn't harm other business metrics.
   A fraud model improvement that increases false positives and tanks user
   engagement is not a win. Always define guardrails before launching.

5. **Network effects & interference:** In recommendation systems, A and B
   users may interact (one user's engagement changes inventory for another).
   Use cluster-based or switchback experiments in these cases.

---

## Section 6 — Model Monitoring

Models degrade silently. Without monitoring, you often find out from an angry
stakeholder, not a pager. There are four distinct failure modes to instrument:

### Data Drift (Covariate Shift)
The input feature distribution P(X) changes over time. The model's learned
decision boundary becomes misaligned with the new data.
- **Detection:** Population Stability Index (PSI) on key features
- **Threshold:** PSI > 0.10 = warning; PSI > 0.20 = retrain
- **Example:** A retail model trained pre-COVID sees completely different
  purchase patterns in March 2020. PSI on `basket_size` spikes to 0.8.

### Concept Drift
The relationship P(y|X) changes. Even if inputs look the same, the correct
label has changed. This is the hardest drift to detect without labels.
- **Detection:** Requires delayed ground truth; compare model accuracy over
  cohorts. Proxy signals: upstream domain changes, major world events.
- **Example:** A credit risk model trained before a recession. The same
  features now predict higher default rates.

### Prediction Drift
The model's output score distribution shifts — more predictions near 0.9 than
before. Catch this before ground truth labels arrive (they're often delayed).
- **Detection:** KS-test or PSI on score distribution vs baseline
- **Advantage:** No label delay — detectable immediately

### Performance Degradation
AUC, precision, recall drop vs the deployment baseline. Requires labels.
- **Detection:** Compare rolling AUC on recent labeled cohort vs launch metric
- **Alerting:** > 5% relative drop → page oncall; > 10% → rollback

---

## Section 7 — CI/CD for ML (MLOps)

**Retraining triggers** (know all four, interviewers love this):
- **Performance-based:** Accuracy drops below threshold (e.g., AUC < 0.90)
- **Data-based:** PSI > 0.20 on key features triggers automated pipeline run
- **Time-based:** Scheduled weekly/monthly retraining (simplest, always safe)
- **Business-based:** Major event (new product launch, regulation change,
  market shock) → manual trigger or code-gate in pipeline

**Cloud MLOps stacks:**

| Platform | Pipelines | Feature Store | Model Registry | Key Differentiator |
|---|---|---|---|---|
| **AWS SageMaker** | SageMaker Pipelines | SageMaker Feature Store | SageMaker Model Registry | Deepest AWS integration; managed everything |
| **Azure ML** | AzureML Pipelines | (via AzureML) | MLflow (workspace = server) | Best for enterprises already on Azure/Teams |
| **GCP Vertex AI** | Vertex Pipelines ($0.03/run) | Vertex Feature Store | Vertex Model Registry | Cheapest managed option; Kubeflow-compatible |
| **Databricks** | Delta Live Tables + MLflow | Databricks Feature Store | Unity Catalog | Best for data-heavy orgs already on Spark |
| **Snowflake ML** | Snowpark ML + tasks | (via Snowpark) | Snowpark Container Services | Best when all data already lives in Snowflake |

---

## Interview Playbook

**"Design a recommendation system for Netflix":**
Candidate generation (ANN search over user/item embeddings) → ranking (deep
learning model with session + historical features from feature store) →
re-ranking (business rules: diversity, freshness, licensing) → A/B testing
framework for algorithm changes + feature store for reuse across surfaces.

**"How would you monitor a fraud model?":**
PSI on transaction features (amount, merchant category, velocity) + prediction
drift on score distribution + delayed ground truth comparison against
chargebacks + guardrail metric on false positive rate to protect user
experience.

**Red flags (what NOT to say):**
- "I'd train a model and deploy it" (no monitoring, no feature store)
- "We'd retrain monthly on a schedule" with no performance-based trigger
- Confusing data drift with concept drift

**Green flags (what signals seniority):**
- Mentioning training-serving skew unprompted
- Point-in-time correctness for feature store training data
- Shadow mode before canary before full rollout
- Bayesian A/B testing for faster iteration without inflating error rates
- Discussing delayed ground truth and how it complicates monitoring
"""

# ─────────────────────────────────────────────────────────────────────────────
# CODE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

CODE_EXAMPLE = '''# MLOps Pipeline — Production Patterns
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import json
from datetime import datetime


# ── Feature Store Simulation ──────────────────────────────────────────────────
class SimpleFeatureStore:
    """Simulates offline (batch) + online (real-time) feature store.

    Key insight: define features ONCE, materialize to both stores.
    Eliminates training-serving skew by construction.
    """

    def __init__(self):
        self._offline = {}   # { entity_id: [{features, timestamp}, ...] }
        self._online  = {}   # { entity_id: {features} }   ← latest snapshot

    def write_batch_features(self, entity_id: str, features: dict, timestamp: str):
        """Write historical features for training (offline store)."""
        self._offline.setdefault(entity_id, []).append(
            {"features": features, "timestamp": timestamp}
        )

    def write_online_features(self, entity_id: str, features: dict):
        """Write real-time features for serving (online store, < 10ms)."""
        self._online[entity_id] = features

    def get_training_data(self, entity_ids: list, as_of_date: str) -> pd.DataFrame:
        """Point-in-time correct feature retrieval — NO data leakage!

        Only returns feature values known BEFORE as_of_date.
        Critical for avoiding inflated offline metrics.
        """
        rows = []
        for eid in entity_ids:
            if eid in self._offline:
                # Only use features whose timestamp <= as_of_date
                valid = [r for r in self._offline[eid]
                         if r["timestamp"] <= as_of_date]
                if valid:
                    rows.append({"entity_id": eid, **valid[-1]["features"]})
        return pd.DataFrame(rows)

    def get_online_features(self, entity_id: str) -> dict:
        """Real-time serving — equivalent to a Redis GET."""
        return self._online.get(entity_id, {})


# ── PSI Drift Detection ───────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """Population Stability Index.

    PSI < 0.1  → no significant drift (green)
    PSI < 0.2  → moderate drift, investigate (amber)
    PSI >= 0.2 → significant drift, retrain (red)
    """
    ref_counts, bins = np.histogram(reference, bins=n_bins)
    cur_counts, _    = np.histogram(current,   bins=bins)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ── A/B Test Power Analysis ───────────────────────────────────────────────────
def ab_test_sample_size(
    baseline_rate: float,
    mde: float,           # minimum detectable effect (absolute)
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum sample size per variant for a two-sided z-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p1, p2  = baseline_rate, baseline_rate + mde
    pooled  = (p1 + p2) / 2
    n = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (mde ** 2)
    return int(np.ceil(n))


# ── Model Registry Entry ──────────────────────────────────────────────────────
model_metadata = {
    "model_id":     f"gbm_fraud_v2_{datetime.now().strftime('%Y%m%d')}",
    "algorithm":    "GradientBoostingClassifier",
    "auc_val":      0.945,
    "psi_train_val": 0.03,   # low PSI → no train/val skew
    "trained_at":   datetime.now().isoformat(),
    "feature_list": ["amount", "hour", "merchant_category", "user_age_days"],
    "status":       "staging",   # staging → production → deprecated
    "retraining_trigger": "performance",   # PSI > 0.2 OR AUC < 0.90
}
print(json.dumps(model_metadata, indent=2))

# ── Full Mini Pipeline ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
store = SimpleFeatureStore()

# Populate offline store with historical feature snapshots
for uid in [f"user_{i}" for i in range(200)]:
    store.write_batch_features(uid,
        {"amount": rng.exponential(50),
         "hour": int(rng.integers(0, 24)),
         "merchant_category": int(rng.integers(0, 10)),
         "user_age_days": int(rng.integers(30, 1000))},
        timestamp="2024-01-14",   # day BEFORE training cutoff
    )
    store.write_online_features(uid,    # real-time snapshot
        {"amount": rng.exponential(50),
         "hour": int(rng.integers(0, 24)),
         "merchant_category": int(rng.integers(0, 10)),
         "user_age_days": int(rng.integers(30, 1000))})

# Point-in-time correct training dataset (as_of 2024-01-15)
train_df = store.get_training_data(
    [f"user_{i}" for i in range(200)], as_of_date="2024-01-15"
)
labels = rng.integers(0, 2, size=len(train_df))
X = train_df[["amount", "hour", "merchant_category", "user_age_days"]].values

model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X, labels)

# Simulate drift: serving distribution shifts
ref_scores = model.predict_proba(X)[:, 1]
drift_data  = X.copy(); drift_data[:, 0] *= 3   # amount 3x → drift
new_scores  = model.predict_proba(drift_data)[:, 1]

print(f"Score PSI (no drift):  {compute_psi(ref_scores, ref_scores):.4f}")
print(f"Score PSI (with drift): {compute_psi(ref_scores, new_scores):.4f}")
print(f"A/B sample size (2% MDE): {ab_test_sample_size(0.10, 0.02):,}")
'''

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Feature Store Sub-tab
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_CONFIGS = {
    "fraud": {
        "features": ["amount", "hour_of_day", "merchant_category", "velocity_1h", "account_age_days"],
        "training_base": [52.3, 14.2, 3.1, 1.8, 420.0],
        "drift_factors": [2.5, 0.0, 0.8, 3.2, 0.0],
    },
    "recommendation": {
        "features": ["watch_time_30d", "genre_affinity", "recency_score", "session_depth", "catalog_coverage"],
        "training_base": [8.4, 0.62, 0.71, 3.2, 0.18],
        "drift_factors": [1.3, 0.15, 0.5, 1.8, 0.06],
    },
    "search": {
        "features": ["query_length", "click_position", "dwell_time_s", "reformulation_rate", "freshness_score"],
        "training_base": [3.1, 2.4, 45.2, 0.22, 0.88],
        "drift_factors": [0.4, 1.1, 15.0, 0.18, 0.12],
    },
}


def _feature_store_demo(dataset_type: str, training_date: str, serving_lag_hours: float):
    try:
        cfg = _DATASET_CONFIGS.get(dataset_type, _DATASET_CONFIGS["fraud"])
        features = cfg["features"]
        base = np.array(cfg["training_base"])
        drift = np.array(cfg["drift_factors"])

        # Serving values shift proportionally to lag
        lag_scale = serving_lag_hours / 24.0
        serving = base + drift * lag_scale

        # Build comparison dataframe
        skew = np.abs((serving - base) / (np.abs(base) + 1e-9)) * 100
        df = pd.DataFrame({
            "Feature": features,
            "Training Value": [f"{v:.3f}" for v in base],
            "Serving Value":  [f"{v:.3f}" for v in serving],
            "Skew %":         skew,
        })

        # Chart: grouped bar
        bar_colors_train   = COLORS["primary"]
        bar_colors_serving = [COLORS["danger"] if s > 15 else
                              COLORS["warning"] if s > 5 else
                              COLORS["success"] for s in skew]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Training (offline)", x=features, y=base,
            marker_color=bar_colors_train, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Serving (online)", x=features, y=serving,
            marker_color=bar_colors_serving, opacity=0.85,
        ))
        fig.update_layout(
            title=f"Training vs Serving Feature Values — {dataset_type.capitalize()} "
                  f"(lag = {serving_lag_hours:.0f}h)",
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Feature Value",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420,
        )

        # Metrics markdown
        max_skew_idx = int(np.argmax(skew))
        severity = "CRITICAL" if skew.max() > 15 else "WARNING" if skew.max() > 5 else "OK"
        sev_emoji = "🔴" if severity == "CRITICAL" else "🟡" if severity == "WARNING" else "🟢"
        md = f"""### Feature Store Analysis — {dataset_type.capitalize()} Model

| Metric | Value |
|--------|-------|
| Training date | `{training_date}` |
| Serving lag | `{serving_lag_hours:.0f} hours` |
| Max skew feature | `{features[max_skew_idx]}` ({skew[max_skew_idx]:.1f}%) |
| Skew severity | {sev_emoji} **{severity}** |
| Features at risk (>5% skew) | `{int((skew > 5).sum())} / {len(features)}` |

{"> ⚠️ **Training-serving skew detected.** Unify feature computation in a feature store." if severity != "OK" else "> ✅ Feature values are consistent between training and serving."}

**Root cause pattern:** A serving lag of {serving_lag_hours:.0f}h means real-time features
(e.g., `{features[max_skew_idx]}`) have evolved since training snapshot.
Solution: point-in-time correct feature retrieval from an offline store for
training; online store (Redis/DynamoDB) mirrors same logic for serving.
"""
        return fig, md

    except Exception as e:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Drift Detection Sub-tab
# ─────────────────────────────────────────────────────────────────────────────

def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    ref_counts, bins = np.histogram(reference, bins=n_bins)
    cur_counts, _    = np.histogram(current, bins=bins)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _drift_detection_demo(drift_level: float, n_features: int):
    try:
        rng = np.random.default_rng(42)
        n_features = int(n_features)
        feature_names = [f"feature_{i+1:02d}" for i in range(n_features)]

        # Generate reference distributions
        ref_data = {f: rng.normal(loc=rng.uniform(0, 5), scale=rng.uniform(0.5, 2),
                                   size=2000) for f in feature_names}

        # Generate drifted distributions — drift level controls how much shift
        psi_values = []
        cur_data = {}
        for i, f in enumerate(feature_names):
            # First few features drift more; later ones stay stable
            feature_drift = drift_level * (1 - i / n_features) * rng.uniform(0.5, 1.5)
            shift = feature_drift * np.std(ref_data[f])
            scale_mult = 1.0 + feature_drift * 0.3
            cur_data[f] = rng.normal(
                loc=np.mean(ref_data[f]) + shift,
                scale=np.std(ref_data[f]) * scale_mult,
                size=2000,
            )
            psi_values.append(_compute_psi(ref_data[f], cur_data[f]))

        psi_arr = np.array(psi_values)
        feature_names_sorted = [x for _, x in sorted(zip(psi_arr, feature_names), reverse=True)]
        psi_sorted = np.sort(psi_arr)[::-1]

        bar_colors = [
            COLORS["danger"]  if p > 0.2 else
            COLORS["warning"] if p > 0.1 else
            COLORS["success"]
            for p in psi_sorted
        ]

        # Top 3 drifted features → distribution overlay
        top3 = feature_names_sorted[:3]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                ["PSI Bar Chart (all features)", "", ""] +
                [f"Distribution: {t}" for t in top3]
            ),
            specs=[[{"colspan": 3}, None, None],
                   [{}, {}, {}]],
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )

        # PSI bar chart (row 1, col 1 spanning 3)
        fig.add_trace(
            go.Bar(
                x=feature_names_sorted, y=psi_sorted,
                marker_color=bar_colors,
                name="PSI",
                showlegend=False,
            ),
            row=1, col=1,
        )
        # Threshold lines
        for threshold, label, color in [(0.1, "Warning (0.1)", COLORS["warning"]),
                                         (0.2, "Critical (0.2)", COLORS["danger"])]:
            fig.add_hline(y=threshold, line_dash="dash", line_color=color,
                          annotation_text=label, annotation_position="top right",
                          row=1, col=1)

        # Distribution overlays (row 2)
        for col_idx, feat in enumerate(top3, start=1):
            bins = np.linspace(
                min(ref_data[feat].min(), cur_data[feat].min()),
                max(ref_data[feat].max(), cur_data[feat].max()),
                40,
            )
            fig.add_trace(go.Histogram(
                x=ref_data[feat], xbins=dict(start=bins[0], end=bins[-1],
                                              size=(bins[-1]-bins[0])/40),
                name="Reference", marker_color=COLORS["primary"],
                opacity=0.6, histnorm="probability density",
                showlegend=(col_idx == 1),
            ), row=2, col=col_idx)
            fig.add_trace(go.Histogram(
                x=cur_data[feat], xbins=dict(start=bins[0], end=bins[-1],
                                              size=(bins[-1]-bins[0])/40),
                name="Current", marker_color=COLORS["danger"],
                opacity=0.6, histnorm="probability density",
                showlegend=(col_idx == 1),
            ), row=2, col=col_idx)

        fig.update_layout(
            title=f"Drift Detection Dashboard — Drift Level {drift_level:.1f}",
            template="plotly_dark",
            height=600,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )

        # Summary stats
        n_critical = int((psi_arr > 0.2).sum())
        n_warning  = int(((psi_arr > 0.1) & (psi_arr <= 0.2)).sum())
        n_ok       = int((psi_arr <= 0.1).sum())
        action = (
            "**🔴 RETRAIN NOW** — multiple features show critical drift."
            if n_critical >= 2 else
            "**🔴 RETRAIN** — at least one feature exceeds critical threshold."
            if n_critical >= 1 else
            "**🟡 INVESTIGATE** — moderate drift detected, monitor closely."
            if n_warning >= 1 else
            "**🟢 STABLE** — no significant drift detected."
        )

        md = f"""### Drift Detection Summary

| Status | Count |
|--------|-------|
| 🟢 Stable (PSI ≤ 0.1) | {n_ok} features |
| 🟡 Warning (0.1 < PSI ≤ 0.2) | {n_warning} features |
| 🔴 Critical (PSI > 0.2) | {n_critical} features |
| Highest PSI | `{feature_names_sorted[0]}` = **{psi_sorted[0]:.4f}** |

**Recommended action:** {action}

**PSI interpretation:**
- PSI measures how much a distribution has shifted between reference and current windows.
- Formula: `Σ (P_cur − P_ref) × ln(P_cur / P_ref)` over histogram bins.
- Typical triggers: PSI > 0.1 → alert; PSI > 0.2 → block serving + trigger retraining pipeline.
"""
        return fig, md

    except Exception as e:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — A/B Test Calculator Sub-tab
# ─────────────────────────────────────────────────────────────────────────────

def _ab_test_sample_size(baseline: float, mde: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    import scipy.stats as stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p1, p2  = baseline, baseline + mde
    pooled  = (p1 + p2) / 2
    n = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2 / (mde ** 2)
    return int(np.ceil(n))


def _ab_test_demo(baseline_conversion: float, mde_relative_pct: float,
                   alpha_str: str, daily_traffic: int):
    try:
        import scipy.stats as stats

        alpha = float(alpha_str)
        mde_abs = baseline_conversion * (mde_relative_pct / 100.0)
        n_per_variant = _ab_test_sample_size(baseline_conversion, mde_abs, alpha=alpha)
        total_n = 2 * n_per_variant
        days_to_sig = max(1, int(np.ceil(total_n / max(daily_traffic, 1))))

        # Power curve — vary MDE relative % from 0.5% to 30%
        mde_range_pct = np.linspace(0.5, 30, 80)
        sample_sizes  = [
            _ab_test_sample_size(baseline_conversion, baseline_conversion * m / 100, alpha=alpha)
            for m in mde_range_pct
        ]
        days_range = [max(1, int(np.ceil(2 * s / max(daily_traffic, 1)))) for s in sample_sizes]

        # Power curve vs sample size for several power levels
        power_levels = [0.70, 0.80, 0.90, 0.95]
        power_colors = [COLORS["warning"], COLORS["primary"], COLORS["success"], COLORS["info"]]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Sample Size vs Min Detectable Effect",
                "Days to Significance vs Daily Traffic",
            ],
            horizontal_spacing=0.12,
        )

        for pwr, col in zip(power_levels, power_colors):
            ss = [_ab_test_sample_size(baseline_conversion,
                                        baseline_conversion * m / 100, alpha=alpha, power=pwr)
                  for m in mde_range_pct]
            fig.add_trace(go.Scatter(
                x=mde_range_pct, y=ss,
                name=f"Power={int(pwr*100)}%",
                line=dict(color=col, width=2),
                mode="lines",
            ), row=1, col=1)

        # Highlight selected MDE
        fig.add_vline(x=mde_relative_pct, line_dash="dash",
                      line_color=COLORS["danger"], row=1, col=1)

        # Days to significance vs daily traffic (range 100–100k)
        traffic_range = np.logspace(2, 5, 80)
        days_vs_traffic = [max(1, int(np.ceil(total_n / t))) for t in traffic_range]
        fig.add_trace(go.Scatter(
            x=traffic_range, y=days_vs_traffic,
            line=dict(color=COLORS["primary"], width=2),
            name="Days to significance",
            showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=daily_traffic, line_dash="dash",
                      line_color=COLORS["danger"], row=1, col=2)
        fig.add_hline(y=days_to_sig, line_dash="dot",
                      line_color=COLORS["warning"], row=1, col=2)

        fig.update_xaxes(title_text="MDE (relative %)", row=1, col=1)
        fig.update_yaxes(title_text="Sample size per variant", row=1, col=1)
        fig.update_xaxes(title_text="Daily traffic", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Days", row=1, col=2)
        fig.update_layout(
            title=f"A/B Test Power Analysis — Baseline={baseline_conversion:.1%}, "
                  f"α={alpha}, MDE={mde_relative_pct:.1f}%",
            template="plotly_dark",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )

        md = f"""### A/B Test Calculator Results

| Parameter | Value |
|-----------|-------|
| Baseline conversion rate | `{baseline_conversion:.2%}` |
| Minimum detectable effect | `{mde_relative_pct:.1f}%` relative → `{mde_abs:.4f}` absolute |
| Significance level (α) | `{alpha}` |
| Statistical power (1−β) | `80%` |
| **Sample size per variant** | **{n_per_variant:,}** |
| **Total sample size** | **{total_n:,}** |
| **Days to significance** | **{days_to_sig} days** @ {daily_traffic:,}/day traffic |

**Interpretation:**
You need **{n_per_variant:,} users per variant** to detect a {mde_relative_pct:.1f}% relative
improvement in conversion rate (from {baseline_conversion:.2%} to {baseline_conversion + mde_abs:.2%})
with 80% power at α={alpha}.

At {daily_traffic:,} daily users split evenly, the test will reach significance in
approximately **{days_to_sig} days**.

> **Tip:** Before launching, define your guardrail metrics (latency, engagement,
> revenue per user). An improvement in one metric that harms guardrails is **not**
> a winner.
"""
        return fig, md

    except Exception as e:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Cloud MLOps Comparison Sub-tab
# ─────────────────────────────────────────────────────────────────────────────

_PLATFORMS = [
    "AWS SageMaker",
    "Azure ML",
    "GCP Vertex AI",
    "Databricks",
    "Snowflake ML",
]

_CAPABILITIES = [
    "Real-time serving (<100ms)",
    "Built-in feature store",
    "AutoML",
    "MLflow compatibility",
    "Cost-sensitive",
    "AWS ecosystem",
    "GCP ecosystem",
    "Azure ecosystem",
]

# Scores 0–3 per (capability, platform)
# Rows = capabilities, cols = platforms
_SCORES = np.array([
    # SageMaker  AzureML  VertexAI  Databricks  Snowflake
    [3,          2,        3,         2,           1],   # Real-time serving
    [3,          2,        3,         3,           1],   # Built-in feature store
    [3,          3,        3,         1,           2],   # AutoML
    [1,          3,        1,         3,           1],   # MLflow compatibility
    [1,          1,        3,         2,           2],   # Cost-sensitive
    [3,          1,        1,         2,           1],   # AWS ecosystem
    [1,          1,        3,         2,           1],   # GCP ecosystem
    [1,          3,        1,         1,           1],   # Azure ecosystem
], dtype=float)

_REQUIREMENT_WEIGHTS = {
    "Real-time serving (<100ms)":  [3, 2, 3, 2, 1],
    "Built-in feature store":      [3, 2, 3, 3, 1],
    "AutoML":                      [3, 3, 3, 1, 2],
    "MLflow compatibility":        [1, 3, 1, 3, 1],
    "Cost-sensitive":              [1, 1, 3, 2, 2],
    "AWS ecosystem":               [3, 1, 1, 2, 1],
    "GCP ecosystem":               [1, 1, 3, 2, 1],
    "Azure ecosystem":             [1, 3, 1, 1, 1],
}


def _cloud_mlops_demo(requirements: list):
    try:
        if not requirements:
            requirements = _CAPABILITIES[:3]

        # Compute weighted scores per platform
        weights = np.zeros(len(_PLATFORMS))
        for req in requirements:
            if req in _REQUIREMENT_WEIGHTS:
                weights += np.array(_REQUIREMENT_WEIGHTS[req])

        # Normalise to 0–100
        max_w = weights.max() if weights.max() > 0 else 1.0
        normalised = (weights / max_w) * 100

        # Build heatmap matrix: rows = selected requirements, cols = platforms
        selected_idx = [_CAPABILITIES.index(r) for r in requirements if r in _CAPABILITIES]
        heatmap_data = _SCORES[selected_idx, :] if selected_idx else _SCORES

        label_map = ["None", "Partial", "Good", "Excellent"]
        text_matrix = [[label_map[int(v)] for v in row] for row in heatmap_data]
        row_labels   = [_CAPABILITIES[i] for i in selected_idx] if selected_idx else _CAPABILITIES

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Capability Heatmap", "Overall Fit Score"],
            vertical_spacing=0.18,
            row_heights=[0.65, 0.35],
        )

        # Heatmap
        fig.add_trace(go.Heatmap(
            z=heatmap_data,
            x=_PLATFORMS,
            y=row_labels,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#374151"],
                [0.33, COLORS["warning"]],
                [0.67, COLORS["info"]],
                [1.0,  COLORS["success"]],
            ],
            zmin=0, zmax=3,
            showscale=True,
            colorbar=dict(
                title="Score",
                tickvals=[0, 1, 2, 3],
                ticktext=["None", "Partial", "Good", "Excellent"],
                len=0.55, y=0.72,
            ),
        ), row=1, col=1)

        # Bar chart of overall fit
        bar_colors = [
            COLORS["success"] if s == normalised.max() else COLORS["primary"]
            for s in normalised
        ]
        fig.add_trace(go.Bar(
            x=_PLATFORMS, y=normalised,
            marker_color=bar_colors,
            text=[f"{v:.0f}%" for v in normalised],
            textposition="outside",
            showlegend=False,
        ), row=2, col=1)

        fig.update_yaxes(title_text="Fit Score (%)", row=2, col=1)
        fig.update_layout(
            title="Cloud MLOps Platform Comparison",
            template="plotly_dark",
            height=680,
        )

        best_idx    = int(np.argmax(normalised))
        best_platform = _PLATFORMS[best_idx]
        best_score    = normalised[best_idx]

        # Platform rationale
        rationale = {
            "AWS SageMaker":  "Deep AWS integration, managed endpoints, SageMaker Pipelines for orchestration. Best when your data already lives in S3/Redshift.",
            "Azure ML":       "Best-in-class MLflow support, Azure DevOps CI/CD, native Azure AD security. Ideal for enterprises already on Microsoft stack.",
            "GCP Vertex AI":  "Cheapest managed pipeline runs ($0.03/run), strong Vertex Feature Store, Kubeflow-compatible. Best value on GCP.",
            "Databricks":     "Unmatched for Spark-heavy data pipelines, MLflow autologging, Unity Catalog lineage. Best for data-intensive ML workloads.",
            "Snowflake ML":   "Best when all data lives in Snowflake. Snowpark ML avoids data movement. Feature store via Snowpark. Inference via container services.",
        }

        rows_md = "\n".join(
            f"| {p} | {normalised[i]:.0f}% | {'⭐ Best fit' if i == best_idx else ''} |"
            for i, p in enumerate(_PLATFORMS)
        )

        md = f"""### Cloud MLOps Platform Recommendation

**Selected requirements:** {", ".join(f"`{r}`" for r in requirements) if requirements else "None selected"}

| Platform | Fit Score | |
|----------|-----------|---|
{rows_md}

**Recommended platform: {best_platform}** ({best_score:.0f}% fit score)

*Why:* {rationale.get(best_platform, "")}

**Decision framework:**
1. Start with your existing cloud — switching costs are high.
2. If you need sub-100ms real-time serving, rule out batch-only stacks.
3. If MLflow is your standard, Azure ML and Databricks lead.
4. If cost is paramount, Vertex AI Pipelines are cheapest per run.
5. If your data never leaves Snowflake, Snowpark ML avoids ETL overhead.
"""
        return fig, md

    except Exception as e:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown("# 🤖 Module 03 — ML System Design & MLOps\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)

    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demos")

    with gr.Tabs():

        # ── Sub-tab 1: Feature Store Simulation ──────────────────────────────
        with gr.Tab("Feature Store"):
            gr.Markdown(
                "Visualise **training-serving skew** — how much feature values drift "
                "between training snapshot and real-time serving as lag increases."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    fs_dataset = gr.Dropdown(
                        choices=["fraud", "recommendation", "search"],
                        value="fraud",
                        label="Dataset / Domain",
                    )
                    fs_date = gr.Textbox(
                        value="2024-06-01",
                        label="Training Date (YYYY-MM-DD)",
                        placeholder="2024-06-01",
                    )
                    fs_lag = gr.Slider(
                        minimum=0, maximum=72, step=1, value=0,
                        label="Serving Lag (hours)",
                        info="Hours between training snapshot and real-time serving",
                    )
                    fs_btn = gr.Button("Run Feature Store Demo", variant="primary")
                with gr.Column(scale=3):
                    fs_plot = gr.Plot(label="Feature Value Comparison")
                    fs_md   = gr.Markdown()

            fs_btn.click(
                fn=_feature_store_demo,
                inputs=[fs_dataset, fs_date, fs_lag],
                outputs=[fs_plot, fs_md],
            )

        # ── Sub-tab 2: Drift Detection Dashboard ─────────────────────────────
        with gr.Tab("Drift Detection"):
            gr.Markdown(
                "Simulate **data drift** across multiple features and visualise PSI "
                "scores. Green < 0.1, Amber 0.1–0.2, Red > 0.2 (retrain)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    dd_drift = gr.Slider(
                        minimum=0.0, maximum=3.0, step=0.1, value=1.0,
                        label="Drift Level",
                        info="0 = no drift, 3 = severe drift",
                    )
                    dd_nfeat = gr.Slider(
                        minimum=5, maximum=20, step=1, value=10,
                        label="Number of Features",
                    )
                    dd_btn = gr.Button("Run Drift Detection", variant="primary")
                with gr.Column(scale=3):
                    dd_plot = gr.Plot(label="Drift Dashboard")
                    dd_md   = gr.Markdown()

            dd_btn.click(
                fn=_drift_detection_demo,
                inputs=[dd_drift, dd_nfeat],
                outputs=[dd_plot, dd_md],
            )

        # ── Sub-tab 3: A/B Test Calculator ───────────────────────────────────
        with gr.Tab("A/B Test Calculator"):
            gr.Markdown(
                "Calculate required **sample size** and **days to significance** "
                "for a model A/B test given your baseline conversion and MDE."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ab_baseline = gr.Slider(
                        minimum=0.01, maximum=0.30, step=0.005, value=0.10,
                        label="Baseline Conversion Rate",
                        info="Current model conversion / click-through / fraud rate",
                    )
                    ab_mde = gr.Slider(
                        minimum=0.5, maximum=20.0, step=0.5, value=5.0,
                        label="Minimum Detectable Effect (% relative)",
                        info="e.g. 5% means detecting 10% → 10.5%",
                    )
                    ab_alpha = gr.Dropdown(
                        choices=["0.01", "0.05", "0.10"],
                        value="0.05",
                        label="Significance Level (α)",
                    )
                    ab_traffic = gr.Slider(
                        minimum=100, maximum=100_000, step=100, value=10_000,
                        label="Daily Traffic (total users)",
                    )
                    ab_btn = gr.Button("Calculate Sample Size", variant="primary")
                with gr.Column(scale=3):
                    ab_plot = gr.Plot(label="Power Curves")
                    ab_md   = gr.Markdown()

            ab_btn.click(
                fn=_ab_test_demo,
                inputs=[ab_baseline, ab_mde, ab_alpha, ab_traffic],
                outputs=[ab_plot, ab_md],
            )

        # ── Sub-tab 4: Cloud MLOps Comparison ────────────────────────────────
        with gr.Tab("Cloud MLOps Comparison"):
            gr.Markdown(
                "Select your **requirements** to get a personalised platform "
                "recommendation across AWS SageMaker, Azure ML, GCP Vertex AI, "
                "Databricks, and Snowflake ML."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cloud_reqs = gr.CheckboxGroup(
                        choices=_CAPABILITIES,
                        value=["Real-time serving (<100ms)", "Built-in feature store"],
                        label="Requirements",
                        info="Select all that apply to your use case",
                    )
                    cloud_btn = gr.Button("Compare Platforms", variant="primary")
                with gr.Column(scale=3):
                    cloud_plot = gr.Plot(label="Platform Comparison Heatmap")
                    cloud_md   = gr.Markdown()

            cloud_btn.click(
                fn=_cloud_mlops_demo,
                inputs=[cloud_reqs],
                outputs=[cloud_plot, cloud_md],
            )
