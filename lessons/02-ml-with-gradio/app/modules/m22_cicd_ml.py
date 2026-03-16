"""
Module 22 — CI/CD for Machine Learning
Level: Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

THEORY = """
## 📖 What Is CI/CD for ML?

**Continuous Integration / Continuous Delivery (CI/CD)** for ML extends traditional software CI/CD to handle the unique challenges of machine learning: data validation, model validation, reproducibility, and deployment gates.

Traditional CI/CD: code changes → tests → deploy
ML CI/CD: code + data + model changes → data tests + model tests + performance gates → deploy

## 🏗️ ML CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    ML CI/CD Pipeline                    │
│                                                         │
│  Code Push / Scheduled  →  Pipeline Triggered           │
│                                                         │
│  Stage 1: Data Validation                               │
│    ✓ Schema check (correct columns/dtypes)              │
│    ✓ Range checks (no impossible values)                │
│    ✓ Missing value threshold                            │
│    ✓ Statistical tests (no sudden distribution shift)   │
│                                                         │
│  Stage 2: Model Training & Validation                   │
│    ✓ Reproducibility (fixed random seed)                │
│    ✓ Cross-validation score                             │
│    ✓ Beat baseline model                                │
│    ✓ Performance gate (accuracy ≥ threshold)            │
│    ✓ Feature importance sanity check                    │
│                                                         │
│  Stage 3: Integration Tests                             │
│    ✓ Prediction latency < budget                        │
│    ✓ Batch prediction produces expected output shape    │
│    ✓ Model handles edge cases (nulls, extreme values)   │
│                                                         │
│  Stage 4: Deploy (if all gates pass)                    │
│    → Canary deployment (5% traffic)                     │
│    → Blue-green deployment                              │
│    → Shadow mode (predict but don't serve)              │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Data Validation
```python
# Great Expectations (popular data validation library)
import great_expectations as ge

df = ge.from_pandas(data)
df.expect_column_values_to_not_be_null("age")
df.expect_column_values_to_be_between("age", 0, 120)
df.expect_column_values_to_be_in_set("gender", ["M", "F", "Other"])
df.expect_column_mean_to_be_between("salary", 30000, 200000)

result = df.validate()
print("Data valid:", result["success"])
```

## ✅ Model Validation Gates
| Gate | Check | Fail action |
|---|---|---|
| **Accuracy gate** | acc ≥ baseline_acc | Block deployment |
| **F1 gate** | macro_f1 ≥ 0.85 | Block deployment |
| **Latency gate** | p99 latency < 100ms | Alert, optional block |
| **Fairness gate** | accuracy gap < 5% across groups | Block deployment |
| **Feature sanity** | top feature makes domain sense | Manual review |

## 🔄 Reproducibility
```python
# Always set random seeds
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Pin dependency versions
# requirements.txt: scikit-learn==1.4.0 (not >=1.0)
# Use virtual environments / Docker

# Hash training data
import hashlib
data_hash = hashlib.sha256(X_train.tobytes()).hexdigest()
# Store in model metadata for exact reproduction
```

## ⚠️ ML-Specific CI/CD Challenges
- **Non-determinism**: Neural networks, some algorithms are non-deterministic by default
- **Slow tests**: Training a large model takes hours — use smaller proxy datasets for CI
- **Data versioning**: Code version alone isn't enough — data + model must be versioned together
- **Online vs offline metrics**: A/B test in production may contradict offline validation
"""

CODE_EXAMPLE = '''
# Simulated ML CI/CD pipeline with automated gates

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

# ── Stage 1: Data Validation ──────────────────────────────────────
def validate_data(X, y):
    checks = {}
    checks["no_nulls"]       = not np.isnan(X).any()
    checks["min_samples"]    = len(X) >= 100
    checks["feature_range"]  = (X >= -100).all() and (X <= 100).all()
    checks["class_balance"]  = min(np.bincount(y)) / max(np.bincount(y)) >= 0.5
    return checks

# ── Stage 2: Model Training + Performance Gate ────────────────────
def train_and_validate(X_train, y_train, X_test, y_test,
                       accuracy_threshold=0.90):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc    = accuracy_score(y_test, model.predict(X_test))
    cv_acc = cross_val_score(model, X_train, y_train, cv=5).mean()
    gate   = acc >= accuracy_threshold
    return model, acc, cv_acc, gate

# ── Stage 3: Latency Check ────────────────────────────────────────
def check_latency(model, X_test, budget_ms=50):
    t0    = time.time()
    _     = model.predict(X_test[:100])
    latency_ms = (time.time() - t0) * 1000 / 100
    return latency_ms, latency_ms < budget_ms

# ── Run pipeline ──────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data_checks = validate_data(X_train, y_train)
model, acc, cv_acc, gate = train_and_validate(X_train, y_train, X_test, y_test)
latency, lat_ok = check_latency(model, X_test)

print("Data checks:", data_checks)
print(f"Accuracy: {acc:.3f}, CV: {cv_acc:.3f}, Gate passed: {gate}")
print(f"Latency: {latency:.2f}ms, Budget OK: {lat_ok}")
print("DEPLOY:", all(data_checks.values()) and gate and lat_ok)
'''


class MLPipeline:
    """Simulated ML CI/CD pipeline with automated quality gates."""

    def __init__(self, X, y, feature_names, acc_threshold=0.90, baseline_model=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.acc_threshold = acc_threshold
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)
        self.baseline_model = baseline_model or LogisticRegression(max_iter=500, random_state=42)
        self.results = {}

    def stage1_data_validation(self):
        checks = {
            "no_nulls":     not np.isnan(self.X_train).any(),
            "min_samples":  len(self.X_train) >= 50,
            "feature_range": float(np.abs(self.X_train).max()) < 1e6,
            "class_balance": min(np.bincount(self.y_train)) / max(np.bincount(self.y_train)) >= 0.2,
            "n_features_ok": self.X_train.shape[1] >= 2,
        }
        self.results["data"] = checks
        return checks

    def stage2_baseline(self):
        self.baseline_model.fit(self.X_train, self.y_train)
        baseline_acc = accuracy_score(self.y_test, self.baseline_model.predict(self.X_test))
        self.results["baseline_acc"] = baseline_acc
        return baseline_acc

    def stage3_train_candidate(self, model, model_name):
        t0 = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - t0

        preds   = model.predict(self.X_test)
        acc     = accuracy_score(self.y_test, preds)
        f1      = f1_score(self.y_test, preds, average="macro")
        cv_accs = cross_val_score(model, self.X_train, self.y_train, cv=5)

        # Latency
        t0 = time.time()
        _ = model.predict(self.X_test[:50])
        latency_ms = (time.time() - t0) * 1000 / 50

        gates = {
            "accuracy_gate":  acc >= self.acc_threshold,
            "beats_baseline": acc >= self.results.get("baseline_acc", 0),
            "f1_gate":        f1 >= 0.85,
            "latency_gate":   latency_ms < 50,
            "cv_stable":      cv_accs.std() < 0.05,
        }
        self.results["candidate"] = {
            "name": model_name, "acc": acc, "f1": f1,
            "cv_mean": cv_accs.mean(), "cv_std": cv_accs.std(),
            "latency_ms": latency_ms, "train_time_ms": train_time * 1000,
            "gates": gates
        }
        return self.results["candidate"]


def run_cicd_demo(dataset_name: str, model_name: str, acc_threshold: float):
    DATASETS = {
        "Breast Cancer": load_breast_cancer,
        "Iris":          load_iris,
    }
    data = DATASETS[dataset_name]()
    X, y = data.data, data.target
    feat_names = list(data.feature_names)

    MODELS = {
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
    }
    model = MODELS[model_name]

    pipeline = MLPipeline(X, y, feat_names, acc_threshold=acc_threshold)

    data_checks = pipeline.stage1_data_validation()
    baseline_acc = pipeline.stage2_baseline()
    candidate    = pipeline.stage3_train_candidate(model, model_name)
    gates        = candidate["gates"]

    all_data_ok  = all(data_checks.values())
    all_gates_ok = all(gates.values())
    deploy       = all_data_ok and all_gates_ok

    # Build pipeline visualization
    stages = ["Data Validation", "Baseline", "Model Train", "Gates", "Deploy"]
    statuses = [
        all_data_ok,
        True,  # baseline always runs
        True,  # training always runs
        all_gates_ok,
        deploy
    ]
    colors = ["#66bb6a" if s else "#ef5350" for s in statuses]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Pipeline Stages", "Gate Results",
                                        "Accuracy Comparison", "CV Score Distribution"])

    # Pipeline stages
    fig.add_trace(go.Bar(
        x=stages, y=[1] * len(stages),
        marker_color=colors,
        text=["✅" if s else "❌" for s in statuses],
        textposition="inside",
        showlegend=False
    ), row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Gate results
    gate_names  = list(gates.keys())
    gate_vals   = list(gates.values())
    gate_colors = ["#66bb6a" if v else "#ef5350" for v in gate_vals]
    fig.add_trace(go.Bar(
        x=[g.replace("_", " ") for g in gate_names],
        y=[1] * len(gate_names),
        marker_color=gate_colors,
        text=["✅ PASS" if v else "❌ FAIL" for v in gate_vals],
        textposition="inside",
        showlegend=False
    ), row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    # Accuracy comparison
    fig.add_trace(go.Bar(
        x=["Baseline (LR)", model_name],
        y=[baseline_acc, candidate["acc"]],
        marker_color=["#bdbdbd", "#42a5f5"],
        text=[f"{baseline_acc:.4f}", f"{candidate['acc']:.4f}"],
        textposition="outside",
        showlegend=False
    ), row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="Accuracy", row=2, col=1)

    # CV distribution (simulate 5-fold)
    rng = np.random.default_rng(42)
    cv_samples = rng.normal(candidate["cv_mean"], max(candidate["cv_std"], 0.005), 100)
    cv_samples = cv_samples.clip(0, 1)
    fig.add_trace(go.Histogram(x=cv_samples, nbinsx=20, name="CV scores",
                               marker_color="#7e57c2"), row=2, col=2)
    fig.add_vline(x=candidate["cv_mean"], line_dash="dash", line_color="red",
                  annotation_text=f"mean={candidate['cv_mean']:.3f}", row=2, col=2)

    fig.update_layout(height=550, title_text=f"ML CI/CD Pipeline — {model_name} on {dataset_name}")

    passed_gates = sum(gates.values())
    total_gates  = len(gates)
    data_checks_str = "\n".join([f"| {k.replace('_', ' ')} | {'✅' if v else '❌'} |"
                                  for k, v in data_checks.items()])

    metrics_md = f"""
### Pipeline Results

**DEPLOY STATUS:** {'✅ APPROVED' if deploy else '❌ BLOCKED'}

#### Stage 1 — Data Validation
| Check | Status |
|---|---|
{data_checks_str}

#### Stage 2 — Model Performance
| Metric | Value | Gate |
|---|---|---|
| Accuracy | `{candidate['acc']:.4f}` | {'✅' if gates['accuracy_gate'] else '❌'} ≥ {acc_threshold} |
| Macro F1 | `{candidate['f1']:.4f}` | {'✅' if gates['f1_gate'] else '❌'} ≥ 0.85 |
| Beats baseline | `{baseline_acc:.4f}` → `{candidate['acc']:.4f}` | {'✅' if gates['beats_baseline'] else '❌'} |
| CV stability (std) | `{candidate['cv_std']:.4f}` | {'✅' if gates['cv_stable'] else '❌'} < 0.05 |
| Latency (50 preds) | `{candidate['latency_ms']:.2f}` ms | {'✅' if gates['latency_gate'] else '❌'} < 50ms |

**Gates passed: {passed_gates}/{total_gates}**
"""
    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🔄 Module 22 — CI/CD for Machine Learning\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nSimulate a full ML CI/CD pipeline with data validation, model training, performance gates, and deploy decision.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=["Breast Cancer", "Iris"],
                value="Breast Cancer"
            )
            model_dd = gr.Dropdown(
                label="Candidate Model",
                choices=["Random Forest", "Gradient Boosting", "Logistic Regression"],
                value="Random Forest"
            )
            threshold_sl = gr.Slider(
                label="Accuracy Gate Threshold",
                minimum=0.80, maximum=0.99, step=0.01, value=0.90
            )
            run_btn = gr.Button("▶ Run Pipeline", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_cicd_demo,
        inputs=[dataset_dd, model_dd, threshold_sl],
        outputs=[plot_out, metrics_out]
    )
