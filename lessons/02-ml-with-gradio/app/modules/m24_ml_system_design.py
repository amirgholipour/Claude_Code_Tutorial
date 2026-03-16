"""
Module 24 — ML System Design Best Practices
Level: Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## 📖 What Is ML System Design?

ML system design is the practice of architecting **production-grade ML systems** that are reliable, scalable, maintainable, and meet business requirements. A model that achieves 95% accuracy in a Jupyter notebook is worthless if it can't serve predictions at scale.

> "Most ML problems are not ML problems — they're data engineering, infrastructure, and product problems."

## 🏗️ ML System Architecture Patterns

### 1. Batch vs Online Serving
| Dimension | Batch | Online (Real-time) |
|---|---|---|
| **Latency** | Hours/days | <100ms |
| **Throughput** | Millions/day | 1000s/second |
| **Complexity** | Low | High |
| **Cost** | Lower | Higher |
| **Use cases** | Recommendation email, churn prediction | Fraud detection, search ranking |
| **Freshness** | Stale predictions | Fresh predictions |

**Batch**: Run model on large dataset overnight, store predictions in DB
**Online**: Model server (REST API) responds to real-time requests

### 2. Feature Store
A centralized repository for storing, sharing, and serving features:

```
Training:   Feature Store ──→ Training job ──→ Model
Inference:  Feature Store ──→ Model server ──→ Prediction
```

**Benefits:**
- No training-serving skew (same features at train and serve time)
- Feature reuse across teams/models
- Point-in-time correctness (no data leakage)

**Tools**: Feast, Tecton, Vertex AI Feature Store, Databricks Feature Store

### 3. Model Serving Patterns
| Pattern | Description | Use case |
|---|---|---|
| **Single model** | One model serves all | Simple products |
| **A/B testing** | Split traffic between models | Safe rollout |
| **Canary deployment** | 5% → 20% → 100% rollout | Risk mitigation |
| **Shadow mode** | New model runs but doesn't serve | Testing without risk |
| **Multi-armed bandit** | Adaptive traffic splitting based on performance | Maximize reward |
| **Ensemble serving** | Combine predictions from multiple models | Maximum accuracy |

### 4. Latency vs Accuracy Tradeoffs
```
Model size:       Small Model ←──────────→ Large Model
Latency:          Low (1ms)               High (500ms)
Accuracy:         Lower                   Higher
Cost:             Low                     High
```

**Optimization techniques:**
- **Model distillation**: Train small "student" model to mimic large "teacher"
- **Quantization**: FP32 → INT8 (4× faster, ~1-2% accuracy loss)
- **Pruning**: Remove unimportant weights
- **Caching**: Cache predictions for frequent inputs
- **Batching**: Accumulate N requests, predict together

### 5. Scalability Considerations
- **Horizontal scaling**: Multiple model server replicas behind load balancer
- **Asynchronous inference**: Queue requests, process async (better for bursty traffic)
- **GPU batching**: Maximize GPU utilization by batching multiple requests
- **Edge deployment**: Run model on device (mobile/IoT) — zero latency, privacy-preserving

### 6. ML System Anti-Patterns
| Anti-pattern | Problem | Fix |
|---|---|---|
| **Training-serving skew** | Different preprocessing at train vs serve time | Use shared feature pipeline |
| **Undifferentiated heavy lifting** | Rebuilding infrastructure from scratch | Use managed services |
| **Direct model coupling** | Client talks directly to model | Add API abstraction layer |
| **No monitoring** | Silent failures accumulate | Add metrics, alerts, dashboards |
| **No rollback plan** | Bad model can't be quickly reverted | Version models, blue-green deploy |

## ✅ Design Checklist
- [ ] What is the latency requirement? (p50, p95, p99)
- [ ] What is the throughput requirement? (requests/second)
- [ ] Batch or online serving?
- [ ] How are features served consistently between training and inference?
- [ ] What is the model update frequency?
- [ ] How is the model monitored in production?
- [ ] What is the rollback strategy?
- [ ] What happens when the model is unavailable? (fallback)
"""

CODE_EXAMPLE = '''
# ML System Design: Online Model Serving with FastAPI

from fastapi import FastAPI
import joblib, numpy as np, time, logging
from pydantic import BaseModel

# ── Model loading ─────────────────────────────────────────────────
app = FastAPI()
model = joblib.load("model_v3.pkl")
scaler = joblib.load("scaler.pkl")
logger = logging.getLogger("inference")

class PredictRequest(BaseModel):
    features: list[float]
    request_id: str = ""

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float

# ── Prediction endpoint ───────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.time()
    X  = np.array(req.features).reshape(1, -1)
    X_sc = scaler.transform(X)

    pred  = int(model.predict(X_sc)[0])
    prob  = float(model.predict_proba(X_sc)[0, pred])
    latency = (time.time() - t0) * 1000

    logger.info({
        "request_id": req.request_id,
        "prediction": pred, "probability": prob,
        "latency_ms": latency, "model": "v3"
    })
    return PredictResponse(
        prediction=pred, probability=prob,
        model_version="v3", latency_ms=latency
    )

# ── Health check + model info ─────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "v3"}

# Run: uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4
'''


def _latency_model(throughput: float, model_size: str, caching: bool, batching: bool) -> dict:
    """Estimate system characteristics based on configuration."""
    base_latency = {"Small (LR/NB)": 1.0, "Medium (RF/GBT)": 5.0, "Large (DNN)": 50.0}[model_size]
    base_accuracy = {"Small (LR/NB)": 0.82, "Medium (RF/GBT)": 0.93, "Large (DNN)": 0.97}[model_size]
    base_cost = {"Small (LR/NB)": 1.0, "Medium (RF/GBT)": 3.0, "Large (DNN)": 10.0}[model_size]

    if batching:
        base_latency *= 1.5     # slightly higher p50 latency due to batching delay
        throughput   *= 3.0     # much higher throughput
        base_cost    *= 0.6     # lower cost per request

    if caching:
        effective_latency = base_latency * 0.2  # 80% cache hit → fast
        base_cost         *= 0.5
    else:
        effective_latency = base_latency

    # Throughput saturation
    max_throughput = {"Small (LR/NB)": 5000, "Medium (RF/GBT)": 1000, "Large (DNN)": 200}[model_size]
    if batching:
        max_throughput *= 3

    saturation = min(throughput / max_throughput, 1.0)
    p99_latency = effective_latency * (1 + 5 * saturation**2)  # latency spikes under load

    return {
        "p50_ms":         round(effective_latency, 2),
        "p99_ms":         round(p99_latency, 2),
        "max_rps":        round(max_throughput, 0),
        "accuracy":       round(base_accuracy, 3),
        "cost_relative":  round(base_cost, 2),
        "saturated":      saturation > 0.8,
    }


def run_system_design(scenario: str, model_size: str, throughput: float,
                      use_caching: bool, use_batching: bool):
    if scenario == "Latency vs Accuracy Tradeoff":
        sizes      = ["Small (LR/NB)", "Medium (RF/GBT)", "Large (DNN)"]
        p50s       = []
        p99s       = []
        accuracies = []
        costs      = []

        for sz in sizes:
            m = _latency_model(throughput, sz, use_caching, use_batching)
            p50s.append(m["p50_ms"])
            p99s.append(m["p99_ms"])
            accuracies.append(m["accuracy"])
            costs.append(m["cost_relative"])

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Latency (ms)", "Accuracy", "Cost (relative)"])

        colors = ["#66bb6a", "#42a5f5", "#ef5350"]

        fig.add_trace(go.Bar(x=sizes, y=p50s, name="p50", marker_color="#42a5f5",
                             text=[f"{v:.1f}" for v in p50s], textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=sizes, y=p99s, name="p99", marker_color="#ef5350",
                             text=[f"{v:.1f}" for v in p99s], textposition="outside"), row=1, col=1)

        fig.add_trace(go.Bar(x=sizes, y=accuracies, marker_color=colors,
                             text=[f"{v:.3f}" for v in accuracies], textposition="outside",
                             showlegend=False), row=1, col=2)
        fig.update_yaxes(range=[0.7, 1.05], row=1, col=2)

        fig.add_trace(go.Bar(x=sizes, y=costs, marker_color=colors,
                             text=[f"{v:.1f}×" for v in costs], textposition="outside",
                             showlegend=False), row=1, col=3)

        fig.update_layout(height=400, barmode="group",
                          title_text=f"Model Size Tradeoffs — Caching: {use_caching} | Batching: {use_batching}")

        metrics_md = f"""
### Latency vs Accuracy vs Cost

| Model | p50 ms | p99 ms | Accuracy | Cost |
|---|---|---|---|---|
{''.join([f"| {sz} | {p:.1f} | {p9:.1f} | {a:.3f} | {c:.1f}× |\n" for sz,p,p9,a,c in zip(sizes,p50s,p99s,accuracies,costs)])}
**Optimizations applied:** {'Caching ✅' if use_caching else 'Caching ❌'} | {'Batching ✅' if use_batching else 'Batching ❌'}

> **Decision**: Choose model size based on SLA. If you must meet p99 < 10ms, Small model is the only option. If latency budget allows 100ms, Medium gives much better accuracy.
"""

    elif scenario == "Batch vs Online Comparison":
        patterns = ["Batch (nightly)", "Near-Real-Time (minutes)", "Online (<100ms)", "Edge (<10ms)"]
        latencies = [43200000, 60000, 50, 5]  # ms
        throughputs = [10_000_000, 100_000, 5000, 100]
        complexities = [1, 3, 7, 9]
        costs = [1, 2.5, 8, 4]

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Latency (log ms)", "Throughput (req/day)",
                                            "Infrastructure Complexity", "Relative Cost"])

        colors = ["#66bb6a", "#42a5f5", "#ffa726", "#ef5350"]

        fig.add_trace(go.Bar(x=patterns, y=[np.log10(l) for l in latencies],
                             marker_color=colors,
                             text=[f"10^{np.log10(l):.1f}ms" for l in latencies],
                             textposition="outside", showlegend=False), row=1, col=1)

        fig.add_trace(go.Bar(x=patterns, y=[np.log10(t) for t in throughputs],
                             marker_color=colors,
                             text=[f"{t:,}" for t in throughputs],
                             textposition="outside", showlegend=False), row=1, col=2)

        fig.add_trace(go.Bar(x=patterns, y=complexities,
                             marker_color=colors,
                             text=complexities, textposition="outside",
                             showlegend=False), row=2, col=1)

        fig.add_trace(go.Bar(x=patterns, y=costs,
                             marker_color=colors,
                             text=[f"{c}×" for c in costs], textposition="outside",
                             showlegend=False), row=2, col=2)

        fig.update_layout(height=500, title_text="Serving Pattern Comparison")

        metrics_md = """
### When to Use Each Pattern

| Pattern | Best for | Example use case |
|---|---|---|
| **Batch** | High volume, latency-tolerant | Nightly churn scores, report generation |
| **Near-Real-Time** | Minutes acceptable | Personalization refresh, feed ranking |
| **Online** | Sub-second required | Fraud detection, search ranking |
| **Edge** | Zero latency, offline | Voice assistants, AR filters |

**Rule of thumb**: Start batch, move to online only when business requires it.
"""

    elif scenario == "Feature Store Architecture":
        # Show feature pipeline diagram as a table
        fig = go.Figure()

        nodes_x = [0.1, 0.3, 0.5, 0.7, 0.9,   # row 1: data sources
                   0.3, 0.5, 0.7,               # row 2: transform
                   0.5,                          # row 3: feature store
                   0.3, 0.7]                     # row 4: consumers

        nodes_y = [0.9]*5 + [0.65]*3 + [0.4] + [0.15]*2

        labels = ["DB", "Logs", "API", "Stream", "Files",
                  "Batch Transform", "Stream Transform", "Aggregations",
                  "Feature Store\n(Online + Offline)",
                  "Training Job", "Inference Server"]

        colors_nodes = (["#4fc3f7"]*5 + ["#81c784"]*3 + ["#ffb74d"] + ["#ce93d8"]*2)

        for x, y, label, color in zip(nodes_x, nodes_y, labels, colors_nodes):
            fig.add_shape(type="rect", x0=x-0.08, y0=y-0.05, x1=x+0.08, y1=y+0.05,
                          fillcolor=color, line_color="white", opacity=0.8)
            fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                               font=dict(size=9, color="black"), align="center")

        # Arrows
        arrow_pairs = [
            (0.1,0.85,0.3,0.7), (0.3,0.85,0.3,0.7), (0.5,0.85,0.5,0.7),
            (0.7,0.85,0.7,0.7), (0.9,0.85,0.7,0.7),
            (0.3,0.6,0.5,0.45), (0.5,0.6,0.5,0.45), (0.7,0.6,0.5,0.45),
            (0.5,0.35,0.3,0.2), (0.5,0.35,0.7,0.2)
        ]
        for x0, y0, x1, y1 in arrow_pairs:
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                               xref="paper", yref="paper", axref="paper", ayref="paper",
                               showarrow=True, arrowhead=2, arrowcolor="#555")

        fig.update_layout(
            xaxis=dict(visible=False, range=[0,1]),
            yaxis=dict(visible=False, range=[0,1]),
            height=450,
            title_text="Feature Store Architecture",
            paper_bgcolor="#fafafa"
        )

        metrics_md = """
### Feature Store: Key Benefits

| Problem (without) | Solution (with Feature Store) |
|---|---|
| **Training-serving skew** | Same feature logic used at train + serve time |
| **Duplicated engineering** | Share features across models/teams |
| **Point-in-time leakage** | Versioned snapshots prevent future data leak |
| **Slow feature computation** | Pre-compute + cache in low-latency store |

> **Offline store**: For training (e.g., Parquet on S3, BigQuery)
> **Online store**: For serving (e.g., Redis, DynamoDB)
"""

    else:
        fig = go.Figure()
        metrics_md = "Select a scenario."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🏗️ Module 24 — ML System Design Best Practices\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nExplore ML system design tradeoffs: latency vs accuracy, batch vs online serving, and feature store architecture.")

    with gr.Row():
        with gr.Column(scale=1):
            scenario_dd = gr.Dropdown(
                label="Scenario",
                choices=["Latency vs Accuracy Tradeoff", "Batch vs Online Comparison",
                         "Feature Store Architecture"],
                value="Latency vs Accuracy Tradeoff"
            )
            model_size_dd = gr.Dropdown(
                label="Model Size",
                choices=["Small (LR/NB)", "Medium (RF/GBT)", "Large (DNN)"],
                value="Medium (RF/GBT)"
            )
            throughput_sl = gr.Slider(label="Traffic (requests/sec)", minimum=10, maximum=2000,
                                      step=10, value=200)
            caching_cb   = gr.Checkbox(label="Enable prediction caching", value=False)
            batching_cb  = gr.Checkbox(label="Enable request batching", value=False)
            run_btn      = gr.Button("▶ Analyze Design", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_system_design,
        inputs=[scenario_dd, model_size_dd, throughput_sl, caching_cb, batching_cb],
        outputs=[plot_out, metrics_out]
    )
