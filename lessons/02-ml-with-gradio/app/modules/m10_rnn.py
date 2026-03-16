"""Module 10 — Recurrent Neural Networks (RNN / LSTM / GRU)
Level: Deep Learning"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## Recurrent Neural Networks (RNNs)

RNNs process **sequential data** where order matters: time series, text, audio.

### Vanilla RNN
At each time step *t*, the hidden state carries information forward:
`hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)`

**Problem — vanishing gradients**: gradients shrink exponentially through many steps,
making it impossible to learn long-range dependencies. Try Demo B below to see this!

### LSTM (Long Short-Term Memory)
Adds a **cell state** Cₜ that flows through time like a highway, plus 3 gates:

| Gate | Purpose |
|------|---------|
| **Forget** fₜ | What to erase from memory |
| **Input** iₜ | What new info to write |
| **Output** oₜ | What to expose as output |

The cell state lets gradients flow back across hundreds of steps.

### GRU (Gated Recurrent Unit)
Simplifies LSTM: merges cell/hidden state, uses 2 gates (reset + update).
- Fewer parameters → faster training
- Often matches LSTM on medium-length sequences
- Good default when LSTM feels like overkill

### Key Concepts for Demos Below

**Demo A — Architecture Comparison**: See how LSTM/GRU outperform Vanilla RNN.
**Demo B — Sequence Length Effect**: Watch Vanilla RNN fail as sequences get longer (vanishing gradients in action).
**Demo C — Multi-Step Forecasting**: See how prediction errors compound over multiple steps ahead.
"""

CODE_EXAMPLE = '''import torch, torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep → prediction

model = LSTMForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(20):
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Multi-step ahead forecasting
def forecast_multistep(model, seed_seq, n_steps):
    """Autoregressive: feed predictions back as input."""
    preds = []
    seq = seed_seq.clone()
    for _ in range(n_steps):
        with torch.no_grad():
            p = model(seq.unsqueeze(0))
        preds.append(p.item())
        seq = torch.cat([seq[1:], p.unsqueeze(0)], dim=0)
    return preds
'''


# ── Model ─────────────────────────────────────────────────────────────────────

class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.0 if num_layers == 1 else 0.1,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# ── Data generation ───────────────────────────────────────────────────────────

def _generate_signal(signal_type: str, n=2000):
    rng = np.random.default_rng(42)
    t = np.linspace(0, 100, n)
    if signal_type == "Sine Wave":
        y = np.sin(t) + 0.1 * rng.standard_normal(n)
    else:  # Compound Signal
        y = (0.02 * t
             + np.sin(2 * np.pi * t / 25)
             + 0.5 * np.sin(2 * np.pi * t / 7)
             + 0.3 * rng.standard_normal(n))
    return t, y


def _make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_model(rnn_type, X_train, y_train, hidden=32, layers=2, epochs=20, lr=0.001):
    """Train an RNN model and return (model, losses, n_params)."""
    Xtr = torch.tensor(X_train).unsqueeze(-1)
    ytr = torch.tensor(y_train).unsqueeze(-1)

    model = RNNModel(rnn_type, 1, hidden, layers, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    losses = []
    bs = 32
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        epoch_loss, nb = 0, 0
        for start in range(0, len(Xtr), bs):
            batch_idx = idx[start:start + bs]
            xb, yb = Xtr[batch_idx], ytr[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        losses.append(epoch_loss / max(nb, 1))

    return model, losses, n_params


def _evaluate(model, X_test, y_test):
    """Return (predictions_array, mse)."""
    Xte = torch.tensor(X_test).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        preds = model(Xte).squeeze().numpy()
    mse = float(np.mean((preds - y_test) ** 2))
    return preds, mse


# ── Demo A: Architecture Comparison ──────────────────────────────────────────

def demo_arch_comparison(signal_type, hidden, epochs, lr):
    try:
        t, y_raw = _generate_signal(signal_type)
        mu, sigma = y_raw.mean(), y_raw.std()
        y_norm = (y_raw - mu) / sigma

        seq_len = 25
        X_all, y_all = _make_sequences(y_norm, seq_len)
        n_train = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:n_train], X_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

        results = {}
        for name, rtype in [("LSTM", "LSTM"), ("GRU", "GRU"), ("Vanilla RNN", "RNN")]:
            model, losses, n_params = _train_model(
                rtype, X_train, y_train, hidden=hidden, layers=2, epochs=epochs, lr=lr
            )
            preds, mse = _evaluate(model, X_test, y_test)
            results[name] = {"preds": preds, "mse": mse, "losses": losses, "params": n_params}

        # Denormalize for plotting
        t_test = t[n_train + seq_len: n_train + seq_len + 200]
        actual = y_raw[n_train + seq_len: n_train + seq_len + 200]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Test Predictions (first 200 steps)", "MSE Comparison"],
            column_widths=[0.65, 0.35]
        )

        # Left: overlay predictions
        fig.add_trace(go.Scatter(
            x=t_test, y=actual, mode="lines", name="Actual",
            line=dict(color="gray", width=2)
        ), row=1, col=1)

        colors = {"LSTM": "#42a5f5", "GRU": "#66bb6a", "Vanilla RNN": "#ef5350"}
        for name in ["LSTM", "GRU", "Vanilla RNN"]:
            pred_denorm = results[name]["preds"][:200] * sigma + mu
            fig.add_trace(go.Scatter(
                x=t_test, y=pred_denorm, mode="lines", name=name,
                line=dict(color=colors[name], width=1.5, dash="dot" if name == "Vanilla RNN" else "solid")
            ), row=1, col=1)

        # Right: MSE bars
        names = list(results.keys())
        mses = [results[n]["mse"] for n in names]
        fig.add_trace(go.Bar(
            x=names, y=mses, marker_color=[colors[n] for n in names],
            text=[f"{m:.5f}" for m in mses], textposition="outside",
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(title=f"Architecture Comparison — {signal_type}",
                          template="plotly_white", height=450)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Test MSE (normalised)", row=1, col=2)

        rows = "\n".join(
            f"| **{n}** | `{results[n]['mse']:.6f}` | `{np.sqrt(results[n]['mse']):.6f}` | {results[n]['params']:,} |"
            for n in names
        )
        md = f"""### Architecture Comparison Results

| Model | MSE | RMSE | Parameters |
|-------|-----|------|------------|
{rows}

> LSTM and GRU typically match or beat Vanilla RNN. On the compound signal (trend + multi-frequency),
> the gap is more pronounced because long-range patterns matter more.
"""
        return fig, md
    except Exception as e:
        return go.Figure().update_layout(title=str(e), height=400), f"**Error:** {e}"


# ── Demo B: Sequence Length Effect ────────────────────────────────────────────

def demo_seq_length(signal_type, hidden, epochs, lr):
    try:
        t, y_raw = _generate_signal(signal_type)
        mu, sigma = y_raw.mean(), y_raw.std()
        y_norm = (y_raw - mu) / sigma

        seq_lengths = [10, 25, 50, 100]
        arch_results = {"LSTM": [], "Vanilla RNN": []}

        for sl in seq_lengths:
            X_all, y_all = _make_sequences(y_norm, sl)
            n_train = int(len(X_all) * 0.8)
            X_train, X_test = X_all[:n_train], X_all[n_train:]
            y_train, y_test = y_all[:n_train], y_all[n_train:]

            for name, rtype in [("LSTM", "LSTM"), ("Vanilla RNN", "RNN")]:
                model, _, _ = _train_model(
                    rtype, X_train, y_train, hidden=hidden, layers=2,
                    epochs=epochs, lr=lr
                )
                _, mse = _evaluate(model, X_test, y_test)
                arch_results[name].append(mse)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["MSE vs Sequence Length", "LSTM Advantage Ratio"],
            column_widths=[0.55, 0.45]
        )

        # Left: MSE vs seq_len for both
        fig.add_trace(go.Scatter(
            x=seq_lengths, y=arch_results["LSTM"], mode="lines+markers",
            name="LSTM", line=dict(color="#42a5f5", width=2),
            marker=dict(size=10)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=seq_lengths, y=arch_results["Vanilla RNN"], mode="lines+markers",
            name="Vanilla RNN", line=dict(color="#ef5350", width=2),
            marker=dict(size=10)
        ), row=1, col=1)

        # Right: ratio (how much better LSTM is)
        ratios = [v / max(l, 1e-8) for l, v in
                  zip(arch_results["LSTM"], arch_results["Vanilla RNN"])]
        fig.add_trace(go.Bar(
            x=[str(s) for s in seq_lengths], y=ratios,
            marker_color=["#66bb6a" if r > 1 else "#bdbdbd" for r in ratios],
            text=[f"{r:.1f}x" for r in ratios], textposition="outside",
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="Equal performance", row=1, col=2)

        fig.update_layout(title=f"Sequence Length Effect — {signal_type}",
                          template="plotly_white", height=450)
        fig.update_xaxes(title_text="Sequence Length", row=1, col=1)
        fig.update_yaxes(title_text="Test MSE", row=1, col=1)
        fig.update_xaxes(title_text="Sequence Length", row=1, col=2)
        fig.update_yaxes(title_text="Vanilla RNN MSE / LSTM MSE", row=1, col=2)

        rows = "\n".join(
            f"| {sl} | `{arch_results['LSTM'][i]:.6f}` | `{arch_results['Vanilla RNN'][i]:.6f}` | {ratios[i]:.1f}x |"
            for i, sl in enumerate(seq_lengths)
        )
        md = f"""### Sequence Length Effect — Vanishing Gradients in Action

| Seq Length | LSTM MSE | Vanilla RNN MSE | RNN/LSTM Ratio |
|------------|----------|-----------------|----------------|
{rows}

> **Key insight**: As sequence length increases, Vanilla RNN performance degrades
> because gradients vanish over long chains. LSTM's cell state "highway" preserves
> gradient flow, maintaining performance even at seq_len=100.
>
> Ratio > 1.0 means LSTM outperforms Vanilla RNN by that factor.
"""
        return fig, md
    except Exception as e:
        return go.Figure().update_layout(title=str(e), height=400), f"**Error:** {e}"


# ── Demo C: Multi-Step Forecasting ───────────────────────────────────────────

def demo_multistep(signal_type, hidden, epochs, lr):
    try:
        t, y_raw = _generate_signal(signal_type)
        mu, sigma = y_raw.mean(), y_raw.std()
        y_norm = (y_raw - mu) / sigma

        seq_len = 25
        X_all, y_all = _make_sequences(y_norm, seq_len)
        n_train = int(len(X_all) * 0.8)
        X_train, y_train = X_all[:n_train], y_all[:n_train]

        model, losses, _ = _train_model(
            "LSTM", X_train, y_train, hidden=hidden, layers=2, epochs=epochs, lr=lr
        )

        # Autoregressive multi-step forecasting
        horizons = [1, 5, 10, 25, 50]
        seed_idx = n_train  # start of test region
        seed_seq = torch.tensor(y_norm[seed_idx:seed_idx + seq_len], dtype=torch.float32).unsqueeze(-1)
        actual_future = y_norm[seed_idx + seq_len: seed_idx + seq_len + max(horizons)]

        model.eval()
        all_preds = []
        seq = seed_seq.clone()
        with torch.no_grad():
            for step in range(max(horizons)):
                p = model(seq.unsqueeze(0))
                all_preds.append(p.item())
                new_val = p.unsqueeze(0)
                seq = torch.cat([seq[1:], new_val], dim=0)

        all_preds = np.array(all_preds)

        # Calculate MSE at each horizon
        horizon_mses = {}
        for h in horizons:
            if h <= len(actual_future):
                horizon_mses[h] = float(np.mean((all_preds[:h] - actual_future[:h]) ** 2))

        # Denormalize
        actual_plot = actual_future * sigma + mu
        preds_plot = all_preds * sigma + mu
        n_plot = min(50, len(actual_future))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Autoregressive Forecast", "Error vs Forecast Horizon"],
            column_widths=[0.55, 0.45]
        )

        # Left: actual vs forecast
        steps = list(range(1, n_plot + 1))
        fig.add_trace(go.Scatter(
            x=steps, y=actual_plot[:n_plot], mode="lines", name="Actual",
            line=dict(color="gray", width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=steps, y=preds_plot[:n_plot], mode="lines", name="LSTM Forecast",
            line=dict(color="#42a5f5", width=2)
        ), row=1, col=1)

        # Shade error region
        fig.add_trace(go.Scatter(
            x=steps + steps[::-1],
            y=list(preds_plot[:n_plot]) + list(actual_plot[:n_plot])[::-1],
            fill="toself", fillcolor="rgba(239,83,80,0.15)",
            line=dict(width=0), name="Error", showlegend=True
        ), row=1, col=1)

        # Right: MSE vs horizon bar
        h_labels = [str(h) for h in horizons if h in horizon_mses]
        h_vals = [horizon_mses[h] for h in horizons if h in horizon_mses]
        colors = ["#66bb6a" if v < 0.1 else "#ff9800" if v < 0.5 else "#ef5350" for v in h_vals]
        fig.add_trace(go.Bar(
            x=h_labels, y=h_vals, marker_color=colors,
            text=[f"{v:.4f}" for v in h_vals], textposition="outside",
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(title=f"Multi-Step Forecasting — {signal_type}",
                          template="plotly_white", height=450)
        fig.update_xaxes(title_text="Steps Ahead", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Forecast Horizon", row=1, col=2)
        fig.update_yaxes(title_text="MSE (normalised)", row=1, col=2)

        rows = "\n".join(
            f"| {h} | `{horizon_mses[h]:.6f}` | `{np.sqrt(horizon_mses[h]):.6f}` |"
            for h in horizons if h in horizon_mses
        )
        md = f"""### Multi-Step Forecasting — Error Accumulation

| Horizon | MSE | RMSE |
|---------|-----|------|
{rows}

> **Key insight**: Each prediction feeds back as input for the next step (autoregressive).
> Small errors compound: by 25–50 steps ahead, the forecast diverges significantly.
> This is why real-world forecasting systems re-anchor predictions with actual observations.
>
> The pink shaded area shows the growing gap between forecast and reality.
"""
        return fig, md
    except Exception as e:
        return go.Figure().update_layout(title=str(e), height=400), f"**Error:** {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown("# 🔄 Module 10 — Recurrent Neural Networks\n*Level: Deep Learning*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("""---
## 🎮 Interactive Demos

Three demos showing different aspects of RNN behavior. Each trains fresh models (may take 10–30s).""")

    # ── Shared controls ──
    with gr.Row():
        signal_dd = gr.Dropdown(
            label="Signal Type",
            choices=["Sine Wave", "Compound Signal"],
            value="Sine Wave",
            info="Compound = trend + two frequencies + noise (harder)"
        )
        hidden_sl = gr.Slider(16, 64, value=32, step=8, label="Hidden Size")
        epochs_sl = gr.Slider(10, 40, value=20, step=5, label="Epochs")
        lr_sl = gr.Slider(0.0005, 0.005, value=0.001, step=0.0005, label="Learning Rate")

    # ── Demo A ──
    gr.Markdown("### A) Architecture Comparison\nTrain LSTM, GRU, and Vanilla RNN on the same data. Compare predictions and MSE.")
    run_a = gr.Button("▶ Run Architecture Comparison", variant="primary")
    with gr.Row():
        plot_a = gr.Plot(label="Architecture Comparison")
    md_a = gr.Markdown()
    run_a.click(fn=demo_arch_comparison, inputs=[signal_dd, hidden_sl, epochs_sl, lr_sl],
                outputs=[plot_a, md_a])

    # ── Demo B ──
    gr.Markdown("### B) Sequence Length Effect (Vanishing Gradients)\nCompare LSTM vs Vanilla RNN as input sequence gets longer. Watch Vanilla RNN fail!")
    run_b = gr.Button("▶ Run Sequence Length Experiment", variant="primary")
    with gr.Row():
        plot_b = gr.Plot(label="Sequence Length Effect")
    md_b = gr.Markdown()
    run_b.click(fn=demo_seq_length, inputs=[signal_dd, hidden_sl, epochs_sl, lr_sl],
                outputs=[plot_b, md_b])

    # ── Demo C ──
    gr.Markdown("### C) Multi-Step Forecasting (Error Accumulation)\nFeed predictions back as input. Watch errors compound over longer horizons.")
    run_c = gr.Button("▶ Run Multi-Step Forecast", variant="primary")
    with gr.Row():
        plot_c = gr.Plot(label="Multi-Step Forecasting")
    md_c = gr.Markdown()
    run_c.click(fn=demo_multistep, inputs=[signal_dd, hidden_sl, epochs_sl, lr_sl],
                outputs=[plot_c, md_c])
