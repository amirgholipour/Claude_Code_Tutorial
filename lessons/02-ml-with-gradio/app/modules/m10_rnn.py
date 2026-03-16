"""Module 10 — Recurrent Neural Networks (RNN / LSTM / GRU)
Level: Deep Learning"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS

import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## Recurrent Neural Networks (RNNs)

RNNs are designed for **sequential data**: time series, text, audio, speech — any data where order matters and context from earlier steps influences later steps.

### Vanilla RNN
At each time step *t*, the network maintains a **hidden state** hₜ that carries information forward:

`hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)`

The same weights are reused at every step (parameter sharing over time). The output at the last step is used for prediction.

**Problem — vanishing gradients**: gradients shrink exponentially through many time steps, making it hard to learn long-range dependencies.

### LSTM (Long Short-Term Memory)
LSTMs add a separate **cell state** Cₜ that flows through time with minimal modification, plus three learnable gates:

| Gate | Purpose |
|------|---------|
| **Forget gate** fₜ | What to erase from cell state |
| **Input gate** iₜ | What new information to write |
| **Output gate** oₜ | What to expose as hidden state |

The cell state acts as a "highway" allowing gradients to flow back across hundreds of steps.

### GRU (Gated Recurrent Unit)
GRU simplifies LSTM by merging the cell state and hidden state into one, using only two gates (reset and update). It is:
- Faster to train (fewer parameters)
- Often matches LSTM on medium-sequence tasks
- Good default when LSTM feels like overkill

### Architectures
- **Sequence-to-one**: process full sequence, take last hidden state → classification / regression
- **Sequence-to-sequence**: produce output at every timestep → translation, labelling
- **Stacked RNNs**: multiple layers, output of one feeds into the next

### Applications
- Time series forecasting (weather, stocks, sensor data)
- Sentiment analysis (sequence → positive/negative)
- Language modelling (predict next word)
- Speech recognition

### Practical Tips
- Normalise input sequences to zero mean / unit variance
- Use LSTM or GRU by default — vanilla RNN rarely used in practice
- `batch_first=True` makes tensors (batch, seq, features) — more intuitive
"""

CODE_EXAMPLE = '''
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # take last timestep only

model = LSTMModel(input_size=1, hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''


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
        out, _ = self.rnn(x)       # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])   # last timestep → scalar


def _make_sequences(series, seq_len):
    """Slide a window of `seq_len` over 1-D series; target = next value."""
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_rnn(rnn_type, hidden_size, num_layers, seq_len, epochs, lr):
    try:
        rnn_map = {"LSTM": "LSTM", "GRU": "GRU", "Vanilla RNN": "RNN"}
        torch_rnn = rnn_map.get(rnn_type, "LSTM")

        # ── Generate sine wave ──────────────────────────────────────────────
        rng = np.random.default_rng(42)
        t   = np.linspace(0, 100, 2000)
        y_raw = np.sin(t) + 0.1 * rng.standard_normal(len(t))

        # Normalise
        mu, sigma = y_raw.mean(), y_raw.std()
        y_norm = (y_raw - mu) / sigma

        X_all, y_all = _make_sequences(y_norm, seq_len)

        # 80/20 train/test split (time-ordered)
        n_total = len(X_all)
        n_train = int(n_total * 0.8)
        X_train, X_test = X_all[:n_train], X_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

        # Tensors: (N, seq_len, 1)
        Xtr = torch.tensor(X_train).unsqueeze(-1)
        ytr = torch.tensor(y_train).unsqueeze(-1)
        Xte = torch.tensor(X_test).unsqueeze(-1)
        yte = torch.tensor(y_test).unsqueeze(-1)

        model     = RNNModel(torch_rnn, 1, hidden_size, num_layers, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        n_params  = sum(p.numel() for p in model.parameters())

        batch_size  = 32
        train_losses = []

        for epoch in range(epochs):
            model.train()
            indices    = torch.randperm(len(Xtr))
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(Xtr), batch_size):
                idx = indices[start: start + batch_size]
                xb, yb = Xtr[idx], ytr[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

        # ── Evaluate on test set ────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            test_preds = model(Xte).squeeze().numpy()
            test_mse   = criterion(torch.tensor(test_preds),
                                   yte.squeeze()).item()

        # De-normalise for plotting
        actual_seg   = y_raw[seq_len: seq_len + 200]          # first 200 actuals
        pred_denorm  = test_preds[:200] * sigma + mu
        actual_denorm = y_raw[n_train + seq_len: n_train + seq_len + 200]

        t_actual = t[seq_len: seq_len + 200]
        t_pred   = t[n_train + seq_len: n_train + seq_len + 200]

        # ── Figure: actual vs predicted | training loss ─────────────────────
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Actual vs Predicted", "Training Loss"])

        fig.add_trace(
            go.Scatter(x=t_actual, y=actual_seg, mode="lines",
                       name="Actual (train region)",
                       line=dict(color=COLORS["info"], width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_pred, y=pred_denorm, mode="lines",
                       name="Predicted (test region)",
                       line=dict(color=COLORS["danger"], width=2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_pred, y=actual_denorm, mode="lines",
                       name="Actual (test region)",
                       line=dict(color=COLORS["success"], width=1, dash="dot")),
            row=1, col=1,
        )

        epoch_list = list(range(1, epochs + 1))
        fig.add_trace(
            go.Scatter(x=epoch_list, y=train_losses, mode="lines+markers",
                       name="Train Loss",
                       line=dict(color=COLORS["primary"])),
            row=1, col=2,
        )

        fig.update_layout(
            title=f"{rnn_type} — Sine Wave Forecasting",
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="MSE Loss", row=1, col=2)

        arch = f"{rnn_type}(input=1, hidden={hidden_size}, layers={num_layers}) → Linear(1)"
        metrics_md = f"""
### Results

| Metric | Value |
|--------|-------|
| **Test MSE** | {test_mse:.6f} |
| **Test RMSE** | {np.sqrt(test_mse):.6f} |
| **Final Train Loss** | {train_losses[-1]:.6f} |
| **Model Parameters** | {n_params:,} |
| **Architecture** | `{arch}` |
| **Sequence Length** | {seq_len} |
"""
        return fig, metrics_md

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(title=f"Error: {e}", template="plotly_white", height=420)
        return err_fig, f"**Error:** {e}"


def build_tab():
    with gr.Tab("M10 — RNN / LSTM"):
        gr.Markdown("## Module 10: Recurrent Neural Networks — Sine Wave Forecasting")

        with gr.Accordion("Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        with gr.Row():
            with gr.Column(scale=1):
                rnn_type_radio = gr.Radio(
                    choices=["LSTM", "GRU", "Vanilla RNN"],
                    value="LSTM",
                    label="RNN Architecture",
                )
                hidden_sl = gr.Slider(16, 128, value=32, step=8,
                                      label="Hidden Size")
                layers_sl = gr.Slider(1, 3, value=2, step=1,
                                      label="Number of Layers")
                seq_sl    = gr.Slider(10, 50, value=20, step=5,
                                      label="Sequence Length")
                epochs_sl = gr.Slider(10, 50, value=20, step=5,
                                      label="Epochs")
                lr_sl     = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001,
                                      label="Learning Rate")
                run_btn   = gr.Button("Train RNN", variant="primary")

            with gr.Column(scale=2):
                pred_plot   = gr.Plot(label="Predictions & Training Loss")
                metrics_out = gr.Markdown()

        run_btn.click(
            fn=run_rnn,
            inputs=[rnn_type_radio, hidden_sl, layers_sl, seq_sl, epochs_sl, lr_sl],
            outputs=[pred_plot, metrics_out],
        )
