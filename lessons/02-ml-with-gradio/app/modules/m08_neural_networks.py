"""Module 08 — Neural Networks (MLP)
Level: Deep Learning"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from utils.plot_utils import learning_curve_plot
from config import COLORS

import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## What is a Neural Network?

A **neural network** is a function approximator built from layers of interconnected neurons. Each neuron computes a weighted sum of its inputs, adds a bias, then passes the result through a non-linear **activation function**.

### Architecture
- **Input layer**: one neuron per feature
- **Hidden layers**: intermediate representations; more layers = more abstraction
- **Output layer**: one neuron per class (classification) or one neuron (regression)

### Forward Propagation
Data flows left-to-right through the network:
`input → layer 1 (linear + activation) → layer 2 → … → output`

Each linear layer computes: **z = Wx + b**, then applies activation **a = f(z)**.

### Activation Functions
| Function | Formula | When to use |
|----------|---------|-------------|
| **ReLU** | max(0, x) | Hidden layers — default choice, fast, avoids vanishing gradient |
| **Sigmoid** | 1/(1+e⁻ˣ) | Binary output (0–1), older networks |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden layers when centred output helps |
| **Softmax** | eˣᵢ/Σeˣ | Final layer for multi-class classification |

### Loss Functions
- **CrossEntropyLoss** — classification: measures divergence between predicted probabilities and true labels
- **MSELoss** — regression: mean squared difference between prediction and target

### Backpropagation
After computing the loss, PyTorch walks backwards through the computation graph applying the **chain rule** to compute ∂Loss/∂W for every weight. This is done automatically by `loss.backward()`.

### Optimizers
- **SGD (Stochastic Gradient Descent)**: W ← W − lr × ∇W — simple, requires tuning momentum
- **Adam**: adaptive per-parameter learning rates using first and second gradient moments — often best default choice

### Key Hyperparameters
- **Learning rate**: how big a step to take per gradient update (too high = diverges, too low = slow)
- **Hidden layer sizes**: controls model capacity
- **Epochs**: number of full passes over training data
- **Batch size**: samples per gradient update — smaller = noisier but more updates per epoch
"""

CODE_EXAMPLE = '''
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Build model: 4 inputs → [64, 32] hidden → 3 outputs
model = MLP(input_size=4, hidden_sizes=[64, 32], output_size=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
'''


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            if h > 0:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(input_size, hidden1, hidden2, output_size):
    hidden_sizes = [s for s in [hidden1, hidden2] if s > 0]
    return MLP(input_size, hidden_sizes, output_size)


def run_mlp(dataset_name, hidden1, hidden2, lr, epochs, batch_size):
    try:
        batch_size = int(batch_size)
        epochs = int(epochs)

        X, y, _, _ = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test, _ = split_and_scale(X, y, test_size=0.2)

        # Further split train into train/val
        n_val = max(1, int(len(X_train) * 0.15))
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train, y_train = X_train[n_val:], y_train[n_val:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_val_t   = torch.tensor(y_val,   dtype=torch.long)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
        y_test_t  = torch.tensor(y_test,  dtype=torch.long)

        n_features = X_train.shape[1]
        n_classes  = len(np.unique(y))

        model     = build_model(n_features, hidden1, hidden2, n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        n_params = sum(p.numel() for p in model.parameters())

        train_losses, val_accs = [], []

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(X_train_t), batch_size):
                idx   = indices[start: start + batch_size]
                xb    = X_train_t[idx]
                yb    = y_train_t[idx]
                optimizer.zero_grad()
                out   = model(xb)
                loss  = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).argmax(dim=1)
                acc = (val_preds == y_val_t).float().mean().item()
            val_accs.append(acc)

        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t).argmax(dim=1)
            test_acc   = (test_preds == y_test_t).float().mean().item()

        # Build dual-axis figure: loss + val accuracy
        epoch_list = list(range(1, epochs + 1))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=epoch_list, y=train_losses, mode="lines+markers",
                       name="Train Loss", line=dict(color=COLORS["primary"])),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=epoch_list, y=val_accs, mode="lines+markers",
                       name="Val Accuracy", line=dict(color=COLORS["success"], dash="dot")),
            secondary_y=True,
        )
        fig.update_layout(
            title=f"MLP Training — {dataset_name}",
            xaxis_title="Epoch",
            template="plotly_white",
            height=420,
            legend=dict(x=0.01, y=0.99),
        )
        fig.update_yaxes(title_text="Cross-Entropy Loss", secondary_y=False)
        fig.update_yaxes(title_text="Validation Accuracy", secondary_y=True,
                         range=[0, 1])

        metrics_md = f"""
### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | {test_acc:.4f} ({test_acc*100:.1f}%) |
| **Final Train Loss** | {train_losses[-1]:.4f} |
| **Final Val Accuracy** | {val_accs[-1]:.4f} |
| **Model Parameters** | {n_params:,} |
| **Architecture** | {n_features} → {hidden1} → {hidden2} → {n_classes} |
| **Dataset** | {dataset_name} ({X_train.shape[0]} train / {X_test.shape[0]} test) |
"""
        return fig, metrics_md

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(title=f"Error: {e}", template="plotly_white", height=420)
        return err_fig, f"**Error:** {e}"


def build_tab():
    with gr.Tab("M08 — Neural Networks"):
        gr.Markdown("## Module 08: Neural Networks (Multi-Layer Perceptron)")

        with gr.Accordion("Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    choices=["iris", "wine", "breast_cancer"],
                    value="iris",
                    label="Dataset",
                )
                hidden1_sl = gr.Slider(8, 256, value=64, step=8,
                                       label="Hidden Layer 1 Size")
                hidden2_sl = gr.Slider(8, 128, value=32, step=8,
                                       label="Hidden Layer 2 Size")
                lr_sl = gr.Slider(0.0001, 0.1, value=0.01, step=0.001,
                                  label="Learning Rate")
                epochs_sl = gr.Slider(10, 100, value=30, step=5, label="Epochs")
                batch_dd = gr.Dropdown(
                    choices=["16", "32", "64"],
                    value="32",
                    label="Batch Size",
                )
                run_btn = gr.Button("Train MLP", variant="primary")

            with gr.Column(scale=2):
                curve_plot = gr.Plot(label="Training Curve")
                metrics_out = gr.Markdown()

        run_btn.click(
            fn=run_mlp,
            inputs=[dataset_dd, hidden1_sl, hidden2_sl, lr_sl, epochs_sl, batch_dd],
            outputs=[curve_plot, metrics_out],
        )
