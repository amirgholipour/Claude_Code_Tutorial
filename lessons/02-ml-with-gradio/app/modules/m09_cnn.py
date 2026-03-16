"""Module 09 — Convolutional Neural Networks
Level: Deep Learning"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from config import COLORS

import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## Convolutional Neural Networks (CNNs)

CNNs are purpose-built for **grid-like data** such as images. Unlike an MLP that treats each pixel independently, a CNN exploits the spatial structure of images through **local connectivity** and **parameter sharing**.

### Convolutional Layers
A convolutional layer slides a small **filter (kernel)** across the input, computing a dot product at each position. This produces a **feature map** highlighting where a particular pattern (edge, curve, texture) appears.

- **Kernel size**: size of the sliding window (e.g., 3×3)
- **Stride**: how many pixels to skip per step
- **Padding**: zeros added around borders to preserve spatial size
- **Feature maps**: one per filter — shallow layers detect edges, deep layers detect complex shapes

### Pooling Layers
Pooling reduces spatial dimensions, making the network:
- **Faster** (fewer parameters downstream)
- **Translation-invariant** (a shifted cat is still a cat)

Max pooling keeps the strongest activation in each region. A 2×2 max pool halves width and height.

### Typical Architecture
```
Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Output
```
Early layers: low-level features (edges, colours)
Later layers: high-level concepts (faces, objects)

### Why CNNs Beat MLPs for Images
| Property | MLP | CNN |
|----------|-----|-----|
| Parameters | O(N²) per layer | O(k²) per filter — shared |
| Spatial awareness | None | Explicit via receptive field |
| Translation invariance | No | Yes (via pooling) |

### Key Concepts
- **Receptive field**: the region of the input a neuron "sees"
- **Depth**: number of filters = number of feature maps produced
- **BatchNorm**: often added after Conv to stabilise training
- **Dropout**: typically applied in the fully-connected head
"""

CODE_EXAMPLE = '''
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# For 8x8 digits:
# Conv(1,16,3,pad=1) -> ReLU -> MaxPool(2) -> 4x4
# Conv(16,32,3,pad=1) -> ReLU -> MaxPool(2) -> 2x2
# Flatten -> Linear(32*2*2=128, 64) -> ReLU -> Linear(64, 10)
'''


class DigitsCNN(nn.Module):
    """CNN for 8x8 digits dataset. Input: (N, 1, 8, 8)"""
    def __init__(self, n_filters_1=16, n_filters_2=32, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 8→4
            nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 4→2
        )
        fc_in = n_filters_2 * 2 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _make_prediction_grid(images, true_labels, pred_labels, n=25):
    """Build a 5x5 plotly grid of digit images with true/pred labels."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[
            f"T:{true_labels[i]} P:{pred_labels[i]}"
            for i in range(n)
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.12,
    )

    for i in range(n):
        img = images[i].reshape(8, 8)
        r, c = divmod(i, cols)
        correct = (true_labels[i] == pred_labels[i])
        fig.add_trace(
            go.Heatmap(
                z=img,
                colorscale="gray_r",
                showscale=False,
                showlegend=False,
            ),
            row=r + 1, col=c + 1,
        )
        # Colour the subplot title green/red by correctness
        idx = i  # subplot_titles are 0-indexed in annotations
        color = COLORS["success"] if correct else COLORS["danger"]
        fig.layout.annotations[idx].font = dict(size=9, color=color)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(
        title="Sample Test Predictions (green=correct, red=wrong)",
        template="plotly_white",
        height=max(300, rows * 120),
        margin=dict(t=60, b=20),
    )
    return fig


def run_cnn(n_filters_1, n_filters_2, epochs, lr, batch_size):
    try:
        batch_size = int(batch_size)
        epochs     = int(epochs)

        X, y, _, _ = load_dataset("digits")          # (1797, 64)  values 0–16
        X = X / 16.0                                  # normalise to [0, 1]

        X_train, X_test, y_train, y_test, _ = split_and_scale(
            X, y, test_size=0.2, scale=None           # already normalised
        )

        # Val split from train
        n_val = max(1, int(len(X_train) * 0.15))
        X_val, y_val   = X_train[:n_val], y_train[:n_val]
        X_train, y_train = X_train[n_val:], y_train[n_val:]

        def to_tensor_4d(arr):
            return torch.tensor(arr, dtype=torch.float32).reshape(-1, 1, 8, 8)

        Xtr = to_tensor_4d(X_train);  ytr = torch.tensor(y_train, dtype=torch.long)
        Xva = to_tensor_4d(X_val);    yva = torch.tensor(y_val,   dtype=torch.long)
        Xte = to_tensor_4d(X_test);   yte = torch.tensor(y_test,  dtype=torch.long)

        model     = DigitsCNN(n_filters_1, n_filters_2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        n_params  = sum(p.numel() for p in model.parameters())

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            indices    = torch.randperm(len(Xtr))
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(Xtr), batch_size):
                idx = indices[start: start + batch_size]
                xb, yb = Xtr[idx], ytr[idx]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xva), yva).item()
            val_losses.append(val_loss)

        model.eval()
        with torch.no_grad():
            test_logits = model(Xte)
            test_preds  = test_logits.argmax(dim=1).numpy()
            test_acc    = (test_preds == yte.numpy()).mean()

        # --- Learning curve plot ---
        epoch_list = list(range(1, epochs + 1))
        curve_fig = go.Figure()
        curve_fig.add_trace(go.Scatter(
            x=epoch_list, y=train_losses, mode="lines+markers",
            name="Train Loss", line=dict(color=COLORS["primary"])))
        curve_fig.add_trace(go.Scatter(
            x=epoch_list, y=val_losses, mode="lines+markers",
            name="Val Loss", line=dict(color=COLORS["warning"])))
        curve_fig.update_layout(
            title="CNN Training — Digits Dataset",
            xaxis_title="Epoch", yaxis_title="Cross-Entropy Loss",
            template="plotly_white", height=400,
        )

        # --- Prediction grid (first 25 test samples) ---
        grid_images = X_test[:25]
        grid_true   = y_test[:25]
        grid_pred   = test_preds[:25]
        grid_fig    = _make_prediction_grid(grid_images, grid_true, grid_pred)

        metrics_md = f"""
### CNN Results — Digits Dataset

| Metric | Value |
|--------|-------|
| **Test Accuracy** | {test_acc:.4f} ({test_acc*100:.1f}%) |
| **Final Train Loss** | {train_losses[-1]:.4f} |
| **Final Val Loss** | {val_losses[-1]:.4f} |
| **Model Parameters** | {n_params:,} |
| **Architecture** | Conv({n_filters_1}) → Conv({n_filters_2}) → FC(64) → 10 |
| **Input shape** | (N, 1, 8, 8) — digits images |
"""
        return curve_fig, grid_fig, metrics_md

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(title=f"Error: {e}", template="plotly_white", height=400)
        return err_fig, err_fig, f"**Error:** {e}"


def build_tab():
    with gr.Tab("M09 — CNN"):
        gr.Markdown("## Module 09: Convolutional Neural Networks")

        with gr.Accordion("Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        with gr.Row():
            with gr.Column(scale=1):
                f1_sl = gr.Slider(8,  64,  value=16, step=8, label="Filters Layer 1")
                f2_sl = gr.Slider(16, 128, value=32, step=8, label="Filters Layer 2")
                ep_sl = gr.Slider(5,  30,  value=10, step=1, label="Epochs")
                lr_sl = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001,
                                  label="Learning Rate")
                bs_dd = gr.Dropdown(choices=["32", "64", "128"], value="64",
                                    label="Batch Size")
                run_btn = gr.Button("Train CNN", variant="primary")

            with gr.Column(scale=2):
                curve_plot = gr.Plot(label="Training / Validation Loss")
                grid_plot  = gr.Plot(label="Test Predictions Grid")
                metrics_out = gr.Markdown()

        run_btn.click(
            fn=run_cnn,
            inputs=[f1_sl, f2_sl, ep_sl, lr_sl, bs_dd],
            outputs=[curve_plot, grid_plot, metrics_out],
        )
