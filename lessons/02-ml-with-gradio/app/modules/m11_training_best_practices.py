"""Module 11 — Training Best Practices
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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

THEORY = """
## Training Best Practices

Training a deep network well requires more than just stacking layers. These techniques dramatically improve **generalisation** — the ability to perform well on unseen data.

### Overfitting
A model **overfits** when it memorises the training data instead of learning general patterns.
Signs: training accuracy ≫ validation accuracy; training loss keeps decreasing while val loss increases.

### Dropout
During training, randomly **zero out** each neuron with probability *p*. This forces the network to develop redundant representations — no single neuron can be relied upon.

- At inference time, all neurons are active (weights scaled by 1−p)
- Typical values: p = 0.2–0.5 for hidden layers
- Introduced in the original Dropout paper (Srivastava et al., 2014)

### Batch Normalisation (BatchNorm)
Normalises the **input to each layer** across the mini-batch to zero mean and unit variance, then applies learnable scale (γ) and shift (β) parameters.

Benefits:
- Reduces sensitivity to weight initialisation
- Allows higher learning rates
- Acts as mild regulariser
- Dramatically speeds up training

Applied after Linear (or Conv) and before activation: `Linear → BN → ReLU`.

### Learning Rate Schedulers
A fixed LR is often suboptimal. Schedulers reduce the LR over training:

| Scheduler | Behaviour |
|-----------|-----------|
| **StepLR** | Multiply LR by γ every N epochs |
| **CosineAnnealingLR** | Smoothly decay LR following a cosine curve to near-zero |
| **ReduceLROnPlateau** | Reduce LR when a monitored metric stops improving |

### Early Stopping
Monitor validation loss. If it doesn't improve for `patience` consecutive epochs, stop training and restore the best weights. Prevents both overfitting and wasted compute.

### Weight Initialisation
| Activation | Recommended Init |
|-----------|-----------------|
| ReLU | He / Kaiming (variance = 2/fan_in) |
| Tanh / Sigmoid | Xavier / Glorot (variance = 1/fan_avg) |
| Default PyTorch | Kaiming uniform for Linear — usually fine |

### Summary: Regularised Model Recipe
```
Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → ReLU → Dropout → Output
```
Combined with a scheduler and early stopping, this setup generalises far better than a bare MLP.
"""

CODE_EXAMPLE = '''
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Regularised model
model = nn.Sequential(
    nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 10)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_one_epoch(model, optimizer, criterion, train_loader)
    val_loss = evaluate(model, criterion, val_loader)
    scheduler.step()            # or scheduler.step(val_loss) for ReduceLROnPlateau

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
'''


def _make_baseline_model(input_size, n_classes):
    """Baseline MLP — no regularisation."""
    return nn.Sequential(
        nn.Linear(input_size, 128), nn.ReLU(),
        nn.Linear(128, 64),          nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def _make_regularised_model(input_size, n_classes, dropout_rate, use_batchnorm):
    """Regularised MLP with optional BatchNorm and Dropout."""
    layers = []

    def block(in_f, out_f):
        layers.append(nn.Linear(in_f, out_f))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_f))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

    block(input_size, 128)
    block(128, 64)
    layers.append(nn.Linear(64, n_classes))
    return nn.Sequential(*layers)


def _get_scheduler(scheduler_type, optimizer, epochs):
    if scheduler_type == "StepLR":
        return StepLR(optimizer, step_size=max(1, epochs // 5), gamma=0.5)
    elif scheduler_type == "CosineAnnealing":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, patience=max(1, epochs // 10),
                                 factor=0.5, verbose=False)
    return None


def _train_model(model, X_tr, y_tr, X_va, y_va, epochs, lr,
                 scheduler_type, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = _get_scheduler(scheduler_type, optimizer, epochs)
    is_plateau = scheduler_type == "ReduceLROnPlateau"

    train_losses, val_losses, val_accs = [], [], []

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    Xva = torch.tensor(X_va, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.long)

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
            val_out  = model(Xva)
            v_loss   = criterion(val_out, yva).item()
            v_acc    = (val_out.argmax(1) == yva).float().mean().item()
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if scheduler:
            if is_plateau:
                scheduler.step(v_loss)
            else:
                scheduler.step()

    return train_losses, val_losses, val_accs


def run_best_practices(dropout_rate, use_batchnorm, scheduler_type, epochs, lr):
    try:
        epochs = int(epochs)

        X, y, _, _ = load_dataset("digits")
        X_train, X_test, y_train, y_test, _ = split_and_scale(X, y, test_size=0.2)

        n_val = max(1, int(len(X_train) * 0.15))
        X_val, y_val   = X_train[:n_val], y_train[:n_val]
        X_train, y_train = X_train[n_val:], y_train[n_val:]

        n_features = X_train.shape[1]
        n_classes  = 10

        # Train baseline
        baseline = _make_baseline_model(n_features, n_classes)
        bl_train_losses, bl_val_losses, bl_val_accs = _train_model(
            baseline, X_train, y_train, X_val, y_val,
            epochs, lr, "None",
        )

        # Train regularised model
        regularised = _make_regularised_model(
            n_features, n_classes, dropout_rate, use_batchnorm
        )
        rg_train_losses, rg_val_losses, rg_val_accs = _train_model(
            regularised, X_train, y_train, X_val, y_val,
            epochs, lr, scheduler_type,
        )

        # Test accuracy for both
        Xte = torch.tensor(X_test, dtype=torch.float32)
        yte = torch.tensor(y_test, dtype=torch.long)

        baseline.eval()
        regularised.eval()
        with torch.no_grad():
            bl_test_acc = (baseline(Xte).argmax(1) == yte).float().mean().item()
            rg_test_acc = (regularised(Xte).argmax(1) == yte).float().mean().item()

        bl_best_epoch = int(np.argmin(bl_val_losses)) + 1
        rg_best_epoch = int(np.argmin(rg_val_losses)) + 1

        bl_params = sum(p.numel() for p in baseline.parameters())
        rg_params = sum(p.numel() for p in regularised.parameters())

        # ── 2-subplot comparison figure ────────────────────────────────────
        epoch_list = list(range(1, epochs + 1))
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Train & Val Loss", "Validation Accuracy"],
        )

        # Loss traces
        fig.add_trace(go.Scatter(
            x=epoch_list, y=bl_train_losses, mode="lines",
            name="Baseline Train Loss",
            line=dict(color=COLORS["info"], dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epoch_list, y=bl_val_losses, mode="lines",
            name="Baseline Val Loss",
            line=dict(color=COLORS["info"])), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epoch_list, y=rg_train_losses, mode="lines",
            name="Regularised Train Loss",
            line=dict(color=COLORS["success"], dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epoch_list, y=rg_val_losses, mode="lines",
            name="Regularised Val Loss",
            line=dict(color=COLORS["success"])), row=1, col=1)

        # Accuracy traces
        fig.add_trace(go.Scatter(
            x=epoch_list, y=bl_val_accs, mode="lines",
            name="Baseline Val Acc",
            line=dict(color=COLORS["warning"])), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=epoch_list, y=rg_val_accs, mode="lines",
            name="Regularised Val Acc",
            line=dict(color=COLORS["danger"])), row=1, col=2)

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
        fig.update_layout(
            title="Baseline vs Regularised Model — Digits Dataset",
            template="plotly_white",
            height=440,
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        )

        reg_desc = []
        if dropout_rate > 0:
            reg_desc.append(f"Dropout({dropout_rate})")
        if use_batchnorm:
            reg_desc.append("BatchNorm")
        if scheduler_type != "None":
            reg_desc.append(scheduler_type)
        reg_str = ", ".join(reg_desc) if reg_desc else "None"

        metrics_md = f"""
### Comparison Results — Digits Dataset

|  | Baseline | Regularised |
|--|----------|-------------|
| **Test Accuracy** | {bl_test_acc:.4f} ({bl_test_acc*100:.1f}%) | {rg_test_acc:.4f} ({rg_test_acc*100:.1f}%) |
| **Final Val Accuracy** | {bl_val_accs[-1]:.4f} | {rg_val_accs[-1]:.4f} |
| **Best Val Loss Epoch** | {bl_best_epoch} | {rg_best_epoch} |
| **Final Val Loss** | {bl_val_losses[-1]:.4f} | {rg_val_losses[-1]:.4f} |
| **Parameters** | {bl_params:,} | {rg_params:,} |
| **Regularisation** | None | {reg_str} |

**Generalisation gap** (train acc − val acc): indicates how much the model overfits.
A well-regularised model should show a smaller gap and higher val/test accuracy.
"""
        return fig, metrics_md

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(title=f"Error: {e}", template="plotly_white", height=440)
        return err_fig, f"**Error:** {e}"


def build_tab():
    with gr.Tab("M11 — Training Best Practices"):
        gr.Markdown("## Module 11: Training Best Practices — Regularisation & Schedulers")

        with gr.Accordion("Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        with gr.Row():
            with gr.Column(scale=1):
                dropout_sl = gr.Slider(0.0, 0.7, value=0.3, step=0.1,
                                       label="Dropout Rate")
                batchnorm_cb = gr.Checkbox(value=True, label="Use Batch Normalisation")
                scheduler_dd = gr.Dropdown(
                    choices=["None", "StepLR", "CosineAnnealing", "ReduceLROnPlateau"],
                    value="CosineAnnealing",
                    label="LR Scheduler",
                )
                epochs_sl = gr.Slider(20, 100, value=50, step=10, label="Epochs")
                lr_sl     = gr.Slider(0.0001, 0.1, value=0.01, step=0.001,
                                      label="Learning Rate")
                run_btn   = gr.Button("Compare Models", variant="primary")

            with gr.Column(scale=2):
                cmp_plot    = gr.Plot(label="Baseline vs Regularised")
                metrics_out = gr.Markdown()

        run_btn.click(
            fn=run_best_practices,
            inputs=[dropout_sl, batchnorm_cb, scheduler_dd, epochs_sl, lr_sl],
            outputs=[cmp_plot, metrics_out],
        )
