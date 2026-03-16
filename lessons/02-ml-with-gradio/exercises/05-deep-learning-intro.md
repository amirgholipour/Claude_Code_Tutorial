# Exercise 5: Deep Learning Introduction

## Goal

Use Modules 08–11 to understand neural networks, train CNNs and RNNs, and learn why training tricks (dropout, batch norm) matter in practice.

## What You'll Learn

- How neural networks learn through backpropagation
- Why CNNs are designed for images (and why MLPs aren't)
- How LSTMs solve the vanishing gradient problem
- How to prevent overfitting with dropout and batch normalization

---

## Part A — Neural Networks (Module 08)

**Open "🧠 08 · Neural Networks" tab**

### Experiment 1: Effect of hidden layer size

1. Dataset: **Iris**, hidden1: **4**, hidden2: **4**, epochs: **30** → Run
2. hidden1: **64**, hidden2: **32** → compare accuracy
3. hidden1: **256**, hidden2: **128** → note if performance improves or overfits

A network with 256 neurons is likely overkill for 150 iris samples — it will overfit.

**Rule of thumb:** Network capacity should match data complexity.

### Experiment 2: Learning rate sensitivity

1. lr: **0.1** → train (watch for unstable/diverging loss)
2. lr: **0.001** → stable but slow convergence
3. lr: **0.01** → usually the sweet spot for Adam optimizer on small datasets

Too high = loss bounces around or explodes. Too low = takes forever to converge.

### Experiment 3: Loss curve diagnosis

Look at the loss curve shape:
- **Both curves decreasing smoothly** → good training
- **Train loss decreases, val loss increases** → overfitting (add dropout!)
- **Both curves barely decrease** → model too small, or learning rate too low
- **Loss oscillates wildly** → learning rate too high

---

## Part B — CNNs (Module 09)

**Open "🖼️ 09 · CNNs" tab**

### Experiment 4: Train a CNN on digit recognition

1. Default settings → Run (trains on sklearn digits: 8×8 grayscale images)
2. Watch the loss curve — CNN should reach ~96–98% accuracy in 10 epochs
3. Look at the prediction grid: which digits get confused?

Commonly confused pairs: 4/9, 3/8, 7/1 — they look similar visually.

### Experiment 5: Effect of filter count

1. n_filters_1: **8** → Run
2. n_filters_1: **32** → compare accuracy
3. n_filters_1: **64** → diminishing returns on 8×8 images

For 8×8 images, 16 filters is enough — the images are tiny. For ImageNet (224×224), you'd need 64+ filters.

### How CNNs work

```
Input: 8×8 pixel image
  ↓ Conv2d(1→16, 3×3) → 16 feature maps of 8×8 (detects edges)
  ↓ MaxPool(2×2)       → 16 feature maps of 4×4 (downsampling)
  ↓ Conv2d(16→32, 3×3) → 32 feature maps of 4×4 (detects shapes)
  ↓ MaxPool(2×2)       → 32 feature maps of 2×2
  ↓ Flatten            → 32×2×2 = 128 values
  ↓ Linear(128→64)     → learns digit patterns
  ↓ Linear(64→10)      → 10 class scores
  ↓ Softmax            → probabilities
```

**Parameter sharing:** The same 3×3 filter slides across all positions. This is why CNNs need far fewer parameters than MLPs for images.

---

## Part C — RNNs (Module 10)

**Open "🔄 10 · RNNs" tab**

### Experiment 6: LSTM vs Vanilla RNN

1. Type: **Vanilla RNN**, hidden_size: **32**, epochs: **20** → Run
2. Type: **LSTM**, same settings → compare MSE
3. LSTM should achieve lower MSE (better at capturing long-range patterns)

The sine wave task requires remembering the pattern from many steps back. Vanilla RNNs struggle because gradients vanish over long sequences.

### Experiment 7: Sequence length

1. seq_len: **10** → Run (short memory, less context)
2. seq_len: **40** → Run (more context available to model)
3. Longer sequences generally help for periodic signals but take more memory

### What the prediction plot shows

- **Blue line**: true sine wave (what really happened)
- **Red line**: LSTM predictions
- A good model should track the blue line closely
- Errors compound: if the first prediction is wrong, subsequent ones drift

---

## Part D — Training Best Practices (Module 11)

**Open "⚙️ 11 · Training Tips" tab**

### Experiment 8: Regularization prevents overfitting

This module compares two models on the `digits` dataset:
- **Baseline**: No regularization
- **Regularized**: Dropout + BatchNorm + LR scheduler

1. Default settings → Run
2. Look at the comparison plots: baseline should overfit (train acc > val acc)
3. Regularized model should have smaller gap between train and val accuracy

### Experiment 9: BatchNorm effect

1. uncheck **use_batchnorm** → Run (slower convergence)
2. check **use_batchnorm** → Run (faster convergence, less sensitive to lr)

BatchNorm normalizes each layer's inputs to have zero mean and unit variance. This:
- Reduces sensitivity to weight initialization
- Allows higher learning rates
- Acts as mild regularization

### Experiment 10: Learning Rate Schedulers

1. scheduler: **None** → Run (constant lr throughout)
2. scheduler: **CosineAnnealing** → Run (lr decays like a cosine wave)
3. scheduler: **StepLR** → Run (lr drops by 0.1 every 30 epochs)

Cosine annealing often works best: starts high (fast progress), gradually decays (fine-tuning at end).

## Deep Learning Decision Guide

```
Your data is...

Images/Video → CNN
  → Small dataset: pre-trained ResNet (Module 12)
  → Large dataset: train from scratch

Sequences (text, time series) → RNN/LSTM
  → Short sequences (<50 steps): GRU (simpler than LSTM)
  → Long sequences: LSTM or Transformer

Tabular/structured data → Gradient Boosting (usually beats DL!)
  → Very large dataset (>100k rows): try a deep MLP
  → Small dataset (<10k rows): stick with Random Forest
```

---

Next: [Exercise 6 — Full Pipeline →](./06-full-pipeline.md)
