# Exercise 3: Classical Machine Learning

## Goal

Use Modules 02–04 to train and evaluate classical ML models, understanding how hyperparameters affect performance.

## What You'll Learn

- How regression algorithms differ (and when regularization helps)
- How classification algorithms draw decision boundaries differently
- How to properly evaluate a model with multiple metrics

## Steps

### Part A — Regression (Module 02)

**Open "📈 02 · Regression" tab**

#### Experiment 1: Overfitting with Polynomials

1. Algorithm: **Polynomial**, Degree: **1** → Run → note R² and RMSE
2. Degree: **3** → Run → R² improves
3. Degree: **5** → Run → check if train score is high but test score drops

This is **overfitting**: the degree-5 polynomial memorizes the training data but fails to generalize.

#### Experiment 2: Regularization rescues overfitting

1. Algorithm: **Ridge**, Alpha: **0.01**, Degree: **5** → Run
2. Increase Alpha to **10.0** → R² may slightly drop but model becomes more stable

Ridge (L2) penalizes large weights, preventing the model from fitting noise. Lasso (L1) goes further — it drives some weights to exactly zero (feature selection).

```
Alpha = 0 → No regularization (pure polynomial fit)
Alpha → ∞ → Weights → 0 (underfitting)
Sweet spot → validation curve
```

### Part B — Classification (Module 03)

**Open "🎯 03 · Classification" tab**

#### Experiment 3: Decision Tree depth and overfitting

1. Algorithm: **Decision Tree**, Dataset: **Iris**, Max Depth: **1** → Run
   - Confusion matrix will show many misclassifications (underfitting)
2. Max Depth: **5** → significant improvement
3. Max Depth: **20** → confusion matrix looks perfect but likely overfitting

The decision tree with depth 20 has memorized every training sample. On a fresh dataset it would perform poorly.

#### Experiment 4: Compare all algorithms

Run each algorithm on **Breast Cancer** (binary classification):

| Algorithm | Expected Accuracy | Key Characteristic |
|---|---|---|
| Logistic Regression | ~95% | Linear boundary, fast |
| k-NN (k=5) | ~93% | Non-parametric, slow at predict time |
| Naive Bayes | ~92% | Assumes feature independence |
| Decision Tree | ~90–95% | Interpretable, prone to overfitting |

The **confusion matrix** tells a richer story than accuracy:
- **False Negative** (missed cancer) is much worse than **False Positive**
- This is why Recall matters more than Precision for cancer detection

### Part C — Model Evaluation (Module 04)

**Open "📏 04 · Model Evaluation" tab**

#### Experiment 5: Cross-validation reveals true performance

1. Algorithm: **Random Forest**, Dataset: **Iris**, eval type: **Cross-Validation**, CV folds: **5** → Run
2. Note mean accuracy and standard deviation
3. Change CV folds to **10** → std should decrease (more reliable estimate)

Cross-validation gives a more honest estimate than a single train/test split because it averages over 5 or 10 different splits.

#### Experiment 6: ROC Curve

1. Dataset: **Breast Cancer** (binary only), eval type: **ROC Curve** → Run
2. Note AUC score — closer to 1.0 is better, 0.5 = random

The ROC curve shows the tradeoff between sensitivity (True Positive Rate) and specificity (False Positive Rate) at different classification thresholds.

## Key Metrics Cheat Sheet

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | TP+TN / All | Balanced classes |
| **Precision** | TP / (TP+FP) | When false positives are costly |
| **Recall** | TP / (TP+FN) | When false negatives are costly (medical!) |
| **F1** | 2 × P×R / (P+R) | Imbalanced classes |
| **ROC-AUC** | Area under ROC curve | Binary classification, threshold-independent |
| **R²** | 1 - SS_res/SS_tot | Regression — explained variance |
| **RMSE** | √(mean(error²)) | Regression — penalizes large errors |

## The Bias-Variance Tradeoff

```
High Bias (Underfitting):          High Variance (Overfitting):
  → Model too simple                 → Model too complex
  → Bad on train AND test            → Good on train, bad on test
  → Fix: more complexity             → Fix: regularization, more data

          Bias ↓ ←──────────────→ Variance ↑
Simple ──────────────────────────────── Complex
(Linear)                        (Deep tree, high-degree poly)
                    ↑
              Sweet spot
```

## Try It Yourself

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
depths = range(1, 15)

train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(), X, y,
    param_name="max_depth", param_range=depths,
    cv=5, scoring="accuracy"
)

# Plot to see the bias-variance tradeoff visually
import matplotlib.pyplot as plt
plt.plot(depths, train_scores.mean(axis=1), label="Train")
plt.plot(depths, val_scores.mean(axis=1), label="Validation")
plt.xlabel("Max Depth"), plt.ylabel("Accuracy"), plt.legend()
plt.show()
```

---

Next: [Exercise 4 — Ensemble Methods →](./04-ensemble-methods.md)
