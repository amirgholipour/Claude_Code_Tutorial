# Exercise 2: Data Exploration

## Goal

Use Module 01 in the Gradio app to perform EDA (Exploratory Data Analysis) on real ML datasets and understand what to look for before training a model.

## What You'll Learn

- Why EDA is the first step in every ML project
- How to read distribution plots, correlation heatmaps, and class balance charts
- How feature scaling affects algorithms
- What "data leakage" is and why it ruins models

## Background

Before training any model, you need to understand your data:

1. **Shape** — how many samples and features?
2. **Types** — numeric, categorical, boolean?
3. **Missing values** — how many? How to handle them?
4. **Distributions** — normally distributed? Skewed? Outliers?
5. **Correlations** — which features are related?
6. **Class balance** — are classes evenly distributed?

Skipping EDA leads to bad models — garbage in, garbage out.

## Steps

### Step 1: Launch the app

```bash
cd lessons/02-ml-with-gradio/app
python app.py
```

Open `http://localhost:7860` and click the **"📊 01 · Data Exploration"** tab.

### Step 2: Explore distributions

1. Select **Iris** dataset → **Distributions** → Run
2. Observe: 4 histogram subplots, one per feature
3. Switch to **Wine** → notice more features (13) → wine features have very different scales

**Question:** Why does feature scale matter?
- k-NN uses distance → large-scale features dominate
- Linear models: regularization treats all weights equally
- Tree models: scale-invariant (they just pick thresholds)

### Step 3: Examine correlations

1. Select **Breast Cancer** → **Correlation Heatmap** → Run
2. Find pairs with correlation > 0.9 (dark red cells)
3. These are **redundant features** — remove one to reduce noise

High correlation between features = **multicollinearity**. It doesn't hurt tree models but degrades linear models.

### Step 4: Check class balance

1. Select **Digits** → **Class Balance** → Run
2. All 10 digit classes should be roughly equal (~180 samples each)
3. Now try **Breast Cancer**: ~62% benign, ~38% malignant

**Class imbalance** matters because:
- A model that always predicts "benign" gets 62% accuracy — but is useless
- Use F1 score or ROC-AUC instead of accuracy for imbalanced data
- Techniques: oversample minority class (SMOTE), undersample majority, class_weight="balanced"

### Step 5: Manual EDA in Python

Try this yourself in a Python REPL or Jupyter notebook:

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Basic stats
print(df.describe().round(2))

# Missing values
print(df.isnull().sum().sum())  # Should be 0 for sklearn datasets

# Class balance
print(df['target'].value_counts(normalize=True).round(3))

# Feature correlations (top 5 most correlated pairs)
corr = df.drop('target', axis=1).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
top_pairs = upper.stack().sort_values(ascending=False).head(5)
print(top_pairs)
```

## Key Concepts

| Concept | What to Look For | Action |
|---|---|---|
| **Skewed distribution** | Long tail in histogram | Log transform the feature |
| **Outliers** | Points far from the bulk | Investigate — remove or clip |
| **High correlation** (>0.9) | Dark cells in heatmap | Drop one of the two features |
| **Class imbalance** | One class >> others | Use F1/AUC, or resampling |
| **Very different scales** | Wine features range from 0.01 to 1000 | Apply StandardScaler |
| **Missing values** | NaN counts | Impute (mean/median) or drop |

## The Data Leakage Trap

**Data leakage** = accidentally using test data information during training. This is the most common mistake in ML competitions and real projects.

```python
# WRONG — scaler learns from entire dataset including test set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)         # leakage!
X_train, X_test = train_test_split(X_scaled)

# CORRECT — scaler only learns from training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)    # learn from train only
X_test = scaler.transform(X_test)          # apply same transform to test
```

The `sklearn.Pipeline` (Module 14) prevents leakage automatically.

---

Next: [Exercise 3 — Classical ML →](./03-classical-ml.md)
