# Exercise 4: Ensemble Methods & Unsupervised Learning

## Goal

Use Modules 05–07 to understand how combining models beats individual models, discover hidden structure in data without labels, and reduce high-dimensional data to 2D for visualization.

## What You'll Learn

- Why ensembles (Random Forest, Gradient Boosting) outperform single models
- How K-Means finds clusters without knowing labels
- How PCA compresses data while preserving structure

---

## Part A — Ensemble Methods (Module 05)

**Open "🌲 05 · Ensemble Methods" tab**

### Experiment 1: Compare All algorithms at once

1. Dataset: **Breast Cancer**, Algorithm: **Compare All** → Run
2. You'll see a bar chart of accuracy for Random Forest, Gradient Boosting, AdaBoost
3. Note which wins and by how much

Random Forest typically wins on this dataset because:
- Breast cancer has 30 features (enough to benefit from feature sampling)
- The data has some noise that boosting can overfit

### Experiment 2: n_estimators effect

1. Algorithm: **Random Forest**, n_estimators: **10** → Run, note accuracy
2. n_estimators: **100** → Run, note accuracy improvement
3. n_estimators: **200** → minimal additional gain (diminishing returns)

**Why more trees helps:** Each tree votes independently. As you add trees, the ensemble average converges and variance drops — but there's a ceiling.

### Experiment 3: Learning rate for Gradient Boosting

1. Algorithm: **Gradient Boosting**, learning_rate: **0.5** → Run
2. learning_rate: **0.05** → compare accuracy
3. Reduce learning_rate but increase n_estimators proportionally → often better

**Low learning rate + many trees = better generalization** (but slower to train).

---

## Part B — Clustering (Module 06)

**Open "🔵 06 · Clustering" tab**

### Experiment 4: K-Means finds natural groups

1. Dataset: **blobs**, Algorithm: **K-Means**, K: **3** → Run
2. The 3 clusters should be clearly separated
3. Try **moons** dataset with K=2 → K-Means fails! (curved clusters)

K-Means assumes **spherical, equal-size clusters**. When that assumption breaks, it fails.

### Experiment 5: Elbow method to choose K

1. Dataset: **blobs**, Algorithm: **K-Means**, enable **Show Elbow Curve** → Run
2. Plot shows inertia (within-cluster SSE) vs K
3. Find the "elbow" — where adding more clusters gives diminishing inertia reduction
4. The elbow at K=3 confirms there are 3 natural clusters

### Experiment 6: DBSCAN handles irregular shapes

1. Dataset: **moons**, Algorithm: **DBSCAN**, eps: **0.3**, min_samples: **5** → Run
2. DBSCAN perfectly finds the two moon-shaped clusters
3. Increase eps to **1.0** → everything merges into one cluster (too loose)
4. Decrease to **0.1** → many noise points (too strict)

DBSCAN's key strength: **finds outliers** as "noise" (shown in black).

---

## Part C — Dimensionality Reduction (Module 07)

**Open "🔍 07 · Dimensionality" tab**

### Experiment 7: PCA visualization

1. Dataset: **Wine** (13 features → can't visualize directly), Method: **PCA**, 2D → Run
2. You'll see a 2D scatter with 3 wine classes
3. Check how much variance the first 2 components explain

**If PCA explains 80%+ of variance in 2 components**, the dataset has low intrinsic dimensionality — many features are redundant.

### Experiment 8: PCA vs t-SNE

1. Dataset: **Digits** (64 features), Method: **PCA** → Run → note cluster separation
2. Same dataset, Method: **t-SNE** → Run (may take 20-30 seconds)
3. t-SNE typically shows cleaner separation of digit clusters

**Why?** PCA is linear (finds directions of max variance). t-SNE is nonlinear (preserves local neighborhood). For complex clusters, t-SNE wins visually.

**Warning:** t-SNE is for visualization only — never use it as preprocessing for ML models (it's non-deterministic and doesn't preserve global structure).

### Experiment 9: PCA for preprocessing (speed up ML)

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import time

X, y = load_digits(return_X_y=True)  # 64 features

# Without PCA
t0 = time.time()
score_raw = cross_val_score(SVC(), X, y, cv=5).mean()
t_raw = time.time() - t0

# With PCA (keep 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)  # reduces to ~40 components
t1 = time.time()
score_pca = cross_val_score(SVC(), X_pca, y, cv=5).mean()
t_pca = time.time() - t1

print(f"Raw: {score_raw:.4f} in {t_raw:.1f}s | PCA: {score_pca:.4f} in {t_pca:.1f}s")
```

Usually PCA gives similar accuracy at significantly less compute time.

## Summary: When to Use What

| Situation | Recommended Approach |
|---|---|
| Labeled data, want best accuracy | Gradient Boosting (tabular) |
| Labeled data, want fast + interpretable | Random Forest |
| No labels, need to find groups | K-Means (spherical clusters), DBSCAN (arbitrary shapes) |
| Too many features for ML | PCA (preprocessing), keep 95% variance |
| Want to visualize high-dimensional clusters | t-SNE (2D plot only) |

---

Next: [Exercise 5 — Deep Learning →](./05-deep-learning-intro.md)
