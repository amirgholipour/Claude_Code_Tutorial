# Exercise 6: Full ML Pipeline

## Goal

Use Modules 12–14 to understand transfer learning, explain model predictions, and build a complete end-to-end pipeline that can be saved and loaded for predictions.

## What You'll Learn

- When to use transfer learning vs training from scratch
- How to explain WHY a model made a prediction (SHAP values)
- How to build a production-ready sklearn Pipeline
- How to save and load models for deployment

---

## Part A — Transfer Learning (Module 12)

**Open "🔀 12 · Transfer Learning" tab**

### Experiment 1: Feature Extraction vs From Scratch

1. Strategy: **Feature Extraction**, n_samples: **50** → Run
2. Strategy: **From Scratch**, n_samples: **50** → compare accuracy

With only 50 labeled samples, feature extraction should win significantly. The pre-trained features (learned from more data) provide a better starting point than learning everything from scratch.

### Experiment 2: Data efficiency curve

Run the experiment with n_samples ranging from 10 to 200 and observe:
- At very small n (10-25): Transfer learning dominates
- At large n (200+): From-scratch can catch up
- **Transfer learning is most valuable when labeled data is scarce**

### Real-world analogy

```
Pre-training (abundant data):     → Model learns "cat ears", "dog faces", "edges"
Transfer (small labeled set):     → New task reuses those features → "cancer cell" patterns
From scratch (small labeled set): → Must learn everything from 50 examples → struggles
```

In practice: always start with a pre-trained model when your dataset is small (<10k examples).

---

## Part B — Explainability (Module 13)

**Open "💡 13 · Explainability" tab**

### Experiment 3: Feature Importance vs Permutation Importance

1. Dataset: **Breast Cancer**, Method: **Feature Importance** → Run
   - Shows how often each feature was used to split in the Random Forest
2. Method: **Permutation Importance** → Run
   - Shuffles each feature and measures accuracy drop
   - More reliable: tests actual impact on predictions

Sometimes they disagree. Permutation importance is generally more trustworthy.

### Experiment 4: SHAP values

1. Dataset: **Iris**, Method: **SHAP Values** → Run
   - Each dot = one sample
   - Color = feature value (red = high, blue = low)
   - X-axis = impact on prediction

**How to read a SHAP beeswarm plot:**
- Points to the right = pushed prediction toward that class
- Points to the left = pushed prediction away from that class
- Spread = feature has high impact on some samples

### Experiment 5: Partial Dependence Plot

1. Dataset: **Wine**, Method: **Partial Dependence** → Run
2. Shows how the model's prediction changes as one feature varies

PDP answers: "All else being equal, how does feature X affect the prediction?"

## Why Explainability Matters in Production

```
Regulatory: GDPR requires "right to explanation" for automated decisions
Medical:    A doctor must understand why the model flagged a scan as cancer
Business:   "Why did the model reject this loan application?"
Debugging:  Model has high accuracy but uses wrong features (spurious correlation)
```

---

## Part C — End-to-End Pipeline (Module 14)

**Open "🔧 14 · ML Pipeline" tab**

### Step 1: Build and train the pipeline

1. **Tab: Build Pipeline**
2. Dataset: **Iris**, Scaler: **StandardScaler**, Algorithm: **Random Forest**
3. CV Folds: **5** → Run & Save
4. See CV scores per fold + mean accuracy
5. Model is saved to `/tmp/ml_course_model.pkl`

### Step 2: Make a prediction

1. **Tab: Predict**
2. Use the sliders to input flower measurements:
   - Sepal length: **5.1**, Sepal width: **3.5**, Petal length: **1.4**, Petal width: **0.2**
   - → Should predict: **Setosa** (confident)
3. Try: Sepal length: **6.3**, Sepal width: **3.3**, Petal length: **6.0**, Petal width: **2.5**
   - → Should predict: **Virginica**

### Step 3: Understand the Pipeline advantage

```python
# Without Pipeline — data leakage risk!
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)   # sees test data during fitting!
X_train, X_test = train_test_split(X_all_scaled)
model.fit(X_train, y_train)

# With Pipeline — no leakage!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)  # scaler only sees X_train
pipeline.predict(X_test)        # scaler applies same transform to test
```

### Step 4: Save and load (the deployment pattern)

```python
import joblib

# Save the entire pipeline (scaler + model together)
joblib.dump(pipeline, 'model.pkl')

# Later, in production:
loaded = joblib.load('model.pkl')
prediction = loaded.predict([[5.1, 3.5, 1.4, 0.2]])
print(prediction)  # ['setosa']
```

`joblib` serializes the entire sklearn pipeline — scaler parameters AND model weights — into one file. When you load it, you get the exact same pipeline ready to predict.

---

## Congratulations — Lesson 2 Complete!

You've built a complete ML course app covering:

| Category | What you covered |
|---|---|
| **Data** | EDA, distributions, correlation, class balance, leakage |
| **Classical ML** | Regression, classification, evaluation metrics |
| **Ensemble** | Random Forest, Gradient Boosting, AdaBoost |
| **Unsupervised** | K-Means, DBSCAN, Agglomerative |
| **Dimensionality** | PCA, t-SNE, LDA |
| **Deep Learning** | MLP, CNN, LSTM/GRU, training tricks |
| **Advanced** | Transfer learning, SHAP, end-to-end pipeline |

**Next step:** Extend the app — add a new module for a topic you want to explore (XGBoost, VAEs, attention mechanisms, time series forecasting, etc.)

---

[← Back to Lesson 2 Overview](../README.md)
