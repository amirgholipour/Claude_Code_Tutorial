# Exercise 7: Data Preparation, Feature Engineering & Selection

## Goal

Use Modules 15–18 to master the data foundation skills that every ML project depends on: cleaning messy data, creating powerful features, selecting the best ones, and understanding your data deeply through advanced EDA.

## What You'll Learn

- How to handle missing values, outliers, and encoding in real data
- How to create new features that improve model performance
- How to select the most informative features without overfitting
- How to perform rigorous statistical EDA beyond basic summaries

---

## Part A — Data Preparation (Module 15)

**Open "🧹 15 · Data Preparation" tab**

### Experiment 1: Missing Value Imputation

1. Imputation: **Mean**, Outliers: **None**, Encoding: **One-Hot** → Run
2. Imputation: **KNN**, Outliers: **None**, Encoding: **One-Hot** → Run
3. Compare the histograms — KNN imputation better preserves the distribution shape

**Takeaway**: Mean imputation is fast but narrows distributions. KNN uses spatial structure for more realistic imputations.

### Experiment 2: Outlier Handling

1. Imputation: **Median**, Outliers: **IQR Cap**, Encoding: **One-Hot** → Run
2. Imputation: **Median**, Outliers: **Z-score Remove**, Encoding: **One-Hot** → Run

Observe: IQR Cap keeps all rows (just clips extreme values); Z-score Remove deletes rows.

**When to cap vs remove:**
- **Cap**: When outliers are measurement errors or data entry mistakes
- **Remove**: When outliers are genuine anomalies you don't want to model

### Experiment 3: Encoding Impact

1. Imputation: **Median**, Outliers: **IQR Cap**, Encoding: **Label** → Run
2. Imputation: **Median**, Outliers: **IQR Cap**, Encoding: **One-Hot** → Run

OHE adds more columns but avoids implying an artificial order between categories.

---

## Part B — Feature Engineering (Module 16)

**Open "⚙️ 16 · Feature Engineering" tab**

### Experiment 4: Polynomial Features

1. Technique: **Polynomial Features**, Degree: **2** → Run
2. Degree: **3** → Run
3. Compare R² improvement vs feature count explosion

**Key insight**: Degree-2 often helps. Degree-3+ risks overfitting on small datasets and creates thousands of features.

### Experiment 5: Log Transform

1. Technique: **Log Transform** → Run
2. Look at the distribution comparison — right-skewed features become more symmetric

The Diabetes dataset contains skewed medical measurements. Log transform often helps linear models significantly.

### Experiment 6: Cyclical Encoding

1. Technique: **Cyclical Encoding (Hour)** → Run
2. Observe the circular scatter plot — hours form a ring

**Why this matters**: If you encode hour=23 and hour=0 as integers (23 and 0), the model thinks they're far apart. Cyclical sin/cos encoding correctly represents their closeness.

---

## Part C — Feature Selection (Module 17)

**Open "🎯 17 · Feature Selection" tab**

### Experiment 7: Filter vs Wrapper vs Embedded

Dataset: **Breast Cancer** (30 features), k: **10**

1. Method: **Filter — ANOVA F-test** → Run — see which features are statistically most different across classes
2. Method: **Filter — Mutual Information** → Run — captures non-linear relationships
3. Method: **Wrapper — RFE** → Run — recursively eliminates weakest features using a Random Forest
4. Method: **Embedded — Tree Importance** → Run — feature importances from the model itself

Compare which features each method selects. They often agree on the most important features but disagree on borderline ones.

### Experiment 8: Finding the Right k

1. Set k: **5** → Run all 4 methods
2. Set k: **20** → Run all 4 methods

**Key question**: Does accuracy improve going from 10 to 20 features? If not, you're adding noise.

### How to read the results

- **Green bars**: Selected features (included in model)
- **Red bars**: Rejected features (excluded)
- **Right chart**: Does using fewer features hurt accuracy? If not, the excluded features were noise.

---

## Part D — Advanced EDA (Module 18)

**Open "🔬 18 · Advanced EDA" tab**

### Experiment 9: Distribution Analysis

Dataset: **Breast Cancer**, EDA Type: **Distribution Analysis**, Feature Index: **0** → Run

Read the stats:
- Skewness > 1 → right-skewed → consider log transform
- Normality test p < 0.05 → NOT normally distributed → tree-based models may be better than linear

Try features 1, 5, 10 — they have very different distributions.

### Experiment 10: Correlation Heatmap

EDA Type: **Correlation Heatmap** → Run

Identify:
- Highly correlated feature pairs (|r| > 0.8) — consider dropping one
- Features with near-zero correlation to others — might be uninformative

### Experiment 11: Class Distribution

EDA Type: **Class Distribution** → Run

Check:
- Balance ratio (minority/majority) — if < 0.5, consider class weights or oversampling
- How does each class differ on the selected feature? (violin plot)

### Experiment 12: Outlier Summary

EDA Type: **Outlier Summary** → Run

Identify which features have the most outliers. High-outlier features are candidates for:
- Capping (IQR/Z-score) — Module 15
- Log transform — Module 16
- Closer inspection (are they real? errors? rare events?)

---

## Summary: The Data Preparation Workflow

```
Raw Data
   ↓
Advanced EDA (Module 18)     ← Understand distributions, correlations, outliers
   ↓
Data Preparation (Module 15) ← Impute missing, handle outliers, encode
   ↓
Feature Engineering (Module 16) ← Create polynomial, transform, cyclical features
   ↓
Feature Selection (Module 17)   ← Remove noise, reduce dimensions
   ↓
Model Training (Modules 02–14)  ← Clean, informative feature set
```

This pipeline prevents the two most common ML failures:
1. **Garbage in, garbage out**: Dirty data makes all models perform poorly
2. **Overfitting**: Too many irrelevant features confuse models

---

[← Back to Lesson 2 Overview](../README.md) | [→ Exercise 8: Time Series & NLP](./08-time-series-nlp.md)
