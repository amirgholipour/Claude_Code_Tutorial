# Exercise 8: Time Series Analysis & NLP

## Goal

Use Modules 19–20 to work with sequential and text data — two domains that require fundamentally different approaches from standard tabular ML.

## What You'll Learn

- How to decompose time series into trend, seasonality, and residual
- How to assess stationarity and use autocorrelation for model selection
- How to represent text as numbers (Bag-of-Words, TF-IDF)
- How to build a text classification pipeline

---

## Part A — Time Series Analysis (Module 19)

**Open "📈 19 · Time Series" tab**

### Experiment 1: Decomposition

Series: **Trend + Seasonal + Noise**, Analysis: **Decomposition** → Run

Observe the 4 panels:
1. **Original**: The raw time series
2. **Trend**: Long-term direction (upward, downward, flat)
3. **Seasonal**: Repeating pattern (period ≈ 30 days)
4. **Residual**: What's left after removing trend + seasonality — should look like random noise

**If residual has clear patterns**: The decomposition didn't capture all structure → try different period.

### Experiment 2: Compare Series Types

Run Decomposition on each series:

| Series | Expected trend? | Strong seasonal? | Stationary? |
|---|---|---|---|
| Trend + Seasonal + Noise | Yes | Yes | No (ADF fails) |
| Strong Seasonality | No | Yes (two periods) | Varies |
| Random Walk | No (but drifts) | No | No (classic non-stationary) |
| Step + Trend | Step at midpoint | No | No |

**Key insight**: Random Walk is the classic non-stationary series. Differencing once (y'(t) = y(t) - y(t-1)) makes it stationary.

### Experiment 3: Forecasting with Naive Methods

Series: **Trend + Seasonal + Noise**, Analysis: **Forecast (Naive Methods)**, Horizon: **30** → Run

Compare the 4 methods:
- **Naive**: Just repeats the last value — terrible for trending series
- **Seasonal Naive**: Repeats the value from 30 days ago — good when seasonality dominates
- **Moving Average**: Smoothed level — good for stable series
- **Linear Extrapolation**: Best when trend is strong and linear

Change to **Strong Seasonality** → Seasonal Naive should win.
Change to **Random Walk** → Moving Average should be competitive.

### Experiment 4: Autocorrelation

Series: **Trend + Seasonal + Noise**, Analysis: **Autocorrelation (ACF/PACF)** → Run

**Reading ACF:**
- Bars above/below the red dashed confidence lines are statistically significant
- Slowly decaying ACF → non-stationary (needs differencing)
- Sharp cutoff at lag q in ACF → MA(q) process
- Sharp cutoff at lag p in PACF → AR(p) process

**ARIMA parameter selection guide:**
```
ACF cuts at q, PACF decays → MA(q) process    → ARIMA(0,0,q)
PACF cuts at p, ACF decays → AR(p) process    → ARIMA(p,0,0)
Both decay slowly           → Mixed ARMA       → ARIMA(p,0,q)
Series non-stationary       → Difference first → ARIMA(p,1,q)
```

---

## Part B — Natural Language Processing (Module 20)

**Open "📝 20 · NLP" tab**

> **Note**: First run downloads the 20 Newsgroups dataset (~15 MB). This takes 10–30 seconds once, then it's cached.

### Experiment 5: Word Frequency Analysis

Demo: **Word Frequency Analysis**, Categories: **4** → Run

Observe the top 20 words after stopword removal. These are the most common domain words across all categories.

**Question**: Are these words useful for classification? If a word appears equally in all categories (like "people", "think", "know"), it's not a good discriminator — this is why TF-IDF down-weights common words.

### Experiment 6: TF-IDF Words per Category

Demo: **Top TF-IDF Words per Category** → Run

Each bar shows the mean TF-IDF score for words that are most characteristic of that category:
- `hockey`, `puck`, `goalie` → rec.sport.hockey
- `nasa`, `orbit`, `shuttle` → sci.space
- `gun`, `weapons`, `firearms` → talk.politics.guns
- `image`, `graphics`, `pixel` → comp.graphics

**Key insight**: TF-IDF correctly identifies category-specific vocabulary because these words are frequent in one category but rare elsewhere.

### Experiment 7: Classifier Comparison

Demo: **Text Classification**, Classifier: **Naive Bayes** → Run
→ Then **Logistic Regression** → Run
→ Then **Linear SVM** → Run

Compare accuracy and the confusion matrix:
- Which categories are most confused? (off-diagonal in confusion matrix)
- Does the SVM outperform Naive Bayes? (usually yes on text)
- Which classifier has the highest F1 across all categories?

**Expected ranking**: Linear SVM ≈ Logistic Regression > Naive Bayes (on most text tasks)

### Experiment 8: Hyperparameter Effect

Demo: **Text Classification**, Classifier: **Logistic Regression**

1. Max TF-IDF Features: **1000** → Run
2. Max TF-IDF Features: **10000** → Run
3. Max TF-IDF Features: **20000** → Run

At what point does adding more features stop helping? Too many features = more noise, more memory, slower training.

### Categories: 2 vs 4

1. Categories: **2** → Accuracy should be very high (easier binary task)
2. Categories: **4** → Accuracy drops (4-class is harder)

**Rule**: Multi-class accuracy decreases as n_classes increases (random baseline = 1/n_classes).

---

## Key Concepts Summary

### Time Series
```
Non-stationary time series:
  • Has trend or seasonality → ADF test fails (p > 0.05)
  • Must be differenced before ARIMA

Model selection from ACF/PACF:
  PACF sharp cutoff at p → AR(p)
  ACF sharp cutoff at q  → MA(q)
  Both gradual           → ARMA(p,q)
```

### NLP Pipeline
```
Raw text
  → Lowercase + remove punctuation
  → Remove stopwords (the, a, is...)
  → Tokenize (split into words)
  → TF-IDF vectorization (word importance scores)
  → Classifier (Naive Bayes / Logistic Regression / SVM)
  → Predicted category
```

### When text data is hard
- Very short texts (tweets) → TF-IDF sparse, needs embeddings (Word2Vec, BERT)
- Multiple languages → needs language-specific preprocessing
- Slang/misspellings → needs spell correction or subword tokenization
- Domain-specific vocabulary → stop words list needs customization

---

[← Exercise 7: Data Preparation](./07-data-preparation.md) | [→ Exercise 9: MLOps & Responsible AI](./09-mlops-responsible-ai.md)
