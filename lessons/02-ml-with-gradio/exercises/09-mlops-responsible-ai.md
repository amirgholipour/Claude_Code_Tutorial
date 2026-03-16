# Exercise 9: MLOps, Responsible AI & Advanced Topics

## Goal

Use Modules 21–26 to understand production ML: monitoring models in production, building automated pipelines, ensuring fairness, making smart system design decisions, tuning hyperparameters, and detecting anomalies.

## What You'll Learn

- How to detect data drift and decide when to retrain
- How to automate model quality gates with CI/CD
- How to measure and mitigate bias in ML models
- How to design ML systems for production constraints
- How to efficiently search for optimal hyperparameters
- How to detect anomalies without labeled data

---

## Part A — MLOps & Model Monitoring (Module 21)

**Open "🚀 21 · MLOps & Monitoring" tab**

### Experiment 1: Data Drift Detection

Demo: **Data Drift Detection (PSI)**, Drift Level: **0.0** → Run
→ Increase to **1.0** → Run
→ Increase to **3.0** → Run

Observe:
- At drift=0: All PSI scores near 0 (green) → ✅ No retrain needed
- At drift=1: Some features show moderate drift (orange)
- At drift=3: Many features critical PSI ≥ 0.2 (red) → 🔴 RETRAIN RECOMMENDED

**PSI thresholds:**
- PSI < 0.1: Safe ✅
- 0.1 ≤ PSI < 0.2: Monitor ⚠️
- PSI ≥ 0.2: Retrain 🔴

### Experiment 2: Model Version Comparison

Demo: **Model Version Comparison** → Run

See how v1 (shallow), v2 (deeper), v3 (full depth) compare on accuracy and training time.

**Production insight**: In a real system, you'd A/B test these versions and deploy the one with the best business metric (not just offline accuracy).

### Experiment 3: Prediction Distribution Shift

Demo: **Prediction Distribution Shift**, Drift Level: **0.0** → **3.0** → Run

When features drift, prediction score distributions shift too. This is **unsupervised monitoring** — you can detect it *before* you receive ground truth labels (which may arrive days/weeks later in production).

---

## Part B — CI/CD for ML (Module 22)

**Open "🔄 22 · CI/CD for ML" tab**

### Experiment 4: Automated Quality Gates

Dataset: **Breast Cancer**, Model: **Random Forest**, Accuracy Gate: **0.90** → Run

Each pipeline stage either passes (✅) or fails (❌):
1. **Data Validation**: Schema, nulls, range checks, class balance
2. **Baseline**: Logistic Regression baseline established
3. **Model Training**: Candidate model trained
4. **Gates**: Accuracy, F1, baseline comparison, CV stability, latency
5. **Deploy**: Only if ALL gates pass

### Experiment 5: Stress Test the Gates

1. Accuracy Gate: **0.99** → Run → Gates should fail (no model reaches 99%)
2. Accuracy Gate: **0.80** → Run → Gates should pass easily
3. Model: **Logistic Regression**, Gate: **0.95** → May fail depending on dataset

**Key lesson**: The accuracy gate threshold is a business decision — set too low and you deploy bad models; set too high and you block good ones.

---

## Part C — Responsible AI (Module 23)

**Open "⚖️ 23 · Responsible AI" tab**

### Experiment 6: Observe Bias

Demo: **Fairness Metrics Comparison**, Bias Strength: **0.0** → Run
→ Bias Strength: **1.0** → Run
→ Bias Strength: **2.0** → Run

At bias=0: Both groups have similar approval rates → Disparate Impact ≈ 1.0 ✅
At bias=2: Group B has much lower approval rate → Disparate Impact < 0.8 ❌

The model learned historical discrimination from the training data.

### Experiment 7: Mitigation

1. Bias Strength: **2.0**, Threshold Adjust: **OFF** → Run (observe DI score)
2. Bias Strength: **2.0**, Threshold Adjust: **ON** → Run

Post-processing threshold adjustment improves DI by lowering the decision threshold for the disadvantaged group.

**Important**: This is a technical fix. The root cause is biased training data. Technical mitigation helps but doesn't eliminate the need for better data collection.

### Experiment 8: Intersectional Analysis

Demo: **Subgroup Performance**, Bias Strength: **1.5** → Run

Observe: Fairness issues are often concentrated in intersections — e.g., "low-income Group B" may have significantly worse outcomes than "high-income Group B" or "low-income Group A".

**Lesson**: Always analyze subgroup × subgroup intersections, not just individual dimensions.

---

## Part D — ML System Design (Module 24)

**Open "🏗️ 24 · System Design" tab**

### Experiment 9: Latency vs Accuracy Tradeoff

Scenario: **Latency vs Accuracy Tradeoff**, Traffic: **200 req/sec**

1. Caching: OFF, Batching: OFF → baseline
2. Caching: ON → observe p50/p99 latency reduction + cost reduction
3. Batching: ON → observe throughput increase

**Real-world decision**: If your SLA requires p99 < 10ms, you must use a Small model. If you can tolerate 100ms, choose Medium for better accuracy.

### Experiment 10: Batch vs Online

Scenario: **Batch vs Online Comparison** → Run

Study the 4 serving patterns:
- **Batch** (nightly): Cheap, high throughput, stale predictions
- **Near-Real-Time** (minutes): Good balance for most products
- **Online** (<100ms): Required for fraud detection, search
- **Edge** (<10ms): Device-level, privacy-preserving

**Rule of thumb**: Start with batch, move to online only when business demands it.

### Feature Store Architecture

Scenario: **Feature Store Architecture** → Run

The diagram shows why feature stores solve the #1 production ML problem: **training-serving skew** — when features computed at training time differ from features computed at serving time.

---

## Part E — Hyperparameter Tuning (Module 25)

**Open "🔧 25 · Hyperparameter Tuning" tab**

### Experiment 11: Validation Curve (bias vs variance)

Dataset: **Breast Cancer**, Demo: **Validation Curve** → Run

Read the chart:
- **Left side** (small max_depth): Both train and val are low → **underfitting**
- **Right side** (large max_depth): Train high, val drops → **overfitting**
- **Sweet spot**: Where validation score peaks

### Experiment 12: Learning Curve

Demo: **Learning Curve** → Run

If the validation score is still rising at the right edge → more data would help
If it has plateaued → data alone won't help; need a better model/features

### Experiment 13: Grid vs Random Search

Demo: **Grid vs Random Search**, Iterations: **20** → Run

Compare:
- How many configs each method evaluated
- Final best score achieved
- Time taken

Increase iterations to 50 → random search finds better configs with more exploration.

---

## Part F — Anomaly Detection (Module 26)

**Open "🚨 26 · Anomaly Detection" tab**

### Experiment 14: Method Comparison

Dataset: **Two Clusters**, Contamination: **0.06**

1. Method: **Isolation Forest** → Run
2. Method: **Local Outlier Factor** → Run
3. Method: **One-Class SVM** → Run

Compare F1 scores. On the "Two Clusters" dataset, all methods should perform similarly.

### Experiment 15: LOF Advantage

Dataset: **Ring (LOF advantage)** → Run all three methods

LOF should outperform Isolation Forest on this dataset because anomalies are defined *relative to local density* — the ring structure means global density is misleading.

**Key insight**: Isolation Forest assumes anomalies are globally sparse. LOF works with local structure. Choose based on your data geometry.

### Experiment 16: Tuning Contamination

Dataset: **Two Clusters**, Method: **Isolation Forest**

1. Contamination: **0.02** → Run (under-estimates anomalies → misses some)
2. Contamination: **0.10** → Run (over-estimates → too many false alarms)
3. Contamination: **0.06** → Run (matches true rate)

Setting contamination = your best estimate of the true anomaly fraction is critical for good precision/recall balance.

---

## Congratulations — Extended Course Complete!

You've now covered the **full ML lifecycle** from data to production:

| Phase | Modules |
|---|---|
| **Data** | EDA (01, 18), Prep (15), Features (16, 17) |
| **Classical ML** | Regression (02), Classification (03), Evaluation (04) |
| **Ensemble** | Random Forest, GBT, AdaBoost (05) |
| **Unsupervised** | Clustering (06), Anomaly Detection (26) |
| **Dimensionality** | PCA, t-SNE, LDA (07) |
| **Deep Learning** | MLP (08), CNN (09), RNN (10), Training (11) |
| **Advanced ML** | Transfer (12), XAI (13), Pipeline (14) |
| **Specialized** | Time Series (19), NLP (20) |
| **Engineering** | MLOps (21), CI/CD (22), Hyperparameter Tuning (25) |
| **Responsible** | Fairness & Bias (23), System Design (24) |

**What's next?** Pick one topic and go deeper:
- Time series: Add ARIMA via `statsmodels`, Prophet for forecasting
- NLP: Upgrade to transformer embeddings (HuggingFace `sentence-transformers`)
- MLOps: Integrate MLflow for real experiment tracking
- Anomaly detection: Add autoencoder-based detection for high-dimensional data

---

[← Exercise 8: Time Series & NLP](./08-time-series-nlp.md) | [← Back to Lesson 2 Overview](../README.md)
