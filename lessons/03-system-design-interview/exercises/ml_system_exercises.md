# ML System Design Exercises

Practice exercises for Module 03 (ML System Design).
Target companies: Google, Meta, Amazon, Microsoft, Nvidia.

---

## Exercise 1 — Design a Content Recommendation System (YouTube/Netflix Level)

**Prompt:**
Design an ML system for YouTube-scale video recommendations:
- 2.7B monthly active users, 500 hours of video uploaded per minute
- Recommend 20 videos for each homepage load
- Optimize for watch time + satisfaction (not just clicks)
- Cold start for new users and new videos
- A/B testing infrastructure for model experimentation

**Your task:**
1. Design the two-tower retrieval + ranking architecture
2. What features would you use for the candidate retrieval model?
3. How would you handle the cold start problem for new videos (< 100 views)?
4. Design the training pipeline: data collection, labeling, training cadence
5. How would you measure success beyond CTR?
6. Describe your A/B testing setup and how you'd decide when to ship a new model

**Key concepts to demonstrate:**
- Two-stage retrieval (ANN + ranking)
- Implicit vs. explicit feedback
- Exploration vs. exploitation (bandits/epsilon-greedy)
- Position bias correction in training data

---

## Exercise 2 — Design a Fraud Detection System (Amazon/Stripe Level)

**Prompt:**
Design an ML system for real-time payment fraud detection:
- 10,000 transactions per second globally
- Must decide in < 200ms (blocks payment if flagged)
- Fraud rate is 0.1% — extreme class imbalance
- False positive rate must be < 0.5% (legitimate users blocked = bad UX)
- Fraud patterns evolve weekly — model must adapt

**Your task:**
1. What ML model family would you choose and why? (rules-based vs. gradient boosting vs. deep learning)
2. Design the feature engineering pipeline for a transaction
3. How would you handle class imbalance in training?
4. Design the online serving architecture to meet the 200ms SLA
5. How would you detect when the model is drifting (fraud patterns changing)?
6. Describe your labeling strategy — labels arrive days after transactions

**Key concepts to demonstrate:**
- Streaming feature computation
- Delayed labeling and temporal validation splits
- Model monitoring: data drift, concept drift
- Threshold calibration for precision-recall trade-off

---

## Exercise 3 — Design a Search Ranking System (Google/Bing Level)

**Prompt:**
Design the ML system powering web search ranking:
- 8.5B searches per day
- Rank 1,000 candidate documents → top 10 results
- Signals: query-document relevance, user engagement, page authority, freshness
- Latency budget: 100ms total for retrieval + ranking
- Different query intents: navigational, informational, transactional

**Your task:**
1. Design the multi-stage ranking pipeline (retrieval → L1 → L2 → L3)
2. What learning-to-rank (LtR) approach would you use? Pointwise / Pairwise / Listwise?
3. How would you collect training labels? (human raters, click data, both?)
4. Design the feature set for the ranker (query features, doc features, query-doc features)
5. How would you handle query understanding (spelling, synonyms, intent)?
6. Describe position bias in click data and how to correct for it

**Key concepts to demonstrate:**
- Learning-to-rank: LambdaMART, Neural LtR
- NDCG as the optimization metric
- Inverse propensity scoring for debiasing
- Query expansion and semantic matching

---

## Exercise 4 — Design an Ad Click-Through Rate (CTR) Prediction System (Meta/Google Ads Level)

**Prompt:**
Design a CTR prediction system for a digital advertising platform:
- 10M ads in inventory, 100B ad impressions/day
- Predict P(click | user, ad, context) for each candidate ad
- Optimize for revenue (bid × predicted CTR)
- Cold start for new advertisers and new ads
- Privacy constraints: GDPR/CCPA — no cross-site tracking in some markets

**Your task:**
1. Choose a model architecture: Logistic Regression vs. GBDT vs. Deep FM vs. Transformer
2. Design the feature set: user features, ad features, contextual features
3. How would you handle the massive categorical feature cardinality? (user IDs, ad IDs)
4. Design the feature store and serving pipeline for < 50ms inference
5. How would you handle the privacy constraints? (federated learning, differential privacy, etc.)
6. Describe your calibration strategy — predicted 0.05 CTR should mean 5% of impressions click

**Key concepts to demonstrate:**
- Embedding tables for high-cardinality categoricals
- Feature hashing trick
- Log-likelihood calibration with Platt scaling
- Online learning for real-time model updates

---

## Exercise 5 — Design an ML Monitoring & Retraining System

**Prompt:**
Design the MLOps infrastructure for monitoring and automatically retraining production ML models:
- 50 models in production, ranging from fraud detection to recommendations
- Models have different retraining cadences: real-time, daily, weekly
- You want to detect model degradation before users notice
- Automated retraining should not ship regressions

**Your task:**
1. What metrics would you monitor for each of: data quality, model quality, business impact?
2. Design the alerting thresholds — how would you avoid alert fatigue?
3. Design the automated retraining trigger: time-based vs. drift-based vs. performance-based
4. Describe the champion/challenger evaluation framework before promoting a new model
5. How would you handle a rollback if a newly promoted model regresses in production?
6. Design the lineage tracking system: which training data, features, and code produced each model version?

**Key concepts to demonstrate:**
- PSI (Population Stability Index) for drift detection
- Shadow mode deployment
- Canary releases for ML models
- Feature and model lineage (MLflow, Weights & Biases)

---

## Scoring Rubric

| Criterion | Description | Max Points |
|---|---|---|
| Problem scoping | Clarified scale, constraints, and success metrics before designing | 3 |
| Architecture clarity | Clear multi-component diagram with data flow | 3 |
| ML model choice | Justified choice with trade-offs vs. alternatives | 3 |
| Feature engineering | Realistic, thoughtful feature set with transformations | 3 |
| Failure analysis | Data drift, cold start, latency spikes, label delay | 3 |

**Target for FAANG ML Engineer interview:** 12/15 or higher

---

## Further Reading

- [Machine Learning System Design Interview](https://www.educative.io/courses/machine-learning-system-design) — Educative
- [Applied ML at Facebook](https://research.fb.com/blog/2017/02/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective/)
- [Google Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [Uber Engineering: Michelangelo ML Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Netflix: System Architectures for Personalization and Recommendation](https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8)
