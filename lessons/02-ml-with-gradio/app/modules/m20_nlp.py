"""
Module 20 — Natural Language Processing (NLP)
Level: Intermediate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

THEORY = """
## 📖 What Is NLP?

**Natural Language Processing (NLP)** is the field of enabling machines to understand, interpret, and generate human language. It sits at the intersection of linguistics, computer science, and machine learning.

NLP powers: search engines, spam filters, sentiment analysis, chatbots, translation, summarization.

## 🏗️ NLP Pipeline

```
Raw Text → Preprocessing → Representation → Model → Output
```

### 1. Text Preprocessing
```python
# Lowercasing
text = text.lower()
# Remove punctuation/special chars
text = re.sub(r"[^a-z0-9 ]", "", text)
# Tokenization (split into words)
tokens = text.split()
# Remove stopwords (common words with little meaning)
stopwords = {"the", "a", "is", "in", "it", "and", "or", "to", ...}
tokens = [t for t in tokens if t not in stopwords]
# Stemming: "running" → "run" (crude, suffix stripping)
# Lemmatization: "better" → "good" (vocabulary-based, more accurate)
```

### 2. Text Representation

#### Bag of Words (BoW)
Count how many times each word appears in a document. Ignores order.
```
["I love cats", "I love dogs"]
→ {"i": [1,1], "love": [1,1], "cats": [1,0], "dogs": [0,1]}
```

#### TF-IDF (Term Frequency — Inverse Document Frequency)
Weights terms by how important they are to a document relative to the corpus:
```
TF(w, d) = count(w in d) / total_words(d)
IDF(w)   = log(N / df(w))    # df = documents containing w
TF-IDF   = TF × IDF
```
**Key insight**: Common words (the, a, is) get low IDF; rare discriminative words get high score.

#### Word Embeddings (Word2Vec, GloVe, FastText)
Dense vectors where similar words are close in space:
```
king - man + woman ≈ queen
```
More powerful than BoW/TF-IDF — captures semantic meaning.

#### Transformers (BERT, GPT)
Context-aware embeddings — same word gets different vector based on surrounding words.

### 3. Classic NLP Models
| Model | Strengths | Weaknesses |
|---|---|---|
| Naive Bayes | Fast, works with small data | Assumes feature independence |
| Logistic Regression + TF-IDF | Interpretable, strong baseline | Linear decision boundary |
| Linear SVM | High accuracy for text | No probability output |
| Random Forest | Handles interactions | Slow on high-dimensional text |

## ✅ Text Classification Workflow
1. Load and explore data
2. Preprocess text (lowercase, clean, tokenize)
3. Vectorize (TF-IDF or BoW)
4. Train classifier
5. Evaluate (accuracy, F1, confusion matrix)
6. Inspect misclassified examples

## ⚠️ Common Pitfalls
- **Vocabulary leakage**: Fitting vectorizer on full dataset → fit on train only
- **Ignoring class imbalance**: Macro F1 is more informative than accuracy for imbalanced classes
- **Stopword removal**: Domain-specific stopwords matter (e.g., "not" is crucial for sentiment)
- **Short texts**: Bag-of-words fails on very short texts (tweets, titles)
"""

CODE_EXAMPLE = '''
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── Load data ─────────────────────────────────────────────────────
categories = ["rec.sport.baseball", "rec.sport.hockey",
              "sci.space", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups(subset="test",  categories=categories,
                           remove=("headers", "footers", "quotes"))

# ── Pipeline: TF-IDF + Naive Bayes ───────────────────────────────
nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf",   MultinomialNB()),
])
nb_pipe.fit(train.data, train.target)
print("NB Accuracy:", accuracy_score(test.target, nb_pipe.predict(test.data)))

# ── Pipeline: TF-IDF + Logistic Regression ────────────────────────
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf",   LogisticRegression(max_iter=1000, C=5.0)),
])
lr_pipe.fit(train.data, train.target)
print("LR Accuracy:", accuracy_score(test.target, lr_pipe.predict(test.data)))
print(classification_report(test.target, lr_pipe.predict(test.data),
                             target_names=categories))
'''


# Simple English stopwords (no external dependency)
STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "with", "this", "that", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "by", "from", "as", "i", "we",
    "you", "he", "she", "they", "my", "your", "our", "not", "no", "all", "more",
    "so", "if", "its", "which", "who", "what", "when", "where", "than", "then",
    "there", "their", "about", "up", "out", "also", "just", "one", "two", "first",
    "s", "re", "ve", "ll", "d", "m", "t", "don"
}


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def _load_newsgroups(n_categories: int = 4):
    # Use 4 diverse topic categories
    categories = [
        "rec.sport.hockey",
        "sci.space",
        "talk.politics.guns",
        "comp.graphics",
    ][:n_categories]

    try:
        train_data = fetch_20newsgroups(
            subset="train", categories=categories,
            remove=("headers", "footers", "quotes"), random_state=42
        )
        test_data = fetch_20newsgroups(
            subset="test", categories=categories,
            remove=("headers", "footers", "quotes"), random_state=42
        )
        return train_data, test_data, categories
    except Exception:
        # Fallback: synthetic data if download fails
        return None, None, categories


def run_nlp_demo(demo_type: str, clf_name: str, max_features: int, n_categories: int):
    train_data, test_data, categories = _load_newsgroups(n_categories)

    if train_data is None:
        return go.Figure(), "❌ Could not load 20 Newsgroups dataset. Check internet connection."

    cat_labels = [c.split(".")[-1] for c in categories[:n_categories]]

    if demo_type == "Text Classification":
        CLFS = {
            "Naive Bayes":         MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, C=5.0),
            "Linear SVM":          LinearSVC(max_iter=2000, C=1.0),
        }
        clf = CLFS[clf_name]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features,
                                      stop_words="english", sublinear_tf=True)),
            ("clf",   clf),
        ])
        pipe.fit(train_data.data, train_data.target)
        preds = pipe.predict(test_data.data)
        acc   = accuracy_score(test_data.target, preds)
        cm    = confusion_matrix(test_data.target, preds)

        fig = go.Figure(go.Heatmap(
            z=cm, x=cat_labels, y=cat_labels,
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
            colorbar=dict(title="Count")
        ))
        fig.update_layout(height=420, title_text=f"Confusion Matrix — {clf_name}")

        report = classification_report(test_data.target, preds,
                                       target_names=cat_labels, output_dict=True)
        rows = "\n".join([
            f"| {c} | {report[c]['precision']:.3f} | {report[c]['recall']:.3f} | {report[c]['f1-score']:.3f} | {int(report[c]['support'])} |"
            for c in cat_labels
        ])
        metrics_md = f"""
### {clf_name} — {n_categories} Categories

**Accuracy**: `{acc:.4f}`

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

> TF-IDF features: `{max_features}` | Dataset: 20 Newsgroups (subset)
"""

    elif demo_type == "Top TF-IDF Words per Category":
        vect = TfidfVectorizer(max_features=max_features, stop_words="english", sublinear_tf=True)
        X    = vect.fit_transform(train_data.data)
        vocab = np.array(vect.get_feature_names_out())

        n_cats = min(n_categories, len(categories))
        fig = make_subplots(rows=1, cols=n_cats,
                            subplot_titles=cat_labels[:n_cats])

        for i, (cat_idx, cat_name) in enumerate(zip(range(n_cats), cat_labels[:n_cats])):
            mask     = train_data.target == cat_idx
            X_cat    = X[mask]
            mean_tfidf = np.asarray(X_cat.mean(axis=0)).flatten()
            top10    = np.argsort(mean_tfidf)[-10:][::-1]
            words    = vocab[top10]
            scores   = mean_tfidf[top10]

            fig.add_trace(go.Bar(
                x=scores[::-1], y=words[::-1],
                orientation="h", name=cat_name,
                marker_color=["#42a5f5", "#66bb6a", "#ffa726", "#ef5350"][i % 4]
            ), row=1, col=i+1)

        fig.update_layout(height=420, showlegend=False,
                          title_text="Top TF-IDF Words per Category")

        metrics_md = f"""
### Top Discriminative Words

Each bar shows the mean TF-IDF score across all documents in that category.

**How to read:** A high mean TF-IDF means the word appears frequently in this category
and rarely in others — it's a strong category signal.

> Dataset: 20 Newsgroups | Categories: {', '.join(cat_labels[:n_cats])}
"""

    elif demo_type == "Word Frequency Analysis":
        # Combine all training texts and compute word freq per category
        all_words = Counter()
        for doc in train_data.data:
            cleaned = _clean_text(doc)
            all_words.update(cleaned.split())

        top_words = all_words.most_common(20)
        words     = [w for w, _ in top_words]
        freqs     = [c for _, c in top_words]

        fig = go.Figure(go.Bar(
            x=freqs[::-1], y=words[::-1],
            orientation="h",
            marker_color="#7e57c2",
            text=freqs[::-1], textposition="outside"
        ))
        fig.update_layout(height=450, title_text="Top 20 Words in Corpus (after stopword removal)")

        metrics_md = f"""
### Corpus Statistics

| Metric | Value |
|---|---|
| Total training documents | `{len(train_data.data)}` |
| Total test documents | `{len(test_data.data)}` |
| Unique words (vocab) | `{len(all_words)}` |
| Top word | `{words[0]}` ({freqs[0]} occurrences) |

> After stopword removal. These high-frequency words may need further domain-specific filtering.
"""

    else:
        fig = go.Figure()
        metrics_md = "Select a demo type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# 📝 Module 20 — Natural Language Processing\n*Level: Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nText classification on the 20 Newsgroups dataset. Compare NLP models and explore word importance.\n\n> **Note**: First run downloads the 20 Newsgroups dataset (~15 MB).")

    with gr.Row():
        with gr.Column(scale=1):
            demo_dd = gr.Dropdown(
                label="Demo Type",
                choices=["Text Classification", "Top TF-IDF Words per Category",
                         "Word Frequency Analysis"],
                value="Text Classification"
            )
            clf_dd = gr.Dropdown(
                label="Classifier",
                choices=["Naive Bayes", "Logistic Regression", "Linear SVM"],
                value="Logistic Regression"
            )
            feat_sl = gr.Slider(label="Max TF-IDF Features", minimum=1000, maximum=20000,
                                step=1000, value=5000)
            cat_sl  = gr.Slider(label="Number of Categories", minimum=2, maximum=4,
                                step=1, value=4)
            run_btn = gr.Button("▶ Run NLP Demo", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_nlp_demo,
        inputs=[demo_dd, clf_dd, feat_sl, cat_sl],
        outputs=[plot_out, metrics_out]
    )
