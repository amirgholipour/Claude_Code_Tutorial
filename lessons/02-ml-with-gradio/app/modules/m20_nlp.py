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
import random

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

THEORY = """
## What Is NLP?

**Natural Language Processing (NLP)** enables machines to understand, interpret,
and generate human language.  It powers search engines, spam filters, sentiment
analysis, chatbots, and more.

## The Classical NLP Pipeline

This module covers the classical NLP pipeline:

```
Raw Text  →  Preprocessing  →  TF-IDF Vectorization  →  Classification
```

### 1. Text Preprocessing
```python
# Lowercasing
text = text.lower()
# Remove punctuation / special chars
text = re.sub(r"[^a-z0-9 ]", "", text)
# Tokenization (split into words)
tokens = text.split()
# Remove stopwords (common words with little meaning)
stopwords = {"the", "a", "is", "in", "it", "and", "or", "to", ...}
tokens = [t for t in tokens if t not in stopwords]
```

### 2. TF-IDF Representation

**TF-IDF (Term Frequency — Inverse Document Frequency)** weights terms by how
important they are to a document relative to the entire corpus:

```
TF(w, d) = count(w in d) / total_words(d)
IDF(w)   = log(N / df(w))       # df = number of documents containing w
TF-IDF   = TF × IDF
```

**Key insight**: Common words (the, a, is) get low IDF; rare, discriminative
words get a high score.  This makes TF-IDF a strong baseline representation
for many text classification tasks.

### 3. Classifiers Used in This Demo

| Model | Strengths | Weaknesses |
|---|---|---|
| Naive Bayes | Fast, works with small data | Assumes feature independence |
| Logistic Regression + TF-IDF | Interpretable, strong baseline | Linear decision boundary |
| Linear SVM | High accuracy for text | No probability output |

### 4. Evaluation

We measure **precision**, **recall**, and **F1-score** per class, plus overall
accuracy.  The confusion matrix shows where the classifier confuses categories.

## Common Pitfalls
- **Vocabulary leakage**: Fitting the vectorizer on the full dataset — always fit on training data only.
- **Ignoring class imbalance**: Macro F1 is more informative than accuracy for imbalanced classes.
- **Stopword removal**: Domain-specific stopwords matter (e.g., "not" is crucial for sentiment).
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


# ---------------------------------------------------------------------------
# Synthetic fallback dataset
# ---------------------------------------------------------------------------
_SYNTHETIC_TEMPLATES = {
    "sports": [
        "The {team} won the {sport} championship this season",
        "A great goal was scored during the {sport} match",
        "The coach announced new players for the {team} roster",
        "Fans celebrated the victory of {team} in the finals",
        "The {sport} league released the schedule for next season",
        "The star player signed a contract extension with {team}",
        "Training camp for {sport} starts next week",
        "The referee made a controversial call in the {sport} game",
        "The {team} defeated their rivals in overtime",
        "Injury report released for the {sport} playoffs",
    ],
    "technology": [
        "The new {device} was released with improved {feature}",
        "Engineers developed a faster {component} for data centers",
        "The software update fixes critical {feature} bugs",
        "A startup raised funding to build {device} hardware",
        "Cloud computing demand drives growth in {component} sales",
        "The {device} benchmark shows major performance gains",
        "Developers are adopting {feature} in their applications",
        "The latest {component} chip uses less power than before",
        "Tech company announces new {device} product line",
        "Open source {feature} framework reaches version five",
    ],
    "politics": [
        "The senator proposed a new {policy} bill in congress",
        "Voters in {region} head to the polls for the election",
        "The government announced changes to {policy} regulations",
        "A debate on {policy} reform dominated the news cycle",
        "The president signed an executive order on {policy}",
        "Opposition party criticizes the new {policy} proposal",
        "The committee held hearings on {policy} spending",
        "Campaign fundraising broke records in {region}",
        "Diplomats met to discuss international {policy} agreements",
        "The governor of {region} vetoed the {policy} legislation",
    ],
    "science": [
        "Researchers discovered a new {subject} phenomenon in the lab",
        "A study published in Nature reveals {subject} insights",
        "Scientists observed unusual {subject} patterns in the data",
        "The {subject} experiment produced unexpected results",
        "A new telescope detected {subject} signals from deep space",
        "The research team won an award for {subject} breakthroughs",
        "Funding for {subject} research increased this year",
        "A peer reviewed paper challenges previous {subject} theories",
        "Laboratory tests confirm the {subject} hypothesis",
        "The conference featured talks on cutting edge {subject} methods",
    ],
}

_FILL_WORDS = {
    "team": ["Eagles", "Tigers", "Wolves", "Sharks", "Bears", "Hawks"],
    "sport": ["hockey", "baseball", "basketball", "soccer", "football"],
    "device": ["laptop", "smartphone", "tablet", "server", "workstation"],
    "feature": ["battery", "processor", "display", "security", "networking"],
    "component": ["GPU", "CPU", "memory", "storage", "motherboard"],
    "policy": ["healthcare", "education", "immigration", "tax", "defense"],
    "region": ["California", "Texas", "Florida", "Ohio", "Virginia"],
    "subject": ["quantum", "biology", "chemistry", "astronomy", "genetics"],
}


def _generate_synthetic_dataset(n_per_category=250, seed=42):
    """Generate a simple synthetic text classification dataset."""
    rng = random.Random(seed)
    categories = list(_SYNTHETIC_TEMPLATES.keys())
    texts, labels, target_names = [], [], categories

    for cat_idx, cat in enumerate(categories):
        templates = _SYNTHETIC_TEMPLATES[cat]
        for _ in range(n_per_category):
            tmpl = rng.choice(templates)
            filled = tmpl
            for placeholder, options in _FILL_WORDS.items():
                if "{" + placeholder + "}" in filled:
                    filled = filled.replace("{" + placeholder + "}", rng.choice(options))
            texts.append(filled)
            labels.append(cat_idx)

    # Shuffle
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)

    # Split 80/20
    split = int(0.8 * len(texts))

    class _Bunch:
        pass

    train = _Bunch()
    train.data = list(texts[:split])
    train.target = np.array(labels[:split])
    train.target_names = target_names

    test = _Bunch()
    test.data = list(texts[split:])
    test.target = np.array(labels[split:])
    test.target_names = target_names

    return train, test, categories


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
        return train_data, test_data, categories, False
    except Exception:
        # Fallback: synthetic data if download fails
        train_data, test_data, synth_cats = _generate_synthetic_dataset()
        return train_data, test_data, synth_cats, True


def run_nlp_demo(demo_type: str, clf_name: str, max_features: int, n_categories: int):
    train_data, test_data, categories, is_synthetic = _load_newsgroups(n_categories)

    if is_synthetic:
        cat_labels = categories[:n_categories]
        fallback_warning = (
            "\n\n> **Warning**: Using synthetic fallback dataset "
            "(20 Newsgroups download failed — no internet?). "
            "Results are for demonstration only.\n"
        )
    else:
        cat_labels = [c.split(".")[-1] for c in categories[:n_categories]]
        fallback_warning = ""

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

        report = classification_report(test_data.target, preds,
                                       target_names=cat_labels, output_dict=True)

        # Build a combined 3-row figure (Gradio gr.Plot accepts one figure)
        from plotly.subplots import make_subplots as _ms

        fig = _ms(rows=3, cols=1,
                  subplot_titles=[f"Confusion Matrix — {clf_name}",
                                  f"Per-Class Precision / Recall / F1 — {clf_name}",
                                  "TF-IDF Heatmap: Top Terms per Category"],
                  row_heights=[0.35, 0.30, 0.35],
                  vertical_spacing=0.10)

        # Row 1: Confusion matrix heatmap
        fig.add_trace(go.Heatmap(
            z=cm, x=cat_labels, y=cat_labels,
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
            showscale=False,
        ), row=1, col=1)

        # Row 2: Precision/Recall/F1 bars
        precisions = [report[c]["precision"] for c in cat_labels]
        recalls = [report[c]["recall"] for c in cat_labels]
        f1s = [report[c]["f1-score"] for c in cat_labels]
        fig.add_trace(go.Bar(name="Precision", x=cat_labels, y=precisions,
                             marker_color="#42a5f5",
                             text=[f"{v:.2f}" for v in precisions],
                             textposition="outside"), row=2, col=1)
        fig.add_trace(go.Bar(name="Recall", x=cat_labels, y=recalls,
                             marker_color="#66bb6a",
                             text=[f"{v:.2f}" for v in recalls],
                             textposition="outside"), row=2, col=1)
        fig.add_trace(go.Bar(name="F1-Score", x=cat_labels, y=f1s,
                             marker_color="#ffa726",
                             text=[f"{v:.2f}" for v in f1s],
                             textposition="outside"), row=2, col=1)

        # Row 3: TF-IDF heatmap
        vect = TfidfVectorizer(max_features=max_features, stop_words="english", sublinear_tf=True)
        X = vect.fit_transform(train_data.data)
        vocab = np.array(vect.get_feature_names_out())
        n_cats = len(cat_labels)
        top_n = 8
        top_terms_set = []
        cat_term_scores = {}
        for cat_idx, cat_name in enumerate(cat_labels):
            mask = train_data.target == cat_idx
            mean_tfidf = np.asarray(X[mask].mean(axis=0)).flatten()
            top_idx = np.argsort(mean_tfidf)[-top_n:][::-1]
            top_terms_set.extend(vocab[top_idx].tolist())
            cat_term_scores[cat_name] = mean_tfidf

        seen = set()
        all_terms = []
        for t in top_terms_set:
            if t not in seen:
                seen.add(t)
                all_terms.append(t)

        term_indices = [np.where(vocab == t)[0][0] for t in all_terms]
        z_hm = np.zeros((n_cats, len(all_terms)))
        for i, cat_name in enumerate(cat_labels):
            for j, ti in enumerate(term_indices):
                z_hm[i, j] = cat_term_scores[cat_name][ti]

        fig.add_trace(go.Heatmap(
            z=z_hm, x=all_terms, y=cat_labels,
            colorscale="YlOrRd",
            text=np.round(z_hm, 4), texttemplate="%{text:.3f}",
            showscale=False,
        ), row=3, col=1)

        fig.update_layout(
            height=1200,
            barmode="group",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        fig.update_yaxes(range=[0, 1.15], row=2, col=1)
        fig.update_xaxes(tickangle=-45, row=3, col=1)

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

> TF-IDF features: `{max_features}` | Dataset: {"Synthetic fallback" if is_synthetic else "20 Newsgroups (subset)"}
{fallback_warning}"""

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

> Dataset: {"Synthetic fallback" if is_synthetic else "20 Newsgroups"} | Categories: {', '.join(cat_labels[:n_cats])}
{fallback_warning}"""

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
{fallback_warning}"""

    else:
        fig = go.Figure()
        metrics_md = "Select a demo type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# Module 20 — Natural Language Processing\n*Level: Intermediate*")

    with gr.Accordion("Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## Interactive Demo\n\nText classification on the 20 Newsgroups dataset. Compare NLP models and explore word importance.\n\n> **Note**: First run downloads the 20 Newsgroups dataset (~15 MB). If download fails, a synthetic fallback dataset is used.")

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
            run_btn = gr.Button("Run NLP Demo", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_nlp_demo,
        inputs=[demo_dd, clf_dd, feat_sl, cat_sl],
        outputs=[plot_out, metrics_out]
    )
