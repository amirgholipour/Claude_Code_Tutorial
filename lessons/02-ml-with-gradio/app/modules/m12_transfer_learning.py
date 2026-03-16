"""Module 12 — Transfer Learning
Level: Advanced"""
import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from utils.plot_utils import feature_importance_bar
from config import COLORS

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

THEORY = """## Transfer Learning — Reusing Knowledge Across Tasks

**Transfer learning** is the technique of reusing knowledge gained from solving one problem (the *source* task) to help solve a different but related problem (the *target* task).

---

### Why It Works

Neural networks (and many ML models) learn hierarchical representations:
- **Lower layers** capture general, broadly useful features — edges, textures, patterns
- **Upper layers** capture task-specific combinations of those patterns

When we transfer the lower layers to a new task, we skip re-learning those general features from scratch.

---

### Two Core Strategies

| Strategy | What Gets Frozen | What Gets Trained | When to Use |
|---|---|---|---|
| **Feature Extraction** | All pre-trained layers (frozen) | Only the new head | Small target dataset, similar domain |
| **Fine-tuning** | No layers frozen (or just the early ones) | All layers (small lr) | Larger target dataset, different domain |

---

### Traditional ML Analogy (sklearn)

We can simulate transfer learning using **PCA as a feature extractor**:

1. **Pre-train** a PCA on a large, rich source dataset — it learns the important directions of variance
2. **Transfer** those learned components as fixed features for the target task
3. **Train only the classifier head** (SVM) on limited target data

This mirrors how CNNs use pre-trained convolutional filters as fixed feature extractors.

---

### Practical Benefits

- **Less data needed** for the target task — the hard representation work is already done
- **Faster training** — only a small head is learned
- **Better performance** — especially when target data is scarce
- **Regularization effect** — pre-trained weights act as a strong prior

---

### When Transfer Learning Helps Most

- Target dataset is **small** (< a few hundred samples per class)
- Source and target domains are **related** (similar data distribution)
- Pre-training used a **large, diverse** dataset
"""

CODE_EXAMPLE = '''from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ─── Simulate Transfer Learning with sklearn ───────────────────────────────

# Step 1: "Pre-train" a PCA feature extractor on a large source dataset
#         (In deep learning, this is pre-training on ImageNet)
X_source, y_source = load_digits(return_X_y=True)
pca_pretrained = PCA(n_components=20)
pca_pretrained.fit(X_source)  # learns the important variance directions

# Step 2: Feature Extraction — freeze PCA, only train SVM head
#         (Equivalent to: load pre-trained ResNet, freeze all layers, add new FC head)
X_target_small = X_source[:50]   # limited labeled data for new task
y_target_small = y_source[:50]

X_features = pca_pretrained.transform(X_target_small)  # fixed transform
clf = SVC(kernel="rbf")
clf.fit(X_features, y_target_small)  # only the head is trained

# Step 3: Fine-tuning — re-fit PCA on combined source + target data (small lr analogy)
pca_finetuned = PCA(n_components=20)
pca_finetuned.fit(X_target_small)   # adapts to target distribution
clf_ft = SVC(kernel="rbf")
clf_ft.fit(pca_finetuned.transform(X_target_small), y_target_small)

# Step 4: From Scratch baseline — no pre-training, no transfer
pca_scratch = PCA(n_components=20)
pca_scratch.fit(X_target_small)     # learns only from small target data
clf_scratch = SVC(kernel="rbf")
clf_scratch.fit(pca_scratch.transform(X_target_small), y_target_small)

# Compare on test set
print(f"Feature Extraction: {accuracy_score(y_test, clf.predict(pca_pretrained.transform(X_test))):.3f}")
print(f"Fine-tuning:        {accuracy_score(y_test, clf_ft.predict(pca_finetuned.transform(X_test))):.3f}")
print(f"From Scratch:       {accuracy_score(y_test, clf_scratch.predict(pca_scratch.transform(X_test))):.3f}")
'''


def _get_source_data(source_dataset: str, target_X: np.ndarray, target_y: np.ndarray):
    """Return the source data used for pre-training the PCA."""
    if source_dataset == "iris only":
        X_src, y_src, _, _ = load_dataset("iris")
    elif source_dataset == "wine only":
        X_src, y_src, _, _ = load_dataset("wine")
    else:
        # "all data (simulate pre-training)" — use the full digits dataset itself
        X_src = target_X
        y_src = target_y
    return X_src, y_src


def _run_single_strategy(
    strategy: str,
    source_X: np.ndarray,
    target_X_train: np.ndarray,
    target_y_train: np.ndarray,
    target_X_test: np.ndarray,
    target_y_test: np.ndarray,
    n_components: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> float:
    """Run one strategy and return test accuracy."""
    # Sample limited training data
    indices = rng.choice(len(target_X_train), size=min(n_samples, len(target_X_train)), replace=False)
    X_small = target_X_train[indices]
    y_small = target_y_train[indices]

    scaler = StandardScaler()

    if strategy == "Feature Extraction":
        # Pre-train PCA on source data, then freeze — only train SVM head
        n_comp = min(n_components, source_X.shape[1], source_X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_src_scaled = scaler.fit_transform(source_X)
        pca.fit(X_src_scaled)
        # Transform target data using frozen (source-fitted) PCA + scaler
        scaler_tgt = StandardScaler()
        X_small_scaled = scaler_tgt.fit_transform(X_small)
        X_test_scaled = scaler_tgt.transform(target_X_test)
        X_small_feat = pca.transform(X_small_scaled)
        X_test_feat = pca.transform(X_test_scaled)
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_small_feat, y_small)
        return accuracy_score(target_y_test, clf.predict(X_test_feat))

    elif strategy == "Fine-tuning":
        # Fit PCA on small target data (fine-tune the feature extractor)
        n_comp = min(n_components, X_small.shape[1], X_small.shape[0] - 1)
        X_small_scaled = scaler.fit_transform(X_small)
        X_test_scaled = scaler.transform(target_X_test)
        pca = PCA(n_components=n_comp)
        pca.fit(X_small_scaled)
        X_small_feat = pca.transform(X_small_scaled)
        X_test_feat = pca.transform(X_test_scaled)
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_small_feat, y_small)
        return accuracy_score(target_y_test, clf.predict(X_test_feat))

    else:  # From Scratch
        # No pre-training at all — raw features directly to classifier
        X_small_scaled = scaler.fit_transform(X_small)
        X_test_scaled = scaler.transform(target_X_test)
        clf = SVC(kernel="rbf", random_state=42)
        clf.fit(X_small_scaled, y_small)
        return accuracy_score(target_y_test, clf.predict(X_test_scaled))


def run_transfer(
    source_dataset: str,
    target_dataset: str,
    n_components: int,
    strategy: str,
    n_samples_max: int,
):
    """
    Simulate transfer learning on a target classification task with limited data.

    Args:
        source_dataset: Dataset used for pre-training the feature extractor
        target_dataset: The downstream classification task
        n_components: Number of PCA components (5–30)
        strategy: "Feature Extraction", "Fine-tuning", or "From Scratch"
        n_samples_max: Maximum training samples to show on the curve

    Returns:
        (comparison_fig, explanation_md)
    """
    try:
        # Load target dataset
        X_tgt, y_tgt, feature_names, target_names = load_dataset(target_dataset)
        X_train_full, X_test, y_train_full, y_test, _ = split_and_scale(
            X_tgt, y_tgt, test_size=0.3, scale=None, random_state=42
        )

        # Load source dataset for pre-training
        X_src, y_src, _, _ = _get_source_data(source_dataset, X_tgt, y_tgt)

        # Sample sizes to test
        max_possible = min(n_samples_max, len(X_train_full))
        sample_sizes = []
        for s in [5, 10, 25, 50, 100, 200, 300, 500]:
            if s <= max_possible:
                sample_sizes.append(s)
        if not sample_sizes or sample_sizes[-1] < max_possible:
            sample_sizes.append(max_possible)
        sample_sizes = sorted(set(sample_sizes))

        rng = np.random.RandomState(42)

        # Always compute all 3 strategies for comparison
        strategies = ["Feature Extraction", "Fine-tuning", "From Scratch"]
        strategy_colors = {
            "Feature Extraction": COLORS["primary"],
            "Fine-tuning": COLORS["success"],
            "From Scratch": COLORS["danger"],
        }

        results = {s: [] for s in strategies}
        for n in sample_sizes:
            for strat in strategies:
                acc = _run_single_strategy(
                    strat, X_src,
                    X_train_full, y_train_full,
                    X_test, y_test,
                    n_components, n, rng,
                )
                results[strat].append(acc)

        # Build grouped bar chart
        fig = go.Figure()
        for strat in strategies:
            dash = "solid" if strat == strategy else "dot"
            width = 3 if strat == strategy else 1.5
            opacity = 1.0 if strat == strategy else 0.45
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=results[strat],
                mode="lines+markers",
                name=strat,
                line=dict(color=strategy_colors[strat], width=width, dash=dash),
                marker=dict(size=8),
                opacity=opacity,
            ))

        fig.update_layout(
            title=f"Transfer Learning: Accuracy vs Training Samples<br>"
                  f"<sup>Target: {target_dataset} | Source: {source_dataset} | "
                  f"PCA components: {n_components}</sup>",
            xaxis_title="Number of Training Samples",
            yaxis_title="Test Accuracy",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Build explanation markdown
        fe_final = results["Feature Extraction"][-1]
        ft_final = results["Fine-tuning"][-1]
        sc_final = results["From Scratch"][-1]
        best_strategy = max(strategies, key=lambda s: results[s][-1])

        fe_small = results["Feature Extraction"][0] if results["Feature Extraction"] else 0
        sc_small = results["From Scratch"][0] if results["From Scratch"] else 0
        small_n = sample_sizes[0]

        explanation_md = f"""### Transfer Learning Results

**Selected strategy:** `{strategy}` (highlighted in chart)

#### Final Accuracy at {sample_sizes[-1]} Samples
| Strategy | Accuracy |
|---|---|
| Feature Extraction (frozen PCA) | `{fe_final:.3f}` |
| Fine-tuning (target-adapted PCA) | `{ft_final:.3f}` |
| From Scratch (no transfer) | `{sc_final:.3f}` |

**Best performer:** `{best_strategy}`

#### Low-Data Regime (n = {small_n} samples)
| Strategy | Accuracy |
|---|---|
| Feature Extraction | `{fe_small:.3f}` |
| From Scratch | `{sc_small:.3f}` |
| **Transfer learning advantage** | `{fe_small - sc_small:+.3f}` |

#### Interpretation
- **Feature Extraction** leverages pre-trained PCA directions from the source dataset. With very few target samples,
  the frozen extractor prevents overfitting and provides a strong prior.
- **Fine-tuning** adapts the feature extractor to the target distribution — beneficial when the target domain
  differs from the source. With small data it may overfit.
- **From Scratch** has no knowledge transfer. It needs more samples to learn useful representations,
  showing **lower accuracy in the low-data regime**.

> The crossover point — where "from scratch" catches up — reveals how much the source domain helps.
> A wider gap at low sample counts = stronger transfer signal.

**Source dataset:** `{source_dataset}` → PCA components used as frozen features
**Target dataset:** `{target_dataset}` → {X_tgt.shape[0]} total samples, {X_tgt.shape[1]} raw features
**PCA components:** {n_components} (capturing top variance directions)
"""
        return fig, explanation_md

    except Exception as e:
        import traceback
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=450)
        return empty_fig, f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def build_tab():
    """Build the Gradio UI for the Transfer Learning module."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Row():
            with gr.Column(scale=1):
                source_dd = gr.Dropdown(
                    choices=["all data (simulate pre-training)", "iris only", "wine only"],
                    value="all data (simulate pre-training)",
                    label="Source Dataset (Pre-training)",
                )
                target_dd = gr.Dropdown(
                    choices=["digits", "iris", "wine", "breast_cancer"],
                    value="digits",
                    label="Target Dataset (Downstream Task)",
                )
                strategy_radio = gr.Radio(
                    choices=["Feature Extraction", "Fine-tuning", "From Scratch"],
                    value="Feature Extraction",
                    label="Strategy to Highlight",
                )
                n_components_slider = gr.Slider(
                    minimum=5, maximum=30, step=1, value=20,
                    label="PCA Components",
                )
                n_samples_slider = gr.Slider(
                    minimum=50, maximum=500, step=50, value=200,
                    label="Max Training Samples",
                )
                run_btn = gr.Button("▶ Run Transfer Learning", variant="primary")

            with gr.Column(scale=3):
                plot_out = gr.Plot(label="Accuracy vs Training Samples")
                metrics_out = gr.Markdown(label="Analysis")

        run_btn.click(
            fn=run_transfer,
            inputs=[source_dd, target_dd, n_components_slider, strategy_radio, n_samples_slider],
            outputs=[plot_out, metrics_out],
        )
