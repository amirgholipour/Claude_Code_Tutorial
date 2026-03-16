"""Module 14 — ML Pipeline
Level: Advanced"""
import gradio as gr
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils import load_dataset, split_and_scale
from config import COLORS

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_SAVE_PATH = "/tmp/ml_course_model.pkl"
IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"]

THEORY = """## End-to-End ML Pipeline — From Data to Deployment

A **production ML pipeline** is a reproducible, end-to-end workflow that carries data from raw input
all the way through to reliable predictions — with no data leakage and minimal manual steps.

---

### The Full ML Workflow

```
Raw Data → EDA → Preprocessing → Feature Engineering → Train/Val Split
    → Model Training → Evaluation → Hyperparameter Tuning → Save → Deploy → Monitor
```

---

### sklearn `Pipeline` — The Right Way to Chain Steps

A `Pipeline` bundles preprocessing and modeling into a **single object**:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)   # scaler.fit_transform + model.fit — in one call
pipeline.predict(X_test)         # scaler.transform + model.predict — no leakage
```

**Why Pipeline prevents data leakage:**
Without a Pipeline, a common mistake is fitting the scaler on `X` (all data) before the train/test split.
The scaler then has information from the test set baked into its mean/std, inflating evaluation metrics.
Pipeline ensures the scaler is **only fit on training data**, even inside cross-validation.

---

### Cross-Validation — Reliable Performance Estimation

K-fold CV splits data into K folds, trains on K-1, tests on 1, repeats K times:

$$\\text{CV Score} = \\frac{1}{K} \\sum_{k=1}^{K} \\text{score}(\\text{model trained without fold } k, \\text{fold } k)$$

- Use `cross_val_score(pipeline, X, y, cv=5)` — Pipeline ensures no leakage per fold
- Report **mean ± std** to characterize both performance and stability
- High std → model is sensitive to the specific training split (high variance)

---

### Model Persistence — Save and Load

```python
import joblib
joblib.dump(pipeline, "model.pkl")    # serialize to disk
pipeline = joblib.load("model.pkl")   # deserialize and use
```

`joblib` is preferred over `pickle` for sklearn objects — it handles large numpy arrays efficiently.

---

### Production Concerns

| Issue | Description | Solution |
|---|---|---|
| **Feature drift** | Input distribution changes over time | Monitor feature statistics |
| **Concept drift** | Relationship between X and y changes | Monitor prediction distribution |
| **Model staleness** | World changes, model becomes wrong | Scheduled retraining |
| **Latency** | Pipeline too slow for real-time use | Profile, simplify, cache |
| **Reproducibility** | Same input → same output | Fix `random_state`, version control models |
"""

CODE_EXAMPLE = '''import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── 1. Build Pipeline ────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),                  # step 1: normalize
    ("model", RandomForestClassifier(              # step 2: classify
        n_estimators=100, random_state=42
    )),
])

# ─── 2. Cross-Validation (no data leakage — scaler is fit per fold) ───────────
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── 3. Final Training on Full Train Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
print(f"Test Accuracy: {accuracy_score(y_test, pipeline.predict(X_test)):.3f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# ─── 4. Save Pipeline to Disk ─────────────────────────────────────────────────
joblib.dump(pipeline, "model.pkl")
print("Model saved!")

# ─── 5. Load and Predict ──────────────────────────────────────────────────────
loaded = joblib.load("model.pkl")
new_sample = [[5.1, 3.5, 1.4, 0.2]]    # iris: sepal L, sepal W, petal L, petal W
prediction = loaded.predict(new_sample)
proba = loaded.predict_proba(new_sample)
print(f"Predicted class: {prediction[0]}, Confidence: {proba.max():.3f}")
'''


def _build_scaler(scaler_type: str):
    if scaler_type == "StandardScaler":
        return StandardScaler()
    elif scaler_type == "MinMaxScaler":
        return MinMaxScaler()
    return None  # "None" — no scaling


def _build_model(algorithm: str, n_estimators: int):
    if algorithm == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    elif algorithm == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_pipeline(
    dataset_name: str,
    scaler_type: str,
    algorithm: str,
    n_estimators: int,
    cv_folds: int,
):
    """
    Build an sklearn Pipeline, run cross-validation, train, evaluate, and save.

    Args:
        dataset_name: Dataset to use
        scaler_type: "None", "StandardScaler", or "MinMaxScaler"
        algorithm: "Logistic Regression", "Random Forest", or "Gradient Boosting"
        n_estimators: Number of trees (for RF / GB)
        cv_folds: Number of CV folds (3–10)

    Returns:
        (cv_fig, results_md, model_path_str)
    """
    try:
        X, y, feature_names, target_names = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test, _ = split_and_scale(
            X, y, test_size=0.2, scale=None, random_state=42
        )

        # Build pipeline steps
        steps = []
        scaler = _build_scaler(scaler_type)
        if scaler is not None:
            steps.append(("scaler", scaler))
        model = _build_model(algorithm, n_estimators)
        steps.append(("model", model))
        pipeline = Pipeline(steps)

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="accuracy")

        # Final fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))

        # Save model
        joblib.dump(pipeline, MODEL_SAVE_PATH)

        # ── CV Score chart ─────────────────────────────────────────────────
        fold_labels = [f"Fold {i+1}" for i in range(cv_folds)]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fold_labels,
            y=cv_scores,
            name="CV Score per Fold",
            marker_color=COLORS["primary"],
            opacity=0.8,
        ))
        fig.add_hline(
            y=cv_scores.mean(),
            line_dash="dash",
            line_color=COLORS["danger"],
            annotation_text=f"Mean: {cv_scores.mean():.3f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"Cross-Validation Scores — {algorithm} on {dataset_name}<br>"
                  f"<sup>Scaler: {scaler_type} | {cv_folds}-fold CV</sup>",
            xaxis_title="Fold",
            yaxis_title="Accuracy",
            yaxis=dict(range=[max(0, cv_scores.min() - 0.1), 1.05]),
            template="plotly_white",
            height=400,
        )

        # ── Report ─────────────────────────────────────────────────────────
        report = classification_report(
            y_test, y_pred,
            target_names=[str(t) for t in target_names],
            output_dict=True
        )
        macro = report.get("macro avg", {})

        pipeline_repr = " → ".join(
            [scaler_type if scaler_type != "None" else "(no scaler)", algorithm]
        )

        results_md = f"""### Pipeline Results

**Pipeline:** `{pipeline_repr}`

| Metric | Value |
|---|---|
| **CV Accuracy (mean)** | `{cv_scores.mean():.4f}` |
| **CV Accuracy (std)** | `±{cv_scores.std():.4f}` |
| **CV Score Range** | `{cv_scores.min():.3f}` – `{cv_scores.max():.3f}` |
| **Test Accuracy** | `{test_acc:.4f}` |
| **Train Accuracy** | `{train_acc:.4f}` |
| **Precision (macro)** | `{macro.get("precision", 0):.4f}` |
| **Recall (macro)** | `{macro.get("recall", 0):.4f}` |
| **F1 Score (macro)** | `{macro.get("f1-score", 0):.4f}` |
| **Train / Test sizes** | `{len(X_train)}` / `{len(X_test)}` |

**Model saved to:** `{MODEL_SAVE_PATH}`

> A tight CV std (< 0.02) indicates a **stable** model. A large gap between train and test accuracy
> indicates **overfitting** — consider reducing model complexity or adding regularization.
>
> Go to the **Predict** tab to test the saved model on custom input.
"""
        return fig, results_md

    except Exception as e:
        import traceback
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white", height=400)
        return empty_fig, f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def run_prediction(f1: float, f2: float, f3: float, f4: float):
    """
    Load the saved iris pipeline and predict class for the given feature values.

    Args:
        f1: Sepal length (cm)
        f2: Sepal width (cm)
        f3: Petal length (cm)
        f4: Petal width (cm)

    Returns:
        prediction_md (str)
    """
    try:
        if not os.path.exists(MODEL_SAVE_PATH):
            return (
                "**No saved model found.**\n\n"
                "Please go to the **Build Pipeline** tab, select `iris` as the dataset, "
                "and click **Run & Save Pipeline** first."
            )

        pipeline = joblib.load(MODEL_SAVE_PATH)
        sample = np.array([[f1, f2, f3, f4]])

        prediction = pipeline.predict(sample)[0]

        # Try to get probabilities
        proba_md = ""
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(sample)[0]
                n_classes = len(proba)
                class_names = IRIS_CLASS_NAMES[:n_classes]
                proba_rows = "\n".join(
                    [f"| `{class_names[i] if i < len(class_names) else i}` | `{proba[i]:.3f}` |"
                     for i in range(n_classes)]
                )
                proba_md = f"""
#### Class Probabilities
| Class | Probability |
|---|---|
{proba_rows}
"""
            except Exception:
                pass

        # Map prediction to name if it's iris (3 classes)
        try:
            pred_name = IRIS_CLASS_NAMES[int(prediction)]
        except (IndexError, ValueError):
            pred_name = str(prediction)

        prediction_md = f"""### Prediction Result

| Input Feature | Value |
|---|---|
| Sepal Length | `{f1:.2f}` cm |
| Sepal Width  | `{f2:.2f}` cm |
| Petal Length | `{f3:.2f}` cm |
| Petal Width  | `{f4:.2f}` cm |

---

## Predicted Class: **{pred_name}** (class `{prediction}`)
{proba_md}
> Prediction made by loaded model from `{MODEL_SAVE_PATH}`.
> This model was trained in the **Build Pipeline** tab.
"""
        return prediction_md

    except Exception as e:
        import traceback
        return f"**Error loading/predicting:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def build_tab():
    """Build the Gradio UI for the ML Pipeline module (with inner Build / Predict tabs)."""
    with gr.Column():
        with gr.Accordion("📖 Theory", open=False):
            gr.Markdown(THEORY)

        with gr.Accordion("💻 Code Example", open=False):
            gr.Code(CODE_EXAMPLE, language="python")

        gr.Markdown("### 🔬 Interactive Demo")

        with gr.Tabs():
            # ── Tab 1: Build Pipeline ──────────────────────────────────────
            with gr.Tab("🔧 Build Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_dd = gr.Dropdown(
                            choices=["iris", "wine", "breast_cancer", "digits"],
                            value="iris",
                            label="Dataset",
                        )
                        scaler_radio = gr.Radio(
                            choices=["None", "StandardScaler", "MinMaxScaler"],
                            value="StandardScaler",
                            label="Scaler",
                        )
                        algorithm_radio = gr.Radio(
                            choices=["Logistic Regression", "Random Forest", "Gradient Boosting"],
                            value="Random Forest",
                            label="Algorithm",
                        )
                        n_estimators_slider = gr.Slider(
                            minimum=10, maximum=300, step=10, value=100,
                            label="n_estimators (RF / GB only)",
                        )
                        cv_folds_slider = gr.Slider(
                            minimum=3, maximum=10, step=1, value=5,
                            label="CV Folds",
                        )
                        run_btn = gr.Button("▶ Run & Save Pipeline", variant="primary")

                    with gr.Column(scale=3):
                        cv_plot = gr.Plot(label="Cross-Validation Scores")
                        results_md = gr.Markdown(label="Results")

                run_btn.click(
                    fn=run_pipeline,
                    inputs=[
                        dataset_dd,
                        scaler_radio,
                        algorithm_radio,
                        n_estimators_slider,
                        cv_folds_slider,
                    ],
                    outputs=[cv_plot, results_md],
                )

            # ── Tab 2: Predict ─────────────────────────────────────────────
            with gr.Tab("🔮 Predict"):
                gr.Markdown(
                    "### Load Saved Model & Predict\n\n"
                    "> **Note:** First go to the **Build Pipeline** tab, select **iris** as the "
                    "dataset, and click **Run & Save Pipeline**. This will save the model to "
                    f"`{MODEL_SAVE_PATH}`. Then come back here to make predictions.\n\n"
                    "Adjust the sliders below to set iris feature values and click **Predict**."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        sepal_length = gr.Slider(
                            minimum=4.0, maximum=8.0, step=0.1, value=5.1,
                            label="Sepal Length (cm)",
                        )
                        sepal_width = gr.Slider(
                            minimum=2.0, maximum=5.0, step=0.1, value=3.5,
                            label="Sepal Width (cm)",
                        )
                        petal_length = gr.Slider(
                            minimum=1.0, maximum=7.0, step=0.1, value=1.4,
                            label="Petal Length (cm)",
                        )
                        petal_width = gr.Slider(
                            minimum=0.1, maximum=2.5, step=0.1, value=0.2,
                            label="Petal Width (cm)",
                        )
                        predict_btn = gr.Button("🔮 Predict", variant="primary")

                    with gr.Column(scale=2):
                        prediction_out = gr.Markdown(label="Prediction")

                predict_btn.click(
                    fn=run_prediction,
                    inputs=[sepal_length, sepal_width, petal_length, petal_width],
                    outputs=[prediction_out],
                )
