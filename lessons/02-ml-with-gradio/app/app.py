"""
ML & Deep Learning Interactive Course
Run: python app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

# ── Import all modules — failures are caught individually below ────────────────
_import_errors = {}

def _try_import(name):
    try:
        mod = __import__(f"modules.{name}", fromlist=[name])
        return mod
    except Exception as e:
        _import_errors[name] = str(e)
        return None

# ── Original 14 modules ───────────────────────────────────────────────────────
m01_data_exploration      = _try_import("m01_data_exploration")
m02_regression            = _try_import("m02_regression")
m03_classification        = _try_import("m03_classification")
m04_model_evaluation      = _try_import("m04_model_evaluation")
m05_ensemble_methods      = _try_import("m05_ensemble_methods")
m06_clustering            = _try_import("m06_clustering")
m07_dimensionality        = _try_import("m07_dimensionality")
m08_neural_networks       = _try_import("m08_neural_networks")
m09_cnn                   = _try_import("m09_cnn")
m10_rnn                   = _try_import("m10_rnn")
m11_training_best_practices = _try_import("m11_training_best_practices")
m12_transfer_learning     = _try_import("m12_transfer_learning")
m13_explainability        = _try_import("m13_explainability")
m14_ml_pipeline           = _try_import("m14_ml_pipeline")

# ── Extended 12 modules ───────────────────────────────────────────────────────
m15_data_preparation      = _try_import("m15_data_preparation")
m16_feature_engineering   = _try_import("m16_feature_engineering")
m17_feature_selection     = _try_import("m17_feature_selection")
m18_eda_advanced          = _try_import("m18_eda_advanced")
m19_time_series           = _try_import("m19_time_series")
m20_nlp                   = _try_import("m20_nlp")
m21_mlops                 = _try_import("m21_mlops")
m22_cicd_ml               = _try_import("m22_cicd_ml")
m23_responsible_ai        = _try_import("m23_responsible_ai")
m24_ml_system_design      = _try_import("m24_ml_system_design")
m25_hyperparameter_tuning = _try_import("m25_hyperparameter_tuning")
m26_anomaly_detection     = _try_import("m26_anomaly_detection")


def _build_tab_safe(module, module_name: str):
    """Call module.build_tab() inside a try/except; show error message on failure."""
    if module is None:
        err = _import_errors.get(module_name, "Unknown import error")
        gr.Markdown(
            f"**Module `{module_name}` failed to load.**\n\n"
            f"```\n{err}\n```\n\n"
            "Check that all dependencies are installed:\n"
            "```bash\npip install -r requirements.txt\n```"
        )
        return

    try:
        module.build_tab()
    except Exception as e:
        import traceback
        gr.Markdown(
            f"**Error building tab for `{module_name}`:**\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )


# ── App ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="ML & Deep Learning Course") as app:
    gr.Markdown("""
# 🧠 Machine Learning & Deep Learning
**Interactive Course — Basic to Advanced (26 Modules)**

> Built with scikit-learn, PyTorch, and Gradio. All demos run on CPU. No API keys required.

| Level | Modules |
|---|---|
| 🟢 Basic | Data Exploration, Regression, Classification, Model Evaluation |
| 🟡 Intermediate | Ensemble Methods, Clustering, Dimensionality Reduction |
| 🔴 Deep Learning | Neural Networks, CNNs, RNNs, Training Best Practices |
| 🟣 Advanced | Transfer Learning, Explainability, ML Pipeline |
| 🔵 Foundation | Data Preparation, Feature Engineering, Feature Selection, Advanced EDA |
| 🟠 Specialized | Time Series, NLP |
| ⚫ Engineering | MLOps & Monitoring, CI/CD for ML |
| 🔶 Responsible | Responsible AI, ML System Design |
| 🔷 Tuning | Hyperparameter Tuning, Anomaly Detection |
""")

    with gr.Tabs():
        # ── BASIC ML ─────────────────────────────────────────────────────────────
        with gr.Tab("📊 01 · Data Exploration"):
            _build_tab_safe(m01_data_exploration, "m01_data_exploration")

        with gr.Tab("📈 02 · Regression"):
            _build_tab_safe(m02_regression, "m02_regression")

        with gr.Tab("🎯 03 · Classification"):
            _build_tab_safe(m03_classification, "m03_classification")

        with gr.Tab("📏 04 · Model Evaluation"):
            _build_tab_safe(m04_model_evaluation, "m04_model_evaluation")

        # ── INTERMEDIATE ML ───────────────────────────────────────────────────────
        with gr.Tab("🌲 05 · Ensemble Methods"):
            _build_tab_safe(m05_ensemble_methods, "m05_ensemble_methods")

        with gr.Tab("🔵 06 · Clustering"):
            _build_tab_safe(m06_clustering, "m06_clustering")

        with gr.Tab("🔍 07 · Dimensionality"):
            _build_tab_safe(m07_dimensionality, "m07_dimensionality")

        # ── DEEP LEARNING ─────────────────────────────────────────────────────────
        with gr.Tab("🧠 08 · Neural Networks"):
            _build_tab_safe(m08_neural_networks, "m08_neural_networks")

        with gr.Tab("🖼️ 09 · CNNs"):
            _build_tab_safe(m09_cnn, "m09_cnn")

        with gr.Tab("🔄 10 · RNNs"):
            _build_tab_safe(m10_rnn, "m10_rnn")

        with gr.Tab("⚙️ 11 · Training Tips"):
            _build_tab_safe(m11_training_best_practices, "m11_training_best_practices")

        # ── ADVANCED ─────────────────────────────────────────────────────────────
        with gr.Tab("🔀 12 · Transfer Learning"):
            _build_tab_safe(m12_transfer_learning, "m12_transfer_learning")

        with gr.Tab("💡 13 · Explainability"):
            _build_tab_safe(m13_explainability, "m13_explainability")

        with gr.Tab("🔧 14 · ML Pipeline"):
            _build_tab_safe(m14_ml_pipeline, "m14_ml_pipeline")

        # ── DATA FOUNDATION ───────────────────────────────────────────────────────
        with gr.Tab("🧹 15 · Data Preparation"):
            _build_tab_safe(m15_data_preparation, "m15_data_preparation")

        with gr.Tab("⚙️ 16 · Feature Engineering"):
            _build_tab_safe(m16_feature_engineering, "m16_feature_engineering")

        with gr.Tab("🎯 17 · Feature Selection"):
            _build_tab_safe(m17_feature_selection, "m17_feature_selection")

        with gr.Tab("🔬 18 · Advanced EDA"):
            _build_tab_safe(m18_eda_advanced, "m18_eda_advanced")

        # ── SPECIALIZED ML ────────────────────────────────────────────────────────
        with gr.Tab("📈 19 · Time Series"):
            _build_tab_safe(m19_time_series, "m19_time_series")

        with gr.Tab("📝 20 · NLP"):
            _build_tab_safe(m20_nlp, "m20_nlp")

        # ── ENGINEERING & OPS ─────────────────────────────────────────────────────
        with gr.Tab("🚀 21 · MLOps & Monitoring"):
            _build_tab_safe(m21_mlops, "m21_mlops")

        with gr.Tab("🔄 22 · CI/CD for ML"):
            _build_tab_safe(m22_cicd_ml, "m22_cicd_ml")

        # ── RESPONSIBLE & DESIGN ──────────────────────────────────────────────────
        with gr.Tab("⚖️ 23 · Responsible AI"):
            _build_tab_safe(m23_responsible_ai, "m23_responsible_ai")

        with gr.Tab("🏗️ 24 · System Design"):
            _build_tab_safe(m24_ml_system_design, "m24_ml_system_design")

        # ── ADVANCED TECHNIQUES ───────────────────────────────────────────────────
        with gr.Tab("🔧 25 · Hyperparameter Tuning"):
            _build_tab_safe(m25_hyperparameter_tuning, "m25_hyperparameter_tuning")

        with gr.Tab("🚨 26 · Anomaly Detection"):
            _build_tab_safe(m26_anomaly_detection, "m26_anomaly_detection")


if __name__ == "__main__":
    app.launch(server_port=7861, server_name="127.0.0.1",
               share=False, show_error=True, theme=gr.themes.Soft())
