"""
App configuration for ML & Deep Learning Course
"""

APP_TITLE = "Machine Learning & Deep Learning"
APP_DESCRIPTION = "Interactive course from basic to advanced — powered by scikit-learn & PyTorch"
APP_VERSION = "1.0.0"

# ── Sklearn dataset registry ──────────────────────────────────────────────────
SKLEARN_DATASETS = {
    "iris": {
        "task": "classification",
        "description": "Iris flowers (150 samples, 4 features, 3 classes)",
        "loader": "load_iris",
    },
    "breast_cancer": {
        "task": "classification",
        "description": "Breast cancer (569 samples, 30 features, 2 classes)",
        "loader": "load_breast_cancer",
    },
    "wine": {
        "task": "classification",
        "description": "Wine recognition (178 samples, 13 features, 3 classes)",
        "loader": "load_wine",
    },
    "diabetes": {
        "task": "regression",
        "description": "Diabetes progression (442 samples, 10 features)",
        "loader": "load_diabetes",
    },
    "digits": {
        "task": "classification",
        "description": "Handwritten digits (1797 samples, 64 features, 10 classes)",
        "loader": "load_digits",
    },
}

CLASSIFICATION_DATASETS = ["iris", "breast_cancer", "wine", "digits"]
REGRESSION_DATASETS = ["diabetes"]
ALL_DATASETS = list(SKLEARN_DATASETS.keys())

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "palette": [
        "#6366F1", "#10B981", "#F59E0B", "#EF4444",
        "#3B82F6", "#8B5CF6", "#EC4899", "#14B8A6",
        "#F97316", "#84CC16",
    ],
}

# ── Deep learning defaults ────────────────────────────────────────────────────
DL_DEFAULTS = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 32,
    "hidden_sizes": [64, 32],
}
