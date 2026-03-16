"""
Module 15 — Data Preparation & Cleaning
Level: Foundation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

THEORY = """
## 📖 What Is Data Preparation?

Data preparation (also called data cleaning or data wrangling) is the process of transforming raw, messy data into a clean, structured format suitable for machine learning. In practice, **80% of a data scientist's time** is spent here — not on modeling.

Real-world data is almost never clean: it has missing values, outliers, inconsistent types, duplicates, and irrelevant features.

## 🏗️ Key Steps

### 1. Missing Value Treatment
- **Drop**: Remove rows/columns with too many nulls (>50% threshold)
- **Mean/Median imputation**: Replace NaN with column average (mean for symmetric, median for skewed)
- **Mode imputation**: For categorical columns
- **KNN imputation**: Use nearby samples to estimate missing values (most accurate, slowest)
- **Forward/backward fill**: For time series data

### 2. Outlier Detection & Treatment
- **IQR method**: Outlier if value < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- **Z-score method**: Outlier if |z| > 3 (>3 standard deviations from mean)
- **Options**: Remove, cap (clip), transform (log), or keep if genuine signal

### 3. Data Type Coercion
- Dates stored as strings → `pd.to_datetime()`
- Numeric IDs stored as floats → cast to int
- Categorical strings → pandas `Categorical` dtype

### 4. Encoding Categorical Variables
- **Label Encoding**: Maps categories to integers (0, 1, 2…) — only for ordinal/tree-based models
- **One-Hot Encoding (OHE)**: Creates binary columns — for linear models & neural networks
- **Target Encoding**: Replace category with mean of target (risk of leakage!)

### 5. Duplicate Removal
- `df.duplicated()` finds exact row duplicates
- `df.drop_duplicates(subset=['id'])` removes based on key columns

## ✅ Decision Guide
| Problem | Solution |
|---|---|
| <5% missing, random | Mean/median imputation |
| >30% missing | Drop column |
| <5% missing, structured | KNN imputation |
| Outliers (real errors) | Remove or cap |
| Outliers (genuine extremes) | Log transform |
| Nominal categories | One-Hot Encoding |
| Ordinal categories | Label Encoding |

## ⚠️ Common Pitfalls
- **Data leakage**: Fitting the imputer on the full dataset (including test set). Always fit on train, transform both.
- **Dropping too aggressively**: A column with 20% missing might still be valuable.
- **Ignoring dtype**: Storing ZIP codes as integers allows meaningless arithmetic.
- **OHE cardinality explosion**: A column with 1000 unique values creates 1000 new columns.
"""

CODE_EXAMPLE = '''
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ── Create dirty dataset ───────────────────────────────────────
df = pd.DataFrame({
    "age":    [25, np.nan, 35, 200, 28, np.nan],   # missing + outlier
    "salary": [50000, 60000, np.nan, 55000, 70000, 65000],
    "city":   ["NYC", "LA", "NYC", np.nan, "LA", "Chicago"],
})

# ── Missing values ─────────────────────────────────────────────
num_imputer = SimpleImputer(strategy="median")
df[["age","salary"]] = num_imputer.fit_transform(df[["age","salary"]])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[["city"]] = cat_imputer.fit_transform(df[["city"]])

# ── Outlier capping (IQR) ──────────────────────────────────────
Q1, Q3 = df["age"].quantile(0.25), df["age"].quantile(0.75)
IQR = Q3 - Q1
df["age"] = df["age"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# ── Encoding ───────────────────────────────────────────────────
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
city_encoded = ohe.fit_transform(df[["city"]])
print("Clean shape:", df.shape)
print("Encoded cities:", ohe.get_feature_names_out())
'''


def _make_dirty_dataset(n=200, missing_rate=0.15, outlier_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)
    age    = rng.normal(35, 10, n).clip(18, 70)
    salary = rng.normal(60000, 15000, n).clip(20000, 150000)
    dept   = rng.choice(["Engineering", "Marketing", "Sales", "HR"], n)

    df = pd.DataFrame({"age": age, "salary": salary, "department": dept})

    # inject missing values
    for col in ["age", "salary", "department"]:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    # inject outliers in age
    n_out = max(1, int(n * outlier_rate))
    out_idx = rng.choice(df.index[df["age"].notna()], size=n_out, replace=False)
    df.loc[out_idx, "age"] = rng.uniform(100, 150, n_out)

    return df


def run_cleaning_demo(impute_method: str, handle_outliers: str, encoding: str):
    df_raw = _make_dirty_dataset()

    stats_before = {
        "missing_age": int(df_raw["age"].isna().sum()),
        "missing_salary": int(df_raw["salary"].isna().sum()),
        "missing_dept": int(df_raw["department"].isna().sum()),
        "outliers_age": int((df_raw["age"] > 80).sum()),
    }

    df = df_raw.copy()

    # ── Impute ──────────────────────────────────────────────────
    num_cols = ["age", "salary"]
    cat_col  = "department"

    if impute_method == "KNN":
        num_imp = KNNImputer(n_neighbors=5)
    elif impute_method == "Median":
        num_imp = SimpleImputer(strategy="median")
    else:
        num_imp = SimpleImputer(strategy="mean")

    df[num_cols] = num_imp.fit_transform(df[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[[cat_col]] = cat_imp.fit_transform(df[[cat_col]])

    # ── Outliers ─────────────────────────────────────────────────
    if handle_outliers == "IQR Cap":
        Q1, Q3 = df["age"].quantile(0.25), df["age"].quantile(0.75)
        IQR = Q3 - Q1
        df["age"] = df["age"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    elif handle_outliers == "Z-score Remove":
        z = np.abs((df["age"] - df["age"].mean()) / df["age"].std())
        df = df[z < 3].reset_index(drop=True)
    elif handle_outliers == "Log Transform":
        df["age"] = np.log1p(df["age"])

    # ── Encode ───────────────────────────────────────────────────
    df_encoded = df.copy()
    if encoding == "One-Hot":
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_arr = ohe.fit_transform(df[[cat_col]])
        enc_cols = ohe.get_feature_names_out([cat_col])
        df_encoded = pd.concat([
            df[num_cols].reset_index(drop=True),
            pd.DataFrame(enc_arr, columns=enc_cols)
        ], axis=1)
    elif encoding == "Label":
        le = LabelEncoder()
        df_encoded["dept_encoded"] = le.fit_transform(df[cat_col])

    stats_after = {
        "missing_age": int(df["age"].isna().sum()),
        "missing_salary": int(df["salary"].isna().sum()),
        "missing_dept": int(df["department"].isna().sum()),
        "outliers_age": int((df["age"] > 80).sum()) if handle_outliers != "Log Transform" else 0,
        "shape": df_encoded.shape,
    }

    # ── Plot ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Age Distribution (raw)", "Age Distribution (cleaned)",
            "Missing Values Before", "Missing Values After"
        ]
    )

    # raw age histogram
    raw_age = df_raw["age"].dropna()
    fig.add_trace(go.Histogram(x=raw_age, nbinsx=30, name="Raw Age",
                               marker_color="#ef5350"), row=1, col=1)

    # cleaned age histogram
    clean_age = df["age"].dropna() if handle_outliers != "Log Transform" else np.expm1(df["age"].dropna())
    fig.add_trace(go.Histogram(x=clean_age, nbinsx=30, name="Clean Age",
                               marker_color="#42a5f5"), row=1, col=2)

    # missing before
    miss_before = [stats_before["missing_age"], stats_before["missing_salary"], stats_before["missing_dept"]]
    fig.add_trace(go.Bar(x=["age", "salary", "department"], y=miss_before,
                         marker_color="#ef9a9a", name="Missing Before"), row=2, col=1)

    # missing after
    miss_after = [stats_after["missing_age"], stats_after["missing_salary"], stats_after["missing_dept"]]
    fig.add_trace(go.Bar(x=["age", "salary", "department"], y=miss_after,
                         marker_color="#a5d6a7", name="Missing After"), row=2, col=2)

    fig.update_layout(height=550, showlegend=False,
                      title_text="Data Cleaning Results")

    metrics_md = f"""
### Cleaning Summary

| Metric | Before | After |
|---|---|---|
| Missing age | {stats_before['missing_age']} | {stats_after['missing_age']} |
| Missing salary | {stats_before['missing_salary']} | {stats_after['missing_salary']} |
| Missing department | {stats_before['missing_dept']} | {stats_after['missing_dept']} |
| Age outliers (>80) | {stats_before['outliers_age']} | {stats_after['outliers_age']} |
| Final shape | — | {stats_after['shape'][0]} rows × {stats_after['shape'][1]} cols |

**Imputation:** {impute_method} | **Outliers:** {handle_outliers} | **Encoding:** {encoding}
"""
    return fig, metrics_md


def build_tab():
    gr.Markdown("# 🧹 Module 15 — Data Preparation & Cleaning\n*Level: Foundation*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nExplore how different cleaning strategies transform a synthetic dirty dataset (200 rows with injected missing values and outliers).")

    with gr.Row():
        with gr.Column(scale=1):
            impute_dd = gr.Dropdown(
                label="Numeric Imputation Strategy",
                choices=["Mean", "Median", "KNN"],
                value="Median"
            )
            outlier_dd = gr.Dropdown(
                label="Outlier Handling",
                choices=["None", "IQR Cap", "Z-score Remove", "Log Transform"],
                value="IQR Cap"
            )
            encoding_dd = gr.Dropdown(
                label="Categorical Encoding",
                choices=["One-Hot", "Label", "None"],
                value="One-Hot"
            )
            run_btn = gr.Button("▶ Clean Data", variant="primary")

        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Before vs After")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_cleaning_demo,
        inputs=[impute_dd, outlier_dd, encoding_dd],
        outputs=[plot_out, metrics_out]
    )
