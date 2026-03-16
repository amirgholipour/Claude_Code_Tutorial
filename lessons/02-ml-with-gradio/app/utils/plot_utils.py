"""
Plotly visualization helpers for ML demos.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import COLORS
PALETTE = COLORS["palette"]


def scatter_2d(X, y, title="", feature_names=None, class_names=None):
    """2D scatter plot colored by class."""
    x_label = feature_names[0] if feature_names else "Feature 0"
    y_label = feature_names[1] if feature_names else "Feature 1"
    classes = np.unique(y)
    fig = go.Figure()
    for i, cls in enumerate(classes):
        mask = y == cls
        label = str(class_names[cls]) if class_names and cls < len(class_names) else f"Class {cls}"
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            name=label, marker=dict(color=PALETTE[i % len(PALETTE)], size=7, opacity=0.8)
        ))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                      template="plotly_white", height=420)
    return fig


def scatter_3d(X, y, title="", feature_names=None, class_names=None):
    """3D scatter plot colored by class."""
    classes = np.unique(y)
    fig = go.Figure()
    for i, cls in enumerate(classes):
        mask = y == cls
        label = str(class_names[cls]) if class_names and cls < len(class_names) else f"Class {cls}"
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=X[mask, 2], mode="markers",
            name=label, marker=dict(color=PALETTE[i % len(PALETTE)], size=5, opacity=0.8)
        ))
    fn = feature_names or ["PC1", "PC2", "PC3"]
    fig.update_layout(title=title, scene=dict(xaxis_title=fn[0], yaxis_title=fn[1], zaxis_title=fn[2]),
                      template="plotly_white", height=480)
    return fig


def confusion_matrix_heatmap(cm, class_names, title="Confusion Matrix"):
    """Plotly heatmap for a confusion matrix."""
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=class_names, y=class_names,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title=title)
    fig.update_layout(template="plotly_white", height=400)
    return fig


def roc_curve_plot(fpr, tpr, auc_score, title="ROC Curve"):
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_score:.3f}",
                             line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(title=title, xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", template="plotly_white", height=400)
    return fig


def regression_scatter(y_true, y_pred, title="Actual vs Predicted"):
    """Scatter plot of actual vs predicted values with ideal line."""
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             name="Predictions", marker=dict(color=COLORS["primary"], opacity=0.6)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                             line=dict(color=COLORS["danger"], dash="dash")))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted",
                      template="plotly_white", height=400)
    return fig


def learning_curve_plot(train_losses, val_losses=None, title="Training Curve", metric="Loss"):
    """Plot train (and optional val) loss/accuracy over epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines+markers",
                             name="Train", line=dict(color=COLORS["primary"])))
    if val_losses:
        fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode="lines+markers",
                                 name="Validation", line=dict(color=COLORS["warning"])))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=metric,
                      template="plotly_white", height=380)
    return fig


def feature_importance_bar(names, importances, title="Feature Importances"):
    """Horizontal bar chart for feature importances."""
    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[names[i] for i in idx],
        orientation="h", marker_color=COLORS["primary"]
    ))
    fig.update_layout(title=title, xaxis_title="Importance", template="plotly_white", height=max(300, len(names) * 22))
    return fig


def elbow_curve(k_values, inertias, title="K-Means Elbow Curve"):
    """Plot inertia vs K for elbow method."""
    fig = go.Figure(go.Scatter(x=k_values, y=inertias, mode="lines+markers",
                               marker=dict(color=COLORS["primary"], size=8)))
    fig.update_layout(title=title, xaxis_title="Number of Clusters (K)",
                      yaxis_title="Inertia (Within-cluster SSE)", template="plotly_white", height=380)
    return fig


def histogram_grid(df, n_cols=3, title="Feature Distributions"):
    """Grid of histograms for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=numeric_cols,
                        vertical_spacing=0.08)
    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, n_cols)
        fig.add_trace(go.Histogram(x=df[col], marker_color=PALETTE[i % len(PALETTE)],
                                   showlegend=False, name=col),
                      row=r + 1, col=c + 1)
    fig.update_layout(title=title, template="plotly_white", height=max(300, n_rows * 220))
    return fig


def correlation_heatmap(df, title="Correlation Matrix"):
    """Correlation heatmap for numeric features."""
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title=title)
    fig.update_layout(template="plotly_white", height=500)
    return fig
