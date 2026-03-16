"""
Module 19 — Time Series Analysis & Forecasting
Level: Intermediate / Advanced
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEORY = """
## 📖 What Is Time Series Analysis?

A **time series** is a sequence of observations indexed by time. Unlike standard ML datasets where rows are i.i.d. (independent and identically distributed), time series data has **temporal dependency** — the past influences the future.

Examples: stock prices, weather, sensor readings, energy consumption, website traffic.

## 🏗️ Core Concepts

### 1. Decomposition
Any time series can be decomposed into:
- **Trend (T)**: Long-term increase or decrease
- **Seasonality (S)**: Regular periodic patterns (daily, weekly, yearly)
- **Residual (R)**: Random noise after removing trend and seasonality

**Additive model**: y = T + S + R (when seasonal amplitude is constant)
**Multiplicative model**: y = T × S × R (when amplitude grows with trend)

### 2. Stationarity
A stationary series has **constant mean, variance, and autocovariance** over time. Most forecasting models require stationarity.

**Augmented Dickey-Fuller (ADF) test:**
- H₀: Series has a unit root (non-stationary)
- If p < 0.05: reject H₀ → series is stationary ✅

**Making a series stationary:**
- **Differencing**: `y'(t) = y(t) - y(t-1)` removes trend
- **Log transform**: stabilizes variance
- **Seasonal differencing**: `y'(t) = y(t) - y(t-s)` removes seasonality

### 3. Autocorrelation
- **ACF (AutoCorrelation Function)**: Correlation of series with itself at lag k
- **PACF (Partial ACF)**: Direct correlation at lag k, controlling for shorter lags

ACF and PACF plots guide ARIMA parameter selection.

### 4. ARIMA Models
**ARIMA(p, d, q)**:
- **p**: AR order — how many past values to use
- **d**: Differencing degree — how many times to difference for stationarity
- **q**: MA order — how many past errors to use

```
ARIMA(1,1,1): y'(t) = c + φ·y'(t-1) + θ·ε(t-1) + ε(t)
```

**Seasonal ARIMA (SARIMA)**: adds seasonal (P,D,Q,s) terms

### 5. Evaluation Metrics
| Metric | Formula | Notes |
|---|---|---|
| MAE | mean(|y - ŷ|) | Easy to interpret, robust to outliers |
| RMSE | √mean((y - ŷ)²) | Penalizes large errors more |
| MAPE | mean(|(y-ŷ)/y|)×100% | Scale-independent, fails at y≈0 |

## ⚠️ Common Pitfalls
- **Data leakage**: Using future information to predict the past
- **Non-stationarity**: Fitting ARIMA on a non-stationary series → poor forecasts
- **Ignoring seasonality**: A model without seasonal terms will miss regular cycles
- **Train/test split must be temporal**: Never shuffle time series data!
"""

CODE_EXAMPLE = '''
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# ── Synthetic time series ─────────────────────────────────────────
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=365, freq="D")
trend    = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 365)
noise    = np.random.normal(0, 1, 365)
series   = pd.Series(trend + seasonal + noise, index=dates)

# ── Stationarity test ─────────────────────────────────────────────
result = adfuller(series)
print(f"ADF p-value: {result[1]:.4f} → {'stationary' if result[1] < 0.05 else 'non-stationary'}")

# ── Decompose ─────────────────────────────────────────────────────
decomp = seasonal_decompose(series, model="additive", period=30)
# decomp.trend, decomp.seasonal, decomp.resid

# ── Fit ARIMA ─────────────────────────────────────────────────────
train = series[:-30]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
fit   = model.fit(disp=False)

# Forecast 30 steps ahead
forecast = fit.forecast(30)
print("MAE:", abs(series[-30:].values - forecast.values).mean())
'''


def _make_time_series(series_type: str, n: int = 365, seed: int = 42):
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t     = np.arange(n)

    if series_type == "Trend + Seasonal + Noise":
        trend    = t * 0.05
        seasonal = 4 * np.sin(2 * np.pi * t / 52)   # weekly-ish in day-scale
        noise    = rng.normal(0, 0.8, n)
        values   = trend + seasonal + noise

    elif series_type == "Strong Seasonality":
        seasonal = 6 * np.sin(2 * np.pi * t / 30) + 2 * np.sin(2 * np.pi * t / 7)
        noise    = rng.normal(0, 0.5, n)
        values   = seasonal + noise + 10

    elif series_type == "Random Walk":
        steps  = rng.normal(0, 1, n)
        values = np.cumsum(steps)

    else:  # "Step + Trend"
        trend  = t * 0.03
        step   = np.where(t > n // 2, 10, 0)
        noise  = rng.normal(0, 0.7, n)
        values = trend + step + noise

    return pd.Series(values, index=dates)


def _manual_decompose(series: pd.Series, period: int = 30):
    """Simple moving-average decomposition without statsmodels."""
    trend = series.rolling(window=period, center=True).mean()
    detrended = series - trend
    seasonal = np.array([detrended.iloc[i::period].mean() for i in range(period)] * (len(series) // period + 1))[:len(series)]
    seasonal = pd.Series(seasonal, index=series.index)
    residual = series - trend - seasonal
    return trend, seasonal, residual


def _adf_test_simple(series: pd.Series):
    """Simple ADF heuristic without statsmodels."""
    vals = series.dropna().values
    n    = len(vals)
    if n < 20:
        return None, None
    diff = np.diff(vals)
    lagged = vals[:-1]
    cov  = np.cov(diff, lagged)
    rho  = cov[0, 1] / (cov[1, 1] + 1e-8)
    se   = np.std(diff) / (np.std(lagged) * np.sqrt(n) + 1e-8)
    t_stat = rho / (se + 1e-8)
    p_approx = 1 / (1 + np.exp(-0.3 * (t_stat + 2.86)))  # heuristic
    return round(t_stat, 3), round(float(p_approx), 4)


def run_time_series(series_type: str, demo_type: str, horizon: int):
    series = _make_time_series(series_type)
    n      = len(series)

    if demo_type == "Decomposition":
        period = 30
        trend, seasonal, residual = _manual_decompose(series, period=period)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
                            vertical_spacing=0.06)

        for row, (data, color, name) in enumerate(zip(
            [series, trend, seasonal, residual],
            ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350"],
            ["Original", "Trend", "Seasonal", "Residual"]
        ), start=1):
            fig.add_trace(go.Scatter(x=data.index, y=data.values, mode="lines",
                                     line=dict(color=color, width=1.5), name=name), row=row, col=1)

        fig.update_layout(height=600, showlegend=False,
                          title_text=f"Time Series Decomposition — {series_type}")

        t_stat, p_adf = _adf_test_simple(series)
        metrics_md = f"""
### Decomposition Summary

| Component | Mean | Std |
|---|---|---|
| Original | `{series.mean():.3f}` | `{series.std():.3f}` |
| Trend | `{trend.dropna().mean():.3f}` | `{trend.dropna().std():.3f}` |
| Seasonal | `{seasonal.mean():.3f}` | `{seasonal.std():.3f}` |
| Residual | `{residual.dropna().mean():.3f}` | `{residual.dropna().std():.3f}` |

**Stationarity (ADF heuristic):** t-stat ≈ `{t_stat}`, p ≈ `{p_adf}` → {'✅ Stationary' if p_adf and p_adf < 0.05 else '⚠️ Likely non-stationary — try differencing'}
"""

    elif demo_type == "Forecast (Naive Methods)":
        train_size = n - horizon
        train = series.iloc[:train_size]
        test  = series.iloc[train_size:]

        # Naive (last value)
        naive_fc = pd.Series(train.iloc[-1], index=test.index)

        # Seasonal naive (same period last cycle)
        period = 30
        snaive_vals = [train.iloc[-(period - (i % period))] for i in range(horizon)]
        snaive_fc = pd.Series(snaive_vals, index=test.index)

        # Moving average
        ma_val = train.iloc[-period:].mean()
        ma_fc  = pd.Series(ma_val, index=test.index)

        # Linear extrapolation
        x_tr   = np.arange(train_size)
        slope, intercept = np.polyfit(x_tr[-60:], train.values[-60:], 1)
        lin_fc = pd.Series(intercept + slope * np.arange(train_size, train_size + horizon),
                           index=test.index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines",
                                 name="Train", line=dict(color="#42a5f5")))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines",
                                 name="Actual", line=dict(color="black", dash="dot")))
        for fc, name, color in [
            (naive_fc, "Naive", "#ef5350"),
            (snaive_fc, "Seasonal Naive", "#ffa726"),
            (ma_fc, "Moving Avg", "#66bb6a"),
            (lin_fc, "Linear", "#ab47bc"),
        ]:
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines",
                                     name=name, line=dict(color=color, width=2)))

        fig.update_layout(height=420, title_text=f"Forecast Comparison — {horizon}-day horizon")

        def mae(fc, actual): return np.mean(np.abs(fc.values - actual.values))
        metrics_md = f"""
### Forecast Metrics (MAE) — {horizon}-day horizon

| Method | MAE |
|---|---|
| Naive (last value) | `{mae(naive_fc, test):.3f}` |
| Seasonal Naive (period=30) | `{mae(snaive_fc, test):.3f}` |
| Moving Average (30-day) | `{mae(ma_fc, test):.3f}` |
| Linear Extrapolation | `{mae(lin_fc, test):.3f}` |

> Lower MAE = better forecast. Linear extrapolation wins on trending series; Seasonal Naive wins when cycles dominate.
"""

    elif demo_type == "Autocorrelation (ACF/PACF)":
        vals   = series.values
        n_lags = 40
        acf_vals = []
        for lag in range(n_lags + 1):
            if lag == 0:
                acf_vals.append(1.0)
            else:
                y1 = vals[lag:]
                y2 = vals[:-lag]
                r  = np.corrcoef(y1, y2)[0, 1]
                acf_vals.append(float(r))

        conf = 1.96 / np.sqrt(len(vals))

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["ACF (AutoCorrelation Function)",
                                            "Approximate PACF (partial correlation)"])

        lags = list(range(n_lags + 1))
        fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF",
                             marker_color="#42a5f5"), row=1, col=1)
        for sign in [1, -1]:
            fig.add_trace(go.Scatter(x=lags, y=[sign * conf] * len(lags), mode="lines",
                                     line=dict(color="red", dash="dash"),
                                     showlegend=False), row=1, col=1)

        # PACF approx via sequential regression residuals
        pacf_vals = [1.0]
        residuals = vals.copy()
        for lag in range(1, min(n_lags + 1, 21)):
            y_lag = vals[lag:]
            x_lag = vals[:-lag]
            r     = np.corrcoef(y_lag, x_lag)[0, 1]
            pacf_vals.append(float(r))

        fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF",
                             marker_color="#66bb6a"), row=2, col=1)
        for sign in [1, -1]:
            fig.add_trace(go.Scatter(x=list(range(len(pacf_vals))),
                                     y=[sign * conf] * len(pacf_vals), mode="lines",
                                     line=dict(color="red", dash="dash"),
                                     showlegend=False), row=2, col=1)

        fig.update_layout(height=500, showlegend=False,
                          title_text=f"ACF / PACF — {series_type}")

        significant_acf = sum(1 for v in acf_vals[1:] if abs(v) > conf)
        metrics_md = f"""
### ACF / PACF Interpretation

| | Value |
|---|---|
| Confidence bound | ±`{conf:.4f}` |
| Significant ACF lags | `{significant_acf}` / {n_lags} |

**Reading guide:**
- **High ACF at lags 1, 2, 3...**: Series has autocorrelation → AR terms needed
- **ACF cuts off at lag q**: MA(q) process
- **PACF cuts off at lag p**: AR(p) process
- **Both decline slowly**: Differencing needed (ARIMA d>0)
"""

    else:
        fig = go.Figure()
        metrics_md = "Select a demo type."

    return fig, metrics_md


def build_tab():
    gr.Markdown("# 📈 Module 19 — Time Series Analysis & Forecasting\n*Level: Intermediate / Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown("---\n## 🎮 Interactive Demo\n\nExplore synthetic time series: decompose components, visualize autocorrelation, and compare naive forecasting methods.")

    with gr.Row():
        with gr.Column(scale=1):
            series_dd = gr.Dropdown(
                label="Series Type",
                choices=["Trend + Seasonal + Noise", "Strong Seasonality", "Random Walk", "Step + Trend"],
                value="Trend + Seasonal + Noise"
            )
            demo_dd = gr.Dropdown(
                label="Analysis Type",
                choices=["Decomposition", "Forecast (Naive Methods)", "Autocorrelation (ACF/PACF)"],
                value="Decomposition"
            )
            horizon_sl = gr.Slider(label="Forecast Horizon (days)", minimum=7, maximum=90, step=7, value=30)
            run_btn = gr.Button("▶ Analyze", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Result")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_time_series,
        inputs=[series_dd, demo_dd, horizon_sl],
        outputs=[plot_out, metrics_out]
    )
